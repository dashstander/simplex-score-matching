import argparse
from functools import partial
from flax import jax_utils
from flax.training import checkpoints
import jax
import jax.numpy as jnp
from jax.random import split as rng_split
import numpy as np
from omegaconf import OmegaConf
import optax
from pathlib import Path
from torch.utils.data import DataLoader
import time
from tqdm import tqdm, trange
import wandb

import ssm.aitchison as aitch
from ssm.data import TokenProbsDataset
from ssm.loss import sliced_score_matching
from ssm.model import create_train_state
from ssm.sde import dirichlet_forward_sde
from ssm.utils import ema_update, psplit, tree_bytes, tree_size



p = argparse.ArgumentParser()
p.add_argument('config', type=str)
p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
p.add_argument('--run-name', type=str, default='SSM Diffusion')
p.add_argument('--checkpoint-dir', type=str, default='checkpoints')



def get_datasets(config):
    data = np.lib.format.open_memmap(config.data.dataset_path, mode='r')
    train_data = TokenProbsDataset(config, data[:-config.data.eval_size, :])
    val_data = TokenProbsDataset(config, data[-config.data.eval_size:, :])
    return train_data, val_data


def save(state, ema_params, dir, index):
    if jax.process_index() == 0:
        checkpoints.save_checkpoint(dir / 'train', jax_utils.unreplicate(state), index)
        checkpoints.save_checkpoint(dir / 'ema', jax_utils.unreplicate(ema_params), index)



@partial(jax.pmap, axis_name='batch')
def forward_noising(texts, times, keys):
    texts = aitch.clr(texts, axis=-1, keepdims=True)
    texts = texts / jnp.var(texts, axis=-1, keepdims=True)
    return dirichlet_forward_sde(texts, times, keys)


@partial(jax.pmap, axis_name='batch')
def apply_model(state, noised_texts, t, keys):
    loss, grads = jax.value_and_grad(sliced_score_matching)(state.params, state, noised_texts, t, keys)
    loss, grads = jax.lax.psum(loss, axis_name='batch'), jax.lax.psum(grads, axis_name='batch')
    return loss, grads


@partial(jax.pmap, axis_name='batch')
def eval_model(state, noised_texts, t, key):
    loss = sliced_score_matching(state.params, state, noised_texts, t, key)
    loss = jax.lax.psum(loss, axis_name='batch')
    return loss


@partial(jax.pmap, axis_name='batch')
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def make_optimizer(config):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.optimizer.init_lr,
        peak_value=config.optimizer.peak_lr,
        warmup_steps=config.optimizer.warmup_steps,
        decay_steps=config.optimizer.decay_steps + config.optimizer.warmup_steps,
        end_value=config.optimizer.lr,
    )
    base_opt = optax.chain(
        optax.adamw(learning_rate=schedule),
        optax.clip(config.optimizer.grad_clip)
    )
    if config.optimizer.grad_accum <= 1:
        return base_opt
    else:
        opt = optax.MultiSteps(base_opt, every_k_schedule=config.optimizer.grad_accum)
        return opt


def main(args):
    config = OmegaConf.load(args.config)
    wandb.init(
        project="simplex-score-matching",
        entity="dstander",
        config=config,
        name=args.run_name
    )

    checkpoint_dir = Path(args.checkpoint_dir)

    num_local_devices = jax.local_device_count()
    num_processes = jax.process_count()
    local_rank = jax.process_index()

    print(f'Beginning training with {num_local_devices} devices.')

    train_data, val_data = get_datasets(config)
    
    key = jax.random.PRNGKey(args.seed)
    
    key, state_key = rng_split(key)
    epoch = 0
    opt = make_optimizer(config)
    train_state = create_train_state(state_key, config, opt)
    ema_params = jax.utils.replicate(train_state.params)
    if args.resume:
        train_state = checkpoints.restore_checkpoint(checkpoint_dir / 'train', train_state)
        ema_params = checkpoints.restore_checkpoint(checkpoint_dir / 'ema', ema_params)

    key = rng_split(key, num_processes)[local_rank]
    
    batch_size = config.data.batch_size
    if local_rank == 0:
        print(f'Initialized model with {tree_size(train_state.params)} parameters, taking up {tree_bytes(train_state.params)/1e9}GB')
    ema_params = jax.utils.replicate(ema_params)
    train_state = jax_utils.replicate(train_state)
    

    def do_eval(state, key):
        data = DataLoader(val_data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
        eval_loss = []
        for batch in tqdm(data):
            if batch.shape[0] != batch_size:
                continue
            key, time_key, sde_key, *local_keys = rng_split(key, 3 + (2 * num_local_devices))
            sde_keys = psplit(jnp.stack(rng_split(sde_key, batch_size)), num_local_devices)
            texts = psplit(batch, num_local_devices)
            times = psplit(jax.random.uniform(time_key, (batch_size,)), num_local_devices)
            noised_texts = forward_noising(texts, times, sde_keys)
            loss = eval_model(state, noised_texts, times, psplit(jnp.stack(local_keys), num_local_devices))
            eval_loss.append(loss)
        return jnp.array(eval_loss).mean()
            
    
    def train_epoch(state, ema_params, num_steps: int, key):
        epoch_start = time.time()
        epoch_losses = []
        i = num_steps
        data = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
        for batch in tqdm(data):
            if batch.shape[0] != batch_size:
                continue
            batch_start = time.time()
            key, time_key, sde_key, *local_keys = rng_split(key, 3 + (2 * num_local_devices))
            sde_keys = psplit(jnp.stack(rng_split(sde_key, batch_size)), num_local_devices)
            texts = psplit(batch, num_local_devices)
            times = psplit(
                jax.random.uniform(time_key, (batch_size,), minval=config.sde.start_time, maxval=config.sde.end_time), num_local_devices)
            noised_texts = forward_noising(texts, times, sde_keys)
            forward_sde_time = time.time() - batch_start
            times = times / config.sde.end_time
            loss, grads = apply_model(state, noised_texts, times, psplit(jnp.stack(local_keys), num_local_devices))
            if jnp.any(loss) < 0:
                print(f'Negative loss, what the fuck? {jax_utils.unreplicate(loss)}')
            state = update_model(state, grads)
            ema_params = ema_update(state.params, ema_params, config.model.ema_decay)
            batch_end = time.time()
            single_loss = jax_utils.unreplicate(loss)
            epoch_losses.append(single_loss)
            batch_log = {'train/loss': single_loss, 'train/time': batch_end - batch_start, 'forward_sde/time': forward_sde_time}
            if i % 5 == 0:
                batch_log['model/gradients'] = grads.unfreeze()
            if i % 100 == 0 and i > 0:
                val_start = time.time()
                key, eval_key = rng_split(key)
                val_loss = do_eval(state, eval_key)
                val_time = time.time() - val_start
                batch_log.update({
                    'val/loss': val_loss,
                    'val/time': val_time
                })
                save(state, ema_params, checkpoint_dir, i)
                if jax.process_index() == 0:
                    tqdm.write(f'Batch {i}, loss {single_loss:g}, val loss {val_loss:g}')
                    checkpoints.save_checkpoint(checkpoint_dir / 'train', jax_utils.unreplicate(state), i)
                    checkpoints.save_checkpoint(checkpoint_dir / 'ema', jax_utils.unreplicate(ema_params), i)

            if jax.process_index() == 0:
                wandb.log(batch_log)
            del batch_log
            i += 1

        epoch_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start
        if jax.process_index() == 0:
            wandb.log({'epoch/loss': epoch_loss, 'epoch/time': epoch_time})
        return state, ema_params, i

    try:
        num_steps = 0
        for epoch in trange(config.data.epochs):
            tqdm.write(f'Epoch {epoch}')
            key, subkey = rng_split(key)
            train_state, ema_params, num_steps = train_epoch(train_state, ema_params, num_steps, subkey)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    args, _ = p.parse_known_args()
    main(args)
