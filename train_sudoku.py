import os
os.environ['GEOMSTATS_BACKEND'] = 'jax'
# ~17GB
os.environ['TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD'] = '17179869184'

import argparse
from confection import Config
from concurrent.futures import ThreadPoolExecutor
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from pathlib import Path
import pickle
import time
from tqdm import tqdm, trange
import wandb

from ssm.data.sudoku import make_batch
from ssm.manifolds import make_sudoku_forward_walker, make_sudoku_solver
from ssm.utils import (
    psplit,
    split_and_stack,
    tree_bytes,
    tree_size,
    unreplicate   
)
from ssm.models.model import TransformerConfig, make_diffusion_fn


p = argparse.ArgumentParser()
p.add_argument('config', type=str)
#p.add_argument('--resume', type=str,
#                   help='the checkpoint to resume from')
p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
p.add_argument('--run-name', type=str, default='SSM Diffusion')
p.add_argument('--checkpoint-dir', type=str, default='checkpoints')


def wandb_log(data):
    if jax.process_index() == 0:
        wandb.log(data)


def save_checkpoint(checkpoint_dir, params, opt_state, epoch, steps, key):
    if jax.process_index() != 0:
        return
    ckpt_path = checkpoint_dir / f'model_{epoch}_{steps}.pkl'
    obj = {
        'params': unreplicate(params),
        #'params_ema': unreplicate(params_ema),
        'opt_state': unreplicate(opt_state),
        'epoch': epoch,
        'key': key
    }
    with open(ckpt_path, 'wb') as f:
        pickle.dump(obj, f)


def make_forward_fn(model, opt, grw_fn, axis_name='batch'):

    def loss_fn(params, x0, masks, key):
        time_key, grw_key, model_key = jax.random.split(key, 3)
        t = jax.random.uniform(time_key, (x0.shape[0],))       
        noised_x, target_score = grw_fn(grw_key, x0, t)
        target_score = target_score / jnp.sum(target_score ** 2, axis=-1, keepdims=True)
        pred_score = model.apply(params, model_key, noised_x, masks, t)
        return jnp.mean((pred_score - target_score)**2)

    def train_step(params, opt_state, key, inputs, masks):
        loss_grads = jax.value_and_grad(loss_fn)(params,inputs, masks, key)
        loss, grads = jax.lax.pmean(loss_grads, axis_name=axis_name)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state
    
    return jax.pmap(train_step, axis_name=axis_name)


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
    return base_opt

def setup_model(config, key):
    batch_size = config['data']['batch_size']
    x_shape = (batch_size, 81, 9)
    t_shape = (batch_size,)
    model_config = TransformerConfig(**config['model'])
    model = hk.transform(make_diffusion_fn(model_config, training=True))
    params = model.init(key, jnp.full(x_shape, 1/3.), jnp.zeros(t_shape))
    opt = make_optimizer(config)
    opt_state = opt.init(params)
    return model, params, opt, opt_state


def setup_forward_diffusion(config, key):
    diffusion = hk.transform(make_sudoku_forward_walker)
    x_init = jnp.full((81, 9), 1./3)
    t_init = jnp.array(2.)
    diff_params = diffusion.init(key, x_init, t_init, config['sde']['num_steps'])
    def forward_fn(x0, t, rng):
        return diffusion.apply(diff_params, rng, x0, t)
    return jax.vmap(forward_fn)


def main(args):
    config = Config().from_disk(args.config)
    wandb.init(
        project="simplex-score-matching",
        entity="dstander",
        config=config,
        name=args.run_name
    )

    checkpoint_dir = Path(args.checkpoint_dir)

    num_local_devices = jax.local_device_count()
    devices = jax.local_devices()
    num_processes = jax.process_count()
    local_rank = jax.process_index()

    print(f'Beginning training with {num_local_devices} devices.')
    
    key, model_key, diffusion_key = jax.random.split(
        jax.random.PRNGKey(args.seed),
        3
    )

    model, params, opt, opt_state = setup_model(config, model_key)
    forward_diffusion_fn = setup_forward_diffusion(config, diffusion_key)
    
    #if args.resume:
    #    train_state = checkpoints.restore_checkpoint(args.checkpoint_dir, train_state)

    train_step_fn = make_forward_fn(model, opt, forward_diffusion_fn)

    key = jax.random.split(key, num_processes)[local_rank]

    #dataset_size = train_data.shape[0]
    if local_rank == 0:
        print(f'Initialized model with {tree_size(params)} parameters, taking up {tree_bytes(params)/1e9}GB')
    
    params = jax.device_put_replicated(params, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)

    def train_epoch(params, opt_state, epoch, key):
        executor = ThreadPoolExecutor(max_workers=10)
        key, data_key = jax.random.split(key)
        data_iterator = executor.map(make_batch, jax.random.split(data_key, 50_000))
        epoch_losses = []
        for i, batch in tqdm(enumerate(data_iterator), total=50_000):
            puzzles, masks = batch
            batch_start = time.time()
            key, subkey = jax.random.split(key)
            local_keys = split_and_stack(subkey, num_local_devices)
            puzzles = psplit(puzzles, num_local_devices)
            masks = psplit(masks, num_local_devices)
            loss, grads = train_step_fn(params, opt_state, local_keys, puzzles, masks)
            batch_end = time.time()
            single_loss = unreplicate(loss)
            epoch_losses.append(single_loss)
            batch_log = {'train/loss': single_loss, 'train/time': batch_end - batch_start}
            wandb_log(batch_log)
            del batch_log
            if i % 1000 == 0:
                tqdm.write(f'Batch {i}, loss {single_loss:g}')
                save_checkpoint(checkpoint_dir, params, opt_state, epoch, i)
        epoch_loss = np.mean(epoch_losses)
        
        wandb_log({'epoch/loss': epoch_loss})
        return params, opt_state

    try:
        for epoch in trange(config.data.epochs):
            tqdm.write(f'Epoch {epoch}')
            key, subkey = jax.random.split(key)
            params, opt_state = train_epoch(params, opt_state, epoch, subkey)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    args, _ = p.parse_known_args()
    main(args)
