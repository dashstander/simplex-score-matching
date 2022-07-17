import argparse
from functools import partial
from flax import jax_utils
from flax.training import checkpoints
from more_itertools import chunked
import jax
import jax.numpy as jnp
from jax.random import split as rng_split
import numpy as np
from omegaconf import OmegaConf
import optax
import ray
import time
from tqdm import tqdm, trange
import wandb

import ssm.aitchison as aitch
from ssm.data import TokenToProbsProcessor
from ssm.utils import (
    psplit,
    tree_bytes,
    tree_size
)
from ssm.model import create_train_state
from ssm.sde import dirichlet_forward_sde


p = argparse.ArgumentParser()
p.add_argument('config', type=str)
p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
p.add_argument('--run-name', type=str, default='SSM Diffusion')
p.add_argument('--checkpoint-dir', type=str, default='checkpoints')


def get_dataset(config):
    data = np.lib.format.open_memmap(config.data.dataset_path, mode='r')
    return data


@partial(jax.pmap, axis_name='batch')
def forward_noising(texts, times, keys):
    texts = aitch.clr(texts, axis=-1, keepdims=True)
    texts = texts / jnp.var(texts, axis=-1, keepdims=True)
    return dirichlet_forward_sde(texts, times, keys)


def loss_fn(params, state, xt, t, keys):
    """
    $x\in\mathbb{R}^{n}$ is in CLR (centered log-ratio) coordinates. It's one softmax away from being on the simplex.
    """
    # v as defined in Sliced Score Matching
    v = jax.random.normal(keys[0], xt.shape)
    def score_fn(x):
        return state.apply_fn({'params': params}, x, t, rngs={'dropout': keys[1]})
    score, score_grad_dot_v = jax.jvp(
        score_fn,
        (xt,),
        (v,)
    ) 
    score_norm = jnp.power(score, 2).sum()
    hutchinson_est = jax.vmap(jnp.tensordot)(v, score_grad_dot_v).sum()
    return 0.5 * score_norm + hutchinson_est


@partial(jax.pmap, axis_name='batch')
def apply_model(state, noised_texts, t, keys):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, state, noised_texts, t, keys)
    loss, grads = jax.lax.psum(loss, axis_name='batch'), jax.lax.psum(grads, axis_name='batch')
    return loss, grads


@partial(jax.pmap, axis_name='batch')
def eval_model(state, noised_texts, t, key):
    loss = loss_fn(state.params, state, noised_texts, t, key)
    loss = jax.lax.psum(loss, axis_name='batch')
    return loss


@partial(jax.pmap, axis_name='batch')
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def make_optimizer(config):
    opt = optax.chain(
        optax.adamw(config.optimizer.lr),    
        optax.clip(config.optimizer.grad_clip)
    )
    return opt


def main(args):
    config = OmegaConf.load(args.config)
    wandb.init(
        project="simplex-score-matching",
        entity="dstander",
        config=config,
        name=args.run_name
    )

    num_local_devices = jax.local_device_count()
    num_processes = jax.process_count()
    local_rank = jax.process_index()

    print(f'Beginning training with {num_local_devices} devices.')

    train_data = get_dataset(config)
    np_seeds = np.random.SeedSequence(args.seed)

    preproc_pool = ray.util.ActorPool(
        [TokenToProbsProcessor.remote(s, config, train_data) for s in np_seeds.spawn(2)]
    )
    
    key = jax.random.PRNGKey(args.seed)
    
    key, state_key = rng_split(key)
    epoch = 0
    opt = make_optimizer(config)
    train_state = create_train_state(state_key, config, opt)
    if args.resume:
        train_state = checkpoints.restore_checkpoint(args.checkpoint_dir, train_state)

    key = rng_split(key, num_processes)[local_rank]
    dataset_size = train_data.shape[0]
    val_size = config.data.val_size
    train_size = dataset_size - val_size
    batch_size = config.data.batch_size
    if local_rank == 0:
        print(f'Initialized model with {tree_size(train_state.params)} parameters, taking up {tree_bytes(train_state.params)/1e9}GB')
    
    train_state = jax_utils.replicate(train_state)
    

    def do_eval(state, key):
        indices = jnp.arange(dataset_size - val_size, dataset_size)
        eval_pool = ray.util.ActorPool(
            [TokenToProbsProcessor.remote(s, config, train_data) for s in np_seeds.spawn(2)]
        )
        data_iterator = eval_pool.map_unordered(
            lambda a, v: a.to_probs.remote(v),
            chunked(indices, batch_size)
        )
        eval_loss = []
        for batch in tqdm(data_iterator, total=train_size // batch_size):
            if batch.shape[0] != batch_size:
                continue
            key, time_key, sde_key, *local_keys = rng_split(key, 3 + num_local_devices)
            sde_keys = psplit(jnp.stack(rng_split(sde_key, num_local_devices)))
            texts = psplit(texts, num_local_devices)
            times = psplit(jax.random.uniform(time_key, (batch_size,)), num_local_devices)
            
            noised_texts = forward_noising(texts, times, sde_keys)
            loss = eval_model(state, noised_texts, times, psplit(jnp.stack(local_keys)))
            eval_loss.append(loss)
        return jnp.array(eval_loss).mean()
            
    
    def train_epoch(state, indices, num_steps: int, key):
        data_iterator = preproc_pool.map_unordered(
            lambda a, v: a.to_probs.remote(v),
            chunked(indices, batch_size)
        )
        epoch_start = time.time()
        epoch_losses = []
        i = num_steps
        
        for batch in tqdm(data_iterator, total=train_size // batch_size):
            if batch.shape[0] != batch_size:
                continue
            batch_start = time.time()
            key, time_key, sde_key, *local_keys = rng_split(key, 3 + (2 * num_local_devices))
            sde_keys = psplit(jnp.stack(rng_split(sde_key, batch_size)), num_local_devices)
            texts = psplit(batch, num_local_devices)
            times = psplit(jax.random.uniform(time_key, (batch_size,)), num_local_devices)
            noised_texts = forward_noising(texts, times, sde_keys)
            forward_sde_time = time.time() - batch_start
            loss, grads = apply_model(state, noised_texts, times, psplit(jnp.stack(local_keys), num_local_devices))
            state = update_model(state, grads)
            batch_end = time.time()
            single_loss = jax_utils.unreplicate(loss)
            epoch_losses.append(single_loss)
            batch_log = {'train/loss': single_loss, 'train/time': batch_end - batch_start, 'forward_sde/time': forward_sde_time}
            i += 1
            if i % 5 == 0:
                batch_log['model/gradients'] = grads
            if i % 50 == 0:
                val_start = time.time()
                key, eval_key = rng_split(key)
                val_loss = do_eval(state, eval_key)
                val_time = time.time() - val_start
                batch_log.update({
                    'val/loss': val_loss,
                    'val/time': val_time
                })
                if jax.process_index() == 0:
                    tqdm.write(f'Batch {i}, loss {single_loss:g}, val loss {val_loss:g}')
                    checkpoints.save_checkpoint(args.checkpoint_dir, state, i)
            if jax.process_index() == 0:
                wandb.log(batch_log)
            del batch_log

        epoch_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start
        if jax.process_index() == 0:
            wandb.log({'epoch/loss': epoch_loss, 'epoch/time': epoch_time})
        return state, i

    try:
        num_steps = 0
        for epoch in trange(config.data.epochs):
            tqdm.write(f'Epoch {epoch}')
            key, index_key, *subkeys = rng_split(key, 3)
            permuted_indices = jax.random.permutation(index_key, train_size).tolist()
            train_state, num_steps = train_epoch(train_state, permuted_indices, num_steps, subkeys[0])
            if jax.process_index() == 0:
                checkpoints.save_checkpoint(args.checkpoint_dir, jax_utils.unreplicate(train_state), epoch, prefix='epoch_')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    args, _ = p.parse_known_args()
    main(args)
