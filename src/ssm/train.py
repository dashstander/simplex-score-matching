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


def save_checkpoint(params, params_ema, opt_state, epoch, key):
    if jax.process_index() != 0:
        return
    obj = {
        'params': unreplicate(params),
        'params_ema': unreplicate(params_ema),
        'opt_state': unreplicate(opt_state),
        'epoch': epoch,
        'key': key
    }
    with open('model.pkl', 'wb') as f:
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

def setup_model(config, args, key):
    x_shape = (args.batch_size, 81, 9)
    t_shape = (args.batch_size,)
    model_config = TransformerConfig(**config['model'])
    model = hk.transform(make_diffusion_fn(model_config, is_training=True))
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
    config = Config().load(args.config)
    wandb.init(
        project="simplex-score-matching",
        entity="dstander",
        config=config,
        name=args.run_name
    )

    num_local_devices = jax.local_device_count()
    devices = jax.local_devices()
    num_processes = jax.process_count()
    local_rank = jax.process_index()

    print(f'Beginning training with {num_local_devices} devices.')

    #train_data = get_dataset(config)
    np_seeds = np.random.SeedSequence(args.seed)

    #preproc_pool = ray.util.ActorPool(
    #    [TokenToProbsProcessor.remote(s, config, train_data) for s in np_seeds.spawn(2)]
    #)
    
    key, model_key, diffusion_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)

    epoch = 0

    model, params, opt, opt_state = setup_model(config, args, model_key)
    forward_diffusion_fn = setup_forward_diffusion(config, diffusion_key)

    params = jax.device_put_replicated(params, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)
    
    #if args.resume:
    #    train_state = checkpoints.restore_checkpoint(args.checkpoint_dir, train_state)

    train_step_fn = make_forward_fn(model, opt, forward_diffusion_fn)

    key = jax.random.split(key, num_processes)[local_rank]

    #dataset_size = train_data.shape[0]
    if local_rank == 0:
        print(f'Initialized model with {tree_size(train_state.params)} parameters, taking up {tree_bytes(train_state.params)/1e9}GB')
    
    
    def train_epoch(state, indices, num_steps: int, key):
        data_iterator = preproc_pool.map_unordered(
            lambda a, v: a.to_probs.remote(v),
            chunked(indices, config.data.batch_size)
        )
        epoch_start = time.time()
        epoch_losses = []
        i = num_steps
        for batch in tqdm(data_iterator, total=dataset_size // config.data.batch_size):
            if batch.shape[0] != config.data.batch_size:
                continue
            batch_start = time.time()
            key, subkey = jax.random.split(key)
            local_keys = split_and_stack(subkey, num_local_devices)
            texts = psplit(batch, num_local_devices)
            loss, grads = train_step_fn(state, texts, local_keys)
            batch_end = time.time()
            single_loss = unreplicate(loss)
            epoch_losses.append(single_loss)
            batch_log = {'train/loss': single_loss, 'train/time': batch_end - batch_start}
            if jax.process_index() == 0:
                wandb.log(batch_log)
            del batch_log
            i += 1
            if i % 50 == 0:
                tqdm.write(f'Batch {i}, loss {single_loss:g}')
                save_checkpoint(args.checkpoint_dir, state, i)
        epoch_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start
        if jax.process_index() == 0:
            wandb.log({'epoch/loss': epoch_loss, 'epoch/time': epoch_time})
        return state, i

    try:
        num_steps = 0
        for epoch in trange(config.data.epochs):
            tqdm.write(f'Epoch {epoch}')
            key, index_key, *subkeys = jax.random.split(key, 3)
            #permuted_indices = jax.random.permutation(index_key, dataset_size).tolist()
            #train_state, num_steps = train_epoch(train_state, permuted_indices, num_steps, subkeys[0])
            #if jax.process_index() == 0:
            #    save_checkpoint(args.checkpoint_dir, jax_utils.unreplicate(train_state), epoch, prefix='epoch_')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    args, _ = p.parse_known_args()
    main(args)
