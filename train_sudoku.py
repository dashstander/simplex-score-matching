import os
os.environ['GEOMSTATS_BACKEND'] = 'jax'
# ~17GB
os.environ['TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD'] = '17179869184'

import argparse
from confection import Config
from concurrent.futures import ThreadPoolExecutor
import haiku as hk
from haiku.data_structures import to_mutable_dict, tree_bytes, tree_size
import jax
import jax.numpy as jnp
import numpy as np
import optax
from pathlib import Path
import pickle
import time
from tqdm import tqdm
import wandb

from ssm.data.sudoku import make_batch
from ssm.manifolds import make_sudoku_forward_walker, make_sudoku_solver
from ssm.utils import (
    psplit,
    split_and_stack,
    unreplicate   
)
from ssm.models.model import TransformerConfig, make_diffusion_fn, normalize


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
    if jax.process_index() == 0:
        ckpt_path = checkpoint_dir / f'model_{epoch}_{steps}.pkl'
        blob = {
            'params': unreplicate(params),
            'opt_state': unreplicate(opt_state),
            'epoch': epoch,
            'key': key
        }
        with open(ckpt_path, 'wb') as f:
            pickle.dump(blob, f)


def make_forward_fn(model, opt, grw_fn, axis_name='batch'):

    def loss_fn(params, x0, masks, key):
        time_key, grw_key, model_key = jax.random.split(key, 3)
        batch_size, seq_len, manif_dim = x0.shape
        t = jax.random.uniform(time_key, (batch_size,))       
        noised_x, target_score = grw_fn(x0, t, split_and_stack(grw_key, batch_size))
        target_score = normalize(target_score)
        pred_score = model.apply(params, model_key, noised_x, t)
        not_masked = 1 - masks
        mse = jnp.square(pred_score - target_score) * not_masked
        loss = jnp.mean(mse)
        return loss

    def train_step(params, opt_state, key, inputs, masks):
        loss_grads = jax.value_and_grad(loss_fn)(params,inputs, masks, key)
        loss, grads = jax.lax.pmean(loss_grads, axis_name=axis_name)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, grads, params, opt_state
    
    return jax.pmap(train_step, axis_name=axis_name)


def load_val_data(config):
    batch_size = config['data']['batch_size']
    num_val_samples = batch_size * config['data']['num_val_batches']
    val_fp = config['data']['val_path']
    data = np.load(val_fp)
    puzzles, masks = data['puzzles'], data['masks']
    puzzles = jax.nn.one_hot(puzzles - 1, num_classes=9)
    masks = jnp.asarray(masks, dtype=jnp.int32)[..., None]
    return puzzles[:num_val_samples], masks[:num_val_samples]

def puzzle_random_init(solutions, masks, key):
    rv = jax.random.normal(key, solutions.shape)
    rv = normalize(jnp.abs(rv))
    return solutions * masks + (1 - masks) * rv

def calc_val_metrics(predicted, solutions, masks):
    not_masked = 1 - masks
    correct = predicted == solutions
    num_correct_vals = 1. * jnp.sum(correct * not_masked)
    num_correct_puzzles = 1. * jax.vmap(jnp.all)(correct).sum()
    pcnt_correct_puzzles =  num_correct_puzzles / predicted.shape[0]
    pcnt_correct_vals = num_correct_vals / not_masked.sum()
    return pcnt_correct_puzzles, pcnt_correct_vals


def do_validation(config, model, params, num_steps, key):
    num_local_devices = jax.local_device_count()
    batch_size = config['data']['batch_size']
    num_batches = config['data']['num_val_batches']
    val_start = time.time()
    key, subkey = jax.random.split(key)
    solve_fn = make_solver(model, params, num_steps, subkey)
    pcnt_solved_puzzles = []
    pcnt_correct_vals = []
    val_puzzles, val_masks = load_val_data(config)
    val_data = zip(jnp.vsplit(val_puzzles, num_batches), jnp.vsplit(val_masks, num_batches))
    for solutions, masks in val_data:
        batch_size = puzzles.shape[0]
        key, puzzle_key, solve_key = jax.random.split(key, 3)
        puzzles = puzzle_random_init(solutions, masks, puzzle_key)
        puzzles = psplit(puzzles, num_local_devices)
        masks = psplit(masks, num_local_devices)
        solve_keys = split_and_stack(solve_key, num_local_devices)
        preds = solve_fn(puzzles, masks, jnp.ones((batch_size,)), solve_keys)
        pred_puzzle = jnp.argmax(preds, axis=-1) + 1
        batch_correct_puzzles, batch_correct_vals = calc_val_metrics(pred_puzzle, solutions, masks)
        pcnt_solved_puzzles.append(batch_correct_puzzles)
        pcnt_correct_vals.append(batch_correct_vals)
    val_time = time.time() - val_start
    val_accuracy = jnp.array(pcnt_correct_vals).mean()
    puzzle_accuracy = jnp.array(pcnt_solved_puzzles).mean()
    return {
        'validation/value_accuracy': val_accuracy,
        'validation/puzzle_accuracy': puzzle_accuracy,
        'val/time': val_time
    }


def make_optimizer(config):

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config['init_lr'],
        peak_value=config['peak_lr'],
        warmup_steps=config['warmup_steps'],
        decay_steps=config['decay_steps'] + config['warmup_steps'],
        end_value=config['lr'],
    )
    base_opt = optax.chain(
        optax.adamw(learning_rate=schedule),
        optax.clip(config['grad_clip'])
    )
    return base_opt

def setup_model(config, key):
    #batch_size = config['data']['batch_size']
    x_shape = (1, 81, 9)
    t_shape = (1,)
    model_config = TransformerConfig(**config['model'])
    model = hk.transform(make_diffusion_fn(model_config, training=True))
    params = model.init(key, jnp.full(x_shape, 1/3.), jnp.zeros(t_shape))
    opt = make_optimizer(config['optimizer'])
    opt_state = opt.init(params)
    return model, params, opt, opt_state


def setup_forward_diffusion(config, key):
    num_steps = config['sde']['num_steps']
    beta_0 = config['sde']['beta_0']
    beta_f = config['sde']['beta_f']
    diffusion = hk.transform(make_sudoku_forward_walker(num_steps, beta_0, beta_f))
    x_init = jnp.full((81, 9), 1./3)
    t_init = jnp.array(2.)
    diff_params = diffusion.init(key, x_init, t_init)
    def forward_fn(x0, t, rng):
        return diffusion.apply(diff_params, rng, x0, t)
    return forward_fn

def make_solver(config, model, params, num_steps, key):
    num_steps = config['sde']['num_steps']
    beta_0 = config['sde']['beta_0']
    beta_f = config['sde']['beta_f']
    x_init = jnp.full((1, 81, 9), 1./3)
    t_init = jnp.array([2., 2.])
    def score_fn(rng, x, masks, time):
        return model.apply(params, rng, x, masks, time)
    solver = hk.transform(make_sudoku_solver(score_fn, num_steps, beta_0, beta_f))
    solver_params = solver.init(key, x_init, x_init, t_init)
    def _solve(x, mask, t, key):
        return solver.apply(solver_params, key, x, mask, t)
    return jax.pmap(_solve, axis_name='batch')


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
    num_params = tree_size(params)
    num_bytes = tree_bytes(params)
    if local_rank == 0:
        print(f'Initialized model with {num_params} parameters, taking up {num_bytes/1e9}GB')
    
    params = jax.device_put_replicated(params, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)
    batch_size = config['data']['batch_size']

    def train_epoch(params, opt_state, epoch, key):
        executor = ThreadPoolExecutor(max_workers=10)
        key, data_key = jax.random.split(key)
        data_iterator = executor.map(
            lambda x: make_batch(x, batch_size),
            jax.random.split(data_key, 50_000)
        )
        epoch_losses = []
        for i, batch in tqdm(enumerate(data_iterator), total=50_000):
            puzzles, masks = batch
            batch_start = time.time()
            key, subkey = jax.random.split(key)
            local_keys = split_and_stack(subkey, num_local_devices)
            puzzles = psplit(puzzles, num_local_devices)
            masks = psplit(masks, num_local_devices)
            loss, grads, params, opt_state = train_step_fn(
                params, opt_state, local_keys, puzzles, masks
            )
            batch_end = time.time()
            single_loss = unreplicate(loss)
            epoch_losses.append(single_loss)

            #print(jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), grads))
            batch_log = {'train/loss': single_loss, 'train/time': batch_end - batch_start}
            #if i % 5 == 0:
            #    batch_log['model/gradients'] = to_mutable_dict(grads)
            wandb_log(batch_log)
            del batch_log
            if i % 1000 == 0:
                tqdm.write(f'Batch {i}, loss {single_loss:g}')
                save_checkpoint(checkpoint_dir, params, opt_state, epoch, i, key)
        epoch_loss = np.mean(epoch_losses)
        
        wandb_log({'epoch/loss': epoch_loss})
        return params, opt_state

    try:
        for epoch in range(config['data']['epochs']):
            print(f'############### Epoch {epoch}\n###################################')
            key, subkey = jax.random.split(key)
            params, opt_state = train_epoch(params, opt_state, epoch, subkey)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    args, _ = p.parse_known_args()
    main(args)
