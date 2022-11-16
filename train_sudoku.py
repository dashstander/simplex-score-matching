import os
os.environ['GEOMSTATS_BACKEND'] = 'jax'
# ~17GB
os.environ['TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD'] = '17179869184'

import argparse
from confection import Config
from functools import partial
import haiku as hk
from haiku.data_structures import to_mutable_dict, tree_bytes, tree_size
import jax
import jax.numpy as jnp
from jax.nn import softmax
import numpy as np
import optax
from pathlib import Path
import pickle
import time
from tqdm import tqdm
import wandb

from ssm.data.sudoku import make_train_loader, make_val_loader
from ssm.manifolds import HypersphereProductManifold
from ssm.sde.solver import HypersphereBackwardsSolver
from ssm.utils import (
    psplit,
    punsplit,
    split_and_stack,
    unreplicate   
)
from ssm.models.transformer import TransformerConfig, make_diffusion_fn, normalize


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


def save_checkpoint(checkpoint_dir, params, ema_params, opt_state, epoch, steps, key):
    if jax.process_index() == 0:
        ckpt_path = checkpoint_dir / f'model_{epoch}_{steps}.pkl'
        blob = {
            'params': unreplicate(params),
            'ema_params': unreplicate(ema_params),
            'opt_state': unreplicate(opt_state),
            'epoch': epoch,
            'key': key
        }
        with open(ckpt_path, 'wb') as f:
            pickle.dump(blob, f)


def sudoku_noise_and_geodesic(key, manifold, x0, t):
    batch_size, seq_len, simplex_dim = x0.shape
    noise = jax.random.dirichlet(
        key,
        alpha=jnp.ones((simplex_dim,)),
        shape=(batch_size, seq_len)
    )
    x_final = jnp.sqrt(noise)
    tangent_vecs =jax.vmap(manifold.log)(x_final, x0)
    scaled_vecs = tangent_vecs * jnp.expand_dims(t, axis=(1, 2))
    return jax.vmap(manifold.exp)(scaled_vecs, x0)


def make_forward_fn(model, ema_update, opt, manifold, axis_name='batch'):

    def loss_fn(params, x0, masks, key):
        time_key, noise_key, model_key = jax.random.split(key, 3)
        batch_size, _, _ = x0.shape
        t = jnp.cos(
            (jnp.pi / 2) * jax.random.uniform(time_key, (batch_size,))
        )
        noised_x = sudoku_noise_and_geodesic(noise_key, manifold, x0, t)
        logits = model.apply(params, model_key, noised_x, masks, t)
        cross_ent = optax.softmax_cross_entropy(logits, x0)
        loss = jnp.mean(cross_ent)
        return loss

    def train_step(params, ema_state, opt_state, key, inputs, masks):
        loss_grads = jax.value_and_grad(loss_fn)(params,inputs, masks, key)
        loss, grads = jax.lax.pmean(loss_grads, axis_name=axis_name)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params, ema_state = ema_update.apply(None, ema_state, None, params)

        return loss, grads, params, ema_params, ema_state, opt_state
    
    return jax.pmap(train_step, axis_name=axis_name)


def puzzle_random_init(solutions, masks, shape, key):
    batch_size, seq_len, dim = shape
    rv = jax.random.dirichlet(key, jnp.ones((dim,)), (batch_size, seq_len))
    #rv = normalize(jnp.abs(rv))
    return solutions * masks + (1 - masks) * rv


def entropy(x, axis=-1):
    p = softmax(x, axis=axis)
    return -1. * jnp.sum(p * jnp.log(p), axis=axis)



@partial(jax.pmap, axis_name='batch')
def calc_val_metrics(preds, solutions, masks):
    entropies = entropy(preds, axis=-1)
    predicted = jnp.argmax(preds, axis=-1) + 1
    solutions = jnp.argmax(solutions, axis=-1) + 1
    masked = jnp.max(masks, axis=-1)
    not_masked = 1 - masked
    correct = predicted == solutions
    correct_vals = 1. * jnp.sum(correct * not_masked)
    correct_puzzles = 1. * jax.vmap(jnp.all)(correct).sum()
    #batch_puzzle_pcnt =  correct_puzzles / predicted.shape[0]
    batch_val_pcnt = correct_vals / not_masked.sum()
    batch_unmasked_ent = jnp.sum(entropies * not_masked, axis=-1) / jnp.count_nonzero(not_masked)
    batch_masked_ent = jnp.sum(entropies *masked , axis=-1) / jnp.count_nonzero(masked)
    return (
        correct_vals,
        correct_puzzles,
        batch_val_pcnt,
        batch_unmasked_ent,
        batch_masked_ent
    )


def validation_metrics(
    preds,
    solutions,
    masks,
    num_solved_puzzles,
    num_correct_vals,
    pcnt_correct_vals,
    masked_entropies,
    unmasked_entropies
):
    nvals, npuzz, pcntval, unmaskent, maskent = calc_val_metrics(preds, solutions, masks)
    num_solved_puzzles.append(jnp.sum(npuzz))
    num_correct_vals.append(jnp.sum(nvals))
    pcnt_correct_vals.append(jnp.mean(pcntval))
    masked_entropies.append(jnp.mean(maskent))
    unmasked_entropies.append(jnp.mean(unmaskent))


def make_validation_fn(config):

    solver = make_solver(config)

    def val_fn(params, key):
        num_local_devices = jax.local_device_count()
        #num_batches = config['data']['num_val_batches']
        key, subkey = jax.random.split(key)
        #solve_fn = make_solver(config, params, subkey1)
        num_solved_puzzles = []
        num_correct_vals = []
        pcnt_correct_vals = []
        masked_entropies = []
        unmasked_entropies = []
        loader = make_val_loader(config, subkey)
        #cpu_dev = jax.devices("cpu")[0]
        for solutions, masks in loader:
            key, puzzle_key, solve_key = jax.random.split(key, 3)
            puzzles = puzzle_random_init(solutions, masks, solutions.shape, puzzle_key)
            puzzles = psplit(puzzles, num_local_devices)
            masks = psplit(masks, num_local_devices)
            solve_keys = split_and_stack(solve_key, num_local_devices)
            preds, path = solver(params, solve_keys, puzzles, masks)
            #preds = punsplit(jax.device_put(preds), cpu_dev)
            validation_metrics(
                preds,
                psplit(solutions, num_local_devices),
                masks,
                num_solved_puzzles,
                num_correct_vals,
                pcnt_correct_vals,
                masked_entropies,
                unmasked_entropies
            )
        #val_time = time.time() - val_start
        num_vals = jnp.array(num_correct_vals).sum()
        val_accuracy = jnp.array(pcnt_correct_vals).mean()
        num_puzzles = jnp.array(num_solved_puzzles).sum()
        masked_entropies = jnp.array(masked_entropies)
        unmasked_entropies = jnp.array(unmasked_entropies)
        return {
            'validation/num_correct_values': num_vals,
            'validation/num_solved_puzzles': num_puzzles,
            'validation/value_accuracy': val_accuracy,
            'validation/masked_val_entropy': masked_entropies,
            'validation/unmasked_val_entropy': unmasked_entropies
        }
    return val_fn


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
    x_shape = (1, 81, 9)
    t_shape = (1,)
    transformer_config = TransformerConfig.from_config(config)
    model = hk.transform(make_diffusion_fn(transformer_config, training=True))
    params = model.init(key, jnp.full(x_shape, 1/3.), jnp.zeros(x_shape), jnp.zeros(t_shape))
    opt = make_optimizer(config['optimizer'])
    opt_state = opt.init(params)
    return model, params, opt, opt_state


def make_solver(config):
    num_steps = config['sde']['num_bwd_steps']
    cfg_weight = config['sde']['cfg_weight']
    transformer_config = TransformerConfig.from_config(config)
    model = hk.transform(make_diffusion_fn(transformer_config, training=False))
    solver = HypersphereBackwardsSolver(9, 81, num_steps, cfg_weight, 1., model)
    def solve_fn(params, rng, x_final, mask):
        return solver.solve(params, rng, x_final, mask)
    return jax.pmap(solve_fn, axis_name='batch')


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
    
    key, model_key = jax.random.split(jax.random.PRNGKey(args.seed))

    sudoku_manifold = HypersphereProductManifold(8, 81)

    model, params, opt, opt_state = setup_model(config, model_key)
    
    #if args.resume:
    #    train_state = checkpoints.restore_checkpoint(args.checkpoint_dir, train_state)
    ema_decay = config['model']['ema']['decay']
    ema_warmup = config['model']['ema']['warmup']
    zero_debias = ema_warmup == 0
    ema_fn = hk.transform_with_state(
        lambda x: hk.EMAParamsTree(
            ema_decay,
            zero_debias=zero_debias,
            warmup_length=ema_warmup
        )(x))
    _, ema_state = ema_fn.init(None, params)

    train_step_fn = make_forward_fn(model, ema_fn, opt, sudoku_manifold)

    validation_fn = make_validation_fn(config)

    key = jax.random.split(key, num_processes)[local_rank]
    num_params = tree_size(params)
    num_bytes = tree_bytes(params)

    if local_rank == 0:
        print(f'Initialized model with {num_params} parameters, taking up {num_bytes/1e9}GB')

    params = jax.device_put_replicated(params, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)
    ema_state = jax.device_put_replicated(ema_state, devices)
    total_size = int(config['data']['num_train_batches'])
    
    def train_epoch(params, ema_state, opt_state, epoch, key):
        key, data_key = jax.random.split(key)
        loader = make_train_loader(config, data_key) 
        epoch_losses = []
        for i, batch in tqdm(enumerate(loader), total=total_size):
            puzzles, masks = batch
            batch_start = time.time()
            key, subkey = jax.random.split(key)
            local_keys = split_and_stack(subkey, num_local_devices)
            puzzles = psplit(puzzles, num_local_devices)
            masks = psplit(masks, num_local_devices)
            loss, grads, params, ema_params, ema_state, opt_state = train_step_fn(
                params, ema_state, opt_state, local_keys, puzzles, masks
            )
            batch_end = time.time()
            single_loss = unreplicate(loss)
            epoch_losses.append(single_loss)
            #print(jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), grads))
            batch_log = {'train/loss': single_loss, 'train/time': batch_end - batch_start}
            if i % 10 == 0:
                grads = to_mutable_dict(grads)
                batch_log['model/gradients'] = jax.tree_map(wandb.Histogram, grads)
            if i % 1000 == 0:
                tqdm.write(f'Batch {i}, loss {single_loss:g}')
                save_checkpoint(checkpoint_dir, params, ema_params, opt_state, epoch, i, key)
                key, subkey = jax.random.split(key)
                val_start = time.time()
                val_log = validation_fn(ema_params, subkey)
                val_time = time.time() - val_start
                batch_log.update(val_log)
                batch_log['validation/time'] = val_time
                for k, v in val_log.items():
                    print(f'{k}: {v}')
            wandb_log(batch_log)
            del batch_log
        epoch_loss = np.mean(epoch_losses)
        
        wandb_log({'epoch/loss': epoch_loss})
        return params, ema_state, opt_state

    try:
        for epoch in range(config['data']['epochs']):
            print(f'############### Epoch {epoch}\n###################################')
            key, subkey = jax.random.split(key)
            params, ema_state, opt_state = train_epoch(
                params, ema_state, opt_state, epoch, subkey
            )
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    args, _ = p.parse_known_args()
    main(args)
