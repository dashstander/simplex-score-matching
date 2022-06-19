import argparse
import gcsfs
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jrand
from omegaconf import OmegaConf
import optax
import pickle
import ray
import time
from tqdm import tqdm, trange
import wandb

from ssm.data import tokenize_and_split
from ssm.utils import (
    psplit,
    t_to_alpha_sigma,
    tokens_to_probs,
    unreplicate
)
from ssm.model import get_and_init_model


p = argparse.ArgumentParser()
p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
p.add_argument('--run-name', type=str, default='SSM Diffusion')


def get_dataset(config):
    fs = gcsfs.GCSFileSystem(project=config.gcs.project)
    data = (
        ray.data.read_numpy(config.data.dataset_path, filesystems=fs)
        .shuffle()
        .repeat(config.data.epochs)
        .window(blocks_per_window=2)
        .map_batch(tokens_to_probs)
    )
    return data


    


def resume_training(checkpoint_file):
    with open(checkpoint_file, mode='rb') as param_file: 
        ckpt = pickle.load(param_file, 'rb')
    epoch = ckpt['epoch']
    params = jax.tree_map(jnp.array, ckpt['params'])
    params_ema = jax.tree_map(jnp.array, ckpt['params_ema'])
    opt_state = jax.tree_map(jnp.array, ckpt['opt_state'])
    key = jax.tree_map(jnp.array, ckpt['key'])
    del ckpt
    return epoch, params, params_ema, opt_state, key


def save(params, params_ema, opt_state, epoch, key):
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


def make_forward_fn(model_fn, opt):

    def compute_loss(params, key, texts, extra_args, is_training):
        keys = jrand.split(key, 2)
        t = jrand.uniform(keys[0], texts.shape[:1])
        alphas, sigmas = t_to_alpha_sigma(t)
        simplex_noise = jrand.normal(keys[1], texts.shape)
        noised_texts = texts * alphas[:, None] + simplex_noise * sigmas[:, None]
        text_targets = simplex_noise * alphas[:, None] - texts * sigmas[:, None]
        v = model_fn(params, key, noised_texts, t, extra_args, is_training)
        return jnp.mean(jnp.square(v - text_targets))
        
    def train_step(params, opt_state, key, inputs, embeddings, extra_args, axis_name='i'):
        loss_grads = jax.value_and_grad(compute_loss)(params, key, inputs, embeddings, extra_args, jnp.array(1))
        loss, grads = jax.lax.pmean(loss_grads, axis_name)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state
    
    return train_step


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

    train_data = get_dataset(config)

    ema_fn = hk.transform_with_state(lambda x: hk.EMAParamsTree(config.ema_decay_rate)(x))

    opt = optax.chain(
        optax.adamw(args.lr),    
        optax.clip(args.grad_clip)
    )
    key = jax.random.PRNGKey(args.seed)
    
    if not args.resume:
        key, subkey = jax.random.split(key)
        epoch = 0
        params, model_fn = get_and_init_model(config, subkey)
        params = jax.tree_map(lambda x: x / 2, params)
        _, ema_state = ema_fn.init(None, params, warmup_length=config.warmup_length)
        params_ema, ema_state = ema_fn.apply(None, ema_state, None, params)
        opt_state = opt.init(params)
    else:
        epoch, params, params_ema, opt_state, key = resume_training(args.resume)

        
    print('Model parameters:', hk.data_structures.tree_size(params))

    params = jax.device_put_replicated(params, jax.local_devices())
    params_ema = jax.device_put_replicated(params_ema, jax.local_devices())
    ema_state = jax.device_put_replicated(ema_state, jax.jax.local_devices())
    opt_state = jax.device_put_replicated(opt_state, jax.local_devices())
    p_ema_update = jax.pmap(ema_fn.apply, in_axes=(None, 0, None, 0))

    key = jax.random.split(key, num_processes)[local_rank]
    jit_start = time.time()
    train_step = jax.pmap(
        make_forward_fn(model_fn, opt),
        axis_name='i'
    )
    jit_time = time.time() - jit_start
    print(f'It took {jit_time}s to compile the train_step function.')
    def train_one_epoch(params, params_ema, opt_state, data, key):
        key, subkey = jrand.split(key)
        seed = jrand.randint(subkey, (1,), 0, 100000)
        # data = train_data.random_shuffle(seed)
        # train_iter = data.iter_batches(prefetch_blocks=4, batch_size=config.batch_size)
        for i, batch in enumerate(tqdm(data.iter_batches(config.batch_size))):
            key, curr_key, *local_keys = jax.random.split(key, 2 + num_local_devices)
            batch = tokens_to_probs(
                curr_key,
                jnp.asarray(batch),
                config.concentration,
                config.vocab_size
            )
            texts = jax.tree_map(lambda x: psplit(x, num_local_devices), batch)
            print('Doing forward and backward passes')
            loss, params, opt_state = train_step(
                params,
                opt_state,
                jnp.stack(local_keys),
                texts,
                {}
            )
            params_ema, ema_state = p_ema_update(None, ema_state, None, params)
            batch_log = {'embedding_loss': unreplicate(loss)}
            wandb.log(batch_log)
            del batch_log
            if i % 50 == 0:
                tqdm.write(f'Epoch {epoch}, iteration {i}, loss {unreplicate(loss):g}')
        return params, params_ema, opt_state, ema_state

    try:
        key, subkey = jax.random.split(key)
        for i in trange(config.epochs):
            tqdm.write(f'Epoch {epoch}')
            key, *subkeys = jax.random.split(key, 3)
            params, params_ema, opt_state, ema_state = train_one_epoch(params, params_ema, opt_state, subkeys[0])
            save(params, params_ema, opt_state, i, subkeys[1])
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    args, _ = p.parse_known_args()
    main(args)
