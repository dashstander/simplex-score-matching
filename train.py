import argparse
import functools
import gcsfs
from flax import jax_utils
from flax.training import checkpoints
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
from omegaconf import OmegaConf
import optax
import ray
import time
from tqdm import tqdm
import wandb

import ssm.aitchison as aitch
from ssm.data import TokenToProbsProcessor
from ssm.utils import (
    psplit,
    t_to_alpha_sigma,
    tokens_to_probs
)
from ssm.model import TransformerConfig, TransformerDiffusion


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
        .repartition(num_blocks=1000)
        .repeat(config.data.epochs)
        .window(blocks_per_window=2)
    )
    return data


@functools.partial(jax.pmap, axis_name='i')
def apply_model(state, texts, key):

    def loss_fn(params, key):
        keys = jrand.split(key, 2)
        batch_dim, seq_len, simplex_dim = texts.shape
        t = jrand.uniform(keys[0], texts.shape[:1])
        alphas, sigmas = t_to_alpha_sigma(t)
        # The noise has to be generated on R^{d - 1}, then transformed to be on the simplex
        real_noise = jrand.normal(keys[1], (batch_dim, seq_len, simplex_dim - 1))
        simplex_noise = aitch.ilr_inv(real_noise)
        # Noise has to be added on the simplex, using special Aitchison addition/multiplication operators
        noised_texts = aitch.add(
             aitch.mul(texts, alphas[:, None]),
             aitch.mul(simplex_noise, sigmas[:, None])
        )
        text_targets = aitch.add(
            aitch.mul(simplex_noise, alphas[:, None]),
            aitch.mul(aitch.mul(texts, sigmas[:, None]), -1.)
        )
        # Normalize so that the (Euclidean) variance of the data == 1
        # Just sends the data to a _different_ simplex, which is fine
        noised_texts = noised_texts / jnp.var(noised_texts, axis=-1)
        text_targets = text_targets / jnp.var(text_targets, axis=-1)
        v = TransformerDiffusion().apply({'params': params}, noised_texts, t)
        return jnp.mean(aitch.dist(v, text_targets))

    outputs = jax.value_and_grad(loss_fn)(state.params, key, texts)
    loss, grads = jax.lax.pmean(outputs, axis_name='i')        
    return loss, grads


@jax.pmap
def update_model(state, grads):
    return state.apply_gradients(grads=grads)
    

@functools.partial(jax.pmap, static_broadcasted_argnums=(1,))
def create_train_state(rng, config):
    transformer_config = TransformerConfig(
        config.tokenizer.vocab_size,
        config.model.embed_dim,
        config.model.model_dim,
        config.model.mlp_dim,
        config.model.num_layers,
        config.model.time_dim,
        config.model.num_heads,
        config.data.seq_len,
        config.model.dropout,
        config.model.attention_dropout
    )
    opt = optax.chain(
        optax.adamw(config.optimizer.lr),    
        optax.clip(config.optimizer.grad_clip)
    )
    model = TransformerDiffusion(transformer_config)
    params = model.init(
        rng,
        jnp.ones([1, config.data.seq_len, config.tokenizer.vocab_size]),
        jnp.ones((1,))    
    )['params']
    return TrainState.create(apply_fn=model.apply, params=params, tx=opt)


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
    np_seeds = np.random.SeedSequence(args.seed)

    preproc_pool = ray.util.ActorPool(
        [TokenToProbsProcessor.remote(s, config) for s in np_seeds.spawn(2)]
    )
    
    key = jax.random.PRNGKey(args.seed)
    
    key, *subkeys = jax.random.split(key, num_local_devices + 1)
    epoch = 0
    state = create_train_state(jnp.stack(subkeys), config)
    if args.resume:
        state = checkpoints.restore_checkpoint(args.checkpoint_dir, state)

    key = jax.random.split(key, num_processes)[local_rank]
    
    def train_epoch(state, data, num_steps: int, key):
        data_iterator = preproc_pool.map_unordered(data.iter_batches(config.batch_size))
        epoch_start = time.time()
        epoch_losses = []
        i = num_steps
        for batch in tqdm(data_iterator):
            batch_start = time.time()
            key, curr_key, *local_keys = jax.random.split(key, 2 + num_local_devices)
            batch = tokens_to_probs(
                curr_key,
                jnp.asarray(batch),
                config.concentration,
                config.vocab_size
            )
            texts = jax.tree_map(lambda x: psplit(x, num_local_devices), batch)
            print('Doing forward and backward passes')
            grads, loss = apply_model(state, texts, local_keys)
            state = update_model(state, grads)
            batch_end = time.time()
            single_loss = jax_utils.unreplicate(loss)
            # params_ema, ema_state = p_ema_update(None, ema_state, None, params)
            epoch_losses.append(single_loss)
            batch_log = {'train/loss': single_loss, 'train/time': batch_end - batch_start}
            if jax.process_index() == 0:
                wandb.log(batch_log)
            del batch_log
            i += 1
            if i % 50 == 0 and jax.process_index() == 0:
                tqdm.write(f'Batch {i}, loss {single_loss:g}')
                checkpoints.save_checkpoint(args.checkpoint_dir, state, i)
        epoch_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start
        if jax.process_index() == 0:
            wandb.log({'epoch/loss': epoch_loss, 'epoch/time': epoch_time})
        return state, i

    try:
        num_steps = 0
        for epoch, epoch_data in tqdm(train_data.iter_epochs()):
            tqdm.write(f'Epoch {epoch}')
            key, *subkeys = jax.random.split(key, 3)
            state, num_steps = train_epoch(state, epoch_data, num_steps, subkeys[0])
            if jax.process_index() == 0:
                checkpoints.save_checkpoint(args.checkpoint_dir, state, epoch, prefix='epoch_')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    args, _ = p.parse_known_args()
    main(args)
