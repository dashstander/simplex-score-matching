import argparse
from functools import partial
from flax import jax_utils
from flax.training import checkpoints
from flax.training.train_state import TrainState
from more_itertools import chunked
import jax
import jax.numpy as jnp
import jax.random as jrand
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
    t_to_alpha_sigma,
    tree_bytes,
    tree_size
)
from ssm.model import normalize_probabilities, TransformerConfig, TransformerDiffusion


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


@partial(jax.pmap, static_broadcasted_argnums=(3,), axis_name='batch')
def apply_model(state, texts, key, ilr_inv_fn):
    vmul = jax.vmap(aitch.mul)
    def loss_fn(params, x, key):
        keys = jrand.split(key, 3)
        batch_dim, seq_len, simplex_dim = x.shape
        t = jrand.uniform(keys[0], (x.shape[0],))
        alphas, sigmas = t_to_alpha_sigma(t)
        # The noise has to be generated on R^{d - 1}, then transformed to be on the simplex
        real_noise = jrand.normal(keys[1], (batch_dim, simplex_dim - 1, seq_len))
        # And once it's on the simplex it has to be transformed by the inverse metric tensor (g^{-1}),
        # which is given by g_inv
        #simplex_noise = aitch.simplex_metric_tensor_inv(
        #    x,
        #    ilr_inv_fn(real_noise).swapaxes(1, 2)
        #)
        simplex_noise = ilr_inv_fn(real_noise).swapaxes(1, 2)
        # Noise has to be added on the simplex, using special Aitchison addition/multiplication operators
        noised_x = aitch.add(
             vmul(x, alphas),
             vmul(simplex_noise, sigmas)
        )
        simplex_targets = aitch.add(
            vmul(simplex_noise, alphas),
            aitch.inverse(vmul(x, sigmas))
        )
        # Train the model to predict in centered log ratio coordinates (essentially logits)
        # Easy softmax transform away from sampling
        targets = normalize_probabilities(simplex_targets) 
        v = state.apply_fn({'params': params}, noised_x, t, rngs={'dropout': keys[3]})
        return jnp.mean((v - targets)**2)

    loss_grads = jax.value_and_grad(loss_fn)(state.params, texts, key)
    loss, grads = jax.lax.pmean(loss_grads, axis_name='batch')
    return loss, grads


@partial(jax.pmap, axis_name='batch')
def update_model(state, grads):
    return state.apply_gradients(grads=grads)
    

def create_train_state(rng, config):
    init_rng, dropout_rng = jrand.split(rng)
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
        {'params': init_rng, 'dropout': dropout_rng},
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

    print(f'Beginning training with {num_local_devices} devices.')

    train_data = get_dataset(config)
    np_seeds = np.random.SeedSequence(args.seed)

    preproc_pool = ray.util.ActorPool(
        [TokenToProbsProcessor.remote(s, config, train_data) for s in np_seeds.spawn(2)]
    )

    _, ilr_inv = aitch.make_isometric_transforms(config.tokenizer.vocab_size)
    
    key = jrand.PRNGKey(args.seed)
    
    #key, *subkeys = jax.random.split(key, num_local_devices + 1)
    key, state_key = jrand.split(key)
    epoch = 0
    train_state = create_train_state(state_key, config)
    if args.resume:
        train_state = checkpoints.restore_checkpoint(args.checkpoint_dir, train_state)

    key = jrand.split(key, num_processes)[local_rank]

    dataset_size = train_data.shape[0]

    if local_rank == 0:
        print(f'Initialized model with {tree_size(train_state.params)} parameters, taking up {tree_bytes(train_state.params)/1e9}GB')
    
    train_state = jax_utils.replicate(train_state)
    #print(train_state.params['Dense_0']['kernel'].shape)
    
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
            key, *local_keys = jax.random.split(key, 1 + num_local_devices)
            texts = psplit(batch, num_local_devices)
            #print(texts.shape)
            # print('Doing forward and backward passes')
            loss, grads = apply_model(state, texts, jnp.stack(local_keys), ilr_inv)
            #loss, grads = jax.lax.pmean(loss_grads, axis_name='batch')
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
        for epoch in trange(config.data.epochs):
            tqdm.write(f'Epoch {epoch}')
            key, index_key, *subkeys = jrand.split(key, 3)
            permuted_indices = jrand.permutation(index_key, dataset_size).tolist()
            train_state, num_steps = train_epoch(train_state, permuted_indices, num_steps, subkeys[0])
            if jax.process_index() == 0:
                checkpoints.save_checkpoint(args.checkpoint_dir, jax_utils.unreplicate(train_state), epoch, prefix='epoch_')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    args, _ = p.parse_known_args()
    main(args)
