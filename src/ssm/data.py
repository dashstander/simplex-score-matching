from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import jax
import jax.numpy as jnp
from ray.data.datasource import DefaultFileMetadataProvider
from ray.data.read_api import read_datasource
from torch.utils.data import Dataset, get_worker_info




def sample(logits, tokenizer, key):
    token_ids = jax.random.categorical(key, logits)
    return tokenizer.decode(token_ids)



class TokenProbsDataset(Dataset):

    def __init__(self, config, data):
        self.rng = np.random.default_rng(get_worker_info().seed)
        self.min_conc = config.data.min_init_prob_concentration
        self.max_conc = config.data.max_init_prob_concentration
        self.vocab_size = config.tokenizer.vocab_size
        self.data = data
        self.length = None
    
    def __len__(self):
        if self.length is None:
            self.length = self.data.shape[0]
        return self.length
    
    def __getitem__(self, idx):
        tokens = self.data[idx, :]
        seq_len = tokens.shape[0]

        def _tokens_to_probs(token_ids):
            concentration = self.rng.uniform(self.min_conc, self.max_conc)
            x = self.rng.random((seq_len, self.vocab_size)) / self.vocab_size
            # At this point E(x.sum()) == 0.5 
            # What we want is for new_val / (x.sum() + new_val) ~ concentration
            # --> new_val == (concentration * x.sum())/(1 - concentration)
            # Then, in the normalized vector, the appropriate token will have ~ concentration weight,
            # and the others will have the rest
            x_sum = x.sum(axis=1)
            conc_val = np.mean((concentration * x_sum) / (1 - concentration))
            np.put_along_axis(x, token_ids[:, None], conc_val, axis=1)
            return x / x.sum(axis=1)[:, None]

        return np.apply_along_axis(_tokens_to_probs, axis=0, arr=tokens)
        






def read_numpy_gcs(
    paths,
    *,
    filesystem = None,
    parallelism: int = 200,
    arrow_open_stream_args = None,
    meta_provider = DefaultFileMetadataProvider(),
    partition_filter = None,
    **numpy_load_args,
):
    return read_datasource(
        NumpyDatasource(),
        parallelism=parallelism,
        paths=paths,
        filesystem=filesystem,
        open_stream_args=arrow_open_stream_args,
        meta_provider=meta_provider,
        partition_filter=partition_filter,
        **numpy_load_args,
    )


def tokenize_and_split(tokenizer, seq_len, batch):
    texts = '[SEP]'.join(batch['text'])
    cls_token = tokenizer.token_to_id('[CLS]')
    pad_token = tokenizer.token_to_id('[PAD]')
    token_ids = np.array(tokenizer.encode(texts).ids)
    leftover = token_ids.shape[0] % (seq_len - 1)
    pads = np.full(((seq_len - 1) - leftover,), pad_token)
    split_tokens = np.append(token_ids, pads).reshape((-1, seq_len - 1))
    num_splits = split_tokens.shape[0]
    extra_cls_tokens = np.full((num_splits, 1), cls_token)
    fully_tokenized = np.hstack([extra_cls_tokens, split_tokens])
    return fully_tokenized


def load_tokens(fp):
    data = np.load(fp)
    return data


def dataloader(path, batch_size, random_seed):
    files = list(path.glob('*.npz'))
    npgen = np.random.default_rng(random_seed)
    with ThreadPoolExecutor(max_workers=8) as executor:
        data_chunks = executor.map(load_tokens, files)
        for future in as_completed(data_chunks):
            data = future.result()
            data = npgen.shuffle(data['tokens'])
            splits = list(range(batch_size, data.shape[0], batch_size))
            for batch in np.split(data, splits):
                if batch.shape[0] == batch_size:
                    yield jnp.asarray(batch)

