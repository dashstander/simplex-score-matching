
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from functools import partial
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm, trange


parser = ArgumentParser()
parser.add_argument('--tokenizer-fp', type=str, default='tokenizer/tokenizer-wiki8192.json')
parser.add_argument('--seq-len', type=int, default=512)
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--chunk-size', type=int, default=1000)



def tokenize_and_split(tokenizer_fp, seq_len, batch):
    tokenizer = Tokenizer.from_file(tokenizer_fp)
    texts = '[SEP]'.join(batch['text'])
    cls_token = tokenizer.token_to_id('[CLS]')
    pad_token = tokenizer.token_to_id('[PAD]')
    token_ids = np.array(tokenizer.encode(texts).ids, dtype=np.uint16)
    leftover = token_ids.shape[0] % (seq_len - 1)
    pads = np.full(((seq_len - 1) - leftover,), pad_token, dtype=np.uint16)
    split_tokens = np.append(token_ids, pads).reshape((-1, seq_len - 1))
    num_splits = split_tokens.shape[0]
    extra_cls_tokens = np.full((num_splits, 1), cls_token, dtype=np.uint16)
    fully_tokenized = np.hstack([extra_cls_tokens, split_tokens])
    return fully_tokenized


def save_chunk(tokenize_fn, path, index , data):
    token_mat = tokenize_fn(data)
    num_rows = token_mat.shape[0]
    fp = path / f"chunk_{index}.npy"
    np.save(fp, token_mat)
    return num_rows


def main(args):
    data = load_dataset("wikipedia", "20220301.en")[args.split]
    fp = Path('/media/dashille/data0/wiki') / args.split
    tokenize_fn = partial(tokenize_and_split, args.tokenizer_fp, args.seq_len)
    save_chunk_fn = partial(save_chunk, tokenize_fn, fp)
    num_rows = data.num_rows
    
    futures = []
    chunk_size = args.chunk_size
    with ThreadPoolExecutor(max_workers=32) as executor:
        for chunk_no, i in enumerate(trange(0, num_rows, chunk_size)):
            j = min(i + chunk_size, num_rows)
            chunk = data[i:j]
            futures.append(executor.submit(save_chunk_fn, chunk_no, chunk))
    total_rows = 0
    for i, fut in tqdm(enumerate(as_completed(futures))):
        try:
            rows = fut.result()
        except Exception:
            print(f'Chunk {i} failed')
            continue
        total_rows += rows
        if i // 50 == 0:
            print(f'Saved {total_rows} records')
    

if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    main(args)
