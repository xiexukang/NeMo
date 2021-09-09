# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import pickle
import random
import tarfile
from typing import List, Tuple

from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import AutoTokenizer

import nemo.collections.nlp.data.text_normalization.constants as constants
from nemo.collections.nlp.data.text_normalization.utils import process_url


def preprocess_file(input_file: str) -> List[Tuple[List[str]]]:
    """
    Performs initial preprocessing, i.e., urls formatting, removal of "_trans" from Ru set

    Args:
        input_file: path to a file in google TN format

    Returns:
        Processed data. Each element is a Tuple(List[semiotic classes], List[written words], List[spoken words])
    """
    print(f"Reading and running initial pre-processing of {input_file}...")
    cur_split = []
    with open(input_file, 'r', encoding='utf-8') as f:
        # Loop through each line of the file
        cur_classes, cur_tokens, cur_outputs = [], [], []
        for linectx, line in tqdm(enumerate(f)):
            es = line.strip().split('\t')
            if len(es) == 2 and es[0] == '<eos>':
                # Update cur_split
                cur_outputs = process_url(cur_tokens, cur_outputs, lang)
                cur_split.append((cur_classes, cur_tokens, cur_outputs))
                # Reset
                cur_classes, cur_tokens, cur_outputs = [], [], []
                continue
            # Remove _trans (for Russian)
            if lang == constants.RUSSIAN:
                es[2] = es[2].replace('_trans', '')
            # Update the current example
            assert len(es) == 3
            cur_classes.append(es[0])
            cur_tokens.append(es[1])
            cur_outputs.append(es[2])
    return cur_split


def create_shard(data, out_dir, shard_id):
    """
        Creates a tarball containing pickled entries from the data.
    """
    tar_file_path = os.path.join(out_dir, f'{shard_id}.tar')
    tar = tarfile.open(tar_file_path, mode='w', dereference=True)

    for idx, entry in enumerate(data):
        entry = {k: v[0] for k, v in entry.items()}
        pickle_file = os.path.join(out_dir, f'entry-{idx:5d}.pkl')
        pickle.dump(entry[0], open(pickle_file, 'wb'))
        tar.add(pickle_file)
        os.remove(pickle_file)
    tar.close()

    return tar_file_path


def write_input_file_entries_to_tarfiles(
    input_file: str,
    tokenizer: AutoTokenizer,
    tokenizer_name: str,
    mode: str,
    max_seq_len: int,
    decoder_data_augmentation: bool = False,
    do_basic_tokenize: bool = False,
    max_insts: int = -1,
):
    """
    Writes current fragment of the overall parallel corpus to tarfiles by:
    (1) Creating a minibatches using a TranslationDataset object.
    (2) Writing each minibatch to a pickle file.
    (3) Adding pickle files to a tarfile until it reaches num_batches_per_tarfile.
    """

    dataset = TextNormalizationDecoderDataset(
        input_file=input_file,
        raw_instances=preprocess_file(input_file),
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        mode=mode,
        max_len=max_seq_len,
        decoder_data_augmentation=decoder_data_augmentation,
        lang=lang,
        do_basic_tokenize=do_basic_tokenize,
        use_cache=False,
        max_insts=max_insts,
    )

    shuffle = True
    shuffle_seed = 2020
    num_shards = 2

    ids = list(range(len(dataset)))
    if shuffle:
        random.seed(shuffle_seed)
        print("Shuffling...")
        random.shuffle(ids)

    # Create shards
    start_indices = []
    end_indices = []
    shard_ids = []
    # Build indices
    for i in range(num_shards):
        shard_id = f"{os.path.basename(input_file)}--{i:04}"
        start_idx = (len(ids) // num_shards) * i
        end_idx = start_idx + (len(ids) // num_shards)
        print(f"Shard {shard_id} includes examples: [{start_idx} ~ {end_idx})")
        shard_ids.append(shard_id)
        if i == num_shards - 1 and (len(ids) - end_idx) > 0:
            # We discard in order to have the same number of entries per shard.
            print(f"{len(ids) - end_idx} example(s) will be discarded.")

        start_indices.append(start_idx)
        end_indices.append(end_idx)

    remainder = len(ids) % num_shards
    num_samples = len(ids) - remainder
    print(f"Number of samples added: {num_samples} out of {len(ids)} from {input_file}.")

    tar_file_paths = [
        create_shard(data=dataset.examples[start_idx:end_idx], out_dir=out_dir, shard_id=shard_ids[i])
        for i, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices))
    ]

    return tar_file_paths, num_samples


"""
python create_tarred_tn_dataset.py \
--fname="/mnt/sdb/DATA/normalization/google_data/DEL/output-00099-of-00100" \
--out_dir="/home/ebakhturina/NeMo/examples/nlp/duplex_text_normalization/tarred"
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NMT dataset pre-processing')
    parser.add_argument(
        '--tokenizer_model', type=str, default='yttm', help='Supports yttm, sentencepiece and HuggingFace tokenizers',
    )
    parser.add_argument('--transformer_name', type=str, default=None, help='Path to tokenizer model')
    parser.add_argument('--pkl_file_prefix', type=str, default='parallel', help='Prefix for tar and pickle files')
    parser.add_argument('--fname', type=str, required=True, help='Path to monolingual data file')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to store dataloader and tokenizer models')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Max Sequence Length')
    parser.add_argument('--min_seq_length', type=int, default=1, help='Min Sequence Length')

    args = parser.parse_args()

    from nemo.collections.nlp.data.text_normalization.decoder_dataset import TextNormalizationDecoderDataset
    from glob import glob

    input_file = args.fname

    transformer_name = 'albert-base-v2'
    mode = "tn"
    max_seq_length = 512
    min_seq_length = 1
    decoder_data_augmentation = False
    lang = "en"
    do_basic_tokenize = False
    max_insts = -1
    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = args.out_dir

    # check if exists do not proceed and re-use
    args.tokens_in_batch = 64
    tokens_in_batch = args.tokens_in_batch
    args.num_batches_per_tarfile = 20
    num_batches_per_tarfile = args.num_batches_per_tarfile
    n_jobs = -2
    fragment_index = 0
    tar_file_prefix = "tn"
    world_size = 1

    tokenizer = AutoTokenizer.from_pretrained(transformer_name)

    # add parallel with glob for all files

    # TODO provide a list of semiotic classes in the config
    # TODO shuffle data

    max_insts = 100

    # result = Parallel(n_jobs=n_jobs)(
    #     delayed(write_input_file_entries_to_tarfiles)(
    #         input_file=input_file,
    #         tokenizer=tokenizer,
    #         tokenizer_name=transformer_name,
    #         mode=mode,
    #         max_seq_len=max_seq_length,
    #         decoder_data_augmentation=decoder_data_augmentation,
    #         do_basic_tokenize=do_basic_tokenize,
    #         max_insts=max_insts,
    #     )
    #     for input_file in [
    #         "/mnt/sdb/DATA/normalization/google_data/DEL/output-00099-of-00100",
    #         "/mnt/sdb/DATA/normalization/google_data/DEL/output-00098-of-00100",
    #     ]
    # )
    #
    # # flatten out the list of the created tar files
    # tar_files_created = [item for sublist in result for item in sublist]
    # num_samples = sum([sublist[1] for sublist in result])

    input_file = "/mnt/sdb/DATA/normalization/google_data/DEL/output-00099-of-00100"
    results_list = write_input_file_entries_to_tarfiles(
        input_file=input_file,
        tokenizer=tokenizer,
        tokenizer_name=transformer_name,
        mode=mode,
        max_seq_len=max_seq_length,
        decoder_data_augmentation=decoder_data_augmentation,
        do_basic_tokenize=do_basic_tokenize,
        max_insts=max_insts,
    )

    # dump metadata to json
    metadata = {}

    tar_file_paths = glob(f'{out_dir}/*.tar')
    assert len(tar_file_paths) == len(tar_files_created)
    metadata["tar_files"] = tar_file_paths
    metadata["num_samples"] = num_samples
    metadata_path = os.path.join(out_dir, 'metadata.json')
    json.dump(metadata, open(metadata_path, 'w'))

    num_tar_files = len(tar_file_paths)
    if num_tar_files < world_size:
        raise ValueError(
            (
                f'Number of tar files found: {num_tar_files} is less than world size: {world_size}. '
                f'There should be at least one tar file per GPU (ideally many tar files per GPU). '
                f'This may be due to dataset size. '
                f'Decrease num_batches_per_tarfile or num_tokens_per_batch to increase the number of tarfiles. '
                f'Also using shard_strategy=replicate will use all available tarfiles for every GPU. '
            )
        )
