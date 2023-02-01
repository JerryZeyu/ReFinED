import os
import pickle
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from typing import Dict

import ujson
def pickle_load_large_file(filepath):
    max_bytes = 2**31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj
def load_umlsID_to_idx(filename: str, is_test: bool = False) -> Dict[str, int]:
    umlsID_to_idx: Dict[str, int] = dict()
    line_num = 0
    with open(filename, "r") as f:
        for line in tqdm(f, total=6000000, desc="Loading qcode_to_idx"):
            line = ujson.loads(line)
            umlsID = line["umlsID"]
            idx = line["index"]
            umlsID_to_idx[umlsID] = idx
            line_num += 1
            if is_test and line_num > 1000:
                break
    return umlsID_to_idx
# TODO FIX THIS SO IT USES CORRECT QCODE_TO_IDX
def create_description_tensor(output_path: str, umlsID_to_idx_filename: str, desc_filename: str, label_filename: str,
                              tokeniser: str = 'roberta-base', is_test: bool = False,
                              include_no_desc: bool = True, keep_all_entities: bool = False):
    labels = pickle_load_large_file(label_filename)
    umlsIDs = list(labels.keys())
    descriptions = pickle_load_large_file(desc_filename)
    umlsID_to_idx = load_umlsID_to_idx(umlsID_to_idx_filename)


    # TODO: check no extra [SEP] tokens between label and description or extra [CLS] or [SEP] at end
    tokenizer = AutoTokenizer.from_pretrained(tokeniser, use_fast=True, add_prefix_space=False)
    descriptions_tns = torch.zeros((len(umlsID_to_idx) + 2, 32), dtype=torch.int32)
    descriptions_tns.fill_(tokenizer.pad_token_id)

    umlsID_has_label = 0
    umlsID_has_desc = 0
    i = 0
    for umlsID, idx in tqdm(umlsID_to_idx.items()):
        if umlsID in labels:
            umlsID_has_label += 1
            label = labels[umlsID]
            if umlsID in descriptions and descriptions[umlsID] is not None:
                umlsID_has_desc += 1
                desc = descriptions[umlsID]
            else:
                if not include_no_desc:
                    continue
                desc = 'no description'

            sentence = (label, desc)
            tokenised = tokenizer.encode_plus(sentence, truncation=True, max_length=32, padding='max_length',
                                              return_tensors='pt')['input_ids']
            descriptions_tns[idx] = tokenised
        i += 1
        if i % 250000 == 0:
            print(f'QCodes processed {i}, Qcodes with label: {umlsID_has_label}, '
                  f'Qcodes with label and description: {umlsID_has_desc}')

    torch.save(descriptions_tns, os.path.join(output_path, 'descriptions_tns.pt'))
OUTPUT_PATH = "data/Corpus_bionorm/UMLS"
create_description_tensor(output_path=OUTPUT_PATH,
                        umlsID_to_idx_filename=os.path.join(OUTPUT_PATH, 'umlsID_to_idx.json'),
                        desc_filename=os.path.join(OUTPUT_PATH, 'umlsID2desc.pkl'),
                        label_filename=os.path.join(OUTPUT_PATH, 'umlsID2title.pkl'))