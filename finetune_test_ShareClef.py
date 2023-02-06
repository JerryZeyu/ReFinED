import os
import sys
import json
import pickle
from collections import OrderedDict
from refined.inference.processor import Refined

def pickle_dump_large_file(obj, filepath):
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])

def pickle_load_large_file(filepath):
    max_bytes = 2**31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj
refined = Refined.from_pretrained(model_name='fine_tuned_models/addUMLS_pretrain_test/f1_0.5210/',
                                          entity_set="wikidata",
                                          use_precomputed_descriptions=True,
                                          data_dir = "data/organised_data_dir",
                                          download_files = False)
docID2context = pickle_load_large_file("Corpus_bionorm/ShareClef/test_docID2context_all.pkl")
docID2results = OrderedDict()
for docID in docID2context.keys():
    text = docID2context[docID]
    spans = refined.process_text(text)
    print(spans)
    docID2results[docID] = spans

with open("Corpus_bionorm/ShareClef/test_docID2results_wikidata_addUMLS_ShareClef_finetune.pickle", "wb") as f_w:
    pickle.dump(docID2results, f_w)
