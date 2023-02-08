import os
import sys
import json
import pickle
from collections import OrderedDict
from refined.resource_management.aws import S3Manager
from refined.inference.processor import Refined_UMLS
from refined.resource_management.resource_manager import ResourceManager
from refined.dataset_reading.entity_linking.dataset_factory import Datasets_BioNorm
from refined.evaluation.evaluation_UMLS import get_datasets_obj, evaluate

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

resource_manager = ResourceManager(S3Manager(),
                                data_dir="data",
                                entity_set="umls",
                                model_name=None
                                       )
#print(preprocessor.umlsID_to_title)
refined = Refined_UMLS.from_pretrained(model_name='fine_tuned_models/test/f1_0.4101/',
                                          entity_set="umls",
                                          use_precomputed_descriptions=True,
                                          data_dir = "data/",
                                          download_files = False)
datasets = Datasets_BioNorm(preprocessor=refined.preprocessor, resource_manager=resource_manager)
evaluation_dataset_name_to_docs = {
        "ShareClef": list(datasets.get_custom_dataset_name_docs_ShareClef(
            split="test",
            include_gold_label=True,
            filter_not_in_kb=True,
            include_spans=True,
        ))
    }
evaluation_metrics = evaluate(refined=refined,
                                  evaluation_dataset_name_to_docs=evaluation_dataset_name_to_docs,
                                  el=True,  # only evaluate EL when training EL
                                  ed=True,  # always evaluate standalone ED
                                  ed_threshold=0.0)
#docID2context = pickle_load_large_file("data/datasets/ShareClef/test_docID2context.pkl")
# docID2results = OrderedDict()
# for docID in docID2context.keys():
#     text = docID2context[docID]
#     spans = refined.process_text(text)
#     print(spans)
#     docID2results[docID] = spans
#
# with open("data/datasets/ShareClef/results/test_docID2results_ShareClef_finetune.pickle", "wb") as f_w:
#     pickle.dump(docID2results, f_w)
