import os
from typing import List

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from refined.data_types.doc_types import Doc
from refined.dataset_reading.entity_linking.wikipedia_dataset import WikipediaDataset
from refined.doc_preprocessing.preprocessor import PreprocessorInferenceOnly_UMLS
from refined.doc_preprocessing.wikidata_mapper import WikidataMapper
from refined.inference.processor import Refined_UMLS
from refined.model_components.config import NER_TAG_TO_IX_UMLS, ModelConfig
from refined.model_components.refined_model import RefinedModel_UMLS
from refined.resource_management.aws import S3Manager
from refined.resource_management.resource_manager import ResourceManager
from refined.evaluation.evaluation import get_datasets_obj, evaluate
from refined.dataset_reading.entity_linking.document_dataset import DocDataset_UMLS
from refined.dataset_reading.entity_linking.dataset_factory import Datasets_BioNorm
from refined.torch_overrides.data_parallel_refined import DataParallelReFinED
from refined.training.fine_tune.fine_tune_UMLS import run_fine_tuning_loops
from refined.training.train.training_args import parse_training_args
from refined.utilities.general_utils import get_logger

LOG = get_logger(name=__name__)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # DDP (ensure batch_elements_included is used)

    training_args = parse_training_args()

    resource_manager = ResourceManager(S3Manager(),
                                       data_dir="data",
                                       entity_set="umls",
                                       model_name=None
                                       )
    preprocessor = PreprocessorInferenceOnly_UMLS(
        data_dir="data",
        debug=training_args.debug,
        max_candidates=training_args.num_candidates_train,
        transformer_name=training_args.transformer_name,
        ner_tag_to_ix=NER_TAG_TO_IX_UMLS,  # for now include default ner_to_tag_ix can make configurable in future
        entity_set="umls",
        use_precomputed_description_embeddings=False
    )
    datasets = Datasets_BioNorm(preprocessor=preprocessor, resource_manager=resource_manager)
    evaluation_dataset_name_to_docs = {
        "ShareClef": list(datasets.get_custom_dataset_name_docs_ShareClef(
            split="dev",
            include_gold_label=True,
            filter_not_in_kb=True,
            include_spans=True,
        ))
    }
    count = 0
    count_num = 0
    evaluation_doc = evaluation_dataset_name_to_docs["ShareClef"]
    for doc in evaluation_doc:
        for span in doc.spans:
            count+=1
            candidate_entities = [item[0] for item in span.candidate_entities]
            gold_entity = span.gold_entity.umls_entity_id
            print(candidate_entities)
            print(gold_entity)
            if gold_entity in candidate_entities:
                count_num += 1
    print(count_num/count)
    print(count_num)
    print(count)


if __name__ == "__main__":
    main()