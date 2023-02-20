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
                                       data_dir=training_args.data_dir,
                                       entity_set=training_args.entity_set,
                                       model_name=None
                                       )
    preprocessor = PreprocessorInferenceOnly_UMLS(
        data_dir=training_args.data_dir,
        debug=training_args.debug,
        max_candidates=training_args.num_candidates_train,
        transformer_name=training_args.transformer_name,
        ner_tag_to_ix=NER_TAG_TO_IX_UMLS,  # for now include default ner_to_tag_ix can make configurable in future
        entity_set=training_args.entity_set,
        use_precomputed_description_embeddings=False,
        return_titles=True
    )
    #print(preprocessor.umlsID_to_title)
    datasets = Datasets_BioNorm(preprocessor=preprocessor, resource_manager=resource_manager)
    training_dataset = DocDataset_UMLS(
        docs=list(datasets.get_custom_dataset_name_docs_ShareClef(split="train", include_gold_label=True)),
        preprocessor=preprocessor
    )
    training_dataloader = DataLoader(
        dataset=training_dataset, batch_size=training_args.batch_size, shuffle=True, num_workers=1,
        collate_fn=training_dataset.collate
    )
    evaluation_dataset_name_to_docs = {
        "ShareClef": list(datasets.get_custom_dataset_name_docs_ShareClef(
            split="dev",
            include_gold_label=True,
            filter_not_in_kb=True,
            include_spans=True,
        ))
    }

    model = RefinedModel_UMLS(
        ModelConfig(data_dir=preprocessor.data_dir,
                    transformer_name=preprocessor.transformer_name,
                    ner_tag_to_ix=preprocessor.ner_tag_to_ix
                    ),
        preprocessor=preprocessor
    )

    if training_args.restore_model_path is not None:
        # TODO load `ModelConfig` file (from the directory) and initialise RefinedModel from that
        # to avoid issues when model config differs
        LOG.info(f'Restored model from {training_args.restore_model_path}')
        checkpoint = torch.load(training_args.restore_model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

    if training_args.n_gpu > 1:
        model = DataParallelReFinED(model, device_ids=list(range(training_args.n_gpu)), output_device=training_args.device)
    model = model.to(training_args.device)

    # wrap a ReFinED processor around the model so evaluation methods can be run easily
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    refined = Refined_UMLS(
        model_file_or_model=model,
        model_config_file_or_model_config=model_to_save.config,
        preprocessor=preprocessor,
        device=training_args.device
    )

    # param_groups = [
    #     {"params": model_to_save.get_desc_params(), "lr": training_args.lr},
    #     {"params": model_to_save.get_ed_params(), "lr": training_args.lr * 100},
    #     {"params": model_to_save.get_parameters_not_to_scale(), "lr": training_args.lr}
    # ]
    param_groups = [
        {"params": model_to_save.get_parameters_not_to_scale(), "lr": training_args.lr}
    ]
    print("training_args.el: ", training_args.el)
    if training_args.el:
        param_groups.append({"params": model_to_save.get_md_params(), "lr": training_args.lr})

    optimizer = AdamW(param_groups, lr=training_args.lr, eps=1e-8)

    total_steps = len(training_dataloader) * training_args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.num_warmup_steps,
        num_training_steps=total_steps / training_args.gradient_accumulation_steps
    )

    scaler = GradScaler()

    if training_args.restore_model_path is not None and training_args.resume:
        LOG.info("Restoring optimizer and scheduler")
        optimizer_checkpoint = torch.load(
            os.path.join(os.path.dirname(training_args.restore_model_path), "optimizer.pt"),
            map_location="cpu",
        )
        scheduler_checkpoint = torch.load(
            os.path.join(os.path.dirname(training_args.restore_model_path), "scheduler.pt"),
            map_location="cpu",
        )
        scaler_checkpoint = torch.load(
            os.path.join(os.path.dirname(training_args.restore_model_path), "scaler.pt"),
            map_location="cpu",
        )
        optimizer.load_state_dict(optimizer_checkpoint)
        scheduler.load_state_dict(scheduler_checkpoint)
        scaler.load_state_dict(scaler_checkpoint)

    run_fine_tuning_loops(
        refined=refined,
        fine_tuning_args=training_args,
        training_dataloader=training_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluation_dataset_name_to_docs=evaluation_dataset_name_to_docs,
        checkpoint_every_n_steps=training_args.checkpoint_every_n_steps
    )


if __name__ == "__main__":
    main()
