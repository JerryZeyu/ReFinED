import os
from pprint import pprint
from typing import Iterable, Optional, Dict

from refined.dataset_reading.entity_linking.dataset_factory import Datasets_BioNorm
from refined.data_types.doc_types import Doc_UMLS
from refined.doc_preprocessing.preprocessor import Preprocessor
from refined.resource_management.resource_manager import ResourceManager
from refined.evaluation.metrics import Metrics
from refined.resource_management.aws import S3Manager
from tqdm.auto import tqdm
from refined.inference.processor import Refined_UMLS
from refined.utilities.general_utils import get_logger

LOG = get_logger(__name__)


def process_annotated_document(
        refined: Refined_UMLS,
        doc: Doc_UMLS,
        el: bool = False,
        ed_threshold: float = 0.0,
        force_prediction: bool = False,
) -> Metrics:
    if force_prediction:
        assert ed_threshold == 0.0, "ed_threshold must be set to 0 to force predictions"
    gold_spans = set()
    gold_spans_list = []
    gold_entity_in_cands = 0
    for span in doc.spans:
        if (span.gold_entity is None or span.gold_entity.umls_entity_id is None
            # only include entity spans that have been annotated as an entity in a KB
                or span.gold_entity.umls_entity_id == "C0"):
            continue
        gold_spans_list.append((span.text, span.start, span.gold_entity.umls_entity_id))
        gold_spans.add((span.text, span.start, span.gold_entity.umls_entity_id))
        if span.gold_entity.umls_entity_id in {umlsID for umlsID, _ in span.candidate_entities}:
            gold_entity_in_cands += 1
    #print("gold_entity_in_cands: ", gold_entity_in_cands)
    #print("el: ", el)
    if len(gold_spans_list) != len(gold_spans):
        print(gold_spans_list)
        print(sorted(gold_spans, key=lambda x: x[1]))
        print(len(gold_spans_list))
        print(len(gold_spans))
        print("*******************")
    # optionally filter NIL gold spans
    # nil_spans is a set of mention spans that are annotated as mentions in the dataset but are not linked to a KB
    # many nil_spans in public datasets should have been linked to an entity but due to the annotation creation
    # method many entity were missed. Furthermore, when some datasets were built the correct entity
    # did not exist in the KB at the time but do exist now. This means models are unfairly penalized for predicting
    # entities for nil_spans.

    predicted_spans = refined.process_text(
        text=doc.text,
        spans=doc.spans if not el else None  # only set to True if the dataset has special spans (e.g. dates)
    )

    pred_spans = set()
    for span in predicted_spans:
        # skip dates and numbers, only consider entities that are linked to a KB
        # pred_spans is used for linkable mentions only
        if span.coarse_type != "MENTION":
            continue
        # print("span.predicted_entity.umls_entity_id: ", span.predicted_entity.umls_entity_id)
        # print("span.entity_linking_model_confidence_score: ", span.entity_linking_model_confidence_score)
        # print(span.top_k_predicted_entities)
        # print("***************************")
        if (
                span.predicted_entity.umls_entity_id is None
                or span.entity_linking_model_confidence_score < ed_threshold
                or span.predicted_entity.umls_entity_id == 'C-1'
        ):
            umlsID = "C0"
        else:
            umlsID = span.predicted_entity.umls_entity_id
        if force_prediction and umlsID == "C0":
            if len(span.top_k_predicted_entities) >= 2:
                umlsID = span.top_k_predicted_entities[1][0].umls_entity_id
        #print("span text: ", span.text)
        #print("span start: ", span.start)
        #print("span umlsID: ", umlsID)
        pred_spans.add((span.text, span.start, umlsID))

    pred_spans = {(text, start, umlsID) for text, start, umlsID in pred_spans if umlsID != "C0"}

    num_gold_spans = len(gold_spans)
    tp = len(pred_spans & gold_spans)
    fp = len(pred_spans - gold_spans)
    fn = len(gold_spans - pred_spans)

    # ignore which entity is linked to (consider just the mention detection (NER) prediction)
    pred_spans_md = {(span.text, span.start, span.coarse_type) for span in predicted_spans}
    gold_spans_md = {(span.text, span.start, span.coarse_type) for span in doc.spans if span.coarse_type == "MENTION"}
    tp_md = len(pred_spans_md & gold_spans_md)
    fp_md = len(pred_spans_md - gold_spans_md)
    fn_md = len(gold_spans_md - pred_spans_md)

    fp_errors = sorted(list(pred_spans - gold_spans), key=lambda x: x[1])[:5]
    fn_errors = sorted(list(gold_spans - pred_spans), key=lambda x: x[1])[:5]

    fp_errors_md = sorted(list(pred_spans_md - gold_spans_md), key=lambda x: x[1])[:5]
    fn_errors_md = sorted(list(gold_spans_md - pred_spans_md), key=lambda x: x[1])[:5]
    metrics = Metrics(
        el=el,
        num_gold_spans=num_gold_spans,
        tp=tp,
        fp=fp,
        fn=fn,
        tp_md=tp_md,
        fp_md=fp_md,
        fn_md=fn_md,
        gold_entity_in_cand=gold_entity_in_cands,
        num_docs=1,
        example_errors=[{'doc_title': doc.text[:20], 'fp_errors': fp_errors, 'fn_errors': fn_errors}],
        example_errors_md=[{'doc_title': doc.text[:20], 'fp_errors_md': fp_errors_md, 'fn_errors_md': fn_errors_md}]
    )
    return metrics


def evaluate_on_docs(
        refined,
        docs: Iterable[Doc_UMLS],
        progress_bar: bool = True,
        dataset_name: str = "dataset",
        ed_threshold: float = 0.0,
        el: bool = False,
        sample_size: Optional[int] = None,
):
    overall_metrics = Metrics.zeros(el=el)
    for doc_idx, doc in tqdm(
            enumerate(list(docs)), disable=not progress_bar, desc=f"Evaluating on {dataset_name}"
    ):
        doc_metrics = process_annotated_document(
            refined=refined,
            doc=doc,
            force_prediction=False,
            ed_threshold=ed_threshold,
            el=el
        )
        overall_metrics += doc_metrics
        if sample_size is not None and doc_idx > sample_size:
            break
    return overall_metrics


def eval_all(
        refined,
        data_dir: Optional[str] = None,
        datasets_dir: Optional[str] = None,
        additional_data_dir: Optional[str] = None,
        include_spans: bool = True,
        filter_not_in_kb: bool = True,
        ed_threshold: float = 0.15,
        el: bool = False,
        download: bool = True,
):
    datasets = get_datasets_obj(preprocessor=refined.preprocessor,
                                data_dir=data_dir,
                                datasets_dir=datasets_dir,
                                additional_data_dir=additional_data_dir,
                                download=download)
    dataset_name_to_docs = get_standard_datasets(datasets, el, filter_not_in_kb, include_spans)
    return evaluate_on_datasets(refined=refined,
                                dataset_name_to_docs=dataset_name_to_docs,
                                el=el,
                                ed_threshold=ed_threshold,
                                )


def get_standard_datasets(datasets: Datasets_BioNorm,
                          el: bool,
                          filter_not_in_kb: bool = True,
                          include_spans: bool = True) -> Dict[str, Iterable[Doc_UMLS]]:
    if not el:
        dataset_name_to_docs = {
            "ShareClef": datasets.get_custom_dataset_name_docs_ShareClef(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
        }
    else:
        dataset_name_to_docs = {
            "ShareClef": datasets.get_custom_dataset_name_docs_ShareClef(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
        }
    return dataset_name_to_docs


def evaluate_on_datasets(refined: Refined_UMLS,
                         dataset_name_to_docs: Dict[str, Iterable[Doc_UMLS]],
                         el: bool,
                         ed_threshold: float = 0.15,
                         ):
    dataset_name_to_metrics = dict()
    for dataset_name, dataset_docs in dataset_name_to_docs.items():
        metrics = evaluate_on_docs(
            refined=refined,
            docs=dataset_docs,
            dataset_name=dataset_name,
            ed_threshold=ed_threshold,
            el=el,
        )
        dataset_name_to_metrics[dataset_name] = metrics
        print("*****************************\n\n")
        print(f"Dataset name: {dataset_name}")
        print(metrics.get_summary())
        print("*****************************\n\n")
    return dataset_name_to_metrics


def get_datasets_obj(preprocessor: Preprocessor,
                     download: bool = True,
                     data_dir: Optional[str] = None,
                     datasets_dir: Optional[str] = None,
                     additional_data_dir: Optional[str] = None,
                     ) -> Datasets_BioNorm:
    if data_dir is None:
        data_dir = os.path.join(os.path.expanduser('~'), '.cache', 'refined')
    if datasets_dir is None:
        datasets_dir = os.path.join(data_dir, 'datasets')
    if additional_data_dir is None:
        additional_data_dir = os.path.join(data_dir, 'additional_data')

    resource_manager = ResourceManager(S3Manager(),
                                       data_dir=datasets_dir,
                                       datasets_dir=datasets_dir,
                                       additional_data_dir=additional_data_dir,
                                       entity_set=None,
                                       model_name=None
                                       )
    if download:
        resource_manager.download_datasets_if_needed()
        resource_manager.download_additional_files_if_needed()

    return Datasets_BioNorm(preprocessor=preprocessor)


def evaluate(evaluation_dataset_name_to_docs: Dict[str, Iterable[Doc_UMLS]],
             refined: Refined_UMLS,
             ed_threshold: float = 0.15,
             el: bool = True,
             ed: bool = True,
             print_errors: bool = True) -> Dict[str, Metrics]:
    dataset_name_to_metrics = dict()
    if el:
        LOG.info("Running entity linking evaluation")
        el_results = evaluate_on_datasets(
            refined=refined,
            dataset_name_to_docs=evaluation_dataset_name_to_docs,
            el=True,
            ed_threshold=ed_threshold,
        )
        for dataset_name, metrics in el_results.items():
            dataset_name_to_metrics[f"{dataset_name}-EL"] = metrics
            if print_errors:
                LOG.info("Printing EL errors")
                pprint(metrics.example_errors[:5])
                LOG.info("Printing MD errors")
                pprint(metrics.example_errors_md[:5])

    if ed:
        LOG.info("Running entity disambiguation evaluation")
        ed_results = evaluate_on_datasets(
            refined=refined,
            dataset_name_to_docs=evaluation_dataset_name_to_docs,
            el=False,
            ed_threshold=ed_threshold
        )
        for dataset_name, metrics in ed_results.items():
            dataset_name_to_metrics[f"{dataset_name}-ED"] = metrics
            if print_errors:
                LOG.info("Printing ED errors")
                pprint(metrics.example_errors[:5])

    return dataset_name_to_metrics
