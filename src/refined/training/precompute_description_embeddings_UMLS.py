import argparse

import torch.cuda

from refined.inference.processor import Refined_UMLS
from refined.utilities.general_utils import get_logger

LOG = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="File path to directory contain model.pt and config.json file of the model.",
    )
    parser.add_argument(
        "--entity_set",
        type=str,
        required=True,
        help="Entity set can be wither 'wikipedia' or 'wikidata'. It determines which set of entities to"
             "precompute description embeddings for. Note that the entity IDs are not the same across entity sets, "
             "which means separate precomputed description embeddings files are needed for Wikipedia and Wikidata.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="file path to directory containing lookup files and wikipedia_data.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device.",
    )
    args = parser.parse_args()
    if args.device is not None:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    LOG.info(f"Using device: {device}.")
    refined = Refined_UMLS.from_pretrained(model_name=args.model_dir,
                                      entity_set=args.entity_set,
                                      data_dir=args.data_dir,
                                      use_precomputed_descriptions=False,
                                      device=device)
    LOG.info('Precomputing description embeddings.')
    refined.precompute_description_embeddings()
    LOG.info('Done.')


if __name__ == '__main__':
    main()
