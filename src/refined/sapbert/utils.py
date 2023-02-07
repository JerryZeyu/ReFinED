import csv
import json
import numpy as np
import torch
import pdb
import pickle
from tqdm import tqdm


def check_k(queries):
    return len(queries[0]['mentions'][0]['candidates'])


def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    queries = data['queries']
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i + 1]  # to get acc@(i+1)
                mention_hit += np.any([candidate['label'] for candidate in candidates])

            # When all mentions in a query are predicted correctly,
            # we consider it as a hit
            if mention_hit == len(mentions):
                hit += 1

        data['acc{}'.format(i + 1)] = hit / len(queries)

    return data


def check_label(predicted_cui, golden_cui):
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|")))) > 0)

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def min_max(x):
    x = np.array(x)
    min = x.min()
    max = x.max()
    y = [(item-min)/(max-min) for item in x]
    return list(y)
def predict_topk(model_wrapper, eval_dictionary, eval_index, eval_queries, topk, agg_mode="cls"):
    encoder = model_wrapper.get_dense_encoder()
    tokenizer = model_wrapper.get_dense_tokenizer()

    # embed dictionary
    # dict_names = [row[0] for row in eval_dictionary]
    # print("[start embedding dictionary...]")
    # dict_dense_embeds = model_wrapper.embed_dense(names=dict_names, show_progress=True, agg_mode=agg_mode)
    # with open("dict_dense_embeds_snomed_disorder_withAbbreviation.pkl", mode="wb") as f:
    #     pickle.dump(dict_dense_embeds, f, protocol=4)
    # with open(eval_index, mode="rb") as f:
    #     dict_dense_embeds = pickle.load(f)
    dict_dense_embeds = eval_index
    # print("dict_dense_embeds.shape:", dict_dense_embeds.shape)
    mean_centering = False
    if mean_centering:
        tgt_space_mean_vec = dict_dense_embeds.mean(0)
        dict_dense_embeds -= tgt_space_mean_vec

    queries = []
    #for eval_query in tqdm(eval_queries, total=len(eval_queries)):
    for eval_query in eval_queries:
        mention = eval_query
        #golden_cui = eval_query[1].replace("+", "|")
        #print("mention: ", mention)
        dict_mentions = []
        mention_dense_embeds = model_wrapper.embed_dense(names=[mention], agg_mode=agg_mode)

        if mean_centering:
            mention_dense_embeds -= tgt_space_mean_vec

        # get score matrix
        dense_score_matrix = model_wrapper.get_score_matrix(
            query_embeds=mention_dense_embeds,
            dict_embeds=dict_dense_embeds,
        )
        score_matrix = dense_score_matrix

        candidate_idxs, candidate_scores = model_wrapper.retrieve_candidate_cuda(
            score_matrix=score_matrix,
            topk=topk+200,
            batch_size=16,
            show_progress=False
        )
        # print(candidate_idxs.shape)
        np_candidates = [eval_dictionary[ind] for ind in candidate_idxs[0].tolist()]  # .squeeze()
        np_candidates_scores = [score for score in candidate_scores[0].tolist()]
        dict_candidates = []
        for np_candidate_idx, np_candidate in enumerate(np_candidates):
            dict_candidates.append((np_candidate[1], np_candidates_scores[np_candidate_idx]))
        dict_candidates_final = []
        for cand, cand_score in dict_candidates:
            temp = [item[0] for item in dict_candidates_final]
            if cand not in temp:
                dict_candidates_final.append((cand, cand_score))
        raw_umlsID = [item[0] for item in dict_candidates_final[0:topk]]
        raw_pem_score = [item[1] for item in dict_candidates_final[0:topk]]
        #pem_score = softmax(np.array(raw_pem_score))
        pem_score = min_max(raw_pem_score)
        pem_score = [item + 1e-6 if item == 0.0 else item for item in pem_score]
        print(pem_score)
        queries.append([(umlsID, pem_score[idx]) for idx, umlsID in enumerate(raw_umlsID)])
        #queries.append(dict_candidates_final[0:topk])
        #queries.append(dict_candidates[0:topk])


    return queries[0]


def predict_topk_fast(model_wrapper, eval_dictionary, eval_queries, topk, agg_mode="cls"):
    """
    for MedMentions only
    """

    encoder = model_wrapper.get_dense_encoder()
    tokenizer = model_wrapper.get_dense_tokenizer()

    # embed dictionary
    # dict_names = [row[0] for row in eval_dictionary]
    print("[start embedding dictionary...]")
    # dict_dense_embeds = model_wrapper.embed_dense(names=dict_names, show_progress=True, batch_size=4096,
    #                                               agg_mode=agg_mode)
    # with open("dict_dense_embeds_snomed_disorder_withAbbreviation.pkl", mode="wb") as f:
    #     pickle.dump(dict_dense_embeds, f, protocol=4)
    with open(eval_dictionary, mode="rb") as f:
       dict_dense_embeds = pickle.load(f)
    print ("dict_dense_embeds.shape:", dict_dense_embeds.shape)

    bs = 32
    candidate_idxs = None
    candidate_scores = None
    print("[computing rankings...]")

    for i in tqdm(np.arange(0, len(eval_queries), bs), total=len(eval_queries) // bs + 1):
        mentions = list(eval_queries[i:i + bs][:, 0])

        mention_dense_embeds = model_wrapper.embed_dense(names=mentions, agg_mode=agg_mode)

        # get score matrix
        dense_score_matrix = model_wrapper.get_score_matrix(
            query_embeds=mention_dense_embeds,
            dict_embeds=dict_dense_embeds
        )
        score_matrix = dense_score_matrix
        candidate_idxs_batch, candidate_scores_batch = model_wrapper.retrieve_candidate_cuda(
            score_matrix=score_matrix,
            topk=topk,
            batch_size=bs,
            show_progress=False
        )
        if candidate_idxs is None:
            candidate_idxs = candidate_idxs_batch
            candidate_scores = candidate_scores_batch
        else:
            candidate_idxs = np.vstack([candidate_idxs, candidate_idxs_batch])
            candidate_scores = np.vstack([candidate_scores, candidate_scores_batch])

    queries = []
    #golden_cuis = list(eval_queries[:, 1])
    mentions = list(eval_queries[:, 0])
    print("[writing json...]")
    for i in tqdm(range(len(eval_queries))):
        # print(candidate_idxs.shape)
        np_candidates = [eval_dictionary[ind] for ind in candidate_idxs[i].tolist()]  # .squeeze()
        np_candidates_scores = [score for score in candidate_scores[i].tolist()]
        dict_candidates = []
        dict_mentions = []
        for np_idx, np_candidate in enumerate(np_candidates):
            dict_candidates.append((np_candidate[1], np_candidates_scores[np_idx]))
        queries.append(dict_candidates)

    return queries[0]


def evaluate(model_wrapper, eval_dictionary, eval_index, eval_queries, topk, agg_mode="cls"):
    # if medmentions or cometa:
    #     result = predict_topk_fast(model_wrapper, eval_dictionary, eval_queries, topk, agg_mode=agg_mode)
    # else:
    #     result = predict_topk(model_wrapper, eval_dictionary, eval_queries, topk, agg_mode=agg_mode)
    result = predict_topk(model_wrapper, eval_dictionary, eval_index, eval_queries, topk, agg_mode=agg_mode)

    return result