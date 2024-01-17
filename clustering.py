"""
This is a more complex example on performing clustering on large scale dataset.

This examples find in a large set of sentences local communities, i.e., groups of sentences that are highly
similar. You can freely configure the threshold what is considered as similar. A high threshold will
only find extremely similar sentences, a lower threshold will find more sentence that are less similar.

A second parameter is 'min_community_size': Only communities with at least a certain number of sentences will be returned.

The method for finding the communities is extremely fast, for clustering 50k sentences it requires only 5 seconds (plus embedding comuptation).

In this example, we download a large set of questions from Quora and then find similar questions in this set.
"""
from sentence_transformers import SentenceTransformer, util
import os
import csv
import time

import pickle
import random

import torch
from torch import Tensor
import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm
from community_detection import *

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def community_detection(embeddings, threshold=0.75, min_community_size=10, batch_size=1024):
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.
    """
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    threshold = torch.tensor(threshold, device=embeddings.device)

    extracted_communities = []

    # Maximum size for community
    min_community_size = min(min_community_size, len(embeddings))
    sort_max_size = min(max(2 * min_community_size, 50), len(embeddings))

    for start_idx in tqdm(range(0, len(embeddings), batch_size)):
        # Compute cosine similarity scores
        cos_scores = cos_sim(embeddings[start_idx:start_idx + batch_size], embeddings)

        # Minimum size for a community
        top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

        # Filter for rows >= min_threshold
        for i in range(len(top_k_values)):
            if top_k_values[i][-1] >= threshold:
                new_cluster = []

                # Only check top k most similar entries
                top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                # Check if we need to increase sort_max_size
                while top_val_large[-1] > threshold and sort_max_size < len(embeddings):
                    sort_max_size = min(2 * sort_max_size, len(embeddings))
                    top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                for idx, val in zip(top_idx_large.tolist(), top_val_large):
                    if val < threshold:
                        break

                    new_cluster.append(idx)

                extracted_communities.append(new_cluster)

        del cos_scores

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for cluster_id, community in tqdm(enumerate(extracted_communities)):
        community = sorted(community)
        non_overlapped_community = []
        for idx in community:
            if idx not in extracted_ids:
                non_overlapped_community.append(idx)

        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            extracted_ids.update(non_overlapped_community)

    unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)

    return unique_communities

# Model for computing sentence embeddings. We use one trained for similar questions detection
model = SentenceTransformer('all-mpnet-base-v2')

data_path = "/sensei-fs/users/ksaifullah/dolphin.jsonl"
data_dict = load_dataset('json', data_files=data_path, split='train')
max_corpus_size = len(data_dict)  # We limit our corpus to only the first 50k questions

# corpus_sentences = data_dict.map(lambda examples: {"text": [examples['instruction'][i]+' '+examples['input'][i] for i in range(len(examples['input']))]}, batched=True)['text']
corpus_sentences = data_dict.map(lambda examples: {"text": [examples['input'][i] for i in range(len(examples['input']))]}, batched=True)
print("Encode the corpus. This might take a while")
corpus_embeddings = model.encode(corpus_sentences['text'], batch_size=1024, show_progress_bar=True, convert_to_numpy=True)
np.save('/sensei-fs/users/ksaifullah/dolphin_embeddings.npy', corpus_embeddings)

corpus_embeddings = np.load('/sensei-fs/users/ksaifullah/dolphin_embeddings.npy')
ids = range(len(corpus_embeddings))
embeddings = {idx: embedding for idx, embedding in zip(ids, corpus_embeddings)}
clusters = {}
print("Start clustering")
start_time = time.time()
#Two parameters to tune:
#min_cluster_size: Only consider cluster that have at least 25 elements
#threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
# clusters = util.community_detection(corpus_embeddings, min_community_size=25, threshold=0.75)
clusters = online_community_detection(ids, embeddings, clusters, chunk_size=10000, threshold=0.75, min_cluster_size=25, cores=4)
end_time = time.time()
print("Minutes taken for clustering: {}".format((end_time-start_time)/60))
with open('/sensei-fs/users/ksaifullah/dolphin_instructions_cluster_sbert.pkl', 'wb') as f:
    pickle.dump(clusters, f)

#Print for all clusters the top 3 and bottom 3 elements
for i, cluster in enumerate(clusters.values()):
    print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
    for sentence_id in cluster[0:3]:
        print("\t", corpus_sentences[int(sentence_id[0])])
    print("\t", "...")
    for sentence_id in cluster[-3:]:
        print("\t", corpus_sentences[int(sentence_id[0])])
    if i==0:
        break