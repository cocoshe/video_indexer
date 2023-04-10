import torch
from jina import Executor, requests, DocumentArray, Document
import numpy as np

class IndexerExecutor(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @requests(on='/index')
    def index(self, docs: DocumentArray, parameters: dict, **kwargs):
        print('-' * 100)
        print(parameters)
        # img_emb = [doc.tags['img_emb'] for doc in docs]
        # text_emb = [doc.tags['text_emb'] for doc in docs]
        img_emb = docs.tensors[0]
        text_emb = docs.tensors[1]
        # print(',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,')
        # print('===img_emb', torch.Tensor(img_emb).shape)
        # print('===text_emb', torch.Tensor(text_emb).shape)
        # img_emb = docs.tensors[0], text_emb = docs.tensors[1]
        scores = []
        for i_emb in img_emb:
            match_score = self.score(i_emb, text_emb)
            scores.append(match_score)
        scores = np.array(scores)  # (seq, 1, 1)
        # print('scores', scores)
        print('scores.shape', scores.shape)
        topk = int(parameters.get('top_k', 5))
        threshold = parameters.get('threshold', 0.1)
        # get top k matches and their scores
        scores = scores.reshape(-1)
        ignore_range = []
        # topk_idxs, topk_scores = scores.argsort()[::-1][:topk], scores[scores.argsort()[::-1][:topk]]
        results = []
        for i in range(topk):
            top_idx, top_score = self.get_next_top(scores, ignore_range)
            left_idx, right_idx = self.get_range(top_idx, top_score, threshold, scores)
            results.append({
                'start': left_idx,
                'end': right_idx,
                'max_score_idx': top_idx,
                'max_score': top_score
            })
            ignore_range += range(left_idx, right_idx + 1)

        result_docs = DocumentArray.empty(len(results))
        for i, doc in enumerate(result_docs):
            doc.tags['start'] = results[i]['start']
            doc.tags['end'] = results[i]['end']
            doc.tags['max_score_idx'] = results[i]['max_score_idx']
            doc.tags['max_score'] = results[i]['max_score']
        return result_docs




    def score(self, img_emb, text_emb):
        # normalize embeddings
        img_emb = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
        text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
        # compute cosine similarity
        scores = np.dot(img_emb, text_emb.T)
        return scores

    def get_next_top(self, scores, ignore_range):
        top_idx, top_score = 0, 0
        for i, score in enumerate(scores):
            if i not in ignore_range and score > top_score:
                top_idx, top_score = i, score
        return top_idx, top_score
    def get_range(self, top_idx, top_score, threshold, scores):
        left_idx = top_idx
        right_idx = top_idx
        for j in range(top_idx):
            prev_idx = top_idx - j
            if top_score - scores[prev_idx] <= threshold:
                left_idx = prev_idx
            else:
                break
        for j in range(len(scores) - top_idx):
            next_idx = top_idx + j
            if top_score - scores[next_idx] <= threshold:
                right_idx = next_idx
            else:
                break

        if right_idx - left_idx > 60:
            left_idx, right_idx = self.get_range(top_idx, top_score, threshold / 2, scores)
        return left_idx, max(right_idx, left_idx + 10)
