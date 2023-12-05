import math
import random
import warnings
import numpy as np
import lightgbm as lgb
import os

from collections import Counter, defaultdict
from scipy.spatial.distance import cosine
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from tqdm import tqdm

from search_engine.bsbi import BSBIIndex
from search_engine.compression import VBEPostings
from search_engine.util import count_bm25_score, count_tfidf_score

warnings.filterwarnings("ignore", category=RuntimeWarning, module="gensim")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

class Letor:
    def __init__(self, ranker, topics_count=200, improved=False):
        self.topics_count = topics_count
        self.improved = improved

        self.tf = None
        self.df = None
        self.dls = None

        self.ranker = ranker
        self.model = None
        self.dictionary = None

        self.test_queries = self.load_queries("qrels-folder/test_queries.txt")
        self.test_qrels = self.load_qrels("qrels-folder/test_qrels.txt")

        self.bsbi = BSBIIndex(data_dir="collections",
                              postings_encoding=VBEPostings,
                              output_dir="index")
        
        self.create_model()

    
    def load_qrels(self, qrel_file):
        qrels = defaultdict(lambda: defaultdict(lambda: 0)) 
        with open(qrel_file) as file:
            for line in file:
                data = line.strip().split()
                if len(data) == 3:
                    qid, did, rel = data
                else:
                    qid, did = data
                    rel = 0
                
                did = int(did)
                qrels[qid][did] = int(rel)

        return qrels


    def load_queries(self, query_file):
        queries = {}
        with open(query_file) as file:
            for line in file:
                qid, query = line.strip().split(maxsplit=1)
                queries[qid] = query.split()

        return queries


    def load_documents(self, document_file):
        documents = defaultdict(list)
        term_frequency = defaultdict(Counter)
        document_frequency = defaultdict(set)
        document_length = defaultdict(int)

        with open(document_file) as file:
            for line in file:
                did, document = line.strip().split(maxsplit=1)
                did = int(did)
                terms = document.split()
                documents[did] = terms

                term_frequency[did] = Counter(terms)

                for term in set(terms):
                    document_frequency[term].add(did)

                document_length[did] = len(terms)


        return documents, term_frequency, document_frequency, document_length


    def load_dataset(self, qrel, queries, documents, negative_samples=1):
        group_qid_count = []
        dataset = []

        for qid in qrel:
            docs_rels = qrel[qid]
            group_qid_count.append(len(docs_rels) + negative_samples)
            
            for did, rel in docs_rels.items():
                dataset.append((queries[qid], (did, documents[did]), rel))

            for _ in range(negative_samples):
                did = random.choice(list(documents.keys()))
                while did in docs_rels:
                    did = random.choice(list(documents.keys()))
                dataset.append((queries[qid], (did, documents[did]), 0))

        assert sum(group_qid_count) == len(dataset), "group_qid_count and dataset size mismatch"
        return dataset, group_qid_count


    def vector_rep(self, text):
        rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == self.topics_count else [0.] * self.topics_count


    # Feature Extraction =======================================================
    def get_feature_default(self, query, doc_data):
        _, doc = doc_data
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_sim = 1 - cosine(v_q, v_d)
        jaccard_sim = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard_sim, cosine_sim]

    
    def get_feature_improved(self, query, doc_data):
        did, doc = doc_data

        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)

        cosine_sim = 1 - cosine(v_q, v_d)
        jaccard_sim = len(q & d) / len(q | d)

        # bm25_score = self.ltr_bm25_score(query, did)
        # tf_idf_score = self.ltr_tf_idf_score(query, did)

        overlap = len(q & d)
        query_freq = sum([1 for word in query if word in d])
        doc_freq = sum([1 for word in d if word in query])


        # NOTE: Jika mengganti fitur yang digunakan, 
        #       ganti juga 'feature_size' pada fungsi 'split_dataset'

        return v_q + v_d + [jaccard_sim, cosine_sim, overlap, query_freq, doc_freq]
        # return v_q + v_d + [jaccard_sim, cosine_sim, overlap, query_freq, doc_freq, bm25_score, tf_idf_score]
        # return [cosine_sim, jaccard_sim]
        # return [cosine_sim, jaccard_sim, overlap, query_freq, doc_freq]
        # return [cosine_sim, jaccard_sim, overlap, query_freq, doc_freq, bm25_score, tf_idf_score]
    # =========================================================================


    def split_dataset(self, dataset):
        X = []
        Y = []

        feature_size = (2 * self.topics_count + 5) if self.improved else 2 * self.topics_count + 2
        # feature_size = (2 * self.topics_count + 7) if self.improved else 2 * self.topics_count + 2
        # feature_size = 2 if self.improved else 2 * self.topics_count + 2
        # feature_size = 5 if self.improved else 2 * self.topics_count + 2
        # feature_size = 7 if self.improved else 2 * self.topics_count + 2
        for (query, doc, rel) in tqdm(dataset, leave=False, desc="Splitting dataset"):
            if self.improved:
                X.append(self.get_feature_improved(query, doc))
            else:
                X.append(self.get_feature_default(query, doc))
            Y.append(rel)

        X = np.array(X)
        Y = np.array(Y)

        assert X.shape[0] == Y.shape[0], "X and Y size mismatch"
        assert X.shape[0] == len(dataset), "X[0] and dataset size mismatch"
        assert X.shape[1] == feature_size, "X[1] and topics_count mismatch"

        return X, Y
    

    def create_model(self):
        train_documents, self.tf, self.df, self.dls = self.load_documents("qrels-folder/train_docs.txt")
        train_qrels = self.load_qrels("qrels-folder/train_qrels.txt")
        train_queries = self.load_queries("qrels-folder/train_queries.txt")

        val_qrels = self.load_qrels("qrels-folder/val_qrels.txt")
        val_queries = self.load_queries("qrels-folder/val_queries.txt")

        train_dataset, train_group_qid_count = self.load_dataset(train_qrels, train_queries, train_documents)
        val_dataset, val_group_qid_count = self.load_dataset(val_qrels, val_queries, train_documents)


        self.dictionary = Dictionary()
        corpus = [
            self.dictionary.doc2bow(doc, allow_update=True) 
            for doc in train_documents.values()
        ]
        self.model = LsiModel(corpus, num_topics=self.topics_count+1)

        X, Y = self.split_dataset(train_dataset)
        val_X, val_Y = self.split_dataset(val_dataset)

        self.ranker.fit(
            X, Y, group=train_group_qid_count,
            eval_set=[(val_X, val_Y)], eval_group=[val_group_qid_count],
        )
    

    def rank(self, query, documents):
        if len(documents) == 0:
            return []
        
        unseen = []
        for doc in tqdm(documents.values(), leave=False, desc="Creating unseen features"):
            if self.improved:
                unseen.append(self.get_feature_improved(query.split(), doc))
            else:
                unseen.append(self.get_feature_default(query.split(), doc))

        unseen = np.array(unseen)
        scores = self.ranker.predict(unseen)

        return sorted(zip(documents.keys(), scores), key=lambda x: x[1], reverse=True)


    # Scoring =================================================================
    def rbp(self, ranking, p=0.8):
        score = 0.
        for i in range(1, len(ranking) + 1):
            pos = i - 1
            score += ranking[pos] * (p ** (i - 1))
        return (1 - p) * score
    
    def dcg(self, ranking):
        score = 0
        for i, r in enumerate(ranking):
            if r == 1:
                score += (1 / math.log(i+2, 2))
        return score

    def ndcg(self, ranking):
        dcg_score = self.dcg(ranking)
        ideal_ranking = sorted(ranking, reverse=True)
        ideal_dcg_score = self.dcg(ideal_ranking)
        if ideal_dcg_score == 0:
            return 0
        return dcg_score / ideal_dcg_score
    

    def ltr_bm25_score(self, query, did, k1=1.2, b=0.75):
        score = 0.
        for term in query:
            tf = self.tf[did][term]
            df = len(self.df[term])
            N = len(self.dls)
            dl = self.dls[did]
            avdl = sum(self.dls.values()) / len(self.dls)

            if tf == 0 or df == 0:
                continue
            
            score += count_bm25_score(tf, df, N, dl, avdl, k1, b)
        return score
    
    def ltr_tf_idf_score(self, query, did):
        score = 0.
        for term in query:
            tf = self.tf[did][term]
            df = len(self.df[term])
            N = len(self.dls)

            if tf == 0 or df == 0:
                continue

            score += count_tfidf_score(tf, df, N)
        return score
    # =========================================================================


    def test(self, verbose=False, K=10):
        without_ltr_scores = []
        with_ltr_scores = []

        for qid, query_terms in tqdm(self.test_queries.items(), leave=False, desc="Testing"):
            query = " ".join(query_terms)
            documents = defaultdict(list)

            ranking_no_ltr = []
            for (score, doc_path) in self.bsbi.retrieve_bm25(query, k=K):
                did = int(os.path.splitext(os.path.basename(doc_path))[0])
                if (did in self.test_qrels[qid]):
                    ranking_no_ltr.append(1)
                else:
                    ranking_no_ltr.append(0)
                    
                with open(f"collections/{doc_path}") as doc_file:
                    documents[did] = (doc_path, doc_file.read().strip().split())
            
            ranking_ltr = []
            for (did, score) in self.rank(query, documents):
                if (did in self.test_qrels[qid]):
                    ranking_ltr.append(1)
                else:
                    ranking_ltr.append(0)


            score_no_ltr = self.ndcg(ranking_no_ltr)
            score_ltr = self.ndcg(ranking_ltr)

            without_ltr_scores.append(score_no_ltr)
            with_ltr_scores.append(score_ltr)

            if verbose:
                print(f"[{qid}] Query : {query}")
                print(f"RBP NoLTR: {score_no_ltr:>.2f}")
                print(f"RBP LTR  : {score_ltr:>.2f}")
                print("==================================\n")


        no_ltr_score = sum(without_ltr_scores) / len(without_ltr_scores)
        ltr_score = sum(with_ltr_scores) / len(with_ltr_scores)
        print(f"NDCG NoLTR: {no_ltr_score:>.2f}")
        print(f"NDCG LTR  : {ltr_score:>.2f}")
        return no_ltr_score, ltr_score


if __name__ == "__main__":
    K = 100
    TOPICS_COUNT = 200
    IMPROVED_VERSION = True

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        boosting_type = "gbdt",
        n_estimators = 100,
        importance_type = "gain",
        metric = "ndcg",
        num_leaves = 40,
        learning_rate = 0.02,
        max_depth = -1,
        verbosity = 0)
    
    letor = Letor(ranker, topics_count=TOPICS_COUNT, improved=IMPROVED_VERSION)
    letor.test(K=K)