import os
import pickle
import contextlib
import heapq
import math
import re
import time

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, count_bm25_score, count_tfidf_score, merge_and_sort_posts_and_tfs
from compression import VBEPostings
from tqdm import tqdm

from mpstemmer import MPStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from operator import itemgetter


class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def pre_processing_text(self, content):
        """
        Melakukan preprocessing pada text, yakni stemming dan removing stopwords
        """
        # https://github.com/ariaghora/mpstemmer/tree/master/mpstemmer

        content = content.lower()
        tokens = re.findall(r'\w+', content)

        stemmer = MPStemmer()
        stemmed = [stemmer.stem(token) for token in tokens]

        # remover = StopWordRemoverFactory().create_stop_word_remover()
        # filtered = [remover.remove(token) for token in stemmed]

        return ' '.join(stemmed)

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk stemming bahasa Indonesia, seperti
        MpStemmer: https://github.com/ariaghora/mpstemmer 
        Jangan gunakan PySastrawi untuk stemming karena kode yang tidak efisien dan lambat.

        JANGAN LUPA BUANG STOPWORDS! Kalian dapat menggunakan PySastrawi 
        untuk menghapus stopword atau menggunakan sumber lain seperti:
        - Satya (https://github.com/datascienceid/stopwords-bahasa-indonesia)
        - Tala (https://github.com/masdevid/ID-Stopwords)

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        td_pairs = []

        files = os.listdir(os.path.join(self.data_dir, block_path))
        for file in tqdm(sorted(files), leave=False):
            with open(os.path.join(self.data_dir, block_path, file), 'r', encoding='utf-8') as f:
                content = f.read()
                content = self.pre_processing_text(content)

                terms = content.split()
                doc_id = self.doc_id_map[os.path.join(block_path, file)]
                for term in terms:
                    term_id = self.term_id_map[term]
                    td_pairs.append((term_id, doc_id))
        
        return td_pairs

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        for term_id, doc_id in sorted(td_pairs):
            if term_id not in term_dict:
                term_dict[term_id] = {"doc_ids": [], "tf": []}
            
            if doc_id not in term_dict[term_id]["doc_ids"]:
                term_dict[term_id]["doc_ids"].append(doc_id)
                term_dict[term_id]["tf"].append(1)
            else:
                term_dict[term_id]["tf"][-1] += 1

        for term_id, postings in term_dict.items():
            index.append(term_id, postings["doc_ids"], postings["tf"])


    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(list(zip(postings, tf_list)),
                                                        list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)


    def retrieve_boolean(self, query):
        """
        Melakukan Boolean Retrieval dari Tugas Pemrograman 1 untuk testing
        """
        self.load()

        query = self.pre_processing_text(query)
        query_terms = query.split()

        result = []
        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as index:
            for term in query_terms:
                term_id = self.term_id_map[term]
            
                try:
                    postings_list, tf_list = index.get_postings_list(term_id)
                    docs_list = [self.doc_id_map[doc_id] for doc_id in postings_list]

                    if len(result) == 0:
                        result = docs_list
                    else:
                        result = set(result).intersection(set(docs_list))
                except KeyError:
                    return []
        
        return result
                    

    def retrieve_tfidf(self, query, k=10, timer=False):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        self.load()

        processed_query = self.pre_processing_text(query)
        query_terms = processed_query.split()

        scores = {}
        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as index:
            if timer:
                start = time.time()

            N = len(index.doc_length)
            for term in query_terms:
                term_id = self.term_id_map[term]
                try:
                    postings_list, tf_list = index.get_postings_list(term_id)
                    df = index.postings_dict[term_id][1]

                    for i, doc_id in enumerate(postings_list):
                        tf = tf_list[i]

                        doc = self.doc_id_map[doc_id]
                        if doc not in scores:
                            scores[doc] = 0
                        scores[doc] += count_tfidf_score(tf, df, N)
                        
                except KeyError:
                    pass

        top_k = [(score, doc) for doc, score in scores.items()]
        r = heapq.nlargest(k, top_k)

        if timer:
            print(f"Retrieval time: {time.time() - start:.5f} seconds")

        return r

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75, timer=False):
        """
        Melakukan Ranked Retrieval dengan skema scoring BM25 dan framework TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        """
        self.load()

        processed_query = self.pre_processing_text(query)
        query_terms = processed_query.split()

        scores = {}
        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as index:
            if timer:
                start = time.time()

            avdl = index.avg_doc_length
            N = len(index.doc_length)
            for term in query_terms:
                term_id = self.term_id_map[term]
                try:
                    postings_list, tf_list = index.get_postings_list(term_id)
                    df = index.postings_dict[term_id][1]

                    for i, doc_id in enumerate(postings_list):
                        tf = tf_list[i]
                        dl = index.doc_length[doc_id]

                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += count_bm25_score(tf, df, N, dl, avdl, k1, b)
                        
                except KeyError:
                    pass

        k_scores = heapq.nlargest(k, scores.items(), key=itemgetter(1))

        if timer:
            print(f"Retrieval time: {time.time() - start:.5f} seconds")

        return [(score, self.doc_id_map[doc_id]) for doc_id, score in k_scores]

    def wand(self, query, scoring_regime, k=10, k1=1.2, b=0.75, timer=False):
        """
        WAND Top-K dengan TF-IDF
        
        ASUMSI: Posting list, TF list, dan upper bound
                untuk semua query terms cukup di memory
        """
        if scoring_regime.lower() not in ["tfidf", "bm25"]:
            raise ValueError("Invalid scoring regime (tfidf or bm25 only)")

        self.load()

        processed_query = self.pre_processing_text(query)
        query_terms = processed_query.split()

        query_index = {}
        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as index:            
            for term in query_terms:
                term_id = self.term_id_map[term]
                try:
                    postings_list, tf_list = index.get_postings_list(term_id)
                    if scoring_regime == "bm25":
                        ub = index.get_bm25_ub(term_id)
                    else:
                        ub = index.get_tfidf_ub(term_id)

                    query_index[term] = {
                        "postings_list": postings_list,
                        "tf_list": tf_list,
                        "ub": ub
                    }
                        
                except KeyError:
                    pass

        top_k = []
        threshold = 0
        pivot = 0
        curr_doc_id = 0
        ptr = {
            term: 0 for term in query_index
        }


        N = len(index.doc_length)
        avdl = index.avg_doc_length

        fully_evaluated = 0
        is_running = True
        if timer:
            start = time.time()
        while is_running and len(query_index) != 0:
            sorted_terms = sorted(query_index, key=lambda x: query_index[x]["postings_list"][ptr[x]])

            max_contrib = 0
            for term in sorted_terms:
                max_contrib += query_index[term]["ub"]
                if max_contrib >= threshold:
                    term_ptr = ptr[term]
                    pivot = query_index[term]["postings_list"][term_ptr]
                    break
            
            first_term = sorted_terms[0]
            first_postings_list = query_index[first_term]["postings_list"]

            if pivot <= curr_doc_id:
                while first_postings_list[ptr[first_term]] <= curr_doc_id:
                    ptr[first_term] += 1

                    if ptr[first_term] >= len(first_postings_list):
                        is_running = False
                        break

            else:
                if first_postings_list[ptr[first_term]] == pivot:
                    curr_doc_id = pivot

                    score = 0
                    for term in sorted_terms:
                        if query_index[term]["postings_list"][ptr[term]] != curr_doc_id:
                            continue

                        postings_list = query_index[term]["postings_list"]
                        tf_list = query_index[term]["tf_list"]

                        term_id = self.term_id_map[term]
                        df = index.postings_dict[term_id][1]
                        tf = tf_list[ptr[term]]

                        if scoring_regime == "bm25":
                            dl = index.doc_length[curr_doc_id]
                            score += count_bm25_score(tf, df, N, dl, avdl, k1, b)
                        else:
                            score += count_tfidf_score(tf, df, N)

                    heapq.heappush(top_k, (score, self.doc_id_map[curr_doc_id]))
                    top_k = heapq.nlargest(k, top_k)
                    threshold = math.ceil(heapq.nsmallest(1, top_k)[0][0])

                    fully_evaluated += 1

                else:
                    while first_postings_list[ptr[first_term]] < pivot:
                        ptr[first_term] += 1

                        if ptr[first_term] >= len(first_postings_list):
                            is_running = False
                            break


        r = heapq.nlargest(k, top_k)
        if timer:
            print(f"Retrieval time: {time.time() - start:.5f} seconds")

            max_doc_count = sum([len(query_index[term]["postings_list"]) for term in query_index])
            print(f"Fully evaluated {fully_evaluated} out of {max_doc_count} documents")

        return r

    def do_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parsing_block
        untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parsing_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.do_indexing()  # memulai indexing!
