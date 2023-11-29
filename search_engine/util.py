import math


class IdMap:
    """
    Ingat kembali di kuliah, bahwa secara praktis, sebuah dokumen dan
    sebuah term akan direpresentasikan sebagai sebuah integer. Oleh
    karena itu, kita perlu maintain mapping antara string term (atau
    dokumen) ke integer yang bersesuaian, dan sebaliknya. Kelas IdMap ini
    akan melakukan hal tersebut.
    """

    def __init__(self):
        """
        Mapping dari string (term atau nama dokumen) ke id disimpan dalam
        python's dictionary; cukup efisien. Mapping sebaliknya disimpan dalam
        python's list.

        contoh:
            str_to_id["halo"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "halo"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Mengembalikan banyaknya term (atau dokumen) yang disimpan di IdMap."""
        return len(self.id_to_str)

    def __get_id(self, s):
        """
        Mengembalikan integer id i yang berkorespondensi dengan sebuah string s.
        Jika s tidak ada pada IdMap, lalu assign sebuah integer id baru dan kembalikan
        integer id baru tersebut.
        """
        if s not in self.str_to_id:
            self.str_to_id[s] = len(self.id_to_str)
            self.id_to_str.append(s)

        return self.str_to_id[s]

    def __get_str(self, i):
        """Mengembalikan string yang terasosiasi dengan index i."""
        return self.id_to_str[i]

    def __getitem__(self, key):
        """
        __getitem__(...) adalah special method di Python, yang mengizinkan sebuah
        collection class (seperti IdMap ini) mempunyai mekanisme akses atau
        modifikasi elemen dengan syntax [..] seperti pada list dan dictionary di Python.

        Silakan search informasi ini di Web search engine favorit Anda. Saya mendapatkan
        link berikut:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        Jika key adalah integer, gunakan __get_str;
        jika key adalah string, gunakan __get_id
        """
        if isinstance(key, int):
            return self.__get_str(key)
        elif isinstance(key, str):
            return self.__get_id(key)
        else:
            raise TypeError("Integer or string only")


def merge_and_sort_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Menggabung (merge) dua lists of tuples (doc id, tf) dan mengembalikan
    hasil penggabungan keduanya (TF perlu diakumulasikan untuk semua tuple
    dengn doc id yang sama), dengan aturan berikut:

    contoh: posts_tfs1 = [(1, 34), (3, 2), (4, 23)]
            posts_tfs2 = [(1, 11), (2, 4), (4, 3 ), (6, 13)]

            return   [(1, 34+11), (2, 4), (3, 2), (4, 23+3), (6, 13)]
                   = [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]

    Parameters
    ----------
    list1: List[(Comparable, int)]
    list2: List[(Comparable, int]
        Dua buah sorted list of tuples yang akan di-merge.

    Returns
    -------
    List[(Comparable, int)]
        Penggabungan yang sudah terurut
    """
    merged = []
    ptr_1 = ptr_2 = 0

    while ptr_1 < len(posts_tfs1) and ptr_2 < len(posts_tfs2):
        if posts_tfs1[ptr_1][0] == posts_tfs2[ptr_2][0]:
            merged.append((posts_tfs1[ptr_1][0], posts_tfs1[ptr_1][1] + posts_tfs2[ptr_2][1]))
            ptr_1 += 1
            ptr_2 += 1
        elif posts_tfs1[ptr_1][0] < posts_tfs2[ptr_2][0]:
            merged.append(posts_tfs1[ptr_1])
            ptr_1 += 1
        else:
            merged.append(posts_tfs2[ptr_2])
            ptr_2 += 1

    while ptr_1 < len(posts_tfs1):
        merged.append(posts_tfs1[ptr_1])
        ptr_1 += 1

    while ptr_2 < len(posts_tfs2):
        merged.append(posts_tfs2[ptr_2])
        ptr_2 += 1

    return merged


def count_tfidf_score(tf, df, N):
    """
    Menghitung skor tf-idf dari suatu term dalam suatu dokumen.

    Parameters
    ----------
    tf: int
        Term frequency dari sebuah term dalam sebuah dokumen
    df: int
        Document frequency dari sebuah term dalam semua dokumen
    N: int
        Banyaknya dokumen dalam koleksi

    Returns
    -------
    float
        Skor tf-idf dari term tersebut dalam dokumen tersebut
    """
    w_tf = 1 + math.log(tf, 10)
    idf = math.log(N / df, 10)
    return w_tf * idf


def count_bm25_score(tf, df, N, dl, avdl, k1, b):
    """
    Menghitung skor BM25 dari suatu term dalam suatu dokumen.

    Parameters
    ----------
    tf: int
        Term frequency dari sebuah term dalam sebuah dokumen
    df: int
        Document frequency dari sebuah term dalam semua dokumen
    N: int
        Banyaknya dokumen dalam koleksi
    dl: int
        Banyaknya term dalam sebuah dokumen
    avdl: float
        Rata-rata banyaknya term dalam sebuah dokumen dalam koleksi
    k1: float
        Parameter k1 untuk BM25
    b: float
        Parameter b untuk BM25

    Returns
    -------
    float
        Skor BM25 dari term tersebut dalam dokumen tersebut
    """
    top = (k1 + 1) * tf
    dl_normalization = (1 - b) + b * (dl / avdl)
    bottom = k1 * dl_normalization + tf
    idf = math.log(N / df, 10)
    
    return idf * (top / bottom)


if __name__ == '__main__':

    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    term_id_map = IdMap()

    assert [term_id_map[term]
            for term in doc] == [0, 1, 2, 3, 1], "term_id salah"
    assert term_id_map[1] == "semua", "term_id salah"
    assert term_id_map[0] == "halo", "term_id salah"
    assert term_id_map["selamat"] == 2, "term_id salah"
    assert term_id_map["pagi"] == 3, "term_id salah"

    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]
    doc_id_map = IdMap()
    assert [doc_id_map[docname]
            for docname in docs] == [0, 1, 2], "docs_id salah"

    assert merge_and_sort_posts_and_tfs([(1, 34), (3, 2), (4, 23)],
                                        [(1, 11), (2, 4), (4, 3), (6, 13)]) == [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)], "merge_and_sort_posts_and_tfs salah"
