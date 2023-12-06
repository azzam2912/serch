import re
from search_engine.bsbi import BSBIIndex
from search_engine.compression import VBEPostings

class SearchClass:
    def __init__(self, data_dir='documents_database', postings_encoding=VBEPostings, output_dir='search_engine/index'):
        self.BSBI_instance = BSBIIndex(data_dir=data_dir, 
                                       postings_encoding=postings_encoding,
                                       output_dir=output_dir)

    def retrieve_result(self, query):
        return self.BSBI_instance.wand(query, "bm25")
