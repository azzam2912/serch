import re
from search_engine.bsbi import BSBIIndex
from search_engine.compression import VBEPostings
import sys

# BSBI_instance = BSBIIndex(data_file='search_engine/documents.csv',
#                           postings_encoding=VBEPostings,
#                           output_dir='search_engine/index')

# query = 'southern methodist university'
# result = BSBI_instance.wand(query, "bm25")
# for (score, doc_id) in result:
#     print(f"Document {doc_id} has score {score:.2f}")

# with open('search_engine/doc_dump.txt') as f:
#     data = f.readlines()
#     doc_index = int(result[0][1].split('-')[1])-1
#     print(data[doc_index])
    
class SearchClass:
    def __init__(self):
        self.BSBI_instance = BSBIIndex(data_file='search_engine/documents.csv',
                          postings_encoding=VBEPostings,
                          output_dir='./search_engine/index')
    
    def retrieve_result(self, query):
        result = self.BSBI_instance.wand(query, "bm25")
        print(result)
        return result
    
