import os
from flask import Flask, render_template, request
from search_engine.search import SearchClass
from json import dumps

app = Flask(__name__)
SC = SearchClass()
DOCS_PATH = "documents_database"

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/search")
def search():
	query = request.args.get("q")
	result = SC.retrieve_result(query=query, k=1000)
	result_json = dumps(generate_result_slice(result, 150))
	return render_template("search.html", query=query, result=result_json, result_len = len(result))

def generate_result_slice(result, n_first_word=50):
    new_result = []
    for pair in result:
        score, doc_id = pair
        doc_id = doc_id.replace(".txt", "")
        doc_content = open_file(doc_id)
        doc_content = doc_content[:n_first_word] + " ... "
        doc_block = doc_id.split("/")[0]
        doc_id = doc_id.split("/")[1]
        new_result.append((score, doc_block, doc_id, doc_content))
    return new_result

def open_file(doc_id):
    doc_path = os.path.join(DOCS_PATH, doc_id + ".txt")
    if os.path.exists(doc_path):
        with open(doc_path, "r") as f:
            doc_content = f.read()
    else:
        doc_content = "File not found"
    return doc_content
    

@app.route("/document/<block>/<doc_id>")
def document(block, doc_id):
    doc_path = f"{block}/{doc_id}"
    doc_content = open_file(doc_path)
    return render_template("document.html", 
                           doc_title="Document " + doc_id,  
                           doc_content=doc_content)

if __name__ == '__main__':
	app.run(debug=True)