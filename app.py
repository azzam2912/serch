from flask import Flask, render_template, request
from search_engine.search import SearchClass

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
	return render_template("index.html")

@app.route("/search")
def search():
	query = request.args.get("q")
	search_instance = SearchClass()
	result = search_instance.retrieve_result(query=query)
	result = generate_result_slice(result, 150)
	return render_template("search.html", query=query, result=result, result_len = len(result))

def generate_result_slice(result, n_first_word=50):
    new_result = []
    for pair in result:
        score, id = pair
        doc_content = open_file(id)
        doc_content = doc_content[:n_first_word] + " ... "
        new_result.append((score, id, doc_content))
    return new_result

def open_file(id):
    inputFn = f"./documents_database/{id}.txt".format(id)
    try:
        with open(inputFn) as inputFileHandle:
            return inputFileHandle.read()
        
    except IOError:
        return " file cannot be retrieved "
    

@app.route("/document/<id>")
def document(id):
    doc_content = open_file(id)
    return render_template("document.html", 
                           doc_title="Document " + id,  
                           doc_content=doc_content)

if __name__ == '__main__':
    # bsbi_instance = SearchClass(data_file='search_engine/documents.csv')
    # bsbi_instance.do_indexing()
	app.run(debug=False)