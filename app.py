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
	return render_template("search.html", query=query, result=result, result_len = len(result))

@app.route("/document/<id>")
def document(id):
	return render_template(
		"document.html",
		doc_title="Document " + id,
		doc_text="This is the text of document " + id)

if __name__ == '__main__':
    # bsbi_instance = SearchClass(data_file='search_engine/documents.csv')
    # bsbi_instance.do_indexing()
	app.run(debug=True)