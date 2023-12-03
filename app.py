from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
	return render_template("index.html")

@app.route("/search")
def search():
	query = request.args.get("q")
	return render_template("search.html", query=query)

@app.route("/document/<id>")
def document(id):
	return render_template(
		"document.html",
		doc_title="Document " + id,
		doc_text="This is the text of document " + id)

if __name__ == '__main__':
	app.run(debug=True)