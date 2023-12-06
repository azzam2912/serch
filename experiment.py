import ir_datasets
dataset = ir_datasets.load("wikir/en1k/test")
print(dataset)
for query in dataset.queries_iter():
    print(query) # namedtuple<query_id, text>