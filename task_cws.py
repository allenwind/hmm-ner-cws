import dataset
from hmm import HiddenMarkovChain
from snippets import find_words

X, y, labels = dataset.load_cws_ctb6("train", with_labels=True)

model = HiddenMarkovChain(labels, task="CWS")
model.fit(X, y)

# X, y = dataset.load_cws_ctb6("test")
# for sentence, labels in zip(X, y):
#     print(model.find(sentence))
#     print(find_words(sentence, labels))
#     input()

for text in dataset.load_sentences():
    print(model.find(text))
