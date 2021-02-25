import dataset
from hmm import HiddenMarkovChain
from snippets import find_entities

X, y, labels = dataset.load_china_people_daily("train", with_labels=True)

model = HiddenMarkovChain(labels, task="NER")
model.fit(X, y)

X, y = dataset.load_china_people_daily("test")
for sentence, labels in zip(X, y):
    print(model.find(sentence))
    print(find_entities(sentence, labels))
    input()
