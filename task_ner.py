import dataset
from hmm import HiddenMarkovChain
from snippets import find_entities
from snippets import find_entities_chunking
from metrics import evaluate_prf

X, y, labels = dataset.load_china_people_daily("train", with_labels=True)

model = HiddenMarkovChain(labels, task="NER")
model.fit(X, y)
model.plot_trans()

if __name__ == "__main__":
    # 测试效果
    X, y = dataset.load_china_people_daily("dev")
    n = 8
    for sentence, labels in zip(X[:n], y[:n]):
        print(model.find(sentence))
        print(find_entities(sentence, labels))

    # 评估指标
    texts, labels = dataset.load_china_people_daily("test")
    y_true = [find_entities_chunking(tags) for text, tags in zip(texts, labels)]
    y_pred = [model.find(text, with_loc=True) for text in texts]
    template = "precision:{:.5f}, recall:{:.5f}, f1:{:.5f}"
    print(template.format(*evaluate_prf(y_true, y_pred)))
