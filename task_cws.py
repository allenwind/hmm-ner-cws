import re
import dataset
from hmm import HiddenMarkovChain
from snippets import find_words

class TokenizerBase:
    """分词的基类，继承该类并在find_word实现分词的核心算法"""

    spaces = re.compile("(\r\n|\s)", re.U)
    english = re.compile("[a-zA-Z0-9]", re.U)
    chinese = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)", re.U)

    def cut(self, text):
        return list(self._cut(text))

    def _cut(self, text):
        # 把长文本切分为句子块
        for block in self.chinese.split(text):
            if not block:
                continue
            if self.chinese.match(block):
                yield from self.cut_block(block)
            else:
                for s in self.spaces.split(block):
                    if self.spaces.match(s):
                        yield s
                    else:
                        yield from s

    def cut_block(self, sentence):
        # 对文本进行分块分句后分词
        buf = ""
        for word in self.find_word(sentence):
            if len(word) == 1 and self.english.match(word):
                buf += word
            else:
                if buf:
                    yield buf
                    buf = ""
                yield word
        if buf:
            yield buf

    def find_word(self, sentence):
        # 从这里实现分词算法的核心
        # 从句子中发现可以构成的词，返回可迭代对象
        raise NotImplementedError

class HMMTokenizer(TokenizerBase, HiddenMarkovChain):
    """分块后的HMM分词"""

    def find_word(self, sentence):
        yield from self.find(sentence)

# 使用无监督的随机数据
X, y, labels = dataset.load_random_sentences("train", nums=1000, with_labels=True)

model = HiddenMarkovChain(labels, task="CWS")
model.fit(X, y)
model.plot_trans()

tokenizer = HMMTokenizer(labels, task="CWS")
tokenizer.fit(X, y)
if __name__ == "__main__":
    from metrics import evaluate_prf
    from snippets import find_words_regions
    from snippets import to_regions

    for text in dataset.load_sentences():
        # 两种方案对比
        print(model.find(text))
        print(tokenizer.cut(text))

    X, y = dataset.load_cws_ctb6("test")
    n = 8
    for sentence, labels in zip(X[:8], y[:8]):
        # 对比
        print(find_words(sentence, labels))
        print(model.find(sentence))
        print(tokenizer.cut(sentence))

    # 评估指标
    X, y, labels = dataset.load_cws_ctb6("train", with_labels=True)
    model = HiddenMarkovChain(labels, task="CWS")
    model.fit(X, y)
    texts, labels = dataset.load_cws_ctb6("test")
    y_true = [find_words_regions(text, tags) for text, tags in zip(texts, labels)]
    y_pred = [to_regions(segments) for segments in [model.find(text) for text in texts]]
    template = "precision:{:.5f}, recall:{:.5f}, f1:{:.5f}"
    print(template.format(*evaluate_prf(y_true, y_pred)))
