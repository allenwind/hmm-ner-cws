import random
import itertools
import re
import time
from functools import wraps
import numpy as np

# NER数据加载

def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, "elapsed time {:.3f}s".format(end-start))
        return result
    return wrapper

def load_file(file, sep=" ", shuffle=True, with_labels=False):
    # 返回逐位置标注形式
    with open(file, encoding="utf-8") as fp:
        text = fp.read()
    lines = text.split("\n\n")
    if shuffle:
        random.shuffle(lines)
    X = []
    y = []
    for line in lines:
        if not line:
            continue
        chars = []
        tags = []
        for item in line.split("\n"):
            char, label = item.split(sep)
            if label.startswith("M"):
                # M -> I
                label = "I" + label[1:]
            chars.append(char)
            tags.append(label)
        X.append("".join(chars))
        y.append(tags)
        assert len(chars) == len(tags)
    if with_labels:
        labels = set(itertools.chain(*y))
        return X, y, sorted(labels)
    return X, y

PATH_MSRA = "dataset/ner/msra/{}.ner"
def load_msra(file, shuffle=True, with_labels=False):
    file = PATH_MSRA.format(file)
    return load_file(file, " ", shuffle, with_labels)

PATH_CPD = "dataset/china-people-daily-ner-corpus/example.{}"
def load_china_people_daily(file, shuffle=True, with_labels=False):
    assert file in ("train", "dev", "test")
    file = PATH_CPD.format(file)
    return load_file(file, " ", shuffle, with_labels)

PATH_CTB6 = "dataset/cws/ctb6/{}.txt"
def load_cws_ctb6(file, shuffle=True, with_labels=False):
    assert file in ("train", "dev", "test")
    file = PATH_CTB6.format(file)
    with open(file, "r") as fp:
        text = fp.read()
    sentences = text.splitlines()
    if shuffle:
        random.shuffle(sentences)
    sentences = [re.split("\s+", sentence) for sentence in sentences]
    sentences = [[w for w in sentence if w] for sentence in sentences]

    X = []
    y = []
    for sentence in sentences:
        X.append("".join(sentence))
        tags = []
        for word in sentence:
            if len(word) == 1:
                tags.append("S")
            else:
                tags.extend(["B"] + ["M"]*(len(word)-2) + ["E"])
        y.append(tags)
        assert len("".join(sentence)) == len(tags)

    if with_labels:
        labels = sorted("BMES")
        return X, y, labels
    return X, y

@timethis
def load_random_sentences(file=None, nums=5000, with_labels=False):
    X = []
    y = []
    for text, tags in gen_random_sentences(nums):
        X.append(text)
        y.append(tags)
    if with_labels:
        labels = sorted("BMES")
        return X, y, labels
    return X, y

def load_sentences():
    # 测试分词效果的句子
    texts = []
    texts.append("守得云开见月明")
    texts.append("乒乓球拍卖完了")
    texts.append("无线电法国别研究")
    texts.append("广东省长假成绩单")
    texts.append("欢迎新老师生前来就餐")
    texts.append("上海浦东开发与建设同步")
    texts.append("独立自主和平等互利的原则")
    texts.append("黑天鹅和灰犀牛是两个突发性事件")
    texts.append("黄马与黑马是马，黄马与黑马不是白马，因此白马不是马。")
    texts.append("The quick brown fox jumps over the lazy dog.")
    texts.append("人的复杂的生理系统的特性注定了一件事情，就是从懂得某个道理到执行之间，是一个漫长的回路。")
    texts.append("除了导致大批旅客包括许多准备前往台北采访空难的新闻记者滞留在香港机场，直到下午2:17分日本亚细亚航空公司开出第一班离港到台北的班机才疏导了滞留在机场的旅客。")
    return texts

def compute_sbme_tags(sentence):
    tags = []
    for word in sentence:
        if len(word) == 1:
            tags.append("S")
        else:
            tags.extend(["B"] + ["M"]*(len(word)-2) + ["E"])
    return np.array(tags)

def gen_random_sentences(nums, maxsize=100):
    try:
        import jieba
    except ImportError as err:
        print(err)
        return ""

    jieba.initialize()
    words = {w:v for w,v in jieba.dt.FREQ.items() if v != 0}
    ws = list(words.keys())
    p = np.array(list(words.values()))
    p = p / np.sum(p)
    for _ in range(nums):
        size = random.randint(10, maxsize)
        sentence = np.random.choice(ws, size, p=p)
        tags = compute_sbme_tags(sentence)
        yield "".join(sentence), tags


if __name__ == "__main__":
    X, y, labels = load_msra("train", with_labels=True)
    print(len(X), len(y))
    print(labels)

    X, y, labels = load_china_people_daily("train", with_labels=True)
    print(len(X), len(y))
    print(labels)

    for text, tags in gen_random_sentences(5):
        print(text)
        print(tags)

    load_random_sentences(nums=100)
