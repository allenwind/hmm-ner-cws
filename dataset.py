import random
import itertools

# NER数据加载

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

if __name__ == "__main__":
    X, y, labels = load_msra("train", with_labels=True)
    print(len(X), len(y))
    print(labels)

    X, y, labels = load_china_people_daily("train", with_labels=True)
    print(len(X), len(y))
    print(labels)
