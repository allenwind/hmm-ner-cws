import itertools
import random
import math
import numpy as np
from collections import *
from snippets import find_entities, find_words
from snippets import find_entities_chunking, find_words_regions

converts = [find_entities, find_entities_chunking, find_words, find_words_regions]

class HiddenMarkovChain:
	
    def __init__(self, tags, task="NER"):
        # 标签集
        self.tags = sorted(set(tags))
        self.tags2id = {i:j for j,i in enumerate(self.tags)}
        self.id2tags = {j:i for i,j in self.tags2id.items()}
        self.state_size = len(tags)
        assert task in ("NER", "CWS")
        self.task = task
        self.reset()

    def reset(self):
        # 初始状态的参数学习
        self.pi = np.zeros((1, self.state_size))
        # 状态转移矩阵
        self.A = np.zeros((self.state_size, self.state_size))
        # 观察矩阵，稀疏形式
        self.B = defaultdict(Counter)
        self.built = False

    def fit(self, X, y):
        if self.built:
            self.reset()
        # 状态转移矩阵参数学习
        for labels in y:
            for label1, label2 in zip(labels[:-1], labels[1:]):
                id1 = self.tags2id[label1]
                id2 = self.tags2id[label2]
                self.A[id1][id2] += 1
        self.A = self.A / np.sum(self.A, axis=1, keepdims=True)
        self.A[self.A == 0] = 1e-10
        # 观察矩阵参数学习
        for sentence, labels in zip(X, y):
            for char, label in zip(sentence, labels):
                self.B[label][char] += 1
        self.logtotal = {tag:math.log(sum(self.B[tag].values())) for tag in self.tags}
        self.built = True

    def predict(self, X):
        # 给定一个batch的观察序列X，预测各个样本每个时间步隐状态的分值scores
        batch_scores = []
        for sentence in X:
            scores = np.zeros((len(sentence), self.state_size))
            for i, char in enumerate(sentence):
                for j, k in self.B.items():
                    if char not in k:
                        # for OOV问题
                        scores[i][self.tags2id[j]] = -self.logtotal[j]
                    else:
                        scores[i][self.tags2id[j]] = math.log(k[char]) - self.logtotal[j]
            batch_scores.append(scores)
        return batch_scores

    def sampling(self, steps):
        init = self.pi
        rs = np.zeros((steps+1, self.state_size))

    def _sampling_from_multi_category(self, p):
        return np.random.multinomial(1, p)

    def find(self, sentence, with_loc=False):
        # 用viterbi求scores最优路径
        scores = self.predict([sentence])[0]
        # log_trans = np.log(np.where(self.A==0, 0.0001, self.A))
        log_trans = np.log(self.A)
        viterbi = self.viterbi_decode(scores, log_trans)
        viterbi = [self.id2tags[i] for i in viterbi]
        if self.task == "NER":
            if with_loc:
                return find_entities_chunking(viterbi)
            else:
                return find_entities(sentence, viterbi)
        else:
            if with_loc:
                return find_words_regions(sentence, viterbi)
            else:
                return find_words(sentence, viterbi)

    def viterbi_decode(self, scores, trans, return_score=False):
        # 使用viterbi算法求最优路径
        # scores.shape = (seq_len, num_tags)
        # trans.shape = (num_tags, num_tags)
        dp = np.zeros_like(scores)
        backpointers = np.zeros_like(scores, dtype=np.int32)
        dp[0] = scores[0]
        for t in range(1, scores.shape[0]):
            # 扩展维度便于广播，计算上一时间步到当前时间步所有路径分值
            v = np.expand_dims(dp[t-1], axis=1) + trans
            # 保存当前时间步各状态的最优路径
            dp[t] = scores[t] + np.max(v, axis=0)
            backpointers[t] = np.argmax(v, axis=0)

        # 回溯状态
        viterbi = [np.argmax(dp[-1])]
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        if return_score:
            viterbi_score = np.max(dp[-1])
            return viterbi, viterbi_score
        return viterbi

    def plot_trans(self):
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except ImportError as err:
            print(err)
            return
        ax = sns.heatmap(
            self.A,
            vmin=0.0,
            vmax=1.0,
            fmt=".2f",
            cmap="copper",
            annot=True,
            cbar=True,
            # xticklabels=self.tags,
            # yticklabels=self.tags,
            linewidths=0.25,
            cbar_kws={"orientation": "horizontal"}
        )
        ax.set_title("Transition Matrix")
        ax.set_xticklabels(self.tags, rotation=0)
        ax.set_yticklabels(self.tags, rotation=0)
        plt.show()
