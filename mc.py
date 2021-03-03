import itertools
import numpy as np

class MarkovChain:

    def __init__(self, states):
        self.states = sorted(states) # 状态集
        self.states2id = {i:j for j,i in enumerate(self.states)}
        self.id2states = {j:i for i,j in self.states2id.items()}
        self.size = len(states)
        self.pi = np.zeros((1, self.size))
        self.trans = np.zeros((self.size, self.size))

    def fit(self, seqs):
        # 初始状态的参数学习
        for seq in seqs:
            init = seq[0]
            state_id = self.states2id[init]
            self.pi[0][state_id] += 1
        self.pi = self.pi / np.sum(self.pi)

        # 状态转移矩阵参数学习
        for seq in seqs:
            for state1, state2 in zip(seq[:-1], seq[1:]):
                id1 = self.states2id[state1]
                id2 = self.states2id[state2]
                self.trans[id1][id2] += 1
        self.trans = self.trans / np.sum(self.trans, axis=1, keepdims=True)

    def forecast(self, steps, init=None):
        if init is None:
            init = self.pi
        r = np.zeros((steps+1, self.size))
        r[0] = init
        for i in range(steps):
            r[i+1] = np.dot(r[i], self.trans)
        
        chars = [self.id2states[i] for i in np.argmax(r, axis=1)]
        return chars

    def plot_trans(self):
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except ImportError as err:
            print(err)
            return
        fig, axes = plt.subplots(1, 2)
        ax = sns.heatmap(
            self.pi.T,
            vmin=0.0,
            vmax=1.0,
            fmt=".2f",
            cmap="copper",
            annot=True,
            cbar=True,
            linewidths=0.25,
            ax=axes[0],
            cbar_kws={"orientation": "horizontal"}
        )
        ax.set_title("Initial State")
        ax.set_yticklabels(self.states, rotation=0)
        ax = sns.heatmap(
            self.trans,
            vmin=0.0,
            vmax=1.0,
            fmt=".2f",
            cmap="copper",
            annot=True,
            cbar=True,
            linewidths=0.25,
            ax=axes[1],
            cbar_kws={"orientation": "horizontal"}
        )
        ax.set_title("Transition Matrix")
        ax.set_xticklabels(self.states, rotation=0)
        ax.set_yticklabels(self.states, rotation=0)
        plt.show()

if __name__ == "__main__":
    import dataset
    X, y, labels = dataset.load_china_people_daily("train", with_labels=True)
    states = set(itertools.chain(*y))

    model = MarkovChain(states)
    model.fit(y)
    model.plot_trans()
