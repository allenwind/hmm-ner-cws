# hmm-ner-cws



## Markov Chain

HMM建立在MarkovChain上，下图是MarkovChain的参数可视化：

![](asset/mc-params.png)

## HMM

经典的HMM模型用于NER和CWS任务。

使用HMM进行中文分词，

```bash
$ python3 task_cws.py
```

使用HMM进行NER，

```bash
$ python3 task_ner.py
```



状态矩阵的可视化：

![](asset/hmm-ner-tranition-matrix.png)
