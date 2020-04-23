Word2Vec
[Word2Vec Tutorial - The Skip-Gram Model · Chris McCormick](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)


假设共有N个词，其中第1个是*ant*，one-hot后为(1, 0, 0, …, 0).T (shape: N*1)

神经网络有3层，输入，隐藏层，输出。
input -> dense(300) -> dense(N, ‘softmax’) (dense没有激活函数)（300是假设特征向量为300维）

则每层含义可看做
```
输入：N*1, 单词ant的one-hot vector
隐藏层：300 * N, 第i列为对应第i个单词的feature_vector_1。因为输入为(1, 0, 0, …, 0).T，所以计算结果即第一列，即ant的feature vector，记做f1，形状是300 * 1
输出层：N * 300, 第i行为第i个单词的feature_vector_2（和feature_vector_1其实没什么关系）,计算结果为（…dot(第i行, f1)…）.T
输出：N*1，第i个值表示在ant的周围随机选一个词，刚好是第i个单词的概率。结果含义即输入词周围出现其他词分别的概率，即context。
```

feature_vector_2和feature_vector_1其实没什么关系，可以看做在提取出的不同的feature vector。
输出结果也不重要，最终留下的是feature_vector_1（其实2也可以应该）

这个模型经过训练得到的feature vector关键是符合规律：
对于两个输入分别对应的最终结果context，以及分别对应隐藏层的相应列，可看做其feature vector。有，context的相似度，与feature vector相似度成正比。 而刚好想要的，就是这样符合这样规律的一组向量，隐藏层参数刚好满足，所以拿来作为feature vector。


哎感觉不知道怎么说。 还是尽量总结下，其实这里的思想是这样的：
神经网络每层的变化其实是映射，而输入的单词，经过one-hot后，互相正交，看不出互相关系的时候，通过构造一个输入输出关系，模型训练结果中，输入被映射到隐藏层的300维后，这300维的向量作为原先one-hot的替代，他们互相不正交，蕴含了更多信息。至少符合预设的输入输出关系规律，那么就拿去做feature vector。
