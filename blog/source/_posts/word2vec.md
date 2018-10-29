---
title: word2vec词向量
date: 2018-10-09 21:43:32
tags: 机器学习
categories: 机器学习
keywords: word2vec,词向量,word embedding,词嵌入,skip-gram,词向量空间
description: word2vec词嵌入
image: word2vec/word2vec_3.png
---

​word2vec整体思路可以说是很简单的（当然可能是因为我学的比较浅 😢），但十分的有趣，直观。

详细的可以看 TensorFlow 官网给出的教程，可以自己具体实现一下[教程](http://www.tensorfly.cn/tfdoc/tutorials/word2vec.html)

这里只进行简单的介绍，对一些重要概念进行梳理：

![word2vec](word2vec/word2vec_1.png)

> 首先，为什么要进行word embeddings?

我们既然要处理自然语言，那么首先就需要将其转换成电脑能看懂的形式。最简单的就是用**one-hot encode** 的方式，及每个向量只有一位为1其余为0，用一个向量表示一个目标（注意这个目标理解为token，可以是字也可以是词，句子、文章什么的这里不考虑~）。但是这样表示的话，都是稀疏向量（sparse vector），不但占位子而且向量之间都毫无关系，这对于后续的处理无疑是很不利的。我们需要一个能够表示各个词之间关系的稠密向量（dense vector）来表示目标。于是就需要进行word embedding,来得到词向量空间。



> 能够用向量表示词之间关系的原理是什么？

依赖于**分布式假设 distributional hypothesis** ，具体可分为两类：

**基于计数的方法**：计算某词汇与其邻近词汇在一个大型语料库中共同出现的频率及其他统计量，然后将这些统计量映射到一个小型且稠密的向量中。

**基于预测方法**：试图直接从某词汇的邻近词汇对其进行预测，在此过程中利用已经学习到的小型且稠密的*嵌套向量*。

😂因为懒，所以这里只了解了一下基于预测的方法。



> 怎么基于预测？

预测，预测。很明显就是要用到这个词的周围词了。预测方法分为两类：

**连续词袋模型（CBOW）**：根据上下文词汇来预测目标词汇。对于很多分布式信息进行了平滑处理（例如将一整段上下文信息视为一个单一观察量）。很多情况下，对于**小型的数据集**，这一处理是有帮助的。

**Skip-Gram模型**：通过目标词汇来预测上下文词汇。将每个“上下文-目标词汇”的组合视为一个新观察量，这种做法在**大型数据集**中会更为有效。

简单来说就是对于"我爱你"这句，爱作为目标词，cbow模型需要由‘爱’推出‘我’、‘你’，而skip-gram模型则是要由‘我’推出‘爱’，‘你’推出‘爱’。在代码具体实现时，只要稍作修改即可。



> 怎么评估这个预测值（代价函数）？

**softmax function ？**对于这种概率型的问题，softmax function 当然可以。但用softmax function在这里有个问题，为了求出概率，他需要将所有的其他词的概率得分都求一遍,虽然这样会比较准确，但开销实在太大，除非你电脑真牛逼，不然没有必要。

**负采样 negative sample ：** 我们没有必要对所有上下文的词都进行评分，我们可以把问题简化为：现在有我们的预测值，目标单词，和一些噪声单词，我们只要目标单词被分配到较高的概率，同时噪声单词的概率很低，我们就可以高兴的认为这个预测是好的。然后这个分配概率怎么求？这可以当做一个简单的二分类逻辑回归问题。于是我们就愉快的得到了一个简单的代价函数。



> 具体的实现是怎么样的？

![img](word2vec/word2vec_2.png)



**projection layer：** 这一层输入的是一个词的one-hot编码，大小为[1*词总量]，为了简单也可以理解为每个词的唯一id值。

**g(embedding)：** 这是一个[词总量 * 词向量长度]的矩阵参数，其中词向量长度是我们自己定义的。这个矩阵为每个词都分配了一个指定长度向量，说白了就是一张映射表，也就是我们期望得到的词向量空间。

**hidden layer：** 这一层就是一个相应的词向量，作为预测值。

**noise classifier：** 这一层是一个逻辑回归。设有参数和偏置。TensorFlow中可以直接用nce函数实现。

**训练：** 使用常规的梯度下降，调整各个参数值。



> 一些心得

**nec 函数：** 

```python
def nce_loss(weights, biases, inputs, labels, num_sampled, num_classes,
             num_true=1,
             sampled_values=None,
             remove_accidental_hits=False,
             partition_strategy="mod",
             name="nce_loss")
```

weights,biases:逻辑回归的参数

input,labels:预测值，真实值

num_sampled:采样数

num_classes：词总数

这些参数都没什么问题，但是sampled_values为什么默认为None?看源码可以发现，其默认使用log_uniform_candidate_sampler函数采样，这个函数P(k)，k越大，被采样到的概率越小。这里的k就是词的编号。很明显我们需要让高频词被采到的概率更大一点，所以在实现的时候需要将高频词放在前面使其编号小。



**gensim：**

虽然自己也跟着教程实现了一下程序，也能用，但实际使用时当然选择别人提供的工具模块。

于是实际用了gensim来进行词嵌入。接口是否简单，没什么好讲。

但需要注意的是gensim里训练出来的词向量空间没有包括低频词！！！！而TensorFlow的demo里是将低频词统一转换成同一个标志一起处理的。而如果没有低频词的向量，那就没法完整的对所有单词进行转换了，所以用gensim时还要自己对低频词进行处理，可以事先预处理文本也可以最后将低频词赋值为0向量。

另外gensim训练出来的是一个model，为了能以后使用方便，最好将其转化为word2id和word2vec两个文件。



最后放张图,什么都没优化，单纯分词+词嵌入：

![img](word2vec/word2vec_3.png)



[程序代码](https://github.com/cenleiding/learning-Machine-Learning/tree/master/gensim_w2v)

