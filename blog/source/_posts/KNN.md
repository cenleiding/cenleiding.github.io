---
title: KNN(K-Nearest Neighbor)&K-近邻学习
date: 2018-10-07 14:20:55
tags: 机器学习
categories: 机器学习
keywords: KNN,机器学习,K-近邻学习
description: KNN算法学习笔记
image: /KNN/KNN_1.png
---

>  本文只对KNN进行一些概念介绍。 

**监督学习**：需要已经标注好的训练集。

**懒惰学习**：没有显示的训练过程，训练时间开销为0，待收到测试样本之后再进行处理。

**复杂度**：由于是懒惰学习，因此对于每个测试值都需要遍历训练样本，需要保存所有的训练样本。时间复杂度和空间复杂度都很高。 

**错误率**：当数据足够大，算法保证错误率不会超过贝叶斯算法错误率的两倍

**机制**：给定测试样本，基于某种距离度量    找出训练集中与其最靠近的K个训练样本，然后基于这K个“邻居”的信息进行预测（对于*分类任务*可以采用投票法，对于*回归任务*可以采用平均法，也可以根据距离远近进行加权）。![image](KNN/KNN_1.png)



**距离度量**：

曼哈顿距离：<img src="http://latex.codecogs.com/gif.latex?d(x,y)=\sum\limits_{i=1}^{n}|x_{i}-y_{i}|"/>

*欧氏距离*：<img src="http://latex.codecogs.com/gif.latex?d(x,y)=\sqrt{\sum\limits_{i=1}^{n}(x_{i}-y_{i})^2}"/>用于数值型数据

海明距离：Hamming distance             用于字符串型数据

**参数选择 **：一般情况下，在分类时较大的K值能够减小噪声的影响，但会使类别之间的界限变得模糊。一个较好的K值能通过各种启发式技术来获取。

[**代码实现 **](https://github.com/cenleiding/learning-Machine-Learning/tree/master/KNN)














