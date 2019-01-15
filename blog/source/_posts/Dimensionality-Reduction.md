---
title: Dimensionality Reduction
date: 2019-01-15 13:23:53
tags: 机器学习
categories: 机器学习
keywords: 降维 机器学习
description: 对降维的简单了解学习
image: /Dimensionality-Reduction/1.jpg
---

![](Dimensionality-Reduction/1.jpg)

​	当我们尝试自己使用机器学习去解决实际问题时，遇到的最大问题往往不是算法上的问题，而是数据上的问题。有时候我们会苦恼于没有数据，而有时候却会因为数据太多而陷入“幸福”的困扰。
​	在之前学习算法时可以看到许多算法都涉及距离计算，而高维空间会给距离计算带来很大的麻烦。实际上，在高维情形下出现的数据样本稀疏、距离计算困难等问题是所有机器学习方法共同面临的严重障碍，被称为**“维度灾难、维度诅咒、维度危机等”** 。

> 降维能带来什么好处？

​	● 降维可以大幅的减少数据存储空间。
​	● 降维可以减少数据计算、训练的时间。
​	● 有些算法对于高维数据处理效果很差。
​	● 有些数据之间存在很强的联系，降维可以减少数据冗余。
​	● 有助于数据的可视化。

> 怎么降维？

​	在机器学习和统计学领域，降维是指在某些限定条件下，降低随机变量个数，得到一组“不相关”主变量的过程。 降维可进一步细分为**变量选择**和**特征提取**两大方法。

接下来，我将从这两个分类出发，简单了解降维的原理和学习基础的算法使用。



## 1. 变量选择

​	变量选择假定数据中包含大量冗余或无关变量（或称特征、属性、指标等），旨在从原有变量中找出主要变量。
​	变量选择一般只能简单的处理一些不是很大的数据集，而且往往需要一些经验支撑。

### 1.1 丢失值比例

​	**Missing Value Ratio**,顾名思义按照变量丢失的情况来选择变量。

​	实际中，我们获得的数据往往是含有损失值的，一般情况下我们会用平均值或者中位数什么的去填补这些缺失值，但是当这个属性的缺失比例过大时，比如大于20%，那么往往没有必要再使用这个属性了。

​	**所以在数据预处理时，可以直接将丢失值比例超出阈值的属性之间抛弃。**



### 1.2 低方差过滤

​	属性值的**方差能够表示一个属性蕴含信息量的多少。**方差小，意味着属性值都很接近，互相之间没有什么区别，也就没法带来有用的信息。

​	**所以在数据预处理时，可以直接将方差很小的属性之间抛弃。**

``` Python
>>> from sklearn.feature_selection import VarianceThreshold
>>> X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
>>> sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
>>> sel.fit_transform(X)
array([[0, 1],
       [1, 0],
       [0, 0],
       [1, 1],
       [1, 0],
       [1, 1]])
```



### 1.3 高相关过滤

​	两个变量之间的高度相关意味着它们具有相似的趋势并且可能携带相似的信息。相似的变量对于提高模型的准确率一般没有帮助，只会降低模型的学习速度。 我们可以计算出变量之间的相关性，如果相关系数超过某个阈值，我们可以删除其中一个变量。

​	可以借助pandas中的.corr获得属性之间的相似度值。
MySQL Notifier

[Main Instruction]
MySQL Notifier 已停止工作

[Content]
Windows 正在查找该问题的解决方案...

[取消]

![](Dimensionality-Reduction/2.png)

​	一般当相似度值大于0.5~0.6时，可以考虑抛弃其中一个变量。

### 1.4 随机森林 

​	我们之前已经比较详细的学习过Random Forest了。知道他是一种集成算法，其每次随机选取一些样本和属性来训练出一个决策树，再根据其准确率为每个决策树分配权重，实际上可以看做一种很巧妙的决策树降低过拟合的方法。因为其随机选择了属性并为之分配了权重，所以我们可以通过它获得属性各自的权重分配情况！

```python
>>> from sklearn.ensemble import RandomForestClassifier

>>> model= RandomForestClassifier(random_state=1)
>>> model.fit(x_train, y_train)
>>> for i, j in sorted(zip(x_train.columns, model.feature_importances_)):
>>>    print(i, j)
ApplicantIncome 0.180924483743
CoapplicantIncome 0.135979758733
Credit_History 0.186436670523
.
.
.
Property_Area_Urban 0.0167025290557
Self_Employed_No 0.0165385567137
Self_Employed_Yes 0.0134763695267
```

​	然后将权重特别小的一些数据进行舍弃。



### 1.5 向后属性消去

​	**Backward Feature Elimination，向后属性消去**，简单来说就是以递归的方式消去属性。

​		● 首先用所有的特征，n个，训练出一个模型，并评估其性能。 
​		● 分别移除一个特征后，再训练出n个模型，评估这些模型的性能，将变化最小的也足够小的模型所对应的特征移除。
​		● 重复这个过程，知道没有特征能够移除。

​	很明显，这个方法十分耗时，毕竟要多次训练评估模型，因此**一般使用线性拟合或逻辑回归作为基础模型。** 

```python 
## 这里使用了SVM
>>> from sklearn.datasets import make_friedman1
>>> from sklearn.feature_selection import RFE
>>> from sklearn.svm import SVR
>>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
>>> estimator = SVR(kernel="linear")
>>> selector = RFE(estimator, n_features_to_select=5, step=1)
>>> selector = selector.fit(X, y)
>>> selector.support_ 
array([ True,  True,  True,  True,  True, False, False, False, False,
       False])
## 还能获得各个属性的排名
>>> selector.ranking_
array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])
```



### 1.6 向前属性消去

​	**Forward Feature Selection，向前属性消去，**看名字就知道和向后属性消去类似。不过就是变成每次挑一个对模型优化最大的特征增加。

​	同样这个方法十分耗时，只能用于属性比较少的情况。

```python
>>> from sklearn.datasets import load_iris
>>> from sklearn.feature_selection import SelectKBest ## 只保留最好的k个属性
>>> from sklearn.feature_selection import f_regression 
>>> iris = load_iris()
>>> X, y = iris.data, iris.target
>>> X.shape
(150, 4)
>>> X_new = SelectKBest(f_regression, k=2).fit_transform(X, y) 
>>> X_new.shape
(150, 2)
```



## 2. 特征提取

​	特征提取可以看作变量选择方法的一般化：变量选择假设在原始数据中，变量数目浩繁，但只有少数几个真正起作用；而特征提取则认为在所有变量可能的函数(比如这些变量各种可能的线性组合)中，只有少数几个真正起作用。

### 2.1 ★主成分分析PCA

​	**Principal Component Analysis,PCA,主成分分析**。 说到降维毫无疑问，最常用也是最出名就是这货。

![](Dimensionality-Reduction/3.gif)

★ 首先让我们顺着推导大概了解一下，PCA是怎么来的。

​	PCA实际上做的就是**寻找一个低维空间，这个空间能使得原高维数据进行投影之后仍保留大量信息。**

> 那么怎么体现投影之后仍保留大量信息？

​	PCA认为这样的空间、超平面大概应具有类似的性质：

​	● 最近重构性：样本点到超平面的距离都足够近，也就意味着样本点尽量少移动。
​	● 最大可分性：样本点在超平面上的投影尽可能分开，因为分的越开方差越大信息包含越多。

​	PCA就是基于这两点出发进行推导、求取低维空间。神奇的是，不论选哪个性质开始推导，结果都一样。

> 推导结果是什么？

​	在推导前当然要进行一些假设：

​	● 假定数据进行了中心化（即所有坐标都减去了均值）$X$，这一步的目的只是为了后续推导方便。
​	● 假定新的空间坐标系为 $W:\{\omega_1,\omega_2,…,\omega_d\}$ ，且$\omega$都为标准正交基向量。也就是说降维到d维。
​	● 从上面任意一个性质出发推导，…跳过具体推导…,最后可以求得$XX^T\omega_i=\lambda_i\omega_i$ 

​	定睛一看，我的天，这不就说 $\omega$ 是$XX^T$的特征向量！ 而且为了保留尽量多的信息，我们选择那些特征值大的特征向量。

> 具体步骤

​	● 设有m条n维数据。
​	● 将原始数据按列组成n行m列矩阵$X$。
​	● 将$X$的每一行（代表一个属性字段）进行零均值化，即减去这一行的均值。
​	● 求出协方差矩阵$XX^T$。
​	● 对矩阵进行特征值分解，特征值及对应的特征向量。
​	● 将特征向量按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵$P$
​	● $Y=PX$即为降维到k维后的数据。	



★ 好吧，顺着推导有点难，让我们试着根据结果，来尝试想象PCA到底在干什么。

> 理解一：

​	我们知道一个属性的信息量多少可以用方差、标准差来表示。

​		$var(X)=\frac{\sum_{i=1}^n(X_i-\bar X)(X_i-\bar X)}{n-1}$

​	而两个属性之间的相似度则可以用协方差来表示。正则正相关，负则负相关，0则无关。

​		$cov(X,Y)=\frac{\sum_{i=1}^{n}(X_i-\bar X)(Y_i-\bar Y)}{n-1}$

​	而我们在PCA中，先行对数据进行了中心化，这就使得矩阵$XX^T$实际上就是协方差矩阵！

​		$C=\Bigg(\begin{array}\ cov(X,X)\ cov(X,Y) \ cov(X,Z)\\ cov(Y,X)\ cov(Y,Y) \ cov(Y,Z)\\ cov(Z,X)\ cov(Z,Y) \ cov(Z,Z)\\\end{array} \Bigg)$

​	协方差矩阵代表了各个属性之间的关系，而协方差矩阵又是实对称矩阵，意味着一定有n个正交特征向量来表示这个“关系空间”。又因为特征值大的特征向量意味着在这个方向信息量就大，也就是说能获得更多的属性关系。所以我们就对协方差矩阵进行了特征分解，然后选取最大的k个特征值的特征向量来构建新的低维空间。
​	所以可以这么理解：**PCA通过协方差矩阵学得了属性之间的特征，然后保留信息量多的特征，舍弃信息量少的特征(向此方向进行压缩)。** 虽然不严谨~

> 理解二：

​	我们知道最后我们通过$Y=PX$将数据降到k维。而P是一个k*n的矩阵。这个降维可以看成一个**线性变换**！ 细节到每一条数据，从原来的1\*n变为1\*k，实则进行了k次线性变换，每次变换都是原数据乘以各个属性的权重。所以也可以这么理解：**PCA学得了属性之间的线性关系，原空间用单一属性做基，现在则用属性的线性组合做基，这样一来可以将原本比较接近的属性进行合并！**



★  杂记

​	● 我们常常会看到PCA使用了**Singular Value Decomposition（SVD，奇异值分解）**， 这个SVD是什么？在上面我们求矩阵特征向量用的是特征分解，需要求$XX^T$ ,但是在这一步中，一些非常小的数容易在平方中丢失！而SVD也是一种矩阵分解方法且也能得到特征向量，但不用求$XX^T$，所以**SVD更稳定，对于稀疏矩阵效果更好！**

  	● PCA不要随便使用，他相当于提前进行了归纳，因而也有可能带来过拟合问题，一般只有到算法效果不好时再考虑PCA等提前优化。

​	● PCA不只是能用来提前优化，一些时候还能帮助可视化数据。

```python
from sklearn import datasets
from sklearn.decomposition import PCA,TruncatedSVD

iris = datasets.load_iris() # 150*4
pca = PCA(n_components=3).fit_transform(iris.data) # 150*3
# 注意当0<n_components<1时，则会考虑信息保留量。0.1意味着至少保留原信息的90%
pca = PCA(n_components=0.1).fit_transform(iris.data) # 150*3
# SVD能够用于稀疏矩阵
svd = TruncatedSVD(n_components=3).fit_transform(iris.data)
```



















## 参考

● https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/

