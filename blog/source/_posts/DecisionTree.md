---
title: DecisionTree
date: 2018-10-07 15:07:16
tags: 机器学习
categories: 机器学习
keywords: DecisionTree,机器学习,决策树
description: DecisionTree算法学习笔记
image: /DecisionTree/DT_1.jpg
---

# 决策树 DecisionTree

## 1.什么是决策树？

![西瓜决策树](DecisionTree/DT_1.jpg)

决策树很好理解，就是模拟了人们平时的思考方式，根据事务的特征，一层层的对事务的属性进行分析划分，最后实现对事务的归类。

`算法用途` ：分类问题，数据挖掘

`优点` ：计算复杂度不高，*易于理解数据中蕴含的知识信息*。

`缺点` ：容易出现过拟合

`适用数据类型` ：数值型、标称型 

## 2.属性划分

