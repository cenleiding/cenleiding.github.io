---
title: TensorFlow 常用函数整理
date: 2018-10-10 21:15:00
tags: TensorFlow.API
categories: 机器学习
keywords: [TensorFlow,常用函数]
description: 对TensorFlow中常用函数的整理
image: /TensorFlow-常用函数/0.jpg
---

刚开始学习TensorFlow，虽然有[官方API](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf)，但是分类实在对新手不友好。。。

所以打算整理一下经常使用的API，方便查找~但由于篇幅有限，更详细的使用介绍请查看官网。

在这先感谢一下[林海山波](https://blog.csdn.net/lenbow) 的整理~



## 运算操作



###基础运算

| 函数                                       | 功能                                       |
| ---------------------------------------- | ---------------------------------------- |
| **tf.matmul(a,b,transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None)** | 矩阵乘法  a*b                                |
| tf.matrix_inverse(input, adjoint=None, name=None) | 矩阵求逆                                     |
| tf.add(x, y, name=None)                  | 求和                                       |
| tf.sub(x, y, name=None)                  | 减法                                       |
| **tf.multiply(x, y, name=None)**         | 乘法 a·b                                   |
| tf.div(x, y, name=None)                  | 除法                                       |
| tf.mod(x, y, name=None)                  | 取模                                       |
| tf.abs(x, name=None)                     | 求绝对值                                     |
| tf.neg(x, name=None)                     | 取负 (y = -x)                              |
| tf.sign(x, name=None)                    | 返回符号 y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0. |
| tf.inv(x, name=None)                     | 取反                                       |
| **tf.square(x, name=None)**              | 计算平方 (y = x * x = x^2)                   |
| tf.round(x, name=None)                   | 舍入最接近的整数 <br /> #‘a’ is [0.9, 2.5, 2.3, -4.4]<br /> tf.round(a) ==> [ 1.0, 3.0, 2.0, -4.0 ] |
| tf.sqrt(x, name=None)                    | 开根号                                      |
| tf.pow(x, y, name=None)                  | 幂次方<br />tensor ‘x’ is [[2, 2], [3, 3]]<br />tensor ‘y’ is [[8, 16], [2, 3]]<br /> tf.pow(x, y) ==> [[256, 65536], [9, 27]] |
| tf.exp(x, name=None)                     | 计算e的次方                                   |
| tf.log(x, name=None)                     | 计算log，一个输入计算e的ln，两输入以第二输入为底              |
| tf.maximum(x, y, name=None)              | 返回最大值 (x > y ? x : y)                    |
| tf.minimum(x, y, name=None)              | 返回最小值 (x < y ? x : y)                    |
| tf.cos(x, name=None)                     | 三角函数cos                                  |
| tf.sin(x, name=None)                     | 三角函数sin                                  |
| tf.tan(x, name=None)                     | 三角函数tan                                  |
| tf.atan(x, name=None)                    | 三角函数ctan                                 |

###reduce 运算

| 函数                                       | 功能            |
| ---------------------------------------- | ------------- |
| tf.reduce_sum (input_tensor,axis=None,keepdims=None,name=None,     reduction_indices=None,keep_dims=None) | 求和            |
| tf.reduce_mean (input_tensor,axis=None,keepdims=None,name=None,     reduction_indices=None,keep_dims=None) | 求平均           |
| tf.reduce_min (input_tensor,axis=None,keepdims=None,name=None,     reduction_indices=None,keep_dims=None) | 最小值           |
| tf.reduce_max (input_tensor,axis=None,keepdims=None,name=None,  reduction_indices=None,keep_dims=None) | 最大值           |
| tf.reduce_join (inputs,axis=None,keep_dims=False,separator='',name=None,     reduction_indices=None) | 连接            |
| tf.reduce_all (input_tensor,axis=None,keepdims=None,name=None,     reduction_indices=None,keep_dims=None) | 与操作           |
| tf.reduce_any (input_tensor,axis=None,keepdims=None,name=None,     reduction_indices=None,keep_dims=None ) | 或操作           |
| tf.reduce_logsumexp(input_tensor,axis=None,keepdims=None,name=None,     reduction_indices=None,keep_dims=None ) | log(sum(exp)) |

```python
x = tf.constant([[1, 1, 1], [1, 1, 1]])
tf.reduce_sum(x)  # 6
tf.reduce_sum(x, 0)  # [2, 2, 2]
tf.reduce_sum(x, 1)  # [3, 3]
tf.reduce_sum(x, 1, keepdims=True)  # [[3], [3]]
tf.reduce_sum(x, [0, 1])  # 6

x = tf.constant([[1., 1.], [2., 2.]])
tf.reduce_mean(x)  # 1.5
tf.reduce_mean(x, 0)  # [1.5, 1.5]
tf.reduce_mean(x, 1)  # [1.,  2.]

# tensor `a` is [["a", "b"], ["c", "d"]]
tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
tf.reduce_join(a, [0, 1]) ==> "acbd"
tf.reduce_join(a, [1, 0]) ==> "abcd"
tf.reduce_join(a, []) ==> [["a", "b"], ["c", "d"]]
tf.reduce_join(a) = tf.reduce_join(a, [1, 0]) ==> "abcd"
```



## 张量操作



### 数据类型转换

| 函数                                       | 功能                   |
| ---------------------------------------- | -------------------- |
| tf.string_to_number(string_tensor, out_type=None, name=None) | 字符串转为数字              |
| tf.to_double(x, name=’ToDouble’)<br /> tf.to_float(x, name=’ToFloat’)<br /> tf.to_int32(x, name=’ToInt32’)<br /> tf.to_int64(x, name=’ToInt64’) | 转为相应数字类型             |
| **tf.cast(x, dtype, name=None)**         | 将x或者x.values转换为dtype |

```python
x = tf.constant([1.8, 2.2], dtype=tf.float32)
tf.cast(x, tf.int32)  # [1, 2], dtype=tf.int32
```



### 形状操作

| 函数                                       | 功能                                       |
| ---------------------------------------- | ---------------------------------------- |
| tf.shape(input, name=None)               | 返回数据的shape                               |
| tf.size(input, name=None)                | 返回数据的元素数量                                |
| tf.rank(input, name=None)                | 返回张量的秩。<br /> 注意：张量的秩与矩阵的秩不一样。<br />指能够唯一确定张量元素的最小索引数<br />秩也被称为 “order”，“degree” 或 “ndims”。 |
| **tf.reshape(tensor, shape, name=None)** | 变形<br />如果shape的一个分量是特殊值-1，<br />则计算该维度的大小，以使总大小保持不变。<br />至多能有一个shape的分量可以是-1。 |

```python
t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
tf.shape(t)  # [2, 2, 3]

t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
tf.size(t)  # 12

# shape of tensor 't' is [2, 2, 3]
t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
tf.rank(t)  # 3

# tensor 't' is [[[1, 1, 1],
#                 [2, 2, 2]],
#                [[3, 3, 3],
#                 [4, 4, 4]],
#                [[5, 5, 5],
#                 [6, 6, 6]]]
# tensor 't' has shape [3, 2, 3]
# pass '[-1]' to flatten 't'
tf.reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
tf.reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
```



### 切片与合并

| 函数                                       | 功能              |
| ---------------------------------------- | --------------- |
| tf.slice(input_, begin, size, name=None) | 对tensor进行切片操作   |
| **tf.split(value,num_or_size_splits,axis=0,num=None,name='split' )** | 对tensor进行分解操作   |
| **tf.concat(values,axis,name='concat')** | 对tensor进行拼接操作   |
| tf.transpose(a,perm=None,name='transpose',conjugate=False) | 对tensor进行转置操作   |
| tf.gather(params,indices,validate_indices=None,name=None,axis=0 ) | 将指定索引切片合并成新的张量  |
| tf.one_hot(indices,depth,on_value=None,off_value=None,axis=None,     dtype=None, name=None ) | 返回一个 one-hot 张量 |

```python
t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]])
tf.slice(t, [1, 0, 0], [1, 1, 3])  # [[[3, 3, 3]]]
tf.slice(t, [1, 0, 0], [1, 2, 3])  # [[[3, 3, 3],
                                   #   [4, 4, 4]]]
tf.slice(t, [1, 0, 0], [2, 1, 3])  # [[[3, 3, 3]],
                                   #  [[5, 5, 5]]]

# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
tf.shape(split0)  # [5, 4]
tf.shape(split1)  # [5, 15]
tf.shape(split2)  # [5, 11]
# Split 'value' into 3 tensors along dimension 1
split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
tf.shape(split0)  # [5, 10]

t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
tf.shape(tf.concat([t3, t4], 0))  # [4, 3]
tf.shape(tf.concat([t3, t4], 1))  # [2, 6]

x = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.transpose(x)  # [[1, 4]
                 #  [2, 5]
                 #  [3, 6]]
    
indices = [0, 1, 2]
depth = 3
tf.one_hot(indices, depth)  # output: [3 x 3]
# [[1., 0., 0.],
#  [0., 1., 0.],
#  [0., 0., 1.]]
indices = [0, 2, -1, 1]
depth = 3
tf.one_hot(indices, depth,
           on_value=5.0, off_value=0.0,
           axis=-1)  # output: [4 x 3]
# [[5.0, 0.0, 0.0],  # one_hot(0)
#  [0.0, 0.0, 5.0],  # one_hot(2)
#  [0.0, 0.0, 0.0],  # one_hot(-1)
#  [0.0, 5.0, 0.0]]  # one_hot(1)
```



### 索引提取

| 函数                                       | 功能       |
| ---------------------------------------- | -------- |
| **tf.argmin(input,axis=None,name=None,dimension=None,output_type=tf.int64)** | 最小值索引    |
| **tf.argmax(input,axis=None,name=None,dimension=None,output_type=tf.int64)** | 最大值索引    |
| tf.unique(x,out_idx=tf.int32,name=None ) | 1-D张量不同值 |

```python
# tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx = unique(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
```





## 神经网络 [.nn模块](https://www.tensorflow.org/versions/r1.12/api_guides/python/nn)



###激活函数 Activation Functions

| 函数                                       | 功能                                       |
| ---------------------------------------- | ---------------------------------------- |
| **tf.nn.relu(features,name=None )**<br />tf.nn.relu6(features,name=None )<br />tf.nn.relu_layer(x,weights, biases,name=None ) | max(features,0)<br /> min(max(features, 0), 6)<br /> Relu(x * weight + biases) |
| tf.nn.elu(features,name=None)            | exp(features)-1 if < 0, features otherwise. |
| **tf.nn.sigmoid(x,name=None)**           | y = 1 / (1 + exp(-x))                    |
| **tf.nn.tanh(x,name=None )**             | 双曲正切激活函数                                 |
| tf.math.softplus(features,name=None)     | log(exp(features) + 1)                   |
| tf.nn.softsign(features,name=None)       | features / (abs(features) + 1)           |
| **tf.nn.dropout(x,keep_prob,noise_shape=None,seed=None,     name=None )** | 独立的，元素放大(1-keep_prob)或者变为0。用于防止过拟合       |
| tf.nn.bias_add(value,bias,data_format=None,name=None ) | Adds bias  to value                      |

###卷积函数 Convolution

| 函数                                       | 功能         |
| ---------------------------------------- | ---------- |
| tf.nn.conv2d<br />tf.nn.conv1d<br />tf.nn.conv3d | 卷积操作，跟多见官网 |



###数据标准化 Normalization

当输入具有不同的比例时，归一化可用于防止神经元饱和，并有助于一般化。

| 函数                                       | 功能                                       |
| ---------------------------------------- | ---------------------------------------- |
| tf.math.l2_normalize <br />tf.linalg.l2_normalize<br />tf.nn.l2_normalize(x,axis=None,epsilon=1e-12,name=None,dim=None )<br /> | 对维度dim进行L2范式标准化<br />output = x/sqrt(max(sum(x**2), epsilon)) |
| tf.nn.moments(x,axes,shift=None,name=None,     keep_dims=False ) | 返回2个张量：均值和方差                             |
| tf.nn.sufficient_statistics(x,axes,shift=None,keep_dims=False,     name=None ) | 计算与均值和方差有关的完全统计量<br />返回4维元组,元素个数，元素总和，元素的平方和，shift结果 |
| tf.nn.local_response_normalization<br />tf.nn.normalize_moments<br />tf.nn.weighted_moments<br />tf.nn.fused_batch_norm<br />tf.nn.batch_normalization<br />tf.nn.batch_norm_with_global_normalization | 具体见官网                                    |

###损失函数 Losses

损失操作测量两个张量之间或张量和零之间的误差。 这可用于在回归任务中测量网络的准确性或用于正则化目的（权重衰减）。

| 函数                                       | 功能                       |
| ---------------------------------------- | ------------------------ |
| tf.nn.l2_loss(t,name=None)               | output = sum(t ** 2) / 2 |
| tf.nn.log_poisson_loss(targets, log_input,compute_full_loss=False,     name=None ) | 计算给定 log_input的对数泊松损失    |

###分类函数 Classification

这部分函数顺便都将各个样本的交叉熵都算完了~~

| 函数                                       | 功能                                       |
| ---------------------------------------- | ---------------------------------------- |
| **tf.nn.sigmoid_cross_entropy_with_logits**(<br /> _sentinel=None,  <br /> labels=None,   <br /> logits=None,     <br /> name=None ) | 计算 sigmoid的交叉熵<br />x = logits, z = labels<br />max(x, 0) - x * z + log(1 + exp(-abs(x)))<br />labels一般为one-hot形式也可以是概率分布<br />logits为网络获得的结果 |
| tf.nn.softmax(    <br /> logits,     <br /> axis=None,   <br /> name=None,   <br /> dim=None ) | softmax = exp(logits)/sum(exp(logits), axis)<br />计算各自概率 |
| tf.nn.log_softmax(   <br />  logits,   <br />  axis=None,   <br />  name=None,  <br />  dim=None ) | logsoftmax = logits - log(sum(exp(logits), axis)) |
| **tf.nn.softmax_cross_entropy_with_logits_v2**(<br />  _sentinel=None,   <br />  labels=None,   <br />  logits=None,   <br />  dim=-1,   <br />  name=None ) | 计算softmax的交叉熵<br />注意不需要在外部对logits进行softmax操作！ |
| tf.nn.sparse_softmax_cross_entropy_with_logits | 计算稀疏softmax交叉熵                           |
| tf.nn.weighted_cross_entropy_with_logits | 计算加权交叉熵                                  |

### 符号嵌入 Embeddings

多用于词嵌入

| 函数                                       | 功能                                       |
| ---------------------------------------- | ---------------------------------------- |
| **tf.nn.embedding_lookup**(    <br /> params,   <br /> ids,  <br /> partition_strategy='mod',   <br /> name=None,   <br /> validate_indices=True,   <br /> max_norm=None ) | Looks up ids in a list of embedding tensors.<br />可以理解为tf.gather的一种应用。 |
| tf.nn.embedding_lookup_sparse(   <br />  params,  <br />  sp_ids,   <br />  sp_weights,    <br />  partition_strategy='mod',   <br />  name=None,   <br />  combiner=None,   <br />  max_norm=None ) | Computes embeddings for the given ids and weights. |

### 循环神经网络 RNN

.nn模块只有一部分rnn功能。更全的功能详见tf.contrib.rnn模块~~~~~~~~~



### 候选采样 Candidate Sampling

对于有巨大量的多分类与多标签模型，如果使用全连接softmax将会占用大量的时间与空间资源，所以采用候选采样方法仅使用一小部分类别与标签作为监督以加速训练。

最经典的就是用于词向量

| 函数                                       | 功能                                       |
| ---------------------------------------- | ---------------------------------------- |
| tf.nn.nce_loss(...)                      | 返回noise-contrastive的训练损失结果               |
| tf.nn.sampled_softmax_loss()             | 返回sampled softmax的训练损失                   |
| tf.nn.uniform_candidate_sampler<br /> tf.nn.log_uniform_candidate_sampler<br /> tf.nn.learned_unigram_candidate_sampler<br /> tf.nn.fixed_unigram_candidate_sampler | TensorFlow提供的采样器，用于在使用上述采样损失函数之一时随机采样候选类。 |

## 保存和恢复

### train.Saver

推荐可以看这篇[文章](https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/)

| Class  =>  tf.train.Saver                |                                          |
| ---------------------------------------- | ---------------------------------------- |
| __init__(    <br />   var_list=None,   <br />   max_to_keep=5,    <br />   keep_checkpoint_every_n_hours=10000.0,    <br />   filename=None <br />   ……) | **var_list**：指定将要保存和恢复的变量。<br />可以是list或dict。默认全加。<br />**max_to_keep**：保存最近检查点的数量<br /> **keep_checkpoint_every_n_hours**：多久自动保存一次<br /> …… |
| save(    <br />  sess,    <br />  save_path,    <br />  global_step=None, <br />  latest_filename=None,   <br />  meta_graph_suffix='meta',   <br />  write_meta_graph=True,   <br />  write_state=True,    <br />  strip_default_attrs=False ) | **sess**:变量所在会话<br /> **save_path**: 保存地址<br /> global_step:几步保存一次<br /> write_meta_graph:是否保存图元文件<br /> …… |
| restore( <br /> sess,   <br /> save_path ) | **sess**:变量恢复会话<br />**save_path**: 文件地址<br /> |
| ……                                       |                                          |

```python 
######################################  存储的文件  #############################################
# v<0.11
# checkpoint   my_test_model.meta   my_test_model.ckpt
# v>=0.11
# checkpoint   my_test_model.meta   my_test_model.data-00000-of-00001    my_test_model.index   

########################################  保存  ##############################################  
saver = tf.train.Saver()
#saver = tf.train.Saver([w1, w2])
#saver = tf.train.Saver({'w1': w1, 'w2': w2})
#saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)

save_path = saver.save(sess, "my-test-model")
#saver.save(sess, 'my_test_model',global_step=1000)

#########################################  恢复  ###############################################

#只恢复变量，模型已经提前手动撸完，而且要保证变量一致
saver = tf.train.Saver()
saver.restore(sess, "my-test-model")
saver。restore(sess,tf.train.latest_checkpoint('/'))#自动获取目录下最近的模型

#恢复模型和数据
new_saver = tf.train.import_meta_graph('my_test_model-1000.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./')) #目录即可

#使用恢复得到的模型
graph = tf.get_default_graph()
# access and create placeholders variables and create feed-dict to feed new data
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
feed_dict ={w1:13.0,w2:17.0}
# access the op that you want to run. 
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
# Add more to the current graph
add_on_op = tf.multiply(op_to_restore,2)

print sess.run(add_on_op,feed_dict)
```



### saved_model

主要用于TensorFlow/serving。实现开发和使用的解耦。

| tf.saved_model.builder.SavedModelBuilder | 建立协议缓冲区，能够保存多个原图，同时共享变量和资源。              |
| ---------------------------------------- | ---------------------------------------- |
| **__ init __(export_dir)**               | **export_dir：保存路径**                      |
| **add_meta_graph_and_variables( <br />sess,   <br />tags,    <br />signature_def_map=None,  <br />main_op=None<br /> ……   )** | 当第一个元图保存时，**必须使用一次！**<br />sess:当前会话<br />tags:用于标识元图的set<br />signature_def_map: signature列表<br />main_op：加载后自动运行的操作 |
| add_meta_graph(<br />tags, <br />signature_def_map=None, <br />main_op=None, <br />…… ) | 与上一个类似，用于加载其他元图。                         |
| **save(as_text=False) **                 | 保存                                       |

| tf.saved_model.signature_def_utils       | 构造签名                                     |
| ---------------------------------------- | ---------------------------------------- |
| **tf.saved_model.signature_def_utils.build_signature_def(<br />inputs=None,<br />outputs=None,<br />method_name=None )** | imputs:字符串与输入张量的映射<br />outputs：字符串与输出张量的映射<br /> method_name：方法名 |
| ……                                       |                                          |

| tf.saved_model.utils                     |                   |
| ---------------------------------------- | ----------------- |
| **tf.saved_model.utils.build_tensor_info(tensor)** | 建立张量原型，tensor:张量名 |
| tf.saved_model.get_tensor_from_tensor_info(<br />tensor_info,<br />graph=None,<br />import_scope=None ) |                   |

| constants                                | 一些常量         |
| ---------------------------------------- | ------------ |
| tf.saved_model.signature_constants<br />`CLASSIFY_INPUTS`<br />`CLASSIFY_METHOD_NAME`<br />`CLASSIFY_OUTPUT_CLASSES`<br />`CLASSIFY_OUTPUT_SCORES`<br />`DEFAULT_SERVING_SIGNATURE_DEF_KEY`<br />`PREDICT_INPUTS`<br />`PREDICT_METHOD_NAME`<br />`PREDICT_OUTPUTS`<br />`REGRESS_INPUTS`<br />`REGRESS_METHOD_NAME`<br />`REGRESS_OUTPUTS` | 对signature命名 |
| tf.saved_model.tag_constants<br />`GPU`<br />`SERVING`<br />`TPU`<br />`TRAINING` | 对tag命名       |



```python
#####################################   保存  ##################################################
## 构造器
builder = tf.saved_model.Builder(export_dir)

with tf.Session(graph=tf.Graph()) as sess:
  ...
  #### Build the signature_def_map。 
  ## 命名可以使用tf.saved_model.signature_constants。也可以自己取。
  ## 输入 SignatureDef
  inputs = {tf.saved_model.signature_constants.PREDICT_INPUTS: 			                                              tf.saved_model.utils.build_tensor_info(x), 
            'keep_prob': tf.saved_model.utils.build_tensor_info(keep_prob)}
  ## 输出 SignatureDef
  outputs = {'output' : tf.saved_model.utils.build_tensor_info(y)}
  ## 对signature 进行封装
  signature = tf.saved_model.signature_def_utils.build_signature_def(
    inputs=inputs, 
    outputs=outputs,
    method_name='test_sig_name')
  
  
  #### 建立缓存区
  builder.add_meta_graph_and_variables(sess,
                                  [tf.saved_model.tag_constants.SERVING],
                                  signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature})
## 保存
builder.save()


####################################### load #############################################
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
input_key = tf.saved_model.signature_constants.PREDICT_INPUTS
output_key = 'output'

meta_graph_def = tf.saved_model.loader.load(sess, 
                                            [tf.saved_model.tag_constants.SERVING],
                                            saved_model_dir)
# 从meta_graph_def中取出SignatureDef对象
signature = meta_graph_def.signature_def
# 从signature中找出具体输入输出的tensor name 
x_tensor_name = signature[signature_key].inputs[input_key].name
y_tensor_name = signature[signature_key].outputs[output_key].name
# 获取tensor 并inference
x = sess.graph.get_tensor_by_name(x_tensor_name)
y = sess.graph.get_tensor_by_name(y_tensor_name)
# _x 实际输入待inference的data
sess.run(y, feed_dict={x:_x})
```



## 图的构建

A TensorFlow computation, represented as a dataflow graph.

A `Graph` contains a set of `tf.Operation` objects, which represent units of computation; and `tf.Tensor` objects, which represent the units of data that flow between operations.

### [tf.Graph](https://www.tensorflow.org/api_docs/python/tf/Graph?hl=zh-cn)

| Class => tf.Graph                    | 注意：该类并不是线程安全的！！         |
| ------------------------------------ | ----------------------- |
| __init__()                           | 建立一个新的空的图               |
| **.as_default()**                    | 一个将某图设置为默认图，并返回一个上下文管理器 |
| .finalize()                          | 完成图的构建，即将其设置为只读模式       |
| **.device(device_name_or_function)** | 指定一个默认驱动，并返回一个上下文管理器    |
| .get_operation_by_name(name)         | 获得指定名字的operation        |
| .get_tensor_by_name(name)            | 获得指定名字的tensor           |
| .name_scope(name)                    | 为节点创建层次化的名称，并返回一个上下文管理器 |
| ……                                   |                         |
| **tf.get_default_graph()**           | 获得当前上下文的默认图             |
| **tf.reset_default_graph()**         | 清除默认图的堆栈，并设置全局图为默认图     |

```python
#获得默认图
g = tf.get_default_graph()

#使用新建的图（1）
g = tf.Graph()
with g.as_default():
  c = tf.constant(5.0)
  assert c.graph is g

#使用新建的图（2）
with tf.Graph().as_default() as g:
  c = tf.constant(5.0)
  assert c.graph is g

#指定驱动
with g.device('/device:GPU:0'):
  
#使用name_scope()
  c = tf.constant(5.0, name="c")
  assert c.op.name == "c"
  c_1 = tf.constant(6.0, name="c")
  assert c_1.op.name == "c_1"
  # Creates a scope called "nested"
  with g.name_scope("nested") as scope:
    nested_c = tf.constant(10.0, name="c")
    assert nested_c.op.name == "nested/c"
    # Creates a nested scope called "inner".
    with g.name_scope("inner"):
      nested_inner_c = tf.constant(20.0, name="c")
      assert nested_inner_c.op.name == "nested/inner/c"
```



### [tf.Operation](https://www.tensorflow.org/api_docs/python/tf/Operation?hl=zh-cn)

Objects of type `Operation` are created by calling a Python op constructor (such as`tf.matmul`) or `tf.Graph.create_op`

| Class => tf.Operation                    | 一般不怎么操作                |
| ---------------------------------------- | ---------------------- |
| tf.Operation.name                        | 操作节点(op)的名称            |
| tf.Operation.type                        | 操作节点(op)的类型，比如”MatMul” |
| tf.Operation.run(feed_dict=None, session=None) | 在会话(Session)中运行该操作     |
|                                          |                        |

### [tf.Tensor](https://www.tensorflow.org/api_docs/python/tf/Tensor?hl=zh-cn)

Represents one of the outputs of an Operation. 计算图中, 操作间传递的数据都是 tensor。

注意：Tensor只是一个operation输出的符号象征，并不保存确切的值，只是提供了计算这个值的方法！

| Class => tf.Tensor                  | 一般不怎么操作                |
| ----------------------------------- | ---------------------- |
| .dtype                              | 数据类型                   |
| .name                               | tensor名称               |
| .op                                 | 产生该tensor的op           |
| .eval(feed_dict=None,session=None ) | 在会话(Session)中运行该Tensor |
| ……                                  |                        |



## 图的运行

| Class => tf.Session                      |                                          |
| ---------------------------------------- | ---------------------------------------- |
| __init__( target='',graph=None,config=None ) | graph:默认加载上下文默认图                         |
| **run(    <br />    fetches,   <br />    feed_dict=None,  <br />    options=None,  <br />    run_metadata=None )** | 此方法运行TensorFlow计算的一个“step”，<br />通过运行必要的图形片段来执行每个操作<br />并评估fetches中的每个Tensor，<br />将feed_dict中的值替换为相应的输入值。<br />fetches:单个图元素，图元素列表或字典<br />feed_dict:图元素与值的映射字典 |
| **.as_default()**                        | 将该会话设置为默认会话，并返回一个上下文管理器。<br />在此会话中可执行tf.Operation.run或tf.Tensor.eval。 |
| ……                                       |                                          |
| **Class => tf.InteractiveSession**       | 使用在交互式上下文环境的tf会话，比如shell，ipython。<br />**与.Session的唯一区别是InteractiveSession在构造时直接将自己设置为了默认session。**于是能够直接调用Operation.run()和Tensor.eval()。 |
| ……                                       |                                          |
| tf.get_default_session()                 | 返回当前线程的默认会话                              |

```python
with tf.Session() as sess:
  sess.run(c)
  with sess.as_default():
  assert tf.get_default_session() is sess
  print(c.eval()) #直接运行~

with tf.InteractiveSession() as sess:
  print(c.eval()) #直接运行
```



## 图的优化训练

只考虑基本的使用，那些什么Gradient Computation and Clipping的自己动手改优化器的骚操作都不在本菜鸟的考虑范围内·······

| Class tf.train.Optimizer                 | 优化类的base类，该类不直接被调用。<br />而较多使用其子类，比如GradientDescentOptimizer |
| ---------------------------------------- | ---------------------------------------- |
| **.minimize(  <br />   loss, <br />   global_step=None,  <br />   var_list=None,   <br />   gate_gradients=GATE_OP,   <br />   aggregation_method=None,<br />   colocate_gradients_with_ops=False,<br />   name=None,   <br />   grad_loss=None )** | **loss**: 用于最小化的tensor<br />**global_step**:如果非空，则每次运行记录步数<br />var_list:更新的变量。<br />gate_gradients: 用于控制并行化的程度。 |
| .compute_gradients(……)                   | 计算梯度（供大佬使用）                              |
| .apply_gradients(……)                     | 对变量应用梯度（供大佬使用）                           |
| ……                                       |                                          |
| **常用实现类**                                |                                          |
| **class tf.train.GradientDescentOptimizer** | 梯度下降优化器                                  |
| **class tf.train.AdamOptimizer**         | Adam优化器                                  |
| **class tf.train.AdadeltaOptimizer**     | Adadelta优化器                              |
| **class tf.train.AdagradOptimizer**      | Adagrad优化器                               |
| class tf.train.MomentumOptimizer         | Momentum优化器                              |
| class tf.train.FtrlOptimizer             | Ftrl优化器                                  |
| class tf.train.RMSPropOptimizer          | RMSProp优化器                               |

**gate_gradients=**

*GATE_NONE* : 并行地计算和应用梯度。提供最大化的并行执行，但是会导致有的数据结果没有再现性。比如两个matmul操作的梯度依赖输入值，使用GATE_NONE可能会出现有一个梯度在其他梯度之前便应用到某个输入中，导致出现不可再现的(non-reproducible)结果。
*GATE_OP* : 对于每个操作Op，确保每一个梯度在使用之前都已经计算完成。这种做法防止了那些具有多个输入，并且梯度计算依赖输入情形中，多输入Ops之间的竞争情况出现。
*GATE_GRAPH* : 确保所有的变量对应的所有梯度在他们任何一个被使用前计算完成。该方式具有最低级别的并行化程度，但是对于想要在应用它们任何一个之前处理完所有的梯度计算时很有帮助的。

```python
#基本使用
global_step = tf.Variable(10, trainable=False, name='global_step')
# Create an optimizer with the desired parameters.
opt = GradientDescentOptimizer(learning_rate=0.1)
# Add Ops to the graph to minimize a cost by updating a list of variables.
# "cost" is a Tensor, and the list of variables contains tf.Variable objects.
opt_op = opt.minimize(cost, var_list=<list of variables>，global_step=global_step)
# Execute opt_op to do one step of training:
opt_op.run()
```



##数据的导入

并不必须，自己处理导入数据更加灵活但是可以简化程序！这里只是简单的应用。

| Class tf.data.Dataset                    | 元素集合输入的管道                                |
| ---------------------------------------- | ---------------------------------------- |
| **.from_tensor_slices(tensors)**         | 创建一个数据集，其元素是给定张量的切片。                     |
| **.map(map_func,num_parallel_calls=None)** | 对数据集元素进行映射                               |
| **.shuffle(buffer_size, seed=None,    reshuffle_each_iteration=None)** | 随机混洗此数据集的元素。                             |
| **.batch(batch_size,drop_remainder=False )** | 将此数据集的连续元素组合成批次，主要用来处理机器学习中的batch_size   |
| **.repeat(count=None)**                  | 将整个序列重复多次，主要用来处理机器学习中的epoch              |
| **.make_one_shot_iterator()**            | **单次**迭代器是最简单的迭代器形式，仅支持对数据集进行一次迭代，不需要显式初始化。 但不支持参数化。 |
| **.make_initializable_iterator(shared_name=None)** | 创建用于枚举此数据集元素的迭代器。需要初始化。但允许使用张量参数化数据集的定义。 |
| **tf.errors.OutOfRangeError**            | 监测数据是否已经用完。                              |
| .from_generator( ... )                   | 创建一个数据集，其元素由生成器生成。                       |
| .from_tensors(tensors)                   | 使用包含给定张量的单个元素创建数据集。                      |
| .list_files(file_pattern,shuffle=None,seed=None) | 与模式匹配的所有文件的数据集。                          |
| .padded_batch()                          | 将此数据集的连续元素组合成批次,允许输入元素的大小不同。             |
| .concatenate(dataset)                    | 合并数据集                                    |
| .filter(predicate)                       | 过滤数据集                                    |
| .flat_map(map_func)                      | 在此数据集中映射map_func并展平结果。                   |
| .range(*args)                            | 根据给定范围生成数据集                              |

```python
# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
dataset = dataset.map(...)  # Parse the record into tensors.
#随机重排输入数据集：它会维持一个固定大小的缓冲区，并从该缓冲区统一地随机选择下一个元素。
dataset = dataset.shuffle(buffersize=1000)
#将数据集中的n个连续元素堆叠为一个元素。
dataset = dataset.batch(32)
# 当count=None/-1 时重复无限次！
dataset = dataset.repeat(count=epoch)  
#生成迭代器
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
sess.run(iterator.initializer)
while True:
  try:
    sess.run(result)
  except tf.errors.OutOfRangeError:
    break

#########       make_initializable_iterator 与 make_one_shot_iterator 的区别   #################
limit = tf.placeholder(dtype=tf.int32, shape=[])
dataset = tf.data.Dataset.from_tensor_slices(tf.range(start=0, limit=limit))
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
#能够使用参数化，feed数据。必须手动初始化，并且可以多次初始化！
sess.run(iterator.initializer, feed_dict={limit: 10})
sess.run(next_element)
//////////////
iterator = dataset.make_one_shot_iterator()
#不能使用参数，且数据过一遍就没了。但不需要初始化。
one_element = iterator.get_next()
sess.run(one_element)

##########################################  .batch    ##########################################
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)
iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()
print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])

##########################################  .repeat   ########################################
#要迭代数据集多个周期，最简单的方法是使用 Dataset.repeat() 转换。
#应用不带参数的 Dataset.repeat() 转换将无限次地重复输入。Dataset.repeat() 转换将其参数连接起来，而不会#在一个周期结束和下一个周期开始时发出信号!所以没法在每个epoch后进行相应处理！！！
dataset = dataset.repeat(num_epochs)
iterator = dataset.make_one_shot_iterator()
next_example, next_label = iterator.get_next()
loss = model_function(next_example, next_label)
training_op = tf.train.AdagradOptimizer(...).minimize(loss)
with tf.train.MonitoredTrainingSession(...) as sess:  #相当于监控了tf.errors.OutOfRangeError
  while not sess.should_stop():
    sess.run(training_op)
#如果想在每个周期结束时收到信号，则可以编写在数据集结束时捕获 tf.errors.OutOfRangeError 的训练循环。
# Compute for 100 epochs.不用.repeat()!!
for _ in range(100):
  sess.run(iterator.initializer)
  while True:
    try:
      sess.run(next_element)
    except tf.errors.OutOfRangeError:
      break
  # [Perform end-of-epoch calculations here.]


```



## RNN-LSTM 

LSTM的简单[介绍](https://cenleiding.github.io/LSTM%E6%98%AF%E4%BB%80%E4%B9%88.html)

| cell                                     |                                          |
| ---------------------------------------- | ---------------------------------------- |
| **Class  tf.nn.rnn_cell.LSTMCell(<br />num_units,<br />use_peepholes=False,<br />cell_clip=None,    <br />initializer=None,<br />forget_bias=1.0,<br />state_is_tuple=True,<br />activation=None,<br />name=None,    <br />dtype=None,<br /> ……)** | **num_units: int，隐藏状态的大小，一般也是输出大小**<br />use_peepholes：是否使用门镜连接<br />cell_clip： float。状态矩阵对输出的影响<br />initializer：初始化方式<br />forget_bias=1.0：遗忘gate的偏见值<br />state_is_tuple=True,状态以2元元组形式接收，返回<br />activation：默认tanh，可以用Keras中的激活函数替代<br /> …… |
| **Class tf.nn.rnn_cell.GRUCell（<br /> num_units, <br /> activation=None,  <br /> reuse=None,     <br /> kernel_initializer=None,   <br /> bias_initializer=None,   <br /> name=None,  <br /> dtype=None,   <br /> kwargs） ** | **num_units: int，隐藏状态的大小，一般也是输出大小**<br />activation：默认tanh，可以用Keras中的激活函数替代<br /> |
| Class DeviceWrapper（<br />cell,  <br />device） | **cell**: 一个RNNCell的实例.<br />**device**:指定设备 |
| Class DropoutWrapper（<br />cell, <br />input_keep_prob=1.0,<br />output_keep_prob=1.0,<br />state_keep_prob=1.0,<br /> ……） | **cell**: 一个RNNCell的实例.<br />input_keep_prob=1.0 ：输入保留<br />**output_keep_prob** =1.0：输出保留<br />state_keep_prob=1.0：状态保留<br /> …… |
| Class MultiRNNCell（<br />cells,    <br />state_is_tuple=True） | **cells**: RNNCells的列表，用于实现多层RNN<br /> state_is_tuple：状态是否表现为元组 |
| **rnn**                                  |                                          |
| **tf.nn.dynamic_rnn(   <br />   cell,    <br />   inputs,  <br />   sequence_length=None,  <br />   initial_state=None,  <br />   dtype=None, <br />    parallel_iterations=None, <br />    swap_memory=False, <br />    time_major=False, <br />    scope=None ) <br /> return: (outputs, state) ** | 根据指定RNNCell生成循环神经网络<br />**cell**: 一个RNNCell的实例<br />**inputs**: 当time_majar=False时,<br />               形状为[batch_size, max_time, ...] <br />**sequence_length**: [batch_size]形状的向量，<br />     表示序列的真实长度,超出长度部分会复制状态和零输出<br />     因此这个参数主要为了性能而不是准确度。<br />      …… <br />initial_state:初始化状态，默认为0<br />**outputs:** 当 time_major == False <br />                形状为[batch_size, max_time, cell.output_size]<br />**state:** 最终的状态，一般为[batch_size, cell.state_size] |
| **tf.nn.bidirectional_dynamic_rnn(   <br />  cell_fw,<br />  cell_bw,  <br />  inputs,  <br />  sequence_length=None, <br />  initial_state_fw=None,  <br />  initial_state_bw=None, <br />  dtype=None, <br />  parallel_iterations=None,  <br />  swap_memory=False,   <br />  time_major=False,    <br />  scope=None )<br /> return：（outputs, output_states）** | 创建一个双向的循环神经网络<br /> **cell_fw**：一个前向的RNNCell实例<br /> **cell_bw**：一个后向的RNNCell实例<br /> **inputs**：当 time_major == False (default)<br />                  形状为 [batch_size, max_time, ...]<br /> **sequence_length**：[batch_size]形状的向量，<br />表示序列的真实长度，若不提供则认为都为max_time<br />initial_state_fw:前向RNN状态初始化，默认为0<br />initial_state_bw:后向RNN状态初始化，默认为0<br /> ……<br />**outputs**：一个元组 (output_fw, output_bw)<br />                   当 time_major == False <br />  形状为 [batch_size, max_time, cell_fw.output_size]<br />  和 [batch_size, max_time, cell_bw.output_size] <br />**output_states**：(output_state_fw, output_state_bw)<br />正向和反向的最终状态输出。 |

```python
#创建RNNCell实例，一般只用设置num_units即可
lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)
gru_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)

#可以添加dropout，一般只用设置output_keep_prob
lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,output_keep_prob=keep_prob)

#可以自动生成多层RNN
mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * layer_num)

#可以通过自带函数获得状态的0初始化矩阵
initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

#运行网络
outputs, state = tf.nn.dynamic_rnn(lstm_cell,inputs=X,
                                   initial_state=init_state)
outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=data,
                                   dtype=tf.float32)
#双向网络
lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)
lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)
(output_fw, output_bw),(output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_cell_fw,
                cell_bw=lstm_cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
output = tf.concat([output_fw, output_bw], axis=-1)

#####几个注意点：
#1.每一层的RNNCell个数是不定的，会根据输入长度max_time动态变化
#2.要对输入进行提前处理，保证每个batch中各个输入长度相同max_time，不足补0
#3.sequence_length表示了每条输入的真正长度，提不提供不会影响最后的结果，因为初始化时是补0的，但提供了可以提高性能。
#4.rnn函数默认对状态矩阵进行0初始化，所以不需要调用.zero_state函数去手动初始化。
```

