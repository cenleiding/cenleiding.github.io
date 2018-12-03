---
title: numpy 学习
date: 2018-11-30 08:57:01
tags: 机器学习
categories: 机器学习
keywords: numpy 学习
description: numpy的文档学习笔记
image: /numpy-学习/numpy_01.png
---

[numpy 参考文档](https://docs.scipy.org/doc/numpy-1.14.0/reference/index.html#numpy-reference)

## 基础

![](/numpy-学习/numpy_01.png)

NumPy 提供一个 N-dimensional 的数组，称为 ndarray，是一些相同类型和大小的items的集合体。

在方法中用 **axis** 来表示 dimensional。



## ndarray 类

[官方文档](https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#the-n-dimensional-array-ndarray)

### ndarray 属性

| [`ndarray.shape`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.shape.html#numpy.ndarray.shape) | **通常用于获取数组的当前形状，也可用于重塑数组。**              |
| ---------------------------------------- | ---------------------------------------- |
| [`ndarray.size`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.size.html#numpy.ndarray.size) | **数组元素个数**                               |
| [`ndarray.ndim`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.ndim.html#numpy.ndarray.ndim) | **数组维度数**                                |
| [`ndarray.nbytes`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.nbytes.html#numpy.ndarray.nbytes) | **数组总字节数**                               |
| [`ndarray.itemsize`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.itemsize.html#numpy.ndarray.itemsize) | 数组元素字节大小                                 |
| [`ndarray.flags`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.flags.html#numpy.ndarray.flags) | 数组内存分布信息                                 |
| [`ndarray.strides`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.strides.html#numpy.ndarray.strides) | Tuple of bytes to step in each dimension when traversing an array. |
| [`ndarray.data`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.data.html#numpy.ndarray.data) | Python buffer object pointing to the start of the array’s data. |
| [`ndarray.base`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.base.html#numpy.ndarray.base) | Base object if memory is from some other object. |

```python
# ndarray.shape
>>> y = np.zeros((2, 3, 4))
>>> y.shape
(2, 3, 4)
>>> y.shape = (3, 8)
>>> y
array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

# ndarray.ndim
>>> x = np.array([1, 2, 3])
>>> x.ndim
1
>>> y = np.zeros((2, 3, 4))
>>> y.ndim
3

# ndarray.size
>>> x = np.zeros((3, 5, 2), dtype=np.complex128)
>>> x.size
30

# ndarray.itemsize/nbytes
>>> x = np.zeros((3,5,2), dtype=np.complex128)
>>> x.nbytes
480
>>> np.prod(x.shape) * x.itemsize
480
```



### ndarray 方法

一般习惯用外部的方法。



### 索引&切片

```python
###  基础 索引&切片
# 注意基础的切片返回的是view，及目标和结果是同一个object。
>>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# start：stop：step
>>> x[1:7:2]
array([1, 3, 5])

# 负数i,以n+i处理
>>> x[-2:10]
array([8, 9])
>>> x[-3:3:-1]
array([7, 6, 5, 4])

# '::' 等同于 ':'。未填的项表示选取所有。 
>>> x[5:]
array([5, 6, 7, 8, 9])

# 当选取元祖维度不够时，未填的项表示选取所有。
>>> x = np.array([[[1],[2],[3]], [[4],[5],[6]]])
>>> x.shape
(2, 3, 1)
>>> x[1:2]     #只对第一维进行选取操作。
array([[[4],
        [5],
        [6]]])
>>> x[1]     #注意：没有取范围，所以维度减1
array([[4],
        [5],
        [6]])
# 如果不用'：'，而是直接取，则会造成维度减少。
>>> x[:,1,:]
array([[2],
       [5]])
>>> x[:,1:2,:]
array([[[2]],
       [[5]]])

# '...'省略号代表所有的':'
>>> x[...,0]
array([[1, 2, 3],
       [4, 5, 6]])
>>> x[:,:,0]         # 等同
array([[1, 2, 3],
       [4, 5, 6]])

# newaxis 则是会直接添加一个轴
>>> x[:,np.newaxis,:,:].shape
(2, 1, 3, 1)

###### 高级索引&切片
## 高级的切片得到的数组是copy的。

#### 数值型高级索引。
>>> x = np.array([[1, 2], [3, 4], [5, 6]])
>>> x[[0, 1, 2], [0, 1, 0]]   # 前一个代表行索引，后一个代表列索引。
array([1, 4, 5])
# 多维的索引数组
>>> x = array([[ 0,  1,  2],
	           [ 3,  4,  5],
               [ 6,  7,  8],
               [ 9, 10, 11]])
>>> rows = np.array([[0, 0],    
                     [3, 3]], dtype=np.intp)
>>> columns = np.array([[0, 2],
                        [0, 2]], dtype=np.intp)
>>> x[rows, columns]     # 结果数组的格式与索引数组的格式一致。
array([[ 0,  2],
       [ 9, 11]])
# 使用np.newaxis使得行列索引数组大小不同，触发广播broadcasting 
>>> rows = np.array([0, 3], dtype=np.intp)
>>> columns = np.array([0, 2], dtype=np.intp)
>>> rows[:, np.newaxis]
array([[0],
       [3]])
>>> x[rows[:, np.newaxis], columns]
array([[ 0,  2],
       [ 9, 11]])
# 使用ix_()触发广播
>>> x[np.ix_(rows, columns)]
array([[ 0,  2],
       [ 9, 11]])

#### Boolean型高级索引
>>> x = np.array([[1., 2.], [np.nan, 3.], [np.nan, np.nan]])
>>> x[~np.isnan(x)]
array([ 1.,  2.,  3.])
>>> x = np.array([1., -1., -2., 3])
>>> x[x < 0] += 20
>>> x
array([  1.,  19.,  18.,   3.])
# 选取和小于等于2的行
>>> x = np.array([[0, 1], [1, 1], [2, 2]])
>>> rowsum = x.sum(-1)
>>> x[rowsum <= 2, :]
array([[0, 1],
       [1, 1]])
```



## 通用函数 ufunc

### 广播  Broadcasting

**执行 broadcast 的前提在于，两个 ndarray 执行的是 element-wise（按位加减等） 的运算，而不是矩阵乘法的运算，矩阵乘法运算时需要维度之间严格匹配。**

**broadcasting rules：**

1.  两个数组shape相等。
2.  两个数组shape不相同，但**其中一个为1**。

```python
# 例子：
Image (3d array):  256 x 256 x 3
Scale (1d array):              3
Result (3d array): 256 x 256 x 3

A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5

A      (2d array):  5 x 4
B      (1d array):      1
Result (2d array):  5 x 4

A      (2d array):  15 x 3 x 5
B      (1d array):  15 x 1 x 5
Result (2d array):  15 x 3 x 5
```



### 数学运算

| [`add`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.add.html#numpy.add) | Add arguments element-wise.              |
| ---------------------------------------- | ---------------------------------------- |
| [`subtract`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.subtract.html#numpy.subtract) | Subtract arguments, element-wise.        |
| [`multiply`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.multiply.html#numpy.multiply) | Multiply arguments element-wise.         |
| [`divide`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.divide.html#numpy.divide) | Returns a true division of the inputs, element-wise. |
| [`logaddexp`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.logaddexp.html#numpy.logaddexp) | Logarithm of the sum of exponentiations of the inputs. |
| [`logaddexp2`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.logaddexp2.html#numpy.logaddexp2) | Logarithm of the sum of exponentiations of the inputs in base-2. |
| [`true_divide`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.true_divide.html#numpy.true_divide) | Returns a true division of the inputs, element-wise. |
| [`floor_divide`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.floor_divide.html#numpy.floor_divide) | Return the largest integer smaller or equal to the division of the inputs. |
| [`negative`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.negative.html#numpy.negative) | Numerical negative, element-wise.        |
| [`positive`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.positive.html#numpy.positive) | Numerical positive, element-wise.        |
| [`power`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.power.html#numpy.power) | First array elements raised to powers from second array, element-wise. |
| [`remainder`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.remainder.html#numpy.remainder) | Return element-wise remainder of division. |
| [`mod`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mod.html#numpy.mod) | Return element-wise remainder of division. |
| [`fmod`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fmod.html#numpy.fmod) | Return the element-wise remainder of division. |
| [`divmod`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.divmod.html#numpy.divmod) | Return element-wise quotient and remainder simultaneously. |
| [`absolute`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.absolute.html#numpy.absolute) | Calculate the absolute value element-wise. |
| [`fabs`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fabs.html#numpy.fabs) | Compute the absolute values element-wise. |
| [`rint`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.rint.html#numpy.rint) | Round elements of the array to the nearest integer. |
| [`sign`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sign.html#numpy.sign) | Returns an element-wise indication of the sign of a number. |
| [`heaviside`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.heaviside.html#numpy.heaviside) | Compute the Heaviside step function.     |
| [`conj`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.conj.html#numpy.conj) | Return the complex conjugate, element-wise. |
| [`exp`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html#numpy.exp) | Calculate the exponential of all elements in the input array. |
| [`exp2`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp2.html#numpy.exp2) | Calculate *2\**p* for all *p* in the input array. |
| [`log`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html#numpy.log) | Natural logarithm, element-wise.         |
| [`log2`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.log2.html#numpy.log2) | Base-2 logarithm of *x*.                 |
| [`log10`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.log10.html#numpy.log10) | Return the base 10 logarithm of the input array, element-wise. |
| [`expm1`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expm1.html#numpy.expm1) | Calculate `exp(x) - 1` for all elements in the array. |
| [`log1p`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.log1p.html#numpy.log1p) | Return the natural logarithm of one plus the input array, element-wise. |
| [`sqrt`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sqrt.html#numpy.sqrt) | Return the non-negative square-root of an array, element-wise. |
| [`square`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.square.html#numpy.square) | Return the element-wise square of the input. |
| [`cbrt`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cbrt.html#numpy.cbrt) | Return the cube-root of an array, element-wise. |
| [`reciprocal`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reciprocal.html#numpy.reciprocal) | Return the reciprocal of the argument, element-wise. |
| [`gcd`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.gcd.html#numpy.gcd) | Returns the greatest common divisor of `|x1|` and `|x2|` |
| [`lcm`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.lcm.html#numpy.lcm) | Returns the lowest common multiple of `|x1|` and `|x2|` |

### 三角运算

| [`sin`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sin.html#numpy.sin) | Trigonometric sine, element-wise.        |
| ---------------------------------------- | ---------------------------------------- |
| [`cos`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cos.html#numpy.cos) | Cosine element-wise.                     |
| [`tan`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.tan.html#numpy.tan) | Compute tangent element-wise.            |
| [`arcsin`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arcsin.html#numpy.arcsin) | Inverse sine, element-wise.              |
| [`arccos`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arccos.html#numpy.arccos) | Trigonometric inverse cosine, element-wise. |
| [`arctan`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan.html#numpy.arctan) | Trigonometric inverse tangent, element-wise. |
| [`arctan2`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan2.html#numpy.arctan2) | Element-wise arc tangent of `x1/x2` choosing the quadrant correctly. |
| [`hypot`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hypot.html#numpy.hypot) | Given the “legs” of a right triangle, return its hypotenuse. |
| [`sinh`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sinh.html#numpy.sinh) | Hyperbolic sine, element-wise.           |
| [`cosh`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cosh.html#numpy.cosh) | Hyperbolic cosine, element-wise.         |
| [`tanh`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.tanh.html#numpy.tanh) | Compute hyperbolic tangent element-wise. |
| [`arcsinh`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arcsinh.html#numpy.arcsinh) | Inverse hyperbolic sine element-wise.    |
| [`arccosh`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arccosh.html#numpy.arccosh) | Inverse hyperbolic cosine, element-wise. |
| [`arctanh`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctanh.html#numpy.arctanh) | Inverse hyperbolic tangent element-wise. |
| [`deg2rad`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.deg2rad.html#numpy.deg2rad) | Convert angles from degrees to radians.  |
| [`rad2deg`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.rad2deg.html#numpy.rad2deg) | Convert angles from radians to degrees.  |

### 对比运算

| [`greater`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.greater.html#numpy.greater) | Return the truth value of (x1 > x2) element-wise. |
| ---------------------------------------- | ---------------------------------------- |
| [`greater_equal`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.greater_equal.html#numpy.greater_equal) | Return the truth value of (x1 >= x2) element-wise. |
| [`less`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.less.html#numpy.less) | Return the truth value of (x1 < x2) element-wise. |
| [`less_equal`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.less_equal.html#numpy.less_equal) | Return the truth value of (x1 =< x2) element-wise. |
| [`not_equal`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.not_equal.html#numpy.not_equal) | Return (x1 != x2) element-wise.          |
| [`equal`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.equal.html#numpy.equal) | Return (x1 == x2) element-wise.          |

| [`logical_and`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.logical_and.html#numpy.logical_and) | Compute the truth value of x1 AND x2 element-wise. |
| ---------------------------------------- | ---------------------------------------- |
| [`logical_or`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.logical_or.html#numpy.logical_or) | Compute the truth value of x1 OR x2 element-wise. |
| [`logical_xor`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.logical_xor.html#numpy.logical_xor) | Compute the truth value of x1 XOR x2, element-wise. |
| [`logical_not`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.logical_not.html#numpy.logical_not) | Compute the truth value of NOT x element-wise. |

| [`maximum`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.maximum.html#numpy.maximum) | Element-wise maximum of array elements. |
| ---------------------------------------- | --------------------------------------- |
| [`minimum`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.minimum.html#numpy.minimum) | Element-wise minimum of array elements. |
| [`fmax`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fmax.html#numpy.fmax) | Element-wise maximum of array elements. |
| [`fmin`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fmin.html#numpy.fmin) | Element-wise minimum of array elements. |

## 常用功能

### 数组生成

[完整文档](https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html)

#### Ones and zeros

| Ones and zeros                           |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`empty`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.empty.html#numpy.empty) | Return a new array of given shape and type, without initializing entries. |
| [`empty_like`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.empty_like.html#numpy.empty_like) | Return a new array with the same shape and type as a given array. |
| [`eye`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.eye.html#numpy.eye) | Return a 2-D array with ones on the diagonal and zeros elsewhere. |
| [`identity`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.identity.html#numpy.identity) | Return the identity array.               |
| [`ones`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html#numpy.ones) | Return a new array of given shape and type, filled with ones. |
| [`ones_like`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones_like.html#numpy.ones_like) | Return an array of ones with the same shape and type as a given array. |
| [`zeros`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html#numpy.zeros) | Return a new array of given shape and type, filled with zeros. |
| [`zeros_like`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros_like.html#numpy.zeros_like) | Return an array of zeros with the same shape and type as a given array. |
| [`full`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html#numpy.full) | Return a new array of given shape and type, filled with *fill_value*. |
| [`full_like`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.full_like.html#numpy.full_like) | Return a full array with the same shape and type as a given array. |

```Python
## 常用例子
# .eye
>>> np.eye(2, dtype=int)
array([[1, 0],
       [0, 1]])
>>> np.eye(3, k=1)
array([[ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 0.,  0.,  0.]])

# .ones()
>>> np.ones(5)
array([ 1.,  1.,  1.,  1.,  1.])
>>> np.ones((5,), dtype=int)
array([1, 1, 1, 1, 1])
>>> np.ones((2, 1))
array([[ 1.],
       [ 1.]])

# .ones_like()
>>> y = np.arange(3, dtype=float)
>>> y
array([ 0.,  1.,  2.])
>>> np.ones_like(y)
array([ 1.,  1.,  1.])

# .zeros()
>>> s = (2,2)
>>> np.zeros(s)
array([[ 0.,  0.],
       [ 0.,  0.]])

# .zeros_like()
>>> y = np.arange(3, dtype=float)
>>> y
array([ 0.,  1.,  2.])
>>> np.zeros_like(y)
array([ 0.,  0.,  0.])

# .full()
>>> np.full((2, 2), np.inf)
array([[ inf,  inf],
       [ inf,  inf]])
>>> np.full((2, 2), 10)
array([[10, 10],
       [10, 10]])

# .full_like()
>>> x = np.arange(6, dtype=int)
>>> np.full_like(x, 1)
array([1, 1, 1, 1, 1, 1])
>>> np.full_like(x, 0.1)
array([0, 0, 0, 0, 0, 0])
>>> np.full_like(x, 0.1, dtype=np.double)
array([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1])
>>> np.full_like(x, np.nan, dtype=np.double)
array([ nan,  nan,  nan,  nan,  nan,  nan])
```

#### From existing data 
| From existing data                       |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`array`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html#numpy.array) | Create an array.                         |
| [`asarray`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.asarray.html#numpy.asarray) | Convert the input to an array.           |
| [`asanyarray`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.asanyarray.html#numpy.asanyarray) | Convert the input to an ndarray, but pass ndarray subclasses through. |
| [`ascontiguousarray`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ascontiguousarray.html#numpy.ascontiguousarray) | Return a contiguous array in memory (C order). |
| [`asmatrix`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.asmatrix.html#numpy.asmatrix) | Interpret the input as a matrix.         |
| [`copy`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.copy.html#numpy.copy) | Return an array copy of the given object. |
| [`frombuffer`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.frombuffer.html#numpy.frombuffer) | Interpret a buffer as a 1-dimensional array. |
| [`fromfile`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fromfile.html#numpy.fromfile) | Construct an array from data in a text or binary file. |
| [`fromfunction`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fromfunction.html#numpy.fromfunction) | Construct an array by executing a function over each coordinate. |
| [`fromiter`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fromiter.html#numpy.fromiter) | Create a new 1-dimensional array from an iterable object. |
| [`fromstring`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fromstring.html#numpy.fromstring) | A new 1-D array initialized from text data in a string. |
| [`loadtxt`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html#numpy.loadtxt) | Load data from a text file.              |

```python
## 常用例子
# .array()
>>> np.array([1, 2, 3])
array([1, 2, 3])
>>> np.array([1, 2, 3], dtype=complex)
array([ 1.+0.j,  2.+0.j,  3.+0.j])

# .copy()
>>> x = np.array([1, 2, 3])
>>> y = x
>>> z = np.copy(x)
>>> x[0] = 10
>>> x[0] == y[0]
True
>>> x[0] == z[0]
False

# .fromfunction()
>>> np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
array([[0, 1, 2],
       [1, 2, 3],
       [2, 3, 4]])
>>> np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
array([[ True, False, False],
       [False,  True, False],
       [False, False,  True]])

# .fromString()
>>> np.fromstring('1 2', dtype=int, sep=' ')
array([1, 2])
>>> np.fromstring('1, 2', dtype=int, sep=',')
array([1, 2])
```

#### Numerical ranges
| Numerical ranges                         |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`arange`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html#numpy.arange) | Return evenly spaced values within a given interval. |
| [`linspace`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html#numpy.linspace) | Return evenly spaced numbers over a specified interval. |
| [`logspace`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.logspace.html#numpy.logspace) | Return numbers spaced evenly on a log scale. |
| [`geomspace`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.geomspace.html#numpy.geomspace) | Return numbers spaced evenly on a log scale (a geometric progression). |
| [`meshgrid`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html#numpy.meshgrid) | Return coordinate matrices from coordinate vectors. |
| [`mgrid`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mgrid.html#numpy.mgrid) | *nd_grid* instance which returns a dense multi-dimensional “meshgrid”. |
| [`ogrid`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ogrid.html#numpy.ogrid) | *nd_grid* instance which returns an open multi-dimensional “meshgrid”. |

```python
## 常用例子
# .arange()
>>> np.arange(3)
array([0, 1, 2])
>>> np.arange(3.0)
array([ 0.,  1.,  2.])
>>> np.arange(3,7)
array([3, 4, 5, 6])
>>> np.arange(3,7,2)
array([3, 5])

#　.linspace()
>>> np.linspace(2.0, 3.0, num=5)
array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
>>> np.linspace(2.0, 3.0, num=5, endpoint=False)
array([ 2. ,  2.2,  2.4,  2.6,  2.8])
>>> np.linspace(2.0, 3.0, num=5, retstep=True)
(array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)
```



### 数组操作

[完整文档](https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html#array-manipulation-routines)

#### Changing array shape
| Changing array shape                     |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`reshape`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html#numpy.reshape) | Gives a new shape to an array without changing its data. |
| [`ravel`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html#numpy.ravel) | Return a contiguous flattened array.     |
| [`ndarray.flat`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flat.html#numpy.ndarray.flat) | A 1-D iterator over the array.           |
| [`ndarray.flatten`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html#numpy.ndarray.flatten) | Return a copy of the array collapsed into one dimension. |

```python
## 常用函数
# .reshape
>>> a = np.arange(6).reshape((3, 2))
>>> a
array([[0, 1],
       [2, 3],
       [4, 5]])
>>> np.reshape(a, (3,-1))       # 形状的一个维度可以是-1，自适应
array([[0, 1],
       [2, 3],
       [4, 5]])

# .ravel  等同于.reshape(-1)
>>> x = np.array([[1, 2, 3], [4, 5, 6]])
>>> print(np.ravel(x))
[1 2 3 4 5 6]
```

#### Transpose-like operations

| Transpose-like operations                |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`moveaxis`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.moveaxis.html#numpy.moveaxis) | Move axes of an array to new positions.  |
| [`rollaxis`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.rollaxis.html#numpy.rollaxis) | Roll the specified axis backwards, until it lies in a given position. 可用moveaxis实现。 |
| [`swapaxes`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.swapaxes.html#numpy.swapaxes) | Interchange two axes of an array.        |
| [`ndarray.T`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.T.html#numpy.ndarray.T) | Same as self.transpose(), except that self is returned if self.ndim < 2. |
| [`transpose`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html#numpy.transpose) | Permute the dimensions of an array.      |

```python
## 常用例子
# .moveaxis()  移动指定轴到指定位置，其他轴不变
>>> x = np.zeros((3, 4, 5))
>>> np.moveaxis(x, 0, -1).shape
(4, 5, 3)
>>> np.moveaxis(x, -1, 0).shape
(5, 3, 4)

# .swapaxes()  交换两个轴
>>> x = np.array([[1,2,3]])
>>> np.swapaxes(x,0,1)
array([[1],
       [2],
       [3]])

#  ndarray.T  .transpose()  转置
>>> x = np.array([[1.,2.],[3.,4.]])
>>> x
array([[ 1.,  2.],
       [ 3.,  4.]])
>>> x.T
array([[ 1.,  3.],
       [ 2.,  4.]])
>>> np.transpose(x)
array([[ 1.,  3.],
       [ 2.,  4.]])
```

#### Changing number of dimensions

| Changing number of dimensions            |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`atleast_1d`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_1d.html#numpy.atleast_1d) | Convert inputs to arrays with at least one dimension. |
| [`atleast_2d`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_2d.html#numpy.atleast_2d) | View inputs as arrays with at least two dimensions. |
| [`atleast_3d`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_3d.html#numpy.atleast_3d) | View inputs as arrays with at least three dimensions. |
| [`broadcast`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast.html#numpy.broadcast) | Produce an object that mimics broadcasting. |
| [`broadcast_to`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_to.html#numpy.broadcast_to) | Broadcast an array to a new shape.       |
| [`broadcast_arrays`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_arrays.html#numpy.broadcast_arrays) | Broadcast any number of arrays against each other. |
| [`expand_dims`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html#numpy.expand_dims)(a, axis) | Expand the shape of an array.            |
| [`squeeze`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html#numpy.squeeze)(a[, axis]) | Remove single-dimensional entries from the shape of an array. |

```python
## 常用函数
# .expand_dims()  添加一个新轴 效果等同于np.newaxis
>>> x = np.array([1,2])
>>> x.shape
(2,)
>>> y = np.expand_dims(x, axis=0) # 等同于 x[np.newaxis,:]和x[np.newaxis]
>>> y
array([[1, 2]])
>>> y.shape
(1, 2)
>>> y = np.expand_dims(x, axis=1)  # 等同于 x[:,np.newaxis]
>>> y
array([[1],
       [2]])
>>> y.shape
(2, 1)

# .squeeze()  压缩数组，去除长度为1的轴
>>> x = np.array([[[0], [1], [2]]])
>>> x.shape
(1, 3, 1)
>>> np.squeeze(x).shape
(3,)
>>> np.squeeze(x, axis=0).shape
(3, 1)
>>> np.squeeze(x, axis=1).shape
Traceback (most recent call last):
```

#### Changing kind of array

| Changing kind of array                   |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`asarray`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.asarray.html#numpy.asarray) | Convert the input to an array.           |
| [`asanyarray`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.asanyarray.html#numpy.asanyarray) | Convert the input to an ndarray, but pass ndarray subclasses through. |
| [`asmatrix`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.asmatrix.html#numpy.asmatrix) | Interpret the input as a matrix.         |
| [`asfarray`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.asfarray.html#numpy.asfarray) | Return an array converted to a float type. |
| [`asfortranarray`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.asfortranarray.html#numpy.asfortranarray) | Return an array laid out in Fortran order in memory. |
| [`ascontiguousarray`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ascontiguousarray.html#numpy.ascontiguousarray) | Return a contiguous array in memory (C order). |
| [`asarray_chkfinite`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.asarray_chkfinite.html#numpy.asarray_chkfinite) | Convert the input to an array, checking for NaNs or Infs. |
| [`asscalar`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.asscalar.html#numpy.asscalar) | Convert an array of size 1 to its scalar equivalent. |
| [`require`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.require.html#numpy.require) | Return an ndarray of the provided type that satisfies requirements. |

```python
## 常用函数
# .asarray()  比.array()更强大一点点
>>> a = [1, 2]   # Convert a list into an array:
>>> np.asarray(a)
array([1, 2])
>>> a = np.array([1, 2])  # Existing arrays are not copied:
>>> np.asarray(a) is a
True
#If dtype is set, array is copied only if dtype does not match:
>>> a = np.array([1, 2], dtype=np.float32) 
>>> np.asarray(a, dtype=np.float32) is a
True
>>> np.asarray(a, dtype=np.float64) is a
False
```

#### Joining arrays

| Joining arrays                           |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`concatenate`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html#numpy.concatenate) | Join a sequence of arrays along an existing axis. |
| [`stack`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.stack.html#numpy.stack) | Join a sequence of arrays along a new axis. |
| [`column_stack`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.column_stack.html#numpy.column_stack) | Stack 1-D arrays as columns into a 2-D array. |
| [`dstack`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dstack.html#numpy.dstack) | Stack arrays in sequence depth wise (along third axis). |
| [`hstack`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html#numpy.hstack) | Stack arrays in sequence horizontally (column wise). |
| [`vstack`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html#numpy.vstack) | Stack arrays in sequence vertically (row wise). |
| [`block`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.block.html#numpy.block) | Assemble an nd-array from nested lists of blocks. |

```Python
### 常用函数
# .concatenate 将两个数组沿已有的轴连接，两个数组除了连接轴形状必须相同！
>>> a = np.array([[1, 2], [3, 4]])   # 2*2
>>> b = np.array([[5, 6]])           # 1*2
>>> np.concatenate((a, b), axis=0)   # 3*2
array([[1, 2],
       [3, 4],
       [5, 6]])
>>> np.concatenate((a, b.T), axis=1)
array([[1, 2, 5],
       [3, 4, 6]])
>>> np.concatenate((a, b), axis=None)
array([1, 2, 3, 4, 5, 6])

# .stack() 将一系列数组沿一个新的轴连接，每个数组的形状必须相同！
>>> a = np.array([1, 2, 3])
>>> b = np.array([2, 3, 4])
>>> np.stack((a, b))           # 维度会加1
array([[1, 2, 3],
       [2, 3, 4]])
>>> np.stack((a, b), axis=-1)  # -1表示在最后加一维。0表示在开头加一维。
array([[1, 2],
       [2, 3],
       [3, 4]])

# .dstack .hstack .vstack 都可以用.stack()和.concatenate实现

```

#### Splitting arrays

| Splitting arrays                         |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`split`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.split.html#numpy.split) | Split an array into multiple sub-arrays. |
| [`array_split`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array_split.html#numpy.array_split) | Split an array into multiple sub-arrays. |
| [`dsplit`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dsplit.html#numpy.dsplit) | Split array into multiple sub-arrays along the 3rd axis (depth). |
| [`hsplit`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hsplit.html#numpy.hsplit) | Split an array into multiple sub-arrays horizontally (column-wise). |
| [`vsplit`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vsplit.html#numpy.vsplit) | Split an array into multiple sub-arrays vertically (row-wise). |

```python
### 常用函数
# .split()
>>> x = np.arange(9.0)
>>> np.split(x, 3)
[array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.,  8.])]
>>> x = np.arange(8.0)
>>> np.split(x, [3, 5, 6, 10])
[array([ 0.,  1.,  2.]),
 array([ 3.,  4.]),
 array([ 5.]),
 array([ 6.,  7.]),
 array([], dtype=float64)]
```



### 输入输出

[完整文档](https://docs.scipy.org/doc/numpy/reference/routines.io.html)

#### NumPy binary files (NPY, NPZ)

| NumPy binary files (NPY,NPZ)             |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`load`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html#numpy.load) | Load arrays or pickled objects from `.npy`, `.npz` or pickled files. |
| [`save`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html#numpy.save) | Save an array to a binary file in NumPy `.npy` format. |
| [`savez`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html#numpy.savez) | Save several arrays into a single file in uncompressed `.npz` format. |
| [`savez_compressed`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed) | Save several arrays into a single file in compressed `.npz` format. |

``` python
### 常用函数
>>> x = np.arange(10)
>>> np.save(outfile, x)
>>> np.load(outfile)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

#### Text files

| Text files                               |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`loadtxt`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html#numpy.loadtxt) | Load data from a text file.              |
| [`savetxt`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savetxt.html#numpy.savetxt) | Save an array to a text file.            |
| [`genfromtxt`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt) | Load data from a text file, with missing values handled as specified. |
| [`fromregex`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fromregex.html#numpy.fromregex) | Construct an array from a text file, using regular expression parsing. |
| [`fromstring`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fromstring.html#numpy.fromstring) | A new 1-D array initialized from text data in a string. |
| [`ndarray.tofile`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tofile.html#numpy.ndarray.tofile) | Write array to a file as text or binary (default). |
| [`ndarray.tolist`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tolist.html#numpy.ndarray.tolist) | Return the array as a (possibly nested) list. |

```python 
### 常用函数
# .loadtxt  每一行的数要一样多。
>>> from io import StringIO   # StringIO behaves like a file object
>>> c = StringIO(u"0 1\n2 3")
>>> np.loadtxt(c)
array([[ 0.,  1.],
       [ 2.,  3.]])

# .savetxt 注意因为只能保存一维或两维的数组，所以不常用！！
>>> x = y = z = np.arange(0.0,5.0,1.0)
>>> np.savetxt('test.out', x, delimiter=',')   # X is an array
>>> np.savetxt('test.out', (x,y,z))   # x,y,z equal sized 1D arrays
>>> np.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation

# .fromstring()  只能转换一维数组，不实用
>>> np.fromstring('1 2', dtype=int, sep=' ')
array([1, 2])
>>> np.fromstring('1, 2', dtype=int, sep=',')
array([1, 2])
```

#### Text formatting options

| Text formatting options                  |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`set_printoptions`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html#numpy.set_printoptions) | Set printing options.                    |
| [`get_printoptions`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.get_printoptions.html#numpy.get_printoptions) | Return the current print options.        |
| [`set_string_function`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.set_string_function.html#numpy.set_string_function) | Set a Python function to be used when pretty printing arrays. |
| [`printoptions`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.printoptions.html#numpy.printoptions) | Context manager for setting print options. |

```python
### 常用例子
# printing 规则：
# 1. 最后的axis，从左往右打印。
# 2. 倒数第2个axis，从上向下打印。
# 3. 其他axis，也从上向下打印，每个切片之间以空行隔开。

# 默认设置
>>> np.set_printoptions(edgeitems=3,infstr='inf',
... linewidth=75, nanstr='nan', precision=8,
... suppress=False, threshold=1000, formatter=None)

# float 显示位数
>>> np.set_printoptions(precision=4)
>>> print(np.array([1.123456789]))
[ 1.1235]

# 打印完整的数组
>>> np.set_printoptions(threshold=np.nan)

# 不用科学计数法表示
>>> np.set_printoptions(suppress=True)
```



### 随机采样(numpy.random) 

[完整文档](https://docs.scipy.org/doc/numpy/reference/routines.random.html)

#### Simple random data

| Simple random data                       |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`rand`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rand.html#numpy.random.rand) | Random values in a given shape.          |
| [`randn`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html#numpy.random.randn) | Return a sample (or samples) from the “standard normal” distribution. |
| [`randint`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html#numpy.random.randint) | Return random integers from *low* (inclusive) to *high* (exclusive). |
| [`random_integers`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.random_integers.html#numpy.random.random_integers) | Random integers of type np.int between *low* and *high*, inclusive. |
| [`random_sample`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.random_sample.html#numpy.random.random_sample) | Return random floats in the half-open interval [0.0, 1.0). |
| [`random`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.random.html#numpy.random.random) | Return random floats in the half-open interval [0.0, 1.0). |
| [`ranf`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.ranf.html#numpy.random.ranf) | Return random floats in the half-open interval [0.0, 1.0). |
| [`sample`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.sample.html#numpy.random.sample) | Return random floats in the half-open interval [0.0, 1.0). |
| [`choice`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html#numpy.random.choice) | Generates a random sample from a given 1-D array |
| [`bytes`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.bytes.html#numpy.random.bytes) | Return random bytes.                     |

```python
### 常用函数

# .rand .random .random_sample 功能一致，生成指定shape的[0,1)的均匀分布
>>> np.random.rand(3,2)             # 只是输入格式不同
array([[ 0.14022471,  0.96360618],  #random
       [ 0.37601032,  0.25528411],  #random
       [ 0.49313049,  0.94909878]]) #random
>>> 5 * np.random.random_sample((3, 2)) - 5   #[-5,0)
array([[-3.99149989, -0.52338984],
       [-2.99091858, -0.79479508],
       [-1.23204345, -1.75224494]])
>>> 5 * np.random.random((3, 2)) - 5   #[-5,0)
array([[-3.99149989, -0.52338984],
       [-2.99091858, -0.79479508],
       [-1.23204345, -1.75224494]])


# .randn 生成指定shape的标准正太分布(normal,Gaussian)
>>> 2.5 * np.random.randn(2, 4) + 3
array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],  #random
       [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]]) #random
```

####　Permutations

| Permutations                             | 随机排列                                     |
| ---------------------------------------- | ---------------------------------------- |
| [`shuffle`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.shuffle.html#numpy.random.shuffle) | Modify a sequence in-place by shuffling its contents. |
| [`permutation`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.permutation.html#numpy.random.permutation) | Randomly permute a sequence, or return a permuted range. |

````python
### 常用函数
# .shuffle() 随机打乱数组，但对于多维数组只打乱第一维。
>>> arr = np.arange(10)
>>> np.random.shuffle(arr)
>>> arr
[1 7 5 2 9 4 3 6 0 8]
>>> arr = np.arange(9).reshape((3, 3))
>>> np.random.shuffle(arr)
>>> arr
array([[3, 4, 5],
       [6, 7, 8],
       [0, 1, 2]])

# .permutation 与 .shuffle()功能相似
````

#### Distributions

| Distributions                            | 输入各个分布的参数，获得相应shape的样本。                  |
| ---------------------------------------- | ---------------------------------------- |
| [`beta`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.beta.html#numpy.random.beta) | Draw samples from a Beta distribution.   |
| [`binomial`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.binomial.html#numpy.random.binomial) | Draw samples from a binomial distribution. |
| [`chisquare`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.chisquare.html#numpy.random.chisquare) | Draw samples from a chi-square distribution. |
| [`dirichlet`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.dirichlet.html#numpy.random.dirichlet) | Draw samples from the Dirichlet distribution. |
| [`exponential`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.exponential.html#numpy.random.exponential) | Draw samples from an exponential distribution. |
| [`f`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.f.html#numpy.random.f) | Draw samples from an F distribution.     |
| [`gamma`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.gamma.html#numpy.random.gamma) | Draw samples from a Gamma distribution.  |
| [`geometric`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.geometric.html#numpy.random.geometric) | Draw samples from the geometric distribution. |
| [`gumbel`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.gumbel.html#numpy.random.gumbel) | Draw samples from a Gumbel distribution. |
| [`hypergeometric`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.hypergeometric.html#numpy.random.hypergeometric) | Draw samples from a Hypergeometric distribution. |
| [`laplace`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.laplace.html#numpy.random.laplace) | **Draw samples from the Laplace or double exponential distribution with specified location (or mean) and scale (decay).** |
| [`logistic`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.logistic.html#numpy.random.logistic) | Draw samples from a logistic distribution. |
| [`lognormal`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.lognormal.html#numpy.random.lognormal) | Draw samples from a log-normal distribution. |
| [`logseries`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.logseries.html#numpy.random.logseries) | Draw samples from a logarithmic series distribution. |
| [`multinomial`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multinomial.html#numpy.random.multinomial) | Draw samples from a multinomial distribution. |
| [`multivariate_normal`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multivariate_normal.html#numpy.random.multivariate_normal) | Draw random samples from a multivariate normal distribution. |
| [`negative_binomial`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.negative_binomial.html#numpy.random.negative_binomial) | Draw samples from a negative binomial distribution. |
| [`noncentral_chisquare`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.noncentral_chisquare.html#numpy.random.noncentral_chisquare) | Draw samples from a noncentral chi-square distribution. |
| [`noncentral_f`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.noncentral_f.html#numpy.random.noncentral_f) | Draw samples from the noncentral F distribution. |
| [`normal`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html#numpy.random.normal) | Draw random samples from a normal (Gaussian) distribution. |
| [`pareto`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.pareto.html#numpy.random.pareto) | Draw samples from a Pareto II or Lomax distribution with specified shape. |
| [`poisson`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.poisson.html#numpy.random.poisson) | Draw samples from a Poisson distribution. |
| [`power`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.power.html#numpy.random.power) | Draws samples in [0, 1] from a power distribution with positive exponent a - 1. |
| [`rayleigh`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rayleigh.html#numpy.random.rayleigh) | Draw samples from a Rayleigh distribution. |
| [`standard_cauchy`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.standard_cauchy.html#numpy.random.standard_cauchy) | Draw samples from a standard Cauchy distribution with mode = 0. |
| [`standard_exponential`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.standard_exponential.html#numpy.random.standard_exponential) | Draw samples from the standard exponential distribution. |
| [`standard_gamma`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.standard_gamma.html#numpy.random.standard_gamma) | Draw samples from a standard Gamma distribution. |
| [`standard_normal`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.standard_normal.html#numpy.random.standard_normal) | Draw samples from a standard Normal distribution (mean=0, stdev=1). |
| [`standard_t`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.standard_t.html#numpy.random.standard_t) | Draw samples from a standard Student’s t distribution with *df* degrees of freedom. |
| [`triangular`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.triangular.html#numpy.random.triangular) | Draw samples from the triangular distribution over the interval `[left, right]`. |
| [`uniform`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.uniform.html#numpy.random.uniform) | Draw samples from a uniform distribution. |
| [`vonmises`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.vonmises.html#numpy.random.vonmises) | Draw samples from a von Mises distribution. |
| [`wald`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.wald.html#numpy.random.wald) | Draw samples from a Wald, or inverse Gaussian, distribution. |
| [`weibull`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.weibull.html#numpy.random.weibull) | Draw samples from a Weibull distribution. |
| [`zipf`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.zipf.html#numpy.random.zipf) | Draw samples from a Zipf distribution.   |



### 排序，搜索，统计

[完整文档](https://docs.scipy.org/doc/numpy/reference/routines.sort.html)

#### Sorting

| Sorting                                  |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`sort`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sort.html#numpy.sort) | Return a sorted copy of an array.        |
| [`lexsort`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.lexsort.html#numpy.lexsort) | Perform an indirect stable sort using a sequence of keys. |
| [`argsort`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html#numpy.argsort) | Returns the indices that would sort an array. |
| [`ndarray.sort`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.sort.html#numpy.ndarray.sort) | Sort an array, in-place.                 |
| [`msort`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.msort.html#numpy.msort) | Return a copy of an array sorted along the first axis. |
| [`sort_complex`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sort_complex.html#numpy.sort_complex) | Sort a complex array using the real part first, then the imaginary part. |
| [`partition`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.partition.html#numpy.partition) | Return a partitioned copy of an array.   |
| [`argpartition`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argpartition.html#numpy.argpartition) | Perform an indirect partition along the given axis using the algorithm specified by the *kind*keyword. |

```python
### 常用函数

# .sort 默认以最后一维排序。返回的是一个copy。
>>> a = np.array([[1,4],[3,1]])
>>> np.sort(a)                # sort along the last axis
array([[1, 4],
       [1, 3]])
>>> np.sort(a, axis=None)     # sort the flattened array
array([1, 1, 3, 4])
>>> np.sort(a, axis=0)        # sort along the first axis
array([[1, 1],
       [3, 4]])

# .ardsort() 功能和.sort一致，但返回的是索引值。
>>> x = np.array([3, 1, 2])
>>> np.argsort(x)
array([1, 2, 0])
```

#### Searching

| Searching                                |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`argmax`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html#numpy.argmax) | Returns the indices of the maximum values along an axis. |
| [`nanargmax`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanargmax.html#numpy.nanargmax) | Return the indices of the maximum values in the specified axis ignoring NaNs. |
| [`argmin`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html#numpy.argmin) | Returns the indices of the minimum values along an axis. |
| [`nanargmin`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanargmin.html#numpy.nanargmin) | Return the indices of the minimum values in the specified axis ignoring NaNs. |
| [`argwhere`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argwhere.html#numpy.argwhere) | Find the indices of array elements that are non-zero, grouped by element. |
| [`nonzero`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html#numpy.nonzero) | Return the indices of the elements that are non-zero. |
| [`flatnonzero`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.flatnonzero.html#numpy.flatnonzero) | Return indices that are non-zero in the flattened version of a. |
| [`where`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html#numpy.where) | Return elements, either from *x* or *y*, depending on *condition*. |
| [`searchsorted`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html#numpy.searchsorted) | Find indices where elements should be inserted to maintain order. |
| [`extract`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.extract.html#numpy.extract) | Return the elements of an array that satisfy some condition. |

```Python
### 常用函数

# .argmax 返回最大值的索引  .nanargmax 与其相似
# .argmin .nanargmin 同理
>>> a = np.arange(6).reshape(2,3)
>>> a
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.argmax(a)             # 默认会将数组进行flattened
5
>>> np.argmax(a, axis=0)
array([1, 1, 1])
>>> np.argmax(a, axis=1)
array([2, 2])

# .argwhere() 寻找非0的值的索引，等同于np.transpose(np.nonzero(a))
# 结合一些简单的对比运算，可以搜索更多的条件。
>>> x = np.arange(6).reshape(2,3)
>>> x
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.argwhere(x>1)
array([[0, 2],
       [1, 0],
       [1, 1],
       [1, 2]])

# .nonzero() 寻找非0值索引
>>> x = np.array([[1,0,0], [0,2,0], [1,1,0]])
>>> x
array([[1, 0, 0],
       [0, 2, 0],
       [1, 1, 0]])
>>> np.nonzero(x)
(array([0, 1, 2, 2]), array([0, 1, 0, 1]))
>>> np.transpose(np.nonzero(x))             # 等同于.argwhere()
array([[0, 0],
       [1, 1],
       [2, 0],
       [2, 1]])
```

#### Counting

| counting                                 |                                          |
| ---------------------------------------- | ---------------------------------------- |
| [`count_nonzero`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.count_nonzero.html#numpy.count_nonzero)(a[, axis]) | Counts the number of non-zero values in the array a |

```python
# .cout_nonzero() 返回非零值的个数。
>>> np.count_nonzero(np.eye(4))     # 默认axis=None,及对flattened数组进行操作。
4
>>> np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]])
5
>>> np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]], axis=0)
array([1, 1, 1, 1, 1])
>>> np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]], axis=1)
array([2, 3])
```





















