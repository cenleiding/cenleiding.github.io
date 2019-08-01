---
title: String-StringBuilder-StringBuffer
date: 2019-06-18 20:13:13
tags: [笔记,计算机基础,String-StringBuilder-StringBuffer]
categories: [计算机基础,String-StringBuilder-StringBuffer]
keywords: [计算机基础,String-StringBuilder-StringBuffer]
description: 纯粹想到什么写什么，做个知识的总结
image: /0.png
---



# String-StringBuilder-StringBuffer



## 1.可变与不可变

- **String： 不可变对象,一旦被创建,就不能修改它的值. 对于已经存在的String对象的修改都是重新创建一个新的对象,然后把新的值保存进去.**

```java
public final class String implements ...
{	// 常量池的引用
    private final char value[];
    // 偏移量
	private final int offset;
	// 字符串长度
	private final int count;
	// hash值
	private final int hash;
	// ...   
}
```

- **StringBuilder-StringBuffer：**都继承自AbstractStringBuilder类，在AbstractStringBuilder中也是使用字符数组保存字符串，但不是final类型，所以**都是可变的。** 

```Java
// 初始为16容量，当容量不够时会扩增数组，所以会导致性能降低。
char[] value;
```



## 2. 线程安全

- **String：** 对象都不可变了，就是个常量，**线程安全**。
- **StringBuffer：**添加了同步锁，**线程安全**。
- **StringBuilder：**没有同步锁，**线程不安全**。



## 3. StringBuilder和StringBuffer的区别

- StringBuilder 基本和StringBuffer 一致（实现同一个抽象类），除了没加同步锁。这样一来，**StringBuilder的效率高于StringBuffer，但线程不安全。**



## 4. 为什么设计成不可变

- 为了实现**字符串池**，字符串池能够在运行时**节约很多空间**。当创建一个String对象时,假如此字符串值已经存在于常量池中,则不会创建一个新的对象,而是引用已经存在的对象。
- 允许**固定HashCode**，避免了重复去计算哈希值，**提高运行性能。**
- 自带**线程安全**。
- **系统安全性**，不论是用户名密码还是类加载都要用字符串。如果可变，黑客直接修改字符数组的值，会造成破坏。



## 5. 字符串常量池

- ### 字符串常量池的设计思想

  - 字符串的分配，和其他的对象分配一样，耗费高昂的时间与空间代价，作为最基础的数据类型，大量频繁的创建字符串，极大程度地影响程序的性能
  - JVM为了提高性能和减少内存开销，在实例化字符串常量的时候进行了一些优化
    - 为字符串开辟一个字符串常量池（位于方法区），类似于缓存区
    - 创建字符串常量时，首先检查字符串常量池是否存在该字符串
    - 存在该字符串，返回引用实例，不存在，实例化该字符串并放入池中
  - 实现的基础
    - 实现该优化的基础是因为字符串是不可变的，可以不用担心数据冲突进行共享
    - 运行时实例创建的全局字符串常量池中有一个表，总是为池中每个唯一的字符串对象维护一个引用,这就意味着它们一直引用着字符串常量池中的对象，所以，在常量池中的这些字符串不会被垃圾收集器回收。



## 其他

- **+** 号

  String 调用“+”号，实际上会创建StringBuilder（或StringBuffer）的append()方法，然后返回toString()，而创建的对象则会回收。**所以在循环中不要用“+”,非常影响性能。**

  ```
  String result = "";
  for (String s : hugeArray) {
      result = result + s;
  }
  
  // 使用StringBuilder
  StringBuilder sb = new StringBuilder();
  for (String s : hugeArray) {
      sb.append(s);
  }
  String result = sb.toString();
  ```

  

- JVM的优化：**s1在编译时直接转化为s3**，所以这时s1的效率最高。（相加的都为字面量！）

```Java
String s1 = “This is only a” + “ simple” + “ test”;
StringBuffer Sb = new StringBuilder(“This is only a”).append(“ simple”).append(“ test”);
String s3 = “This is only a simple test”;  
```



- String s = "abc"  和  String s = new String("abc") 是不同的！

![](/String、StringBuilder、StringBuffer/1.png)

  ![](/String、StringBuilder、StringBuffer/2.png)



- .substring() 不会增加常量池对象！只会增加堆的对象！

  会返回 new String(value, beginIndex, subLen)，会新建一个对象，但value值没变！意味着常量池并不会增加新的对象！

  **创建子字符串所需的空间和时间与其长度无关是许多基础字符串处理算法的效率的关键所在！**
  
  **也是常量池的一大优势所在！**

```java
public String substring(int beginIndex, int endIndex) {
        if (beginIndex < 0) {
            throw new StringIndexOutOfBoundsException(beginIndex);
        }
        if (endIndex > value.length) {
            throw new StringIndexOutOfBoundsException(endIndex);
        }
        int subLen = endIndex - beginIndex;
        if (subLen < 0) {
            throw new StringIndexOutOfBoundsException(subLen);
        }
        return ((beginIndex == 0) && (endIndex == value.length)) ? this
                : new String(value, beginIndex, subLen);
    }
```

