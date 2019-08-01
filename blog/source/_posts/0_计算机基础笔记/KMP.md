---
title: KMP
date: 2019-07-08 20:13:13
tags: [笔记,计算机基础,KMP]
categories: [计算机基础,KMP]
keywords: [计算机基础,KMP]
description: 纯粹想到什么写什么，做个知识的总结
image: /0.png
---



# KMP简单实现

```java
//        ABC ABCD
//i=0,j=1 ABCD
//i=1,j=2 ABCD
//i=2,j=3 ABCD 
//i=3,j=0 ABCD 失败j回退
//i=4,j=1     ABCD
//i=4,j=2     ABCD
//i=4,j=3     ABCD
//i=4,j=4     ABCD 成功
public static void main(String[] args){
    String str1 = "BBC ABCDAB ABCDABCDABDE";
    String str2 = "ABCDABD";//[0,0,0,0,1,2,0]
    
    // kmpSearch 和 kmpNext 的代码基本相同，实际上都是在找一个字串（i）能和目标str2有多少重叠 
    int[] next = kmpNext(str2);
    int index = kmpSearch(str1,str2,next);
}

// 返回匹配的首位置
public static int kmpSearch(String str1,String str2,int[] next){
    for(int i=0,j=0;i<str1.length();i++){
       while(j>0 && str1.charAt(i) != str2.charAt(j)){
           j = next[j-1]; // 只需j回退指定位置，i不用回退。
       } 
        if(str1.charAt(i) == str2.charAt[j]){
            j++;
        }
        if(j == str2.length()){
            return i-j+1;
        }
    }
    return -1；
}


//获得匹配字符串的回退数组
public static int[] kmpNext(String dest){
    int[] next = new int[dest.length()];
    next[0]=0;
    //设置每一位的回退距离
    for(int i=1,j=0;i<dest.length();i++){
        //不符合，j回退
        while(j>0 && dest.charAt(i) != dest.charAt(j)){
            j = next[j-1];
        }
        if(dest.charAt(i) == dest.charAt(j)){
            j++;
        }
        next[i] = j;
    }
    return next;
}

```



 

