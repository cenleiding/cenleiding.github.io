---
title: docker发布TensorFlow
date: 2018-11-20 21:10:04
tags: [docker,TensorFlow serving]
categories: 随笔
keywords: [docker,端口问题,TensorFlow serving] 
description: 使用docker发布TensorFlow的模型，并解决windows端口映射问题
image: 
---

只是为了简单发布 TensorFlow/serving 服务…… 

docker下载地址https://store.docker.com/editions/community/docker-ce-desktop-windows

使用`Docker Quickstart terminal ` 快速初始化，建立Linux虚拟机，分配虚拟ip等等

## 环境

```java
//查看docker环境
$ docker version
$ docker info

//从仓库下载image,可以利用kitematic方便从docker hub查找
$ docker pull [OPTIONS] NAME[:TAG|@DIGEST]		
// 镜像加速
$ docker pull registry.docker-cn.com/myname/myrepo:mytag
```

## image

```java
// 查看所有image
$ docker image ls
// 删除相应image
$ docker image rm [imageID]
```

## container 操作

```java
// 查看所有容器
$ docker ps -a
// 查看容器日志
$ docker logs [OPTIONS] CONTAINER
```

## container 生命周期

```java
// 创建新的容器并运行
docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
option：
-p: 端口映射，格式为：主机(宿主)端口:容器端口;
-i: 以交互模式运行容器，通常与 -t 同时使用；
-t: 为容器重新分配一个伪输入终端，通常与 -i 同时使用；
-e: 设置环境变量；
-d: 后台运行容器，并返回容器ID；
-m :设置容器使用内存最大值；
--name="": 为容器指定一个名称；
--mount type=bind,source=,target=  :文件挂载
  详见：https://docs.docker.com/storage/bind-mounts/#choosing-the--v-or---mount-flag
  
// start/stop/restart
$ docker start [OPTIONS] CONTAINER [CONTAINER...]
$ docker stop [OPTIONS] CONTAINER [CONTAINER...]
$ docker restart [OPTIONS] CONTAINER [CONTAINER...]

// 杀掉一个运行中的container
$ docker kill [OPTIONS] CONTAINER [CONTAINER...]

// 移除一个container
$ docker rm [OPTIONS] CONTAINER [CONTAINER...]
```



## 例子：tensorflow/serving

用于tensorflow模型的web发布

一. pull别人已有的image

```java
$ docker pull tensorflow/serving
```

二.将pb文件放到虚拟机共享文件之下

```java
// 注意 虚拟机默认共享文件目录：c:Users.
// 可以通过virtual Box 查看共享文件夹，也可以添加新的共享文件夹。	
```

三.启动容器

```java
// tensorflow/serving image 环境变量
// MODEL_NAME(defaults to model)
// MODEL_BASE_PATH(defaults to /models)
// --model_name=${MODEL_NAME}  模型名
// --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} 模型存储路径
$ docker run \
$ -p 8501:8501 \
$ -e MODEL_NAME=half_plus_two \
$ -i -t -d \
$ --name=test \
$ --mount type=bind,source=/c/Users/saved_model_half_plus_two_cpu,target=/models/half_plus_two \
$ tensorflow/serving
```

四.URL

详见https://www.tensorflow.org/serving/api_rest?hl=zh-cn

```java
//Model status API
GET http://host:port/v1/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]

//Model Metadata API
GET http://host:port/v1/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/metadata

// Predict API
POST http://host:port/v1/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]:predict
{
  // (Optional) Serving signature to use.
  // If unspecifed default serving signature is used.
  //add_meta_graph_and_variables(signature_def_map={name:signature}中的name)
  "signature_name": <string>,

  // Input Tensors in row ("instances") or columnar ("inputs") format.
  // A request can have either of them but NOT both.
  "instances": <value>|<(nested)list>|<list-of-objects>
  "inputs": <value>|<(nested)list>|<object>
}
examples：
{
 "instances": [
   {
     "tag": "foo",
     "signal": [1, 2, 3, 4, 5],
     "sensor": [[1, 2], [3, 4]]
   },
   {
     "tag": "bar",
     "signal": [3, 4, 1, 2, 5]],
     "sensor": [[4, 5], [6, 8]]
   }
 ]
}

//Classify and Regress API
POST http://host:port/v1/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]:(classify|regress)
{
  // Optional: serving signature to use.
  // If unspecifed default serving signature is used.
  "signature_name": <string>,

  // Optional: Common context shared by all examples.
  // Features that appear here MUST NOT appear in examples (below).
  "context": {
    "<feature_name3>": <value>|<list>
    "<feature_name4>": <value>|<list>
  },

  // List of Example objects
  "examples": [
    {
      // Example 1
      "<feature_name1>": <value>|<list>,
      "<feature_name2>": <value>|<list>,
      ...
    },
    {
      // Example 2
      "<feature_name1>": <value>|<list>,
      "<feature_name2>": <value>|<list>,
      ...
    }
    ...
  ]
}
```

## 巨坑--windows端口映射问题

https://github.com/moby/moby/issues/15740

https://blog.sixeyed.com/published-ports-on-windows-containers-dont-do-loopback/

在linux下，端口映射完全没有问题。但是在windows下，就有个巨大问题。

我们按照步骤做完之后发现，并不能愉快的访问http://localhost:... 而且也没法通过局域网访问!!!!!!
但是我们用`$ docker ps` 查看，发现` 0:0:0:0:8080:8080` 端口明明已经转好了！

```
原来docker是运行在Linux上的，在Windows中运行docker，实际上还是在Windows下先安装了一个Linux环境，然后在这个系统中运行的docker。也就是说，服务中使用的localhost指的是这个Linux环境的地址，而不是我们的宿主环境Windows。
```

好的那么我们仔细看一下几个ip：

**容器ip：** 使用`$ docker inspect` 查看容器ip发现是：`172.17.0.2`

**虚拟机ip：**可以到kitematic看，也可以通过`$ docker-machine ip` 查看，发现默认为：`192.168.99.100`

**主机ip：** `172.16.119.212` 

试验一下，发现只能访问**虚拟机ip** 。**这意味着-p只是映射了 容器=》linux虚拟机 之间的端口**，这导致windows下只能用本机访问虚拟机ip,**没法通过局域网访问**， 这往往很蠢。。

**解决方法：**

既然我们缺失了 **主机ip=>虚拟机ip** 的映射，那么我们加上映射不就可以了？
可以在 **VitualBox =》 网络 =》 端口转发 **  中设置转发规则，将**主机ip 和 子系统ip ** 进行映射。

这样我们就可以通过别的电脑来访问容器中的程序了。



