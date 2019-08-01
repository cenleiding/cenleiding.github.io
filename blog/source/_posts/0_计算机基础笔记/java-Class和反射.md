---
title: Class和反射
date: 2019-06-18 20:13:13
tags: [笔记,计算机基础,Class和反射]
categories: [计算机基础,Class和反射]
keywords: [计算机基础,Class和反射]
description: 纯粹想到什么写什么，做个知识的总结
image: /0.png
---



# Class和反射

## Class对象

　　我们写的每一个类都是对象，是 java.lang.Class 类的对象。当我们编写并且编译一个新创建的类就会产生一个对应Class对象并且这个Class对象会被保存在同名.class文件里(**编译后的字节码文件保存的就是Class对象，保存了类的相关类型信息**)。

　　**当我们new一个新对象或者引用静态成员变量时，Java虚拟机(JVM)中的类加载器子系统会将对应Class对象加载到JVM中，**然后JVM再根据这个类型信息相关的Class对象创建我们需要实例对象或者提供静态变量的引用值。需要特别注意的是，**手动编写的每个class类，无论创建多少个实例对象，在JVM中都只有一个Class对象，即在内存中每个类有且只有一个相对应的Class对象。**


```java
public final class Class<T> implements java.io.Serializable,GenericDeclaration,Type, AnnotatedElement{
    /*
     * 匿名构造器，只能JVM来创建类对象！
     * Private constructor. 
     * Only the Java Virtual Machine
     * creates Class objects.
     */
    private Class(ClassLoader loader) {
        classLoader = loader;
    }
}
```

### 获取类对象

```java
class MyClass{}
MyClass mClass1 = new MyClass();

// 每个类都有一个隐含的静态成员class
Class c1 = MyClass.class;

// 每个类都有getClass()方法
Class c2 = mClass1.getClass();

// 最最常用，只需一条字符串,不需要有对象不需要已经导入包。
// 注意，Class.forName()需要传入类的全路径,如果当前类与参数类在同一包下即可省略包名
// 实现类的动态加载，如果目标类还未被加载，则JVM会去加载该类。
try {
    mClass = Class.forName("custom.OtherClass");
    } catch (ClassNotFoundException e) {
    e.printStackTrace();
    }
```

注意点：

- 实例类的getClass方法和Class类的静态方法forName都将会触发类的初始化阶段，而字面常量获取Class对象的方式则不会触发初始化。

- 初始化是类加载的最后一个阶段，也就是说完成这个阶段后类也就加载到内存中(Class对象在加载阶段已被创建)，此时可以对类进行各种必要的操作了（如new对象，调用静态成员等），注意在这个阶段，才真正开始执行类中定义的Java程序代码或者字节码。

关于类加载的初始化阶段，在虚拟机规范严格规定了有且只有5种场景必须对类进行初始化：

- 使用new关键字实例化对象时、读取或者设置一个类的静态字段(不包含编译期常量)以及调用静态方法的时候，必须触发类加载的初始化过程(类加载过程最终阶段)。

- 使用反射包(java.lang.reflect)的方法对类进行反射调用时，如果类还没有被初始化，则需先进行初始化，这点对反射很重要。

- 当初始化一个类的时候，如果其父类还没进行初始化则需先触发其父类的初始化。

-  当Java虚拟机启动时，用户需要指定一个要执行的主类(包含main方法的类)，虚拟机会先初始化这个主类

- 当使用JDK 1.7 的动态语言支持时，如果一个java.lang.invoke.MethodHandle 实例最后解析结果为REF_getStatic、REF_putStatic、REF_invokeStatic的方法句柄，并且这个方法句柄对应类没有初始化时，必须触发其初始化

### 获取操作类信息

```java
class MyClass{}
MyClass mClass1 = new MyClass();

// 获得类对象
Class mClass = MyClass.class;

// 获取所有 public 访问权限的变量
// 包括本类声明的和从父类继承的
Field[] fields = mClass.getFields();

// 获取所有本类声明的变量（不问访问权限）
Field[] fields = mClass.getDeclaredFields();

// 获取所有 public 访问权限的方法
// 包括自己声明和从父类继承的
Method[] mMethods = mClass.getMethods();

// 获取所有本类的的方法（不问访问权限）
Method[] mMethods = mClass.getDeclaredMethods();

// 调用指定对象的私有方法
// 第一个参数为要获取的私有方法的名称
// 第二个为要获取方法的参数的类型，参数为 Class...，没有参数就是null
// 方法参数也可这么写 ：new Class[]{String.class , int.class}
Method privateMethod =mClass.getDeclaredMethod("privateMethod", String.class, int.class);
if (privateMethod != null) {
    //获取私有方法的访问权
    //只是获取访问权，并不是修改实际权限
    privateMethod.setAccessible(true);
    //使用 invoke 反射调用私有方法
    //privateMethod 是获取到的私有方法
    //mClass1 要操作的对象
    //后面两个参数传实参
    privateMethod.invoke(mClass1, "Java Reflect ", 666);
}

// 修改指定对象的私有变量
Field privateField = mClass.getDeclaredField("MSG");
// 操作私有变量
if (privateField != null) {
    //获取私有变量的访问权
    privateField.setAccessible(true);
    //调用 set(object , value) 修改变量的值
    //privateField 是获取到的私有变量
    //mClass1 要操作的对象
    //"Modified" 为要修改成的值
        privateField.set(mClass1, "Modified");
}
```

### 类型强制转换

关键在于向上转型之后对象的class并不会变。

```java
class A {}
class B extends A {}

public class C {
  static void test(Object x) {
    print("Testing x of type " + x.getClass());
    print("x instanceof A " + (x instanceof A));
    print("x instanceof B "+ (x instanceof B));
    print("A.isInstance(x) "+ A.class.isInstance(x));
    print("B.isInstance(x) " + B.class.isInstance(x));
    print("x.getClass() == A.class " + (x.getClass() == A.class));
    print("x.getClass() == B.class " + (x.getClass() == B.class));
    print("x.getClass().equals(A.class)) "+ (x.getClass().equals(A.class)));
    print("x.getClass().equals(B.class)) " + (x.getClass().equals(B.class)));
  }
  public static void main(String[] args) {
    test(new A());
    test(new B());
  } 
}
// 虽然向上转型了但是class信息不会变
Testing x of type class com.zejian.A
x instanceof A true
x instanceof B false 
A.isInstance(x) true
B.isInstance(x) false
x.getClass() == A.class true
x.getClass() == B.class false
x.getClass().equals(A.class)) true
x.getClass().equals(B.class)) false
---------------------------------------------
Testing x of type class com.zejian.B
x instanceof A true
x instanceof B true
A.isInstance(x) true
B.isInstance(x) true
x.getClass() == A.class false
x.getClass() == B.class true
x.getClass().equals(A.class)) false
x.getClass().equals(B.class)) true
```





## 反射机制

　　Java 反射机制在程序**运行时**，对于任意一个类，都能够知道这个类的所有属性和方法；对于任意一个对象，都能够调用它的任意一个方法和属性。这种**动态获取的信息**以及**动态调用对象的方法**的功能称为 java 的反射机制。其使得我们可以在程序运行时加载、探索以及使用编译期间完全未知的 .class文件。

用途：各类框架的实现支柱。框架可以主动管理类文件，可以根据配置文件动态生成相应类等，框架可以成为管理者。

### Constructor类

用于新建对象。

```java
// 获取类对象
Class<?> clazz = Class.forName("reflect.User");

//第一种方法，实例化默认构造方法，User必须无参构造函数,否则将抛异常
User user = (User) clazz.newInstance();

//获取带String参数的public构造函数
Constructor cs1 =clazz.getConstructor(String.class);
User user1= (User) cs1.newInstance("xiaolong");

//取得指定带int和String参数构造函数,该方法是私有构造private
Constructor cs2=clazz.getDeclaredConstructor(int.class,String.class);
cs2.setAccessible(true);
User user2= (User) cs2.newInstance(25,"lidakang");

// 还有很多其他各种用途的接口。。
```

### Field类

Field 提供有关类或接口的单个字段的信息，以及对它的动态访问权限。反射的字段可能是一个类（静态）字段或实例字段。

```java
Class<?> clazz = Class.forName("reflect.Student");
//获取指定字段名称的Field类,注意字段修饰符必须为public而且存在该字段,
// 否则抛NoSuchFieldException
Field field = clazz.getField("age");

//获取所有修饰符为public的字段,包含父类字段,注意修饰符为public才会获取
Field fields[] = clazz.getFields();
for (Field f:fields) {
    System.out.println("f:"+f.getDeclaringClass());
}

//获取当前类所字段(包含private字段),注意不包含父类的字段
Field fields2[] = clazz.getDeclaredFields();
for (Field f:fields2) {
    System.out.println("f2:"+f.getDeclaringClass());
}

//获取指定字段名称的Field类,可以是任意修饰符的自动,注意不包含父类的字段
Field field2 = clazz.getDeclaredField("desc");

// 还有很多其他各种用途的接口。。
```

### Method类

Method 提供关于类或接口上单独某个方法（以及如何访问该方法）的信息，所反映的方法可能是类方法或实例方法（包括抽象方法）。

```java
Class clazz = Class.forName("reflect.Circle");
//根据参数获取public的Method,包含继承自父类的方法
Method method = clazz.getMethod("draw",int.class,String.class);

//获取所有public的方法:
Method[] methods =clazz.getMethods();
for (Method m:methods){
    System.out.println("m::"+m);
}
```











