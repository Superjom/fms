编译&&安装
============
首先安装第三方依赖，所有的第三方依赖在third中

    cd third
    mkdir -p local/glog
    cd ../glog-0.3.3
    ./configure --prefix=`pwd`/../local/glog
    make && make install

其次，本代码需要支持C++11，可以采用gcc4.8+编译

需要将g++编译器的绝对地址设定到 Makefile 中 **CXX** 选项

    cd src
    make

生成的二进制文件在bin中

FM框架
=======

总体的模型框架都是，FM为底层打分，上层采用不同的损失函数来拟合不同类型的目标。

KL-FM
==========
利用FM回归概率

底层FM打分，然后采用sigmoid函数将打分归一成概率(0~1)

利用KL距离作为损失函数

使用
-----
总体流程拆成 train/predict

    ./bin/kl_fm_train.out   //进行训练
    ./bin/kl_fm_predict.out //进行预测

数据格式
---------
Data格式(标准的libFM格式):

    target id1:v1 id2:v2 id3:v3
    0.7 2323:0.7 34343:0.2 3432232:-0.7

IDData格式(默认命中的id的value为1，否则为0):

    target id1 id2 id3
    0.7 2323 34343 3432232


性能比较
----------
支持多线程

174W数据/10维参数

单线程: 10轮/3m46s

10线程: 10轮/56s

PairFM
=========
**待测试!**
利用FM结合pairwise rank 学习order.

使用
-----
目前只支持 train 以及 fm模型输出

    ./bin/pair_fm_train.out

数据格式
---------
暂时只支持IDData，输入每行是一个list，工具会自动拆分出两两组合进行学习。

ListInstance的IDData格式：

    common-prefix-features \t score1 key:val key:val \t score2 key:val key:val \t score3 key:val key:val key:val 

其中， common-prefix-features 指的是共同的feature前缀(类似共同的prequery的特征)，格式也是 key:val key:val，在模型计算时，会将common-prefix-features加到后面每个Instance中。

具体的格式的例子如下：

    1:2.7 3:3.1 \t 0.72 4:1.2 5:7.2 \t 0.8 4:0.6 5:1.1

TODO
------
1. 一个pair中，前后关系的定义(> < =三者的定义)
2. log输出的损失为以list为单位的pair正确率，可能变化较小；若改成以pair为单位会比较直观
3. 只包含了二阶FM的打分，后续需要添加一阶打分
