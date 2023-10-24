# 基于深度图高斯过程的网络流量预测模型

## 预安装程序库

our implementation is mainly based on `tensorflow 1.x` and `gpflow 1.x`:

```
python 3.x (3.7 tested)
pip install tensorflow-gpu==1.15
pip install keras==2.3.1
pip install gpflow==1.5
pip install gpuinfo
pip install dpkt
```
Besides, some basic packages like `numpy` are also needed.
It's maybe easy to wrap the codes for TF2.0 and GPflow2, but it's not tested yet.
## 使用说明

主程序入口为main.ipynb，在安装好的jupyter notebook环境下，可以直接点击Restart & Run all进行运行。可通过设置变量filename决定输入文件。

sample_output.html为样例输出结果。

## 输出说明

主要输出结果位于Results小节，以样例输出为例：

```
metrics:	[mae | rmse | mape]
ours DGP:	 [11.231, 28.742, 0.858]
Last baseline:	 [17.419, 43.516, 3.537]
5s mean baseline:	 [15.852, 37.469, 1.69]
5s median baseline:	 [14.597, 36.219, 2.281]
```

- 在此评估了三个指标（MAE, RMSE, MAPE），均为值越小越好。

- Last baseline使用上一时刻的流量值作为预测输出。

- 5s mean baseline使用前5s的流量均值作为预测输出。

- 5s median baseline使用前5s的流量中位数作为预测输出。

  在可视化结果部分，橙色曲线为真实流量值，蓝色曲线为预测输出流量值，浅蓝色阴影部代表两个标准差。

## 方法概述

程序基于最新的最新的DGPG（Stochastic Deep Gaussian Processes over Graphs，深度图高斯过程 ）模型，用于对网络中的数据流量进行预测。在预测均值的同时，算法还可以对方差进行预测，从而刻画网络流量的非确定性。重新同时和一些简单的baseline进行了比较，在结果上得到了显著的提升。代码同时包括了可视化模块，对预测输出的均值和方差进行展示。