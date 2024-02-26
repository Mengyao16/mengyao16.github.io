---
title: DeepFM&DCN&XDeepFM
date: 2024-01-28 12:12:12 +/-TTTT
categories: [推荐系统]
tags: [精排]    
math: true
---
# DeepFM

## 模型结构
<div align=center>
<img src="/assets/img/DeepFM-model.png" alt="alt text" width="600"/>
</div>

- 将FM与DNN部分结合起来，共享输入embedding，分别学习低阶特征交互和高阶特征交互
- FM部分：
<div align=center>
<img src="/assets/img/DeepFM-Wide.png" alt="alt text" width="600"/>
</div>

$$y_{FM}=<w,x>+\sum_{j_{1}=1}^{d}\sum_{j_{2}=j_{1}+1}^{d}<V_{i},V_{j}>x_{j_{1}}·x_{j_{2}}$$

第一项为一阶特征，第二项为二阶特征

- Deep部分:DNN网络
<div align=center>
<img src="/assets/img/DeepFM-Deep.png" alt="alt text" width="600"/>
</div>

- 损失函数
  $$\hat{y}=sigmoid(y_{FM}+y_{DNN})$$

# DCN-v2
DeepFM的FM部分仅能学习二阶特征之间的关系，本文提出DCN-v2模型，通过cross network学习显式特征交互，通过deep network学习隐式特征交互。
有限高阶、自动叉乘、参数共享。
## 模型结构
DCN-v2中提出两种模型结构，分别是Cross network和Deep network级联与并联。
<div align=center>
<img src="/assets/img/DCN-v2-model.png" alt="alt text" width="600"/>
</div>

## Cross network
Cross network通过n层cross layer显式学习特征交互。cross layer计算方式：
<div align=center>
<img src="/assets/img/DCN-v2-layer.png" alt="alt text" width="400"/>
</div>

$$x_{0}\in\mathbb{R}^{d}$$表示input layer编码的embedding，$$x_l$$和$$x_{l+1}$$分别表示第$l$层cross layer的输入和输出，$$W_{l}\in\mathbb{R}^{d\times d}$$

为了提升计算效率，使用$$UV^{T}$$来代替参数$$W$$，$$U,V\in\mathbb{R}^{d\times r},r\ll d/2$$

# XDeepFM
- DCN的不足:
  - 特征交叉是一种bit-wise的方式，那么同一个field embedding的每个bit都会相互影响。
XDeepFM使用CIN(Compressed Interaction Network，压缩交互网络)将vector-wise的思想引入cross部分。
## 网络结构
Linear和DNN部分和DeepFM一致，重点在于CIN模块。
<div align=center>
<img src="/assets/img/xDeepFM-model.png" alt="alt text" width="600"/>
</div>

## CIN网络
CIN网络的输入来自embedding层，设有$m$个特征field，每个特征的编码维度为$D$,则输入矩阵$$X^{0}\in \mathbb{R}^{m\times D}$$

令$$X^{k}\in \mathbb{R}^{H_{k}\times D} $$为第k层的输出，$H_{k}$表示第k层的向量个数，$D$维度不变，第k层的vector计算方式为

$$
X^{k}_{h,*} = \sum_{i=1}^{H_{k-1}}\sum_{j=1}^{m}W_{ij}^{k,h}(X_{i,*}^{k-1}\circ X_{j,*}^{0})
$$

1. 取k-1层的$X^{k\in \mathbb{R}^{H_{k-1}\times D} }$,中的k-1个vector与$$X^{0}\in \mathbb{R}^{m\times D}$$中的m个vector一次做Hadamard积并加权求和
2. 同一层不同vector的区别在与参数$W$的不同
3. CIN在每一层只包含l+1阶特征，每一层都输出中间结果


  


