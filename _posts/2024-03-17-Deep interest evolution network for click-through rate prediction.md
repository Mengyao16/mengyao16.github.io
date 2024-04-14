---
title: 2019-Deep interest evolution network for click-through rate prediction
date: 2024-03-17 12:12:12 +/-TTTT
categories: [推荐系统]
tags: [多兴趣建模]    
math: true
---

# Motivation
但是大多该类模型将用户的行为直接看做兴趣，而用户的潜在兴趣往往很难通过行为来完全表示。因此需要挖掘行为背后的用户真实兴趣，并考虑用户兴趣的动态变化。

# 方法

模型结构

<div align=center>
<img src="/assets/img/DIEN.png" alt="alt text" width="700"/>
</div>

## 兴趣抽取层
### 网络结构
兴趣抽取层的作用是从用户行为序列中学习用户的隐式兴趣，本文将用户的历史行为作为时序输入，通过GRU网络来提取用户的兴趣

<div align=center>
<img src="/assets/img/GRU.png" alt="alt text" width="300"/>
</div>

### 损失函数
但是用户对目标item的点击行为可能由用户的最后一个兴趣触发，本文通过引入辅助loss，通过行为 $$b_{t+1}$$ 来指导 $$h_{t}$$ 的学习，即正样本为用户行为序列中的下一个行为，负样本为用户历史行为中采样的负样本
辅助损失函数：

<div align=center>
<img src="/assets/img/DIEN-aux-loss.png" alt="alt text" width="250"/>
</div>

整体损失函数：
$$L = L_{target} + \alpha * L_{aux}$$


## 兴趣进化层
主要目标是刻画用户兴趣的进化过程，对兴趣演变顺序建模的好处：
- 追踪用户的interest可以使我们学习final interest的表达时包含更多的历史信息。
- 可以根据interest的变化趋势更好地进行CTR预测。
文中使用attention + GRU的方式来提取兴趣进化过程，

- attention的计算方式

$$
a_{t}= \frac{exp(h_{t}We_{a})}{ {\textstyle \sum_{j=1}^{t}}exp(h_{t}We_{a}) } 
$$

- attention与GRU的结合方式
  - AIGRU,对第二个GRU的输入乘以attention权重
  
  $$
  i_{t}' = h_{t} * a_{t}
  $$

  - AGRU：使用attention代替GRU中的更新门

  $$
  h_{t}' = (1-a_{t})h_{t-1}'+a_{t}\tilde{h}_{t}' 
  $$

  - AUGRU：将以上的标量操作修改为向量操作
  
  $$
  \tilde{u}_{t}' = a_{t}u_{t}'
  $$

  $$
  h_{t}' = (1-\tilde{u}_{t}')\circ (h_{t-1})' + \tilde{u}_{t}'\circ \tilde{h}_{t}'
  $$

