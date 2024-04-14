---
title: 2029-AutoInt Automatic Feature Interaction Learning via Self-Attentive Neural Networks
date: 2024-04-14 12:12:12 +/-TTTT
categories: [推荐系统]
tags: [特征交叉]    
math: true
---
## 背景
推荐系统CTR预估任务中的两个难点：
(1)输入特征多是稀疏高维特征，(2)高阶特征交叉依赖人工设计
本文提出AutoInt方法，基于多头注意力结构自动学习高阶特征交叉信息

## 方法
本文中采用Key-value 注意力机制来决定特征怎样结合是有意义的
<div align=center>
<img src="/assets/img/AutoInt-4.png" alt="alt text" width="400"/>
</div>
每个注意力机制头有三个矩阵，$$W_{Query},W_{Key},W_{Value}$$

以特征m为例，在第h和attention head中，
特征m和特征k的相关性定义为
<div align=center>
<img src="/assets/img/AutoInt-1.png" alt="alt text" width="300"/>
</div>
第m个特征的embedding表示为第m个向量与其他向量相关性的加权和
<div align=center>
<img src="/assets/img/AutoInt-2.png" alt="alt text" width="200"/>
</div>
第m个向量最终表示为多头输出的拼接

最后，通过残差连接保留低阶特征
<div align=center>
<img src="/assets/img/AutoInt-3.png" alt="alt text" width="200"/>
</div>

