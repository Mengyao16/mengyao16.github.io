---
title: Transformer源码阅读
date: 2024-01-06 12:12:12 +/-TTTT
categories: [推荐系统, 源码阅读]
tags: [source code]     # TAG names should always be lowercase
math: true
---

# Transformer代码理解
代码来源：https://github.com/Kyubyong/transformer/blob/master/modules.py

input : 输入向量，维度为 $batch\times seq\times feature$,$seq$表示序列长度，$feature$表示词向量的embedding维度
epsilon : 防止除数变成0
outputs : 输出向量，维度与input相同


## module.py

### Layer Normalization
层标准化函数实现在特征维度上标准化，在最后一个维度上计算均值和标准差

```python
def ln(inputs, epsilon = 1e-8, scope="ln"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 获得输入向量维度列表[batch,seq,feature]
        inputs_shape = inputs.get_shape()
        # params_shape = feature
        params_shape = inputs_shape[-1:]
        # 计算输入向量inputs在最后一个维度(特征维度)上的均值和方差,[batch,seq,1]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        # 线性变化beta,shape:[feature],gamma:[feature]
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        # 归一化之后的结果
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        # 归一化之后做线性变换，可学习参数kx+b
        outputs = gamma * normalized + beta
    return outputs
```
### token embedding
此函数用于初始化词向量，token embedding是可学习的想来根，输入参数vocab_size表示单词表长度V，num_units表示词嵌入维度E
zero_pad=True将第一行token embedding置0
```python
def get_token_embeddings(vocab_size, num_units, zero_pad=True):
   '''
    Returns
    weight variable: (V, E)
    '''
    with tf.variable_scope("shared_weight_matrix"):
        embeddings = tf.get_variable('weight_mat',
                                   dtype=tf.float32,
                                   shape=(vocab_size, num_units),
                                   initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings
```

### multihead_attention
```python
def multihead_attention(queries, keys, values, key_masks,
                        num_heads=8, 
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        # tf.layers.dense相当于全连接层,output = activate(input*kernel + bias)
        Q = tf.layers.dense(queries, d_model, use_bias=True) # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=True) # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=True) # (N, T_k, d_model)
        
        # Split and concat
        # Q矩阵被split成[N, T_q, d_model/num_heads]维度的矩阵list
        # concat操作后变成[h*N,T_q, d_model/h]
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)

        # Restore shape
        # 合并多头注意力
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, d_model)
              
        # Residual connection
        # 残差连接
        outputs += queries
              
        # Normalize
        outputs = ln(outputs)
 
    return outputs
```
### scaled_dot_product_attention
核心attention计算公式

$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_{k}}})V
$$

```python
def scaled_dot_product_attention(Q, K, V, key_masks,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 获得Q矩阵最后一位的维度，也就是query的编码维度
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        # Q矩阵*乘K矩阵的转置矩阵[N, T_q, d_k]*[N, d_k, T_k]
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, key_masks=key_masks, type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # softmax 默认axis=-1,在最后一个维度上做softmax
        outputs = tf.nn.softmax(outputs)
        # attention [N,T_k,T_q],这里为什么要做转置？这段应该没有核心计算作用
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)


        # weighted sum (context vectors)
        # output:(N, T_q, T_k),V:(N, T_k, d_v)
        outputs = tf.matmul(outputs, V)  
        # output (N, T_q, d_v)


    return outputs
```
### mask
```python
def mask(inputs, key_masks=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N, 1, T_k)
    type: string. "key" | "future"

    e.g.,
    >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
    >> key_masks = tf.constant([[0., 0., 1.],
                                [0., 1., 1.]])
    >> mask(inputs, key_masks=key_masks, type="key")
    array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],

       [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
    """
    # 一个绝对值非常大的负数
    padding_num = -2 ** 32 + 1
    # padding mask操作
    if type in ("k", "key", "keys"):
        # 数据类型转化操作
        key_masks = tf.to_float(key_masks)
        # 输入key_masks 维度(N,seqlen)
        # tf.shape(inputs)[0] = h * N
        # tf.shape(key_masks)[0] = N
        # key_masks = tf.tile(key_masks,[h, 1])
        # 输出key_masks (h*N, seqlen)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1]) # (h*N, seqlen)
        # 输出key_masks (h*N, 1, seqlen)
        key_masks = tf.expand_dims(key_masks, 1)  
        # (h*N, seq_len, T_k) + (h*N, 1, seqlen) *padding_num
        outputs = inputs + key_masks * padding_num
   # seq mask操作
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs
```

### feedforward
前馈神经网络部分
```python
def ff(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3
    
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        # 全连接层+relu激活
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        # 全连接层
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = ln(outputs)
    
    return outputs
```
## train.py

### encoder

```python
def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # x: int32 tensor. (N, T1)
            # x_seqlens: int32 tensor. (N,)
            # sents1: str tensor. (N,)
            x, seqlens, sents1 = xs

            # src_masks
            # boolean矩阵，如果=0，为True，否则False
            src_masks = tf.math.equal(x, 0) # (N, T1)

            # embedding
            # 构建特征维度矩阵
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            # 乘以特征维度的开方做标准化
            enc *= self.hp.d_model**0.5 # scale
            # 加上位置向量编码
            enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory, sents1, src_masks
```
### decoder
```python
def decode(self, ys, memory, src_masks, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)
        src_masks: (N, T1)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys

            # tgt_masks
            tgt_masks = tf.math.equal(decoder_inputs, 0)  # (N, T2)

            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)  # (N, T2, d_model)
            dec *= self.hp.d_model ** 0.5  # scale

            dec += positional_encoding(dec, self.hp.maxlen2)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              key_masks=tgt_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings) # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat, y, sents2
```
### train
```python
def train(self, xs, ys):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        # 编码器输出memory: (N, T1, d_model)
        memory, sents1, src_masks = self.encode(xs)
        # 解码器输入  src_masks: (N, T1)
        logits, preds, y, sents2 = self.decode(ys, memory, src_masks)

        # train scheme
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<pad>"]))  # 0: <pad>
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

```
