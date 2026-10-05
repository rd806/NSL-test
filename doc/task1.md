# 任务1：基础实现

## 前置知识

### Tensor（张量）

Tensor 是一种多维数组。

> 可以认为：
> * 标量 = 0 维张量
> * 向量 = 1 维张量
> * 矩阵 = 2 维张量
> * 更高维数组 = n 维张量

Tensor 是深度学习框架中的多维数组，既能像 NumPy 一样做数值计算，又能利用 GPU 加速，还支持自动求导，是神经网络训练中的基本数据单位。

### `@` 运算符

`@` 在 PyTorch 中作为矩阵乘法运算符，例如执行

```py
C = A @ B
```

就是进行矩阵运算：

$$\bm{C} = \bm{A} \times \bm{B}$$

`@` 对最后两维做矩阵乘法，前面的维度按广播处理。

## 任务实现

### gelu

直接使用提供的公式：

$$\frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]$$

其中的双曲余弦使用 `torch.tanh` 方法。

> 三次方运算使用 `x**3`， 它一般情况下比 `torch.pow(x, 3)` 更快。

### softmax

该函数的主要作用是把一组实数转换成概率分布（每个值在 0~1 之间，且总和为 1）。查阅资料得知当前主流的 `softmax` 定义为 

$$\text{softmax}(\textbf{z})_i=\frac{e^z_i}{\displaystyle \sum_{j=1}^{K}{e^z_j}}$$

其中 $z$ 为一个向量，相比于普通的加权平均算法，对数值作指数运算可以放大相应的权重。
实际运算时，若 $z_i$ 的值过大会出现未定义的错误，此时可以先减去最大值，

$$\text{softmax}(\textbf{z})_i=\frac{e^{z_i-m}}{\displaystyle \sum_{j=1}^{K}{e^{z_j-m}}}$$

这样最大的数的权重就可以保持为1，即所有权重都将被压缩至 $[0,1]$ 区间内。

### layer_norm

主要作用是对样本的特征向量进行标准化。因此需要求出输入数据的均值和方差：

$$\mu =\frac{1}{d}\sum_{i=1}^{d}{x_i}，{\sigma^2} =\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2$$

然后对其进行标准化（参考正态分布的标准化处理）：

$$\hat{x}_i=\frac{x_i-\mu}{\sigma}$$

但是实际运行时可能出现方差恰好为0的情况，因此还需要修改（输入正好有一个eps）：

$$\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2 + \varepsilon}}$$

但是，强行标准化可能会带来一些问题，最后输出时还要加上 $\gamma$ 和 $\beta$：

> 这里还需要我进一步学习，这是查资料和任务提示的结果。

$$y_i=\gamma_i \hat{x}_i +\beta_i$$

### ffn 

标准 FFN 的公式如下：

$$\text{FFN}(x)=W_2\sigma(W_1x+b_1)+b_2$$

在本任务中，$b_1$ 和 $b_2$ 来自获取到的GPT-2 的 TensorFlow checkpoint 里的变量名 `[c_fc]` 和 `[c_proj]`

### attention

查阅资料得知 `attention` 是一个加权求和：

$$ \text{Attention}(Q,K,V)=\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V $$

其中 $Q$，$K$，$V$ 均已在输入给出，关键是得到 $d_k$ 和 $K^T$。
* 对于 $d_k$：在 Attention 中，$d_k$ 表示 Key 的维度，在设计网络时人为设定。在本任务中，由于得到的恰好为张量，则可以直接读取最后一维的值，即 `d_k = k.shape[-1]`。
* 对于 $K_T$：张量的转置针对最后两维的数据，因此可以直接使用 `K.transpose(-2, -1)` 计算。

### mha

包含3个子任务：
1. 从张量 $x$ 中得到 $qkv$：这里使用 `chunk` 函数调用：`x.chunk(3, dim=-1)`，按最后一个维度均匀划分为 3 份。
2. 构建 mask 矩阵：该矩阵的样式如下，实现思路是先生成一个全为负无穷的方阵，然后保留主对角线向上偏移一格的右上角范围内的元素。
3. 张量拼接：使用 `torch.cat(tensors, dim)` 函数，沿最后一个维度拼接（即 `dim = -1`）。

$$\text{mask} = \displaystyle \left [{\begin{matrix}0 & -\infty & -\infty & \cdots & -\infty \\0 & 0 & -\infty & \cdots & -\infty \\ 0 & 0 & 0 & \cdots & -\infty \\ \cdots & \cdots & \cdots & \cdots & -\infty \\ 0 & 0 & 0 & \cdots & 0 \end{matrix}} \right ]$$