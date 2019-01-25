# 多层感知机

我们已经介绍了包括线性回归和softmax回归在内的单层神经网络。然而深度学习主要关注多层模型。在本节中，我们将以多层感知机（multilayer perceptron，MLP）为例，介绍多层神经网络的概念。

## 隐藏层

多层感知机在单层神经网络的基础上引入了一到多个隐藏层（hidden layer）。隐藏层位于输入层和输出层之间。图3.3展示了一个多层感知机的神经网络图。

![带有隐藏层的多层感知机。它含有一个隐藏层，该层中有5个隐藏单元](../img/mlp.svg)

在图3.3所示的多层感知机中，输入和输出个数分别为4和3，中间的隐藏层中包含了5个隐藏单元（hidden unit）。由于输入层不涉及计算，图3.3中的多层感知机的层数为2。由图3.3可见，隐藏层中的神经元和输入层中各个输入完全连接，输出层中的神经元和隐藏层中的各个神经元也完全连接。因此，多层感知机中的隐藏层和输出层都是全连接层。


具体来说，给定一个小批量样本$\boldsymbol{X} \in \mathbb{R}^{n \times d}$，其批量大小为$n$，输入个数为$d$。假设多层感知机只有一个隐藏层，其中隐藏单元个数为$h$。记隐藏层的输出（也称为隐藏层变量或隐藏变量）为$\boldsymbol{H}$，有$\boldsymbol{H} \in \mathbb{R}^{n \times h}$。因为隐藏层和输出层均是全连接层，可以设隐藏层的权重参数和偏差参数分别为$\boldsymbol{W}_h \in \mathbb{R}^{d \times h}$和 $\boldsymbol{b}_h \in \mathbb{R}^{1 \times h}$，输出层的权重和偏差参数分别为$\boldsymbol{W}_o \in \mathbb{R}^{h \times q}$和$\boldsymbol{b}_o \in \mathbb{R}^{1 \times q}$。

我们先来看一种含单隐藏层的多层感知机的设计。其输出$\boldsymbol{O} \in \mathbb{R}^{n \times q}$的计算为

$$
\begin{aligned}
\boldsymbol{H} &= \boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h,\\
\boldsymbol{O} &= \boldsymbol{H} \boldsymbol{W}_o + \boldsymbol{b}_o,
\end{aligned}      
$$

也就是将隐藏层的输出直接作为输出层的输入。如果将以上两个式子联立起来，可以得到

$$
\boldsymbol{O} = (\boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h)\boldsymbol{W}_o + \boldsymbol{b}_o = \boldsymbol{X} \boldsymbol{W}_h\boldsymbol{W}_o + \boldsymbol{b}_h \boldsymbol{W}_o + \boldsymbol{b}_o.
$$

从联立后的式子可以看出，虽然神经网络引入了隐藏层，却依然等价于一个单层神经网络：其中输出层权重参数为$\boldsymbol{W}_h\boldsymbol{W}_o$，偏差参数为$\boldsymbol{b}_h \boldsymbol{W}_o + \boldsymbol{b}_o$。不难发现，即便再添加更多的隐藏层，以上设计依然只能与仅含输出层的单层神经网络等价。


## 激活函数

上述问题的根源在于全连接层只是对数据做仿射变换（affine transformation），而多个仿射变换的叠加仍然是一个仿射变换。解决问题的一个方法是引入非线性变换，例如对隐藏变量使用按元素运算的非线性函数进行变换，然后再作为下一个全连接层的输入。这个非线性函数被称为激活函数（activation function）。下面我们介绍几个常用的激活函数。

### ReLU函数

ReLU（rectified linear unit）函数提供了一个很简单的非线性变换。给定元素$x$，该函数定义为

$$\text{ReLU}(x) = \max(x, 0).$$

可以看出，ReLU函数只保留正数元素，并将负数元素清零。为了直观地观察这一非线性变换，我们先定义一个绘图函数`xyplot`。

```{.python .input  n=4}
%matplotlib inline
import d2lzh as d2l
from mxnet import autograd, nd

def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
```

我们接下来通过`NDArray`提供的`relu`函数来绘制ReLU函数。可以看到，该激活函数是一个两段线性函数。

```{.python .input  n=5}
x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = x.relu()
xyplot(x, y, 'relu')
```

```{.json .output n=5}
[
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (http://matplotlib.org/) -->\n<svg height=\"184.15625pt\" version=\"1.1\" viewBox=\"0 0 323.940625 184.15625\" width=\"323.940625pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <defs>\n  <style type=\"text/css\">\n*{stroke-linecap:butt;stroke-linejoin:round;}\n  </style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 184.15625 \nL 323.940625 184.15625 \nL 323.940625 -0 \nL 0 -0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 34.240625 146.6 \nL 313.240625 146.6 \nL 313.240625 10.7 \nL 34.240625 10.7 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"ma74d9f65b6\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"46.922443\" xlink:href=\"#ma74d9f65b6\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- \u22128 -->\n      <defs>\n       <path d=\"M 10.59375 35.5 \nL 73.1875 35.5 \nL 73.1875 27.203125 \nL 10.59375 27.203125 \nz\n\" id=\"DejaVuSans-2212\"/>\n       <path d=\"M 31.78125 34.625 \nQ 24.75 34.625 20.71875 30.859375 \nQ 16.703125 27.09375 16.703125 20.515625 \nQ 16.703125 13.921875 20.71875 10.15625 \nQ 24.75 6.390625 31.78125 6.390625 \nQ 38.8125 6.390625 42.859375 10.171875 \nQ 46.921875 13.96875 46.921875 20.515625 \nQ 46.921875 27.09375 42.890625 30.859375 \nQ 38.875 34.625 31.78125 34.625 \nz\nM 21.921875 38.8125 \nQ 15.578125 40.375 12.03125 44.71875 \nQ 8.5 49.078125 8.5 55.328125 \nQ 8.5 64.0625 14.71875 69.140625 \nQ 20.953125 74.21875 31.78125 74.21875 \nQ 42.671875 74.21875 48.875 69.140625 \nQ 55.078125 64.0625 55.078125 55.328125 \nQ 55.078125 49.078125 51.53125 44.71875 \nQ 48 40.375 41.703125 38.8125 \nQ 48.828125 37.15625 52.796875 32.3125 \nQ 56.78125 27.484375 56.78125 20.515625 \nQ 56.78125 9.90625 50.3125 4.234375 \nQ 43.84375 -1.421875 31.78125 -1.421875 \nQ 19.734375 -1.421875 13.25 4.234375 \nQ 6.78125 9.90625 6.78125 20.515625 \nQ 6.78125 27.484375 10.78125 32.3125 \nQ 14.796875 37.15625 21.921875 38.8125 \nz\nM 18.3125 54.390625 \nQ 18.3125 48.734375 21.84375 45.5625 \nQ 25.390625 42.390625 31.78125 42.390625 \nQ 38.140625 42.390625 41.71875 45.5625 \nQ 45.3125 48.734375 45.3125 54.390625 \nQ 45.3125 60.0625 41.71875 63.234375 \nQ 38.140625 66.40625 31.78125 66.40625 \nQ 25.390625 66.40625 21.84375 63.234375 \nQ 18.3125 60.0625 18.3125 54.390625 \nz\n\" id=\"DejaVuSans-38\"/>\n      </defs>\n      <g transform=\"translate(39.551349 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-2212\"/>\n       <use x=\"83.789062\" xlink:href=\"#DejaVuSans-38\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_2\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"78.826387\" xlink:href=\"#ma74d9f65b6\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- \u22126 -->\n      <defs>\n       <path d=\"M 33.015625 40.375 \nQ 26.375 40.375 22.484375 35.828125 \nQ 18.609375 31.296875 18.609375 23.390625 \nQ 18.609375 15.53125 22.484375 10.953125 \nQ 26.375 6.390625 33.015625 6.390625 \nQ 39.65625 6.390625 43.53125 10.953125 \nQ 47.40625 15.53125 47.40625 23.390625 \nQ 47.40625 31.296875 43.53125 35.828125 \nQ 39.65625 40.375 33.015625 40.375 \nz\nM 52.59375 71.296875 \nL 52.59375 62.3125 \nQ 48.875 64.0625 45.09375 64.984375 \nQ 41.3125 65.921875 37.59375 65.921875 \nQ 27.828125 65.921875 22.671875 59.328125 \nQ 17.53125 52.734375 16.796875 39.40625 \nQ 19.671875 43.65625 24.015625 45.921875 \nQ 28.375 48.1875 33.59375 48.1875 \nQ 44.578125 48.1875 50.953125 41.515625 \nQ 57.328125 34.859375 57.328125 23.390625 \nQ 57.328125 12.15625 50.6875 5.359375 \nQ 44.046875 -1.421875 33.015625 -1.421875 \nQ 20.359375 -1.421875 13.671875 8.265625 \nQ 6.984375 17.96875 6.984375 36.375 \nQ 6.984375 53.65625 15.1875 63.9375 \nQ 23.390625 74.21875 37.203125 74.21875 \nQ 40.921875 74.21875 44.703125 73.484375 \nQ 48.484375 72.75 52.59375 71.296875 \nz\n\" id=\"DejaVuSans-36\"/>\n      </defs>\n      <g transform=\"translate(71.455293 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-2212\"/>\n       <use x=\"83.789062\" xlink:href=\"#DejaVuSans-36\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_3\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"110.730331\" xlink:href=\"#ma74d9f65b6\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- \u22124 -->\n      <defs>\n       <path d=\"M 37.796875 64.3125 \nL 12.890625 25.390625 \nL 37.796875 25.390625 \nz\nM 35.203125 72.90625 \nL 47.609375 72.90625 \nL 47.609375 25.390625 \nL 58.015625 25.390625 \nL 58.015625 17.1875 \nL 47.609375 17.1875 \nL 47.609375 0 \nL 37.796875 0 \nL 37.796875 17.1875 \nL 4.890625 17.1875 \nL 4.890625 26.703125 \nz\n\" id=\"DejaVuSans-34\"/>\n      </defs>\n      <g transform=\"translate(103.359237 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-2212\"/>\n       <use x=\"83.789062\" xlink:href=\"#DejaVuSans-34\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_4\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"142.634275\" xlink:href=\"#ma74d9f65b6\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- \u22122 -->\n      <defs>\n       <path d=\"M 19.1875 8.296875 \nL 53.609375 8.296875 \nL 53.609375 0 \nL 7.328125 0 \nL 7.328125 8.296875 \nQ 12.9375 14.109375 22.625 23.890625 \nQ 32.328125 33.6875 34.8125 36.53125 \nQ 39.546875 41.84375 41.421875 45.53125 \nQ 43.3125 49.21875 43.3125 52.78125 \nQ 43.3125 58.59375 39.234375 62.25 \nQ 35.15625 65.921875 28.609375 65.921875 \nQ 23.96875 65.921875 18.8125 64.3125 \nQ 13.671875 62.703125 7.8125 59.421875 \nL 7.8125 69.390625 \nQ 13.765625 71.78125 18.9375 73 \nQ 24.125 74.21875 28.421875 74.21875 \nQ 39.75 74.21875 46.484375 68.546875 \nQ 53.21875 62.890625 53.21875 53.421875 \nQ 53.21875 48.921875 51.53125 44.890625 \nQ 49.859375 40.875 45.40625 35.40625 \nQ 44.1875 33.984375 37.640625 27.21875 \nQ 31.109375 20.453125 19.1875 8.296875 \nz\n\" id=\"DejaVuSans-32\"/>\n      </defs>\n      <g transform=\"translate(135.263181 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-2212\"/>\n       <use x=\"83.789062\" xlink:href=\"#DejaVuSans-32\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_5\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"174.538219\" xlink:href=\"#ma74d9f65b6\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 0 -->\n      <defs>\n       <path d=\"M 31.78125 66.40625 \nQ 24.171875 66.40625 20.328125 58.90625 \nQ 16.5 51.421875 16.5 36.375 \nQ 16.5 21.390625 20.328125 13.890625 \nQ 24.171875 6.390625 31.78125 6.390625 \nQ 39.453125 6.390625 43.28125 13.890625 \nQ 47.125 21.390625 47.125 36.375 \nQ 47.125 51.421875 43.28125 58.90625 \nQ 39.453125 66.40625 31.78125 66.40625 \nz\nM 31.78125 74.21875 \nQ 44.046875 74.21875 50.515625 64.515625 \nQ 56.984375 54.828125 56.984375 36.375 \nQ 56.984375 17.96875 50.515625 8.265625 \nQ 44.046875 -1.421875 31.78125 -1.421875 \nQ 19.53125 -1.421875 13.0625 8.265625 \nQ 6.59375 17.96875 6.59375 36.375 \nQ 6.59375 54.828125 13.0625 64.515625 \nQ 19.53125 74.21875 31.78125 74.21875 \nz\n\" id=\"DejaVuSans-30\"/>\n      </defs>\n      <g transform=\"translate(171.356969 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_6\">\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"206.442163\" xlink:href=\"#ma74d9f65b6\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 2 -->\n      <g transform=\"translate(203.260913 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-32\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_7\">\n     <g id=\"line2d_7\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"238.346107\" xlink:href=\"#ma74d9f65b6\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 4 -->\n      <g transform=\"translate(235.164857 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-34\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_8\">\n     <g id=\"line2d_8\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"270.250051\" xlink:href=\"#ma74d9f65b6\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 6 -->\n      <g transform=\"translate(267.068801 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-36\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_9\">\n     <g id=\"line2d_9\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"302.153995\" xlink:href=\"#ma74d9f65b6\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 8 -->\n      <g transform=\"translate(298.972745 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-38\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_10\">\n     <!-- x -->\n     <defs>\n      <path d=\"M 54.890625 54.6875 \nL 35.109375 28.078125 \nL 55.90625 0 \nL 45.3125 0 \nL 29.390625 21.484375 \nL 13.484375 0 \nL 2.875 0 \nL 24.125 28.609375 \nL 4.6875 54.6875 \nL 15.28125 54.6875 \nL 29.78125 35.203125 \nL 44.28125 54.6875 \nz\n\" id=\"DejaVuSans-78\"/>\n     </defs>\n     <g transform=\"translate(170.78125 174.876562)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-78\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_10\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"m6ec7694dbe\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"34.240625\" xlink:href=\"#m6ec7694dbe\" y=\"140.422727\"/>\n      </g>\n     </g>\n     <g id=\"text_11\">\n      <!-- 0 -->\n      <g transform=\"translate(20.878125 144.221946)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_11\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"34.240625\" xlink:href=\"#m6ec7694dbe\" y=\"109.145399\"/>\n      </g>\n     </g>\n     <g id=\"text_12\">\n      <!-- 2 -->\n      <g transform=\"translate(20.878125 112.944618)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-32\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_12\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"34.240625\" xlink:href=\"#m6ec7694dbe\" y=\"77.868071\"/>\n      </g>\n     </g>\n     <g id=\"text_13\">\n      <!-- 4 -->\n      <g transform=\"translate(20.878125 81.66729)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-34\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_13\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"34.240625\" xlink:href=\"#m6ec7694dbe\" y=\"46.590743\"/>\n      </g>\n     </g>\n     <g id=\"text_14\">\n      <!-- 6 -->\n      <g transform=\"translate(20.878125 50.389962)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-36\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_14\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"34.240625\" xlink:href=\"#m6ec7694dbe\" y=\"15.313415\"/>\n      </g>\n     </g>\n     <g id=\"text_15\">\n      <!-- 8 -->\n      <g transform=\"translate(20.878125 19.112634)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-38\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_16\">\n     <!-- relu(x) -->\n     <defs>\n      <path d=\"M 41.109375 46.296875 \nQ 39.59375 47.171875 37.8125 47.578125 \nQ 36.03125 48 33.890625 48 \nQ 26.265625 48 22.1875 43.046875 \nQ 18.109375 38.09375 18.109375 28.8125 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 20.953125 51.171875 25.484375 53.578125 \nQ 30.03125 56 36.53125 56 \nQ 37.453125 56 38.578125 55.875 \nQ 39.703125 55.765625 41.0625 55.515625 \nz\n\" id=\"DejaVuSans-72\"/>\n      <path d=\"M 56.203125 29.59375 \nL 56.203125 25.203125 \nL 14.890625 25.203125 \nQ 15.484375 15.921875 20.484375 11.0625 \nQ 25.484375 6.203125 34.421875 6.203125 \nQ 39.59375 6.203125 44.453125 7.46875 \nQ 49.3125 8.734375 54.109375 11.28125 \nL 54.109375 2.78125 \nQ 49.265625 0.734375 44.1875 -0.34375 \nQ 39.109375 -1.421875 33.890625 -1.421875 \nQ 20.796875 -1.421875 13.15625 6.1875 \nQ 5.515625 13.8125 5.515625 26.8125 \nQ 5.515625 40.234375 12.765625 48.109375 \nQ 20.015625 56 32.328125 56 \nQ 43.359375 56 49.78125 48.890625 \nQ 56.203125 41.796875 56.203125 29.59375 \nz\nM 47.21875 32.234375 \nQ 47.125 39.59375 43.09375 43.984375 \nQ 39.0625 48.390625 32.421875 48.390625 \nQ 24.90625 48.390625 20.390625 44.140625 \nQ 15.875 39.890625 15.1875 32.171875 \nz\n\" id=\"DejaVuSans-65\"/>\n      <path d=\"M 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 0 \nL 9.421875 0 \nz\n\" id=\"DejaVuSans-6c\"/>\n      <path d=\"M 8.5 21.578125 \nL 8.5 54.6875 \nL 17.484375 54.6875 \nL 17.484375 21.921875 \nQ 17.484375 14.15625 20.5 10.265625 \nQ 23.53125 6.390625 29.59375 6.390625 \nQ 36.859375 6.390625 41.078125 11.03125 \nQ 45.3125 15.671875 45.3125 23.6875 \nL 45.3125 54.6875 \nL 54.296875 54.6875 \nL 54.296875 0 \nL 45.3125 0 \nL 45.3125 8.40625 \nQ 42.046875 3.421875 37.71875 1 \nQ 33.40625 -1.421875 27.6875 -1.421875 \nQ 18.265625 -1.421875 13.375 4.4375 \nQ 8.5 10.296875 8.5 21.578125 \nz\nM 31.109375 56 \nz\n\" id=\"DejaVuSans-75\"/>\n      <path d=\"M 31 75.875 \nQ 24.46875 64.65625 21.28125 53.65625 \nQ 18.109375 42.671875 18.109375 31.390625 \nQ 18.109375 20.125 21.3125 9.0625 \nQ 24.515625 -2 31 -13.1875 \nL 23.1875 -13.1875 \nQ 15.875 -1.703125 12.234375 9.375 \nQ 8.59375 20.453125 8.59375 31.390625 \nQ 8.59375 42.28125 12.203125 53.3125 \nQ 15.828125 64.359375 23.1875 75.875 \nz\n\" id=\"DejaVuSans-28\"/>\n      <path d=\"M 8.015625 75.875 \nL 15.828125 75.875 \nQ 23.140625 64.359375 26.78125 53.3125 \nQ 30.421875 42.28125 30.421875 31.390625 \nQ 30.421875 20.453125 26.78125 9.375 \nQ 23.140625 -1.703125 15.828125 -13.1875 \nL 8.015625 -13.1875 \nQ 14.5 -2 17.703125 9.0625 \nQ 20.90625 20.125 20.90625 31.390625 \nQ 20.90625 42.671875 17.703125 53.65625 \nQ 14.5 64.65625 8.015625 75.875 \nz\n\" id=\"DejaVuSans-29\"/>\n     </defs>\n     <g transform=\"translate(14.798438 95.199219)rotate(-90)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-72\"/>\n      <use x=\"41.082031\" xlink:href=\"#DejaVuSans-65\"/>\n      <use x=\"102.605469\" xlink:href=\"#DejaVuSans-6c\"/>\n      <use x=\"130.388672\" xlink:href=\"#DejaVuSans-75\"/>\n      <use x=\"193.767578\" xlink:href=\"#DejaVuSans-28\"/>\n      <use x=\"232.78125\" xlink:href=\"#DejaVuSans-78\"/>\n      <use x=\"291.960938\" xlink:href=\"#DejaVuSans-29\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_15\">\n    <path clip-path=\"url(#p2e8b5e69fa)\" d=\"M 46.922443 140.422727 \nL 174.538219 140.422727 \nL 300.558807 16.877273 \nL 300.558807 16.877273 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 34.240625 146.6 \nL 34.240625 10.7 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 313.240625 146.6 \nL 313.240625 10.7 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 34.240625 146.6 \nL 313.240625 146.6 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 34.240625 10.7 \nL 313.240625 10.7 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p2e8b5e69fa\">\n   <rect height=\"135.9\" width=\"279\" x=\"34.240625\" y=\"10.7\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 360x180 with 1 Axes>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

显然，当输入为负数时，ReLU函数的导数为0；当输入为正数时，ReLU函数的导数为1。尽管输入为0时ReLU函数不可导，但是我们可以取此处的导数为0。下面绘制ReLU函数的导数。

```{.python .input}
y.backward()
xyplot(x, x.grad, 'grad of relu')
```

### sigmoid函数

sigmoid函数可以将元素的值变换到0和1之间：

$$\text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

sigmoid函数在早期的神经网络中较为普遍，但它目前逐渐被更简单的ReLU函数取代。在后面“循环神经网络”一章中我们会介绍如何利用它值域在0到1之间这一特性来控制信息在神经网络中的流动。下面绘制了sigmoid函数。当输入接近0时，sigmoid函数接近线性变换。

```{.python .input  n=8}
with autograd.record():
    y = x.sigmoid()
xyplot(x, y, 'sigmoid')
```

依据链式法则，sigmoid函数的导数

$$\text{sigmoid}'(x) = \text{sigmoid}(x)\left(1-\text{sigmoid}(x)\right).$$


下面绘制了sigmoid函数的导数。当输入为0时，sigmoid函数的导数达到最大值0.25；当输入越偏离0时，sigmoid函数的导数越接近0。

```{.python .input}
y.backward()
xyplot(x, x.grad, 'grad of sigmoid')
```

### tanh函数

tanh（双曲正切）函数可以将元素的值变换到-1和1之间：

$$\text{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

我们接着绘制tanh函数。当输入接近0时，tanh函数接近线性变换。虽然该函数的形状和sigmoid函数的形状很像，但tanh函数在坐标系的原点上对称。

```{.python .input  n=9}
with autograd.record():
    y = x.tanh()
xyplot(x, y, 'tanh')
```

依据链式法则，tanh函数的导数

$$\text{tanh}'(x) = 1 - \text{tanh}^2(x).$$

下面绘制了tanh函数的导数。当输入为0时，tanh函数的导数达到最大值1；当输入越偏离0时，tanh函数的导数越接近0。

```{.python .input}
y.backward()
xyplot(x, x.grad, 'grad of tanh')
```

## 多层感知机

多层感知机就是含有至少一个隐藏层的由全连接层组成的神经网络，且每个隐藏层的输出通过激活函数进行变换。多层感知机的层数和各隐藏层中隐藏单元个数都是超参数。以单隐藏层为例并沿用本节之前定义的符号，多层感知机按以下方式计算输出：

$$
\begin{aligned}
\boldsymbol{H} &= \phi(\boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h),\\
\boldsymbol{O} &= \boldsymbol{H} \boldsymbol{W}_o + \boldsymbol{b}_o,
\end{aligned}
$$
 
其中$\phi$表示激活函数。在分类问题中，我们可以对输出$\boldsymbol{O}$做softmax运算，并使用softmax回归中的交叉熵损失函数。
在回归问题中，我们将输出层的输出个数设为1，并将输出$\boldsymbol{O}$直接提供给线性回归中使用的平方损失函数。



## 小结

* 多层感知机在输出层与输入层之间加入了一个或多个全连接隐藏层，并通过激活函数对隐藏层输出进行变换。
* 常用的激活函数包括ReLU函数、sigmoid函数和tanh函数。


## 练习

* 应用链式法则，推导出sigmoid函数和tanh函数的导数的数学表达式。
* 查阅资料，了解其他的激活函数。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6447)

![](../img/qr_mlp.svg)
