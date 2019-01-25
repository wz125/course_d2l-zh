# softmax回归的简洁实现

我们在[“线性回归的简洁实现”](linear-regression-gluon.md)一节中已经了解了使用Gluon实现模型的便利。下面，让我们再次使用Gluon来实现一个softmax回归模型。首先导入所需的包或模块。

```{.python .input  n=20}
%matplotlib inline
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init,nd
from mxnet.gluon import loss as gloss, nn, utils as gutils
```

## 获取和读取数据

我们仍然使用Fashion-MNIST数据集和上一节中设置的批量大小。

```{.python .input  n=26}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
for X,y in train_iter:
    print(X.shape,y.shape,y.size)
    break
```

```{.json .output n=26}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(256, 1, 28, 28) (256,) 256\n"
 }
]
```

## 定义和初始化模型

在[“softmax回归”](softmax-regression.md)一节中提到，softmax回归的输出层是一个全连接层。因此，我们添加一个输出个数为10的全连接层。我们使用均值为0、标准差为0.01的正态分布随机初始化模型的权重参数。 
`w.输出一个值y，相当于与10个输入层全连接得出`

```{.python .input  n=7}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

## softmax和交叉熵损失函数

如果做了上一节的练习，那么你可能意识到了分开定义softmax运算和交叉熵损失函数可能会造成数值不稳定。因此，Gluon提供了一个包括softmax运算和交叉熵损失计算的函数。它的数值稳定性更好。

```{.python .input  n=8}
loss = gloss.SoftmaxCrossEntropyLoss()
#todo learn
```

## 定义优化算法

我们使用学习率为0.1的小批量随机梯度下降作为优化算法。

```{.python .input  n=23}
#gluon.Trainer(params ,optimizer ,optimizer_params ,...)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
print(net.collect_params())

#example class mxnet.optimizer.SGD
sgd = mx.optimizer.Optimizer.create_optimizer('sgd')
#mxnet.ndarray.sgd_update: weight = weight - learning_rate * (gradient + wd * weight)
```

```{.json .output n=23}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "sequential0_ (\n  Parameter dense0_weight (shape=(10, 784), dtype=float32)\n  Parameter dense0_bias (shape=(10,), dtype=float32)\n)\n"
 },
 {
  "data": {
   "text/plain": "mxnet.optimizer.optimizer.SGD"
  },
  "execution_count": 23,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 训练模型

接下来，我们使用上一节中定义的训练函数来训练模型。 `方便演示 把utils函数直接拷贝过来参考`

```{.python .input  n=27}
#from d2l/utils.py
def d2l_get_batch(batch, ctx):
    """Return features and labels on ctx."""
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])
#from d2l/utils.py
def d2l_evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = d2l_get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n
#from d2l/utils.py
def d2l_train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    """Train and evaluate a model with CPU."""
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m = 0.0, 0.0, 0,0
        for X, y in train_iter: #train_iter 的数量是 dataset总数据/batch_size=600000/256=235
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                sgd(params, lr, batch_size)
            else:
                #Makes one step of parameter update. 
                #Should be called after autograd.backward() and outside of record() scope.
                trainer.step(batch_size) 
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size #y.size=256 数值
            m +=1
        test_acc = d2l_evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, m %d'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,m))

num_epochs = 5
d2l_train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
```

```{.json .output n=27}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 1, loss 0.4241, train acc 0.855, test acc 0.855, n 235\nepoch 2, loss 0.4225, train acc 0.855, test acc 0.848, n 235\nepoch 3, loss 0.4205, train acc 0.855, test acc 0.853, n 235\nepoch 4, loss 0.4197, train acc 0.856, test acc 0.852, n 235\nepoch 5, loss 0.4183, train acc 0.856, test acc 0.855, n 235\n"
 }
]
```

## Q&A
+ dataset  "train_iter" 的数量从什么地方可得到 ？

## 小结

* Gluon提供的函数往往具有更好的数值稳定性。
* 可以使用Gluon更简洁地实现softmax回归。

## 练习

* 尝试调一调超参数，如批量大小、迭代周期和学习率，看看结果会怎样。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/740)

![](../img/qr_softmax-regression-gluon.svg)
