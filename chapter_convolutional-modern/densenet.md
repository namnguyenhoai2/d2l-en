```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Mạng kết nối dày đặc (DenseNet)
:label:`sec_densenet`

ResNet đã thay đổi đáng kể cách nhìn về việc tham số hóa các hàm trong mạng sâu. *DenseNet* (mạng tích chập dày đặc) ở một mức độ nào đó là phần mở rộng hợp lý của điều này :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.
DenseNet được đặc trưng bởi cả mẫu kết nối, trong đó
mỗi lớp kết nối với tất cả các lớp trước đó,
và phép toán nối (thay vì toán tử cộng trong ResNet) để bảo toàn và tái sử dụng các đặc trưng
từ các lớp trước.
Để hiểu cách suy ra nó, hãy đi qua một chút toán học.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
```

## Từ ResNet đến DenseNet

Hãy nhắc lại khai triển Taylor cho các hàm. Tại điểm $x = 0$, nó có thể được viết là

$$f(x) = f(0) + x \cdot \left[f'(0) + x \cdot \left[\frac{f''(0)}{2!}  + x \cdot \left[\frac{f'''(0)}{3!}  + \cdots \right]\right]\right].$$


Điểm then chốt là nó phân rã một hàm thành các hạng có bậc ngày càng cao. Tương tự, ResNet phân rã các hàm thành

$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}).$$

Nói cách khác, ResNet phân rã $f$ thành một hạng tuyến tính đơn giản và một hạng
phi tuyến phức tạp hơn.
Nếu chúng ta muốn nắm bắt thông tin vượt ngoài hai hạng thì sao (không nhất thiết phải cộng chúng)?
Một lời giải như vậy là DenseNet :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.

![Khác biệt chính giữa ResNet (trái) và DenseNet (phải) trong các kết nối xuyên lớp: sử dụng phép cộng và sử dụng phép nối. ](../img/densenet-block.svg)
:label:`fig_densenet_block`

Như minh họa trong :numref:`fig_densenet_block`, khác biệt then chốt giữa ResNet và DenseNet là ở trường hợp sau, các đầu ra được *nối* (ký hiệu là $[,]$) thay vì được cộng.
Do đó, chúng ta thực hiện ánh xạ từ $\mathbf{x}$ sang các giá trị của nó sau khi áp dụng một chuỗi hàm ngày càng phức tạp:

$$\mathbf{x} \to \left[
\mathbf{x},
f_1(\mathbf{x}),
f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right), f_3\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right), f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right)\right]\right), \ldots\right].$$

Cuối cùng, tất cả các hàm này được kết hợp trong MLP để giảm lại số lượng đặc trưng. Về mặt triển khai, điều này khá đơn giản:
thay vì cộng các hạng, chúng ta nối chúng lại. Tên DenseNet xuất phát từ việc đồ thị phụ thuộc giữa các biến trở nên khá dày đặc. Lớp cuối cùng của một chuỗi như vậy được kết nối dày đặc với tất cả các lớp trước đó. Các kết nối dày đặc được thể hiện trong :numref:`fig_densenet`.

![Các kết nối dày đặc trong DenseNet. Lưu ý rằng số chiều tăng theo độ sâu.](../img/densenet.svg)
:label:`fig_densenet`

Các thành phần chính cấu thành DenseNet là *khối dày đặc* và *lớp chuyển tiếp*. Thành phần thứ nhất xác định cách đầu vào và đầu ra được nối, trong khi thành phần thứ hai kiểm soát số lượng kênh để không trở nên quá lớn, 
vì phép mở rộng $\mathbf{x} \to \left[\mathbf{x}, f_1(\mathbf{x}),
f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right), \ldots \right]$ có thể có số chiều rất cao.


## [**Khối dày đặc**]

DenseNet sử dụng cấu trúc "batch normalization, activation, và convolution" đã được sửa đổi của ResNet (xem bài tập trong :numref:`sec_resnet`).
Trước tiên, chúng ta triển khai cấu trúc khối convolution này.

```{.python .input}
%%tab mxnet
def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

```{.python .input}
%%tab pytorch
def conv_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1))
```

```{.python .input}
%%tab tensorflow
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3), padding='same')

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x,y], axis=-1)
        return y
```

```{.python .input}
%%tab jax
class ConvBlock(nn.Module):
    num_channels: int
    training: bool = True

    @nn.compact
    def __call__(self, X):
        Y = nn.relu(nn.BatchNorm(not self.training)(X))
        Y = nn.Conv(self.num_channels, kernel_size=(3, 3), padding=(1, 1))(Y)
        Y = jnp.concatenate((X, Y), axis=-1)
        return Y
```

Một *khối dày đặc* bao gồm nhiều khối convolution, mỗi khối sử dụng cùng số lượng kênh đầu ra. Tuy nhiên, trong lan truyền thuận, chúng ta nối đầu vào và đầu ra của mỗi khối convolution theo chiều kênh. Lazy evaluation cho phép chúng ta tự động điều chỉnh số chiều.

```{.python .input}
%%tab mxnet
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels):
        super().__init__()
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate input and output of each block along the channels
            X = np.concatenate((X, Y), axis=1)
        return X
```

```{.python .input}
%%tab pytorch
class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate input and output of each block along the channels
            X = torch.cat((X, Y), dim=1)
        return X
```

```{.python .input}
%%tab tensorflow
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x
```

```{.python .input}
%%tab jax
class DenseBlock(nn.Module):
    num_convs: int
    num_channels: int
    training: bool = True

    def setup(self):
        layer = []
        for i in range(self.num_convs):
            layer.append(ConvBlock(self.num_channels, self.training))
        self.net = nn.Sequential(layer)

    def __call__(self, X):
        return self.net(X)
```

Trong ví dụ sau,
chúng ta [**định nghĩa một instance `DenseBlock`**] với hai khối convolution có 10 kênh đầu ra.
Khi sử dụng đầu vào có ba kênh, chúng ta sẽ nhận được đầu ra có $3 + 10 + 10=23$ kênh. Số kênh của khối convolution kiểm soát mức tăng số lượng kênh đầu ra so với số lượng kênh đầu vào. Điều này cũng được gọi là *tốc độ tăng trưởng*.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
blk = DenseBlock(2, 10)
if tab.selected('mxnet'):
    X = np.random.uniform(size=(4, 3, 8, 8))
    blk.initialize()
if tab.selected('pytorch'):
    X = torch.randn(4, 3, 8, 8)
if tab.selected('tensorflow'):
    X = tf.random.uniform((4, 8, 8, 3))
Y = blk(X)
Y.shape
```

```{.python .input}
%%tab jax
blk = DenseBlock(2, 10)
X = jnp.zeros((4, 8, 8, 3))
Y = blk.init_with_output(d2l.get_key(), X)[0]
Y.shape
```

## [**Lớp chuyển tiếp**]

Vì mỗi khối dày đặc sẽ làm tăng số lượng kênh, việc thêm quá nhiều khối sẽ dẫn đến một mô hình phức tạp quá mức. Một *lớp chuyển tiếp* được dùng để kiểm soát độ phức tạp của mô hình. Nó giảm số lượng kênh bằng cách sử dụng convolution $1\times 1$. Hơn nữa, nó giảm một nửa chiều cao và chiều rộng thông qua average pooling với stride bằng 2.

```{.python .input}
%%tab mxnet
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
%%tab pytorch
def transition_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

```{.python .input}
%%tab tensorflow
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)
```

```{.python .input}
%%tab jax
class TransitionBlock(nn.Module):
    num_channels: int
    training: bool = True

    @nn.compact
    def __call__(self, X):
        X = nn.BatchNorm(not self.training)(X)
        X = nn.relu(X)
        X = nn.Conv(self.num_channels, kernel_size=(1, 1))(X)
        X = nn.avg_pool(X, window_shape=(2, 2), strides=(2, 2))
        return X
```

[**Áp dụng một lớp chuyển tiếp**] với 10 kênh cho đầu ra của khối dày đặc trong ví dụ trước. Điều này giảm số lượng kênh đầu ra xuống 10, đồng thời giảm một nửa chiều cao và chiều rộng.

```{.python .input}
%%tab mxnet
blk = transition_block(10)
blk.initialize()
blk(Y).shape
```

```{.python .input}
%%tab pytorch
blk = transition_block(10)
blk(Y).shape
```

```{.python .input}
%%tab tensorflow
blk = TransitionBlock(10)
blk(Y).shape
```

```{.python .input}
%%tab jax
blk = TransitionBlock(10)
blk.init_with_output(d2l.get_key(), Y)[0].shape
```

## [**Mô hình DenseNet**]

Tiếp theo, chúng ta sẽ xây dựng một mô hình DenseNet. DenseNet trước tiên sử dụng cùng lớp convolution đơn và lớp max-pooling như trong ResNet.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class DenseNet(d2l.Classifier):
    def b1(self):
        if tab.selected('mxnet'):
            net = nn.Sequential()
            net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
                nn.BatchNorm(), nn.Activation('relu'),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            return net
        if tab.selected('pytorch'):
            return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                nn.LazyBatchNorm2d(), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if tab.selected('tensorflow'):
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(
                    64, kernel_size=7, strides=2, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(
                    pool_size=3, strides=2, padding='same')])
```

```{.python .input}
%%tab jax
class DenseNet(d2l.Classifier):
    num_channels: int = 64
    growth_rate: int = 32
    arch: tuple = (4, 4, 4, 4)
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def b1(self):
        return nn.Sequential([
            nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
            nn.BatchNorm(not self.training),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3),
                                  strides=(2, 2), padding='same')
        ])
```

Sau đó, tương tự bốn mô-đun được tạo thành từ các residual block mà ResNet sử dụng,
DenseNet sử dụng bốn khối dày đặc.
Cũng như với ResNet, chúng ta có thể đặt số lớp convolution được dùng trong mỗi khối dày đặc. Ở đây, chúng ta đặt là 4, nhất quán với mô hình ResNet-18 trong :numref:`sec_resnet`. Hơn nữa, chúng ta đặt số kênh (tức là tốc độ tăng trưởng) cho các lớp convolution trong khối dày đặc là 32, do đó mỗi khối dày đặc sẽ thêm 128 kênh.

Trong ResNet, chiều cao và chiều rộng được giảm giữa mỗi mô-đun bằng một residual block có stride bằng 2. Ở đây, chúng ta dùng lớp chuyển tiếp để giảm một nửa chiều cao, chiều rộng và số lượng kênh. Tương tự ResNet, một lớp global pooling và một lớp fully connected được kết nối ở cuối để tạo ra đầu ra.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(DenseNet)
def __init__(self, num_channels=64, growth_rate=32, arch=(4, 4, 4, 4),
             lr=0.1, num_classes=10):
    super(DenseNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add(DenseBlock(num_convs, growth_rate))
            # The number of output channels in the previous dense block
            num_channels += num_convs * growth_rate
            # A transition layer that halves the number of channels is added
            # between the dense blocks
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add(transition_block(num_channels))
        self.net.add(nn.BatchNorm(), nn.Activation('relu'),
                     nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add_module(f'dense_blk{i+1}', DenseBlock(num_convs,
                                                              growth_rate))
            # The number of output channels in the previous dense block
            num_channels += num_convs * growth_rate
            # A transition layer that halves the number of channels is added
            # between the dense blocks
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add_module(f'tran_blk{i+1}', transition_block(
                    num_channels))
        self.net.add_module('last', nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add(DenseBlock(num_convs, growth_rate))
            # The number of output channels in the previous dense block
            num_channels += num_convs * growth_rate
            # A transition layer that halves the number of channels is added
            # between the dense blocks
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add(TransitionBlock(num_channels))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes)]))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(DenseNet)
def create_net(self):
    net = self.b1()
    for i, num_convs in enumerate(self.arch):
        net.layers.extend([DenseBlock(num_convs, self.growth_rate,
                                      training=self.training)])
        # The number of output channels in the previous dense block
        num_channels = self.num_channels + (num_convs * self.growth_rate)
        # A transition layer that halves the number of channels is added
        # between the dense blocks
        if i != len(self.arch) - 1:
            num_channels //= 2
            net.layers.extend([TransitionBlock(num_channels,
                                               training=self.training)])
    net.layers.extend([
        nn.BatchNorm(not self.training),
        nn.relu,
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                              strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)
    ])
    return net
```

## [**Huấn luyện**]

Vì ở đây chúng ta sử dụng một mạng sâu hơn, trong phần này chúng ta sẽ giảm chiều cao và chiều rộng đầu vào từ 224 xuống 96 để đơn giản hóa việc tính toán.

```{.python .input}
%%tab mxnet, pytorch, jax
model = DenseNet(lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = DenseNet(lr=0.01)
    trainer.fit(model, data)
```

## Tóm tắt và Thảo luận

Các thành phần chính cấu thành DenseNet là khối dày đặc và lớp chuyển tiếp. Đối với thành phần sau, chúng ta cần kiểm soát số chiều khi cấu thành mạng bằng cách thêm các lớp chuyển tiếp để thu nhỏ lại số lượng kênh.
Về các kết nối xuyên lớp, trái với ResNet, nơi đầu vào và đầu ra được cộng với nhau, DenseNet nối đầu vào và đầu ra theo chiều kênh.
Mặc dù các phép nối này
tái sử dụng đặc trưng để đạt hiệu quả tính toán,
đáng tiếc là chúng dẫn đến mức tiêu thụ bộ nhớ GPU cao.
Do đó,
việc áp dụng DenseNet có thể đòi hỏi các triển khai tiết kiệm bộ nhớ hơn, điều có thể làm tăng thời gian huấn luyện :cite:`pleiss2017memory`.


## Bài tập

1. Tại sao chúng ta dùng average pooling thay vì max-pooling trong lớp chuyển tiếp?
1. Một trong các ưu điểm được đề cập trong bài báo DenseNet là các tham số mô hình của nó ít hơn ResNet. Tại sao lại như vậy?
1. Một vấn đề mà DenseNet bị chỉ trích là mức tiêu thụ bộ nhớ cao.
    1. Điều này có thực sự đúng không? Hãy thử thay đổi hình dạng đầu vào thành $224\times 224$ để so sánh thực nghiệm mức tiêu thụ bộ nhớ GPU thực tế.
    1. Bạn có thể nghĩ ra một phương án khác để giảm mức tiêu thụ bộ nhớ không? Bạn cần thay đổi framework như thế nào?
1. Triển khai các phiên bản DenseNet khác nhau được trình bày trong Bảng 1 của bài báo DenseNet :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.
1. Thiết kế một mô hình dựa trên MLP bằng cách áp dụng ý tưởng DenseNet. Áp dụng nó cho tác vụ dự đoán giá nhà trong :numref:`sec_kaggle_house`.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/87)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/88)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/331)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18008)
:end_tab: