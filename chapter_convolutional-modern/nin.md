```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Mạng trong Mạng (NiN)
:label:`sec_nin`

LeNet, AlexNet và VGG đều có chung một mẫu thiết kế:
trích xuất đặc trưng bằng cách khai thác cấu trúc *không gian*
thông qua một chuỗi các lớp convolution và pooling,
sau đó hậu xử lý các biểu diễn bằng các lớp fully connected.
Các cải tiến của AlexNet và VGG so với LeNet chủ yếu nằm ở
cách những mạng sau này mở rộng chiều rộng và chiều sâu của hai mô-đun này.

Thiết kế này đặt ra hai thách thức lớn.
Thứ nhất, các lớp fully connected ở cuối
kiến trúc tiêu thụ số lượng tham số khổng lồ. Chẳng hạn, ngay cả một
mô hình đơn giản như VGG-11 cũng cần một ma trận khổng lồ, chiếm gần
400MB RAM ở độ chính xác đơn (FP32). Đây là một trở ngại đáng kể đối với việc tính toán, đặc biệt trên
các thiết bị di động và nhúng. Xét cho cùng, ngay cả điện thoại di động cao cấp cũng chỉ có tối đa 8GB RAM. Vào thời điểm VGG được phát minh, con số này ít hơn một bậc độ lớn (iPhone 4S có 512MB). Vì vậy, sẽ rất khó để biện minh cho việc dành phần lớn bộ nhớ cho một bộ phân loại ảnh.

Thứ hai, cũng không thể thêm các lớp fully connected
sớm hơn trong mạng để tăng mức độ phi tuyến: làm vậy sẽ phá hủy
cấu trúc không gian và có thể yêu cầu thậm chí nhiều bộ nhớ hơn.

Các khối *network in network* (*NiN*) :cite:`Lin.Chen.Yan.2013` đưa ra một phương án thay thế,
có thể giải quyết cả hai vấn đề bằng một chiến lược đơn giản.
Chúng được đề xuất dựa trên một nhận định rất đơn giản: (i) sử dụng convolution $1 \times 1$ để bổ sung
các tính phi tuyến cục bộ trên các kích hoạt kênh và (ii) sử dụng global average pooling để tổng hợp
trên mọi vị trí trong lớp biểu diễn cuối cùng. Lưu ý rằng global average pooling sẽ không
hiệu quả nếu không có các tính phi tuyến được bổ sung. Hãy tìm hiểu chi tiết về điều này.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
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
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## (**Khối NiN**)

Nhắc lại :numref:`subsec_1x1`. Trong đó, chúng ta đã nói rằng đầu vào và đầu ra của các lớp convolution
gồm các tensor bốn chiều với các trục
tương ứng với mẫu, kênh, chiều cao và chiều rộng.
Cũng hãy nhớ rằng đầu vào và đầu ra của các lớp fully connected
thường là các tensor hai chiều tương ứng với mẫu và đặc trưng.
Ý tưởng đằng sau NiN là áp dụng một lớp fully connected
tại mỗi vị trí pixel (cho mỗi chiều cao và chiều rộng).
Convolution $1 \times 1$ thu được có thể được xem như
một lớp fully connected hoạt động độc lập tại mỗi vị trí pixel.

:numref:`fig_nin` minh họa các khác biệt chính về cấu trúc
giữa VGG và NiN, cũng như các khối của chúng.
Lưu ý cả sự khác biệt trong các khối NiN (convolution ban đầu được theo sau bởi các convolution $1 \times 1$, trong khi VGG duy trì các convolution $3 \times 3$) lẫn ở phần cuối, nơi chúng ta không còn cần một lớp fully connected khổng lồ.

![So sánh kiến trúc của VGG và NiN, cũng như các khối của chúng.](../img/nin.svg)
:width:`600px`
:label:`fig_nin`

```{.python .input}
%%tab mxnet
def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding,
                      activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk
```

```{.python .input}
%%tab pytorch
def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())
```

```{.python .input}
%%tab tensorflow
def nin_block(out_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(out_channels, kernel_size, strides=strides,
                           padding=padding),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(out_channels, 1),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(out_channels, 1),
    tf.keras.layers.Activation('relu')])
```

```{.python .input}
%%tab jax
def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential([
        nn.Conv(out_channels, kernel_size, strides, padding),
        nn.relu,
        nn.Conv(out_channels, kernel_size=(1, 1)), nn.relu,
        nn.Conv(out_channels, kernel_size=(1, 1)), nn.relu])
```

## [**Mô hình NiN**]

NiN sử dụng các kích thước convolution ban đầu giống AlexNet (nó được đề xuất không lâu sau đó).
Các kích thước kernel lần lượt là $11\times 11$, $5\times 5$ và $3\times 3$,
và số lượng kênh đầu ra tương ứng với AlexNet. Mỗi khối NiN được theo sau bởi một lớp max-pooling
với stride bằng 2 và kích thước cửa sổ là $3\times 3$.

Khác biệt đáng kể thứ hai giữa NiN với cả AlexNet lẫn VGG
là NiN hoàn toàn tránh các lớp fully connected.
Thay vào đó, NiN sử dụng một khối NiN có số lượng kênh đầu ra bằng số lớp nhãn, sau đó là một lớp average pooling *toàn cục*,
tạo ra một vector logits.
Thiết kế này giảm đáng kể số lượng tham số mô hình cần thiết, mặc dù có thể làm tăng thời gian huấn luyện.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class NiN(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nin_block(96, kernel_size=11, strides=4, padding=0),
                nn.MaxPool2D(pool_size=3, strides=2),
                nin_block(256, kernel_size=5, strides=1, padding=2),
                nn.MaxPool2D(pool_size=3, strides=2),
                nin_block(384, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding=1),
                nn.GlobalAvgPool2D(),
                nn.Flatten())
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nin_block(96, kernel_size=11, strides=4, padding=0),
                nn.MaxPool2d(3, stride=2),
                nin_block(256, kernel_size=5, strides=1, padding=2),
                nn.MaxPool2d(3, stride=2),
                nin_block(384, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2d(3, stride=2),
                nn.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten())
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                nin_block(96, kernel_size=11, strides=4, padding='valid'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                nin_block(256, kernel_size=5, strides=1, padding='same'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                nin_block(384, kernel_size=3, strides=1, padding='same'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding='same'),
                tf.keras.layers.GlobalAvgPool2D(),
                tf.keras.layers.Flatten()])
```

```{.python .input}
%%tab jax
class NiN(d2l.Classifier):
    lr: float = 0.1
    num_classes = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nin_block(96, kernel_size=(11, 11), strides=(4, 4), padding=(0, 0)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nin_block(256, kernel_size=(5, 5), strides=(1, 1), padding=(2, 2)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nin_block(384, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nn.Dropout(0.5, deterministic=not self.training),
            nin_block(self.num_classes, kernel_size=(3, 3), strides=1, padding=(1, 1)),
            lambda x: nn.avg_pool(x, (5, 5)),  # global avg pooling
            lambda x: x.reshape((x.shape[0], -1))  # flatten
        ])
```

Chúng ta tạo một mẫu dữ liệu để xem [**hình dạng đầu ra của từng khối**].

```{.python .input}
%%tab mxnet, pytorch
NiN().layer_summary((1, 1, 224, 224))
```

```{.python .input}
%%tab tensorflow
NiN().layer_summary((1, 224, 224, 1))
```

```{.python .input}
%%tab jax
NiN(training=False).layer_summary((1, 224, 224, 1))
```

## [**Huấn luyện**]

Như trước, chúng ta sử dụng Fashion-MNIST để huấn luyện mô hình với cùng
optimizer đã dùng cho AlexNet và VGG.

```{.python .input}
%%tab mxnet, pytorch, jax
model = NiN(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = NiN(lr=0.05)
    trainer.fit(model, data)
```

## Tóm tắt

NiN có số lượng tham số ít hơn đáng kể so với AlexNet và VGG. Điều này chủ yếu xuất phát từ việc nó không cần các lớp fully connected khổng lồ. Thay vào đó, nó sử dụng global average pooling để tổng hợp trên tất cả các vị trí ảnh sau giai đoạn cuối của phần thân mạng. Điều này loại bỏ nhu cầu về các phép giảm chiều tốn kém (được học) và thay chúng bằng một phép lấy trung bình đơn giản. Điều khiến các nhà nghiên cứu ngạc nhiên vào thời điểm đó là phép lấy trung bình này không làm giảm độ chính xác. Lưu ý rằng lấy trung bình trên một biểu diễn độ phân giải thấp (với nhiều kênh) cũng làm tăng mức độ bất biến với phép dịch chuyển mà mạng có thể xử lý.

Việc chọn ít convolution hơn với kernel rộng và thay chúng bằng các convolution $1 \times 1$ càng hỗ trợ mục tiêu giảm số lượng tham số. Nó có thể đáp ứng một lượng đáng kể tính phi tuyến giữa các kênh tại bất kỳ vị trí nào. Cả convolution $1 \times 1$ và global average pooling đều ảnh hưởng đáng kể đến các thiết kế CNN về sau.

## Bài tập

1. Vì sao mỗi khối NiN có hai lớp convolution $1\times 1$? Tăng số lượng của chúng lên ba. Giảm số lượng của chúng xuống một. Có những thay đổi gì?
1. Điều gì thay đổi nếu bạn thay các convolution $1 \times 1$ bằng các convolution $3 \times 3$?
1. Điều gì xảy ra nếu bạn thay global average pooling bằng một lớp fully connected (tốc độ, độ chính xác, số lượng tham số)?
1. Hãy tính mức sử dụng tài nguyên của NiN.
    1. Số lượng tham số là bao nhiêu?
    1. Lượng tính toán là bao nhiêu?
    1. Lượng bộ nhớ cần thiết trong quá trình huấn luyện là bao nhiêu?
    1. Lượng bộ nhớ cần thiết trong quá trình dự đoán là bao nhiêu?
1. Những vấn đề có thể gặp phải khi giảm biểu diễn $384 \times 5 \times 5$ xuống biểu diễn $10 \times 5 \times 5$ trong một bước là gì?
1. Sử dụng các quyết định thiết kế cấu trúc trong VGG đã dẫn đến VGG-11, VGG-16 và VGG-19 để thiết kế một họ mạng tương tự NiN.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/79)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/80)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18003)
:end_tab: