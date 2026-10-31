```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Các Layer Tùy chỉnh

Một yếu tố đằng sau thành công của deep learning
là sự sẵn có của nhiều loại layer
có thể được kết hợp theo những cách sáng tạo
để thiết kế các kiến trúc phù hợp
cho rất nhiều tác vụ khác nhau.
Ví dụ, các nhà nghiên cứu đã phát minh ra các layer
chuyên biệt để xử lý hình ảnh, văn bản,
lặp qua dữ liệu tuần tự,
và
thực hiện dynamic programming.
Sớm hay muộn, bạn sẽ cần
một layer chưa tồn tại trong framework deep learning.
Trong những trường hợp này, bạn phải xây dựng một layer tùy chỉnh.
Trong phần này, chúng tôi sẽ hướng dẫn bạn cách thực hiện.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
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
import jax
from jax import numpy as jnp
```

## (**Các Layer Không có Tham số**)

Trước tiên, chúng ta xây dựng một layer tùy chỉnh
không có bất kỳ tham số nào riêng.
Điều này hẳn sẽ quen thuộc nếu bạn nhớ lại phần
giới thiệu về module trong :numref:`sec_model_construction`.
Lớp `CenteredLayer` dưới đây chỉ đơn giản
trừ giá trị trung bình khỏi đầu vào của nó.
Để xây dựng nó, chúng ta chỉ cần kế thừa
từ lớp layer cơ sở và triển khai hàm lan truyền xuôi.

```{.python .input}
%%tab mxnet
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab pytorch
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab tensorflow
class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, X):
        return X - tf.reduce_mean(X)
```

```{.python .input}
%%tab jax
class CenteredLayer(nn.Module):
    def __call__(self, X):
        return X - X.mean()
```

Hãy xác minh layer của chúng ta hoạt động như mong muốn bằng cách truyền một số dữ liệu qua nó.

```{.python .input}
%%tab all
layer = CenteredLayer()
layer(d2l.tensor([1.0, 2, 3, 4, 5]))
```

Giờ đây, chúng ta có thể [**tích hợp layer của mình như một thành phần
để xây dựng các mô hình phức tạp hơn.**]

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
```

```{.python .input}
%%tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(128), CenteredLayer()])
```

Để kiểm tra thêm tính hợp lý, chúng ta có thể truyền dữ liệu ngẫu nhiên
qua mạng và kiểm tra xem giá trị trung bình có thực sự bằng 0 hay không.
Vì đang làm việc với các số dấu phẩy động,
chúng ta vẫn có thể thấy một số khác 0 rất nhỏ
do lượng tử hóa.

:begin_tab:`jax`
Ở đây, chúng ta sử dụng phương thức `init_with_output`, phương thức này trả về cả đầu ra của
mạng lẫn các tham số. Trong trường hợp này, chúng ta chỉ tập trung vào
đầu ra.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet
Y = net(d2l.rand(4, 8))
Y.mean()
```

```{.python .input}
%%tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

```{.python .input}
%%tab jax
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (4, 8)))
Y.mean()
```

## [**Các Layer có Tham số**]

Bây giờ khi đã biết cách định nghĩa các layer đơn giản,
hãy chuyển sang định nghĩa các layer có tham số
có thể được điều chỉnh thông qua huấn luyện.
Chúng ta có thể dùng các hàm có sẵn để tạo tham số, chúng
cung cấp một số chức năng quản lý cơ bản.
Cụ thể, chúng quản lý việc truy cập, khởi tạo,
chia sẻ, lưu và tải các tham số mô hình.
Nhờ đó, cùng với các lợi ích khác, chúng ta sẽ không cần viết
các quy trình tuần tự hóa tùy chỉnh cho từng layer tùy chỉnh.

Bây giờ, hãy triển khai phiên bản riêng của chúng ta cho layer fully connected.
Hãy nhớ rằng layer này cần hai tham số,
một tham số biểu diễn weight và tham số còn lại biểu diễn bias.
Trong triển khai này, chúng ta đặt sẵn activation ReLU làm mặc định.
Layer này cần hai đối số đầu vào: `in_units` và `units`, lần lượt
biểu thị số lượng đầu vào và đầu ra.

```{.python .input}
%%tab mxnet
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
%%tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```{.python .input}
%%tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

```{.python .input}
%%tab jax
class MyDense(nn.Module):
    in_units: int
    units: int

    def setup(self):
        self.weight = self.param('weight', nn.initializers.normal(stddev=1),
                                 (self.in_units, self.units))
        self.bias = self.param('bias', nn.initializers.zeros, self.units)

    def __call__(self, X):
        linear = jnp.matmul(X, self.weight) + self.bias
        return nn.relu(linear)
```

:begin_tab:`mxnet, tensorflow, jax`
Tiếp theo, chúng ta khởi tạo lớp `MyDense`
và truy cập các tham số mô hình của nó.
:end_tab:

:begin_tab:`pytorch`
Tiếp theo, chúng ta khởi tạo lớp `MyLinear`
và truy cập các tham số mô hình của nó.
:end_tab:

```{.python .input}
%%tab mxnet
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
%%tab pytorch
linear = MyLinear(5, 3)
linear.weight
```

```{.python .input}
%%tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

```{.python .input}
%%tab jax
dense = MyDense(5, 3)
params = dense.init(d2l.get_key(), jnp.zeros((3, 5)))
params
```

Chúng ta có thể [**trực tiếp thực hiện các phép tính lan truyền xuôi bằng các layer tùy chỉnh.**]

```{.python .input}
%%tab mxnet
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
%%tab pytorch
linear(torch.rand(2, 5))
```

```{.python .input}
%%tab tensorflow
dense(tf.random.uniform((2, 5)))
```

```{.python .input}
%%tab jax
dense.apply(params, jax.random.uniform(d2l.get_key(),
                                       (2, 5)))
```

Chúng ta cũng có thể (**xây dựng mô hình bằng các layer tùy chỉnh.**)
Khi đã có nó, chúng ta có thể sử dụng nó giống như layer fully connected có sẵn.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

```{.python .input}
%%tab jax
net = nn.Sequential([MyDense(64, 8), MyDense(8, 1)])
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (2, 64)))
Y
```

## Tóm tắt

Chúng ta có thể thiết kế các layer tùy chỉnh thông qua lớp layer cơ sở. Điều này cho phép chúng ta định nghĩa các layer mới linh hoạt, hoạt động khác với bất kỳ layer hiện có nào trong thư viện.
Sau khi được định nghĩa, các layer tùy chỉnh có thể được gọi trong các ngữ cảnh và kiến trúc tùy ý.
Các layer có thể có các tham số cục bộ, được tạo thông qua các hàm có sẵn.


## Bài tập

1. Thiết kế một layer nhận đầu vào và tính phép rút gọn tensor,
   tức là, nó trả về $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
1. Thiết kế một layer trả về nửa đầu các hệ số Fourier của dữ liệu.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/279)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17993)
:end_tab: