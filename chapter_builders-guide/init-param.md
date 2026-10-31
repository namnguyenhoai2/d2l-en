```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Khởi tạo tham số

Giờ đây khi đã biết cách truy cập các tham số,
hãy xem cách khởi tạo chúng đúng cách.
Chúng ta đã thảo luận về sự cần thiết của việc khởi tạo phù hợp trong :numref:`sec_numerical_stability`.
Framework deep learning cung cấp các phương thức khởi tạo ngẫu nhiên mặc định cho các layer của nó.
Tuy nhiên, chúng ta thường muốn khởi tạo các trọng số
theo nhiều quy tắc khác nhau. Framework cung cấp hầu hết các
quy tắc thường dùng, đồng thời cũng cho phép tạo một initializer tùy chỉnh.

```{.python .input}
%%tab mxnet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

:begin_tab:`mxnet`
Theo mặc định, MXNet khởi tạo các tham số trọng số bằng cách lấy ngẫu nhiên từ phân phối đều $U(-0.07, 0.07)$,
và đặt các tham số bias về 0.
Module `init` của MXNet cung cấp nhiều
phương thức khởi tạo dựng sẵn.
:end_tab:

:begin_tab:`pytorch`
Theo mặc định, PyTorch khởi tạo các ma trận trọng số và bias
theo phân phối đều bằng cách lấy mẫu từ một khoảng được tính dựa trên chiều đầu vào và đầu ra.
Module `nn.init` của PyTorch cung cấp nhiều
phương thức khởi tạo dựng sẵn.
:end_tab:

:begin_tab:`tensorflow`
Theo mặc định, Keras khởi tạo các ma trận trọng số theo phân phối đều bằng cách lấy mẫu từ một khoảng được tính dựa trên chiều đầu vào và đầu ra, còn các tham số bias đều được đặt về 0.
TensorFlow cung cấp nhiều phương thức khởi tạo cả trong module gốc lẫn module `keras.initializers`.
:end_tab:

:begin_tab:`jax`
Theo mặc định, Flax khởi tạo trọng số bằng `jax.nn.initializers.lecun_normal`,
nghĩa là lấy mẫu từ một phân phối chuẩn bị cắt cụt có tâm tại 0 với
độ lệch chuẩn được đặt bằng căn bậc hai của $1 / \textrm{fan}_{\textrm{in}}$
trong đó `fan_in` là số lượng unit đầu vào trong tensor trọng số. Các tham số bias
đều được đặt về 0.
Module `nn.initializers` của Jax cung cấp nhiều
phương thức khởi tạo dựng sẵn.
:end_tab:

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X).shape
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8), nn.relu, nn.Dense(1)])
X = jax.random.uniform(d2l.get_key(), (2, 4))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

## [**Khởi tạo dựng sẵn**]

Hãy bắt đầu bằng cách gọi các initializer dựng sẵn.
Đoạn mã dưới đây khởi tạo tất cả các tham số trọng số
là các biến ngẫu nhiên Gaussian
với độ lệch chuẩn 0.01, trong khi các tham số bias được đặt về 0.

```{.python .input}
%%tab mxnet
# Here force_reinit ensures that parameters are freshly initialized even if
# they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
%%tab jax
weight_init = nn.initializers.normal(0.01)
bias_init = nn.initializers.zeros

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

Chúng ta cũng có thể khởi tạo tất cả các tham số
thành một giá trị hằng cho trước (ví dụ: 1).

```{.python .input}
%%tab mxnet
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
%%tab jax
weight_init = nn.initializers.constant(1)

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

[**Chúng ta cũng có thể áp dụng các initializer khác nhau cho từng block nhất định.**]
Ví dụ, bên dưới chúng ta khởi tạo layer đầu tiên
bằng Xavier initializer
và khởi tạo layer thứ hai
thành giá trị hằng là 42.

```{.python .input}
%%tab mxnet
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input}
%%tab pytorch
def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(42)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8, kernel_init=nn.initializers.xavier_uniform(),
                              bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=nn.initializers.constant(42),
                              bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
params['params']['layers_0']['kernel'][:, 0], params['params']['layers_2']['kernel']
```

### [**Khởi tạo tùy chỉnh**]

Đôi khi, các phương thức khởi tạo chúng ta cần
không được framework deep learning cung cấp.
Trong ví dụ dưới đây, chúng ta định nghĩa một initializer
cho bất kỳ tham số trọng số $w$ nào bằng phân phối kỳ lạ sau:

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \textrm{ with probability } \frac{1}{4} \\
            0    & \textrm{ with probability } \frac{1}{2} \\
        U(-10, -5) & \textrm{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
Ở đây, chúng ta định nghĩa một lớp con của lớp `Initializer`.
Thông thường, chúng ta chỉ cần triển khai hàm `_init_weight`,
hàm này nhận một đối số tensor (`data`)
và gán cho nó các giá trị khởi tạo mong muốn.
:end_tab:

:begin_tab:`pytorch`
Một lần nữa, chúng ta triển khai một hàm `my_init` để áp dụng cho `net`.
:end_tab:

:begin_tab:`tensorflow`
Ở đây, chúng ta định nghĩa một lớp con của `Initializer` và triển khai
hàm `__call__` trả về một tensor mong muốn dựa trên shape và kiểu dữ liệu.
:end_tab:

:begin_tab:`jax`
Các hàm khởi tạo Jax nhận `PRNGKey`, `shape` và
`dtype` làm đối số. Ở đây, chúng ta triển khai hàm `my_init` trả về một
tensor mong muốn dựa trên shape và kiểu dữ liệu.
:end_tab:

```{.python .input}
%%tab mxnet
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input}
%%tab pytorch
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input}
%%tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

```{.python .input}
%%tab jax
def my_init(key, shape, dtype=jnp.float_):
    data = jax.random.uniform(key, shape, minval=-10, maxval=10)
    return data * (jnp.abs(data) >= 5)

net = nn.Sequential([nn.Dense(8, kernel_init=my_init), nn.relu, nn.Dense(1)])
params = net.init(d2l.get_key(), X)
print(params['params']['layers_0']['kernel'][:, :2])
```

:begin_tab:`mxnet, pytorch, tensorflow`
Lưu ý rằng chúng ta luôn có thể
thiết lập trực tiếp các tham số.
:end_tab:

:begin_tab:`jax`
Khi khởi tạo tham số trong JAX và Flax, dictionary tham số
được trả về có kiểu `flax.core.frozen_dict.FrozenDict`. Trong
hệ sinh thái Jax, không nên trực tiếp thay đổi giá trị của một array, do đó các kiểu dữ liệu
thường là bất biến. Có thể dùng `params.unfreeze()` để thực hiện thay đổi.
:end_tab:

```{.python .input}
%%tab mxnet
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input}
%%tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

## Tóm tắt

Chúng ta có thể khởi tạo tham số bằng các initializer dựng sẵn và tùy chỉnh.

## Bài tập

Tra cứu tài liệu trực tuyến để biết thêm các initializer dựng sẵn.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/8089)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/8090)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/8091)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17991)
:end_tab: