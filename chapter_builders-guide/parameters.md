```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Quản lý tham số

Sau khi đã chọn một kiến trúc
và thiết lập các hyperparameter,
chúng ta tiếp tục đến vòng lặp huấn luyện,
nơi mục tiêu là tìm các giá trị tham số
giảm thiểu hàm mất mát của chúng ta.
Sau khi huấn luyện, chúng ta sẽ cần các tham số này
để đưa ra các dự đoán trong tương lai.
Ngoài ra, đôi khi chúng ta cũng muốn
trích xuất các tham số,
có thể để tái sử dụng chúng trong một ngữ cảnh khác,
lưu mô hình vào đĩa để
có thể thực thi trong phần mềm khác,
hoặc kiểm tra chúng với hy vọng
thu được hiểu biết khoa học.

Phần lớn thời gian, chúng ta có thể
bỏ qua các chi tiết vụn vặt
về cách khai báo và thao tác tham số,
dựa vào các framework deep learning
để thực hiện phần việc nặng nhọc.
Tuy nhiên, khi rời xa
các kiến trúc xếp chồng với các layer tiêu chuẩn,
đôi khi chúng ta sẽ cần đi sâu vào
việc khai báo và thao tác tham số.
Trong phần này, chúng ta sẽ trình bày:

* Truy cập tham số để debug, chẩn đoán và trực quan hóa.
* Chia sẻ tham số giữa các thành phần mô hình khác nhau.

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

(**Chúng ta bắt đầu bằng cách tập trung vào một MLP có một hidden layer.**)

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
net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

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

## [**Truy cập tham số**]
:label:`subsec_param-access`

Hãy bắt đầu với cách truy cập các tham số
từ những mô hình mà bạn đã biết.

:begin_tab:`mxnet, pytorch, tensorflow`
Khi một mô hình được định nghĩa thông qua lớp `Sequential`,
trước tiên chúng ta có thể truy cập bất kỳ layer nào bằng cách lập chỉ mục
vào mô hình như thể nó là một danh sách.
Các tham số của mỗi layer được đặt thuận tiện
trong thuộc tính của nó.
:end_tab:

:begin_tab:`jax`
Flax và JAX tách rời mô hình và các tham số, như bạn
có thể đã quan sát trong các mô hình được định nghĩa trước đó.
Khi một mô hình được định nghĩa thông qua lớp `Sequential`,
trước tiên chúng ta cần khởi tạo mạng để tạo ra
từ điển tham số. Chúng ta có thể truy cập
các tham số của bất kỳ layer nào thông qua các khóa của từ điển này.
:end_tab:

Chúng ta có thể kiểm tra các tham số của fully connected layer thứ hai như sau.

```{.python .input}
%%tab mxnet
net[1].params
```

```{.python .input}
%%tab pytorch
net[2].state_dict()
```

```{.python .input}
%%tab tensorflow
net.layers[2].weights
```

```{.python .input}
%%tab jax
params['params']['layers_2']
```

Chúng ta có thể thấy rằng fully connected layer này
chứa hai tham số,
lần lượt tương ứng với
trọng số và bias của layer đó.


### [**Tham số mục tiêu**]

Lưu ý rằng mỗi tham số được biểu diễn
dưới dạng một instance của lớp tham số.
Để làm bất kỳ việc hữu ích nào với các tham số,
trước tiên chúng ta cần truy cập các giá trị số bên dưới.
Có nhiều cách để thực hiện việc này.
Một số đơn giản hơn, trong khi những cách khác tổng quát hơn.
Đoạn mã sau trích xuất bias
từ layer mạng nơ-ron thứ hai, trả về một instance của lớp tham số, và
sau đó truy cập giá trị của tham số đó.

```{.python .input}
%%tab mxnet
type(net[1].bias), net[1].bias.data()
```

```{.python .input}
%%tab pytorch
type(net[2].bias), net[2].bias.data
```

```{.python .input}
%%tab tensorflow
type(net.layers[2].weights[1]), tf.convert_to_tensor(net.layers[2].weights[1])
```

```{.python .input}
%%tab jax
bias = params['params']['layers_2']['bias']
type(bias), bias
```

:begin_tab:`mxnet,pytorch`
Tham số là các đối tượng phức tạp,
chứa giá trị, gradient,
và thông tin bổ sung.
Đó là lý do chúng ta cần yêu cầu giá trị một cách tường minh.

Ngoài giá trị, mỗi tham số cũng cho phép chúng ta truy cập gradient. Vì chúng ta chưa gọi backpropagation cho mạng này, nó đang ở trạng thái ban đầu.
:end_tab:

:begin_tab:`jax`
Không giống các framework khác, JAX không theo dõi gradient trên
các tham số mạng nơ-ron; thay vào đó, các tham số và mạng được tách rời.
Nó cho phép người dùng biểu đạt phép tính của họ dưới dạng một
hàm Python, và sử dụng phép biến đổi `grad` cho cùng mục đích.
:end_tab:

```{.python .input}
%%tab mxnet
net[1].weight.grad()
```

```{.python .input}
%%tab pytorch
net[2].weight.grad == None
```

### [**Tất cả tham số cùng lúc**]

Khi cần thực hiện thao tác trên tất cả các tham số,
việc truy cập từng tham số một có thể trở nên tẻ nhạt.
Tình huống này có thể đặc biệt khó xử lý
khi làm việc với các module phức tạp hơn, chẳng hạn các module lồng nhau,
vì chúng ta sẽ cần đệ quy
qua toàn bộ cây để trích xuất
tham số của từng sub-module. Dưới đây, chúng ta minh họa cách truy cập tham số của tất cả các layer.

```{.python .input}
%%tab mxnet
net.collect_params()
```

```{.python .input}
%%tab pytorch
[(name, param.shape) for name, param in net.named_parameters()]
```

```{.python .input}
%%tab tensorflow
net.get_weights()
```

```{.python .input}
%%tab jax
jax.tree_util.tree_map(lambda x: x.shape, params)
```

## [**Tham số ràng buộc**]

Thông thường, chúng ta muốn chia sẻ tham số giữa nhiều layer.
Hãy xem cách thực hiện điều này một cách gọn gàng.
Trong phần sau, chúng ta cấp phát một fully connected layer
rồi sử dụng riêng các tham số của nó
để thiết lập tham số cho một layer khác.
Ở đây, chúng ta cần chạy forward propagation
`net(X)` trước khi truy cập các tham số.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))

net(X)
# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
%%tab pytorch
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
%%tab tensorflow
# tf.keras behaves a bit differently. It removes the duplicate layer
# automatically
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# Check whether the parameters are different
print(len(net.layers) == 3)
```

```{.python .input}
%%tab jax
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8)
net = nn.Sequential([nn.Dense(8), nn.relu,
                     shared, nn.relu,
                     shared, nn.relu,
                     nn.Dense(1)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)

# Check whether the parameters are different
print(len(params['params']) == 3)
```

Ví dụ này cho thấy các tham số
của layer thứ hai và thứ ba được ràng buộc.
Chúng không chỉ bằng nhau, mà còn
được biểu diễn bởi chính xác cùng một tensor.
Do đó, nếu chúng ta thay đổi một trong các tham số,
tham số còn lại cũng thay đổi.

:begin_tab:`mxnet, pytorch, tensorflow`
Bạn có thể tự hỏi,
khi các tham số được ràng buộc,
điều gì xảy ra với gradient?
Vì các tham số mô hình chứa gradient,
gradient của hidden layer thứ hai
và hidden layer thứ ba được cộng lại
trong quá trình backpropagation.
:end_tab:


## Tóm tắt

Chúng ta có một số cách để truy cập và ràng buộc các tham số mô hình.


## Bài tập

1. Sử dụng mô hình `NestMLP` được định nghĩa trong :numref:`sec_model_construction` và truy cập các tham số của những layer khác nhau.
1. Xây dựng một MLP chứa một layer tham số dùng chung và huấn luyện nó. Trong quá trình huấn luyện, quan sát các tham số mô hình và gradient của từng layer.
1. Vì sao chia sẻ tham số là một ý tưởng tốt?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/269)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17990)
:end_tab: