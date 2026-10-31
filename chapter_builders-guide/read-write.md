```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# I/O tệp

Cho đến nay, chúng ta đã thảo luận về cách xử lý dữ liệu và cách
xây dựng, huấn luyện, cũng như kiểm thử các mô hình deep learning.
Tuy nhiên, vào một thời điểm nào đó, hy vọng chúng ta sẽ đủ hài lòng
với các mô hình đã học để muốn
lưu kết quả nhằm sử dụng sau này trong nhiều ngữ cảnh khác nhau
(có thể thậm chí để đưa ra dự đoán khi deploy).
Ngoài ra, khi chạy một quá trình huấn luyện dài,
thực hành tốt nhất là định kỳ lưu các kết quả trung gian (checkpointing)
để bảo đảm rằng chúng ta không mất lượng tính toán trị giá vài ngày
nếu vô tình vấp phải dây nguồn của máy chủ.
Vì vậy, đã đến lúc tìm hiểu cách nạp và lưu trữ
cả các vector trọng số riêng lẻ lẫn toàn bộ mô hình.
Phần này đề cập đến cả hai vấn đề.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
import numpy as np
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import flax
from flax import linen as nn
from flax.training import checkpoints
import jax
from jax import numpy as jnp
```

## (**Nạp và Lưu Tensor**)

Đối với từng tensor riêng lẻ, chúng ta có thể trực tiếp
gọi các hàm `load` và `save`
để lần lượt đọc và ghi chúng.
Cả hai hàm đều yêu cầu chúng ta cung cấp một tên,
và `save` yêu cầu biến cần lưu làm đầu vào.

```{.python .input}
%%tab mxnet
x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
%%tab pytorch
x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
%%tab tensorflow
x = tf.range(4)
np.save('x-file.npy', x)
```

```{.python .input}
%%tab jax
x = jnp.arange(4)
jnp.save('x-file.npy', x)
```

Bây giờ, chúng ta có thể đọc dữ liệu từ tệp đã lưu trở lại bộ nhớ.

```{.python .input}
%%tab mxnet
x2 = npx.load('x-file')
x2
```

```{.python .input}
%%tab pytorch
x2 = torch.load('x-file')
x2
```

```{.python .input}
%%tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

```{.python .input}
%%tab jax
x2 = jnp.load('x-file.npy', allow_pickle=True)
x2
```

Chúng ta có thể [**lưu một danh sách tensor và đọc chúng trở lại bộ nhớ.**]

```{.python .input}
%%tab mxnet
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

```{.python .input}
%%tab jax
y = jnp.zeros(4)
jnp.save('xy-files.npy', [x, y])
x2, y2 = jnp.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

Chúng ta thậm chí có thể [**ghi và đọc một dictionary ánh xạ
từ chuỗi sang tensor.**]
Điều này thuận tiện khi chúng ta muốn
đọc hoặc ghi tất cả trọng số trong một mô hình.

```{.python .input}
%%tab mxnet
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
%%tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
%%tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

```{.python .input}
%%tab jax
mydict = {'x': x, 'y': y}
jnp.save('mydict.npy', mydict)
mydict2 = jnp.load('mydict.npy', allow_pickle=True)
mydict2
```

## [**Nạp và Lưu Tham số Mô hình**]

Việc lưu các vector trọng số riêng lẻ (hoặc các tensor khác) rất hữu ích,
nhưng sẽ trở nên rất tẻ nhạt nếu chúng ta muốn lưu
(và sau đó nạp) toàn bộ một mô hình.
Dù sao đi nữa, chúng ta có thể có hàng trăm
nhóm tham số nằm rải rác khắp nơi.
Vì lý do này, framework deep learning cung cấp các chức năng tích hợp sẵn
để nạp và lưu toàn bộ mạng.
Một chi tiết quan trọng cần lưu ý là thao tác này
lưu *tham số* mô hình chứ không phải toàn bộ mô hình.
Ví dụ, nếu có một MLP 3 lớp,
chúng ta cần chỉ định kiến trúc riêng.
Lý do là bản thân các mô hình có thể chứa mã tùy ý,
do đó chúng không thể được serialize một cách tự nhiên.
Vì vậy, để khôi phục một mô hình, chúng ta cần
tạo kiến trúc trong mã
rồi nạp các tham số từ ổ đĩa.
(**Hãy bắt đầu với MLP quen thuộc của chúng ta.**)

```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

```{.python .input}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        self.hidden = nn.Dense(256)
        self.output = nn.Dense(10)

    def __call__(self, x):
        return self.output(nn.relu(self.hidden(x)))

net = MLP()
X = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (2, 20))
Y, params = net.init_with_output(jax.random.PRNGKey(d2l.get_seed()), X)
```

Tiếp theo, chúng ta [**lưu các tham số của mô hình thành một tệp**] có tên "mlp.params".

```{.python .input}
%%tab mxnet
net.save_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
%%tab tensorflow
net.save_weights('mlp.params')
```

```{.python .input}
%%tab jax
checkpoints.save_checkpoint('ckpt_dir', params, step=1, overwrite=True)
```

Để khôi phục mô hình, chúng ta khởi tạo một bản sao
của mô hình MLP gốc.
Thay vì khởi tạo ngẫu nhiên các tham số mô hình,
chúng ta [**đọc trực tiếp các tham số được lưu trong tệp**].

```{.python .input}
%%tab mxnet
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

```{.python .input}
%%tab tensorflow
clone = MLP()
clone.load_weights('mlp.params')
```

```{.python .input}
%%tab jax
clone = MLP()
cloned_params = flax.core.freeze(checkpoints.restore_checkpoint('ckpt_dir',
                                                                target=None))
```

Vì cả hai instance đều có cùng tham số mô hình,
kết quả tính toán cho cùng đầu vào `X` phải giống nhau.
Hãy xác minh điều này.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
%%tab jax
Y_clone = clone.apply(cloned_params, X)
Y_clone == Y
```

## Tóm tắt

Các hàm `save` và `load` có thể được dùng để thực hiện I/O tệp cho các đối tượng tensor.
Chúng ta có thể lưu và nạp toàn bộ tập tham số của một mạng thông qua một parameter dictionary.
Việc lưu kiến trúc phải được thực hiện trong mã thay vì trong các tham số.

## Bài tập

1. Ngay cả khi không cần deploy các mô hình đã huấn luyện sang một thiết bị khác, những lợi ích thực tế của việc lưu trữ tham số mô hình là gì?
1. Giả sử chúng ta chỉ muốn tái sử dụng các phần của một mạng để tích hợp vào một mạng có kiến trúc khác. Bạn sẽ sử dụng, chẳng hạn, hai lớp đầu tiên từ một mạng trước đó trong một mạng mới như thế nào?
1. Bạn sẽ lưu kiến trúc mạng và các tham số như thế nào? Bạn sẽ áp đặt những hạn chế nào đối với kiến trúc?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/327)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17994)
:end_tab: