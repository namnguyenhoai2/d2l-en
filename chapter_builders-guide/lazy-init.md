```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Khởi tạo trì hoãn
:label:`sec_lazy_init`

Cho đến nay, có thể có vẻ như chúng ta đã may mắn
khi thiết lập mạng một cách khá tùy tiện.
Cụ thể, chúng ta đã thực hiện những điều không trực quan sau đây,
mà có thể trông như không thể hoạt động:

* Chúng ta đã định nghĩa kiến trúc mạng
  mà không chỉ định số chiều đầu vào.
* Chúng ta đã thêm các layer mà không chỉ định
  số chiều đầu ra của layer trước đó.
* Chúng ta thậm chí đã "khởi tạo" các tham số này
  trước khi cung cấp đủ thông tin để xác định
  mô hình của chúng ta cần chứa bao nhiêu tham số.

Bạn có thể ngạc nhiên rằng mã của chúng ta vẫn chạy được.
Xét cho cùng, framework deep learning không thể nào biết
số chiều đầu vào của một mạng là bao nhiêu.
Mấu chốt ở đây là framework *trì hoãn việc khởi tạo*,
chờ đến lần đầu tiên chúng ta truyền dữ liệu qua mô hình
để suy ra kích thước của từng layer một cách tự động.


Sau này, khi làm việc với mạng nơ-ron tích chập,
kỹ thuật này sẽ trở nên thuận tiện hơn nữa
vì số chiều đầu vào
(ví dụ: độ phân giải của một hình ảnh)
sẽ ảnh hưởng đến số chiều
của từng layer tiếp theo.
Do đó, khả năng thiết lập các tham số
mà không cần biết,
ngay tại thời điểm viết mã,
giá trị của số chiều
có thể đơn giản hóa đáng kể việc chỉ định
và sau đó chỉnh sửa các mô hình của chúng ta.
Tiếp theo, chúng ta sẽ tìm hiểu sâu hơn về cơ chế khởi tạo.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
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
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

Để bắt đầu, hãy khởi tạo một MLP.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])
```

Tại thời điểm này, mạng không thể biết
số chiều của trọng số ở layer đầu vào
vì số chiều đầu vào vẫn chưa xác định.

:begin_tab:`mxnet, pytorch, tensorflow`
Do đó, framework vẫn chưa khởi tạo bất kỳ tham số nào.
Chúng ta xác nhận điều này bằng cách thử truy cập các tham số bên dưới.
:end_tab:

:begin_tab:`jax`
Như đã đề cập trong :numref:`subsec_param-access`, tham số và định nghĩa mạng được tách rời
trong Jax và Flax, và người dùng tự xử lý cả hai. Các mô hình Flax không trạng thái
nên không có thuộc tính `parameters`.
:end_tab:

```{.python .input}
%%tab mxnet
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
%%tab pytorch
net[0].weight
```

```{.python .input}
%%tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
Lưu ý rằng mặc dù các đối tượng tham số tồn tại,
số chiều đầu vào của mỗi layer được liệt kê là -1.
MXNet sử dụng giá trị đặc biệt -1 để biểu thị
rằng số chiều tham số vẫn chưa xác định.
Tại thời điểm này, việc cố gắng truy cập `net[0].weight.data()`
sẽ kích hoạt lỗi runtime, cho biết rằng mạng
phải được khởi tạo trước khi có thể truy cập các tham số.
Bây giờ hãy xem điều gì xảy ra khi chúng ta cố gắng khởi tạo
tham số thông qua phương thức `initialize`.
:end_tab:

:begin_tab:`tensorflow`
Lưu ý rằng các đối tượng layer đều tồn tại nhưng trọng số trống.
Việc dùng `net.get_weights()` sẽ gây ra lỗi vì các trọng số
vẫn chưa được khởi tạo.
:end_tab:

```{.python .input}
%%tab mxnet
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
Như chúng ta có thể thấy, không có gì thay đổi.
Khi chưa biết số chiều đầu vào,
các lệnh gọi initialize không thực sự khởi tạo các tham số.
Thay vào đó, lệnh gọi này đăng ký với MXNet rằng chúng ta muốn
(và tùy chọn, theo phân phối nào)
khởi tạo các tham số.
:end_tab:

Tiếp theo, hãy truyền dữ liệu qua mạng
để framework cuối cùng khởi tạo các tham số.

```{.python .input}
%%tab mxnet
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
%%tab pytorch
X = torch.rand(2, 20)
net(X)

net[0].weight.shape
```

```{.python .input}
%%tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

```{.python .input}
%%tab jax
params = net.init(d2l.get_key(), jnp.zeros((2, 20)))
jax.tree_util.tree_map(lambda x: x.shape, params).tree_flatten_with_keys()
```

Ngay khi biết số chiều đầu vào,
20,
framework có thể xác định hình dạng ma trận trọng số của layer đầu tiên bằng cách thay giá trị 20 vào.
Sau khi nhận ra hình dạng của layer đầu tiên, framework tiếp tục
đến layer thứ hai,
và cứ thế qua computational graph
cho đến khi biết được tất cả các hình dạng.
Lưu ý rằng trong trường hợp này,
chỉ layer đầu tiên cần khởi tạo trì hoãn,
nhưng framework khởi tạo tuần tự.
Khi tất cả hình dạng tham số đã được biết,
framework cuối cùng có thể khởi tạo các tham số.

:begin_tab:`pytorch`
Phương thức sau đây
truyền đầu vào giả
qua mạng
để chạy thử
nhằm suy ra mọi hình dạng tham số
và sau đó khởi tạo các tham số.
Phương thức này sẽ được sử dụng sau khi không mong muốn các khởi tạo ngẫu nhiên mặc định.
:end_tab:

:begin_tab:`jax`
Việc khởi tạo tham số trong Flax luôn được thực hiện thủ công và do
người dùng xử lý. Phương thức sau nhận đầu vào giả và một từ điển khóa làm đối số.
Từ điển khóa này chứa các rngs để khởi tạo tham số mô hình
và dropout rng để tạo dropout mask cho các mô hình có
dropout layer. Chúng ta sẽ tìm hiểu thêm về dropout trong :numref:`sec_dropout`.
Cuối cùng, phương thức khởi tạo mô hình và trả về các tham số.
Chúng ta cũng đã sử dụng nó ngầm trong các phần trước.
:end_tab:

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, dummy_input, key):
    params = self.init(key, *dummy_input)  # dummy_input tuple unpacked
    return params
```

## Tóm tắt

Khởi tạo trì hoãn có thể thuận tiện, cho phép framework tự động suy ra hình dạng tham số, giúp dễ dàng chỉnh sửa kiến trúc và loại bỏ một nguồn lỗi phổ biến.
Chúng ta có thể truyền dữ liệu qua mô hình để framework cuối cùng khởi tạo các tham số.


## Bài tập

1. Điều gì xảy ra nếu bạn chỉ định số chiều đầu vào cho layer đầu tiên nhưng không chỉ định cho các layer tiếp theo? Bạn có nhận được khởi tạo ngay lập tức không?
1. Điều gì xảy ra nếu bạn chỉ định các số chiều không khớp?
1. Bạn cần làm gì nếu đầu vào có số chiều thay đổi? Gợi ý: hãy xem parameter tying.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/8092)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/281)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17992)
:end_tab: