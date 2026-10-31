```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Các tầng và mô-đun
:label:`sec_model_construction`

Khi lần đầu giới thiệu mạng nơ-ron,
chúng ta tập trung vào các mô hình tuyến tính với một đầu ra duy nhất.
Ở đây, toàn bộ mô hình chỉ gồm một nơ-ron duy nhất.
Lưu ý rằng một nơ-ron đơn lẻ
(i) nhận một tập đầu vào;
(ii) tạo ra một đầu ra vô hướng tương ứng;
và (iii) có một tập tham số liên quan có thể được cập nhật
để tối ưu hóa một hàm mục tiêu cần quan tâm.
Sau đó, khi bắt đầu suy nghĩ về các mạng có nhiều đầu ra,
chúng ta đã tận dụng phép toán vector hóa
để mô tả đặc trưng cho cả một tầng nơ-ron.
Tương tự như từng nơ-ron riêng lẻ,
các tầng (i) nhận một tập đầu vào,
(ii) tạo ra các đầu ra tương ứng,
và (iii) được mô tả bởi một tập tham số có thể điều chỉnh.
Khi tìm hiểu hồi quy softmax,
một tầng duy nhất chính là mô hình.
Tuy nhiên, ngay cả khi sau đó
giới thiệu các MLP,
chúng ta vẫn có thể xem mô hình
là giữ nguyên cấu trúc cơ bản này.

Điều thú vị là đối với MLP,
cả toàn bộ mô hình lẫn các tầng cấu thành của nó
đều chia sẻ cấu trúc này.
Toàn bộ mô hình nhận đầu vào thô (các đặc trưng),
tạo ra đầu ra (các dự đoán),
và sở hữu các tham số
(các tham số kết hợp từ mọi tầng cấu thành).
Tương tự, mỗi tầng riêng lẻ nhận đầu vào
(do tầng trước đó cung cấp),
tạo ra đầu ra (đầu vào cho tầng tiếp theo),
và sở hữu một tập tham số có thể điều chỉnh được cập nhật
theo tín hiệu lan truyền ngược
từ tầng tiếp theo.


Mặc dù bạn có thể nghĩ rằng nơ-ron, tầng và mô hình
đã cung cấp đủ các trừu tượng để làm việc,
trên thực tế, chúng ta thường thấy thuận tiện
khi nói về các thành phần
lớn hơn một tầng riêng lẻ
nhưng nhỏ hơn toàn bộ mô hình.
Ví dụ, kiến trúc ResNet-152,
vốn cực kỳ phổ biến trong thị giác máy tính,
có hàng trăm tầng.
Các tầng này bao gồm những mẫu lặp lại của *các nhóm tầng*. Việc triển khai một mạng như vậy từng tầng một có thể trở nên tẻ nhạt.
Mối bận tâm này không chỉ là giả thuyết---các
mẫu thiết kế như vậy phổ biến trong thực tế.
Kiến trúc ResNet được đề cập ở trên
đã chiến thắng các cuộc thi thị giác máy tính ImageNet và COCO năm 2015
ở cả nhận dạng lẫn phát hiện :cite:`He.Zhang.Ren.ea.2016`
và vẫn là một kiến trúc được ưu tiên cho nhiều tác vụ thị giác.
Các kiến trúc tương tự, trong đó các tầng được sắp xếp
theo nhiều mẫu lặp lại khác nhau,
hiện nay cũng phổ biến rộng rãi ở các lĩnh vực khác,
bao gồm xử lý ngôn ngữ tự nhiên và tiếng nói.

Để triển khai các mạng phức tạp này,
chúng ta giới thiệu khái niệm *mô-đun* mạng nơ-ron.
Một mô-đun có thể mô tả một tầng đơn lẻ,
một thành phần gồm nhiều tầng,
hoặc chính toàn bộ mô hình!
Một lợi ích của việc làm việc với trừu tượng mô-đun
là chúng có thể được kết hợp thành các cấu trúc lớn hơn,
thường theo cách đệ quy. Điều này được minh họa trong :numref:`fig_blocks`. Bằng cách định nghĩa mã để tạo các mô-đun
có độ phức tạp tùy ý theo yêu cầu,
chúng ta có thể viết mã nhỏ gọn đáng ngạc nhiên
mà vẫn triển khai được các mạng nơ-ron phức tạp.

![Nhiều tầng được kết hợp thành các mô-đun, tạo nên các mẫu lặp lại của những mô hình lớn hơn.](../img/blocks.svg)
:label:`fig_blocks`


Từ góc độ lập trình, một mô-đun được biểu diễn bằng một *lớp*.
Mọi lớp con của nó phải định nghĩa một phương thức lan truyền xuôi
biến đổi đầu vào thành đầu ra
và phải lưu trữ mọi tham số cần thiết.
Lưu ý rằng một số mô-đun hoàn toàn không cần bất kỳ tham số nào.
Cuối cùng, một mô-đun phải có phương thức lan truyền ngược,
nhằm mục đích tính gradient.
May mắn thay, nhờ một số phép thuật hậu trường
do tự động vi phân
(được giới thiệu trong :numref:`sec_autograd`)
cung cấp, khi định nghĩa mô-đun riêng,
chúng ta chỉ cần quan tâm đến các tham số
và phương thức lan truyền xuôi.

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
```

```{.python .input}
%%tab jax
from typing import List
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

[**Để bắt đầu, chúng ta xem lại đoạn mã
đã dùng để triển khai MLP**]
(:numref:`sec_mlp`).
Đoạn mã sau tạo một mạng
có một tầng ẩn kết nối đầy đủ
với 256 đơn vị và kích hoạt ReLU,
tiếp theo là một tầng đầu ra kết nối đầy đủ
với mười đơn vị (không có hàm kích hoạt).

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X).shape
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

X = torch.rand(2, 20)
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X).shape
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])

# get_key is a d2l saved function returning jax.random.PRNGKey(random_seed)
X = jax.random.uniform(d2l.get_key(), (2, 20))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

:begin_tab:`mxnet`
Trong ví dụ này, chúng ta xây dựng
mô hình bằng cách khởi tạo một `nn.Sequential`,
gán đối tượng trả về cho biến `net`.
Tiếp theo, chúng ta gọi lặp lại phương thức `add` của nó,
nối thêm các tầng theo thứ tự
mà chúng cần được thực thi.
Tóm lại, `nn.Sequential` định nghĩa một dạng đặc biệt của `Block`,
lớp biểu diễn một *mô-đun* trong Gluon.
Nó duy trì một danh sách có thứ tự gồm các `Block` cấu thành.
Phương thức `add` chỉ đơn giản tạo điều kiện
để thêm từng `Block` kế tiếp vào danh sách.
Lưu ý rằng mỗi tầng là một thực thể của lớp `Dense`
mà bản thân nó là lớp con của `Block`.
Phương thức lan truyền xuôi (`forward`) cũng rất đơn giản:
nó liên kết từng `Block` trong danh sách lại với nhau,
truyền đầu ra của mỗi phần làm đầu vào cho phần tiếp theo.
Lưu ý rằng cho đến nay, chúng ta vẫn gọi các mô hình
thông qua cấu trúc `net(X)` để nhận đầu ra.
Đây thực chất chỉ là cách viết tắt cho `net.forward(X)`,
một thủ thuật Python tinh tế được thực hiện thông qua
phương thức `__call__` của lớp `Block`.
:end_tab:

:begin_tab:`pytorch`
Trong ví dụ này, chúng ta xây dựng
mô hình bằng cách khởi tạo một `nn.Sequential`, với các tầng theo thứ tự
mà chúng cần được thực thi được truyền vào làm đối số.
Tóm lại, (**`nn.Sequential` định nghĩa một dạng đặc biệt của `Module`**),
lớp biểu diễn một mô-đun trong PyTorch.
Nó duy trì một danh sách có thứ tự gồm các `Module` cấu thành.
Lưu ý rằng mỗi trong hai tầng kết nối đầy đủ là một thực thể của lớp `Linear`
mà bản thân nó là lớp con của `Module`.
Phương thức lan truyền xuôi (`forward`) cũng rất đơn giản:
nó liên kết từng mô-đun trong danh sách lại với nhau,
truyền đầu ra của mỗi mô-đun làm đầu vào cho mô-đun tiếp theo.
Lưu ý rằng cho đến nay, chúng ta vẫn gọi các mô hình
thông qua cấu trúc `net(X)` để nhận đầu ra.
Đây thực chất chỉ là cách viết tắt cho `net.__call__(X)`.
:end_tab:

:begin_tab:`tensorflow`
Trong ví dụ này, chúng ta xây dựng
mô hình bằng cách khởi tạo một `keras.models.Sequential`, với các tầng theo thứ tự
mà chúng cần được thực thi được truyền vào làm đối số.
Tóm lại, `Sequential` định nghĩa một dạng đặc biệt của `keras.Model`,
lớp biểu diễn một mô-đun trong Keras.
Nó duy trì một danh sách có thứ tự gồm các `Model` cấu thành.
Lưu ý rằng mỗi trong hai tầng kết nối đầy đủ là một thực thể của lớp `Dense`
mà bản thân nó là lớp con của `Model`.
Phương thức lan truyền xuôi (`call`) cũng rất đơn giản:
nó liên kết từng mô-đun trong danh sách lại với nhau,
truyền đầu ra của mỗi mô-đun làm đầu vào cho mô-đun tiếp theo.
Lưu ý rằng cho đến nay, chúng ta vẫn gọi các mô hình
thông qua cấu trúc `net(X)` để nhận đầu ra.
Đây thực chất chỉ là cách viết tắt cho `net.call(X)`,
một thủ thuật Python tinh tế được thực hiện thông qua
phương thức `__call__` của lớp mô-đun.
:end_tab:

## [**Một mô-đun tùy chỉnh**]

Có lẽ cách dễ nhất để phát triển trực giác
về cách một mô-đun hoạt động
là tự triển khai một mô-đun.
Trước khi làm điều đó,
chúng ta tóm tắt ngắn gọn chức năng cơ bản
mà mỗi mô-đun phải cung cấp:


1. Nhận dữ liệu đầu vào làm đối số cho phương thức lan truyền xuôi.
1. Tạo đầu ra bằng cách để phương thức lan truyền xuôi trả về một giá trị. Lưu ý rằng đầu ra có thể có hình dạng khác với đầu vào. Ví dụ, tầng kết nối đầy đủ đầu tiên trong mô hình ở trên nhận đầu vào có số chiều tùy ý nhưng trả về đầu ra có số chiều 256.
1. Tính gradient của đầu ra theo đầu vào, có thể được truy cập qua phương thức lan truyền ngược. Thông thường, việc này diễn ra tự động.
1. Lưu trữ và cung cấp quyền truy cập vào các tham số cần thiết
   để thực hiện phép tính lan truyền xuôi.
1. Khởi tạo các tham số mô hình khi cần.


Trong đoạn mã sau,
chúng ta lập trình một mô-đun từ đầu
tương ứng với một MLP
có một tầng ẩn với 256 đơn vị ẩn,
và một tầng đầu ra 10 chiều.
Lưu ý rằng lớp `MLP` bên dưới kế thừa lớp biểu diễn một mô-đun.
Chúng ta sẽ phụ thuộc nhiều vào các phương thức của lớp cha,
chỉ cung cấp constructor riêng (phương thức `__init__` trong Python) và phương thức lan truyền xuôi.

```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self):
        # Call the constructor of the MLP parent class nn.Block to perform
        # the necessary initialization
        super().__init__()
        self.hidden = nn.Dense(256, activation='relu')
        self.out = nn.Dense(10)

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        # Call the constructor of the parent class nn.Module to perform
        # the necessary initialization
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        # Call the constructor of the parent class tf.keras.Model to perform
        # the necessary initialization
        super().__init__()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def call(self, X):
        return self.out(self.hidden((X)))
```

```{.python .input}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        # Define the layers
        self.hidden = nn.Dense(256)
        self.out = nn.Dense(10)

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def __call__(self, X):
        return self.out(nn.relu(self.hidden(X)))
```

Trước tiên, hãy tập trung vào phương thức lan truyền xuôi.
Lưu ý rằng nó nhận `X` làm đầu vào,
tính biểu diễn ẩn
có áp dụng hàm kích hoạt,
và xuất ra các logits của nó.
Trong phần triển khai `MLP` này,
cả hai tầng đều là biến thực thể.
Để thấy vì sao điều này hợp lý, hãy tưởng tượng
khởi tạo hai MLP, `net1` và `net2`,
và huấn luyện chúng trên các dữ liệu khác nhau.
Một cách tự nhiên, chúng ta sẽ mong đợi chúng
biểu diễn hai mô hình đã học khác nhau.

Chúng ta [**khởi tạo các layer của MLP**]
trong constructor
(**và sau đó gọi các layer này**)
trong mỗi lần gọi phương thức forward propagation.
Hãy lưu ý một vài chi tiết quan trọng.
Trước hết, phương thức `__init__` tùy chỉnh của chúng ta
gọi phương thức `__init__` của lớp cha
thông qua `super().__init__()`,
giúp chúng ta không phải vất vả viết lại
mã boilerplate áp dụng cho hầu hết các module.
Sau đó, chúng ta khởi tạo hai layer fully connected,
gán chúng cho `self.hidden` và `self.out`.
Lưu ý rằng trừ khi chúng ta triển khai một layer mới,
chúng ta không cần lo về phương thức backpropagation
hay việc khởi tạo parameter.
Hệ thống sẽ tự động tạo các phương thức này.
Hãy thử điều này.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
net = MLP()
if tab.selected('mxnet'):
    net.initialize()
net(X).shape
```

```{.python .input}
%%tab jax
net = MLP()
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

Một ưu điểm quan trọng của abstraction module là tính linh hoạt.
Chúng ta có thể kế thừa một module để tạo các layer
(chẳng hạn như lớp layer fully connected),
toàn bộ model (chẳng hạn như lớp `MLP` ở trên),
hoặc các thành phần có độ phức tạp trung gian khác nhau.
Chúng ta sẽ khai thác tính linh hoạt này
xuyên suốt các chương tiếp theo,
chẳng hạn khi đề cập đến
convolutional neural networks.


## [**Module Sequential**]
:label:`subsec_model-construction-sequential`

Bây giờ, chúng ta có thể xem xét kỹ hơn
cách lớp `Sequential` hoạt động.
Hãy nhớ rằng `Sequential` được thiết kế
để nối chuỗi các module khác lại với nhau.
Để xây dựng `MySequential` đơn giản hóa của riêng mình,
chúng ta chỉ cần định nghĩa hai phương thức chính:

1. Một phương thức để lần lượt thêm từng module vào một danh sách.
1. Một phương thức forward propagation để truyền input qua chuỗi module, theo đúng thứ tự chúng được thêm vào.

Lớp `MySequential` sau đây cung cấp cùng
chức năng như lớp `Sequential` mặc định.

```{.python .input}
%%tab mxnet
class MySequential(nn.Block):
    def add(self, block):
        # Here, block is an instance of a Block subclass, and we assume that
        # it has a unique name. We save it in the member variable _children of
        # the Block class, and its type is OrderedDict. When the MySequential
        # instance calls the initialize method, the system automatically
        # initializes all members of _children
        self._children[block.name] = block

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._children.values():
            X = block(X)
        return X
```

```{.python .input}
%%tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, X):
        for module in self.children():            
            X = module(X)
        return X
```

```{.python .input}
%%tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = args

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

```{.python .input}
%%tab jax
class MySequential(nn.Module):
    modules: List

    def __call__(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

:begin_tab:`mxnet`
Phương thức `add` thêm một block
vào ordered dictionary `_children`.
Bạn có thể tự hỏi tại sao mọi `Block` của Gluon
đều có thuộc tính `_children`
và tại sao chúng ta dùng nó thay vì chỉ
tự định nghĩa một Python list.
Tóm lại, ưu điểm chủ yếu của `_children`
là trong quá trình khởi tạo parameter của block,
Gluon biết cần tìm bên trong dictionary `_children`
để tìm các sub-block có
parameter cũng cần được khởi tạo.
:end_tab:

:begin_tab:`pytorch`
Trong phương thức `__init__`, chúng ta thêm từng module
bằng cách gọi phương thức `add_modules`. Các module này sau đó có thể được truy cập bằng phương thức `children`.
Theo cách này, hệ thống biết các module đã được thêm vào,
và sẽ khởi tạo đúng parameter của từng module.
:end_tab:

Khi phương thức forward propagation của `MySequential` được gọi,
mỗi module đã thêm sẽ được thực thi
theo thứ tự chúng được thêm vào.
Bây giờ, chúng ta có thể triển khai lại một MLP
bằng lớp `MySequential` của mình.

```{.python .input}
%%tab mxnet
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X).shape
```

```{.python .input}
%%tab pytorch
net = MySequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X).shape
```

```{.python .input}
%%tab jax
net = MySequential([nn.Dense(256), nn.relu, nn.Dense(10)])
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

Lưu ý rằng cách sử dụng `MySequential` này
giống hệt với mã mà trước đây chúng ta đã viết
cho lớp `Sequential`
(như được mô tả trong :numref:`sec_mlp`).


## [**Thực thi Code trong Phương thức Forward Propagation**]

Lớp `Sequential` giúp việc xây dựng model trở nên dễ dàng,
cho phép chúng ta lắp ghép các kiến trúc mới
mà không cần định nghĩa lớp riêng.
Tuy nhiên, không phải mọi kiến trúc đều là chuỗi nối đơn giản.
Khi cần linh hoạt hơn,
chúng ta sẽ muốn định nghĩa các block riêng.
Ví dụ, chúng ta có thể muốn thực thi
control flow của Python trong phương thức forward propagation.
Hơn nữa, chúng ta có thể muốn thực hiện
các phép toán học tùy ý,
thay vì chỉ dựa vào các neural network layer được định nghĩa sẵn.

Bạn có thể đã nhận thấy rằng cho đến giờ,
mọi phép toán trong network của chúng ta
đều tác động lên activation
và parameter của network.
Tuy nhiên, đôi khi chúng ta có thể muốn
đưa vào các hạng tử
không phải là kết quả của các layer trước đó
cũng không phải là parameter có thể cập nhật.
Chúng ta gọi chúng là *constant parameters*.
Ví dụ, giả sử chúng ta muốn một layer
tính hàm
$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$,
trong đó $\mathbf{x}$ là input, $\mathbf{w}$ là parameter của chúng ta,
và $c$ là một hằng số xác định
không được cập nhật trong quá trình optimization.
Vì vậy, chúng ta triển khai một lớp `FixedHiddenMLP` như sau.

```{.python .input}
%%tab mxnet
class FixedHiddenMLP(nn.Block):
    def __init__(self):
        super().__init__()
        # Random weight parameters created with the get_constant method
        # are not updated during training (i.e., constant parameters)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # Use the created constant parameters, as well as the relu and dot
        # functions
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.dense(X)
        # Control flow
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
%%tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20))
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)        
        X = F.relu(X @ self.rand_weight + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
%%tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Random weight parameters created with tf.constant are not updated
        # during training (i.e., constant parameters)
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # Use the created constant parameters, as well as the relu and
        # matmul functions
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.dense(X)
        # Control flow
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

```{.python .input}
%%tab jax
class FixedHiddenMLP(nn.Module):
    # Random weight parameters that will not compute gradients and
    # therefore keep constant during training
    rand_weight: jnp.array = jax.random.uniform(d2l.get_key(), (20, 20))

    def setup(self):
        self.dense = nn.Dense(20)

    def __call__(self, X):
        X = self.dense(X)
        X = nn.relu(X @ self.rand_weight + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.dense(X)
        # Control flow
        while jnp.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

Trong model này,
chúng ta triển khai một hidden layer có trọng số
(`self.rand_weight`) được khởi tạo ngẫu nhiên
khi khởi tạo và sau đó giữ nguyên không đổi.
Trọng số này không phải là model parameter
và do đó không bao giờ được cập nhật bởi backpropagation.
Sau đó, network truyền output của layer "cố định" này
qua một layer fully connected.

Lưu ý rằng trước khi trả về output,
model của chúng ta đã thực hiện một điều bất thường.
Chúng ta chạy một while-loop, kiểm tra
điều kiện rằng chuẩn $\ell_1$ của nó lớn hơn $1$,
và chia output vector cho $2$
cho đến khi nó thỏa mãn điều kiện.
Cuối cùng, chúng ta trả về tổng các phần tử trong `X`.
Theo hiểu biết của chúng tôi, không có neural network tiêu chuẩn nào
thực hiện phép toán này.
Lưu ý rằng phép toán cụ thể này có thể không hữu ích
trong bất kỳ tác vụ thực tế nào.
Mục đích của chúng tôi chỉ là cho bạn thấy cách tích hợp
code tùy ý vào luồng
tính toán của neural network.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
net = FixedHiddenMLP()
if tab.selected('mxnet'):
    net.initialize()
net(X)
```

```{.python .input}
%%tab jax
net = FixedHiddenMLP()
params = net.init(d2l.get_key(), X)
net.apply(params, X)
```

Chúng ta có thể [**kết hợp linh hoạt nhiều
cách lắp ghép module với nhau.**]
Trong ví dụ sau, chúng ta lồng các module
theo một số cách sáng tạo.

```{.python .input}
%%tab mxnet
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())
chimera.initialize()
chimera(X)
```

```{.python .input}
%%tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
                                 nn.LazyLinear(32), nn.ReLU())
        self.linear = nn.LazyLinear(16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.LazyLinear(20), FixedHiddenMLP())
chimera(X)
```

```{.python .input}
%%tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)
```

```{.python .input}
%%tab jax
class NestMLP(nn.Module):
    def setup(self):
        self.net = nn.Sequential([nn.Dense(64), nn.relu,
                                  nn.Dense(32), nn.relu])
        self.dense = nn.Dense(16)

    def __call__(self, X):
        return self.dense(self.net(X))


chimera = nn.Sequential([NestMLP(), nn.Dense(20), FixedHiddenMLP()])
params = chimera.init(d2l.get_key(), X)
chimera.apply(params, X)
```

## Tóm tắt

Các layer riêng lẻ có thể là module.
Nhiều layer có thể tạo thành một module.
Nhiều module có thể tạo thành một module.

Một module có thể chứa code.
Module xử lý nhiều công việc quản lý, bao gồm khởi tạo parameter và backpropagation.
Các chuỗi kết hợp tuần tự của layer và module được xử lý bởi module `Sequential`.


## Bài tập

1. Những loại vấn đề nào sẽ xảy ra nếu bạn thay đổi `MySequential` để lưu trữ module trong một Python list?
1. Hãy triển khai một module nhận hai module làm đối số, chẳng hạn `net1` và `net2`, rồi trả về output được nối của cả hai network trong forward propagation. Đây cũng được gọi là một *parallel module*.
1. Giả sử bạn muốn nối nhiều instance của cùng một network. Hãy triển khai một factory function tạo ra nhiều instance của cùng một module và xây dựng một network lớn hơn từ chúng.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/264)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17989)
:end_tab: