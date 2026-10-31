```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Thiết kế hướng đối tượng cho việc triển khai
:label:`sec_oo-design`

Trong phần giới thiệu về hồi quy tuyến tính,
chúng ta đã lần lượt tìm hiểu nhiều thành phần khác nhau,
bao gồm
dữ liệu, mô hình, hàm mất mát
và thuật toán tối ưu hóa.
Thực vậy,
hồi quy tuyến tính là
một trong những mô hình machine learning đơn giản nhất.
Tuy nhiên, việc huấn luyện mô hình này
sử dụng nhiều thành phần giống với những thành phần mà các mô hình khác trong cuốn sách này yêu cầu.
Do đó,
trước khi đi sâu vào các chi tiết triển khai,
chúng ta nên
thiết kế một số API
được sử dụng xuyên suốt tài liệu.
Bằng cách xem các thành phần trong deep learning
như các đối tượng,
chúng ta có thể bắt đầu
định nghĩa các class cho những đối tượng này
và các tương tác giữa chúng.
Thiết kế hướng đối tượng
này cho việc triển khai
sẽ giúp
đơn giản hóa đáng kể phần trình bày, và bạn thậm chí có thể muốn sử dụng nó trong các dự án của mình.


Lấy cảm hứng từ các thư viện mã nguồn mở như [PyTorch Lightning](https://www.pytorchlightning.ai/),
ở mức độ khái quát,
chúng ta muốn có ba class:
(i) `Module` chứa các mô hình, hàm mất mát và phương thức tối ưu hóa;
(ii) `DataModule` cung cấp các data loader cho việc huấn luyện và validation;
(iii) cả hai class được kết hợp bằng class `Trainer`, cho phép chúng ta
huấn luyện các mô hình trên nhiều nền tảng phần cứng khác nhau.
Phần lớn code trong cuốn sách này điều chỉnh từ `Module` và `DataModule`. Chúng ta sẽ chỉ đề cập đến class `Trainer` khi thảo luận về GPU, CPU, huấn luyện song song và các thuật toán tối ưu hóa.

```{.python .input}
%%tab mxnet
import time
import numpy as np
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
import time
import numpy as np
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import time
import numpy as np
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from dataclasses import field
from d2l import jax as d2l
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
import numpy as np
import jax
import time
from typing import Any
```

## Các tiện ích
:label:`oo-design-utilities`

Chúng ta cần một vài tiện ích để đơn giản hóa việc lập trình hướng đối tượng trong Jupyter notebook. Một trong những thách thức là các định nghĩa class thường là những khối code khá dài. Tính dễ đọc của notebook đòi hỏi các đoạn code ngắn, xen kẽ với phần giải thích, một yêu cầu không tương thích với phong cách lập trình phổ biến trong các thư viện Python. Hàm tiện ích đầu tiên cho phép chúng ta đăng ký các hàm làm method trong một class *sau khi* class đã được tạo. Thực tế, chúng ta có thể làm điều đó *ngay cả sau khi* đã tạo các instance của class! Hàm này cho phép chúng ta chia phần triển khai của một class thành nhiều khối code.

```{.python .input}
%%tab all
def add_to_class(Class):  #@save
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

Hãy xem nhanh cách sử dụng hàm này. Chúng ta dự định triển khai một class `A` với một method `do`. Thay vì đặt code cho cả `A` và `do` trong cùng một khối code, trước tiên chúng ta có thể khai báo class `A` và tạo một instance `a`.

```{.python .input}
%%tab all
class A:
    def __init__(self):
        self.b = 1

a = A()
```

Tiếp theo, chúng ta định nghĩa method `do` như bình thường, nhưng không nằm trong phạm vi của class `A`. Thay vào đó, chúng ta decorate method này bằng `add_to_class`, với class `A` làm đối số. Nhờ đó, method có thể truy cập các biến thành viên của `A` đúng như khi nó được định nghĩa bên trong `A`. Hãy xem điều gì xảy ra khi chúng ta gọi method này cho instance `a`.

```{.python .input}
%%tab all
@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

a.do()
```

Tiện ích thứ hai là một class lưu tất cả các đối số trong method `__init__` của một class dưới dạng các thuộc tính của class. Điều này cho phép chúng ta mở rộng ngầm chữ ký lời gọi constructor mà không cần thêm code.

```{.python .input}
%%tab all
class HyperParameters:  #@save
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
```

Chúng ta tạm hoãn phần triển khai của tiện ích này đến :numref:`sec_utils`. Để sử dụng, chúng ta định nghĩa một class kế thừa từ `HyperParameters` và gọi `save_hyperparameters` trong method `__init__`.

```{.python .input}
%%tab all
# Call the fully implemented HyperParameters class saved in d2l
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
```

Tiện ích cuối cùng cho phép chúng ta vẽ tiến trình của thí nghiệm một cách tương tác trong khi thí nghiệm đang diễn ra. Để phân biệt với [TensorBoard](https://www.tensorflow.org/tensorboard) mạnh mẽ hơn nhiều (và cũng phức tạp hơn), chúng ta đặt tên cho tiện ích này là `ProgressBoard`. Phần triển khai được tạm hoãn đến :numref:`sec_utils`. Hiện tại, hãy xem nó hoạt động như thế nào.

Method `draw` vẽ một điểm `(x, y)` trên hình, với `label` được chỉ định trong chú giải. Đối số tùy chọn `every_n` làm mượt đường bằng cách chỉ hiển thị $1/n$ điểm trên hình. Giá trị của chúng được tính trung bình từ $n$ điểm lân cận trong hình ban đầu.

```{.python .input}
%%tab all
class ProgressBoard(d2l.HyperParameters):  #@save
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented
```

Trong ví dụ sau, chúng ta vẽ `sin` và `cos` với độ mượt khác nhau. Nếu chạy khối code này, bạn sẽ thấy các đường phát triển dưới dạng animation.

```{.python .input}
%%tab all
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```

## Các mô hình
:label:`subsec_oo-design-models`

Class `Module` là class cơ sở của tất cả các mô hình mà chúng ta sẽ triển khai. Tối thiểu, chúng ta cần ba method. Method đầu tiên, `__init__`, lưu trữ các tham số có thể học; method `training_step` nhận một batch dữ liệu để trả về giá trị loss; và cuối cùng, `configure_optimizers` trả về phương thức tối ưu hóa, hoặc một danh sách các phương thức tối ưu hóa, được sử dụng để cập nhật các tham số có thể học. Ngoài ra, chúng ta có thể định nghĩa `validation_step` để báo cáo các thước đo đánh giá.
Đôi khi, chúng ta đặt code tính toán đầu ra vào một method `forward` riêng để có thể tái sử dụng dễ dàng hơn.

:begin_tab:`jax`
Với sự ra đời của [dataclasses](https://docs.python.org/3/library/dataclasses.html)
trong Python 3.7, các class được decorate bằng `@dataclass` sẽ tự động bổ sung
các method đặc biệt như `__init__` và `__repr__`. Các biến thành viên được định nghĩa
bằng type annotation. Tất cả module Flax đều là dataclass của Python 3.7.
:end_tab:

```{.python .input}
%%tab pytorch
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

```{.python .input}
%%tab mxnet, tensorflow, jax
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    if tab.selected('mxnet', 'tensorflow'):
        def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
            super().__init__()
            self.save_hyperparameters()
            self.board = ProgressBoard()
        if tab.selected('tensorflow'):
            self.training = None

    if tab.selected('jax'):
        # No need for save_hyperparam when using Python dataclass
        plot_train_per_epoch: int = field(default=2, init=False)
        plot_valid_per_epoch: int = field(default=1, init=False)
        # Use default_factory to make sure new plots are generated on each run
        board: ProgressBoard = field(default_factory=lambda: ProgressBoard(),
                                     init=False)

    def loss(self, y_hat, y):
        raise NotImplementedError

    if tab.selected('mxnet', 'tensorflow'):
        def forward(self, X):
            assert hasattr(self, 'net'), 'Neural network is defined'
            return self.net(X)

    if tab.selected('tensorflow'):
        def call(self, X, *args, **kwargs):
            if kwargs and "training" in kwargs:
                self.training = kwargs['training']
            return self.forward(X, *args)

    if tab.selected('jax'):
        # JAX & Flax do not have a forward-method-like syntax. Flax uses setup
        # and built-in __call__ magic methods for forward pass. Adding here
        # for consistency
        def forward(self, X, *args, **kwargs):
            assert hasattr(self, 'net'), 'Neural network is defined'
            return self.net(X, *args, **kwargs)

        def __call__(self, X, *args, **kwargs):
            return self.forward(X, *args, **kwargs)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        if tab.selected('mxnet', 'tensorflow'):
            self.board.draw(x, d2l.numpy(value), (
                'train_' if train else 'val_') + key, every_n=int(n))
        if tab.selected('jax'):
            self.board.draw(x, d2l.to(value, d2l.cpu()),
                            ('train_' if train else 'val_') + key,
                            every_n=int(n))

    if tab.selected('mxnet', 'tensorflow'):
        def training_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=True)
            return l

        def validation_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=False)

    if tab.selected('jax'):
        def training_step(self, params, batch, state):
            l, grads = jax.value_and_grad(self.loss)(params, batch[:-1],
                                                     batch[-1], state)
            self.plot("loss", l, train=True)
            return l, grads

        def validation_step(self, params, batch, state):
            l = self.loss(params, batch[:-1], batch[-1], state)
            self.plot('loss', l, train=False)
        
        def apply_init(self, dummy_input, key):
            """To be defined later in :numref:`sec_lazy_init`"""
            raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
```

:begin_tab:`mxnet`
Bạn có thể nhận thấy rằng `Module` là một subclass của `nn.Block`, class cơ sở của các mạng neural trong Gluon.
Nó cung cấp các tính năng thuận tiện để xử lý mạng neural. Ví dụ, nếu chúng ta định nghĩa một method `forward`, chẳng hạn như `forward(self, X)`, thì với một instance `a`, chúng ta có thể gọi method này bằng `a(X)`. Điều này hoạt động vì nó gọi method `forward` trong method `__call__` tích hợp sẵn. Bạn có thể tìm thêm chi tiết và ví dụ về `nn.Block` trong :numref:`sec_model_construction`.
:end_tab:

:begin_tab:`pytorch`
Bạn có thể nhận thấy rằng `Module` là một subclass của `nn.Module`, class cơ sở của các mạng neural trong PyTorch.
Nó cung cấp các tính năng thuận tiện để xử lý mạng neural. Ví dụ, nếu chúng ta định nghĩa một method `forward`, chẳng hạn như `forward(self, X)`, thì với một instance `a`, chúng ta có thể gọi method này bằng `a(X)`. Điều này hoạt động vì nó gọi method `forward` trong method `__call__` tích hợp sẵn. Bạn có thể tìm thêm chi tiết và ví dụ về `nn.Module` trong :numref:`sec_model_construction`.
:end_tab:

:begin_tab:`tensorflow`
Bạn có thể nhận thấy rằng `Module` là một subclass của `tf.keras.Model`, class cơ sở của các mạng neural trong TensorFlow.
Nó cung cấp các tính năng thuận tiện để xử lý mạng neural. Ví dụ, nó gọi method `call` trong method `__call__` tích hợp sẵn. Ở đây, chúng ta chuyển hướng `call` đến method `forward`, đồng thời lưu các đối số của nó dưới dạng một thuộc tính của class. Chúng ta làm vậy để code của mình tương tự hơn với cách triển khai của các framework khác.
:end_tab:

:begin_tab:`jax`
Bạn có thể nhận thấy rằng `Module` là một subclass của `linen.Module`, class cơ sở của các mạng neural trong Flax.
Nó cung cấp các tính năng thuận tiện để xử lý mạng neural. Ví dụ, nó xử lý các tham số của mô hình, cung cấp decorator `nn.compact` để đơn giản hóa code, gọi method `__call__` cùng nhiều chức năng khác.
Ở đây, chúng ta cũng chuyển hướng `__call__` đến method `forward`. Chúng ta làm vậy để code của mình tương tự hơn với cách triển khai của các framework khác.
:end_tab:

##  Dữ liệu
:label:`oo-design-data`

Class `DataModule` là class cơ sở cho dữ liệu. Rất thường xuyên, method `__init__` được sử dụng để chuẩn bị dữ liệu. Việc này bao gồm tải xuống và tiền xử lý nếu cần. Method `train_dataloader` trả về data loader cho tập dữ liệu huấn luyện. Data loader là một generator (Python) tạo ra một batch dữ liệu mỗi khi được sử dụng. Batch này sau đó được truyền vào method `training_step` của `Module` để tính loss. Có thể tùy chọn `val_dataloader` để trả về data loader của tập validation. Nó hoạt động tương tự, ngoại trừ việc tạo ra các batch dữ liệu cho method `validation_step` trong `Module`.

```{.python .input}
%%tab all
class DataModule(d2l.HyperParameters):  #@save
    """The base class of data."""
    if tab.selected('mxnet', 'pytorch'):
        def __init__(self, root='../data', num_workers=4):
            self.save_hyperparameters()

    if tab.selected('tensorflow', 'jax'):
        def __init__(self, root='../data'):
            self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```

## Huấn luyện
:label:`oo-design-training`

:begin_tab:`pytorch, mxnet, tensorflow`
Class `Trainer` huấn luyện các tham số có thể học trong class `Module` bằng dữ liệu được chỉ định trong `DataModule`. Method chính là `fit`, nhận hai đối số: `model`, một instance của `Module`, và `data`, một instance của `DataModule`. Sau đó, method lặp qua toàn bộ tập dữ liệu `max_epochs` lần để huấn luyện mô hình. Như trước đây, chúng ta sẽ tạm hoãn phần triển khai method này đến các chương sau.
:end_tab:

:begin_tab:`jax`
Lớp `Trainer` huấn luyện các tham số có thể học được `params` bằng dữ liệu được chỉ định trong `DataModule`. Phương thức chính là `fit`, nhận ba đối số: `model`, một thực thể của `Module`, `data`, một thực thể của `DataModule`, và `key`, một `PRNGKeyArray` của JAX. Ở đây, chúng ta đặt đối số `key` là tùy chọn để đơn giản hóa interface, nhưng bạn nên luôn truyền đối số này và khởi tạo các tham số mô hình bằng một root key trong JAX và Flax. Sau đó, phương thức lặp qua toàn bộ dataset `max_epochs` lần để huấn luyện mô hình. Như trước đây, chúng ta sẽ trì hoãn việc triển khai phương thức này đến các chương sau.
:end_tab:

```{.python .input}
%%tab all
class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        def fit(self, model, data):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    if tab.selected('jax'):
        def fit(self, model, data, key=None):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()

            if key is None:
                root_key = d2l.get_key()
            else:
                root_key = key
            params_key, dropout_key = jax.random.split(root_key)
            key = {'params': params_key, 'dropout': dropout_key}

            dummy_input = next(iter(self.train_dataloader))[:-1]
            variables = model.apply_init(dummy_input, key=key)
            params = variables['params']

            if 'batch_stats' in variables.keys():
                # Here batch_stats will be used later (e.g., for batch norm)
                batch_stats = variables['batch_stats']
            else:
                batch_stats = {}

            # Flax uses optax under the hood for a single state obj TrainState.
            # More will be discussed later in the dropout and batch
            # normalization section
            class TrainState(train_state.TrainState):
                batch_stats: Any
                dropout_rng: jax.random.PRNGKeyArray

            self.state = TrainState.create(apply_fn=model.apply,
                                           params=params,
                                           batch_stats=batch_stats,
                                           dropout_rng=dropout_key,
                                           tx=model.configure_optimizers())
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
```

## Tóm tắt

Để làm nổi bật thiết kế hướng đối tượng
cho phần triển khai deep learning trong tương lai,
các lớp trên chỉ đơn giản minh họa cách các đối tượng của chúng
lưu trữ dữ liệu và tương tác với nhau.
Chúng ta sẽ tiếp tục bổ sung các triển khai cho những lớp này,
chẳng hạn như thông qua `@add_to_class`,
trong phần còn lại của cuốn sách.
Ngoài ra,
các lớp được triển khai đầy đủ này
được lưu trong thư viện [D2L](https://github.com/d2l-ai/d2l-en/tree/master/d2l),
một *bộ công cụ gọn nhẹ* giúp việc xây dựng mô hình có cấu trúc cho deep learning trở nên dễ dàng.
Cụ thể, thư viện này hỗ trợ tái sử dụng nhiều component giữa các dự án mà hầu như không cần thay đổi gì. Chẳng hạn, chúng ta có thể chỉ thay optimizer, chỉ thay model, chỉ thay dataset, v.v.;
mức độ modularity này mang lại lợi ích xuyên suốt cuốn sách về tính ngắn gọn và đơn giản (đó là lý do chúng tôi bổ sung nó), đồng thời cũng có thể mang lại lợi ích tương tự cho các dự án của riêng bạn.


## Bài tập

1. Tìm các triển khai đầy đủ của những lớp trên được lưu trong thư viện [D2L](https://github.com/d2l-ai/d2l-en/tree/master/d2l). Chúng tôi đặc biệt khuyến nghị bạn xem xét chi tiết phần triển khai sau khi đã làm quen hơn với việc xây dựng mô hình deep learning.
1. Xóa câu lệnh `save_hyperparameters` trong lớp `B`. Bạn vẫn có thể in `self.a` và `self.b` không? Tùy chọn: nếu đã tìm hiểu sâu hơn về phần triển khai đầy đủ của lớp `HyperParameters`, bạn có thể giải thích tại sao không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/6645)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/6646)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/6647)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17974)
:end_tab: