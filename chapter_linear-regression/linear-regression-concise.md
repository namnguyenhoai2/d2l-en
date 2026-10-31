```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Triển khai hồi quy tuyến tính ngắn gọn
:label:`sec_linear_concise`

Trong thập kỷ qua, deep learning đã chứng kiến một dạng bùng nổ kiểu Cambrian.
Số lượng khổng lồ các kỹ thuật, ứng dụng và thuật toán đã vượt xa
tiến bộ của những thập kỷ trước.
Điều này là nhờ sự kết hợp tình cờ của nhiều yếu tố,
một trong số đó là các công cụ miễn phí mạnh mẽ
do một số framework deep learning mã nguồn mở cung cấp.
Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010`,
DistBelief :cite:`Dean.Corrado.Monga.ea.2012`,
và Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014`
có thể được xem là
thế hệ đầu tiên của các mô hình như vậy
được áp dụng rộng rãi.
Trái ngược với những công trình nền tảng trước đây
như SN2 (Simulateur Neuristique) :cite:`Bottou.Le-Cun.1988`,
vốn cung cấp trải nghiệm lập trình tương tự Lisp,
các framework hiện đại cung cấp automatic differentiation
và sự tiện lợi của Python.
Những framework này cho phép chúng ta tự động hóa và mô-đun hóa
công việc lặp đi lặp lại khi triển khai các thuật toán học dựa trên gradient.

Trong :numref:`sec_linear_scratch`, chúng ta chỉ dựa vào
(i) tensor để lưu trữ dữ liệu và thực hiện đại số tuyến tính;
và (ii) automatic differentiation để tính gradient.
Trên thực tế, vì data iterator, hàm loss, optimizer
và các layer của neural network
rất phổ biến, các thư viện hiện đại cũng triển khai sẵn những thành phần này.
Trong phần này, (**chúng tôi sẽ hướng dẫn bạn cách triển khai
mô hình hồi quy tuyến tính**) từ :numref:`sec_linear_scratch`
(**một cách ngắn gọn bằng cách sử dụng các API cấp cao**) của các framework deep learning.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## Định nghĩa mô hình

Khi triển khai hồi quy tuyến tính từ đầu
trong :numref:`sec_linear_scratch`,
chúng ta định nghĩa tường minh các tham số của mô hình
và viết mã để tạo đầu ra
bằng các phép toán đại số tuyến tính cơ bản.
Bạn *nên* biết cách thực hiện việc này.
Nhưng khi mô hình của bạn trở nên phức tạp hơn,
và khi bạn phải làm việc này gần như mỗi ngày,
bạn sẽ thấy sự hỗ trợ đó rất hữu ích.
Tình huống này tương tự như việc tự viết một blog từ đầu.
Thực hiện một hoặc hai lần thì vừa bổ ích vừa mang tính hướng dẫn,
nhưng bạn sẽ là một web developer kém
nếu dành cả tháng để phát minh lại những thứ đã có sẵn.

Đối với các phép toán tiêu chuẩn,
chúng ta có thể [**sử dụng các layer được định nghĩa sẵn trong framework,**]
nhờ đó tập trung vào các layer dùng để xây dựng mô hình
thay vì phải lo lắng về cách triển khai chúng.
Hãy nhớ lại kiến trúc của một mạng một layer
được mô tả trong :numref:`fig_single_neuron`.
Layer này được gọi là *fully connected*,
vì mỗi đầu vào của nó được kết nối
với từng đầu ra
thông qua phép nhân matrix--vector.

:begin_tab:`mxnet`
Trong Gluon, layer fully connected được định nghĩa trong class `Dense`.
Vì chúng ta chỉ muốn tạo ra một đầu ra vô hướng duy nhất,
nên đặt số đó bằng 1.
Đáng chú ý là để thuận tiện,
Gluon không yêu cầu chúng ta chỉ định
input shape cho từng layer.
Do đó, chúng ta không cần cho Gluon biết
có bao nhiêu đầu vào đi vào linear layer này.
Khi lần đầu truyền dữ liệu qua mô hình,
chẳng hạn khi thực thi `net(X)` sau đây,
Gluon sẽ tự động suy luận số lượng đầu vào của mỗi layer và
do đó khởi tạo mô hình chính xác.
Chúng ta sẽ mô tả chi tiết hơn cách thức này hoạt động ở phần sau.
:end_tab:

:begin_tab:`pytorch`
Trong PyTorch, layer fully connected được định nghĩa trong các class `Linear` và `LazyLinear` (có từ phiên bản 1.8.0).
Class sau
cho phép người dùng chỉ định *chỉ*
chiều đầu ra,
trong khi class trước
còn yêu cầu
số lượng đầu vào đi vào layer này.
Việc chỉ định input shape khá bất tiện và có thể đòi hỏi các phép tính không tầm thường
(chẳng hạn như trong các convolutional layer).
Vì vậy, để đơn giản, chúng ta sẽ sử dụng các layer "lazy" như vậy
bất cứ khi nào có thể.
:end_tab:

:begin_tab:`tensorflow`
Trong Keras, layer fully connected được định nghĩa trong class `Dense`.
Vì chúng ta chỉ muốn tạo ra một đầu ra vô hướng duy nhất,
nên đặt số đó bằng 1.
Đáng chú ý là để thuận tiện,
Keras không yêu cầu chúng ta chỉ định
input shape cho từng layer.
Chúng ta không cần cho Keras biết
có bao nhiêu đầu vào đi vào linear layer này.
Khi lần đầu thử truyền dữ liệu qua mô hình,
chẳng hạn khi thực thi `net(X)` sau đây,
Keras sẽ tự động suy luận
số lượng đầu vào của mỗi layer.
Chúng ta sẽ mô tả chi tiết hơn cách thức này hoạt động ở phần sau.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(1)
            self.net.initialize(init.Normal(sigma=0.01))
        if tab.selected('tensorflow'):
            initializer = tf.initializers.RandomNormal(stddev=0.01)
            self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        if tab.selected('pytorch'):
            self.net = nn.LazyLinear(1)
            self.net.weight.data.normal_(0, 0.01)
            self.net.bias.data.fill_(0)
```

```{.python .input}
%%tab jax
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    lr: float

    def setup(self):
        self.net = nn.Dense(1, kernel_init=nn.initializers.normal(0.01))
```

Trong method `forward`, chúng ta chỉ cần gọi method tích hợp sẵn `__call__` của các layer được định nghĩa trước để tính đầu ra.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

## Định nghĩa hàm loss

:begin_tab:`mxnet`
Module `loss` định nghĩa nhiều hàm loss hữu ích.
Để tăng tốc và thuận tiện, chúng ta không tự triển khai
mà thay vào đó chọn `loss.L2Loss` tích hợp sẵn.
Vì `loss` mà nó trả về là
sai số bình phương cho từng mẫu,
chúng ta sử dụng `mean` để lấy trung bình loss trên minibatch.
:end_tab:

:begin_tab:`pytorch`
[**Class `MSELoss` tính mean squared error (không có hệ số $1/2$ trong :eqref:`eq_mse`).**]
Theo mặc định, `MSELoss` trả về loss trung bình trên các mẫu.
Cách này nhanh hơn (và dễ sử dụng hơn) so với việc tự triển khai.
:end_tab:

:begin_tab:`tensorflow`
Class `MeanSquaredError` tính mean squared error (không có hệ số $1/2$ trong :eqref:`eq_mse`).
Theo mặc định, nó trả về loss trung bình trên các mẫu.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    if tab.selected('mxnet'):
        fn = gluon.loss.L2Loss()
        return fn(y_hat, y).mean()
    if tab.selected('pytorch'):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)
    return d2l.reduce_mean(optax.l2_loss(y_hat, y))
```

## Định nghĩa thuật toán tối ưu hóa

:begin_tab:`mxnet`
Minibatch SGD là một công cụ tiêu chuẩn
để tối ưu hóa neural network
và vì vậy Gluon hỗ trợ nó cùng với một số
biến thể của thuật toán này thông qua class `Trainer`.
Lưu ý rằng class `Trainer` của Gluon
đại diện cho thuật toán tối ưu hóa,
trong khi class `Trainer` mà chúng ta tạo trong :numref:`sec_oo-design`
chứa method training,
tức là liên tục gọi optimizer
để cập nhật các tham số của mô hình.
Khi khởi tạo `Trainer`,
chúng ta chỉ định các tham số cần tối ưu,
có thể lấy từ model `net` thông qua `net.collect_params()`,
thuật toán tối ưu hóa mong muốn sử dụng (`sgd`),
và một dictionary chứa các hyperparameter
mà thuật toán tối ưu hóa yêu cầu.
:end_tab:

:begin_tab:`pytorch`
Minibatch SGD là một công cụ tiêu chuẩn
để tối ưu hóa neural network
và vì vậy PyTorch hỗ trợ nó cùng với một số
biến thể của thuật toán này trong module `optim`.
Khi (**khởi tạo một instance của `SGD`,**)
chúng ta chỉ định các tham số cần tối ưu,
có thể lấy từ model thông qua `self.parameters()`,
và learning rate (`self.lr`)
mà thuật toán tối ưu hóa yêu cầu.
:end_tab:

:begin_tab:`tensorflow`
Minibatch SGD là một công cụ tiêu chuẩn
để tối ưu hóa neural network
và vì vậy Keras hỗ trợ nó cùng với một số
biến thể của thuật toán này trong module `optimizers`.
:end_tab:

```{.python .input}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet'):
        return gluon.Trainer(self.collect_params(),
                             'sgd', {'learning_rate': self.lr})
    if tab.selected('pytorch'):
        return torch.optim.SGD(self.parameters(), self.lr)
    if tab.selected('tensorflow'):
        return tf.keras.optimizers.SGD(self.lr)
    if tab.selected('jax'):
        return optax.sgd(self.lr)
```

## Training

Bạn có thể đã nhận thấy rằng việc biểu diễn mô hình bằng
các API cấp cao của một framework deep learning
đòi hỏi ít dòng mã hơn.
Chúng ta không phải cấp phát từng tham số,
định nghĩa hàm loss hoặc triển khai minibatch SGD.
Khi bắt đầu làm việc với các mô hình phức tạp hơn nhiều,
những ưu điểm của API cấp cao sẽ càng đáng kể.

Giờ đây, khi đã có đầy đủ các thành phần cơ bản,
[**vòng lặp training thực chất giống hệt
vòng lặp mà chúng ta đã triển khai từ đầu.**]
Vì vậy, chúng ta chỉ cần gọi method `fit` (được giới thiệu trong :numref:`oo-design-training`),
method này dựa trên việc triển khai method `fit_epoch`
trong :numref:`sec_linear_scratch`,
để training mô hình.

```{.python .input}
%%tab all
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Dưới đây, chúng ta
[**so sánh các tham số của mô hình được học
bằng cách training trên dữ liệu hữu hạn
với các tham số thực tế**]
đã tạo ra dataset của chúng ta.
Để truy cập các tham số,
chúng ta truy cập trọng số và bias
của layer cần dùng.
Giống như trong phần triển khai từ đầu,
hãy lưu ý rằng các tham số ước lượng của chúng ta
gần với các tham số thực tương ứng.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    if tab.selected('mxnet'):
        return (self.net.weight.data(), self.net.bias.data())
    if tab.selected('pytorch'):
        return (self.net.weight.data, self.net.bias.data)
    if tab.selected('tensorflow'):
        return (self.get_weights()[0], self.get_weights()[1])

w, b = model.get_w_b()
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self, state):
    net = state.params['net']
    return net['kernel'], net['bias']

w, b = model.get_w_b(trainer.state)
```

```{.python .input}
print(f'error in estimating w: {data.w - d2l.reshape(w, data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```

## Tóm tắt

Phần này chứa
lần triển khai đầu tiên của một deep network (trong cuốn sách này)
tận dụng những tiện ích do
các framework deep learning hiện đại cung cấp,
chẳng hạn như MXNet :cite:`Chen.Li.Li.ea.2015`,
JAX :cite:`Frostig.Johnson.Leary.2018`,
PyTorch :cite:`Paszke.Gross.Massa.ea.2019`,
và Tensorflow :cite:`Abadi.Barham.Chen.ea.2016`.
Chúng ta đã sử dụng các giá trị mặc định của framework để tải dữ liệu, định nghĩa một layer,
một hàm loss, một optimizer và một vòng lặp training.
Bất cứ khi nào framework cung cấp đầy đủ các tính năng cần thiết,
thông thường bạn nên sử dụng chúng,
vì các thành phần do thư viện triển khai
thường được tối ưu hóa mạnh về hiệu năng
và được kiểm thử kỹ lưỡng về độ tin cậy.
Đồng thời, hãy cố gắng không quên
rằng các module này *có thể* được triển khai trực tiếp.
Điều này đặc biệt quan trọng đối với các nhà nghiên cứu tiềm năng
muốn đi đầu trong việc phát triển mô hình,
nơi bạn sẽ phát minh ra các thành phần mới
mà không thể tồn tại trong bất kỳ thư viện hiện tại nào.

:begin_tab:`mxnet`
Trong Gluon, module `data` cung cấp các công cụ để xử lý dữ liệu,
module `nn` định nghĩa một số lượng lớn các layer mạng neural,
và module `loss` định nghĩa nhiều hàm loss phổ biến.
Ngoài ra, `initializer` cho phép truy cập
nhiều lựa chọn để khởi tạo tham số.
Để thuận tiện cho người dùng,
số chiều và bộ nhớ lưu trữ được tự động suy luận.
Một hệ quả của khởi tạo trì hoãn (lazy initialization) là
bạn không được cố truy cập các tham số
trước khi chúng được khởi tạo (initialized).
:end_tab:

:begin_tab:`pytorch`
Trong PyTorch, module `data` cung cấp các công cụ để xử lý dữ liệu,
module `nn` định nghĩa một số lượng lớn các layer mạng neural và các hàm loss phổ biến.
Chúng ta có thể khởi tạo các tham số bằng cách thay thế giá trị của chúng
bằng các phương thức kết thúc bằng `_`.
Lưu ý rằng chúng ta cần chỉ định các chiều đầu vào của mạng.
Hiện tại, việc này khá đơn giản, nhưng nó có thể gây ra những ảnh hưởng đáng kể
khi chúng ta muốn thiết kế các mạng phức tạp với nhiều layer.
Cần cân nhắc cẩn thận cách tham số hóa các mạng này
để cho phép tính di động (portability).
:end_tab:

:begin_tab:`tensorflow`
Trong TensorFlow, module `data` cung cấp các công cụ để xử lý dữ liệu,
module `keras` định nghĩa một số lượng lớn các layer mạng neural và các hàm loss phổ biến.
Ngoài ra, module `initializers` cung cấp nhiều phương thức khác nhau để khởi tạo tham số mô hình.
Số chiều và bộ nhớ lưu trữ cho các mạng được tự động suy luận
(nhưng hãy cẩn thận, không cố truy cập các tham số trước khi chúng được khởi tạo).
:end_tab:

## Bài tập

1. Bạn cần thay đổi learning rate như thế nào nếu thay tổng loss trên minibatch
   bằng giá trị trung bình của loss trên minibatch?
1. Xem tài liệu của framework để biết những hàm loss nào được cung cấp. Cụ thể,
   hãy thay squared loss bằng hàm robust loss của Huber. Nghĩa là, sử dụng hàm loss
   $$l(y,y') = \begin{cases}|y-y'| -\frac{\sigma}{2} & \textrm{ if } |y-y'| > \sigma \\ \frac{1}{2 \sigma} (y-y')^2 & \textrm{ otherwise}\end{cases}$$
1. Làm thế nào để truy cập gradient của các trọng số trong mô hình?
1. Nghiệm thay đổi như thế nào nếu bạn thay đổi learning rate và số epoch? Nó có tiếp tục được cải thiện không?
1. Nghiệm thay đổi như thế nào khi bạn thay đổi lượng dữ liệu được tạo ra?
    1. Vẽ sai số ước lượng của $\hat{\mathbf{w}} - \mathbf{w}$ và $\hat{b} - b$ theo lượng dữ liệu. Gợi ý: tăng lượng dữ liệu theo logarithm thay vì tuyến tính, tức là 5, 10, 20, 50, ..., 10,000 thay vì 1000, 2000, ..., 10,000.
    2. Tại sao đề xuất trong gợi ý lại phù hợp?


:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/204)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17977)
:end_tab: