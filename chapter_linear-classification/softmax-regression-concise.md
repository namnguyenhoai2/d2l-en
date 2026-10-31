```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Cài đặt Ngắn gọn Hồi quy Softmax
:label:`sec_softmax_concise`



Cũng như các framework deep learning cấp cao
đã giúp việc cài đặt hồi quy tuyến tính trở nên dễ dàng hơn
(xem :numref:`sec_linear_concise`),
chúng cũng tiện lợi tương tự trong trường hợp này.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
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
from functools import partial
import jax
from jax import numpy as jnp
import optax
```

## Định nghĩa Mô hình

Như trong :numref:`sec_linear_concise`, 
chúng ta xây dựng lớp fully connected 
bằng lớp tích hợp sẵn. 
Phương thức `__call__` tích hợp sẵn sau đó gọi `forward` 
mỗi khi cần áp dụng mạng lên một đầu vào nào đó.

:begin_tab:`mxnet`
Mặc dù đầu vào `X` là một tensor bậc bốn, 
lớp `Dense` tích hợp sẵn 
sẽ tự động chuyển đổi `X` thành tensor bậc hai 
bằng cách giữ nguyên số chiều dọc theo trục đầu tiên.
:end_tab:

:begin_tab:`pytorch`
Chúng ta sử dụng một lớp `Flatten` để chuyển đổi tensor bậc bốn `X` sang bậc hai 
bằng cách giữ nguyên số chiều dọc theo trục đầu tiên.

:end_tab:

:begin_tab:`tensorflow`
Chúng ta sử dụng một lớp `Flatten` để chuyển đổi tensor bậc bốn `X` 
bằng cách giữ nguyên số chiều dọc theo trục đầu tiên.
:end_tab:

:begin_tab:`jax`
Flax cho phép người dùng viết lớp mạng theo cách gọn hơn
bằng cách sử dụng decorator `@nn.compact`. Với `@nn.compact`, người dùng
có thể đơn giản viết toàn bộ logic mạng bên trong một phương thức
“forward pass” duy nhất mà không cần định nghĩa phương thức `setup` tiêu chuẩn trong
dataclass.
:end_tab:

```{.python .input}
%%tab pytorch
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input}
%%tab mxnet, tensorflow
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(num_outputs)
            self.net.initialize()
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            self.net.add(tf.keras.layers.Flatten())
            self.net.add(tf.keras.layers.Dense(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input}
%%tab jax
class SoftmaxRegression(d2l.Classifier):  #@save
    num_outputs: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_outputs)(X)
        return X
```

## Xem lại Softmax
:label:`subsec_softmax-implementation-revisited`

Trong :numref:`sec_softmax_scratch`, chúng ta đã tính đầu ra của mô hình
và áp dụng hàm mất mát cross-entropy. Mặc dù điều này hoàn toàn
hợp lý về mặt toán học, nó lại rủi ro về mặt tính toán do
hiện tượng underflow và overflow số học khi lũy thừa.

Hãy nhớ rằng hàm softmax tính xác suất theo
$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$.
Nếu một vài $o_k$ rất lớn, tức là rất dương,
thì $\exp(o_k)$ có thể lớn hơn số lớn nhất
mà chúng ta có thể biểu diễn với một số kiểu dữ liệu nhất định. Điều này được gọi là *overflow*. Tương tự,
nếu mọi đối số đều là số âm rất lớn, chúng ta sẽ gặp *underflow*.
Ví dụ, số dấu phẩy động độ chính xác đơn xấp xỉ
bao phủ khoảng từ $10^{-38}$ đến $10^{38}$. Do đó, nếu hạng tử lớn nhất trong $\mathbf{o}$
nằm ngoài khoảng $[-90, 90]$, kết quả sẽ không ổn định.
Một cách giải quyết vấn đề này là trừ $\bar{o} \stackrel{\textrm{def}}{=} \max_k o_k$ khỏi
mọi phần tử:

$$
\hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} =
\frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} =
\frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}.
$$

Theo cách xây dựng, chúng ta biết rằng $o_j - \bar{o} \leq 0$ với mọi $j$. Do đó, đối với bài toán
phân loại $q$ lớp, mẫu số nằm trong khoảng $[1, q]$. Hơn nữa,
tử số không bao giờ vượt quá $1$, nhờ đó ngăn chặn overflow số học. Underflow số học chỉ
xảy ra khi $\exp(o_j - \bar{o})$ được tính về mặt số học thành $0$. Tuy nhiên, sau vài bước
chúng ta có thể gặp rắc rối khi muốn tính $\log \hat{y}_j$ dưới dạng $\log 0$.
Đặc biệt, trong backpropagation,
chúng ta có thể phải đối mặt với cả một màn hình
đầy các kết quả `NaN` (Không phải là Số) đáng sợ.

May mắn thay, chúng ta được cứu bởi thực tế rằng
mặc dù đang tính các hàm mũ,
cuối cùng chúng ta lại muốn lấy log của chúng
(khi tính hàm mất mát cross-entropy).
Bằng cách kết hợp softmax và cross-entropy,
chúng ta có thể hoàn toàn tránh được các vấn đề ổn định số học. Ta có:

$$
\log \hat{y}_j =
\log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} =
o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o}).
$$

Điều này tránh được cả overflow lẫn underflow.
Chúng ta vẫn cần giữ sẵn hàm softmax thông thường
trong trường hợp muốn đánh giá xác suất đầu ra của mô hình.
Nhưng thay vì truyền các xác suất softmax vào hàm mất mát mới,
chúng ta chỉ
[**truyền các logits và tính softmax cùng log của nó
trong một lần bên trong hàm mất mát cross-entropy,**]
vốn thực hiện các thao tác thông minh như ["mẹo LogSumExp"](https://en.wikipedia.org/wiki/LogSumExp).

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    if tab.selected('mxnet'):
        fn = gluon.loss.SoftmaxCrossEntropyLoss()
        l = fn(Y_hat, Y)
        return l.mean() if averaged else l
    if tab.selected('pytorch'):
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return fn(Y, Y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    # To be used later (e.g., for batch norm)
    Y_hat = state.apply_fn({'params': params}, *X,
                           mutable=False, rngs=None)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    # The returned empty dictionary is a placeholder for auxiliary data,
    # which will be used later (e.g., for batch norm)
    return (fn(Y_hat, Y).mean(), {}) if averaged else (fn(Y_hat, Y), {})
```

## Huấn luyện

Tiếp theo, chúng ta huấn luyện mô hình. Chúng ta sử dụng các ảnh Fashion-MNIST, được làm phẳng thành các vector đặc trưng 784 chiều.

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

Như trước đây, thuật toán này hội tụ đến một nghiệm
có độ chính xác khá tốt,
mặc dù lần này với ít dòng mã hơn trước.


## Tóm tắt

Các API cấp cao rất tiện lợi trong việc che giấu khỏi người dùng những khía cạnh có thể nguy hiểm, chẳng hạn như độ ổn định số học. Hơn nữa, chúng cho phép người dùng thiết kế mô hình một cách ngắn gọn chỉ với rất ít dòng mã. Đây vừa là phúc lành vừa là lời nguyền. Lợi ích rõ ràng là chúng làm mọi thứ trở nên rất dễ tiếp cận, ngay cả với những kỹ sư chưa từng học một lớp thống kê nào trong đời (thực tế, họ là một phần đối tượng mục tiêu của cuốn sách). Nhưng việc che giấu các góc cạnh sắc bén cũng đi kèm cái giá phải trả: làm giảm động lực tự thêm các thành phần mới và khác biệt, vì có rất ít kinh nghiệm thực hành để làm điều đó. Hơn nữa, nó khiến việc *sửa* mọi thứ trở nên khó khăn hơn mỗi khi lớp bảo vệ của
một framework không bao phủ hoàn toàn mọi trường hợp biên. Một lần nữa, điều này là do thiếu sự quen thuộc.

Vì vậy, chúng tôi đặc biệt khuyến nghị bạn xem lại *cả hai* phiên bản tối giản và thanh lịch của nhiều cách cài đặt tiếp theo. Mặc dù chúng tôi nhấn mạnh tính dễ hiểu, các cách cài đặt này thường vẫn có hiệu năng khá tốt (convolution là ngoại lệ lớn ở đây). Mục tiêu của chúng tôi là cho phép bạn xây dựng dựa trên chúng khi bạn phát minh ra điều gì đó mới mà không framework nào có thể cung cấp.


## Bài tập

1. Deep learning sử dụng nhiều định dạng số khác nhau, bao gồm độ chính xác kép FP64 (được dùng cực kỳ hiếm), độ chính xác đơn FP32, BFLOAT16 (tốt cho các biểu diễn nén), FP16 (rất không ổn định), TF32 (một định dạng mới của NVIDIA), và INT8. Hãy tính đối số nhỏ nhất và lớn nhất của hàm mũ sao cho kết quả không dẫn đến underflow hoặc overflow số học.
1. INT8 là một định dạng rất hạn chế gồm các số khác không từ $1$ đến $255$. Làm thế nào bạn có thể mở rộng dynamic range của nó mà không sử dụng thêm bit? Phép nhân và phép cộng tiêu chuẩn có còn hoạt động không?
1. Tăng số epoch huấn luyện. Vì sao độ chính xác validation có thể giảm sau một thời gian? Chúng ta có thể khắc phục điều này như thế nào?
1. Điều gì xảy ra khi bạn tăng learning rate? Hãy so sánh các đường cong loss với nhiều learning rate khác nhau. Giá trị nào hoạt động tốt hơn? Khi nào?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/260)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17983)
:end_tab: