# Mạng nơ-ron hồi quy sâu

:label:`sec_deep_rnn`

Cho đến nay, chúng ta tập trung vào việc định nghĩa các mạng
gồm một đầu vào dạng chuỗi,
một lớp RNN ẩn duy nhất
và một lớp đầu ra.
Mặc dù chỉ có một lớp ẩn
giữa đầu vào tại mỗi bước thời gian
và đầu ra tương ứng,
vẫn có một nghĩa theo đó các mạng này là mạng sâu.
Các đầu vào từ bước thời gian đầu tiên có thể ảnh hưởng
đến các đầu ra tại bước thời gian cuối cùng $T$
(thường là sau hàng trăm hoặc hàng nghìn bước).
Các đầu vào này đi qua $T$ lần áp dụng
của lớp hồi quy trước khi đến
đầu ra cuối cùng.
Tuy nhiên, chúng ta thường cũng muốn duy trì khả năng
biểu diễn các mối quan hệ phức tạp
giữa các đầu vào tại một bước thời gian nhất định
và các đầu ra tại chính bước thời gian đó.
Do đó, chúng ta thường xây dựng các RNN sâu
không chỉ theo hướng thời gian
mà còn theo hướng từ đầu vào đến đầu ra.
Đây chính là khái niệm về độ sâu
mà chúng ta đã gặp
trong quá trình phát triển MLP
và CNN sâu.


Phương pháp tiêu chuẩn để xây dựng loại RNN sâu này
cực kỳ đơn giản: chúng ta xếp chồng các RNN lên nhau.
Với một chuỗi có độ dài $T$, RNN đầu tiên tạo ra
một chuỗi đầu ra cũng có độ dài $T$.
Đến lượt mình, các đầu ra này tạo thành đầu vào cho lớp RNN tiếp theo.
Trong phần ngắn này, chúng ta minh họa mẫu thiết kế này
và trình bày một ví dụ đơn giản về cách lập trình các RNN xếp chồng như vậy.
Bên dưới, trong :numref:`fig_deep_rnn`, chúng ta minh họa
một RNN sâu với $L$ lớp ẩn.
Mỗi trạng thái ẩn hoạt động trên một đầu vào tuần tự
và tạo ra một đầu ra tuần tự.
Hơn nữa, bất kỳ ô RNN nào (hộp trắng trong :numref:`fig_deep_rnn`) tại mỗi bước thời gian
đều phụ thuộc vào cả
giá trị của cùng lớp đó
tại bước thời gian trước
và giá trị của lớp trước
tại cùng bước thời gian.

![Kiến trúc của một RNN sâu.](../img/deep-rnn.svg)
:label:`fig_deep_rnn`

Về mặt hình thức, giả sử chúng ta có một đầu vào minibatch
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$
(số lượng mẫu $=n$; số lượng đầu vào trong mỗi mẫu $=d$) tại bước thời gian $t$.
Tại cùng bước thời gian,
gọi trạng thái ẩn của lớp ẩn thứ $l^\textrm{th}$ ($l=1,\ldots,L$) là $\mathbf{H}_t^{(l)} \in \mathbb{R}^{n \times h}$
(số lượng nút ẩn $=h$)
và biến của lớp đầu ra là
$\mathbf{O}_t \in \mathbb{R}^{n \times q}$
(số lượng đầu ra: $q$).
Đặt $\mathbf{H}_t^{(0)} = \mathbf{X}_t$,
trạng thái ẩn của
lớp ẩn thứ $l^\textrm{th}$
sử dụng hàm kích hoạt $\phi_l$
được tính như sau:

$$\mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{\textrm{xh}}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{\textrm{hh}}^{(l)}  + \mathbf{b}_\textrm{h}^{(l)}),$$
:eqlabel:`eq_deep_rnn_H`

trong đó các trọng số $\mathbf{W}_{\textrm{xh}}^{(l)} \in \mathbb{R}^{h \times h}$ và $\mathbf{W}_{\textrm{hh}}^{(l)} \in \mathbb{R}^{h \times h}$, cùng với
bias $\mathbf{b}_\textrm{h}^{(l)} \in \mathbb{R}^{1 \times h}$,
là các tham số mô hình của lớp ẩn thứ $l^\textrm{th}$.

Cuối cùng, phép tính của lớp đầu ra
chỉ dựa trên trạng thái ẩn
của lớp ẩn cuối cùng $L^\textrm{th}$:

$$\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q},$$

trong đó trọng số $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$
và bias $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$
là các tham số mô hình của lớp đầu ra.

Cũng như với MLP, số lượng lớp ẩn $L$
và số lượng nút ẩn $h$ là các hyperparameter
mà chúng ta có thể điều chỉnh.
Độ rộng lớp RNN phổ biến ($h$) nằm trong khoảng $(64, 2056)$,
và độ sâu phổ biến ($L$) nằm trong khoảng $(1, 8)$.
Ngoài ra, chúng ta có thể dễ dàng tạo ra một RNN có cổng sâu
bằng cách thay thế phép tính trạng thái ẩn trong :eqref:`eq_deep_rnn_H`
bằng phép tính từ một LSTM hoặc GRU.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn
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

## Tự triển khai

Để triển khai một RNN nhiều lớp từ đầu,
chúng ta có thể xem mỗi lớp là một thực thể `RNNScratch`
với các tham số có thể học riêng.

```{.python .input}
%%tab mxnet, tensorflow
class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnns = [d2l.RNNScratch(num_inputs if i==0 else num_hiddens,
                                    num_hiddens, sigma)
                     for i in range(num_layers)]
```

```{.python .input}
%%tab pytorch
class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnns = nn.Sequential(*[d2l.RNNScratch(
            num_inputs if i==0 else num_hiddens, num_hiddens, sigma)
                                    for i in range(num_layers)])
```

```{.python .input}
%%tab jax
class StackedRNNScratch(d2l.Module):
    num_inputs: int
    num_hiddens: int
    num_layers: int
    sigma: float = 0.01

    def setup(self):
        self.rnns = [d2l.RNNScratch(self.num_inputs if i==0 else self.num_hiddens,
                                    self.num_hiddens, self.sigma)
                     for i in range(self.num_layers)]
```

Phép tính forward của mạng nhiều lớp
chỉ đơn giản là thực hiện phép tính forward
từng lớp một.

```{.python .input}
%%tab all
@d2l.add_to_class(StackedRNNScratch)
def forward(self, inputs, Hs=None):
    outputs = inputs
    if Hs is None: Hs = [None] * self.num_layers
    for i in range(self.num_layers):
        outputs, Hs[i] = self.rnns[i](outputs, Hs[i])
        outputs = d2l.stack(outputs, 0)
    return outputs, Hs
```

Ví dụ, chúng ta huấn luyện một mô hình GRU sâu trên
tập dữ liệu *The Time Machine* (giống như trong :numref:`sec_rnn-scratch`).
Để đơn giản, chúng ta đặt số lớp là 2.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn_block = StackedRNNScratch(num_inputs=len(data.vocab),
                                  num_hiddens=32, num_layers=2)
    model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=2)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn_block = StackedRNNScratch(num_inputs=len(data.vocab),
                                  num_hiddens=32, num_layers=2)
        model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=2)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## Triển khai cô đọng

:begin_tab:`pytorch, mxnet, tensorflow`
May mắn là nhiều chi tiết về tổ chức cần thiết
để triển khai nhiều lớp của một RNN
đã có sẵn trong các API cấp cao.
Phần triển khai cô đọng của chúng ta sẽ sử dụng các chức năng tích hợp sẵn này.
Mã này mở rộng mã mà chúng ta đã sử dụng trước đó trong :numref:`sec_gru`,
cho phép chúng ta chỉ định rõ ràng số lớp
thay vì chọn giá trị mặc định chỉ có một lớp.
:end_tab:

:begin_tab:`jax`
Flax áp dụng cách tiếp cận tối giản khi triển khai
các RNN. Việc định nghĩa số lớp trong một RNN hoặc kết hợp RNN với dropout
không được cung cấp sẵn.
Phần triển khai cô đọng của chúng ta sẽ sử dụng toàn bộ chức năng tích hợp sẵn và
bổ sung các tính năng `num_layers` và `dropout`.
Mã này mở rộng mã mà chúng ta đã sử dụng trước đó trong :numref:`sec_gru`,
cho phép chỉ định rõ ràng số lớp
thay vì chọn giá trị mặc định là một lớp.
:end_tab:

```{.python .input}
%%tab mxnet
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model."""
    def __init__(self, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
```

```{.python .input}
%%tab pytorch
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model."""
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers,
                          dropout=dropout)
```

```{.python .input}
%%tab tensorflow
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model."""
    def __init__(self, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        gru_cells = [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
                     for _ in range(num_layers)]
        self.rnn = tf.keras.layers.RNN(gru_cells, return_sequences=True,
                                       return_state=True, time_major=True)

    def forward(self, X, state=None):
        outputs, *state = self.rnn(X, state)
        return outputs, state
```

```{.python .input}
%%tab jax
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model."""
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    @nn.compact
    def __call__(self, X, state=None, training=False):
        outputs = X
        new_state = []
        if state is None:
            batch_size = X.shape[1]
            state = [nn.GRUCell.initialize_carry(jax.random.PRNGKey(0),
                    (batch_size,), self.num_hiddens)] * self.num_layers

        GRU = nn.scan(nn.GRUCell, variable_broadcast="params",
                      in_axes=0, out_axes=0, split_rngs={"params": False})

        # Introduce a dropout layer after every GRU layer except last
        for i in range(self.num_layers - 1):
            layer_i_state, X = GRU()(state[i], outputs)
            new_state.append(layer_i_state)
            X = nn.Dropout(self.dropout, deterministic=not training)(X)

        # Final GRU layer without dropout
        out_state, X = GRU()(state[-1], X)
        new_state.append(out_state)
        return X, jnp.array(new_state)
```

Các quyết định về kiến trúc, chẳng hạn như lựa chọn hyperparameter,
rất tương tự với các quyết định trong :numref:`sec_gru`.
Chúng ta chọn cùng số lượng đầu vào và đầu ra
vì số lượng này bằng số token khác nhau, tức là `vocab_size`.
Số lượng nút ẩn vẫn là 32.
Điểm khác biệt duy nhất là hiện tại chúng ta
(**chọn một số lượng lớp ẩn đáng kể
bằng cách chỉ định giá trị của `num_layers`.**)

```{.python .input}
%%tab mxnet
gru = GRU(num_hiddens=32, num_layers=2)
model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)

# Running takes > 1h (pending fix from MXNet)
# trainer.fit(model, data)
# model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab pytorch, tensorflow, jax
if tab.selected('tensorflow', 'jax'):
    gru = GRU(num_hiddens=32, num_layers=2)
if tab.selected('pytorch'):
    gru = GRU(num_inputs=len(data.vocab), num_hiddens=32, num_layers=2)
if tab.selected('pytorch', 'jax'):
    model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)
trainer.fit(model, data)
```

```{.python .input}
%%tab pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

```{.python .input}
%%tab jax
model.predict('it has', 20, data.vocab, trainer.state.params)
```

## Tóm tắt

Trong các RNN sâu, thông tin trạng thái ẩn được truyền
đến bước thời gian tiếp theo của lớp hiện tại
và bước thời gian hiện tại của lớp tiếp theo.
Có nhiều biến thể khác nhau của RNN sâu, chẳng hạn như LSTM, GRU hoặc RNN cơ bản.
Thuận tiện là tất cả các mô hình này đều có sẵn
dưới dạng các thành phần của API cấp cao trong các framework deep learning.
Việc khởi tạo mô hình đòi hỏi sự cẩn thận.
Nhìn chung, RNN sâu đòi hỏi khá nhiều công sức
(chẳng hạn như điều chỉnh learning rate và clipping)
để bảo đảm hội tụ đúng cách.

## Bài tập

1. Thay GRU bằng LSTM và so sánh độ chính xác cũng như tốc độ huấn luyện.
1. Tăng dữ liệu huấn luyện để bao gồm nhiều cuốn sách. Bạn có thể giảm perplexity xuống thấp đến mức nào?
1. Khi lập mô hình văn bản, bạn có muốn kết hợp các nguồn của những tác giả khác nhau không? Tại sao đây là một ý hay? Điều gì có thể xảy ra?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/340)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1058)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/3862)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18018)
:end_tab: