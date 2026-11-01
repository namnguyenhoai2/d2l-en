# Các đơn vị hồi quy có cổng (GRU)
:label:`sec_gru`


Khi RNN và đặc biệt là kiến trúc LSTM (:numref:`sec_lstm`)
nhanh chóng trở nên phổ biến trong những năm 2010,
một số nhà nghiên cứu bắt đầu thử nghiệm
các kiến trúc đơn giản hóa với hy vọng
giữ lại ý tưởng cốt lõi là kết hợp
trạng thái bên trong và các cơ chế gating nhân
nhưng nhằm mục tiêu tăng tốc độ tính toán.
Đơn vị hồi quy có cổng (GRU) :cite:`Cho.Van-Merrienboer.Bahdanau.ea.2014` 
đưa ra một phiên bản tinh gọn của ô nhớ LSTM,
thường đạt hiệu năng tương đương
nhưng có ưu điểm là tính toán nhanh hơn
:cite:`Chung.Gulcehre.Cho.ea.2014`.

```{.python .input  n=5}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=6}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn
npx.set_np()
```

```{.python .input  n=7}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=8}
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

## Cổng Reset và Cổng Update

Ở đây, ba cổng của LSTM được thay thế bằng hai cổng:
*cổng reset* và *cổng update*.
Tương tự như LSTM, các cổng này sử dụng hàm kích hoạt sigmoid,
buộc giá trị của chúng nằm trong khoảng $(0, 1)$.
Theo trực giác, cổng reset kiểm soát mức độ
mà chúng ta vẫn muốn ghi nhớ từ trạng thái trước đó.
Tương tự, cổng update cho phép chúng ta kiểm soát
mức độ trạng thái mới chỉ là bản sao của trạng thái cũ.
:numref:`fig_gru_1` minh họa các đầu vào cho cả
cổng reset và cổng update trong GRU,
với đầu vào của bước thời gian hiện tại
và trạng thái ẩn của bước thời gian trước đó.
Đầu ra của các cổng được tạo ra
bởi hai lớp kết nối đầy đủ
với hàm kích hoạt sigmoid.

![Tính toán cổng reset và cổng update trong một mô hình GRU.](../img/gru-1.svg)
:label:`fig_gru_1`

Về mặt toán học, với một bước thời gian $t$,
giả sử đầu vào là một minibatch
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$ 
(số lượng ví dụ $=n$; số lượng đầu vào $=d$)
và trạng thái ẩn của bước thời gian trước đó
là $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$ 
(số lượng đơn vị ẩn $=h$).
Khi đó, cổng reset $\mathbf{R}_t \in \mathbb{R}^{n \times h}$
và cổng update $\mathbf{Z}_t \in \mathbb{R}^{n \times h}$ được tính như sau:

$$
\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xr}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hr}} + \mathbf{b}_\textrm{r}),\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xz}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hz}} + \mathbf{b}_\textrm{z}),
\end{aligned}
$$

trong đó $\mathbf{W}_{\textrm{xr}}, \mathbf{W}_{\textrm{xz}} \in \mathbb{R}^{d \times h}$
và $\mathbf{W}_{\textrm{hr}}, \mathbf{W}_{\textrm{hz}} \in \mathbb{R}^{h \times h}$
là các tham số trọng số, còn $\mathbf{b}_\textrm{r}, \mathbf{b}_\textrm{z} \in \mathbb{R}^{1 \times h}$
là các tham số bias.


## Trạng thái ẩn ứng viên

Tiếp theo, chúng ta tích hợp cổng reset $\mathbf{R}_t$
với cơ chế cập nhật thông thường
trong :eqref:`rnn_h_with_state`,
dẫn đến
*trạng thái ẩn ứng viên*
$\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$ tại bước thời gian $t$:

$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{\textrm{hh}} + \mathbf{b}_\textrm{h}),$$
:eqlabel:`gru_tilde_H`

trong đó $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$ và $\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$
là các tham số trọng số,
$\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$
là bias,
và ký hiệu $\odot$ là toán tử tích Hadamard (theo từng phần tử).
Ở đây, chúng ta sử dụng hàm kích hoạt tanh.

Kết quả là một *ứng viên*, vì chúng ta vẫn cần
tích hợp tác động của cổng update.
So sánh với :eqref:`rnn_h_with_state`,
ảnh hưởng của các trạng thái trước đó
giờ đây có thể được giảm bớt bằng phép nhân
theo từng phần tử giữa
$\mathbf{R}_t$ và $\mathbf{H}_{t-1}$
trong :eqref:`gru_tilde_H`.
Khi các phần tử trong cổng reset $\mathbf{R}_t$ gần bằng 1,
chúng ta thu được một RNN thông thường như trong :eqref:`rnn_h_with_state`.
Đối với mọi phần tử của cổng reset $\mathbf{R}_t$ gần bằng 0,
trạng thái ẩn ứng viên là kết quả của một MLP với $\mathbf{X}_t$ làm đầu vào.
Do đó, mọi trạng thái ẩn đã tồn tại đều được *reset* về các giá trị mặc định.

:numref:`fig_gru_2` minh họa luồng tính toán sau khi áp dụng cổng reset.

![Tính toán trạng thái ẩn ứng viên trong một mô hình GRU.](../img/gru-2.svg)
:label:`fig_gru_2`


## Trạng thái ẩn

Cuối cùng, chúng ta cần tích hợp tác động của cổng update $\mathbf{Z}_t$.
Cổng này xác định mức độ mà trạng thái ẩn mới $\mathbf{H}_t \in \mathbb{R}^{n \times h}$ 
giống với trạng thái cũ $\mathbf{H}_{t-1}$ so với mức độ
nó giống với trạng thái ứng viên mới $\tilde{\mathbf{H}}_t$.
Có thể sử dụng cổng update $\mathbf{Z}_t$ cho mục đích này
bằng cách lấy các tổ hợp lồi theo từng phần tử
của $\mathbf{H}_{t-1}$ và $\tilde{\mathbf{H}}_t$.
Điều này dẫn đến phương trình cập nhật cuối cùng cho GRU:

$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$$


Khi cổng update $\mathbf{Z}_t$ gần bằng 1,
chúng ta chỉ giữ lại trạng thái cũ.
Trong trường hợp này, thông tin từ $\mathbf{X}_t$ bị bỏ qua,
về cơ bản bỏ qua bước thời gian $t$ trong chuỗi phụ thuộc.
Ngược lại, khi $\mathbf{Z}_t$ gần bằng 0,
trạng thái tiềm ẩn mới $\mathbf{H}_t$ tiến gần đến trạng thái tiềm ẩn ứng viên $\tilde{\mathbf{H}}_t$.
:numref:`fig_gru_3` cho thấy luồng tính toán sau khi cổng update có tác dụng.

![Tính toán trạng thái ẩn trong một mô hình GRU.](../img/gru-3.svg)
:label:`fig_gru_3`


Tóm lại, GRU có hai đặc điểm phân biệt sau:

* Cổng reset giúp nắm bắt các phụ thuộc ngắn hạn trong chuỗi.
* Cổng update giúp nắm bắt các phụ thuộc dài hạn trong chuỗi.

## Triển khai từ đầu

Để hiểu rõ hơn về mô hình GRU, hãy triển khai mô hình này từ đầu.

### (**Khởi tạo các tham số mô hình**)

Bước đầu tiên là khởi tạo các tham số mô hình.
Chúng ta lấy các trọng số từ một phân phối Gaussian
với độ lệch chuẩn bằng `sigma` và đặt bias bằng 0.
Siêu tham số `num_hiddens` xác định số lượng đơn vị ẩn.
Chúng ta khởi tạo tất cả trọng số và bias liên quan đến cổng update,
cổng reset và trạng thái ẩn ứng viên.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class GRUScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        if tab.selected('mxnet'):
            init_weight = lambda *shape: d2l.randn(*shape) * sigma
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              d2l.zeros(num_hiddens))            
        if tab.selected('pytorch'):
            init_weight = lambda *shape: nn.Parameter(d2l.randn(*shape) * sigma)
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              nn.Parameter(d2l.zeros(num_hiddens)))
        if tab.selected('tensorflow'):
            init_weight = lambda *shape: tf.Variable(d2l.normal(shape) * sigma)
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              tf.Variable(d2l.zeros(num_hiddens)))            
            
        self.W_xz, self.W_hz, self.b_z = triple()  # Update gate
        self.W_xr, self.W_hr, self.b_r = triple()  # Reset gate
        self.W_xh, self.W_hh, self.b_h = triple()  # Candidate hidden state        
```

```{.python .input}
%%tab jax
class GRUScratch(d2l.Module):
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        init_weight = lambda name, shape: self.param(name,
                                                     nn.initializers.normal(self.sigma),
                                                     shape)
        triple = lambda name : (
            init_weight(f'W_x{name}', (self.num_inputs, self.num_hiddens)),
            init_weight(f'W_h{name}', (self.num_hiddens, self.num_hiddens)),
            self.param(f'b_{name}', nn.initializers.zeros, (self.num_hiddens)))

        self.W_xz, self.W_hz, self.b_z = triple('z')  # Update gate
        self.W_xr, self.W_hr, self.b_r = triple('r')  # Reset gate
        self.W_xh, self.W_hh, self.b_h = triple('h')  # Candidate hidden state
```

### Định nghĩa mô hình

Bây giờ chúng ta đã sẵn sàng để [**định nghĩa phép tính forward của GRU**].
Cấu trúc của nó giống với cấu trúc của một ô RNN cơ bản,
ngoại trừ việc các phương trình cập nhật phức tạp hơn.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(GRUScratch)
def forward(self, inputs, H=None):
    if H is None:
        # Initial state with shape: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          ctx=inputs.ctx)
        if tab.selected('pytorch'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
        if tab.selected('tensorflow'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens))
    outputs = []
    for X in inputs:
        Z = d2l.sigmoid(d2l.matmul(X, self.W_xz) +
                        d2l.matmul(H, self.W_hz) + self.b_z)
        R = d2l.sigmoid(d2l.matmul(X, self.W_xr) + 
                        d2l.matmul(H, self.W_hr) + self.b_r)
        H_tilde = d2l.tanh(d2l.matmul(X, self.W_xh) + 
                           d2l.matmul(R * H, self.W_hh) + self.b_h)
        H = Z * H + (1 - Z) * H_tilde
        outputs.append(H)
    return outputs, H
```

```{.python .input}
%%tab jax
@d2l.add_to_class(GRUScratch)
def forward(self, inputs, H=None):
    # Use lax.scan primitive instead of looping over the
    # inputs, since scan saves time in jit compilation
    def scan_fn(H, X):
        Z = d2l.sigmoid(d2l.matmul(X, self.W_xz) + d2l.matmul(H, self.W_hz) +
                        self.b_z)
        R = d2l.sigmoid(d2l.matmul(X, self.W_xr) +
                        d2l.matmul(H, self.W_hr) + self.b_r)
        H_tilde = d2l.tanh(d2l.matmul(X, self.W_xh) +
                           d2l.matmul(R * H, self.W_hh) + self.b_h)
        H = Z * H + (1 - Z) * H_tilde
        return H, H  # return carry, y

    if H is None:
        batch_size = inputs.shape[1]
        carry = jnp.zeros((batch_size, self.num_hiddens))
    else:
        carry = H

    # scan takes the scan_fn, initial carry state, xs with leading axis to be scanned
    carry, outputs = jax.lax.scan(scan_fn, carry, inputs)
    return outputs, carry
```

### Huấn luyện

[**Huấn luyện**] một mô hình ngôn ngữ trên tập dữ liệu *The Time Machine*
hoàn toàn giống với cách thực hiện trong :numref:`sec_rnn-scratch`.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = d2l.RNNLMScratch(gru, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = d2l.RNNLMScratch(gru, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1)
trainer.fit(model, data)
```

## [**Triển khai cô đọng**]

Trong các API cấp cao, chúng ta có thể khởi tạo trực tiếp một mô hình GRU.
Điều này đóng gói mọi chi tiết cấu hình mà chúng ta đã nêu rõ ở trên.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class GRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.rnn = rnn.GRU(num_hiddens)
        if tab.selected('pytorch'):
            self.rnn = nn.GRU(num_inputs, num_hiddens)
        if tab.selected('tensorflow'):
            self.rnn = tf.keras.layers.GRU(num_hiddens, return_sequences=True, 
                                           return_state=True)
```

```{.python .input}
%%tab jax
class GRU(d2l.RNN):
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H=None, training=False):
        if H is None:
            batch_size = inputs.shape[1]
            H = nn.GRUCell.initialize_carry(jax.random.PRNGKey(0),
                                            (batch_size,), self.num_hiddens)

        GRU = nn.scan(nn.GRUCell, variable_broadcast="params",
                      in_axes=0, out_axes=0, split_rngs={"params": False})

        H, outputs = GRU()(H, inputs)
        return outputs, H
```

Mã chạy nhanh hơn đáng kể trong quá trình huấn luyện vì sử dụng các toán tử đã biên dịch
thay vì Python.

```{.python .input}
%%tab all
if tab.selected('mxnet', 'pytorch', 'tensorflow'):
    gru = GRU(num_inputs=len(data.vocab), num_hiddens=32)
if tab.selected('jax'):
    gru = GRU(num_hiddens=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=4)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=4)
trainer.fit(model, data)
```

Sau khi huấn luyện, chúng ta in ra perplexity trên tập huấn luyện
và chuỗi được dự đoán tiếp theo prefix đã cung cấp.

```{.python .input}
%%tab mxnet, pytorch
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

So với LSTM, GRU đạt hiệu năng tương tự nhưng thường nhẹ hơn về mặt tính toán.
Nhìn chung, so với RNN đơn giản, các RNN có cổng, chẳng hạn như LSTM và GRU,
có thể nắm bắt tốt hơn các phụ thuộc trong những chuỗi có khoảng cách bước thời gian lớn.
GRU chứa RNN cơ bản như một trường hợp cực hạn khi cổng reset được bật.
Chúng cũng có thể bỏ qua các chuỗi con bằng cách bật cổng update.


## Bài tập

1. Giả sử rằng chúng ta chỉ muốn sử dụng đầu vào tại bước thời gian $t'$ để dự đoán đầu ra tại bước thời gian $t > t'$. Giá trị tốt nhất của các cổng reset và update cho mỗi bước thời gian là gì?
1. Điều chỉnh các hyperparameter và phân tích ảnh hưởng của chúng đến thời gian chạy, perplexity và chuỗi đầu ra.
1. So sánh runtime, perplexity và các chuỗi đầu ra của các triển khai `rnn.RNN` và `rnn.GRU` với nhau.
1. Điều gì xảy ra nếu bạn chỉ triển khai một phần của GRU, chẳng hạn như chỉ sử dụng cổng reset hoặc chỉ sử dụng cổng update?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/342)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1056)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/3860)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18017)
:end_tab: