# Bộ nhớ dài-ngắn hạn (LSTM)
:label:`sec_lstm`


Không lâu sau khi các RNN kiểu Elman đầu tiên được huấn luyện bằng lan truyền ngược 
:cite:`elman1990finding`, các vấn đề trong việc học các phụ thuộc dài hạn
(do gradient biến mất và bùng nổ)
đã trở nên rõ rệt, khi Bengio và Hochreiter 
thảo luận về vấn đề này
:cite:`bengio1994learning,Hochreiter.Bengio.Frasconi.ea.2001`.
Hochreiter đã trình bày vấn đề này từ năm 
1991 trong luận văn thạc sĩ của mình, mặc dù các kết quả 
không được biết đến rộng rãi vì luận văn được viết bằng tiếng Đức.
Mặc dù việc cắt gradient giúp xử lý gradient bùng nổ, 
việc xử lý gradient biến mất dường như 
đòi hỏi một giải pháp phức tạp hơn. 
Một trong những kỹ thuật đầu tiên và thành công nhất 
để xử lý gradient biến mất 
là mô hình bộ nhớ dài-ngắn hạn (long short-term memory, LSTM) 
do :citet:`Hochreiter.Schmidhuber.1997` đề xuất. 
LSTM tương tự như các mạng nơ-ron hồi quy tiêu chuẩn, 
nhưng ở đây mỗi node hồi quy thông thường
được thay thế bằng một *memory cell*.
Mỗi memory cell chứa một *trạng thái bên trong*,
tức là một node có một cạnh hồi quy tự kết nối với trọng số cố định bằng 1,
bảo đảm rằng gradient có thể truyền qua nhiều bước thời gian 
mà không biến mất hoặc bùng nổ.

Thuật ngữ "long short-term memory" xuất phát từ trực giác sau đây.
Các mạng nơ-ron hồi quy đơn giản 
có *bộ nhớ dài hạn* dưới dạng các trọng số.
Các trọng số thay đổi chậm trong quá trình huấn luyện, 
mã hóa tri thức tổng quát về dữ liệu.
Chúng cũng có *bộ nhớ ngắn hạn*
dưới dạng các activation tạm thời,
truyền từ mỗi node đến các node tiếp theo.
Mô hình LSTM giới thiệu một kiểu lưu trữ trung gian thông qua memory cell.
Một memory cell là một đơn vị kết hợp, 
được xây dựng từ các node đơn giản hơn 
theo một mẫu kết nối cụ thể,
với điểm mới là sự xuất hiện của các node nhân.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
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

## Memory Cell có cổng

Mỗi memory cell được trang bị một *trạng thái bên trong*
và một số cổng nhân, xác định liệu
(i) một input nhất định có tác động đến trạng thái bên trong hay không (*input gate*),
(ii) trạng thái bên trong có được xóa về $0$ hay không (*forget gate*),
và (iii) trạng thái bên trong của một neuron nhất định 
có được phép tác động đến output của cell hay không 
(*output gate*). 


### Hidden State có cổng

Điểm khác biệt then chốt giữa vanilla RNN và LSTM
là LSTM hỗ trợ gating cho hidden state.
Điều này có nghĩa là chúng ta có các cơ chế chuyên biệt để xác định
khi nào hidden state cần được *cập nhật* và
khi nào nó cần được *reset*.
Các cơ chế này được học và giải quyết những mối quan ngại đã nêu ở trên.
Chẳng hạn, nếu token đầu tiên có tầm quan trọng lớn,
chúng ta sẽ học cách không cập nhật hidden state sau lần quan sát đầu tiên.
Tương tự, chúng ta sẽ học cách bỏ qua các quan sát tạm thời không liên quan.
Cuối cùng, chúng ta sẽ học cách reset latent state bất cứ khi nào cần.
Chúng ta sẽ thảo luận chi tiết về điều này dưới đây.

### Input Gate, Forget Gate và Output Gate

Dữ liệu đưa vào các gate của LSTM là
input tại bước thời gian hiện tại và
hidden state của bước thời gian trước đó,
như minh họa trong :numref:`fig_lstm_0`.
Ba lớp fully connected với các hàm activation sigmoid
tính toán giá trị của input gate, forget gate và output gate.
Do activation sigmoid,
tất cả giá trị của ba gate
đều nằm trong khoảng $(0, 1)$.
Ngoài ra, chúng ta cần một *input node*,
thường được tính bằng activation *tanh*. 
Theo trực giác, *input gate* xác định mức độ
giá trị của input node cần được cộng 
vào trạng thái bên trong hiện tại của memory cell.
*Forget gate* xác định có giữ lại
giá trị hiện tại của memory hay xóa nó. 
Còn *output gate* xác định liệu 
memory cell có nên ảnh hưởng đến output
tại bước thời gian hiện tại hay không. 


![Tính toán input gate, forget gate và output gate trong mô hình LSTM.](../img/lstm-0.svg)
:label:`fig_lstm_0`

Về mặt toán học, giả sử có $h$ hidden unit, 
kích thước batch là $n$ và số lượng input là $d$.
Do đó, input là $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ 
và hidden state của bước thời gian trước đó 
là $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$. 
Tương ứng, các gate tại bước thời gian $t$
được định nghĩa như sau: input gate là $\mathbf{I}_t \in \mathbb{R}^{n \times h}$, 
forget gate là $\mathbf{F}_t \in \mathbb{R}^{n \times h}$, 
và output gate là $\mathbf{O}_t \in \mathbb{R}^{n \times h}$. 
Chúng được tính như sau:

$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xi}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hi}} + \mathbf{b}_\textrm{i}),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xf}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hf}} + \mathbf{b}_\textrm{f}),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xo}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{ho}} + \mathbf{b}_\textrm{o}),
\end{aligned}
$$

trong đó $\mathbf{W}_{\textrm{xi}}, \mathbf{W}_{\textrm{xf}}, \mathbf{W}_{\textrm{xo}} \in \mathbb{R}^{d \times h}$ và $\mathbf{W}_{\textrm{hi}}, \mathbf{W}_{\textrm{hf}}, \mathbf{W}_{\textrm{ho}} \in \mathbb{R}^{h \times h}$ là các tham số trọng số,
còn $\mathbf{b}_\textrm{i}, \mathbf{b}_\textrm{f}, \mathbf{b}_\textrm{o} \in \mathbb{R}^{1 \times h}$ là các tham số bias.
Lưu ý rằng broadcasting 
(xem :numref:`subsec_broadcasting`)
được kích hoạt trong quá trình cộng.
Chúng ta sử dụng các hàm sigmoid
(như đã giới thiệu trong :numref:`sec_mlp`) 
để ánh xạ các giá trị input vào khoảng $(0, 1)$.


### Input Node

Tiếp theo, chúng ta thiết kế memory cell. 
Vì chưa xác định tác động của các gate khác nhau, 
trước tiên chúng ta giới thiệu *input node* 
$\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$.
Cách tính node này tương tự như cách tính ba gate được mô tả ở trên, 
nhưng sử dụng hàm $\tanh$ có miền giá trị $(-1, 1)$ làm activation function. 
Điều này dẫn đến phương trình sau tại bước thời gian $t$:

$$\tilde{\mathbf{C}}_t = \textrm{tanh}(\mathbf{X}_t \mathbf{W}_{\textrm{xc}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hc}} + \mathbf{b}_\textrm{c}),$$

trong đó $\mathbf{W}_{\textrm{xc}} \in \mathbb{R}^{d \times h}$ và $\mathbf{W}_{\textrm{hc}} \in \mathbb{R}^{h \times h}$ là các tham số trọng số, còn $\mathbf{b}_\textrm{c} \in \mathbb{R}^{1 \times h}$ là một tham số bias.

Một minh họa nhanh về input node được trình bày trong :numref:`fig_lstm_1`.

![Tính toán input node trong mô hình LSTM.](../img/lstm-1.svg)
:label:`fig_lstm_1`


### Trạng thái bên trong của Memory Cell

Trong LSTM, input gate $\mathbf{I}_t$ chi phối 
mức độ dữ liệu mới được đưa vào thông qua $\tilde{\mathbf{C}}_t$, 
còn forget gate $\mathbf{F}_t$ xác định 
mức độ trạng thái bên trong cũ của cell $\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$ được giữ lại. 
Sử dụng toán tử tích Hadamard (theo từng phần tử) $\odot$,
chúng ta có phương trình cập nhật sau:

$$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.$$

Nếu forget gate luôn bằng 1 và input gate luôn bằng 0, 
trạng thái bên trong của memory cell $\mathbf{C}_{t-1}$
sẽ giữ nguyên mãi mãi, 
được truyền không thay đổi đến mỗi bước thời gian tiếp theo.
Tuy nhiên, input gate và forget gate
mang lại cho mô hình sự linh hoạt để học 
khi nào cần giữ nguyên giá trị này
và khi nào cần làm thay đổi giá trị đó để phản hồi 
với các input tiếp theo. 
Trong thực tế, thiết kế này làm giảm vấn đề gradient biến mất,
tạo ra các mô hình dễ huấn luyện hơn nhiều,
đặc biệt khi làm việc với các dataset có độ dài chuỗi lớn. 

Do đó, chúng ta có sơ đồ luồng trong :numref:`fig_lstm_2`.

![Tính toán trạng thái bên trong của memory cell trong mô hình LSTM.](../img/lstm-2.svg)

:label:`fig_lstm_2`


### Hidden State

Cuối cùng, chúng ta cần xác định cách tính đầu ra
của memory cell, tức hidden state $\mathbf{H}_t \in \mathbb{R}^{n \times h}$, được các layer khác nhìn thấy.
Đây là lúc output gate phát huy tác dụng.
Trong LSTM, trước tiên chúng ta áp dụng $\tanh$ cho trạng thái nội bộ của memory cell
rồi thực hiện một phép nhân từng phần tử khác,
lần này với output gate.
Điều này đảm bảo rằng các giá trị của $\mathbf{H}_t$
luôn nằm trong khoảng $(-1, 1)$:

$$\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t).$$


Khi output gate gần bằng 1,
chúng ta cho phép trạng thái nội bộ của memory cell tác động đến các layer tiếp theo mà không bị hạn chế,
trong khi với các giá trị của output gate gần bằng 0,
chúng ta ngăn memory hiện tại tác động đến các layer khác của mạng
tại time step hiện tại.
Lưu ý rằng một memory cell có thể tích lũy thông tin
qua nhiều time step mà không tác động đến phần còn lại của mạng
(miễn là output gate nhận các giá trị gần bằng 0),
sau đó đột ngột tác động đến mạng tại một time step tiếp theo
ngay khi output gate chuyển từ các giá trị gần bằng 0
sang các giá trị gần bằng 1. :numref:`fig_lstm_3` minh họa bằng hình ảnh luồng dữ liệu.

![Tính hidden state trong mô hình LSTM.](../img/lstm-3.svg)
:label:`fig_lstm_3`



## Triển khai từ đầu

Bây giờ, chúng ta sẽ triển khai một LSTM từ đầu.
Tương tự như các thí nghiệm trong :numref:`sec_rnn-scratch`,
trước tiên chúng ta tải dataset *The Time Machine*.

### [**Khởi tạo các tham số của mô hình**]

Tiếp theo, chúng ta cần xác định và khởi tạo các tham số của mô hình.
Như trước đây, hyperparameter `num_hiddens`
quy định số lượng hidden unit.
Chúng ta khởi tạo các trọng số theo phân phối Gaussian
với độ lệch chuẩn 0.01,
và đặt các bias bằng 0.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LSTMScratch(d2l.Module):
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

        self.W_xi, self.W_hi, self.b_i = triple()  # Input gate
        self.W_xf, self.W_hf, self.b_f = triple()  # Forget gate
        self.W_xo, self.W_ho, self.b_o = triple()  # Output gate
        self.W_xc, self.W_hc, self.b_c = triple()  # Input node
```

```{.python .input}
%%tab jax
class LSTMScratch(d2l.Module):
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

        self.W_xi, self.W_hi, self.b_i = triple('i')  # Input gate
        self.W_xf, self.W_hf, self.b_f = triple('f')  # Forget gate
        self.W_xo, self.W_ho, self.b_o = triple('o')  # Output gate
        self.W_xc, self.W_hc, self.b_c = triple('c')  # Input node
```

:begin_tab:`pytorch, mxnet, tensorflow`
[**Mô hình thực tế**] được định nghĩa như mô tả ở trên,
bao gồm ba gate và một input node.
Lưu ý rằng chỉ hidden state được truyền đến output layer.
:end_tab:

:begin_tab:`jax`
[**Mô hình thực tế**] được định nghĩa như mô tả ở trên,
bao gồm ba gate và một input node.
Lưu ý rằng chỉ hidden state được truyền đến output layer.
Một vòng lặp for dài trong phương thức `forward` sẽ dẫn đến thời gian
biên dịch JIT cực kỳ dài trong lần chạy đầu tiên. Để giải quyết vấn đề này, thay vì sử dụng
vòng lặp for để cập nhật state ở mỗi time step,
JAX có phép biến đổi tiện ích `jax.lax.scan` để đạt được cùng hành vi.
Nó nhận một state ban đầu có tên là `carry` và một mảng `inputs`,
được quét theo axis đầu tiên của nó. Phép biến đổi `scan` cuối cùng
trả về state cuối cùng và các output được xếp chồng như mong đợi.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LSTMScratch)
def forward(self, inputs, H_C=None):
    if H_C is None:
        # Initial state with shape: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          ctx=inputs.ctx)
            C = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          ctx=inputs.ctx)
        if tab.selected('pytorch'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
            C = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
        if tab.selected('tensorflow'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens))
            C = d2l.zeros((inputs.shape[1], self.num_hiddens))
    else:
        H, C = H_C
    outputs = []
    for X in inputs:
        I = d2l.sigmoid(d2l.matmul(X, self.W_xi) +
                        d2l.matmul(H, self.W_hi) + self.b_i)
        F = d2l.sigmoid(d2l.matmul(X, self.W_xf) +
                        d2l.matmul(H, self.W_hf) + self.b_f)
        O = d2l.sigmoid(d2l.matmul(X, self.W_xo) +
                        d2l.matmul(H, self.W_ho) + self.b_o)
        C_tilde = d2l.tanh(d2l.matmul(X, self.W_xc) +
                           d2l.matmul(H, self.W_hc) + self.b_c)
        C = F * C + I * C_tilde
        H = O * d2l.tanh(C)
        outputs.append(H)
    return outputs, (H, C)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LSTMScratch)
def forward(self, inputs, H_C=None):
    # Use lax.scan primitive instead of looping over the
    # inputs, since scan saves time in jit compilation.
    def scan_fn(carry, X):
        H, C = carry
        I = d2l.sigmoid(d2l.matmul(X, self.W_xi) + (
            d2l.matmul(H, self.W_hi)) + self.b_i)
        F = d2l.sigmoid(d2l.matmul(X, self.W_xf) +
                        d2l.matmul(H, self.W_hf) + self.b_f)
        O = d2l.sigmoid(d2l.matmul(X, self.W_xo) +
                        d2l.matmul(H, self.W_ho) + self.b_o)
        C_tilde = d2l.tanh(d2l.matmul(X, self.W_xc) +
                           d2l.matmul(H, self.W_hc) + self.b_c)
        C = F * C + I * C_tilde
        H = O * d2l.tanh(C)
        return (H, C), H  # return carry, y

    if H_C is None:
        batch_size = inputs.shape[1]
        carry = jnp.zeros((batch_size, self.num_hiddens)), \
                jnp.zeros((batch_size, self.num_hiddens))
    else:
        carry = H_C

    # scan takes the scan_fn, initial carry state, xs with leading axis to be scanned
    carry, outputs = jax.lax.scan(scan_fn, carry, inputs)
    return outputs, carry
```

### [**Huấn luyện**] và dự đoán

Hãy huấn luyện một mô hình LSTM bằng cách khởi tạo class `RNNLMScratch` từ :numref:`sec_rnn-scratch`.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1)
trainer.fit(model, data)
```

## [**Triển khai cô đọng**]

Sử dụng các API cấp cao,
chúng ta có thể khởi tạo trực tiếp một mô hình LSTM.
Mô hình này đóng gói tất cả chi tiết cấu hình
mà chúng ta đã trình bày tường minh ở trên.
Code chạy nhanh hơn đáng kể vì sử dụng
các operator đã được biên dịch thay vì Python
cho nhiều chi tiết mà trước đó chúng ta phải viết rõ ràng.

```{.python .input}
%%tab mxnet
class LSTM(d2l.RNN):
    def __init__(self, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = rnn.LSTM(num_hiddens)

    def forward(self, inputs, H_C=None):
        if H_C is None: H_C = self.rnn.begin_state(
            inputs.shape[1], ctx=inputs.ctx)
        return self.rnn(inputs, H_C)
```

```{.python .input}
%%tab pytorch
class LSTM(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.LSTM(num_inputs, num_hiddens)

    def forward(self, inputs, H_C=None):
        return self.rnn(inputs, H_C)
```

```{.python .input}
%%tab tensorflow
class LSTM(d2l.RNN):
    def __init__(self, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = tf.keras.layers.LSTM(
                num_hiddens, return_sequences=True,
                return_state=True, time_major=True)

    def forward(self, inputs, H_C=None):
        outputs, *H_C = self.rnn(inputs, H_C)
        return outputs, H_C
```

```{.python .input}
%%tab jax
class LSTM(d2l.RNN):
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H_C=None, training=False):
        if H_C is None:
            batch_size = inputs.shape[1]
            H_C = nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0),
                                                        (batch_size,),
                                                        self.num_hiddens)

        LSTM = nn.scan(nn.OptimizedLSTMCell, variable_broadcast="params",
                       in_axes=0, out_axes=0, split_rngs={"params": False})

        H_C, outputs = LSTM()(H_C, inputs)
        return outputs, H_C
```

```{.python .input}
%%tab all
if tab.selected('pytorch'):
    lstm = LSTM(num_inputs=len(data.vocab), num_hiddens=32)
if tab.selected('mxnet', 'tensorflow', 'jax'):
    lstm = LSTM(num_hiddens=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    model = d2l.RNNLM(lstm, vocab_size=len(data.vocab), lr=4)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(lstm, vocab_size=len(data.vocab), lr=4)
trainer.fit(model, data)
```

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

LSTM là mô hình autoregressive biến ẩn điển hình với cơ chế kiểm soát state phức tạp.
Trong nhiều năm qua, nhiều biến thể của mô hình này đã được đề xuất, chẳng hạn như nhiều layer, residual connection và các kiểu regularization khác nhau. Tuy nhiên, việc huấn luyện LSTM và các mô hình chuỗi khác (chẳng hạn GRU) khá tốn kém do dependency dài hạn của chuỗi.
Sau này, chúng ta sẽ tìm hiểu các mô hình thay thế như Transformer, có thể được sử dụng trong một số trường hợp.


## Tóm tắt

Mặc dù LSTM được công bố vào năm 1997,
chúng trở nên nổi tiếng rộng rãi
sau một số thành công trong các cuộc thi dự đoán vào giữa những năm 2000,
và trở thành mô hình thống trị cho việc học chuỗi từ năm 2011
cho đến khi các mô hình Transformer nổi lên, bắt đầu từ năm 2017.
Ngay cả Transformer cũng kế thừa một số ý tưởng then chốt
từ những đổi mới trong thiết kế kiến trúc do LSTM giới thiệu.


LSTM có ba loại gate:
input gate, forget gate và output gate
kiểm soát luồng thông tin.
Đầu ra của hidden layer trong LSTM bao gồm hidden state và trạng thái nội bộ của memory cell.
Chỉ hidden state được truyền vào output layer, còn
trạng thái nội bộ của memory cell hoàn toàn vẫn ở bên trong.
LSTM có thể giảm thiểu vanishing gradient và exploding gradient.



## Bài tập

1. Điều chỉnh các hyperparameter và phân tích ảnh hưởng của chúng đến thời gian chạy, perplexity và chuỗi đầu ra.
1. Bạn cần thay đổi mô hình như thế nào để tạo ra các từ hoàn chỉnh thay vì chỉ các chuỗi ký tự?
1. So sánh chi phí tính toán của GRU, LSTM và RNN thông thường với cùng một số chiều hidden. Đặc biệt chú ý đến chi phí huấn luyện và suy luận.
1. Vì candidate memory cell đảm bảo miền giá trị nằm trong khoảng từ $-1$ đến $1$ bằng cách sử dụng hàm $\tanh$, tại sao hidden state lại cần sử dụng hàm $\tanh$ một lần nữa để đảm bảo miền giá trị đầu ra nằm trong khoảng từ $-1$ đến $1$?
1. Triển khai một mô hình LSTM để dự đoán time series thay vì dự đoán chuỗi ký tự.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/343)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1057)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/3861)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18016)
:end_tab: