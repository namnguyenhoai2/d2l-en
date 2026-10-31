# Triển khai Ngắn gọn Mạng Nơ-ron Hồi quy
:label:`sec_rnn-concise`

Giống như hầu hết các phần triển khai từ đầu của chúng ta,
:numref:`sec_rnn-scratch` được thiết kế
để giúp bạn hiểu rõ cách từng thành phần hoạt động.
Nhưng khi bạn sử dụng RNN hằng ngày
hoặc viết mã production,
bạn sẽ muốn phụ thuộc nhiều hơn vào các thư viện
giúp giảm cả thời gian triển khai
(bằng cách cung cấp mã thư viện cho các mô hình và hàm phổ biến)
lẫn thời gian tính toán
(bằng cách tối ưu triệt để các phần triển khai thư viện này).
Phần này sẽ chỉ cho bạn cách triển khai
cùng một mô hình ngôn ngữ hiệu quả hơn
bằng API cấp cao được cung cấp
bởi framework deep learning của bạn.
Như trước đây, chúng ta bắt đầu bằng việc tải
dataset *The Time Machine*.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
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
from jax import numpy as jnp
```

## [**Định nghĩa Mô hình**]

Chúng ta định nghĩa lớp sau
bằng RNN được triển khai
bởi các API cấp cao.

:begin_tab:`mxnet`
Cụ thể, để khởi tạo hidden state,
chúng ta gọi phương thức thành viên `begin_state`.
Phương thức này trả về một danh sách chứa
một hidden state ban đầu
cho mỗi ví dụ trong minibatch,
có shape là
(số hidden layer, kích thước batch, số hidden unit).
Với một số mô hình sẽ được giới thiệu sau
(ví dụ: long short-term memory),
danh sách này cũng sẽ chứa thông tin khác.
:end_tab:

:begin_tab:`jax`
Tại thời điểm hiện tại, Flax không cung cấp RNNCell để triển khai ngắn gọn Vanilla RNN.
Có các biến thể RNN nâng cao hơn như LSTM và GRU
được cung cấp trong API `linen` của Flax.
:end_tab:

```{.python .input}
%%tab mxnet
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()        
        self.rnn = rnn.RNN(num_hiddens)
        
    def forward(self, inputs, H=None):
        if H is None:
            H, = self.rnn.begin_state(inputs.shape[1], ctx=inputs.ctx)
        outputs, (H, ) = self.rnn(inputs, (H, ))
        return outputs, H
```

```{.python .input}
%%tab pytorch
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)
        
    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

```{.python .input}
%%tab tensorflow
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()            
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, return_sequences=True, return_state=True,
            time_major=True)
        
    def forward(self, inputs, H=None):
        outputs, H = self.rnn(inputs, H)
        return outputs, H
```

```{.python .input}
%%tab jax
class RNN(nn.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H=None):
        raise NotImplementedError
```

Kế thừa từ lớp `RNNLMScratch` trong :numref:`sec_rnn-scratch`, 
lớp `RNNLM` sau đây định nghĩa một mô hình ngôn ngữ hoàn chỉnh dựa trên RNN.
Lưu ý rằng chúng ta cần tạo một fully connected output layer riêng biệt.

```{.python .input}
%%tab pytorch
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)
        
    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        if tab.selected('mxnet'):
            self.linear = nn.Dense(self.vocab_size, flatten=False)
            self.initialize()
        if tab.selected('tensorflow'):
            self.linear = tf.keras.layers.Dense(self.vocab_size)
        
    def output_layer(self, hiddens):
        if tab.selected('mxnet'):
            return d2l.swapaxes(self.linear(hiddens), 0, 1)        
        if tab.selected('tensorflow'):
            return d2l.transpose(self.linear(hiddens), (1, 0, 2))
```

```{.python .input}
%%tab jax
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    training: bool = True

    def setup(self):
        self.linear = nn.Dense(self.vocab_size)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state, self.training)
        return self.output_layer(rnn_outputs)
```

## Huấn luyện và Dự đoán

Trước khi huấn luyện mô hình, hãy [**đưa ra một dự đoán
bằng mô hình được khởi tạo với trọng số ngẫu nhiên.**]
Do chúng ta chưa huấn luyện network,
nó sẽ tạo ra các dự đoán vô nghĩa.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'tensorflow'):
    rnn = RNN(num_hiddens=32)
if tab.selected('pytorch'):
    rnn = RNN(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
model.predict('it has', 20, data.vocab)
```

Tiếp theo, chúng ta [**huấn luyện mô hình bằng cách tận dụng API cấp cao**].

```{.python .input}
%%tab pytorch, mxnet, tensorflow
if tab.selected('mxnet', 'pytorch'):
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

So với :numref:`sec_rnn-scratch`,
mô hình này đạt được perplexity tương đương,
nhưng chạy nhanh hơn nhờ các phần triển khai được tối ưu.
Như trước đây, chúng ta có thể tạo các token dự đoán
theo chuỗi tiền tố đã chỉ định.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

## Tóm tắt

Các API cấp cao trong framework deep learning cung cấp các phần triển khai RNN tiêu chuẩn.
Những thư viện này giúp bạn tránh lãng phí thời gian triển khai lại các mô hình tiêu chuẩn.
Hơn nữa,
các phần triển khai của framework thường được tối ưu cao,
  đem lại mức tăng hiệu năng tính toán đáng kể
  so với các phần triển khai từ đầu.

## Bài tập

1. Bạn có thể làm cho mô hình RNN overfit bằng các API cấp cao không?
1. Triển khai mô hình autoregressive của :numref:`sec_sequence` bằng RNN.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1053)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/2211)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18015)
:end_tab: