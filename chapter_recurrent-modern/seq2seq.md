```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

#  Học sequence-to-sequence cho dịch máy
:label:`sec_seq2seq`

Trong các bài toán được gọi là sequence-to-sequence, chẳng hạn như dịch máy
(như đã thảo luận trong :numref:`sec_machine_translation`),
trong đó đầu vào và đầu ra đều là các chuỗi không căn chỉnh có độ dài thay đổi,
chúng ta thường dựa vào các kiến trúc encoder--decoder
(:numref:`sec_encoder-decoder`).
Trong phần này,
chúng ta sẽ minh họa việc áp dụng
một kiến trúc encoder--decoder,
trong đó cả encoder và decoder
đều được triển khai dưới dạng RNN,
cho bài toán dịch máy
:cite:`Sutskever.Vinyals.Le.2014,Cho.Van-Merrienboer.Gulcehre.ea.2014`.

Ở đây, RNN encoder sẽ nhận một chuỗi có độ dài thay đổi làm đầu vào
và chuyển đổi chuỗi đó thành một hidden state có hình dạng cố định.
Sau đó, trong :numref:`chap_attention-and-transformers`,
chúng ta sẽ giới thiệu các cơ chế attention,
cho phép truy cập các đầu vào đã được mã hóa
mà không cần nén toàn bộ đầu vào
thành một biểu diễn có độ dài cố định duy nhất.

Sau đó, để tạo chuỗi đầu ra,
mỗi lần một token,
mô hình decoder,
bao gồm một RNN riêng biệt,
sẽ dự đoán từng token đích kế tiếp
dựa trên cả chuỗi đầu vào
và các token đứng trước trong đầu ra.
Trong quá trình huấn luyện, decoder thường
được điều kiện hóa bởi các token đứng trước
trong nhãn "ground truth" chính thức.
Tuy nhiên, tại thời điểm kiểm thử, chúng ta muốn điều kiện hóa
mỗi đầu ra của decoder dựa trên các token đã được dự đoán.
Lưu ý rằng nếu bỏ qua encoder,
decoder trong một kiến trúc sequence-to-sequence
hoạt động giống hệt một language model thông thường.
:numref:`fig_seq2seq` minh họa
cách sử dụng hai RNN
cho việc học sequence-to-sequence
trong dịch máy.


![Học sequence-to-sequence với encoder RNN và decoder RNN.](../img/seq2seq.svg)
:label:`fig_seq2seq`

Trong :numref:`fig_seq2seq`,
token đặc biệt "&lt;eos&gt;"
đánh dấu kết thúc chuỗi.
Mô hình của chúng ta có thể dừng dự đoán
sau khi token này được tạo ra.
Tại bước thời gian ban đầu của decoder RNN,
có hai quyết định thiết kế đặc biệt cần lưu ý:
Thứ nhất, chúng ta bắt đầu mọi đầu vào bằng một
token đặc biệt đánh dấu bắt đầu chuỗi "&lt;bos&gt;".
Thứ hai, chúng ta có thể đưa
hidden state cuối cùng của encoder
vào decoder
tại mọi bước thời gian giải mã :cite:`Cho.Van-Merrienboer.Gulcehre.ea.2014`.
Trong một số thiết kế khác,
chẳng hạn như thiết kế của :citet:`Sutskever.Vinyals.Le.2014`,
hidden state cuối cùng của RNN encoder
chỉ được sử dụng
để khởi tạo hidden state của decoder
tại bước giải mã đầu tiên.

```{.python .input}
%%tab mxnet
import collections
from d2l import mxnet as d2l
import math
from mxnet import np, npx, init, gluon, autograd
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import collections
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
%%tab jax
import collections
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
import jax
from jax import numpy as jnp
import math
import optax
```

## Teacher Forcing

Mặc dù việc chạy encoder trên chuỗi đầu vào
tương đối đơn giản,
việc xử lý đầu vào và đầu ra
của decoder đòi hỏi sự cẩn thận hơn.
Cách tiếp cận phổ biến nhất đôi khi được gọi là *teacher forcing*.
Ở đây, chuỗi đích ban đầu (các nhãn token)
được đưa vào decoder làm đầu vào.
Cụ thể hơn,
token đặc biệt đánh dấu bắt đầu chuỗi
và chuỗi đích ban đầu,
không bao gồm token cuối cùng,
được nối với nhau để làm đầu vào cho decoder,
trong khi đầu ra của decoder (các nhãn dùng để huấn luyện) là
chuỗi đích ban đầu,
được dịch đi một token:
"&lt;bos&gt;", "Ils", "regardent", "." $\rightarrow$
"Ils", "regardent", ".", "&lt;eos&gt;" (:numref:`fig_seq2seq`).

Cài đặt của chúng ta trong
:numref:`subsec_loading-seq-fixed-len`
đã chuẩn bị dữ liệu huấn luyện cho teacher forcing,
trong đó việc dịch các token để học tự giám sát
tương tự như quá trình huấn luyện language model trong
:numref:`sec_language-model`.
Một cách tiếp cận khác là
đưa token *được dự đoán*
từ bước thời gian trước
làm đầu vào hiện tại cho decoder.


Sau đây, chúng ta giải thích chi tiết hơn
thiết kế được mô tả trong :numref:`fig_seq2seq`.
Chúng ta sẽ huấn luyện mô hình này cho bài toán dịch máy
trên tập dữ liệu Anh--Pháp được giới thiệu trong
:numref:`sec_machine_translation`.

## Encoder

Hãy nhớ rằng encoder chuyển đổi một chuỗi đầu vào có độ dài thay đổi
thành một *biến ngữ cảnh* có hình dạng cố định $\mathbf{c}$ (xem :numref:`fig_seq2seq`).


Xét một ví dụ chuỗi đơn (batch size bằng 1).
Giả sử chuỗi đầu vào là $x_1, \ldots, x_T$,
trong đó $x_t$ là token thứ $t^{\textrm{th}}$.
Tại bước thời gian $t$, RNN chuyển đổi
vector đặc trưng đầu vào $\mathbf{x}_t$ của $x_t$
và hidden state $\mathbf{h} _{t-1}$
từ bước thời gian trước
thành hidden state hiện tại $\mathbf{h}_t$.
Chúng ta có thể sử dụng một hàm $f$ để biểu diễn
phép biến đổi của recurrent layer trong RNN:

$$\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1}). $$

Nói chung, encoder chuyển đổi
các hidden state tại mọi bước thời gian
thành một biến ngữ cảnh thông qua một hàm tùy chỉnh $q$:

$$\mathbf{c} =  q(\mathbf{h}_1, \ldots, \mathbf{h}_T).$$

Ví dụ, trong :numref:`fig_seq2seq`,
biến ngữ cảnh chính là hidden state $\mathbf{h}_T$
tương ứng với biểu diễn của RNN encoder
sau khi xử lý token cuối cùng của chuỗi đầu vào.

Trong ví dụ này, chúng ta đã sử dụng một RNN một chiều
để thiết kế encoder,
trong đó hidden state chỉ phụ thuộc vào chuỗi con đầu vào
tại và trước bước thời gian của hidden state.
Chúng ta cũng có thể xây dựng encoder bằng RNN hai chiều.
Trong trường hợp này, một hidden state phụ thuộc vào chuỗi con trước và sau bước thời gian
(bao gồm cả đầu vào tại bước thời gian hiện tại),
nhờ đó mã hóa thông tin của toàn bộ chuỗi.


Bây giờ, hãy [**triển khai RNN encoder**].
Lưu ý rằng chúng ta sử dụng một *embedding layer*
để thu được vector đặc trưng cho mỗi token trong chuỗi đầu vào.
Trọng số của một embedding layer là một ma trận,
trong đó số hàng tương ứng với
kích thước của vocabulary đầu vào (`vocab_size`)
và số cột tương ứng với
số chiều của vector đặc trưng (`embed_size`).
Với mọi chỉ số token đầu vào $i$,
embedding layer lấy hàng thứ $i^{\textrm{th}}$ (bắt đầu từ 0)
của ma trận trọng số
để trả về vector đặc trưng của token đó.
Ở đây, chúng ta triển khai encoder bằng một GRU nhiều layer.

```{.python .input}
%%tab mxnet
class Seq2SeqEncoder(d2l.Encoder):  #@save
    """The RNN encoder for sequence-to-sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
        self.initialize(init.Xavier())
            
    def forward(self, X, *args):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(d2l.transpose(X))
        # embs shape: (num_steps, batch_size, embed_size)    
        outputs, state = self.rnn(embs)
        # outputs shape: (num_steps, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

```{.python .input}
%%tab pytorch
def init_seq2seq(module):  #@save
    """Initialize weights for sequence-to-sequence learning."""
    if type(module) == nn.Linear:
         nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])
```

```{.python .input}
%%tab pytorch
class Seq2SeqEncoder(d2l.Encoder):  #@save
    """The RNN encoder for sequence-to-sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size, num_hiddens, num_layers, dropout)
        self.apply(init_seq2seq)
            
    def forward(self, X, *args):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int64))
        # embs shape: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs)
        # outputs shape: (num_steps, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

```{.python .input}
%%tab tensorflow
class Seq2SeqEncoder(d2l.Encoder):  #@save
    """The RNN encoder for sequence-to-sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
            
    def call(self, X, *args):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(d2l.transpose(X))
        # embs shape: (num_steps, batch_size, embed_size)    
        outputs, state = self.rnn(embs)
        # outputs shape: (num_steps, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

```{.python .input}
%%tab jax
class Seq2SeqEncoder(d2l.Encoder):  #@save
    """The RNN encoder for sequence-to-sequence learning."""
    vocab_size: int
    embed_size: int
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.embed_size)
        self.rnn = d2l.GRU(self.num_hiddens, self.num_layers, self.dropout)

    def __call__(self, X, *args, training=False):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int32))
        # embs shape: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs, training=training)
        # outputs shape: (num_steps, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

Hãy sử dụng một ví dụ cụ thể
để [**minh họa cách triển khai encoder ở trên.**]
Dưới đây, chúng ta khởi tạo một encoder GRU hai layer
có số hidden unit là 16.
Với một minibatch gồm các đầu vào chuỗi `X`
(batch size $=4$; số bước thời gian $=9$),
các hidden state của layer cuối cùng
tại mọi bước thời gian
(`enc_outputs` được trả về bởi các recurrent layer của encoder)
là một tensor có hình dạng
(số bước thời gian, batch size, số hidden unit).

```{.python .input}
%%tab all
vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 9
encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
X = d2l.zeros((batch_size, num_steps))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    enc_outputs, enc_state = encoder(X)
if tab.selected('jax'):
    (enc_outputs, enc_state), _ = encoder.init_with_output(d2l.get_key(), X)

d2l.check_shape(enc_outputs, (num_steps, batch_size, num_hiddens))
```

Vì ở đây chúng ta sử dụng GRU,
hình dạng của các hidden state nhiều layer
tại bước thời gian cuối cùng là
(số hidden layer, batch size, số hidden unit).

```{.python .input}
%%tab all
if tab.selected('mxnet', 'pytorch', 'jax'):
    d2l.check_shape(enc_state, (num_layers, batch_size, num_hiddens))
if tab.selected('tensorflow'):
    d2l.check_len(enc_state, num_layers)
    d2l.check_shape(enc_state[0], (batch_size, num_hiddens))
```

## [**Decoder**]
:label:`sec_seq2seq_decoder`

Với một chuỗi đầu ra đích $y_1, y_2, \ldots, y_{T'}$
tại mỗi bước thời gian $t'$
(chúng ta sử dụng $t^\prime$ để phân biệt với các bước thời gian của chuỗi đầu vào),
decoder gán một xác suất dự đoán
cho mỗi token có thể xuất hiện tại bước $y_{t'+1}$
dựa trên các token trước đó trong đích
$y_1, \ldots, y_{t'}$
và biến ngữ cảnh
$\mathbf{c}$, tức là $P(y_{t'+1} \mid y_1, \ldots, y_{t'}, \mathbf{c})$.

Để dự đoán token tiếp theo $t^\prime+1$ trong chuỗi đích,
RNN decoder nhận token đích của bước trước $y_{t^\prime}$,
hidden state của RNN từ bước thời gian trước $\mathbf{s}_{t^\prime-1}$,
và biến ngữ cảnh $\mathbf{c}$ làm đầu vào,
sau đó chuyển đổi chúng thành hidden state
$\mathbf{s}_{t^\prime}$ tại bước thời gian hiện tại.
Chúng ta có thể sử dụng một hàm $g$ để biểu diễn
phép biến đổi của hidden layer trong decoder:

$$\mathbf{s}_{t^\prime} = g(y_{t^\prime-1}, \mathbf{c}, \mathbf{s}_{t^\prime-1}).$$
:eqlabel:`eq_seq2seq_s_t`

Sau khi thu được hidden state của decoder,
chúng ta có thể sử dụng một output layer và phép toán softmax
để tính phân phối dự đoán
$p(y_{t^{\prime}+1} \mid y_1, \ldots, y_{t^\prime}, \mathbf{c})$
trên token đầu ra tiếp theo ${t^\prime+1}$.

Theo :numref:`fig_seq2seq`,
khi triển khai decoder như sau,
chúng ta trực tiếp sử dụng hidden state tại bước thời gian cuối cùng
của encoder
để khởi tạo hidden state của decoder.
Điều này yêu cầu
RNN encoder và RNN decoder
có cùng số layer và hidden unit.
Để đưa thêm thông tin của chuỗi đầu vào đã mã hóa,
biến ngữ cảnh được nối
với đầu vào của decoder tại mọi bước thời gian.
Để dự đoán phân phối xác suất của token đầu ra,
chúng ta sử dụng một fully connected layer
để chuyển đổi hidden state
tại layer cuối cùng của RNN decoder.

```{.python .input}
%%tab mxnet
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)
        self.initialize(init.Xavier())
            
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs 

    def forward(self, X, state):
        # X shape: (batch_size, num_steps)
        # embs shape: (num_steps, batch_size, embed_size)
        embs = self.embedding(d2l.transpose(X))
        enc_output, hidden_state = state
        # context shape: (batch_size, num_hiddens)
        context = enc_output[-1]
        # Broadcast context to (num_steps, batch_size, num_hiddens)
        context = np.tile(context, (embs.shape[0], 1, 1))
        # Concat at the feature dimension
        embs_and_context = d2l.concat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = d2l.swapaxes(self.dense(outputs), 0, 1)
        # outputs shape: (batch_size, num_steps, vocab_size)
        # hidden_state shape: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

```{.python .input}
%%tab pytorch
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size+num_hiddens, num_hiddens,
                           num_layers, dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)
            
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def forward(self, X, state):
        # X shape: (batch_size, num_steps)
        # embs shape: (num_steps, batch_size, embed_size)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int32))
        enc_output, hidden_state = state
        # context shape: (batch_size, num_hiddens)
        context = enc_output[-1]
        # Broadcast context to (num_steps, batch_size, num_hiddens)
        context = context.repeat(embs.shape[0], 1, 1)
        # Concat at the feature dimension
        embs_and_context = d2l.concat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = d2l.swapaxes(self.dense(outputs), 0, 1)
        # outputs shape: (batch_size, num_steps, vocab_size)
        # hidden_state shape: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

```{.python .input}
%%tab tensorflow
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
        self.dense = tf.keras.layers.Dense(vocab_size)
            
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def call(self, X, state):
        # X shape: (batch_size, num_steps)
        # embs shape: (num_steps, batch_size, embed_size)
        embs = self.embedding(d2l.transpose(X))
        enc_output, hidden_state = state
        # context shape: (batch_size, num_hiddens)
        context = enc_output[-1]
        # Broadcast context to (num_steps, batch_size, num_hiddens)
        context = tf.tile(tf.expand_dims(context, 0), (embs.shape[0], 1, 1))
        # Concat at the feature dimension
        embs_and_context = d2l.concat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = d2l.transpose(self.dense(outputs), (1, 0, 2))
        # outputs shape: (batch_size, num_steps, vocab_size)
        # hidden_state shape: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

```{.python .input}
%%tab jax
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    vocab_size: int
    embed_size: int
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.embed_size)
        self.rnn = d2l.GRU(self.num_hiddens, self.num_layers, self.dropout)
        self.dense = nn.Dense(self.vocab_size)

    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def __call__(self, X, state, training=False):
        # X shape: (batch_size, num_steps)
        # embs shape: (num_steps, batch_size, embed_size)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int32))
        enc_output, hidden_state = state
        # context shape: (batch_size, num_hiddens)
        context = enc_output[-1]
        # Broadcast context to (num_steps, batch_size, num_hiddens)
        context = jnp.tile(context, (embs.shape[0], 1, 1))
        # Concat at the feature dimension
        embs_and_context = d2l.concat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state,
                                         training=training)
        outputs = d2l.swapaxes(self.dense(outputs), 0, 1)
        # outputs shape: (batch_size, num_steps, vocab_size)
        # hidden_state shape: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

Để [**minh họa decoder đã triển khai**],
dưới đây chúng ta khởi tạo decoder với cùng các hyperparameter như encoder nói trên.
Như có thể thấy, hình dạng đầu ra của decoder trở thành (batch size, số bước thời gian, kích thước vocabulary),
trong đó chiều cuối cùng của tensor lưu phân phối token được dự đoán.

```{.python .input}
%%tab all
decoder = Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers)
if tab.selected('mxnet', 'pytorch', 'tensorflow'):
    state = decoder.init_state(encoder(X))
    dec_outputs, state = decoder(X, state)
if tab.selected('jax'):
    state = decoder.init_state(encoder.init_with_output(d2l.get_key(), X)[0])
    (dec_outputs, state), _ = decoder.init_with_output(d2l.get_key(), X,
                                                       state)


d2l.check_shape(dec_outputs, (batch_size, num_steps, vocab_size))
if tab.selected('mxnet', 'pytorch', 'jax'):
    d2l.check_shape(state[1], (num_layers, batch_size, num_hiddens))
if tab.selected('tensorflow'):
    d2l.check_len(state[1], num_layers)
    d2l.check_shape(state[1][0], (batch_size, num_hiddens))
```

Các layer trong mô hình RNN encoder--decoder ở trên
được tóm tắt trong :numref:`fig_seq2seq_details`.

![Các lớp trong mô hình encoder--decoder RNN.](../img/seq2seq-details.svg)
:label:`fig_seq2seq_details`



## Encoder--Decoder cho việc học Sequence-to-Sequence


Kết hợp tất cả lại trong code sẽ cho kết quả như sau:

```{.python .input}
%%tab pytorch, tensorflow, mxnet
class Seq2Seq(d2l.EncoderDecoder):  #@save
    """The RNN encoder--decoder for sequence to sequence learning."""
    def __init__(self, encoder, decoder, tgt_pad, lr):
        super().__init__(encoder, decoder)
        self.save_hyperparameters()
        
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        
    def configure_optimizers(self):
        # Adam optimizer is used here
        if tab.selected('mxnet'):
            return gluon.Trainer(self.parameters(), 'adam',
                                 {'learning_rate': self.lr})
        if tab.selected('pytorch'):
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        if tab.selected('tensorflow'):
            return tf.keras.optimizers.Adam(learning_rate=self.lr)
```

```{.python .input}
%%tab jax
class Seq2Seq(d2l.EncoderDecoder):  #@save
    """The RNN encoder--decoder for sequence to sequence learning."""
    encoder: nn.Module
    decoder: nn.Module
    tgt_pad: int
    lr: float

    def validation_step(self, params, batch, state):
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        # Adam optimizer is used here
        return optax.adam(learning_rate=self.lr)
```

## Hàm mất mát với Masking

Ở mỗi bước thời gian, decoder dự đoán 
một phân phối xác suất cho các token đầu ra.
Tương tự như trong language modeling, 
chúng ta có thể áp dụng softmax 
để thu được phân phối
và tính toán cross-entropy loss để tối ưu hóa.
Hãy nhớ lại từ :numref:`sec_machine_translation`
rằng các token padding đặc biệt
được thêm vào cuối các sequence
và do đó các sequence có độ dài khác nhau
có thể được tải hiệu quả
theo các minibatch có cùng shape.
Tuy nhiên, việc dự đoán các token padding
nên được loại trừ khỏi phép tính loss.
Để thực hiện điều này, chúng ta có thể 
[**mask các phần tử không liên quan bằng giá trị zero**]
để phép nhân 
của bất kỳ dự đoán không liên quan nào
với zero sẽ cho kết quả bằng zero.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(Seq2Seq)
def loss(self, Y_hat, Y):
    l = super(Seq2Seq, self).loss(Y_hat, Y, averaged=False)
    mask = d2l.astype(d2l.reshape(Y, -1) != self.tgt_pad, d2l.float32)
    return d2l.reduce_sum(l * mask) / d2l.reduce_sum(mask)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(Seq2Seq)
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=False):
    Y_hat = state.apply_fn({'params': params}, *X,
                           rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    l = fn(Y_hat, Y)
    mask = d2l.astype(d2l.reshape(Y, -1) != self.tgt_pad, d2l.float32)
    return d2l.reduce_sum(l * mask) / d2l.reduce_sum(mask), {}
```

## [**Training**]
:label:`sec_seq2seq_training`

Bây giờ chúng ta có thể [**tạo và training một mô hình encoder--decoder RNN**]
cho việc học sequence-to-sequence trên dataset machine translation.

```{.python .input}
%%tab all
data = d2l.MTFraEng(batch_size=128) 
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
if tab.selected('mxnet', 'pytorch', 'jax'):
    encoder = Seq2SeqEncoder(
        len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(
        len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
if tab.selected('mxnet', 'pytorch'):
    model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                    lr=0.005)
if tab.selected('jax'):
    model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                    lr=0.005, training=True)
if tab.selected('mxnet', 'pytorch', 'jax'):
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        encoder = Seq2SeqEncoder(
            len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
        decoder = Seq2SeqDecoder(
            len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
        model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.005)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1)
trainer.fit(model, data)
```

## [**Prediction**]

Để dự đoán sequence đầu ra
ở mỗi bước, 
token được dự đoán ở
bước thời gian trước đó được đưa vào decoder làm input.
Một chiến lược đơn giản là lấy mẫu token
được decoder gán xác suất cao nhất
khi dự đoán ở mỗi bước.
Giống như trong quá trình training, ở bước thời gian ban đầu
token bắt đầu sequence ("&lt;bos&gt;") 
được đưa vào decoder.
Quá trình dự đoán này
được minh họa trong :numref:`fig_seq2seq_predict`.
Khi token kết thúc sequence ("&lt;eos&gt;") được dự đoán,
việc dự đoán sequence đầu ra hoàn tất.


![Dự đoán sequence đầu ra từng token một bằng encoder--decoder RNN.](../img/seq2seq-predict.svg)
:label:`fig_seq2seq_predict`

Trong phần tiếp theo, chúng ta sẽ giới thiệu 
các chiến lược phức tạp hơn 
dựa trên beam search (:numref:`sec_beam-search`).

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(d2l.EncoderDecoder)  #@save
def predict_step(self, batch, device, num_steps,
                 save_attention_weights=False):
    if tab.selected('mxnet', 'pytorch'):
        batch = [d2l.to(a, device) for a in batch]
    src, tgt, src_valid_len, _ = batch
    if tab.selected('mxnet', 'pytorch'):
        enc_all_outputs = self.encoder(src, src_valid_len)
    if tab.selected('tensorflow'):
        enc_all_outputs = self.encoder(src, src_valid_len, training=False)
    dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
    outputs, attention_weights = [d2l.expand_dims(tgt[:, 0], 1), ], []
    for _ in range(num_steps):
        if tab.selected('mxnet', 'pytorch'):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
        if tab.selected('tensorflow'):
            Y, dec_state = self.decoder(outputs[-1], dec_state, training=False)
        outputs.append(d2l.argmax(Y, 2))
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weights.append(self.decoder.attention_weights)
    return d2l.concat(outputs[1:], 1), attention_weights
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.EncoderDecoder)  #@save
def predict_step(self, params, batch, num_steps,
                 save_attention_weights=False):
    src, tgt, src_valid_len, _ = batch
    enc_all_outputs, inter_enc_vars = self.encoder.apply(
        {'params': params['encoder']}, src, src_valid_len, training=False,
        mutable='intermediates')
    # Save encoder attention weights if inter_enc_vars containing encoder
    # attention weights is not empty. (to be covered later)
    enc_attention_weights = []
    if bool(inter_enc_vars) and save_attention_weights:
        # Encoder Attention Weights saved in the intermediates collection
        enc_attention_weights = inter_enc_vars[
            'intermediates']['enc_attention_weights'][0]

    dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
    outputs, attention_weights = [d2l.expand_dims(tgt[:,0], 1), ], []
    for _ in range(num_steps):
        (Y, dec_state), inter_dec_vars = self.decoder.apply(
            {'params': params['decoder']}, outputs[-1], dec_state,
            training=False, mutable='intermediates')
        outputs.append(d2l.argmax(Y, 2))
        # Save attention weights (to be covered later)
        if save_attention_weights:
            # Decoder Attention Weights saved in the intermediates collection
            dec_attention_weights = inter_dec_vars[
                'intermediates']['dec_attention_weights'][0]
            attention_weights.append(dec_attention_weights)
    return d2l.concat(outputs[1:], 1), (attention_weights,
                                        enc_attention_weights)
```

## Đánh giá các Sequence được dự đoán

Chúng ta có thể đánh giá một sequence được dự đoán
bằng cách so sánh nó với
sequence mục tiêu (ground truth).
Nhưng chính xác thì thước đo phù hợp 
để so sánh độ tương đồng giữa hai sequence là gì?


Bilingual Evaluation Understudy (BLEU),
mặc dù ban đầu được đề xuất để đánh giá kết quả
machine translation :cite:`Papineni.Roukos.Ward.ea.2002`,
đã được sử dụng rộng rãi để đo lường
chất lượng của các sequence đầu ra cho nhiều ứng dụng khác nhau.
Về nguyên tắc, với mọi $n$-gram (:numref:`subsec_markov-models-and-n-grams`) trong sequence được dự đoán,
BLEU đánh giá xem $n$-gram này có xuất hiện
trong sequence mục tiêu hay không.

Ký hiệu $p_n$ là precision của một $n$-gram,
được định nghĩa là tỷ lệ 
giữa số $n$-gram khớp nhau trong
sequence được dự đoán và sequence mục tiêu
với số $n$-gram trong sequence được dự đoán.
Cụ thể, với sequence mục tiêu $A$, $B$, $C$, $D$, $E$, $F$,
và sequence được dự đoán $A$, $B$, $B$, $C$, $D$,
ta có $p_1 = 4/5$,  $p_2 = 3/4$, $p_3 = 1/3$, và $p_4 = 0$.
Bây giờ, gọi $\textrm{len}_{\textrm{label}}$ và $\textrm{len}_{\textrm{pred}}$
lần lượt là số token trong sequence mục tiêu 
và sequence được dự đoán.
Khi đó, BLEU được định nghĩa như sau

$$ \exp\left(\min\left(0, 1 - \frac{\textrm{len}_{\textrm{label}}}{\textrm{len}_{\textrm{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},$$
:eqlabel:`eq_bleu`

trong đó $k$ là $n$-gram dài nhất được dùng để matching.

Dựa trên định nghĩa BLEU trong :eqref:`eq_bleu`,
bất cứ khi nào sequence được dự đoán giống với sequence mục tiêu, BLEU bằng 1.
Ngoài ra,
vì việc matching các $n$-gram dài hơn khó hơn,
BLEU gán trọng số lớn hơn
khi một $n$-gram dài hơn có precision cao.
Cụ thể, khi $p_n$ cố định,
$p_n^{1/2^n}$ tăng khi $n$ tăng (bài báo gốc sử dụng $p_n^{1/n}$).
Hơn nữa,
vì
việc dự đoán các sequence ngắn hơn
thường cho giá trị $p_n$ cao hơn,
hệ số đứng trước số hạng tích trong :eqref:`eq_bleu`
sẽ phạt các sequence được dự đoán ngắn hơn.
Ví dụ, khi $k=2$,
với sequence mục tiêu $A$, $B$, $C$, $D$, $E$, $F$ và sequence được dự đoán $A$, $B$,
mặc dù $p_1 = p_2 = 1$, hệ số phạt $\exp(1-6/2) \approx 0.14$ làm giảm BLEU.

Chúng ta [**implement thước đo BLEU**] như sau.

```{.python .input}
%%tab all
def bleu(pred_seq, label_seq, k):  #@save
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```

Cuối cùng,
chúng ta sử dụng encoder--decoder RNN đã được training
để [**dịch một vài câu tiếng Anh sang tiếng Pháp**]
và tính BLEU của các kết quả.

```{.python .input}
%%tab all
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds, _ = model.predict_step(
        data.build(engs, fras), d2l.try_gpu(), data.num_steps)
if tab.selected('jax'):
    preds, _ = model.predict_step(trainer.state.params, data.build(engs, fras),
                                  data.num_steps)
for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)        
    print(f'{en} => {translation}, bleu,'
          f'{bleu(" ".join(translation), fr, k=2):.3f}')
```

## Tóm tắt

Theo thiết kế của kiến trúc encoder--decoder, chúng ta có thể sử dụng hai RNN để thiết kế một mô hình cho việc học sequence-to-sequence.
Trong quá trình training encoder--decoder, phương pháp teacher forcing đưa các sequence đầu ra ban đầu (thay vì các dự đoán) vào decoder.
Khi triển khai encoder và decoder, chúng ta có thể sử dụng các RNN nhiều lớp.
Chúng ta có thể sử dụng mask để lọc các phép tính không liên quan, chẳng hạn như khi tính loss.
Để đánh giá các sequence đầu ra,
BLEU là một thước đo phổ biến, dùng để matching các $n$-gram giữa sequence được dự đoán và sequence mục tiêu.


## Bài tập

1. Bạn có thể điều chỉnh các hyperparameter để cải thiện kết quả dịch không?
1. Chạy lại experiment mà không sử dụng mask trong phép tính loss. Bạn quan sát được kết quả gì? Tại sao?
1. Nếu encoder và decoder khác nhau về số lớp hoặc số hidden unit, làm thế nào chúng ta có thể khởi tạo hidden state của decoder?
1. Trong quá trình training, hãy thay teacher forcing bằng việc đưa dự đoán ở bước thời gian trước đó vào decoder. Điều này ảnh hưởng thế nào đến performance?
1. Chạy lại experiment bằng cách thay GRU bằng LSTM.
1. Có cách nào khác để thiết kế output layer của decoder không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/345)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1062)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/3865)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18022)
:end_tab: