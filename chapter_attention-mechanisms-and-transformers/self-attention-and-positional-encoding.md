```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Self-Attention và Positional Encoding
:label:`sec_self-attention-and-positional-encoding`

Trong deep learning, chúng ta thường sử dụng CNN hoặc RNN để mã hóa các chuỗi.
Bây giờ, với cơ chế attention,
hãy hình dung việc đưa một chuỗi token
vào một cơ chế attention
sao cho ở mỗi bước,
mỗi token có query, key và value riêng.
Ở đây, khi tính giá trị biểu diễn của một token tại lớp kế tiếp,
token đó có thể attention (thông qua vector query của nó) đến bất kỳ token nào khác
(đối sánh dựa trên các vector key của chúng).
Bằng cách sử dụng toàn bộ tập điểm tương thích query-key,
chúng ta có thể tính, cho mỗi token, một biểu diễn
bằng cách xây dựng tổng có trọng số phù hợp
trên các token khác.
Vì mọi token đều attention đến mọi token khác
(không giống trường hợp các bước decoder attention đến các bước encoder),
các kiến trúc như vậy thường được mô tả là các mô hình *self-attention* :cite:`Lin.Feng.Santos.ea.2017,Vaswani.Shazeer.Parmar.ea.2017`, 
và ở nơi khác được mô tả là mô hình *intra-attention* :cite:`Cheng.Dong.Lapata.2016,Parikh.Tackstrom.Das.ea.2016,Paulus.Xiong.Socher.2017`.
Trong phần này, chúng ta sẽ thảo luận về việc mã hóa chuỗi bằng self-attention,
bao gồm cả việc sử dụng thông tin bổ sung về thứ tự chuỗi.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import math
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
from jax import numpy as jnp
import jax
```

## [**Self-Attention**]

Cho một chuỗi các token đầu vào
$\mathbf{x}_1, \ldots, \mathbf{x}_n$ trong đó bất kỳ $\mathbf{x}_i \in \mathbb{R}^d$ nào ($1 \leq i \leq n$),
đầu ra self-attention của nó là
một chuỗi có cùng độ dài
$\mathbf{y}_1, \ldots, \mathbf{y}_n$,
trong đó

$$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$$

theo định nghĩa về attention pooling trong
:eqref:`eq_attention_pooling`.
Sử dụng multi-head attention,
đoạn mã sau
tính self-attention của một tensor
có shape (batch size, số bước thời gian hoặc độ dài chuỗi tính theo token, $d$).
Tensor đầu ra có cùng shape.

```{.python .input}
%%tab pytorch
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab mxnet
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
%%tab jax
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab tensorflow
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab mxnet
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab tensorflow
batch_size, num_queries, valid_lens = 2, 4, tf.constant([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens, training=False),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab jax
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention.init_with_output(d2l.get_key(), X, X, X, valid_lens,
                                           training=False)[0][0],
                (batch_size, num_queries, num_hiddens))
```

## So sánh CNN, RNN và Self-Attention
:label:`subsec_cnn-rnn-self-attention`

Hãy
so sánh các kiến trúc dùng để ánh xạ
một chuỗi gồm $n$ token
sang một chuỗi khác có cùng độ dài,
trong đó mỗi token đầu vào hoặc đầu ra được biểu diễn bởi
một vector $d$ chiều.
Cụ thể,
chúng ta sẽ xem xét CNN, RNN và self-attention.
Chúng ta sẽ so sánh
độ phức tạp tính toán, 
các phép toán tuần tự,
và độ dài đường đi tối đa của chúng.
Lưu ý rằng các phép toán tuần tự ngăn cản tính toán song song,
trong khi đường đi ngắn hơn giữa
bất kỳ tổ hợp vị trí nào trong chuỗi
giúp việc học các phụ thuộc dài hạn
trong chuỗi dễ hơn :cite:`Hochreiter.Bengio.Frasconi.ea.2001`.


![So sánh các kiến trúc CNN (đã lược bỏ token padding), RNN và self-attention.](../img/cnn-rnn-self-attention.svg)
:label:`fig_cnn-rnn-self-attention`



Hãy xem bất kỳ chuỗi văn bản nào như một "ảnh một chiều". Tương tự, CNN một chiều có thể xử lý các đặc trưng cục bộ như $n$-gram trong văn bản.
Cho một chuỗi có độ dài $n$,
xét một lớp convolution có kernel size là $k$,
và số lượng input channel lẫn output channel đều là $d$.
Độ phức tạp tính toán của lớp convolution là $\mathcal{O}(knd^2)$.
Như :numref:`fig_cnn-rnn-self-attention` minh họa,
CNN có cấu trúc phân cấp,
do đó có $\mathcal{O}(1)$ phép toán tuần tự
và độ dài đường đi tối đa là $\mathcal{O}(n/k)$.
Ví dụ, $\mathbf{x}_1$ và $\mathbf{x}_5$
nằm trong receptive field của một CNN hai lớp
với kernel size 3 trong :numref:`fig_cnn-rnn-self-attention`.

Khi cập nhật hidden state của RNN,
phép nhân giữa ma trận trọng số $d \times d$
và hidden state $d$ chiều có
độ phức tạp tính toán là $\mathcal{O}(d^2)$.
Vì độ dài chuỗi là $n$,
độ phức tạp tính toán của recurrent layer
là $\mathcal{O}(nd^2)$.
Theo :numref:`fig_cnn-rnn-self-attention`,
có $\mathcal{O}(n)$ phép toán tuần tự
không thể song song hóa
và độ dài đường đi tối đa cũng là $\mathcal{O}(n)$.

Trong self-attention,
query, key và value
đều là các ma trận $n \times d$.
Xét scaled dot product attention trong
:eqref:`eq_softmax_QK_V`,
trong đó một ma trận $n \times d$ được nhân với
một ma trận $d \times n$,
sau đó ma trận đầu ra $n \times n$ được nhân
với một ma trận $n \times d$.
Do đó,
self-attention
có độ phức tạp tính toán $\mathcal{O}(n^2d)$.
Như có thể thấy từ :numref:`fig_cnn-rnn-self-attention`,
mỗi token được kết nối trực tiếp
với bất kỳ token nào khác thông qua self-attention.
Vì vậy,
việc tính toán có thể song song với $\mathcal{O}(1)$ phép toán tuần tự
và độ dài đường đi tối đa cũng là $\mathcal{O}(1)$.

Tóm lại,
cả CNN lẫn self-attention đều hỗ trợ tính toán song song
và self-attention có độ dài đường đi tối đa ngắn nhất.
Tuy nhiên, độ phức tạp tính toán bậc hai theo độ dài chuỗi
khiến self-attention chậm đến mức khó khả thi đối với các chuỗi rất dài.





## [**Positional Encoding**]
:label:`subsec_positional-encoding`


Không giống RNN, vốn xử lý lặp lại
các token của một chuỗi từng token một,
self-attention loại bỏ
các phép toán tuần tự để ưu tiên
tính toán song song.
Lưu ý rằng riêng self-attention
không bảo toàn thứ tự của chuỗi.
Chúng ta làm gì nếu thực sự quan trọng
rằng mô hình biết
chuỗi đầu vào đến theo thứ tự nào?

Cách tiếp cận chủ đạo để bảo toàn
thông tin về thứ tự của các token
là biểu diễn thông tin này cho mô hình
dưới dạng một đầu vào bổ sung
gắn với từng token.
Các đầu vào này được gọi là *positional encodings*,
và chúng có thể được học hoặc cố định *a priori*.
Bây giờ, chúng ta mô tả một phương pháp đơn giản cho positional encoding cố định
dựa trên các hàm sine và cosine :cite:`Vaswani.Shazeer.Parmar.ea.2017`.

Giả sử biểu diễn đầu vào
$\mathbf{X} \in \mathbb{R}^{n \times d}$ 
chứa các embedding $d$ chiều
cho $n$ token của một chuỗi.
Positional encoding xuất ra
$\mathbf{X} + \mathbf{P}$
bằng cách sử dụng ma trận positional embedding
$\mathbf{P} \in \mathbb{R}^{n \times d}$ có cùng shape,
trong đó phần tử tại hàng $i^\textrm{th}$ 
và cột $(2j)^\textrm{th}$
hoặc $(2j + 1)^\textrm{th}$ là

$$\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).\end{aligned}$$
:eqlabel:`eq_positional-encoding-def`

Thoạt nhìn,
thiết kế hàm lượng giác này
có vẻ kỳ lạ.
Trước khi giải thích thiết kế này,
trước hết hãy triển khai nó trong lớp `PositionalEncoding` sau đây.

```{.python .input}
%%tab mxnet
class PositionalEncoding(nn.Block):  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)
```

```{.python .input}
%%tab pytorch
class PositionalEncoding(nn.Module):  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
```

```{.python .input}
%%tab tensorflow
class PositionalEncoding(tf.keras.layers.Layer):  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # Create a long enough P
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)
        
    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)
```

```{.python .input}
%%tab jax
class PositionalEncoding(nn.Module):  #@save
    """Positional encoding."""
    num_hiddens: int
    dropout: float
    max_len: int = 1000

    def setup(self):
        # Create a long enough P
        self.P = d2l.zeros((1, self.max_len, self.num_hiddens))
        X = d2l.arange(self.max_len, dtype=jnp.float32).reshape(
            -1, 1) / jnp.power(10000, jnp.arange(
            0, self.num_hiddens, 2, dtype=jnp.float32) / self.num_hiddens)
        self.P = self.P.at[:, :, 0::2].set(jnp.sin(X))
        self.P = self.P.at[:, :, 1::2].set(jnp.cos(X))

    @nn.compact
    def __call__(self, X, training=False):
        # Flax sow API is used to capture intermediate variables
        self.sow('intermediates', 'P', self.P)
        X = X + self.P[:, :X.shape[1], :]
        return nn.Dropout(self.dropout)(X, deterministic=not training)
```

Trong ma trận positional embedding $\mathbf{P}$,
[**các hàng tương ứng với các vị trí trong một chuỗi
và các cột biểu diễn các chiều positional encoding khác nhau**].
Trong ví dụ dưới đây,
chúng ta có thể thấy rằng
các cột $6^{\textrm{th}}$ và $7^{\textrm{th}}$
của ma trận positional embedding
có tần số cao hơn
so với các cột $8^{\textrm{th}}$ và $9^{\textrm{th}}$.
Độ lệch giữa
các cột $6^{\textrm{th}}$ và $7^{\textrm{th}}$ (tương tự với các cột $8^{\textrm{th}}$ và $9^{\textrm{th}}$)
là do sự luân phiên giữa các hàm sine và cosine.

```{.python .input}
%%tab mxnet
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.initialize()
X = pos_encoding(np.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
%%tab pytorch
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(d2l.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
%%tab tensorflow
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(tf.zeros((1, num_steps, encoding_dim)), training=False)
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(np.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in np.arange(6, 10)])
```

```{.python .input}
%%tab jax
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
params = pos_encoding.init(d2l.get_key(), d2l.zeros((1, num_steps, encoding_dim)))
X, inter_vars = pos_encoding.apply(params, d2l.zeros((1, num_steps, encoding_dim)),
                                   mutable='intermediates')
P = inter_vars['intermediates']['P'][0]  # retrieve intermediate value P
P = P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

### Thông tin vị trí tuyệt đối

Để xem tần số giảm đơn điệu
theo chiều encoding liên quan thế nào đến thông tin vị trí tuyệt đối,
hãy in [**các biểu diễn nhị phân**] của $0, 1, \ldots, 7$.
Như có thể thấy, bit thấp nhất, bit thấp thứ hai,
và bit thấp thứ ba luân phiên sau mỗi số,
mỗi hai số và mỗi bốn số, tương ứng.

```{.python .input}
%%tab all
for i in range(8):
    print(f'{i} in binary is {i:>03b}')
```

Trong biểu diễn nhị phân, một bit cao hơn
có tần số thấp hơn một bit thấp hơn.
Tương tự, như được minh họa trong heat map bên dưới,
[**positional encoding làm giảm
tần số theo chiều encoding**]
bằng cách sử dụng các hàm lượng giác.
Vì đầu ra là các số float,
các biểu diễn liên tục như vậy
hiệu quả về không gian hơn
so với biểu diễn nhị phân.

```{.python .input}
%%tab mxnet
P = np.expand_dims(np.expand_dims(P[0, :, :], 0), 0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab pytorch
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab tensorflow
P = tf.expand_dims(tf.expand_dims(P[0, :, :], axis=0), axis=0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab jax
P = jnp.expand_dims(jnp.expand_dims(P[0, :, :], axis=0), axis=0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

### Thông tin vị trí tương đối

Bên cạnh việc nắm bắt thông tin vị trí tuyệt đối,
positional encoding ở trên
cũng cho phép
một mô hình dễ dàng học cách attention theo các vị trí tương đối.
Điều này là do
với bất kỳ độ lệch vị trí cố định $\delta$ nào,
positional encoding tại vị trí $i + \delta$
có thể được biểu diễn bằng một phép chiếu tuyến tính
của positional encoding tại vị trí $i$.


Phép chiếu này có thể được giải thích
bằng toán học.
Ký hiệu
$\omega_j = 1/10000^{2j/d}$,
bất kỳ cặp $(p_{i, 2j}, p_{i, 2j+1})$ nào
trong :eqref:`eq_positional-encoding-def`
có thể
được chiếu tuyến tính thành $(p_{i+\delta, 2j}, p_{i+\delta, 2j+1})$
với bất kỳ độ lệch cố định $\delta$ nào:

$$\begin{aligned}
\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\
=& 
\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
\end{aligned}$$

trong đó ma trận chiếu $2\times 2$ không phụ thuộc vào bất kỳ chỉ số vị trí $i$ nào.

## Tóm tắt

Trong self-attention, các query, key và value đều đến từ cùng một nguồn.
Cả CNN và self-attention đều hỗ trợ tính toán song song,
và self-attention có độ dài đường đi cực đại ngắn nhất.
Tuy nhiên, độ phức tạp tính toán bậc hai
theo độ dài chuỗi
khiến self-attention chậm đến mức không khả thi
đối với các chuỗi rất dài.
Để sử dụng thông tin về thứ tự chuỗi,
chúng ta có thể đưa thông tin vị trí tuyệt đối hoặc tương đối vào
bằng cách thêm positional encoding vào các biểu diễn đầu vào.

## Bài tập

1. Giả sử chúng ta thiết kế một kiến trúc sâu để biểu diễn một chuỗi bằng cách xếp chồng các lớp self-attention có positional encoding. Những vấn đề có thể xảy ra là gì?
1. Bạn có thể thiết kế một phương pháp positional encoding có thể học được không?
1. Chúng ta có thể gán các embedding đã học khác nhau theo các offset khác nhau giữa query và key được so sánh trong self-attention không? Gợi ý: bạn có thể tham khảo relative position embeddings :cite:`shaw2018self,huang2018music`.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/1651)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1652)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/3870)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18030)
:end_tab: