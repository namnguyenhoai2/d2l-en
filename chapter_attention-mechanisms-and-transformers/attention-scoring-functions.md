```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Hàm chấm điểm Attention
:label:`sec_attention-scoring-functions`


Trong :numref:`sec_attention-pooling`,
chúng ta đã sử dụng một số kernel dựa trên khoảng cách khác nhau, bao gồm kernel Gaussian để mô hình hóa
tương tác giữa truy vấn và khóa. Hóa ra, các hàm khoảng cách có chi phí tính toán cao hơn một chút so với tích vô hướng. Do đó, 
cùng với phép toán softmax để đảm bảo các trọng số attention không âm,
phần lớn công việc đã tập trung vào các *hàm chấm điểm attention* $a$ trong :eqref:`eq_softmax_attention` và :numref:`fig_attention_output` có cách tính đơn giản hơn. 

![Tính đầu ra của attention pooling dưới dạng trung bình có trọng số của các giá trị, trong đó trọng số được tính bằng hàm chấm điểm attention $\mathit{a}$ và phép toán softmax.](../img/attention-output.svg)
:label:`fig_attention_output`

```{.python .input}
%%tab mxnet
import math
from d2l import mxnet as d2l
from mxnet import np, npx
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
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
import math
```

## [**Attention Tích vô hướng**]


Hãy xem lại trong chốc lát hàm attention (không có phép lũy thừa) từ kernel Gaussian:

$$
a(\mathbf{q}, \mathbf{k}_i) = -\frac{1}{2} \|\mathbf{q} - \mathbf{k}_i\|^2  = \mathbf{q}^\top \mathbf{k}_i -\frac{1}{2} \|\mathbf{k}_i\|^2  -\frac{1}{2} \|\mathbf{q}\|^2.
$$

Trước hết, lưu ý rằng hạng tử cuối chỉ phụ thuộc vào $\mathbf{q}$. Do đó, nó giống hệt nhau đối với mọi cặp $(\mathbf{q}, \mathbf{k}_i)$. Chuẩn hóa trọng số attention về $1$, như được thực hiện trong :eqref:`eq_softmax_attention`, đảm bảo rằng hạng tử này biến mất hoàn toàn. Thứ hai, lưu ý rằng cả batch normalization và layer normalization (sẽ được thảo luận sau) đều tạo ra các activation có chuẩn $\|\mathbf{k}_i\|$ được chặn tốt, và thường không đổi. Chẳng hạn, điều này xảy ra bất cứ khi nào các khóa $\mathbf{k}_i$ được tạo bởi một layer norm. Do đó, chúng ta có thể loại bỏ nó khỏi định nghĩa của $a$ mà không làm thay đổi đáng kể kết quả. 

Cuối cùng, chúng ta cần kiểm soát bậc độ lớn của các đối số trong hàm mũ. Giả sử rằng mọi phần tử của truy vấn $\mathbf{q} \in \mathbb{R}^d$ và khóa $\mathbf{k}_i \in \mathbb{R}^d$ là các biến ngẫu nhiên độc lập và cùng phân phối, có trung bình bằng không và phương sai đơn vị. Tích vô hướng giữa hai vector có trung bình bằng không và phương sai là $d$. Để đảm bảo phương sai của tích vô hướng vẫn bằng $1$ bất kể độ dài vector, chúng ta sử dụng hàm chấm điểm *scaled dot product attention*. Nghĩa là, chúng ta co giãn tích vô hướng theo $1/\sqrt{d}$. Nhờ đó, ta có được hàm attention phổ biến đầu tiên, được sử dụng, chẳng hạn, trong Transformers :cite:`Vaswani.Shazeer.Parmar.ea.2017`:

$$ a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i / \sqrt{d}.$$
:eqlabel:`eq_dot_product_attention`

Lưu ý rằng các trọng số attention $\alpha$ vẫn cần được chuẩn hóa. Chúng ta có thể đơn giản hóa thêm thông qua :eqref:`eq_softmax_attention` bằng cách sử dụng phép toán softmax:

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(\mathbf{q}^\top \mathbf{k}_i / \sqrt{d})}{\sum_{j=1} \exp(\mathbf{q}^\top \mathbf{k}_j / \sqrt{d})}.$$
:eqlabel:`eq_attn-scoring-alpha`

Hóa ra, mọi cơ chế attention phổ biến đều sử dụng softmax, vì vậy chúng ta sẽ chỉ giới hạn ở nó trong phần còn lại của chương này.

## Các hàm tiện ích

Chúng ta cần một vài hàm để triển khai cơ chế attention hiệu quả. Điều này bao gồm các công cụ xử lý chuỗi có độ dài biến đổi (phổ biến trong xử lý ngôn ngữ tự nhiên) và các công cụ đánh giá hiệu quả trên minibatch (nhân ma trận theo batch). 


### [**Phép toán Softmax có Mask**]

Một trong những ứng dụng phổ biến nhất của cơ chế attention là các mô hình chuỗi. Vì vậy, chúng ta cần có khả năng xử lý các chuỗi có độ dài khác nhau. Trong một số trường hợp, các chuỗi như vậy có thể nằm trong cùng một minibatch, đòi hỏi phải padding bằng các token giả cho những chuỗi ngắn hơn (xem :numref:`sec_machine_translation` để biết ví dụ). Các token đặc biệt này không mang ý nghĩa. Chẳng hạn, giả sử chúng ta có ba câu sau:

```
Dive  into  Deep    Learning 
Learn to    code    <blank>
Hello world <blank> <blank>
```


Vì không muốn các khoảng trống trong mô hình attention, chúng ta chỉ cần giới hạn $\sum_{i=1}^n \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$ thành $\sum_{i=1}^l \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$ với độ dài thực tế của câu là $l \leq n$. Vì đây là một vấn đề rất phổ biến, nó có một tên gọi: *phép toán softmax có mask*. 

Hãy triển khai nó. Thực tế, phần triển khai này có một mẹo nhỏ là đặt các giá trị của $\mathbf{v}_i$, với $i > l$, thành không. Hơn nữa, nó đặt các trọng số attention thành một số âm lớn, chẳng hạn $-10^{6}$, để trên thực tế đóng góp của chúng vào gradient và giá trị triệt tiêu. Điều này được thực hiện vì các kernel và toán tử đại số tuyến tính được tối ưu hóa mạnh cho GPU, và việc lãng phí một ít tính toán sẽ nhanh hơn so với việc có mã chứa các câu lệnh điều kiện (if then else).

```{.python .input}
%%tab mxnet
def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    if valid_lens is None:
        return npx.softmax(X)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = valid_lens.repeat(shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = npx.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, True,
                              value=-1e6, axis=1)
        return npx.softmax(X).reshape(shape)
```

```{.python .input}
%%tab pytorch
def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor 
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X
    
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

```{.python .input}
%%tab tensorflow
def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[
            None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)

        if len(X.shape) == 3:
            return tf.where(tf.expand_dims(mask, axis=-1), X, value)
        else:
            return tf.where(mask, X, value)
    
    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])
            
        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0    
        X = _sequence_mask(tf.reshape(X, shape=(-1, shape[-1])), valid_lens,
                           value=-1e6)    
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)
```

```{.python .input}
%%tab jax
def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = jnp.arange((maxlen),
                          dtype=jnp.float32)[None, :] < valid_len[:, None]
        return jnp.where(mask, X, value)

    if valid_lens is None:
        return nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = jnp.repeat(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.softmax(X.reshape(shape), axis=-1)
```

Để [**minh họa cách hàm này hoạt động**],
hãy xét một minibatch gồm hai ví dụ có kích thước $2 \times 4$,
trong đó độ dài hợp lệ của chúng lần lượt là $2$ và $3$. 
Kết quả của phép toán softmax có mask là
các giá trị vượt quá độ dài hợp lệ của từng cặp vector đều được mask thành không.

```{.python .input}
%%tab mxnet
masked_softmax(np.random.uniform(size=(2, 2, 4)), d2l.tensor([2, 3]))
```

```{.python .input}
%%tab pytorch
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
```

```{.python .input}
%%tab tensorflow
masked_softmax(tf.random.uniform(shape=(2, 2, 4)), tf.constant([2, 3]))
```

```{.python .input}
%%tab jax
masked_softmax(jax.random.uniform(d2l.get_key(), (2, 2, 4)), jnp.array([2, 3]))
```

Nếu cần kiểm soát chi tiết hơn để chỉ định độ dài hợp lệ cho từng vector trong hai vector của mỗi ví dụ, chúng ta chỉ cần sử dụng tensor độ dài hợp lệ hai chiều. Điều này cho kết quả:

```{.python .input}
%%tab mxnet
masked_softmax(np.random.uniform(size=(2, 2, 4)),
               d2l.tensor([[1, 3], [2, 4]]))
```

```{.python .input}
%%tab pytorch
masked_softmax(torch.rand(2, 2, 4), d2l.tensor([[1, 3], [2, 4]]))
```

```{.python .input}
%%tab tensorflow
masked_softmax(tf.random.uniform((2, 2, 4)), tf.constant([[1, 3], [2, 4]]))
```

```{.python .input}
%%tab jax
masked_softmax(jax.random.uniform(d2l.get_key(), (2, 2, 4)),
               jnp.array([[1, 3], [2, 4]]))
```

### Nhân ma trận theo Batch
:label:`subsec_batch_dot`

Một phép toán thường dùng khác là nhân các batch ma trận với nhau. Điều này hữu ích khi chúng ta có các minibatch gồm truy vấn, khóa và giá trị. Cụ thể hơn, giả sử rằng 

$$\mathbf{Q} = [\mathbf{Q}_1, \mathbf{Q}_2, \ldots, \mathbf{Q}_n]  \in \mathbb{R}^{n \times a \times b}, \\
    \mathbf{K} = [\mathbf{K}_1, \mathbf{K}_2, \ldots, \mathbf{K}_n]  \in \mathbb{R}^{n \times b \times c}.
$$

Khi đó, phép nhân ma trận theo batch (BMM) tính tích theo từng phần tử

$$\textrm{BMM}(\mathbf{Q}, \mathbf{K}) = [\mathbf{Q}_1 \mathbf{K}_1, \mathbf{Q}_2 \mathbf{K}_2, \ldots, \mathbf{Q}_n \mathbf{K}_n] \in \mathbb{R}^{n \times a \times c}.$$
:eqlabel:`eq_batch-matrix-mul`

Hãy xem cách thực hiện điều này trong một deep learning framework.

```{.python .input}
%%tab mxnet
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(npx.batch_dot(Q, K), (2, 3, 6))
```

```{.python .input}
%%tab pytorch
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(torch.bmm(Q, K), (2, 3, 6))
```

```{.python .input}
%%tab tensorflow
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(tf.matmul(Q, K).numpy(), (2, 3, 6))
```

```{.python .input}
%%tab jax
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(jax.lax.batch_matmul(Q, K), (2, 3, 6))
```

## [**Scaled Dot Product Attention**]

Hãy quay lại attention tích vô hướng được giới thiệu trong :eqref:`eq_dot_product_attention`. 
Nhìn chung, nó yêu cầu cả truy vấn và khóa
có cùng độ dài vector, giả sử là $d$, mặc dù điều này có thể dễ dàng được xử lý bằng cách thay 
$\mathbf{q}^\top \mathbf{k}$ bằng $\mathbf{q}^\top \mathbf{M} \mathbf{k}$, trong đó $\mathbf{M}$ là một ma trận được chọn phù hợp để chuyển đổi giữa hai không gian. Hiện tại, hãy giả sử các chiều khớp nhau. 

Trong thực tế, chúng ta thường xét các minibatch để tăng hiệu quả,
chẳng hạn tính attention cho $n$ truy vấn và $m$ cặp khóa-giá trị,
trong đó truy vấn và khóa có độ dài $d$
còn giá trị có độ dài $v$. Vì vậy, scaled dot product attention 
của các truy vấn $\mathbf Q\in\mathbb R^{n\times d}$,
khóa $\mathbf K\in\mathbb R^{m\times d}$,
và giá trị $\mathbf V\in\mathbb R^{m\times v}$
có thể được viết là 

$$ \mathrm{softmax}\left(\frac{\mathbf Q \mathbf K^\top }{\sqrt{d}}\right) \mathbf V \in \mathbb{R}^{n\times v}.$$
:eqlabel:`eq_softmax_QK_V`

Lưu ý rằng khi áp dụng cho một minibatch, chúng ta cần phép nhân ma trận theo batch được giới thiệu trong :eqref:`eq_batch-matrix-mul`. Trong phần triển khai scaled dot product attention sau đây,
chúng ta sử dụng dropout để regularization mô hình.

```{.python .input}
%%tab mxnet
class DotProductAttention(nn.Block):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set transpose_b=True to swap the last two dimensions of keys
        scores = npx.batch_dot(queries, keys, transpose_b=True) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab pytorch
class DotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab tensorflow
class DotProductAttention(tf.keras.layers.Layer):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def call(self, queries, keys, values, valid_lens=None, **kwargs):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
            tf.cast(d, dtype=tf.float32))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)
```

```{.python .input}
%%tab jax
class DotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    dropout: float

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    @nn.compact
    def __call__(self, queries, keys, values, valid_lens=None,
                 training=False):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.swapaxes(1, 2)
        scores = queries@(keys.swapaxes(1, 2)) / math.sqrt(d)
        attention_weights = masked_softmax(scores, valid_lens)
        dropout_layer = nn.Dropout(self.dropout, deterministic=not training)
        return dropout_layer(attention_weights)@values, attention_weights
```

Để [**minh họa cách lớp `DotProductAttention` hoạt động**],
chúng ta sử dụng cùng các khóa, giá trị và độ dài hợp lệ từ ví dụ đồ chơi trước đó về additive attention. Với mục đích minh họa, chúng ta giả sử có một minibatch kích thước $2$, tổng cộng $10$ khóa và giá trị, và số chiều của các giá trị là $4$. Cuối cùng, chúng ta giả sử độ dài hợp lệ của mỗi quan sát lần lượt là $2$ và $6$. Với điều đó, ta kỳ vọng đầu ra là một tensor $2 \times 1 \times 4$, tức là một hàng cho mỗi ví dụ trong minibatch.

```{.python .input}
%%tab mxnet
queries = d2l.normal(0, 1, (2, 1, 2))
keys = d2l.normal(0, 1, (2, 10, 2))
values = d2l.normal(0, 1, (2, 10, 4))
valid_lens = d2l.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
attention.initialize()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab pytorch
queries = d2l.normal(0, 1, (2, 1, 2))
keys = d2l.normal(0, 1, (2, 10, 2))
values = d2l.normal(0, 1, (2, 10, 4))
valid_lens = d2l.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
attention.eval()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab tensorflow
queries = tf.random.normal(shape=(2, 1, 2))
keys = tf.random.normal(shape=(2, 10, 2))
values = tf.random.normal(shape=(2, 10, 4))
valid_lens = tf.constant([2, 6])

attention = DotProductAttention(dropout=0.5)
d2l.check_shape(attention(queries, keys, values, valid_lens, training=False),
                (2, 1, 4))
```

```{.python .input}
%%tab jax
queries = jax.random.normal(d2l.get_key(), (2, 1, 2))
keys = jax.random.normal(d2l.get_key(), (2, 10, 2))
values = jax.random.normal(d2l.get_key(), (2, 10, 4))
valid_lens = d2l.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
(output, attention_weights), params = attention.init_with_output(
    d2l.get_key(), queries, keys, values, valid_lens)
print(output)
```

Hãy kiểm tra xem các trọng số attention có thực sự triệt tiêu đối với mọi thứ vượt quá cột thứ hai và thứ sáu tương ứng hay không (do đặt độ dài hợp lệ thành $2$ và $6$).

```{.python .input}
%%tab pytorch, mxnet, tensorflow
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

```{.python .input}
%%tab jax
d2l.show_heatmaps(d2l.reshape(attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## [**Attention cộng**]
:label:`subsec_additive-attention`

Khi các query $\mathbf{q}$ và key $\mathbf{k}$ là các vector có số chiều khác nhau,
ta có thể dùng một ma trận để xử lý sự không khớp thông qua $\mathbf{q}^\top \mathbf{M} \mathbf{k}$, hoặc có thể dùng attention cộng
làm hàm tính điểm. Một lợi ích khác là, đúng như tên gọi, attention này có tính cộng. Điều này có thể giúp tiết kiệm một phần nhỏ chi phí tính toán.
Với một query $\mathbf{q} \in \mathbb{R}^q$
và một key $\mathbf{k} \in \mathbb{R}^k$,
hàm tính điểm *attention cộng* :cite:`Bahdanau.Cho.Bengio.2014` được cho bởi

$$a(\mathbf q, \mathbf k) = \mathbf w_v^\top \textrm{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k) \in \mathbb{R},$$
:eqlabel:`eq_additive-attn`

trong đó $\mathbf W_q\in\mathbb R^{h\times q}$, $\mathbf W_k\in\mathbb R^{h\times k}$,
và $\mathbf w_v\in\mathbb R^{h}$ là các tham số có thể học. Hạng tử này sau đó được đưa vào softmax để bảo đảm vừa không âm vừa được chuẩn hóa.
Một cách diễn giải tương đương của :eqref:`eq_additive-attn` là query và key được nối lại
rồi đưa vào một MLP có một hidden layer duy nhất.
Sử dụng $\tanh$ làm hàm kích hoạt và vô hiệu hóa các hạng tử bias,
ta triển khai attention cộng như sau:

```{.python .input}
%%tab mxnet
class AdditiveAttention(nn.Block):  #@save
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # Use flatten=False to only transform the last axis so that the
        # shapes for the other axes are kept the same
        self.W_k = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_q = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.w_v = nn.Dense(1, use_bias=False, flatten=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1,
        # no. of key-value pairs, num_hiddens). Sum them up with
        # broadcasting
        features = np.expand_dims(queries, axis=2) + np.expand_dims(
            keys, axis=1)
        features = np.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores:
        # (batch_size, no. of queries, no. of key-value pairs)
        scores = np.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab pytorch
class AdditiveAttention(nn.Module):  #@save
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab tensorflow
class AdditiveAttention(tf.keras.layers.Layer):  #@save
    """Additive attention."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.w_v = tf.keras.layers.Dense(1, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, queries, keys, values, valid_lens, **kwargs):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = tf.expand_dims(queries, axis=2) + tf.expand_dims(
            keys, axis=1)
        features = tf.nn.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = tf.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return tf.matmul(self.dropout(
            self.attention_weights, **kwargs), values)
```

```{.python .input}
%%tab jax
class AdditiveAttention(nn.Module):  #@save
    num_hiddens: int
    dropout: float

    def setup(self):
        self.W_k = nn.Dense(self.num_hiddens, use_bias=False)
        self.W_q = nn.Dense(self.num_hiddens, use_bias=False)
        self.w_v = nn.Dense(1, use_bias=False)

    @nn.compact
    def __call__(self, queries, keys, values, valid_lens, training=False):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = jnp.expand_dims(queries, axis=2) + jnp.expand_dims(keys, axis=1)
        features = nn.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        attention_weights = masked_softmax(scores, valid_lens)
        dropout_layer = nn.Dropout(self.dropout, deterministic=not training)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return dropout_layer(attention_weights)@values, attention_weights
```

Hãy [**xem cách `AdditiveAttention` hoạt động**]. Trong ví dụ đồ chơi, ta chọn query, key và value có kích thước
lần lượt là $(2, 1, 20)$, $(2, 10, 2)$ và $(2, 10, 4)$. Điều này giống hệt lựa chọn của ta cho `DotProductAttention`, ngoại trừ giờ đây các query có $20$ chiều. Tương tự, ta chọn $(2, 6)$ làm các độ dài hợp lệ cho các chuỗi trong minibatch.

```{.python .input}
%%tab mxnet
queries = d2l.normal(0, 1, (2, 1, 20))

attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.initialize()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab pytorch
queries = d2l.normal(0, 1, (2, 1, 20))

attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.eval()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab tensorflow
queries = tf.random.normal(shape=(2, 1, 20))

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
d2l.check_shape(attention(queries, keys, values, valid_lens, training=False),
                (2, 1, 4))
```

```{.python .input}
%%tab jax
queries = jax.random.normal(d2l.get_key(), (2, 1, 20))
attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
(output, attention_weights), params = attention.init_with_output(
    d2l.get_key(), queries, keys, values, valid_lens)
print(output)
```

Khi xem xét hàm attention, ta thấy một hành vi về mặt định tính khá tương tự với `DotProductAttention`. Nghĩa là, chỉ các hạng tử nằm trong độ dài hợp lệ đã chọn $(2, 6)$ là khác không.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

```{.python .input}
%%tab jax
d2l.show_heatmaps(d2l.reshape(attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## Tóm tắt

Trong phần này, chúng ta đã giới thiệu hai hàm tính điểm attention chủ chốt: tích vô hướng và attention cộng. Chúng là những công cụ hiệu quả để tổng hợp trên các chuỗi có độ dài thay đổi. Đặc biệt, attention tích vô hướng là nền tảng chính của các kiến trúc Transformer hiện đại. Khi query và key là các vector có độ dài khác nhau,
ta có thể dùng hàm tính điểm attention cộng thay thế. Tối ưu hóa các layer này là một trong những lĩnh vực tiến bộ then chốt trong những năm gần đây. Chẳng hạn, [Thư viện Transformer của NVIDIA](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html) và Megatron :cite:`shoeybi2019megatron` phụ thuộc rất nhiều vào các biến thể hiệu quả của cơ chế attention. Chúng ta sẽ đi sâu hơn đáng kể vào vấn đề này khi xem xét Transformer trong các phần sau.

## Bài tập

1. Triển khai attention dựa trên khoảng cách bằng cách sửa đổi mã `DotProductAttention`. Lưu ý rằng bạn chỉ cần chuẩn bình phương của các key $\|\mathbf{k}_i\|^2$ để triển khai hiệu quả.
1. Sửa đổi attention tích vô hướng để cho phép query và key có số chiều khác nhau bằng cách sử dụng một ma trận để điều chỉnh số chiều.
1. Chi phí tính toán thay đổi như thế nào theo số chiều của key, query, value và số lượng của chúng? Còn yêu cầu về băng thông bộ nhớ thì sao?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/346)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1064)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/3867)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18027)
:end_tab: