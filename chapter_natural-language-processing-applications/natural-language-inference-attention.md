# Suy luận ngôn ngữ tự nhiên: Sử dụng Attention
:label:`sec_natural-language-inference-attention`

Chúng ta đã giới thiệu tác vụ suy luận ngôn ngữ tự nhiên và bộ dữ liệu SNLI trong :numref:`sec_natural-language-inference-and-dataset`. Trước nhiều mô hình dựa trên các kiến trúc phức tạp và sâu, :citet:`Parikh.Tackstrom.Das.ea.2016` đã đề xuất giải quyết bài toán suy luận ngôn ngữ tự nhiên bằng các cơ chế attention và gọi đó là một “mô hình attention có thể phân rã”.
Mô hình này không có các lớp recurrent hoặc convolutional, nhưng đạt kết quả tốt nhất tại thời điểm đó trên bộ dữ liệu SNLI với số lượng tham số ít hơn nhiều.
Trong phần này, chúng ta sẽ mô tả và triển khai phương pháp dựa trên attention này (với các MLP) cho bài toán suy luận ngôn ngữ tự nhiên, như được minh họa trong :numref:`fig_nlp-map-nli-attention`.

![Phần này đưa GloVe đã được pretrained vào một kiến trúc dựa trên attention và các MLP cho bài toán suy luận ngôn ngữ tự nhiên.](../img/nlp-map-nli-attention.svg)
:label:`fig_nlp-map-nli-attention`


## Mô hình

Thay vì bảo toàn thứ tự của các token trong premise và hypothesis,
chúng ta chỉ cần căn chỉnh các token trong một chuỗi văn bản với từng token trong chuỗi còn lại, và ngược lại,
sau đó so sánh và tổng hợp thông tin đó để dự đoán các mối quan hệ logic
giữa premise và hypothesis.
Tương tự việc căn chỉnh các token giữa câu nguồn và câu đích trong machine translation,
việc căn chỉnh các token giữa premise và hypothesis
có thể được thực hiện dễ dàng bằng các cơ chế attention.

![Suy luận ngôn ngữ tự nhiên bằng các cơ chế attention.](../img/nli-attention.svg)
:label:`fig_nli_attention`

:numref:`fig_nli_attention` minh họa phương pháp suy luận ngôn ngữ tự nhiên sử dụng các cơ chế attention.
Ở mức tổng quát, phương pháp này bao gồm ba bước được huấn luyện đồng thời: attending, comparing và aggregating.
Sau đây, chúng ta sẽ minh họa từng bước.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

### Attending

Bước đầu tiên là căn chỉnh các token trong một chuỗi văn bản với từng token trong chuỗi còn lại.
Giả sử premise là “i do need sleep” và hypothesis là “i am tired”.
Do có sự tương đồng về ngữ nghĩa,
chúng ta có thể muốn căn chỉnh “i” trong hypothesis với “i” trong premise,
và căn chỉnh “tired” trong hypothesis với “sleep” trong premise.
Tương tự, chúng ta có thể muốn căn chỉnh “i” trong premise với “i” trong hypothesis,
và căn chỉnh “need” cùng “sleep” trong premise với “tired” trong hypothesis.
Lưu ý rằng việc căn chỉnh này là *mềm* bằng cách sử dụng trung bình có trọng số,
trong đó lý tưởng nhất là các trọng số lớn được gán cho những token cần căn chỉnh.
Để dễ minh họa, :numref:`fig_nli_attention` trình bày việc căn chỉnh theo cách *cứng*.

Bây giờ, chúng ta mô tả chi tiết hơn việc căn chỉnh mềm bằng các cơ chế attention.
Ký hiệu $\mathbf{A} = (\mathbf{a}_1, \ldots, \mathbf{a}_m)$
và $\mathbf{B} = (\mathbf{b}_1, \ldots, \mathbf{b}_n)$ lần lượt là premise và hypothesis,
có số lượng token lần lượt là $m$ và $n$,
trong đó $\mathbf{a}_i, \mathbf{b}_j \in \mathbb{R}^{d}$ ($i = 1, \ldots, m, j = 1, \ldots, n$) là một vector từ có $d$ chiều.
Để căn chỉnh mềm, chúng ta tính các trọng số attention $e_{ij} \in \mathbb{R}$ như sau

$$e_{ij} = f(\mathbf{a}_i)^\top f(\mathbf{b}_j),$$
:eqlabel:`eq_nli_e`

trong đó hàm $f$ là một MLP được định nghĩa trong hàm `mlp` sau đây.
Số chiều đầu ra của $f$ được chỉ định bởi đối số `num_hiddens` của `mlp`.

```{.python .input}
#@tab mxnet
def mlp(num_hiddens, flatten):
    net = nn.Sequential()
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    return net
```

```{.python .input}
#@tab pytorch
def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)
```

Cần nhấn mạnh rằng, trong :eqref:`eq_nli_e`,
$f$ nhận các đầu vào $\mathbf{a}_i$ và $\mathbf{b}_j$ một cách riêng biệt, thay vì nhận cả hai dưới dạng một cặp đầu vào.
Thủ thuật *phân rã* này chỉ cần $m + n$ lần áp dụng $f$ (độ phức tạp tuyến tính) thay vì $mn$ lần áp dụng
(độ phức tạp bậc hai).


Sau khi chuẩn hóa các trọng số attention trong :eqref:`eq_nli_e`,
chúng ta tính trung bình có trọng số của tất cả các vector token trong hypothesis
để thu được biểu diễn của hypothesis được căn chỉnh mềm với token có chỉ số $i$ trong premise:

$$
\boldsymbol{\beta}_i = \sum_{j=1}^{n}\frac{\exp(e_{ij})}{ \sum_{k=1}^{n} \exp(e_{ik})} \mathbf{b}_j.
$$

Tương tự, chúng ta tính sự căn chỉnh mềm của các token trong premise cho từng token có chỉ số $j$ trong hypothesis:

$$
\boldsymbol{\alpha}_j = \sum_{i=1}^{m}\frac{\exp(e_{ij})}{ \sum_{k=1}^{m} \exp(e_{kj})} \mathbf{a}_i.
$$

Dưới đây, chúng ta định nghĩa class `Attend` để tính sự căn chỉnh mềm của các hypothesis (`beta`) với các premise đầu vào `A` và sự căn chỉnh mềm của các premise (`alpha`) với các hypothesis đầu vào `B`.

```{.python .input}
#@tab mxnet
class Attend(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (b`atch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = npx.batch_dot(f_A, f_B, transpose_b=True)
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = npx.batch_dot(npx.softmax(e), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = npx.batch_dot(npx.softmax(e.transpose(0, 2, 1)), A)
        return beta, alpha
```

```{.python .input}
#@tab pytorch
class Attend(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (`batch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha
```

### Comparing

Ở bước tiếp theo, chúng ta so sánh một token trong một chuỗi với chuỗi còn lại được căn chỉnh mềm với token đó.
Lưu ý rằng trong căn chỉnh mềm, tất cả token từ một chuỗi, dù có thể có các trọng số attention khác nhau, đều sẽ được so sánh với một token trong chuỗi còn lại.
Để dễ minh họa, :numref:`fig_nli_attention` ghép các token với những token được căn chỉnh theo cách *cứng*.
Ví dụ, giả sử bước attending xác định rằng “need” và “sleep” trong premise đều được căn chỉnh với “tired” trong hypothesis, thì cặp “tired--need sleep” sẽ được so sánh.

Trong bước comparing, chúng ta đưa phép nối (toán tử $[\cdot, \cdot]$) giữa các token từ một chuỗi và các token được căn chỉnh từ chuỗi còn lại vào một hàm $g$ (một MLP):

$$\mathbf{v}_{A,i} = g([\mathbf{a}_i, \boldsymbol{\beta}_i]), i = 1, \ldots, m\\ \mathbf{v}_{B,j} = g([\mathbf{b}_j, \boldsymbol{\alpha}_j]), j = 1, \ldots, n.$$

:eqlabel:`eq_nli_v_ab`


Trong :eqref:`eq_nli_v_ab`, $\mathbf{v}_{A,i}$ là phép so sánh giữa token $i$ trong premise và tất cả các token trong hypothesis được căn chỉnh mềm với token $i$;
còn $\mathbf{v}_{B,j}$ là phép so sánh giữa token $j$ trong hypothesis và tất cả các token trong premise được căn chỉnh mềm với token $j$.
Class `Compare` sau đây định nghĩa bước comparing này.

```{.python .input}
#@tab mxnet
class Compare(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(np.concatenate([A, beta], axis=2))
        V_B = self.g(np.concatenate([B, alpha], axis=2))
        return V_A, V_B
```

```{.python .input}
#@tab pytorch
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B
```

### Aggregating

Với hai tập vector so sánh $\mathbf{v}_{A,i}$ ($i = 1, \ldots, m$) và $\mathbf{v}_{B,j}$ ($j = 1, \ldots, n$),
ở bước cuối cùng, chúng ta sẽ tổng hợp thông tin này để suy ra mối quan hệ logic.
Trước hết, chúng ta tính tổng của cả hai tập:

$$
\mathbf{v}_A = \sum_{i=1}^{m} \mathbf{v}_{A,i}, \quad \mathbf{v}_B = \sum_{j=1}^{n}\mathbf{v}_{B,j}.
$$

Tiếp theo, chúng ta đưa phép nối của hai kết quả tóm tắt vào hàm $h$ (một MLP) để thu được kết quả phân loại của mối quan hệ logic:

$$
\hat{\mathbf{y}} = h([\mathbf{v}_A, \mathbf{v}_B]).
$$

Bước aggregating được định nghĩa trong class `Aggregate` sau đây.

```{.python .input}
#@tab mxnet
class Aggregate(nn.Block):
    def __init__(self, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_hiddens=num_hiddens, flatten=True)
        self.h.add(nn.Dense(num_outputs))

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(axis=1)
        V_B = V_B.sum(axis=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.h(np.concatenate([V_A, V_B], axis=1))
        return Y_hat
```

```{.python .input}
#@tab pytorch
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat
```

### Kết hợp tất cả

Bằng cách kết hợp các bước attending, comparing và aggregating,
chúng ta định nghĩa mô hình attention có thể phân rã để huấn luyện đồng thời cả ba bước này.

```{.python .input}
#@tab mxnet
class DecomposableAttention(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_hiddens)
        self.compare = Compare(num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_hiddens, 3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

```{.python .input}
#@tab pytorch
class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

## Huấn luyện và đánh giá mô hình

Bây giờ, chúng ta sẽ huấn luyện và đánh giá mô hình attention có thể phân rã đã định nghĩa trên bộ dữ liệu SNLI.
Chúng ta bắt đầu bằng việc đọc bộ dữ liệu.


### Đọc bộ dữ liệu

Chúng ta tải xuống và đọc bộ dữ liệu SNLI bằng hàm được định nghĩa trong :numref:`sec_natural-language-inference-and-dataset`. Kích thước batch và độ dài chuỗi lần lượt được đặt là $256$ và $50$.

```{.python .input}
#@tab all
batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = d2l.load_data_snli(batch_size, num_steps)
```

### Tạo mô hình

Chúng ta sử dụng embedding GloVe 100 chiều đã được pretrained để biểu diễn các token đầu vào.
Do đó, trong :eqref:`eq_nli_e`, chúng ta định trước số chiều của các vector $\mathbf{a}_i$ và $\mathbf{b}_j$ là 100.
Số chiều đầu ra của các hàm $f$ trong :eqref:`eq_nli_e` và $g$ trong :eqref:`eq_nli_v_ab` được đặt là 200.
Sau đó, chúng ta tạo một instance của mô hình, khởi tạo các tham số của mô hình,
và tải embedding GloVe để khởi tạo các vector của token đầu vào.

```{.python .input}
#@tab mxnet
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
net.initialize(init.Xavier(), ctx=devices)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
```

```{.python .input}
#@tab pytorch
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds);
```

### Huấn luyện và đánh giá mô hình

Không giống hàm `split_batch` trong :numref:`sec_multi_gpu` nhận các đầu vào đơn như các chuỗi văn bản (hoặc hình ảnh),
chúng ta định nghĩa hàm `split_batch_multi_inputs` để nhận nhiều đầu vào như premise và hypothesis trong các minibatch.

```{.python .input}
#@tab mxnet
#@save
def split_batch_multi_inputs(X, y, devices):
    """Split multi-input `X` and `y` into multiple devices."""
    X = list(zip(*[gluon.utils.split_and_load(
        feature, devices, even_split=False) for feature in X]))
    return (X, gluon.utils.split_and_load(y, devices, even_split=False))
```

Bây giờ, chúng ta có thể huấn luyện và đánh giá mô hình trên bộ dữ liệu SNLI.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.001, 4
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 4
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

### Sử dụng mô hình

Cuối cùng, hãy định nghĩa hàm dự đoán để xuất ra mối quan hệ logic giữa một cặp premise và hypothesis.

```{.python .input}
#@tab mxnet
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis."""
    premise = np.array(vocab[premise], ctx=d2l.try_gpu())
    hypothesis = np.array(vocab[hypothesis], ctx=d2l.try_gpu())
    label = np.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), axis=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

```{.python .input}
#@tab pytorch
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis."""
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

Chúng ta có thể sử dụng mô hình đã huấn luyện để thu được kết quả suy luận ngôn ngữ tự nhiên cho một cặp câu mẫu.

```{.python .input}
#@tab all
predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'])
```

## Tóm tắt

* Mô hình attention có thể phân rã gồm ba bước để dự đoán các mối quan hệ logic giữa tiền đề và giả thuyết: attending, comparing và aggregating.
* Với các cơ chế attention, chúng ta có thể căn chỉnh các token trong một chuỗi văn bản với mọi token trong chuỗi còn lại và ngược lại. Kiểu căn chỉnh này là mềm, sử dụng trung bình có trọng số; trong đó, lý tưởng nhất là các trọng số lớn được gán cho những token cần được căn chỉnh.
* Mẹo phân rã dẫn đến độ phức tạp tuyến tính phù hợp hơn so với độ phức tạp bậc hai khi tính các trọng số attention.
* Chúng ta có thể sử dụng các vector từ được pretrained làm biểu diễn đầu vào cho các tác vụ xử lý ngôn ngữ tự nhiên downstream, chẳng hạn như suy luận ngôn ngữ tự nhiên.


## Bài tập

1. Huấn luyện mô hình với các tổ hợp hyperparameter khác. Bạn có thể đạt được độ chính xác cao hơn trên tập kiểm tra không?
1. Những nhược điểm chính của mô hình attention có thể phân rã đối với bài toán suy luận ngôn ngữ tự nhiên là gì?
1. Giả sử chúng ta muốn xác định mức độ tương đồng về ngữ nghĩa (ví dụ: một giá trị liên tục từ 0 đến 1) cho mọi cặp câu. Chúng ta nên thu thập và gắn nhãn dataset như thế nào? Bạn có thể thiết kế một mô hình sử dụng các cơ chế attention không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/395)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1530)
:end_tab: