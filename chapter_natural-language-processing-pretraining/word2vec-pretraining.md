# Pretraining word2vec
:label:`sec_word2vec_pretraining`


Tiếp tục triển khai mô hình skip-gram
được định nghĩa trong
:numref:`sec_word2vec`.
Sau đó,
chúng ta sẽ pretrain word2vec bằng phương pháp lấy mẫu âm
trên tập dữ liệu PTB.
Trước hết,
hãy lấy data iterator
và vocabulary cho tập dữ liệu này
bằng cách gọi hàm `d2l.load_data_ptb`,
được mô tả trong :numref:`sec_word2vec_data`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

## Mô hình Skip-Gram

Chúng ta triển khai mô hình skip-gram
bằng cách sử dụng các embedding layer và phép nhân ma trận theo batch.
Trước tiên, hãy ôn lại
cách embedding layer hoạt động.


### Embedding Layer

Như được mô tả trong :numref:`sec_seq2seq`,
embedding layer
ánh xạ chỉ mục của một token thành vector đặc trưng của token đó.
Trọng số của layer này
là một ma trận có số hàng bằng
kích thước của từ điển (`input_dim`) và
số cột bằng
số chiều của vector cho mỗi token (`output_dim`).
Sau khi một mô hình word embedding được huấn luyện,
đây chính là trọng số chúng ta cần.

```{.python .input}
#@tab mxnet
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

```{.python .input}
#@tab pytorch
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

Đầu vào của một embedding layer là
chỉ mục của một token (từ).
Với mọi chỉ mục token $i$,
biểu diễn vector của token đó
có thể được lấy từ
hàng thứ $i^\textrm{th}$ của ma trận trọng số
trong embedding layer.
Vì số chiều của vector (`output_dim`)
được đặt là 4,
embedding layer
trả về các vector có shape (2, 3, 4)
cho một minibatch gồm các chỉ mục token có shape
(2, 3).

```{.python .input}
#@tab all
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### Định nghĩa Lan truyền tiến

Trong quá trình lan truyền tiến,
đầu vào của mô hình skip-gram
bao gồm
các chỉ mục của từ trung tâm `center`
có shape (batch size, 1)
và
các chỉ mục được nối của từ ngữ cảnh và từ nhiễu `contexts_and_negatives`
có shape (batch size, `max_len`),
trong đó `max_len`
được định nghĩa
trong :numref:`subsec_word2vec-minibatch-loading`.
Hai biến này trước tiên được chuyển từ
chỉ mục token thành các vector thông qua embedding layer,
sau đó phép nhân ma trận theo batch của chúng
(được mô tả trong :numref:`subsec_batch_dot`)
trả về
một đầu ra có shape (batch size, 1, `max_len`).
Mỗi phần tử trong đầu ra là tích vô hướng của
một vector từ trung tâm và một vector từ ngữ cảnh hoặc từ nhiễu.

```{.python .input}
#@tab mxnet
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

```{.python .input}
#@tab pytorch
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
```

Hãy in shape đầu ra của hàm `skip_gram` này với một số đầu vào mẫu.

```{.python .input}
#@tab mxnet
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

```{.python .input}
#@tab pytorch
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

## Huấn luyện

Trước khi huấn luyện mô hình skip-gram bằng phương pháp lấy mẫu âm,
trước hết hãy định nghĩa hàm mất mát của mô hình.


### Binary Cross-Entropy Loss

Theo định nghĩa của hàm mất mát
cho phương pháp lấy mẫu âm trong :numref:`subsec_negative-sampling`, 
chúng ta sẽ sử dụng 
binary cross-entropy loss.

```{.python .input}
#@tab mxnet
loss = gluon.loss.SigmoidBCELoss()
```

```{.python .input}
#@tab pytorch
class SigmoidBCELoss(nn.Module):
    # Binary cross-entropy loss with masking
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

Hãy nhớ lại phần mô tả
về biến mask
và biến label trong
:numref:`subsec_word2vec-minibatch-loading`.
Đoạn mã sau
tính toán
binary cross-entropy loss
cho các biến đã cho.

```{.python .input}
#@tab all
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

Dưới đây cho thấy
cách tính các kết quả trên
(theo cách kém hiệu quả hơn)
bằng cách sử dụng
hàm kích hoạt sigmoid
trong binary cross-entropy loss.
Chúng ta có thể xem
hai đầu ra là
hai loss đã chuẩn hóa
được lấy trung bình trên các dự đoán không bị mask.

```{.python .input}
#@tab all
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### Khởi tạo Tham số Mô hình

Chúng ta định nghĩa hai embedding layer
cho tất cả các từ trong vocabulary
khi chúng lần lượt được dùng làm từ trung tâm
và từ ngữ cảnh.
Số chiều của vector từ
`embed_size` được đặt là 100.

```{.python .input}
#@tab mxnet
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```

```{.python .input}
#@tab pytorch
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

### Định nghĩa Vòng lặp Huấn luyện

Vòng lặp huấn luyện được định nghĩa bên dưới. Do có padding, phép tính hàm mất mát hơi khác so với các hàm huấn luyện trước đó.

```{.python .input}
#@tab mxnet
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(module):
        if type(module) == nn.Embedding:
            nn.init.xavier_uniform_(module.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

Bây giờ chúng ta có thể huấn luyện một mô hình skip-gram bằng phương pháp lấy mẫu âm.

```{.python .input}
#@tab all
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

## Áp dụng Word Embedding
:label:`subsec_apply-word-embed`


Sau khi huấn luyện mô hình word2vec,
chúng ta có thể sử dụng độ tương đồng cosine
của các vector từ trong mô hình đã huấn luyện
để
tìm những từ trong từ điển
có ngữ nghĩa tương đồng nhất
với một từ đầu vào.

```{.python .input}
#@tab mxnet
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

```{.python .input}
#@tab pytorch
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

## Tóm tắt

* Chúng ta có thể huấn luyện mô hình skip-gram bằng phương pháp lấy mẫu âm sử dụng embedding layer và binary cross-entropy loss.
* Các ứng dụng của word embedding bao gồm việc tìm những từ có ngữ nghĩa tương đồng với một từ cho trước dựa trên độ tương đồng cosine của các vector từ.


## Bài tập

1. Sử dụng mô hình đã huấn luyện, hãy tìm những từ có ngữ nghĩa tương đồng với các từ đầu vào khác. Bạn có thể cải thiện kết quả bằng cách tinh chỉnh các hyperparameter không?
1. Khi corpus huấn luyện rất lớn, chúng ta thường lấy mẫu các từ ngữ cảnh và từ nhiễu cho các từ trung tâm trong minibatch hiện tại *khi cập nhật tham số mô hình*. Nói cách khác, cùng một từ trung tâm có thể có các từ ngữ cảnh hoặc từ nhiễu khác nhau trong những epoch huấn luyện khác nhau. Phương pháp này có những lợi ích gì? Hãy thử triển khai phương pháp huấn luyện này.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/384)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1335)
:end_tab: