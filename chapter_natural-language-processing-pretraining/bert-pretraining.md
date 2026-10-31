# Tiền huấn luyện BERT
:label:`sec_bert-pretraining`

Với mô hình BERT được triển khai trong :numref:`sec_bert`
và các ví dụ tiền huấn luyện được tạo từ tập dữ liệu WikiText-2 trong :numref:`sec_bert-dataset`, trong phần này, chúng ta sẽ tiền huấn luyện BERT trên tập dữ liệu WikiText-2.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

Trước tiên, chúng ta tải tập dữ liệu WikiText-2 dưới dạng các minibatch
gồm các ví dụ tiền huấn luyện cho mô hình ngôn ngữ bị che và dự đoán câu tiếp theo.
Kích thước batch là 512 và độ dài tối đa của một chuỗi đầu vào BERT là 64.
Lưu ý rằng trong mô hình BERT gốc, độ dài tối đa là 512.

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
```

## Tiền huấn luyện BERT

BERT gốc có hai phiên bản với kích thước mô hình khác nhau :cite:`Devlin.Chang.Lee.ea.2018`.
Mô hình base ($\textrm{BERT}_{\textrm{BASE}}$) sử dụng 12 tầng (các khối encoder của Transformer)
với 768 hidden unit (hidden size) và 12 đầu self-attention.
Mô hình large ($\textrm{BERT}_{\textrm{LARGE}}$) sử dụng 24 tầng
với 1024 hidden unit và 16 đầu self-attention.
Đáng chú ý, mô hình trước có 110 triệu tham số, trong khi mô hình sau có 340 triệu tham số.
Để dễ minh họa,
chúng ta định nghĩa [**một BERT nhỏ, sử dụng 2 tầng, 128 hidden unit và 2 đầu self-attention**].

```{.python .input}
#@tab mxnet
net = d2l.BERTModel(len(vocab), num_hiddens=128, ffn_num_hiddens=256,
                    num_heads=2, num_blks=2, dropout=0.2)
devices = d2l.try_all_gpus()
net.initialize(init.Xavier(), ctx=devices)
loss = gluon.loss.SoftmaxCELoss()
```

```{.python .input}
#@tab pytorch
net = d2l.BERTModel(len(vocab), num_hiddens=128, 
                    ffn_num_hiddens=256, num_heads=2, num_blks=2, dropout=0.2)
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()
```

Trước khi định nghĩa vòng lặp huấn luyện,
chúng ta định nghĩa một hàm hỗ trợ `_get_batch_loss_bert`.
Với một shard gồm các ví dụ huấn luyện,
hàm này [**tính loss cho cả hai tác vụ mô hình ngôn ngữ bị che và dự đoán câu tiếp theo**].
Lưu ý rằng loss cuối cùng trong quá trình tiền huấn luyện BERT
chỉ là tổng của loss mô hình ngôn ngữ bị che
và loss dự đoán câu tiếp theo.

```{.python .input}
#@tab mxnet
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X_shards,
                         segments_X_shards, valid_lens_x_shards,
                         pred_positions_X_shards, mlm_weights_X_shards,
                         mlm_Y_shards, nsp_y_shards):
    mlm_ls, nsp_ls, ls = [], [], []
    for (tokens_X_shard, segments_X_shard, valid_lens_x_shard,
         pred_positions_X_shard, mlm_weights_X_shard, mlm_Y_shard,
         nsp_y_shard) in zip(
        tokens_X_shards, segments_X_shards, valid_lens_x_shards,
        pred_positions_X_shards, mlm_weights_X_shards, mlm_Y_shards,
        nsp_y_shards):
        # Forward pass
        _, mlm_Y_hat, nsp_Y_hat = net(
            tokens_X_shard, segments_X_shard, valid_lens_x_shard.reshape(-1),
            pred_positions_X_shard)
        # Compute masked language model loss
        mlm_l = loss(
            mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y_shard.reshape(-1),
            mlm_weights_X_shard.reshape((-1, 1)))
        mlm_l = mlm_l.sum() / (mlm_weights_X_shard.sum() + 1e-8)
        # Compute next sentence prediction loss
        nsp_l = loss(nsp_Y_hat, nsp_y_shard)
        nsp_l = nsp_l.mean()
        mlm_ls.append(mlm_l)
        nsp_ls.append(nsp_l)
        ls.append(mlm_l + nsp_l)
        npx.waitall()
    return mlm_ls, nsp_ls, ls
```

```{.python .input}
#@tab pytorch
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # Compute masked language model loss
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Compute next sentence prediction loss
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
```

Gọi hai hàm hỗ trợ nêu trên,
hàm `train_bert` sau đây
định nghĩa quy trình để [**tiền huấn luyện BERT (`net`) trên tập dữ liệu WikiText-2 (`train_iter`)**].
Việc huấn luyện BERT có thể mất rất nhiều thời gian.
Thay vì chỉ định số epoch huấn luyện
như trong hàm `train_ch13` (xem :numref:`sec_image_augmentation`),
đầu vào `num_steps` của hàm sau
chỉ định số bước lặp dùng để huấn luyện.

```{.python .input}
#@tab mxnet
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': 0.01})
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for batch in train_iter:
            (tokens_X_shards, segments_X_shards, valid_lens_x_shards,
             pred_positions_X_shards, mlm_weights_X_shards,
             mlm_Y_shards, nsp_y_shards) = [gluon.utils.split_and_load(
                elem, devices, even_split=False) for elem in batch]
            timer.start()
            with autograd.record():
                mlm_ls, nsp_ls, ls = _get_batch_loss_bert(
                    net, loss, vocab_size, tokens_X_shards, segments_X_shards,
                    valid_lens_x_shards, pred_positions_X_shards,
                    mlm_weights_X_shards, mlm_Y_shards, nsp_y_shards)
            for l in ls:
                l.backward()
            trainer.step(1)
            mlm_l_mean = sum([float(l) for l in mlm_ls]) / len(mlm_ls)
            nsp_l_mean = sum([float(l) for l in nsp_ls]) / len(nsp_ls)
            metric.add(mlm_l_mean, nsp_l_mean, batch[0].shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net(*next(iter(train_iter))[:4])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

Chúng ta có thể vẽ cả loss mô hình ngôn ngữ bị che và loss dự đoán câu tiếp theo
trong quá trình tiền huấn luyện BERT.

```{.python .input}
#@tab all
train_bert(train_iter, net, loss, len(vocab), devices, 50)
```

## [**Biểu diễn văn bản bằng BERT**]

Sau khi tiền huấn luyện BERT,
chúng ta có thể sử dụng nó để biểu diễn một văn bản đơn, các cặp văn bản hoặc bất kỳ token nào trong đó.
Hàm sau trả về các biểu diễn BERT (`net`) cho tất cả token
trong `tokens_a` và `tokens_b`.

```{.python .input}
#@tab mxnet
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = np.expand_dims(np.array(vocab[tokens], ctx=devices[0]),
                               axis=0)
    segments = np.expand_dims(np.array(segments, ctx=devices[0]), axis=0)
    valid_len = np.expand_dims(np.array(len(tokens), ctx=devices[0]), axis=0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

```{.python .input}
#@tab pytorch
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

[**Xét câu "a crane is flying".**]
Hãy nhớ lại biểu diễn đầu vào của BERT như đã thảo luận trong :numref:`subsec_bert_input_rep`.
Sau khi chèn các token đặc biệt “&lt;cls&gt;” (dùng cho việc phân loại)
và “&lt;sep&gt;” (dùng để phân tách),
chuỗi đầu vào BERT có độ dài là sáu.
Vì chỉ số của token “&lt;cls&gt;” là zero,
`encoded_text[:, 0, :]` là biểu diễn BERT của toàn bộ câu đầu vào.
Để đánh giá token đa nghĩa "crane",
chúng ta cũng in ra ba phần tử đầu tiên trong biểu diễn BERT của token này.

```{.python .input}
#@tab all
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# Tokens: '<cls>', 'a', 'crane', 'is', 'flying', '<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
```

[**Bây giờ hãy xét cặp câu
"a crane driver came" và "he just left".**]
Tương tự, `encoded_pair[:, 0, :]` là kết quả được mã hóa của toàn bộ cặp câu từ BERT đã được tiền huấn luyện.
Lưu ý rằng ba phần tử đầu tiên của token đa nghĩa "crane" khác với khi ngữ cảnh khác đi.
Điều này cho thấy các biểu diễn BERT phụ thuộc vào ngữ cảnh.

```{.python .input}
#@tab all
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# Tokens: '<cls>', 'a', 'crane', 'driver', 'came', '<sep>', 'he', 'just',
# 'left', '<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
```

Trong :numref:`chap_nlp_app`, chúng ta sẽ fine-tune một mô hình BERT đã được tiền huấn luyện
cho các ứng dụng xử lý ngôn ngữ tự nhiên downstream.

## Tóm tắt

* BERT gốc có hai phiên bản, trong đó mô hình base có 110 triệu tham số và mô hình large có 340 triệu tham số.
* Sau khi tiền huấn luyện BERT, chúng ta có thể sử dụng nó để biểu diễn một văn bản đơn, các cặp văn bản hoặc bất kỳ token nào trong đó.
* Trong thí nghiệm, cùng một token có biểu diễn BERT khác nhau khi ngữ cảnh của chúng khác nhau. Điều này cho thấy các biểu diễn BERT phụ thuộc vào ngữ cảnh.


## Bài tập

1. Trong thí nghiệm, chúng ta có thể thấy loss mô hình ngôn ngữ bị che cao hơn đáng kể so với loss dự đoán câu tiếp theo. Tại sao?
2. Đặt độ dài tối đa của chuỗi đầu vào BERT là 512 (giống mô hình BERT gốc). Sử dụng các cấu hình của mô hình BERT gốc, chẳng hạn như $\textrm{BERT}_{\textrm{LARGE}}$. Bạn có gặp lỗi khi chạy phần này không? Tại sao?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/390)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1497)
:end_tab: