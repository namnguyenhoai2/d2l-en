# Suy luận ngôn ngữ tự nhiên: Fine-Tuning BERT
:label:`sec_natural-language-inference-bert`

Trong các phần trước của chương này,
chúng ta đã thiết kế một kiến trúc dựa trên attention
(trong :numref:`sec_natural-language-inference-attention`)
cho tác vụ suy luận ngôn ngữ tự nhiên
trên tập dữ liệu SNLI (như được mô tả trong :numref:`sec_natural-language-inference-and-dataset`).
Bây giờ chúng ta xem xét lại tác vụ này bằng cách fine-tuning BERT.
Như đã thảo luận trong :numref:`sec_finetuning-bert`,
suy luận ngôn ngữ tự nhiên là một bài toán phân loại cặp văn bản ở cấp độ chuỗi,
và để fine-tuning BERT chỉ cần thêm một kiến trúc dựa trên MLP,
như minh họa trong :numref:`fig_nlp-map-nli-bert`.

![Phần này đưa BERT pretrained vào một kiến trúc dựa trên MLP để thực hiện suy luận ngôn ngữ tự nhiên.](../img/nlp-map-nli-bert.svg)
:label:`fig_nlp-map-nli-bert`

Trong phần này,
chúng ta sẽ tải xuống một phiên bản nhỏ pretrained của BERT,
sau đó fine-tuning nó
cho tác vụ suy luận ngôn ngữ tự nhiên trên tập dữ liệu SNLI.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import json
import multiprocessing
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import json
import multiprocessing
import torch
from torch import nn
import os
```

## [**Tải BERT Pretrained**]

Chúng ta đã giải thích cách pretrain BERT trên tập dữ liệu WikiText-2 trong
:numref:`sec_bert-dataset` và :numref:`sec_bert-pretraining`
(lưu ý rằng mô hình BERT ban đầu được pretrained trên các corpora lớn hơn nhiều).
Như đã thảo luận trong :numref:`sec_bert-pretraining`,
mô hình BERT ban đầu có hàng trăm triệu tham số.
Trong phần sau,
chúng ta cung cấp hai phiên bản BERT pretrained:
"bert.base" có kích thước gần bằng mô hình BERT base ban đầu và cần nhiều tài nguyên tính toán để fine-tuning,
trong khi "bert.small" là một phiên bản nhỏ nhằm hỗ trợ việc minh họa.

```{.python .input}
#@tab mxnet
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.zip',
                             '7b3820b35da691042e5d34c0971ac3edbd80d3f4')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.zip',
                              'a4e718a47137ccd1809c9107ab4f5edd317bae2c')
```

```{.python .input}
#@tab pytorch
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')
```

Mỗi mô hình BERT pretrained đều chứa một tệp "vocab.json" xác định tập vocabulary
và một tệp "pretrained.params" chứa các tham số pretrained.
Chúng ta triển khai hàm `load_pretrained_model` sau đây để [**tải các tham số BERT pretrained**].

```{.python .input}
#@tab mxnet
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_blks, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, ffn_num_hiddens, num_heads, 
                         num_blks, dropout, max_len)
    # Load pretrained BERT parameters
    bert.load_parameters(os.path.join(data_dir, 'pretrained.params'),
                         ctx=devices)
    return bert, vocab
```

```{.python .input}
#@tab pytorch
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_blks, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(
        len(vocab), num_hiddens, ffn_num_hiddens=ffn_num_hiddens, num_heads=4,
        num_blks=2, dropout=0.2, max_len=max_len)
    # Load pretrained BERT parameters
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab
```

Để thuận tiện cho việc minh họa trên hầu hết các máy,
trong phần này chúng ta sẽ tải và fine-tuning phiên bản nhỏ ("bert.small") của BERT pretrained.
Trong bài tập, chúng ta sẽ trình bày cách fine-tuning "bert.base" lớn hơn nhiều để cải thiện đáng kể độ chính xác kiểm tra.

```{.python .input}
#@tab all
devices = d2l.try_all_gpus()
bert, vocab = load_pretrained_model(
    'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_blks=2, dropout=0.1, max_len=512, devices=devices)
```

## [**Tập dữ liệu để Fine-Tuning BERT**]

Đối với tác vụ downstream là suy luận ngôn ngữ tự nhiên trên tập dữ liệu SNLI,
chúng ta định nghĩa một lớp dataset tùy chỉnh `SNLIBERTDataset`.
Trong mỗi ví dụ,
premise và hypothesis tạo thành một cặp chuỗi văn bản
và được đóng gói thành một chuỗi đầu vào BERT như minh họa trong :numref:`fig_bert-two-seqs`.
Hãy nhớ lại từ :numref:`subsec_bert_input_rep` rằng segment ID
được dùng để phân biệt premise và hypothesis trong một chuỗi đầu vào BERT.
Với độ dài tối đa được định trước của một chuỗi đầu vào BERT (`max_len`),
token cuối cùng của chuỗi dài hơn trong cặp văn bản đầu vào sẽ liên tục bị loại bỏ cho đến khi
đạt `max_len`.
Để tăng tốc việc tạo tập dữ liệu SNLI
cho fine-tuning BERT,
chúng ta sử dụng 4 tiến trình worker để tạo song song các ví dụ huấn luyện hoặc kiểm tra.

```{.python .input}
#@tab mxnet
class SNLIBERTDataset(gluon.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = np.array(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # Use 4 worker processes
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (np.array(all_token_ids, dtype='int32'),
                np.array(all_segments, dtype='int32'), 
                np.array(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Reserve slots for '<CLS>', '<SEP>', and '<SEP>' tokens for the BERT
        # input
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # Use 4 worker processes
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long), 
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Reserve slots for '<CLS>', '<SEP>', and '<SEP>' tokens for the BERT
        # input
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

Sau khi tải xuống tập dữ liệu SNLI,
chúng ta [**tạo các ví dụ huấn luyện và kiểm tra**]
bằng cách khởi tạo lớp `SNLIBERTDataset`.
Các ví dụ này sẽ được đọc theo minibatch trong quá trình huấn luyện và kiểm tra
mô hình suy luận ngôn ngữ tự nhiên.

```{.python .input}
#@tab mxnet
# Reduce `batch_size` if there is an out of memory error. In the original BERT
# model, `max_len` = 512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = gluon.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

```{.python .input}
#@tab pytorch
# Reduce `batch_size` if there is an out of memory error. In the original BERT
# model, `max_len` = 512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

## Fine-Tuning BERT

Như :numref:`fig_bert-two-seqs` chỉ ra,
việc fine-tuning BERT cho suy luận ngôn ngữ tự nhiên
chỉ yêu cầu thêm một MLP gồm hai lớp fully connected
(xem `self.hidden` và `self.output` trong lớp `BERTClassifier` sau đây).
[**MLP này biến đổi biểu diễn BERT của token đặc biệt “&lt;cls&gt;”**],
mã hóa thông tin của cả premise và hypothesis,
(**thành ba đầu ra của suy luận ngôn ngữ tự nhiên**):
entailment, contradiction và neutral.

```{.python .input}
#@tab mxnet
class BERTClassifier(nn.Block):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Dense(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

```{.python .input}
#@tab pytorch
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.LazyLinear(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

Trong phần sau,
mô hình BERT pretrained `bert` được đưa vào instance `BERTClassifier` `net` để
phục vụ ứng dụng downstream.
Trong các triển khai fine-tuning BERT thông thường,
chỉ các tham số của output layer của MLP bổ sung (`net.output`) được học từ đầu.
Tất cả tham số của BERT encoder pretrained (`net.encoder`) và hidden layer của MLP bổ sung (`net.hidden`) sẽ được fine-tuning.

```{.python .input}
#@tab mxnet
net = BERTClassifier(bert)
net.output.initialize(ctx=devices)
```

```{.python .input}
#@tab pytorch
net = BERTClassifier(bert)
```

Hãy nhớ rằng
trong :numref:`sec_bert`
cả lớp `MaskLM` và lớp `NextSentencePred`
đều có các tham số trong những MLP mà chúng sử dụng.
Các tham số này là một phần của các tham số trong mô hình BERT pretrained
`bert`, và do đó là một phần của các tham số trong `net`.
Tuy nhiên, các tham số này chỉ được dùng để tính toán
loss masked language modeling
và loss next sentence prediction
trong quá trình pretraining.
Hai hàm loss này không liên quan đến các ứng dụng downstream trong quá trình fine-tuning,
do đó các tham số của những MLP được sử dụng trong
`MaskLM` và `NextSentencePred` không được cập nhật (bị giữ gradient cũ) khi BERT được fine-tuning.

Để cho phép các tham số có gradient cũ,
flag `ignore_stale_grad=True` được thiết lập trong hàm `step` của `d2l.train_batch_ch13`.
Chúng ta sử dụng hàm này để huấn luyện và đánh giá mô hình `net` bằng tập huấn luyện
(`train_iter`) và tập kiểm tra (`test_iter`) của SNLI.
Do tài nguyên tính toán hạn chế, [**độ chính xác huấn luyện**] và độ chính xác kiểm tra
có thể được cải thiện hơn nữa: chúng ta để phần thảo luận về vấn đề này trong các bài tập.

```{.python .input}
#@tab mxnet
lr, num_epochs = 1e-4, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               d2l.split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 1e-4, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
net(next(iter(train_iter))[0])
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## Tóm tắt

* Chúng ta có thể fine-tuning mô hình BERT pretrained cho các ứng dụng downstream, chẳng hạn như suy luận ngôn ngữ tự nhiên trên tập dữ liệu SNLI.
* Trong quá trình fine-tuning, mô hình BERT trở thành một phần của mô hình dành cho ứng dụng downstream. Các tham số chỉ liên quan đến loss pretraining sẽ không được cập nhật trong quá trình fine-tuning.



## Bài tập

1. Fine-tuning một mô hình BERT pretrained lớn hơn nhiều, có kích thước gần bằng mô hình BERT base ban đầu, nếu tài nguyên tính toán của bạn cho phép. Thiết lập các đối số trong hàm `load_pretrained_model` như sau: thay 'bert.small' bằng 'bert.base', tăng các giá trị của `num_hiddens=256`, `ffn_num_hiddens=512`, `num_heads=4` và `num_blks=2` lần lượt lên 768, 3072, 12 và 12. Bằng cách tăng số epoch fine-tuning (và có thể điều chỉnh các hyperparameter khác), bạn có thể đạt độ chính xác kiểm tra cao hơn 0.86 không?
1. Làm thế nào để truncate một cặp chuỗi theo tỷ lệ độ dài của chúng? Hãy so sánh phương pháp truncate cặp chuỗi này với phương pháp được sử dụng trong lớp `SNLIBERTDataset`. Ưu và nhược điểm của chúng là gì?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/397)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1526)
:end_tab: