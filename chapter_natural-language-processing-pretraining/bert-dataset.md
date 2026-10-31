# Tập dữ liệu để pretraining BERT
:label:`sec_bert-dataset`

Để pretrain mô hình BERT như được triển khai trong :numref:`sec_bert`,
chúng ta cần tạo tập dữ liệu ở định dạng lý tưởng để hỗ trợ
hai tác vụ pretraining:
masked language modeling và next sentence prediction.
Một mặt,
mô hình BERT ban đầu được pretrain trên sự kết hợp của
hai corpus khổng lồ là BookCorpus và English Wikipedia (xem :numref:`subsec_bert_pretraining_tasks`),
khiến việc chạy mô hình trở nên khó khăn đối với hầu hết độc giả của cuốn sách này.
Mặt khác,
mô hình BERT pretrained có sẵn
có thể không phù hợp với các ứng dụng trong những lĩnh vực cụ thể như y học.
Do đó, việc pretrain BERT trên một tập dữ liệu tùy chỉnh ngày càng trở nên phổ biến.
Để minh họa việc pretrain BERT,
chúng ta sử dụng corpus nhỏ hơn là WikiText-2 :cite:`Merity.Xiong.Bradbury.ea.2016`.

So với tập dữ liệu PTB được sử dụng để pretrain word2vec trong :numref:`sec_word2vec_data`,
WikiText-2 (i) giữ lại dấu câu ban đầu, phù hợp cho next sentence prediction; (ii) giữ lại kiểu viết hoa ban đầu và các chữ số; (iii) có kích thước lớn hơn hai lần.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import random

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import os
import random
import torch
```

Trong [**tập dữ liệu WikiText-2**],
mỗi dòng biểu diễn một đoạn văn, trong đó
khoảng trắng được chèn giữa mọi dấu câu và token đứng trước nó.
Các đoạn văn có ít nhất hai câu được giữ lại.
Để tách câu, chúng ta chỉ sử dụng dấu chấm làm dấu phân cách cho đơn giản.
Chúng ta dành phần thảo luận về các kỹ thuật tách câu phức tạp hơn cho phần bài tập
ở cuối mục này.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

#@save
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # Uppercase letters are converted to lowercase ones
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs
```

## Định nghĩa các hàm hỗ trợ cho các tác vụ pretraining

Sau đây,
chúng ta bắt đầu triển khai các hàm hỗ trợ cho hai tác vụ pretraining của BERT:
next sentence prediction và masked language modeling.
Các hàm hỗ trợ này sẽ được gọi sau đó
khi chuyển corpus văn bản thô
thành tập dữ liệu có định dạng lý tưởng để pretrain BERT.

### [**Tạo tác vụ Next Sentence Prediction**]

Theo mô tả trong :numref:`subsec_nsp`,
hàm `_get_next_sentence` tạo một ví dụ huấn luyện
cho tác vụ phân loại nhị phân.

```{.python .input}
#@tab all
#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```

Hàm sau đây tạo các ví dụ huấn luyện cho next sentence prediction
từ đầu vào `paragraph` bằng cách gọi hàm `_get_next_sentence`.
Ở đây, `paragraph` là một danh sách các câu, trong đó mỗi câu là một danh sách token.
Đối số `max_len` chỉ định độ dài tối đa của một chuỗi đầu vào BERT trong quá trình pretraining.

```{.python .input}
#@tab all
#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # Consider 1 '<cls>' token and 2 '<sep>' tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
```

### [**Tạo tác vụ Masked Language Modeling**]
:label:`subsec_prepare_mlm_data`

Để tạo các ví dụ huấn luyện
cho tác vụ masked language modeling
từ một chuỗi đầu vào BERT,
chúng ta định nghĩa hàm `_replace_mlm_tokens` sau đây.
Trong các đầu vào của hàm, `tokens` là một danh sách các token biểu diễn một chuỗi đầu vào BERT,
`candidate_pred_positions` là một danh sách các chỉ số token của chuỗi đầu vào BERT
không bao gồm các chỉ số của các special token (special token không được dự đoán trong tác vụ masked language modeling),
và `num_mlm_preds` cho biết số lượng dự đoán (nhắc lại rằng 15% token ngẫu nhiên được chọn để dự đoán).
Theo định nghĩa của tác vụ masked language modeling trong :numref:`subsec_mlm`,
tại mỗi vị trí dự đoán, đầu vào có thể được thay thế bằng
một token đặc biệt “&lt;mask&gt;” hoặc một token ngẫu nhiên, hoặc được giữ nguyên.
Cuối cùng, hàm trả về các token đầu vào sau khi có thể được thay thế,
các chỉ số token tại đó thực hiện dự đoán và các nhãn cho những dự đoán này.

```{.python .input}
#@tab all
#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # For the input of a masked language model, make a new copy of tokens and
    # replace some of them by '<mask>' or random tokens
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Shuffle for getting 15% random tokens for prediction in the masked
    # language modeling task
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% of the time: keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% of the time: replace the word with a random word
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels
```

Bằng cách gọi hàm `_replace_mlm_tokens` nói trên,
hàm sau đây nhận một chuỗi đầu vào BERT (`tokens`)
làm đầu vào và trả về các chỉ số của các token đầu vào
(sau khi có thể được thay thế token như mô tả trong :numref:`subsec_mlm`),
các chỉ số token tại đó thực hiện dự đoán,
và các chỉ số nhãn cho những dự đoán này.

```{.python .input}
#@tab all
#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # `tokens` is a list of strings
    for i, token in enumerate(tokens):
        # Special tokens are not predicted in the masked language modeling
        # task
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 15% of random tokens are predicted in the masked language modeling task
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```

## Chuyển văn bản thành tập dữ liệu pretraining

Bây giờ chúng ta gần như đã sẵn sàng để tùy chỉnh một lớp `Dataset` cho việc pretrain BERT.
Trước đó,
chúng ta vẫn cần định nghĩa một hàm hỗ trợ `_pad_bert_inputs`
để [**thêm các token đặc biệt “&lt;pad&gt;” vào đầu vào.**]
Đối số `examples` của hàm chứa các đầu ra từ các hàm hỗ trợ `_get_nsp_data_from_paragraph` và `_get_mlm_data_from_tokens` cho hai tác vụ pretraining.

```{.python .input}
#@tab mxnet
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype='int32'))
        all_segments.append(np.array(segments + [0] * (
            max_len - len(segments)), dtype='int32'))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(np.array(len(token_ids), dtype='float32'))
        all_pred_positions.append(np.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype='int32'))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype='float32'))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```{.python .input}
#@tab pytorch
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

Kết hợp các hàm hỗ trợ
tạo ví dụ huấn luyện cho hai tác vụ pretraining,
cùng với hàm hỗ trợ đệm đầu vào,
chúng ta tùy chỉnh lớp `_WikiTextDataset` sau đây thành [**tập dữ liệu WikiText-2 để pretrain BERT**].
Bằng cách triển khai hàm `__getitem__ `,
chúng ta có thể truy cập tùy ý các ví dụ pretraining (masked language modeling và next sentence prediction)
được tạo từ một cặp câu trong corpus WikiText-2.

Mô hình BERT ban đầu sử dụng các embedding WordPiece với kích thước vocabulary là 30000 :cite:`Wu.Schuster.Chen.ea.2016`.
Phương pháp tokenization của WordPiece là một sửa đổi nhỏ của
thuật toán byte pair encoding ban đầu trong :numref:`subsec_Byte_Pair_Encoding`.
Để đơn giản, chúng ta sử dụng hàm `d2l.tokenize` cho việc tokenization.
Các token không thường gặp xuất hiện ít hơn năm lần sẽ bị loại bỏ.

```{.python .input}
#@tab mxnet
#@save
class _WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

Bằng cách sử dụng hàm `_read_wiki` và lớp `_WikiTextDataset`,
chúng ta định nghĩa `load_data_wiki` sau đây để [**tải xuống tập dữ liệu WikiText-2 và tạo các ví dụ pretraining**] từ đó.

```{.python .input}
#@tab mxnet
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    return train_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

Đặt batch size là 512 và độ dài tối đa của một chuỗi đầu vào BERT là 64,
chúng ta [**in ra các shape của một minibatch gồm các ví dụ pretraining BERT**].
Lưu ý rằng trong mỗi chuỗi đầu vào BERT,
$10$ ($64 \times 0.15$) vị trí được dự đoán cho tác vụ masked language modeling.

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```

Cuối cùng, hãy xem kích thước vocabulary.
Ngay cả sau khi loại bỏ các token không thường gặp,
kích thước này vẫn lớn hơn hai lần so với tập dữ liệu PTB.

```{.python .input}
#@tab all
len(vocab)
```

## Tóm tắt

* So với tập dữ liệu PTB, tập dữ liệu WikiText-2 giữ lại dấu câu, kiểu viết hoa và các chữ số ban đầu, đồng thời có kích thước lớn hơn hai lần.
* Chúng ta có thể truy cập tùy ý các ví dụ pretraining (masked language modeling và next sentence prediction) được tạo từ một cặp câu trong corpus WikiText-2.


## Bài tập

1. Để đơn giản, dấu chấm được sử dụng làm dấu phân cách duy nhất để tách câu. Hãy thử các kỹ thuật tách câu khác, chẳng hạn như spaCy và NLTK. Lấy NLTK làm ví dụ. Trước tiên, bạn cần cài đặt NLTK: `pip install nltk`. Trong code, trước hết `import nltk`. Sau đó, tải xuống bộ tách câu Punkt: `nltk.download('punkt')`. Để tách các câu như `sentences = 'This is great ! Why not ?'`, việc gọi `nltk.tokenize.sent_tokenize(sentences)` sẽ trả về một danh sách gồm hai chuỗi câu: `['This is great !', 'Why not ?']`.
1. Kích thước vocabulary là bao nhiêu nếu chúng ta không loại bỏ bất kỳ token không thường gặp nào?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/389)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1496)
:end_tab: