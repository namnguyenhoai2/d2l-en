# Suy luận ngôn ngữ tự nhiên và tập dữ liệu
:label:`sec_natural-language-inference-and-dataset`

Trong :numref:`sec_sentiment`, chúng ta đã thảo luận về bài toán phân tích cảm xúc.
Nhiệm vụ này nhằm phân loại một chuỗi văn bản đơn lẻ vào các danh mục được định trước,
chẳng hạn như một tập hợp các cực tính cảm xúc.
Tuy nhiên, khi cần quyết định liệu một câu có thể được suy ra từ một câu khác hay không,
hoặc loại bỏ sự dư thừa bằng cách xác định các câu tương đương về mặt ngữ nghĩa,
thì việc biết cách phân loại một chuỗi văn bản là chưa đủ.
Thay vào đó, chúng ta cần có khả năng suy luận trên các cặp chuỗi văn bản.


## Suy luận ngôn ngữ tự nhiên

*Suy luận ngôn ngữ tự nhiên* nghiên cứu liệu một *giả thuyết*
có thể được suy ra từ một *tiền đề* hay không, trong đó cả hai đều là các chuỗi văn bản.
Nói cách khác, suy luận ngôn ngữ tự nhiên xác định mối quan hệ logic giữa một cặp chuỗi văn bản.
Các mối quan hệ như vậy thường thuộc ba loại:

* *Entailment*: giả thuyết có thể được suy ra từ tiền đề.
* *Contradiction*: phủ định của giả thuyết có thể được suy ra từ tiền đề.
* *Neutral*: tất cả các trường hợp còn lại.

Suy luận ngôn ngữ tự nhiên còn được gọi là tác vụ nhận diện quan hệ kéo theo trong văn bản.
Ví dụ, cặp sau được gán nhãn *entailment* vì “thể hiện tình cảm” trong giả thuyết có thể được suy ra từ “ôm nhau” trong tiền đề.

> Tiền đề: Hai người phụ nữ đang ôm nhau.

> Giả thuyết: Hai người phụ nữ đang thể hiện tình cảm.

Sau đây là một ví dụ về *contradiction*, vì “chạy ví dụ lập trình” cho thấy “không ngủ” chứ không phải “đang ngủ”.

> Tiền đề: Một người đàn ông đang chạy ví dụ lập trình từ Dive into Deep Learning.

> Giả thuyết: Người đàn ông đang ngủ.

Ví dụ thứ ba cho thấy mối quan hệ *neutral* vì không thể suy ra “nổi tiếng” cũng như “không nổi tiếng” từ sự việc “đang biểu diễn cho chúng ta”.

> Tiền đề: Các nhạc sĩ đang biểu diễn cho chúng ta.

> Giả thuyết: Các nhạc sĩ nổi tiếng.

Suy luận ngôn ngữ tự nhiên đã trở thành một chủ đề trọng tâm trong việc tìm hiểu ngôn ngữ tự nhiên.
Nó có nhiều ứng dụng, từ
truy hồi thông tin đến trả lời câu hỏi trong miền mở.
Để nghiên cứu bài toán này, trước tiên chúng ta sẽ tìm hiểu một bộ dữ liệu chuẩn phổ biến về suy luận ngôn ngữ tự nhiên.


## Bộ dữ liệu Stanford Natural Language Inference (SNLI)

[**Kho ngữ liệu Stanford Natural Language Inference (SNLI)**] là một tập hợp gồm hơn 500000 cặp câu tiếng Anh được gán nhãn :cite:`Bowman.Angeli.Potts.ea.2015`.
Chúng ta tải xuống và lưu bộ dữ liệu SNLI đã giải nén tại đường dẫn `../data/snli_1.0`.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### [**Đọc bộ dữ liệu**]

Bộ dữ liệu SNLI gốc chứa nhiều thông tin phong phú hơn những gì chúng ta thực sự cần trong các thử nghiệm. Vì vậy, chúng ta định nghĩa một hàm `read_snli` để chỉ trích xuất một phần của bộ dữ liệu, sau đó trả về các danh sách tiền đề, giả thuyết và nhãn tương ứng.

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

Bây giờ, hãy [**in 3 cặp đầu tiên**] của tiền đề và giả thuyết, cùng với các nhãn của chúng (“0”, “1” và “2” lần lượt tương ứng với “entailment”, “contradiction” và “neutral”).

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

Tập huấn luyện có khoảng 550000 cặp,
còn tập kiểm tra có khoảng 10000 cặp.
Phần sau cho thấy
ba [**nhãn “entailment”, “contradiction” và “neutral” được phân bố cân bằng**] trong
cả tập huấn luyện và tập kiểm tra.

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### [**Định nghĩa một lớp để tải bộ dữ liệu**]

Dưới đây, chúng ta định nghĩa một lớp để tải bộ dữ liệu SNLI bằng cách kế thừa từ lớp `Dataset` trong Gluon. Đối số `num_steps` trong hàm khởi tạo của lớp chỉ định độ dài của một chuỗi văn bản để mỗi minibatch các chuỗi có cùng hình dạng.
Nói cách khác,
các token sau `num_steps` token đầu tiên trong chuỗi dài hơn sẽ bị cắt bớt, trong khi các token đặc biệt “&lt;pad&gt;” sẽ được thêm vào các chuỗi ngắn hơn cho đến khi độ dài của chúng đạt `num_steps`.
Bằng cách triển khai hàm `__getitem__`, chúng ta có thể truy cập tùy ý vào tiền đề, giả thuyết và nhãn bằng chỉ mục `idx`.

```{.python .input}
#@tab mxnet
#@save
class SNLIDataset(gluon.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

```{.python .input}
#@tab pytorch
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

### [**Kết hợp mọi thứ**]

Bây giờ, chúng ta có thể gọi hàm `read_snli` và lớp `SNLIDataset` để tải xuống bộ dữ liệu SNLI và trả về các thực thể `DataLoader` cho cả tập huấn luyện và tập kiểm tra, cùng với từ vựng của tập huấn luyện.
Điều đáng chú ý là chúng ta phải sử dụng từ vựng được xây dựng từ tập huấn luyện
làm từ vựng của tập kiểm tra.
Kết quả là, mọi token mới trong tập kiểm tra sẽ không được mô hình đã huấn luyện trên tập huấn luyện nhận biết.

```{.python .input}
#@tab mxnet
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

Ở đây, chúng ta đặt kích thước batch là 128 và độ dài chuỗi là 50,
sau đó gọi hàm `load_data_snli` để lấy các iterator dữ liệu và từ vựng.
Tiếp theo, chúng ta in kích thước của từ vựng.

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

Bây giờ, chúng ta in hình dạng của minibatch đầu tiên.
Trái với phân tích cảm xúc,
chúng ta có hai đầu vào `X[0]` và `X[1]` đại diện cho các cặp tiền đề và giả thuyết.

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## Tóm tắt

* Suy luận ngôn ngữ tự nhiên nghiên cứu liệu một giả thuyết có thể được suy ra từ một tiền đề hay không, trong đó cả hai đều là các chuỗi văn bản.
* Trong suy luận ngôn ngữ tự nhiên, các mối quan hệ giữa tiền đề và giả thuyết bao gồm entailment, contradiction và neutral.
* Kho ngữ liệu Stanford Natural Language Inference (SNLI) là một bộ dữ liệu chuẩn phổ biến về suy luận ngôn ngữ tự nhiên.


## Bài tập

1. Dịch máy từ lâu đã được đánh giá dựa trên việc khớp $n$-gram bề mặt giữa bản dịch đầu ra và bản dịch chuẩn. Bạn có thể thiết kế một phương pháp đánh giá kết quả dịch máy bằng cách sử dụng suy luận ngôn ngữ tự nhiên không?
1. Làm thế nào để thay đổi các hyperparameter nhằm giảm kích thước từ vựng?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/394)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1388)
:end_tab: