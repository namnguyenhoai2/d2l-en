# Phân tích cảm xúc và Tập dữ liệu
:label:`sec_sentiment`


Cùng với sự phổ biến rộng rãi của mạng xã hội trực tuyến
và các nền tảng đánh giá,
một lượng lớn
dữ liệu thể hiện ý kiến
đã được ghi lại,
mang lại tiềm năng to lớn trong việc
hỗ trợ các quy trình ra quyết định.
*Phân tích cảm xúc*
nghiên cứu cảm xúc của mọi người
trong các văn bản do họ tạo ra,
chẳng hạn như các bài đánh giá sản phẩm,
bình luận trên blog,
và
thảo luận trên diễn đàn.
Phân tích cảm xúc được ứng dụng rộng rãi
trong nhiều lĩnh vực đa dạng như 
chính trị (ví dụ: phân tích cảm xúc của công chúng đối với các chính sách),
tài chính (ví dụ: phân tích cảm xúc của thị trường),
và 
marketing (ví dụ: nghiên cứu sản phẩm và quản lý thương hiệu).

Vì cảm xúc
có thể được phân loại
thành các cực tính hoặc thang đo rời rạc (ví dụ: tích cực và tiêu cực),
chúng ta có thể xem 
phân tích cảm xúc 
là một tác vụ phân loại văn bản,
chuyển đổi một chuỗi văn bản có độ dài thay đổi
thành một danh mục văn bản có độ dài cố định.
Trong chương này,
chúng ta sẽ sử dụng [bộ dữ liệu đánh giá phim lớn](https://ai.stanford.edu/%7Eamaas/data/sentiment/) của Stanford
để phân tích cảm xúc. 
Bộ dữ liệu này gồm một tập huấn luyện và một tập kiểm thử, 
mỗi tập chứa 25000 bài đánh giá phim được tải xuống từ IMDb.
Trong cả hai tập dữ liệu, 
số lượng nhãn
"tích cực" và "tiêu cực" là bằng nhau,
biểu thị các cực tính cảm xúc khác nhau.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

##  Đọc Tập dữ liệu

Trước tiên, hãy tải xuống và giải nén tập dữ liệu đánh giá IMDb này
vào đường dẫn `../data/aclImdb`.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['aclImdb'] = (d2l.DATA_URL + 'aclImdb_v1.tar.gz', 
                          '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

Tiếp theo, hãy đọc các tập dữ liệu huấn luyện và kiểm thử. Mỗi ví dụ là một bài đánh giá cùng với nhãn của nó: 1 cho "tích cực" và 0 cho "tiêu cực".

```{.python .input}
#@tab all
#@save
def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels."""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[:60])
```

## Tiền xử lý Tập dữ liệu

Coi mỗi từ là một token
và lọc các từ xuất hiện ít hơn 5 lần,
chúng ta tạo một vocabulary từ tập dữ liệu huấn luyện.

```{.python .input}
#@tab all
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
```

Sau khi tokenization,
hãy vẽ histogram của
độ dài các bài đánh giá tính theo token.

```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.xlabel('# tokens per review')
d2l.plt.ylabel('count')
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```

Như dự đoán,
các bài đánh giá có độ dài thay đổi.
Để xử lý
một minibatch gồm các bài đánh giá như vậy trong mỗi lần,
chúng ta đặt độ dài của mỗi bài đánh giá thành 500 bằng cách cắt ngắn và padding,
tương tự như 
bước tiền xử lý 
cho tập dữ liệu machine translation
trong :numref:`sec_machine_translation`.

```{.python .input}
#@tab all
num_steps = 500  # sequence length
train_features = d2l.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)
```

## Tạo Data Iterator

Bây giờ chúng ta có thể tạo các data iterator.
Ở mỗi lần lặp, một minibatch gồm các ví dụ sẽ được trả về.

```{.python .input}
#@tab mxnet
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

```{.python .input}
#@tab pytorch
train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

## Kết hợp Tất cả

Cuối cùng, chúng ta gói các bước trên vào hàm `load_data_imdb`.
Hàm này trả về các data iterator cho dữ liệu huấn luyện và kiểm thử cùng vocabulary của tập dữ liệu đánh giá IMDb.

```{.python .input}
#@tab mxnet
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

## Tóm tắt

* Phân tích cảm xúc nghiên cứu cảm xúc của mọi người trong các văn bản do họ tạo ra. Đây được xem là một bài toán phân loại văn bản, chuyển đổi một chuỗi văn bản có độ dài thay đổi
thành một danh mục văn bản có độ dài cố định.
* Sau khi tiền xử lý, chúng ta có thể tải bộ dữ liệu đánh giá phim lớn của Stanford (tập dữ liệu đánh giá IMDb) vào các data iterator cùng với một vocabulary.


## Bài tập


1. Những hyperparameter nào trong phần này có thể được điều chỉnh để tăng tốc quá trình huấn luyện các mô hình phân tích cảm xúc?
1. Bạn có thể triển khai một hàm để tải tập dữ liệu [đánh giá Amazon](https://snap.stanford.edu/data/web-Amazon.html) vào các data iterator và nhãn phục vụ cho việc phân tích cảm xúc không?


:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/391)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1387)
:end_tab: