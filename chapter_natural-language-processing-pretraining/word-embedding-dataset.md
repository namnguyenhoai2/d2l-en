# Bộ dữ liệu để pretrain word embedding
:label:`sec_word2vec_data`

Giờ đây, khi đã biết các chi tiết kỹ thuật
của các mô hình word2vec và các phương pháp training xấp xỉ,
hãy cùng tìm hiểu cách triển khai chúng.
Cụ thể,
chúng ta sẽ lấy mô hình skip-gram trong :numref:`sec_word2vec`
và negative sampling trong :numref:`sec_approx_train`
làm ví dụ.
Trong phần này,
chúng ta bắt đầu với dataset
dùng để pretrain mô hình word embedding:
định dạng ban đầu của dữ liệu
sẽ được chuyển đổi
thành các minibatch
có thể được lặp qua trong quá trình training.

```{.python .input}
#@tab mxnet
import collections
from d2l import mxnet as d2l
import math
from mxnet import gluon, np
import os
import random
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
import os
import random
```

## Đọc Dataset

Dataset mà chúng ta sử dụng ở đây
là [Penn Tree Bank (PTB)]( https://catalog.ldc.upenn.edu/LDC99T42).
Corpus này được lấy mẫu
từ các bài viết của Wall Street Journal,
sau đó được chia thành các tập training, validation và test.
Ở định dạng ban đầu,
mỗi dòng trong file văn bản
đại diện cho một câu gồm các từ được phân tách bằng dấu cách.
Ở đây, chúng ta coi mỗi từ là một token.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    """Load the PTB dataset into a list of text lines."""
    data_dir = d2l.download_extract('ptb')
    # Read the training set
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# sentences: {len(sentences)}'
```

Sau khi đọc tập training,
chúng ta xây dựng một vocabulary cho corpus,
trong đó mọi từ xuất hiện
ít hơn 10 lần sẽ được thay thế bằng
token "&lt;unk&gt;".
Lưu ý rằng dataset ban đầu
cũng chứa các token "&lt;unk&gt;" đại diện cho các từ hiếm (không xác định).

```{.python .input}
#@tab all
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size: {len(vocab)}'
```

## Subsampling

Dữ liệu văn bản
thường có các từ xuất hiện với tần suất cao
như "the", "a" và "in":
chúng thậm chí có thể xuất hiện hàng tỷ lần trong
các corpus rất lớn.
Tuy nhiên,
những từ này thường đồng xuất hiện
với nhiều từ khác nhau
trong các context window, nên cung cấp rất ít tín hiệu hữu ích.
Ví dụ,
hãy xét từ "chip" trong một context window:
theo trực giác,
việc nó đồng xuất hiện với một từ có tần suất thấp như "intel"
hữu ích hơn trong quá trình training
so với
việc đồng xuất hiện với một từ có tần suất cao như "a".
Hơn nữa, training với một lượng lớn các từ (có tần suất cao)
sẽ chậm.
Do đó, khi training các mô hình word embedding,
các từ có tần suất cao có thể được *subsample* :cite:`Mikolov.Sutskever.Chen.ea.2013`.
Cụ thể,
mỗi từ đã được đánh chỉ số $w_i$
trong dataset sẽ bị loại bỏ với xác suất


$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

trong đó $f(w_i)$ là tỷ lệ giữa
số lượng từ $w_i$
và tổng số từ trong dataset,
còn hằng số $t$ là một hyperparameter
($10^{-4}$ trong thí nghiệm).
Ta có thể thấy rằng chỉ khi
tần suất tương đối
$f(w_i) > t$ thì từ (có tần suất cao) $w_i$ mới có thể bị loại bỏ,
và tần suất tương đối của từ càng cao
thì xác suất bị loại bỏ càng lớn.

```{.python .input}
#@tab all
#@save
def subsample(sentences, vocab):
    """Subsample high-frequency words."""
    # Exclude unknown tokens ('<unk>')
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = collections.Counter([
        token for line in sentences for token in line])
    num_tokens = sum(counter.values())

    # Return True if `token` is kept during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

Đoạn code sau
vẽ histogram của
số lượng token trong mỗi câu
trước và sau khi subsampling.
Đúng như dự kiến,
subsampling rút ngắn đáng kể các câu
bằng cách loại bỏ những từ có tần suất cao,
từ đó giúp tăng tốc độ training.

```{.python .input}
#@tab all
d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence',
                            'count', sentences, subsampled);
```

Đối với từng token,
tỷ lệ lấy mẫu của từ có tần suất cao "the" nhỏ hơn 1/20.

```{.python .input}
#@tab all
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

Ngược lại,
các từ có tần suất thấp như "join" được giữ lại hoàn toàn.

```{.python .input}
#@tab all
compare_counts('join')
```

Sau khi subsampling,
chúng ta ánh xạ các token thành các chỉ số của chúng trong corpus.

```{.python .input}
#@tab all
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## Trích xuất Center Word và Context Word


Hàm `get_centers_and_contexts`
sau đây trích xuất tất cả
center word và các context word tương ứng
từ `corpus`.
Hàm lấy mẫu đồng đều một số nguyên trong khoảng từ 1 đến `max_window_size`
một cách ngẫu nhiên làm kích thước context window.
Với mỗi center word,
những từ
có khoảng cách đến nó
không vượt quá
kích thước context window đã lấy mẫu
sẽ là các context word của nó.

```{.python .input}
#@tab all
#@save
def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram."""
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at `i`
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

Tiếp theo, chúng ta tạo một dataset nhân tạo gồm hai câu lần lượt có 7 và 3 từ.
Cho kích thước context window tối đa là 2
và in tất cả center word cùng các context word tương ứng.

```{.python .input}
#@tab all
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

Khi training trên dataset PTB,
chúng ta đặt kích thước context window tối đa là 5.
Đoạn code sau trích xuất tất cả center word và context word tương ứng trong dataset.

```{.python .input}
#@tab all
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}'
```

## Negative Sampling

Chúng ta sử dụng negative sampling để training xấp xỉ.
Để lấy mẫu các noise word theo
một phân phối được định trước,
chúng ta định nghĩa class `RandomGenerator` sau đây,
trong đó phân phối lấy mẫu (có thể chưa chuẩn hóa) được truyền vào
thông qua đối số `sampling_weights`.

```{.python .input}
#@tab all
#@save
class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights."""
    def __init__(self, sampling_weights):
        # Exclude 
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Cache `k` random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

Ví dụ,
chúng ta có thể lấy 10 biến ngẫu nhiên $X$
từ các chỉ số 1, 2 và 3
với xác suất lấy mẫu $P(X=1)=2/9, P(X=2)=3/9$ và $P(X=3)=4/9$ như sau.

```{.python .input}
#@tab mxnet
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

Với một cặp gồm center word và context word,
chúng ta lấy mẫu ngẫu nhiên `K` noise word (5 trong thí nghiệm).
Theo đề xuất trong bài báo word2vec,
xác suất lấy mẫu $P(w)$ của
một noise word $w$
được
đặt bằng tần suất tương đối của nó
trong dictionary
lũy thừa
0.75 :cite:`Mikolov.Sutskever.Chen.ea.2013`.

```{.python .input}
#@tab all
#@save
def get_negatives(all_contexts, vocab, counter, K):
    """Return noise words in negative sampling."""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## Nạp Training Example theo Minibatch
:label:`subsec_word2vec-minibatch-loading`

Sau khi
tất cả center word
cùng với
context word và noise word được lấy mẫu
đã được trích xuất,
chúng sẽ được chuyển đổi thành
các minibatch gồm các example
có thể được nạp lặp đi lặp lại
trong quá trình training.



Trong một minibatch,
example thứ $i^\textrm{th}$ bao gồm một center word
và $n_i$ context word cùng $m_i$ noise word.
Do kích thước context window thay đổi,
$n_i+m_i$ sẽ khác nhau đối với các $i$ khác nhau.
Vì vậy,
với mỗi example,
chúng ta nối các context word và noise word của nó trong
biến `contexts_negatives`,
sau đó thêm các số 0 cho đến khi độ dài của chuỗi nối
đạt $\max_i n_i+m_i$ (`max_len`).
Để loại trừ phần padding
khỏi phép tính loss,
chúng ta định nghĩa một biến mask `masks`.
Có sự tương ứng một-một
giữa các phần tử trong `masks` và các phần tử trong `contexts_negatives`,
trong đó các số 0 (các phần tử còn lại là số 1) trong `masks`
tương ứng với phần padding trong `contexts_negatives`.


Để phân biệt các example dương và âm,
chúng ta tách context word khỏi noise word trong `contexts_negatives` thông qua một biến `labels`.
Tương tự như `masks`,
cũng có sự tương ứng một-một
giữa các phần tử trong `labels` và các phần tử trong `contexts_negatives`,
trong đó các số 1 (các phần tử còn lại là số 0) trong `labels`
tương ứng với các context word (example dương) trong `contexts_negatives`.


Ý tưởng trên được triển khai trong hàm `batchify` sau đây.
Đầu vào `data` của hàm là một danh sách có độ dài
bằng kích thước batch,
trong đó mỗi phần tử là một example
gồm
center word `center`, context word `context` và noise word `negative` của nó.
Hàm này trả về
một minibatch có thể được nạp để thực hiện các phép tính
trong quá trình training,
chẳng hạn như bao gồm biến mask.

```{.python .input}
#@tab all
#@save
def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling."""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))
```

Hãy kiểm thử hàm này bằng một minibatch gồm hai example.

```{.python .input}
#@tab all
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## Kết hợp Tất cả

Cuối cùng, chúng ta định nghĩa hàm `load_data_ptb` để đọc dataset PTB và trả về data iterator cùng vocabulary.

```{.python .input}
#@tab mxnet
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(
        dataset, batch_size, shuffle=True,batchify_fn=batchify,
        num_workers=d2l.get_dataloader_workers())
    return data_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=batchify,
                                      num_workers=num_workers)
    return data_iter, vocab
```

Hãy in minibatch đầu tiên của data iterator.

```{.python .input}
#@tab all
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## Tóm tắt

* Các từ có tần suất cao có thể không hữu ích trong quá trình training. Chúng ta có thể subsample chúng để tăng tốc độ training.
* Để đạt hiệu quả tính toán, chúng ta nạp các example theo minibatch. Chúng ta có thể định nghĩa các biến khác để phân biệt phần padding với phần không padding, cũng như các example dương với các example âm.



## Bài tập

1. Thời gian chạy của code trong phần này thay đổi như thế nào nếu không sử dụng subsampling?
1. Class `RandomGenerator` lưu vào cache `k` kết quả lấy mẫu ngẫu nhiên. Hãy đặt `k` thành các giá trị khác và quan sát ảnh hưởng của nó đến tốc độ nạp dữ liệu.
1. Những hyperparameter nào khác trong code của phần này có thể ảnh hưởng đến tốc độ nạp dữ liệu?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/383)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1330)
:end_tab: