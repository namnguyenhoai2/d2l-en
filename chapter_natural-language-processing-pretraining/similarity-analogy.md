# Độ tương đồng và phép loại suy từ
:label:`sec_synonyms`

Trong :numref:`sec_word2vec_pretraining`, 
chúng ta đã huấn luyện một mô hình word2vec trên một tập dữ liệu nhỏ 
và áp dụng mô hình đó
để tìm các từ tương đồng về mặt ngữ nghĩa 
với một từ đầu vào.
Trong thực tế,
các vector từ được pretrained
trên những kho ngữ liệu lớn có thể được
áp dụng cho các tác vụ
xử lý ngôn ngữ tự nhiên
ở các bước tiếp theo,
sẽ được trình bày sau
trong :numref:`chap_nlp_app`.
Để minh họa
một cách trực quan ngữ nghĩa của các vector từ đã được pretrained
từ những kho ngữ liệu lớn,
hãy áp dụng chúng
cho các tác vụ độ tương đồng và phép loại suy từ.

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

## Tải các Vector Từ Đã Pretrained

Dưới đây liệt kê các embedding GloVe đã được pretrained với số chiều 50, 100 và 300,
có thể tải xuống từ [trang web GloVe](https://nlp.stanford.edu/projects/glove/).
Các embedding fastText đã được pretrained có sẵn trong nhiều ngôn ngữ.
Ở đây, chúng ta xét một phiên bản tiếng Anh (300 chiều, "wiki.en") có thể tải xuống từ
[trang web fastText](https://fasttext.cc/).

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

Để tải các embedding GloVe và fastText đã được pretrained này, chúng ta định nghĩa `TokenEmbedding` class sau.

```{.python .input}
#@tab all
#@save
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

Dưới đây, chúng ta tải các embedding GloVe 50 chiều
(được pretrained trên một tập con của Wikipedia).
Khi tạo một instance `TokenEmbedding`,
tệp embedding được chỉ định sẽ phải được tải xuống nếu
chưa được tải xuống trước đó.

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

Xuất kích thước từ vựng. Từ vựng chứa 400000 từ (token) và một token đặc biệt dành cho từ chưa biết.

```{.python .input}
#@tab all
len(glove_6b50d)
```

Chúng ta có thể lấy chỉ mục của một từ trong từ vựng và ngược lại.

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## Áp dụng các Vector Từ Đã Pretrained

Sử dụng các vector GloVe đã tải,
chúng ta sẽ minh họa ngữ nghĩa của chúng
bằng cách áp dụng chúng
cho các tác vụ độ tương đồng và phép loại suy từ sau đây.


### Độ Tương đồng Từ

Tương tự như :numref:`subsec_apply-word-embed`,
để tìm các từ tương đồng về mặt ngữ nghĩa
với một từ đầu vào
dựa trên độ tương đồng cosine giữa
các vector từ,
chúng ta triển khai hàm `knn`
($k$-nearest neighbors) sau.

```{.python .input}
#@tab mxnet
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```{.python .input}
#@tab pytorch
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

Sau đó, chúng ta
tìm kiếm các từ tương tự
bằng cách sử dụng các vector từ đã pretrained
từ instance `TokenEmbedding` `embed`.

```{.python .input}
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

Từ vựng của các vector từ đã pretrained
trong `glove_6b50d` chứa 400000 từ và một token đặc biệt dành cho từ chưa biết.
Không tính từ đầu vào và token dành cho từ chưa biết,
trong từ vựng này,
hãy tìm
ba từ tương đồng nhất về mặt ngữ nghĩa
với từ "chip".

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

Dưới đây xuất ra các từ tương tự
với "baby" và "beautiful".

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### Phép Loại Suy Từ

Bên cạnh việc tìm các từ tương tự,
chúng ta cũng có thể áp dụng các vector từ
cho các tác vụ phép loại suy từ.
Ví dụ,
“man”:“woman”::“son”:“daughter”
là dạng của một phép loại suy từ:
“man” đối với “woman” cũng như “son” đối với “daughter”.
Cụ thể,
tác vụ hoàn thành phép loại suy từ
có thể được định nghĩa như sau:
với một phép loại suy từ
$a : b :: c : d$, cho trước ba từ đầu tiên $a$, $b$ và $c$, hãy tìm $d$.
Ký hiệu vector của từ $w$ là $\textrm{vec}(w)$.
Để hoàn thành phép loại suy,
chúng ta sẽ tìm từ
có vector tương đồng nhất
với kết quả của $\textrm{vec}(c)+\textrm{vec}(b)-\textrm{vec}(a)$.

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

Hãy kiểm chứng phép loại suy "nam-nữ" bằng các vector từ đã tải.

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

Dưới đây hoàn thành một
phép loại suy “thủ đô-quốc gia”:
“beijing”:“china”::“tokyo”:“japan”.
Điều này minh họa
ngữ nghĩa trong các vector từ đã pretrained.

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

Đối với
phép loại suy “tính từ-tính từ so sánh nhất”
chẳng hạn như
“bad”:“worst”::“big”:“biggest”,
chúng ta có thể thấy rằng các vector từ đã pretrained
có thể nắm bắt thông tin cú pháp.

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

Để cho thấy khái niệm
thì quá khứ được nắm bắt trong các vector từ đã pretrained,
chúng ta có thể kiểm tra cú pháp bằng
phép loại suy "thì hiện tại-thì quá khứ": “do”:“did”::“go”:“went”.

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## Tóm tắt

* Trong thực tế, các vector từ được pretrained trên những kho ngữ liệu lớn có thể được áp dụng cho các tác vụ xử lý ngôn ngữ tự nhiên ở các bước tiếp theo.
* Các vector từ đã pretrained có thể được áp dụng cho các tác vụ độ tương đồng và phép loại suy từ.


## Bài tập

1. Kiểm tra các kết quả của fastText bằng `TokenEmbedding('wiki.en')`.
1. Khi từ vựng cực kỳ lớn, làm thế nào chúng ta có thể tìm các từ tương tự hoặc hoàn thành một phép loại suy từ nhanh hơn?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/387)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1336)
:end_tab: