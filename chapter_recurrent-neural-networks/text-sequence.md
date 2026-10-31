# Chuyển đổi Văn bản Thô thành Dữ liệu Chuỗi
:label:`sec_text-sequence`

Xuyên suốt cuốn sách này,
chúng ta sẽ thường xuyên làm việc với dữ liệu văn bản
được biểu diễn dưới dạng chuỗi
các từ, ký tự hoặc mảnh từ.
Để bắt đầu, chúng ta sẽ cần một số công cụ cơ bản
để chuyển đổi văn bản thô
thành các chuỗi có dạng phù hợp.
Các quy trình tiền xử lý điển hình
thực hiện các bước sau:

1. Tải văn bản dưới dạng chuỗi vào bộ nhớ.
1. Tách các chuỗi thành token (ví dụ: từ hoặc ký tự).
1. Xây dựng từ điển từ vựng để liên kết mỗi phần tử từ vựng với một chỉ số số.
1. Chuyển đổi văn bản thành các chuỗi chỉ số số.

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=2}
%%tab mxnet
import collections
import re
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
import collections
import re
from d2l import torch as d2l
import torch
import random
```

```{.python .input  n=4}
%%tab tensorflow
import collections
import re
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
import collections
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import random
import re
```

## Đọc Tập dữ liệu

Ở đây, chúng ta sẽ làm việc với tác phẩm của H. G. Wells
[The Time Machine](http://www.gutenberg.org/ebooks/35),
một cuốn sách có hơn 30.000 từ một chút.
Mặc dù các ứng dụng thực tế thường
liên quan đến những tập dữ liệu lớn hơn đáng kể,
điều này là đủ để minh họa
quy trình tiền xử lý.
Phương thức `_download` sau đây
(**đọc văn bản thô vào một chuỗi**).

```{.python .input  n=5}
%%tab all
class TimeMachine(d2l.DataModule): #@save
    """The Time Machine dataset."""
    def _download(self):
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root,
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

data = TimeMachine()
raw_text = data._download()
raw_text[:60]
```

Để đơn giản, chúng ta bỏ qua dấu câu và chữ hoa/chữ thường khi tiền xử lý văn bản thô.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

text = data._preprocess(raw_text)
text[:60]
```

## Phân tách Token

*Token* là các đơn vị nguyên tử (không thể chia nhỏ) của văn bản.
Mỗi bước thời gian tương ứng với 1 token,
nhưng chính xác token là gì là một lựa chọn thiết kế.
Ví dụ, chúng ta có thể biểu diễn câu
"Baby needs a new pair of shoes"
dưới dạng một chuỗi gồm 7 từ,
trong đó tập hợp tất cả các từ tạo thành
một từ vựng lớn (thường là hàng chục
hoặc hàng trăm nghìn từ).
Hoặc chúng ta có thể biểu diễn cùng câu đó
dưới dạng một chuỗi dài hơn nhiều gồm 30 ký tự,
sử dụng một từ vựng nhỏ hơn nhiều
(chỉ có 256 ký tự ASCII riêng biệt).
Bên dưới, chúng ta token hóa văn bản đã tiền xử lý
thành một chuỗi ký tự.

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _tokenize(self, text):
    return list(text)

tokens = data._tokenize(text)
','.join(tokens[:30])
```

## Từ vựng

Các token này vẫn là chuỗi.
Tuy nhiên, đầu vào cho các mô hình của chúng ta
cuối cùng phải bao gồm
các đầu vào dạng số.
[**Tiếp theo, chúng ta giới thiệu một lớp
để xây dựng *từ vựng*,
tức là các đối tượng liên kết
mỗi giá trị token riêng biệt
với một chỉ số duy nhất.**]
Trước tiên, chúng ta xác định tập token duy nhất trong *corpus* huấn luyện.
Sau đó, chúng ta gán một chỉ số số cho mỗi token duy nhất.
Các phần tử từ vựng hiếm thường bị loại bỏ để thuận tiện.
Bất cứ khi nào gặp một token trong lúc huấn luyện hoặc kiểm thử
chưa từng được thấy trước đó hoặc đã bị loại khỏi từ vựng,
chúng ta biểu diễn nó bằng token đặc biệt "&lt;unk&gt;",
cho biết đây là một giá trị *không xác định*.

```{.python .input  n=8}
%%tab all
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']
```

Bây giờ, chúng ta [**xây dựng một từ vựng**] cho tập dữ liệu của mình,
chuyển đổi chuỗi các chuỗi
thành một danh sách các chỉ số số.
Lưu ý rằng chúng ta không mất bất kỳ thông tin nào
và có thể dễ dàng chuyển đổi tập dữ liệu của mình
trở lại biểu diễn ban đầu (chuỗi).

```{.python .input  n=9}
%%tab all
vocab = Vocab(tokens)
indices = vocab[tokens[:10]]
print('indices:', indices)
print('words:', vocab.to_tokens(indices))
```

## Kết hợp Tất cả

Sử dụng các lớp và phương thức ở trên,
chúng ta [**đóng gói mọi thứ vào
phương thức `build` sau đây của lớp `TimeMachine`**],
phương thức này trả về `corpus`, một danh sách các chỉ số token, và `vocab`,
từ vựng của corpus *The Time Machine*.
Các thay đổi chúng ta đã thực hiện ở đây là:
(i) chúng ta token hóa văn bản thành ký tự, không phải từ,
để đơn giản hóa việc huấn luyện trong các phần sau;
(ii) `corpus` là một danh sách đơn, không phải danh sách các danh sách token,
vì mỗi dòng văn bản trong tập dữ liệu *The Time Machine*
không nhất thiết là một câu hoặc đoạn văn.

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def build(self, raw_text, vocab=None):
    tokens = self._tokenize(self._preprocess(raw_text))
    if vocab is None: vocab = Vocab(tokens)
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab

corpus, vocab = data.build(raw_text)
len(corpus), len(vocab)
```

## Thống kê Ngôn ngữ Khám phá
:label:`subsec_natural-lang-stat`

Sử dụng corpus thực tế và lớp `Vocab` được định nghĩa trên các từ,
chúng ta có thể kiểm tra các thống kê cơ bản về việc sử dụng từ trong corpus của mình.
Bên dưới, chúng ta xây dựng một từ vựng từ các từ được dùng trong *The Time Machine*
và in ra mười từ xuất hiện thường xuyên nhất.

```{.python .input  n=11}
%%tab all
words = text.split()
vocab = Vocab(words)
vocab.token_freqs[:10]
```

Lưu ý rằng (**mười từ xuất hiện thường xuyên nhất**)
không mang nhiều tính mô tả.
Bạn thậm chí có thể hình dung rằng
chúng ta có thể thấy một danh sách rất tương tự
nếu chọn ngẫu nhiên bất kỳ cuốn sách nào.
Các mạo từ như "the" và "a",
đại từ như "i" và "my",
và giới từ như "of", "to", và "in"
xuất hiện thường xuyên vì chúng đảm nhiệm các vai trò cú pháp phổ biến.
Những từ phổ biến nhưng không đặc biệt mang tính mô tả như vậy
thường được gọi là (***stop words***) và,
trong các thế hệ trước của bộ phân loại văn bản
dựa trên cái gọi là biểu diễn bag-of-words,
chúng thường bị lọc bỏ.
Tuy nhiên, chúng mang ý nghĩa và
không cần thiết phải lọc chúng ra
khi làm việc với các mô hình mạng nơ-ron hiện đại
dựa trên RNN và Transformer.
Nếu nhìn sâu hơn xuống danh sách,
bạn sẽ nhận thấy
tần suất từ giảm nhanh.
Từ phổ biến thứ $10^{\textrm{th}}$
có tần suất chưa đến $1/5$ so với từ phổ biến nhất.
Tần suất từ có xu hướng tuân theo phân phối luật lũy thừa
(cụ thể là Zipfian) khi chúng ta đi xuống theo thứ hạng.
Để có hình dung rõ hơn, chúng ta [**vẽ biểu đồ tần suất từ**].

```{.python .input  n=12}
%%tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

Sau khi coi vài từ đầu tiên là các ngoại lệ,
tất cả các từ còn lại gần đúng theo một đường thẳng trên biểu đồ log--log.
Hiện tượng này được mô tả bởi *định luật Zipf*,
phát biểu rằng tần suất $n_i$
của từ phổ biến thứ $i^\textrm{th}$ là:

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

tương đương với

$$\log n_i = -\alpha \log i + c,$$

trong đó $\alpha$ là số mũ đặc trưng cho
phân phối và $c$ là một hằng số.
Điều này hẳn đã khiến chúng ta phải suy nghĩ nếu muốn
mô hình hóa từ bằng thống kê đếm.
Xét cho cùng, chúng ta sẽ ước lượng quá cao đáng kể tần suất của phần đuôi, còn được gọi là các từ không thường xuyên. Nhưng [**còn các tổ hợp từ khác thì sao, chẳng hạn như hai từ liên tiếp (bigram), ba từ liên tiếp (trigram)**], và xa hơn nữa?
Hãy xem liệu tần suất bigram có hành xử giống tần suất từ đơn lẻ (unigram) hay không.

```{.python .input  n=13}
%%tab all
bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
bigram_vocab = Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

Có một điều đáng chú ý ở đây. Trong mười cặp từ thường xuyên nhất, chín cặp chỉ gồm các stop word và chỉ một cặp liên quan đến nội dung thực tế của cuốn sách---"the time". Hơn nữa, hãy xem liệu tần suất trigram có hành xử theo cùng cách hay không.

```{.python .input  n=14}
%%tab all
trigram_tokens = ['--'.join(triple) for triple in zip(
    words[:-2], words[1:-1], words[2:])]
trigram_vocab = Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

Bây giờ, hãy [**trực quan hóa tần suất token**] trong ba mô hình này: unigram, bigram và trigram.

```{.python .input  n=15}
%%tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

Hình này khá thú vị.
Thứ nhất, ngoài các từ unigram, các chuỗi từ
dường như cũng tuân theo định luật Zipf,
mặc dù có số mũ
$\alpha$ nhỏ hơn trong :eqref:`eq_zipf_law`,
phụ thuộc vào độ dài chuỗi.
Thứ hai, số lượng $n$-gram riêng biệt không quá lớn.
Điều này cho chúng ta hy vọng rằng ngôn ngữ có khá nhiều cấu trúc.
Thứ ba, nhiều $n$-gram xuất hiện rất hiếm.
Điều này khiến một số phương pháp không phù hợp cho mô hình hóa ngôn ngữ
và thúc đẩy việc sử dụng các mô hình deep learning.
Chúng ta sẽ thảo luận điều này trong phần tiếp theo.


## Tóm tắt

Văn bản là một trong những dạng dữ liệu chuỗi phổ biến nhất gặp trong deep learning.
Các lựa chọn phổ biến về thành phần tạo nên một token là ký tự, từ và mảnh từ.
Để tiền xử lý văn bản, chúng ta thường (i) tách văn bản thành token; (ii) xây dựng từ vựng để ánh xạ các chuỗi token thành chỉ số số; và (iii) chuyển đổi dữ liệu văn bản thành các chỉ số token để mô hình xử lý.
Trong thực tế, tần suất từ có xu hướng tuân theo định luật Zipf. Điều này không chỉ đúng với các từ riêng lẻ (unigram), mà còn đúng với $n$-gram.


## Bài tập

1. Trong thí nghiệm của phần này, token hóa văn bản thành từ và thay đổi giá trị đối số `min_freq` của instance `Vocab`. Mô tả định tính cách các thay đổi trong `min_freq` ảnh hưởng đến kích thước từ vựng thu được.
1. Ước lượng số mũ của phân phối Zipfian cho unigram, bigram và trigram trong corpus này.
1. Tìm một số nguồn dữ liệu khác (tải xuống một tập dữ liệu machine learning tiêu chuẩn, chọn một cuốn sách thuộc phạm vi công cộng khác,
   scrape một website, v.v.). Với mỗi nguồn, token hóa dữ liệu ở cả cấp độ từ và ký tự. Kích thước từ vựng so với corpus *The Time Machine* như thế nào tại các giá trị tương đương của `min_freq`? Ước lượng số mũ của phân phối Zipfian tương ứng với các phân phối unigram và bigram cho các corpus này. Chúng so sánh thế nào với các giá trị bạn quan sát được đối với corpus *The Time Machine*?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1049)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18011)
:end_tab: