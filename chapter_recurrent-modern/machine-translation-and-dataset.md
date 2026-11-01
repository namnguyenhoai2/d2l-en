```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Dịch máy và tập dữ liệu
:label:`sec_machine_translation`

Một trong những đột phá lớn thúc đẩy 
sự quan tâm rộng rãi đến các RNN hiện đại
là một bước tiến quan trọng trong lĩnh vực ứng dụng của 
*dịch máy* thống kê.
Trong đó, mô hình được cung cấp một câu bằng một ngôn ngữ
và phải dự đoán câu tương ứng bằng một ngôn ngữ khác.
Lưu ý rằng ở đây các câu có thể có độ dài khác nhau,
và các từ tương ứng trong hai câu 
có thể không xuất hiện theo cùng một thứ tự,
do sự khác biệt 
trong cấu trúc ngữ pháp của hai ngôn ngữ.


Nhiều bài toán có dạng ánh xạ 
giữa hai chuỗi “không được căn chỉnh” như vậy.
Các ví dụ bao gồm ánh xạ 
từ lời nhắc trong hội thoại sang câu trả lời
hoặc từ câu hỏi sang đáp án.
Nhìn chung, các bài toán như vậy được gọi là 
bài toán *sequence-to-sequence* (seq2seq)
và đây là trọng tâm của 
phần còn lại của chương này
cũng như phần lớn :numref:`chap_attention-and-transformers`.

Trong phần này, chúng ta giới thiệu bài toán dịch máy
và một tập dữ liệu mẫu mà chúng ta sẽ sử dụng trong các ví dụ tiếp theo.
Trong nhiều thập kỷ, các cách xây dựng thống kê cho bài toán dịch giữa các ngôn ngữ
đã rất phổ biến :cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990`,
ngay cả trước khi các nhà nghiên cứu xây dựng được các phương pháp tiếp cận bằng mạng nơ-ron
(các phương pháp này thường được xếp chung dưới thuật ngữ *neural machine translation*).

Trước tiên, chúng ta cần một số mã mới để xử lý dữ liệu.
Không giống như mô hình hóa ngôn ngữ mà chúng ta đã thấy trong :numref:`sec_language-model`,
ở đây mỗi ví dụ gồm hai chuỗi văn bản riêng biệt,
một chuỗi bằng ngôn ngữ nguồn và chuỗi còn lại (bản dịch) bằng ngôn ngữ đích.
Các đoạn mã sau đây sẽ cho thấy cách 
nạp dữ liệu đã tiền xử lý vào các minibatch để huấn luyện.

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
import os
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

```{.python .input  n=4}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
import os
```

## [**Tải xuống và tiền xử lý tập dữ liệu**]

Để bắt đầu, chúng ta tải xuống một tập dữ liệu Anh--Pháp
gồm [cặp câu song ngữ từ Dự án Tatoeba](http://www.manythings.org/anki/).
Mỗi dòng trong tập dữ liệu là một cặp được phân tách bằng tab,
gồm một chuỗi văn bản tiếng Anh (gọi là *nguồn*)
và chuỗi văn bản tiếng Pháp đã dịch (gọi là *đích*).
Lưu ý rằng mỗi chuỗi văn bản
có thể chỉ là một câu,
hoặc một đoạn văn gồm nhiều câu.

```{.python .input  n=5}
%%tab all
class MTFraEng(d2l.DataModule):  #@save
    """The English-French dataset."""
    def _download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root, 
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()
```

```{.python .input}
%%tab all
data = MTFraEng() 
raw_text = data._download()
print(raw_text[:75])
```

Sau khi tải xuống tập dữ liệu,
chúng ta [**tiến hành một số bước tiền xử lý**]
cho dữ liệu văn bản thô.
Ví dụ, chúng ta thay thế khoảng trắng không ngắt bằng khoảng trắng,
chuyển các chữ cái viết hoa thành chữ thường,
và chèn khoảng trắng giữa các từ và dấu câu.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _preprocess(self, text):
    # Replace non-breaking space with space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Insert space between words and punctuation marks
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)
```

```{.python .input}
%%tab all
text = data._preprocess(raw_text)
print(text[:80])
```

## [**Tokenization**]

Không giống với tokenization ở cấp ký tự
trong :numref:`sec_language-model`,
đối với dịch máy,
ở đây chúng ta ưu tiên tokenization ở cấp từ
(các mô hình hiện đại đạt kết quả tốt nhất ngày nay sử dụng
các kỹ thuật tokenization phức tạp hơn).
Phương thức `_tokenize` sau đây
tokenize `max_examples` cặp chuỗi văn bản đầu tiên,
trong đó mỗi token là một từ hoặc một dấu câu.
Chúng ta nối thêm token đặc biệt “&lt;eos&gt;”
vào cuối mỗi chuỗi để biểu thị
điểm kết thúc chuỗi.
Khi một mô hình dự đoán
bằng cách tạo ra từng token của chuỗi,
việc tạo ra token “&lt;eos&gt;”
có thể cho biết chuỗi đầu ra đã hoàn tất.
Cuối cùng, phương thức dưới đây trả về
hai danh sách các danh sách token: `src` và `tgt`.
Cụ thể, `src[i]` là danh sách các token của
chuỗi văn bản thứ $i^\textrm{th}$ trong ngôn ngữ nguồn (ở đây là tiếng Anh)
và `tgt[i]` là danh sách tương ứng trong ngôn ngữ đích (ở đây là tiếng Pháp).

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _tokenize(self, text, max_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):
        if max_examples and i > max_examples: break
        parts = line.split('\t')
        if len(parts) == 2:
            # Skip empty tokens
            src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
    return src, tgt
```

```{.python .input}
%%tab all
src, tgt = data._tokenize(text)
src[:6], tgt[:6]
```

Hãy [**vẽ biểu đồ histogram về số lượng token trong mỗi chuỗi văn bản.**]
Trong tập dữ liệu Anh--Pháp đơn giản này,
hầu hết các chuỗi văn bản có ít hơn 20 token.

```{.python .input  n=8}
%%tab all
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs."""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)
```

```{.python .input}
%%tab all
show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', src, tgt);
```

## Nạp các chuỗi có độ dài cố định
:label:`subsec_loading-seq-fixed-len`

Hãy nhớ rằng trong mô hình hóa ngôn ngữ,
[**mỗi chuỗi ví dụ**],
dù là một đoạn của một câu
hay một phần trải dài qua nhiều câu,
(**đều có độ dài cố định.**)
Điều này được xác định bởi đối số `num_steps`
(số bước thời gian hoặc token) từ :numref:`sec_language-model`.
Trong dịch máy, mỗi ví dụ là
một cặp chuỗi văn bản nguồn và đích,
trong đó hai chuỗi văn bản có thể có độ dài khác nhau.

Để đạt hiệu quả tính toán,
chúng ta vẫn có thể xử lý một minibatch gồm các chuỗi văn bản
cùng một lúc bằng cách *cắt ngắn* và *đệm*.
Giả sử rằng mọi chuỗi trong cùng một minibatch
phải có cùng độ dài `num_steps`.
Nếu một chuỗi văn bản có ít hơn `num_steps` token,
chúng ta sẽ tiếp tục nối token đặc biệt “&lt;pad&gt;”
vào cuối chuỗi cho đến khi độ dài đạt `num_steps`.
Nếu không, chúng ta sẽ cắt ngắn chuỗi văn bản
bằng cách chỉ lấy `num_steps` token đầu tiên
và loại bỏ phần còn lại.
Theo cách này, mọi chuỗi văn bản
sẽ có cùng độ dài
để được nạp vào các minibatch có cùng hình dạng.
Ngoài ra, chúng ta cũng ghi lại độ dài của chuỗi nguồn, không tính các token đệm.
Thông tin này sẽ cần thiết cho một số mô hình mà chúng ta sẽ tìm hiểu sau.


Vì tập dữ liệu dịch máy
gồm các cặp ngôn ngữ,
chúng ta có thể xây dựng hai từ vựng
một cách riêng biệt cho ngôn ngữ nguồn
và ngôn ngữ đích.
Với tokenization ở cấp từ,
kích thước từ vựng sẽ lớn hơn đáng kể
so với tokenization ở cấp ký tự.
Để giảm vấn đề này,
ở đây chúng ta coi các token không thường xuyên
xuất hiện ít hơn hai lần
là cùng một token không xác định (“&lt;unk&gt;”).
Như sẽ giải thích sau (:numref:`fig_seq2seq`),
khi huấn luyện với các chuỗi đích,
đầu ra của decoder (các token nhãn)
có thể giống với đầu vào của decoder (các token đích),
nhưng dịch đi một token;
và token bắt đầu chuỗi đặc biệt “&lt;bos&gt;”
sẽ được dùng làm token đầu vào đầu tiên
để dự đoán chuỗi đích (:numref:`fig_seq2seq_predict`).

```{.python .input  n=9}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
    super(MTFraEng, self).__init__()
    self.save_hyperparameters()
    self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
        self._download())
```

```{.python .input}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
    def _build_array(sentences, vocab, is_tgt=False):
        pad_or_trim = lambda seq, t: (
            seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
        sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]
        if vocab is None:
            vocab = d2l.Vocab(sentences, min_freq=2)
        array = d2l.tensor([vocab[s] for s in sentences])
        valid_len = d2l.reduce_sum(
            d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
        return array, vocab, valid_len
    src, tgt = self._tokenize(self._preprocess(raw_text), 
                              self.num_train + self.num_val)
    src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
    tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
    return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
            src_vocab, tgt_vocab)
```

## [**Đọc tập dữ liệu**]

Cuối cùng, chúng ta định nghĩa phương thức `get_dataloader`
để trả về bộ lặp dữ liệu.

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(self.arrays, train, idx)
```

Hãy [**đọc minibatch đầu tiên từ tập dữ liệu Anh--Pháp.**]

```{.python .input  n=11}
%%tab all
data = MTFraEng(batch_size=3)
src, tgt, src_valid_len, label = next(iter(data.train_dataloader()))
print('source:', d2l.astype(src, d2l.int32))
print('decoder input:', d2l.astype(tgt, d2l.int32))
print('source len excluding pad:', d2l.astype(src_valid_len, d2l.int32))
print('label:', d2l.astype(label, d2l.int32))
```

Chúng ta hiển thị một cặp chuỗi nguồn và đích
được xử lý bởi phương thức `_build_arrays` ở trên
(ở dạng chuỗi).

```{.python .input  n=12}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def build(self, src_sentences, tgt_sentences):
    raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])
    arrays, _, _ = self._build_arrays(
        raw_text, self.src_vocab, self.tgt_vocab)
    return arrays
```

```{.python .input  n=13}
%%tab all
src, tgt, _,  _ = data.build(['hi .'], ['salut .'])
print('source:', data.src_vocab.to_tokens(d2l.astype(src[0], d2l.int32)))
print('target:', data.tgt_vocab.to_tokens(d2l.astype(tgt[0], d2l.int32)))
```

## Tóm tắt

Trong xử lý ngôn ngữ tự nhiên, *dịch máy* là bài toán tự động ánh xạ một chuỗi biểu diễn cho một đoạn văn bản trong ngôn ngữ *nguồn* thành một chuỗi biểu diễn cho một bản dịch hợp lý trong ngôn ngữ *đích*. Khi sử dụng tokenization ở cấp từ, kích thước từ vựng sẽ lớn hơn đáng kể so với tokenization ở cấp ký tự, nhưng độ dài chuỗi sẽ ngắn hơn nhiều. Để giảm kích thước từ vựng lớn, chúng ta có thể coi các token không thường xuyên là một token “không xác định” nào đó. Chúng ta có thể cắt ngắn và đệm các chuỗi văn bản để tất cả chúng có cùng độ dài nhằm nạp vào các minibatch. Các triển khai hiện đại thường gom các chuỗi có độ dài tương tự vào cùng bucket để tránh lãng phí quá nhiều phép tính cho việc đệm.


## Bài tập

1. Hãy thử các giá trị khác nhau cho đối số `max_examples` trong phương thức `_tokenize`. Điều này ảnh hưởng như thế nào đến kích thước từ vựng của ngôn ngữ nguồn và ngôn ngữ đích?
1. Văn bản trong một số ngôn ngữ như tiếng Trung và tiếng Nhật không có dấu hiệu phân cách từ (ví dụ: khoảng trắng). Trong những trường hợp như vậy, tokenization ở cấp từ có còn là một ý tưởng tốt không? Tại sao có hoặc tại sao không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1060)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/3863)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18020)
:end_tab: