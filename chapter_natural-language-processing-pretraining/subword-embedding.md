# Nhúng từ con
:label:`sec_fasttext`

Trong tiếng Anh,
các từ như
"helps", "helped" và "helping" là
những dạng biến tố của cùng một từ "help".
Mối quan hệ
giữa "dog" và "dogs"
cũng giống như
mối quan hệ
giữa "cat" và "cats",
và
mối quan hệ
giữa "boy" và "boyfriend"
cũng giống như
mối quan hệ
giữa "girl" và "girlfriend".
Trong các ngôn ngữ khác
như tiếng Pháp và tiếng Tây Ban Nha,
nhiều động từ có hơn 40 dạng biến tố,
trong khi ở tiếng Phần Lan,
một danh từ có thể có tới 15 cách.
Trong ngôn ngữ học,
hình thái học nghiên cứu sự hình thành từ và mối quan hệ giữa các từ.
Tuy nhiên,
cấu trúc bên trong của từ
không được khám phá trong word2vec
cũng như trong GloVe.

## Mô hình fastText

Hãy nhớ lại cách các từ được biểu diễn trong word2vec.
Trong cả mô hình skip-gram
và mô hình continuous bag-of-words,
các dạng biến tố khác nhau của cùng một từ
được biểu diễn trực tiếp bằng các vector khác nhau
mà không dùng chung tham số.
Để sử dụng thông tin hình thái học,
mô hình *fastText*
đề xuất cách tiếp cận *nhúng từ con* (subword embedding),
trong đó một từ con là một character $n$-gram :cite:`Bojanowski.Grave.Joulin.ea.2017`.
Thay vì học các biểu diễn vector ở cấp độ từ,
fastText có thể được xem là
skip-gram ở cấp độ từ con,
trong đó mỗi *từ trung tâm* được biểu diễn bằng tổng
các vector từ con của nó.

Hãy minh họa cách thu được
các từ con cho mỗi từ trung tâm trong fastText
bằng từ "where".
Trước tiên, thêm các ký tự đặc biệt “&lt;” và “&gt;”
vào đầu và cuối từ để phân biệt tiền tố và hậu tố với các từ con khác.
Sau đó, trích xuất các character $n$-gram từ từ đó.
Ví dụ, khi $n=3$,
ta thu được tất cả các từ con có độ dài 3: "&lt;wh", "whe", "her", "ere", "re&gt;", và từ con đặc biệt "&lt;where&gt;".


Trong fastText, với mọi từ $w$,
ký hiệu $\mathcal{G}_w$
là hợp của tất cả các từ con có độ dài từ 3 đến 6
và từ con đặc biệt của nó.
Từ vựng
là hợp của các từ con của tất cả các từ.
Gọi $\mathbf{z}_g$
là vector của từ con $g$ trong từ điển,
vector $\mathbf{v}_w$ của
từ $w$ khi làm từ trung tâm
trong mô hình skip-gram
là tổng các vector từ con của nó:

$$\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

Phần còn lại của fastText giống với mô hình skip-gram. So với mô hình skip-gram,
từ vựng trong fastText lớn hơn,
dẫn đến nhiều tham số mô hình hơn.
Ngoài ra,
để tính biểu diễn của một từ,
phải cộng tất cả các vector từ con của từ đó,
dẫn đến độ phức tạp tính toán cao hơn.
Tuy nhiên,
nhờ các tham số dùng chung từ các từ con giữa những từ có cấu trúc tương tự,
các từ hiếm và thậm chí cả các từ ngoài từ vựng
có thể nhận được các biểu diễn vector tốt hơn trong fastText.



## Byte Pair Encoding
:label:`subsec_Byte_Pair_Encoding`

Trong fastText, tất cả các từ con được trích xuất phải có độ dài được chỉ định, chẳng hạn từ $3$ đến $6$, vì vậy không thể xác định trước kích thước từ vựng.
Để cho phép các từ con có độ dài thay đổi trong một từ vựng có kích thước cố định,
ta có thể áp dụng một thuật toán nén
gọi là *byte pair encoding* (BPE) để trích xuất các từ con :cite:`Sennrich.Haddow.Birch.2015`.

Byte pair encoding thực hiện phân tích thống kê trên tập dữ liệu huấn luyện để phát hiện các ký hiệu phổ biến trong một từ,
chẳng hạn các ký tự liên tiếp có độ dài tùy ý.
Bắt đầu từ các ký hiệu có độ dài 1,
byte pair encoding lặp đi lặp lại việc gộp cặp ký hiệu liên tiếp xuất hiện thường xuyên nhất để tạo ra các ký hiệu mới dài hơn.
Lưu ý rằng để tăng hiệu quả, các cặp vượt qua ranh giới giữa các từ không được xét đến.
Cuối cùng, ta có thể dùng những ký hiệu này làm từ con để phân đoạn các từ.
Byte pair encoding và các biến thể của nó đã được sử dụng cho các biểu diễn đầu vào trong những mô hình pretraining xử lý ngôn ngữ tự nhiên phổ biến như GPT-2 :cite:`Radford.Wu.Child.ea.2019` và RoBERTa :cite:`Liu.Ott.Goyal.ea.2019`.
Sau đây, ta sẽ minh họa cách byte pair encoding hoạt động.

Trước tiên, ta khởi tạo từ vựng các ký hiệu gồm tất cả các ký tự thường tiếng Anh, một ký hiệu đặc biệt biểu thị kết thúc từ `'_'`, và một ký hiệu đặc biệt biểu thị ký hiệu không xác định `'[UNK]'`.

```{.python .input}
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

Vì không xét các cặp ký hiệu vượt qua ranh giới giữa các từ,
ta chỉ cần một từ điển `raw_token_freqs` ánh xạ các từ tới tần suất của chúng (số lần xuất hiện)
trong một tập dữ liệu.
Lưu ý rằng ký hiệu đặc biệt `'_'` được thêm vào mỗi từ để
ta có thể dễ dàng khôi phục một chuỗi từ (ví dụ: "a taller man")
từ một chuỗi các ký hiệu đầu ra (ví dụ: "a_ tall er_ man").
Vì bắt đầu quá trình gộp từ một từ vựng chỉ gồm các ký tự đơn và các ký hiệu đặc biệt, dấu cách được chèn giữa mọi cặp ký tự liên tiếp trong mỗi từ (các khóa của từ điển `token_freqs`).
Nói cách khác, dấu cách là dấu phân cách giữa các ký hiệu trong một từ.

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

Ta định nghĩa hàm `get_max_freq_pair` sau đây,
hàm trả về cặp ký hiệu liên tiếp xuất hiện thường xuyên nhất trong một từ,
trong đó các từ được lấy từ các khóa của từ điển đầu vào `token_freqs`.

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # Key of `pairs` is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of `pairs` with the max value
```

Là một phương pháp greedy dựa trên tần suất của các ký hiệu liên tiếp,
byte pair encoding sẽ sử dụng hàm `merge_symbols` sau đây để gộp cặp ký hiệu liên tiếp xuất hiện thường xuyên nhất nhằm tạo ra các ký hiệu mới.

```{.python .input}
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

Bây giờ, ta lặp lại thuật toán byte pair encoding trên các khóa của từ điển `token_freqs`. Trong lần lặp đầu tiên, cặp ký hiệu liên tiếp xuất hiện thường xuyên nhất là `'t'` và `'a'`, do đó byte pair encoding gộp chúng để tạo ra ký hiệu mới `'ta'`. Trong lần lặp thứ hai, byte pair encoding tiếp tục gộp `'ta'` và `'l'` để tạo ra một ký hiệu mới khác `'tal'`.

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

Sau 10 lần lặp byte pair encoding, ta có thể thấy rằng danh sách `symbols` hiện chứa thêm 10 ký hiệu được gộp lần lượt từ các ký hiệu khác.

```{.python .input}
#@tab all
print(symbols)
```

Với cùng tập dữ liệu được chỉ định trong các khóa của từ điển `raw_token_freqs`,
mỗi từ trong tập dữ liệu giờ đây được phân đoạn thành các từ con "fast_", "fast", "er_", "tall_" và "tall"
do kết quả của thuật toán byte pair encoding.
Ví dụ, các từ "faster_" và "taller_" lần lượt được phân đoạn thành "fast er_" và "tall er_".

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

Lưu ý rằng kết quả của byte pair encoding phụ thuộc vào tập dữ liệu được sử dụng.
Ta cũng có thể sử dụng các từ con được học từ một tập dữ liệu
để phân đoạn các từ thuộc một tập dữ liệu khác.
Là một phương pháp greedy, hàm `segment_BPE` sau đây cố gắng tách các từ thành những từ con dài nhất có thể từ đối số đầu vào `symbols`.

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Segment token with the longest possible subwords from symbols
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```

Sau đây, ta sử dụng các từ con trong danh sách `symbols`, vốn được học từ tập dữ liệu nói trên,
để phân đoạn `tokens` đại diện cho một tập dữ liệu khác.

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## Tóm tắt

* Mô hình fastText đề xuất cách tiếp cận nhúng từ con. Dựa trên mô hình skip-gram trong word2vec, mô hình biểu diễn một từ trung tâm bằng tổng các vector từ con của từ đó.
* Byte pair encoding thực hiện phân tích thống kê trên tập dữ liệu huấn luyện để phát hiện các ký hiệu phổ biến trong một từ. Là một phương pháp greedy, byte pair encoding lặp đi lặp lại việc gộp cặp ký hiệu liên tiếp xuất hiện thường xuyên nhất.
* Nhúng từ con có thể cải thiện chất lượng biểu diễn của các từ hiếm và các từ ngoài từ điển.

## Bài tập

1. Ví dụ, trong tiếng Anh có khoảng $3\times 10^8$ $6$-gram có thể có. Vấn đề là gì khi có quá nhiều từ con? Làm thế nào để giải quyết vấn đề đó? Gợi ý: tham khảo phần cuối của Mục 3.2 trong bài báo fastText :cite:`Bojanowski.Grave.Joulin.ea.2017`.
1. Làm thế nào để thiết kế một mô hình nhúng từ con dựa trên mô hình continuous bag-of-words?
1. Để có một từ vựng kích thước $m$, cần bao nhiêu phép gộp khi kích thước từ vựng ký hiệu ban đầu là $n$?
1. Làm thế nào để mở rộng ý tưởng byte pair encoding để trích xuất các cụm từ?



:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/386)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/4587)
:end_tab: