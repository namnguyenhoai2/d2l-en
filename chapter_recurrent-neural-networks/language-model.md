# Mô hình ngôn ngữ
:label:`sec_language-model`

Trong :numref:`sec_text-sequence`, chúng ta đã thấy cách ánh xạ các chuỗi văn bản thành token, trong đó các token này có thể được xem là một chuỗi các quan sát rời rạc như từ hoặc ký tự. Giả sử các token trong một chuỗi văn bản có độ dài $T$ lần lượt là $x_1, x_2, \ldots, x_T$.
Mục tiêu của *mô hình ngôn ngữ*
là ước lượng xác suất đồng thời của toàn bộ chuỗi:

$$P(x_1, x_2, \ldots, x_T),$$

trong đó các công cụ thống kê
trong :numref:`sec_sequence`
có thể được áp dụng.

Mô hình ngôn ngữ vô cùng hữu ích. Chẳng hạn, một mô hình ngôn ngữ lý tưởng có thể tự sinh văn bản tự nhiên, chỉ bằng cách lấy mẫu từng token một $x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$.
Không giống như một con khỉ dùng máy đánh chữ, toàn bộ văn bản được sinh ra từ mô hình như vậy sẽ trông như ngôn ngữ tự nhiên, ví dụ như văn bản tiếng Anh. Hơn nữa, nó đủ để tạo ra một cuộc đối thoại có ý nghĩa, chỉ bằng cách điều kiện hóa văn bản theo các đoạn đối thoại trước đó.
Rõ ràng, chúng ta vẫn còn rất xa mới có thể thiết kế một hệ thống như vậy, vì nó cần phải *hiểu* văn bản thay vì chỉ tạo ra nội dung hợp lý về mặt ngữ pháp.

Tuy vậy, ngay cả ở dạng hạn chế, mô hình ngôn ngữ vẫn mang lại giá trị lớn.
Ví dụ, các cụm từ "to recognize speech" và "to wreck a nice beach" nghe rất giống nhau.
Điều này có thể gây mơ hồ trong nhận dạng giọng nói,
và có thể dễ dàng được giải quyết bằng một mô hình ngôn ngữ loại bỏ cách dịch thứ hai vì quá vô lý.
Tương tự, trong một thuật toán tóm tắt tài liệu,
sẽ hữu ích nếu biết rằng "dog bites man" xuất hiện thường xuyên hơn nhiều so với "man bites dog", hoặc rằng "I want to eat grandma" là một phát biểu khá đáng sợ, trong khi "I want to eat, grandma" thì lành tính hơn nhiều.

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
```

## Học mô hình ngôn ngữ

Câu hỏi hiển nhiên là chúng ta nên mô hình hóa một tài liệu, hay thậm chí một chuỗi token, như thế nào.
Giả sử rằng chúng ta token hóa dữ liệu văn bản ở cấp độ từ.
Hãy bắt đầu bằng cách áp dụng các quy tắc xác suất cơ bản:

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).$$

Ví dụ,
xác suất của một chuỗi văn bản chứa bốn từ được cho bởi:

$$\begin{aligned}&P(\textrm{deep}, \textrm{learning}, \textrm{is}, \textrm{fun}) \\
=&P(\textrm{deep}) P(\textrm{learning}  \mid  \textrm{deep}) P(\textrm{is}  \mid  \textrm{deep}, \textrm{learning}) P(\textrm{fun}  \mid  \textrm{deep}, \textrm{learning}, \textrm{is}).\end{aligned}$$

### Mô hình Markov và $n$-gram
:label:`subsec_markov-models-and-n-grams`

Trong số các phân tích mô hình chuỗi ở :numref:`sec_sequence`,
hãy áp dụng các mô hình Markov cho việc mô hình hóa ngôn ngữ.
Một phân phối trên các chuỗi thỏa mãn tính chất Markov bậc nhất nếu $P(x_{t+1} \mid x_t, \ldots, x_1) = P(x_{t+1} \mid x_t)$. Các bậc cao hơn tương ứng với các phụ thuộc dài hơn. Điều này dẫn đến một số phép xấp xỉ mà chúng ta có thể áp dụng để mô hình hóa một chuỗi:

$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}
$$

Các công thức xác suất liên quan đến một, hai và ba biến thường lần lượt được gọi là các mô hình *unigram*, *bigram* và *trigram*.
Để tính mô hình ngôn ngữ, chúng ta cần tính
xác suất của các từ và xác suất có điều kiện của một từ khi biết
một vài từ đứng trước.
Lưu ý rằng
các xác suất như vậy là
các tham số của mô hình ngôn ngữ.



### Tần suất từ

Ở đây, chúng ta
giả sử rằng tập dữ liệu huấn luyện là một kho ngữ liệu văn bản lớn, chẳng hạn như toàn bộ
các mục Wikipedia, [Project Gutenberg](https://en.wikipedia.org/wiki/Project_Gutenberg),
và toàn bộ văn bản được đăng trên
web.
Xác suất của các từ có thể được tính từ tần suất tương đối của một từ
nhất định trong tập dữ liệu huấn luyện.
Ví dụ, ước lượng $\hat{P}(\textrm{deep})$ có thể được tính là
xác suất bất kỳ câu nào bắt đầu bằng từ "deep". Một
cách tiếp cận kém chính xác hơn một chút là đếm tất cả các lần xuất hiện của
từ "deep" và chia cho tổng số từ trong
kho ngữ liệu.
Cách này hoạt động khá tốt, đặc biệt đối với các
từ thường gặp. Tiếp theo, chúng ta có thể thử ước lượng

$$\hat{P}(\textrm{learning} \mid \textrm{deep}) = \frac{n(\textrm{deep, learning})}{n(\textrm{deep})},$$

trong đó $n(x)$ và $n(x, x')$ lần lượt là số lần xuất hiện của các từ đơn
và các cặp từ liên tiếp.
Đáng tiếc là,
việc ước lượng
xác suất của một cặp từ khó hơn phần nào, vì
cụm "deep learning" xuất hiện ít thường xuyên hơn rất nhiều.
Đặc biệt, với một số tổ hợp từ hiếm gặp, có thể khó
tìm đủ số lần xuất hiện để có được các ước lượng chính xác.
Như các kết quả thực nghiệm trong :numref:`subsec_natural-lang-stat` cho thấy,
mọi việc trở nên tệ hơn với các tổ hợp ba từ trở lên.
Sẽ có nhiều tổ hợp ba từ hợp lý mà có lẽ chúng ta không thấy trong tập dữ liệu.
Nếu không đưa ra giải pháp gán cho các tổ hợp từ như vậy một số đếm khác không, chúng ta sẽ không thể sử dụng chúng trong mô hình ngôn ngữ. Nếu tập dữ liệu nhỏ hoặc các từ rất hiếm, có thể chúng ta sẽ không tìm thấy dù chỉ một lần xuất hiện.

### Làm trơn Laplace

Một chiến lược phổ biến là thực hiện một dạng *làm trơn Laplace*.
Giải pháp là
cộng một hằng số nhỏ vào mọi số đếm.
Ký hiệu $n$ là tổng số từ trong
tập huấn luyện
và $m$ là số từ duy nhất.
Giải pháp này hữu ích với các từ đơn, ví dụ thông qua

$$\begin{aligned}
	\hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}, \\
	\hat{P}(x' \mid x) & = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}, \\
	\hat{P}(x'' \mid x,x') & = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}.
\end{aligned}$$

Ở đây $\epsilon_1,\epsilon_2$ và $\epsilon_3$ là các siêu tham số.
Lấy $\epsilon_1$ làm ví dụ:
khi $\epsilon_1 = 0$, không áp dụng làm trơn;
khi $\epsilon_1$ tiến tới vô cực dương,
$\hat{P}(x)$ tiến tới xác suất đồng đều $1/m$.
Phương pháp trên là một biến thể khá sơ khai so với những gì
các kỹ thuật khác có thể thực hiện :cite:`Wood.Gasthaus.Archambeau.ea.2011`.


Đáng tiếc là các mô hình như thế này nhanh chóng trở nên cồng kềnh
vì các lý do sau.
Thứ nhất,
như đã thảo luận trong :numref:`subsec_natural-lang-stat`,
nhiều $n$-gram xuất hiện rất hiếm,
khiến làm trơn Laplace khá không phù hợp cho mô hình hóa ngôn ngữ.
Thứ hai, chúng ta cần lưu trữ mọi số đếm.
Thứ ba, cách này hoàn toàn bỏ qua ý nghĩa của các từ. Ví dụ,
"cat" và "feline" nên xuất hiện trong những ngữ cảnh có liên quan.
Việc điều chỉnh các mô hình như vậy cho các ngữ cảnh bổ sung là khá khó,
trong khi các mô hình ngôn ngữ dựa trên deep learning rất phù hợp để
xử lý điều này.
Cuối cùng, các chuỗi từ dài
gần như chắc chắn là mới, do đó một mô hình chỉ đơn giản
đếm tần suất các chuỗi từ đã thấy trước đó chắc chắn sẽ hoạt động kém trong trường hợp này.
Vì vậy, trong phần còn lại của chương, chúng ta tập trung sử dụng mạng nơ-ron để mô hình hóa ngôn ngữ.


## Độ hỗn loạn
:label:`subsec_perplexity`

Tiếp theo, hãy thảo luận cách đo lường chất lượng của mô hình ngôn ngữ, sau đó chúng ta sẽ dùng để đánh giá các mô hình ở những phần tiếp theo.
Một cách là kiểm tra mức độ bất ngờ của văn bản.
Một mô hình ngôn ngữ tốt có thể dự đoán chính xác cao các token tiếp theo.
Hãy xem xét các phần tiếp nối sau của cụm từ "It is raining", được đề xuất bởi các mô hình ngôn ngữ khác nhau:

1. "It is raining outside"
1. "It is raining banana tree"
1. "It is raining piouw;kcj pwepoiut"

Xét về chất lượng, Ví dụ 1 rõ ràng là tốt nhất. Các từ hợp lý và mạch lạc về logic.
Dù có thể không phản ánh thật chính xác từ nào sẽ theo sau về mặt ngữ nghĩa ("in San Francisco" và "in winter" đều có thể là những phần mở rộng hoàn toàn hợp lý), mô hình vẫn có thể nắm bắt loại từ nào sẽ theo sau.
Ví dụ 2 kém hơn đáng kể vì tạo ra một phần mở rộng vô nghĩa. Tuy nhiên, ít nhất mô hình đã học được cách đánh vần từ và một mức độ tương quan nào đó giữa các từ. Cuối cùng, Ví dụ 3 cho thấy một mô hình được huấn luyện kém, không khớp dữ liệu đúng cách.

Chúng ta có thể đo chất lượng của mô hình bằng cách tính likelihood của chuỗi.
Đáng tiếc, đây là một con số khó hiểu và khó so sánh.
Xét cho cùng, các chuỗi ngắn có khả năng xuất hiện cao hơn nhiều so với các chuỗi dài,
vì vậy việc đánh giá mô hình trên kiệt tác
*War and Peace* của Tolstoy chắc chắn sẽ tạo ra likelihood nhỏ hơn nhiều so với, chẳng hạn, trên tiểu thuyết vừa *The Little Prince* của Saint-Exupery. Điều còn thiếu là một đại lượng tương đương với giá trị trung bình.

Lý thuyết thông tin hữu ích trong trường hợp này.
Chúng ta đã định nghĩa entropy, độ bất ngờ và cross-entropy
khi giới thiệu softmax regression
(:numref:`subsec_info_theory_basics`).
Nếu muốn nén văn bản, chúng ta có thể xem xét
việc dự đoán token tiếp theo khi biết tập token hiện tại.
Một mô hình ngôn ngữ tốt hơn sẽ cho phép chúng ta dự đoán token tiếp theo chính xác hơn.
Do đó, nó sẽ cho phép chúng ta dùng ít bit hơn để nén chuỗi.
Vì vậy, chúng ta có thể đo nó bằng cross-entropy loss lấy trung bình
trên tất cả $n$ token của một chuỗi:

$$\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1),$$
:eqlabel:`eq_avg_ce_for_lm`

trong đó $P$ được cho bởi một mô hình ngôn ngữ và $x_t$ là token thực tế được quan sát tại bước thời gian $t$ từ chuỗi.
Điều này giúp hiệu năng trên các tài liệu có độ dài khác nhau có thể so sánh được. Vì các lý do lịch sử, các nhà khoa học trong xử lý ngôn ngữ tự nhiên thường sử dụng một đại lượng gọi là *perplexity*. Tóm lại, nó là hàm mũ của :eqref:`eq_avg_ce_for_lm`:

$$\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right).$$

Perplexity có thể được hiểu rõ nhất là nghịch đảo của trung bình hình học của số lượng lựa chọn thực tế mà chúng ta có khi quyết định token nào sẽ được chọn tiếp theo. Hãy xem xét một số trường hợp:

* Trong kịch bản tốt nhất, mô hình luôn ước lượng hoàn hảo xác suất của token đích là 1. Trong trường hợp này, perplexity của mô hình là 1.
* Trong kịch bản xấu nhất, mô hình luôn dự đoán xác suất của token đích là 0. Trong tình huống này, perplexity là dương vô cực.
* Ở mức cơ sở, mô hình dự đoán một phân phối đều trên tất cả các token có sẵn trong từ vựng. Trong trường hợp này, perplexity bằng số lượng token duy nhất trong từ vựng. Thực tế, nếu chúng ta lưu trữ chuỗi mà không có bất kỳ nén nào, đây sẽ là điều tốt nhất có thể làm để mã hóa nó. Do đó, điều này cung cấp một cận trên không tầm thường mà mọi mô hình hữu ích đều phải vượt qua.

## Phân vùng chuỗi
:label:`subsec_partitioning-seqs`

Chúng ta sẽ thiết kế các mô hình ngôn ngữ bằng mạng nơ-ron
và sử dụng perplexity để đánh giá
mô hình dự đoán token tiếp theo tốt đến mức nào
với tập token hiện có
trong các chuỗi văn bản.
Trước khi giới thiệu mô hình,
hãy giả sử rằng mô hình
xử lý một minibatch gồm các chuỗi có độ dài xác định trước
mỗi lần.
Bây giờ câu hỏi là làm thế nào để [**đọc ngẫu nhiên các minibatch gồm chuỗi đầu vào và chuỗi đích**].


Giả sử tập dữ liệu có dạng một chuỗi gồm $T$ chỉ số token trong `corpus`.
Chúng ta sẽ
phân vùng nó
thành các chuỗi con, trong đó mỗi chuỗi con có $n$ token (bước thời gian).
Để duyệt qua
(gần như) tất cả token của toàn bộ tập dữ liệu
trong mỗi epoch
và thu được tất cả các chuỗi con có độ dài $n$ khả dĩ,
chúng ta có thể đưa vào tính ngẫu nhiên.
Cụ thể hơn,
ở đầu mỗi epoch,
loại bỏ $d$ token đầu tiên,
trong đó $d\in [0,n)$ được lấy mẫu ngẫu nhiên theo phân phối đều.
Phần còn lại của chuỗi
sau đó được phân vùng
thành $m=\lfloor (T-d)/n \rfloor$ chuỗi con.
Ký hiệu $\mathbf x_t = [x_t, \ldots, x_{t+n-1}]$ là chuỗi con độ dài $n$ bắt đầu từ token $x_t$ tại bước thời gian $t$. 
$m$ chuỗi con đã phân vùng thu được
là 
$\mathbf x_d, \mathbf x_{d+n}, \ldots, \mathbf x_{d+n(m-1)}.$
Mỗi chuỗi con sẽ được sử dụng làm chuỗi đầu vào cho mô hình ngôn ngữ.


Đối với mô hình hóa ngôn ngữ,
mục tiêu là dự đoán token tiếp theo dựa trên các token đã thấy cho đến thời điểm hiện tại; do đó, các đích (nhãn) là chuỗi gốc được dịch đi một token.
Chuỗi đích cho bất kỳ chuỗi đầu vào nào $\mathbf x_t$
là $\mathbf x_{t+1}$ có độ dài $n$.

![Thu được năm cặp chuỗi đầu vào và chuỗi đích từ các chuỗi con đã phân vùng có độ dài 5.](../img/lang-model-data.svg) 
:label:`fig_lang_model_data`

:numref:`fig_lang_model_data` minh họa một ví dụ về việc thu được năm cặp chuỗi đầu vào và chuỗi đích với $n=5$ và $d=2$.

```{.python .input  n=5}
%%tab all
@d2l.add_to_class(d2l.TimeMachine)  #@save
def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
    super(d2l.TimeMachine, self).__init__()
    self.save_hyperparameters()
    corpus, self.vocab = self.build(self._download())
    array = d2l.tensor([corpus[i:i+num_steps+1] 
                        for i in range(len(corpus)-num_steps)])
    self.X, self.Y = array[:,:-1], array[:,1:]
```

Để huấn luyện các mô hình ngôn ngữ,
chúng ta sẽ lấy mẫu ngẫu nhiên
các cặp chuỗi đầu vào và chuỗi đích
theo minibatch.
Data loader sau đây tạo ngẫu nhiên một minibatch từ tập dữ liệu mỗi lần.
Đối số `batch_size` chỉ định số lượng ví dụ chuỗi con trong mỗi minibatch
và `num_steps` là độ dài chuỗi con tính theo token.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(d2l.TimeMachine)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(
        self.num_train, self.num_train + self.num_val)
    return self.get_tensorloader([self.X, self.Y], train, idx)
```

Như có thể thấy dưới đây, 
một minibatch gồm các chuỗi đích
có thể được thu được 
bằng cách dịch các chuỗi đầu vào
đi một token.

```{.python .input  n=7}
%%tab all
data = d2l.TimeMachine(batch_size=2, num_steps=10)
for X, Y in data.train_dataloader():
    print('X:', X, '\nY:', Y)
    break
```

## Tóm tắt và Thảo luận

Các mô hình ngôn ngữ ước lượng xác suất đồng thời của một chuỗi văn bản. Đối với các chuỗi dài, $n$-gram cung cấp một mô hình thuận tiện bằng cách cắt ngắn sự phụ thuộc. Tuy nhiên, có rất nhiều cấu trúc nhưng không đủ tần suất để xử lý hiệu quả các tổ hợp từ không thường xuyên thông qua làm trơn Laplace. Vì vậy, chúng ta sẽ tập trung vào mô hình hóa ngôn ngữ bằng mạng nơ-ron trong các phần tiếp theo.
Để huấn luyện các mô hình ngôn ngữ, chúng ta có thể lấy mẫu ngẫu nhiên các cặp chuỗi đầu vào và chuỗi đích theo minibatch. Sau khi huấn luyện, chúng ta sẽ sử dụng perplexity để đo chất lượng của mô hình ngôn ngữ.

Các mô hình ngôn ngữ có thể được mở rộng bằng cách tăng kích thước dữ liệu, kích thước mô hình và lượng tính toán huấn luyện. Các mô hình ngôn ngữ lớn có thể thực hiện các tác vụ mong muốn bằng cách dự đoán văn bản đầu ra dựa trên chỉ dẫn bằng văn bản đầu vào. Như chúng ta sẽ thảo luận sau (ví dụ: :numref:`sec_large-pretraining-transformers`),
ở thời điểm hiện tại
các mô hình ngôn ngữ lớn là nền tảng của các hệ thống hiện đại nhất trên nhiều tác vụ đa dạng.


## Bài tập

1. Giả sử có 100.000 từ trong tập dữ liệu huấn luyện. Một four-gram cần lưu trữ bao nhiêu tần suất từ và tần suất liền kề đa từ?
1. Bạn sẽ mô hình hóa một đoạn hội thoại như thế nào?
1. Bạn có thể nghĩ ra những phương pháp nào khác để đọc dữ liệu chuỗi dài?
1. Xét phương pháp của chúng ta về việc loại bỏ một số lượng ngẫu nhiên đồng đều các token đầu tiên ở đầu mỗi epoch.
    1. Liệu nó có thực sự dẫn đến một phân phối hoàn toàn đồng đều trên các chuỗi trong tài liệu không?
    1. Bạn cần làm gì để mọi thứ trở nên đồng đều hơn? 
1. Nếu chúng ta muốn một ví dụ chuỗi là một câu hoàn chỉnh, điều này gây ra loại vấn đề nào trong việc lấy mẫu minibatch? Chúng ta có thể khắc phục nó như thế nào?

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
[Thảo luận](https://discuss.d2l.ai/t/18012)
:end_tab: