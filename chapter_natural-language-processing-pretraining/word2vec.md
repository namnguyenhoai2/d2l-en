# Nhúng từ (word2vec)
:label:`sec_word2vec`


Ngôn ngữ tự nhiên là một hệ thống phức tạp được sử dụng để biểu đạt ý nghĩa.
Trong hệ thống này, từ là đơn vị cơ bản của ý nghĩa.
Đúng như tên gọi,
*vector từ* là các vector được sử dụng để biểu diễn từ,
và cũng có thể được xem là các vector đặc trưng hoặc các biểu diễn của từ.
Kỹ thuật ánh xạ từ vào các vector thực
được gọi là *nhúng từ*.
Trong những năm gần đây,
nhúng từ đã dần trở thành
kiến thức cơ bản của xử lý ngôn ngữ tự nhiên.


## Vector One-Hot Là Một Lựa Chọn Không Tốt

Chúng ta đã sử dụng vector one-hot để biểu diễn các từ (ký tự cũng được xem là từ) trong :numref:`sec_rnn-scratch`.
Giả sử số lượng từ khác nhau trong từ điển (kích thước từ điển) là $N$,
và mỗi từ tương ứng với
một số nguyên khác nhau (chỉ số) từ $0$ đến $N-1$.
Để thu được biểu diễn vector one-hot
cho bất kỳ từ nào có chỉ số $i$,
chúng ta tạo một vector có độ dài $N$ với tất cả các phần tử bằng 0
và đặt phần tử ở vị trí $i$ thành 1.
Theo cách này, mỗi từ được biểu diễn bởi một vector có độ dài $N$, và nó
có thể được sử dụng trực tiếp bởi các mạng neural.


Mặc dù vector từ one-hot dễ xây dựng,
chúng thường không phải là một lựa chọn tốt.
Một lý do chính là vector từ one-hot không thể biểu đạt chính xác mức độ tương đồng giữa các từ khác nhau, chẳng hạn như *độ tương đồng cosine* mà chúng ta thường sử dụng.
Với các vector $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$, độ tương đồng cosine của chúng là cosin của góc giữa hai vector:


$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$$


Vì độ tương đồng cosine giữa vector one-hot của bất kỳ hai từ khác nhau nào cũng bằng 0,
vector one-hot không thể mã hóa sự tương đồng giữa các từ.


## word2vec Tự Giám sát

Công cụ [word2vec](https://code.google.com/archive/p/word2vec/) được đề xuất để giải quyết vấn đề trên.
Công cụ này ánh xạ mỗi từ vào một vector có độ dài cố định, và các vector này có thể biểu đạt tốt hơn mối quan hệ tương đồng và tương tự giữa các từ khác nhau.
Công cụ word2vec bao gồm hai mô hình, đó là *skip-gram* :cite:`Mikolov.Sutskever.Chen.ea.2013` và *continuous bag of words* (CBOW) :cite:`Mikolov.Chen.Corrado.ea.2013`.
Để tạo ra các biểu diễn có ý nghĩa về mặt ngữ nghĩa,
việc huấn luyện các mô hình này dựa trên
các xác suất có điều kiện
có thể được xem là việc dự đoán
một số từ bằng cách sử dụng một số từ xung quanh chúng
trong các corpus.
Vì sự giám sát đến từ dữ liệu không có nhãn,
cả skip-gram và continuous bag of words
đều là các mô hình tự giám sát.

Sau đây, chúng ta sẽ giới thiệu hai mô hình này cùng các phương pháp huấn luyện chúng.


## Mô hình Skip-Gram
:label:`subsec_skip-gram`

Mô hình *skip-gram* giả định rằng một từ có thể được sử dụng để sinh ra các từ xung quanh nó trong một chuỗi văn bản.
Hãy lấy chuỗi văn bản "the", "man", "loves", "his", "son" làm ví dụ.
Chúng ta chọn "loves" làm *từ trung tâm* và đặt kích thước cửa sổ ngữ cảnh là 2.
Như được minh họa trong :numref:`fig_skip_gram`,
với từ trung tâm "loves",
mô hình skip-gram xét
xác suất có điều kiện để sinh ra các *từ ngữ cảnh*: "the", "man", "his" và "son",
là những từ cách từ trung tâm không quá 2 từ:

$$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$$

Giả sử rằng
các từ ngữ cảnh được sinh độc lập
khi biết từ trung tâm (tức là độc lập có điều kiện).
Trong trường hợp này, xác suất có điều kiện trên
có thể được viết lại thành

$$P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).$$

![Mô hình skip-gram xét xác suất có điều kiện để sinh ra các từ ngữ cảnh xung quanh khi biết một từ trung tâm.](../img/skip-gram.svg)
:label:`fig_skip_gram`

Trong mô hình skip-gram, mỗi từ
có hai biểu diễn dạng vector $d$ chiều
để tính các xác suất có điều kiện.
Cụ thể hơn,
với bất kỳ từ nào có chỉ số $i$ trong từ điển,
ký hiệu $\mathbf{v}_i\in\mathbb{R}^d$
và $\mathbf{u}_i\in\mathbb{R}^d$
là hai vector của từ đó
khi được sử dụng lần lượt như một từ *trung tâm* và một từ *ngữ cảnh*.
Xác suất có điều kiện để sinh ra bất kỳ
từ ngữ cảnh $w_o$ nào (có chỉ số $o$ trong từ điển) khi biết từ trung tâm $w_c$ (có chỉ số $c$ trong từ điển) có thể được mô hình hóa bằng
phép toán softmax trên các tích vô hướng của vector:


$$P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)},$$
:eqlabel:`eq_skip-gram-softmax`

trong đó tập chỉ số của từ vựng $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$.
Cho một chuỗi văn bản có độ dài $T$, trong đó từ ở bước thời gian $t$ được ký hiệu là $w^{(t)}$.
Giả sử rằng
các từ ngữ cảnh được sinh độc lập
khi biết bất kỳ từ trung tâm nào.
Với kích thước cửa sổ ngữ cảnh là $m$,
hàm hợp lý của mô hình skip-gram
là xác suất sinh ra tất cả các từ ngữ cảnh
khi biết bất kỳ từ trung tâm nào:


$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

trong đó có thể bỏ qua mọi bước thời gian nhỏ hơn $1$ hoặc lớn hơn $T$.

### Huấn luyện

Các tham số của mô hình skip-gram là vector từ trung tâm và vector từ ngữ cảnh cho mỗi từ trong từ vựng.
Trong quá trình huấn luyện, chúng ta học các tham số mô hình bằng cách cực đại hóa hàm hợp lý (tức là ước lượng hợp lý cực đại). Điều này tương đương với việc cực tiểu hóa hàm mất mát sau:

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \textrm{log}\, P(w^{(t+j)} \mid w^{(t)}).$$

Khi sử dụng gradient descent ngẫu nhiên để cực tiểu hóa hàm mất mát,
ở mỗi vòng lặp
chúng ta có thể
lấy mẫu ngẫu nhiên một chuỗi con ngắn hơn để tính gradient (ngẫu nhiên) cho chuỗi con này và cập nhật các tham số mô hình.
Để tính gradient (ngẫu nhiên) này,
chúng ta cần thu được
các gradient của
logarit của xác suất có điều kiện đối với vector từ trung tâm và vector từ ngữ cảnh.
Nói chung, theo :eqref:`eq_skip-gram-softmax`
logarit của xác suất có điều kiện
liên quan đến bất kỳ cặp từ trung tâm $w_c$ và
từ ngữ cảnh $w_o$ nào là


$$\log P(w_o \mid w_c) =\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$
:eqlabel:`eq_skip-gram-log`

Bằng phép lấy đạo hàm, chúng ta có thể thu được gradient của nó
theo vector từ trung tâm $\mathbf{v}_c$ như sau

$$\begin{aligned}\frac{\partial \textrm{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j.\end{aligned}$$
:eqlabel:`eq_skip-gram-grad`


Lưu ý rằng phép tính trong :eqref:`eq_skip-gram-grad` yêu cầu các xác suất có điều kiện của tất cả các từ trong từ điển khi $w_c$ là từ trung tâm.
Gradient của các vector từ còn lại có thể được thu được theo cách tương tự.


Sau khi huấn luyện, với bất kỳ từ nào có chỉ số $i$ trong từ điển, chúng ta thu được cả hai vector từ
$\mathbf{v}_i$ (với vai trò từ trung tâm) và $\mathbf{u}_i$ (với vai trò từ ngữ cảnh).
Trong các ứng dụng xử lý ngôn ngữ tự nhiên, các vector từ trung tâm của mô hình skip-gram thường được
sử dụng làm biểu diễn từ.


## Mô hình Continuous Bag of Words (CBOW)


Mô hình *continuous bag of words* (CBOW) tương tự mô hình skip-gram.
Điểm khác biệt chính
so với mô hình skip-gram là
mô hình continuous bag of words
giả định rằng một từ trung tâm được sinh ra
dựa trên các từ ngữ cảnh xung quanh nó trong chuỗi văn bản.
Ví dụ,
trong cùng chuỗi văn bản "the", "man", "loves", "his" và "son", với "loves" là từ trung tâm và kích thước cửa sổ ngữ cảnh bằng 2,
mô hình continuous bag of words
xét
xác suất có điều kiện để sinh ra từ trung tâm "loves" dựa trên các từ ngữ cảnh "the", "man", "his" và "son" (như được minh họa trong :numref:`fig_cbow`), đó là

$$P(\textrm{"loves"}\mid\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}).$$

![Mô hình continuous bag of words xét xác suất có điều kiện để sinh ra từ trung tâm khi biết các từ ngữ cảnh xung quanh nó.](../img/cbow.svg)
:label:`fig_cbow`


Vì có nhiều từ ngữ cảnh
trong mô hình continuous bag of words,
các vector của những từ ngữ cảnh này được lấy trung bình
trong phép tính xác suất có điều kiện.
Cụ thể,
với mọi từ có chỉ số $i$ trong từ điển,
ký hiệu $\mathbf{v}_i\in\mathbb{R}^d$
và $\mathbf{u}_i\in\mathbb{R}^d$
là hai vector của từ đó
khi được dùng làm từ *ngữ cảnh* và từ *trung tâm*
(ý nghĩa được hoán đổi trong mô hình skip-gram), tương ứng.
Xác suất có điều kiện để sinh ra bất kỳ
từ *trung tâm* $w_c$ nào (có chỉ số $c$ trong từ điển) khi biết các từ ngữ cảnh xung quanh $w_{o_1}, \ldots, w_{o_{2m}}$ (có các chỉ số $o_1, \ldots, o_{2m}$ trong từ điển) có thể được mô hình hóa như sau:



$$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\exp\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \exp\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots + \mathbf{v}_{o_{2m}}) \right)}.$$
:eqlabel:`fig_cbow-full`


Để viết ngắn gọn, đặt $\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$ và $\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots + \mathbf{v}_{o_{2m}} \right)/(2m)$. Khi đó, :eqref:`fig_cbow-full` có thể được đơn giản hóa thành

$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}.$$

Xét một chuỗi văn bản có độ dài $T$, trong đó từ tại bước thời gian $t$ được ký hiệu là $w^{(t)}$.
Với kích thước cửa sổ ngữ cảnh là $m$,
hàm likelihood của mô hình continuous bag of words
là xác suất sinh ra tất cả các từ trung tâm
khi biết các từ ngữ cảnh tương ứng:


$$ \prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

### Huấn luyện

Việc huấn luyện các mô hình continuous bag of words
gần như giống hệt
việc huấn luyện các mô hình skip-gram.
Ước lượng hợp lý cực đại của
mô hình continuous bag of words tương đương với việc tối thiểu hóa hàm mất mát sau:



$$  -\sum_{t=1}^T  \textrm{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

Lưu ý rằng

$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right).$$

Bằng phép vi phân, ta có thể thu được gradient
theo bất kỳ vector từ ngữ cảnh nào $\mathbf{v}_{o_i}$($i = 1, \ldots, 2m$)
như sau:


$$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).$$
:eqlabel:`eq_cbow-gradient`


Các gradient của những vector từ khác có thể được tính theo cách tương tự.
Khác với mô hình skip-gram,
mô hình continuous bag of words
thường
sử dụng các vector từ ngữ cảnh làm biểu diễn từ.




## Tóm tắt

* Vector từ là các vector được dùng để biểu diễn từ, đồng thời cũng có thể được xem là các vector đặc trưng hoặc các biểu diễn của từ. Kỹ thuật ánh xạ từ sang các vector thực được gọi là word embedding.
* Công cụ word2vec chứa cả mô hình skip-gram và continuous bag of words.
* Mô hình skip-gram giả định rằng một từ có thể được dùng để sinh ra các từ xung quanh nó trong một chuỗi văn bản; trong khi mô hình continuous bag of words giả định rằng một từ trung tâm được sinh ra dựa trên các từ ngữ cảnh xung quanh nó.



## Bài tập

1. Độ phức tạp tính toán để tính mỗi gradient là bao nhiêu? Vấn đề có thể xảy ra nếu kích thước từ điển rất lớn là gì?
1. Một số cụm từ cố định trong tiếng Anh gồm nhiều từ, chẳng hạn như "new york". Làm thế nào để huấn luyện các vector từ của chúng? Gợi ý: xem Mục 4 trong bài báo word2vec :cite:`Mikolov.Sutskever.Chen.ea.2013`.
1. Hãy cùng suy ngẫm về thiết kế của word2vec bằng cách lấy mô hình skip-gram làm ví dụ. Mối quan hệ giữa tích vô hướng của hai vector từ trong mô hình skip-gram và độ tương đồng cosine là gì? Với một cặp từ có ngữ nghĩa tương tự nhau, tại sao độ tương đồng cosine của các vector từ của chúng (được huấn luyện bằng mô hình skip-gram) có thể cao?

[Thảo luận](https://discuss.d2l.ai/t/381)