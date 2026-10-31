# Huấn luyện xấp xỉ
:label:`sec_approx_train`

Hãy nhớ lại các thảo luận của chúng ta trong :numref:`sec_word2vec`.
Ý tưởng chính của mô hình skip-gram là
sử dụng các phép toán softmax để tính
xác suất có điều kiện của việc
sinh ra một từ ngữ cảnh $w_o$
dựa trên từ trung tâm $w_c$ đã cho
trong :eqref:`eq_skip-gram-softmax`,
với hàm mất mát log tương ứng bằng
đối của :eqref:`eq_skip-gram-log`.



Do bản chất của phép toán softmax,
vì một từ ngữ cảnh có thể là bất kỳ từ nào trong
từ điển $\mathcal{V}$,
đối của :eqref:`eq_skip-gram-log`
chứa phép tổng
của số hạng nhiều bằng toàn bộ kích thước của từ vựng.
Do đó,
việc tính gradient
cho mô hình skip-gram
trong :eqref:`eq_skip-gram-grad`
và
cho mô hình continuous bag-of-words
trong :eqref:`eq_cbow-gradient`
đều chứa
phép tổng.
Đáng tiếc là,
chi phí tính toán
cho các gradient như vậy,
vốn tính tổng trên
một từ điển lớn
(thường có
hàng trăm nghìn hoặc hàng triệu từ),
là rất lớn!

Để giảm độ phức tạp tính toán nói trên, phần này sẽ giới thiệu hai phương pháp huấn luyện xấp xỉ:
*negative sampling* và *hierarchical softmax*.
Do sự tương đồng
giữa mô hình skip-gram và
mô hình continuous bag-of-words,
chúng ta sẽ chỉ lấy mô hình skip-gram làm ví dụ
để mô tả hai phương pháp huấn luyện xấp xỉ này.

## Negative Sampling
:label:`subsec_negative-sampling`


Negative sampling sửa đổi hàm mục tiêu ban đầu.
Với cửa sổ ngữ cảnh của một từ trung tâm $w_c$,
việc một từ (ngữ cảnh) bất kỳ $w_o$
xuất hiện trong cửa sổ ngữ cảnh này
được xem là một sự kiện có xác suất
được mô hình hóa bởi


$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$

trong đó $\sigma$ sử dụng định nghĩa của hàm kích hoạt sigmoid:

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$
:eqlabel:`eq_sigma-f`

Hãy bắt đầu bằng cách
tối đa hóa xác suất đồng thời của
tất cả các sự kiện như vậy trong
các chuỗi văn bản để huấn luyện word embedding.
Cụ thể,
với một chuỗi văn bản có độ dài $T$,
ký hiệu $w^{(t)}$ là từ tại bước thời gian $t$
và gọi kích thước cửa sổ ngữ cảnh là $m$,
ta xét việc tối đa hóa xác suất đồng thời


$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).$$
:eqlabel:`eq-negative-sample-pos`


Tuy nhiên,
:eqref:`eq-negative-sample-pos`
chỉ xét các sự kiện
liên quan đến các ví dụ dương.
Do đó,
xác suất đồng thời trong
:eqref:`eq-negative-sample-pos`
được tối đa hóa thành 1
chỉ khi tất cả các vector từ đều bằng vô cực.
Dĩ nhiên,
kết quả như vậy là vô nghĩa.
Để làm cho hàm mục tiêu
có ý nghĩa hơn,
*negative sampling*
bổ sung các ví dụ âm được lấy mẫu
từ một phân phối định trước.

Ký hiệu $S$
là sự kiện mà
một từ ngữ cảnh $w_o$ xuất hiện
trong cửa sổ ngữ cảnh của một từ trung tâm $w_c$.
Với sự kiện liên quan đến $w_o$ này,
từ một phân phối định trước $P(w)$,
lấy mẫu $K$ *noise word*
không xuất hiện trong cửa sổ ngữ cảnh này.
Ký hiệu $N_k$
là sự kiện mà
một noise word $w_k$ ($k=1, \ldots, K$)
không xuất hiện
trong cửa sổ ngữ cảnh của $w_c$.
Giả sử rằng
các sự kiện liên quan đến
cả ví dụ dương và các ví dụ âm
$S, N_1, \ldots, N_K$ là độc lập lẫn nhau.
Negative sampling
viết lại xác suất đồng thời (chỉ liên quan đến các ví dụ dương)
trong :eqref:`eq-negative-sample-pos`
thành

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

trong đó xác suất có điều kiện được xấp xỉ thông qua
các sự kiện $S, N_1, \ldots, N_K$:

$$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$
:eqlabel:`eq-negative-sample-conditional-prob`

Ký hiệu
$i_t$ và $h_k$
lần lượt là chỉ số của
một từ $w^{(t)}$ tại bước thời gian $t$
của một chuỗi văn bản
và một noise word $w_k$.
Hàm mất mát log đối với các xác suất có điều kiện trong :eqref:`eq-negative-sample-conditional-prob` là

$$
\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$


Ta có thể thấy rằng
giờ đây chi phí tính toán gradient
ở mỗi bước huấn luyện
không liên quan gì đến kích thước từ điển,
mà phụ thuộc tuyến tính vào $K$.
Khi đặt hyperparameter $K$
thành một giá trị nhỏ hơn,
chi phí tính toán gradient
ở mỗi bước huấn luyện với negative sampling
sẽ nhỏ hơn.




## Hierarchical Softmax

Là một phương pháp huấn luyện xấp xỉ khác,
*hierarchical softmax*
sử dụng cây nhị phân,
một cấu trúc dữ liệu
được minh họa trong :numref:`fig_hi_softmax`,
trong đó mỗi nút lá
của cây biểu diễn
một từ trong từ điển $\mathcal{V}$.

![Hierarchical softmax cho huấn luyện xấp xỉ, trong đó mỗi nút lá của cây biểu diễn một từ trong từ điển.](../img/hi-softmax.svg)
:label:`fig_hi_softmax`

Ký hiệu $L(w)$
là số lượng nút (bao gồm cả hai đầu mút)
trên đường đi
từ nút gốc đến nút lá biểu diễn từ $w$
trong cây nhị phân.
Gọi $n(w,j)$ là nút thứ $j^\textrm{th}$ trên đường đi này,
với vector từ ngữ cảnh của nó là
$\mathbf{u}_{n(w, j)}$.
Ví dụ,
$L(w_3) = 4$ trong :numref:`fig_hi_softmax`.
Hierarchical softmax xấp xỉ xác suất có điều kiện trong :eqref:`eq_skip-gram-softmax` như sau:


$$P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \textrm{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right),$$

trong đó hàm $\sigma$
được định nghĩa trong :eqref:`eq_sigma-f`,
và $\textrm{leftChild}(n)$ là nút con bên trái của nút $n$: nếu $x$ đúng thì $[\![x]\!] = 1$; ngược lại, $[\![x]\!] = -1$.

Để minh họa,
hãy tính
xác suất có điều kiện
của việc sinh ra từ $w_3$
với từ $w_c$ trong :numref:`fig_hi_softmax`.
Việc này yêu cầu các tích vô hướng
giữa vector từ
$\mathbf{v}_c$ của $w_c$
và
các vector của những nút không phải lá
trên đường đi (đường đi được in đậm trong :numref:`fig_hi_softmax`) từ gốc đến $w_3$,
được duyệt theo thứ tự trái, phải, rồi trái:


$$P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c).$$

Vì $\sigma(x)+\sigma(-x) = 1$,
nên
tổng các xác suất có điều kiện của
việc sinh ra tất cả các từ trong
từ điển $\mathcal{V}$
dựa trên bất kỳ từ $w_c$ nào
bằng một:

$$\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1.$$
:eqlabel:`eq_hi-softmax-sum-one`

May mắn là, vì $L(w_o)-1$ có bậc $\mathcal{O}(\textrm{log}_2|\mathcal{V}|)$ do cấu trúc cây nhị phân,
khi kích thước từ điển $\mathcal{V}$ rất lớn,
chi phí tính toán cho mỗi bước huấn luyện sử dụng hierarchical softmax
được giảm đáng kể so với
chi phí khi không sử dụng huấn luyện xấp xỉ.

## Tóm tắt

* Negative sampling xây dựng hàm mất mát bằng cách xét các sự kiện độc lập lẫn nhau có liên quan đến cả ví dụ dương và ví dụ âm. Chi phí tính toán cho việc huấn luyện phụ thuộc tuyến tính vào số lượng noise word ở mỗi bước.
* Hierarchical softmax xây dựng hàm mất mát bằng cách sử dụng đường đi từ nút gốc đến nút lá trong cây nhị phân. Chi phí tính toán cho việc huấn luyện phụ thuộc vào logarit của kích thước từ điển ở mỗi bước.

## Bài tập

1. Làm thế nào để lấy mẫu noise word trong negative sampling?
1. Hãy chứng minh :eqref:`eq_hi-softmax-sum-one` đúng.
1. Làm thế nào để huấn luyện mô hình continuous bag-of-words lần lượt bằng negative sampling và hierarchical softmax?

[Thảo luận](https://discuss.d2l.ai/t/382)