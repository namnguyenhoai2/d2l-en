# Mạng Nơ-ron Hồi quy
:label:`sec_rnn`


Trong :numref:`sec_language-model`, chúng ta đã mô tả các mô hình Markov và $n$-gram cho mô hình hóa ngôn ngữ, trong đó xác suất có điều kiện của token $x_t$ tại bước thời gian $t$ chỉ phụ thuộc vào $n-1$ token trước đó.
Nếu muốn kết hợp ảnh hưởng có thể có của các token xuất hiện trước bước thời gian $t-(n-1)$ lên $x_t$,
chúng ta cần tăng $n$.
Tuy nhiên, số lượng tham số mô hình cũng sẽ tăng theo cấp số nhân, vì chúng ta cần lưu trữ $|\mathcal{V}|^n$ số cho tập từ vựng $\mathcal{V}$.
Do đó, thay vì mô hình hóa $P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$, tốt hơn là sử dụng mô hình biến tiềm ẩn,

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$

trong đó $h_{t-1}$ là một *trạng thái ẩn* lưu trữ thông tin chuỗi cho đến bước thời gian $t-1$.
Nói chung,
trạng thái ẩn tại bất kỳ bước thời gian $t$ nào có thể được tính dựa trên cả đầu vào hiện tại $x_{t}$ và trạng thái ẩn trước đó $h_{t-1}$:

$$h_t = f(x_{t}, h_{t-1}).$$
:eqlabel:`eq_ht_xt`

Với một hàm $f$ đủ mạnh trong :eqref:`eq_ht_xt`, mô hình biến tiềm ẩn không phải là một phép xấp xỉ. Xét cho cùng, $h_t$ có thể đơn giản lưu trữ toàn bộ dữ liệu mà nó đã quan sát cho đến thời điểm đó.
Tuy nhiên, điều này có thể khiến cả việc tính toán lẫn lưu trữ trở nên tốn kém.

Hãy nhớ rằng chúng ta đã thảo luận về các tầng ẩn với các đơn vị ẩn trong :numref:`chap_perceptrons`.
Cần lưu ý rằng
tầng ẩn và trạng thái ẩn đề cập đến hai khái niệm hoàn toàn khác nhau.
Như đã giải thích, tầng ẩn là các tầng bị che khuất khỏi tầm nhìn trên đường đi từ đầu vào đến đầu ra.
Về mặt kỹ thuật, trạng thái ẩn là *đầu vào* cho bất kỳ thao tác nào chúng ta thực hiện tại một bước nhất định,
và chúng chỉ có thể được tính bằng cách xem dữ liệu ở các bước thời gian trước đó.

*Mạng nơ-ron hồi quy* (RNN) là các mạng nơ-ron có trạng thái ẩn. Trước khi giới thiệu mô hình RNN, trước hết chúng ta sẽ xem lại mô hình MLP được giới thiệu trong :numref:`sec_mlp`.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
```

## Mạng Nơ-ron không có Trạng thái ẩn

Hãy xem xét một MLP có một tầng ẩn duy nhất.
Gọi hàm kích hoạt của tầng ẩn là $\phi$.
Với một minibatch gồm các mẫu $\mathbf{X} \in \mathbb{R}^{n \times d}$ có batch size là $n$ và $d$ đầu vào, đầu ra tầng ẩn $\mathbf{H} \in \mathbb{R}^{n \times h}$ được tính như sau

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{\textrm{xh}} + \mathbf{b}_\textrm{h}).$$
:eqlabel:`rnn_h_without_state`

Trong :eqref:`rnn_h_without_state`, chúng ta có tham số trọng số $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$, tham số bias $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$, và số lượng đơn vị ẩn $h$ của tầng ẩn.
Với các thành phần này, chúng ta áp dụng broadcasting (xem :numref:`subsec_broadcasting`) trong phép cộng.
Tiếp theo, đầu ra tầng ẩn $\mathbf{H}$ được dùng làm đầu vào của tầng đầu ra, được cho bởi

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q},$$

trong đó $\mathbf{O} \in \mathbb{R}^{n \times q}$ là biến đầu ra, $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$ là tham số trọng số, và $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$ là tham số bias của tầng đầu ra. Nếu đây là một bài toán phân loại, chúng ta có thể sử dụng $\mathrm{softmax}(\mathbf{O})$ để tính phân phối xác suất của các danh mục đầu ra.

Điều này hoàn toàn tương tự với bài toán hồi quy mà chúng ta đã giải trước đó trong :numref:`sec_sequence`, nên chúng ta lược bỏ chi tiết.
Chỉ cần nói rằng chúng ta có thể chọn ngẫu nhiên các cặp đặc trưng-nhãn và học các tham số của mạng thông qua vi phân tự động và stochastic gradient descent.

## Mạng Nơ-ron Hồi quy với Trạng thái ẩn
:label:`subsec_rnn_w_hidden_states`

Mọi thứ hoàn toàn khác khi chúng ta có trạng thái ẩn. Hãy xem xét cấu trúc này chi tiết hơn.

Giả sử chúng ta có
một minibatch đầu vào
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$
tại bước thời gian $t$.
Nói cách khác,
với một minibatch gồm $n$ mẫu chuỗi,
mỗi hàng của $\mathbf{X}_t$ tương ứng với một mẫu tại bước thời gian $t$ trong chuỗi.
Tiếp theo,
ký hiệu $\mathbf{H}_t  \in \mathbb{R}^{n \times h}$ là đầu ra tầng ẩn tại bước thời gian $t$.
Không giống MLP, ở đây chúng ta lưu đầu ra tầng ẩn $\mathbf{H}_{t-1}$ từ bước thời gian trước và đưa vào một tham số trọng số mới $\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$ để mô tả cách sử dụng đầu ra tầng ẩn của bước thời gian trước trong bước thời gian hiện tại. Cụ thể, việc tính đầu ra tầng ẩn của bước thời gian hiện tại được xác định bởi đầu vào của bước thời gian hiện tại cùng với đầu ra tầng ẩn của bước thời gian trước:

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}  + \mathbf{b}_\textrm{h}).$$
:eqlabel:`rnn_h_with_state`

So với :eqref:`rnn_h_without_state`, :eqref:`rnn_h_with_state` bổ sung thêm một hạng tử $\mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$ và do đó
hiện thực hóa :eqref:`eq_ht_xt`.
Từ mối quan hệ giữa các đầu ra tầng ẩn $\mathbf{H}_t$ và $\mathbf{H}_{t-1}$ tại các bước thời gian kề nhau,
chúng ta biết rằng các biến này đã nắm bắt và lưu giữ thông tin lịch sử của chuỗi cho đến bước thời gian hiện tại của chúng, giống như trạng thái hoặc bộ nhớ tại bước thời gian hiện tại của mạng nơ-ron. Vì vậy, đầu ra tầng ẩn như vậy được gọi là một *trạng thái ẩn*.
Vì trạng thái ẩn sử dụng cùng định nghĩa của bước thời gian trước trong bước thời gian hiện tại, phép tính của :eqref:`rnn_h_with_state` là *hồi quy*. Do đó, như đã nói, các mạng nơ-ron có trạng thái ẩn
dựa trên tính toán hồi quy được gọi là
*mạng nơ-ron hồi quy*.
Các tầng thực hiện
phép tính trong :eqref:`rnn_h_with_state`
trong RNN
được gọi là *tầng hồi quy*.


Có nhiều cách khác nhau để xây dựng RNN.
Các RNN có trạng thái ẩn được định nghĩa bởi :eqref:`rnn_h_with_state` là rất phổ biến.
Tại bước thời gian $t$,
đầu ra của tầng đầu ra tương tự như phép tính trong MLP:

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}.$$

Các tham số của RNN
bao gồm các trọng số $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}, \mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$,
và bias $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$
của tầng ẩn,
cùng với các trọng số $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$
và bias $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$
của tầng đầu ra.
Cần lưu ý rằng
ngay cả tại các bước thời gian khác nhau,
RNN luôn sử dụng các tham số mô hình này.
Do đó, chi phí tham số hóa của một RNN
không tăng khi số lượng bước thời gian tăng lên.

:numref:`fig_rnn` minh họa logic tính toán của một RNN tại ba bước thời gian kề nhau.
Tại bất kỳ bước thời gian $t$ nào,
việc tính trạng thái ẩn có thể được xem là:
(i) ghép đầu vào $\mathbf{X}_t$ tại bước thời gian hiện tại $t$ với trạng thái ẩn $\mathbf{H}_{t-1}$ tại bước thời gian trước $t-1$;
(ii) đưa kết quả ghép vào một tầng fully connected có hàm kích hoạt $\phi$.
Đầu ra của tầng fully connected đó là trạng thái ẩn $\mathbf{H}_t$ của bước thời gian hiện tại $t$.
Trong trường hợp này,
các tham số mô hình là phép ghép của $\mathbf{W}_{\textrm{xh}}$ và $\mathbf{W}_{\textrm{hh}}$, cùng một bias $\mathbf{b}_\textrm{h}$, tất cả đều từ :eqref:`rnn_h_with_state`.
Trạng thái ẩn $\mathbf{H}_t$ của bước thời gian hiện tại $t$ sẽ tham gia vào việc tính trạng thái ẩn $\mathbf{H}_{t+1}$ của bước thời gian tiếp theo $t+1$.
Hơn nữa, $\mathbf{H}_t$ cũng sẽ được
đưa vào tầng đầu ra fully connected
để tính đầu ra
$\mathbf{O}_t$ của bước thời gian hiện tại $t$.

![Một RNN với trạng thái ẩn.](../img/rnn.svg)
:label:`fig_rnn`

Chúng ta vừa đề cập rằng phép tính $\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$ cho trạng thái ẩn tương đương với
phép nhân ma trận của
phép nối $\mathbf{X}_t$ và $\mathbf{H}_{t-1}$
và
phép nối $\mathbf{W}_{\textrm{xh}}$ và $\mathbf{W}_{\textrm{hh}}$.
Mặc dù điều này có thể được chứng minh về mặt toán học,
dưới đây chúng ta chỉ sử dụng một đoạn mã đơn giản để minh họa.
Trước hết,
chúng ta định nghĩa các ma trận `X`, `W_xh`, `H` và `W_hh`, có các hình dạng lần lượt là (3, 1), (1, 4), (3, 4) và (4, 4).
Nhân `X` với `W_xh`, và `H` với `W_hh`, rồi cộng hai tích này lại,
ta thu được một ma trận có hình dạng (3, 4).

```{.python .input}
%%tab mxnet, pytorch
X, W_xh = d2l.randn(3, 1), d2l.randn(1, 4)
H, W_hh = d2l.randn(3, 4), d2l.randn(4, 4)
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
%%tab tensorflow
X, W_xh = d2l.normal((3, 1)), d2l.normal((1, 4))
H, W_hh = d2l.normal((3, 4)), d2l.normal((4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
%%tab jax
X, W_xh = jax.random.normal(d2l.get_key(), (3, 1)), jax.random.normal(
                                                        d2l.get_key(), (1, 4))
H, W_hh = jax.random.normal(d2l.get_key(), (3, 4)), jax.random.normal(
                                                        d2l.get_key(), (4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

Bây giờ chúng ta nối các ma trận `X` và `H`
theo các cột (trục 1),
và các ma trận
`W_xh` và `W_hh` theo các hàng (trục 0).
Hai phép nối này
lần lượt tạo ra
các ma trận có hình dạng (3, 5)
và (5, 4).
Nhân hai ma trận đã nối này,
ta thu được cùng ma trận đầu ra có hình dạng (3, 4)
như ở trên.

```{.python .input}
%%tab all
d2l.matmul(d2l.concat((X, H), 1), d2l.concat((W_xh, W_hh), 0))
```

## Mô hình ngôn ngữ cấp ký tự dựa trên RNN

Nhắc lại rằng đối với mô hình hóa ngôn ngữ trong :numref:`sec_language-model`,
chúng ta hướng đến việc dự đoán token tiếp theo dựa trên
token hiện tại và các token trước đó;
do đó, chúng ta dịch chuỗi gốc đi một token
để làm mục tiêu (nhãn).
:citet:`Bengio.Ducharme.Vincent.ea.2003` là những người đầu tiên đề xuất
sử dụng mạng nơ-ron cho mô hình hóa ngôn ngữ.
Dưới đây, chúng ta minh họa cách RNN có thể được dùng để xây dựng một mô hình ngôn ngữ.
Giả sử kích thước minibatch là một và chuỗi văn bản là "machine".
Để đơn giản hóa việc huấn luyện trong các phần tiếp theo,
chúng ta token hóa văn bản thành các ký tự thay vì từ
và xét một *mô hình ngôn ngữ cấp ký tự*.
:numref:`fig_rnn_train` minh họa cách dự đoán ký tự tiếp theo dựa trên các ký tự hiện tại và trước đó thông qua một RNN cho mô hình hóa ngôn ngữ cấp ký tự.

![Một mô hình ngôn ngữ cấp ký tự dựa trên RNN. Chuỗi đầu vào và chuỗi mục tiêu lần lượt là "machin" và "achine".](../img/rnn-train.svg)
:label:`fig_rnn_train`

Trong quá trình huấn luyện,
chúng ta chạy phép toán softmax trên đầu ra từ lớp đầu ra ở mỗi bước thời gian, rồi sử dụng hàm mất mát cross-entropy để tính sai số giữa đầu ra của mô hình và mục tiêu.
Do phép tính lặp của trạng thái ẩn trong lớp ẩn, đầu ra $\mathbf{O}_3$ ở bước thời gian 3 trong :numref:`fig_rnn_train` được xác định bởi chuỗi văn bản "m", "a" và "c". Vì ký tự tiếp theo của chuỗi trong dữ liệu huấn luyện là "h", mất mát ở bước thời gian 3 sẽ phụ thuộc vào phân phối xác suất của ký tự tiếp theo được tạo ra dựa trên chuỗi đặc trưng "m", "a", "c" và mục tiêu "h" của bước thời gian này.

Trong thực tế, mỗi token được biểu diễn bằng một vector $d$ chiều, và chúng ta sử dụng kích thước batch $n>1$. Do đó, đầu vào $\mathbf X_t$ tại bước thời gian $t$ sẽ là một ma trận $n\times d$, giống hệt những gì chúng ta đã thảo luận trong :numref:`subsec_rnn_w_hidden_states`.

Trong các phần tiếp theo, chúng ta sẽ triển khai RNN
cho các mô hình ngôn ngữ cấp ký tự.


## Tóm tắt

Một mạng nơ-ron sử dụng phép tính lặp cho các trạng thái ẩn được gọi là mạng nơ-ron hồi tiếp (RNN).
Trạng thái ẩn của một RNN có thể nắm bắt thông tin lịch sử của chuỗi cho đến bước thời gian hiện tại. Với phép tính lặp, số lượng tham số của mô hình RNN không tăng khi số bước thời gian tăng lên. Về ứng dụng, RNN có thể được dùng để tạo các mô hình ngôn ngữ cấp ký tự.


## Bài tập

1. Nếu chúng ta sử dụng một RNN để dự đoán ký tự tiếp theo trong một chuỗi văn bản, mỗi đầu ra cần có số chiều là bao nhiêu?
1. Vì sao RNN có thể biểu diễn xác suất có điều kiện của một token tại một bước thời gian nào đó dựa trên tất cả các token trước đó trong chuỗi văn bản?
1. Điều gì xảy ra với gradient nếu bạn backpropagate qua một chuỗi dài?
1. Một số vấn đề liên quan đến mô hình ngôn ngữ được mô tả trong phần này là gì?


:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/337)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1050)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1051)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/180013)
:end_tab: