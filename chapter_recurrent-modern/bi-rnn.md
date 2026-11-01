# Mạng nơ-ron hồi quy hai chiều
:label:`sec_bi_rnn`

Cho đến nay, ví dụ đang được sử dụng của chúng ta về một tác vụ học chuỗi là mô hình hóa ngôn ngữ,
trong đó mục tiêu là dự đoán token tiếp theo dựa trên tất cả các token trước đó trong một chuỗi.
Trong trường hợp này, chúng ta chỉ muốn điều kiện hóa dựa trên ngữ cảnh bên trái,
vì vậy việc kết nối một chiều của RNN tiêu chuẩn có vẻ phù hợp.
Tuy nhiên, còn nhiều ngữ cảnh tác vụ học chuỗi khác
mà trong đó hoàn toàn có thể điều kiện hóa dự đoán tại mỗi bước thời gian
dựa trên cả ngữ cảnh bên trái và bên phải.
Ví dụ, hãy xem xét việc nhận diện từ loại.
Tại sao chúng ta không nên tính đến ngữ cảnh theo cả hai hướng
khi xác định từ loại gắn với một từ nhất định?

Một tác vụ phổ biến khác---thường hữu ích như một bài tập pretraining
trước khi fine-tuning một mô hình trên tác vụ thực tế cần quan tâm---
là che ngẫu nhiên các token trong một tài liệu văn bản, sau đó huấn luyện
một mô hình chuỗi để dự đoán giá trị của các token bị thiếu.
Lưu ý rằng tùy thuộc vào phần nằm sau chỗ trống,
giá trị có khả năng nhất của token bị thiếu sẽ thay đổi đáng kể:

* I am `___`.
* I am `___` hungry.
* I am `___` hungry, and I can eat half a pig.

Trong câu đầu tiên, “happy” có vẻ là một ứng viên phù hợp.
Các từ “not” và “very” có vẻ hợp lý trong câu thứ hai,
nhưng “not” dường như không phù hợp với câu thứ ba.


May mắn thay, một kỹ thuật đơn giản có thể biến đổi bất kỳ RNN một chiều nào
thành RNN hai chiều :cite:`Schuster.Paliwal.1997`.
Chúng ta chỉ cần triển khai hai lớp RNN một chiều
được nối với nhau theo các hướng ngược nhau
và hoạt động trên cùng một đầu vào (:numref:`fig_birnn`).
Đối với lớp RNN thứ nhất,
đầu vào đầu tiên là $\mathbf{x}_1$
và đầu vào cuối cùng là $\mathbf{x}_T$,
nhưng đối với lớp RNN thứ hai,
đầu vào đầu tiên là $\mathbf{x}_T$
và đầu vào cuối cùng là $\mathbf{x}_1$.
Để tạo đầu ra của lớp RNN hai chiều này,
chúng ta chỉ cần nối các đầu ra tương ứng
của hai lớp RNN một chiều bên dưới lại với nhau.


![Kiến trúc của RNN hai chiều.](../img/birnn.svg)
:label:`fig_birnn`


Về mặt hình thức, tại bất kỳ bước thời gian $t$ nào,
chúng ta xét một đầu vào minibatch $\mathbf{X}_t \in \mathbb{R}^{n \times d}$
(số lượng ví dụ $=n$; số lượng đầu vào trong mỗi ví dụ $=d$)
và gọi hàm kích hoạt của lớp ẩn là $\phi$.
Trong kiến trúc hai chiều,
các trạng thái ẩn tiến và lùi tại bước thời gian này lần lượt là
$\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$
và $\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$,
trong đó $h$ là số lượng đơn vị ẩn.
Các cập nhật trạng thái ẩn tiến và lùi như sau:


$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{\textrm{hh}}^{(f)}  + \mathbf{b}_\textrm{h}^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{\textrm{hh}}^{(b)}  + \mathbf{b}_\textrm{h}^{(b)}),
\end{aligned}
$$

trong đó các trọng số $\mathbf{W}_{\textrm{xh}}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{\textrm{hh}}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{\textrm{xh}}^{(b)} \in \mathbb{R}^{d \times h}, \textrm{ and } \mathbf{W}_{\textrm{hh}}^{(b)} \in \mathbb{R}^{h \times h}$, cùng các bias $\mathbf{b}_\textrm{h}^{(f)} \in \mathbb{R}^{1 \times h}$ và $\mathbf{b}_\textrm{h}^{(b)} \in \mathbb{R}^{1 \times h}$ đều là các tham số của mô hình.

Tiếp theo, chúng ta nối các trạng thái ẩn tiến và lùi
$\overrightarrow{\mathbf{H}}_t$ và $\overleftarrow{\mathbf{H}}_t$
để thu được trạng thái ẩn $\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$
đưa vào lớp đầu ra.
Trong các RNN hai chiều sâu với nhiều lớp ẩn,
thông tin này được truyền dưới dạng *đầu vào* cho lớp hai chiều tiếp theo.
Cuối cùng, lớp đầu ra tính toán đầu ra
$\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (số lượng đầu ra $=q$):

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}.$$

Ở đây, ma trận trọng số $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{2h \times q}$
và bias $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$
là các tham số của lớp đầu ra.
Mặc dù về mặt kỹ thuật, hai hướng có thể có số lượng đơn vị ẩn khác nhau,
lựa chọn thiết kế này hiếm khi được sử dụng trong thực tế.
Bây giờ, chúng ta sẽ trình bày một cách triển khai đơn giản của RNN hai chiều.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import npx, np
from mxnet.gluon import rnn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
```

## Triển khai từ đầu

Nếu muốn triển khai RNN hai chiều từ đầu, chúng ta có thể
đưa vào hai instance `RNNScratch` một chiều
với các tham số có thể học riêng biệt.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class BiRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.f_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.num_hiddens *= 2  # The output dimension will be doubled
```

```{.python .input}
%%tab jax
class BiRNNScratch(d2l.Module):
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        self.f_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.num_hiddens *= 2  # The output dimension will be doubled
```

Các trạng thái của RNN tiến và lùi
được cập nhật riêng biệt,
trong khi đầu ra của hai RNN này được nối lại với nhau.

```{.python .input}
%%tab all
@d2l.add_to_class(BiRNNScratch)
def forward(self, inputs, Hs=None):
    f_H, b_H = Hs if Hs is not None else (None, None)
    f_outputs, f_H = self.f_rnn(inputs, f_H)
    b_outputs, b_H = self.b_rnn(reversed(inputs), b_H)
    outputs = [d2l.concat((f, b), -1) for f, b in zip(
        f_outputs, reversed(b_outputs))]
    return outputs, (f_H, b_H)
```

## Triển khai ngắn gọn

:begin_tab:`pytorch, mxnet, tensorflow`
Sử dụng các API cấp cao,
chúng ta có thể triển khai RNN hai chiều một cách ngắn gọn hơn.
Ở đây, chúng ta lấy một mô hình GRU làm ví dụ.
:end_tab:

:begin_tab:`jax`
Flax API không cung cấp các lớp RNN, do đó không có
khái niệm về đối số `bidirectional`. Nếu cần một lớp hai chiều,
ta phải đảo ngược các đầu vào theo cách thủ công như trong phần triển khai từ đầu.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
class BiGRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.rnn = rnn.GRU(num_hiddens, bidirectional=True)
        if tab.selected('pytorch'):
            self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional=True)
        self.num_hiddens *= 2
```

## Tóm tắt

Trong RNN hai chiều, trạng thái ẩn tại mỗi bước thời gian đồng thời được xác định bởi dữ liệu trước và sau bước thời gian hiện tại. RNN hai chiều chủ yếu hữu ích cho việc mã hóa chuỗi và ước lượng các quan sát dựa trên ngữ cảnh hai chiều. RNN hai chiều rất tốn kém khi huấn luyện do các chuỗi gradient dài.

## Bài tập

1. Nếu các hướng khác nhau sử dụng số lượng đơn vị ẩn khác nhau, hình dạng của $\mathbf{H}_t$ sẽ thay đổi như thế nào?
1. Hãy thiết kế một RNN hai chiều với nhiều lớp ẩn.
1. Hiện tượng đa nghĩa phổ biến trong các ngôn ngữ tự nhiên. Ví dụ, từ “bank” có các nghĩa khác nhau trong các ngữ cảnh “i went to the bank to deposit cash” và “i went to the bank to sit down”. Làm thế nào để thiết kế một mô hình mạng nơ-ron sao cho khi được cung cấp một chuỗi ngữ cảnh và một từ, mô hình trả về biểu diễn vector của từ trong đúng ngữ cảnh? Những loại kiến trúc mạng nơ-ron nào phù hợp hơn để xử lý hiện tượng đa nghĩa?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/339)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1059)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18019)
:end_tab: