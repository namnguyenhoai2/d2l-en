# Adagrad
:label:`sec_adagrad`

Hãy bắt đầu bằng cách xem xét các bài toán học có những đặc trưng xuất hiện không thường xuyên.


## Đặc trưng thưa và Learning Rate

Hãy tưởng tượng rằng chúng ta đang huấn luyện một language model. Để đạt độ chính xác tốt, thông thường chúng ta muốn giảm learning rate khi tiếp tục huấn luyện, thường với tốc độ $\mathcal{O}(t^{-\frac{1}{2}})$ hoặc chậm hơn. Bây giờ, hãy xem xét một mô hình được huấn luyện trên các đặc trưng thưa, tức là các đặc trưng chỉ xuất hiện không thường xuyên. Điều này phổ biến trong ngôn ngữ tự nhiên, ví dụ, xác suất chúng ta gặp từ *preconditioning* thấp hơn rất nhiều so với từ *learning*. Tuy nhiên, điều này cũng phổ biến trong các lĩnh vực khác như quảng cáo tính toán và collaborative filtering được cá nhân hóa. Xét cho cùng, có nhiều thứ chỉ được một số ít người quan tâm.

Các tham số liên kết với những đặc trưng không thường xuyên chỉ nhận được các cập nhật có ý nghĩa khi những đặc trưng này xuất hiện. Với learning rate giảm dần, chúng ta có thể rơi vào tình huống các tham số cho đặc trưng phổ biến hội tụ khá nhanh về giá trị tối ưu, trong khi với các đặc trưng không thường xuyên, chúng ta vẫn chưa quan sát chúng đủ thường xuyên để xác định giá trị tối ưu. Nói cách khác, learning rate hoặc giảm quá chậm đối với đặc trưng thường xuyên, hoặc quá nhanh đối với đặc trưng không thường xuyên.

Một cách xử lý khả dĩ cho vấn đề này là đếm số lần chúng ta thấy một đặc trưng cụ thể và dùng số đó làm đồng hồ để điều chỉnh learning rate. Tức là, thay vì chọn learning rate có dạng $\eta = \frac{\eta_0}{\sqrt{t + c}}$, chúng ta có thể dùng $\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$. Ở đây, $s(i, t)$ đếm số phần tử khác không của đặc trưng $i$ mà chúng ta đã quan sát cho đến thời điểm $t$. Việc này thực ra khá dễ triển khai mà không có overhead đáng kể. Tuy nhiên, nó thất bại khi dữ liệu không thực sự thưa mà chỉ là dữ liệu trong đó gradient thường rất nhỏ và chỉ hiếm khi lớn. Xét cho cùng, không rõ phải xác định ranh giới ở đâu để một giá trị được xem là một đặc trưng đã quan sát hay không.

Adagrad của :citet:`Duchi.Hazan.Singer.2011` giải quyết vấn đề này bằng cách thay bộ đếm khá thô $s(i, t)$ bằng một đại lượng tổng hợp từ bình phương các gradient đã quan sát trước đó. Cụ thể, nó dùng $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$ để điều chỉnh learning rate. Cách này có hai lợi ích: thứ nhất, chúng ta không còn cần quyết định chính xác khi nào một gradient đủ lớn. Thứ hai, nó tự động scale theo độ lớn của gradient. Các tọa độ thường xuyên tương ứng với gradient lớn sẽ được scale down đáng kể, trong khi các tọa độ khác có gradient nhỏ được xử lý nhẹ nhàng hơn nhiều. Trên thực tế, điều này tạo ra một quy trình tối ưu hóa rất hiệu quả cho quảng cáo tính toán và các bài toán liên quan. Nhưng điều này che giấu một số lợi ích bổ sung vốn có của Adagrad, được hiểu rõ nhất trong ngữ cảnh preconditioning.


## Preconditioning

Các bài toán tối ưu lồi phù hợp để phân tích đặc tính của thuật toán. Xét cho cùng, với hầu hết các bài toán không lồi, rất khó để suy ra các đảm bảo lý thuyết có ý nghĩa, nhưng *trực giác* và *sự thấu hiểu* thường vẫn có thể áp dụng. Hãy xem xét bài toán tối thiểu hóa $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$.

Như đã thấy trong :numref:`sec_momentum`, ta có thể viết lại bài toán này theo eigendecomposition $\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$ để thu được một bài toán đơn giản hơn nhiều, trong đó mỗi tọa độ có thể được giải riêng lẻ:

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

Ở đây, chúng ta dùng $\bar{\mathbf{x}} = \mathbf{U} \mathbf{x}$ và do đó $\bar{\mathbf{c}} = \mathbf{U} \mathbf{c}$. Bài toán đã hiệu chỉnh có nghiệm tối thiểu là $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$ và giá trị nhỏ nhất là $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$. Điều này dễ tính toán hơn nhiều vì $\boldsymbol{\Lambda}$ là một ma trận đường chéo chứa các eigenvalue của $\mathbf{Q}$.

Nếu ta nhiễu loạn nhẹ $\mathbf{c}$, ta hy vọng chỉ thấy những thay đổi nhỏ trong nghiệm tối thiểu của $f$. Đáng tiếc, điều này không đúng. Mặc dù những thay đổi nhỏ trong $\mathbf{c}$ dẫn đến những thay đổi nhỏ tương ứng trong $\bar{\mathbf{c}}$, điều này không đúng với nghiệm tối thiểu của $f$ (và tương ứng là của $\bar{f}$). Khi các eigenvalue $\boldsymbol{\Lambda}_i$ lớn, chúng ta sẽ chỉ thấy những thay đổi nhỏ trong $\bar{x}_i$ và trong giá trị nhỏ nhất của $\bar{f}$. Ngược lại, với $\boldsymbol{\Lambda}_i$ nhỏ, những thay đổi trong $\bar{x}_i$ có thể rất lớn. Tỷ lệ giữa eigenvalue lớn nhất và nhỏ nhất được gọi là condition number của một bài toán tối ưu hóa.

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

Nếu condition number $\kappa$ lớn, rất khó giải chính xác bài toán tối ưu hóa. Chúng ta cần đảm bảo xử lý cẩn thận một dải động lớn các giá trị. Phân tích của chúng ta dẫn đến một câu hỏi hiển nhiên, dù hơi ngây thơ: liệu chúng ta có thể đơn giản “sửa” bài toán bằng cách biến dạng không gian sao cho mọi eigenvalue đều bằng $1$ không? Về lý thuyết, điều này khá dễ: ta chỉ cần eigenvalue và eigenvector của $\mathbf{Q}$ để rescale bài toán từ $\mathbf{x}$ thành một bài toán trong $\mathbf{z} \stackrel{\textrm{def}}{=} \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$. Trong hệ tọa độ mới, $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$ có thể được đơn giản hóa thành $\|\mathbf{z}\|^2$. Than ôi, đây là một đề xuất khá không thực tế. Việc tính eigenvalue và eigenvector nói chung *tốn kém hơn nhiều* so với việc giải bài toán thực tế.

Mặc dù việc tính chính xác eigenvalue có thể tốn kém, việc ước lượng chúng và tính chúng dù chỉ gần đúng có thể đã tốt hơn nhiều so với không làm gì cả. Cụ thể, chúng ta có thể dùng các phần tử trên đường chéo của $\mathbf{Q}$ và rescale tương ứng. Cách này rẻ hơn *nhiều* so với việc tính eigenvalue.

$$\tilde{\mathbf{Q}} = \textrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \textrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

Trong trường hợp này, ta có $\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$ và cụ thể $\tilde{\mathbf{Q}}_{ii} = 1$ với mọi $i$. Trong hầu hết trường hợp, điều này đơn giản hóa condition number đáng kể. Chẳng hạn, với các trường hợp đã thảo luận trước đó, điều này sẽ loại bỏ hoàn toàn vấn đề đang xét vì bài toán được căn chỉnh theo trục.

Đáng tiếc, chúng ta lại gặp một vấn đề khác: trong deep learning, thông thường ta thậm chí không có quyền truy cập vào đạo hàm bậc hai của hàm mục tiêu: với $\mathbf{x} \in \mathbb{R}^d$, đạo hàm bậc hai ngay cả trên một minibatch cũng có thể cần $\mathcal{O}(d^2)$ không gian và công sức để tính, khiến nó không khả thi trên thực tế. Ý tưởng khéo léo của Adagrad là dùng một proxy cho đường chéo khó nắm bắt đó của Hessian, vừa tương đối rẻ để tính vừa hiệu quả---chính là độ lớn của gradient.

Để hiểu vì sao cách này hoạt động, hãy xem xét $\bar{f}(\bar{\mathbf{x}})$. Ta có

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$

trong đó $\bar{\mathbf{x}}_0$ là nghiệm tối thiểu của $\bar{f}$. Do đó, độ lớn của gradient phụ thuộc cả vào $\boldsymbol{\Lambda}$ lẫn khoảng cách tới điểm tối ưu. Nếu $\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$ không thay đổi, đây sẽ là tất cả những gì cần thiết. Xét cho cùng, trong trường hợp này, độ lớn của gradient $\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$ là đủ. Vì AdaGrad là một thuật toán stochastic gradient descent, chúng ta sẽ thấy các gradient có phương sai khác không ngay cả tại điểm tối ưu. Do đó, ta có thể an toàn dùng phương sai của gradient làm một proxy rẻ cho thang đo của Hessian. Phân tích kỹ lưỡng nằm ngoài phạm vi của phần này (sẽ cần vài trang). Chúng tôi giới thiệu người đọc tham khảo :cite:`Duchi.Hazan.Singer.2011` để biết chi tiết.

## Thuật toán

Hãy chính thức hóa phần thảo luận ở trên. Chúng ta dùng biến $\mathbf{s}_t$ để tích lũy phương sai gradient trong quá khứ như sau.

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

Ở đây, các phép toán được áp dụng theo từng tọa độ. Nghĩa là, $\mathbf{v}^2$ có các phần tử là $v_i^2$. Tương tự, $\frac{1}{\sqrt{v}}$ có các phần tử là $\frac{1}{\sqrt{v_i}}$ và $\mathbf{u} \cdot \mathbf{v}$ có các phần tử là $u_i v_i$. Như trước, $\eta$ là learning rate và $\epsilon$ là một hằng số cộng thêm, đảm bảo rằng chúng ta không chia cho $0$. Cuối cùng, chúng ta khởi tạo $\mathbf{s}_0 = \mathbf{0}$.

Cũng như trong trường hợp momentum, chúng ta cần theo dõi một biến phụ trợ, trong trường hợp này để cho phép một learning rate riêng cho từng tọa độ. Điều này không làm tăng đáng kể chi phí của Adagrad so với SGD, đơn giản vì chi phí chính thường là tính $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$ và đạo hàm của nó.

Lưu ý rằng việc tích lũy gradient bình phương trong $\mathbf{s}_t$ có nghĩa là $\mathbf{s}_t$ về cơ bản tăng với tốc độ tuyến tính (trong thực tế chậm hơn tuyến tính một chút, vì gradient ban đầu giảm dần). Điều này dẫn đến learning rate $\mathcal{O}(t^{-\frac{1}{2}})$, mặc dù được điều chỉnh theo từng tọa độ. Đối với các bài toán lồi, điều này hoàn toàn phù hợp. Tuy nhiên, trong deep learning, chúng ta có thể muốn giảm learning rate chậm hơn đáng kể. Điều này dẫn đến một số biến thể của Adagrad mà chúng ta sẽ thảo luận trong các chương tiếp theo. Bây giờ, hãy xem nó hoạt động như thế nào trong một bài toán lồi bậc hai. Chúng ta sử dụng cùng bài toán như trước:

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Chúng ta sẽ triển khai Adagrad bằng cùng learning rate đã dùng trước đó, tức là $\eta = 0.4$. Như có thể thấy, quỹ đạo lặp của biến độc lập mượt mà hơn. Tuy nhiên, do hiệu ứng tích lũy của $\boldsymbol{s}_t$, learning rate liên tục giảm, nên biến độc lập không di chuyển nhiều trong các giai đoạn lặp sau.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
#@tab all
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

Khi tăng learning rate lên $2$, chúng ta thấy hành vi tốt hơn nhiều. Điều này đã cho thấy việc giảm learning rate có thể khá quyết liệt, ngay cả trong trường hợp không có nhiễu, và chúng ta cần đảm bảo rằng các tham số hội tụ phù hợp.

```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

## Triển khai từ đầu

Cũng như phương pháp momentum, Adagrad cần duy trì một biến trạng thái có cùng hình dạng với các tham số.

```{.python .input}
#@tab mxnet
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(s + tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

So với thí nghiệm trong :numref:`sec_minibatch_sgd`, chúng ta sử dụng một learning rate lớn hơn để huấn luyện mô hình.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## Triển khai ngắn gọn

Sử dụng instance `Trainer` của thuật toán `adagrad`, chúng ta có thể gọi thuật toán Adagrad trong Gluon.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate' : 0.1}, data_iter)
```

## Tóm tắt

* Adagrad giảm learning rate động theo từng tọa độ.
* Nó sử dụng độ lớn của gradient để điều chỉnh tốc độ đạt được tiến triển — các tọa độ có gradient lớn được bù bằng learning rate nhỏ hơn.
* Việc tính đạo hàm bậc hai chính xác thường không khả thi trong các bài toán deep learning do các ràng buộc về bộ nhớ và tính toán. Gradient có thể là một đại diện hữu ích.
* Nếu bài toán tối ưu hóa có cấu trúc khá không đồng đều, Adagrad có thể giúp giảm thiểu sự méo lệch.
* Adagrad đặc biệt hiệu quả với các đặc trưng thưa, nơi learning rate cần giảm chậm hơn đối với các hạng tử xuất hiện không thường xuyên.
* Trong các bài toán deep learning, Adagrad đôi khi có thể quá quyết liệt trong việc giảm learning rate. Chúng ta sẽ thảo luận các chiến lược để giảm thiểu điều này trong ngữ cảnh của :numref:`sec_adam`.

## Bài tập

1. Chứng minh rằng với ma trận trực giao $\mathbf{U}$ và vector $\mathbf{c}$, điều sau đây đúng: $\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$. Tại sao điều này có nghĩa là độ lớn của các nhiễu loạn không thay đổi sau một phép đổi biến trực giao?
1. Thử Adagrad cho $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ và cả cho hàm mục tiêu được xoay 45 độ, tức là $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Nó có hoạt động khác đi không?
1. Chứng minh [định lý vòng tròn Gerschgorin](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem), phát biểu rằng các trị riêng $\lambda_i$ của một ma trận $\mathbf{M}$ thỏa mãn $|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$ với ít nhất một lựa chọn của $j$.
1. Định lý Gerschgorin cho chúng ta biết gì về các trị riêng của ma trận được tiền điều kiện hóa theo đường chéo $\textrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \textrm{diag}^{-\frac{1}{2}}(\mathbf{M})$?
1. Thử Adagrad cho một mạng deep phù hợp, chẳng hạn như :numref:`sec_lenet` khi áp dụng cho Fashion-MNIST.
1. Bạn cần sửa đổi Adagrad như thế nào để đạt được tốc độ giảm learning rate ít quyết liệt hơn?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/355)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1072)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1073)
:end_tab: