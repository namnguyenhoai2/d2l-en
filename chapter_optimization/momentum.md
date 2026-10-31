# Momentum
:label:`sec_momentum`

Trong :numref:`sec_sgd`, chúng ta đã xem xét điều gì xảy ra khi thực hiện stochastic gradient descent, tức là khi thực hiện tối ưu hóa mà chỉ có một biến thể nhiễu của gradient. Cụ thể, chúng ta nhận thấy rằng với gradient nhiễu, cần đặc biệt thận trọng khi chọn learning rate trước sự hiện diện của nhiễu. Nếu giảm nó quá nhanh, sự hội tụ sẽ đình trệ. Nếu quá nới lỏng, chúng ta sẽ không hội tụ đến một nghiệm đủ tốt vì nhiễu liên tục đẩy chúng ta rời xa điểm tối ưu.

## Kiến thức cơ bản

Trong phần này, chúng ta sẽ khám phá các thuật toán tối ưu hóa hiệu quả hơn, đặc biệt dành cho một số loại bài toán tối ưu hóa thường gặp trong thực tế.


### Trung bình rò rỉ

Phần trước đã thảo luận về minibatch SGD như một cách để tăng tốc tính toán. Nó cũng có tác dụng phụ hữu ích là việc lấy trung bình các gradient làm giảm lượng phương sai. Minibatch stochastic gradient descent có thể được tính như sau:

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

Để giữ ký hiệu đơn giản, ở đây chúng ta dùng $\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$ làm stochastic gradient descent cho mẫu $i$ sử dụng các trọng số được cập nhật tại thời điểm $t-1$.
Sẽ rất tốt nếu chúng ta có thể hưởng lợi từ hiệu ứng giảm phương sai ngay cả ngoài việc lấy trung bình gradient trên một minibatch. Một lựa chọn để thực hiện điều này là thay thế phép tính gradient bằng một "trung bình rò rỉ":

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

với một $\beta \in (0, 1)$. Điều này thực chất thay gradient tức thời bằng một gradient đã được lấy trung bình trên nhiều gradient *trong quá khứ*. $\mathbf{v}$ được gọi là *vận tốc*. Nó tích lũy các gradient trong quá khứ, tương tự như cách một quả bóng nặng lăn xuống địa hình của hàm mục tiêu tích hợp các lực trong quá khứ. Để xem chi tiết hơn điều gì đang xảy ra, hãy khai triển đệ quy $\mathbf{v}_t$ thành

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

$\beta$ lớn tương ứng với việc lấy trung bình trên phạm vi dài, trong khi $\beta$ nhỏ chỉ tương ứng với một hiệu chỉnh nhẹ so với phương pháp gradient. Gradient thay thế mới không còn chỉ theo hướng giảm dốc nhất trên một mẫu cụ thể, mà theo hướng trung bình có trọng số của các gradient trong quá khứ. Điều này cho phép chúng ta đạt được phần lớn lợi ích của việc lấy trung bình trên một batch mà không phải trả chi phí thực sự tính gradient trên batch đó. Chúng ta sẽ xem lại quy trình lấy trung bình này chi tiết hơn sau.

Lập luận trên là nền tảng cho những gì ngày nay được gọi là các phương pháp gradient *tăng tốc*, chẳng hạn như gradient có momentum. Chúng có thêm lợi ích là hiệu quả hơn nhiều trong những trường hợp bài toán tối ưu hóa kém điều kiện (tức là có một số hướng mà tiến độ chậm hơn nhiều so với các hướng khác, giống như một hẻm núi hẹp). Hơn nữa, chúng cho phép chúng ta lấy trung bình các gradient liên tiếp để có được các hướng giảm ổn định hơn. Thật vậy, khía cạnh tăng tốc ngay cả đối với các bài toán lồi không nhiễu là một trong những lý do chính khiến momentum hoạt động và hoạt động hiệu quả đến vậy.

Như có thể mong đợi, nhờ hiệu quả của nó, momentum là một chủ đề được nghiên cứu kỹ trong tối ưu hóa cho deep learning và nhiều lĩnh vực khác. Xem, chẳng hạn, [bài viết trình bày](https://distill.pub/2017/momentum/) xuất sắc của :citet:`Goh.2017` để có phân tích chuyên sâu và hình động tương tác. Nó được đề xuất bởi :citet:`Polyak.1964`. :citet:`Nesterov.2018` có thảo luận lý thuyết chi tiết trong bối cảnh tối ưu hóa lồi. Momentum trong deep learning đã được biết là có lợi từ lâu. Xem, chẳng hạn, thảo luận của :citet:`Sutskever.Martens.Dahl.ea.2013` để biết chi tiết.

### Một bài toán kém điều kiện

Để hiểu rõ hơn các đặc tính hình học của phương pháp momentum, chúng ta xem lại gradient descent, dù với một hàm mục tiêu kém dễ chịu hơn đáng kể. Nhớ rằng trong :numref:`sec_gd` chúng ta đã dùng $f(\mathbf{x}) = x_1^2 + 2 x_2^2$, tức là một hàm mục tiêu ellipsoid bị biến dạng vừa phải. Chúng ta tiếp tục làm biến dạng hàm này bằng cách kéo giãn nó theo hướng $x_1$ thông qua

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Như trước, $f$ đạt cực tiểu tại $(0, 0)$. Hàm này *rất* phẳng theo hướng $x_1$. Hãy xem điều gì xảy ra khi chúng ta thực hiện gradient descent như trước trên hàm mới này. Chúng ta chọn learning rate là $0.4$.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

Theo cấu trúc, gradient theo hướng $x_2$ *lớn hơn nhiều* và thay đổi nhanh hơn nhiều so với theo hướng ngang $x_1$. Do đó, chúng ta mắc kẹt giữa hai lựa chọn không mong muốn: nếu chọn learning rate nhỏ, chúng ta đảm bảo nghiệm không phân kỳ theo hướng $x_2$ nhưng phải chấp nhận hội tụ chậm theo hướng $x_1$. Ngược lại, với learning rate lớn, chúng ta tiến nhanh theo hướng $x_1$ nhưng phân kỳ theo $x_2$. Ví dụ dưới đây minh họa điều gì xảy ra ngay cả sau khi tăng nhẹ learning rate từ $0.4$ lên $0.6$. Sự hội tụ theo hướng $x_1$ được cải thiện nhưng chất lượng nghiệm tổng thể kém hơn nhiều.

```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### Phương pháp Momentum

Phương pháp momentum cho phép chúng ta giải quyết bài toán gradient descent được mô tả ở trên. Quan sát quỹ đạo tối ưu hóa ở trên, chúng ta có thể trực giác rằng việc lấy trung bình các gradient trong quá khứ sẽ hoạt động tốt. Xét cho cùng, theo hướng $x_1$, điều này sẽ tổng hợp các gradient cùng hướng tốt, từ đó tăng quãng đường chúng ta đi được ở mỗi bước. Ngược lại, theo hướng $x_2$, nơi các gradient dao động, một gradient tổng hợp sẽ giảm kích thước bước do các dao động triệt tiêu lẫn nhau.
Sử dụng $\mathbf{v}_t$ thay cho gradient $\mathbf{g}_t$ cho ta các phương trình cập nhật sau:

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

Lưu ý rằng với $\beta = 0$, chúng ta thu được gradient descent thông thường. Trước khi đi sâu hơn vào các đặc tính toán học, hãy nhanh chóng xem thuật toán hoạt động thế nào trong thực tế.

```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Như có thể thấy, ngay cả với cùng learning rate đã dùng trước đó, momentum vẫn hội tụ tốt. Hãy xem điều gì xảy ra khi chúng ta giảm tham số momentum. Giảm một nửa xuống $\beta = 0.25$ dẫn đến một quỹ đạo hầu như không hội tụ. Tuy vậy, nó vẫn tốt hơn nhiều so với không dùng momentum (khi nghiệm phân kỳ).

```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Lưu ý rằng chúng ta có thể kết hợp momentum với stochastic gradient descent và đặc biệt là minibatch stochastic gradient descent. Thay đổi duy nhất là trong trường hợp đó, chúng ta thay các gradient $\mathbf{g}_{t, t-1}$ bằng $\mathbf{g}_t$. Cuối cùng, để thuận tiện, chúng ta khởi tạo $\mathbf{v}_0 = 0$ tại thời điểm $t=0$. Hãy xem việc lấy trung bình rò rỉ thực sự tác động thế nào đến các cập nhật.

### Trọng số mẫu hiệu dụng

Nhớ rằng $\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$. Ở giới hạn, các hạng tử cộng lại thành $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$. Nói cách khác, thay vì thực hiện một bước có kích thước $\eta$ trong gradient descent hoặc stochastic gradient descent, chúng ta thực hiện một bước có kích thước $\frac{\eta}{1-\beta}$, đồng thời xử lý một hướng giảm có khả năng ổn định hơn nhiều. Đây là hai lợi ích trong một. Để minh họa cách trọng số hoạt động với các lựa chọn $\beta$ khác nhau, hãy xem sơ đồ bên dưới.

```{.python .input}
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## Thí nghiệm thực tế

Hãy xem momentum hoạt động thế nào trong thực tế, tức là khi được sử dụng trong bối cảnh của một optimizer phù hợp. Để làm điều này, chúng ta cần một implementation có khả năng mở rộng hơn đôi chút.

### Implementation từ đầu

So với stochastic gradient descent (minibatch), phương pháp momentum cần duy trì một tập các biến phụ trợ, tức là vận tốc. Nó có cùng hình dạng với các gradient (và các biến của bài toán tối ưu hóa). Trong implementation bên dưới, chúng ta gọi các biến này là `states`.

```{.python .input}
#@tab mxnet,pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input}
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```{.python .input}
#@tab mxnet
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```{.python .input}
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

Hãy xem cách này hoạt động trong thực tế.

```{.python .input}
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

Khi tăng hyperparameter momentum `momentum` lên 0.9, nó tương ứng với một kích thước mẫu hiệu dụng lớn hơn đáng kể là $\frac{1}{1 - 0.9} = 10$. Chúng ta giảm nhẹ learning rate xuống $0.01$ để giữ mọi thứ trong tầm kiểm soát.

```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

Việc giảm thêm learning rate sẽ xử lý mọi vấn đề của các bài toán tối ưu hóa không trơn. Đặt nó thành $0.005$ sẽ mang lại các đặc tính hội tụ tốt.

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### Cài đặt Ngắn gọn

Có rất ít việc cần làm trong Gluon vì solver `sgd` tiêu chuẩn đã tích hợp sẵn momentum. Việc đặt các tham số tương ứng sẽ cho quỹ đạo rất tương tự.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## Phân tích Lý thuyết

Cho đến nay, ví dụ 2D về $f(x) = 0.1 x_1^2 + 2 x_2^2$ có vẻ khá gượng ép. Bây giờ chúng ta sẽ thấy rằng ví dụ này thực ra khá điển hình cho các loại bài toán có thể gặp, ít nhất là trong trường hợp tối thiểu hóa các hàm mục tiêu bậc hai lồi.

### Hàm Lồi Bậc hai

Xét hàm

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

Đây là một hàm bậc hai tổng quát. Với các ma trận xác định dương $\mathbf{Q} \succ 0$, tức là các ma trận có các trị riêng dương, hàm này có một điểm tối thiểu tại $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$ với giá trị nhỏ nhất là $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Do đó, chúng ta có thể viết lại $h$ thành

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

Gradient được cho bởi $\partial_{\mathbf{x}} h(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$. Nghĩa là, nó được xác định bởi khoảng cách giữa $\mathbf{x}$ và điểm tối thiểu, nhân với $\mathbf{Q}$. Do đó, vận tốc cũng là một tổ hợp tuyến tính của các hạng tử $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$.

Vì $\mathbf{Q}$ là xác định dương, nó có thể được phân rã thành hệ riêng thông qua $\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$ với ma trận trực giao (quay) $\mathbf{O}$ và ma trận đường chéo $\boldsymbol{\Lambda}$ gồm các trị riêng dương. Điều này cho phép chúng ta thực hiện phép đổi biến từ $\mathbf{x}$ sang $\mathbf{z} \stackrel{\textrm{def}}{=} \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$ để thu được biểu thức đơn giản hơn nhiều:

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

Ở đây $b' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Vì $\mathbf{O}$ chỉ là một ma trận trực giao, điều này không làm nhiễu loạn gradient theo cách đáng kể. Biểu diễn theo $\mathbf{z}$, gradient descent trở thành

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

Điểm quan trọng trong biểu thức này là gradient descent *không trộn lẫn* giữa các không gian riêng khác nhau. Nghĩa là, khi được biểu diễn theo hệ riêng của $\mathbf{Q}$, bài toán tối ưu hóa diễn ra theo từng tọa độ. Điều này cũng đúng với

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

Qua đó, chúng ta vừa chứng minh định lý sau: gradient descent có và không có momentum đối với một hàm bậc hai lồi có thể được phân rã thành tối ưu hóa theo từng tọa độ theo hướng của các vector riêng của ma trận bậc hai.

### Hàm Vô hướng

Dựa trên kết quả trên, hãy xem điều gì xảy ra khi chúng ta tối thiểu hóa hàm $f(x) = \frac{\lambda}{2} x^2$. Với gradient descent, ta có

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

Bất cứ khi nào $|1 - \eta \lambda| < 1$, quá trình tối ưu hóa này hội tụ với tốc độ mũ vì sau $t$ bước ta có $x_t = (1 - \eta \lambda)^t x_0$. Điều này cho thấy tốc độ hội tụ ban đầu được cải thiện khi chúng ta tăng learning rate $\eta$ cho đến khi $\eta \lambda = 1$. Vượt quá mức đó, mọi thứ sẽ phân kỳ, và khi $\eta \lambda > 2$, bài toán tối ưu hóa sẽ phân kỳ.

```{.python .input}
#@tab all
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

Để phân tích sự hội tụ trong trường hợp có momentum, chúng ta bắt đầu bằng cách viết lại các phương trình cập nhật theo hai đại lượng vô hướng: một cho $x$ và một cho vận tốc $v$. Điều này cho:

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

Chúng ta dùng $\mathbf{R}$ để ký hiệu ma trận $2 \times 2$ chi phối hành vi hội tụ. Sau $t$ bước, lựa chọn ban đầu $[v_0, x_0]$ trở thành $\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$. Do đó, tốc độ hội tụ được quyết định bởi các trị riêng của $\mathbf{R}$. Xem bài viết [Distill post](https://distill.pub/2017/momentum/) của :citet:`Goh.2017` để xem một hình động tuyệt vời và :citet:`Flammarion.Bach.2015` để xem phân tích chi tiết. Có thể chứng minh rằng với $0 < \eta \lambda < 2 + 2 \beta$, vận tốc hội tụ. Đây là một khoảng tham số khả thi lớn hơn so với $0 < \eta \lambda < 2$ của gradient descent. Điều này cũng gợi ý rằng nhìn chung các giá trị lớn của $\beta$ là mong muốn. Các chi tiết sâu hơn đòi hỏi một lượng đáng kể kiến thức kỹ thuật, và chúng tôi đề xuất người đọc quan tâm tham khảo các ấn phẩm gốc.

## Tóm tắt

* Momentum thay thế gradient bằng trung bình suy giảm của các gradient trong quá khứ. Điều này tăng tốc hội tụ đáng kể.
* Nó hữu ích cho cả gradient descent không nhiễu và stochastic gradient descent có nhiễu.
* Momentum ngăn quá trình tối ưu hóa bị đình trệ, điều vốn dễ xảy ra hơn nhiều với stochastic gradient descent.
* Số lượng gradient hiệu dụng được cho bởi $\frac{1}{1-\beta}$ do việc giảm trọng số theo hàm mũ của dữ liệu quá khứ.
* Trong trường hợp các bài toán bậc hai lồi, điều này có thể được phân tích tường minh và chi tiết.
* Việc cài đặt khá đơn giản nhưng yêu cầu chúng ta lưu trữ thêm một vector trạng thái (vận tốc $\mathbf{v}$).

## Bài tập

1. Sử dụng các tổ hợp khác của hyperparameter momentum và learning rate, rồi quan sát và phân tích các kết quả thực nghiệm khác nhau.
1. Thử gradient descent và momentum cho một bài toán bậc hai có nhiều trị riêng, tức là $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$, ví dụ $\lambda_i = 2^{-i}$. Vẽ biểu đồ cách các giá trị của $x$ giảm dần với khởi tạo $x_i = 1$.
1. Suy ra giá trị nhỏ nhất và điểm tối thiểu của $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$.
1. Điều gì thay đổi khi chúng ta thực hiện stochastic gradient descent với momentum? Điều gì xảy ra khi dùng minibatch stochastic gradient descent với momentum? Hãy thử nghiệm với các tham số?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/354)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1070)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1071)
:end_tab: