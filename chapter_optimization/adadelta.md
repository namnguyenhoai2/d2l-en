# Adadelta
:label:`sec_adadelta`

Adadelta là một biến thể khác của AdaGrad (:numref:`sec_adagrad`). Điểm khác biệt chính nằm ở việc nó làm giảm mức độ mà learning rate thích ứng theo từng tọa độ. Hơn nữa, theo truyền thống, nó được xem là không có learning rate vì nó sử dụng chính độ lớn của thay đổi làm thước đo cho những thay đổi trong tương lai. Thuật toán này được đề xuất trong :citet:`Zeiler.2012`. Dựa trên phần thảo luận về các thuật toán trước đó, nó khá dễ hiểu.

## Thuật toán

Tóm lại, Adadelta sử dụng hai biến trạng thái: $\mathbf{s}_t$ để lưu trung bình rò rỉ (leaky average) của moment bậc hai của gradient, và $\Delta\mathbf{x}_t$ để lưu trung bình rò rỉ của moment bậc hai của thay đổi các tham số trong chính mô hình. Lưu ý rằng chúng tôi sử dụng ký hiệu và cách đặt tên ban đầu của các tác giả để tương thích với các công bố và implementation khác (không có lý do thực sự nào khác để dùng các biến Hy Lạp khác nhằm biểu thị một tham số có cùng mục đích trong momentum, Adagrad, RMSProp và Adadelta).

Dưới đây là các chi tiết kỹ thuật của Adadelta. Với tham số đang xét là $\rho$, ta nhận được các cập nhật rò rỉ sau, tương tự như :numref:`sec_rmsprop`:

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

Điểm khác biệt so với :numref:`sec_rmsprop` là chúng ta thực hiện cập nhật với gradient đã được tái tỷ lệ $\mathbf{g}_t'$, tức là

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

Vậy gradient đã được tái tỷ lệ $\mathbf{g}_t'$ là gì? Ta có thể tính nó như sau:

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

trong đó $\Delta \mathbf{x}_{t-1}$ là trung bình rò rỉ của các gradient đã tái tỷ lệ bình phương $\mathbf{g}_t'$. Chúng ta khởi tạo $\Delta \mathbf{x}_{0}$ bằng $0$ và cập nhật nó ở mỗi bước bằng $\mathbf{g}_t'$, tức là

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

và $\epsilon$ (một giá trị nhỏ như $10^{-5}$) được thêm vào để duy trì tính ổn định số.

## Cài đặt

Adadelta cần duy trì hai biến trạng thái cho mỗi biến, $\mathbf{s}_t$ và $\Delta\mathbf{x}_t$. Điều này dẫn đến implementation sau.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-place updates via [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-place updates via [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

Việc chọn $\rho = 0.9$ tương ứng với thời gian bán rã là 10 cho mỗi lần cập nhật tham số. Giá trị này thường hoạt động khá tốt. Ta có hành vi sau.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

Để có một implementation ngắn gọn, chúng ta chỉ cần sử dụng thuật toán Adadelta từ các API cấp cao. Điều này cho ta one-liner sau đây để gọi hàm một cách cô đọng hơn nhiều.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
# adadelta is not converging at default learning rate
# but it is converging at lr = 5.0
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

## Tóm tắt

* Adadelta không có tham số learning rate. Thay vào đó, nó sử dụng chính tốc độ thay đổi của các tham số để điều chỉnh learning rate.
* Adadelta cần hai biến trạng thái để lưu các moment bậc hai của gradient và sự thay đổi trong các tham số.
* Adadelta sử dụng các trung bình rò rỉ để duy trì một ước lượng liên tục của các thống kê phù hợp.

## Bài tập

1. Điều chỉnh giá trị của $\rho$. Điều gì xảy ra?
1. Hãy chỉ ra cách implementation thuật toán mà không sử dụng $\mathbf{g}_t'$. Vì sao đây có thể là một ý tưởng hay?
1. Adadelta có thực sự không cần learning rate không? Bạn có thể tìm các bài toán tối ưu hóa khiến Adadelta thất bại không?
1. So sánh Adadelta với Adagrad và RMS prop để thảo luận về hành vi hội tụ của chúng.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/357)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1076)
:end_tab:


:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1077)
:end_tab: