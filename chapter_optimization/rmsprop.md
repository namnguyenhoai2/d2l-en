# RMSProp
:label:`sec_rmsprop`


Một trong những vấn đề chính ở :numref:`sec_adagrad` là learning rate giảm theo một lịch trình được xác định trước, thực chất là $\mathcal{O}(t^{-\frac{1}{2}})$. Mặc dù điều này nhìn chung phù hợp với các bài toán lồi, nó có thể không lý tưởng cho các bài toán không lồi, chẳng hạn như những bài toán gặp phải trong deep learning. Tuy vậy, tính thích ứng theo từng tọa độ của Adagrad lại rất đáng mong muốn với vai trò là một preconditioner.

:citet:`Tieleman.Hinton.2012` đã đề xuất thuật toán RMSProp như một cách khắc phục đơn giản nhằm tách việc lập lịch tốc độ khỏi learning rate thích ứng theo tọa độ. Vấn đề là Adagrad tích lũy bình phương của gradient $\mathbf{g}_t$ vào một vector trạng thái $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$. Do đó, $\mathbf{s}_t$ tiếp tục tăng không giới hạn do thiếu chuẩn hóa, về cơ bản tăng tuyến tính khi thuật toán hội tụ.

Một cách khắc phục vấn đề này là sử dụng $\mathbf{s}_t / t$. Với các phân phối hợp lý của $\mathbf{g}_t$, đại lượng này sẽ hội tụ. Không may là có thể mất rất lâu trước khi hành vi giới hạn bắt đầu trở nên đáng kể, vì quy trình ghi nhớ toàn bộ quỹ đạo của các giá trị. Một phương án khác là sử dụng trung bình rò rỉ (leaky average) theo cùng cách đã dùng trong phương pháp momentum, tức là $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$ với một tham số $\gamma > 0$. Giữ nguyên mọi phần khác sẽ tạo ra RMSProp.

## Thuật toán

Hãy viết chi tiết các phương trình.

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

Hằng số $\epsilon > 0$ thường được đặt thành $10^{-6}$ để đảm bảo rằng chúng ta không gặp phải phép chia cho 0 hoặc kích thước bước quá lớn. Với phần mở rộng này, giờ đây chúng ta có thể kiểm soát learning rate $\eta$ độc lập với việc scale được áp dụng theo từng tọa độ. Xét theo các trung bình rò rỉ, chúng ta có thể áp dụng cùng lập luận như đã áp dụng trước đó cho phương pháp momentum. Khai triển định nghĩa của $\mathbf{s}_t$ ta được

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

Như trước đây trong :numref:`sec_momentum`, chúng ta sử dụng $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$. Do đó, tổng các trọng số được chuẩn hóa thành $1$, với thời gian bán rã của một quan sát là $\gamma^{-1}$. Hãy trực quan hóa các trọng số cho 40 bước thời gian trước đó với nhiều lựa chọn khác nhau của $\gamma$.

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
from d2l import torch as d2l
import torch
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input}
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## Triển khai từ đầu

Như trước, chúng ta sử dụng hàm bậc hai $f(\mathbf{x})=0.1x_1^2+2x_2^2$ để quan sát quỹ đạo của RMSProp. Hãy nhớ rằng trong :numref:`sec_adagrad`, khi sử dụng Adagrad với learning rate là 0.4, các biến chỉ di chuyển rất chậm ở các giai đoạn sau của thuật toán vì learning rate giảm quá nhanh. Vì $\eta$ được kiểm soát riêng, điều này không xảy ra với RMSProp.

```{.python .input}
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

Tiếp theo, chúng ta triển khai RMSProp để sử dụng trong một deep network. Việc này cũng đơn giản tương tự.

```{.python .input}
#@tab mxnet,pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```{.python .input}
#@tab mxnet
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

Chúng ta đặt learning rate ban đầu là 0.01 và hạng trọng số $\gamma$ là 0.9. Nghĩa là, $\mathbf{s}$ tổng hợp trung bình trên $1/(1-\gamma) = 10$ quan sát gần nhất của gradient bình phương.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## Triển khai ngắn gọn

Vì RMSProp là một thuật toán khá phổ biến, nó cũng có sẵn trong instance `Trainer`. Tất cả những gì chúng ta cần làm là khởi tạo nó bằng một thuật toán có tên `rmsprop`, gán $\gamma$ cho tham số `gamma1`.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## Tóm tắt

* RMSProp rất giống Adagrad ở chỗ cả hai đều sử dụng bình phương của gradient để scale các hệ số.
* RMSProp có điểm chung với momentum là phép lấy trung bình rò rỉ. Tuy nhiên, RMSProp sử dụng kỹ thuật này để điều chỉnh preconditioner theo từng hệ số.
* Trên thực tế, learning rate cần được người thực nghiệm lập lịch.
* Hệ số $\gamma$ quyết định độ dài lịch sử khi điều chỉnh scale theo từng tọa độ.

## Bài tập

1. Điều gì xảy ra trong thực nghiệm nếu đặt $\gamma = 1$? Tại sao?
1. Xoay bài toán tối ưu để tối thiểu hóa $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Điều gì xảy ra với sự hội tụ?
1. Hãy thử xem RMSProp hoạt động thế nào trên một bài toán machine learning thực tế, chẳng hạn như huấn luyện trên Fashion-MNIST. Thử nghiệm các lựa chọn khác nhau để điều chỉnh learning rate.
1. Bạn có muốn điều chỉnh $\gamma$ khi quá trình tối ưu tiến triển không? RMSProp nhạy cảm với điều này đến mức nào?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/356)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1074)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1075)
:end_tab: