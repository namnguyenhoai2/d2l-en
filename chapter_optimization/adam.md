# Adam
:label:`sec_adam`

Trong các phần thảo luận dẫn đến phần này, chúng ta đã gặp một số kỹ thuật để tối ưu hóa hiệu quả. Hãy cùng điểm lại chi tiết chúng tại đây:

* Chúng ta đã thấy rằng :numref:`sec_sgd` hiệu quả hơn Gradient Descent khi giải các bài toán tối ưu hóa, chẳng hạn nhờ khả năng chống chịu vốn có đối với dữ liệu dư thừa.
* Chúng ta đã thấy rằng :numref:`sec_minibatch_sgd` mang lại hiệu quả bổ sung đáng kể nhờ vectorization, sử dụng các tập quan sát lớn hơn trong một minibatch. Đây là chìa khóa để xử lý song song hiệu quả trên nhiều máy, nhiều GPU và nói chung.
* :numref:`sec_momentum` đã bổ sung một cơ chế tổng hợp lịch sử các gradient trước đó để tăng tốc hội tụ.
* :numref:`sec_adagrad` đã sử dụng scaling theo từng tọa độ để cho phép một preconditioner hiệu quả về mặt tính toán.
* :numref:`sec_rmsprop` đã tách scaling theo từng tọa độ khỏi việc điều chỉnh learning rate.

Adam :cite:`Kingma.Ba.2014` kết hợp tất cả các kỹ thuật này thành một thuật toán học hiệu quả. Đúng như dự đoán, đây là một thuật toán đã trở nên khá phổ biến, là một trong những thuật toán tối ưu hóa mạnh mẽ và hiệu quả hơn để sử dụng trong deep learning. Tuy nhiên, nó không hoàn toàn không có vấn đề. Cụ thể, :cite:`Reddi.Kale.Kumar.2019` chỉ ra rằng có những tình huống Adam có thể phân kỳ do kiểm soát phương sai kém. Trong một công trình tiếp theo, :citet:`Zaheer.Reddi.Sachan.ea.2018` đã đề xuất một bản vá nhanh cho Adam, gọi là Yogi, nhằm giải quyết các vấn đề này. Chúng ta sẽ tìm hiểu thêm về điều này sau. Bây giờ, hãy xem lại thuật toán Adam.

## Thuật toán

Một trong những thành phần then chốt của Adam là nó sử dụng trung bình động có trọng số theo hàm mũ (exponential weighted moving averages, còn gọi là leaky averaging) để ước lượng cả momentum lẫn moment bậc hai của gradient. Nghĩa là, nó sử dụng các biến trạng thái

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

Ở đây $\beta_1$ và $\beta_2$ là các tham số trọng số không âm. Các lựa chọn thường dùng là $\beta_1 = 0.9$ và $\beta_2 = 0.999$. Nghĩa là, ước lượng phương sai thay đổi *chậm hơn nhiều* so với hạng momentum. Lưu ý rằng nếu khởi tạo $\mathbf{v}_0 = \mathbf{s}_0 = 0$, ban đầu chúng ta sẽ có một lượng bias đáng kể hướng về các giá trị nhỏ hơn. Điều này có thể được xử lý bằng cách sử dụng thực tế rằng $\sum_{i=0}^{t-1} \beta^i = \frac{1 - \beta^t}{1 - \beta}$ để chuẩn hóa lại các hạng. Tương ứng, các biến trạng thái đã chuẩn hóa được cho bởi

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \textrm{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

Với các ước lượng phù hợp, giờ đây chúng ta có thể viết các phương trình cập nhật. Trước hết, chúng ta rescale gradient theo cách rất giống RMSProp để thu được

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

Không giống RMSProp, bản cập nhật của chúng ta sử dụng momentum $\hat{\mathbf{v}}_t$ thay vì chính gradient. Hơn nữa, có một khác biệt nhỏ về hình thức là việc rescale diễn ra bằng $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$ thay vì $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$. Cách đầu tiên có thể hoạt động tốt hơn một chút trong thực tế, do đó có sự khác biệt so với RMSProp. Thông thường, chúng ta chọn $\epsilon = 10^{-6}$ để đạt sự cân bằng tốt giữa tính ổn định số và độ chính xác.

Giờ đây chúng ta đã có đầy đủ các thành phần để tính toán các cập nhật. Điều này có phần hơi thiếu kịch tính, và chúng ta có một cập nhật đơn giản có dạng

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

Khi xem xét thiết kế của Adam, nguồn cảm hứng của nó rất rõ ràng. Momentum và scale hiện diện rõ trong các biến trạng thái. Định nghĩa khá đặc biệt của chúng buộc chúng ta phải khử bias cho các hạng (điều này có thể được khắc phục bằng một điều kiện khởi tạo và cập nhật hơi khác). Thứ hai, việc kết hợp cả hai hạng khá trực tiếp, dựa trên RMSProp. Cuối cùng, learning rate tường minh $\eta$ cho phép chúng ta kiểm soát độ dài bước để giải quyết các vấn đề hội tụ.

## Triển khai

Việc triển khai Adam từ đầu không quá khó. Để thuận tiện, chúng ta lưu bộ đếm bước thời gian $t$ trong dictionary `hyperparams`. Ngoài ra, mọi thứ đều đơn giản.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adam_states(feature_dim):
    v_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
```

Chúng ta đã sẵn sàng sử dụng Adam để huấn luyện mô hình. Chúng ta dùng learning rate $\eta = 0.01$.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

Một cách triển khai ngắn gọn hơn rất trực tiếp vì `adam` là một trong các thuật toán được cung cấp như một phần của thư viện tối ưu hóa Gluon `trainer`. Do đó, chúng ta chỉ cần truyền các tham số cấu hình cho một bản triển khai trong Gluon.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## Yogi

Một trong các vấn đề của Adam là nó có thể không hội tụ ngay cả trong các thiết lập lồi khi ước lượng moment bậc hai trong $\mathbf{s}_t$ tăng vọt. Để khắc phục, :citet:`Zaheer.Reddi.Sachan.ea.2018` đã đề xuất một cập nhật (và khởi tạo) tinh chỉnh cho $\mathbf{s}_t$. Để hiểu điều gì đang xảy ra, hãy viết lại cập nhật Adam như sau:

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$

Bất cứ khi nào $\mathbf{g}_t^2$ có phương sai lớn hoặc các cập nhật thưa thớt, $\mathbf{s}_t$ có thể quên các giá trị trước đó quá nhanh. Một cách khắc phục khả dĩ là thay $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$ bằng $\mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$. Giờ đây, độ lớn của cập nhật không còn phụ thuộc vào mức độ sai lệch. Điều này cho ra các cập nhật Yogi

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$

Các tác giả cũng khuyên nên khởi tạo momentum trên một batch ban đầu lớn hơn thay vì chỉ dựa trên ước lượng điểm ban đầu. Chúng ta bỏ qua chi tiết vì chúng không thiết yếu cho phần thảo luận và vì ngay cả không có điều này, khả năng hội tụ vẫn khá tốt.

```{.python .input}
#@tab mxnet
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab pytorch
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab tensorflow
def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

## Tóm tắt

* Adam kết hợp các đặc điểm của nhiều thuật toán tối ưu hóa thành một quy tắc cập nhật khá mạnh mẽ.
* Được xây dựng dựa trên RMSProp, Adam cũng sử dụng EWMA trên gradient ngẫu nhiên của minibatch.
* Adam sử dụng hiệu chỉnh bias để điều chỉnh cho khởi đầu chậm khi ước lượng momentum và moment bậc hai.
* Với các gradient có phương sai đáng kể, chúng ta có thể gặp các vấn đề về hội tụ. Chúng có thể được cải thiện bằng cách sử dụng minibatch lớn hơn hoặc chuyển sang một ước lượng được cải tiến cho $\mathbf{s}_t$. Yogi cung cấp một lựa chọn thay thế như vậy.

## Bài tập

1. Điều chỉnh learning rate và quan sát, phân tích các kết quả thực nghiệm.
1. Bạn có thể viết lại các cập nhật momentum và moment bậc hai sao cho không cần hiệu chỉnh bias không?
1. Tại sao bạn cần giảm learning rate $\eta$ khi chúng ta hội tụ?
1. Hãy thử xây dựng một trường hợp mà Adam phân kỳ và Yogi hội tụ.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/358)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1078)
:end_tab:


:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1079)
:end_tab: