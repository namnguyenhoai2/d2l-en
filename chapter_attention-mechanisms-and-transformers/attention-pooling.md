# Attention Pooling theo Độ tương đồng

:label:`sec_attention-pooling`

Giờ đây, khi đã giới thiệu các thành phần chính của cơ chế attention, hãy sử dụng chúng trong một bối cảnh khá cổ điển, cụ thể là hồi quy và phân loại thông qua ước lượng mật độ kernel :cite:`Nadaraya.1964,Watson.1964`. Phần mở rộng này chỉ nhằm cung cấp thêm kiến thức nền: hoàn toàn không bắt buộc và có thể bỏ qua nếu cần. 
Về cốt lõi, các bộ ước lượng Nadaraya--Watson dựa vào một kernel độ tương đồng $\alpha(\mathbf{q}, \mathbf{k})$ liên hệ các query $\mathbf{q}$ với các key $\mathbf{k}$. Một số kernel phổ biến là

$$\begin{aligned}
\alpha(\mathbf{q}, \mathbf{k}) & = \exp\left(-\frac{1}{2} \|\mathbf{q} - \mathbf{k}\|^2 \right) && \textrm{Gaussian;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = 1 \textrm{ if } \|\mathbf{q} - \mathbf{k}\| \leq 1 && \textrm{Boxcar;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = \mathop{\mathrm{max}}\left(0, 1 - \|\mathbf{q} - \mathbf{k}\|\right) && \textrm{Epanechikov.}
\end{aligned}
$$

Chúng ta có thể chọn thêm rất nhiều phương án khác. Hãy xem một [bài viết Wikipedia](https://en.wikipedia.org/wiki/Kernel_(statistics)) để có phần tổng quan mở rộng hơn và tìm hiểu cách lựa chọn kernel liên quan đến ước lượng mật độ kernel, đôi khi còn được gọi là *Parzen Windows* :cite:`parzen1957consistent`. Tất cả các kernel đều mang tính heuristic và có thể được tinh chỉnh. Chẳng hạn, chúng ta có thể điều chỉnh độ rộng, không chỉ trên phạm vi toàn cục mà thậm chí theo từng tọa độ. Dù vậy, tất cả chúng đều dẫn đến phương trình sau cho cả hồi quy lẫn phân loại:

$$f(\mathbf{q}) = \sum_i \mathbf{v}_i \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}.$$

Trong trường hợp hồi quy (vô hướng) với các quan sát $(\mathbf{x}_i, y_i)$ lần lượt cho feature và label, $\mathbf{v}_i = y_i$ là các vô hướng, $\mathbf{k}_i = \mathbf{x}_i$ là các vector, và query $\mathbf{q}$ biểu thị vị trí mới mà tại đó $f$ cần được đánh giá. Trong trường hợp phân loại (đa lớp), chúng ta dùng one-hot encoding của $y_i$ để thu được $\mathbf{v}_i$. Một thuộc tính thuận tiện của bộ ước lượng này là không cần huấn luyện. Hơn nữa, nếu thu hẹp kernel một cách phù hợp khi lượng dữ liệu tăng lên, phương pháp này là nhất quán :cite:`mack1982weak`, tức là nó sẽ hội tụ về một nghiệm tối ưu theo thống kê nào đó. Hãy bắt đầu bằng cách xem xét một số kernel.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

d2l.use_svg_display()
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np

d2l.use_svg_display()
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import linen as nn
```

## [**Kernel và Dữ liệu**]

Tất cả các kernel $\alpha(\mathbf{k}, \mathbf{q})$ được định nghĩa trong phần này đều *bất biến đối với phép tịnh tiến và phép quay*; nghĩa là, nếu chúng ta dịch chuyển và xoay $\mathbf{k}$ và $\mathbf{q}$ theo cùng một cách, giá trị của $\alpha$ vẫn không đổi. Để đơn giản, do đó ta chọn các đối số vô hướng $k, q \in \mathbb{R}$ và chọn key $k = 0$ làm gốc tọa độ. Khi đó ta có:

```{.python .input}
%%tab all
# Define some kernels
def gaussian(x):
    return d2l.exp(-x**2 / 2)

def boxcar(x):
    return d2l.abs(x) < 1.0

def constant(x):
    return 1.0 + 0 * x
 
if tab.selected('pytorch'):
    def epanechikov(x):
        return torch.max(1 - d2l.abs(x), torch.zeros_like(x))
if tab.selected('mxnet'):
    def epanechikov(x):
        return np.maximum(1 - d2l.abs(x), 0)
if tab.selected('tensorflow'):
    def epanechikov(x):
        return tf.maximum(1 - d2l.abs(x), 0)
if tab.selected('jax'):
    def epanechikov(x):
        return jnp.maximum(1 - d2l.abs(x), 0)
```

```{.python .input}
%%tab all
fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))

kernels = (gaussian, boxcar, constant, epanechikov)
names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')
x = d2l.arange(-2.5, 2.5, 0.1)
for kernel, name, ax in zip(kernels, names, axes):
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        ax.plot(d2l.numpy(x), d2l.numpy(kernel(x)))
    if tab.selected('jax'):
        ax.plot(x, kernel(x))
    ax.set_xlabel(name)

d2l.plt.show()
```

Các kernel khác nhau tương ứng với những khái niệm khác nhau về phạm vi và độ trơn. Chẳng hạn, kernel boxcar chỉ attention đến các quan sát trong khoảng cách $1$ (hoặc một hyperparameter được xác định theo cách khác) và thực hiện điều này không phân biệt.

Để thấy ước lượng Nadaraya--Watson hoạt động như thế nào, hãy định nghĩa một số dữ liệu huấn luyện. Sau đây, chúng ta sử dụng quan hệ phụ thuộc

$$y_i = 2\sin(x_i) + x_i + \epsilon,$$

trong đó $\epsilon$ được lấy mẫu từ phân phối chuẩn có trung bình bằng không và phương sai đơn vị. Chúng ta lấy 40 mẫu huấn luyện.

```{.python .input}
%%tab all
def f(x):
    return 2 * d2l.sin(x) + x

n = 40
if tab.selected('pytorch'):
    x_train, _ = torch.sort(d2l.rand(n) * 5)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('mxnet'):
    x_train = np.sort(d2l.rand(n) * 5, axis=None)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('tensorflow'):
    x_train = tf.sort(d2l.rand((n,1)) * 5, 0)
    y_train = f(x_train) + d2l.normal((n, 1))
if tab.selected('jax'):
    x_train = jnp.sort(jax.random.uniform(d2l.get_key(), (n,)) * 5)
    y_train = f(x_train) + jax.random.normal(d2l.get_key(), (n,))
x_val = d2l.arange(0, 5, 0.1)
y_val = f(x_val)
```

## [**Attention Pooling qua Hồi quy Nadaraya--Watson**]

Giờ đây, khi đã có dữ liệu và các kernel, tất cả những gì ta cần là một hàm tính các ước lượng hồi quy kernel. Lưu ý rằng chúng ta cũng muốn thu được các trọng số kernel tương đối để thực hiện một số chẩn đoán nhỏ. Vì vậy, trước hết ta tính kernel giữa tất cả feature huấn luyện (biến đồng biến) `x_train` và tất cả feature xác thực `x_val`. Điều này tạo ra một ma trận, sau đó chúng ta chuẩn hóa nó. Khi nhân với các label huấn luyện `y_train`, ta thu được các ước lượng.

Hãy nhớ lại attention pooling trong :eqref:`eq_attention_pooling`. Hãy coi mỗi feature xác thực là một query, còn mỗi cặp feature--label huấn luyện là một cặp key--value. Do đó, các trọng số kernel tương đối đã chuẩn hóa (`attention_w` bên dưới) chính là *trọng số attention*.

```{.python .input}
%%tab all
def nadaraya_watson(x_train, y_train, x_val, kernel):
    dists = d2l.reshape(x_train, (-1, 1)) - d2l.reshape(x_val, (1, -1))
    # Each column/row corresponds to each query/key
    k = d2l.astype(kernel(dists), d2l.float32)
    # Normalization over keys for each query
    attention_w = k / d2l.reduce_sum(k, 0)
    if tab.selected('pytorch'):
        y_hat = y_train@attention_w
    if tab.selected('mxnet'):
        y_hat = np.dot(y_train, attention_w)
    if tab.selected('tensorflow'):
        y_hat = d2l.transpose(d2l.transpose(y_train)@attention_w)
    if tab.selected('jax'):
        y_hat = y_train@attention_w
    return y_hat, attention_w
```

Hãy xem các kernel khác nhau tạo ra kiểu ước lượng nào.

```{.python .input}
%%tab all
def plot(x_train, y_train, x_val, y_val, kernels, names, attention=False):
    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        if attention:
            if tab.selected('pytorch', 'mxnet', 'tensorflow'):
                pcm = ax.imshow(d2l.numpy(attention_w), cmap='Reds')
            if tab.selected('jax'):
                pcm = ax.imshow(attention_w, cmap='Reds')
        else:
            ax.plot(x_val, y_hat)
            ax.plot(x_val, y_val, 'm--')
            ax.plot(x_train, y_train, 'o', alpha=0.5);
        ax.set_xlabel(name)
        if not attention:
            ax.legend(['y_hat', 'y'])
    if attention:
        fig.colorbar(pcm, ax=axes, shrink=0.7)
```

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names)
```

Điều đầu tiên dễ nhận thấy là cả ba kernel không tầm thường (Gaussian, Boxcar và Epanechikov) đều tạo ra các ước lượng khá hữu dụng và không quá xa hàm thực. Chỉ kernel hằng, dẫn đến ước lượng tầm thường $f(x) = \frac{1}{n} \sum_i y_i$, mới cho kết quả khá phi thực tế. Hãy xem xét kỹ hơn trọng số attention:

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

Phần trực quan hóa cho thấy rõ vì sao các ước lượng của Gaussian, Boxcar và Epanechikov rất giống nhau: xét cho cùng, chúng được suy ra từ các trọng số attention rất tương tự nhau, bất chấp dạng hàm khác nhau của kernel. Điều này đặt ra câu hỏi liệu có phải lúc nào cũng như vậy hay không.

## [**Điều chỉnh Attention Pooling**]

Chúng ta có thể thay kernel Gaussian bằng một kernel có độ rộng khác. Tức là, chúng ta có thể dùng 
$\alpha(\mathbf{q}, \mathbf{k}) = \exp\left(-\frac{1}{2 \sigma^2} \|\mathbf{q} - \mathbf{k}\|^2 \right)$ trong đó $\sigma^2$ xác định độ rộng của kernel. Hãy xem liệu điều này có ảnh hưởng đến kết quả hay không.

```{.python .input}
%%tab all
sigmas = (0.1, 0.2, 0.5, 1)
names = ['Sigma ' + str(sigma) for sigma in sigmas]

def gaussian_with_width(sigma): 
    return (lambda x: d2l.exp(-x**2 / (2*sigma**2)))

kernels = [gaussian_with_width(sigma) for sigma in sigmas]
plot(x_train, y_train, x_val, y_val, kernels, names)
```

Rõ ràng, kernel càng hẹp thì ước lượng càng ít trơn. Đồng thời, nó thích ứng tốt hơn với các biến thiên cục bộ. Hãy xem các trọng số attention tương ứng.

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

Như dự đoán, kernel càng hẹp thì phạm vi các trọng số attention lớn càng hẹp. Cũng rõ ràng rằng việc chọn cùng một độ rộng có thể không lý tưởng. Thực tế, :citet:`Silverman86` đã đề xuất một heuristic phụ thuộc vào mật độ cục bộ. Nhiều "mẹo" như vậy khác đã được đề xuất. Chẳng hạn, :citet:`norelli2022asif` đã sử dụng một kỹ thuật nội suy láng giềng gần tương tự để thiết kế các biểu diễn hình ảnh và văn bản đa phương thức.

Người đọc tinh ý có thể tự hỏi vì sao chúng ta lại đi sâu vào một phương pháp đã hơn nửa thế kỷ tuổi. Thứ nhất, đây là một trong những tiền thân sớm nhất của các cơ chế attention hiện đại. Thứ hai, nó rất phù hợp để trực quan hóa. Thứ ba, và cũng quan trọng không kém, nó cho thấy những giới hạn của các cơ chế attention được thiết kế thủ công. Một chiến lược tốt hơn nhiều là *học* cơ chế này, bằng cách học các biểu diễn cho query và key. Đây là nội dung chúng ta sẽ bắt đầu trong các phần tiếp theo.


## Tóm tắt

Hồi quy kernel Nadaraya--Watson là một tiền thân sớm của các cơ chế attention hiện nay. 
Nó có thể được sử dụng trực tiếp với rất ít hoặc không cần huấn luyện hay tinh chỉnh, cho cả phân loại lẫn hồi quy. 
Trọng số attention được gán theo độ tương đồng (hoặc khoảng cách) giữa query và key, cũng như theo số lượng quan sát tương tự có sẵn.

## Bài tập

1. Các ước lượng mật độ Parzen windows được cho bởi $\hat{p}(\mathbf{x}) = \frac{1}{n} \sum_i k(\mathbf{x}, \mathbf{x}_i)$. Hãy chứng minh rằng, đối với phân loại nhị phân, hàm $\hat{p}(\mathbf{x}, y=1) - \hat{p}(\mathbf{x}, y=-1)$ thu được bằng Parzen windows tương đương với phân loại Nadaraya--Watson.
1. Cài đặt stochastic gradient descent để học một giá trị tốt cho độ rộng kernel trong hồi quy Nadaraya--Watson.
    1. Điều gì xảy ra nếu bạn chỉ sử dụng các ước lượng ở trên để tối thiểu hóa trực tiếp $(f(\mathbf{x_i}) - y_i)^2$? Gợi ý: $y_i$ là một phần của các hạng tử dùng để tính $f$.
    1. Loại bỏ $(\mathbf{x}_i, y_i)$ khỏi ước lượng cho $f(\mathbf{x}_i)$ và tối ưu theo độ rộng kernel. Bạn vẫn quan sát thấy overfitting chứ?
1. Giả sử mọi $\mathbf{x}$ đều nằm trên mặt cầu đơn vị, tức là tất cả đều thỏa mãn $\|\mathbf{x}\| = 1$. Bạn có thể đơn giản hóa hạng tử $\|\mathbf{x} - \mathbf{x}_i\|^2$ trong số mũ không? Gợi ý: sau này chúng ta sẽ thấy rằng điều này liên hệ rất chặt chẽ với dot product attention.
1. Hãy nhớ rằng :citet:`mack1982weak` đã chứng minh ước lượng Nadaraya--Watson là nhất quán. Bạn nên giảm scale cho cơ chế attention nhanh đến mức nào khi có thêm dữ liệu? Hãy đưa ra một số trực giác cho câu trả lời của bạn. Điều này có phụ thuộc vào số chiều của dữ liệu không? Phụ thuộc như thế nào?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1599)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/3866)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18026)
:end_tab: