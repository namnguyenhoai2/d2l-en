```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Tính ổn định số và Khởi tạo
:label:`sec_numerical_stability`


Cho đến nay, mọi mô hình mà chúng ta đã triển khai
đều yêu cầu khởi tạo các tham số
theo một phân phối được chỉ định trước nào đó.
Cho đến lúc này, chúng ta đã mặc nhiên chấp nhận sơ đồ khởi tạo,
bỏ qua các chi tiết về cách đưa ra những lựa chọn này.
Bạn thậm chí có thể đã có ấn tượng rằng những lựa chọn này
không đặc biệt quan trọng.
Ngược lại, việc lựa chọn sơ đồ khởi tạo
đóng vai trò đáng kể trong quá trình học của mạng nơ-ron,
và có thể là yếu tố then chốt để duy trì tính ổn định số.
Hơn nữa, các lựa chọn này có thể liên hệ theo những cách thú vị
với việc lựa chọn hàm kích hoạt phi tuyến.
Hàm chúng ta chọn và cách khởi tạo tham số
có thể quyết định tốc độ hội tụ của thuật toán tối ưu hóa.
Những lựa chọn kém ở đây có thể khiến chúng ta gặp phải
gradient bùng nổ hoặc tiêu biến trong khi huấn luyện.
Trong phần này, chúng ta sẽ tìm hiểu các chủ đề này chi tiết hơn
và thảo luận một số kinh nghiệm thực hành hữu ích
mà bạn sẽ thấy hữu ích
trong suốt sự nghiệp deep learning của mình.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from jax import grad, vmap
```

## Gradient tiêu biến và bùng nổ

Xét một mạng sâu có $L$ lớp,
đầu vào $\mathbf{x}$ và đầu ra $\mathbf{o}$.
Với mỗi lớp $l$ được xác định bởi một phép biến đổi $f_l$
được tham số hóa bởi các trọng số $\mathbf{W}^{(l)}$,
có đầu ra lớp ẩn là $\mathbf{h}^{(l)}$ (đặt $\mathbf{h}^{(0)} = \mathbf{x}$),
mạng của chúng ta có thể được biểu diễn như sau:

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \textrm{ and thus } \mathbf{o} = f_L \circ \cdots \circ f_1(\mathbf{x}).$$

Nếu tất cả đầu ra lớp ẩn và đầu vào đều là vector,
ta có thể viết gradient của $\mathbf{o}$ theo
bất kỳ tập tham số $\mathbf{W}^{(l)}$ nào như sau:

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\textrm{def}}{=}} \cdots \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\textrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\textrm{def}}{=}}.$$

Nói cách khác, gradient này là
tích của $L-l$ ma trận
$\mathbf{M}^{(L)} \cdots \mathbf{M}^{(l+1)}$
và vector gradient $\mathbf{v}^{(l)}$.
Do đó, chúng ta dễ gặp phải cùng các
vấn đề tràn dưới số học thường xuất hiện
khi nhân quá nhiều xác suất với nhau.
Khi xử lý xác suất, một thủ thuật phổ biến là
chuyển sang không gian log, tức là chuyển
áp lực từ phần định trị sang số mũ
của biểu diễn số.
Đáng tiếc, vấn đề ở trên của chúng ta nghiêm trọng hơn:
ban đầu, các ma trận $\mathbf{M}^{(l)}$ có thể có rất nhiều trị riêng khác nhau.
Chúng có thể nhỏ hoặc lớn, và
tích của chúng có thể *rất lớn* hoặc *rất nhỏ*.

Các rủi ro do gradient không ổn định gây ra
không chỉ dừng ở biểu diễn số.
Gradient có độ lớn khó đoán
cũng đe dọa sự ổn định của các thuật toán tối ưu hóa.
Chúng ta có thể phải đối mặt với các cập nhật tham số
hoặc (i) quá lớn, phá hủy mô hình
(vấn đề *gradient bùng nổ*);
hoặc (ii) quá nhỏ
(vấn đề *gradient tiêu biến*),
khiến việc học trở nên bất khả thi vì các tham số
hầu như không thay đổi ở mỗi lần cập nhật.


### (**Gradient tiêu biến**)

Một nguyên nhân thường gặp gây ra vấn đề gradient tiêu biến
là lựa chọn hàm kích hoạt $\sigma$
được thêm sau các phép toán tuyến tính của mỗi lớp.
Trong lịch sử, hàm sigmoid
$1/(1 + \exp(-x))$ (được giới thiệu trong :numref:`sec_mlp`)
từng phổ biến vì nó giống một hàm ngưỡng.
Do các mạng nơ-ron nhân tạo thời kỳ đầu được lấy cảm hứng
từ mạng nơ-ron sinh học,
ý tưởng về các nơ-ron hoặc kích hoạt *hoàn toàn* hoặc *không hề*
(như các nơ-ron sinh học) có vẻ hấp dẫn.
Hãy xem kỹ hơn sigmoid
để hiểu tại sao nó có thể gây ra gradient tiêu biến.

```{.python .input}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab jax
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.sigmoid(x)
grad_sigmoid = vmap(grad(jax.nn.sigmoid))
d2l.plot(x, [y, grad_sigmoid(x)],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

Như bạn có thể thấy, (**gradient của sigmoid tiêu biến
cả khi đầu vào của nó lớn lẫn khi đầu vào của nó nhỏ**).
Hơn nữa, khi lan truyền ngược qua nhiều lớp,
trừ khi chúng ta nằm trong vùng Goldilocks, nơi
đầu vào của nhiều sigmoid gần bằng không,
gradient của tích tổng thể có thể tiêu biến.
Khi mạng của chúng ta có nhiều lớp,
trừ khi cẩn thận, gradient
có khả năng sẽ bị chặn tại một lớp nào đó.
Thực tế, vấn đề này từng gây khó khăn cho việc huấn luyện mạng sâu.
Do đó, ReLU, vốn ổn định hơn
(nhưng kém hợp lý hơn về mặt sinh học thần kinh),
đã trở thành lựa chọn mặc định của các practitioner.


### [**Gradient bùng nổ**]

Vấn đề ngược lại, khi gradient bùng nổ,
cũng có thể gây bối rối tương tự.
Để minh họa rõ hơn,
chúng ta lấy 100 ma trận ngẫu nhiên Gaussian
và nhân chúng với một ma trận ban đầu.
Với thang đo mà chúng ta đã chọn
(lựa chọn phương sai $\sigma^2=1$),
tích ma trận bùng nổ.
Khi điều này xảy ra do việc khởi tạo
một mạng sâu, chúng ta không có cơ hội nào để khiến
một optimizer gradient descent hội tụ.

```{.python .input}
%%tab mxnet
M = np.random.normal(size=(4, 4))
print('a single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))
print('after multiplying 100 matrices', M)
```

```{.python .input}
%%tab pytorch
M = torch.normal(0, 1, size=(4, 4))
print('a single matrix \n',M)
for i in range(100):
    M = M @ torch.normal(0, 1, size=(4, 4))
print('after multiplying 100 matrices\n', M)
```

```{.python .input}
%%tab tensorflow
M = tf.random.normal((4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))
print('after multiplying 100 matrices\n', M.numpy())
```

```{.python .input}
%%tab jax
get_key = lambda: jax.random.PRNGKey(d2l.get_seed())  # Generate PRNG keys
M = jax.random.normal(get_key(), (4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = jnp.matmul(M, jax.random.normal(get_key(), (4, 4)))
print('after multiplying 100 matrices\n', M)
```

### Phá vỡ tính đối xứng

Một vấn đề khác trong thiết kế mạng nơ-ron
là tính đối xứng vốn có trong quá trình tham số hóa chúng.
Giả sử chúng ta có một MLP đơn giản
với một lớp ẩn và hai đơn vị.
Trong trường hợp này, ta có thể hoán vị các trọng số $\mathbf{W}^{(1)}$
của lớp thứ nhất và tương tự hoán vị
các trọng số của lớp đầu ra
để thu được cùng một hàm.
Không có gì đặc biệt để phân biệt
đơn vị ẩn thứ nhất và thứ hai.
Nói cách khác, ta có tính đối xứng hoán vị
giữa các đơn vị ẩn của mỗi lớp.

Đây không chỉ là một phiền toái về mặt lý thuyết.
Xét MLP một lớp ẩn với hai đơn vị ẩn đã đề cập ở trên.
Để minh họa,
giả sử lớp đầu ra biến đổi hai đơn vị ẩn thành chỉ một đơn vị đầu ra.
Hãy hình dung điều gì xảy ra nếu chúng ta khởi tạo
tất cả tham số của lớp ẩn
là $\mathbf{W}^{(1)} = c$ với một hằng số $c$ nào đó.
Trong trường hợp này, khi lan truyền xuôi,
mỗi đơn vị ẩn nhận cùng đầu vào và tham số
nên tạo ra cùng một kích hoạt
được đưa vào đơn vị đầu ra.
Trong quá trình lan truyền ngược,
lấy đạo hàm của đơn vị đầu ra theo các tham số $\mathbf{W}^{(1)}$ sẽ cho gradient mà mọi phần tử đều có cùng giá trị.
Vì vậy, sau mỗi lần lặp dựa trên gradient (ví dụ: minibatch stochastic gradient descent),
mọi phần tử của $\mathbf{W}^{(1)}$ vẫn có cùng giá trị.
Các lần lặp như vậy
sẽ không bao giờ tự *phá vỡ tính đối xứng*
và chúng ta có thể không bao giờ khai thác được
khả năng biểu đạt của mạng.
Lớp ẩn sẽ hoạt động
như thể nó chỉ có một đơn vị duy nhất.
Lưu ý rằng mặc dù minibatch stochastic gradient descent không phá vỡ tính đối xứng này,
regularization dropout (sẽ được giới thiệu sau) thì có thể!


## Khởi tạo tham số

Một cách để giải quyết---hoặc ít nhất giảm nhẹ---
các vấn đề nêu trên là khởi tạo cẩn thận.
Như chúng ta sẽ thấy sau,
sự cẩn thận bổ sung trong quá trình tối ưu hóa
và regularization phù hợp có thể cải thiện thêm tính ổn định.


### Khởi tạo mặc định

Trong các phần trước, ví dụ trong :numref:`sec_linear_concise`,
chúng ta đã dùng phân phối chuẩn
để khởi tạo giá trị các trọng số.
Nếu không chỉ định phương thức khởi tạo, framework sẽ
dùng phương thức khởi tạo ngẫu nhiên mặc định, thường hoạt động tốt trong thực tế
với các bài toán có quy mô vừa phải.






### Khởi tạo Xavier
:label:`subsec_xavier`

Hãy xem xét phân phối thang đo của
một đầu ra $o_{i}$ cho một lớp fully connected nào đó
*không có phi tuyến tính*.
Với $n_\textrm{in}$ đầu vào $x_j$
và các trọng số tương ứng $w_{ij}$ của lớp này,
một đầu ra được cho bởi

$$o_{i} = \sum_{j=1}^{n_\textrm{in}} w_{ij} x_j.$$

Các trọng số $w_{ij}$ đều được lấy mẫu
độc lập từ cùng một phân phối.
Hơn nữa, giả sử rằng phân phối này
có trung bình bằng không và phương sai $\sigma^2$.
Lưu ý rằng điều này không có nghĩa phân phối phải là Gaussian,
mà chỉ cần trung bình và phương sai tồn tại.
Hiện tại, giả sử rằng các đầu vào của lớp $x_j$
cũng có trung bình bằng không và phương sai $\gamma^2$
và chúng độc lập với $w_{ij}$ cũng như độc lập với nhau.
Trong trường hợp này, ta có thể tính trung bình của $o_i$:

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\textrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\textrm{in}} E[w_{ij}] E[x_j] \\&= 0, \end{aligned}$$

và phương sai:

$$
\begin{aligned}
    \textrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\textrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

Một cách để giữ phương sai cố định
là đặt $n_\textrm{in} \sigma^2 = 1$.
Bây giờ hãy xét backpropagation.
Ở đây, chúng ta gặp một vấn đề tương tự,
mặc dù các gradient được truyền từ những layer gần đầu ra hơn.
Sử dụng cùng lập luận như đối với forward propagation,
ta thấy rằng phương sai của các gradient có thể bùng nổ
trừ khi $n_\textrm{out} \sigma^2 = 1$,
trong đó $n_\textrm{out}$ là số lượng đầu ra của layer này.
Điều này đặt chúng ta vào tình thế tiến thoái lưỡng nan:
ta không thể đồng thời thỏa mãn cả hai điều kiện.
Thay vào đó, ta chỉ cố gắng thỏa mãn:

$$
\begin{aligned}
\frac{1}{2} (n_\textrm{in} + n_\textrm{out}) \sigma^2 = 1 \textrm{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\textrm{in} + n_\textrm{out}}}.
\end{aligned}
$$

Đây là lập luận nền tảng cho *Xavier initialization*,
hiện nay đã trở thành tiêu chuẩn và mang lại lợi ích thực tế,
được đặt theo tên tác giả đầu tiên trong nhóm tạo ra nó :cite:`Glorot.Bengio.2010`.
Thông thường, Xavier initialization
lấy mẫu các trọng số từ một phân phối Gaussian
có trung bình bằng không và phương sai
$\sigma^2 = \frac{2}{n_\textrm{in} + n_\textrm{out}}$.
Ta cũng có thể điều chỉnh điều này để
chọn phương sai khi lấy mẫu trọng số
từ một phân phối đều.
Lưu ý rằng phân phối đều $U(-a, a)$ có phương sai $\frac{a^2}{3}$.
Thay $\frac{a^2}{3}$ vào điều kiện của ta đối với $\sigma^2$
cho ta khởi tạo theo

$$U\left(-\sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}, \sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}\right).$$

Mặc dù giả định về việc không có các phi tuyến
trong lập luận toán học ở trên
có thể dễ dàng bị vi phạm trong neural network,
phương pháp Xavier initialization
lại cho thấy hiệu quả tốt trong thực tế.


### Mở rộng

Lập luận ở trên mới chỉ chạm nhẹ vào bề mặt
của các phương pháp hiện đại để khởi tạo tham số.
Một deep learning framework thường triển khai hơn một tá heuristic khác nhau.
Hơn nữa, khởi tạo tham số vẫn tiếp tục là
một lĩnh vực nghiên cứu cơ bản sôi động trong deep learning.
Trong số đó có các heuristic chuyên biệt cho
các tham số được ràng buộc (chia sẻ), super-resolution,
sequence model, và những tình huống khác.
Ví dụ,
:citet:`Xiao.Bahri.Sohl-Dickstein.ea.2018` đã chứng minh khả năng huấn luyện
neural network 10.000 layer mà không cần các thủ thuật kiến trúc
bằng cách sử dụng một phương pháp khởi tạo được thiết kế cẩn thận.

Nếu bạn quan tâm đến chủ đề này, chúng tôi gợi ý
tìm hiểu sâu các nội dung của module này,
đọc các bài báo đã đề xuất và phân tích từng heuristic,
sau đó khám phá các công bố mới nhất về chủ đề này.
Có lẽ bạn sẽ tình cờ bắt gặp, hoặc thậm chí phát minh,
một ý tưởng thông minh và đóng góp một implementation cho các deep learning framework.


## Tóm tắt

Gradient biến mất và bùng nổ là những vấn đề phổ biến trong deep network. Cần hết sức cẩn thận khi khởi tạo tham số để bảo đảm gradient và tham số luôn được kiểm soát tốt.
Cần có các heuristic khởi tạo để bảo đảm các gradient ban đầu không quá lớn cũng không quá nhỏ.
Khởi tạo ngẫu nhiên là yếu tố then chốt để bảo đảm tính đối xứng được phá vỡ trước khi tối ưu hóa.
Xavier initialization đề xuất rằng, đối với mỗi layer, phương sai của bất kỳ đầu ra nào không bị ảnh hưởng bởi số lượng đầu vào, và phương sai của bất kỳ gradient nào không bị ảnh hưởng bởi số lượng đầu ra.
Hàm kích hoạt ReLU giảm nhẹ vấn đề gradient biến mất. Điều này có thể tăng tốc độ hội tụ.

## Bài tập

1. Bạn có thể thiết kế những trường hợp khác mà neural network có thể thể hiện tính đối xứng cần được phá vỡ, ngoài tính đối xứng hoán vị trong các layer của một MLP không?
1. Ta có thể khởi tạo tất cả tham số trọng số trong linear regression hoặc softmax regression bằng cùng một giá trị không?
1. Hãy tìm các cận giải tích cho các trị riêng của tích hai ma trận. Điều này cho bạn biết gì về việc bảo đảm gradient được điều kiện hóa tốt?
1. Nếu biết rằng một số hạng phân kỳ, ta có thể khắc phục điều này sau đó không? Hãy xem bài báo về layerwise adaptive rate scaling để lấy cảm hứng :cite:`You.Gitman.Ginsburg.2017`.


:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/235)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17986)
:end_tab: