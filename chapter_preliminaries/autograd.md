```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Automatic Differentiation
:label:`sec_autograd`

Hãy nhớ lại từ :numref:`sec_calculus` 
rằng việc tính đạo hàm là bước then chốt
trong tất cả các thuật toán tối ưu hóa
mà chúng ta sẽ sử dụng để huấn luyện mạng sâu.
Mặc dù các phép tính khá đơn giản,
việc tự tính chúng có thể tẻ nhạt và dễ xảy ra sai sót,
và những vấn đề này chỉ trở nên nghiêm trọng hơn
khi mô hình của chúng ta ngày càng phức tạp.

May mắn thay, tất cả các framework deep learning hiện đại
đều đảm nhiệm công việc này thay chúng ta
bằng cách cung cấp *automatic differentiation*
(thường được viết tắt là *autograd*).
Khi truyền dữ liệu qua từng hàm liên tiếp,
framework sẽ xây dựng một *computational graph*
theo dõi cách mỗi giá trị phụ thuộc vào các giá trị khác.
Để tính đạo hàm,
automatic differentiation
sẽ đi ngược qua graph này
và áp dụng chain rule.
Thuật toán tính toán áp dụng chain rule
theo cách này được gọi là *backpropagation*.

Mặc dù các thư viện autograd đã trở thành
một chủ đề được quan tâm nhiều trong thập kỷ qua,
chúng có lịch sử lâu dài.
Trên thực tế, những tài liệu tham khảo sớm nhất về autograd
đã có từ hơn nửa thế kỷ trước :cite:`Wengert.1964`.
Các ý tưởng cốt lõi đằng sau backpropagation hiện đại
bắt nguồn từ một luận án tiến sĩ năm 1980 :cite:`Speelpenning.1980`
và tiếp tục được phát triển vào cuối những năm 1980 :cite:`Griewank.1989`.
Mặc dù backpropagation đã trở thành phương pháp mặc định
để tính gradient, đây không phải là lựa chọn duy nhất.
Chẳng hạn, ngôn ngữ lập trình Julia sử dụng
forward propagation :cite:`Revels.Lubin.Papamarkou.2016`.
Trước khi tìm hiểu các phương pháp,
trước tiên hãy nắm vững package autograd.

```{.python .input}
%%tab mxnet
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from jax import numpy as jnp
```

## Một hàm đơn giản

Giả sử rằng chúng ta quan tâm đến việc
(**lấy đạo hàm của hàm
$y = 2\mathbf{x}^{\top}\mathbf{x}$
theo vector cột $\mathbf{x}$.**)
Để bắt đầu, chúng ta gán cho `x` một giá trị ban đầu.

```{.python .input  n=1}
%%tab mxnet
x = np.arange(4.0)
x
```

```{.python .input  n=7}
%%tab pytorch
x = torch.arange(4.0)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(4, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(4.0)
x
```

:begin_tab:`mxnet, pytorch, tensorflow`
[**Trước khi tính gradient
của $y$ theo $\mathbf{x}$,
chúng ta cần một nơi để lưu gradient.**]
Nói chung, chúng ta tránh cấp phát bộ nhớ mới
mỗi khi lấy một đạo hàm
vì deep learning yêu cầu
tính liên tiếp các đạo hàm
theo cùng một tập tham số
rất nhiều lần,
và chúng ta có thể có nguy cơ hết bộ nhớ.
Lưu ý rằng gradient của một hàm vô hướng
theo một vector $\mathbf{x}$
là một vector có
cùng hình dạng với $\mathbf{x}$.
:end_tab:

```{.python .input  n=8}
%%tab mxnet
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```

```{.python .input  n=9}
%%tab pytorch
# Can also create x = torch.arange(4.0, requires_grad=True)
x.requires_grad_(True)
x.grad  # The gradient is None by default
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(x)
```

(**Bây giờ chúng ta tính hàm của `x` và gán kết quả cho `y`.**)

```{.python .input  n=10}
%%tab mxnet
# Our code is inside an `autograd.record` scope to build the computational
# graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input  n=11}
%%tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
%%tab tensorflow
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

```{.python .input}
%%tab jax
y = lambda x: 2 * jnp.dot(x, x)
y(x)
```

:begin_tab:`mxnet`
[**Bây giờ chúng ta có thể lấy gradient của `y`
theo `x`**] bằng cách gọi
method `backward` của nó.
Tiếp theo, chúng ta có thể truy cập gradient
thông qua attribute `grad` của `x`.
:end_tab:

:begin_tab:`pytorch`
[**Bây giờ chúng ta có thể lấy gradient của `y`
theo `x`**] bằng cách gọi
method `backward` của nó.
Tiếp theo, chúng ta có thể truy cập gradient
thông qua attribute `grad` của `x`.
:end_tab:

:begin_tab:`tensorflow`
[**Bây giờ chúng ta có thể tính gradient của `y`
theo `x`**] bằng cách gọi
method `gradient`.
:end_tab:

:begin_tab:`jax`
[**Bây giờ chúng ta có thể lấy gradient của `y`
theo `x`**] bằng cách truyền qua
phép biến đổi `grad`.
:end_tab:

```{.python .input}
%%tab mxnet
y.backward()
x.grad
```

```{.python .input  n=12}
%%tab pytorch
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

```{.python .input}
%%tab jax
from jax import grad
# The `grad` transform returns a Python function that
# computes the gradient of the original function
x_grad = grad(y)(x)
x_grad
```

(**Chúng ta đã biết rằng gradient của hàm $y = 2\mathbf{x}^{\top}\mathbf{x}$
theo $\mathbf{x}$ phải là $4\mathbf{x}$.**)
Bây giờ chúng ta có thể kiểm tra rằng phép tính gradient tự động
và kết quả mong đợi là giống hệt nhau.

```{.python .input  n=13}
%%tab mxnet
x.grad == 4 * x
```

```{.python .input  n=14}
%%tab pytorch
x.grad == 4 * x
```

```{.python .input}
%%tab tensorflow
x_grad == 4 * x
```

```{.python .input}
%%tab jax
x_grad == 4 * x
```

:begin_tab:`mxnet`
[**Bây giờ hãy tính
một hàm khác của `x`
và lấy gradient của nó.**]
Lưu ý rằng MXNet đặt lại gradient buffer
mỗi khi chúng ta ghi nhận một gradient mới.
:end_tab:

:begin_tab:`pytorch`
[**Bây giờ hãy tính
một hàm khác của `x`
và lấy gradient của nó.**]
Lưu ý rằng PyTorch không tự động
đặt lại gradient buffer
khi chúng ta ghi nhận một gradient mới.
Thay vào đó, gradient mới
được cộng vào gradient đã lưu trước đó.
Hành vi này rất hữu ích
khi chúng ta muốn tối ưu hóa tổng
của nhiều hàm mục tiêu.
Để đặt lại gradient buffer,
chúng ta có thể gọi `x.grad.zero_()` như sau:
:end_tab:

:begin_tab:`tensorflow`
[**Bây giờ hãy tính
một hàm khác của `x`
và lấy gradient của nó.**]
Lưu ý rằng TensorFlow đặt lại gradient buffer
mỗi khi chúng ta ghi nhận một gradient mới.
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient
```

```{.python .input  n=20}
%%tab pytorch
x.grad.zero_()  # Reset the gradient
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

```{.python .input}
%%tab jax
y = lambda x: x.sum()
grad(y)(x)
```

## Backward cho các biến không vô hướng

Khi `y` là một vector,
cách biểu diễn tự nhiên nhất
của đạo hàm của `y`
theo một vector `x`
là một ma trận được gọi là *Jacobian*
chứa các đạo hàm riêng
của từng thành phần của `y`
theo từng thành phần của `x`.
Tương tự, với `y` và `x` bậc cao hơn,
kết quả của phép lấy đạo hàm có thể là một tensor
có bậc còn cao hơn.

Mặc dù Jacobian xuất hiện trong một số
kỹ thuật machine learning nâng cao,
thông thường hơn, chúng ta muốn cộng
gradient của từng thành phần của `y`
theo toàn bộ vector `x`,
tạo ra một vector có cùng hình dạng với `x`.
Ví dụ, chúng ta thường có một vector
biểu diễn giá trị của hàm mất mát
được tính riêng cho từng mẫu trong
một *batch* gồm các mẫu huấn luyện.
Ở đây, chúng ta chỉ muốn (**cộng các gradient
được tính riêng cho từng mẫu**).

:begin_tab:`mxnet`
MXNet xử lý vấn đề này bằng cách giảm tất cả tensor về vô hướng
bằng phép cộng trước khi tính gradient.
Nói cách khác, thay vì trả về Jacobian
$\partial_{\mathbf{x}} \mathbf{y}$,
nó trả về gradient của tổng
$\partial_{\mathbf{x}} \sum_i y_i$.
:end_tab:

:begin_tab:`pytorch`
Vì các framework deep learning khác nhau
trong cách diễn giải gradient của
các tensor không vô hướng,
PyTorch thực hiện một số bước để tránh nhầm lẫn.
Việc gọi `backward` trên một giá trị không vô hướng sẽ gây lỗi
trừ khi chúng ta cho PyTorch biết cách giảm đối tượng đó về một vô hướng.
Chính xác hơn, chúng ta cần cung cấp một vector $\mathbf{v}$
sao cho `backward` sẽ tính
$\mathbf{v}^\top \partial_{\mathbf{x}} \mathbf{y}$
thay vì $\partial_{\mathbf{x}} \mathbf{y}$.
Phần tiếp theo có thể gây khó hiểu,
nhưng vì những lý do sẽ trở nên rõ ràng sau này,
đối số này (đại diện cho $\mathbf{v}$) được đặt tên là `gradient`.
Để xem mô tả chi tiết hơn, hãy xem
[bài đăng trên Medium](https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29) của Yang Zhang.
:end_tab:

:begin_tab:`tensorflow`
Theo mặc định, TensorFlow trả về gradient của tổng.
Nói cách khác, thay vì trả về
Jacobian $\partial_{\mathbf{x}} \mathbf{y}$,
nó trả về gradient của tổng
$\partial_{\mathbf{x}} \sum_i y_i$.
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x  
y.backward()
x.grad  # Equals the gradient of y = sum(x * x)
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as y = tf.reduce_sum(x * x)
```

```{.python .input}
%%tab jax
y = lambda x: x * x
# grad is only defined for scalar output functions
grad(lambda x: y(x).sum())(x)
```

## Tách phép tính

Đôi khi, chúng ta muốn [**đưa một số phép tính
ra ngoài computational graph đã được ghi nhận.**]
Ví dụ, giả sử chúng ta sử dụng đầu vào
để tạo ra một số hạng trung gian phụ
mà chúng ta không muốn tính gradient.
Trong trường hợp này, chúng ta cần *detach*
computational graph tương ứng
khỏi kết quả cuối cùng.
Ví dụ minh họa đơn giản sau sẽ làm rõ hơn:
giả sử chúng ta có `z = x * y` và `y = x * x`
nhưng muốn tập trung vào ảnh hưởng *trực tiếp* của `x` lên `z`
thay vì ảnh hưởng truyền qua `y`.
Trong trường hợp này, chúng ta có thể tạo một biến mới `u`
có cùng giá trị với `y`
nhưng *nguồn gốc* của nó (cách nó được tạo ra)
đã bị xóa.
Do đó, `u` không có nút tổ tiên nào trong graph
và gradient không truyền qua `u` đến `x`.
Ví dụ, lấy gradient của `z = x * u`
sẽ cho kết quả `u`,
(không phải `3 * x * x` như bạn có thể
đã dự đoán vì `z = x * x * x`).

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input  n=21}
%%tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
%%tab tensorflow
# Set persistent=True to preserve the compute graph. 
# This lets us run t.gradient more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

```{.python .input}
%%tab jax
import jax

y = lambda x: x * x
# jax.lax primitives are Python wrappers around XLA operations
u = jax.lax.stop_gradient(y(x))
z = lambda x: u * x

grad(lambda x: z(x).sum())(x) == y(x)
```

Lưu ý rằng mặc dù quy trình này
tách các nút tổ tiên của `y`
khỏi graph dẫn đến `z`,
computational graph dẫn đến `y`
vẫn tồn tại, vì vậy chúng ta có thể tính
gradient của `y` theo `x`.

```{.python .input}
%%tab mxnet
y.backward()
x.grad == 2 * x
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
%%tab tensorflow
t.gradient(y, x) == 2 * x
```

```{.python .input}
%%tab jax
grad(lambda x: y(x).sum())(x) == 2 * x
```

## Gradient và luồng điều khiển Python

Cho đến nay, chúng ta đã xem xét các trường hợp trong đó đường đi từ đầu vào đến đầu ra được xác định rõ ràng thông qua một hàm như `z = x * x * x`.
Lập trình mang lại cho chúng ta nhiều tự do hơn trong cách tính toán kết quả.
Chẳng hạn, chúng ta có thể khiến chúng phụ thuộc vào các biến phụ hoặc điều kiện hóa các lựa chọn dựa trên những kết quả trung gian.
Một lợi ích của việc sử dụng automatic differentiation
là [**ngay cả khi**] việc xây dựng computational graph của
(**một hàm đòi hỏi phải đi qua mê cung của luồng điều khiển Python**)
(ví dụ: các câu lệnh điều kiện, vòng lặp và những lời gọi hàm tùy ý),
(**chúng ta vẫn có thể tính gradient của biến thu được.**)
Để minh họa điều này, hãy xem đoạn mã sau, trong đó
số lần lặp của vòng lặp `while`
và việc đánh giá câu lệnh `if`
đều phụ thuộc vào giá trị của đầu vào `a`.

```{.python .input}
%%tab mxnet
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab jax
def f(a):
    b = a * 2
    while jnp.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Bên dưới, chúng ta gọi hàm này và truyền vào một giá trị ngẫu nhiên làm đầu vào.
Vì đầu vào là một biến ngẫu nhiên,
chúng ta không biết computational graph sẽ có dạng nào.
Tuy nhiên, bất cứ khi nào thực thi `f(a)`
trên một đầu vào cụ thể, chúng ta sẽ tạo ra
một computational graph cụ thể
và sau đó có thể chạy `backward`.

```{.python .input}
%%tab mxnet
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
%%tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
%%tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

```{.python .input}
%%tab jax
from jax import random
a = random.normal(random.PRNGKey(1), ())
d = f(a)
d_grad = grad(f)(a)
```

Mặc dù hàm `f` của chúng ta, vì mục đích minh họa, có phần khiên cưỡng,
sự phụ thuộc của nó vào đầu vào khá đơn giản:
đó là một hàm *tuyến tính* của `a`
với hệ số tỷ lệ được xác định từng đoạn.
Do đó, `f(a) / a` là một vector gồm các phần tử hằng số
và hơn nữa, `f(a) / a` cần phải khớp với
gradient của `f(a)` theo `a`.

```{.python .input}
%%tab mxnet
a.grad == d / a
```

```{.python .input}
%%tab pytorch
a.grad == d / a
```

```{.python .input}
%%tab tensorflow
d_grad == d / a
```

```{.python .input}
%%tab jax
d_grad == d / a
```

Luồng điều khiển động rất phổ biến trong deep learning.
Chẳng hạn, khi xử lý văn bản, computational graph
phụ thuộc vào độ dài của đầu vào.
Trong những trường hợp này, automatic differentiation
trở nên thiết yếu đối với việc mô hình hóa thống kê
vì không thể tính gradient *a priori*.

## Thảo luận

Đến đây, bạn đã có một cái nhìn sơ bộ về sức mạnh của automatic differentiation.
Việc phát triển các thư viện để tính đạo hàm
một cách tự động và hiệu quả
đã thúc đẩy năng suất đáng kể
cho những người thực hành deep learning,
giúp họ tập trung vào những công việc ít mang tính thủ công hơn.
Hơn nữa, autograd cho phép chúng ta thiết kế những mô hình lớn
mà việc tính gradient bằng giấy bút
sẽ tốn thời gian đến mức không khả thi.
Điều thú vị là, trong khi chúng ta sử dụng autograd để *tối ưu hóa* các mô hình
(theo nghĩa thống kê),
việc *tối ưu hóa* chính các thư viện autograd
(theo nghĩa tính toán)
lại là một chủ đề phong phú
được các nhà thiết kế framework đặc biệt quan tâm.
Tại đây, các công cụ từ compiler và thao tác graph
được tận dụng để tính toán kết quả
theo cách nhanh chóng và tiết kiệm bộ nhớ nhất.

Hiện tại, hãy cố ghi nhớ những điều cơ bản sau: (i) gắn gradient cho những biến mà theo đó chúng ta muốn tính đạo hàm; (ii) ghi lại quá trình tính toán giá trị đích; (iii) thực thi hàm backpropagation; và (iv) truy cập gradient thu được.


## Bài tập

1. Tại sao đạo hàm bậc hai tốn nhiều chi phí tính toán hơn nhiều so với đạo hàm bậc nhất?
1. Sau khi chạy hàm để thực hiện backpropagation, hãy chạy lại ngay lập tức và xem điều gì xảy ra. Hãy tìm hiểu nguyên nhân.
1. Trong ví dụ về luồng điều khiển, nơi chúng ta tính đạo hàm của `d` theo `a`, điều gì sẽ xảy ra nếu chúng ta thay đổi biến `a` thành một vector hoặc ma trận ngẫu nhiên? Khi đó, kết quả của phép tính `f(a)` không còn là một scalar. Điều gì sẽ xảy ra với kết quả? Chúng ta phân tích điều này như thế nào?
1. Cho $f(x) = \sin(x)$. Hãy vẽ đồ thị của $f$ và đạo hàm $f'$. Không được tận dụng thực tế rằng $f'(x) = \cos(x)$, mà hãy sử dụng automatic differentiation để thu được kết quả.
1. Cho $f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$. Hãy viết dependency graph lần theo các kết quả từ $x$ đến $f(x)$.
1. Sử dụng chain rule để tính đạo hàm $\frac{df}{dx}$ của hàm nói trên, đặt từng số hạng lên dependency graph mà bạn đã xây dựng trước đó.
1. Với graph và các kết quả đạo hàm trung gian, bạn có một số lựa chọn khi tính gradient. Hãy đánh giá kết quả một lần, bắt đầu từ $x$ đến $f$, và một lần, bắt đầu từ $f$ rồi lần ngược về $x$. Đường đi từ $x$ đến $f$ thường được gọi là *forward differentiation*, trong khi đường đi từ $f$ đến $x$ được gọi là backward differentiation.
1. Khi nào bạn nên sử dụng forward differentiation, và khi nào nên sử dụng backward differentiation? Gợi ý: hãy cân nhắc lượng dữ liệu trung gian cần thiết, khả năng song song hóa các bước, cũng như kích thước của các ma trận và vector liên quan.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/200)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17970)
:end_tab: