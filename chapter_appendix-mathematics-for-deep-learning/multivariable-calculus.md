# Giải tích nhiều biến
:label:`sec_multivariable_calculus`

Giờ đây, khi đã hiểu khá rõ về đạo hàm của hàm một biến, hãy quay lại câu hỏi ban đầu, trong đó chúng ta xét một hàm mất mát có thể phụ thuộc vào hàng tỷ trọng số.

## Đạo hàm trong không gian nhiều chiều
Điều mà :numref:`sec_single_variable_calculus` cho chúng ta biết là nếu thay đổi chỉ một trong hàng tỷ trọng số này và giữ cố định tất cả các trọng số còn lại, chúng ta biết điều gì sẽ xảy ra! Đây không gì khác hơn là một hàm một biến, vì vậy ta có thể viết

$$L(w_1+\epsilon_1, w_2, \ldots, w_N) \approx L(w_1, w_2, \ldots, w_N) + \epsilon_1 \frac{d}{dw_1} L(w_1, w_2, \ldots, w_N).$$
:eqlabel:`eq_part_der`

Ta sẽ gọi đạo hàm theo một biến trong khi giữ cố định các biến còn lại là *đạo hàm riêng*, và sử dụng ký hiệu $\frac{\partial}{\partial w_1}$ cho đạo hàm trong :eqref:`eq_part_der`.

Bây giờ, hãy thay đổi $w_2$ một chút thành $w_2 + \epsilon_2$:

$$
\begin{aligned}
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N) & \approx L(w_1, w_2+\epsilon_2, \ldots, w_N) + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1\epsilon_2\frac{\partial}{\partial w_2}\frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N).
\end{aligned}
$$

Một lần nữa, ta đã sử dụng ý tưởng rằng $\epsilon_1\epsilon_2$ là một số hạng bậc cao hơn và có thể bỏ qua, tương tự như cách ta có thể bỏ qua $\epsilon^{2}$ trong phần trước, cùng với điều đã thấy trong :eqref:`eq_part_der`. Tiếp tục theo cách này, ta có thể viết

$$
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \approx L(w_1, w_2, \ldots, w_N) + \sum_i \epsilon_i \frac{\partial}{\partial w_i} L(w_1, w_2, \ldots, w_N).
$$

Biểu thức này có vẻ phức tạp, nhưng ta có thể làm cho nó trở nên quen thuộc hơn bằng cách nhận thấy rằng tổng ở vế phải chính xác có dạng của một tích vô hướng. Vì vậy, nếu đặt

$$
\boldsymbol{\epsilon} = [\epsilon_1, \ldots, \epsilon_N]^\top \; \textrm{and} \;
\nabla_{\mathbf{x}} L = \left[\frac{\partial L}{\partial x_1}, \ldots, \frac{\partial L}{\partial x_N}\right]^\top,
$$

thì

$$L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).$$
:eqlabel:`eq_nabla_use`

Ta sẽ gọi vector $\nabla_{\mathbf{w}} L$ là *gradient* của $L$.

Phương trình :eqref:`eq_nabla_use` đáng để chúng ta suy ngẫm một chút. Nó có đúng định dạng mà ta đã gặp trong trường hợp một chiều, chỉ khác là mọi thứ đã được chuyển thành các vector và tích vô hướng. Phương trình này cho phép ta ước lượng hàm $L$ sẽ thay đổi như thế nào trước bất kỳ nhiễu loạn nào của đầu vào. Như ta sẽ thấy trong phần tiếp theo, đây sẽ là một công cụ quan trọng giúp hiểu về mặt hình học cách chúng ta có thể học bằng thông tin chứa trong gradient.

Nhưng trước hết, hãy xem phép xấp xỉ này hoạt động như thế nào qua một ví dụ. Giả sử ta đang làm việc với hàm

$$
f(x, y) = \log(e^x + e^y) \textrm{ with gradient } \nabla f (x, y) = \left[\frac{e^x}{e^x+e^y}, \frac{e^y}{e^x+e^y}\right].
$$

Nếu xét một điểm như $(0, \log(2))$, ta thấy rằng

$$
f(x, y) = \log(3) \textrm{ with gradient } \nabla f (x, y) = \left[\frac{1}{3}, \frac{2}{3}\right].
$$

Vì vậy, nếu muốn xấp xỉ $f$ tại $(\epsilon_1, \log(2) + \epsilon_2)$, ta thấy rằng trường hợp cụ thể của :eqref:`eq_nabla_use` là:

$$
f(\epsilon_1, \log(2) + \epsilon_2) \approx \log(3) + \frac{1}{3}\epsilon_1 + \frac{2}{3}\epsilon_2.
$$

Ta có thể kiểm tra điều này bằng code để xem phép xấp xỉ chính xác đến đâu.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import autograd, np, npx
npx.set_np()

def f(x, y):
    return np.log(np.exp(x) + np.exp(y))
def grad_f(x, y):
    return np.array([np.exp(x) / (np.exp(x) + np.exp(y)),
                     np.exp(y) / (np.exp(x) + np.exp(y))])

epsilon = np.array([0.01, -0.03])
grad_approx = f(0, np.log(2)) + epsilon.dot(grad_f(0, np.log(2)))
true_value = f(0 + epsilon[0], np.log(2) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch
import numpy as np

def f(x, y):
    return torch.log(torch.exp(x) + torch.exp(y))
def grad_f(x, y):
    return torch.tensor([torch.exp(x) / (torch.exp(x) + torch.exp(y)),
                     torch.exp(y) / (torch.exp(x) + torch.exp(y))])

epsilon = torch.tensor([0.01, -0.03])
grad_approx = f(torch.tensor([0.]), torch.log(
    torch.tensor([2.]))) + epsilon.dot(
    grad_f(torch.tensor([0.]), torch.log(torch.tensor(2.))))
true_value = f(torch.tensor([0.]) + epsilon[0], torch.log(
    torch.tensor([2.])) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf
import numpy as np

def f(x, y):
    return tf.math.log(tf.exp(x) + tf.exp(y))
def grad_f(x, y):
    return tf.constant([(tf.exp(x) / (tf.exp(x) + tf.exp(y))).numpy(),
                        (tf.exp(y) / (tf.exp(x) + tf.exp(y))).numpy()])

epsilon = tf.constant([0.01, -0.03])
grad_approx = f(tf.constant([0.]), tf.math.log(
    tf.constant([2.]))) + tf.tensordot(
    epsilon, grad_f(tf.constant([0.]), tf.math.log(tf.constant(2.))), axes=1)
true_value = f(tf.constant([0.]) + epsilon[0], tf.math.log(
    tf.constant([2.])) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

## Hình học của Gradient và Gradient Descent
Hãy xét lại biểu thức trong :eqref:`eq_nabla_use`:

$$
L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).
$$

Giả sử ta muốn sử dụng biểu thức này để giúp cực tiểu hóa hàm mất mát $L$. Trước hết, hãy tìm hiểu về mặt hình học thuật toán gradient descent được mô tả lần đầu trong :numref:`sec_autograd`. Ta sẽ thực hiện như sau:

1. Bắt đầu với một lựa chọn ngẫu nhiên cho các tham số ban đầu $\mathbf{w}$.
2. Tìm hướng $\mathbf{v}$ khiến $L$ giảm nhanh nhất tại $\mathbf{w}$.
3. Tiến một bước nhỏ theo hướng đó: $\mathbf{w} \rightarrow \mathbf{w} + \epsilon\mathbf{v}$.
4. Lặp lại.

Điều duy nhất ta chưa biết chính xác cách thực hiện là tính vector $\mathbf{v}$ trong bước thứ hai. Ta sẽ gọi hướng như vậy là *hướng giảm dốc nhất*. Dựa trên cách hiểu hình học về tích vô hướng trong :numref:`sec_geometry-linear-algebraic-ops`, ta thấy có thể viết lại :eqref:`eq_nabla_use` như sau:

$$
L(\mathbf{w} + \mathbf{v}) \approx L(\mathbf{w}) + \mathbf{v}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}) = L(\mathbf{w}) + \|\nabla_{\mathbf{w}} L(\mathbf{w})\|\cos(\theta).
$$

Lưu ý rằng để thuận tiện, ta đã chọn hướng có độ dài bằng một, và dùng $\theta$ cho góc giữa $\mathbf{v}$ và $\nabla_{\mathbf{w}} L(\mathbf{w})$. Nếu muốn tìm hướng làm $L$ giảm nhanh nhất có thể, ta muốn biểu thức này có giá trị âm nhất có thể. Hướng được chọn chỉ xuất hiện trong phương trình này thông qua $\cos(\theta)$, vì vậy ta muốn cosine này có giá trị âm nhất có thể. Nhắc lại dạng của hàm cosine, ta có thể làm cho nó âm nhất bằng cách đặt $\cos(\theta) = -1$, hay tương đương với việc góc giữa gradient và hướng được chọn là $\pi$ radian, hoặc tương đương $180$ độ. Cách duy nhất để đạt được điều này là đi theo hướng ngược lại hoàn toàn: chọn $\mathbf{v}$ theo hướng ngược hẳn với $\nabla_{\mathbf{w}} L(\mathbf{w})$!

Điều này đưa chúng ta đến một trong những khái niệm toán học quan trọng nhất trong machine learning: hướng giảm dốc nhất trỏ theo hướng $-\nabla_{\mathbf{w}}L(\mathbf{w})$. Vì vậy, thuật toán mô tả không chính thức của chúng ta có thể được viết lại như sau.

1. Bắt đầu với một lựa chọn ngẫu nhiên cho các tham số ban đầu $\mathbf{w}$.
2. Tính $\nabla_{\mathbf{w}} L(\mathbf{w})$.
3. Tiến một bước nhỏ theo hướng ngược lại: $\mathbf{w} \leftarrow \mathbf{w} - \epsilon\nabla_{\mathbf{w}} L(\mathbf{w})$.
4. Lặp lại.


Thuật toán cơ bản này đã được nhiều nhà nghiên cứu sửa đổi và điều chỉnh theo nhiều cách, nhưng khái niệm cốt lõi vẫn giống nhau trong tất cả các biến thể. Sử dụng gradient để tìm hướng làm giảm hàm mất mát nhanh nhất có thể, rồi cập nhật các tham số để tiến một bước theo hướng đó.

## Lưu ý về Tối ưu hóa Toán học
Trong suốt cuốn sách này, chúng ta tập trung chủ yếu vào các kỹ thuật tối ưu hóa số vì một lý do thực tế: tất cả các hàm gặp trong bối cảnh deep learning đều quá phức tạp để có thể cực tiểu hóa một cách tường minh.

Tuy nhiên, việc xem xét cách hiểu hình học mà ta thu được ở trên cho chúng ta biết gì về việc tối ưu hóa trực tiếp các hàm là một bài tập hữu ích.

Giả sử ta muốn tìm giá trị của $\mathbf{x}_0$ làm cực tiểu một hàm nào đó $L(\mathbf{x})$. Hơn nữa, giả sử có người cho ta một giá trị và nói rằng đó là giá trị cực tiểu của $L$. Có điều gì ta có thể kiểm tra để xem câu trả lời của họ có hợp lý hay không?

Một lần nữa, hãy xét :eqref:`eq_nabla_use`:
$$
L(\mathbf{x}_0 + \boldsymbol{\epsilon}) \approx L(\mathbf{x}_0) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{x}} L(\mathbf{x}_0).
$$

Nếu gradient khác không, ta biết rằng có thể tiến một bước theo hướng $-\epsilon \nabla_{\mathbf{x}} L(\mathbf{x}_0)$ để tìm một giá trị của $L$ nhỏ hơn. Vì vậy, nếu thực sự đang ở điểm cực tiểu thì điều này không thể xảy ra! Ta có thể kết luận rằng nếu $\mathbf{x}_0$ là một điểm cực tiểu, thì $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$. Ta gọi các điểm thỏa mãn $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$ là *điểm tới hạn*.

Điều này rất hữu ích, bởi vì trong một số trường hợp hiếm gặp, ta *có thể* tìm tường minh tất cả các điểm tại đó gradient bằng không, rồi tìm điểm có giá trị nhỏ nhất.

Ví dụ cụ thể, hãy xét hàm
$$
f(x) = 3x^4 - 4x^3 -12x^2.
$$

Hàm này có đạo hàm
$$
\frac{df}{dx} = 12x^3 - 12x^2 -24x = 12x(x-2)(x+1).
$$

Các vị trí duy nhất có thể là điểm cực tiểu là $x = -1, 0, 2$, tại đó hàm nhận các giá trị lần lượt là $-5,0, -32$, vì vậy ta có thể kết luận rằng hàm đạt giá trị nhỏ nhất khi $x = 2$. Một đồ thị phác thảo nhanh cũng xác nhận điều này.

```{.python .input}
#@tab mxnet
x = np.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

Điều này làm nổi bật một thực tế quan trọng cần biết khi làm việc cả về mặt lý thuyết lẫn số: các điểm duy nhất mà tại đó ta có thể tối thiểu hóa (hoặc tối đa hóa) một hàm sẽ có gradient bằng không; tuy nhiên, không phải mọi điểm có gradient bằng không đều là cực tiểu (hoặc cực đại) *toàn cục* thực sự.

## Quy tắc chuỗi nhiều biến
Giả sử ta có một hàm của bốn biến ($w, x, y$ và $z$), được tạo ra bằng cách hợp thành nhiều số hạng:

$$\begin{aligned}f(u, v) & = (u+v)^{2} \\u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.\end{aligned}$$
:eqlabel:`eq_multi_func_def`

Những chuỗi phương trình như vậy thường xuất hiện khi làm việc với neural network, vì vậy việc hiểu cách tính gradient của các hàm này là rất quan trọng. Ta có thể bắt đầu nhận thấy những gợi ý trực quan về mối liên hệ này trong :numref:`fig_chain-1` nếu xem xét những biến nào liên hệ trực tiếp với nhau.

![Các quan hệ hàm ở trên, trong đó các nút biểu diễn các giá trị và các cạnh biểu diễn sự phụ thuộc hàm.](../img/chain-net1.svg)
:label:`fig_chain-1`

Không có gì ngăn cản ta hợp thành mọi thứ từ :eqref:`eq_multi_func_def` và viết ra

$$
f(w, x, y, z) = \left(\left((w+x+y+z)^2+(w+x-y-z)^2\right)^2+\left((w+x+y+z)^2-(w+x-y-z)^2\right)^2\right)^2.
$$

Sau đó, ta có thể lấy đạo hàm chỉ bằng cách sử dụng các đạo hàm một biến, nhưng nếu làm vậy, ta sẽ nhanh chóng bị ngập trong các số hạng, trong đó có nhiều số hạng lặp lại! Thật vậy, có thể thấy rằng, chẳng hạn:

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = 2 \left(2 \left(2 (w + x + y + z) - 2 (w + x - y - z)\right) \left((w + x + y + z)^{2}- (w + x - y - z)^{2}\right) + \right.\\
& \left. \quad 2 \left(2 (w + x - y - z) + 2 (w + x + y + z)\right) \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)\right) \times \\
& \quad \left(\left((w + x + y + z)^{2}- (w + x - y - z)^2\right)^{2}+ \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)^{2}\right).
\end{aligned}
$$

Nếu sau đó ta cũng muốn tính $\frac{\partial f}{\partial x}$, ta sẽ lại thu được một phương trình tương tự với nhiều số hạng lặp lại, cũng như nhiều số hạng lặp lại *chung* giữa hai đạo hàm. Điều này thể hiện một lượng công việc bị lãng phí khổng lồ, và nếu phải tính đạo hàm theo cách này thì toàn bộ cuộc cách mạng deep learning đã dừng lại trước khi bắt đầu!

Hãy chia nhỏ bài toán. Trước hết, ta sẽ cố gắng hiểu $f$ thay đổi như thế nào khi ta thay đổi $a$, về cơ bản giả sử rằng $w, x, y$ và $z$ đều không tồn tại. Ta sẽ lập luận như khi lần đầu làm việc với gradient. Hãy lấy $a$ và cộng thêm một lượng nhỏ $\epsilon$ vào đó.

$$
\begin{aligned}
& f(u(a+\epsilon, b), v(a+\epsilon, b)) \\
\approx & f\left(u(a, b) + \epsilon\frac{\partial u}{\partial a}(a, b), v(a, b) + \epsilon\frac{\partial v}{\partial a}(a, b)\right) \\
\approx & f(u(a, b), v(a, b)) + \epsilon\left[\frac{\partial f}{\partial u}(u(a, b), v(a, b))\frac{\partial u}{\partial a}(a, b) + \frac{\partial f}{\partial v}(u(a, b), v(a, b))\frac{\partial v}{\partial a}(a, b)\right].
\end{aligned}
$$

Dòng đầu tiên xuất phát từ định nghĩa của đạo hàm riêng, còn dòng thứ hai xuất phát từ định nghĩa của gradient. Việc theo dõi chính xác nơi ta tính mỗi đạo hàm, như trong biểu thức $\frac{\partial f}{\partial u}(u(a, b), v(a, b))$, gây khó khăn về mặt ký hiệu, vì vậy ta thường viết tắt thành biểu thức dễ nhớ hơn nhiều sau đây:

$$
\frac{\partial f}{\partial a} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}.
$$

Việc suy nghĩ về ý nghĩa của quá trình này rất hữu ích. Ta đang cố gắng hiểu một hàm có dạng $f(u(a, b), v(a, b))$ thay đổi giá trị như thế nào khi $a$ thay đổi. Có hai con đường mà điều này có thể xảy ra: con đường $a \rightarrow u \rightarrow f$ và con đường $a \rightarrow v \rightarrow f$. Ta có thể tính cả hai đóng góp này thông qua quy tắc chuỗi: $\frac{\partial w}{\partial u} \cdot \frac{\partial u}{\partial x}$ và $\frac{\partial w}{\partial v} \cdot \frac{\partial v}{\partial x}$ tương ứng, rồi cộng chúng lại.

Hãy tưởng tượng ta có một mạng lưới hàm khác, trong đó các hàm bên phải phụ thuộc vào những hàm được nối với chúng ở bên trái, như minh họa trong :numref:`fig_chain-2`.

![Một ví dụ tinh tế hơn về quy tắc chuỗi.](../img/chain-net2.svg)
:label:`fig_chain-2`

Để tính một biểu thức như $\frac{\partial f}{\partial y}$, ta cần tính tổng trên tất cả (trong trường hợp này là $3$) đường đi từ $y$ đến $f$, cho

$$
\frac{\partial f}{\partial y} = \frac{\partial f}{\partial a} \frac{\partial a}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial b} \frac{\partial b}{\partial v} \frac{\partial v}{\partial y}.
$$

Hiểu quy tắc chuỗi theo cách này sẽ mang lại lợi ích lớn khi cố gắng hiểu cách gradient lan truyền qua các mạng, cũng như lý do những lựa chọn kiến trúc khác nhau, chẳng hạn như trong LSTM (:numref:`sec_lstm`) hoặc các lớp residual (:numref:`sec_resnet`), có thể giúp định hình quá trình học bằng cách kiểm soát luồng gradient.

## Thuật toán lan truyền ngược

Hãy quay lại ví dụ trong :eqref:`eq_multi_func_def` của phần trước, trong đó

$$
\begin{aligned}
f(u, v) & = (u+v)^{2} \\
u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\
a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.
\end{aligned}
$$

Nếu muốn tính, chẳng hạn, $\frac{\partial f}{\partial w}$, ta có thể áp dụng quy tắc chuỗi nhiều biến để thấy rằng:

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial w} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial w}, \\
\frac{\partial u}{\partial w} & = \frac{\partial u}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial u}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial v}{\partial w} & = \frac{\partial v}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial v}{\partial b}\frac{\partial b}{\partial w}.
\end{aligned}
$$

Hãy thử sử dụng phép phân rã này để tính $\frac{\partial f}{\partial w}$. Lưu ý rằng ở đây ta chỉ cần các đạo hàm riêng từng bước:

$$
\begin{aligned}
\frac{\partial f}{\partial u} = 2(u+v), & \quad\frac{\partial f}{\partial v} = 2(u+v), \\
\frac{\partial u}{\partial a} = 2(a+b), & \quad\frac{\partial u}{\partial b} = 2(a+b), \\
\frac{\partial v}{\partial a} = 2(a-b), & \quad\frac{\partial v}{\partial b} = -2(a-b), \\
\frac{\partial a}{\partial w} = 2(w+x+y+z), & \quad\frac{\partial b}{\partial w} = 2(w+x-y-z).
\end{aligned}
$$

Nếu viết điều này thành code, ta sẽ thu được một biểu thức khá dễ quản lý.

```{.python .input}
#@tab all
# Compute the value of the function from inputs to outputs
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'    f at {w}, {x}, {y}, {z} is {f}')

# Compute the single step partials
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)

# Compute the final result from inputs to outputs
du_dw, dv_dw = du_da*da_dw + du_db*db_dw, dv_da*da_dw + dv_db*db_dw
df_dw = df_du*du_dw + df_dv*dv_dw
print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
```

Tuy nhiên, lưu ý rằng điều này vẫn không giúp việc tính một biểu thức như $\frac{\partial f}{\partial x}$ trở nên dễ dàng. Lý do nằm ở *cách* ta chọn áp dụng quy tắc chuỗi. Nếu xem xét những gì đã làm ở trên, ta luôn giữ $\partial w$ ở mẫu số khi có thể. Theo cách này, ta đã chọn áp dụng quy tắc chuỗi bằng cách xem $w$ thay đổi mọi biến khác như thế nào. Nếu đó là điều ta muốn, đây sẽ là một ý tưởng hay. Tuy nhiên, hãy nhớ lại động lực từ deep learning: ta muốn xem mỗi tham số thay đổi *loss* như thế nào. Về bản chất, ta muốn áp dụng quy tắc chuỗi bằng cách giữ $\partial f$ ở tử số bất cứ khi nào có thể!

Nói rõ hơn, lưu ý rằng ta có thể viết

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial w} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial f}{\partial a} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}, \\
\frac{\partial f}{\partial b} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial b}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial b}.
\end{aligned}
$$

Lưu ý rằng việc áp dụng quy tắc chain rule này yêu cầu chúng ta tính một cách tường minh $\frac{\partial f}{\partial u}, \frac{\partial f}{\partial v}, \frac{\partial f}{\partial a}, \frac{\partial f}{\partial b}, \; \textrm{and} \; \frac{\partial f}{\partial w}$. Không có gì ngăn cản chúng ta đưa thêm các phương trình sau:

$$
\begin{aligned}
\frac{\partial f}{\partial x} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial x} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial x}, \\
\frac{\partial f}{\partial y} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial y}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial y}, \\
\frac{\partial f}{\partial z} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial z}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial z}.
\end{aligned}
$$

và sau đó theo dõi cách $f$ thay đổi khi chúng ta thay đổi *bất kỳ* node nào trong toàn bộ network. Hãy triển khai điều này.

```{.python .input}
#@tab all
# Compute the value of the function from inputs to outputs
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'f at {w}, {x}, {y}, {z} is {f}')

# Compute the derivative using the decomposition above
# First compute the single step partials
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)
da_dx, db_dx = 2*(w + x + y + z), 2*(w + x - y - z)
da_dy, db_dy = 2*(w + x + y + z), -2*(w + x - y - z)
da_dz, db_dz = 2*(w + x + y + z), -2*(w + x - y - z)

# Now compute how f changes when we change any value from output to input
df_da, df_db = df_du*du_da + df_dv*dv_da, df_du*du_db + df_dv*dv_db
df_dw, df_dx = df_da*da_dw + df_db*db_dw, df_da*da_dx + df_db*db_dx
df_dy, df_dz = df_da*da_dy + df_db*db_dy, df_da*da_dz + df_db*db_dz

print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
print(f'df/dx at {w}, {x}, {y}, {z} is {df_dx}')
print(f'df/dy at {w}, {x}, {y}, {z} is {df_dy}')
print(f'df/dz at {w}, {x}, {y}, {z} is {df_dz}')
```

Việc chúng ta tính các đạo hàm từ $f$ ngược về phía các đầu vào, thay vì từ các đầu vào tiến về phía các đầu ra (như trong đoạn code đầu tiên ở trên), chính là lý do thuật toán này có tên là *backpropagation*. Lưu ý rằng có hai bước:
1. Tính giá trị của hàm và các đạo hàm riêng tại từng bước từ đầu đến cuối. Mặc dù chưa được thực hiện ở trên, việc này có thể được kết hợp thành một *forward pass* duy nhất.
2. Tính gradient của $f$ từ cuối về đầu. Chúng ta gọi đây là *backwards pass*.

Đây chính xác là điều mọi thuật toán deep learning đều triển khai để cho phép tính gradient của loss đối với mọi weight trong network chỉ trong một pass. Thật đáng kinh ngạc khi chúng ta có được một phép phân rã như vậy.

Để xem cách đóng gói điều này, hãy cùng xem nhanh ví dụ sau.

```{.python .input}
#@tab mxnet
# Initialize as ndarrays, then attach gradients
w, x, y, z = np.array(-1), np.array(0), np.array(-2), np.array(1)

w.attach_grad()
x.attach_grad()
y.attach_grad()
z.attach_grad()

# Do the computation like usual, tracking gradients
with autograd.record():
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Execute backward pass
f.backward()

print(f'df/dw at {w}, {x}, {y}, {z} is {w.grad}')
print(f'df/dx at {w}, {x}, {y}, {z} is {x.grad}')
print(f'df/dy at {w}, {x}, {y}, {z} is {y.grad}')
print(f'df/dz at {w}, {x}, {y}, {z} is {z.grad}')
```

```{.python .input}
#@tab pytorch
# Initialize as ndarrays, then attach gradients
w = torch.tensor([-1.], requires_grad=True)
x = torch.tensor([0.], requires_grad=True)
y = torch.tensor([-2.], requires_grad=True)
z = torch.tensor([1.], requires_grad=True)
# Do the computation like usual, tracking gradients
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2

# Execute backward pass
f.backward()

print(f'df/dw at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {w.grad.data.item()}')
print(f'df/dx at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {x.grad.data.item()}')
print(f'df/dy at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {y.grad.data.item()}')
print(f'df/dz at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {z.grad.data.item()}')
```

```{.python .input}
#@tab tensorflow
# Initialize as ndarrays, then attach gradients
w = tf.Variable(tf.constant([-1.]))
x = tf.Variable(tf.constant([0.]))
y = tf.Variable(tf.constant([-2.]))
z = tf.Variable(tf.constant([1.]))
# Do the computation like usual, tracking gradients
with tf.GradientTape(persistent=True) as t:
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Execute backward pass
w_grad = t.gradient(f, w).numpy()
x_grad = t.gradient(f, x).numpy()
y_grad = t.gradient(f, y).numpy()
z_grad = t.gradient(f, z).numpy()

print(f'df/dw at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {w_grad}')
print(f'df/dx at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {x_grad}')
print(f'df/dy at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {y_grad}')
print(f'df/dz at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {z_grad}')
```

Tất cả những gì chúng ta đã thực hiện ở trên có thể được tự động hóa bằng cách gọi `f.backwards()`.


## Hessians
Cũng như trong giải tích một biến, việc xem xét các đạo hàm bậc cao hơn rất hữu ích để hiểu cách thu được một xấp xỉ tốt hơn cho một hàm so với chỉ sử dụng gradient.

Có một vấn đề ngay lập tức xuất hiện khi làm việc với các đạo hàm bậc cao của hàm nhiều biến: số lượng các đạo hàm này rất lớn. Nếu có hàm $f(x_1, \ldots, x_n)$ gồm $n$ biến, thì chúng ta có thể lấy tổng cộng $n^{2}$ đạo hàm bậc hai, cụ thể là với mọi lựa chọn của $i$ và $j$:

$$
\frac{d^2f}{dx_idx_j} = \frac{d}{dx_i}\left(\frac{d}{dx_j}f\right).
$$

Theo truyền thống, các đạo hàm này được sắp xếp thành một ma trận gọi là *Hessian*:

$$\mathbf{H}_f = \begin{bmatrix} \frac{d^2f}{dx_1dx_1} & \cdots & \frac{d^2f}{dx_1dx_n} \\ \vdots & \ddots & \vdots \\ \frac{d^2f}{dx_ndx_1} & \cdots & \frac{d^2f}{dx_ndx_n} \\ \end{bmatrix}.$$
:eqlabel:`eq_hess_def`

Không phải mọi phần tử của ma trận này đều độc lập. Thật vậy, ta có thể chứng minh rằng miễn là cả hai *đạo hàm riêng hỗn hợp* (đạo hàm riêng theo nhiều hơn một biến) tồn tại và liên tục, thì với mọi $i$ và $j$,

$$
\frac{d^2f}{dx_idx_j} = \frac{d^2f}{dx_jdx_i}.
$$

Điều này có được bằng cách trước hết xét việc nhiễu một hàm theo hướng của $x_i$, sau đó nhiễu theo $x_j$, rồi so sánh kết quả với trường hợp trước hết nhiễu theo $x_j$, sau đó theo $x_i$, dựa trên thực tế rằng cả hai thứ tự này đều dẫn đến cùng một thay đổi cuối cùng trong đầu ra của $f$.

Cũng như với hàm một biến, chúng ta có thể dùng các đạo hàm này để hiểu rõ hơn nhiều về cách hàm hoạt động lân cận một điểm. Cụ thể, chúng ta có thể dùng chúng để tìm đa thức bậc hai khớp tốt nhất lân cận một điểm $\mathbf{x}_0$, như đã thấy trong trường hợp một biến.

Hãy xem một ví dụ. Giả sử $f(x_1, x_2) = a + b_1x_1 + b_2x_2 + c_{11}x_1^{2} + c_{12}x_1x_2 + c_{22}x_2^{2}$. Đây là dạng tổng quát của một đa thức bậc hai theo hai biến. Nếu xét giá trị của hàm, gradient và Hessian :eqref:`eq_hess_def` của nó, tất cả đều tại điểm zero:

$$
\begin{aligned}
f(0,0) & = a, \\
\nabla f (0,0) & = \begin{bmatrix}b_1 \\ b_2\end{bmatrix}, \\
\mathbf{H} f (0,0) & = \begin{bmatrix}2 c_{11} & c_{12} \\ c_{12} & 2c_{22}\end{bmatrix},
\end{aligned}
$$

chúng ta có thể khôi phục đa thức ban đầu bằng cách viết

$$
f(\mathbf{x}) = f(0) + \nabla f (0) \cdot \mathbf{x} + \frac{1}{2}\mathbf{x}^\top \mathbf{H} f (0) \mathbf{x}.
$$

Nói chung, nếu tính khai triển này tại một điểm bất kỳ $\mathbf{x}_0$, ta thấy rằng

$$
f(\mathbf{x}) = f(\mathbf{x}_0) + \nabla f (\mathbf{x}_0) \cdot (\mathbf{x}-\mathbf{x}_0) + \frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H} f (\mathbf{x}_0) (\mathbf{x}-\mathbf{x}_0).
$$

Điều này đúng với đầu vào có số chiều bất kỳ và cung cấp đa thức bậc hai xấp xỉ tốt nhất cho mọi hàm tại một điểm. Để làm ví dụ, hãy vẽ đồ thị của hàm

$$
f(x, y) = xe^{-x^2-y^2}.
$$

Có thể tính được gradient và Hessian là
$$
\nabla f(x, y) = e^{-x^2-y^2}\begin{pmatrix}1-2x^2 \\ -2xy\end{pmatrix} \; \textrm{and} \; \mathbf{H}f(x, y) = e^{-x^2-y^2}\begin{pmatrix} 4x^3 - 6x & 4x^2y - 2y \\ 4x^2y-2y &4xy^2-2x\end{pmatrix}.
$$

Do đó, sau một vài phép biến đổi đại số, ta thấy đa thức bậc hai xấp xỉ tại $[-1,0]^\top$ là

$$
f(x, y) \approx e^{-1}\left(-1 - (x+1) +(x+1)^2+y^2\right).
$$

```{.python .input}
#@tab mxnet
# Construct grid and compute function
x, y = np.meshgrid(np.linspace(-2, 2, 101),
                   np.linspace(-2, 2, 101), indexing='ij')
z = x*np.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = np.exp(-1)*(-1 - (x + 1) + (x + 1)**2 + y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), w.asnumpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Construct grid and compute function
x, y = torch.meshgrid(torch.linspace(-2, 2, 101),
                   torch.linspace(-2, 2, 101))

z = x*torch.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = torch.exp(torch.tensor([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# Construct grid and compute function
x, y = tf.meshgrid(tf.linspace(-2., 2., 101),
                   tf.linspace(-2., 2., 101))

z = x*tf.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = tf.exp(tf.constant([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

Đây là cơ sở cho Newton's Algorithm được thảo luận trong :numref:`sec_gd`, trong đó chúng ta thực hiện numerical optimization bằng cách lặp đi lặp lại việc tìm đa thức bậc hai khớp tốt nhất, rồi chính xác tìm cực tiểu của đa thức bậc hai đó.

## Một chút về phép tính vi phân ma trận
Các đạo hàm của những hàm liên quan đến ma trận thường đặc biệt gọn gàng. Phần này có thể chứa nhiều ký hiệu và trở nên nặng nề, vì vậy có thể bỏ qua trong lần đọc đầu tiên; tuy nhiên, việc biết rằng đạo hàm của các hàm liên quan đến những phép toán ma trận phổ biến thường đơn giản hơn nhiều so với dự đoán ban đầu là rất hữu ích, đặc biệt vì các phép toán ma trận giữ vai trò trung tâm trong các ứng dụng deep learning.

Hãy bắt đầu bằng một ví dụ. Giả sử chúng ta có một vector cột cố định $\boldsymbol{\beta}$ và muốn xét hàm tích $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$, đồng thời tìm hiểu tích vô hướng thay đổi như thế nào khi thay đổi $\mathbf{x}$.

Một khái niệm ký hiệu hữu ích khi làm việc với đạo hàm ma trận trong ML được gọi là *denominator layout matrix derivative*, trong đó chúng ta sắp xếp các đạo hàm riêng theo hình dạng của vector, ma trận hoặc tensor nằm ở mẫu số của vi phân. Trong trường hợp này, ta sẽ viết

$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix},
$$

trong đó chúng ta giữ nguyên hình dạng của vector cột $\mathbf{x}$.

Nếu viết hàm theo từng thành phần, ta có

$$
f(\mathbf{x}) = \sum_{i = 1}^{n} \beta_ix_i = \beta_1x_1 + \cdots + \beta_nx_n.
$$

Nếu bây giờ lấy đạo hàm riêng theo, chẳng hạn, $\beta_1$, hãy lưu ý rằng mọi thành phần đều bằng zero ngoại trừ số hạng đầu tiên, vốn chỉ là $x_1$ nhân với $\beta_1$, do đó ta thu được

$$
\frac{df}{dx_1} = \beta_1,
$$

hay tổng quát hơn rằng

$$
\frac{df}{dx_i} = \beta_i.
$$

Bây giờ chúng ta có thể ghép lại thành một ma trận để thấy

$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix} = \begin{bmatrix}
\beta_1 \\
\vdots \\
\beta_n
\end{bmatrix} = \boldsymbol{\beta}.
$$

Điều này minh họa một vài đặc điểm của phép tính ma trận mà chúng ta sẽ thường gặp trong phần này:

* Thứ nhất, các phép tính sẽ trở nên khá phức tạp.
* Thứ hai, kết quả cuối cùng gọn gàng hơn nhiều so với quá trình trung gian và sẽ luôn có dạng tương tự như trường hợp một biến.  Trong trường hợp này, hãy lưu ý rằng $\frac{d}{dx}(bx) = b$ và $\frac{d}{d\mathbf{x}} (\boldsymbol{\beta}^\top\mathbf{x}) = \boldsymbol{\beta}$ đều tương tự nhau.
* Thứ ba, các phép chuyển vị thường có thể xuất hiện như thể từ hư không.  Lý do cốt lõi là quy ước khớp với hình dạng của mẫu số; do đó, khi nhân các ma trận, chúng ta sẽ cần lấy phép chuyển vị để khớp lại với hình dạng của biểu thức ban đầu.

Để tiếp tục xây dựng trực giác, hãy thử một phép tính khó hơn một chút.  Giả sử chúng ta có một vector cột $\mathbf{x}$ và một ma trận vuông $A$, đồng thời muốn tính

$$\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}).$$
:eqlabel:`eq_mat_goal_1`

Để hướng tới một ký hiệu dễ thao tác hơn, hãy xét bài toán này bằng ký hiệu Einstein.  Trong trường hợp này, chúng ta có thể viết hàm dưới dạng

$$
\mathbf{x}^\top A \mathbf{x} = x_ia_{ij}x_j.
$$

Để tính đạo hàm, chúng ta cần biết, với mọi $k$, giá trị của

$$
\frac{d}{dx_k}(\mathbf{x}^\top A \mathbf{x}) = \frac{d}{dx_k}x_ia_{ij}x_j.
$$

Theo quy tắc tích, biểu thức này là

$$
\frac{d}{dx_k}x_ia_{ij}x_j = \frac{dx_i}{dx_k}a_{ij}x_j + x_ia_{ij}\frac{dx_j}{dx_k}.
$$

Với một số hạng như $\frac{dx_i}{dx_k}$, không khó để thấy rằng nó bằng một khi $i=k$ và bằng không trong các trường hợp khác.  Điều này có nghĩa là mọi số hạng mà $i$ và $k$ khác nhau sẽ triệt tiêu khỏi tổng này, vì vậy các số hạng duy nhất còn lại trong tổng thứ nhất là những số hạng có $i=k$.  Lập luận tương tự cũng đúng với số hạng thứ hai, trong đó chúng ta cần $j=k$.  Do đó,

$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{kj}x_j + x_ia_{ik}.
$$

Bây giờ, tên của các chỉ số trong ký hiệu Einstein là tùy ý---việc $i$ và $j$ khác nhau không còn quan trọng đối với phép tính này, vì vậy chúng ta có thể đánh lại chỉ số để cả hai cùng sử dụng $i$, từ đó có

$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{ki}x_i + x_ia_{ik} = (a_{ki} + a_{ik})x_i.
$$

Đây là lúc chúng ta bắt đầu cần một chút thực hành để tiếp tục.  Hãy thử nhận diện kết quả này dưới dạng các phép toán ma trận.  $a_{ki} + a_{ik}$ là thành phần thứ $k, i$ của $\mathbf{A} + \mathbf{A}^\top$.  Do đó,

$$
\frac{d}{dx_k}x_ia_{ij}x_j = [\mathbf{A} + \mathbf{A}^\top]_{ki}x_i.
$$

Tương tự, số hạng này hiện là tích của ma trận $\mathbf{A} + \mathbf{A}^\top$ với vector $\mathbf{x}$, vì vậy chúng ta thấy rằng

$$
\left[\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x})\right]_k = \frac{d}{dx_k}x_ia_{ij}x_j = [(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}]_k.
$$

Như vậy, ta thấy phần tử thứ $k$ của đạo hàm cần tìm trong :eqref:`eq_mat_goal_1` chính là phần tử thứ $k$ của vector ở vế phải, nên hai biểu thức này bằng nhau.  Do đó,

$$
\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}.
$$

Phép tính này đòi hỏi nhiều công sức hơn đáng kể so với phép tính trước, nhưng kết quả cuối cùng lại rất ngắn gọn.  Hơn nữa, hãy xét phép tính sau đây đối với đạo hàm thông thường của một biến:

$$
\frac{d}{dx}(xax) = \frac{dx}{dx}ax + xa\frac{dx}{dx} = (a+a)x.
$$

Tương đương, $\frac{d}{dx}(ax^2) = 2ax = (a+a)x$.  Một lần nữa, chúng ta nhận được một kết quả trông khá giống với kết quả của trường hợp một biến, nhưng có thêm một phép chuyển vị.

Đến đây, quy luật này có vẻ khá đáng ngờ, vì vậy hãy thử tìm hiểu lý do.  Khi tính các đạo hàm ma trận như thế này, trước hết hãy giả sử rằng biểu thức nhận được sẽ là một biểu thức ma trận khác: một biểu thức có thể được viết dưới dạng tích và tổng của các ma trận cùng với các phép chuyển vị của chúng.  Nếu tồn tại một biểu thức như vậy, nó phải đúng với mọi ma trận.  Đặc biệt, nó phải đúng với các ma trận kích thước $1 \times 1$, trong trường hợp đó tích ma trận chỉ là tích của các số, tổng ma trận chỉ là tổng, còn phép chuyển vị hoàn toàn không làm thay đổi gì!  Nói cách khác, bất kỳ biểu thức nào chúng ta nhận được *phải* khớp với biểu thức một biến.  Điều này có nghĩa là, sau một thời gian thực hành, chúng ta thường có thể đoán các đạo hàm ma trận chỉ bằng cách biết biểu thức một biến tương ứng phải có dạng như thế nào!

Hãy thử áp dụng điều này.  Giả sử $\mathbf{X}$ là một ma trận $n \times m$, $\mathbf{U}$ là một ma trận $n \times r$ và $\mathbf{V}$ là một ma trận $r \times m$. Hãy thử tính

$$\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2} = \;?$$
:eqlabel:`eq_mat_goal_2`

Phép tính này quan trọng trong một lĩnh vực có tên là phân tích nhân tử ma trận.  Tuy nhiên, đối với chúng ta, đây chỉ là một đạo hàm cần tính.  Hãy thử hình dung biểu thức này trong trường hợp các ma trận có kích thước $1\times1$.  Khi đó, chúng ta nhận được biểu thức

$$
\frac{d}{dv} (x-uv)^{2}= -2(x-uv)u,
$$

trong đó đạo hàm khá tiêu chuẩn.  Nếu cố chuyển biểu thức này trở lại dạng ma trận, chúng ta nhận được

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2(\mathbf{X} - \mathbf{U}\mathbf{V})\mathbf{U}.
$$

Tuy nhiên, khi xem xét kỹ, biểu thức này không hoàn toàn đúng.  Hãy nhớ rằng $\mathbf{X}$ có kích thước $n \times m$, cũng như $\mathbf{U}\mathbf{V}$, vì vậy ma trận $2(\mathbf{X} - \mathbf{U}\mathbf{V})$ có kích thước $n \times m$.  Mặt khác, $\mathbf{U}$ có kích thước $n \times r$, và chúng ta không thể nhân một ma trận $n \times m$ với một ma trận $n \times r$ vì các kích thước không khớp!

Chúng ta muốn có $\frac{d}{d\mathbf{V}}$, có cùng hình dạng với $\mathbf{V}$, tức là $r \times m$.  Vì vậy, bằng cách nào đó chúng ta cần lấy một ma trận $n \times m$ và một ma trận $n \times r$, nhân chúng với nhau (có thể kèm theo một số phép chuyển vị) để thu được ma trận $r \times m$.  Ta có thể thực hiện điều này bằng cách nhân $U^\top$ với $(\mathbf{X} - \mathbf{U}\mathbf{V})$.  Do đó, chúng ta có thể đoán nghiệm của :eqref:`eq_mat_goal_2` là

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$

Để chứng minh điều này đúng, chúng ta không thể bỏ qua việc trình bày một phép tính chi tiết.  Nếu đã tin rằng quy tắc kinh nghiệm này đúng, bạn có thể bỏ qua phần suy diễn sau đây.  Để tính

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^2,
$$

chúng ta phải tìm, với mọi $a$ và $b$,

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \frac{d}{dv_{ab}} \sum_{i, j}\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)^2.
$$

Nhắc lại rằng mọi phần tử của $\mathbf{X}$ và $\mathbf{U}$ đều là hằng số khi xét $\frac{d}{dv_{ab}}$, ta có thể đưa đạo hàm vào bên trong tổng và áp dụng chain rule cho bình phương để nhận được

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \sum_{i, j}2\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)\left(-\sum_k u_{ik}\frac{dv_{kj}}{dv_{ab}} \right).
$$

Như trong phép suy ra trước, ta có thể lưu ý rằng $\frac{dv_{kj}}{dv_{ab}}$ chỉ khác không khi $k=a$ và $j=b$. Nếu một trong hai điều kiện này không được thỏa mãn, số hạng trong tổng bằng không, và ta có thể tự do loại bỏ nó. Ta thấy rằng

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}\left(x_{ib} - \sum_k u_{ik}v_{kb}\right)u_{ia}.
$$

Một điểm tinh tế quan trọng ở đây là yêu cầu $k=a$ không xuất hiện bên trong tổng bên trong, vì $k$ đó là một biến giả mà ta đang lấy tổng trong số hạng bên trong. Để có một ví dụ dễ ký hiệu hơn, hãy xem xét tại sao

$$
\frac{d}{dx_1} \left(\sum_i x_i \right)^{2}= 2\left(\sum_i x_i \right).
$$

Từ đây, ta có thể bắt đầu xác định các thành phần của tổng. Trước hết,

$$
\sum_k u_{ik}v_{kb} = [\mathbf{U}\mathbf{V}]_{ib}.
$$

Vì vậy, toàn bộ biểu thức bên trong tổng là

$$
x_{ib} - \sum_k u_{ik}v_{kb} = [\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$

Điều này có nghĩa là giờ đây ta có thể viết đạo hàm của mình là

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}u_{ia}.
$$

Ta muốn biểu thức này có dạng phần tử $a, b$ của một ma trận để có thể sử dụng kỹ thuật như trong ví dụ trước và thu được một biểu thức ma trận. Điều đó có nghĩa là ta cần đổi thứ tự các chỉ số của $u_{ia}$. Nếu nhận thấy rằng $u_{ia} = [\mathbf{U}^\top]_{ai}$, ta có thể viết

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i} [\mathbf{U}^\top]_{ai}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$

Đây là một phép nhân ma trận, do đó ta có thể kết luận rằng

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2[\mathbf{U}^\top(\mathbf{X}-\mathbf{U}\mathbf{V})]_{ab}.
$$

và do đó ta có thể viết nghiệm của :eqref:`eq_mat_goal_2`

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$

Điều này khớp với nghiệm mà ta đã dự đoán ở trên!

Tại thời điểm này, việc đặt câu hỏi sau là hoàn toàn hợp lý: “Tại sao tôi không thể chỉ viết ra các phiên bản ma trận của tất cả các quy tắc giải tích mà mình đã học? Rõ ràng quá trình này vẫn mang tính cơ học. Tại sao chúng ta không làm luôn cho xong!” Và quả thực có những quy tắc như vậy, đồng thời :cite:`Petersen.Pedersen.ea.2008` cung cấp một bản tóm tắt xuất sắc. Tuy nhiên, do có rất nhiều cách để kết hợp các phép toán ma trận so với các giá trị đơn, số lượng quy tắc đạo hàm ma trận cũng nhiều hơn rất nhiều so với các quy tắc của một biến. Thông thường, tốt nhất là làm việc với các chỉ số hoặc để automatic differentiation xử lý khi phù hợp.

## Tóm tắt

* Trong không gian nhiều chiều, ta có thể định nghĩa các gradient, phục vụ cùng mục đích như đạo hàm trong một chiều. Chúng cho phép ta thấy một hàm nhiều biến thay đổi như thế nào khi ta thực hiện một thay đổi nhỏ tùy ý đối với các đầu vào.
* Có thể xem thuật toán lan truyền ngược là một phương pháp tổ chức quy tắc chuỗi nhiều biến để cho phép tính hiệu quả nhiều đạo hàm riêng.
* Giải tích ma trận cho phép ta viết các đạo hàm của các biểu thức ma trận theo những cách cô đọng.

## Bài tập
1. Cho một vector cột $\boldsymbol{\beta}$, hãy tính đạo hàm của cả $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$ và $g(\mathbf{x}) = \mathbf{x}^\top\boldsymbol{\beta}$. Tại sao bạn nhận được cùng một đáp án?
2. Cho $\mathbf{v}$ là một vector $n$ chiều. $\frac{\partial}{\partial\mathbf{v}}\|\mathbf{v}\|_2$ bằng bao nhiêu?
3. Cho $L(x, y) = \log(e^x + e^y)$. Hãy tính gradient. Tổng các thành phần của gradient là bao nhiêu?
4. Cho $f(x, y) = x^2y + xy^2$. Hãy chứng minh rằng điểm tới hạn duy nhất là $(0,0)$. Bằng cách xét $f(x, x)$, hãy xác định xem $(0,0)$ là cực đại, cực tiểu hay không thuộc loại nào trong hai loại đó.
5. Giả sử ta đang cực tiểu hóa một hàm $f(\mathbf{x}) = g(\mathbf{x}) + h(\mathbf{x})$. Ta có thể diễn giải hình học điều kiện $\nabla f = 0$ theo $g$ và $h$ như thế nào?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/413)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1090)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1091)
:end_tab: