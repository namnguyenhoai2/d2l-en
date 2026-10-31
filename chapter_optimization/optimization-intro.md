# Tối ưu hóa và Deep Learning
:label:`sec_optimization-intro`

Trong phần này, chúng ta sẽ thảo luận về mối quan hệ giữa tối ưu hóa và deep learning, cũng như những thách thức khi sử dụng tối ưu hóa trong deep learning.
Đối với một bài toán deep learning, thông thường trước tiên chúng ta sẽ định nghĩa một *hàm mất mát (loss function)*. Khi đã có hàm mất mát, chúng ta có thể sử dụng một thuật toán tối ưu hóa để cố gắng giảm thiểu mất mát.
Trong tối ưu hóa, một hàm mất mát thường được gọi là *hàm mục tiêu (objective function)* của bài toán tối ưu hóa. Theo truyền thống và quy ước, hầu hết các thuật toán tối ưu hóa đều quan tâm đến việc *tối thiểu hóa*. Nếu cần tối đa hóa một mục tiêu, có một cách giải quyết đơn giản: chỉ cần đảo dấu của mục tiêu.

## Mục tiêu của Tối ưu hóa

Mặc dù tối ưu hóa cung cấp một cách để giảm thiểu hàm mất mát cho deep
learning, về bản chất, mục tiêu của tối ưu hóa và deep learning
khác nhau một cách cơ bản.
Mục tiêu thứ nhất chủ yếu là giảm thiểu một
mục tiêu, trong khi mục tiêu thứ hai là tìm một mô hình phù hợp, với một
lượng dữ liệu hữu hạn.
Trong :numref:`sec_generalization_basics`,
chúng ta đã thảo luận chi tiết về sự khác biệt giữa hai mục tiêu này.
Chẳng hạn,
lỗi huấn luyện và lỗi tổng quát hóa thường khác nhau: vì hàm mục tiêu
của thuật toán tối ưu hóa thường là một hàm mất mát dựa trên
tập dữ liệu huấn luyện, nên mục tiêu của tối ưu hóa là giảm lỗi huấn luyện.
Tuy nhiên, mục tiêu của deep learning (hoặc rộng hơn là suy luận thống kê) là
giảm lỗi tổng quát hóa.
Để đạt được mục tiêu sau, chúng ta cần chú ý
đến overfitting, bên cạnh việc sử dụng thuật toán tối ưu hóa để
giảm lỗi huấn luyện.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

Để minh họa các mục tiêu khác nhau đã đề cập ở trên,
hãy xem xét
rủi ro thực nghiệm và rủi ro.
Như đã mô tả
trong :numref:`subsec_empirical-risk-and-risk`,
rủi ro thực nghiệm
là mất mát trung bình
trên tập dữ liệu huấn luyện,
trong khi rủi ro là mất mát kỳ vọng
trên toàn bộ tổng thể dữ liệu.
Dưới đây, chúng ta định nghĩa hai hàm:
hàm rủi ro `f`
và hàm rủi ro thực nghiệm `g`.
Giả sử rằng chúng ta chỉ có một lượng dữ liệu huấn luyện hữu hạn.
Do đó, ở đây `g` kém mượt hơn `f`.

```{.python .input}
#@tab all
def f(x):
    return x * d2l.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

Biểu đồ bên dưới minh họa rằng điểm cực tiểu của rủi ro thực nghiệm trên một tập dữ liệu huấn luyện có thể nằm ở vị trí khác với điểm cực tiểu của rủi ro (lỗi tổng quát hóa).

```{.python .input}
#@tab all
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

## Những Thách thức Tối ưu hóa trong Deep Learning

Trong chương này, chúng ta sẽ tập trung cụ thể vào hiệu năng của các thuật toán tối ưu hóa trong việc giảm thiểu hàm mục tiêu, thay vì
lỗi tổng quát hóa của một mô hình.
Trong :numref:`sec_linear_regression`
chúng ta đã phân biệt giữa nghiệm giải tích và nghiệm số trong
các bài toán tối ưu hóa.
Trong deep learning, hầu hết các hàm mục tiêu đều
phức tạp và không có nghiệm giải tích. Thay vào đó, chúng ta phải sử dụng các
thuật toán tối ưu hóa số.
Tất cả các thuật toán tối ưu hóa trong chương này
đều thuộc
nhóm này.

Có nhiều thách thức trong tối ưu hóa deep learning. Một số thách thức khó giải quyết nhất là cực tiểu cục bộ, điểm yên ngựa và gradient biến mất.
Hãy cùng xem xét chúng.

### Cực tiểu Cục bộ

Với bất kỳ hàm mục tiêu nào $f(x)$,
nếu giá trị của $f(x)$ tại $x$ nhỏ hơn các giá trị của $f(x)$ tại bất kỳ điểm nào khác lân cận $x$, thì $f(x)$ có thể là một cực tiểu cục bộ.
Nếu giá trị của $f(x)$ tại $x$ là giá trị nhỏ nhất của hàm mục tiêu trên toàn bộ miền,
thì $f(x)$ là cực tiểu toàn cục.

Ví dụ, với hàm

$$f(x) = x \cdot \textrm{cos}(\pi x) \textrm{ for } -1.0 \leq x \leq 2.0,$$

chúng ta có thể xấp xỉ cực tiểu cục bộ và cực tiểu toàn cục của hàm này.

```{.python .input}
#@tab all
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

Hàm mục tiêu của các mô hình deep learning thường có nhiều điểm tối ưu cục bộ.
Khi nghiệm số của một bài toán tối ưu hóa ở gần điểm tối ưu cục bộ, nghiệm số thu được ở lần lặp cuối cùng có thể chỉ giảm thiểu hàm mục tiêu *cục bộ*, thay vì *toàn cục*, vì gradient của các nghiệm của hàm mục tiêu tiến gần đến hoặc trở thành không.
Chỉ một mức độ nhiễu nào đó mới có thể đẩy tham số ra khỏi cực tiểu cục bộ. Trên thực tế, đây là một trong các đặc tính có lợi của
minibatch stochastic gradient descent, trong đó sự biến thiên tự nhiên của gradient giữa các minibatch có thể đẩy các tham số ra khỏi cực tiểu cục bộ.

### Điểm Yên ngựa

Bên cạnh cực tiểu cục bộ, điểm yên ngựa là một nguyên nhân khác khiến gradient biến mất. Một *điểm yên ngựa* là bất kỳ vị trí nào mà mọi gradient của một hàm đều bằng không nhưng vị trí đó không phải là cực tiểu toàn cục lẫn cực tiểu cục bộ.
Xét hàm $f(x) = x^3$. Đạo hàm bậc nhất và bậc hai của hàm này đều bằng không tại $x=0$. Quá trình tối ưu hóa có thể bị đình trệ tại điểm này, dù nó không phải là một cực tiểu.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

Các điểm yên ngựa trong không gian nhiều chiều thậm chí còn khó xử lý hơn, như ví dụ dưới đây cho thấy. Xét hàm $f(x, y) = x^2 - y^2$. Hàm này có điểm yên ngựa tại $(0, 0)$. Đây là một cực đại theo $y$ và một cực tiểu theo $x$. Hơn nữa, nó *trông* giống một cái yên ngựa, cũng là nguồn gốc tên gọi của tính chất toán học này.

```{.python .input}
#@tab mxnet
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

```{.python .input}
#@tab pytorch, tensorflow
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

Chúng ta giả sử đầu vào của một hàm là một vector $k$ chiều và
đầu ra của nó là một vô hướng, do đó ma trận Hessian của nó sẽ có $k$ trị riêng.
Nghiệm của
hàm có thể là cực tiểu cục bộ, cực đại cục bộ hoặc điểm yên ngựa tại một
vị trí mà gradient của hàm bằng không:

* Khi các trị riêng của ma trận Hessian của hàm tại vị trí có gradient bằng không đều dương, hàm có một cực tiểu cục bộ.
* Khi các trị riêng của ma trận Hessian của hàm tại vị trí có gradient bằng không đều âm, hàm có một cực đại cục bộ.
* Khi các trị riêng của ma trận Hessian của hàm tại vị trí có gradient bằng không gồm cả âm và dương, hàm có một điểm yên ngựa.

Đối với các bài toán nhiều chiều, xác suất có ít nhất *một số* trị riêng âm là khá cao. Điều này khiến các điểm yên ngựa có khả năng xuất hiện nhiều hơn cực tiểu cục bộ. Chúng ta sẽ thảo luận một số ngoại lệ của tình huống này trong phần tiếp theo khi giới thiệu tính lồi. Tóm lại, hàm lồi là các hàm mà trị riêng của Hessian không bao giờ âm. Tuy nhiên, đáng tiếc là hầu hết các bài toán deep learning không thuộc nhóm này. Dù vậy, đây vẫn là một công cụ tuyệt vời để nghiên cứu các thuật toán tối ưu hóa.

### Gradient Biến mất

Có lẽ vấn đề khó xử lý nhất là gradient biến mất.
Hãy nhớ lại các activation function thường dùng và đạo hàm của chúng trong :numref:`subsec_activation-functions`.
Ví dụ, giả sử rằng chúng ta muốn giảm thiểu hàm $f(x) = \tanh(x)$ và tình cờ bắt đầu tại $x = 4$. Như có thể thấy, gradient của $f$ gần như bằng không.
Cụ thể hơn, $f'(x) = 1 - \tanh^2(x)$ và do đó $f'(4) = 0.0013$.
Vì vậy, quá trình tối ưu hóa sẽ bị mắc kẹt trong một thời gian dài trước khi đạt được tiến triển. Đây hóa ra là một trong những lý do khiến việc huấn luyện các mô hình deep learning khá khó khăn trước khi activation function ReLU được giới thiệu.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

Như chúng ta đã thấy, tối ưu hóa cho deep learning đầy rẫy thách thức. May mắn thay, có một loạt thuật toán mạnh mẽ hoạt động tốt và dễ sử dụng, ngay cả với người mới bắt đầu. Hơn nữa, thực sự không cần phải tìm được nghiệm tốt nhất *duy nhất*. Các điểm tối ưu cục bộ hoặc thậm chí các nghiệm xấp xỉ của chúng vẫn rất hữu ích.

## Tóm tắt

* Việc giảm thiểu lỗi huấn luyện *không* đảm bảo rằng chúng ta tìm được tập tham số tốt nhất để giảm thiểu lỗi tổng quát hóa.
* Các bài toán tối ưu hóa có thể có nhiều cực tiểu cục bộ.
* Bài toán có thể có nhiều điểm yên ngựa hơn nữa, vì thông thường các bài toán không lồi.
* Gradient biến mất có thể khiến quá trình tối ưu hóa bị đình trệ. Việc tái tham số hóa bài toán thường hữu ích. Khởi tạo tốt các tham số cũng có thể mang lại lợi ích.

## Bài tập

1. Xét một MLP đơn giản với một hidden layer duy nhất, chẳng hạn có $d$ chiều trong hidden layer và một đầu ra duy nhất. Hãy chỉ ra rằng với bất kỳ cực tiểu cục bộ nào, có ít nhất $d!$ nghiệm tương đương hoạt động giống hệt nhau.
1. Giả sử chúng ta có một ma trận ngẫu nhiên đối xứng $\mathbf{M}$ với các phần tử
   $M_{ij} = M_{ji}$ đều được lấy mẫu từ một phân phối xác suất nào đó
   $p_{ij}$. Hơn nữa, giả sử rằng $p_{ij}(x) = p_{ij}(-x)$, tức là
   phân phối là đối xứng (xem, chẳng hạn, :citet:`Wigner.1958` để biết chi tiết).
    1. Chứng minh rằng phân phối trên các trị riêng cũng đối xứng. Nghĩa là, với bất kỳ vector riêng $\mathbf{v}$ nào, xác suất để trị riêng liên kết $\lambda$ thỏa mãn $P(\lambda > 0) = P(\lambda < 0)$.
    1. Tại sao điều trên *không* ngụ ý rằng $P(\lambda > 0) = 0.5$?
1. Bạn có thể nghĩ đến những thách thức nào khác trong tối ưu hóa deep learning?
1. Giả sử bạn muốn giữ thăng bằng một quả bóng (thật) trên một cái yên ngựa (thật).
    1. Tại sao điều này khó?
    1. Bạn có thể khai thác hiệu ứng này cho các thuật toán tối ưu hóa không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/349)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/487)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/489)
:end_tab: