# Hạ Gradient
:label:`sec_gd`

Trong phần này, chúng ta sẽ giới thiệu các khái niệm cơ bản làm nền tảng cho *hạ gradient*.
Mặc dù hiếm khi được sử dụng trực tiếp trong deep learning, việc hiểu gradient descent là chìa khóa để hiểu các thuật toán stochastic gradient descent.
Ví dụ, bài toán tối ưu hóa có thể phân kỳ do learning rate quá lớn. Hiện tượng này đã có thể thấy trong gradient descent. Tương tự, preconditioning là một kỹ thuật phổ biến trong gradient descent và được kế thừa trong các thuật toán nâng cao hơn.
Hãy bắt đầu với một trường hợp đặc biệt đơn giản.


## Hạ Gradient Một Chiều

Gradient descent trong một chiều là một ví dụ tuyệt vời để giải thích vì sao thuật toán gradient descent có thể làm giảm giá trị của hàm mục tiêu. Xét một hàm số thực khả vi liên tục $f: \mathbb{R} \rightarrow \mathbb{R}$. Sử dụng khai triển Taylor, ta có

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

Tức là, trong xấp xỉ bậc một, $f(x+\epsilon)$ được xác định bởi giá trị hàm $f(x)$ và đạo hàm bậc nhất $f'(x)$ tại $x$. Có cơ sở để giả định rằng với $\epsilon$ nhỏ, việc di chuyển theo hướng gradient âm sẽ làm giảm $f$. Để đơn giản, ta chọn kích thước bước cố định $\eta > 0$ và chọn $\epsilon = -\eta f'(x)$. Thay giá trị này vào khai triển Taylor ở trên, ta được

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$
:eqlabel:`gd-taylor-2`

Nếu đạo hàm $f'(x) \neq 0$ không triệt tiêu, ta đạt được tiến triển vì $\eta f'^2(x)>0$. Hơn nữa, ta luôn có thể chọn $\eta$ đủ nhỏ để các số hạng bậc cao trở nên không đáng kể. Do đó, ta có

$$f(x - \eta f'(x)) \lessapprox f(x).$$

Điều này có nghĩa là nếu ta sử dụng

$$x \leftarrow x - \eta f'(x)$$

để lặp $x$, giá trị của hàm $f(x)$ có thể giảm. Vì vậy, trong gradient descent, trước tiên ta chọn một giá trị ban đầu $x$ và một hằng số $\eta > 0$, rồi dùng chúng để liên tục lặp $x$ cho đến khi đạt điều kiện dừng, ví dụ khi độ lớn của gradient $|f'(x)|$ đủ nhỏ hoặc số lần lặp đã đạt đến một giá trị nhất định.

Để đơn giản, ta chọn hàm mục tiêu $f(x)=x^2$ nhằm minh họa cách triển khai gradient descent. Mặc dù biết rằng $x=0$ là nghiệm làm nhỏ nhất $f(x)$, ta vẫn sử dụng hàm đơn giản này để quan sát sự thay đổi của $x$.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x
```

Tiếp theo, ta sử dụng $x=10$ làm giá trị ban đầu và giả sử $\eta=0.2$. Khi dùng gradient descent để lặp $x$ 10 lần, ta có thể thấy rằng cuối cùng giá trị của $x$ tiến gần đến nghiệm tối ưu.

```{.python .input}
#@tab all
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

Tiến trình tối ưu hóa theo $x$ có thể được vẽ như sau.

```{.python .input}
#@tab all
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### Learning Rate
:label:`subsec_gd-learningrate`

Learning rate $\eta$ có thể được thiết lập bởi người thiết kế thuật toán. Nếu sử dụng learning rate quá nhỏ, $x$ sẽ được cập nhật rất chậm, đòi hỏi nhiều lần lặp hơn để đạt được nghiệm tốt hơn. Để cho thấy điều gì xảy ra trong trường hợp đó, hãy xét tiến trình của cùng bài toán tối ưu hóa với $\eta = 0.05$. Như có thể thấy, ngay cả sau 10 bước, ta vẫn còn rất xa nghiệm tối ưu.

```{.python .input}
#@tab all
show_trace(gd(0.05, f_grad), f)
```

Ngược lại, nếu sử dụng learning rate quá cao, $\left|\eta f'(x)\right|$ có thể quá lớn khiến công thức khai triển Taylor bậc một không còn phù hợp. Nghĩa là, số hạng $\mathcal{O}(\eta^2 f'^2(x))$ trong :eqref:`gd-taylor-2` có thể trở nên đáng kể. Trong trường hợp này, ta không thể đảm bảo rằng việc lặp $x$ sẽ làm giảm giá trị của $f(x)$. Ví dụ, khi đặt learning rate là $\eta=1.1$, $x$ vượt qua nghiệm tối ưu $x=0$ và dần phân kỳ.

```{.python .input}
#@tab all
show_trace(gd(1.1, f_grad), f)
```

### Cực Tiểu Cục Bộ

Để minh họa điều xảy ra với các hàm không lồi, hãy xét trường hợp $f(x) = x \cdot \cos(cx)$ với một hằng số $c$. Hàm này có vô số cực tiểu cục bộ. Tùy thuộc vào lựa chọn learning rate và mức độ well-conditioned của bài toán, ta có thể kết thúc ở một trong nhiều nghiệm. Ví dụ dưới đây minh họa cách một learning rate cao đến mức không thực tế sẽ dẫn đến một cực tiểu cục bộ kém.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## Hạ Gradient Đa Biến

Bây giờ khi đã có trực giác tốt hơn về trường hợp một biến, hãy xét tình huống $\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$. Tức là, hàm mục tiêu $f: \mathbb{R}^d \to \mathbb{R}$ ánh xạ vector thành vô hướng. Theo đó, gradient của nó cũng là đa biến. Đây là một vector gồm $d$ đạo hàm riêng:

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

Mỗi phần tử đạo hàm riêng $\partial f(\mathbf{x})/\partial x_i$ trong gradient biểu thị tốc độ thay đổi của $f$ tại $\mathbf{x}$ theo đầu vào $x_i$. Tương tự trường hợp một biến trước đó, ta có thể sử dụng xấp xỉ Taylor tương ứng cho các hàm đa biến để hình dung việc cần làm. Cụ thể, ta có

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

Nói cách khác, bỏ qua các số hạng bậc hai theo $\boldsymbol{\epsilon}$, hướng giảm nhanh nhất được cho bởi gradient âm $-\nabla f(\mathbf{x})$. Chọn một learning rate phù hợp $\eta > 0$ cho ta thuật toán gradient descent nguyên mẫu:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

Để xem thuật toán hoạt động thế nào trong thực tế, hãy xây dựng hàm mục tiêu $f(\mathbf{x})=x_1^2+2x_2^2$ với vector hai chiều $\mathbf{x} = [x_1, x_2]^\top$ làm đầu vào và một vô hướng làm đầu ra. Gradient được cho bởi $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$. Ta sẽ quan sát quỹ đạo của $\mathbf{x}$ bằng gradient descent từ vị trí ban đầu $[-5, -2]$.

Đầu tiên, ta cần thêm hai hàm trợ giúp. Hàm thứ nhất sử dụng một hàm cập nhật và áp dụng nó 20 lần lên giá trị ban đầu. Hàm trợ giúp thứ hai trực quan hóa quỹ đạo của $\mathbf{x}$.

```{.python .input}
#@tab all
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """Optimize a 2D objective function with a customized trainer."""
    # `s1` and `s2` are internal state variables that will be used in Momentum, adagrad, RMSProp
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results
```

```{.python .input}
#@tab mxnet
def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-55, 1, 1),
                          d2l.arange(-30, 1, 1))
    x1, x2 = x1.asnumpy()*0.1, x2.asnumpy()*0.1
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab tensorflow
def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab pytorch
def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

Tiếp theo, ta quan sát quỹ đạo của biến tối ưu hóa $\mathbf{x}$ với learning rate $\eta = 0.1$. Ta có thể thấy rằng sau 20 bước, giá trị của $\mathbf{x}$ tiến gần đến giá trị cực tiểu tại $[0, 0]$. Tiến trình diễn ra khá ổn định, dù tương đối chậm.

```{.python .input}
#@tab all
def f_2d(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # Gradient of the objective function
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## Các Phương Pháp Adaptive

Như có thể thấy trong :numref:`subsec_gd-learningrate`, việc chọn learning rate $\eta$ “vừa phải” là khó. Nếu chọn quá nhỏ, ta đạt được rất ít tiến triển. Nếu chọn quá lớn, nghiệm sẽ dao động và trong trường hợp xấu nhất thậm chí có thể phân kỳ. Điều gì xảy ra nếu ta có thể xác định $\eta$ tự động hoặc hoàn toàn không cần phải chọn learning rate?
Các phương pháp bậc hai, không chỉ xét giá trị và gradient của hàm mục tiêu
mà còn xét *độ cong* của nó, có thể hữu ích trong trường hợp này. Mặc dù các phương pháp này không thể áp dụng trực tiếp cho deep learning do chi phí tính toán, chúng cung cấp trực giác hữu ích về cách thiết kế các thuật toán tối ưu hóa nâng cao mô phỏng nhiều đặc tính mong muốn của các thuật toán được trình bày bên dưới.


### Phương Pháp Newton

Khi xem lại khai triển Taylor của một hàm $f: \mathbb{R}^d \rightarrow \mathbb{R}$, không cần dừng lại sau số hạng đầu tiên. Thực tế, ta có thể viết nó là

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

Để tránh ký hiệu cồng kềnh, ta định nghĩa $\mathbf{H} \stackrel{\textrm{def}}{=} \nabla^2 f(\mathbf{x})$ là Hessian của $f$, một ma trận $d \times d$. Với $d$ nhỏ và các bài toán đơn giản, $\mathbf{H}$ dễ tính toán. Tuy nhiên, đối với các mạng nơ-ron sâu, $\mathbf{H}$ có thể lớn đến mức không khả thi, do chi phí lưu trữ $\mathcal{O}(d^2)$ phần tử. Hơn nữa, việc tính nó thông qua backpropagation cũng có thể quá tốn kém. Trước mắt, hãy bỏ qua các cân nhắc đó và xem ta sẽ có được thuật toán nào.

Xét cho cùng, giá trị cực tiểu của $f$ thỏa mãn $\nabla f = 0$.
Theo các quy tắc giải tích trong :numref:`subsec_calculus-grad`,
lấy đạo hàm của :eqref:`gd-hot-taylor` theo $\boldsymbol{\epsilon}$ và bỏ qua các số hạng bậc cao, ta có

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \textrm{ and hence }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

Tức là, ta cần nghịch đảo Hessian $\mathbf{H}$ như một phần của bài toán tối ưu hóa.

Ví dụ đơn giản, với $f(x) = \frac{1}{2} x^2$ ta có $\nabla f(x) = x$ và $\mathbf{H} = 1$. Do đó, với mọi $x$ ta thu được $\epsilon = -x$. Nói cách khác, chỉ cần *một* bước là đủ để hội tụ hoàn hảo mà không cần bất kỳ điều chỉnh nào! Than ôi, ở đây ta đã gặp may một chút: khai triển Taylor là chính xác vì $f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$.

Hãy xem điều gì xảy ra trong các bài toán khác.
Với một hàm cosin hyperbolic lồi $f(x) = \cosh(cx)$ cho một hằng số $c$, ta có thể thấy rằng
cực tiểu toàn cục tại $x=0$ đạt được
sau vài vòng lặp.

```{.python .input}
#@tab all
c = d2l.tensor(0.5)

def f(x):  # Objective function
    return d2l.cosh(c * x)

def f_grad(x):  # Gradient of the objective function
    return c * d2l.sinh(c * x)

def f_hess(x):  # Hessian of the objective function
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

Bây giờ, hãy xét một hàm *không lồi*, chẳng hạn $f(x) = x \cos(c x)$ với một hằng số $c$. Xét cho cùng, lưu ý rằng trong phương pháp Newton, ta phải chia cho Hessian. Điều này có nghĩa là nếu đạo hàm bậc hai là *âm*, ta có thể đi theo hướng *làm tăng* giá trị của $f$.
Đó là một khiếm khuyết nghiêm trọng của thuật toán.
Hãy xem điều gì xảy ra trong thực tế.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # Hessian of the objective function
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

Việc này đã sai một cách thảm hại. Làm thế nào để khắc phục? Một cách là "sửa" Hessian bằng cách lấy giá trị tuyệt đối của nó. Một chiến lược khác là đưa learning rate trở lại. Điều này dường như đi ngược mục đích ban đầu, nhưng không hẳn vậy. Có thông tin bậc hai cho phép ta thận trọng khi độ cong lớn và thực hiện các bước dài hơn khi hàm mục tiêu phẳng hơn.
Hãy xem cách này hoạt động với learning rate nhỏ hơn một chút, chẳng hạn $\eta = 0.5$. Như ta có thể thấy, đây là một thuật toán khá hiệu quả.

```{.python .input}
#@tab all
show_trace(newton(0.5), f)
```

### Phân tích hội tụ

Ta chỉ phân tích tốc độ hội tụ của phương pháp Newton đối với một hàm mục tiêu lồi, khả vi ba lần $f$, trong đó đạo hàm bậc hai khác không, tức là $f'' > 0$. Chứng minh đa biến là phần mở rộng trực tiếp của lập luận một chiều bên dưới và được lược bỏ vì không giúp ích nhiều cho trực giác của chúng ta.

Ký hiệu $x^{(k)}$ là giá trị của $x$ tại vòng lặp thứ $k^\textrm{th}$ và đặt $e^{(k)} \stackrel{\textrm{def}}{=} x^{(k)} - x^*$ là khoảng cách tới nghiệm tối ưu tại vòng lặp thứ $k^\textrm{th}$. Theo khai triển Taylor, điều kiện $f'(x^*) = 0$ có thể được viết thành

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$

đúng với một $\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$ nào đó. Chia khai triển trên cho $f''(x^{(k)})$ ta được

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$

Nhắc lại rằng ta có cập nhật $x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$.
Thay phương trình cập nhật này vào và lấy giá trị tuyệt đối của hai vế, ta có

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$

Do đó, bất cứ khi nào ta ở trong một vùng có $\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$ bị chặn, ta có sai số giảm theo bậc hai

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$


Ngoài lề, các nhà nghiên cứu tối ưu hóa gọi đây là hội tụ *tuyến tính*, trong khi một điều kiện như $\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$ sẽ được gọi là tốc độ hội tụ *hằng số*.
Lưu ý rằng phân tích này đi kèm một số lưu ý quan trọng.
Thứ nhất, ta không thực sự có nhiều đảm bảo về thời điểm sẽ đến được vùng hội tụ nhanh. Thay vào đó, ta chỉ biết rằng một khi đến được vùng đó, quá trình hội tụ sẽ rất nhanh. Thứ hai, phân tích này yêu cầu $f$ có tính chất tốt đến các đạo hàm bậc cao hơn. Điều này quy về việc đảm bảo rằng $f$ không có bất kỳ đặc tính "bất ngờ" nào về cách nó có thể thay đổi giá trị.



### Tiền điều kiện hóa

Không có gì ngạc nhiên khi việc tính toán và lưu trữ toàn bộ Hessian rất tốn kém. Vì vậy, ta muốn tìm các lựa chọn thay thế. Một cách để cải thiện tình hình là *tiền điều kiện hóa* (preconditioning). Phương pháp này tránh tính toàn bộ Hessian mà chỉ tính các phần tử *trên đường chéo*. Điều này dẫn đến các thuật toán cập nhật có dạng

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \textrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$


Mặc dù cách này không tốt bằng phương pháp Newton đầy đủ, nó vẫn tốt hơn nhiều so với không sử dụng nó.
Để hiểu tại sao đây có thể là một ý tưởng hay, hãy xét trường hợp một biến biểu thị chiều cao tính bằng milimét và biến còn lại biểu thị chiều cao tính bằng kilômét. Giả sử thang đo tự nhiên của cả hai đều là mét, ta có sự không tương thích nghiêm trọng trong việc tham số hóa. May mắn thay, việc dùng tiền điều kiện hóa sẽ loại bỏ điều này. Trên thực tế, tiền điều kiện hóa với gradient descent tương đương với việc chọn một learning rate khác nhau cho từng biến (tọa độ của vector $\mathbf{x}$).
Như ta sẽ thấy sau, tiền điều kiện hóa thúc đẩy một số đổi mới trong các thuật toán tối ưu hóa stochastic gradient descent.

### Gradient Descent với Line Search

Một trong những vấn đề chính của gradient descent là ta có thể vượt quá mục tiêu hoặc đạt tiến triển không đủ. Một cách khắc phục đơn giản là dùng line search kết hợp với gradient descent. Tức là, ta sử dụng hướng do $\nabla f(\mathbf{x})$ cung cấp, rồi thực hiện tìm kiếm nhị phân để xác định learning rate $\eta$ nào tối thiểu hóa $f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$.

Thuật toán này hội tụ nhanh (để xem phân tích và chứng minh, xem ví dụ :citet:`Boyd.Vandenberghe.2004`). Tuy nhiên, đối với mục đích deep learning, điều này không thực sự khả thi, vì mỗi bước của line search sẽ yêu cầu ta đánh giá hàm mục tiêu trên toàn bộ tập dữ liệu. Chi phí để thực hiện điều này là quá lớn.

## Tóm tắt

* Learning rate rất quan trọng. Quá lớn thì ta phân kỳ, quá nhỏ thì ta không tiến triển.
* Gradient descent có thể mắc kẹt ở các cực tiểu cục bộ.
* Trong không gian nhiều chiều, việc điều chỉnh learning rate rất phức tạp.
* Tiền điều kiện hóa có thể giúp điều chỉnh thang đo.
* Phương pháp Newton nhanh hơn nhiều khi nó đã bắt đầu hoạt động đúng trong các bài toán lồi.
* Hãy thận trọng khi sử dụng phương pháp Newton mà không có điều chỉnh nào cho các bài toán không lồi.

## Bài tập

1. Thử nghiệm các learning rate và hàm mục tiêu khác nhau cho gradient descent.
1. Cài đặt line search để tối thiểu hóa một hàm lồi trong khoảng $[a, b]$.
    1. Bạn có cần đạo hàm cho tìm kiếm nhị phân, tức là để quyết định chọn $[a, (a+b)/2]$ hay $[(a+b)/2, b]$ không?
    1. Tốc độ hội tụ của thuật toán nhanh đến mức nào?
    1. Cài đặt thuật toán và áp dụng nó để tối thiểu hóa $\log (\exp(x) + \exp(-2x -3))$.
1. Thiết kế một hàm mục tiêu xác định trên $\mathbb{R}^2$ mà gradient descent hoạt động cực kỳ chậm. Gợi ý: đặt thang đo khác nhau cho các tọa độ khác nhau.
1. Cài đặt phiên bản nhẹ của phương pháp Newton bằng tiền điều kiện hóa:
    1. Sử dụng Hessian đường chéo làm preconditioner.
    1. Sử dụng giá trị tuyệt đối của chúng thay vì các giá trị thực tế (có thể có dấu).
    1. Áp dụng điều này cho bài toán ở trên.
1. Áp dụng thuật toán ở trên cho một số hàm mục tiêu (lồi hoặc không lồi). Điều gì xảy ra nếu bạn xoay các tọa độ đi $45$ độ?

[Thảo luận](https://discuss.d2l.ai/t/351)