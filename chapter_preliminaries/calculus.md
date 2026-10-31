```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Giải tích
:label:`sec_calculus`

Trong một thời gian dài, cách tính
diện tích của một hình tròn vẫn là một bí ẩn.
Sau đó, tại Hy Lạp cổ đại, nhà toán học Archimedes
đã nảy ra ý tưởng thông minh
là nội tiếp một chuỗi đa giác
có số lượng đỉnh tăng dần
bên trong một hình tròn
(:numref:`fig_circle_area`).
Với một đa giác có $n$ đỉnh,
ta thu được $n$ tam giác.
Chiều cao của mỗi tam giác tiến gần đến bán kính $r$
khi ta phân hoạch hình tròn ngày càng mịn hơn.
Đồng thời, đáy của nó tiến gần đến $2 \pi r/n$,
vì tỷ số giữa cung tròn và dây cung tiến gần đến 1
khi số lượng đỉnh lớn.
Do đó, diện tích của đa giác tiến gần đến
$n \cdot r \cdot \frac{1}{2} (2 \pi r/n) = \pi r^2$.

![Tìm diện tích hình tròn như một quá trình lấy giới hạn.](../img/polygon-circle.svg)
:label:`fig_circle_area`

Quy trình lấy giới hạn này là nền tảng của cả
*vi phân* và *tích phân*.
Loại thứ nhất có thể cho chúng ta biết cách tăng
hoặc giảm giá trị của một hàm
bằng cách điều chỉnh các đối số của nó.
Điều này hữu ích cho các *bài toán tối ưu hóa*
mà chúng ta gặp trong deep learning,
nơi chúng ta liên tục cập nhật các tham số
để giảm hàm mất mát.
Tối ưu hóa giải quyết cách khớp các mô hình với dữ liệu huấn luyện,
và giải tích là tiền đề then chốt của nó.
Tuy nhiên, đừng quên rằng mục tiêu cuối cùng của chúng ta
là hoạt động tốt trên dữ liệu *chưa từng thấy trước đây*.
Vấn đề đó được gọi là *khái quát hóa*
và sẽ là trọng tâm chính của các chương khác.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from matplotlib_inline import backend_inline
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

## Đạo hàm và Phép vi phân

Nói một cách đơn giản, *đạo hàm* là tốc độ thay đổi
của một hàm theo sự thay đổi trong các đối số của nó.
Đạo hàm có thể cho chúng ta biết hàm mất mát
sẽ tăng hoặc giảm nhanh đến mức nào nếu chúng ta
*tăng* hoặc *giảm* từng tham số
một lượng nhỏ vô cùng.
Về mặt hình thức, với các hàm $f: \mathbb{R} \rightarrow \mathbb{R}$
ánh xạ từ vô hướng sang vô hướng,
[**đạo hàm của $f$ tại điểm $x$ được định nghĩa là**]

(**$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.$$**)
:eqlabel:`eq_derivative`

Biểu thức ở vế phải này được gọi là *giới hạn*
và nó cho chúng ta biết điều gì xảy ra
với giá trị của một biểu thức
khi một biến được chỉ định
tiến đến một giá trị cụ thể.
Giới hạn này cho chúng ta biết
tỷ số giữa một nhiễu loạn $h$
và sự thay đổi trong giá trị hàm
$f(x + h) - f(x)$ hội tụ về đâu
khi ta thu nhỏ độ lớn của nó về không.

Khi $f'(x)$ tồn tại, $f$ được gọi
là *khả vi* tại $x$;
và khi $f'(x)$ tồn tại với mọi $x$
trên một tập, ví dụ khoảng $[a,b]$,
ta nói rằng $f$ khả vi trên tập này.
Không phải mọi hàm đều khả vi,
bao gồm nhiều hàm mà ta muốn tối ưu hóa,
chẳng hạn như accuracy và diện tích dưới
đường cong đặc trưng hoạt động của bộ thu (AUC).
Tuy nhiên, vì việc tính đạo hàm của mất mát
là một bước thiết yếu trong hầu như mọi
thuật toán huấn luyện mạng nơ-ron sâu,
chúng ta thường tối ưu hóa một *surrogate* khả vi thay vào đó.


Ta có thể diễn giải đạo hàm
$f'(x)$
là tốc độ thay đổi *tức thời*
của $f(x)$ theo $x$.
Hãy xây dựng trực giác qua một ví dụ.
(**Đặt $u = f(x) = 3x^2-4x$.**)

```{.python .input}
%%tab mxnet
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab pytorch
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab tensorflow
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab jax
def f(x):
    return 3 * x ** 2 - 4 * x
```

[**Khi đặt $x=1$, ta thấy $\frac{f(x+h) - f(x)}{h}$**] (**tiến đến $2$
khi $h$ tiến đến $0$.**)
Mặc dù thí nghiệm này không có
tính chặt chẽ của một chứng minh toán học,
ta có thể nhanh chóng thấy rằng thực sự $f'(1) = 2$.

```{.python .input}
%%tab all
for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')
```

Có một số quy ước ký hiệu tương đương cho đạo hàm.
Cho $y = f(x)$, các biểu thức sau là tương đương:

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

trong đó các ký hiệu $\frac{d}{dx}$ và $D$ là các *toán tử vi phân*.
Dưới đây, chúng tôi trình bày đạo hàm của một số hàm phổ biến:

$$\begin{aligned} \frac{d}{dx} C & = 0 && \textrm{for any constant $C$} \\ \frac{d}{dx} x^n & = n x^{n-1} && \textrm{for } n \neq 0 \\ \frac{d}{dx} e^x & = e^x \\ \frac{d}{dx} \ln x & = x^{-1}. \end{aligned}$$

Các hàm được cấu thành từ những hàm khả vi
thường cũng tự chúng khả vi.
Các quy tắc sau hữu ích
khi làm việc với các tổ hợp
của bất kỳ hàm khả vi nào
$f$ và $g$, cùng hằng số $C$.

$$\begin{aligned} \frac{d}{dx} [C f(x)] & = C \frac{d}{dx} f(x) && \textrm{Constant multiple rule} \\ \frac{d}{dx} [f(x) + g(x)] & = \frac{d}{dx} f(x) + \frac{d}{dx} g(x) && \textrm{Sum rule} \\ \frac{d}{dx} [f(x) g(x)] & = f(x) \frac{d}{dx} g(x) + g(x) \frac{d}{dx} f(x) && \textrm{Product rule} \\ \frac{d}{dx} \frac{f(x)}{g(x)} & = \frac{g(x) \frac{d}{dx} f(x) - f(x) \frac{d}{dx} g(x)}{g^2(x)} && \textrm{Quotient rule} \end{aligned}$$

Dùng các quy tắc này, ta có thể tìm đạo hàm của $3 x^2 - 4x$ bằng

$$\frac{d}{dx} [3 x^2 - 4x] = 3 \frac{d}{dx} x^2 - 4 \frac{d}{dx} x = 6x - 4.$$

Thay $x = 1$ vào cho thấy rằng thực sự
đạo hàm bằng $2$ tại vị trí này.
Lưu ý rằng đạo hàm cho chúng ta biết
*độ dốc* của một hàm
tại một vị trí cụ thể.

## Tiện ích trực quan hóa

[**Ta có thể trực quan hóa độ dốc của các hàm bằng thư viện `matplotlib`**].
Ta cần định nghĩa một vài hàm.
Đúng như tên gọi, `use_svg_display`
chỉ dẫn `matplotlib` xuất đồ họa
ở định dạng SVG để hình ảnh sắc nét hơn.
Chú thích `#@save` là một modifier đặc biệt
cho phép ta lưu bất kỳ hàm,
class hoặc khối mã nào khác vào package `d2l`
để có thể gọi lại sau này
mà không cần lặp lại mã,
ví dụ, thông qua `d2l.use_svg_display()`.

```{.python .input}
%%tab all
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')
```

Một cách thuận tiện, ta có thể đặt kích thước hình bằng `set_figsize`.
Vì câu lệnh import `from matplotlib import pyplot as plt`
đã được đánh dấu qua `#@save` trong package `d2l`, ta có thể gọi `d2l.plt`.

```{.python .input}
%%tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

Hàm `set_axes` có thể liên kết các trục
với các thuộc tính, bao gồm nhãn, phạm vi
và tỷ lệ.

```{.python .input}
%%tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

Với ba hàm này, ta có thể định nghĩa một hàm `plot`
để chồng nhiều đường cong lên nhau.
Phần lớn mã ở đây chỉ nhằm bảo đảm
rằng kích thước và hình dạng của các đầu vào khớp nhau.

```{.python .input}
%%tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))
    
    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
        
    set_figsize(figsize)
    if axes is None:
        axes = d2l.plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

Bây giờ ta có thể [**vẽ hàm $u = f(x)$ và tiếp tuyến $y = 2x - 3$ của nó tại $x=1$**],
trong đó hệ số $2$ là độ dốc của tiếp tuyến.

```{.python .input}
%%tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## Đạo hàm riêng và Gradient
:label:`subsec_calculus-grad`

Cho đến nay, chúng ta chỉ lấy đạo hàm
các hàm của một biến.
Trong deep learning, ta cũng cần làm việc
với các hàm có *nhiều* biến.
Chúng tôi sẽ giới thiệu ngắn gọn các khái niệm về đạo hàm
áp dụng cho những hàm *nhiều biến* như vậy.


Cho $y = f(x_1, x_2, \ldots, x_n)$ là một hàm có $n$ biến.
*Đạo hàm riêng* của $y$
theo tham số thứ $i^\textrm{th}$ $x_i$ là

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$


Để tính $\frac{\partial y}{\partial x_i}$,
ta có thể coi $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ là hằng số
và tính đạo hàm của $y$ theo $x_i$.
Các quy ước ký hiệu sau cho đạo hàm riêng
đều phổ biến và đều có cùng nghĩa:

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = \partial_{x_i} f = \partial_i f = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

Ta có thể ghép các đạo hàm riêng
của một hàm nhiều biến
theo tất cả các biến của nó
để thu được một vector được gọi là
*gradient* của hàm.
Giả sử đầu vào của hàm
$f: \mathbb{R}^n \rightarrow \mathbb{R}$
là một vector $n$ chiều
$\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$
và đầu ra là một vô hướng.
Gradient của hàm $f$
theo $\mathbf{x}$
là một vector gồm $n$ đạo hàm riêng:

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \left[\partial_{x_1} f(\mathbf{x}), \partial_{x_2} f(\mathbf{x}), \ldots
\partial_{x_n} f(\mathbf{x})\right]^\top.$$

Khi không có sự mơ hồ,
$\nabla_{\mathbf{x}} f(\mathbf{x})$
thường được thay thế
bằng $\nabla f(\mathbf{x})$.
Các quy tắc sau hữu ích
khi lấy đạo hàm các hàm nhiều biến:

* Với mọi $\mathbf{A} \in \mathbb{R}^{m \times n}$, ta có $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$ và $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$.
* Với các ma trận vuông $\mathbf{A} \in \mathbb{R}^{n \times n}$, ta có $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ và cụ thể là
$\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$.

Tương tự, với mọi ma trận $\mathbf{X}$, 
ta có $\nabla_{\mathbf{X}} \|\mathbf{X} \|_\textrm{F}^2 = 2\mathbf{X}$. 



## Quy tắc Chuỗi

Trong deep learning, các gradient cần quan tâm
thường khó tính toán
vì chúng ta làm việc với 
các hàm lồng nhau sâu 
(của các hàm (của các hàm...)).
May mắn thay, *quy tắc chuỗi* xử lý được điều này. 
Quay lại với các hàm một biến,
giả sử rằng $y = f(g(x))$
và các hàm cơ sở 
$y=f(u)$ và $u=g(x)$ 
đều khả vi.
Quy tắc chuỗi phát biểu rằng 


$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$



Quay lại với các hàm nhiều biến,
giả sử rằng $y = f(\mathbf{u})$ có các biến
$u_1, u_2, \ldots, u_m$, 
trong đó mỗi $u_i = g_i(\mathbf{x})$ 
có các biến $x_1, x_2, \ldots, x_n$,
nghĩa là $\mathbf{u} = g(\mathbf{x})$.
Khi đó, quy tắc chuỗi phát biểu rằng

$$\frac{\partial y}{\partial x_{i}} = \frac{\partial y}{\partial u_{1}} \frac{\partial u_{1}}{\partial x_{i}} + \frac{\partial y}{\partial u_{2}} \frac{\partial u_{2}}{\partial x_{i}} + \ldots + \frac{\partial y}{\partial u_{m}} \frac{\partial u_{m}}{\partial x_{i}} \ \textrm{ and so } \ \nabla_{\mathbf{x}} y =  \mathbf{A} \nabla_{\mathbf{u}} y,$$

trong đó $\mathbf{A} \in \mathbb{R}^{n \times m}$ là một *ma trận*
chứa đạo hàm của vector $\mathbf{u}$
theo vector $\mathbf{x}$.
Do đó, việc tính gradient yêu cầu 
tính tích vector--ma trận. 
Đây là một trong những lý do then chốt khiến đại số tuyến tính 
là một nền tảng không thể thiếu 
trong việc xây dựng các hệ thống deep learning. 



## Thảo luận

Mặc dù chúng ta mới chỉ chạm đến bề mặt của một chủ đề sâu rộng,
một số khái niệm đã trở nên rõ nét: 
thứ nhất, các quy tắc hợp thành cho phép lấy đạo hàm
được áp dụng một cách thường quy, giúp
chúng ta tính gradient *tự động*.
Nhiệm vụ này không đòi hỏi sự sáng tạo, do đó 
chúng ta có thể tập trung năng lực nhận thức vào những việc khác.
Thứ hai, việc tính đạo hàm của các hàm có giá trị vector 
yêu cầu chúng ta nhân các ma trận khi lần theo 
đồ thị phụ thuộc của các biến từ đầu ra đến đầu vào. 
Cụ thể, đồ thị này được duyệt theo hướng *thuận* 
khi chúng ta tính giá trị của một hàm 
và theo hướng *ngược* 
khi chúng ta tính gradient. 
Các chương sau sẽ giới thiệu chính thức backpropagation,
một quy trình tính toán để áp dụng quy tắc chuỗi.

Từ góc nhìn của tối ưu hóa, gradient cho phép chúng ta 
xác định cách di chuyển các tham số của một mô hình
để giảm loss,
và mỗi bước của các thuật toán tối ưu hóa được sử dụng 
xuyên suốt cuốn sách này sẽ cần tính gradient.

## Bài tập

1. Cho đến nay, chúng ta đã mặc nhiên thừa nhận các quy tắc đạo hàm. 
   Hãy sử dụng định nghĩa và giới hạn để chứng minh các tính chất 
   cho (i) $f(x) = c$, (ii) $f(x) = x^n$, (iii) $f(x) = e^x$ và (iv) $f(x) = \log x$.
1. Tương tự, hãy chứng minh quy tắc tích, tổng và thương từ các nguyên lý cơ bản. 
1. Chứng minh rằng quy tắc nhân với hằng số là một trường hợp đặc biệt của quy tắc tích. 
1. Tính đạo hàm của $f(x) = x^x$. 
1. Việc $f'(x) = 0$ tại một $x$ nào đó có nghĩa là gì? 
   Hãy đưa ra một ví dụ về hàm $f$ 
   và một vị trí $x$ mà điều này có thể đúng. 
1. Vẽ đồ thị hàm số $y = f(x) = x^3 - \frac{1}{x}$ 
   và vẽ đường tiếp tuyến của nó tại $x = 1$.
1. Tìm gradient của hàm 
   $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$.
1. Gradient của hàm 
   $f(\mathbf{x}) = \|\mathbf{x}\|_2$ là gì? Điều gì xảy ra khi $\mathbf{x} = \mathbf{0}$?
1. Bạn có thể viết quy tắc chuỗi cho trường hợp 
   $u = f(x, y, z)$ và $x = x(a, b)$, $y = y(a, b)$, và $z = z(a, b)$ không?
1. Cho một hàm khả nghịch $f(x)$, 
   hãy tính đạo hàm của hàm nghịch đảo $f^{-1}(x)$. 
   Ở đây, ta có $f^{-1}(f(x)) = x$ và ngược lại $f(f^{-1}(y)) = y$. 
   Gợi ý: sử dụng các tính chất này trong phép suy diễn của bạn. 

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/32)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/33)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/197)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17969)
:end_tab: