# Tính lồi
:label:`sec_convexity`

Tính lồi đóng vai trò thiết yếu trong việc thiết kế các thuật toán tối ưu hóa. 
Điều này phần lớn là do việc phân tích và kiểm thử các thuật toán trong bối cảnh như vậy dễ dàng hơn nhiều. 
Nói cách khác,
nếu thuật toán hoạt động kém ngay cả trong bối cảnh lồi,
thì thông thường chúng ta không nên kỳ vọng vào kết quả tốt trong các trường hợp khác. 
Hơn nữa, mặc dù các bài toán tối ưu hóa trong deep learning nhìn chung là không lồi, chúng thường thể hiện một số đặc tính của các bài toán lồi ở gần cực tiểu cục bộ. Điều này có thể dẫn đến các biến thể tối ưu hóa mới thú vị như :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`.

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

## Định nghĩa

Trước khi phân tích tính lồi,
chúng ta cần định nghĩa *tập lồi* và *hàm lồi*.
Chúng dẫn đến các công cụ toán học thường được áp dụng trong machine learning.


### Tập lồi

Tập hợp là nền tảng của tính lồi. Nói đơn giản, một tập $\mathcal{X}$ trong không gian vector là *lồi* nếu với mọi $a, b \in \mathcal{X}$, đoạn thẳng nối $a$ và $b$ cũng nằm trong $\mathcal{X}$. Về mặt toán học, điều này có nghĩa là với mọi $\lambda \in [0, 1]$, ta có

$$\lambda  a + (1-\lambda)  b \in \mathcal{X} \textrm{ whenever } a, b \in \mathcal{X}.$$

Điều này nghe có vẻ hơi trừu tượng. Hãy xem :numref:`fig_pacman`. Tập đầu tiên không lồi vì tồn tại các đoạn thẳng không được chứa trong nó.
Hai tập còn lại không có vấn đề này.

![Tập thứ nhất là không lồi, còn hai tập còn lại là lồi.](../img/pacman.svg)
:label:`fig_pacman`

Các định nghĩa tự chúng không đặc biệt hữu ích nếu bạn không thể làm gì với chúng.
Trong trường hợp này, chúng ta có thể xem xét các giao như minh họa trong :numref:`fig_convex_intersect`.
Giả sử $\mathcal{X}$ và $\mathcal{Y}$ là các tập lồi. Khi đó $\mathcal{X} \cap \mathcal{Y}$ cũng lồi. Để thấy điều này, xét bất kỳ $a, b \in \mathcal{X} \cap \mathcal{Y}$. Vì $\mathcal{X}$ và $\mathcal{Y}$ là các tập lồi, các đoạn thẳng nối $a$ và $b$ đều nằm trong cả $\mathcal{X}$ lẫn $\mathcal{Y}$. Do đó, chúng cũng phải nằm trong $\mathcal{X} \cap \mathcal{Y}$, qua đó chứng minh định lý của chúng ta.

![Giao của hai tập lồi là lồi.](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

Ta có thể củng cố kết quả này mà không cần nhiều nỗ lực: với các tập lồi $\mathcal{X}_i$, giao của chúng $\cap_{i} \mathcal{X}_i$ là lồi.
Để thấy mệnh đề đảo không đúng, hãy xét hai tập rời nhau $\mathcal{X} \cap \mathcal{Y} = \emptyset$. Bây giờ chọn $a \in \mathcal{X}$ và $b \in \mathcal{Y}$. Đoạn thẳng trong :numref:`fig_nonconvex` nối $a$ và $b$ phải chứa một phần không nằm trong cả $\mathcal{X}$ lẫn $\mathcal{Y}$, vì ta đã giả sử $\mathcal{X} \cap \mathcal{Y} = \emptyset$. Do đó, đoạn thẳng cũng không nằm trong $\mathcal{X} \cup \mathcal{Y}$, từ đó chứng minh rằng nói chung hợp của các tập lồi không nhất thiết là lồi.

![Hợp của hai tập lồi không nhất thiết là lồi.](../img/nonconvex.svg)
:label:`fig_nonconvex`

Thông thường, các bài toán trong deep learning được xác định trên các tập lồi. Ví dụ, $\mathbb{R}^d$,
tập các vector số thực $d$ chiều,
là một tập lồi (xét cho cùng, đoạn thẳng giữa bất kỳ hai điểm nào trong $\mathbb{R}^d$ vẫn nằm trong $\mathbb{R}^d$). Trong một số trường hợp, chúng ta làm việc với các biến có độ dài bị chặn, chẳng hạn như các quả cầu bán kính $r$ được xác định bởi $\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \textrm{ and } \|\mathbf{x}\| \leq r\}$.

### Hàm lồi

Giờ đây khi đã có các tập lồi, ta có thể giới thiệu *hàm lồi* $f$.
Với một tập lồi $\mathcal{X}$, một hàm $f: \mathcal{X} \to \mathbb{R}$ là *lồi* nếu với mọi $x, x' \in \mathcal{X}$ và mọi $\lambda \in [0, 1]$, ta có

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

Để minh họa, hãy vẽ một vài hàm và kiểm tra hàm nào thỏa mãn yêu cầu.
Bên dưới, chúng ta định nghĩa một vài hàm, bao gồm cả hàm lồi và không lồi.

```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2  # Convex
g = lambda x: d2l.cos(np.pi * x)  # Nonconvex
h = lambda x: d2l.exp(0.5 * x)  # Convex

x, segment = d2l.arange(-2, 2, 0.01), d2l.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

Đúng như dự kiến, hàm cosine là *không lồi*, trong khi parabol và hàm mũ là lồi. Lưu ý rằng yêu cầu $\mathcal{X}$ là một tập lồi là cần thiết để điều kiện này có ý nghĩa. Nếu không, kết quả của $f(\lambda x + (1-\lambda) x')$ có thể không được xác định rõ ràng.


### Bất đẳng thức Jensen

Với một hàm lồi $f$,
một trong những công cụ toán học hữu ích nhất
là *bất đẳng thức Jensen*.
Đây là sự khái quát hóa của định nghĩa tính lồi:

$$\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right)    \textrm{ and }    E_X[f(X)]  \geq f\left(E_X[X]\right),$$
:eqlabel:`eq_jensens-inequality`

trong đó $\alpha_i$ là các số thực không âm sao cho $\sum_i \alpha_i = 1$ và $X$ là một biến ngẫu nhiên.
Nói cách khác, kỳ vọng của một hàm lồi không nhỏ hơn hàm lồi của một kỳ vọng, trong đó biểu thức sau thường đơn giản hơn. 
Để chứng minh bất đẳng thức thứ nhất, ta áp dụng lặp lại định nghĩa tính lồi cho từng hạng tử trong tổng.


Một ứng dụng phổ biến của bất đẳng thức Jensen là
chặn một biểu thức phức tạp hơn bằng một biểu thức đơn giản hơn.
Ví dụ,
nó có thể được áp dụng
cho log-likelihood của các biến ngẫu nhiên được quan sát một phần. Cụ thể, ta dùng

$$E_{Y \sim P(Y)}[-\log P(X \mid Y)] \geq -\log P(X),$$

vì $\int P(Y) P(X \mid Y) dY = P(X)$.
Điều này có thể được sử dụng trong các phương pháp biến phân. Ở đây, $Y$ thường là biến ngẫu nhiên không được quan sát, $P(Y)$ là ước đoán tốt nhất về cách nó có thể được phân phối, và $P(X)$ là phân phối sau khi đã tích phân $Y$. Chẳng hạn, trong phân cụm, $Y$ có thể là các nhãn cụm và $P(X \mid Y)$ là mô hình sinh khi áp dụng các nhãn cụm.



## Tính chất

Các hàm lồi có nhiều tính chất hữu ích. Dưới đây, chúng ta mô tả một vài tính chất thường dùng.


### Cực tiểu cục bộ là cực tiểu toàn cục

Trước hết và quan trọng nhất, các cực tiểu cục bộ của hàm lồi cũng là các cực tiểu toàn cục. 
Ta có thể chứng minh điều này bằng phản chứng như sau.

Xét một hàm lồi $f$ được xác định trên một tập lồi $\mathcal{X}$.
Giả sử $x^{\ast} \in \mathcal{X}$ là một cực tiểu cục bộ:
tồn tại một giá trị dương nhỏ $p$ sao cho với $x \in \mathcal{X}$ thỏa mãn $0 < |x - x^{\ast}| \leq p$, ta có $f(x^{\ast}) < f(x)$.

Giả sử cực tiểu cục bộ $x^{\ast}$
không phải là cực tiểu toàn cục của $f$:
tồn tại $x' \in \mathcal{X}$ sao cho $f(x') < f(x^{\ast})$. 
Cũng tồn tại 
$\lambda \in [0, 1)$ chẳng hạn như $\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$
sao cho
$0 < |\lambda x^{\ast} + (1-\lambda) x' - x^{\ast}| \leq p$. 

Tuy nhiên,
theo định nghĩa của hàm lồi, ta có

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &= f(x^{\ast}),
\end{aligned}$$

điều này mâu thuẫn với phát biểu rằng $x^{\ast}$ là một cực tiểu cục bộ.
Do đó, không tồn tại $x' \in \mathcal{X}$ sao cho $f(x') < f(x^{\ast})$. Cực tiểu cục bộ $x^{\ast}$ cũng là cực tiểu toàn cục.

Ví dụ, hàm lồi $f(x) = (x-1)^2$ có một cực tiểu cục bộ tại $x=1$, đồng thời cũng là cực tiểu toàn cục.

```{.python .input}
#@tab all
f = lambda x: (x - 1) ** 2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

Việc các cực tiểu cục bộ của hàm lồi cũng là các cực tiểu toàn cục là rất thuận tiện. 
Điều này có nghĩa rằng khi ta tối thiểu hóa các hàm, ta không thể “mắc kẹt”. 
Tuy nhiên, lưu ý rằng điều này không có nghĩa là không thể có nhiều hơn một cực tiểu toàn cục, hoặc thậm chí có thể không tồn tại cực tiểu nào. Ví dụ, hàm $f(x) = \mathrm{max}(|x|-1, 0)$ đạt giá trị nhỏ nhất trên khoảng $[-1, 1]$. Ngược lại, hàm $f(x) = \exp(x)$ không đạt giá trị nhỏ nhất trên $\mathbb{R}$: khi $x \to -\infty$, nó tiệm cận về $0$, nhưng không có $x$ nào để $f(x) = 0$.

### Tập dưới của hàm lồi là lồi

Ta có thể thuận tiện
định nghĩa các tập lồi
thông qua *tập dưới* của các hàm lồi.
Cụ thể,
với một hàm lồi $f$ được xác định trên tập lồi $\mathcal{X}$,
bất kỳ tập dưới nào

$$\mathcal{S}_b \stackrel{\textrm{def}}{=} \{x | x \in \mathcal{X} \textrm{ and } f(x) \leq b\}$$

đều lồi. 

Hãy chứng minh nhanh điều này. Nhắc lại rằng với mọi $x, x' \in \mathcal{S}_b$, ta cần chỉ ra rằng $\lambda x + (1-\lambda) x' \in \mathcal{S}_b$ miễn là $\lambda \in [0, 1]$. 
Vì $f(x) \leq b$ và $f(x') \leq b$,
theo định nghĩa tính lồi, ta có 

$$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b.$$


### Tính lồi và đạo hàm bậc hai

Bất cứ khi nào đạo hàm bậc hai của một hàm $f: \mathbb{R}^n \rightarrow \mathbb{R}$ tồn tại, việc kiểm tra xem $f$ có lồi hay không là rất dễ dàng. 
Tất cả những gì ta cần làm là kiểm tra liệu Hessian của $f$ có nửa xác định dương hay không: $\nabla^2f \succeq 0$, tức là, 
ký hiệu ma trận Hessian $\nabla^2f$ là $\mathbf{H}$,
$\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$
với mọi $\mathbf{x} \in \mathbb{R}^n$.
Chẳng hạn, hàm $f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2$ là lồi vì $\nabla^2 f = \mathbf{1}$, tức là Hessian của nó là một ma trận đơn vị.


Về mặt hình thức, một hàm một chiều khả vi hai lần $f: \mathbb{R} \rightarrow \mathbb{R}$ là lồi
khi và chỉ khi đạo hàm bậc hai của nó $f'' \geq 0$. Với mọi hàm đa chiều khả vi hai lần $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$,
nó là lồi khi và chỉ khi Hessian $\nabla^2f \succeq 0$.

Trước hết, ta cần chứng minh trường hợp một chiều.
Để thấy rằng 
tính lồi của $f$ kéo theo 
$f'' \geq 0$, ta sử dụng thực tế rằng

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

Vì đạo hàm bậc hai được cho bởi giới hạn của các sai phân hữu hạn, suy ra rằng

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

Để thấy rằng 
$f'' \geq 0$ kéo theo $f$ là lồi,
ta sử dụng thực tế rằng $f'' \geq 0$ kéo theo $f'$ là một hàm đơn điệu không giảm. Gọi $a < x < b$ là ba điểm trong $\mathbb{R}$,
trong đó $x = (1-\lambda)a + \lambda b$ và $\lambda \in (0, 1)$.
Theo định lý giá trị trung bình,
tồn tại $\alpha \in [a, x]$ và $\beta \in [x, b]$
sao cho

$$f'(\alpha) = \frac{f(x) - f(a)}{x-a} \textrm{ and } f'(\beta) = \frac{f(b) - f(x)}{b-x}.$$


Theo tính đơn điệu, $f'(\beta) \geq f'(\alpha)$, do đó

$$\frac{x-a}{b-a}f(b) + \frac{b-x}{b-a}f(a) \geq f(x).$$

Vì $x = (1-\lambda)a + \lambda b$,
ta có

$$\lambda f(b) + (1-\lambda)f(a) \geq f((1-\lambda)a + \lambda b),$$

qua đó chứng minh tính lồi.

Tiếp theo, ta cần một bổ đề trước khi 
chứng minh trường hợp đa chiều:
$f: \mathbb{R}^n \rightarrow \mathbb{R}$
là lồi khi và chỉ khi với mọi $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$

$$g(z) \stackrel{\textrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y}) \textrm{ where } z \in [0,1]$$ 

là lồi.

Để chứng minh rằng tính lồi của $f$ kéo theo $g$ là lồi,
ta có thể chỉ ra rằng với mọi $a, b, \lambda \in [0, 1]$ (do đó
$0 \leq \lambda a + (1-\lambda) b \leq 1$)

$$\begin{aligned} &g(\lambda a + (1-\lambda) b)\\
=&f\left(\left(\lambda a + (1-\lambda) b\right)\mathbf{x} + \left(1-\lambda a - (1-\lambda) b\right)\mathbf{y} \right)\\
=&f\left(\lambda \left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) \left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \right)\\
\leq& \lambda f\left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) f\left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \\
=& \lambda g(a) + (1-\lambda) g(b).
\end{aligned}$$

Để chứng minh chiều ngược lại,
ta có thể chỉ ra rằng với 
mọi $\lambda \in [0, 1]$ 

$$\begin{aligned} &f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y})\\
=&g(\lambda \cdot 1 + (1-\lambda) \cdot 0)\\
\leq& \lambda g(1)  + (1-\lambda) g(0) \\
=& \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y}).
\end{aligned}$$


Cuối cùng,
sử dụng bổ đề ở trên và kết quả của trường hợp một chiều,
trường hợp đa chiều
có thể được chứng minh như sau.
Một hàm đa chiều $f: \mathbb{R}^n \rightarrow \mathbb{R}$ là lồi
khi và chỉ khi với mọi $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ $g(z) \stackrel{\textrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y})$, trong đó $z \in [0,1]$,
là lồi.
Theo trường hợp một chiều,
điều này đúng khi và chỉ khi
$g'' = (\mathbf{x} - \mathbf{y})^\top \mathbf{H}(\mathbf{x} - \mathbf{y}) \geq 0$ ($\mathbf{H} \stackrel{\textrm{def}}{=} \nabla^2f$)
với mọi $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$,
điều này tương đương với $\mathbf{H} \succeq 0$
theo định nghĩa về ma trận nửa xác định dương.


## Ràng buộc

Một trong những đặc tính hay của tối ưu hóa lồi là nó cho phép chúng ta xử lý các ràng buộc một cách hiệu quả. Nghĩa là, nó cho phép ta giải các bài toán *tối ưu hóa có ràng buộc* có dạng:

$$\begin{aligned} \mathop{\textrm{minimize~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \textrm{ subject to } & c_i(\mathbf{x}) \leq 0 \textrm{ for all } i \in \{1, \ldots, n\},
\end{aligned}$$

trong đó $f$ là hàm mục tiêu và các hàm $c_i$ là các hàm ràng buộc. Để thấy điều này có ý nghĩa gì, xét trường hợp $c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$. Trong trường hợp này, các tham số $\mathbf{x}$ bị ràng buộc trong quả cầu đơn vị. Nếu ràng buộc thứ hai là $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$, thì điều này tương ứng với mọi $\mathbf{x}$ nằm trong một nửa không gian. Thỏa mãn đồng thời cả hai ràng buộc tương đương với việc chọn một lát cắt của một quả cầu.

### Lagrangian

Nhìn chung, giải một bài toán tối ưu hóa có ràng buộc là khó. Một cách tiếp cận xuất phát từ vật lý với trực giác khá đơn giản. Hãy tưởng tượng một quả bóng bên trong một chiếc hộp. Quả bóng sẽ lăn đến vị trí thấp nhất và lực hấp dẫn sẽ được cân bằng bởi các lực mà các mặt của hộp có thể tác động lên quả bóng. Tóm lại, gradient của hàm mục tiêu (tức là trọng lực) sẽ được bù bởi gradient của hàm ràng buộc (quả bóng cần ở trong hộp nhờ các thành hộp “đẩy ngược lại”). 
Lưu ý rằng một số ràng buộc có thể không hoạt động:
các thành hộp không chạm vào quả bóng
sẽ không thể tác động bất kỳ lực nào lên quả bóng.


Bỏ qua phép suy dẫn của *Lagrangian* $L$,
lập luận trên
có thể được biểu diễn thông qua bài toán tối ưu hóa điểm yên ngựa sau:

$$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n) = f(\mathbf{x}) + \sum_{i=1}^n \alpha_i c_i(\mathbf{x}) \textrm{ where } \alpha_i \geq 0.$$

Ở đây, các biến $\alpha_i$ ($i=1,\ldots,n$) là những *bội số Lagrange* đảm bảo rằng các ràng buộc được thực thi đúng cách. Chúng được chọn vừa đủ lớn để đảm bảo $c_i(\mathbf{x}) \leq 0$ với mọi $i$. Chẳng hạn, với bất kỳ $\mathbf{x}$ nào mà $c_i(\mathbf{x}) < 0$ một cách tự nhiên, ta sẽ chọn $\alpha_i = 0$. Hơn nữa, đây là một bài toán tối ưu hóa điểm yên ngựa, trong đó ta muốn *tối đa hóa* $L$ theo tất cả $\alpha_i$ và đồng thời *tối thiểu hóa* nó theo $\mathbf{x}$. Có rất nhiều tài liệu giải thích cách suy ra hàm $L(\mathbf{x}, \alpha_1, \ldots, \alpha_n)$. Với mục đích của chúng ta, chỉ cần biết rằng điểm yên ngựa của $L$ là nơi bài toán tối ưu hóa có ràng buộc ban đầu được giải một cách tối ưu.

### Phạt

Một cách để thỏa mãn các bài toán tối ưu hóa có ràng buộc, ít nhất là *xấp xỉ*, là điều chỉnh Lagrangian $L$. 
Thay vì thỏa mãn $c_i(\mathbf{x}) \leq 0$, ta chỉ cần thêm $\alpha_i c_i(\mathbf{x})$ vào hàm mục tiêu $f(x)$. Điều này đảm bảo rằng các ràng buộc sẽ không bị vi phạm quá nghiêm trọng.

Thực tế, chúng ta đã luôn sử dụng thủ thuật này. Hãy xét weight decay trong :numref:`sec_weight_decay`. Trong đó, ta thêm $\frac{\lambda}{2} \|\mathbf{w}\|^2$ vào hàm mục tiêu để đảm bảo rằng $\mathbf{w}$ không tăng quá lớn. Theo quan điểm tối ưu hóa có ràng buộc, ta có thể thấy rằng điều này sẽ đảm bảo $\|\mathbf{w}\|^2 - r^2 \leq 0$ với một bán kính $r$ nào đó. Điều chỉnh giá trị của $\lambda$ cho phép ta thay đổi kích thước của $\mathbf{w}$.

Nhìn chung, thêm các khoản phạt là một cách tốt để đảm bảo sự thỏa mãn ràng buộc xấp xỉ. Trên thực tế, điều này tỏ ra ổn định hơn nhiều so với việc thỏa mãn chính xác. Hơn nữa, đối với các bài toán không lồi, nhiều đặc tính khiến cách tiếp cận chính xác trở nên hấp dẫn trong trường hợp lồi (ví dụ: tính tối ưu) không còn đúng nữa.

### Phép chiếu

Một chiến lược thay thế để thỏa mãn các ràng buộc là phép chiếu. Một lần nữa, chúng ta đã gặp chúng trước đây, ví dụ khi xử lý gradient clipping trong :numref:`sec_rnn-scratch`. Ở đó, chúng ta đảm bảo rằng gradient có độ dài bị chặn bởi $\theta$ thông qua

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, \theta/\|\mathbf{g}\|).$$

Điều này hóa ra là một *phép chiếu* của $\mathbf{g}$ lên quả cầu có bán kính $\theta$. Tổng quát hơn, phép chiếu lên một tập lồi $\mathcal{X}$ được định nghĩa là

$$\textrm{Proj}_\mathcal{X}(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in \mathcal{X}} \|\mathbf{x} - \mathbf{x}'\|,$$

tức là điểm gần $\mathbf{x}$ nhất trong $\mathcal{X}$. 

![Phép Chiếu Lồi.](../img/projections.svg)
:label:`fig_projections`

Định nghĩa toán học của phép chiếu có thể nghe hơi trừu tượng. :numref:`fig_projections` giải thích điều này rõ ràng hơn một chút. Trong đó, chúng ta có hai tập lồi, một hình tròn và một hình thoi. 
Các điểm nằm bên trong cả hai tập (màu vàng) không thay đổi trong quá trình chiếu. 
Các điểm nằm ngoài cả hai tập (màu đen) được chiếu tới 
các điểm nằm trong tập (màu đỏ) gần nhất với các điểm ban đầu (màu đen).
Mặc dù với các quả cầu $\ell_2$, điều này giữ nguyên hướng, nhưng nói chung không nhất thiết phải như vậy, như có thể thấy trong trường hợp hình thoi.


Một trong những ứng dụng của phép chiếu lồi là tính toán các vector trọng số thưa. Trong trường hợp này, chúng ta chiếu các vector trọng số lên một quả cầu $\ell_1$,
đây là phiên bản tổng quát của trường hợp hình thoi trong :numref:`fig_projections`.


## Tóm tắt

Trong ngữ cảnh deep learning, mục đích chính của các hàm lồi là tạo cơ sở cho các thuật toán tối ưu hóa và giúp chúng ta hiểu chúng một cách chi tiết. Tiếp theo, chúng ta sẽ xem cách gradient descent và stochastic gradient descent có thể được suy ra tương ứng.


* Giao của các tập lồi là lồi. Hợp của chúng thì không.
* Kỳ vọng của một hàm lồi không nhỏ hơn hàm lồi của một kỳ vọng (bất đẳng thức Jensen).
* Một hàm khả vi hai lần là lồi khi và chỉ khi Hessian của nó (một ma trận các đạo hàm cấp hai) là nửa xác định dương.
* Các ràng buộc lồi có thể được thêm vào thông qua Lagrangian. Trong thực tế, ta có thể đơn giản thêm chúng dưới dạng một penalty vào hàm mục tiêu.
* Phép chiếu ánh xạ tới các điểm trong tập lồi gần nhất với các điểm ban đầu.

## Bài tập

1. Giả sử rằng chúng ta muốn kiểm tra tính lồi của một tập bằng cách vẽ tất cả các đoạn thẳng giữa các điểm trong tập và kiểm tra xem các đoạn thẳng đó có nằm trong tập hay không.
    1. Chứng minh rằng chỉ cần kiểm tra các điểm trên biên là đủ.
    1. Chứng minh rằng chỉ cần kiểm tra các đỉnh của tập là đủ.
1. Ký hiệu $\mathcal{B}_p[r] \stackrel{\textrm{def}}{=} \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \textrm{ and } \|\mathbf{x}\|_p \leq r\}$ là quả cầu bán kính $r$ sử dụng chuẩn $p$. Chứng minh rằng $\mathcal{B}_p[r]$ là lồi với mọi $p \geq 1$.
1. Với các hàm lồi $f$ và $g$, hãy chỉ ra rằng $\mathrm{max}(f, g)$ cũng lồi. Chứng minh rằng $\mathrm{min}(f, g)$ không lồi.
1. Chứng minh rằng phép chuẩn hóa của hàm softmax là lồi. Cụ thể hơn, hãy chứng minh tính lồi của
    $f(x) = \log \sum_i \exp(x_i)$.
1. Chứng minh rằng các không gian con tuyến tính, tức là $\mathcal{X} = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$, là các tập lồi.
1. Chứng minh rằng trong trường hợp các không gian con tuyến tính với $\mathbf{b} = \mathbf{0}$, phép chiếu $\textrm{Proj}_\mathcal{X}$ có thể được viết là $\mathbf{M} \mathbf{x}$ với một ma trận $\mathbf{M}$ nào đó.
1. Chỉ ra rằng với các hàm lồi khả vi hai lần $f$, ta có thể viết $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$ với một $\xi \in [0, \epsilon]$ nào đó.
1. Với một tập lồi $\mathcal{X}$ và hai vector $\mathbf{x}$ và $\mathbf{y}$, hãy chứng minh rằng phép chiếu không bao giờ làm tăng khoảng cách, tức là $\|\mathbf{x} - \mathbf{y}\| \geq \|\textrm{Proj}_\mathcal{X}(\mathbf{x}) - \textrm{Proj}_\mathcal{X}(\mathbf{y})\|$.


[Thảo luận](https://discuss.d2l.ai/t/350)