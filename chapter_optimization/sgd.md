# Hạ Gradient Ngẫu nhiên
:label:`sec_sgd`

Trong các chương trước, chúng ta liên tục sử dụng hạ gradient ngẫu nhiên trong quy trình huấn luyện, nhưng chưa giải thích tại sao nó hoạt động.
Để làm rõ vấn đề này,
chúng ta vừa mô tả các nguyên lý cơ bản của hạ gradient
trong :numref:`sec_gd`.
Trong phần này, chúng ta sẽ tiếp tục thảo luận về
*hạ gradient ngẫu nhiên* một cách chi tiết hơn.

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
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

## Cập nhật Gradient Ngẫu nhiên

Trong deep learning, hàm mục tiêu thường là trung bình các hàm mất mát của từng mẫu trong tập dữ liệu huấn luyện.
Với một tập dữ liệu huấn luyện gồm $n$ mẫu,
chúng ta giả sử rằng $f_i(\mathbf{x})$ là hàm mất mát
đối với mẫu huấn luyện có chỉ số $i$,
trong đó $\mathbf{x}$ là vector tham số.
Khi đó, ta có hàm mục tiêu

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

Gradient của hàm mục tiêu tại $\mathbf{x}$ được tính như sau

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

Nếu sử dụng hạ gradient, chi phí tính toán cho mỗi lần lặp của biến độc lập là $\mathcal{O}(n)$, tăng tuyến tính theo $n$. Do đó, khi tập dữ liệu huấn luyện lớn hơn, chi phí của hạ gradient cho mỗi lần lặp cũng sẽ cao hơn.

Hạ gradient ngẫu nhiên (SGD) giảm chi phí tính toán ở mỗi lần lặp. Tại mỗi lần lặp của hạ gradient ngẫu nhiên, ta lấy mẫu ngẫu nhiên đồng đều một chỉ số $i\in\{1,\ldots, n\}$ cho các mẫu dữ liệu, rồi tính gradient $\nabla f_i(\mathbf{x})$ để cập nhật $\mathbf{x}$:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

trong đó $\eta$ là learning rate. Ta có thể thấy rằng chi phí tính toán cho mỗi lần lặp giảm từ $\mathcal{O}(n)$ của hạ gradient xuống hằng số $\mathcal{O}(1)$. Hơn nữa, chúng tôi muốn nhấn mạnh rằng gradient ngẫu nhiên $\nabla f_i(\mathbf{x})$ là một ước lượng không chệch của gradient đầy đủ $\nabla f(\mathbf{x})$ vì

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

Điều này có nghĩa là, xét trung bình, gradient ngẫu nhiên là một ước lượng tốt của gradient.

Bây giờ, chúng ta sẽ so sánh nó với hạ gradient bằng cách thêm nhiễu ngẫu nhiên có trung bình bằng 0 và phương sai bằng 1 vào gradient để mô phỏng hạ gradient ngẫu nhiên.

```{.python .input}
#@tab all
def f(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Gradient of the objective function
    return 2 * x1, 4 * x2
```

```{.python .input}
#@tab mxnet
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab pytorch
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab tensorflow
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab all
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Như có thể thấy, quỹ đạo của các biến trong hạ gradient ngẫu nhiên nhiễu hơn nhiều so với quỹ đạo chúng ta quan sát được ở hạ gradient trong :numref:`sec_gd`. Điều này là do tính ngẫu nhiên của gradient. Nghĩa là, ngay cả khi đến gần điểm cực tiểu, chúng ta vẫn chịu ảnh hưởng của sự bất định do gradient tức thời thông qua $\eta \nabla f_i(\mathbf{x})$ đưa vào. Ngay cả sau 50 bước, chất lượng vẫn chưa tốt. Tệ hơn nữa, nó sẽ không cải thiện sau các bước bổ sung (chúng tôi khuyến khích bạn thử nghiệm với số bước lớn hơn để xác nhận điều này). Điều này chỉ để lại một lựa chọn: thay đổi learning rate $\eta$. Tuy nhiên, nếu chọn nó quá nhỏ, ban đầu chúng ta sẽ không đạt được tiến triển đáng kể nào. Mặt khác, nếu chọn nó quá lớn, chúng ta sẽ không nhận được một nghiệm tốt, như đã thấy ở trên. Cách duy nhất để giải quyết các mục tiêu mâu thuẫn này là giảm learning rate *một cách động* khi quá trình tối ưu hóa tiến triển.

Đây cũng là lý do thêm hàm learning rate `lr` vào hàm bước `sgd`. Trong ví dụ trên, mọi chức năng lập lịch learning rate đều không hoạt động vì ta đặt hàm `lr` liên quan là hằng số.

## Learning Rate Động

Việc thay $\eta$ bằng learning rate phụ thuộc thời gian $\eta(t)$ làm tăng độ phức tạp trong việc kiểm soát sự hội tụ của thuật toán tối ưu hóa. Cụ thể, chúng ta cần xác định $\eta$ nên giảm nhanh đến mức nào. Nếu quá nhanh, chúng ta sẽ dừng tối ưu hóa quá sớm. Nếu giảm quá chậm, chúng ta lãng phí quá nhiều thời gian cho việc tối ưu hóa. Sau đây là một vài chiến lược cơ bản được dùng để điều chỉnh $\eta$ theo thời gian (chúng ta sẽ thảo luận các chiến lược nâng cao hơn sau):

$$
\begin{aligned}
    \eta(t) & = \eta_i \textrm{ if } t_i \leq t \leq t_{i+1}  && \textrm{piecewise constant} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \textrm{exponential decay} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \textrm{polynomial decay}
\end{aligned}
$$

Trong kịch bản *hằng từng đoạn* đầu tiên, chúng ta giảm learning rate, chẳng hạn mỗi khi tiến trình tối ưu hóa bị chững lại. Đây là một chiến lược phổ biến để huấn luyện các mạng sâu. Hoặc, chúng ta có thể giảm nó quyết liệt hơn nhiều bằng *phân rã mũ*. Đáng tiếc là điều này thường dẫn đến dừng sớm trước khi thuật toán hội tụ. Một lựa chọn phổ biến là *phân rã đa thức* với $\alpha = 0.5$. Trong trường hợp tối ưu hóa lồi, có một số chứng minh cho thấy tốc độ này có tính chất tốt.

Hãy xem phân rã mũ trông như thế nào trong thực tế.

```{.python .input}
#@tab all
def exponential_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

Đúng như dự kiến, phương sai trong các tham số giảm đáng kể. Tuy nhiên, điều này phải đánh đổi bằng việc không hội tụ đến nghiệm tối ưu $\mathbf{x} = (0, 0)$. Ngay cả sau 1000 bước lặp, chúng ta vẫn còn rất xa nghiệm tối ưu. Thật vậy, thuật toán hoàn toàn không hội tụ. Mặt khác, nếu sử dụng phân rã đa thức, trong đó learning rate giảm theo nghịch đảo căn bậc hai của số bước, sự hội tụ sẽ tốt hơn chỉ sau 50 bước.

```{.python .input}
#@tab all
def polynomial_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Còn có nhiều lựa chọn khác để thiết lập learning rate. Chẳng hạn, chúng ta có thể bắt đầu với một tốc độ nhỏ, sau đó tăng nhanh rồi lại giảm, dù chậm hơn. Chúng ta thậm chí có thể luân phiên giữa các learning rate nhỏ hơn và lớn hơn. Có rất nhiều loại schedule như vậy. Hiện tại, hãy tập trung vào các schedule learning rate mà việc phân tích lý thuyết toàn diện là khả thi, tức là các learning rate trong bối cảnh lồi. Đối với các bài toán không lồi tổng quát, rất khó để có được các đảm bảo hội tụ có ý nghĩa, vì nhìn chung việc tối thiểu hóa các bài toán không lồi phi tuyến là NP hard. Để xem một khảo sát, hãy tham khảo, chẳng hạn, [ghi chú bài giảng](https://www.stat.cmu.edu/%7Eryantibs/convexopt-F15/lectures/26-nonconvex.pdf) xuất sắc của Tibshirani 2015.



## Phân tích Hội tụ cho các Mục tiêu Lồi

Phân tích hội tụ sau đây của hạ gradient ngẫu nhiên đối với các hàm mục tiêu lồi
là tùy chọn và chủ yếu nhằm cung cấp thêm trực giác về bài toán.
Chúng ta giới hạn ở một trong những chứng minh đơn giản nhất :cite:`Nesterov.Vial.2000`.
Các kỹ thuật chứng minh nâng cao hơn đáng kể vẫn tồn tại, chẳng hạn khi hàm mục tiêu có tính chất đặc biệt tốt.


Giả sử hàm mục tiêu $f(\boldsymbol{\xi}, \mathbf{x})$ lồi theo $\mathbf{x}$
với mọi $\boldsymbol{\xi}$.
Cụ thể hơn,
chúng ta xét cập nhật hạ gradient ngẫu nhiên:

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$

trong đó $f(\boldsymbol{\xi}_t, \mathbf{x})$
là hàm mục tiêu
đối với mẫu huấn luyện $\boldsymbol{\xi}_t$
được rút từ một phân phối nào đó
ở bước $t$ và $\mathbf{x}$ là tham số mô hình.
Ký hiệu

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$

là rủi ro kỳ vọng và $R^*$ là giá trị nhỏ nhất của nó theo $\mathbf{x}$. Cuối cùng, gọi $\mathbf{x}^*$ là điểm tối thiểu hóa (chúng ta giả sử nó tồn tại trong miền mà $\mathbf{x}$ được xác định). Trong trường hợp này, ta có thể theo dõi khoảng cách giữa tham số hiện tại $\mathbf{x}_t$ tại thời điểm $t$ và điểm tối thiểu hóa rủi ro $\mathbf{x}^*$ để xem liệu nó có cải thiện theo thời gian hay không:

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$
:eqlabel:`eq_sgd-xt+1-xstar`

Chúng ta giả sử rằng chuẩn $\ell_2$ của gradient ngẫu nhiên $\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$ bị chặn bởi một hằng số $L$, do đó ta có

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$
:eqlabel:`eq_sgd-L`


Ta chủ yếu quan tâm đến việc khoảng cách giữa $\mathbf{x}_t$ và $\mathbf{x}^*$ thay đổi như thế nào *trong kỳ vọng*. Thực tế, với bất kỳ chuỗi bước cụ thể nào, khoảng cách hoàn toàn có thể tăng lên, tùy thuộc vào $\boldsymbol{\xi}_t$ mà ta gặp phải. Vì vậy, ta cần chặn tích vô hướng.
Do với mọi hàm lồi $f$, ta có
$f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$
với mọi $\mathbf{x}$ và $\mathbf{y}$,
theo tính lồi, ta có

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$
:eqlabel:`eq_sgd-f-xi-xstar`

Thay cả hai bất đẳng thức :eqref:`eq_sgd-L` và :eqref:`eq_sgd-f-xi-xstar` vào :eqref:`eq_sgd-xt+1-xstar`, ta thu được một cận cho khoảng cách giữa các tham số tại thời điểm $t+1$ như sau:

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$
:eqlabel:`eqref_sgd-xt-diff`

Điều này có nghĩa là ta đạt được tiến triển miễn là chênh lệch giữa loss hiện tại và loss tối ưu lớn hơn $\eta_t L^2/2$. Vì chênh lệch này chắc chắn hội tụ về không, suy ra learning rate $\eta_t$ cũng cần *giảm về không*.

Tiếp theo, ta lấy kỳ vọng của :eqref:`eqref_sgd-xt-diff`. Điều này cho

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$

Bước cuối cùng bao gồm việc cộng các bất đẳng thức với $t \in \{1, \ldots, T\}$. Vì tổng là tổng triệt tiêu và ta bỏ đi hạng tử dưới, ta thu được

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$
:eqlabel:`eq_sgd-x1-xstar`

Lưu ý rằng ta đã sử dụng việc $\mathbf{x}_1$ được cho trước, do đó có thể bỏ kỳ vọng. Cuối cùng, định nghĩa

$$\bar{\mathbf{x}} \stackrel{\textrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$

Vì

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$

theo bất đẳng thức Jensen (đặt $i=t$, $\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$ trong :eqref:`eq_jensens-inequality`) và tính lồi của $R$, suy ra $E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$, do đó

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$

Thay điều này vào bất đẳng thức :eqref:`eq_sgd-x1-xstar` cho ta cận

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

trong đó $r^2 \stackrel{\textrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$ là một cận của khoảng cách giữa lựa chọn tham số ban đầu và kết quả cuối cùng. Tóm lại, tốc độ hội tụ phụ thuộc vào cách chuẩn của gradient ngẫu nhiên bị chặn ($L$) và giá trị tham số ban đầu cách điểm tối ưu bao xa ($r$). Lưu ý rằng cận được biểu diễn theo $\bar{\mathbf{x}}$ thay vì $\mathbf{x}_T$. Điều này là do $\bar{\mathbf{x}}$ là một phiên bản được làm trơn của quỹ đạo tối ưu hóa.
Khi biết $r$, $L$ và $T$, ta có thể chọn learning rate $\eta = r/(L \sqrt{T})$. Điều này cho cận trên là $rL/\sqrt{T}$. Nghĩa là, ta hội tụ với tốc độ $\mathcal{O}(1/\sqrt{T})$ về nghiệm tối ưu.





## Gradient ngẫu nhiên và mẫu hữu hạn

Cho đến nay, khi nói về stochastic gradient descent, ta đã trình bày hơi đơn giản hóa. Ta giả định rằng mình lấy các mẫu $x_i$, thường kèm theo nhãn $y_i$, từ một phân phối $p(x, y)$ và sử dụng chúng để cập nhật các tham số mô hình theo một cách nào đó. Cụ thể, với kích thước mẫu hữu hạn, ta chỉ lập luận rằng phân phối rời rạc $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$
với một số hàm $\delta_{x_i}$ và $\delta_{y_i}$
cho phép ta thực hiện stochastic gradient descent trên đó.

Tuy nhiên, đây không thực sự là điều ta đã làm. Trong các ví dụ minh họa ở phần hiện tại, ta chỉ thêm nhiễu vào một gradient vốn không ngẫu nhiên, tức là ta giả vờ có các cặp $(x_i, y_i)$. Hóa ra điều này được biện minh ở đây (xem phần bài tập để biết thảo luận chi tiết). Đáng lo hơn là trong tất cả các thảo luận trước đó, rõ ràng ta đã không làm như vậy. Thay vào đó, ta duyệt qua toàn bộ các mẫu *chính xác một lần*. Để thấy vì sao cách này tốt hơn, hãy xét trường hợp ngược lại, tức là ta lấy mẫu $n$ quan sát từ phân phối rời rạc *có hoàn lại*. Xác suất chọn ngẫu nhiên phần tử $i$ là $1/n$. Do đó, xác suất chọn nó *ít nhất* một lần là

$$P(\textrm{choose~} i) = 1 - P(\textrm{omit~} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$

Lập luận tương tự cho thấy xác suất chọn một mẫu nào đó (tức là ví dụ huấn luyện) *chính xác một lần* được cho bởi

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$

Lấy mẫu có hoàn lại dẫn đến phương sai tăng và hiệu quả sử dụng dữ liệu giảm so với lấy mẫu *không hoàn lại*. Do đó, trong thực tế, ta thực hiện cách sau (và đây là lựa chọn mặc định trong toàn bộ cuốn sách này). Cuối cùng, lưu ý rằng các lần duyệt lặp lại qua tập dữ liệu huấn luyện sẽ đi qua tập này theo một thứ tự ngẫu nhiên *khác*.


## Tóm tắt

* Với các bài toán lồi, ta có thể chứng minh rằng stochastic gradient descent sẽ hội tụ về nghiệm tối ưu với nhiều lựa chọn learning rate khác nhau.
* Với deep learning, điều này nói chung không đúng. Tuy nhiên, việc phân tích các bài toán lồi mang lại cho ta hiểu biết hữu ích về cách tiếp cận tối ưu hóa, cụ thể là giảm learning rate dần dần, nhưng không quá nhanh.
* Vấn đề xảy ra khi learning rate quá nhỏ hoặc quá lớn. Trong thực tế, một learning rate phù hợp thường chỉ được tìm thấy sau nhiều lần thử nghiệm.
* Khi tập dữ liệu huấn luyện có nhiều ví dụ hơn, việc tính toán mỗi vòng lặp của gradient descent tốn kém hơn, vì vậy stochastic gradient descent được ưu tiên trong các trường hợp này.
* Các đảm bảo về tính tối ưu của stochastic gradient descent nói chung không có sẵn trong các trường hợp không lồi vì số lượng cực tiểu cục bộ cần kiểm tra có thể là cấp số mũ.




## Bài tập

1. Thử nghiệm với các lịch learning rate khác nhau cho stochastic gradient descent và với các số vòng lặp khác nhau. Cụ thể, hãy vẽ khoảng cách đến nghiệm tối ưu $(0, 0)$ theo số vòng lặp.
1. Chứng minh rằng với hàm $f(x_1, x_2) = x_1^2 + 2 x_2^2$, việc thêm nhiễu chuẩn vào gradient tương đương với việc tối thiểu hóa hàm loss $f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$, trong đó $\mathbf{x}$ được lấy từ một phân phối chuẩn.
1. So sánh sự hội tụ của stochastic gradient descent khi bạn lấy mẫu từ $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ có hoàn lại và khi lấy mẫu không hoàn lại.
1. Bạn sẽ thay đổi solver stochastic gradient descent như thế nào nếu một gradient nào đó (hay đúng hơn là một tọa độ liên kết với nó) luôn lớn hơn tất cả các gradient khác?
1. Giả sử $f(x) = x^2 (1 + \sin x)$. $f$ có bao nhiêu cực tiểu cục bộ? Bạn có thể thay đổi $f$ sao cho để tối thiểu hóa nó, cần phải đánh giá tất cả các cực tiểu cục bộ không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/352)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/497)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1067)
:end_tab: