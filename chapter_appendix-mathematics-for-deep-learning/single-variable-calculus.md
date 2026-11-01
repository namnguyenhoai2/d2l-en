# Giải tích một biến
:label:`sec_single_variable_calculus`

Trong :numref:`sec_calculus`, chúng ta đã tìm hiểu các thành phần cơ bản của giải tích vi phân. Phần này sẽ đi sâu hơn vào những nền tảng của giải tích và cách chúng ta có thể hiểu cũng như áp dụng chúng trong bối cảnh machine learning.

## Giải tích vi phân
Về bản chất, giải tích vi phân nghiên cứu cách các hàm số biến đổi dưới những thay đổi nhỏ. Để thấy vì sao đây là nền tảng quan trọng của deep learning, hãy cùng xem xét một ví dụ.

Giả sử chúng ta có một deep neural network, trong đó các trọng số được nối lại thành một vector duy nhất $\mathbf{w} = (w_1, \ldots, w_n)$ để thuận tiện. Với một tập dữ liệu huấn luyện, chúng ta xét loss của neural network trên tập dữ liệu này, được ký hiệu là $\mathcal{L}(\mathbf{w})$.

Hàm này cực kỳ phức tạp, vì nó mã hóa hiệu năng của mọi mô hình có thể có thuộc kiến trúc đã cho trên tập dữ liệu này. Do đó, gần như không thể xác định được tập trọng số $\mathbf{w}$ nào sẽ tối thiểu hóa loss. Vì vậy, trong thực tế, chúng ta thường bắt đầu bằng cách khởi tạo các trọng số một cách *ngẫu nhiên*, sau đó lặp đi lặp lại việc thực hiện những bước nhỏ theo hướng làm loss giảm nhanh nhất có thể.

Khi đó, câu hỏi đặt ra thoạt nhìn cũng không dễ hơn: làm thế nào để tìm được hướng khiến các trọng số giảm nhanh nhất có thể? Để tìm hiểu, trước hết hãy xét trường hợp chỉ có một trọng số: $L(\mathbf{w}) = L(x)$ với một giá trị thực duy nhất $x$.

Hãy lấy $x$ và cố gắng tìm hiểu điều gì xảy ra khi ta thay đổi nó một lượng nhỏ thành $x + \epsilon$. Để cụ thể hóa, hãy nghĩ đến một số như $\epsilon = 0.0000001$. Để giúp hình dung điều này, hãy vẽ đồ thị của một hàm ví dụ, $f(x) = \sin(x^x)$, trên khoảng $[0, 3]$.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Plot a function in a normal range
x_big = np.arange(0.01, 3.01, 0.01)
ys = np.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch

# Plot a function in a normal range
x_big = torch.arange(0.01, 3.01, 0.01)
ys = torch.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # Define pi in TensorFlow

# Plot a function in a normal range
x_big = tf.range(0.01, 3.01, 0.01)
ys = tf.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

Ở quy mô lớn này, hành vi của hàm không đơn giản. Tuy nhiên, nếu thu hẹp khoảng xét xuống một khoảng nhỏ hơn như $[1.75,2.25]$, ta thấy đồ thị trở nên đơn giản hơn nhiều.

```{.python .input}
#@tab mxnet
# Plot a the same function in a tiny range
x_med = np.arange(1.75, 2.25, 0.001)
ys = np.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Plot a the same function in a tiny range
x_med = torch.arange(1.75, 2.25, 0.001)
ys = torch.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Plot a the same function in a tiny range
x_med = tf.range(1.75, 2.25, 0.001)
ys = tf.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

Nếu tiếp tục đến mức cực đoan, khi phóng to một đoạn rất nhỏ, hành vi trở nên đơn giản hơn nhiều: nó chỉ là một đường thẳng.

```{.python .input}
#@tab mxnet
# Plot a the same function in a tiny range
x_small = np.arange(2.0, 2.01, 0.0001)
ys = np.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Plot a the same function in a tiny range
x_small = torch.arange(2.0, 2.01, 0.0001)
ys = torch.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Plot a the same function in a tiny range
x_small = tf.range(2.0, 2.01, 0.0001)
ys = tf.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

Đây là nhận xét then chốt của giải tích một biến: hành vi của các hàm quen thuộc có thể được mô hình hóa bằng một đường thẳng trong một khoảng đủ nhỏ. Điều này có nghĩa là với hầu hết các hàm, ta có thể kỳ vọng hợp lý rằng khi dịch chuyển giá trị $x$ của hàm đi một chút, đầu ra $f(x)$ cũng sẽ dịch chuyển một chút. Câu hỏi duy nhất cần trả lời là: “Mức thay đổi của đầu ra so với mức thay đổi của đầu vào lớn đến đâu? Nó bằng một nửa? Lớn gấp đôi?”

Vì vậy, ta có thể xét tỷ số giữa mức thay đổi trong đầu ra của một hàm khi đầu vào của hàm thay đổi một lượng nhỏ. Ta có thể viết điều này một cách hình thức như sau:

$$
\frac{L(x+\epsilon) - L(x)}{(x+\epsilon) - x} = \frac{L(x+\epsilon) - L(x)}{\epsilon}.
$$

Chừng đó đã đủ để bắt đầu thử nghiệm trong code. Chẳng hạn, giả sử ta biết rằng $L(x) = x^{2} + 1701(x-4)^3$, khi đó ta có thể xem giá trị này lớn đến đâu tại điểm $x = 4$ như sau.

```{.python .input}
#@tab all
# Define our function
def L(x):
    return x**2 + 1701*(x-4)**3

# Print the difference divided by epsilon for several epsilon
for epsilon in [0.1, 0.001, 0.0001, 0.00001]:
    print(f'epsilon = {epsilon:.5f} -> {(L(4+epsilon) - L(4)) / epsilon:.5f}')
```

Nếu quan sát kỹ, ta sẽ nhận thấy đầu ra của phép tính này gần một cách đáng ngờ với $8$. Quả thực, nếu giảm $\epsilon$, ta sẽ thấy giá trị này ngày càng tiến gần đến $8$. Vì vậy, ta có thể kết luận chính xác rằng giá trị cần tìm (mức độ mà một thay đổi trong đầu vào làm thay đổi đầu ra) tại điểm $x=4$ là $8$. Một nhà toán học biểu diễn sự thật này như sau:

$$
\lim_{\epsilon \rightarrow 0}\frac{L(4+\epsilon) - L(4)}{\epsilon} = 8.
$$

Một chút mở rộng mang tính lịch sử: trong vài thập kỷ đầu của nghiên cứu neural network, các nhà khoa học đã sử dụng thuật toán này (*phương pháp sai phân hữu hạn*) để đánh giá cách một hàm loss thay đổi dưới một nhiễu nhỏ: chỉ cần thay đổi các trọng số rồi xem loss thay đổi như thế nào. Cách này không hiệu quả về mặt tính toán, vì cần hai lần đánh giá hàm loss để xem một thay đổi của một biến ảnh hưởng đến loss như thế nào. Nếu thử làm điều này với chỉ vài nghìn tham số, ta sẽ cần vài nghìn lần đánh giá network trên toàn bộ tập dữ liệu! Vấn đề này chỉ được giải quyết vào năm 1986, khi *thuật toán backpropagation* được giới thiệu trong :citet:`Rumelhart.Hinton.Williams.ea.1988`, cung cấp một cách tính toán tác động của *bất kỳ* thay đổi đồng thời nào của các trọng số lên loss, với thời gian tính toán tương đương một lần dự đoán của network trên tập dữ liệu.

Quay lại ví dụ của chúng ta, giá trị $8$ này khác nhau với các giá trị khác nhau của $x$, vì vậy việc định nghĩa nó như một hàm của $x$ là hợp lý. Nói chính xác hơn, tốc độ thay đổi phụ thuộc vào giá trị này được gọi là *đạo hàm*, được viết như sau:

$$\frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon}.$$
:eqlabel:`eq_der_def`

Các tài liệu khác nhau có thể sử dụng những ký hiệu khác nhau cho đạo hàm. Chẳng hạn, tất cả các ký hiệu dưới đây đều biểu thị cùng một ý:

$$
\frac{df}{dx} = \frac{d}{dx}f = f' = \nabla_xf = D_xf = f_x.
$$

Hầu hết tác giả sẽ chọn một ký hiệu duy nhất và sử dụng nhất quán; tuy nhiên, ngay cả điều đó cũng không được đảm bảo. Tốt nhất là làm quen với tất cả các ký hiệu này. Trong suốt tài liệu này, chúng ta sẽ sử dụng ký hiệu $\frac{df}{dx}$, trừ khi muốn lấy đạo hàm của một biểu thức phức tạp; khi đó, chúng ta sẽ dùng $\frac{d}{dx}f$ để viết các biểu thức như
$$
\frac{d}{dx}\left[x^4+\cos\left(\frac{x^2+1}{2x-1}\right)\right].
$$

Trong nhiều trường hợp, việc khai triển lại định nghĩa đạo hàm :eqref:`eq_der_def` sẽ hữu ích về mặt trực giác để thấy một hàm thay đổi như thế nào khi ta thực hiện một thay đổi nhỏ của $x$:

$$\begin{aligned} \frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon} & \implies \frac{df}{dx}(x) \approx \frac{f(x+\epsilon) - f(x)}{\epsilon} \\ & \implies \epsilon \frac{df}{dx}(x) \approx f(x+\epsilon) - f(x) \\ & \implies f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x). \end{aligned}$$
:eqlabel:`eq_small_change`

Phương trình cuối cùng đáng được nhấn mạnh. Nó cho chúng ta biết rằng nếu lấy bất kỳ hàm nào và thay đổi đầu vào một lượng nhỏ, đầu ra sẽ thay đổi một lượng nhỏ đó được nhân với đạo hàm.

Theo cách này, ta có thể hiểu đạo hàm là hệ số tỷ lệ cho biết mức thay đổi của đầu ra do một thay đổi trong đầu vào.

## Các quy tắc của giải tích
:label:`sec_derivative_table`

Bây giờ chúng ta chuyển sang nhiệm vụ tìm hiểu cách tính đạo hàm của một hàm tường minh. Một trình bày hình thức đầy đủ về giải tích sẽ suy ra mọi thứ từ những nguyên lý đầu tiên. Tuy nhiên, ở đây chúng ta sẽ không đi theo hướng đó, mà thay vào đó sẽ cung cấp cách hiểu về những quy tắc phổ biến thường gặp.

### Các đạo hàm thường gặp
Như đã thấy trong :numref:`sec_calculus`, khi tính đạo hàm, ta thường có thể sử dụng một loạt quy tắc để rút gọn phép tính về một vài hàm cốt lõi. Chúng tôi nhắc lại các quy tắc này để tiện tham khảo.

* **Đạo hàm của hằng số.** $\frac{d}{dx}c = 0$.
* **Đạo hàm của hàm tuyến tính.** $\frac{d}{dx}(ax) = a$.
* **Quy tắc lũy thừa.** $\frac{d}{dx}x^n = nx^{n-1}$.
* **Đạo hàm của hàm mũ.** $\frac{d}{dx}e^x = e^x$.
* **Đạo hàm của logarithm.** $\frac{d}{dx}\log(x) = \frac{1}{x}$.

### Các quy tắc đạo hàm
Nếu phải tính riêng và lưu trữ từng đạo hàm cần thiết trong một bảng, giải tích vi phân sẽ gần như không thể thực hiện. Toán học đã mang đến một điều hữu ích: chúng ta có thể khái quát các đạo hàm trên và tính những đạo hàm phức tạp hơn, chẳng hạn tìm đạo hàm của $f(x) = \log\left(1+(x-1)^{10}\right)$. Như đã đề cập trong :numref:`sec_calculus`, chìa khóa để làm điều đó là hệ thống hóa những gì xảy ra khi kết hợp các hàm theo nhiều cách khác nhau, quan trọng nhất là: tổng, tích và hợp.

* **Quy tắc tổng.** $\frac{d}{dx}\left(g(x) + h(x)\right) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$.
* **Quy tắc tích.** $\frac{d}{dx}\left(g(x)\cdot h(x)\right) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$.
* **Quy tắc chuỗi.** $\frac{d}{dx}g(h(x)) = \frac{dg}{dh}(h(x))\cdot \frac{dh}{dx}(x)$.

Hãy xem chúng ta có thể sử dụng :eqref:`eq_small_change` để hiểu các quy tắc này như thế nào. Với quy tắc tổng, hãy xét chuỗi lập luận sau:

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon) + h(x+\epsilon) \\
& \approx g(x) + \epsilon \frac{dg}{dx}(x) + h(x) + \epsilon \frac{dh}{dx}(x) \\
& = g(x) + h(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right) \\
& = f(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right).
\end{aligned}
$$

Bằng cách so sánh kết quả này với thực tế rằng $f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x)$, ta thấy rằng $\frac{df}{dx}(x) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$ như mong muốn. Trực giác ở đây là: khi thay đổi đầu vào $x$, $g$ và $h$ cùng đóng góp vào sự thay đổi của đầu ra lần lượt là $\frac{dg}{dx}(x)$ và $\frac{dh}{dx}(x)$.


Tích phức tạp hơn và đòi hỏi một quan sát mới về cách xử lý các biểu thức này. Trước hết, ta sẽ bắt đầu như trước bằng cách sử dụng :eqref:`eq_small_change`:

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon)\cdot h(x+\epsilon) \\
& \approx \left(g(x) + \epsilon \frac{dg}{dx}(x)\right)\cdot\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& = g(x)\cdot h(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x) \\
& = f(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x). \\
\end{aligned}
$$


Điều này tương tự như phép tính ở trên, và quả thực ta thấy đáp án của mình ($\frac{df}{dx}(x) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$) nằm cạnh $\epsilon$, nhưng có một vấn đề là số hạng có bậc $\epsilon^{2}$. Ta sẽ gọi đây là một *số hạng bậc cao hơn*, vì lũy thừa của $\epsilon^2$ cao hơn lũy thừa của $\epsilon^1$. Trong một phần sau, ta sẽ thấy rằng đôi khi chúng ta muốn theo dõi các số hạng này; tuy nhiên, hiện tại hãy lưu ý rằng nếu $\epsilon = 0.0000001$, thì $\epsilon^{2}= 0.0000000000001$, nhỏ hơn rất nhiều. Khi cho $\epsilon \rightarrow 0$, ta có thể an toàn bỏ qua các số hạng bậc cao hơn. Theo quy ước chung trong phụ lục này, ta sẽ dùng "$\approx$" để biểu thị rằng hai số hạng bằng nhau sai khác các số hạng bậc cao hơn. Tuy nhiên, nếu muốn chặt chẽ hơn, ta có thể xét thương sai phân

$$
\frac{f(x+\epsilon) - f(x)}{\epsilon} = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x) + \epsilon \frac{dg}{dx}(x)\frac{dh}{dx}(x),
$$

và thấy rằng khi cho $\epsilon \rightarrow 0$, số hạng bên phải cũng tiến về không.

Cuối cùng, với chain rule, ta lại có thể tiến hành như trước bằng cách sử dụng :eqref:`eq_small_change` và thấy rằng

$$
\begin{aligned}
f(x+\epsilon) & = g(h(x+\epsilon)) \\
& \approx g\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& \approx g(h(x)) + \epsilon \frac{dh}{dx}(x) \frac{dg}{dh}(h(x))\\
& = f(x) + \epsilon \frac{dg}{dh}(h(x))\frac{dh}{dx}(x),
\end{aligned}
$$

trong đó, ở dòng thứ hai, ta xem hàm $g$ có đầu vào ($h(x)$) bị dịch chuyển bởi một lượng rất nhỏ $\epsilon \frac{dh}{dx}(x)$.

Các quy tắc này cung cấp cho chúng ta một tập công cụ linh hoạt để tính toán về cơ bản mọi biểu thức mong muốn. Chẳng hạn,

$$
\begin{aligned}
\frac{d}{dx}\left[\log\left(1+(x-1)^{10}\right)\right] & = \left(1+(x-1)^{10}\right)^{-1}\frac{d}{dx}\left[1+(x-1)^{10}\right]\\
& = \left(1+(x-1)^{10}\right)^{-1}\left(\frac{d}{dx}[1] + \frac{d}{dx}[(x-1)^{10}]\right) \\
& = \left(1+(x-1)^{10}\right)^{-1}\left(0 + 10(x-1)^9\frac{d}{dx}[x-1]\right) \\
& = 10\left(1+(x-1)^{10}\right)^{-1}(x-1)^9 \\
& = \frac{10(x-1)^9}{1+(x-1)^{10}}.
\end{aligned}
$$

Trong đó, mỗi dòng đã sử dụng các quy tắc sau:

1. Chain rule và đạo hàm của logarithm.
2. Quy tắc tổng.
3. Đạo hàm của hằng số, chain rule và quy tắc lũy thừa.
4. Quy tắc tổng, đạo hàm của hàm tuyến tính, đạo hàm của hằng số.

Sau khi thực hiện ví dụ này, có hai điều cần thấy rõ:

1. Mọi hàm mà ta có thể viết bằng cách sử dụng tổng, tích, hằng số, lũy thừa, hàm mũ và logarithm đều có thể được tính đạo hàm một cách máy móc bằng cách làm theo các quy tắc này.
2. Để một người thực hiện các quy tắc này có thể rất tẻ nhạt và dễ xảy ra sai sót!

May mắn thay, hai sự thật này cùng gợi ý một hướng đi: đây là một ứng viên hoàn hảo cho việc cơ giới hóa! Quả thực, backpropagation, nội dung mà chúng ta sẽ quay lại sau trong phần này, chính là như vậy.

### Xấp xỉ tuyến tính
Khi làm việc với đạo hàm, việc diễn giải về mặt hình học phép xấp xỉ được sử dụng ở trên thường rất hữu ích. Cụ thể, hãy lưu ý rằng phương trình

$$
f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x),
$$

xấp xỉ giá trị của $f$ bằng một đường thẳng đi qua điểm $(x, f(x))$ và có hệ số góc $\frac{df}{dx}(x)$. Theo cách này, ta nói rằng đạo hàm cung cấp một phép xấp xỉ tuyến tính cho hàm $f$, như minh họa dưới đây:

```{.python .input}
#@tab mxnet
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Compute some linear approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) *
                 torch.cos(torch.tensor(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Compute sin
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) *
                 tf.cos(tf.constant(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

### Đạo hàm bậc cao hơn

Bây giờ hãy làm một điều thoạt nhìn có vẻ kỳ lạ. Lấy một hàm $f$ và tính đạo hàm $\frac{df}{dx}$. Điều này cho ta tốc độ thay đổi của $f$ tại mọi điểm.

Tuy nhiên, đạo hàm, $\frac{df}{dx}$, có thể được xem như một hàm, vì vậy không có gì ngăn cản ta tính đạo hàm của $\frac{df}{dx}$ để nhận được $\frac{d^2f}{dx^2} = \frac{df}{dx}\left(\frac{df}{dx}\right)$. Ta sẽ gọi đây là đạo hàm bậc hai của $f$. Hàm này là tốc độ thay đổi của tốc độ thay đổi của $f$, hay nói cách khác, là cách tốc độ thay đổi đang thay đổi. Ta có thể áp dụng đạo hàm nhiều lần để thu được cái gọi là đạo hàm bậc $n$. Để ký hiệu gọn gàng, ta sẽ biểu diễn đạo hàm bậc $n$ như sau:

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$

Hãy thử tìm hiểu *tại sao* đây lại là một khái niệm hữu ích. Dưới đây, ta minh họa $f^{(2)}(x)$, $f^{(1)}(x)$ và $f(x)$.

Trước hết, xét trường hợp đạo hàm bậc hai $f^{(2)}(x)$ là một hằng số dương. Điều này có nghĩa là hệ số góc của đạo hàm bậc nhất là dương. Do đó, đạo hàm bậc nhất $f^{(1)}(x)$ có thể bắt đầu bằng giá trị âm, trở thành bằng không tại một điểm, rồi cuối cùng trở thành dương. Điều này cho ta biết hệ số góc của hàm ban đầu $f$ và do đó, bản thân hàm $f$ giảm, phẳng dần, rồi tăng. Nói cách khác, hàm $f$ cong lên và có một cực tiểu duy nhất, như được minh họa trong :numref:`fig_positive-second`.

![Nếu giả sử đạo hàm bậc hai là một hằng số dương, thì đạo hàm bậc nhất tăng, điều này cho thấy bản thân hàm có một cực tiểu.](../img/posSecDer.svg)
:label:`fig_positive-second`


Thứ hai, nếu đạo hàm bậc hai là một hằng số âm, điều đó có nghĩa là đạo hàm bậc nhất đang giảm. Điều này ngụ ý rằng đạo hàm bậc nhất có thể bắt đầu bằng giá trị dương, trở thành bằng không tại một điểm, rồi trở thành âm. Do đó, bản thân hàm $f$ tăng, phẳng dần, rồi giảm. Nói cách khác, hàm $f$ cong xuống và có một cực đại duy nhất, như được minh họa trong :numref:`fig_negative-second`.

![Nếu giả sử đạo hàm bậc hai là một hằng số âm, thì đạo hàm bậc nhất giảm, điều này cho thấy bản thân hàm có một cực đại.](../img/negSecDer.svg)
:label:`fig_negative-second`


Thứ ba, nếu đạo hàm bậc hai luôn bằng không, thì đạo hàm bậc nhất sẽ không bao giờ thay đổi---nó là một hằng số! Điều này có nghĩa là $f$ tăng (hoặc giảm) với một tốc độ cố định, và bản thân $f$ là một đường thẳng, như được minh họa trong :numref:`fig_zero-second`.

![Nếu giả sử đạo hàm cấp hai bằng không, thì đạo hàm cấp một là hằng số, điều này cho thấy bản thân hàm số là một đường thẳng.](../img/zeroSecDer.svg)
:label:`fig_zero-second`

Tóm lại, đạo hàm cấp hai có thể được hiểu là mô tả cách hàm số $f$ uốn cong. Đạo hàm cấp hai dương dẫn đến một đường cong hướng lên, trong khi đạo hàm cấp hai âm có nghĩa là $f$ cong xuống, còn đạo hàm cấp hai bằng không có nghĩa là $f$ hoàn toàn không uốn cong.

Hãy tiến thêm một bước. Xét hàm số $g(x) = ax^{2}+ bx + c$. Khi đó, ta có thể tính được

$$
\begin{aligned}
\frac{dg}{dx}(x) & = 2ax + b \\
\frac{d^2g}{dx^2}(x) & = 2a.
\end{aligned}
$$

Nếu đã có một hàm số ban đầu $f(x)$, ta có thể tính hai đạo hàm đầu tiên và tìm các giá trị của $a, b$ và $c$ sao cho chúng khớp với phép tính này. Tương tự như phần trước, nơi ta thấy rằng đạo hàm cấp một cho phép xấp xỉ tốt nhất bằng một đường thẳng, cách xây dựng này cung cấp phép xấp xỉ tốt nhất bằng một đa thức bậc hai. Hãy trực quan hóa điều này với $f(x) = \sin(x)$.

```{.python .input}
#@tab mxnet
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0) -
                              (xs - x0)**2 * np.sin(x0) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) *
                 torch.cos(torch.tensor(x0)) - (xs - x0)**2 *
                 torch.sin(torch.tensor(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Compute sin
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) *
                 tf.cos(tf.constant(x0)) - (xs - x0)**2 *
                 tf.sin(tf.constant(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

Trong phần tiếp theo, chúng ta sẽ mở rộng ý tưởng này thành ý tưởng về một *chuỗi Taylor*.

### Chuỗi Taylor


*Chuỗi Taylor* cung cấp một phương pháp để xấp xỉ hàm số $f(x)$ nếu ta biết các giá trị của $n$ đạo hàm đầu tiên tại một điểm $x_0$, tức là $\left\{ f(x_0), f^{(1)}(x_0), f^{(2)}(x_0), \ldots, f^{(n)}(x_0) \right\}$. Ý tưởng là tìm một đa thức bậc $n$ khớp với tất cả các đạo hàm đã cho tại $x_0$.

Ở phần trước, ta đã xét trường hợp $n=2$, và một chút biến đổi đại số cho thấy

$$
f(x) \approx \frac{1}{2}\frac{d^2f}{dx^2}(x_0)(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

Như có thể thấy ở trên, mẫu số $2$ được dùng để triệt tiêu thừa số $2$ xuất hiện khi ta lấy hai đạo hàm của $x^2$, trong khi các số hạng khác đều bằng không. Lập luận tương tự cũng áp dụng cho đạo hàm cấp một và giá trị của chính hàm số.

Nếu tiếp tục lập luận này đến $n=3$, ta sẽ kết luận rằng

$$
f(x) \approx \frac{\frac{d^3f}{dx^3}(x_0)}{6}(x-x_0)^3 + \frac{\frac{d^2f}{dx^2}(x_0)}{2}(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

trong đó $6 = 3 \times 2 = 3!$ xuất phát từ hằng số đứng trước khi ta lấy ba đạo hàm của $x^3$.


Hơn nữa, ta có thể có được một đa thức bậc $n$ bằng cách

$$
P_n(x) = \sum_{i = 0}^{n} \frac{f^{(i)}(x_0)}{i!}(x-x_0)^{i}.
$$

trong đó ký hiệu

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$


Thật vậy, $P_n(x)$ có thể được xem là phép xấp xỉ đa thức bậc $n$ tốt nhất cho hàm số $f(x)$ của chúng ta.

Mặc dù chúng ta sẽ không đi sâu vào sai số của các phép xấp xỉ trên, nhưng đáng nói đến trường hợp giới hạn vô hạn. Trong trường hợp này, với các hàm số có tính chất tốt (được gọi là hàm giải tích thực) như $\cos(x)$ hoặc $e^{x}$, ta có thể viết ra vô hạn số hạng và xấp xỉ chính xác cùng một hàm số

$$
f(x) = \sum_{n = 0}^\infty \frac{f^{(n)}(x_0)}{n!}(x-x_0)^{n}.
$$

Hãy lấy $f(x) = e^{x}$ làm ví dụ. Vì $e^{x}$ là đạo hàm của chính nó, ta biết rằng $f^{(n)}(x) = e^{x}$. Do đó, $e^{x}$ có thể được tái tạo bằng cách lấy chuỗi Taylor tại $x_0 = 0$, tức là

$$
e^{x} = \sum_{n = 0}^\infty \frac{x^{n}}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots.
$$

Hãy xem cách thức hoạt động của điều này trong code và quan sát cách việc tăng bậc của phép xấp xỉ Taylor đưa chúng ta đến gần hơn với hàm số mong muốn $e^x$.

```{.python .input}
#@tab mxnet
# Compute the exponential function
xs = np.arange(0, 3, 0.01)
ys = np.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab pytorch
# Compute the exponential function
xs = torch.arange(0, 3, 0.01)
ys = torch.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab tensorflow
# Compute the exponential function
xs = tf.range(0, 3, 0.01)
ys = tf.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

Chuỗi Taylor có hai ứng dụng chính:

1. *Ứng dụng lý thuyết*: Thông thường, khi cố gắng tìm hiểu một hàm số quá phức tạp, việc sử dụng chuỗi Taylor cho phép chúng ta chuyển nó thành một đa thức mà ta có thể trực tiếp xử lý.

2. *Ứng dụng số*: Một số hàm như $e^{x}$ hoặc $\cos(x)$ rất khó để máy tính tính toán. Máy có thể lưu các bảng giá trị với độ chính xác cố định (và điều này thường được thực hiện), nhưng vẫn còn những câu hỏi như “Chữ số thứ 1000 của $\cos(1)$ là gì?” Chuỗi Taylor thường hữu ích để trả lời những câu hỏi như vậy.


## Tóm tắt

* Đạo hàm có thể được dùng để diễn tả cách hàm số thay đổi khi ta thay đổi đầu vào một lượng nhỏ.
* Các đạo hàm cơ bản có thể được kết hợp bằng các quy tắc đạo hàm để tạo ra các đạo hàm phức tạp tùy ý.
* Có thể lấy đạo hàm lặp đi lặp lại để thu được đạo hàm cấp hai hoặc cấp cao hơn. Mỗi lần tăng cấp cung cấp thông tin chi tiết hơn về hành vi của hàm số.
* Sử dụng thông tin từ các đạo hàm của một ví dụ dữ liệu duy nhất, ta có thể xấp xỉ các hàm số có tính chất tốt bằng những đa thức thu được từ chuỗi Taylor.


## Bài tập

1. Đạo hàm của $x^3-4x+1$ là gì?
2. Đạo hàm của $\log(\frac{1}{x})$ là gì?
3. Đúng hay Sai: Nếu $f'(x) = 0$ thì $f$ đạt cực đại hoặc cực tiểu tại $x$?
4. Cực tiểu của $f(x) = x\log(x)$ nằm ở đâu với $x\ge0$ (trong đó ta giả sử rằng $f$ nhận giá trị giới hạn bằng $0$ tại $f(0)$)?


:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/412)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1088)
:end_tab:


:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1089)
:end_tab: