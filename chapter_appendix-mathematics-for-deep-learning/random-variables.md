# Biến ngẫu nhiên
:label:`sec_random_variables`

Trong :numref:`sec_prob`, chúng ta đã tìm hiểu những kiến thức cơ bản về cách làm việc với các biến ngẫu nhiên rời rạc; trong trường hợp của chúng ta, đó là những biến ngẫu nhiên nhận một tập hữu hạn các giá trị có thể có hoặc nhận các số nguyên. Trong phần này, chúng ta sẽ phát triển lý thuyết về *biến ngẫu nhiên liên tục*, tức là những biến ngẫu nhiên có thể nhận bất kỳ giá trị thực nào.

## Biến ngẫu nhiên liên tục

Biến ngẫu nhiên liên tục là một chủ đề tinh tế hơn đáng kể so với biến ngẫu nhiên rời rạc. Một phép tương tự khá phù hợp là bước nhảy về mặt kỹ thuật tương đương với bước nhảy từ việc cộng các danh sách số sang việc lấy tích phân các hàm. Vì vậy, chúng ta sẽ cần dành thời gian để phát triển lý thuyết này.

### Từ rời rạc đến liên tục

Để hiểu những thách thức kỹ thuật bổ sung khi làm việc với các biến ngẫu nhiên liên tục, hãy thực hiện một thí nghiệm tưởng tượng. Giả sử chúng ta đang phóng phi tiêu vào bia và muốn biết xác suất phi tiêu chạm chính xác vào vị trí cách tâm bia $2 \textrm{cm}$.

Trước tiên, hãy tưởng tượng rằng chúng ta đo với độ chính xác một chữ số, tức là chia thành các khoảng cho $0 \textrm{cm}$, $1 \textrm{cm}$, $2 \textrm{cm}$, v.v. Chúng ta phóng, chẳng hạn, $100$ phi tiêu vào bia, và nếu có $20$ chiếc rơi vào khoảng $2\textrm{cm}$, chúng ta kết luận rằng $20\%$ số phi tiêu được phóng chạm vào bia ở vị trí cách tâm $2 \textrm{cm}$.

Tuy nhiên, khi xem xét kỹ hơn, điều này không khớp với câu hỏi của chúng ta! Chúng ta muốn sự bằng nhau chính xác, trong khi các khoảng này chứa tất cả những phi tiêu rơi vào khoảng từ $1.5\textrm{cm}$ đến $2.5\textrm{cm}$.

Không nản chí, chúng ta tiếp tục đi xa hơn. Chúng ta đo chính xác hơn nữa, chẳng hạn $1.9\textrm{cm}$, $2.0\textrm{cm}$, $2.1\textrm{cm}$, và giờ thấy rằng có lẽ $3$ trong số $100$ phi tiêu rơi vào ô $2.0\textrm{cm}$. Vì vậy, chúng ta kết luận xác suất là $3\%$.

Tuy nhiên, điều này không giải quyết được gì cả! Chúng ta chỉ đẩy vấn đề xuống thêm một chữ số. Hãy khái quát hóa một chút. Hãy tưởng tượng chúng ta biết xác suất mà $k$ chữ số đầu tiên khớp với $2.00000\ldots$ và muốn biết xác suất khớp ở $k+1$ chữ số đầu tiên. Có thể hợp lý khi giả sử chữ số thứ ${k+1}^{\textrm{th}}$ về cơ bản là một lựa chọn ngẫu nhiên từ tập $\{0, 1, 2, \ldots, 9\}$. Ít nhất, chúng ta không thể hình dung một quá trình có ý nghĩa về mặt vật lý nào buộc số micromet cách tâm phải có xu hướng kết thúc bằng $7$ thay vì $3$.

Điều này có nghĩa là về bản chất, mỗi chữ số chính xác bổ sung mà chúng ta yêu cầu sẽ làm giảm xác suất khớp đi một hệ số $10$. Nói cách khác, chúng ta kỳ vọng rằng

$$
P(\textrm{distance is}\; 2.00\ldots, \;\textrm{to}\; k \;\textrm{digits} ) \approx p\cdot10^{-k}.
$$

Giá trị $p$ về cơ bản mã hóa điều xảy ra với một vài chữ số đầu tiên, còn $10^{-k}$ xử lý phần còn lại.

Lưu ý rằng nếu biết vị trí chính xác đến $k=4$ chữ số sau dấu thập phân, điều đó có nghĩa là giá trị nằm trong khoảng, chẳng hạn, $[1.99995,2.00005]$, là một khoảng có độ dài $2.00005-1.99995 = 10^{-4}$. Do đó, nếu gọi độ dài của khoảng này là $\epsilon$, chúng ta có thể nói rằng

$$
P(\textrm{distance is in an}\; \epsilon\textrm{-sized interval around}\; 2 ) \approx \epsilon \cdot p.
$$

Hãy tiến thêm một bước cuối cùng. Suốt thời gian qua, chúng ta chỉ suy nghĩ về điểm $2$ mà chưa nghĩ đến các điểm khác. Về cơ bản thì không có gì khác biệt, nhưng giá trị $p$ có thể sẽ khác. Ít nhất, chúng ta kỳ vọng người phóng phi tiêu có khả năng chạm vào một điểm gần tâm, chẳng hạn $2\textrm{cm}$, cao hơn so với $20\textrm{cm}$. Vì vậy, giá trị $p$ không cố định mà phụ thuộc vào điểm $x$. Điều này cho chúng ta kỳ vọng rằng

$$P(\textrm{distance is in an}\; \epsilon \textrm{-sized interval around}\; x ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_deriv`

Thật vậy, :eqref:`eq_pdf_deriv` chính là định nghĩa của *hàm mật độ xác suất*. Đây là một hàm $p(x)$ mã hóa xác suất tương đối của việc chạm gần điểm này so với điểm khác. Hãy hình dung xem một hàm như vậy có thể trông như thế nào.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Plot the probability density function for some random variable
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2)/np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2)/np.sqrt(2 * np.pi)

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch

# Plot the probability density function for some random variable
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi)) + \
    0.8*torch.exp(-(x + 1)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi))

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # Define pi in TensorFlow

# Plot the probability density function for some random variable
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi)) + \
    0.8*tf.exp(-(x + 1)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi))

d2l.plot(x, p, 'x', 'Density')
```

Những vị trí mà giá trị hàm lớn cho biết các vùng mà chúng ta có nhiều khả năng tìm thấy giá trị ngẫu nhiên hơn. Các phần thấp là những vùng mà chúng ta ít có khả năng tìm thấy giá trị ngẫu nhiên.

### Hàm mật độ xác suất

Bây giờ hãy khảo sát vấn đề này sâu hơn. Chúng ta đã thấy trực giác về hàm mật độ xác suất của một biến ngẫu nhiên $X$, cụ thể là hàm mật độ là một hàm $p(x)$ sao cho

$$P(X \; \textrm{is in an}\; \epsilon \textrm{-sized interval around}\; x ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_def`

Nhưng điều này kéo theo những tính chất nào của $p(x)$?

Thứ nhất, xác suất không bao giờ âm, vì vậy chúng ta cũng kỳ vọng $p(x) \ge 0$.

Thứ hai, hãy tưởng tượng chúng ta chia $\mathbb{R}$ thành vô hạn khoảng có độ rộng $\epsilon$, chẳng hạn các khoảng $(\epsilon\cdot i, \epsilon \cdot (i+1)]$. Với mỗi khoảng như vậy, từ :eqref:`eq_pdf_def` chúng ta biết xác suất xấp xỉ là

$$
P(X \; \textrm{is in an}\; \epsilon\textrm{-sized interval around}\; x ) \approx \epsilon \cdot p(\epsilon \cdot i),
$$

vì vậy tổng của tất cả các khoảng phải là

$$
P(X\in\mathbb{R}) \approx \sum_i \epsilon \cdot p(\epsilon\cdot i).
$$

Đây không gì khác hơn là phép xấp xỉ một tích phân đã được thảo luận trong :numref:`sec_integral_calculus`, do đó chúng ta có thể nói rằng

$$
P(X\in\mathbb{R}) = \int_{-\infty}^{\infty} p(x) \; dx.
$$

Chúng ta biết rằng $P(X\in\mathbb{R}) = 1$, vì biến ngẫu nhiên chắc chắn phải nhận *một* số nào đó. Do đó, với mọi hàm mật độ,

$$
\int_{-\infty}^{\infty} p(x) \; dx = 1.
$$

Thật vậy, khi tìm hiểu sâu hơn, chúng ta thấy rằng với mọi $a$ và $b$,

$$
P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.
$$

Trong code, chúng ta có thể xấp xỉ giá trị này bằng cách sử dụng các phương pháp xấp xỉ rời rạc như trước. Trong trường hợp này, chúng ta có thể xấp xỉ xác suất rơi vào vùng màu xanh.

```{.python .input}
#@tab mxnet
# Approximate probability using numerical integration
epsilon = 0.01
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2) / np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2) / np.sqrt(2 * np.pi)

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {np.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab pytorch
# Approximate probability using numerical integration
epsilon = 0.01
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi)) +\
    0.8*torch.exp(-(x + 1)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {torch.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab tensorflow
# Approximate probability using numerical integration
epsilon = 0.01
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi)) +\
    0.8*tf.exp(-(x + 1)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.numpy().tolist()[300:800], p.numpy().tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {tf.reduce_sum(epsilon*p[300:800])}'
```

Hóa ra hai tính chất này mô tả chính xác không gian các hàm mật độ xác suất có thể có (hay *p.d.f.*, cách viết tắt thường gặp). Đó là các hàm không âm $p(x) \ge 0$ sao cho

$$\int_{-\infty}^{\infty} p(x) \; dx = 1.$$
:eqlabel:`eq_pdf_int_one`

Chúng ta diễn giải hàm này bằng cách sử dụng tích phân để thu được xác suất biến ngẫu nhiên nằm trong một khoảng cụ thể:

$$P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.$$
:eqlabel:`eq_pdf_int_int`

Trong :numref:`sec_distributions`, chúng ta sẽ thấy một số phân phối phổ biến, nhưng hãy tiếp tục làm việc ở mức trừu tượng.

### Hàm phân phối tích lũy

Trong phần trước, chúng ta đã tìm hiểu khái niệm p.d.f. Trong thực tế, đây là một phương pháp thường gặp để thảo luận về các biến ngẫu nhiên liên tục, nhưng nó có một nhược điểm đáng kể: các giá trị của p.d.f. bản thân chúng không phải là xác suất, mà là một hàm cần được lấy tích phân để cho ra xác suất. Không có gì sai khi mật độ lớn hơn $10$, miễn là nó không lớn hơn $10$ trên một khoảng có độ dài lớn hơn $1/10$. Điều này có thể phản trực giác, vì vậy người ta thường cũng suy nghĩ theo khái niệm *hàm phân phối tích lũy*, hay c.d.f.; hàm này *là* một xác suất.

Cụ thể, bằng cách sử dụng :eqref:`eq_pdf_int_int`, chúng ta định nghĩa c.d.f. của biến ngẫu nhiên $X$ có mật độ $p(x)$ là

$$
F(x) = \int _ {-\infty}^{x} p(x) \; dx = P(X \le x).
$$

Hãy xem xét một vài tính chất.

* $F(x) \rightarrow 0$ khi $x\rightarrow -\infty$.
* $F(x) \rightarrow 1$ khi $x\rightarrow \infty$.
* $F(x)$ không giảm ($y > x \implies F(y) \ge F(x)$).
* $F(x)$ liên tục (không có bước nhảy) nếu $X$ là một biến ngẫu nhiên liên tục.

Đối với ý thứ tư, lưu ý rằng điều này sẽ không đúng nếu $X$ là biến rời rạc, chẳng hạn nhận các giá trị $0$ và $1$, mỗi giá trị có xác suất $1/2$. Khi đó

$$
F(x) = \begin{cases}
0 & x < 0, \\
\frac{1}{2} & x < 1, \\
1 & x \ge 1.
\end{cases}
$$

Trong ví dụ này, chúng ta thấy một trong những lợi ích của việc làm việc với c.d.f.: khả năng xử lý các biến ngẫu nhiên liên tục hoặc rời rạc trong cùng một framework, hoặc thậm chí các tổ hợp của cả hai (tung đồng xu: nếu là ngửa thì trả về kết quả gieo xúc xắc, nếu là sấp thì trả về khoảng cách từ điểm phóng phi tiêu đến tâm của bia).

### Giá trị trung bình

Giả sử chúng ta đang làm việc với một biến ngẫu nhiên $X$. Bản thân phân phối có thể khó diễn giải. Thường sẽ hữu ích nếu có thể tóm tắt ngắn gọn hành vi của một biến ngẫu nhiên. Những con số giúp chúng ta nắm bắt hành vi của một biến ngẫu nhiên được gọi là *thống kê tóm tắt*. Những thống kê thường gặp nhất là *giá trị trung bình*, *phương sai* và *độ lệch chuẩn*.

*Giá trị trung bình* biểu diễn giá trị trung bình của một biến ngẫu nhiên. Nếu chúng ta có một biến ngẫu nhiên rời rạc $X$, nhận các giá trị $x_i$ với các xác suất $p_i$, thì giá trị trung bình được cho bởi trung bình có trọng số: cộng các giá trị nhân với xác suất biến ngẫu nhiên nhận giá trị đó:

$$\mu_X = E[X] = \sum_i x_i p_i.$$
:eqlabel:`eq_exp_def`

Cách chúng ta nên diễn giải giá trị trung bình (dù cần thận trọng) là nó cho biết về cơ bản vị trí mà biến ngẫu nhiên có xu hướng nhận giá trị.

Là một ví dụ tối giản mà chúng ta sẽ xem xét trong suốt phần này, hãy cho $X$ là biến ngẫu nhiên nhận giá trị $a-2$ với xác suất $p$, $a+2$ với xác suất $p$ và $a$ với xác suất $1-2p$. Sử dụng :eqref:`eq_exp_def`, ta có thể tính rằng, với mọi lựa chọn khả dĩ của $a$ và $p$, giá trị trung bình là

$$
\mu_X = E[X] = \sum_i x_i p_i = (a-2)p + a(1-2p) + (a+2)p = a.
$$

Như vậy, ta thấy giá trị trung bình là $a$. Điều này phù hợp với trực giác, vì $a$ là vị trí mà quanh đó chúng ta đã lấy biến ngẫu nhiên làm tâm.

Vì các tính chất này hữu ích, hãy tóm tắt một vài tính chất.

* Với mọi biến ngẫu nhiên $X$ và các số $a$, $b$, ta có $\mu_{aX+b} = a\mu_X + b$.
* Nếu có hai biến ngẫu nhiên $X$ và $Y$, ta có $\mu_{X+Y} = \mu_X+\mu_Y$.

Giá trị trung bình hữu ích để hiểu hành vi trung bình của một biến ngẫu nhiên, tuy nhiên chỉ riêng giá trị trung bình vẫn chưa đủ để có được một hiểu biết trực giác đầy đủ. Kiếm $\$10 \pm \$1$ cho mỗi lần bán hàng rất khác với kiếm $\$10 \pm \$15$ cho mỗi lần bán hàng, dù chúng có cùng giá trị trung bình. Trường hợp thứ hai có mức dao động lớn hơn nhiều và do đó thể hiện rủi ro lớn hơn nhiều. Vì vậy, để hiểu hành vi của một biến ngẫu nhiên, ít nhất chúng ta cần thêm một đại lượng nữa: một thước đo cho biết biến ngẫu nhiên dao động rộng đến mức nào.

### Phương sai

Điều này dẫn chúng ta đến việc xem xét *phương sai* của một biến ngẫu nhiên. Đây là một thước đo định lượng cho biết một biến ngẫu nhiên lệch khỏi giá trị trung bình bao xa. Hãy xét biểu thức $X - \mu_X$. Đây là độ lệch của biến ngẫu nhiên so với giá trị trung bình của nó. Giá trị này có thể dương hoặc âm, vì vậy chúng ta cần làm gì đó để biến nó thành số dương, nhằm đo độ lớn của độ lệch.

Một cách hợp lý là xét $\left|X-\mu_X\right|$, và quả thực điều này dẫn đến một đại lượng hữu ích gọi là *độ lệch tuyệt đối trung bình* (mean absolute deviation). Tuy nhiên, do có mối liên hệ với các lĩnh vực khác của toán học và thống kê, người ta thường sử dụng một cách khác.

Cụ thể, họ xét $(X-\mu_X)^2.$ Nếu lấy giá trị trung bình của đại lượng này để xem kích thước điển hình của nó, ta thu được phương sai

$$\sigma_X^2 = \textrm{Var}(X) = E\left[(X-\mu_X)^2\right] = E[X^2] - \mu_X^2.$$
:eqlabel:`eq_var_def`

Đẳng thức cuối cùng trong :eqref:`eq_var_def` có được bằng cách khai triển định nghĩa ở giữa và áp dụng các tính chất của kỳ vọng.

Hãy xét lại ví dụ trong đó $X$ là biến ngẫu nhiên nhận giá trị $a-2$ với xác suất $p$, $a+2$ với xác suất $p$ và $a$ với xác suất $1-2p$. Trong trường hợp này, $\mu_X = a$, vì vậy tất cả những gì cần tính là $E\left[X^2\right]$. Ta có thể dễ dàng tính được:

$$
E\left[X^2\right] = (a-2)^2p + a^2(1-2p) + (a+2)^2p = a^2 + 8p.
$$

Do đó, theo :eqref:`eq_var_def`, phương sai của chúng ta là

$$
\sigma_X^2 = \textrm{Var}(X) = E[X^2] - \mu_X^2 = a^2 + 8p - a^2 = 8p.
$$

Kết quả này một lần nữa hợp lý. Giá trị lớn nhất của $p$ là $1/2$, tương ứng với việc dùng phép tung đồng xu để chọn $a-2$ hoặc $a+2$. Phương sai bằng $4$ trong trường hợp này phản ánh thực tế rằng cả $a-2$ và $a+2$ đều cách giá trị trung bình $2$ đơn vị, và $2^2 = 4$. Ở đầu kia của phổ, nếu $p=0$, biến ngẫu nhiên này luôn nhận giá trị $0$, nên hoàn toàn không có phương sai.

Dưới đây là một vài tính chất của phương sai:

* Với mọi biến ngẫu nhiên $X$, $\textrm{Var}(X) \ge 0$, trong đó $\textrm{Var}(X) = 0$ khi và chỉ khi $X$ là một hằng số.
* Với mọi biến ngẫu nhiên $X$ và các số $a$, $b$, ta có $\textrm{Var}(aX+b) = a^2\textrm{Var}(X)$.
* Nếu có hai biến ngẫu nhiên *độc lập* $X$ và $Y$, ta có $\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y)$.

Khi diễn giải các giá trị này, có thể xuất hiện một chút vướng mắc. Cụ thể, hãy thử hình dung điều gì xảy ra nếu chúng ta theo dõi đơn vị trong suốt phép tính. Giả sử chúng ta đang làm việc với số sao được gán cho một sản phẩm trên trang web. Khi đó, $a$, $a-2$ và $a+2$ đều được đo bằng đơn vị sao. Tương tự, giá trị trung bình $\mu_X$ cũng được đo bằng đơn vị sao (vì là một trung bình có trọng số). Tuy nhiên, khi đến phương sai, chúng ta ngay lập tức gặp vấn đề: ta muốn xét $(X-\mu_X)^2$, đại lượng có đơn vị là *sao bình phương*. Điều này có nghĩa là bản thân phương sai không thể so sánh trực tiếp với các phép đo ban đầu. Để phương sai dễ diễn giải, chúng ta cần quay lại đơn vị ban đầu.

### Độ lệch chuẩn

Thống kê tóm tắt này luôn có thể được suy ra từ phương sai bằng cách lấy căn bậc hai! Vì vậy, chúng ta định nghĩa *độ lệch chuẩn* là

$$
\sigma_X = \sqrt{\textrm{Var}(X)}.
$$

Trong ví dụ của chúng ta, điều này có nghĩa là độ lệch chuẩn là $\sigma_X = 2\sqrt{2p}$. Nếu đang làm việc với đơn vị sao trong ví dụ về đánh giá, $\sigma_X$ cũng có đơn vị sao.

Các tính chất của phương sai có thể được phát biểu lại cho độ lệch chuẩn.

* Với mọi biến ngẫu nhiên $X$, $\sigma_{X} \ge 0$.
* Với mọi biến ngẫu nhiên $X$ và các số $a$, $b$, ta có $\sigma_{aX+b} = |a|\sigma_{X}$
* Nếu có hai biến ngẫu nhiên *độc lập* $X$ và $Y$, ta có $\sigma_{X+Y} = \sqrt{\sigma_{X}^2 + \sigma_{Y}^2}$.

Lúc này, việc đặt câu hỏi “Nếu độ lệch chuẩn có cùng đơn vị với biến ngẫu nhiên ban đầu, liệu nó có biểu thị điều gì mà chúng ta có thể vẽ ra liên quan đến biến ngẫu nhiên đó không?” là hoàn toàn tự nhiên. Câu trả lời chắc chắn là có! Quả thực, cũng giống như giá trị trung bình cho biết vị trí điển hình của biến ngẫu nhiên, độ lệch chuẩn cho biết khoảng dao động điển hình của biến ngẫu nhiên đó. Chúng ta có thể phát biểu chặt chẽ điều này bằng bất đẳng thức Chebyshev:

$$P\left(X \not\in [\mu_X - \alpha\sigma_X, \mu_X + \alpha\sigma_X]\right) \le \frac{1}{\alpha^2}.$$
:eqlabel:`eq_chebyshev`

Nói bằng lời, trong trường hợp $\alpha=10$, $99\%$ các mẫu từ bất kỳ biến ngẫu nhiên nào đều nằm trong phạm vi $10$ độ lệch chuẩn tính từ giá trị trung bình. Điều này mang lại cách diễn giải trực tiếp cho các thống kê tóm tắt tiêu chuẩn của chúng ta.

Để thấy phát biểu này tinh tế như thế nào, hãy xem lại ví dụ đang xét, trong đó $X$ là biến ngẫu nhiên nhận giá trị $a-2$ với xác suất $p$, $a+2$ với xác suất $p$ và $a$ với xác suất $1-2p$. Ta đã thấy giá trị trung bình là $a$ và độ lệch chuẩn là $2\sqrt{2p}$. Điều này có nghĩa là, nếu áp dụng bất đẳng thức Chebyshev :eqref:`eq_chebyshev` với $\alpha = 2$, ta thấy biểu thức là

$$
P\left(X \not\in [a - 4\sqrt{2p}, a + 4\sqrt{2p}]\right) \le \frac{1}{4}.
$$

Điều này có nghĩa là $75\%$ thời gian, biến ngẫu nhiên này sẽ nằm trong khoảng trên với mọi giá trị của $p$. Bây giờ, hãy chú ý rằng khi $p \rightarrow 0$, khoảng này cũng hội tụ về điểm duy nhất $a$. Nhưng chúng ta biết biến ngẫu nhiên chỉ nhận các giá trị $a-2$, $a$ và $a+2$, nên cuối cùng chúng ta có thể chắc chắn rằng $a-2$ và $a+2$ sẽ nằm ngoài khoảng! Câu hỏi là điều đó xảy ra tại giá trị $p$ nào. Vì vậy, ta muốn giải: với giá trị $p$ nào thì $a+4\sqrt{2p} = a+2$; phương trình này được giải khi $p=1/8$. Đây *chính xác* là giá trị $p$ đầu tiên mà điều đó có thể xảy ra mà không vi phạm khẳng định rằng không quá $1/4$ số mẫu từ phân phối nằm ngoài khoảng ( $1/8$ ở bên trái và $1/8$ ở bên phải).

Hãy trực quan hóa điều này. Chúng ta sẽ biểu diễn xác suất nhận được ba giá trị bằng ba thanh dọc có chiều cao tỷ lệ với xác suất. Khoảng sẽ được vẽ bằng một đường ngang ở giữa. Biểu đồ đầu tiên cho thấy điều gì xảy ra khi $p > 1/8$, trong đó khoảng chứa an toàn tất cả các điểm.

```{.python .input}
#@tab mxnet
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * np.sqrt(2 * p),
                   a + 4 * np.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, 0.2)
```

```{.python .input}
#@tab pytorch
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * torch.sqrt(2 * p),
                   a + 4 * torch.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, torch.tensor(0.2))
```

```{.python .input}
#@tab tensorflow
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * tf.sqrt(2 * p),
                   a + 4 * tf.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * tf.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * tf.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, tf.constant(0.2))
```

Biểu đồ thứ hai cho thấy tại $p = 1/8$, khoảng vừa chạm đúng hai điểm. Điều này cho thấy bất đẳng thức là *chặt*, vì không thể chọn khoảng nhỏ hơn mà vẫn giữ cho bất đẳng thức đúng.

```{.python .input}
#@tab mxnet
# Plot interval when p = 1/8
plot_chebyshev(0.0, 0.125)
```

```{.python .input}
#@tab pytorch
# Plot interval when p = 1/8
plot_chebyshev(0.0, torch.tensor(0.125))
```

```{.python .input}
#@tab tensorflow
# Plot interval when p = 1/8
plot_chebyshev(0.0, tf.constant(0.125))
```

Biểu đồ thứ ba cho thấy khi $p < 1/8$, khoảng chỉ chứa điểm trung tâm. Điều này không làm bất đẳng thức mất hiệu lực, vì chúng ta chỉ cần bảo đảm rằng không quá $1/4$ xác suất nằm ngoài khoảng; điều đó có nghĩa là khi $p < 1/8$, có thể bỏ qua hai điểm $a-2$ và $a+2$.

```{.python .input}
#@tab mxnet
# Plot interval when p < 1/8
plot_chebyshev(0.0, 0.05)
```

```{.python .input}
#@tab pytorch
# Plot interval when p < 1/8
plot_chebyshev(0.0, torch.tensor(0.05))
```

```{.python .input}
#@tab tensorflow
# Plot interval when p < 1/8
plot_chebyshev(0.0, tf.constant(0.05))
```

### Giá trị trung bình và phương sai trong trường liên tục

Cho đến đây, mọi thứ đều được trình bày theo các biến ngẫu nhiên rời rạc, nhưng trường hợp biến ngẫu nhiên liên tục cũng tương tự. Để hiểu trực giác cách thức hoạt động, hãy tưởng tượng chia trục số thực thành các khoảng có độ dài $\epsilon$, được cho bởi $(\epsilon i, \epsilon (i+1)]$. Khi làm vậy, biến ngẫu nhiên liên tục của chúng ta trở thành biến rời rạc và chúng ta có thể sử dụng :eqref:`eq_exp_def` để viết

$$
\begin{aligned}
\mu_X & \approx \sum_{i} (\epsilon i)P(X \in (\epsilon i, \epsilon (i+1)]) \\
& \approx \sum_{i} (\epsilon i)p_X(\epsilon i)\epsilon, \\
\end{aligned}
$$

trong đó $p_X$ là mật độ của $X$. Đây là một xấp xỉ của tích phân $xp_X(x)$, vì vậy ta có thể kết luận rằng

$$
\mu_X = \int_{-\infty}^\infty xp_X(x) \; dx.
$$

Tương tự, sử dụng :eqref:`eq_var_def`, phương sai có thể được viết là

$$
\sigma^2_X = E[X^2] - \mu_X^2 = \int_{-\infty}^\infty x^2p_X(x) \; dx - \left(\int_{-\infty}^\infty xp_X(x) \; dx\right)^2.
$$

Mọi điều đã nêu ở trên về giá trị trung bình, phương sai và độ lệch chuẩn vẫn áp dụng trong trường hợp này. Chẳng hạn, nếu xét biến ngẫu nhiên có mật độ

$$
p(x) = \begin{cases}
1 & x \in [0,1], \\
0 & \textrm{otherwise}.
\end{cases}
$$

ta có thể tính

$$
\mu_X = \int_{-\infty}^\infty xp(x) \; dx = \int_0^1 x \; dx = \frac{1}{2}.
$$

và

$$
\sigma_X^2 = \int_{-\infty}^\infty x^2p(x) \; dx - \left(\frac{1}{2}\right)^2 = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}.
$$

Để cảnh báo, hãy xem xét thêm một ví dụ, được gọi là *phân phối Cauchy*. Đây là phân phối có hàm mật độ xác suất được cho bởi

$$
p(x) = \frac{1}{1+x^2}.
$$

```{.python .input}
#@tab mxnet
# Plot the Cauchy distribution p.d.f.
x = np.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
# Plot the Cauchy distribution p.d.f.
x = torch.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
# Plot the Cauchy distribution p.d.f.
x = tf.range(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

Hàm này trông khá vô hại; quả thực, tra một bảng tích phân sẽ cho thấy diện tích bên dưới nó bằng một, và do đó nó xác định một biến ngẫu nhiên liên tục.

Để thấy vấn đề phát sinh ở đâu, hãy thử tính phương sai của biến này. Việc này sẽ bao gồm sử dụng :eqref:`eq_var_def` để tính

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx.
$$

Hàm bên trong có dạng như sau:

```{.python .input}
#@tab mxnet
# Plot the integrand needed to compute the variance
x = np.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab pytorch
# Plot the integrand needed to compute the variance
x = torch.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab tensorflow
# Plot the integrand needed to compute the variance
x = tf.range(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

Hàm này rõ ràng có diện tích vô hạn bên dưới nó, vì về cơ bản nó là hằng số một với một chỗ lõm nhỏ gần số không; quả thực, ta có thể chứng minh rằng

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx = \infty.
$$

Điều này có nghĩa là nó không có phương sai hữu hạn được xác định rõ.

Tuy nhiên, xem xét sâu hơn còn cho thấy một kết quả đáng lo ngại hơn nữa. Hãy thử tính giá trị trung bình bằng :eqref:`eq_exp_def`. Sử dụng công thức đổi biến, ta có

$$
\mu_X = \int_{-\infty}^{\infty} \frac{x}{1+x^2} \; dx = \frac{1}{2}\int_1^\infty \frac{1}{u} \; du.
$$

Tích phân bên trong chính là định nghĩa của logarithm, nên về bản chất, biểu thức này là $\log(\infty) = \infty$; do đó, cũng không tồn tại giá trị trung bình được xác định rõ!

Các nhà khoa học machine learning định nghĩa mô hình của họ sao cho trong hầu hết trường hợp, chúng ta không cần xử lý những vấn đề này, và phần lớn thời gian sẽ làm việc với các biến ngẫu nhiên có giá trị trung bình và phương sai được xác định rõ. Tuy nhiên, đôi khi các biến ngẫu nhiên có *đuôi nặng* (tức là những biến ngẫu nhiên mà xác suất nhận các giá trị lớn đủ cao để khiến những đại lượng như giá trị trung bình hoặc phương sai không xác định) lại hữu ích trong việc mô hình hóa các hệ thống vật lý. Vì vậy, việc biết rằng chúng tồn tại là điều đáng giá.

### Hàm mật độ kết hợp

Phần trình bày trên giả định rằng ta đang làm việc với một biến ngẫu nhiên có giá trị thực duy nhất. Nhưng nếu ta xử lý hai hoặc nhiều biến ngẫu nhiên có thể tương quan cao thì sao? Đây là tình huống phổ biến trong machine learning: hãy hình dung các biến ngẫu nhiên như $R_{i, j}$, mã hóa giá trị màu đỏ của điểm ảnh tại tọa độ $(i, j)$ trong một hình ảnh, hoặc $P_t$, là biến ngẫu nhiên biểu diễn giá cổ phiếu tại thời điểm $t$. Các điểm ảnh gần nhau thường có màu tương tự, và các thời điểm gần nhau thường có mức giá tương tự. Ta không thể xem chúng là các biến ngẫu nhiên tách biệt mà vẫn kỳ vọng xây dựng được một mô hình thành công (trong :numref:`sec_naive_bayes`, ta sẽ thấy một mô hình hoạt động kém do giả định như vậy). Ta cần phát triển ngôn ngữ toán học để xử lý các biến ngẫu nhiên liên tục có tương quan này.

May mắn là, với các tích phân nhiều biến trong :numref:`sec_integral_calculus`, ta có thể xây dựng một ngôn ngữ như vậy. Giả sử, để đơn giản, ta có hai biến ngẫu nhiên $X, Y$ có thể tương quan. Khi đó, tương tự như trường hợp một biến, ta có thể đặt câu hỏi:

$$
P(X \;\textrm{is in an}\; \epsilon \textrm{-sized interval around}\; x \; \textrm{and} \;Y \;\textrm{is in an}\; \epsilon \textrm{-sized interval around}\; y ).
$$

Lập luận tương tự như trong trường hợp một biến cho thấy biểu thức này xấp xỉ

$$
P(X \;\textrm{is in an}\; \epsilon \textrm{-sized interval around}\; x \; \textrm{and} \;Y \;\textrm{is in an}\; \epsilon \textrm{-sized interval around}\; y ) \approx \epsilon^{2}p(x, y),
$$

với một hàm nào đó $p(x, y)$. Hàm này được gọi là mật độ kết hợp của $X$ và $Y$. Các tính chất tương tự như trong trường hợp một biến cũng đúng ở đây. Cụ thể:

* $p(x, y) \ge 0$;
* $\int _ {\mathbb{R}^2} p(x, y) \;dx \;dy = 1$;
* $P((X, Y) \in \mathcal{D}) = \int _ {\mathcal{D}} p(x, y) \;dx \;dy$.

Theo cách này, ta có thể xử lý nhiều biến ngẫu nhiên có thể tương quan với nhau. Nếu muốn làm việc với nhiều hơn hai biến ngẫu nhiên, ta có thể mở rộng mật độ đa biến đến số lượng tọa độ tùy ý bằng cách xét $p(\mathbf{x}) = p(x_1, \ldots, x_n)$. Các tính chất không âm và có tích phân tổng bằng một vẫn được giữ nguyên.

### Phân phối biên

Khi làm việc với nhiều biến, chúng ta thường muốn bỏ qua các mối quan hệ giữa chúng và hỏi: “biến này được phân phối như thế nào?” Một phân phối như vậy được gọi là *phân phối biên*.

Cụ thể, giả sử ta có hai biến ngẫu nhiên $X, Y$ với mật độ kết hợp được cho bởi $p _ {X, Y}(x, y)$. Ta sẽ sử dụng chỉ số dưới để cho biết mật độ tương ứng với những biến ngẫu nhiên nào. Bài toán tìm phân phối biên là lấy hàm này và sử dụng nó để tìm $p _ X(x)$.

Cũng như với hầu hết các khái niệm, tốt nhất là quay lại hình dung trực quan để xác định điều gì cần đúng. Nhắc lại rằng mật độ là hàm $p _ X$ sao cho

$$
P(X \in [x, x+\epsilon]) \approx \epsilon \cdot p _ X(x).
$$

Không có sự xuất hiện của $Y$, nhưng nếu những gì ta được cho chỉ là $p _{X, Y}$, ta cần đưa $Y$ vào theo một cách nào đó. Trước hết, ta có thể nhận thấy điều này tương đương với

$$
P(X \in [x, x+\epsilon] \textrm{, and } Y \in \mathbb{R}) \approx \epsilon \cdot p _ X(x).
$$

Mật độ của ta không trực tiếp cho biết điều gì xảy ra trong trường hợp này; ta cũng cần chia thành các khoảng nhỏ theo $y$, vì vậy ta có thể viết

$$
\begin{aligned}
\epsilon \cdot p _ X(x) & \approx \sum _ {i} P(X \in [x, x+\epsilon] \textrm{, and } Y \in [\epsilon \cdot i, \epsilon \cdot (i+1)]) \\
& \approx \sum _ {i} \epsilon^{2} p _ {X, Y}(x, \epsilon\cdot i).
\end{aligned}
$$

![Bằng cách cộng theo các cột trong mảng xác suất, ta có thể thu được phân phối biên chỉ cho biến ngẫu nhiên được biểu diễn trên trục $\mathit{x}$.](../img/marginal.svg)
:label:`fig_marginal`

Điều này cho ta biết rằng để lấy tổng các giá trị của mật độ dọc theo một dãy hình vuông trên một đường thẳng, như minh họa trong :numref:`fig_marginal`. Quả thực, sau khi khử một thừa số epsilon ở cả hai vế và nhận ra rằng tổng ở vế phải là tích phân theo $y$, ta có thể kết luận rằng

$$
\begin{aligned}
 p _ X(x) &  \approx \sum _ {i} \epsilon p _ {X, Y}(x, \epsilon\cdot i) \\
 & \approx \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
\end{aligned}
$$

Do đó,

$$
p _ X(x) = \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
$$

Điều này cho biết rằng để có được một phân phối biên, ta tích phân theo các biến mà ta không quan tâm. Quá trình này thường được gọi là *integrating out* hoặc *marginalized out* các biến không cần thiết.

### Hiệp phương sai

Khi làm việc với nhiều biến ngẫu nhiên, có thêm một đại lượng tóm tắt hữu ích cần biết: *hiệp phương sai*. Đại lượng này đo mức độ hai biến ngẫu nhiên biến động cùng nhau.

Giả sử ta có hai biến ngẫu nhiên $X$ và $Y$; trước hết, giả sử chúng là rời rạc, nhận các giá trị $(x_i, y_j)$ với xác suất $p_{ij}$. Trong trường hợp này, hiệp phương sai được định nghĩa là

$$\sigma_{XY} = \textrm{Cov}(X, Y) = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij}. = E[XY] - E[X]E[Y].$$
:eqlabel:`eq_cov_def`

Để hình dung trực quan, hãy xét cặp biến ngẫu nhiên sau. Giả sử $X$ nhận các giá trị $1$ và $3$, còn $Y$ nhận các giá trị $-1$ và $3$. Giả sử ta có các xác suất sau

$$
\begin{aligned}
P(X = 1 \; \textrm{and} \; Y = -1) & = \frac{p}{2}, \\
P(X = 1 \; \textrm{and} \; Y = 3) & = \frac{1-p}{2}, \\
P(X = 3 \; \textrm{and} \; Y = -1) & = \frac{1-p}{2}, \\
P(X = 3 \; \textrm{and} \; Y = 3) & = \frac{p}{2},
\end{aligned}
$$

trong đó $p$ là một tham số trong $[0,1]$ mà ta được quyền lựa chọn. Lưu ý rằng nếu $p=1$ thì cả hai luôn đồng thời nhận giá trị nhỏ nhất hoặc lớn nhất, còn nếu $p=0$ thì chúng chắc chắn đồng thời nhận các giá trị đảo ngược (một biến lớn khi biến kia nhỏ và ngược lại). Nếu $p=1/2$, thì cả bốn khả năng đều có xác suất như nhau và hai biến không nên có quan hệ với nhau. Hãy tính hiệp phương sai. Trước hết, lưu ý rằng $\mu_X = 2$ và $\mu_Y = 1$, nên ta có thể tính bằng :eqref:`eq_cov_def`:

$$
\begin{aligned}
\textrm{Cov}(X, Y) & = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij} \\
& = (1-2)(-1-1)\frac{p}{2} + (1-2)(3-1)\frac{1-p}{2} + (3-2)(-1-1)\frac{1-p}{2} + (3-2)(3-1)\frac{p}{2} \\
& = 4p-2.
\end{aligned}
$$

Khi $p=1$ (trường hợp cả hai đồng thời dương hoặc âm cực đại), hiệp phương sai bằng $2$. Khi $p=0$ (trường hợp hai biến bị đảo ngược), hiệp phương sai bằng $-2$. Cuối cùng, khi $p=1/2$ (trường hợp hai biến không liên quan), hiệp phương sai bằng $0$. Như vậy, ta thấy rằng hiệp phương sai đo lường mối quan hệ giữa hai biến ngẫu nhiên này.

Một lưu ý nhanh về hiệp phương sai là nó chỉ đo lường các mối quan hệ tuyến tính. Những mối quan hệ phức tạp hơn, chẳng hạn như $X = Y^2$ trong đó $Y$ được chọn ngẫu nhiên từ $\{-2, -1, 0, 1, 2\}$ với xác suất bằng nhau, có thể bị bỏ sót. Quả thực, một phép tính nhanh cho thấy các biến ngẫu nhiên này có hiệp phương sai bằng không, mặc dù một biến là hàm xác định của biến kia.

Đối với các biến ngẫu nhiên liên tục, câu chuyện cũng gần như tương tự. Đến đây, ta đã khá quen với việc chuyển đổi giữa trường hợp rời rạc và liên tục, nên sẽ đưa ra dạng tương tự trong trường hợp liên tục của :eqref:`eq_cov_def` mà không trình bày phép suy ra.

$$
\sigma_{XY} = \int_{\mathbb{R}^2} (x-\mu_X)(y-\mu_Y)p(x, y) \;dx \;dy.
$$

Để trực quan hóa, hãy xem một tập hợp các biến ngẫu nhiên có hiệp phương sai có thể điều chỉnh.

```{.python .input}
#@tab mxnet
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = covs[i]*X + np.random.normal(0, 1, (500))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = covs[i]*X + torch.randn(500)

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = tf.random.normal((500, ))
    Y = covs[i]*X + tf.random.normal((500, ))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

Hãy xem xét một số tính chất của hiệp phương sai:

* Với mọi biến ngẫu nhiên $X$, $\textrm{Cov}(X, X) = \textrm{Var}(X)$.
* Với mọi biến ngẫu nhiên $X, Y$ và các số $a$ và $b$, $\textrm{Cov}(aX+b, Y) = \textrm{Cov}(X, aY+b) = a\textrm{Cov}(X, Y)$.
* Nếu $X$ và $Y$ độc lập thì $\textrm{Cov}(X, Y) = 0$.

Ngoài ra, ta có thể dùng hiệp phương sai để mở rộng một mối quan hệ đã thấy trước đây. Nhắc lại rằng nếu $X$ và $Y$ là hai biến ngẫu nhiên độc lập thì

$$
\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y).
$$

Với kiến thức về hiệp phương sai, ta có thể mở rộng mối quan hệ này. Thực vậy, một số phép biến đổi đại số cho thấy trong trường hợp tổng quát,

$$
\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y) + 2\textrm{Cov}(X, Y).
$$

Điều này cho phép ta tổng quát hóa quy tắc cộng phương sai cho các biến ngẫu nhiên có tương quan.

### Tương quan

Tương tự như khi xét giá trị trung bình và phương sai, bây giờ hãy xem xét các đơn vị đo. Nếu $X$ được đo bằng một đơn vị (chẳng hạn inch), còn $Y$ được đo bằng một đơn vị khác (chẳng hạn đô la), thì hiệp phương sai được đo bằng tích của hai đơn vị này $\textrm{inches} \times \textrm{dollars}$. Những đơn vị này có thể khó diễn giải. Trong trường hợp này, điều ta thường muốn là một phép đo mức độ liên quan không phụ thuộc đơn vị. Quả thực, ta thường không quan tâm đến độ tương quan định lượng chính xác, mà muốn biết liệu tương quan có cùng chiều hay không và mối quan hệ mạnh đến mức nào.

Để xem điều gì là hợp lý, hãy thực hiện một thí nghiệm tư duy. Giả sử ta chuyển các biến ngẫu nhiên được đo bằng inch và đô la sang inch và cent. Khi đó, biến ngẫu nhiên $Y$ được nhân với $100$. Nếu áp dụng định nghĩa, ta thấy rằng $\textrm{Cov}(X, Y)$ sẽ được nhân với $100$. Như vậy, trong trường hợp này, việc thay đổi đơn vị làm thay đổi hiệp phương sai theo hệ số $100$. Do đó, để tìm một phép đo tương quan không phụ thuộc đơn vị, ta cần chia cho một đại lượng khác cũng được nhân với $100$. Quả thực, ta có một ứng viên rõ ràng: độ lệch chuẩn! Cụ thể, nếu định nghĩa *hệ số tương quan* là

$$\rho(X, Y) = \frac{\textrm{Cov}(X, Y)}{\sigma_{X}\sigma_{Y}},$$
:eqlabel:`eq_cor_def`

ta thấy đây là một giá trị không có đơn vị. Một chút toán học có thể cho thấy rằng số này nằm giữa $-1$ và $1$, trong đó $1$ biểu thị tương quan dương cực đại, còn $-1$ biểu thị tương quan âm cực đại.

Quay lại ví dụ rời rạc cụ thể ở trên, ta thấy rằng $\sigma_X = 1$ và $\sigma_Y = 2$, vì vậy có thể tính tương quan giữa hai biến ngẫu nhiên bằng :eqref:`eq_cor_def` để được

$$
\rho(X, Y) = \frac{4p-2}{1\cdot 2} = 2p-1.
$$

Giá trị này nằm trong khoảng từ $-1$ đến $1$, với hành vi đúng như dự kiến: $1$ nghĩa là tương quan mạnh nhất, còn $-1$ nghĩa là tương quan yếu nhất.

Một ví dụ khác, xét $X$ là một biến ngẫu nhiên bất kỳ và $Y=aX+b$ là một hàm tuyến tính xác định bất kỳ của $X$. Khi đó, ta có thể tính được rằng

$$\sigma_{Y} = \sigma_{aX+b} = |a|\sigma_{X},$$

$$\textrm{Cov}(X, Y) = \textrm{Cov}(X, aX+b) = a\textrm{Cov}(X, X) = a\textrm{Var}(X),$$

và do :eqref:`eq_cor_def`,

$$
\rho(X, Y) = \frac{a\textrm{Var}(X)}{|a|\sigma_{X}^2} = \frac{a}{|a|} = \textrm{sign}(a).
$$

Như vậy, ta thấy rằng tương quan bằng $+1$ với mọi $a > 0$ và bằng $-1$ với mọi $a < 0$, cho thấy tương quan đo lường mức độ và chiều hướng mà hai biến ngẫu nhiên có liên hệ với nhau, chứ không đo quy mô của biến thiên.

Hãy lại vẽ một tập hợp các biến ngẫu nhiên có tương quan có thể điều chỉnh.

```{.python .input}
#@tab mxnet
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = cors[i] * X + np.sqrt(1 - cors[i]**2) * np.random.normal(0, 1, 500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = cors[i] * X + torch.sqrt(torch.tensor(1) -
                                 cors[i]**2) * torch.randn(500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = tf.random.normal((500, ))
    Y = cors[i] * X + tf.sqrt(tf.constant(1.) -
                                 cors[i]**2) * tf.random.normal((500, ))

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

Dưới đây là một số tính chất của tương quan.

* Với mọi biến ngẫu nhiên $X$, $\rho(X, X) = 1$.
* Với mọi biến ngẫu nhiên $X, Y$ và các số $a$ và $b$, $\rho(aX+b, Y) = \rho(X, aY+b) = \rho(X, Y)$.
* Nếu $X$ và $Y$ độc lập và có phương sai khác không thì $\rho(X, Y) = 0$.

Cuối cùng, có thể bạn cảm thấy một số công thức này quen thuộc. Quả thực, nếu khai triển mọi thứ với giả định rằng $\mu_X = \mu_Y = 0$, ta được

$$
\rho(X, Y) = \frac{\sum_{i, j} x_iy_ip_{ij}}{\sqrt{\sum_{i, j}x_i^2 p_{ij}}\sqrt{\sum_{i, j}y_j^2 p_{ij}}}.
$$

Biểu thức này giống như một tổng của các tích, chia cho căn bậc hai của các tổng. Đây chính là công thức tính cosin của góc giữa hai vector $\mathbf{v}, \mathbf{w}$, trong đó các tọa độ khác nhau được gán trọng số bởi $p_{ij}$:

$$
\cos(\theta) = \frac{\mathbf{v}\cdot \mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|} = \frac{\sum_{i} v_iw_i}{\sqrt{\sum_{i}v_i^2}\sqrt{\sum_{i}w_i^2}}.
$$

Thật vậy, nếu xem các chuẩn là có liên hệ với độ lệch chuẩn và các tương quan là cosin của các góc, ta có thể áp dụng phần lớn trực giác hình học vào việc suy nghĩ về các biến ngẫu nhiên.

## Tóm tắt
* Các biến ngẫu nhiên liên tục là những biến ngẫu nhiên có thể nhận một liên tục các giá trị. Chúng có một số khó khăn về mặt kỹ thuật, khiến việc xử lý chúng phức tạp hơn so với các biến ngẫu nhiên rời rạc.
* Hàm mật độ xác suất cho phép ta làm việc với các biến ngẫu nhiên liên tục bằng cách cung cấp một hàm trong đó diện tích dưới đường cong trên một khoảng nào đó cho biết xác suất tìm thấy một điểm mẫu trong khoảng đó.
* Hàm phân phối tích lũy là xác suất quan sát biến ngẫu nhiên nhỏ hơn một ngưỡng cho trước. Hàm này cung cấp một góc nhìn thay thế hữu ích, thống nhất các biến rời rạc và liên tục.
* Giá trị trung bình là giá trị trung bình của một biến ngẫu nhiên.
* Phương sai là bình phương kỳ vọng của độ chênh lệch giữa biến ngẫu nhiên và giá trị trung bình của nó.
* Độ lệch chuẩn là căn bậc hai của phương sai. Có thể xem nó là đại lượng đo khoảng giá trị mà biến ngẫu nhiên có thể nhận.
* Bất đẳng thức Chebyshev cho phép ta biến trực giác này thành lập luận chặt chẽ bằng cách đưa ra một khoảng cụ thể chứa biến ngẫu nhiên trong phần lớn thời gian.
* Mật độ kết hợp cho phép ta làm việc với các biến ngẫu nhiên có tương quan. Ta có thể lấy biên từ mật độ kết hợp bằng cách tích phân theo các biến ngẫu nhiên không cần thiết để nhận được phân phối của biến ngẫu nhiên mong muốn.
* Hiệp phương sai và hệ số tương quan cung cấp một cách đo lường mọi mối quan hệ tuyến tính giữa hai biến ngẫu nhiên có tương quan.

## Bài tập
1. Giả sử ta có một biến ngẫu nhiên với mật độ được cho bởi $p(x) = \frac{1}{x^2}$ với $x \ge 1$ và $p(x) = 0$ trong các trường hợp khác. $P(X > 2)$ bằng bao nhiêu?
2. Phân phối Laplace là một biến ngẫu nhiên có mật độ được cho bởi $p(x = \frac{1}{2}e^{-|x|}$. Giá trị trung bình và độ lệch chuẩn của hàm này bằng bao nhiêu? Gợi ý, $\int_0^\infty xe^{-x} \; dx = 1$ và $\int_0^\infty x^2e^{-x} \; dx = 2$.
3. Tôi đi đến chỗ bạn trên phố và nói: "Tôi có một biến ngẫu nhiên với giá trị trung bình là $1$, độ lệch chuẩn là $2$, và tôi quan sát thấy $25\%$ mẫu của mình có giá trị lớn hơn $9$." Bạn có tin tôi không? Tại sao có hoặc tại sao không?
4. Giả sử bạn có hai biến ngẫu nhiên $X, Y$, với mật độ kết hợp được cho bởi $p_{XY}(x, y) = 4xy$ với $x, y \in [0,1]$ và $p_{XY}(x, y) = 0$ trong các trường hợp khác. Hiệp phương sai của $X$ và $Y$ bằng bao nhiêu?


:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/415)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1094)
:end_tab:


:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1095)
:end_tab: