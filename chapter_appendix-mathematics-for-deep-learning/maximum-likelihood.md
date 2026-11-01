# Maximum Likelihood
:label:`sec_maximum_likelihood`

Một trong những cách tư duy thường gặp nhất trong machine learning là quan điểm maximum likelihood. Đây là khái niệm cho rằng khi làm việc với một mô hình xác suất có các tham số chưa biết, những tham số khiến dữ liệu có xác suất cao nhất là những tham số có khả năng đúng nhất.

## Nguyên lý Maximum Likelihood

Điều này có một cách diễn giải theo Bayesian, có thể hữu ích để suy nghĩ về vấn đề. Giả sử chúng ta có một mô hình với các tham số $\boldsymbol{\theta}$ và một tập hợp các ví dụ dữ liệu $X$. Cụ thể hơn, ta có thể hình dung $\boldsymbol{\theta}$ là một giá trị duy nhất biểu diễn xác suất đồng xu cho mặt ngửa khi được tung, còn $X$ là một chuỗi các lần tung đồng xu độc lập. Chúng ta sẽ xem xét sâu ví dụ này ở phần sau.

Nếu muốn tìm giá trị có khả năng đúng nhất của các tham số trong mô hình, điều đó có nghĩa là chúng ta muốn tìm

$$\mathop{\mathrm{argmax}} P(\boldsymbol{\theta}\mid X).$$
:eqlabel:`eq_max_like`

Theo quy tắc Bayes, biểu thức này tương đương với

$$
\mathop{\mathrm{argmax}} \frac{P(X \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(X)}.
$$

Biểu thức $P(X)$, là xác suất tạo ra dữ liệu không phụ thuộc vào tham số, hoàn toàn không phụ thuộc vào $\boldsymbol{\theta}$, vì vậy có thể bỏ qua mà không làm thay đổi lựa chọn tốt nhất của $\boldsymbol{\theta}$. Tương tự, giờ đây ta có thể giả định rằng không có tiên nghiệm nào cho rằng một tập tham số tốt hơn các tập khác, do đó có thể khẳng định rằng $P(\boldsymbol{\theta})$ cũng không phụ thuộc vào theta! Chẳng hạn, điều này hợp lý trong ví dụ tung đồng xu, vì xác suất đồng xu cho mặt ngửa có thể là bất kỳ giá trị nào trong $[0,1]$ mà không có niềm tin tiên nghiệm nào về việc đồng xu có cân bằng hay không (thường được gọi là *tiên nghiệm không mang thông tin*). Vì vậy, ta thấy rằng việc áp dụng quy tắc Bayes cho thấy lựa chọn tốt nhất của $\boldsymbol{\theta}$ là ước lượng maximum likelihood cho $\boldsymbol{\theta}$:

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}).
$$

Theo thuật ngữ thông dụng, xác suất của dữ liệu khi biết các tham số ($P(X \mid \boldsymbol{\theta})$) được gọi là *likelihood*.

### Một ví dụ cụ thể

Hãy xem cách thức này hoạt động trong một ví dụ cụ thể. Giả sử chúng ta có một tham số duy nhất $\theta$ biểu diễn xác suất một lần tung đồng xu cho mặt ngửa. Khi đó, xác suất cho mặt sấp là $1-\theta$, và nếu dữ liệu quan sát được $X$ là một chuỗi gồm $n_H$ lần ngửa và $n_T$ lần sấp, ta có thể sử dụng thực tế rằng các xác suất độc lập được nhân với nhau để thấy rằng

$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$

Nếu tung $13$ lần và nhận được chuỗi "HHHTHTTHHHHHT", trong đó có $n_H = 9$ lần ngửa và $n_T = 4$ lần sấp, ta thấy rằng

$$
P(X \mid \theta) = \theta^9(1-\theta)^4.
$$

Một điểm thuận lợi của ví dụ này là chúng ta biết trước đáp án. Quả thực, nếu nói bằng lời rằng: “Tôi tung đồng xu 13 lần và có 9 lần ra mặt ngửa, vậy phỏng đoán tốt nhất của chúng ta về xác suất đồng xu cho mặt ngửa là bao nhiêu?”, mọi người đều sẽ đoán đúng là $9/13$. Phương pháp maximum likelihood này cho chúng ta một cách để suy ra con số đó từ những nguyên lý đầu tiên, theo một cách có thể tổng quát hóa cho những tình huống phức tạp hơn rất nhiều.

Với ví dụ của chúng ta, đồ thị của $P(X \mid \theta)$ như sau:

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

theta = np.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

theta = torch.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

theta = tf.range(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

Giá trị của nó đạt cực đại ở một điểm gần với giá trị kỳ vọng $9/13 \approx 0.7\ldots$. Để xem điểm đó có chính xác hay không, chúng ta có thể dùng phép tính vi tích phân. Lưu ý rằng tại điểm cực đại, gradient của hàm là phẳng. Vì vậy, ta có thể tìm ước lượng maximum likelihood :eqref:`eq_max_like` bằng cách tìm các giá trị của $\theta$ tại đó đạo hàm bằng 0, rồi chọn giá trị cho xác suất cao nhất. Ta tính được:

$$
\begin{aligned}
0 & = \frac{d}{d\theta} P(X \mid \theta) \\
& = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\
& = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\
& = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned}
$$

Phương trình này có ba nghiệm: $0$, $1$ và $9/13$. Hai nghiệm đầu rõ ràng là các điểm cực tiểu chứ không phải cực đại, vì chúng gán xác suất bằng $0$ cho chuỗi của chúng ta. Giá trị cuối cùng *không* gán xác suất bằng 0 cho chuỗi, do đó phải là ước lượng maximum likelihood $\hat \theta = 9/13$.

## Tối ưu hóa số và Negative Log-Likelihood

Ví dụ trước khá đơn giản, nhưng nếu chúng ta có hàng tỷ tham số và ví dụ dữ liệu thì sao?

Trước hết, hãy lưu ý rằng nếu giả định tất cả các ví dụ dữ liệu là độc lập, chúng ta không còn có thể thực tế xem xét trực tiếp likelihood, vì nó là tích của rất nhiều xác suất. Quả thực, mỗi xác suất đều nằm trong $[0,1]$, thường có giá trị khoảng $1/2$, và tích $(1/2)^{1000000000}$ nhỏ hơn rất nhiều so với độ chính xác của máy. Chúng ta không thể làm việc trực tiếp với giá trị đó.

Tuy nhiên, hãy nhớ rằng logarithm biến tích thành tổng, khi đó

$$
\log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots
$$

Con số này hoàn toàn nằm trong phạm vi biểu diễn, ngay cả với một số thực 32-bit độ chính xác đơn. Vì vậy, chúng ta nên xét *log-likelihood*, được định nghĩa là

$$
\log(P(X \mid \boldsymbol{\theta})).
$$

Vì hàm $x \mapsto \log(x)$ là hàm tăng, việc cực đại hóa likelihood tương đương với việc cực đại hóa log-likelihood. Quả thực, trong :numref:`sec_naive_bayes`, chúng ta sẽ thấy lập luận này được áp dụng khi làm việc với ví dụ cụ thể về bộ phân loại naive Bayes.

Chúng ta thường làm việc với các hàm mất mát, trong đó mục tiêu là tối thiểu hóa mất mát. Ta có thể chuyển maximum likelihood thành bài toán tối thiểu hóa một hàm mất mát bằng cách lấy $-\log(P(X \mid \boldsymbol{\theta}))$, được gọi là *negative log-likelihood*.

Để minh họa, hãy xét lại bài toán tung đồng xu ở trên và giả vờ rằng chúng ta không biết nghiệm dạng đóng. Ta có thể tính được

$$
-\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta)).
$$

Biểu thức này có thể được viết thành code và tối ưu hóa dễ dàng, ngay cả khi có hàng tỷ lần tung đồng xu.

```{.python .input}
#@tab mxnet
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = np.array(0.5)
theta.attach_grad()

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    with autograd.record():
        loss = -(n_H * np.log(theta) + n_T * np.log(1 - theta))
    loss.backward()
    theta -= lr * theta.grad

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab pytorch
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = torch.tensor(0.5, requires_grad=True)

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab tensorflow
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = tf.Variable(tf.constant(0.5))

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    with tf.GradientTape() as t:
        loss = -(n_H * tf.math.log(theta) + n_T * tf.math.log(1 - theta))
    theta.assign_sub(lr * t.gradient(loss, theta))

# Check output
theta, n_H / (n_H + n_T)
```

Sự thuận tiện về mặt số học không phải là lý do duy nhất khiến mọi người thích sử dụng negative log-likelihood. Có một số lý do khác khiến nó được ưu tiên.

Lý do thứ hai để xét log-likelihood là việc áp dụng các quy tắc giải tích trở nên đơn giản hơn. Như đã thảo luận ở trên, do các giả định về tính độc lập, hầu hết các xác suất chúng ta gặp trong machine learning đều là tích của các xác suất riêng lẻ.

$$
P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$

Điều này có nghĩa là nếu áp dụng trực tiếp quy tắc đạo hàm của tích để tính đạo hàm, ta nhận được

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} P(X\mid\boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right)\cdot P(x_2\mid\boldsymbol{\theta})\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_2\mid\boldsymbol{\theta})\right)\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot P(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
\end{aligned}
$$

Việc này cần $n(n-1)$ phép nhân cùng với $(n-1)$ phép cộng, nên có độ phức tạp thời gian bậc hai theo đầu vào! Việc nhóm các hạng tử một cách đủ khéo léo sẽ giảm độ phức tạp này xuống thời gian tuyến tính, nhưng cần phải suy nghĩ thêm. Với negative log-likelihood, thay vào đó ta có

$$
-\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta})),
$$

từ đó suy ra

$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
$$

Việc này chỉ cần $n$ phép chia và $n-1$ phép cộng, do đó có độ phức tạp thời gian tuyến tính theo đầu vào.

Lý do thứ ba và cũng là lý do cuối cùng để xét negative log-likelihood là mối liên hệ của nó với lý thuyết thông tin, nội dung sẽ được thảo luận chi tiết trong :numref:`sec_information_theory`. Đây là một lý thuyết toán học chặt chẽ, cung cấp cách đo lường mức độ thông tin hoặc tính ngẫu nhiên trong một biến ngẫu nhiên. Đối tượng nghiên cứu cốt lõi của lĩnh vực này là entropy, được định nghĩa là

$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$

đại lượng đo tính ngẫu nhiên của một nguồn. Lưu ý rằng đây không gì khác hơn là xác suất $-\log$ trung bình, và do đó nếu lấy negative log-likelihood rồi chia cho số lượng ví dụ dữ liệu, ta nhận được một biến thể của entropy có tên là cross-entropy. Chỉ riêng cách diễn giải lý thuyết này cũng đã đủ thuyết phục để báo cáo negative log-likelihood trung bình trên tập dữ liệu như một cách đo hiệu năng của mô hình.

## Maximum Likelihood cho các biến liên tục

Mọi thứ chúng ta đã thực hiện cho đến nay đều giả định rằng ta đang làm việc với các biến ngẫu nhiên rời rạc, nhưng nếu muốn làm việc với các biến ngẫu nhiên liên tục thì sao?

Tóm lại, hoàn toàn không có gì thay đổi, ngoại trừ việc ta thay mọi trường hợp của xác suất bằng mật độ xác suất. Nhắc lại rằng ta viết các mật độ bằng chữ $p$ thường, điều này có nghĩa là, chẳng hạn, bây giờ ta viết

$$
-\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)).
$$

Câu hỏi đặt ra là: “Tại sao điều này lại đúng?” Xét cho cùng, lý do chúng ta đưa vào các mật độ là vì xác suất nhận được những kết quả cụ thể tự nó bằng không; vậy chẳng phải xác suất sinh ra dữ liệu của chúng ta sẽ bằng không với mọi tập tham số hay sao?

Đúng là như vậy, và việc hiểu tại sao ta có thể chuyển sang dùng mật độ là một bài tập theo dõi điều gì xảy ra với các epsilon.

Trước hết, hãy định nghĩa lại mục tiêu của chúng ta. Giả sử rằng đối với các biến ngẫu nhiên liên tục, ta không còn muốn tính xác suất nhận được chính xác giá trị cần tìm, mà thay vào đó là xác suất khớp trong một khoảng $\epsilon$. Để đơn giản, ta giả sử dữ liệu của mình là các quan sát lặp lại $x_1, \ldots, x_N$ của các biến ngẫu nhiên phân phối đồng nhất $X_1, \ldots, X_N$. Như đã thấy trước đây, điều này có thể được viết là

$$
\begin{aligned}
&P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta}) \\
\approx &\epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}).
\end{aligned}
$$

Do đó, nếu lấy logarithm âm của biểu thức này, ta thu được

$$
\begin{aligned}
&-\log(P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta})) \\
\approx & -N\log(\epsilon) - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
\end{aligned}
$$

Nếu xem xét biểu thức này, ta thấy $\epsilon$ chỉ xuất hiện trong hằng số cộng $-N\log(\epsilon)$. Hằng số này hoàn toàn không phụ thuộc vào các tham số $\boldsymbol{\theta}$, vì vậy lựa chọn tối ưu của $\boldsymbol{\theta}$ không phụ thuộc vào lựa chọn $\epsilon$ của chúng ta! Dù ta yêu cầu bốn chữ số hay bốn trăm chữ số, lựa chọn tốt nhất của $\boldsymbol{\theta}$ vẫn không thay đổi. Do đó, ta có thể tự do bỏ epsilon để thấy rằng đại lượng cần tối ưu là

$$
- \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
$$

Như vậy, ta thấy rằng quan điểm maximum likelihood có thể áp dụng cho các biến ngẫu nhiên liên tục dễ dàng như cho các biến rời rạc, bằng cách thay các xác suất bằng các mật độ xác suất.

## Tóm tắt
* Nguyên lý maximum likelihood cho biết mô hình khớp tốt nhất với một tập dữ liệu cho trước là mô hình sinh ra dữ liệu với xác suất cao nhất.
* Thay vào đó, mọi người thường làm việc với negative log-likelihood vì nhiều lý do: tính ổn định số, chuyển tích thành tổng (và nhờ đó đơn giản hóa việc tính gradient), cũng như mối liên hệ về mặt lý thuyết với lý thuyết thông tin.
* Mặc dù dễ được giải thích nhất trong bối cảnh rời rạc, nguyên lý này cũng có thể được mở rộng tự do sang bối cảnh liên tục bằng cách cực đại hóa mật độ xác suất được gán cho các điểm dữ liệu.

## Bài tập
1. Giả sử bạn biết rằng một biến ngẫu nhiên không âm có mật độ $\alpha e^{-\alpha x}$ với một giá trị $\alpha>0$. Bạn thu được một quan sát duy nhất từ biến ngẫu nhiên này, có giá trị là $3$. Ước lượng maximum likelihood của $\alpha$ là bao nhiêu?
2. Giả sử bạn có một tập dữ liệu gồm các mẫu $\{x_i\}_{i=1}^N$ được lấy từ một Gaussian có giá trị trung bình chưa biết nhưng phương sai bằng $1$. Ước lượng maximum likelihood của giá trị trung bình là bao nhiêu?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/416)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1096)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1097)
:end_tab: