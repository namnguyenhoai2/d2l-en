# Các phân phối
:label:`sec_distributions`

Sau khi đã học cách làm việc với xác suất trong cả bối cảnh rời rạc và liên tục, hãy cùng tìm hiểu một số phân phối phổ biến thường gặp. Tùy thuộc vào lĩnh vực machine learning, chúng ta có thể cần làm quen với nhiều phân phối hơn đáng kể, hoặc trong một số lĩnh vực deep learning thì có thể không cần đến chúng. Tuy nhiên, đây là một danh sách cơ bản đáng để làm quen. Trước tiên, hãy import một số thư viện phổ biến.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from math import erf, factorial
import numpy as np
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from math import erf, factorial
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # Define pi in torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from math import erf, factorial
import tensorflow as tf
import tensorflow_probability as tfp

tf.pi = tf.acos(tf.zeros(1)) * 2  # Define pi in TensorFlow
```

## Bernoulli

Đây là biến ngẫu nhiên đơn giản nhất thường gặp. Biến ngẫu nhiên này biểu diễn một lần tung đồng xu, cho kết quả $1$ với xác suất $p$ và $0$ với xác suất $1-p$. Nếu có một biến ngẫu nhiên $X$ tuân theo phân phối này, ta sẽ viết

$$
X \sim \textrm{Bernoulli}(p).
$$

Hàm phân phối tích lũy là

$$F(x) = \begin{cases} 0 & x < 0, \\ 1-p & 0 \le x < 1, \\ 1 & x >= 1 . \end{cases}$$
:eqlabel:`eq_bernoulli_cdf`

Hàm khối xác suất được biểu diễn trong hình dưới đây.

```{.python .input}
#@tab all
p = 0.3

d2l.set_figsize()
d2l.plt.stem([0, 1], [1 - p, p], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Bây giờ, hãy vẽ hàm phân phối tích lũy :eqref:`eq_bernoulli_cdf`.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, tf.constant([F(y) for y in x]), 'x', 'c.d.f.')
```

Nếu $X \sim \textrm{Bernoulli}(p)$, thì:

* $\mu_X = p$,
* $\sigma_X^2 = p(1-p)$.

Ta có thể lấy mẫu một mảng có hình dạng tùy ý từ một biến ngẫu nhiên Bernoulli như sau.

```{.python .input}
#@tab mxnet
1*(np.random.rand(10, 10) < p)
```

```{.python .input}
#@tab pytorch
1*(torch.rand(10, 10) < p)
```

```{.python .input}
#@tab tensorflow
tf.cast(tf.random.uniform((10, 10)) < p, dtype=tf.float32)
```

## Uniform rời rạc

Biến ngẫu nhiên phổ biến tiếp theo là uniform rời rạc. Trong phần thảo luận này, ta giả sử nó có miền giá trị là các số nguyên $\{1, 2, \ldots, n\}$, tuy nhiên có thể tự do chọn bất kỳ tập giá trị nào khác. Trong ngữ cảnh này, từ *uniform* có nghĩa là mọi giá trị có thể xảy ra đều có khả năng như nhau. Xác suất của mỗi giá trị $i \in \{1, 2, 3, \ldots, n\}$ là $p_i = \frac{1}{n}$. Ta sẽ ký hiệu một biến ngẫu nhiên $X$ có phân phối này là

$$
X \sim U(n).
$$

Hàm phân phối tích lũy là

$$F(x) = \begin{cases} 0 & x < 1, \\ \frac{k}{n} & k \le x < k+1 \textrm{ with } 1 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_discrete_uniform_cdf`

Trước tiên, hãy vẽ hàm khối xác suất.

```{.python .input}
#@tab all
n = 5

d2l.plt.stem([i+1 for i in range(n)], n*[1 / n], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Bây giờ, hãy vẽ hàm phân phối tích lũy :eqref:`eq_discrete_uniform_cdf`.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else np.floor(x) / n

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else torch.floor(x) / n

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else tf.floor(x) / n

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

Nếu $X \sim U(n)$, thì:

* $\mu_X = \frac{1+n}{2}$,
* $\sigma_X^2 = \frac{n^2-1}{12}$.

Ta có thể lấy mẫu một mảng có hình dạng tùy ý từ một biến ngẫu nhiên uniform rời rạc như sau.

```{.python .input}
#@tab mxnet
np.random.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.uniform((10, 10), 1, n, dtype=tf.int32)
```

## Uniform liên tục

Tiếp theo, hãy thảo luận về phân phối uniform liên tục. Ý tưởng đằng sau biến ngẫu nhiên này là: nếu tăng $n$ trong phân phối uniform rời rạc rồi scale nó để nằm trong khoảng $[a, b]$, ta sẽ tiến tới một biến ngẫu nhiên liên tục chỉ chọn một giá trị bất kỳ trong $[a, b]$, với mọi giá trị có xác suất như nhau. Ta sẽ ký hiệu phân phối này là

$$
X \sim U(a, b).
$$

Hàm mật độ xác suất là

$$p(x) = \begin{cases} \frac{1}{b-a} & x \in [a, b], \\ 0 & x \not\in [a, b].\end{cases}$$
:eqlabel:`eq_cont_uniform_pdf`

Hàm phân phối tích lũy là

$$F(x) = \begin{cases} 0 & x < a, \\ \frac{x-a}{b-a} & x \in [a, b], \\ 1 & x >= b . \end{cases}$$
:eqlabel:`eq_cont_uniform_cdf`

Trước tiên, hãy vẽ hàm mật độ xác suất :eqref:`eq_cont_uniform_pdf`.

```{.python .input}
#@tab mxnet
a, b = 1, 3

x = np.arange(0, 4, 0.01)
p = (x > a)*(x < b)/(b - a)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
a, b = 1, 3

x = torch.arange(0, 4, 0.01)
p = (x > a).type(torch.float32)*(x < b).type(torch.float32)/(b-a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
a, b = 1, 3

x = tf.range(0, 4, 0.01)
p = tf.cast(x > a, tf.float32) * tf.cast(x < b, tf.float32) / (b - a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

Bây giờ, hãy vẽ hàm phân phối tích lũy :eqref:`eq_cont_uniform_cdf`.

```{.python .input}
#@tab mxnet
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

Nếu $X \sim U(a, b)$, thì:

* $\mu_X = \frac{a+b}{2}$,
* $\sigma_X^2 = \frac{(b-a)^2}{12}$.

Ta có thể lấy mẫu một mảng có hình dạng tùy ý từ một biến ngẫu nhiên uniform như sau. Lưu ý rằng theo mặc định, nó lấy mẫu từ $U(0,1)$, vì vậy nếu muốn một khoảng khác, ta cần scale nó.

```{.python .input}
#@tab mxnet
(b - a) * np.random.rand(10, 10) + a
```

```{.python .input}
#@tab pytorch
(b - a) * torch.rand(10, 10) + a
```

```{.python .input}
#@tab tensorflow
(b - a) * tf.random.uniform((10, 10)) + a
```

## Binomial

Hãy làm cho mọi thứ phức tạp hơn một chút và xem xét biến ngẫu nhiên *binomial*. Biến ngẫu nhiên này bắt nguồn từ việc thực hiện một chuỗi gồm $n$ phép thử độc lập, mỗi phép thử có xác suất thành công là $p$, rồi hỏi xem ta kỳ vọng sẽ quan sát được bao nhiêu lần thành công.

Hãy biểu diễn điều này bằng toán học. Mỗi phép thử là một biến ngẫu nhiên độc lập $X_i$, trong đó ta dùng $1$ để biểu diễn thành công và $0$ để biểu diễn thất bại. Vì mỗi phép thử là một lần tung đồng xu độc lập, thành công với xác suất $p$, ta có thể nói rằng $X_i \sim \textrm{Bernoulli}(p)$. Khi đó, biến ngẫu nhiên binomial là

$$
X = \sum_{i=1}^n X_i.
$$

Trong trường hợp này, ta sẽ viết

$$
X \sim \textrm{Binomial}(n, p).
$$

Để có được hàm phân phối tích lũy, ta cần nhận thấy rằng việc có đúng $k$ lần thành công có thể xảy ra theo $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ cách, trong đó mỗi cách có xác suất xảy ra là $p^k(1-p)^{n-k}$. Do đó, hàm phân phối tích lũy là

$$F(x) = \begin{cases} 0 & x < 0, \\ \sum_{m \le k} \binom{n}{m} p^m(1-p)^{n-m}  & k \le x < k+1 \textrm{ with } 0 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_binomial_cdf`

Trước tiên, hãy vẽ hàm khối xác suất.

```{.python .input}
#@tab mxnet
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = np.array([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = d2l.tensor([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = tf.constant([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Bây giờ, hãy vẽ hàm phân phối tích lũy :eqref:`eq_binomial_cdf`.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 11, 0.01)
cmf = np.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 11, 0.01)
cmf = torch.cumsum(pmf, dim=0)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 11, 0.01)
cmf = tf.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

Nếu $X \sim \textrm{Binomial}(n, p)$, thì:

* $\mu_X = np$,
* $\sigma_X^2 = np(1-p)$.

Điều này suy ra từ tính tuyến tính của kỳ vọng đối với tổng của $n$ biến ngẫu nhiên Bernoulli, cùng với thực tế rằng phương sai của tổng các biến ngẫu nhiên độc lập bằng tổng các phương sai. Có thể lấy mẫu biến này như sau.

```{.python .input}
#@tab mxnet
np.random.binomial(n, p, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.binomial.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

## Poisson

Bây giờ, hãy thực hiện một thí nghiệm tưởng tượng. Ta đang đứng tại một trạm xe buýt và muốn biết trong một phút tới sẽ có bao nhiêu xe buýt đến. Hãy bắt đầu bằng cách xét $X^{(1)} \sim \textrm{Bernoulli}(p)$, tức đơn giản là xác suất có một xe buýt đến trong khoảng thời gian một phút. Đối với các trạm xe buýt xa trung tâm đô thị, đây có thể là một xấp xỉ khá tốt. Có thể ta sẽ không bao giờ thấy nhiều hơn một xe buýt trong một phút.

Tuy nhiên, nếu đang ở một khu vực đông đúc, có khả năng, thậm chí nhiều khả năng, là hai xe buýt sẽ đến. Ta có thể mô hình hóa điều này bằng cách chia biến ngẫu nhiên thành hai phần: 30 giây đầu và 30 giây sau. Trong trường hợp này, ta có thể viết

$$
X^{(2)} \sim X^{(2)}_1 + X^{(2)}_2,
$$

trong đó $X^{(2)}$ là tổng cộng, còn $X^{(2)}_i \sim \textrm{Bernoulli}(p/2)$. Khi đó, phân phối tổng là $X^{(2)} \sim \textrm{Binomial}(2, p/2)$.

Tại sao phải dừng lại ở đây? Hãy tiếp tục chia phút đó thành $n$ phần. Theo cùng lập luận như trên, ta thấy rằng

$$X^{(n)} \sim \textrm{Binomial}(n, p/n).$$
:eqlabel:`eq_eq_poisson_approx`

Hãy xét các biến ngẫu nhiên này. Từ phần trước, ta biết rằng :eqref:`eq_eq_poisson_approx` có kỳ vọng $\mu_{X^{(n)}} = n(p/n) = p$ và phương sai $\sigma_{X^{(n)}}^2 = n(p/n)(1-(p/n)) = p(1-p/n)$. Nếu cho $n \rightarrow \infty$, ta có thể thấy các giá trị này ổn định thành $\mu_{X^{(\infty)}} = p$ và phương sai $\sigma_{X^{(\infty)}}^2 = p$. Điều này cho thấy *có thể có* một biến ngẫu nhiên nào đó mà ta có thể định nghĩa trong giới hạn chia vô hạn này.

Điều này không quá bất ngờ, vì trong thế giới thực, ta chỉ cần đếm số xe buýt đến; tuy nhiên, thật hữu ích khi thấy rằng mô hình toán học của chúng ta được xác định chặt chẽ. Thảo luận này có thể được trình bày một cách hình thức dưới dạng *law of rare events*.

Theo sát lập luận này, ta có thể đi đến mô hình sau. Ta nói rằng $X \sim \textrm{Poisson}(\lambda)$ nếu đó là một biến ngẫu nhiên nhận các giá trị $\{0,1,2, \ldots\}$ với xác suất

$$p_k = \frac{\lambda^ke^{-\lambda}}{k!}.$$
:eqlabel:`eq_poisson_mass`

Giá trị $\lambda > 0$ được gọi là *rate* (hoặc tham số *shape*), và biểu thị số lượt đến trung bình mà ta kỳ vọng trong một đơn vị thời gian.

Ta có thể lấy tổng của hàm khối xác suất này để nhận được hàm phân phối tích lũy.

$$F(x) = \begin{cases} 0 & x < 0, \\ e^{-\lambda}\sum_{m = 0}^k \frac{\lambda^m}{m!} & k \le x < k+1 \textrm{ with } 0 \le k. \end{cases}$$
:eqlabel:`eq_poisson_cdf`

Trước tiên, hãy vẽ hàm khối xác suất :eqref:`eq_poisson_mass`.

```{.python .input}
#@tab mxnet
lam = 5.0

xs = [i for i in range(20)]
pmf = np.array([np.exp(-lam) * lam**k / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
lam = 5.0

xs = [i for i in range(20)]
pmf = torch.tensor([torch.exp(torch.tensor(-lam)) * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
lam = 5.0

xs = [i for i in range(20)]
pmf = tf.constant([tf.exp(tf.constant(-lam)).numpy() * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Bây giờ, hãy vẽ hàm phân phối tích lũy :eqref:`eq_poisson_cdf`.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 21, 0.01)
cmf = np.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 21, 0.01)
cmf = torch.cumsum(pmf, dim=0)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 21, 0.01)
cmf = tf.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

Như đã thấy ở trên, kỳ vọng và phương sai đặc biệt ngắn gọn. Nếu $X \sim \textrm{Poisson}(\lambda)$, thì:

* $\mu_X = \lambda$,
* $\sigma_X^2 = \lambda$.

Ta có thể lấy mẫu như sau.

```{.python .input}
#@tab mxnet
np.random.poisson(lam, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.poisson.Poisson(lam)
m.sample((10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Poisson(lam)
m.sample((10, 10))
```

## Gaussian
Bây giờ, hãy thử một thí nghiệm khác nhưng có liên quan. Giả sử một lần nữa chúng ta thực hiện $n$ phép đo $\textrm{Bernoulli}(p)$ độc lập $X_i$. Phân phối của tổng các phép đo này là $X^{(n)} \sim \textrm{Binomial}(n, p)$. Thay vì lấy giới hạn khi $n$ tăng và $p$ giảm, hãy cố định $p$, rồi cho $n \rightarrow \infty$. Khi đó, $\mu_{X^{(n)}} = np \rightarrow \infty$ và $\sigma_{X^{(n)}}^2 = np(1-p) \rightarrow \infty$, vì vậy không có lý do gì để cho rằng giới hạn này sẽ được xác định rõ.

Tuy nhiên, vẫn chưa phải hết hy vọng! Hãy làm cho kỳ vọng và phương sai có tính chất phù hợp bằng cách định nghĩa

$$
Y^{(n)} = \frac{X^{(n)} - \mu_{X^{(n)}}}{\sigma_{X^{(n)}}}.
$$

Có thể thấy biến này có kỳ vọng bằng không và phương sai bằng một, do đó việc tin rằng nó sẽ hội tụ đến một phân phối giới hạn nào đó là hợp lý. Nếu vẽ các phân phối này, chúng ta sẽ càng tin tưởng hơn rằng điều đó sẽ xảy ra.

```{.python .input}
#@tab mxnet
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = np.array([p**i * (1-p)**(n-i) * binom(n, i) for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/np.sqrt(n*p*(1 - p)) for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = torch.tensor([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/torch.sqrt(torch.tensor(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = tf.constant([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/tf.sqrt(tf.constant(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

Một điều cần lưu ý: so với trường hợp Poisson, giờ đây chúng ta chia cho độ lệch chuẩn, nghĩa là ép các kết quả có thể xảy ra vào những vùng ngày càng nhỏ hơn. Đây là dấu hiệu cho thấy giới hạn của chúng ta sẽ không còn rời rạc mà sẽ liên tục.

Việc suy ra kết quả này nằm ngoài phạm vi của tài liệu, nhưng *định lý giới hạn trung tâm* phát biểu rằng khi $n \rightarrow \infty$, kết quả sẽ là Phân phối Gaussian (đôi khi còn gọi là phân phối chuẩn). Cụ thể hơn, với mọi $a, b$:

$$
\lim_{n \rightarrow \infty} P(Y^{(n)} \in [a, b]) = P(\mathcal{N}(0,1) \in [a, b]),
$$

trong đó, ta nói một biến ngẫu nhiên có phân phối chuẩn với kỳ vọng $\mu$ và phương sai $\sigma^2$, được viết là $X \sim \mathcal{N}(\mu, \sigma^2)$, nếu $X$ có mật độ

$$p_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.$$
:eqlabel:`eq_gaussian_pdf`

Trước tiên, hãy vẽ hàm mật độ xác suất :eqref:`eq_gaussian_pdf`.

```{.python .input}
#@tab mxnet
mu, sigma = 0, 1

x = np.arange(-3, 3, 0.01)
p = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
mu, sigma = 0, 1

x = torch.arange(-3, 3, 0.01)
p = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
mu, sigma = 0, 1

x = tf.range(-3, 3, 0.01)
p = 1 / tf.sqrt(2 * tf.pi * sigma**2) * tf.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

Bây giờ, hãy vẽ hàm phân phối tích lũy. Nội dung này nằm ngoài phạm vi của phụ lục, nhưng c.d.f. Gaussian không có công thức dạng đóng theo các hàm sơ cấp hơn. Chúng ta sẽ sử dụng `erf`, công cụ cung cấp cách tính tích phân này bằng phương pháp số.

```{.python .input}
#@tab mxnet
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2.0

d2l.plot(x, np.array([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * torch.sqrt(d2l.tensor(2.))))) / 2.0

d2l.plot(x, torch.tensor([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * tf.sqrt(tf.constant(2.))))) / 2.0

d2l.plot(x, [phi(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

Những độc giả tinh ý sẽ nhận ra một số biểu thức này. Thực vậy, chúng ta đã gặp tích phân này trong :numref:`sec_integral_calculus`. Chính xác là chúng ta cần phép tính đó để thấy rằng $p_X(x)$ có tổng diện tích bằng một và do đó là một mật độ hợp lệ.

Việc chọn làm việc với các phép tung đồng xu giúp các phép tính ngắn gọn hơn, nhưng không có gì căn bản phụ thuộc vào lựa chọn đó. Thực vậy, nếu lấy một tập hợp bất kỳ các biến ngẫu nhiên độc lập cùng phân phối $X_i$, rồi lập

$$
X^{(N)} = \sum_{i=1}^N X_i.
$$

thì

$$
\frac{X^{(N)} - \mu_{X^{(N)}}}{\sigma_{X^{(N)}}}
$$

sẽ xấp xỉ Gaussian. Cần thêm một số điều kiện để kết quả này đúng, phổ biến nhất là $E[X^4] < \infty$, nhưng tư tưởng chính đã rõ ràng.

Định lý giới hạn trung tâm là lý do Gaussian đóng vai trò nền tảng trong xác suất, thống kê và machine learning. Bất cứ khi nào có thể nói rằng một đại lượng được đo là tổng của nhiều đóng góp nhỏ độc lập, chúng ta có thể giả định rằng đại lượng đó sẽ gần với Gaussian.

Gaussian còn có nhiều tính chất hấp dẫn khác, và ở đây chúng ta muốn thảo luận thêm một tính chất. Gaussian được gọi là một *phân phối entropy cực đại*. Chúng ta sẽ tìm hiểu sâu hơn về entropy trong :numref:`sec_information_theory`, nhưng tại thời điểm này, tất cả những gì cần biết là entropy là một phép đo tính ngẫu nhiên. Theo nghĩa toán học chặt chẽ, ta có thể xem Gaussian là lựa chọn *ngẫu nhiên nhất* của một biến ngẫu nhiên có kỳ vọng và phương sai cố định. Vì vậy, nếu biết biến ngẫu nhiên có một kỳ vọng và phương sai nào đó, Gaussian theo một nghĩa nào đó là lựa chọn phân phối thận trọng nhất mà ta có thể đưa ra.

Để kết thúc phần này, hãy nhớ rằng nếu $X \sim \mathcal{N}(\mu, \sigma^2)$ thì:

* $\mu_X = \mu$,
* $\sigma_X^2 = \sigma^2$.

Ta có thể lấy mẫu từ phân phối Gaussian (hoặc phân phối chuẩn tắc) như dưới đây.

```{.python .input}
#@tab mxnet
np.random.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.normal((10, 10), mu, sigma)
```

## Họ exponential
:label:`subsec_exponential_family`

Một tính chất chung của tất cả các phân phối được liệt kê ở trên là chúng đều thuộc *họ exponential*. Họ exponential là một tập hợp các phân phối có mật độ có thể được biểu diễn dưới dạng sau:

$$p(\mathbf{x} \mid \boldsymbol{\eta}) = h(\mathbf{x}) \cdot \exp \left( \boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) - A(\boldsymbol{\eta}) \right)$$
:eqlabel:`eq_exp_pdf`

Vì định nghĩa này có thể hơi tinh tế, hãy cùng xem xét kỹ hơn.

Thứ nhất, $h(\mathbf{x})$ được gọi là *độ đo nền tảng* hoặc *base measure*. Có thể xem đây là lựa chọn ban đầu về độ đo mà chúng ta điều chỉnh bằng trọng số exponential.

Thứ hai, ta có vector $\boldsymbol{\eta} = (\eta_1, \eta_2, ..., \eta_l) \in
\mathbb{R}^l$ được gọi là *các tham số tự nhiên* hoặc *các tham số chính tắc*. Các tham số này xác định cách độ đo nền tảng được điều chỉnh. Các tham số tự nhiên đi vào độ đo mới bằng cách lấy tích vô hướng của chúng với một hàm $T(\cdot)$ nào đó của $\mathbf{x}= (x_1, x_2, ..., x_n) \in
\mathbb{R}^n$, rồi lũy thừa hóa kết quả. Vector $T(\mathbf{x})= (T_1(\mathbf{x}),
T_2(\mathbf{x}), ..., T_l(\mathbf{x}))$ được gọi là *các thống kê đủ* cho $\boldsymbol{\eta}$. Tên gọi này xuất phát từ việc thông tin được biểu diễn bởi $T(\mathbf{x})$ là đủ để tính mật độ xác suất và không cần thêm thông tin nào khác từ mẫu $\mathbf{x}$.

Thứ ba, ta có $A(\boldsymbol{\eta})$, được gọi là *hàm cumulant*, có tác dụng bảo đảm rằng phân phối :eqref:`eq_exp_pdf` ở trên tích phân bằng một, tức là

$$A(\boldsymbol{\eta})  = \log \left[\int h(\mathbf{x}) \cdot \exp
\left(\boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) \right) d\mathbf{x} \right].$$

Để cụ thể hóa, hãy xét Gaussian. Giả sử $\mathbf{x}$ là một biến đơn biến, ta đã thấy nó có mật độ

$$
\begin{aligned}
p(x \mid \mu, \sigma) &= \frac{1}{\sqrt{2 \pi \sigma^2}} \cdot \exp 
\left\{ \frac{-(x-\mu)^2}{2 \sigma^2} \right\} \\
&= \frac{1}{\sqrt{2 \pi}} \cdot \exp \left\{ \frac{\mu}{\sigma^2}x
-\frac{1}{2 \sigma^2} x^2 - \left( \frac{1}{2 \sigma^2} \mu^2
+\log(\sigma) \right) \right\}.
\end{aligned}
$$

Điều này khớp với định nghĩa của họ exponential với:

* *độ đo nền tảng*: $h(x) = \frac{1}{\sqrt{2 \pi}}$,
* *các tham số tự nhiên*: $\boldsymbol{\eta} = \begin{bmatrix} \eta_1 \\ \eta_2
\end{bmatrix} = \begin{bmatrix} \frac{\mu}{\sigma^2} \\
\frac{1}{2 \sigma^2} \end{bmatrix}$,
* *các thống kê đủ*: $T(x) = \begin{bmatrix}x\\-x^2\end{bmatrix}$, và
* *hàm cumulant*: $A({\boldsymbol\eta}) = \frac{1}{2 \sigma^2} \mu^2 + \log(\sigma)
= \frac{\eta_1^2}{4 \eta_2} - \frac{1}{2}\log(2 \eta_2)$.

Cần lưu ý rằng lựa chọn cụ thể cho từng thành phần trên phần nào mang tính tùy ý. Thực vậy, điều quan trọng là phân phối có thể được biểu diễn dưới dạng này, chứ không phải dạng chính xác của biểu diễn.

Như đã đề cập trong :numref:`subsec_softmax_and_derivatives`, một kỹ thuật được sử dụng rộng rãi là giả định rằng đầu ra cuối cùng $\mathbf{y}$ tuân theo một phân phối thuộc họ exponential. Họ exponential là một họ phân phối phổ biến và mạnh mẽ, thường xuyên xuất hiện trong machine learning.


## Tóm tắt
* Biến ngẫu nhiên Bernoulli có thể được dùng để mô hình hóa các sự kiện có kết quả yes/no.
* Phân phối uniform rời rạc mô hình hóa việc lựa chọn từ một tập hợp hữu hạn các khả năng.
* Phân phối uniform liên tục lựa chọn một giá trị từ một khoảng.
* Phân phối nhị thức mô hình hóa một chuỗi các biến ngẫu nhiên Bernoulli và đếm số lần thành công.
* Biến ngẫu nhiên Poisson mô hình hóa sự xuất hiện của các sự kiện hiếm.
* Biến ngẫu nhiên Gaussian mô hình hóa kết quả của việc cộng một số lượng lớn các biến ngẫu nhiên độc lập.
* Tất cả các phân phối trên đều thuộc họ exponential.

## Bài tập

1. Độ lệch chuẩn của một biến ngẫu nhiên là hiệu $X-Y$ của hai biến ngẫu nhiên nhị thức độc lập $X, Y \sim \textrm{Binomial}(16, 1/2)$ là bao nhiêu?
2. Nếu lấy một biến ngẫu nhiên Poisson $X \sim \textrm{Poisson}(\lambda)$ và xét $(X - \lambda)/\sqrt{\lambda}$ khi $\lambda \rightarrow \infty$, ta có thể chứng minh rằng biến này trở nên xấp xỉ Gaussian. Tại sao điều này hợp lý?
3. Hàm khối lượng xác suất của tổng hai biến ngẫu nhiên đều rời rạc trên $n$ phần tử là gì?


:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/417)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1098)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1099)
:end_tab: