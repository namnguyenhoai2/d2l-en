```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("gpytorch")
```

# Suy luận Gaussian Process

Trong phần này, chúng ta sẽ trình bày cách thực hiện suy luận hậu nghiệm và đưa ra dự đoán bằng các prior GP đã giới thiệu ở phần trước. Chúng ta sẽ bắt đầu với hồi quy, trong đó có thể thực hiện suy luận dưới dạng _tường minh_ (_closed form_). Đây là phần “GP trong một nutshell”, giúp bạn nhanh chóng bắt đầu sử dụng Gaussian process trong thực tế. Trước tiên, chúng ta sẽ tự viết mã cho tất cả các phép toán cơ bản, sau đó giới thiệu [GPyTorch](https://gpytorch.ai/), giúp việc làm việc với Gaussian process hiện đại và tích hợp với mạng neural sâu trở nên thuận tiện hơn nhiều. Chúng ta sẽ tìm hiểu sâu hơn các chủ đề nâng cao này trong phần tiếp theo. Trong phần đó, chúng ta cũng sẽ xem xét các trường hợp cần suy luận xấp xỉ --- phân loại, point process hoặc bất kỳ likelihood không Gaussian nào.

## Suy luận hậu nghiệm cho bài toán hồi quy

Mô hình _quan sát_ liên hệ hàm mà chúng ta muốn học, $f(x)$, với các quan sát của chúng ta $y(x)$, trong đó cả hai đều được lập chỉ mục bởi một đầu vào $x$. Trong bài toán phân loại, $x$ có thể là các pixel của một hình ảnh, còn $y$ có thể là nhãn lớp tương ứng. Trong bài toán hồi quy, $y$ thường biểu diễn một đầu ra liên tục, chẳng hạn như nhiệt độ bề mặt đất, mực nước biển, nồng độ $CO_2$, v.v.

Trong hồi quy, chúng ta thường giả định rằng các đầu ra được cho bởi một hàm tiềm ẩn không nhiễu $f(x)$ cộng với nhiễu Gaussian độc lập và cùng phân phối $\epsilon(x)$:

$$y(x) = f(x) + \epsilon(x),$$
:eqlabel:`eq_gp-regression`

với $\epsilon(x) \sim \mathcal{N}(0,\sigma^2)$. Gọi $\mathbf{y} = y(X) = (y(x_1),\dots,y(x_n))^{\top}$ là vector các quan sát huấn luyện, và $\textbf{f} = (f(x_1),\dots,f(x_n))^{\top}$ là vector các giá trị của hàm tiềm ẩn không nhiễu, được truy vấn tại các đầu vào huấn luyện $X = {x_1, \dots, x_n}$.

Chúng ta sẽ giả định $f(x) \sim \mathcal{GP}(m,k)$, nghĩa là mọi tập hợp các giá trị hàm $\textbf{f}$ đều có phân phối Gaussian đa biến đồng thời, với vector trung bình $\mu_i = m(x_i)$ và ma trận hiệp phương sai $K_{ij} = k(x_i,x_j)$. Kernel RBF $k(x_i,x_j) = a^2 \exp\left(-\frac{1}{2\ell^2}||x_i-x_j||^2\right)$ là một lựa chọn chuẩn cho hàm hiệp phương sai. Để đơn giản về mặt ký hiệu, chúng ta sẽ giả định hàm trung bình $m(x)=0$; các phép suy diễn của chúng ta có thể dễ dàng được tổng quát hóa sau này.

Giả sử chúng ta muốn đưa ra dự đoán tại một tập hợp các đầu vào $$X_* = x_{*1},x_{*2},\dots,x_{*m}.$$ Khi đó, chúng ta muốn tìm $x^2$ và $p(\mathbf{f}_* | \mathbf{y}, X)$. Trong bối cảnh hồi quy, chúng ta có thể thuận tiện tìm được phân phối này bằng cách sử dụng các đồng nhất thức Gaussian, sau khi xác định phân phối đồng thời của $\mathbf{f}_* = f(X_*)$ và $\mathbf{y}$.

Nếu đánh giá phương trình :eqref:`eq_gp-regression` tại các đầu vào huấn luyện $X$, ta có $\mathbf{y} = \mathbf{f} + \mathbf{\epsilon}$. Theo định nghĩa của Gaussian process (xem phần trước), $\mathbf{f} \sim \mathcal{N}(0,K(X,X))$, trong đó $K(X,X)$ là ma trận $n \times n$ được tạo bằng cách đánh giá hàm hiệp phương sai (hay còn gọi là _kernel_) trên mọi cặp đầu vào $x_i, x_j \in X$. $\mathbf{\epsilon}$ đơn giản là một vector gồm các mẫu iid từ $\mathcal{N}(0,\sigma^2)$ và do đó có phân phối $\mathcal{N}(0,\sigma^2I)$. Vì vậy, $\mathbf{y}$ là tổng của hai biến Gaussian đa biến độc lập, do đó có phân phối $\mathcal{N}(0, K(X,X) + \sigma^2I)$. Ta cũng có thể chứng minh rằng $\textrm{cov}(\mathbf{f}_*, \mathbf{y}) = \textrm{cov}(\mathbf{y},\mathbf{f}_*)^{\top} = K(X_*,X)$, trong đó $K(X_*,X)$ là ma trận $m \times n$ được tạo bằng cách đánh giá kernel trên mọi cặp đầu vào kiểm thử và huấn luyện.

$$
\begin{bmatrix}
\mathbf{y} \\
\mathbf{f}_*
\end{bmatrix}
\sim
\mathcal{N}\left(0, 
\mathbf{A} = \begin{bmatrix}
K(X,X)+\sigma^2I & K(X,X_*) \\
K(X_*,X) & K(X_*,X_*)
\end{bmatrix}
\right)
$$

Sau đó, chúng ta có thể sử dụng các đồng nhất thức Gaussian tiêu chuẩn để tìm phân phối có điều kiện từ phân phối đồng thời (xem, chẳng hạn, Bishop Chương 2),
$\mathbf{f}_* | \mathbf{y}, X, X_* \sim \mathcal{N}(m_*,S_*)$, trong đó $m_* = K(X_*,X)[K(X,X)+\sigma^2I]^{-1}\textbf{y}$, và $S = K(X_*,X_*) - K(X_*,X)[K(X,X)+\sigma^2I]^{-1}K(X,X_*)$.

Thông thường, chúng ta không cần sử dụng toàn bộ ma trận hiệp phương sai dự đoán $S$, mà thay vào đó sử dụng đường chéo của $S$ để biểu diễn độ bất định của từng dự đoán. Vì lý do này, ta thường viết phân phối dự đoán cho một điểm kiểm thử đơn $x_*$ thay vì một tập hợp các điểm kiểm thử.

Ma trận kernel có các tham số $\theta$ mà chúng ta cũng muốn ước lượng, chẳng hạn như biên độ $a$ và lengthscale $\ell$ của kernel RBF ở trên. Vì mục đích này, chúng ta sử dụng _marginal likelihood_, $p(\textbf{y} | \theta, X)$, mà chúng ta đã suy ra khi tính các phân phối biên để xác định phân phối đồng thời của $\textbf{y},\textbf{f}_*$. Như chúng ta sẽ thấy, marginal likelihood phân tách thành các thành phần độ phù hợp của mô hình và độ phức tạp của mô hình, đồng thời tự động mã hóa nguyên tắc Occam's razor để học các hyperparameter. Để có phần thảo luận đầy đủ, xem MacKay Ch. 28 :cite:`mackay2003information`, và Rasmussen and Williams Ch. 5 :cite:`rasmussen2006gaussian`.

```{.python .input}
from d2l import torch as d2l
import numpy as np
from scipy.spatial import distance_matrix
from scipy import optimize
import matplotlib.pyplot as plt
import math
import torch
import gpytorch
import os

d2l.set_figsize()
```

## Các phương trình để dự đoán và học hyperparameter của kernel trong hồi quy GP

Dưới đây là các phương trình bạn sẽ sử dụng để học hyperparameter và đưa ra dự đoán trong hồi quy Gaussian process. Một lần nữa, chúng ta giả định một vector các đích hồi quy $\textbf{y}$, được lập chỉ mục bởi các đầu vào $X = \{x_1,\dots,x_n\}$, và muốn đưa ra dự đoán tại một đầu vào kiểm thử $x_*$. Chúng ta giả định nhiễu Gaussian cộng tính, có trung bình bằng không, độc lập và cùng phân phối, với phương sai $\sigma^2$. Chúng ta sử dụng prior Gaussian process $f(x) \sim \mathcal{GP}(m,k)$ cho hàm tiềm ẩn không nhiễu, với hàm trung bình $m$ và hàm kernel $k$. Bản thân kernel có các tham số $\theta$ mà chúng ta muốn học. Ví dụ, nếu sử dụng kernel RBF, $k(x_i,x_j) = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right)$, chúng ta muốn học $\theta = \{a^2, \ell^2\}$. Gọi $K(X,X)$ là ma trận $n \times n$ tương ứng với việc đánh giá kernel trên mọi cặp trong số $n$ đầu vào huấn luyện. Gọi $K(x_*,X)$ là vector $1 \times n$ được tạo bằng cách đánh giá $k(x_*, x_i)$, $i=1,\dots,n$. Gọi $\mu$ là vector trung bình được tạo bằng cách đánh giá hàm trung bình $m(x)$ tại mọi điểm huấn luyện $x$.

Thông thường, khi làm việc với Gaussian process, chúng ta thực hiện theo quy trình hai bước.
1. Học các hyperparameter của kernel $\hat{\theta}$ bằng cách cực đại hóa marginal likelihood theo các hyperparameter này.
2. Sử dụng trung bình dự đoán làm bộ dự đoán điểm, và dùng 2 lần độ lệch chuẩn dự đoán để tạo thành một tập tin cậy 95%, với điều kiện là các hyperparameter đã học $\hat{\theta}$.

Log marginal likelihood đơn giản là mật độ Gaussian lấy log, có dạng:
$$\log p(\textbf{y} | \theta, X) = -\frac{1}{2}\textbf{y}^{\top}[K_{\theta}(X,X) + \sigma^2I]^{-1}\textbf{y} - \frac{1}{2}\log|K_{\theta}(X,X)| + c$$

Phân phối dự đoán có dạng:
$$p(y_* | x_*, \textbf{y}, \theta) = \mathcal{N}(a_*,v_*)$$
$$a_* = k_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}(\textbf{y}-\mu) + \mu$$
$$v_* = k_{\theta}(x_*,x_*) - K_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}k_{\theta}(X,x_*)$$

## Diễn giải các phương trình dùng để học và dự đoán

Có một số điểm chính cần lưu ý về các phân phối dự đoán của Gaussian process:

* Mặc dù lớp mô hình rất linh hoạt, ta vẫn có thể thực hiện suy luận Bayesian _chính xác_ cho hồi quy GP dưới dạng _tường minh_. Ngoài việc học các hyperparameter của kernel, không có bước _huấn luyện_ nào. Ta có thể viết chính xác các phương trình cần sử dụng để đưa ra dự đoán. Gaussian process tương đối đặc biệt ở khía cạnh này, và điều đó góp phần rất lớn vào tính tiện dụng, tính linh hoạt cũng như sự phổ biến lâu dài của chúng.

* Trung bình dự đoán $a_*$ là một tổ hợp tuyến tính của các đích huấn luyện $\textbf{y}$, được gán trọng số bởi kernel $k_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}$. Như chúng ta sẽ thấy, kernel (và các hyperparameter của nó) vì vậy đóng vai trò then chốt trong các thuộc tính tổng quát hóa của mô hình.

* Trung bình dự đoán phụ thuộc tường minh vào các giá trị đích $\textbf{y}$, nhưng phương sai dự đoán thì không. Thay vào đó, độ bất định dự đoán tăng lên khi đầu vào kiểm thử $x_*$ di chuyển xa khỏi các vị trí đích $X$, theo sự chi phối của hàm kernel. Tuy nhiên, độ bất định vẫn phụ thuộc ngầm vào các giá trị đích $\textbf{y}$ thông qua các hyperparameter của kernel $\theta$, vốn được học từ dữ liệu.

* Marginal likelihood được phân tách thành các thành phần về độ phù hợp của mô hình và độ phức tạp của mô hình (định thức log). Marginal likelihood có xu hướng chọn các hyperparameter tạo ra độ phù hợp đơn giản nhất nhưng vẫn nhất quán với dữ liệu.

* Các nút thắt tính toán chính đến từ việc giải một hệ tuyến tính và tính định thức log trên ma trận xác định dương đối xứng $n \times n$ $K(X,X)$ cho $n$ điểm huấn luyện. Một cách ngây thơ, mỗi phép toán này đều cần $\mathcal{O}(n^3)$ phép tính, đồng thời cần $\mathcal{O}(n^2)$ bộ nhớ để lưu từng phần tử của ma trận kernel (hiệp phương sai), thường bắt đầu bằng một phép phân rã Cholesky. Trong quá khứ, các nút thắt này đã giới hạn GP ở những bài toán có ít hơn khoảng 10.000 điểm huấn luyện, đồng thời khiến GP mang tiếng là “chậm”, một nhận định đã không còn chính xác trong gần một thập kỷ qua. Trong các chủ đề nâng cao, chúng ta sẽ thảo luận cách mở rộng GP cho các bài toán có hàng triệu điểm.

* Với các lựa chọn kernel phổ biến, $K(X,X)$ thường gần như suy biến, điều này có thể gây ra các vấn đề số khi thực hiện phân rã Cholesky hoặc các phép toán khác nhằm giải hệ tuyến tính. May mắn là trong hồi quy, chúng ta thường làm việc với $K_{\theta}(X,X)+\sigma^2I$, sao cho phương sai nhiễu $\sigma^2$ được cộng vào đường chéo của $K(X,X)$, cải thiện đáng kể tính điều kiện của ma trận. Nếu phương sai nhiễu nhỏ, hoặc khi thực hiện hồi quy không nhiễu, thông lệ là cộng một lượng nhỏ “jitter” vào đường chéo, cỡ $10^{-6}$, để cải thiện tính điều kiện.

## Ví dụ thực hành từ đầu

Hãy tạo một số dữ liệu hồi quy, sau đó khớp dữ liệu bằng một GP, tự triển khai từng bước từ đầu.  
Chúng ta sẽ lấy mẫu dữ liệu từ  
$$y(x) = \sin(x) + \frac{1}{2}\sin(4x) + \epsilon,$$ với $\epsilon \sim \mathcal{N}(0,\sigma^2)$. Hàm không nhiễu mà chúng ta muốn tìm là $f(x) = \sin(x) + \frac{1}{2}\sin(4x)$. Trước tiên, chúng ta sẽ sử dụng độ lệch chuẩn của nhiễu $\sigma = 0.25$.

```{.python .input}
def data_maker1(x, sig):
    return np.sin(x) + 0.5 * np.sin(4 * x) + np.random.randn(x.shape[0]) * sig

sig = 0.25
train_x, test_x = np.linspace(0, 5, 50), np.linspace(0, 5, 500)
train_y, test_y = data_maker1(train_x, sig=sig), data_maker1(test_x, sig=0.)

d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y)
d2l.plt.xlabel("x", fontsize=20)
d2l.plt.ylabel("Observations y", fontsize=20)
d2l.plt.show()
```

Ở đây, chúng ta thấy các quan sát nhiễu dưới dạng những hình tròn, và hàm không nhiễu màu xanh mà chúng ta muốn tìm.

Bây giờ, hãy chỉ định một GP prior trên hàm ẩn không nhiễu, $f(x)\sim \mathcal{GP}(m,k)$. Chúng ta sẽ sử dụng hàm trung bình $m(x) = 0$ và một hàm hiệp phương sai RBF (kernel)
$$k(x_i,x_j) = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right).$$

```{.python .input}
mean = np.zeros(test_x.shape[0])
cov = d2l.rbfkernel(test_x, test_x, ls=0.2)
```

Chúng ta bắt đầu với length-scale bằng 0.2. Trước khi khớp dữ liệu, điều quan trọng là xem xét liệu chúng ta đã chỉ định một prior hợp lý hay chưa. Hãy trực quan hóa một số hàm được lấy mẫu từ prior này, cũng như miền tin cậy 95\% (chúng ta tin rằng có 95\% khả năng hàm thực nằm trong miền này).

```{.python .input}
prior_samples = np.random.multivariate_normal(mean=mean, cov=cov, size=5)
d2l.plt.plot(test_x, prior_samples.T, color='black', alpha=0.5)
d2l.plt.plot(test_x, mean, linewidth=2.)
d2l.plt.fill_between(test_x, mean - 2 * np.diag(cov), mean + 2 * np.diag(cov), 
                 alpha=0.25)
d2l.plt.show()
```

Các mẫu này có trông hợp lý không? Những đặc tính cấp cao của các hàm có phù hợp với loại dữ liệu mà chúng ta đang cố gắng mô hình hóa không?

Bây giờ, hãy tạo trung bình và phương sai của phân phối dự đoán hậu nghiệm tại một điểm kiểm tra bất kỳ $x_*$.

$$
\bar{f}_{*} = K(x, x_*)^T (K(x, x) + \sigma^2 I)^{-1}y
$$

$$
V(f_{*}) = K(x_*, x_*) - K(x, x_*)^T (K(x, x) + \sigma^2 I)^{-1}K(x, x_*)
$$

Trước khi đưa ra dự đoán, chúng ta cần học các hyperparameter của kernel $\theta$ và phương sai nhiễu $\sigma^2$. Hãy khởi tạo length-scale bằng 0.75, vì các hàm prior của chúng ta có vẻ biến thiên quá nhanh so với dữ liệu đang khớp. Chúng ta cũng sẽ phỏng đoán độ lệch chuẩn của nhiễu $\sigma$ là 0.75.

Để học các tham số này, chúng ta sẽ cực đại hóa marginal likelihood theo các tham số đó.

$$
\log p(y | X) = \log \int p(y | f, X)p(f | X)df
$$
$$
\log p(y | X) = -\frac{1}{2}y^T(K(x, x) + \sigma^2 I)^{-1}y - \frac{1}{2}\log |K(x, x) + \sigma^2 I| - \frac{n}{2}\log 2\pi
$$

Có lẽ các hàm prior của chúng ta biến thiên quá nhanh. Hãy phỏng đoán length-scale là 0.4. Chúng ta cũng sẽ phỏng đoán độ lệch chuẩn của nhiễu là 0.75. Đây chỉ là các giá trị khởi tạo của hyperparameter --- chúng ta sẽ học các tham số này từ marginal likelihood.

```{.python .input}
ell_est = 0.4
post_sig_est = 0.5

def neg_MLL(pars):
    K = d2l.rbfkernel(train_x, train_x, ls=pars[0])
    kernel_term = -0.5 * train_y @ \
        np.linalg.inv(K + pars[1] ** 2 * np.eye(train_x.shape[0])) @ train_y
    logdet = -0.5 * np.log(np.linalg.det(K + pars[1] ** 2 * \
                                         np.eye(train_x.shape[0])))
    const = -train_x.shape[0] / 2. * np.log(2 * np.pi)
    
    return -(kernel_term + logdet + const)


learned_hypers = optimize.minimize(neg_MLL, x0=np.array([ell_est,post_sig_est]), 
                                   bounds=((0.01, 10.), (0.01, 10.)))
ell = learned_hypers.x[0]
post_sig_est = learned_hypers.x[1]
```

Trong trường hợp này, chúng ta học được length-scale là 0.299 và độ lệch chuẩn của nhiễu là 0.24. Lưu ý rằng nhiễu học được cực kỳ gần với nhiễu thực, điều này cho thấy GP của chúng ta được đặc tả rất tốt cho bài toán này.

Nhìn chung, việc cân nhắc cẩn thận khi lựa chọn kernel và khởi tạo hyperparameter là hết sức quan trọng. Mặc dù tối ưu hóa marginal likelihood tương đối bền vững trước cách khởi tạo, nó không miễn nhiễm với các giá trị khởi tạo kém. Hãy thử chạy script trên với nhiều cách khởi tạo khác nhau và xem bạn nhận được những kết quả nào.

Bây giờ, hãy đưa ra dự đoán bằng các hyperparameter đã học.

```{.python .input}
K_x_xstar = d2l.rbfkernel(train_x, test_x, ls=ell)
K_x_x = d2l.rbfkernel(train_x, train_x, ls=ell)
K_xstar_xstar = d2l.rbfkernel(test_x, test_x, ls=ell)

post_mean = K_x_xstar.T @ np.linalg.inv((K_x_x + \
                post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ train_y
post_cov = K_xstar_xstar - K_x_xstar.T @ np.linalg.inv((K_x_x + \
                post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ K_x_xstar

lw_bd = post_mean - 2 * np.sqrt(np.diag(post_cov))
up_bd = post_mean + 2 * np.sqrt(np.diag(post_cov))

d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y, linewidth=2.)
d2l.plt.plot(test_x, post_mean, linewidth=2.)
d2l.plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
d2l.plt.legend(['Observed Data', 'True Function', 'Predictive Mean', '95% Set on True Func'])
d2l.plt.show()
```

Chúng ta thấy trung bình hậu nghiệm màu cam gần như khớp hoàn hảo với hàm không nhiễu thực! Lưu ý rằng miền tin cậy 95\% được trình bày là dành cho hàm _không nhiễu_ (thực) ẩn, chứ không phải các điểm dữ liệu. Chúng ta thấy miền tin cậy này hoàn toàn bao hàm hàm thực và có vẻ không quá rộng cũng không quá hẹp. Chúng ta không muốn, cũng không kỳ vọng, nó bao hàm các điểm dữ liệu. Nếu muốn có một miền tin cậy cho các quan sát, chúng ta nên tính

```{.python .input}
lw_bd_observed = post_mean - 2 * np.sqrt(np.diag(post_cov) + post_sig_est ** 2)
up_bd_observed = post_mean + 2 * np.sqrt(np.diag(post_cov) + post_sig_est ** 2)
```

Có hai nguồn bất định: bất định _epistemic_, đại diện cho bất định _có thể giảm_, và bất định _aleatoric_ hay bất định _không thể giảm_. Bất định _epistemic_ ở đây thể hiện sự không chắc chắn về các giá trị thực của hàm không nhiễu. Bất định này sẽ tăng khi chúng ta di chuyển ra xa các điểm dữ liệu, vì ở xa dữ liệu có nhiều giá trị hàm khác nhau hơn phù hợp với dữ liệu của chúng ta. Khi quan sát ngày càng nhiều dữ liệu, niềm tin của chúng ta về hàm thực trở nên chắc chắn hơn và bất định epistemic biến mất. Bất định _aleatoric_ trong trường hợp này là nhiễu quan sát, vì dữ liệu được cung cấp kèm theo nhiễu này và không thể giảm bớt.

Bất định _epistemic_ trong dữ liệu được biểu diễn bởi phương sai của hàm không nhiễu ẩn np.diag(post\_cov). Bất định _aleatoric_ được biểu diễn bởi phương sai nhiễu post_sig_est**2.

Đáng tiếc là mọi người thường bất cẩn trong cách biểu diễn bất định, với nhiều bài báo trình bày các thanh sai số hoàn toàn không được định nghĩa, không làm rõ liệu chúng ta đang trực quan hóa bất định epistemic, aleatoric hay cả hai, đồng thời nhầm lẫn giữa phương sai nhiễu với độ lệch chuẩn của nhiễu, độ lệch chuẩn với sai số chuẩn, khoảng tin cậy với miền tin cậy, v.v. Nếu không xác định chính xác bất định đang biểu diễn điều gì thì về cơ bản nó không có ý nghĩa.

Theo tinh thần chú ý sát sao đến ý nghĩa của bất định, điều quan trọng là phải lưu ý rằng chúng ta đang lấy _hai lần_ _căn bậc hai_ của ước lượng phương sai cho hàm không nhiễu. Vì phân phối dự đoán của chúng ta là Gaussian, đại lượng này cho phép chúng ta tạo ra một miền tin cậy 95\%, biểu diễn niềm tin của chúng ta về khoảng có 95\% khả năng chứa hàm thực. _Phương sai_ nhiễu nằm trên một thang đo hoàn toàn khác và khó diễn giải hơn nhiều.

Cuối cùng, hãy xem xét 20 mẫu hậu nghiệm. Các mẫu này cho chúng ta biết những loại hàm nào mà chúng ta tin rằng có thể khớp với dữ liệu của mình, theo hậu nghiệm.

```{.python .input}
post_samples = np.random.multivariate_normal(post_mean, post_cov, size=20)
d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y, linewidth=2.)
d2l.plt.plot(test_x, post_mean, linewidth=2.)
d2l.plt.plot(test_x, post_samples.T, color='gray', alpha=0.25)
d2l.plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
plt.legend(['Observed Data', 'True Function', 'Predictive Mean', 'Posterior Samples'])
d2l.plt.show()
```

Trong các ứng dụng hồi quy cơ bản, cách phổ biến nhất là sử dụng trung bình và độ lệch chuẩn dự đoán hậu nghiệm lần lượt làm dự đoán điểm và thước đo bất định. Trong các ứng dụng nâng cao hơn, chẳng hạn như Bayesian optimization với các hàm acquisition Monte Carlo, hoặc Gaussian process cho RL dựa trên mô hình, việc lấy các mẫu hậu nghiệm thường là cần thiết. Tuy nhiên, ngay cả khi không thực sự bắt buộc trong các ứng dụng cơ bản, những mẫu này giúp chúng ta có thêm trực giác về độ phù hợp với dữ liệu và thường hữu ích khi đưa vào phần trực quan hóa.

## Đơn giản hóa công việc với GPyTorch

Như đã thấy, việc triển khai hoàn toàn từ đầu một hồi quy Gaussian process cơ bản thực ra khá dễ. Tuy nhiên, ngay khi muốn khám phá nhiều lựa chọn kernel, xem xét approximate inference (vốn cần thiết ngay cả cho classification), kết hợp GP với neural network, hoặc thậm chí có một tập dữ liệu lớn hơn khoảng 10.000 điểm, việc triển khai từ đầu trở nên cồng kềnh và khó xử lý. Một số phương pháp hiệu quả nhất cho scalable GP inference, chẳng hạn như SKI (còn được gọi là KISS-GP), có thể yêu cầu hàng trăm dòng code để triển khai các routine đại số tuyến tính số nâng cao.

Trong những trường hợp này, thư viện _GPyTorch_ sẽ giúp công việc của chúng ta dễ dàng hơn nhiều. Chúng ta sẽ thảo luận thêm về GPyTorch trong các notebook sắp tới về số học Gaussian process và các phương pháp nâng cao. Thư viện GPyTorch có [nhiều ví dụ](https://github.com/cornellius-gp/gpytorch/tree/master/examples). Để làm quen với package, chúng ta sẽ đi qua [ví dụ hồi quy đơn giản](https://github.com/cornellius-gp/gpytorch/blob/master/examples/01_Exact_GPs/Simple_GP_Regression.ipynb), cho thấy cách điều chỉnh nó để tái tạo các kết quả trên bằng GPyTorch. Việc chỉ tái tạo bài toán hồi quy cơ bản ở trên mà cần từng này code có thể khiến bạn thấy nhiều, và theo một nghĩa nào đó thì đúng là như vậy. Nhưng chúng ta có thể ngay lập tức sử dụng nhiều kernel, các kỹ thuật inference có khả năng mở rộng và approximate inference, chỉ bằng cách thay đổi vài dòng code bên dưới, thay vì phải viết có thể đến hàng nghìn dòng code mới.

```{.python .input}
# First let's convert our data into tensors for use with PyTorch
train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)
test_y = torch.tensor(test_y)

# We are using exact GP inference with a zero mean and RBF kernel
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

Khối mã này chuyển dữ liệu sang đúng định dạng cho GPyTorch và chỉ định rằng chúng ta cũng đang sử dụng exact inference, cùng với hàm mean (zero) và hàm kernel (RBF) mà chúng ta muốn dùng. Chúng ta có thể dễ dàng sử dụng bất kỳ kernel nào khác, chẳng hạn bằng cách gọi `gpytorch.kernels.matern_kernel()` hoặc `gpyotrch.kernels.spectral_mixture_kernel()`. Cho đến nay, chúng ta mới chỉ thảo luận về exact inference, trong đó có thể suy luận một phân phối dự đoán mà không cần thực hiện bất kỳ phép xấp xỉ nào. Với Gaussian process, chúng ta chỉ có thể thực hiện exact inference khi có Gaussian likelihood; cụ thể hơn, khi giả định rằng các quan sát của chúng ta được sinh ra từ một hàm không có nhiễu được biểu diễn bởi một Gaussian process, cộng với Gaussian noise. Trong các notebook sau, chúng ta sẽ xét những thiết lập khác, chẳng hạn như classification, trong đó chúng ta không thể đưa ra các giả định này.

```{.python .input}
# Initialize Gaussian likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)
training_iter = 50
# Find optimal model hyperparameters
model.train()
likelihood.train()
# Use the adam optimizer, includes GaussianLikelihood parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  
# Set our loss as the negative log GP marginal likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
```

Ở đây, chúng ta chỉ rõ likelihood mà mình muốn sử dụng (Gaussian), objective sẽ dùng để training các kernel hyperparameter (ở đây là marginal likelihood), và procedure muốn dùng để tối ưu objective đó (trong trường hợp này là Adam). Lưu ý rằng mặc dù đang sử dụng Adam, vốn là một optimizer “stochastic”, nhưng trong trường hợp này đó là full-batch Adam. Vì marginal likelihood không factorize theo từng data instance, chúng ta không thể sử dụng optimizer trên các “mini-batch” dữ liệu mà vẫn được đảm bảo hội tụ. GPyTorch cũng hỗ trợ các optimizer khác, chẳng hạn như L-BFGS. Không giống deep learning tiêu chuẩn, việc tối ưu marginal likelihood tốt thường tương ứng chặt chẽ với khả năng generalization tốt, điều này thường khiến chúng ta ưu tiên các optimizer mạnh như L-BFGS, với điều kiện chi phí của chúng không quá lớn.

```{.python .input}
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    if i % 10 == 0:
        print(f'Iter {i+1:d}/{training_iter:d} - Loss: {loss.item():.3f} '
              f'squared lengthscale: '
              f'{model.covar_module.base_kernel.lengthscale.item():.3f} '
              f'noise variance: {model.likelihood.noise.item():.3f}')
    optimizer.step()
```

Ở đây, chúng ta thực sự chạy procedure tối ưu, đồng thời xuất các giá trị của loss sau mỗi 10 iteration.

```{.python .input}
# Get into evaluation (predictive posterior) mode
test_x = torch.tensor(test_x)
model.eval()
likelihood.eval()
observed_pred = likelihood(model(test_x)) 
```

Khối mã trên cho phép chúng ta đưa ra dự đoán trên các test input.

```{.python .input}
with torch.no_grad():
    # Initialize plot
    f, ax = d2l.plt.subplots(1, 1, figsize=(4, 3))
    # Get upper and lower bounds for 95\% credible set (in this case, in
    # observation space)
    lower, upper = observed_pred.confidence_region()
    ax.scatter(train_x.numpy(), train_y.numpy())
    ax.plot(test_x.numpy(), test_y.numpy(), linewidth=2.)
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), linewidth=2.)
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.25)
    ax.set_ylim([-1.5, 1.5])
    ax.legend(['True Function', 'Predictive Mean', 'Observed Data',
               '95% Credible Set'])
```

Cuối cùng, chúng ta vẽ kết quả fit.

Ta thấy các kết quả fit gần như giống hệt nhau. Có một vài điểm cần lưu ý: GPyTorch làm việc với length-scale _bình phương_ và observation noise _bình phương_. Ví dụ, độ lệch chuẩn noise học được trong mã from scratch là khoảng 0.283. Noise variance được GPyTorch tìm thấy là $0.81 \approx 0.283^2$. Trong biểu đồ GPyTorch, chúng ta cũng hiển thị credible set trong _observation space_ thay vì latent function space, để cho thấy rằng chúng thực sự bao phủ các datapoint đã quan sát.

## Tóm tắt

Chúng ta có thể kết hợp một Gaussian process prior với dữ liệu để tạo thành posterior, rồi sử dụng posterior này để đưa ra dự đoán. Chúng ta cũng có thể tạo marginal likelihood, vốn hữu ích cho việc tự động học các kernel hyperparameter; những hyperparameter này kiểm soát các thuộc tính như tốc độ biến thiên của Gaussian process. Cơ chế tạo posterior và học kernel hyperparameter cho regression rất đơn giản, chỉ cần khoảng một tá dòng mã. Notebook này là tài liệu tham khảo hữu ích cho bất kỳ độc giả nào muốn nhanh chóng “bắt đầu sử dụng” Gaussian process. Chúng ta cũng đã giới thiệu thư viện GPyTorch. Mặc dù mã GPyTorch cho regression cơ bản tương đối dài, nó có thể dễ dàng được sửa đổi để sử dụng các kernel function khác hoặc các chức năng nâng cao hơn mà chúng ta sẽ thảo luận trong những notebook sau, chẳng hạn như scalable inference hoặc non-Gaussian likelihood cho classification.


## Bài tập

1. Chúng ta đã nhấn mạnh tầm quan trọng của việc _học_ kernel hyperparameter, cũng như ảnh hưởng của hyperparameter và kernel đến các thuộc tính generalization của Gaussian process. Hãy thử bỏ qua bước học hyperparameter, thay vào đó đoán nhiều length-scale và noise variance khác nhau, rồi kiểm tra ảnh hưởng của chúng lên các dự đoán. Điều gì xảy ra khi bạn sử dụng length-scale lớn? Length-scale nhỏ? Noise variance lớn? Noise variance nhỏ?
2. Chúng ta đã nói rằng marginal likelihood không phải là một objective lồi, nhưng các hyperparameter như length-scale và noise variance vẫn có thể được ước lượng một cách đáng tin cậy trong GP regression. Điều này nhìn chung là đúng --- thực tế, marginal likelihood học các length-scale hyperparameter tốt hơn _rất nhiều_ so với các phương pháp truyền thống trong spatial statistics, vốn liên quan đến việc fit các hàm autocorrelation thực nghiệm (“covariogram”). Có thể lập luận rằng đóng góp lớn nhất của machine learning cho nghiên cứu Gaussian process, ít nhất là trước những công trình gần đây về scalable inference, chính là việc đưa marginal likelihood vào quá trình học hyperparameter.

*Tuy nhiên*, các cặp kết hợp khác nhau, ngay cả với những parameter này, vẫn đưa ra các cách giải thích hợp lý khác nhau về nhiều dataset, dẫn đến các local optima trong objective của chúng ta. Nếu sử dụng length-scale lớn, chúng ta giả định rằng hàm nền tảng thực sự biến thiên chậm. Nếu dữ liệu quan sát _thực sự_ biến thiên đáng kể, thì cách duy nhất để một length-scale lớn trở nên hợp lý là có noise-variance lớn. Mặt khác, nếu sử dụng length-scale nhỏ, kết quả fit sẽ rất nhạy với các biến thiên trong dữ liệu, khiến còn rất ít khả năng giải thích các biến thiên bằng noise (aleatoric uncertainty).

Hãy thử xem bạn có thể tìm thấy các local optima này không: khởi tạo với length-scale rất lớn cùng noise lớn, và length-scale nhỏ cùng noise nhỏ. Bạn có hội tụ đến các nghiệm khác nhau không?
  
3. Chúng ta đã nói rằng một ưu điểm nền tảng của các phương pháp Bayesian là khả năng biểu diễn tự nhiên _epistemic_ uncertainty. Trong ví dụ trên, chúng ta không thể quan sát đầy đủ các ảnh hưởng của epistemic uncertainty. Thay vào đó, hãy thử dự đoán với `test_x = np.linspace(0, 10, 1000)`. Điều gì xảy ra với credible set 95\% khi các dự đoán của bạn tiến ra ngoài phạm vi dữ liệu? Nó có bao phủ hàm thực trong khoảng đó không? Điều gì xảy ra nếu bạn chỉ trực quan hóa aleatoric uncertainty trong vùng đó?

4. Hãy thử chạy ví dụ trên với lần lượt 10,000, 20,000 và 40,000 training point, rồi đo runtime. Thời gian training tăng theo quy mô như thế nào? Ngoài ra, runtime thay đổi như thế nào theo số lượng test point? Điều đó có khác nhau giữa predictive mean và predictive variance không? Hãy trả lời câu hỏi này cả bằng cách tính về mặt lý thuyết độ phức tạp thời gian của training và testing, lẫn bằng cách chạy đoạn mã trên với số lượng point khác nhau.

5. Hãy thử chạy ví dụ GPyTorch với các covariance function khác nhau, chẳng hạn như Matern kernel. Kết quả thay đổi như thế nào? Còn spectral mixture kernel trong thư viện GPyTorch thì sao? Có kernel nào dễ training marginal likelihood hơn những kernel khác không? Có kernel nào hữu ích hơn cho dự đoán dài hạn so với dự đoán ngắn hạn không?

6. Trong ví dụ GPyTorch, chúng ta đã vẽ predictive distribution có bao gồm observation noise, trong khi ở ví dụ “from scratch”, chúng ta chỉ đưa vào epistemic uncertainty. Hãy thực hiện lại ví dụ GPyTorch, nhưng lần này chỉ vẽ epistemic uncertainty, rồi so sánh với kết quả from scratch. Các predictive distribution bây giờ có trông giống nhau không? (Chúng phải giống nhau.)

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/12117)
:end_tab: