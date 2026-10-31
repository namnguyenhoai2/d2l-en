```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['pytorch'])
```

# Prior của Gaussian Process

Hiểu về Gaussian process (GP) rất quan trọng để suy luận về việc xây dựng mô hình và khả năng tổng quát hóa, cũng như đạt được hiệu năng tiên tiến trong nhiều ứng dụng, bao gồm active learning và tinh chỉnh hyperparameter trong deep learning. GP xuất hiện ở khắp nơi, và chúng ta nên biết chúng là gì cũng như cách sử dụng chúng.

Trong phần này, chúng ta giới thiệu các _prior_ của Gaussian process trên các hàm. Trong notebook tiếp theo, chúng ta sẽ trình bày cách sử dụng các prior này để thực hiện _suy luận posterior_ và đưa ra dự đoán. Phần tiếp theo có thể được xem như “GP tóm lược”, nhanh chóng cung cấp những gì bạn cần để áp dụng Gaussian process trong thực tế.

```{.python .input}
from d2l import torch as d2l
import numpy as np
from scipy.spatial import distance_matrix

d2l.set_figsize()
```

## Định nghĩa

Gaussian process được định nghĩa là _một tập hợp các biến ngẫu nhiên, trong đó mọi số lượng hữu hạn biến đều có phân phối Gaussian đồng thời_. Nếu hàm $f(x)$ là một Gaussian process, với _hàm trung bình_ $m(x)$ và _hàm hiệp phương sai_ hoặc _kernel_ $k(x,x')$, $f(x) \sim \mathcal{GP}(m, k)$, thì mọi tập hợp các giá trị hàm được truy vấn tại mọi tập hợp các điểm đầu vào $x$ (thời điểm, vị trí không gian, pixel ảnh, v.v.) đều có phân phối Gaussian đa biến đồng thời với vector trung bình $\mu$ và ma trận hiệp phương sai $K$: $f(x_1),\dots,f(x_n) \sim \mathcal{N}(\mu, K)$, trong đó $\mu_i = E[f(x_i)] = m(x_i)$ và $K_{ij} = \textrm{Cov}(f(x_i),f(x_j)) = k(x_i,x_j)$.

Định nghĩa này có vẻ trừu tượng và khó tiếp cận, nhưng thực tế Gaussian process là những đối tượng rất đơn giản. Mọi hàm

$$f(x) = w^{\top} \phi(x) = \langle w, \phi(x) \rangle,$$ :eqlabel:`eq_gp-function`

với $w$ được lấy từ một phân phối Gaussian (chuẩn), và $\phi$ là một vector bất kỳ gồm các hàm cơ sở, chẳng hạn $\phi(x) = (1, x, x^2, ..., x^d)^{\top}$,
đều là một Gaussian process. Hơn nữa, mọi Gaussian process f(x) đều có thể được biểu diễn dưới dạng phương trình :eqref:`eq_gp-function`. Hãy cùng xem xét một vài ví dụ cụ thể để bắt đầu làm quen với Gaussian process, từ đó chúng ta có thể thấy chúng thực sự đơn giản và hữu ích như thế nào.

## Một Gaussian Process đơn giản

Giả sử $f(x) = w_0 + w_1 x$, và $w_0, w_1 \sim \mathcal{N}(0,1)$, trong đó $w_0, w_1, x$ đều là các đại lượng một chiều. Ta có thể viết tương đương hàm này dưới dạng tích vô hướng $f(x) = (w_0, w_1)(1, x)^{\top}$. Trong :eqref:`eq_gp-function` ở trên, $w = (w_0, w_1)^{\top}$ và $\phi(x) = (1,x)^{\top}$.

Với mọi $x$, $f(x)$ là tổng của hai biến ngẫu nhiên Gaussian. Vì các biến Gaussian đóng dưới phép cộng, $f(x)$ cũng là một biến ngẫu nhiên Gaussian với mọi $x$. Cụ thể, với mọi $x$ cụ thể, ta có thể tính được $f(x)$ là $\mathcal{N}(0,1+x^2)$. Tương tự, phân phối đồng thời của mọi tập hợp các giá trị hàm, $(f(x_1),\dots,f(x_n))$, với mọi tập hợp đầu vào $x_1,\dots,x_n$, là một phân phối Gaussian đa biến. Do đó, $f(x)$ là một Gaussian process.

Tóm lại, $f(x)$ là một _hàm ngẫu nhiên_, hay một _phân phối trên các hàm_. Ta có thể hiểu thêm về phân phối này bằng cách lấy mẫu lặp lại các giá trị của $w_0, w_1$ và trực quan hóa các hàm $f(x)$ tương ứng, vốn là những đường thẳng có độ dốc và tung độ gốc khác nhau, như sau:

```{.python .input}
def lin_func(x, n_sample):
    preds = np.zeros((n_sample, x.shape[0]))
    for ii in range(n_sample):
        w = np.random.normal(0, 1, 2)
        y = w[0] + w[1] * x
        preds[ii, :] = y
    return preds

x_points = np.linspace(-5, 5, 50)
outs = lin_func(x_points, 10)
lw_bd = -2 * np.sqrt((1 + x_points ** 2))
up_bd = 2 * np.sqrt((1 + x_points ** 2))

d2l.plt.fill_between(x_points, lw_bd, up_bd, alpha=0.25)
d2l.plt.plot(x_points, np.zeros(len(x_points)), linewidth=4, color='black')
d2l.plt.plot(x_points, outs.T)
d2l.plt.xlabel("x", fontsize=20)
d2l.plt.ylabel("f(x)", fontsize=20)
d2l.plt.show()
```

Nếu thay vào đó $w_0$ và $w_1$ được lấy từ $\mathcal{N}(0,\alpha^2)$, bạn hình dung việc thay đổi $\alpha$ sẽ ảnh hưởng đến phân phối trên các hàm như thế nào?

## Từ không gian trọng số đến không gian hàm

Trong biểu đồ trên, chúng ta đã thấy cách một phân phối trên các tham số của mô hình tạo ra một phân phối trên các hàm. Mặc dù chúng ta thường có hình dung về những hàm mình muốn mô hình hóa --- chẳng hạn chúng trơn, tuần hoàn, biến thiên nhanh, v.v. --- việc suy luận về các tham số, vốn phần lớn không thể diễn giải được, lại tương đối tẻ nhạt. May mắn thay, Gaussian process cung cấp một cơ chế đơn giản để suy luận _trực tiếp_ về các hàm. Vì một phân phối Gaussian được xác định hoàn toàn bởi hai moment đầu tiên của nó, tức trung bình và ma trận hiệp phương sai, nên Gaussian process cũng được xác định bởi hàm trung bình và hàm hiệp phương sai.

Trong ví dụ trên, hàm trung bình là

$$m(x) = E[f(x)] = E[w_0 + w_1x] = E[w_0] + E[w_1]x = 0+0 = 0.$$

Tương tự, hàm hiệp phương sai là

$$k(x,x') = \textrm{Cov}(f(x),f(x')) = E[f(x)f(x')]-E[f(x)]E[f(x')] = E[w_0^2 + w_0w_1x' + w_1w_0x + w_1^2xx'] = 1 + xx'.$$

Giờ đây, ta có thể chỉ định và lấy mẫu trực tiếp từ phân phối trên các hàm mà không cần lấy mẫu từ phân phối trên các tham số. Chẳng hạn, để lấy mẫu từ $f(x)$, ta chỉ cần tạo phân phối Gaussian đa biến tương ứng với bất kỳ tập hợp $x$ nào mà ta muốn truy vấn, rồi lấy mẫu trực tiếp từ phân phối đó. Chúng ta sẽ dần thấy cách biểu diễn này mang lại nhiều lợi thế như thế nào.

Trước hết, ta lưu ý rằng về cơ bản, cùng một phép suy diễn cho mô hình đường thẳng đơn giản ở trên có thể được áp dụng để tìm hàm trung bình và hàm hiệp phương sai cho _bất kỳ_ mô hình nào có dạng $f(x) = w^{\top} \phi(x)$, với $w \sim \mathcal{N}(u,S)$. Trong trường hợp này, hàm trung bình $m(x) = u^{\top}\phi(x)$, và hàm hiệp phương sai $k(x,x') = \phi(x)^{\top}S\phi(x')$. Vì $\phi(x)$ có thể biểu diễn một vector gồm bất kỳ hàm cơ sở phi tuyến nào, ta đang xét một lớp mô hình rất tổng quát, bao gồm cả các mô hình có số lượng tham số _vô hạn_.

## Kernel Radial Basis Function (RBF)

Kernel _radial basis function_ (RBF) là hàm hiệp phương sai phổ biến nhất đối với Gaussian process, cũng như đối với các kernel machine nói chung.
Kernel này có dạng $k_{\textrm{RBF}}(x,x') = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right)$, trong đó $a$ là tham số biên độ, còn $\ell$ là hyperparameter _lengthscale_.

Hãy suy diễn kernel này bắt đầu từ không gian trọng số. Xét hàm

$$f(x) = \sum_{i=1}^J w_i \phi_i(x), w_i  \sim \mathcal{N}\left(0,\frac{\sigma^2}{J}\right), \phi_i(x) = \exp\left(-\frac{(x-c_i)^2}{2\ell^2 }\right).$$

$f(x)$ là tổng của các hàm radial basis, có độ rộng $\ell$, tâm tại các điểm $c_i$, như minh họa trong hình dưới đây.


Ta có thể nhận ra $f(x)$ có dạng $w^{\top} \phi(x)$, trong đó $w = (w_1,\dots,w_J)^{\top}$ và $\phi(x)$ là một vector chứa từng hàm radial basis. Khi đó, hàm hiệp phương sai của Gaussian process này là

$$k(x,x') = \frac{\sigma^2}{J} \sum_{i=1}^{J} \phi_i(x)\phi_i(x').$$

Bây giờ hãy xem điều gì xảy ra khi ta tăng số lượng tham số (và hàm cơ sở) lên vô hạn. Đặt $c_J = \log J$, $c_1 = -\log J$, và $c_{i+1}-c_{i} = \Delta c = 2\frac{\log J}{J}$, và $J \to \infty$. Hàm hiệp phương sai trở thành tổng Riemann:

$$k(x,x') = \lim_{J \to \infty} \frac{\sigma^2}{J} \sum_{i=1}^{J} \phi_i(x)\phi_i(x') = \int_{c_0}^{c_\infty} \phi_c(x)\phi_c(x') dc.$$

Bằng cách đặt $c_0 = -\infty$ và $c_\infty = \infty$, ta trải vô hạn hàm cơ sở trên toàn bộ trục số thực, mỗi hàm
cách nhau một khoảng $\Delta c \to 0$:

$$k(x,x') = \int_{-\infty}^{\infty} \exp(-\frac{(x-c)^2}{2\ell^2}) \exp(-\frac{(x'-c)^2}{2\ell^2 }) dc = \sqrt{\pi}\ell \sigma^2 \exp(-\frac{(x-x')^2}{2(\sqrt{2} \ell)^2}) \propto k_{\textrm{RBF}}(x,x').$$

Ta nên dừng lại một chút để nắm rõ những gì vừa thực hiện. Bằng cách chuyển sang biểu diễn trong không gian hàm, ta đã suy ra cách biểu diễn một mô hình có số lượng tham số _vô hạn_ bằng một lượng tính toán hữu hạn. Gaussian process với kernel RBF là một _bộ xấp xỉ phổ quát_, có khả năng biểu diễn mọi hàm liên tục với độ chính xác tùy ý. Ta có thể hình dung trực giác lý do qua phép suy diễn ở trên. Ta có thể thu gọn mỗi hàm radial basis thành một khối lượng điểm bằng cách cho $\ell \to 0$, rồi gán cho mỗi khối lượng điểm một độ cao bất kỳ.

Vì vậy, Gaussian process với kernel RBF là một mô hình có số lượng tham số vô hạn và linh hoạt hơn nhiều so với bất kỳ neural network hữu hạn nào. Có lẽ mọi ồn ào về các neural network _thừa tham số_ là không đúng trọng tâm. Như chúng ta sẽ thấy, GP với kernel RBF không overfit, mà thực tế còn mang lại khả năng tổng quát hóa đặc biệt thuyết phục trên các tập dữ liệu nhỏ. Hơn nữa, các ví dụ trong :cite:`zhang2021understanding`, chẳng hạn như khả năng khớp hoàn hảo các ảnh có nhãn ngẫu nhiên nhưng vẫn tổng quát hóa tốt trên các bài toán có cấu trúc, (có thể được tái tạo hoàn hảo bằng Gaussian process) :cite:`wilson2020bayesian`. Neural network không khác biệt rõ rệt như cách chúng ta thường nghĩ.

Ta có thể xây dựng thêm trực giác về Gaussian process với kernel RBF và các hyperparameter như _length-scale_ bằng cách lấy mẫu trực tiếp từ phân phối trên các hàm. Như trước đây, quy trình này gồm các bước đơn giản:

1. Chọn các điểm đầu vào $x$ mà ta muốn truy vấn GP: $x_1,\dots,x_n$.
2. Tính $m(x_i)$, $i = 1,\dots,n$, và $k(x_i,x_j)$ với $i,j = 1,\dots,n$ để lần lượt tạo vector trung bình và ma trận hiệp phương sai $\mu$ và $K$, trong đó $(f(x_1),\dots,f(x_n)) \sim \mathcal{N}(\mu, K)$.
3. Lấy mẫu từ phân phối Gaussian đa biến này để thu được các giá trị hàm của mẫu.
4. Lấy mẫu thêm nhiều lần để trực quan hóa nhiều hàm mẫu hơn được truy vấn tại các điểm đó.

Chúng ta minh họa quy trình này trong hình dưới đây.

```{.python .input}
def rbfkernel(x1, x2, ls=4.):  #@save
    dist = distance_matrix(np.expand_dims(x1, 1), np.expand_dims(x2, 1))
    return np.exp(-(1. / ls / 2) * (dist ** 2))

x_points = np.linspace(0, 5, 50)
meanvec = np.zeros(len(x_points))
covmat = rbfkernel(x_points,x_points, 1)

prior_samples= np.random.multivariate_normal(meanvec, covmat, size=5);
d2l.plt.plot(x_points, prior_samples.T, alpha=0.5)
d2l.plt.show()
```

## Kernel Neural Network

Nghiên cứu về quá trình Gaussian trong machine learning bắt nguồn từ nghiên cứu về neural network. Radford Neal đang theo đuổi các Bayesian neural network ngày càng lớn hơn, và cuối cùng đã chỉ ra vào năm 1994 (sau đó được công bố vào năm 1996, vì đây là một trong những bài bị NeurIPS từ chối tai tiếng nhất) rằng các mạng như vậy với số lượng hidden unit vô hạn sẽ trở thành các quá trình Gaussian với những hàm kernel cụ thể :cite:`neal1996bayesian`. Mối quan tâm đối với phép suy luận này đã xuất hiện trở lại, với những ý tưởng như neural tangent kernel được sử dụng để nghiên cứu các tính chất tổng quát hóa của neural network :cite:`matthews2018gaussian` :cite:`novak2018bayesian`. Ta có thể suy ra kernel của neural network như sau.

Xét một hàm neural network $f(x)$ với một hidden layer:

$$f(x) = b + \sum_{i=1}^{J} v_i h(x; u_i).$$

$b$ là bias, $v_i$ là các trọng số từ hidden layer đến output, $h$ là một hàm truyền của hidden unit bất kỳ bị chặn, $u_i$ là các trọng số từ input đến hidden layer, và $J$ là số lượng hidden unit. Cho $b$ và $v_i$ độc lập với kỳ vọng bằng 0 và phương sai lần lượt là $\sigma_b^2$ và $\sigma_v^2/J$, đồng thời cho $u_i$ có các phân phối độc lập và đồng nhất. Khi đó, ta có thể sử dụng định lý giới hạn trung tâm để chỉ ra rằng mọi tập hợp các giá trị hàm $f(x_1),\dots,f(x_n)$ đều có phân phối Gaussian đa biến đồng thời.

Hàm trung bình và hàm hiệp phương sai của quá trình Gaussian tương ứng là:

$$m(x) = E[f(x)] = 0$$

$$k(x,x') = \textrm{cov}[f(x),f(x')] = E[f(x)f(x')] = \sigma_b^2 + \frac{1}{J} \sum_{i=1}^{J} \sigma_v^2 E[h_i(x; u_i)h_i(x'; u_i)]$$

Trong một số trường hợp, về cơ bản ta có thể tính hàm hiệp phương sai này dưới dạng đóng. Cho $h(x; u) = \textrm{erf}(u_0 + \sum_{j=1}^{P} u_j x_j)$, trong đó $\textrm{erf}(z) = \frac{2}{\sqrt{\pi}} \int_{0}^{z} e^{-t^2} dt$, và $u \sim \mathcal{N}(0,\Sigma)$. Khi đó $k(x,x') = \frac{2}{\pi} \textrm{sin}(\frac{2 \tilde{x}^{\top} \Sigma \tilde{x}'}{\sqrt{(1 + 2 \tilde{x}^{\top} \Sigma \tilde{x})(1 + 2 \tilde{x}'^{\top} \Sigma \tilde{x}')}})$.

RBF kernel là _stationary_, nghĩa là _bất biến theo phép tịnh tiến_ (_translation invariant_), và do đó có thể được viết dưới dạng một hàm của $\tau = x-x'$. Theo trực giác, tính stationary có nghĩa là các thuộc tính cấp cao của hàm, chẳng hạn như tốc độ biến thiên, không thay đổi khi ta di chuyển trong không gian input. Tuy nhiên, neural network kernel là _non-stationary_. Bên dưới, ta minh họa các hàm mẫu từ một quá trình Gaussian với kernel này. Có thể thấy hàm này có diện mạo khác biệt về mặt định tính khi ở gần gốc tọa độ.

## Tóm tắt


Bước đầu tiên khi thực hiện Bayesian inference là xác định một prior. Quá trình Gaussian có thể được sử dụng để xác định toàn bộ một prior trên các hàm. Bắt đầu từ cách nhìn nhận mô hình truyền thống trong “weight space”, ta có thể tạo ra một prior trên các hàm bằng cách bắt đầu với dạng hàm của một mô hình và đưa vào một phân phối trên các tham số của mô hình đó. Ngoài ra, ta có thể xác định trực tiếp một phân phối prior trong function space, với các thuộc tính được kernel kiểm soát. Cách tiếp cận trong function space có nhiều ưu điểm. Ta có thể xây dựng các mô hình thực sự tương ứng với vô hạn tham số, nhưng chỉ cần sử dụng một lượng tính toán hữu hạn! Hơn nữa, mặc dù các mô hình này rất linh hoạt, chúng cũng đưa ra những giả định mạnh về loại hàm nào có khả năng xảy ra theo prior, từ đó dẫn đến khả năng tổng quát hóa tương đối tốt trên các tập dữ liệu nhỏ.

Các giả định của những mô hình trong function space được kernel kiểm soát một cách trực quan; các kernel thường mã hóa những thuộc tính cấp cao hơn của hàm, chẳng hạn như độ trơn và tính tuần hoàn. Nhiều kernel là stationary, nghĩa là bất biến theo phép tịnh tiến. Các hàm được lấy mẫu từ một quá trình Gaussian với stationary kernel có gần như cùng các thuộc tính cấp cao (chẳng hạn như tốc độ biến thiên), bất kể ta quan sát ở đâu trong không gian input.

Quá trình Gaussian là một lớp mô hình tương đối tổng quát, bao gồm nhiều ví dụ về các mô hình mà ta đã quen thuộc, chẳng hạn như đa thức, chuỗi Fourier, v.v., miễn là ta có Gaussian prior trên các tham số. Chúng cũng bao gồm các neural network với số lượng tham số vô hạn, ngay cả khi các tham số không có phân phối Gaussian. Mối liên hệ này, được Radford Neal phát hiện, đã thúc đẩy các nhà nghiên cứu machine learning rời xa neural network và chuyển sang quá trình Gaussian.


## Bài tập

1. Vẽ các hàm prior mẫu từ một GP với Ornstein-Uhlenbeck (OU) kernel, $k_{\textrm{OU}}(x,x') = \exp\left(-\frac{1}{2\ell}||x - x'|\right)$. Nếu cố định lengthscale $\ell$ ở cùng một giá trị, các hàm này trông khác với các hàm mẫu từ một GP với RBF kernel như thế nào?

2. Việc thay đổi _amplitude_ $a^2$ của RBF kernel ảnh hưởng thế nào đến phân phối trên các hàm?

3. Giả sử ta tạo thành $u(x) = f(x) + 2g(x)$, trong đó $f(x) \sim \mathcal{GP}(m_1,k_1)$ và $g(x) \sim \mathcal{GP}(m_2,k_2)$. $u(x)$ có phải là một quá trình Gaussian không, và nếu có thì hàm trung bình và hàm hiệp phương sai của nó là gì?

4. Giả sử ta tạo thành $g(x) = a(x)f(x)$, trong đó $f(x) \sim \mathcal{GP}(0,k)$ và $a(x) = x^2$. $g(x)$ có phải là một quá trình Gaussian không, và nếu có thì hàm trung bình và hàm hiệp phương sai của nó là gì? $a(x)$ có tác động gì? Các hàm mẫu được lấy từ $g(x)$ trông như thế nào?

5. Giả sử ta tạo thành $u(x) = f(x)g(x)$, trong đó $f(x) \sim \mathcal{GP}(m_1,k_1)$ và $g(x) \sim \mathcal{GP}(m_2,k_2)$. $u(x)$ có phải là một quá trình Gaussian không, và nếu có thì hàm trung bình và hàm hiệp phương sai của nó là gì?

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/12116)
:end_tab: