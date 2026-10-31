# Giới thiệu về Gaussian Processes

Trong nhiều trường hợp, machine learning về bản chất là việc ước lượng các tham số từ dữ liệu. Những tham số này thường rất nhiều và tương đối khó diễn giải --- chẳng hạn như các trọng số của một neural network. Ngược lại, Gaussian processes cung cấp một cơ chế để trực tiếp suy luận về các thuộc tính cấp cao của những hàm có thể khớp với dữ liệu của chúng ta. Ví dụ, chúng ta có thể biết được liệu các hàm này có biến thiên nhanh, có tính tuần hoàn, bao hàm các độc lập có điều kiện hay có tính bất biến theo phép tịnh tiến hay không. Gaussian processes cho phép chúng ta dễ dàng đưa những thuộc tính này vào model bằng cách trực tiếp xác định một phân phối Gaussian trên các giá trị hàm có thể khớp với dữ liệu.

Hãy bắt đầu với một vài ví dụ để hình dung cách Gaussian processes hoạt động.

Giả sử chúng ta quan sát dataset sau đây, gồm các regression target (đầu ra), $y$, được đánh chỉ mục bởi các input, $x$. Ví dụ, target có thể là những thay đổi trong nồng độ carbon dioxide, còn input có thể là những thời điểm mà các target này được ghi nhận. Dữ liệu có những đặc điểm nào? Có vẻ dữ liệu biến thiên nhanh đến mức nào? Chúng ta có các điểm dữ liệu được thu thập ở những khoảng thời gian đều đặn, hay có những input bị thiếu? Bạn sẽ hình dung việc điền vào các vùng bị thiếu hoặc dự báo cho đến $x=25$ như thế nào?

![Dữ liệu quan sát được.](../img/gp-observed-data.svg)

Để khớp dữ liệu bằng một Gaussian process, trước hết chúng ta xác định một prior distribution về những loại hàm mà chúng ta cho là hợp lý. Ở đây, chúng ta hiển thị một số hàm mẫu từ một Gaussian process. Prior này có vẻ hợp lý không? Lưu ý rằng ở đây chúng ta không tìm kiếm những hàm khớp với dataset, mà thay vào đó xác định các thuộc tính cấp cao hợp lý của những nghiệm, chẳng hạn như tốc độ biến thiên theo các input. Lưu ý rằng trong các notebook tiếp theo về priors và inference, chúng ta sẽ thấy code để tái tạo tất cả các biểu đồ trong notebook này.

![Các hàm prior mẫu mà chúng ta có thể muốn biểu diễn bằng model của mình.](../img/gp-sample-prior-functions.svg)

Sau khi điều kiện hóa theo dữ liệu, chúng ta có thể sử dụng prior này để suy ra một posterior distribution trên các hàm có thể khớp với dữ liệu. Ở đây, chúng ta hiển thị các hàm posterior mẫu.

![Các hàm posterior mẫu sau khi chúng ta đã quan sát dữ liệu.](../img/gp-sample-posterior-functions.svg)

Chúng ta thấy rằng mỗi hàm này hoàn toàn nhất quán với dữ liệu, đi chính xác qua từng quan sát. Để sử dụng các posterior sample này nhằm đưa ra dự đoán, chúng ta có thể lấy trung bình các giá trị của mọi hàm mẫu khả dĩ từ posterior, tạo ra đường cong bên dưới, có màu xanh lam đậm. Lưu ý rằng chúng ta thực sự không cần lấy vô hạn mẫu để tính kỳ vọng này; như sẽ thấy sau, chúng ta có thể tính kỳ vọng dưới dạng đóng.

![Các posterior sample cùng với posterior mean, có thể được sử dụng để đưa ra point prediction, bằng màu xanh lam.](../img/gp-posterior-samples.svg)

Chúng ta cũng có thể muốn có một biểu diễn về độ bất định, để biết mình nên tin tưởng các dự đoán đến mức nào. Theo trực giác, độ bất định sẽ lớn hơn ở những nơi có nhiều biến thiên hơn giữa các hàm posterior mẫu, vì điều này cho thấy có nhiều giá trị khả dĩ hơn mà hàm thực có thể nhận. Loại độ bất định này được gọi là _epistemic uncertainty_, tức _reducible uncertainty_ gắn với việc thiếu thông tin. Khi thu thập thêm dữ liệu, loại độ bất định này biến mất, vì ngày càng có ít nghiệm hơn nhất quán với những gì chúng ta quan sát được. Tương tự như posterior mean, chúng ta có thể tính posterior variance (mức độ biến thiên của các hàm trong posterior) dưới dạng đóng. Bằng vùng tô bóng, chúng ta biểu diễn hai lần posterior standard deviation ở mỗi phía của mean, tạo thành một _credible interval_ có xác suất 95% chứa giá trị thực của hàm với mọi input $x$.

![Các posterior sample, bao gồm 95% credible set.](../img/gp-posterior-samples-95.svg)

Biểu đồ trông gọn gàng hơn nếu loại bỏ các posterior sample, chỉ trực quan hóa dữ liệu, posterior mean và 95% credible set. Hãy chú ý rằng độ bất định tăng lên khi rời xa dữ liệu, đây là một thuộc tính của epistemic uncertainty.

![Point prediction và credible set.](../img/gp-point-predictions.svg)

Các thuộc tính của Gaussian process mà chúng ta sử dụng để khớp dữ liệu được kiểm soát chặt chẽ bởi thứ được gọi là _covariance function_, còn được biết đến là _kernel_. Covariance function mà chúng ta sử dụng được gọi là _RBF (Radial Basis Function) kernel_, có dạng
$$ k_{\textrm{RBF}}(x,x') = \textrm{Cov}(f(x),f(x')) = a^2 \exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right) $$

Các _hyperparameter_ của kernel này có thể diễn giải được. Tham số _amplitude_ $a$ kiểm soát thang đo theo chiều dọc mà trên đó hàm biến thiên, còn tham số _length-scale_
$\ell$
kiểm soát tốc độ biến thiên (mức độ uốn lượn) của hàm. $a$ lớn hơn có nghĩa là các giá trị hàm lớn hơn, còn
$\ell$
lớn hơn có nghĩa là các hàm biến thiên chậm hơn. Hãy xem điều gì xảy ra với các hàm prior và posterior mẫu khi chúng ta thay đổi $a$ và
$\ell$.

_Length-scale_ có ảnh hưởng đặc biệt rõ rệt đến các dự đoán và độ bất định của một GP. Khi
$||x-x'|| = \ell$
, covariance giữa một cặp giá trị hàm là $a^2\exp(-0.5)$. Ở những khoảng cách lớn hơn
$\ell$
, các giá trị hàm trở nên gần như không tương quan. Điều này có nghĩa là nếu chúng ta muốn đưa ra dự đoán tại một điểm $x_*$, thì các giá trị hàm có input $x$ sao cho
$||x-x'||>\ell$
sẽ không ảnh hưởng đáng kể đến các dự đoán của chúng ta.

Hãy xem việc thay đổi lengthscale ảnh hưởng như thế nào đến các hàm prior và posterior mẫu cũng như credible set. Các phép khớp ở trên sử dụng length-scale bằng $2$. Bây giờ hãy xét
$\ell = 0.1, 0.5, 2, 5, 10$
. Length-scale bằng $0.1$ rất nhỏ so với phạm vi của miền input mà chúng ta đang xét, là $25$. Ví dụ, các giá trị của hàm tại $x=5$ và $x=10$ về cơ bản sẽ không tương quan ở length-scale như vậy. Ngược lại, với length-scale bằng $10$, các giá trị hàm tại những input này sẽ tương quan rất cao. Lưu ý rằng thang đo theo chiều dọc thay đổi trong các hình sau.


![priorpoint1](../img/gp-priorpoint1.svg)
![postpoint1](../img/gp-postpoint1.svg)

![priorpoint5](../img/gp-priorpoint5.svg)
![postpoint5](../img/gp-postpoint5.svg)

![prior2](../img/gp-prior2.svg)
![post2](../img/gp-post2.svg)

![prior5](../img/gp-prior5.svg)
![post5](../img/gp-post5.svg)

Hãy chú ý rằng khi length-scale tăng, mức độ “uốn lượn” của các hàm giảm xuống và độ bất định của chúng ta cũng giảm. Nếu length-scale nhỏ, độ bất định sẽ nhanh chóng tăng lên khi chúng ta rời xa dữ liệu, vì các điểm dữ liệu trở nên kém hữu ích hơn trong việc cung cấp thông tin về các giá trị hàm.

Bây giờ, hãy thay đổi tham số amplitude, đồng thời giữ length-scale cố định ở $2$. Lưu ý rằng thang đo theo chiều dọc được giữ cố định đối với các prior sample và thay đổi đối với các posterior sample, để bạn có thể thấy rõ cả thang đo tăng lên của hàm lẫn các phép khớp với dữ liệu.


![priorap1](../img/gp-priorap1.svg)
![postapoint1](../img/gp-postapoint1.svg)

![priora2](../img/gp-priora2.svg)
![posta2](../img/gp-posta2.svg)

![priora8](../img/gp-priora8.svg)
![posta8](../img/gp-posta8.svg)

Chúng ta thấy rằng tham số amplitude ảnh hưởng đến thang đo của hàm, nhưng không ảnh hưởng đến tốc độ biến thiên. Đến đây, chúng ta cũng nhận thấy rằng hiệu năng generalization của quy trình sẽ phụ thuộc vào việc có các giá trị hợp lý cho những hyperparameter này. Các giá trị $\ell=2$ và $a=1$ có vẻ tạo ra những phép khớp hợp lý, trong khi một số giá trị khác thì không. May mắn là có một cách mạnh mẽ và tự động để xác định các hyperparameter này, sử dụng thứ được gọi là _marginal likelihood_, mà chúng ta sẽ quay lại trong notebook về inference.

Vậy thực sự GP là gì? Như đã nói từ đầu, GP đơn giản phát biểu rằng mọi tập hợp các giá trị hàm
$f(x_1),\dots,f(x_n)$,
được đánh chỉ mục bởi mọi tập hợp các input
$x_1,\dots,x_n$
, đều có một joint multivariate Gaussian distribution. Vector mean $\mu$ của phân phối này được xác định bởi một _mean function_, thường được chọn là một hằng số hoặc bằng không. Ma trận covariance của phân phối này được xác định bởi _kernel_ được đánh giá trên mọi cặp input $x$.

$$\begin{bmatrix}f(x) \\f(x_1) \\ \vdots \\ f(x_n) \end{bmatrix}\sim \mathcal{N}\left(\mu, \begin{bmatrix}k(x,x) & k(x, x_1) & \dots & k(x,x_n) \\ k(x_1,x) & k(x_1,x_1) & \dots & k(x_1,x_n) \\ \vdots & \vdots & \ddots & \vdots \\ k(x_n, x) & k(x_n, x_1) & \dots & k(x_n,x_n) \end{bmatrix}\right)$$
:eqlabel:`eq_gp_prior`:

Phương trình :eqref:`eq_gp_prior` chỉ ra một prior GP. Ta có thể tính phân phối có điều kiện của $f(x)$ với mọi $x$ khi biết $f(x_1), \dots, f(x_n)$, tức các giá trị hàm mà ta đã quan sát. Phân phối có điều kiện này được gọi là _posterior_, và đó là phân phối ta sử dụng để đưa ra dự đoán.

Cụ thể,

$$f(x) | f(x_1), \dots, f(x_n) \sim \mathcal{N}(m,s^2)$$ 

trong đó

$$m = k(x,x_{1:n}) k(x_{1:n},x_{1:n})^{-1} f(x_{1:n})$$ 

$$s^2 = k(x,x) - k(x,x_{1:n})k(x_{1:n},x_{1:n})^{-1}k(x,x_{1:n})$$ 

trong đó $k(x,x_{1:n})$ là một vector $1 \times n$ được tạo bằng cách tính $k(x,x_{i})$ với $i=1,\dots,n$, còn $k(x_{1:n},x_{1:n})$ là một ma trận $n \times n$ được tạo bằng cách tính $k(x_i,x_j)$ với $i,j = 1,\dots,n$. $m$ là giá trị ta có thể sử dụng làm dự đoán điểm cho mọi $x$, còn $s^2$ được sử dụng để biểu diễn độ bất định: nếu muốn tạo một khoảng có xác suất 95% rằng $f(x)$ nằm trong khoảng đó, ta sẽ sử dụng $m \pm 2s$. Các giá trị trung bình dự đoán và độ bất định trong tất cả các hình trên đều được tạo bằng các phương trình này. Các điểm dữ liệu quan sát được là
$f(x_1), \dots, f(x_n)$
và ta đã chọn một tập các điểm $x$ có độ phân giải cao để đưa ra dự đoán.

Giả sử ta quan sát một điểm dữ liệu duy nhất, $f(x_1)$, và muốn xác định giá trị của $f(x)$ tại một $x$ nào đó. Vì $f(x)$ được mô tả bởi một Gaussian process, ta biết rằng phân phối đồng thời của
$(f(x), f(x_1))$ 
là Gaussian:

$$
\begin{bmatrix}
f(x) \\ 
f(x_1) \\
\end{bmatrix}
\sim
\mathcal{N}\left(\mu, 
\begin{bmatrix}
k(x,x) & k(x, x_1) \\
k(x_1,x) & k(x_1,x_1)
\end{bmatrix}
\right)
$$

Biểu thức ngoài đường chéo $k(x,x_1) = k(x_1,x)$
cho ta biết các giá trị hàm sẽ tương quan với nhau như thế nào --- $f(x)$
sẽ được xác định từ $f(x_1)$ mạnh đến mức nào.
Ta đã thấy rằng nếu sử dụng một length-scale lớn, so với khoảng cách giữa $x$ và $x_1$,
$||x-x_1||$, thì các giá trị hàm sẽ tương quan rất mạnh. Ta có thể hình dung quá trình xác định $f(x)$ từ $f(x_1)$ cả trong không gian hàm lẫn trong phân phối đồng thời của $f(x_1), f(x)$. Trước tiên, hãy xét một $x$ sao cho $k(x,x_1) = 0.9$, và $k(x,x)=1$, nghĩa là giá trị của $f(x)$ tương quan ở mức vừa phải với giá trị của $f(x_1)$. Trong phân phối đồng thời, các đường đồng xác suất sẽ là những ellipse tương đối hẹp.

Giả sử ta quan sát được $f(x_1) = 1.2$.
Để điều kiện hóa theo giá trị này của $f(x_1)$,
ta có thể vẽ một đường nằm ngang tại $1.2$ trên đồ thị mật độ và thấy rằng giá trị của $f(x)$
phần lớn bị giới hạn trong $[0.64,1.52]$. Ta cũng đã vẽ đồ thị này trong không gian hàm, biểu diễn
điểm $f(x_1)$ đã quan sát bằng màu cam, và 1 độ lệch chuẩn của phân phối dự đoán Gaussian process cho $f(x)$
bằng màu xanh lam, xung quanh giá trị trung bình là $1.08$.

![Các đường đồng xác suất của mật độ Gaussian hai biến trên $f(x_1)$ và $f(x)$ với $k(x,x_1) = 0.9$.](https://user-images.githubusercontent.com/6753639/206867364-b4707db5-0c2e-4ae4-a412-8292bca4d08d.svg)
![Phân phối dự đoán Gaussian process trong không gian hàm tại $f(x)$, với $k(x,x_1) = 0.9$.](https://user-images.githubusercontent.com/6753639/206867367-3815720c-93c8-4b4b-80e7-296db1d3553b.svg)

Bây giờ, giả sử ta có một tương quan mạnh hơn, $k(x,x_1) = 0.95$.
Các ellipse lúc này hẹp hơn nữa, và giá trị của $f(x)$
được xác định bởi $f(x_1)$ mạnh hơn. Khi vẽ một đường nằm ngang tại $1.2$, ta thấy các đường đồng mức của $f(x)$
chủ yếu hỗ trợ các giá trị trong khoảng $[0.83, 1.45]$. Một lần nữa, ta cũng biểu diễn đồ thị trong không gian hàm, với một độ lệch
chuẩn xung quanh giá trị dự đoán trung bình là $1.14$.

![Các đường đồng xác suất của mật độ Gaussian hai biến trên $f(x_1)$ và $f(x)$ với $k(x,x_1) = 0.95$.](https://user-images.githubusercontent.com/6753639/206867797-20e42783-31de-4c50-8103-e9441ba6d0a9.svg)
![Phân phối dự đoán Gaussian process trong không gian hàm tại $f(x)$, với $k(x,x_1)$ = 0.95.](https://user-images.githubusercontent.com/6753639/206867800-d9fc7add-649d-492c-8848-cab07c8fb83e.svg)

Ta thấy predictor là giá trị trung bình posterior của Gaussian process gần với $1.2$ hơn, vì tương quan hiện mạnh hơn. Ta cũng thấy độ bất định của mình (các error bar) đã giảm đi phần nào. Mặc dù các giá trị hàm này có tương quan mạnh, độ bất định của ta vẫn khá lớn, vì ta chỉ mới quan sát một điểm dữ liệu duy nhất!

Quy trình này có thể cung cấp posterior trên $f(x)$ với mọi $x$, với bất kỳ số lượng điểm nào mà ta đã quan sát. Giả sử ta quan sát $f(x_1), f(x_2)$. Bây giờ, ta trực quan hóa posterior của $f(x)$ tại một $x=x'$ cụ thể trong không gian hàm. Phân phối chính xác của $f(x)$ được cho bởi các phương trình trên. $f(x)$ có phân phối Gaussian, với giá trị trung bình

$$m = k(x,x_{1:3}) k(x_{1:3},x_{1:3})^{-1} f(x_{1:3})$$

và phương sai

$$s^2 = k(x,x) - k(x,x_{1:3})k(x_{1:3},x_{1:3})^{-1}k(x,x_{1:3})$$

Trong notebook nhập môn này, ta đã xét các quan sát _không có nhiễu_. Như ta sẽ thấy, việc đưa nhiễu quan sát vào rất đơn giản. Nếu giả sử dữ liệu được tạo từ một hàm ẩn không có nhiễu $f(x)$ cộng với nhiễu Gaussian iid
$\epsilon(x) \sim \mathcal{N}(0,\sigma^2)$
có phương sai $\sigma^2$, thì hàm covariance của ta đơn giản trở thành
$k(x_i,x_j) \to k(x_i,x_j) + \delta_{ij}\sigma^2$,
trong đó $\delta_{ij} = 1$ nếu $i=j$ và bằng $0$ trong các trường hợp khác.

Ta đã bắt đầu hình thành trực giác về cách sử dụng Gaussian process để xác định prior và posterior trên các nghiệm, cũng như cách hàm kernel ảnh hưởng đến các thuộc tính của những nghiệm này. Trong các notebook tiếp theo, ta sẽ trình bày chính xác cách xác định một prior Gaussian process, giới thiệu và suy ra nhiều hàm kernel khác nhau, sau đó đi qua cơ chế tự động học các hyperparameter của kernel và hình thành posterior Gaussian process để đưa ra dự đoán. Mặc dù cần thời gian và thực hành để làm quen với các khái niệm như “phân phối trên các hàm”, cơ chế thực tế để tìm các phương trình dự đoán GP lại khá đơn giản --- giúp ta dễ dàng thực hành và hình thành hiểu biết trực quan về các khái niệm này.

## Tóm tắt

Trong machine learning điển hình, ta xác định một hàm với một số tham số tự do (chẳng hạn như một neural network và các trọng số của nó), rồi tập trung vào việc ước lượng các tham số đó, vốn có thể không dễ diễn giải. Với Gaussian process, thay vào đó ta trực tiếp suy luận về các phân phối trên các hàm, nhờ đó có thể suy luận về các thuộc tính cấp cao của nghiệm. Các thuộc tính này được kiểm soát bởi một hàm covariance (kernel), thường có một vài hyperparameter rất dễ diễn giải. Các hyperparameter này bao gồm _length-scale_, tham số kiểm soát tốc độ biến thiên (mức độ “uốn lượn”) của các hàm. Một hyperparameter khác là amplitude, tham số kiểm soát quy mô theo chiều dọc mà các hàm của ta biến thiên.

Việc biểu diễn nhiều hàm khác nhau có thể khớp với dữ liệu, rồi kết hợp tất cả chúng thành một phân phối dự đoán, là một đặc điểm riêng của các phương pháp Bayesian. Vì có nhiều biến thiên hơn giữa các nghiệm khả dĩ ở xa dữ liệu, độ bất định của ta theo trực giác sẽ tăng khi di chuyển ra xa dữ liệu.

Một Gaussian process biểu diễn một phân phối trên các hàm bằng cách xác định một phân phối chuẩn đa biến (Gaussian) trên mọi giá trị hàm có thể có. Ta có thể dễ dàng thao tác với các phân phối Gaussian để tìm phân phối của một giá trị hàm dựa trên các giá trị của bất kỳ tập hợp giá trị nào khác. Nói cách khác, nếu quan sát một tập điểm, ta có thể điều kiện hóa theo các điểm này và suy ra phân phối về giá trị của hàm tại bất kỳ đầu vào nào khác. Cách ta mô hình hóa tương quan giữa các điểm này được xác định bởi hàm covariance và chính là yếu tố quyết định các thuộc tính khái quát hóa của Gaussian process. Mặc dù cần thời gian để làm quen với Gaussian process, chúng dễ sử dụng, có nhiều ứng dụng, đồng thời giúp ta hiểu và phát triển các lớp mô hình khác, chẳng hạn như neural network.

## Bài tập

1. Sự khác nhau giữa độ bất định nhận thức (epistemic uncertainty) và độ bất định do quan sát là gì?
2. Ngoài tốc độ biến thiên và amplitude, ta có thể muốn xét những thuộc tính nào khác của các hàm, và đâu là những ví dụ trong thế giới thực về các hàm có những thuộc tính đó?
3. Hàm covariance RBF mà ta đã xét cho rằng covariance (và correlation) giữa các quan sát giảm theo khoảng cách của chúng trong không gian đầu vào (thời gian, vị trí không gian, v.v.). Đây có phải là một giả định hợp lý không? Tại sao có hoặc tại sao không?
4. Tổng của hai biến Gaussian có phải là Gaussian không? Tích của hai biến Gaussian có phải là Gaussian không? Nếu $(a,b)$ có phân phối Gaussian đồng thời, thì $a|b$ (a với điều kiện b) có phải là Gaussian không? $a$ có phải là Gaussian không?
5. Lặp lại bài tập trong đó ta quan sát một điểm dữ liệu tại $f(x_1) = 1.2$, nhưng bây giờ giả sử ta quan sát thêm $f(x_2) = 1.4$. Cho $k(x,x_1) = 0.9$ và $k(x,x_2) = 0.8$. Ta sẽ chắc chắn hơn hay kém chắc chắn hơn về giá trị của $f(x)$ so với khi chỉ quan sát $f(x_1)$? Giá trị trung bình và credible set 95\% cho giá trị $f(x)$ lúc này là gì?
6. Bạn có nghĩ việc tăng ước lượng về nhiễu quan sát sẽ làm tăng hay giảm ước lượng về length-scale của hàm ground truth không?
7. Khi di chuyển ra xa dữ liệu, giả sử độ bất định trong phân phối dự đoán của ta tăng đến một mức nào đó rồi dừng tăng. Tại sao điều đó có thể xảy ra?

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/12115)
:end_tab: