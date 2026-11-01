# Thống kê
:label:`sec_statistics`

Không còn nghi ngờ gì nữa, để trở thành một người thực hành deep learning hàng đầu, khả năng huấn luyện các mô hình hiện đại và có độ chính xác cao là điều vô cùng quan trọng. Tuy nhiên, thường khó xác định khi nào những cải thiện là đáng kể, hoặc chỉ là kết quả của các dao động ngẫu nhiên trong quá trình huấn luyện. Để có thể thảo luận về độ không chắc chắn trong các giá trị ước lượng, chúng ta cần tìm hiểu một số kiến thức về thống kê.


Tài liệu tham khảo sớm nhất về *thống kê* có thể được truy nguyên đến học giả người Ả Rập Al-Kindi vào thế kỷ thứ $9^{\textrm{th}}$, người đã mô tả chi tiết cách sử dụng thống kê và phân tích tần suất để giải mã các thông điệp được mã hóa. Sau 800 năm, thống kê hiện đại xuất hiện ở Đức vào những năm 1700, khi các nhà nghiên cứu tập trung vào việc thu thập và phân tích dữ liệu nhân khẩu học và kinh tế. Ngày nay, thống kê là một ngành khoa học liên quan đến việc thu thập, xử lý, phân tích, diễn giải và trực quan hóa dữ liệu. Hơn nữa, lý thuyết cốt lõi của thống kê đã được sử dụng rộng rãi trong các nghiên cứu thuộc giới học thuật, ngành công nghiệp và chính phủ.


Cụ thể hơn, thống kê có thể được chia thành *thống kê mô tả* và *suy luận thống kê*. Loại thứ nhất tập trung vào việc tóm tắt và minh họa các đặc điểm của một tập hợp dữ liệu quan sát, được gọi là một *mẫu*. Mẫu được lấy từ một *tổng thể*, biểu thị toàn bộ tập hợp các cá thể, vật phẩm hoặc sự kiện tương tự mà thí nghiệm của chúng ta quan tâm. Trái với thống kê mô tả, *suy luận thống kê* tiếp tục suy ra các đặc điểm của một tổng thể từ các *mẫu* đã cho, dựa trên giả định rằng phân phối của mẫu có thể tái hiện phân phối của tổng thể ở một mức độ nhất định.


Bạn có thể thắc mắc: “Sự khác biệt cốt yếu giữa machine learning và thống kê là gì?” Về cơ bản, thống kê tập trung vào bài toán suy luận. Loại bài toán này bao gồm việc mô hình hóa mối quan hệ giữa các biến, chẳng hạn như suy luận nhân quả, và kiểm định ý nghĩa thống kê của các tham số mô hình, chẳng hạn như kiểm thử A/B. Ngược lại, machine learning nhấn mạnh việc đưa ra các dự đoán chính xác mà không cần lập trình tường minh và hiểu chức năng của từng tham số.


Trong phần này, chúng ta sẽ giới thiệu ba loại phương pháp suy luận thống kê: đánh giá và so sánh các bộ ước lượng, thực hiện kiểm định giả thuyết và xây dựng khoảng tin cậy. Những phương pháp này có thể giúp chúng ta suy ra các đặc điểm của một tổng thể cho trước, tức là tham số thực $\theta$. Để đơn giản, chúng ta giả định rằng tham số thực $\theta$ của một tổng thể cho trước là một giá trị vô hướng. Việc mở rộng sang trường hợp $\theta$ là một vector hoặc tensor là khá trực tiếp, vì vậy chúng ta bỏ qua trường hợp này trong phần thảo luận.



## Đánh giá và So sánh các Bộ ước lượng

Trong thống kê, một *bộ ước lượng* là một hàm của các mẫu đã cho được dùng để ước lượng tham số thực $\theta$. Chúng ta sẽ viết $\hat{\theta}_n = \hat{f}(x_1, \ldots, x_n)$ cho giá trị ước lượng của $\theta$ sau khi quan sát các mẫu {$x_1, x_2, \ldots, x_n$}.

Trước đây, chúng ta đã thấy những ví dụ đơn giản về các bộ ước lượng trong phần :numref:`sec_maximum_likelihood`. Nếu bạn có một số mẫu từ một biến ngẫu nhiên Bernoulli, ước lượng hợp lý cực đại cho xác suất biến ngẫu nhiên nhận giá trị một có thể được tính bằng cách đếm số lượng số một đã quan sát rồi chia cho tổng số mẫu. Tương tự, một bài tập yêu cầu bạn chứng minh rằng ước lượng hợp lý cực đại của giá trị trung bình của một Gaussian khi biết một số mẫu được cho bởi giá trị trung bình của tất cả các mẫu. Những bộ ước lượng này gần như không bao giờ cho giá trị thực của tham số, nhưng lý tưởng nhất là với số lượng mẫu lớn, giá trị ước lượng sẽ gần với giá trị thực.

Ví dụ, dưới đây chúng ta biểu diễn mật độ thực của một biến ngẫu nhiên Gaussian có giá trị trung bình bằng không và phương sai bằng một, cùng với một tập hợp các mẫu từ Gaussian đó. Chúng ta xây dựng tọa độ $y$ sao cho mọi điểm đều hiển thị được và mối quan hệ với mật độ ban đầu trở nên rõ ràng hơn.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()

# Sample datapoints and create y coordinate
epsilon = 0.1
random.seed(8675309)
xs = np.random.normal(loc=0, scale=1, size=(300,))

ys = [np.sum(np.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))
             / np.sqrt(2*np.pi*epsilon**2)) / len(xs) for i in range(len(xs))]

# Compute true density
xd = np.arange(np.min(xs), np.max(xs), 0.01)
yd = np.exp(-xd**2/2) / np.sqrt(2 * np.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=np.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(np.mean(xs)):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  #define pi in torch

# Sample datapoints and create y coordinate
epsilon = 0.1
torch.manual_seed(8675309)
xs = torch.randn(size=(300,))

ys = torch.tensor(
    [torch.sum(torch.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))\
               / torch.sqrt(2*torch.pi*epsilon**2)) / len(xs)\
     for i in range(len(xs))])

# Compute true density
xd = torch.arange(torch.min(xs), torch.max(xs), 0.01)
yd = torch.exp(-xd**2/2) / torch.sqrt(2 * torch.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=torch.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(torch.mean(xs).item()):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

tf.pi = tf.acos(tf.zeros(1)) * 2  # define pi in TensorFlow

# Sample datapoints and create y coordinate
epsilon = 0.1
xs = tf.random.normal((300,))

ys = tf.constant(
    [(tf.reduce_sum(tf.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2)) \
               / tf.sqrt(2*tf.pi*epsilon**2)) / tf.cast(
        tf.size(xs), dtype=tf.float32)).numpy() \
     for i in range(tf.size(xs))])

# Compute true density
xd = tf.range(tf.reduce_min(xs), tf.reduce_max(xs), 0.01)
yd = tf.exp(-xd**2/2) / tf.sqrt(2 * tf.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=tf.reduce_mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(tf.reduce_mean(xs).numpy()):.2f}')
d2l.plt.show()
```

Có nhiều cách để tính một bộ ước lượng của tham số $\hat{\theta}_n$. Trong phần này, chúng ta giới thiệu ba phương pháp phổ biến để đánh giá và so sánh các bộ ước lượng: sai số bình phương trung bình, độ lệch chuẩn và độ chệch thống kê.

### Sai số Bình phương Trung bình

Có lẽ độ đo đơn giản nhất được dùng để đánh giá các bộ ước lượng là bộ ước lượng *sai số bình phương trung bình (MSE)* (hay hàm mất mát $l_2$), được định nghĩa như sau:

$$\textrm{MSE} (\hat{\theta}_n, \theta) = E[(\hat{\theta}_n - \theta)^2].$$
:eqlabel:`eq_mse_est`

Điều này cho phép chúng ta định lượng độ lệch bình phương trung bình so với giá trị thực. MSE luôn không âm. Nếu bạn đã đọc :numref:`sec_linear_regression`, bạn sẽ nhận ra đây là hàm mất mát hồi quy được sử dụng phổ biến nhất. Khi được dùng làm độ đo để đánh giá một bộ ước lượng, giá trị của nó càng gần không thì bộ ước lượng càng gần với tham số thực $\theta$.


### Độ chệch Thống kê

MSE cung cấp một độ đo tự nhiên, nhưng chúng ta có thể dễ dàng hình dung nhiều hiện tượng khác nhau có thể khiến nó lớn. Hai hiện tượng quan trọng về bản chất là dao động trong bộ ước lượng do tính ngẫu nhiên của tập dữ liệu và sai số có tính hệ thống trong bộ ước lượng do quy trình ước lượng.


Trước tiên, hãy đo sai số có tính hệ thống. Với một bộ ước lượng $\hat{\theta}_n$, biểu diễn toán học của *độ chệch thống kê* có thể được định nghĩa như sau:

$$\textrm{bias}(\hat{\theta}_n) = E(\hat{\theta}_n - \theta) = E(\hat{\theta}_n) - \theta.$$
:eqlabel:`eq_bias`

Lưu ý rằng khi $\textrm{bias}(\hat{\theta}_n) = 0$, kỳ vọng của bộ ước lượng $\hat{\theta}_n$ bằng với giá trị thực của tham số. Trong trường hợp này, chúng ta nói $\hat{\theta}_n$ là một bộ ước lượng không chệch. Nhìn chung, một bộ ước lượng không chệch tốt hơn một bộ ước lượng chệch vì kỳ vọng của nó bằng với tham số thực.


Tuy nhiên, cần lưu ý rằng các bộ ước lượng chệch thường xuyên được sử dụng trong thực tế. Có những trường hợp không tồn tại bộ ước lượng không chệch nếu không có thêm giả định, hoặc việc tính toán chúng là bất khả thi. Điều này có vẻ là một khiếm khuyết đáng kể của một bộ ước lượng, nhưng phần lớn các bộ ước lượng gặp trong thực tế ít nhất đều không chệch tiệm cận, theo nghĩa là độ chệch tiến về không khi số lượng mẫu khả dụng tiến đến vô hạn: $\lim_{n \rightarrow \infty} \textrm{bias}(\hat{\theta}_n) = 0$.


### Phương sai và Độ lệch chuẩn

Tiếp theo, hãy đo tính ngẫu nhiên trong bộ ước lượng. Nhắc lại từ :numref:`sec_random_variables`, *độ lệch chuẩn* (hay *sai số chuẩn*) được định nghĩa là căn bậc hai của phương sai. Chúng ta có thể đo mức độ dao động của một bộ ước lượng bằng cách đo độ lệch chuẩn hoặc phương sai của bộ ước lượng đó.

$$\sigma_{\hat{\theta}_n} = \sqrt{\textrm{Var} (\hat{\theta}_n )} = \sqrt{E[(\hat{\theta}_n - E(\hat{\theta}_n))^2]}.$$
:eqlabel:`eq_var_est`

Điều quan trọng là phải so sánh :eqref:`eq_var_est` với :eqref:`eq_mse_est`. Trong phương trình này, chúng ta không so sánh với giá trị $\theta$ thực của tổng thể, mà thay vào đó so sánh với $E(\hat{\theta}_n)$, giá trị trung bình kỳ vọng của mẫu. Vì vậy, chúng ta không đo khoảng cách mà bộ ước lượng có xu hướng lệch khỏi giá trị thực, mà đo sự dao động của chính bộ ước lượng đó.


### Đánh đổi giữa Độ chệch và Phương sai

Trực giác cho thấy hai thành phần chính này góp phần tạo nên sai số bình phương trung bình. Điều khá đáng ngạc nhiên là chúng ta có thể chứng minh rằng đây thực sự là một *phân rã* của sai số bình phương trung bình thành hai phần đóng góp này cùng với một phần thứ ba. Nói cách khác, chúng ta có thể viết sai số bình phương trung bình dưới dạng tổng của bình phương độ chệch, phương sai và sai số không thể giảm.

$$
\begin{aligned}
\textrm{MSE} (\hat{\theta}_n, \theta) &= E[(\hat{\theta}_n - \theta)^2] \\
 &= E[(\hat{\theta}_n)^2] + E[\theta^2] - 2E[\hat{\theta}_n\theta] \\
 &= \textrm{Var} [\hat{\theta}_n] + E[\hat{\theta}_n]^2 + \textrm{Var} [\theta] + E[\theta]^2 - 2E[\hat{\theta}_n]E[\theta] \\
 &= (E[\hat{\theta}_n] - E[\theta])^2 + \textrm{Var} [\hat{\theta}_n] + \textrm{Var} [\theta] \\
 &= (E[\hat{\theta}_n - \theta])^2 + \textrm{Var} [\hat{\theta}_n] + \textrm{Var} [\theta] \\
 &= (\textrm{bias} [\hat{\theta}_n])^2 + \textrm{Var} (\hat{\theta}_n) + \textrm{Var} [\theta].\\
\end{aligned}
$$

Công thức trên được gọi là *đánh đổi giữa độ chệch và phương sai*. Sai số bình phương trung bình có thể được chia thành ba nguồn sai số: sai số do độ chệch cao, sai số do phương sai cao và sai số không thể giảm. Sai số do độ chệch thường xuất hiện trong một mô hình đơn giản (chẳng hạn như mô hình hồi quy tuyến tính), mô hình không thể trích xuất các mối quan hệ nhiều chiều giữa các đặc trưng và đầu ra. Nếu một mô hình chịu sai số do độ chệch cao, chúng ta thường nói mô hình đó bị *underfitting* hoặc thiếu *tính linh hoạt*, như đã giới thiệu trong (:numref:`sec_generalization_basics`). Phương sai cao thường là kết quả của một mô hình quá phức tạp, khiến mô hình overfit dữ liệu huấn luyện. Do đó, một mô hình *overfitting* nhạy cảm với những dao động nhỏ trong dữ liệu. Nếu một mô hình chịu phương sai cao, chúng ta thường nói mô hình đó bị *overfitting* và thiếu *khả năng tổng quát hóa*, như đã giới thiệu trong (:numref:`sec_generalization_basics`). Sai số không thể giảm là kết quả của nhiễu trong chính $\theta$.


### Đánh giá các Bộ ước lượng trong Code

Vì độ lệch chuẩn của một estimator đã được triển khai bằng cách פשוט gọi `a.std()` cho một tensor `a`, chúng ta sẽ bỏ qua phần này nhưng triển khai độ chệch thống kê và sai số bình phương trung bình.

```{.python .input}
#@tab mxnet
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(np.mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(np.mean(np.square(data - true_theta)))
```

```{.python .input}
#@tab pytorch
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(torch.mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(torch.mean(torch.square(data - true_theta)))
```

```{.python .input}
#@tab tensorflow
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(tf.reduce_mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(tf.reduce_mean(tf.square(data - true_theta)))
```

Để minh họa phương trình của sự đánh đổi giữa độ chệch và phương sai, hãy mô phỏng một phân phối chuẩn $\mathcal{N}(\theta, \sigma^2)$ với $10,000$ mẫu. Ở đây, chúng ta sử dụng $\theta = 1$ và $\sigma = 4$. Vì estimator là một hàm của các mẫu đã cho, ở đây chúng ta sử dụng giá trị trung bình của các mẫu làm estimator cho $\theta$ thực trong phân phối chuẩn này $\mathcal{N}(\theta, \sigma^2)$ .

```{.python .input}
#@tab mxnet
theta_true = 1
sigma = 4
sample_len = 10000
samples = np.random.normal(theta_true, sigma, sample_len)
theta_est = np.mean(samples)
theta_est
```

```{.python .input}
#@tab pytorch
theta_true = 1
sigma = 4
sample_len = 10000
samples = torch.normal(theta_true, sigma, size=(sample_len, 1))
theta_est = torch.mean(samples)
theta_est
```

```{.python .input}
#@tab tensorflow
theta_true = 1
sigma = 4
sample_len = 10000
samples = tf.random.normal((sample_len, 1), theta_true, sigma)
theta_est = tf.reduce_mean(samples)
theta_est
```

Hãy kiểm chứng phương trình đánh đổi bằng cách tính tổng của bình phương độ chệch và phương sai của estimator. Trước tiên, hãy tính MSE của estimator.

```{.python .input}
#@tab all
mse(samples, theta_true)
```

Tiếp theo, chúng ta tính $\textrm{Var} (\hat{\theta}_n) + [\textrm{bias} (\hat{\theta}_n)]^2$ như dưới đây. Như bạn có thể thấy, hai giá trị khớp nhau với độ chính xác số học.

```{.python .input}
#@tab mxnet
bias = stat_bias(theta_true, theta_est)
np.square(samples.std()) + np.square(bias)
```

```{.python .input}
#@tab pytorch
bias = stat_bias(theta_true, theta_est)
torch.square(samples.std(unbiased=False)) + torch.square(bias)
```

```{.python .input}
#@tab tensorflow
bias = stat_bias(theta_true, theta_est)
tf.square(tf.math.reduce_std(samples)) + tf.square(bias)
```

## Tiến hành kiểm định giả thuyết


Chủ đề thường gặp nhất trong suy luận thống kê là kiểm định giả thuyết. Mặc dù kiểm định giả thuyết trở nên phổ biến vào đầu thế kỷ $20^{th}$, việc sử dụng đầu tiên có thể được truy nguyên về John Arbuthnot vào những năm 1700. John đã theo dõi hồ sơ sinh trong 80 năm ở London và kết luận rằng mỗi năm có nhiều bé trai được sinh ra hơn bé gái. Sau đó, kiểm định ý nghĩa hiện đại kế thừa những đóng góp quan trọng của Karl Pearson, người phát minh ra $p$-value và kiểm định chi-squared của Pearson, William Gosset, cha đẻ của phân phối t của Student, và Ronald Fisher, người khởi xướng giả thuyết không và kiểm định ý nghĩa.

*Một kiểm định giả thuyết* là một cách đánh giá bằng chứng chống lại phát biểu mặc định về một tổng thể. Chúng ta gọi phát biểu mặc định này là *giả thuyết không* $H_0$, và cố gắng bác bỏ nó bằng dữ liệu quan sát được. Ở đây, chúng ta sử dụng $H_0$ làm điểm khởi đầu cho việc kiểm định ý nghĩa thống kê. *Giả thuyết đối* $H_A$ (hoặc $H_1$) là một phát biểu trái ngược với giả thuyết không. Giả thuyết không thường được phát biểu dưới dạng khẳng định, trong đó nêu ra mối quan hệ giữa các biến. Giả thuyết này cần phản ánh vấn đề một cách rõ ràng nhất có thể và có thể được kiểm định bằng lý thuyết thống kê.

Hãy tưởng tượng bạn là một nhà hóa học. Sau hàng nghìn giờ làm việc trong phòng thí nghiệm, bạn phát triển một loại thuốc mới có thể cải thiện đáng kể khả năng hiểu toán của một người. Để chứng minh tác dụng kỳ diệu của nó, bạn cần kiểm định nó. Đương nhiên, bạn có thể cần một số tình nguyện viên dùng thuốc và xem liệu thuốc có giúp họ học toán tốt hơn hay không. Bạn sẽ bắt đầu như thế nào?

Trước tiên, bạn cần cẩn thận chọn ngẫu nhiên hai nhóm tình nguyện viên, sao cho không có sự khác biệt giữa khả năng hiểu toán của họ khi được đo bằng một số chỉ số. Hai nhóm này thường được gọi là nhóm thử nghiệm và nhóm đối chứng. *Nhóm thử nghiệm* (hoặc *nhóm điều trị*) là nhóm những người sẽ sử dụng thuốc, trong khi *nhóm đối chứng* đại diện cho nhóm người được giữ lại làm chuẩn đối chiếu, tức là có môi trường được thiết lập giống hệt ngoại trừ việc không dùng thuốc này. Bằng cách đó, ảnh hưởng của tất cả các biến được giảm thiểu, ngoại trừ tác động của biến độc lập trong quá trình điều trị.

Thứ hai, sau một thời gian dùng thuốc, bạn cần đo khả năng hiểu toán của hai nhóm bằng cùng một chỉ số, chẳng hạn yêu cầu các tình nguyện viên làm cùng một bài kiểm tra sau khi học một công thức toán mới. Sau đó, bạn có thể thu thập kết quả của họ và so sánh. Trong trường hợp này, giả thuyết không của chúng ta sẽ là không có sự khác biệt giữa hai nhóm, còn giả thuyết đối sẽ là có sự khác biệt.

Điều này vẫn chưa hoàn toàn mang tính hình thức. Có nhiều chi tiết bạn phải cân nhắc cẩn thận. Ví dụ, chỉ số nào phù hợp để kiểm tra khả năng hiểu toán của họ? Bạn cần bao nhiêu tình nguyện viên cho thử nghiệm để có thể tự tin khẳng định thuốc của mình có hiệu quả? Bạn nên tiến hành thử nghiệm trong bao lâu? Làm thế nào để quyết định liệu có sự khác biệt giữa hai nhóm? Bạn chỉ quan tâm đến hiệu suất trung bình, hay cả phạm vi biến thiên của điểm số? Và còn nhiều câu hỏi khác.

Theo cách này, kiểm định giả thuyết cung cấp một khuôn khổ cho việc thiết kế thí nghiệm và suy luận về mức độ chắc chắn của các kết quả quan sát được. Nếu giờ đây chúng ta có thể cho thấy giả thuyết không rất khó có khả năng đúng, chúng ta có thể tự tin bác bỏ nó.

Để hoàn thiện câu chuyện về cách làm việc với kiểm định giả thuyết, giờ đây chúng ta cần giới thiệu một số thuật ngữ bổ sung và hình thức hóa một số khái niệm đã nêu ở trên.


### Ý nghĩa thống kê

*Ý nghĩa thống kê* đo xác suất bác bỏ nhầm giả thuyết không, $H_0$, trong khi lẽ ra không nên bác bỏ, tức là

$$ \textrm{statistical significance }= 1 - \alpha = 1 - P(\textrm{reject } H_0 \mid H_0 \textrm{ is true} ).$$

Nó còn được gọi là *sai lầm loại I* hoặc *dương tính giả*. $\alpha$ được gọi là *mức ý nghĩa* và giá trị thường được sử dụng của nó là $5\%$, tức là $1-\alpha = 95\%$. Mức ý nghĩa có thể được hiểu là mức rủi ro mà chúng ta sẵn sàng chấp nhận khi bác bỏ một giả thuyết không đúng.

:numref:`fig_statistical_significance` cho thấy các giá trị quan sát và xác suất của một phân phối chuẩn cho trước trong kiểm định giả thuyết hai mẫu. Nếu dữ liệu quan sát của ví dụ nằm ngoài ngưỡng $95\%$, thì đó sẽ là một quan sát rất khó xảy ra theo giả định giả thuyết không. Do đó, có thể giả thuyết không có vấn đề và chúng ta sẽ bác bỏ nó.

![Ý nghĩa thống kê.](../img/statistical-significance.svg)
:label:`fig_statistical_significance`


### Độ mạnh thống kê

*Độ mạnh thống kê* (hoặc *độ nhạy*) đo xác suất bác bỏ giả thuyết không, $H_0$, khi lẽ ra nó phải bị bác bỏ, tức là

$$ \textrm{statistical power }= 1 - \beta = 1 - P(\textrm{ fail to reject } H_0  \mid H_0 \textrm{ is false} ).$$

Hãy nhớ rằng *sai lầm loại I* là sai lầm do bác bỏ giả thuyết không khi nó đúng, trong khi *sai lầm loại II* phát sinh do không bác bỏ giả thuyết không khi nó sai. Sai lầm loại II thường được ký hiệu là $\beta$, do đó độ mạnh thống kê tương ứng là $1-\beta$.


Theo trực giác, độ mạnh thống kê có thể được hiểu là khả năng kiểm định phát hiện một khác biệt thực tế có độ lớn tối thiểu nào đó ở mức ý nghĩa thống kê mong muốn. $80\%$ là ngưỡng độ mạnh thống kê thường được sử dụng. Độ mạnh thống kê càng cao thì khả năng phát hiện những khác biệt thực sự càng lớn.

Một trong những ứng dụng phổ biến nhất của độ mạnh thống kê là xác định số lượng mẫu cần thiết. Xác suất bác bỏ giả thuyết không khi nó sai phụ thuộc vào mức độ sai của nó (được gọi là *effect size*) và số lượng mẫu bạn có. Như bạn có thể dự đoán, effect size nhỏ sẽ đòi hỏi số lượng mẫu rất lớn để có thể phát hiện với xác suất cao. Mặc dù việc suy ra chi tiết nằm ngoài phạm vi của phụ lục ngắn này, hãy xét ví dụ sau: ta muốn có khả năng bác bỏ giả thuyết không rằng mẫu của mình đến từ một Gaussian có trung bình bằng không và phương sai bằng một, đồng thời tin rằng trung bình của mẫu thực sự gần bằng một; khi đó, ta có thể đạt được tỷ lệ sai sót chấp nhận được chỉ với cỡ mẫu là $8$. Tuy nhiên, nếu cho rằng trung bình thực của tổng thể mẫu gần bằng $0.01$, thì chúng ta sẽ cần cỡ mẫu gần $80000$ để phát hiện sự khác biệt.

Chúng ta có thể hình dung độ mạnh như một bộ lọc nước. Trong phép tương tự này, một kiểm định giả thuyết có độ mạnh cao giống như một hệ thống lọc nước chất lượng cao, giúp giảm các chất có hại trong nước nhiều nhất có thể. Mặt khác, một khác biệt nhỏ giống như một bộ lọc nước chất lượng thấp, nơi một số chất tương đối nhỏ có thể dễ dàng thoát qua các khe hở. Tương tự, nếu độ mạnh thống kê không đủ cao, kiểm định có thể không phát hiện được khác biệt nhỏ hơn.


### Thống kê kiểm định

*Một thống kê kiểm định* $T(x)$ là một đại lượng vô hướng tóm tắt một đặc trưng nào đó của dữ liệu mẫu. Mục tiêu của việc định nghĩa thống kê như vậy là để nó cho phép chúng ta phân biệt giữa các phân phối khác nhau và tiến hành kiểm định giả thuyết. Quay lại ví dụ về nhà hóa học, nếu muốn chứng minh một tổng thể có hiệu suất tốt hơn tổng thể kia, việc chọn giá trị trung bình làm thống kê kiểm định có thể là hợp lý. Những lựa chọn khác nhau về thống kê kiểm định có thể dẫn đến độ mạnh thống kê của kiểm định khác nhau đáng kể.

Thông thường, $T(X)$ (phân phối của thống kê kiểm định dưới giả thuyết không của chúng ta) sẽ tuân theo, ít nhất là xấp xỉ, một phân phối xác suất phổ biến, chẳng hạn như phân phối chuẩn, khi được xét dưới giả thuyết không. Nếu có thể suy ra một cách tường minh phân phối như vậy và sau đó đo thống kê kiểm định trên tập dữ liệu, chúng ta có thể an toàn bác bỏ giả thuyết không nếu thống kê nằm rất xa ngoài phạm vi mà ta kỳ vọng. Việc định lượng điều này dẫn chúng ta đến khái niệm $p$-value.


### $p$-value

*$p$-value* (hoặc *giá trị xác suất*) là xác suất để $T(X)$ cực đoan ít nhất ở mức bằng thống kê kiểm định quan sát được $T(x)$, với giả định rằng giả thuyết không là *đúng*, tức là

$$ p\textrm{-value} = P_{H_0}(T(X) \geq T(x)).$$

Nếu $p$-value nhỏ hơn hoặc bằng mức ý nghĩa thống kê được xác định trước và cố định $\alpha$, ta có thể bác bỏ giả thuyết không. Nếu không, ta sẽ kết luận rằng chưa có đủ bằng chứng để bác bỏ giả thuyết không. Với một phân phối tổng thể cho trước, *vùng bác bỏ* sẽ là khoảng chứa tất cả các điểm có $p$-value nhỏ hơn mức ý nghĩa thống kê $\alpha$.


### Kiểm định một phía và kiểm định hai phía

Thông thường có hai loại kiểm định ý nghĩa: kiểm định một phía và kiểm định hai phía. *Kiểm định một phía* (hay *kiểm định một đuôi*) được áp dụng khi giả thuyết không và giả thuyết đối chỉ có một hướng. Ví dụ, giả thuyết không có thể phát biểu rằng tham số thực $\theta$ nhỏ hơn hoặc bằng một giá trị $c$. Giả thuyết đối sẽ là $\theta$ lớn hơn $c$. Nghĩa là vùng bác bỏ chỉ nằm ở một phía của phân phối lấy mẫu. Ngược lại với kiểm định một phía, *kiểm định hai phía* (hay *kiểm định hai đuôi*) được áp dụng khi vùng bác bỏ nằm ở cả hai phía của phân phối lấy mẫu. Một ví dụ trong trường hợp này là giả thuyết không phát biểu rằng tham số thực $\theta$ bằng một giá trị $c$. Giả thuyết đối sẽ là $\theta$ khác $c$.


### Các bước chung của kiểm định giả thuyết

Sau khi làm quen với các khái niệm trên, hãy cùng xem qua các bước chung của kiểm định giả thuyết.

1. Phát biểu câu hỏi và thiết lập giả thuyết không $H_0$.
2. Thiết lập mức ý nghĩa thống kê $\alpha$ và lực thống kê ($1 - \beta$).
3. Thu thập mẫu thông qua các thí nghiệm. Số lượng mẫu cần thiết sẽ phụ thuộc vào lực thống kê và kích thước hiệu ứng kỳ vọng.
4. Tính thống kê kiểm định và $p$-value.
5. Quyết định giữ lại hay bác bỏ giả thuyết không dựa trên $p$-value và mức ý nghĩa thống kê $\alpha$.

Để thực hiện kiểm định giả thuyết, trước tiên ta xác định một giả thuyết không và một mức độ rủi ro mà ta sẵn sàng chấp nhận. Sau đó, ta tính thống kê kiểm định của mẫu, trong đó một giá trị cực đoan của thống kê kiểm định được xem là bằng chứng chống lại giả thuyết không. Nếu thống kê kiểm định nằm trong vùng bác bỏ, ta có thể bác bỏ giả thuyết không để ủng hộ giả thuyết đối.

Kiểm định giả thuyết được áp dụng trong nhiều tình huống khác nhau, chẳng hạn như các thử nghiệm lâm sàng và thử nghiệm A/B.


## Xây dựng khoảng tin cậy


Khi ước lượng giá trị của một tham số $\theta$, các ước lượng điểm như $\hat \theta$ có tính hữu dụng hạn chế vì chúng không thể hiện khái niệm nào về độ bất định. Thay vào đó, sẽ tốt hơn nhiều nếu ta có thể tạo ra một khoảng chứa tham số thực $\theta$ với xác suất cao. Nếu bạn quan tâm đến những ý tưởng như vậy từ một thế kỷ trước, hẳn bạn sẽ rất hào hứng khi đọc tác phẩm "Outline of a Theory of Statistical Estimation Based on the Classical Theory of Probability" của Jerzy Neyman :cite:`Neyman.1937`, người đầu tiên giới thiệu khái niệm khoảng tin cậy vào năm 1937.

Để hữu ích, khoảng tin cậy nên nhỏ nhất có thể với một mức độ chắc chắn nhất định. Hãy cùng xem cách suy ra khoảng này.


### Định nghĩa

Về mặt toán học, một *khoảng tin cậy* cho tham số thực $\theta$ là một khoảng $C_n$ được tính từ dữ liệu mẫu sao cho

$$P_{\theta} (C_n \ni \theta) \geq 1 - \alpha, \forall \theta.$$
:eqlabel:`eq_confidence`

Ở đây $\alpha \in (0, 1)$, và $1 - \alpha$ được gọi là *mức tin cậy* hoặc *độ bao phủ* của khoảng. Đây chính là $\alpha$ giống với mức ý nghĩa mà chúng ta đã thảo luận ở trên.

Lưu ý rằng :eqref:`eq_confidence` nói về biến $C_n$, chứ không nói về $\theta$ cố định. Để nhấn mạnh điều này, ta viết $P_{\theta} (C_n \ni \theta)$ thay vì $P_{\theta} (\theta \in C_n)$.

### Diễn giải

Rất dễ diễn giải khoảng tin cậy $95\%$ là một khoảng mà bạn có thể chắc chắn $95\%$ rằng tham số thực nằm trong đó, tuy nhiên điều này đáng tiếc là không đúng. Tham số thực là cố định, còn khoảng mới là đại lượng ngẫu nhiên. Vì vậy, cách diễn giải tốt hơn là: nếu bạn tạo ra một số lượng lớn khoảng tin cậy bằng quy trình này, thì $95\%$ số khoảng được tạo ra sẽ chứa tham số thực.

Điều này có vẻ quá chú trọng tiểu tiết, nhưng nó có thể dẫn đến những hệ quả thực tế trong việc diễn giải kết quả. Cụ thể, ta vẫn có thể thỏa mãn :eqref:`eq_confidence` bằng cách xây dựng các khoảng mà ta *gần như chắc chắn* không chứa giá trị thực, miễn là ta chỉ làm như vậy đủ hiếm. Ta kết thúc phần này bằng ba phát biểu hấp dẫn nhưng sai. Có thể tìm thấy phần thảo luận chuyên sâu về những điểm này trong :citet:`Morey.Hoekstra.Rouder.ea.2016`.

* **Ngụy biện 1**. Các khoảng tin cậy hẹp có nghĩa là ta có thể ước lượng tham số một cách chính xác.
* **Ngụy biện 2**. Các giá trị nằm trong khoảng tin cậy có khả năng là giá trị thực cao hơn các giá trị nằm ngoài khoảng.
* **Ngụy biện 3**. Xác suất một khoảng tin cậy $95\%$ cụ thể đã quan sát chứa giá trị thực là $95\%$.

Có thể nói rằng khoảng tin cậy là những đối tượng tinh tế. Tuy nhiên, nếu bạn hiểu rõ cách diễn giải, chúng có thể là những công cụ mạnh mẽ.

### Ví dụ Gaussian

Hãy cùng thảo luận về ví dụ kinh điển nhất: khoảng tin cậy cho trung bình của một phân phối Gaussian có trung bình và phương sai chưa biết. Giả sử ta thu thập $n$ mẫu $\{x_i\}_{i=1}^n$ từ Gaussian $\mathcal{N}(\mu, \sigma^2)$. Ta có thể tính các ước lượng cho trung bình và phương sai bằng cách lấy

$$\hat\mu_n = \frac{1}{n}\sum_{i=1}^n x_i \;\textrm{and}\; \hat\sigma^2_n = \frac{1}{n-1}\sum_{i=1}^n (x_i - \hat\mu)^2.$$

Nếu bây giờ ta xét biến ngẫu nhiên

$$
T = \frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}},
$$

ta thu được một biến ngẫu nhiên tuân theo một phân phối nổi tiếng có tên là *phân phối t-Student với* $n-1$ *bậc tự do*.

Phân phối này đã được nghiên cứu rất kỹ, và người ta biết rằng, chẳng hạn, khi $n\rightarrow \infty$, nó xấp xỉ một Gaussian chuẩn. Do đó, bằng cách tra các giá trị của hàm phân phối tích lũy Gaussian trong một bảng, ta có thể kết luận rằng giá trị của $T$ nằm trong khoảng $[-1.96, 1.96]$ ít nhất $95\%$ số lần. Với các giá trị hữu hạn của $n$, khoảng này cần lớn hơn một chút, nhưng các giá trị đó đã được biết rõ và tính sẵn trong các bảng.

Vì vậy, ta có thể kết luận rằng với $n$ lớn,

$$
P\left(\frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}} \in [-1.96, 1.96]\right) \ge 0.95.
$$

Biến đổi biểu thức này bằng cách nhân cả hai vế với $\hat\sigma_n/\sqrt{n}$ rồi cộng thêm $\hat\mu_n$, ta thu được

$$
P\left(\mu \in \left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right]\right) \ge 0.95.
$$

Như vậy, ta biết rằng mình đã tìm được khoảng tin cậy $95\%$:
$$\left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right].$$
:eqlabel:`eq_gauss_confidence`

Có thể nói rằng :eqref:`eq_gauss_confidence` là một trong những công thức được sử dụng nhiều nhất trong thống kê. Hãy kết thúc phần thảo luận về thống kê bằng cách triển khai công thức này. Để đơn giản, ta giả sử mình đang ở chế độ tiệm cận. Các giá trị nhỏ của $N$ nên sử dụng giá trị đúng của `t_star` được lấy bằng lập trình hoặc từ bảng $t$.

```{.python .input}
#@tab mxnet
# Number of samples
N = 1000

# Sample dataset
samples = np.random.normal(loc=0, scale=1, size=(N,))

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = np.mean(samples)
sigma_hat = samples.std(ddof=1)
(mu_hat - t_star*sigma_hat/np.sqrt(N), mu_hat + t_star*sigma_hat/np.sqrt(N))
```

```{.python .input}
#@tab pytorch
# PyTorch uses Bessel's correction by default, which means the use of ddof=1
# instead of default ddof=0 in numpy. We can use unbiased=False to imitate
# ddof=0.

# Number of samples
N = 1000

# Sample dataset
samples = torch.normal(0, 1, size=(N,))

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = torch.mean(samples)
sigma_hat = samples.std(unbiased=True)
(mu_hat - t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)),\
 mu_hat + t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)))
```

```{.python .input}
#@tab tensorflow
# Number of samples
N = 1000

# Sample dataset
samples = tf.random.normal((N,), 0, 1)

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = tf.reduce_mean(samples)
sigma_hat = tf.math.reduce_std(samples)
(mu_hat - t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)), \
 mu_hat + t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)))
```

## Tóm tắt

* Thống kê tập trung vào các bài toán suy luận, trong khi deep learning nhấn mạnh việc đưa ra các dự đoán chính xác mà không cần lập trình và tìm hiểu một cách tường minh.
* Có ba phương pháp suy luận thống kê phổ biến: đánh giá và so sánh các ước lượng, thực hiện kiểm định giả thuyết và xây dựng khoảng tin cậy.
* Có ba ước lượng phổ biến nhất: độ chệch thống kê, độ lệch chuẩn và sai số bình phương trung bình.
* Khoảng tin cậy là một khoảng ước lượng của tham số thực của tổng thể mà ta có thể xây dựng dựa trên các mẫu đã cho.
* Kiểm định giả thuyết là một cách đánh giá bằng chứng chống lại phát biểu mặc định về một tổng thể.


## Bài tập

1. Cho $X_1, X_2, \ldots, X_n \overset{\textrm{iid}}{\sim} \textrm{Unif}(0, \theta)$, trong đó "iid" là viết tắt của *độc lập và cùng phân phối*. Xét các ước lượng sau của $\theta$:
$$\hat{\theta} = \max \{X_1, X_2, \ldots, X_n \};$$
$$\tilde{\theta} = 2 \bar{X_n} = \frac{2}{n} \sum_{i=1}^n X_i.$$
    * Tìm độ chệch thống kê, độ lệch chuẩn và sai số bình phương trung bình của $\hat{\theta}.$
    * Tìm độ chệch thống kê, độ lệch chuẩn và sai số bình phương trung bình của $\tilde{\theta}.$
    * Ước lượng nào tốt hơn?
1. Với ví dụ về nhà hóa học trong phần giới thiệu, bạn có thể suy ra 5 bước để thực hiện kiểm định giả thuyết hai phía không? Cho mức ý nghĩa thống kê $\alpha = 0.05$ và lực thống kê $1 - \beta = 0.8$.
1. Chạy mã khoảng tin cậy với $N=2$ và $\alpha = 0.5$ cho $100$ tập dữ liệu được tạo độc lập, rồi vẽ các khoảng thu được (trong trường hợp này `t_star = 1.0`). Bạn sẽ thấy một số khoảng rất ngắn nhưng cách rất xa so với việc chứa trung bình thực $0$. Điều này có mâu thuẫn với cách diễn giải khoảng tin cậy không? Bạn có cảm thấy thoải mái khi sử dụng các khoảng ngắn để biểu thị những ước lượng có độ chính xác cao không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/419)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1102)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1103)
:end_tab: