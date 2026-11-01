# Naive Bayes
:label:`sec_naive_bayes`

Trong các phần trước, chúng ta đã tìm hiểu về lý thuyết xác suất và các biến ngẫu nhiên. Để áp dụng lý thuyết này vào thực tế, hãy giới thiệu bộ phân loại *naive Bayes*. Bộ phân loại này chỉ sử dụng những nền tảng xác suất cơ bản để cho phép chúng ta thực hiện việc phân loại chữ số.

Học máy luôn liên quan đến việc đưa ra các giả định. Nếu muốn phân loại một ví dụ dữ liệu mới mà chúng ta chưa từng thấy trước đây, chúng ta phải đưa ra một số giả định về những ví dụ dữ liệu nào tương tự nhau. Bộ phân loại naive Bayes, một thuật toán phổ biến và đặc biệt dễ hiểu, giả định rằng tất cả các đặc trưng độc lập với nhau để đơn giản hóa việc tính toán. Trong phần này, chúng ta sẽ áp dụng mô hình này để nhận dạng các ký tự trong hình ảnh.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import gluon, np, npx
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import torchvision
d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
d2l.use_svg_display()
```

## Nhận dạng ký tự quang học

MNIST :cite:`LeCun.Bottou.Bengio.ea.1998` là một trong những bộ dữ liệu được sử dụng rộng rãi. Bộ dữ liệu này chứa 60.000 hình ảnh dùng để huấn luyện và 10.000 hình ảnh dùng để kiểm định. Mỗi hình ảnh chứa một chữ số viết tay từ 0 đến 9. Nhiệm vụ là phân loại mỗi hình ảnh vào chữ số tương ứng.

Gluon cung cấp một lớp `MNIST` trong mô-đun `data.vision` để
tự động lấy bộ dữ liệu từ Internet.
Sau đó, Gluon sẽ sử dụng bản sao cục bộ đã được tải xuống.
Chúng ta xác định đang yêu cầu tập huấn luyện hay tập kiểm tra
bằng cách đặt giá trị của tham số `train` thành `True` hoặc `False`, tương ứng.
Mỗi hình ảnh là một ảnh grayscale có cả chiều rộng và chiều cao bằng $28$, với shape ($28$,$28$,$1$). Chúng ta sử dụng một phép biến đổi tùy chỉnh để loại bỏ chiều kênh cuối cùng. Ngoài ra, bộ dữ liệu biểu diễn mỗi pixel bằng một số nguyên unsigned 8-bit. Chúng ta lượng tử hóa chúng thành các đặc trưng nhị phân để đơn giản hóa bài toán.

```{.python .input}
#@tab mxnet
def transform(data, label):
    return np.floor(data.astype('float32') / 128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
```

```{.python .input}
#@tab pytorch
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    lambda x: torch.floor(x * 255 / 128).squeeze(dim=0)
])

mnist_train = torchvision.datasets.MNIST(
    root='./temp', train=True, transform=data_transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./temp', train=False, transform=data_transform, download=True)
```

```{.python .input}
#@tab tensorflow
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.mnist.load_data()

# Original pixel values of MNIST range from 0-255 (as the digits are stored as
# uint8). For this section, pixel values that are greater than 128 (in the
# original image) are converted to 1 and values that are less than 128 are
# converted to 0. See section 18.9.2 and 18.9.3 for why
train_images = tf.floor(tf.constant(train_images / 128, dtype = tf.float32))
test_images = tf.floor(tf.constant(test_images / 128, dtype = tf.float32))

train_labels = tf.constant(train_labels, dtype = tf.int32)
test_labels = tf.constant(test_labels, dtype = tf.int32)
```

Chúng ta có thể truy cập một ví dụ cụ thể, trong đó chứa hình ảnh và nhãn tương ứng.

```{.python .input}
#@tab mxnet
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab pytorch
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab tensorflow
image, label = train_images[2], train_labels[2]
image.shape, label.numpy()
```

Ví dụ của chúng ta, được lưu trong biến `image`, tương ứng với một hình ảnh có chiều cao và chiều rộng là $28$ pixel.

```{.python .input}
#@tab all
image.shape, image.dtype
```

Mã của chúng ta lưu nhãn của mỗi hình ảnh dưới dạng một scalar. Kiểu của nó là một số nguyên $32$-bit.

```{.python .input}
#@tab mxnet
label, type(label), label.dtype
```

```{.python .input}
#@tab pytorch
label, type(label)
```

```{.python .input}
#@tab tensorflow
label.numpy(), label.dtype
```

Chúng ta cũng có thể truy cập nhiều ví dụ cùng lúc.

```{.python .input}
#@tab mxnet
images, labels = mnist_train[10:38]
images.shape, labels.shape
```

```{.python .input}
#@tab pytorch
images = torch.stack([mnist_train[i][0] for i in range(10, 38)], dim=0)
labels = torch.tensor([mnist_train[i][1] for i in range(10, 38)])
images.shape, labels.shape
```

```{.python .input}
#@tab tensorflow
images = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
labels = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
images.shape, labels.shape
```

Hãy trực quan hóa các ví dụ này.

```{.python .input}
#@tab all
d2l.show_images(images, 2, 9);
```

## Mô hình xác suất cho bài toán phân loại

Trong một bài toán phân loại, chúng ta ánh xạ một ví dụ vào một danh mục. Ở đây, một ví dụ là hình ảnh grayscale $28\times 28$, còn một danh mục là một chữ số. (Tham khảo :numref:`sec_softmax` để xem phần giải thích chi tiết hơn.)
Một cách tự nhiên để diễn đạt bài toán phân loại là thông qua câu hỏi xác suất: nhãn nào có khả năng xảy ra cao nhất khi biết các đặc trưng (tức là các pixel của hình ảnh)? Ký hiệu các đặc trưng của ví dụ là $\mathbf x\in\mathbb R^d$ và nhãn là $y\in\mathbb R$. Ở đây, các đặc trưng là các pixel của hình ảnh; chúng ta có thể reshape một hình ảnh $2$-chiều thành một vector sao cho $d=28^2=784$, còn các nhãn là các chữ số.
Xác suất của nhãn khi biết các đặc trưng là $p(y  \mid  \mathbf{x})$. Nếu có thể tính được các xác suất này, tức là $p(y  \mid  \mathbf{x})$ với $y=0, \ldots,9$ trong ví dụ của chúng ta, thì bộ phân loại sẽ đưa ra dự đoán $\hat{y}$ theo biểu thức:

$$\hat{y} = \mathrm{argmax} \> p(y  \mid  \mathbf{x}).$$

Thật không may, điều này đòi hỏi chúng ta phải ước lượng $p(y  \mid  \mathbf{x})$ cho mọi giá trị của $\mathbf{x} = x_1, ..., x_d$. Hãy tưởng tượng rằng mỗi đặc trưng có thể nhận một trong $2$ giá trị. Ví dụ, đặc trưng $x_1 = 1$ có thể biểu thị rằng từ apple xuất hiện trong một tài liệu nhất định, còn $x_1 = 0$ biểu thị rằng từ đó không xuất hiện. Nếu có $30$ đặc trưng nhị phân như vậy, điều đó có nghĩa là chúng ta phải sẵn sàng phân loại bất kỳ giá trị nào trong số $2^{30}$ (hơn 1 tỷ!) giá trị có thể có của vector đầu vào $\mathbf{x}$.

Hơn nữa, việc học nằm ở đâu? Nếu cần phải xem mọi ví dụ có thể có để dự đoán nhãn tương ứng, thì thực ra chúng ta không học một quy luật nào mà chỉ ghi nhớ bộ dữ liệu.

## Bộ phân loại Naive Bayes

May mắn thay, bằng cách đưa ra một số giả định về tính độc lập có điều kiện, chúng ta có thể đưa vào một inductive bias và xây dựng một mô hình có khả năng khái quát hóa từ một tập hợp ví dụ huấn luyện tương đối nhỏ. Trước hết, hãy sử dụng định lý Bayes để biểu diễn bộ phân loại như sau:

$$\hat{y} = \mathrm{argmax}_y \> p(y  \mid  \mathbf{x}) = \mathrm{argmax}_y \> \frac{p( \mathbf{x}  \mid  y) p(y)}{p(\mathbf{x})}.$$

Lưu ý rằng mẫu số là hạng tử chuẩn hóa $p(\mathbf{x})$, không phụ thuộc vào giá trị của nhãn $y$. Vì vậy, chúng ta chỉ cần quan tâm đến việc so sánh tử số trên các giá trị khác nhau của $y$. Ngay cả khi việc tính mẫu số trở nên bất khả thi, chúng ta vẫn có thể bỏ qua nó, miễn là có thể tính được tử số. May mắn thay, ngay cả khi muốn khôi phục hằng số chuẩn hóa, chúng ta vẫn có thể làm được. Chúng ta luôn có thể khôi phục hạng tử chuẩn hóa vì $\sum_y p(y  \mid  \mathbf{x}) = 1$.

Bây giờ, hãy tập trung vào $p( \mathbf{x}  \mid  y)$. Sử dụng quy tắc chuỗi của xác suất, chúng ta có thể biểu diễn hạng tử $p( \mathbf{x}  \mid  y)$ như sau:

$$p(x_1  \mid y) \cdot p(x_2  \mid  x_1, y) \cdot ... \cdot p( x_d  \mid  x_1, ..., x_{d-1}, y).$$

Tự bản thân, biểu thức này không giúp chúng ta tiến xa hơn. Chúng ta vẫn phải ước lượng khoảng $2^d$ tham số. Tuy nhiên, nếu giả định rằng *các đặc trưng độc lập có điều kiện với nhau khi biết nhãn*, thì tình hình đột nhiên trở nên khả quan hơn nhiều, vì hạng tử này được đơn giản hóa thành $\prod_i p(x_i  \mid  y)$, cho chúng ta bộ dự đoán

$$\hat{y} = \mathrm{argmax}_y \> \prod_{i=1}^d p(x_i  \mid  y) p(y).$$

Nếu có thể ước lượng $p(x_i=1  \mid  y)$ cho mọi $i$ và $y$, rồi lưu giá trị đó trong $P_{xy}[i, y]$, trong đó $P_{xy}$ là ma trận $d\times n$ với $n$ là số lớp và $y\in\{1, \ldots, n\}$, thì chúng ta cũng có thể sử dụng giá trị này để ước lượng $p(x_i = 0 \mid y)$, tức là

$$
p(x_i = t_i \mid y) =
\begin{cases}
    P_{xy}[i, y] & \textrm{for } t_i=1 ;\\
    1 - P_{xy}[i, y] & \textrm{for } t_i = 0 .
\end{cases}
$$

Ngoài ra, chúng ta ước lượng $p(y)$ cho mọi $y$ và lưu nó trong $P_y[y]$, với $P_y$ là một vector có độ dài $n$. Khi đó, với bất kỳ ví dụ mới nào $\mathbf t = (t_1, t_2, \ldots, t_d)$, chúng ta có thể tính

$$\begin{aligned}\hat{y} &= \mathrm{argmax}_ y \ p(y)\prod_{i=1}^d   p(x_t = t_i \mid y) \\ &= \mathrm{argmax}_y \ P_y[y]\prod_{i=1}^d \ P_{xy}[i, y]^{t_i}\, \left(1 - P_{xy}[i, y]\right)^{1-t_i}\end{aligned}$$
:eqlabel:`eq_naive_bayes_estimation`

với mọi $y$. Như vậy, giả định về tính độc lập có điều kiện đã chuyển độ phức tạp của mô hình từ phụ thuộc theo hàm mũ vào số lượng đặc trưng $\mathcal{O}(2^dn)$ thành phụ thuộc tuyến tính, tức là $\mathcal{O}(dn)$.


## Huấn luyện

Vấn đề hiện tại là chúng ta chưa biết $P_{xy}$ và $P_y$. Vì vậy, trước tiên chúng ta cần ước lượng các giá trị của chúng dựa trên một số dữ liệu huấn luyện. Đây chính là quá trình *huấn luyện* mô hình. Việc ước lượng $P_y$ không quá khó. Vì chỉ xử lý $10$ lớp, chúng ta có thể đếm số lần xuất hiện $n_y$ của mỗi chữ số rồi chia cho tổng lượng dữ liệu $n$. Ví dụ, nếu chữ số 8 xuất hiện $n_8 = 5,800$ lần và tổng cộng có $n = 60,000$ hình ảnh, thì ước lượng xác suất là $p(y=8) = 0.0967$.

```{.python .input}
#@tab mxnet
X, Y = mnist_train[:]  # All training examples

n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], dim=0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab tensorflow
X = train_images
Y = train_labels

n_y = tf.Variable(tf.zeros(10))
for y in range(10):
    n_y[y].assign(tf.reduce_sum(tf.cast(Y == y, tf.float32)))
P_y = n_y / tf.reduce_sum(n_y)
P_y
```

Bây giờ hãy chuyển sang phần hơi khó hơn một chút là $P_{xy}$. Vì đã chọn các hình ảnh đen trắng, $p(x_i  \mid  y)$ biểu thị xác suất pixel $i$ được bật đối với lớp $y$. Giống như trước, chúng ta có thể đếm số lần $n_{iy}$ mà một sự kiện xảy ra rồi chia cho tổng số lần xuất hiện của $y$, tức là $n_y$. Tuy nhiên, có một vấn đề đáng lưu ý: một số pixel có thể không bao giờ có màu đen (chẳng hạn, với các hình ảnh được crop ngay ngắn, các pixel ở góc có thể luôn có màu trắng). Một cách thuận tiện để các nhà thống kê xử lý vấn đề này là thêm các số đếm giả vào tất cả các lần xuất hiện. Vì vậy, thay vì $n_{iy}$, chúng ta sử dụng $n_{iy}+1$, và thay vì $n_y$, chúng ta sử dụng $n_{y}+2$ (vì pixel $i$ có thể nhận hai giá trị - đen hoặc trắng). Phương pháp này còn được gọi là *Laplace Smoothing*. Cách làm này có vẻ tùy ý, nhưng có thể được giải thích dưới góc nhìn Bayes bằng mô hình Beta-binomial.

```{.python .input}
#@tab mxnet
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y.asnumpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab pytorch
n_x = torch.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = torch.tensor(X.numpy()[Y.numpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab tensorflow
n_x = tf.Variable(tf.zeros((10, 28, 28)))
for y in range(10):
    n_x[y].assign(tf.cast(tf.reduce_sum(
        X.numpy()[Y.numpy() == y], axis=0), tf.float32))
P_xy = (n_x + 1) / tf.reshape((n_y + 2), (10, 1, 1))

d2l.show_images(P_xy, 2, 5);
```

Bằng cách trực quan hóa các xác suất $10\times 28\times 28$ này (cho mỗi pixel của mỗi lớp), chúng ta có thể thu được một số chữ số trông khá kỳ lạ.

Bây giờ chúng ta có thể sử dụng :eqref:`eq_naive_bayes_estimation` để dự đoán một hình ảnh mới. Với $\mathbf x$, các hàm sau đây tính $p(\mathbf x \mid y)p(y)$ cho mọi $y$.

```{.python .input}
#@tab mxnet
def bayes_pred(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(axis=1)  # p(x|y)
    return np.array(p_xy) * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab pytorch
def bayes_pred(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(dim=1)  # p(x|y)
    return p_xy * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab tensorflow
def bayes_pred(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = tf.math.reduce_prod(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy * P_y

image, label = train_images[0], train_labels[0]
bayes_pred(image)
```

Điều này đã sai hoàn toàn! Để tìm hiểu lý do, hãy xem các xác suất theo từng pixel. Chúng thường là các số nằm giữa $0.001$ và $1$. Chúng ta đang nhân $784$ số trong đó. Ở đây, cần lưu ý rằng chúng ta đang tính các số này trên máy tính, nên số mũ có một phạm vi cố định. Điều xảy ra là chúng ta gặp phải hiện tượng *tràn dưới số học* (numerical underflow), tức là việc nhân tất cả các số nhỏ này tạo ra một số còn nhỏ hơn nữa, cho đến khi số đó bị làm tròn xuống thành zero. Chúng ta đã thảo luận về vấn đề lý thuyết này trong :numref:`sec_maximum_likelihood`, nhưng giờ đây có thể thấy rõ hiện tượng này trong thực tế.

Như đã thảo luận trong phần đó, chúng ta khắc phục vấn đề này bằng cách sử dụng tính chất $\log a b = \log a + \log b$, tức là chuyển sang tính tổng các logarithm.
Ngay cả khi cả $a$ và $b$ đều là các số nhỏ, giá trị logarithm vẫn nằm trong một phạm vi phù hợp.

```{.python .input}
#@tab mxnet
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab pytorch
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab tensorflow
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*tf.math.log(a).numpy())
```

Vì logarithm là một hàm tăng, chúng ta có thể viết lại :eqref:`eq_naive_bayes_estimation` như sau:

$$ \hat{y} = \mathrm{argmax}_y \ \log P_y[y] + \sum_{i=1}^d \Big[t_i\log P_{xy}[x_i, y] + (1-t_i) \log (1 - P_{xy}[x_i, y]) \Big].$$

Chúng ta có thể triển khai phiên bản ổn định sau:

```{.python .input}
#@tab mxnet
log_P_xy = np.log(P_xy)
log_P_xy_neg = np.log(1 - P_xy)
log_P_y = np.log(P_y)

def bayes_pred_stable(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab pytorch
log_P_xy = torch.log(P_xy)
log_P_xy_neg = torch.log(1 - P_xy)
log_P_y = torch.log(P_y)

def bayes_pred_stable(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab tensorflow
log_P_xy = tf.math.log(P_xy)
log_P_xy_neg = tf.math.log(1 - P_xy)
log_P_y = tf.math.log(P_y)

def bayes_pred_stable(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = tf.math.reduce_sum(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

Bây giờ chúng ta có thể kiểm tra xem dự đoán có chính xác hay không.

```{.python .input}
#@tab mxnet
# Convert label which is a scalar tensor of int32 dtype to a Python scalar
# integer for comparison
py.argmax(axis=0) == int(label)
```

```{.python .input}
#@tab pytorch
py.argmax(dim=0) == label
```

```{.python .input}
#@tab tensorflow
tf.argmax(py, axis=0, output_type = tf.int32) == label
```

Nếu dự đoán một vài ví dụ validation, chúng ta có thể thấy rằng bộ phân loại Bayes hoạt động khá tốt.

```{.python .input}
#@tab mxnet
def predict(X):
    return [bayes_pred_stable(x).argmax(axis=0).astype(np.int32) for x in X]

X, y = mnist_test[:18]
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab pytorch
def predict(X):
    return [bayes_pred_stable(x).argmax(dim=0).type(torch.int32).item()
            for x in X]

X = torch.stack([mnist_test[i][0] for i in range(18)], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(18)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab tensorflow
def predict(X):
    return [tf.argmax(
        bayes_pred_stable(x), axis=0, output_type = tf.int32).numpy()
            for x in X]

X = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
y = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

Cuối cùng, hãy tính độ chính xác tổng thể của bộ phân loại.

```{.python .input}
#@tab mxnet
X, y = mnist_test[:]
preds = np.array(predict(X), dtype=np.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))])
preds = torch.tensor(predict(X), dtype=torch.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab tensorflow
X = test_images
y = test_labels
preds = tf.constant(predict(X), dtype=tf.int32)
# Validation accuracy
tf.reduce_sum(tf.cast(preds == y, tf.float32)).numpy() / len(y)
```

Các mạng deep hiện đại đạt tỷ lệ lỗi nhỏ hơn $0.01$. Hiệu năng tương đối kém này là do các giả định thống kê không chính xác mà chúng ta đã đưa ra trong mô hình: chúng ta giả định rằng từng pixel đều được sinh ra *độc lập*, chỉ phụ thuộc vào label. Rõ ràng con người không viết các chữ số theo cách này, và giả định sai này đã dẫn đến sự thất bại của bộ phân loại (Bayes) quá ngây thơ của chúng ta.

## Tóm tắt
* Bằng cách sử dụng định lý Bayes, có thể xây dựng một bộ phân loại bằng cách giả định rằng tất cả các đặc trưng quan sát được là độc lập.
* Có thể huấn luyện bộ phân loại này trên một dataset bằng cách đếm số lần xuất hiện của các tổ hợp giữa label và giá trị pixel.
* Bộ phân loại này từng là tiêu chuẩn vàng trong nhiều thập kỷ đối với các tác vụ như phát hiện spam.

## Bài tập
1. Xét dataset $[[0,0], [0,1], [1,0], [1,1]]$ với các label được cho bởi XOR của hai phần tử $[0,1,1,0]$. Các xác suất của một bộ phân loại Naive Bayes được xây dựng trên dataset này là gì? Bộ phân loại có phân loại chính xác các điểm của chúng ta không? Nếu không, những giả định nào đã bị vi phạm?
1. Giả sử chúng ta không sử dụng Laplace smoothing khi ước lượng xác suất, và tại thời điểm testing xuất hiện một ví dụ dữ liệu chứa một giá trị chưa từng được quan sát trong quá trình training. Mô hình sẽ đưa ra output gì?
1. Bộ phân loại naive Bayes là một ví dụ cụ thể của Bayesian network, trong đó sự phụ thuộc giữa các biến ngẫu nhiên được mã hóa bằng một cấu trúc đồ thị. Mặc dù lý thuyết đầy đủ nằm ngoài phạm vi của phần này (xem :citet:`Koller.Friedman.2009` để biết đầy đủ chi tiết), hãy giải thích tại sao việc cho phép có sự phụ thuộc rõ ràng giữa hai biến đầu vào trong mô hình XOR lại cho phép tạo ra một bộ phân loại hoạt động chính xác.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/418)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1100)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1101)
:end_tab: