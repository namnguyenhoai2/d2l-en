# Hạ dốc Gradient ngẫu nhiên theo Minibatch
:label:`sec_minibatch_sgd`

Cho đến nay, chúng ta đã gặp hai thái cực trong cách tiếp cận học dựa trên gradient: :numref:`sec_gd` sử dụng toàn bộ tập dữ liệu để tính gradient và cập nhật tham số, mỗi lần một lượt quét. Ngược lại, :numref:`sec_sgd` xử lý từng ví dụ huấn luyện riêng lẻ để đạt được tiến triển.
Mỗi phương pháp đều có những nhược điểm riêng.
Gradient descent không đặc biệt *hiệu quả về dữ liệu* khi dữ liệu rất tương đồng.
Stochastic gradient descent không đặc biệt *hiệu quả về tính toán* vì CPU và GPU không thể khai thác toàn bộ sức mạnh của vectorization.
Điều này gợi ý rằng có thể tồn tại một phương án ở giữa,
và thực tế đó chính là điều chúng ta đã sử dụng trong các ví dụ đã thảo luận cho đến nay.

## Vectorization và Cache

Cốt lõi của quyết định sử dụng minibatch là hiệu quả tính toán. Điều này dễ hiểu nhất khi xem xét parallelization trên nhiều GPU và nhiều server. Trong trường hợp này, chúng ta cần gửi ít nhất một ảnh đến mỗi GPU. Với 8 GPU trên mỗi server và 16 server, kích thước minibatch đã không thể nhỏ hơn 128.

Mọi việc tinh tế hơn đôi chút khi xét GPU đơn lẻ hoặc thậm chí CPU. Các thiết bị này có nhiều loại bộ nhớ, thường có nhiều loại đơn vị tính toán và các ràng buộc băng thông khác nhau giữa chúng.
Ví dụ, CPU có một số lượng nhỏ thanh ghi, sau đó là cache L1, L2 và trong một số trường hợp cả L3 (được chia sẻ giữa các lõi xử lý khác nhau).
Các cache này có kích thước và độ trễ tăng dần (đồng thời băng thông giảm dần).
Chỉ cần nói rằng, bộ xử lý có thể thực hiện nhiều phép toán hơn rất nhiều so với khả năng cung cấp của giao diện bộ nhớ chính.

Thứ nhất, một CPU 2GHz với 16 lõi và vectorization AVX-512 có thể xử lý tối đa $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$ byte mỗi giây. Khả năng của GPU dễ dàng vượt con số này gấp 100 lần. Mặt khác, một bộ xử lý server tầm trung có thể không có băng thông lớn hơn nhiều so với 100 GB/s, tức là chưa đến một phần mười mức cần thiết để cung cấp đủ dữ liệu cho bộ xử lý. Tệ hơn nữa, không phải mọi truy cập bộ nhớ đều như nhau: các giao diện bộ nhớ thường rộng 64 bit hoặc hơn (ví dụ, trên GPU lên đến 384 bit), vì vậy việc đọc một byte đơn lẻ phải chịu chi phí của một lần truy cập rộng hơn nhiều.

Thứ hai, lần truy cập đầu tiên có overhead đáng kể, trong khi truy cập tuần tự tương đối rẻ (điều này thường được gọi là burst read). Còn nhiều điều khác cần lưu ý, chẳng hạn như caching khi có nhiều socket, chiplet và các cấu trúc khác.
Xem [bài viết Wikipedia này](https://en.wikipedia.org/wiki/Cache_hierarchy)
để thảo luận sâu hơn.

Cách giảm bớt các ràng buộc này là sử dụng một hệ phân cấp cache CPU đủ nhanh để thực sự cung cấp dữ liệu cho bộ xử lý. Đây *chính là* động lực thúc đẩy batching trong deep learning. Để đơn giản, hãy xét phép nhân ma trận-ma trận, giả sử $\mathbf{A} = \mathbf{B}\mathbf{C}$. Chúng ta có một số lựa chọn để tính $\mathbf{A}$. Ví dụ, chúng ta có thể thử các cách sau:

1. Chúng ta có thể tính $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}$, tức là tính theo từng phần tử bằng các dot product.
1. Chúng ta có thể tính $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}$, tức là tính từng cột một. Tương tự, chúng ta có thể tính $\mathbf{A}$ từng hàng $\mathbf{A}_{i,:}$ một.
1. Chúng ta có thể đơn giản tính $\mathbf{A} = \mathbf{B} \mathbf{C}$.
1. Chúng ta có thể chia $\mathbf{B}$ và $\mathbf{C}$ thành các ma trận khối nhỏ hơn và tính $\mathbf{A}$ từng khối một.

Nếu làm theo lựa chọn đầu tiên, chúng ta sẽ cần sao chép một vector hàng và một vector cột vào CPU mỗi khi muốn tính một phần tử $\mathbf{A}_{ij}$. Tệ hơn nữa, do các phần tử ma trận được căn chỉnh tuần tự, chúng ta phải truy cập nhiều vị trí rời rạc đối với một trong hai vector khi đọc chúng từ bộ nhớ. Lựa chọn thứ hai thuận lợi hơn nhiều. Với lựa chọn này, chúng ta có thể giữ vector cột $\mathbf{C}_{:,j}$ trong cache CPU trong khi tiếp tục duyệt qua $\mathbf{B}$. Điều này giảm một nửa yêu cầu về băng thông bộ nhớ, nhờ đó truy cập nhanh hơn tương ứng. Tất nhiên, lựa chọn 3 là mong muốn nhất. Đáng tiếc, hầu hết ma trận có thể không hoàn toàn vừa cache (xét cho cùng, đây chính là vấn đề chúng ta đang bàn). Tuy nhiên, lựa chọn 4 mang lại một phương án thay thế hữu ích trong thực tế: chúng ta có thể chuyển các khối của ma trận vào cache và nhân chúng cục bộ. Các thư viện được tối ưu hóa sẽ lo việc này cho chúng ta. Hãy xem các phép toán này hiệu quả đến mức nào trong thực tế.

Ngoài hiệu quả tính toán, overhead do Python và chính deep learning framework tạo ra cũng đáng kể. Hãy nhớ rằng mỗi khi thực thi một lệnh, trình thông dịch Python gửi một lệnh đến MXNet engine, nơi cần chèn lệnh đó vào computational graph và xử lý nó trong quá trình scheduling. Overhead như vậy có thể gây bất lợi đáng kể. Tóm lại, nên sử dụng vectorization (và ma trận) bất cứ khi nào có thể.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
import time
npx.set_np()

A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import time
import torch
from torch import nn

A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
import time

A = tf.Variable(d2l.zeros((256, 256)))
B = tf.Variable(d2l.normal([256, 256], 0, 1))
C = tf.Variable(d2l.normal([256, 256], 0, 1))
```

Vì chúng ta sẽ thường xuyên benchmark thời gian chạy trong phần còn lại của cuốn sách, hãy định nghĩa một timer.

```{.python .input}
#@tab all
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

timer = Timer()
```

Phép gán theo từng phần tử chỉ đơn giản duyệt qua tất cả các hàng và cột của $\mathbf{B}$ và $\mathbf{C}$ tương ứng để gán giá trị cho $\mathbf{A}$.

```{.python .input}
#@tab mxnet
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

Một chiến lược nhanh hơn là thực hiện phép gán theo cột.

```{.python .input}
#@tab mxnet
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
timer.stop()
```

Cuối cùng, cách hiệu quả nhất là thực hiện toàn bộ phép toán trong một khối. 
Lưu ý rằng việc nhân hai ma trận bất kỳ $\mathbf{B} \in \mathbb{R}^{m \times n}$ và $\mathbf{C} \in \mathbb{R}^{n \times p}$ cần xấp xỉ $2mnp$ phép toán dấu phẩy động,
khi phép nhân vô hướng và phép cộng được tính là các phép toán riêng biệt (trong thực tế chúng được fused).
Do đó, nhân hai ma trận $256 \times 256$
cần $0.03$ tỷ phép toán dấu phẩy động.
Hãy xem tốc độ tương ứng của các phép toán này.

```{.python .input}
#@tab mxnet
# Compute A = BC in one go
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab pytorch
# Compute A = BC in one go
timer.start()
A = torch.mm(B, C)
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## Minibatch

:label:`sec_minibatches`

Trước đây, chúng ta mặc nhiên cho rằng sẽ đọc *minibatch* dữ liệu thay vì từng quan sát riêng lẻ để cập nhật tham số. Bây giờ, chúng ta sẽ đưa ra một lý giải ngắn gọn cho điều này. Xử lý từng quan sát riêng lẻ yêu cầu thực hiện nhiều phép nhân ma trận-vector (hoặc thậm chí vector-vector) đơn lẻ, vốn khá tốn kém và gây ra overhead đáng kể từ deep learning framework bên dưới. Điều này áp dụng cả khi đánh giá một network trên dữ liệu (thường được gọi là inference) lẫn khi tính gradient để cập nhật tham số. Nghĩa là, điều này áp dụng bất cứ khi nào chúng ta thực hiện $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$ trong đó

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

Chúng ta có thể tăng hiệu quả *tính toán* của phép toán này bằng cách áp dụng nó cho một minibatch các quan sát mỗi lần. Nghĩa là, chúng ta thay gradient $\mathbf{g}_t$ trên một quan sát đơn lẻ bằng gradient trên một batch nhỏ

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$

Hãy xem điều này ảnh hưởng thế nào đến các đặc tính thống kê của $\mathbf{g}_t$: vì cả $\mathbf{x}_t$ lẫn mọi phần tử của minibatch $\mathcal{B}_t$ đều được lấy ngẫu nhiên đồng đều từ tập huấn luyện, kỳ vọng của gradient vẫn không đổi. Mặt khác, phương sai giảm đáng kể. Vì gradient minibatch được cấu thành từ $b \stackrel{\textrm{def}}{=} |\mathcal{B}_t|$ gradient độc lập được lấy trung bình, độ lệch chuẩn của nó giảm đi một hệ số $b^{-\frac{1}{2}}$. Bản thân điều này là một điều tốt, vì nó có nghĩa là các cập nhật được căn chỉnh với full gradient một cách đáng tin cậy hơn.

Một cách ngây thơ, điều này có thể cho thấy việc chọn minibatch lớn $\mathcal{B}_t$ sẽ luôn là mong muốn. Than ôi, sau một ngưỡng nào đó, mức giảm thêm về độ lệch chuẩn là tối thiểu so với mức tăng tuyến tính của chi phí tính toán. Trong thực tế, chúng ta chọn minibatch đủ lớn để mang lại hiệu quả tính toán tốt nhưng vẫn vừa với bộ nhớ của GPU. Để minh họa mức tiết kiệm, hãy xem một số code. Trong đó, chúng ta thực hiện cùng phép nhân ma trận-ma trận, nhưng lần này được chia thành các "minibatch" gồm 64 cột mỗi lần.

```{.python .input}
#@tab mxnet
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

Như có thể thấy, việc tính toán trên minibatch về cơ bản hiệu quả tương đương với trên toàn bộ ma trận. Cần có một lưu ý. Trong :numref:`sec_batch_norm` chúng ta đã sử dụng một dạng regularization phụ thuộc nhiều vào lượng phương sai trong minibatch. Khi tăng kích thước minibatch, phương sai giảm và cùng với đó là lợi ích từ việc chèn nhiễu do batch normalization. Xem, chẳng hạn, :citet:`Ioffe.2017` để biết chi tiết về cách rescale và tính toán các hạng tử phù hợp.

## Đọc Dataset

Hãy xem cách các minibatch được tạo hiệu quả từ dữ liệu. Trong phần sau, chúng ta sử dụng một dataset do NASA phát triển để kiểm tra tiếng ồn cánh [từ các loại máy bay khác nhau](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) nhằm so sánh các thuật toán tối ưu hóa này. Để thuận tiện, chúng ta chỉ sử dụng $1,500$ mẫu đầu tiên. Dữ liệu được whitening trong bước preprocessing, tức là chúng ta loại bỏ giá trị trung bình và rescale phương sai về $1$ cho mỗi tọa độ.

```{.python .input}
#@tab mxnet
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## Cài đặt từ đầu

Hãy nhớ lại phần cài đặt minibatch stochastic gradient descent từ :numref:`sec_linear_scratch`. Sau đây, chúng ta cung cấp một phần cài đặt tổng quát hơn một chút. Để thuận tiện, nó có cùng call signature với các thuật toán tối ưu hóa khác được giới thiệu sau trong chương này. Cụ thể, chúng ta thêm đầu vào trạng thái `states` và đặt hyperparameter trong dictionary `hyperparams`. Ngoài ra, chúng ta sẽ lấy trung bình loss của từng mẫu minibatch trong hàm huấn luyện, do đó gradient trong thuật toán tối ưu hóa không cần được chia cho batch size.

```{.python .input}
#@tab mxnet
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

```{.python .input}
#@tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, states, hyperparams):
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr']*grad)
```

Tiếp theo, chúng ta cài đặt một hàm huấn luyện tổng quát để hỗ trợ sử dụng các thuật toán tối ưu hóa khác được giới thiệu sau trong chương này. Hàm này khởi tạo một mô hình linear regression và có thể được dùng để huấn luyện mô hình bằng minibatch stochastic gradient descent cùng các thuật toán khác được giới thiệu tiếp theo.

```{.python .input}
#@tab mxnet
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab pytorch
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # Train
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

Hãy xem quá trình tối ưu hóa diễn ra như thế nào với batch gradient descent. Có thể thực hiện điều này bằng cách đặt minibatch size thành 1500 (tức là tổng số mẫu). Kết quả là các tham số mô hình chỉ được cập nhật một lần mỗi epoch. Tiến triển rất ít. Thực tế, sau 6 bước, tiến triển bị chững lại.

```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

Khi batch size bằng 1, chúng ta sử dụng stochastic gradient descent để tối ưu hóa. Để đơn giản khi cài đặt, chúng ta chọn một learning rate không đổi (dù nhỏ). Trong stochastic gradient descent, các tham số mô hình được cập nhật mỗi khi một mẫu được xử lý. Trong trường hợp của chúng ta, điều này tương ứng với 1500 lần cập nhật mỗi epoch. Như có thể thấy, mức giảm của giá trị hàm mục tiêu chậm lại sau một epoch. Mặc dù cả hai quy trình đều xử lý 1500 mẫu trong một epoch, stochastic gradient descent tốn nhiều thời gian hơn gradient descent trong thử nghiệm của chúng ta. Điều này là do stochastic gradient descent cập nhật tham số thường xuyên hơn và việc xử lý từng quan sát riêng lẻ kém hiệu quả hơn.

```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

Cuối cùng, khi batch size bằng 100, chúng ta sử dụng minibatch stochastic gradient descent để tối ưu hóa. Thời gian cần thiết cho mỗi epoch ngắn hơn thời gian cần cho stochastic gradient descent và batch gradient descent.

```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

Khi giảm batch size xuống 10, thời gian cho mỗi epoch tăng lên vì workload của mỗi batch được thực thi kém hiệu quả hơn.

```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

Bây giờ chúng ta có thể so sánh thời gian với loss của bốn thử nghiệm trước đó. Như có thể thấy, mặc dù stochastic gradient descent hội tụ nhanh hơn GD xét theo số lượng mẫu đã xử lý, nó cần nhiều thời gian hơn GD để đạt cùng loss vì việc tính gradient từng mẫu một không hiệu quả bằng. Minibatch stochastic gradient descent có thể cân bằng giữa tốc độ hội tụ và hiệu quả tính toán. Minibatch size là 10 hiệu quả hơn stochastic gradient descent; minibatch size là 100 thậm chí vượt GD xét về runtime.

```{.python .input}
#@tab all
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## Cài đặt ngắn gọn

Trong Gluon, chúng ta có thể sử dụng lớp `Trainer` để gọi các thuật toán tối ưu hóa. Lớp này được dùng để cài đặt một hàm huấn luyện tổng quát. Chúng ta sẽ sử dụng nó xuyên suốt chương hiện tại.

```{.python .input}
#@tab mxnet
#@save
def train_concise_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input}
#@tab pytorch
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # Initialization
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # `MSELoss` computes squared error without the 1/2 factor
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input}
#@tab tensorflow
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    loss = tf.keras.losses.MeanSquaredError()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
                # `MeanSquaredError` computes squared error without the 1/2
                # factor
                r = (d2l.evaluate_loss(net, data_iter, loss) / 2,)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

Sử dụng Gluon để lặp lại thử nghiệm trước cho thấy hành vi giống hệt.

```{.python .input}
#@tab mxnet
data_iter, _ = get_data_ch11(10)
train_concise_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

```{.python .input}
#@tab pytorch
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

## Tóm tắt

* Vectorization giúp code hiệu quả hơn nhờ giảm overhead từ deep learning framework và nhờ tính cục bộ bộ nhớ cùng caching tốt hơn trên CPU và GPU.
* Có sự đánh đổi giữa hiệu quả thống kê từ stochastic gradient descent và hiệu quả tính toán từ việc xử lý các batch dữ liệu lớn cùng lúc.
* Minibatch stochastic gradient descent mang lại ưu điểm của cả hai: hiệu quả tính toán và hiệu quả thống kê.
* Trong minibatch stochastic gradient descent, chúng ta xử lý các batch dữ liệu thu được từ một hoán vị ngẫu nhiên của dữ liệu huấn luyện (tức là mỗi quan sát chỉ được xử lý một lần mỗi epoch, dù theo thứ tự ngẫu nhiên).
* Nên giảm dần learning rate trong quá trình huấn luyện.
* Nhìn chung, minibatch stochastic gradient descent nhanh hơn stochastic gradient descent và gradient descent để hội tụ đến rủi ro nhỏ hơn, khi đo theo thời gian thực.

## Bài tập

1. Thay đổi batch size và learning rate, rồi quan sát tốc độ giảm của giá trị hàm mục tiêu và thời gian tiêu tốn trong mỗi epoch.
1. Đọc tài liệu MXNet và sử dụng hàm `set_learning_rate` của lớp `Trainer` để giảm learning rate của minibatch stochastic gradient descent xuống 1/10 giá trị trước đó sau mỗi epoch.
1. So sánh minibatch stochastic gradient descent với một biến thể thực sự *lấy mẫu có hoàn lại* từ tập huấn luyện. Điều gì xảy ra?
1. Một vị thần đèn độc ác sao chép dataset của bạn mà không cho bạn biết (tức là mỗi quan sát xuất hiện hai lần và dataset của bạn tăng lên gấp đôi kích thước ban đầu, nhưng không ai nói cho bạn biết). Hành vi của stochastic gradient descent, minibatch stochastic gradient descent và gradient descent thay đổi như thế nào?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/353)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1068)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1069)
:end_tab: