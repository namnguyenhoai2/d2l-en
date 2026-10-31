```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Mạng nơ-ron tích chập (LeNet)
:label:`sec_lenet`

Giờ đây, chúng ta đã có đầy đủ các thành phần cần thiết để xây dựng
một CNN hoạt động hoàn chỉnh.
Trong lần tiếp cận dữ liệu ảnh trước đây, chúng ta đã áp dụng
một mô hình tuyến tính với hồi quy softmax (:numref:`sec_softmax_scratch`)
và một MLP (:numref:`sec_mlp-implementation`)
cho các ảnh quần áo trong bộ dữ liệu Fashion-MNIST.
Để xử lý được dữ liệu như vậy, trước hết chúng ta đã làm phẳng từng ảnh từ ma trận $28\times28$
thành một vector có độ dài cố định gồm $784$ chiều,
sau đó xử lý chúng trong các lớp kết nối đầy đủ.
Giờ đây khi đã nắm được các lớp tích chập,
chúng ta có thể giữ lại cấu trúc không gian của ảnh.
Một lợi ích bổ sung khi thay các lớp kết nối đầy đủ bằng các lớp tích chập
là chúng ta sẽ có các mô hình gọn hơn đáng kể, cần ít tham số hơn nhiều.

Trong phần này, chúng ta sẽ giới thiệu *LeNet*,
một trong những CNN được công bố sớm nhất
và thu hút sự chú ý rộng rãi nhờ hiệu năng trên các tác vụ thị giác máy tính.
Mô hình được Yann LeCun, khi đó là nhà nghiên cứu tại AT&T Bell Labs,
giới thiệu (và được đặt theo tên ông)
nhằm nhận dạng các chữ số viết tay trong ảnh :cite:`LeCun.Bottou.Bengio.ea.1998`.
Công trình này là đỉnh cao
của một thập kỷ nghiên cứu phát triển công nghệ;
nhóm của LeCun đã công bố nghiên cứu đầu tiên huấn luyện thành công
CNN bằng backpropagation :cite:`LeCun.Boser.Denker.ea.1989`.

Vào thời điểm đó, LeNet đạt kết quả xuất sắc,
sánh ngang hiệu năng của support vector machines,
khi ấy là một phương pháp thống trị trong học có giám sát, với tỷ lệ lỗi dưới 1% cho mỗi chữ số.
LeNet sau đó được điều chỉnh để nhận dạng chữ số
phục vụ xử lý tiền gửi tại các máy ATM.
Cho đến ngày nay, một số máy ATM vẫn chạy mã
mà Yann LeCun và đồng nghiệp Leon Bottou đã viết vào những năm 1990!

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
from types import FunctionType
```

## LeNet

Ở mức tổng quan, (**LeNet (LeNet-5) gồm hai phần:
(i) một encoder tích chập gồm hai lớp tích chập; và
(ii) một khối dense gồm ba lớp kết nối đầy đủ**).
Kiến trúc được tóm tắt trong :numref:`img_lenet`.

![Luồng dữ liệu trong LeNet. Đầu vào là một chữ số viết tay, đầu ra là xác suất trên 10 kết quả có thể có.](../img/lenet.svg)
:label:`img_lenet`

Các đơn vị cơ bản trong mỗi khối tích chập
là một lớp tích chập, một hàm kích hoạt sigmoid,
và một phép average pooling tiếp theo.
Lưu ý rằng mặc dù ReLU và max-pooling hoạt động tốt hơn,
chúng vẫn chưa được phát hiện vào thời điểm đó.
Mỗi lớp tích chập sử dụng kernel $5\times 5$
và hàm kích hoạt sigmoid.
Các lớp này ánh xạ đầu vào được sắp xếp theo không gian
thành một số feature map hai chiều, thường là
tăng số lượng kênh.
Lớp tích chập đầu tiên có 6 kênh đầu ra,
trong khi lớp thứ hai có 16 kênh.
Mỗi phép pooling $2\times2$ (stride 2)
giảm số chiều đi $4$ lần thông qua downsampling theo không gian.
Khối tích chập tạo ra đầu ra có shape
(batch size, số lượng kênh, chiều cao, chiều rộng).

Để truyền đầu ra từ khối tích chập
sang khối dense,
chúng ta phải làm phẳng từng mẫu trong minibatch.
Nói cách khác, chúng ta lấy đầu vào bốn chiều này và biến đổi nó
thành đầu vào hai chiều mà các lớp kết nối đầy đủ mong đợi:
nhắc lại rằng, biểu diễn hai chiều mà ta cần sử dụng chiều thứ nhất để lập chỉ mục các mẫu trong minibatch
và chiều thứ hai để biểu diễn vector phẳng của mỗi mẫu.
Khối dense của LeNet có ba lớp kết nối đầy đủ,
với lần lượt 120, 84 và 10 đầu ra.
Vì chúng ta vẫn đang thực hiện phân loại,
lớp đầu ra 10 chiều tương ứng
với số lượng lớp đầu ra có thể có.

Mặc dù để thực sự hiểu
điều đang diễn ra bên trong LeNet có thể cần một chút công sức,
chúng tôi hy vọng rằng đoạn mã sau đây sẽ thuyết phục bạn
rằng việc triển khai những mô hình như vậy bằng các deep learning framework hiện đại
đơn giản đáng kể.
Chúng ta chỉ cần khởi tạo một khối `Sequential`
và nối chuỗi các lớp phù hợp,
sử dụng khởi tạo Xavier như
đã giới thiệu trong :numref:`subsec_xavier`.

```{.python .input}
%%tab pytorch
def init_cnn(module):  #@save
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
```

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LeNet(d2l.Classifier):  #@save
    """The LeNet-5 model."""
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(channels=6, kernel_size=5, padding=2,
                          activation='sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120, activation='sigmoid'),
                nn.Dense(84, activation='sigmoid'),
                nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.LazyLinear(120), nn.Sigmoid(),
                nn.LazyLinear(84), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       activation='sigmoid', padding='same'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                                       activation='sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(120, activation='sigmoid'),
                tf.keras.layers.Dense(84, activation='sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class LeNet(d2l.Classifier):  #@save
    """The LeNet-5 model."""
    lr: float = 0.1
    num_classes: int = 10
    kernel_init: FunctionType = nn.initializers.xavier_uniform

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(features=6, kernel_size=(5, 5), padding='SAME',
                    kernel_init=self.kernel_init()),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(features=16, kernel_size=(5, 5), padding='VALID',
                    kernel_init=self.kernel_init()),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(features=120, kernel_init=self.kernel_init()),
            nn.sigmoid,
            nn.Dense(features=84, kernel_init=self.kernel_init()),
            nn.sigmoid,
            nn.Dense(features=self.num_classes, kernel_init=self.kernel_init())
        ])
```

Chúng tôi đã điều chỉnh đôi chút khi tái tạo LeNet, cụ thể là thay lớp kích hoạt Gaussian bằng
một lớp softmax. Điều này đơn giản hóa đáng kể việc triển khai, đặc biệt vì
Gaussian decoder ngày nay hiếm khi được sử dụng. Ngoài điểm đó, mạng này khớp với
kiến trúc LeNet-5 gốc.

:begin_tab:`pytorch, mxnet, tensorflow`
Hãy xem điều gì xảy ra bên trong mạng. Bằng cách đưa một ảnh
một kênh (đen trắng)
$28 \times 28$ qua mạng
và in shape đầu ra tại mỗi lớp,
chúng ta có thể [**kiểm tra mô hình**] để đảm bảo
các phép toán của nó phù hợp với
những gì ta mong đợi từ :numref:`img_lenet_vert`.
:end_tab:

:begin_tab:`jax`
Hãy xem điều gì xảy ra bên trong mạng. Bằng cách đưa một ảnh
một kênh (đen trắng)
$28 \times 28$ qua mạng
và in shape đầu ra tại mỗi lớp,
chúng ta có thể [**kiểm tra mô hình**] để đảm bảo
các phép toán của nó phù hợp với
những gì ta mong đợi từ :numref:`img_lenet_vert`.
Flax cung cấp `nn.tabulate`, một phương thức tiện dụng để tóm tắt các lớp và
tham số trong mạng của chúng ta. Ở đây, chúng ta sử dụng phương thức `bind` để tạo một mô hình bound.
Các biến giờ đây được bound với lớp `d2l.Module`, tức là mô hình bound này
trở thành một đối tượng có trạng thái, sau đó có thể được dùng để truy cập
thuộc tính đối tượng `net` của `Sequential`
và `layers` bên trong nó. Lưu ý rằng phương thức `bind`
chỉ nên được dùng để thử nghiệm tương tác và không phải là
thay thế trực tiếp cho phương thức `apply`.
:end_tab:

![Ký hiệu rút gọn cho LeNet-5.](../img/lenet-vert.svg)
:label:`img_lenet_vert`

```{.python .input}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = d2l.randn(*X_shape)
    for layer in self.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
        
model = LeNet()
model.layer_summary((1, 1, 28, 28))
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = d2l.normal(X_shape)
    for layer in self.net.layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 28, 28, 1))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape, key=d2l.get_key()):
    X = jnp.zeros(X_shape)
    params = self.init(key, X)
    bound_model = self.clone().bind(params, mutable=['batch_stats'])
    _ = bound_model(X)
    for layer in bound_model.net.layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 28, 28, 1))
```

Lưu ý rằng chiều cao và chiều rộng của biểu diễn
tại mỗi lớp trong toàn bộ khối tích chập
đều giảm đi (so với lớp trước đó).
Lớp tích chập đầu tiên sử dụng padding hai pixel
để bù cho sự giảm chiều cao và chiều rộng
vốn sẽ xảy ra khi dùng kernel $5 \times 5$.
Ngoài lề, kích thước ảnh $28 \times 28$ pixel trong bộ dữ liệu OCR MNIST gốc
là kết quả của việc *cắt bớt* hai hàng (và cột) pixel từ
các bản quét gốc có kích thước $32 \times 32$ pixel. Việc này chủ yếu được thực hiện để
tiết kiệm dung lượng (giảm 30%) vào thời điểm mà megabyte còn rất đáng giá.

Ngược lại, lớp tích chập thứ hai không dùng padding,
do đó chiều cao và chiều rộng đều giảm bốn pixel.
Khi đi lên qua chồng lớp,
số lượng kênh tăng dần theo từng lớp
từ 1 ở đầu vào lên 6 sau lớp tích chập đầu tiên
và 16 sau lớp tích chập thứ hai.
Tuy nhiên, mỗi lớp pooling giảm một nửa chiều cao và chiều rộng.
Cuối cùng, mỗi lớp kết nối đầy đủ giảm số chiều,
và cuối cùng tạo ra đầu ra có số chiều
khớp với số lượng lớp.


## Huấn luyện

Giờ đây khi đã triển khai mô hình,
hãy [**chạy một thí nghiệm để xem mô hình LeNet-5 hoạt động thế nào trên Fashion-MNIST**].

Mặc dù CNN có ít tham số hơn,
chúng vẫn có thể tốn chi phí tính toán hơn
so với các MLP có độ sâu tương tự
vì mỗi tham số tham gia vào nhiều phép nhân hơn.
Nếu bạn có quyền truy cập GPU, đây có thể là thời điểm phù hợp
để sử dụng nó nhằm tăng tốc huấn luyện.
Lưu ý rằng
lớp `d2l.Trainer` đảm nhiệm toàn bộ các chi tiết.
Theo mặc định, nó khởi tạo các tham số mô hình trên
các thiết bị khả dụng.
Tương tự như với MLP, hàm mất mát của chúng ta là cross-entropy,
và chúng ta tối thiểu hóa nó bằng stochastic gradient descent theo minibatch.

```{.python .input}
%%tab pytorch, mxnet, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = LeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = LeNet(lr=0.1)
    trainer.fit(model, data)
```

## Tóm tắt

Chúng ta đã đạt được tiến bộ đáng kể trong chương này. Chúng ta đã chuyển từ các MLP của thập niên 1980 sang các CNN của thập niên 1990 và đầu những năm 2000. Các kiến trúc được đề xuất, chẳng hạn dưới dạng LeNet-5, vẫn còn ý nghĩa ngay cả đến ngày nay. Cần so sánh tỷ lệ lỗi trên Fashion-MNIST có thể đạt được bằng LeNet-5 với cả kết quả tốt nhất có thể của MLP (:numref:`sec_mlp-implementation`) lẫn kết quả của các kiến trúc tiên tiến hơn đáng kể như ResNet (:numref:`sec_resnet`). LeNet giống với loại sau hơn nhiều so với loại trước. Một trong những khác biệt chính, như chúng ta sẽ thấy, là lượng tính toán lớn hơn đã cho phép các kiến trúc phức tạp hơn đáng kể.

Khác biệt thứ hai là mức độ dễ dàng tương đối khi chúng ta triển khai LeNet. Những gì từng là một thách thức kỹ thuật đòi hỏi hàng tháng viết mã C++ và assembly, kỹ thuật hóa để cải tiến SN, một công cụ deep learning ban đầu dựa trên Lisp :cite:`Bottou.Le-Cun.1988`, rồi cuối cùng mới thử nghiệm mô hình, nay có thể hoàn thành trong vài phút. Chính sự tăng năng suất đáng kinh ngạc này đã dân chủ hóa mạnh mẽ việc phát triển mô hình deep learning. Trong chương tiếp theo, chúng ta sẽ đi sâu vào chủ đề này để xem nó sẽ dẫn ta tới đâu.

## Bài tập

1. Hãy hiện đại hóa LeNet. Triển khai và kiểm thử các thay đổi sau:
    1. Thay average pooling bằng max-pooling.
    1. Thay lớp softmax bằng ReLU.
1. Hãy thử thay đổi kích thước của mạng kiểu LeNet để cải thiện độ chính xác, bên cạnh max-pooling và ReLU.
    1. Điều chỉnh kích thước cửa sổ convolution.
    1. Điều chỉnh số lượng kênh đầu ra.
    1. Điều chỉnh số lượng lớp convolution.
    1. Điều chỉnh số lượng lớp fully connected.
    1. Điều chỉnh learning rate và các chi tiết huấn luyện khác (ví dụ: khởi tạo và số epoch).
1. Thử nghiệm mạng đã cải tiến trên bộ dữ liệu MNIST gốc.
1. Hiển thị activation của lớp thứ nhất và thứ hai của LeNet đối với các đầu vào khác nhau (ví dụ: áo len và áo khoác).
1. Điều gì xảy ra với các activation khi bạn đưa những hình ảnh khác biệt đáng kể vào mạng (ví dụ: mèo, ô tô hoặc thậm chí nhiễu ngẫu nhiên)?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/73)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/74)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/275)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18000)
:end_tab: