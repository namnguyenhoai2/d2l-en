```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Mạng sử dụng các khối (VGG)
:label:`sec_vgg`

Mặc dù AlexNet đã cung cấp bằng chứng thực nghiệm rằng các CNN sâu
có thể đạt được kết quả tốt, nó không đưa ra một khuôn mẫu tổng quát
để định hướng các nhà nghiên cứu sau này trong việc thiết kế mạng mới.
Trong các phần tiếp theo, chúng ta sẽ giới thiệu một số khái niệm heuristic
thường được dùng để thiết kế mạng sâu.

Tiến bộ trong lĩnh vực này phản chiếu sự phát triển của VLSI (very large scale integration)
trong thiết kế chip,
khi các kỹ sư chuyển từ việc bố trí transistor
sang các phần tử logic rồi đến các khối logic :cite:`Mead.1980`.
Tương tự, thiết kế kiến trúc mạng nơ-ron
đã dần trở nên trừu tượng hơn,
khi các nhà nghiên cứu chuyển từ tư duy theo
từng nơ-ron riêng lẻ sang toàn bộ các tầng,
và hiện nay là các khối, tức những mẫu tầng lặp lại. Một thập kỷ sau, điều này hiện đã
tiến tới việc các nhà nghiên cứu sử dụng toàn bộ các mô hình đã huấn luyện để tái sử dụng chúng cho những tác vụ khác,
dù có liên quan. Những mô hình tiền huấn luyện lớn như vậy thường được gọi là
*foundation models* :cite:`bommasani2021opportunities`.

Quay lại thiết kế mạng. Ý tưởng sử dụng các khối lần đầu xuất hiện từ
Visual Geometry Group (VGG) tại Đại học Oxford,
trong mạng *VGG* cùng tên của họ :cite:`Simonyan.Zisserman.2014`.
Có thể dễ dàng hiện thực các cấu trúc lặp lại này trong mã
bằng bất kỳ framework deep learning hiện đại nào, sử dụng vòng lặp và chương trình con.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
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
```

## (**Các khối VGG**)
:label:`subsec_vgg-blocks`

Khối xây dựng cơ bản của CNN
là một chuỗi gồm:
(i) một tầng tích chập
với padding để duy trì độ phân giải,
(ii) một phi tuyến như ReLU,
(iii) một tầng pooling như
max-pooling để giảm độ phân giải. Một trong những vấn đề của
cách tiếp cận này là độ phân giải không gian giảm khá nhanh. Cụ thể,
điều này áp đặt giới hạn cứng là $\log_2 d$ tầng tích chập cho mạng trước khi toàn bộ
các chiều ($d$) bị dùng hết. Chẳng hạn, với ImageNet, sẽ không thể có
nhiều hơn 8 tầng tích chập theo cách này.

Ý tưởng then chốt của :citet:`Simonyan.Zisserman.2014` là sử dụng *nhiều* phép tích chập giữa các lần giảm mẫu
qua max-pooling dưới dạng một khối. Họ chủ yếu quan tâm đến việc mạng sâu hay
mạng rộng hoạt động tốt hơn. Chẳng hạn, việc áp dụng liên tiếp hai phép tích chập $3 \times 3$
tác động đến cùng các pixel như một phép tích chập $5 \times 5$ duy nhất. Đồng thời, phép sau sử dụng xấp xỉ
nhiều tham số ($25 \cdot c^2$) như ba phép tích chập $3 \times 3$ ($3 \cdot 9 \cdot c^2$).
Trong một phân tích khá chi tiết, họ chỉ ra rằng các mạng sâu và hẹp vượt trội đáng kể so với các mạng nông tương ứng. Điều này đã đưa deep learning vào hành trình hướng tới các mạng ngày càng sâu hơn, với hơn 100 tầng cho các ứng dụng điển hình.
Xếp chồng các phép tích chập $3 \times 3$
đã trở thành tiêu chuẩn vàng trong các mạng sâu về sau (một quyết định thiết kế chỉ mới được xem xét lại gần đây bởi
:citet:`liu2022convnet`). Do đó, các hiện thực nhanh cho phép tích chập nhỏ đã trở thành yếu tố thiết yếu trên GPU :cite:`lavin2016fast`.

Quay lại VGG: một khối VGG gồm một *chuỗi* các phép tích chập với kernel $3\times3$ và padding bằng 1
(giữ nguyên chiều cao và chiều rộng), theo sau bởi một tầng max-pooling $2 \times 2$ với stride bằng 2
(giảm một nửa chiều cao và chiều rộng sau mỗi khối).
Trong mã bên dưới, chúng ta định nghĩa một hàm có tên `vgg_block`
để hiện thực một khối VGG.

Hàm bên dưới nhận hai đối số,
tương ứng với số tầng tích chập `num_convs`
và số kênh đầu ra `num_channels`.

```{.python .input  n=2}
%%tab mxnet
def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input  n=3}
%%tab pytorch
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```{.python .input  n=4}
%%tab tensorflow
def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(
            tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                   padding='same', activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
%%tab jax
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv(out_channels, kernel_size=(3, 3), padding=(1, 1)))
        layers.append(nn.relu)
    layers.append(lambda x: nn.max_pool(x, window_shape=(2, 2), strides=(2, 2)))
    return nn.Sequential(layers)
```

## [**Mạng VGG**]
:label:`subsec_vgg-network`

Tương tự AlexNet và LeNet,
Mạng VGG có thể được chia thành hai phần:
phần thứ nhất chủ yếu gồm các tầng tích chập và pooling
và phần thứ hai gồm các tầng fully connected giống hệt các tầng trong AlexNet.
Điểm khác biệt chính là
các tầng tích chập được nhóm thành các phép biến đổi phi tuyến
giữ nguyên số chiều, theo sau bởi một bước giảm độ phân giải, như
minh họa trong :numref:`fig_vgg`.

![Từ AlexNet đến VGG. Điểm khác biệt chính là VGG gồm các khối tầng, trong khi các tầng của AlexNet đều được thiết kế riêng lẻ.](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

Phần tích chập của mạng kết nối liên tiếp một số khối VGG từ :numref:`fig_vgg` (cũng được định nghĩa trong hàm `vgg_block`).
Việc nhóm các phép tích chập này là một mẫu đã
gần như không thay đổi trong thập kỷ qua, mặc dù lựa chọn cụ thể về
các phép toán đã trải qua những thay đổi đáng kể.
Biến `arch` gồm một danh sách các tuple (mỗi tuple cho một khối),
trong đó mỗi tuple chứa hai giá trị: số tầng tích chập
và số kênh đầu ra,
chính xác là các đối số cần thiết để gọi
hàm `vgg_block`. Vì vậy, VGG định nghĩa một *họ* mạng thay vì chỉ
một hiện thân cụ thể. Để xây dựng một mạng cụ thể, chúng ta chỉ cần lặp qua `arch` để ghép các khối.

```{.python .input  n=5}
%%tab pytorch, mxnet, tensorflow
class VGG(d2l.Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            conv_blks = []
            for (num_convs, out_channels) in arch:
                conv_blks.append(vgg_block(num_convs, out_channels))
            self.net = nn.Sequential(
                *conv_blks, nn.Flatten(),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(num_classes))
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(
                tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes)]))
```

```{.python .input  n=5}
%%tab jax
class VGG(d2l.Classifier):
    arch: list
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        conv_blks = []
        for (num_convs, out_channels) in self.arch:
            conv_blks.append(vgg_block(num_convs, out_channels))

        self.net = nn.Sequential([
            *conv_blks,
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(4096), nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(4096), nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(self.num_classes)])
```

Mạng VGG nguyên bản có năm khối tích chập,
trong đó hai khối đầu tiên mỗi khối có một tầng tích chập
và ba khối sau mỗi khối chứa hai tầng tích chập.
Khối đầu tiên có 64 kênh đầu ra
và mỗi khối tiếp theo tăng gấp đôi số kênh đầu ra,
cho đến khi con số đó đạt 512.
Vì mạng này sử dụng tám tầng tích chập
và ba tầng fully connected, nó thường được gọi là VGG-11.

```{.python .input  n=6}
%%tab pytorch, mxnet
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 1, 224, 224))
```

```{.python .input  n=7}
%%tab tensorflow
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 224, 224, 1))
```

```{.python .input}
%%tab jax
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
    training=False).layer_summary((1, 224, 224, 1))
```

Như bạn có thể thấy, chúng ta giảm một nửa chiều cao và chiều rộng ở mỗi khối,
cuối cùng đạt chiều cao và chiều rộng là 7
trước khi làm phẳng các biểu diễn
để xử lý bằng phần fully connected của mạng.
:citet:`Simonyan.Zisserman.2014` đã mô tả một số biến thể khác của VGG.
Thực tế, việc đề xuất các *họ* mạng với
sự đánh đổi tốc độ--độ chính xác khác nhau khi giới thiệu một kiến trúc mới đã trở thành thông lệ.

## Huấn luyện

[**Vì VGG-11 đòi hỏi tính toán nhiều hơn AlexNet,
chúng ta xây dựng một mạng có số lượng kênh nhỏ hơn.**]
Điều này là quá đủ để huấn luyện trên Fashion-MNIST.
Quy trình [**huấn luyện mô hình**] tương tự như của AlexNet trong :numref:`sec_alexnet`.
Một lần nữa, hãy quan sát sự tương đồng sát sao giữa loss trên tập validation và tập huấn luyện,
cho thấy chỉ có một lượng overfitting nhỏ.

```{.python .input  n=8}
%%tab mxnet, pytorch, jax
model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input  n=9}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
    trainer.fit(model, data)
```

## Tóm tắt

Có thể lập luận rằng VGG là mạng nơ-ron tích chập hiện đại thực sự đầu tiên. Mặc dù AlexNet đã giới thiệu nhiều thành phần làm cho deep learning hiệu quả ở quy mô lớn, chính VGG được cho là đã đưa vào các đặc tính then chốt như các khối gồm nhiều phép tích chập và ưu tiên các mạng sâu, hẹp. Đây cũng là mạng đầu tiên thực sự là cả một họ các mô hình được tham số hóa tương tự, mang lại cho người thực hành nhiều sự đánh đổi giữa độ phức tạp và tốc độ. Đây cũng là nơi các framework deep learning hiện đại tỏa sáng. Không còn cần phải tạo các tệp cấu hình XML để chỉ định một mạng, mà thay vào đó có thể lắp ghép các mạng nói trên bằng mã Python đơn giản.

Gần đây hơn, ParNet :cite:`Goyal.Bochkovskiy.Deng.ea.2021` đã chứng minh rằng có thể đạt hiệu năng cạnh tranh bằng một kiến trúc nông hơn nhiều thông qua số lượng lớn các phép tính song song. Đây là một phát triển đầy hứng khởi và có hy vọng rằng nó sẽ ảnh hưởng đến thiết kế kiến trúc trong tương lai. Tuy nhiên, trong phần còn lại của chương, chúng ta sẽ đi theo con đường tiến bộ khoa học trong thập kỷ qua.

## Bài tập


1. So với AlexNet, VGG chậm hơn nhiều về mặt tính toán, đồng thời cũng cần nhiều bộ nhớ GPU hơn.
    1. So sánh số lượng tham số cần thiết cho AlexNet và VGG.
    1. So sánh số lượng phép toán dấu phẩy động được dùng trong các tầng tích chập và trong các tầng fully connected.
    1. Làm thế nào bạn có thể giảm chi phí tính toán do các tầng fully connected tạo ra?
1. Khi hiển thị các kích thước liên kết với các tầng khác nhau của mạng, chúng ta chỉ thấy thông tin liên kết với tám khối (cộng thêm một số phép biến đổi phụ trợ), mặc dù mạng có 11 tầng. Ba tầng còn lại đã đi đâu?
1. Sử dụng Bảng 1 trong bài báo VGG :cite:`Simonyan.Zisserman.2014` để xây dựng các mô hình phổ biến khác, chẳng hạn VGG-16 hoặc VGG-19.
1. Việc tăng độ phân giải của Fashion-MNIST lên tám lần từ kích thước $28 \times 28$ thành $224 \times 224$ là rất lãng phí. Hãy thử sửa đổi kiến trúc mạng và phép chuyển đổi độ phân giải, chẳng hạn thành 56 hoặc 84 kích thước cho đầu vào. Bạn có thể làm điều đó mà không làm giảm độ chính xác của mạng không? Hãy tham khảo bài báo VGG :cite:`Simonyan.Zisserman.2014` để có ý tưởng về việc thêm nhiều phép phi tuyến hơn trước khi giảm mẫu.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/77)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/78)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/277)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18002)
:end_tab: