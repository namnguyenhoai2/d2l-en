```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Mạng Đa Nhánh  (GoogLeNet)
:label:`sec_googlenet`

Năm 2014, *GoogLeNet*
đã chiến thắng ImageNet Challenge :cite:`Szegedy.Liu.Jia.ea.2015`, sử dụng một kiến trúc
kết hợp các ưu điểm của NiN :cite:`Lin.Chen.Yan.2013`, các block lặp lại :cite:`Simonyan.Zisserman.2014`,
và sự kết hợp của các kernel tích chập. Đây cũng được cho là mạng đầu tiên thể hiện sự phân tách rõ ràng giữa stem (nạp dữ liệu), body (xử lý dữ liệu) và head (dự đoán) trong một CNN. Mẫu thiết kế này vẫn được duy trì kể từ đó trong việc thiết kế các mạng sâu: *stem* được tạo bởi hai hoặc ba phép tích chập đầu tiên hoạt động trên ảnh. Chúng trích xuất các đặc trưng cấp thấp từ những ảnh đầu vào. Tiếp theo là *body* gồm các block tích chập. Cuối cùng, *head* ánh xạ các đặc trưng thu được đến bài toán phân loại, phân đoạn, phát hiện hoặc theo dõi cần giải quyết.

Đóng góp then chốt của GoogLeNet là thiết kế phần body của mạng. Nó giải quyết bài toán lựa chọn
kernel tích chập một cách khéo léo. Trong khi các công trình khác cố gắng xác định phép tích chập nào, từ $1 \times 1$ đến $11 \times 11$, sẽ là tốt nhất, GoogLeNet chỉ đơn giản *nối* các phép tích chập đa nhánh.
Trong phần sau, chúng ta sẽ giới thiệu một phiên bản GoogLeNet được đơn giản hóa đôi chút: thiết kế ban đầu bao gồm một số thủ thuật để ổn định quá trình huấn luyện thông qua các hàm mất mát trung gian, được áp dụng tại nhiều tầng của mạng.
Chúng không còn cần thiết nhờ sự xuất hiện của các thuật toán huấn luyện được cải tiến.

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
from torch.nn import functional as F
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
from jax import numpy as jnp
import jax
```

## (**Các Block Inception**)

Block tích chập cơ bản trong GoogLeNet được gọi là *block Inception*,
bắt nguồn từ meme "we need to go deeper" trong bộ phim *Inception*.

![Cấu trúc của block Inception.](../img/inception.svg)
:label:`fig_inception`

Như được minh họa trong :numref:`fig_inception`,
block inception gồm bốn nhánh song song.
Ba nhánh đầu tiên sử dụng các tầng tích chập
với kích thước cửa sổ $1\times 1$, $3\times 3$ và $5\times 5$
để trích xuất thông tin ở các kích thước không gian khác nhau.
Hai nhánh ở giữa cũng thêm một phép tích chập $1\times 1$ trên đầu vào
để giảm số lượng kênh, qua đó giảm độ phức tạp của mô hình.
Nhánh thứ tư sử dụng một tầng max-pooling $3\times 3$,
tiếp theo là một tầng tích chập $1\times 1$
để thay đổi số lượng kênh.
Cả bốn nhánh đều sử dụng padding phù hợp để đầu vào và đầu ra có cùng chiều cao và chiều rộng.
Cuối cùng, các đầu ra của mỗi nhánh được nối lại
theo chiều kênh và tạo thành đầu ra của block.
Các hyperparameter thường được tinh chỉnh của block Inception
là số kênh đầu ra trên mỗi tầng, tức là cách phân bổ năng lực giữa các phép tích chập có kích thước khác nhau.

```{.python .input}
%%tab mxnet
class Inception(nn.Block):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # Branch 2
        self.b2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.b2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        # Branch 3
        self.b3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.b3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        # Branch 4
        self.b4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.b4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        b1 = self.b1_1(x)
        b2 = self.b2_2(self.b2_1(x))
        b3 = self.b3_2(self.b3_1(x))
        b4 = self.b4_2(self.b4_1(x))
        return np.concatenate((b1, b2, b3, b4), axis=1)
```

```{.python .input}
%%tab pytorch
class Inception(nn.Module):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)
```

```{.python .input}
%%tab tensorflow
class Inception(tf.keras.Model):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.b1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        self.b2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.b2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',
                                           activation='relu')
        self.b3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.b3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',
                                           activation='relu')
        self.b4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.b4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')

    def call(self, x):
        b1 = self.b1_1(x)
        b2 = self.b2_2(self.b2_1(x))
        b3 = self.b3_2(self.b3_1(x))
        b4 = self.b4_2(self.b4_1(x))
        return tf.keras.layers.Concatenate()([b1, b2, b3, b4])
```

```{.python .input}
%%tab jax
class Inception(nn.Module):
    # `c1`--`c4` are the number of output channels for each branch
    c1: int
    c2: tuple
    c3: tuple
    c4: int

    def setup(self):
        # Branch 1
        self.b1_1 = nn.Conv(self.c1, kernel_size=(1, 1))
        # Branch 2
        self.b2_1 = nn.Conv(self.c2[0], kernel_size=(1, 1))
        self.b2_2 = nn.Conv(self.c2[1], kernel_size=(3, 3), padding='same')
        # Branch 3
        self.b3_1 = nn.Conv(self.c3[0], kernel_size=(1, 1))
        self.b3_2 = nn.Conv(self.c3[1], kernel_size=(5, 5), padding='same')
        # Branch 4
        self.b4_1 = lambda x: nn.max_pool(x, window_shape=(3, 3),
                                          strides=(1, 1), padding='same')
        self.b4_2 = nn.Conv(self.c4, kernel_size=(1, 1))

    def __call__(self, x):
        b1 = nn.relu(self.b1_1(x))
        b2 = nn.relu(self.b2_2(nn.relu(self.b2_1(x))))
        b3 = nn.relu(self.b3_2(nn.relu(self.b3_1(x))))
        b4 = nn.relu(self.b4_2(self.b4_1(x)))
        return jnp.concatenate((b1, b2, b3, b4), axis=-1)
```

Để hiểu trực quan hơn vì sao mạng này hoạt động tốt đến vậy,
hãy xem xét sự kết hợp của các filter.
Chúng khảo sát ảnh với nhiều kích thước filter khác nhau.
Điều này có nghĩa là các chi tiết ở những phạm vi khác nhau
có thể được nhận diện hiệu quả bởi các filter có kích thước khác nhau.
Đồng thời, chúng ta có thể phân bổ số lượng tham số khác nhau
cho các filter khác nhau.


## [**Mô hình GoogLeNet**]

Như minh họa trong :numref:`fig_inception_full`, GoogLeNet sử dụng một chồng gồm tổng cộng 9 block inception, được sắp xếp thành ba nhóm với max-pooling ở giữa,
và global average pooling trong head để tạo ra các dự đoán.
Max-pooling giữa các block inception làm giảm số chiều.
Ở stem, module đầu tiên tương tự AlexNet và LeNet.

![Kiến trúc GoogLeNet.](../img/inception-full-90.svg)
:label:`fig_inception_full`

Bây giờ, chúng ta có thể triển khai GoogLeNet từng phần. Hãy bắt đầu với stem.
Module đầu tiên sử dụng một tầng tích chập $7\times 7$ với 64 kênh.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class GoogleNet(d2l.Classifier):
    def b1(self):
        if tab.selected('mxnet'):
            net = nn.Sequential()
            net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3,
                              activation='relu'),
                    nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            return net
        if tab.selected('pytorch'):
            return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if tab.selected('tensorflow'):
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2,
                                          padding='same')])
```

```{.python .input}
%%tab jax
class GoogleNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10

    def setup(self):
        self.net = nn.Sequential([self.b1(), self.b2(), self.b3(), self.b4(),
                                  self.b5(), nn.Dense(self.num_classes)])

    def b1(self):
        return nn.Sequential([
                nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
                nn.relu,
                lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2),
                                      padding='same')])
```

Module thứ hai sử dụng hai tầng tích chập:
đầu tiên là một tầng tích chập $1\times 1$ với 64 kênh,
tiếp theo là một tầng tích chập $3\times 3$ làm số lượng kênh tăng gấp ba. Điều này tương ứng với nhánh thứ hai trong block Inception và hoàn tất thiết kế của body. Tại thời điểm này, chúng ta có 192 kênh.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b2(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
               nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 1, activation='relu'),
            tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
    if tab.selected('jax'):
        return nn.Sequential([nn.Conv(64, kernel_size=(1, 1)),
                              nn.relu,
                              nn.Conv(192, kernel_size=(3, 3), padding='same'),
                              nn.relu,
                              lambda x: nn.max_pool(x, window_shape=(3, 3),
                                                    strides=(2, 2),
                                                    padding='same')])
```

Module thứ ba nối tiếp hai block Inception hoàn chỉnh.
Số kênh đầu ra của block Inception thứ nhất là
$64+128+32+32=256$. Điều này tương ứng với
tỷ lệ số kênh đầu ra
giữa bốn nhánh là $2:4:1:1$. Để đạt được điều này, trước hết chúng ta giảm số chiều đầu vào
đi $\frac{1}{2}$ và $\frac{1}{12}$ lần lượt ở nhánh thứ hai và thứ ba
để đạt được lần lượt $96 = 192/2$ và $16 = 192/12$ kênh.

Số kênh đầu ra của block Inception thứ hai
được tăng lên thành $128+192+96+64=480$, tạo ra tỷ lệ $128:192:96:64 = 4:6:3:2$. Như trước,
chúng ta cần giảm số chiều trung gian trong kênh thứ hai và thứ ba. Một
tỷ lệ $\frac{1}{2}$ và $\frac{1}{8}$ lần lượt là đủ, cho ra $128$ và $32$ kênh
tương ứng. Điều này được thể hiện qua các đối số của các constructor block `Inception` sau đây.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b3(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(64, (96, 128), (16, 32), 32),
               Inception(128, (128, 192), (32, 96), 64),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(64, (96, 128), (16, 32), 32),
                             Inception(128, (128, 192), (32, 96), 64),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.models.Sequential([
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
    if tab.selected('jax'):
        return nn.Sequential([Inception(64, (96, 128), (16, 32), 32),
                              Inception(128, (128, 192), (32, 96), 64),
                              lambda x: nn.max_pool(x, window_shape=(3, 3),
                                                    strides=(2, 2),
                                                    padding='same')])
```

Module thứ tư phức tạp hơn.
Nó nối tiếp năm block Inception,
và chúng lần lượt có $192+208+48+64=512$, $160+224+64+64=512$,
$128+256+64+64=512$, $112+288+64+64=528$,
và $256+320+128+128=832$ kênh đầu ra.
Số lượng kênh được phân bổ cho các nhánh này tương tự
như trong module thứ ba:
nhánh thứ hai có tầng tích chập $3\times 3$
cho ra số kênh lớn nhất,
tiếp theo là nhánh thứ nhất chỉ có tầng tích chập $1\times 1$,
nhánh thứ ba có tầng tích chập $5\times 5$,
và nhánh thứ tư có tầng max-pooling $3\times 3$.
Nhánh thứ hai và thứ ba trước tiên sẽ giảm
số lượng kênh theo tỷ lệ.
Các tỷ lệ này hơi khác nhau giữa các block Inception khác nhau.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b4(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(192, (96, 208), (16, 48), 64),
                Inception(160, (112, 224), (24, 64), 64),
                Inception(128, (128, 256), (24, 64), 64),
                Inception(112, (144, 288), (32, 64), 64),
                Inception(256, (160, 320), (32, 128), 128),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(192, (96, 208), (16, 48), 64),
                             Inception(160, (112, 224), (24, 64), 64),
                             Inception(128, (128, 256), (24, 64), 64),
                             Inception(112, (144, 288), (32, 64), 64),
                             Inception(256, (160, 320), (32, 128), 128),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
    if tab.selected('jax'):
        return nn.Sequential([Inception(192, (96, 208), (16, 48), 64),
                              Inception(160, (112, 224), (24, 64), 64),
                              Inception(128, (128, 256), (24, 64), 64),
                              Inception(112, (144, 288), (32, 64), 64),
                              Inception(256, (160, 320), (32, 128), 128),
                              lambda x: nn.max_pool(x, window_shape=(3, 3),
                                                    strides=(2, 2),
                                                    padding='same')])
```

Module thứ năm có hai block Inception với $256+320+128+128=832$
và $384+384+128+128=1024$ kênh đầu ra.
Số lượng kênh được phân bổ cho mỗi nhánh
giống như trong module thứ ba và thứ tư,
nhưng khác về các giá trị cụ thể.
Cần lưu ý rằng block thứ năm được theo sau bởi tầng đầu ra.
Block này sử dụng tầng global average pooling
để đưa chiều cao và chiều rộng của mỗi kênh về 1, giống như trong NiN.
Cuối cùng, chúng ta chuyển đầu ra thành một mảng hai chiều
rồi đưa qua một tầng kết nối đầy đủ
có số lượng đầu ra bằng số lớp nhãn.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b5(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(256, (160, 320), (32, 128), 128),
                Inception(384, (192, 384), (48, 128), 128),
                nn.GlobalAvgPool2D())
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(256, (160, 320), (32, 128), 128),
                             Inception(384, (192, 384), (48, 128), 128),
                             nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Flatten()])
    if tab.selected('jax'):
        return nn.Sequential([Inception(256, (160, 320), (32, 128), 128),
                              Inception(384, (192, 384), (48, 128), 128),
                              # Flax does not provide a GlobalAvgPool2D layer
                              lambda x: nn.avg_pool(x,
                                                    window_shape=x.shape[1:3],
                                                    strides=x.shape[1:3],
                                                    padding='valid'),
                              lambda x: x.reshape((x.shape[0], -1))])
```

Sau khi đã định nghĩa tất cả các block từ `b1` đến `b5`, việc còn lại chỉ là lắp ghép chúng thành một mạng hoàn chỉnh.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(GoogleNet)
def __init__(self, lr=0.1, num_classes=10):
    super(GoogleNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
                     nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), self.b4(),
                                 self.b5(), nn.LazyLinear(num_classes))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.Sequential([
            self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
            tf.keras.layers.Dense(num_classes)])
```

Mô hình GoogLeNet có độ phức tạp tính toán cao. Hãy lưu ý số lượng lớn các hyperparameter tương đối tùy ý, chẳng hạn số kênh được chọn, số block trước khi giảm số chiều, cách phân chia tương đối năng lực giữa các kênh, v.v. Phần lớn điều này là do
vào thời điểm GoogLeNet được giới thiệu, các công cụ tự động để định nghĩa mạng hoặc khám phá thiết kế
vẫn chưa có sẵn. Chẳng hạn, hiện nay chúng ta xem là hiển nhiên rằng một framework deep learning đủ năng lực có thể tự động suy luận số chiều của các tensor đầu vào. Khi đó, nhiều cấu hình như vậy phải được người thực nghiệm chỉ định rõ ràng, do đó thường làm chậm quá trình thử nghiệm tích cực. Hơn nữa, các công cụ cần thiết cho việc khám phá tự động vẫn đang trong giai đoạn biến động và các thử nghiệm ban đầu chủ yếu là những quá trình khám phá brute-force tốn kém, thuật toán di truyền và các chiến lược tương tự.

Hiện tại, thay đổi duy nhất chúng ta sẽ thực hiện là
[**giảm chiều cao và chiều rộng đầu vào từ 224 xuống 96
để có thời gian huấn luyện hợp lý trên Fashion-MNIST.**]
Điều này đơn giản hóa việc tính toán. Hãy xem xét
các thay đổi về shape của đầu ra giữa các module khác nhau.

```{.python .input}
%%tab mxnet, pytorch
model = GoogleNet().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow, jax
model = GoogleNet().layer_summary((1, 96, 96, 1))
```

## [**Huấn luyện**]

Như trước, chúng ta huấn luyện mô hình bằng tập dữ liệu Fashion-MNIST.
Chúng ta chuyển đổi độ phân giải của nó thành $96 \times 96$ pixel
trước khi gọi quy trình huấn luyện.

```{.python .input}
%%tab mxnet, pytorch, jax
model = GoogleNet(lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = GoogleNet(lr=0.01)
    trainer.fit(model, data)
```

## Thảo luận

Một đặc điểm then chốt của GoogLeNet là nó thực sự *ít tốn kém* hơn về mặt tính toán so với các mô hình tiền nhiệm
trong khi đồng thời cho độ chính xác được cải thiện. Điều này đánh dấu sự khởi đầu của một cách thiết kế
mạng có chủ đích hơn nhiều, cân bằng chi phí đánh giá mạng với việc giảm lỗi. Nó cũng đánh dấu sự khởi đầu của thử nghiệm ở cấp độ block với các hyperparameter thiết kế mạng, dù vào thời điểm đó mọi thứ đều hoàn toàn thủ công. Chúng ta sẽ xem lại chủ đề này trong :numref:`sec_cnn-design` khi thảo luận các chiến lược khám phá cấu trúc mạng.

Trong các phần tiếp theo, chúng ta sẽ gặp một số lựa chọn thiết kế (ví dụ: batch normalization, residual connections và channel grouping) cho phép cải thiện mạng đáng kể. Hiện tại, bạn có thể tự hào vì đã triển khai được thứ được cho là CNN hiện đại thực sự đầu tiên.

## Bài tập

1. GoogLeNet thành công đến mức đã trải qua nhiều lần lặp lại, từng bước cải thiện tốc độ và độ chính xác. Hãy thử triển khai và chạy một số phiên bản trong đó. Chúng bao gồm:
    1. Thêm một tầng batch normalization :cite:`Ioffe.Szegedy.2015`, như được mô tả sau trong :numref:`sec_batch_norm`.
    1. Điều chỉnh Inception block (độ rộng, lựa chọn và thứ tự các phép tích chập), như được mô tả trong :citet:`Szegedy.Vanhoucke.Ioffe.ea.2016`.
    1. Sử dụng label smoothing để regularization mô hình, như được mô tả trong :citet:`Szegedy.Vanhoucke.Ioffe.ea.2016`.
    1. Tiếp tục điều chỉnh Inception block bằng cách thêm residual connection :cite:`Szegedy.Ioffe.Vanhoucke.ea.2017`, như được mô tả sau trong :numref:`sec_resnet`.
1. Kích thước ảnh tối thiểu cần thiết để GoogLeNet hoạt động là bao nhiêu?
1. Bạn có thể thiết kế một biến thể của GoogLeNet hoạt động với độ phân giải gốc $28 \times 28$ pixel của Fashion-MNIST không? Bạn sẽ cần thay đổi stem, body và head của mạng như thế nào, nếu có cần thay đổi?
1. So sánh kích thước tham số mô hình của AlexNet, VGG, NiN và GoogLeNet. Hai kiến trúc mạng sau giảm đáng kể kích thước tham số mô hình bằng cách nào?
1. So sánh lượng tính toán cần thiết trong GoogLeNet và AlexNet. Điều này ảnh hưởng như thế nào đến việc thiết kế một chip accelerator, chẳng hạn về kích thước bộ nhớ, băng thông bộ nhớ, kích thước cache, lượng tính toán và lợi ích của các phép toán chuyên biệt?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/81)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/82)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/316)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18004)
:end_tab: