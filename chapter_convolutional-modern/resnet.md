```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Mạng phần dư (ResNet) và ResNeXt
:label:`sec_resnet`

Khi thiết kế các mạng ngày càng sâu hơn, việc hiểu cách thêm các lớp có thể làm tăng độ phức tạp và khả năng biểu đạt của mạng trở nên cấp thiết.
Quan trọng hơn nữa là khả năng thiết kế các mạng mà việc thêm lớp khiến mạng có khả năng biểu đạt cao hơn một cách rõ rệt, thay vì chỉ khác đi.
Để đạt được một số tiến bộ, chúng ta cần một chút toán học.

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

## Các lớp hàm

Xét $\mathcal{F}$, lớp các hàm mà một kiến trúc mạng cụ thể (cùng với learning rate và các thiết lập hyperparameter khác) có thể đạt tới.
Tức là, với mọi $f \in \mathcal{F}$, tồn tại một tập tham số nào đó (ví dụ: trọng số và bias) có thể thu được thông qua quá trình huấn luyện trên một dataset phù hợp.
Hãy giả sử rằng $f^*$ là hàm "sự thật" mà chúng ta thực sự muốn tìm.
Nếu nó nằm trong $\mathcal{F}$, chúng ta đang ở trong tình thế thuận lợi, nhưng thông thường sẽ không may mắn đến vậy.
Thay vào đó, chúng ta sẽ cố gắng tìm một $f^*_\mathcal{F}$ là lựa chọn tốt nhất của mình trong $\mathcal{F}$.
Ví dụ,
với một dataset có đặc trưng $\mathbf{X}$
và nhãn $\mathbf{y}$,
chúng ta có thể cố gắng tìm nó bằng cách giải bài toán tối ưu hóa sau:

$$f^*_\mathcal{F} \stackrel{\textrm{def}}{=} \mathop{\mathrm{argmin}}_f L(\mathbf{X}, \mathbf{y}, f) \textrm{ subject to } f \in \mathcal{F}.$$

Chúng ta biết rằng regularization :cite:`tikhonov1977solutions,morozov2012methods` có thể kiểm soát độ phức tạp của $\mathcal{F}$
và đạt được tính nhất quán, do đó tập dữ liệu huấn luyện lớn hơn
thường dẫn đến $f^*_\mathcal{F}$ tốt hơn.
Sẽ hợp lý khi giả sử rằng nếu chúng ta thiết kế một kiến trúc khác và mạnh hơn $\mathcal{F}'$ thì sẽ đạt được kết quả tốt hơn. Nói cách khác, chúng ta kỳ vọng rằng $f^*_{\mathcal{F}'}$ sẽ "tốt hơn" $f^*_{\mathcal{F}}$. Tuy nhiên, nếu $\mathcal{F} \not\subseteq \mathcal{F}'$ thì không có gì đảm bảo điều này sẽ xảy ra. Thực tế, $f^*_{\mathcal{F}'}$ hoàn toàn có thể tệ hơn.
Như minh họa trong :numref:`fig_functionclasses`,
đối với các lớp hàm không lồng nhau, một lớp hàm lớn hơn không phải lúc nào cũng tiến gần hơn đến hàm "sự thật" $f^*$. Ví dụ,
ở bên trái của :numref:`fig_functionclasses`,
mặc dù $\mathcal{F}_3$ gần $f^*$ hơn $\mathcal{F}_1$, $\mathcal{F}_6$ lại đi xa hơn và không có gì đảm bảo rằng việc tiếp tục tăng độ phức tạp có thể giảm khoảng cách tới $f^*$.
Với các lớp hàm lồng nhau,
trong đó $\mathcal{F}_1 \subseteq \cdots \subseteq \mathcal{F}_6$
ở bên phải của :numref:`fig_functionclasses`,
chúng ta có thể tránh được vấn đề nói trên của các lớp hàm không lồng nhau.


![Đối với các lớp hàm không lồng nhau, một lớp hàm lớn hơn (được biểu thị bằng diện tích) không đảm bảo rằng chúng ta sẽ tiến gần hơn đến hàm "sự thật" ($\mathit{f}^*$). Điều này không xảy ra với các lớp hàm lồng nhau.](../img/functionclasses.svg)
:label:`fig_functionclasses`

Do đó,
chỉ khi các lớp hàm lớn hơn chứa các lớp nhỏ hơn thì chúng ta mới được đảm bảo rằng việc tăng chúng sẽ làm tăng nghiêm ngặt khả năng biểu đạt của mạng.
Đối với mạng nơ-ron sâu,
nếu chúng ta có thể
huấn luyện lớp mới được thêm vào thành một hàm đồng nhất $f(\mathbf{x}) = \mathbf{x}$, mô hình mới sẽ hiệu quả không kém mô hình ban đầu. Vì mô hình mới có thể tìm được một nghiệm tốt hơn để khớp với dataset huấn luyện, lớp được thêm vào có thể giúp giảm lỗi huấn luyện dễ dàng hơn.

Đây là câu hỏi mà :citet:`He.Zhang.Ren.ea.2016` đã xem xét khi nghiên cứu các mô hình computer vision rất sâu.
Cốt lõi của *mạng phần dư* (*ResNet*) mà họ đề xuất là ý tưởng rằng mỗi lớp bổ sung cần
dễ dàng hơn
bao gồm hàm đồng nhất như một trong các phần tử của nó.
Các cân nhắc này khá sâu sắc nhưng đã dẫn đến một
giải pháp đơn giản đáng ngạc nhiên, đó là *khối phần dư*.
Nhờ đó, ResNet đã chiến thắng ImageNet Large Scale Visual Recognition Challenge vào năm 2015. Thiết kế này có ảnh hưởng sâu rộng đến cách
xây dựng các mạng nơ-ron sâu. Ví dụ, các khối phần dư đã được thêm vào recurrent network :cite:`prakash2016neural,kim2017residual`. Tương tự, Transformer :cite:`Vaswani.Shazeer.Parmar.ea.2017` sử dụng chúng để xếp chồng hiệu quả nhiều lớp mạng. Chúng cũng được dùng trong graph neural network :cite:`Kipf.Welling.2016` và, như một khái niệm cơ bản, đã được sử dụng rộng rãi trong computer vision :cite:`Redmon.Farhadi.2018,Ren.He.Girshick.ea.2015`. 
Lưu ý rằng mạng phần dư xuất hiện sau highway network :cite:`srivastava2015highway`, vốn có chung một phần động cơ nhưng không có tham số hóa thanh lịch xoay quanh hàm đồng nhất.

## (**Khối phần dư**)
:label:`subsec_residual-blks`

Hãy tập trung vào một phần cục bộ của mạng nơ-ron, như minh họa trong :numref:`fig_residual_block`. Ký hiệu đầu vào là $\mathbf{x}$.
Chúng ta giả sử rằng $f(\mathbf{x})$, ánh xạ nền tảng mong muốn mà ta muốn thu được bằng học, sẽ được dùng làm đầu vào cho hàm kích hoạt ở phía trên.
Ở bên trái,
phần nằm trong hộp nét đứt
phải trực tiếp học $f(\mathbf{x})$.
Ở bên phải,
phần nằm trong hộp nét đứt
cần
học *ánh xạ phần dư* $g(\mathbf{x}) = f(\mathbf{x}) - \mathbf{x}$,
đây chính là nguồn gốc tên gọi của khối phần dư.
Nếu ánh xạ đồng nhất $f(\mathbf{x}) = \mathbf{x}$ là ánh xạ nền tảng mong muốn,
thì ánh xạ phần dư trở thành $g(\mathbf{x}) = 0$ và do đó dễ học hơn:
chúng ta chỉ cần đưa trọng số và bias
của
lớp trọng số phía trên (ví dụ: fully connected layer và convolutional layer)
trong hộp nét đứt
về 0.
Hình bên phải minh họa *khối phần dư* của ResNet,
trong đó đường liền mang đầu vào lớp
$\mathbf{x}$ đến toán tử cộng
được gọi là *kết nối phần dư* (hoặc *kết nối tắt*).
Với các khối phần dư, đầu vào có thể
lan truyền thuận nhanh hơn qua các kết nối phần dư giữa các lớp.
Thực tế,
khối phần dư
có thể được xem là
một trường hợp đặc biệt của khối Inception đa nhánh:
nó có hai nhánh,
một trong số đó là ánh xạ đồng nhất.

![Trong một khối thông thường (bên trái), phần nằm trong hộp nét đứt phải trực tiếp học ánh xạ $\mathit{f}(\mathbf{x})$. Trong một khối phần dư (bên phải), phần nằm trong hộp nét đứt cần học ánh xạ phần dư $\mathit{g}(\mathbf{x}) = \mathit{f}(\mathbf{x}) - \mathbf{x}$, giúp ánh xạ đồng nhất $\mathit{f}(\mathbf{x}) = \mathbf{x}$ dễ học hơn.](../img/residual-block.svg)
:label:`fig_residual_block`


ResNet sử dụng thiết kế các convolutional layer $3\times 3$ đầy đủ của VGG. Khối phần dư có hai convolutional layer $3\times 3$ với cùng số lượng output channel. Mỗi convolutional layer được theo sau bởi một batch normalization layer và một hàm kích hoạt ReLU. Sau đó, chúng ta bỏ qua hai phép toán convolution này và cộng trực tiếp đầu vào trước hàm kích hoạt ReLU cuối cùng.
Kiểu thiết kế này yêu cầu đầu ra của hai convolutional layer phải có cùng shape với đầu vào để chúng có thể được cộng lại. Nếu muốn thay đổi số lượng channel, chúng ta cần thêm một convolutional layer $1\times 1$ để biến đổi đầu vào thành shape mong muốn cho phép toán cộng. Hãy xem đoạn code bên dưới.

```{.python .input}
%%tab mxnet
class Residual(nn.Block):  #@save
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return npx.relu(Y + X)
```

```{.python .input}
%%tab pytorch
class Residual(nn.Module):  #@save
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

```{.python .input}
%%tab tensorflow
class Residual(tf.keras.Model):  #@save
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(num_channels, padding='same',
                                            kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                            padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                                                strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)
```

```{.python .input}
%%tab jax
class Residual(nn.Module):  #@save
    """The Residual block of ResNet models."""
    num_channels: int
    use_1x1conv: bool = False
    strides: tuple = (1, 1)
    training: bool = True

    def setup(self):
        self.conv1 = nn.Conv(self.num_channels, kernel_size=(3, 3),
                             padding='same', strides=self.strides)
        self.conv2 = nn.Conv(self.num_channels, kernel_size=(3, 3),
                             padding='same')
        if self.use_1x1conv:
            self.conv3 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                                 strides=self.strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm(not self.training)
        self.bn2 = nn.BatchNorm(not self.training)

    def __call__(self, X):
        Y = nn.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return nn.relu(Y)
```

Đoạn code này tạo ra hai loại mạng: một loại cộng đầu vào vào đầu ra trước khi áp dụng phi tuyến ReLU bất cứ khi nào `use_1x1conv=False`; và một loại điều chỉnh channel và độ phân giải bằng convolution $1 \times 1$ trước khi cộng. :numref:`fig_resnet_block` minh họa điều này.

![Khối ResNet có và không có convolution $1 \times 1$, phép toán này biến đổi đầu vào thành shape mong muốn cho phép toán cộng.](../img/resnet-block.svg)
:label:`fig_resnet_block`

Bây giờ, hãy xem xét [**trường hợp đầu vào và đầu ra có cùng shape**], khi không cần convolution $1 \times 1$.

```{.python .input}
%%tab mxnet, pytorch
if tab.selected('mxnet'):
    blk = Residual(3)
    blk.initialize()
if tab.selected('pytorch'):
    blk = Residual(3)
X = d2l.randn(4, 3, 6, 6)
blk(X).shape
```

```{.python .input}
%%tab tensorflow
blk = Residual(3)
X = d2l.normal((4, 6, 6, 3))
Y = blk(X)
Y.shape
```

```{.python .input}
%%tab jax
blk = Residual(3)
X = jax.random.normal(d2l.get_key(), (4, 6, 6, 3))
blk.init_with_output(d2l.get_key(), X)[0].shape
```

Chúng ta cũng có lựa chọn [**giảm một nửa chiều cao và chiều rộng đầu ra trong khi tăng số lượng output channel**].
Trong trường hợp này, chúng ta sử dụng convolution $1 \times 1$ thông qua `use_1x1conv=True`. Điều này hữu ích ở đầu mỗi khối ResNet để giảm số chiều không gian thông qua `strides=2`.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
blk = Residual(6, use_1x1conv=True, strides=2)
if tab.selected('mxnet'):
    blk.initialize()
blk(X).shape
```

```{.python .input}
%%tab jax
blk = Residual(6, use_1x1conv=True, strides=(2, 2))
blk.init_with_output(d2l.get_key(), X)[0].shape
```

## [**Mô hình ResNet**]

Hai lớp đầu tiên của ResNet giống với GoogLeNet mà chúng ta đã mô tả trước đó: convolutional layer $7\times 7$ với 64 output channel và stride bằng 2, theo sau là max-pooling layer $3\times 3$ với stride bằng 2. Điểm khác biệt là batch normalization layer được thêm sau mỗi convolutional layer trong ResNet.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class ResNet(d2l.Classifier):
    def b1(self):
        if tab.selected('mxnet'):
            net = nn.Sequential()
            net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
                    nn.BatchNorm(), nn.Activation('relu'),
                    nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            return net
        if tab.selected('pytorch'):
            return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                nn.LazyBatchNorm2d(), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if tab.selected('tensorflow'):
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, kernel_size=7, strides=2,
                                       padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2,
                                          padding='same')])
```

```{.python .input}
%%tab jax
class ResNet(d2l.Classifier):
    arch: tuple
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def b1(self):
        return nn.Sequential([
            nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
            nn.BatchNorm(not self.training), nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2),
                                  padding='same')])
```

GoogLeNet sử dụng bốn mô-đun được tạo thành từ các khối Inception.
Tuy nhiên, ResNet sử dụng bốn mô-đun được tạo thành từ các khối residual, trong đó mỗi mô-đun sử dụng một số khối residual có cùng số kênh đầu ra.
Số kênh trong mô-đun đầu tiên bằng số kênh đầu vào. Vì một lớp max-pooling với stride bằng 2 đã được sử dụng, nên không cần giảm chiều cao và chiều rộng. Trong khối residual đầu tiên của mỗi mô-đun tiếp theo, số kênh được tăng gấp đôi so với mô-đun trước đó, còn chiều cao và chiều rộng được giảm một nửa.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels))
    return nn.Sequential(*blk)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = tf.keras.models.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

```{.python .input}
%%tab jax
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(num_channels, use_1x1conv=True,
                                strides=(2, 2), training=self.training))
        else:
            blk.append(Residual(num_channels, training=self.training))
    return nn.Sequential(blk)
```

Sau đó, chúng ta thêm tất cả các mô-đun vào ResNet. Ở đây, hai khối residual được dùng cho mỗi mô-đun. Cuối cùng, giống như GoogLeNet, chúng ta thêm một lớp global average pooling, theo sau là đầu ra từ lớp fully connected.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(ResNet)
def __init__(self, arch, lr=0.1, num_classes=10):
    super(ResNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1())
        for i, b in enumerate(arch):
            self.net.add(self.block(*b, first_block=(i==0)))
        self.net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add(self.block(*b, first_block=(i==0)))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(units=num_classes)]))
```

```{.python .input}
# %%tab jax
@d2l.add_to_class(ResNet)
def create_net(self):
    net = nn.Sequential([self.b1()])
    for i, b in enumerate(self.arch):
        net.layers.extend([self.block(*b, first_block=(i==0))])
    net.layers.extend([nn.Sequential([
        # Flax does not provide a GlobalAvg2D layer
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                              strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)])])
    return net
```

Mỗi mô-đun có bốn lớp convolutional (không tính lớp convolutional $1\times 1$). Cùng với lớp convolutional $7\times 7$ đầu tiên và lớp fully connected cuối cùng, tổng cộng có 18 lớp. Do đó, mô hình này thường được gọi là ResNet-18.
Bằng cách cấu hình số lượng kênh và khối residual khác nhau trong mô-đun, chúng ta có thể tạo các mô hình ResNet khác nhau, chẳng hạn ResNet-152 sâu hơn với 152 lớp. Mặc dù kiến trúc chính của ResNet tương tự GoogLeNet, cấu trúc của ResNet đơn giản hơn và dễ chỉnh sửa hơn. Tất cả các yếu tố này đã dẫn đến việc ResNet được sử dụng nhanh chóng và rộng rãi. :numref:`fig_resnet18` minh họa ResNet-18 hoàn chỉnh.

![Kiến trúc ResNet-18.](../img/resnet18-90.svg)
:label:`fig_resnet18`

Trước khi huấn luyện ResNet, hãy [**quan sát cách hình dạng đầu vào thay đổi qua các mô-đun khác nhau trong ResNet**]. Giống như trong tất cả các kiến trúc trước đó, độ phân giải giảm trong khi số kênh tăng cho đến khi một lớp global average pooling tổng hợp toàn bộ đặc trưng.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)),
                       lr, num_classes)
```

```{.python .input}
%%tab jax
class ResNet18(ResNet):
    arch: tuple = ((2, 64), (2, 128), (2, 256), (2, 512))
    lr: float = 0.1
    num_classes: int = 10
```

```{.python .input}
%%tab pytorch, mxnet
ResNet18().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow
ResNet18().layer_summary((1, 96, 96, 1))
```

```{.python .input}
%%tab jax
ResNet18(training=False).layer_summary((1, 96, 96, 1))
```

## [**Huấn luyện**]

Chúng ta huấn luyện ResNet trên tập dữ liệu Fashion-MNIST, giống như trước đây. ResNet là một kiến trúc khá mạnh mẽ và linh hoạt. Biểu đồ thể hiện loss huấn luyện và validation cho thấy khoảng cách đáng kể giữa hai đường, trong đó training loss thấp hơn đáng kể. Đối với một mạng có độ linh hoạt này, thêm dữ liệu huấn luyện sẽ mang lại lợi ích rõ rệt trong việc thu hẹp khoảng cách và cải thiện độ chính xác.

```{.python .input}
%%tab mxnet, pytorch, jax
model = ResNet18(lr=0.01)
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
    model = ResNet18(lr=0.01)
    trainer.fit(model, data)
```

## ResNeXt
:label:`subsec_resnext`

Một trong các thách thức khi thiết kế ResNet là sự đánh đổi giữa tính phi tuyến và số chiều trong một block nhất định. Nghĩa là, chúng ta có thể tăng tính phi tuyến bằng cách tăng số lớp hoặc tăng chiều rộng của các convolution. Một chiến lược thay thế là tăng số kênh có thể truyền thông tin giữa các block. Không may là cách sau đi kèm chi phí tăng bậc hai, vì chi phí tính toán để nhận $c_\textrm{i}$ kênh và tạo ra $c_\textrm{o}$ kênh tỉ lệ với $\mathcal{O}(c_\textrm{i} \cdot c_\textrm{o})$ (xem phần thảo luận của chúng ta trong :numref:`sec_channels`). 

Chúng ta có thể lấy cảm hứng từ block Inception trong :numref:`fig_inception`, nơi thông tin đi qua block theo các nhóm riêng biệt. Áp dụng ý tưởng về nhiều nhóm độc lập vào block ResNet trong :numref:`fig_resnet_block` đã dẫn đến thiết kế ResNeXt :cite:`Xie.Girshick.Dollar.ea.2017`.
Khác với tập hợp đa dạng các phép biến đổi trong Inception, 
ResNeXt áp dụng *cùng một* phép biến đổi ở mọi nhánh,
do đó giảm thiểu nhu cầu tinh chỉnh thủ công từng nhánh. 

![Block ResNeXt. Việc sử dụng grouped convolution với $\mathit{g}$ nhóm nhanh hơn $\mathit{g}$ lần so với dense convolution. Đây là một bottleneck residual block khi số kênh trung gian $\mathit{b}$ nhỏ hơn $\mathit{c}$.](../img/resnext-block.svg)
:label:`fig_resnext_block`

Việc chia một convolution từ $c_\textrm{i}$ kênh sang $c_\textrm{o}$ kênh thành một trong $g$ nhóm có kích thước $c_\textrm{i}/g$, tạo ra $g$ đầu ra có kích thước $c_\textrm{o}/g$, được gọi rất phù hợp là *grouped convolution*. Chi phí tính toán (theo tỉ lệ) được giảm từ $\mathcal{O}(c_\textrm{i} \cdot c_\textrm{o})$ xuống $\mathcal{O}(g \cdot (c_\textrm{i}/g) \cdot (c_\textrm{o}/g)) = \mathcal{O}(c_\textrm{i} \cdot c_\textrm{o} / g)$, tức là nhanh hơn $g$ lần. Tuyệt hơn nữa, số tham số cần để tạo đầu ra cũng được giảm từ một ma trận $c_\textrm{i} \times c_\textrm{o}$ xuống $g$ ma trận nhỏ hơn có kích thước $(c_\textrm{i}/g) \times (c_\textrm{o}/g)$, một lần nữa giảm $g$ lần. Trong phần sau, chúng ta giả sử cả $c_\textrm{i}$ và $c_\textrm{o}$ đều chia hết cho $g$. 

Thách thức duy nhất trong thiết kế này là không có thông tin nào được trao đổi giữa $g$ nhóm. Block ResNeXt trong 
:numref:`fig_resnext_block` khắc phục điều này theo hai cách: grouped convolution với kernel $3 \times 3$ được đặt giữa hai convolution $1 \times 1$. Convolution thứ hai thực hiện đồng thời nhiệm vụ thay đổi số kênh trở lại. Lợi ích là chúng ta chỉ phải trả chi phí $\mathcal{O}(c \cdot b)$ cho các kernel $1 \times 1$ và có thể dùng chi phí $\mathcal{O}(b^2 / g)$ cho các kernel $3 \times 3$. Tương tự cách triển khai residual block trong
:numref:`subsec_residual-blks`, kết nối residual được thay thế (và do đó được tổng quát hóa) bằng một convolution $1 \times 1$.

Hình bên phải trong :numref:`fig_resnext_block` cung cấp bản tóm tắt súc tích hơn nhiều về network block thu được. Nó cũng sẽ đóng vai trò quan trọng trong thiết kế các CNN hiện đại tổng quát ở :numref:`sec_cnn-design`. Lưu ý rằng ý tưởng grouped convolution bắt nguồn từ cách triển khai AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012`. Khi phân phối mạng trên hai GPU có bộ nhớ hạn chế, cách triển khai này coi mỗi GPU là một kênh riêng mà không gây tác động xấu nào. 

Cách triển khai sau đây của lớp `ResNeXtBlock` nhận `groups` ($g$) làm đối số, với 
`bot_channels` ($b$) kênh trung gian (bottleneck). Cuối cùng, khi cần giảm chiều cao và chiều rộng của biểu diễn, chúng ta thêm stride bằng $2$ bằng cách đặt `use_1x1conv=True, strides=2`.

```{.python .input}
%%tab mxnet
class ResNeXtBlock(nn.Block):  #@save
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul,
                 use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.Conv2D(bot_channels, kernel_size=1, padding=0,
                               strides=1)
        self.conv2 = nn.Conv2D(bot_channels, kernel_size=3, padding=1, 
                               strides=strides, groups=bot_channels//groups)
        self.conv3 = nn.Conv2D(num_channels, kernel_size=1, padding=0,
                               strides=1)
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
        self.bn3 = nn.BatchNorm()
        if use_1x1conv:
            self.conv4 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
            self.bn4 = nn.BatchNorm()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = npx.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return npx.relu(Y + X)
```

```{.python .input}
%%tab pytorch
class ResNeXtBlock(nn.Module):  #@save
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=bot_channels//groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1, 
                                       stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)
```

```{.python .input}
%%tab tensorflow
class ResNeXtBlock(tf.keras.Model):  #@save
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = tf.keras.layers.Conv2D(bot_channels, 1, strides=1)
        self.conv2 = tf.keras.layers.Conv2D(bot_channels, 3, strides=strides,
                                            padding="same",
                                            groups=bot_channels//groups)
        self.conv3 = tf.keras.layers.Conv2D(num_channels, 1, strides=1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        if use_1x1conv:
            self.conv4 = tf.keras.layers.Conv2D(num_channels, 1,
                                                strides=strides)
            self.bn4 = tf.keras.layers.BatchNormalization()
        else:
            self.conv4 = None

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = tf.keras.activations.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return tf.keras.activations.relu(Y + X)
```

```{.python .input}
%%tab jax
class ResNeXtBlock(nn.Module):  #@save
    """The ResNeXt block."""
    num_channels: int
    groups: int
    bot_mul: int
    use_1x1conv: bool = False
    strides: tuple = (1, 1)
    training: bool = True

    def setup(self):
        bot_channels = int(round(self.num_channels * self.bot_mul))
        self.conv1 = nn.Conv(bot_channels, kernel_size=(1, 1),
                               strides=(1, 1))
        self.conv2 = nn.Conv(bot_channels, kernel_size=(3, 3),
                               strides=self.strides, padding='same',
                               feature_group_count=bot_channels//self.groups)
        self.conv3 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                               strides=(1, 1))
        self.bn1 = nn.BatchNorm(not self.training)
        self.bn2 = nn.BatchNorm(not self.training)
        self.bn3 = nn.BatchNorm(not self.training)
        if self.use_1x1conv:
            self.conv4 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                                       strides=self.strides)
            self.bn4 = nn.BatchNorm(not self.training)
        else:
            self.conv4 = None

    def __call__(self, X):
        Y = nn.relu(self.bn1(self.conv1(X)))
        Y = nn.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return nn.relu(Y + X)
```

Cách sử dụng của nó hoàn toàn tương tự với `ResNetBlock` đã được thảo luận trước đó. Ví dụ, khi sử dụng (`use_1x1conv=False, strides=1`), đầu vào và đầu ra có cùng hình dạng. Hoặc, đặt `use_1x1conv=True, strides=2` sẽ giảm một nửa chiều cao và chiều rộng đầu ra.

```{.python .input}
%%tab mxnet, pytorch
blk = ResNeXtBlock(32, 16, 1)
if tab.selected('mxnet'):
    blk.initialize()
X = d2l.randn(4, 32, 96, 96)
blk(X).shape
```

```{.python .input}
%%tab tensorflow
blk = ResNeXtBlock(32, 16, 1)
X = d2l.normal((4, 96, 96, 32))
Y = blk(X)
Y.shape
```

```{.python .input}
%%tab jax
blk = ResNeXtBlock(32, 16, 1)
X = jnp.zeros((4, 96, 96, 32))
blk.init_with_output(d2l.get_key(), X)[0].shape
```

## Tóm tắt và Thảo luận

Các lớp hàm lồng nhau là mong muốn vì chúng cho phép chúng ta thu được các lớp hàm *mạnh hơn* một cách chặt chẽ, thay vì đồng thời trở nên *khác biệt* một cách tinh vi khi tăng năng lực. Một cách để thực hiện điều này là để các lớp bổ sung chỉ đơn giản truyền đầu vào sang đầu ra. Các kết nối residual cho phép điều đó. Do đó, điều này thay đổi inductive bias từ các hàm đơn giản có dạng $f(\mathbf{x}) = 0$ thành các hàm đơn giản có dạng $f(\mathbf{x}) = \mathbf{x}$. 


Ánh xạ residual có thể học hàm đồng nhất dễ dàng hơn, chẳng hạn bằng cách đưa các tham số trong lớp trọng số về không. Chúng ta có thể huấn luyện một mạng nơ-ron *sâu* hiệu quả nhờ các residual block. Đầu vào có thể lan truyền xuôi nhanh hơn qua các kết nối residual giữa các lớp. Do đó, chúng ta có thể huấn luyện các mạng sâu hơn nhiều. Ví dụ, bài báo ResNet gốc :cite:`He.Zhang.Ren.ea.2016` cho phép lên đến 152 lớp. Một lợi ích khác của mạng residual là nó cho phép chúng ta thêm các lớp, được khởi tạo là hàm đồng nhất, *trong khi* huấn luyện. Xét cho cùng, hành vi mặc định của một lớp là để dữ liệu đi qua mà không thay đổi. Điều này trong một số trường hợp có thể tăng tốc huấn luyện các mạng rất lớn. 

Trước các kết nối residual,
các đường bypass với các đơn vị gating đã được giới thiệu
để huấn luyện hiệu quả các highway network có hơn 100 lớp
:cite:`srivastava2015highway`.
Sử dụng các hàm đồng nhất làm đường bypass,
ResNet hoạt động nổi bật
trên nhiều tác vụ computer vision.
Các kết nối residual đã ảnh hưởng lớn đến thiết kế của các mạng nơ-ron sâu sau này, dù có bản chất convolutional hay tuần tự.
Như chúng ta sẽ giới thiệu sau,
kiến trúc Transformer :cite:`Vaswani.Shazeer.Parmar.ea.2017`
áp dụng các kết nối residual (cùng với các lựa chọn thiết kế khác) và phổ biến
trong những lĩnh vực đa dạng như
ngôn ngữ, thị giác, giọng nói và reinforcement learning.

ResNeXt là một ví dụ về cách thiết kế mạng nơ-ron tích chập đã phát triển theo thời gian: bằng cách sử dụng tính toán tiết kiệm hơn và đánh đổi nó với kích thước của các activation (số lượng kênh), mô hình cho phép tạo ra các mạng nhanh hơn và chính xác hơn với chi phí thấp hơn. Một cách khác để xem grouped convolution là coi các trọng số convolution như một ma trận block-diagonal. Lưu ý rằng có khá nhiều “mẹo” như vậy giúp tạo ra các mạng hiệu quả hơn. Ví dụ, ShiftNet :cite:`wu2018shift` mô phỏng hiệu ứng của một convolution $3 \times 3$ chỉ bằng cách cộng các activation đã dịch chuyển vào các kênh, mang lại độ phức tạp hàm tăng lên mà lần này không tốn bất kỳ chi phí tính toán nào. 

Một đặc điểm chung của các thiết kế mà chúng ta đã thảo luận cho đến nay là việc thiết kế mạng khá thủ công, chủ yếu dựa vào sự khéo léo của người thiết kế để tìm ra các siêu tham số mạng “đúng”. Mặc dù rõ ràng là khả thi, cách này cũng rất tốn thời gian của con người và không có bảo đảm rằng kết quả đạt được là tối ưu theo bất kỳ nghĩa nào. Trong :numref:`sec_cnn-design` chúng ta sẽ thảo luận một số chiến lược để thu được các mạng chất lượng cao theo cách tự động hóa hơn. Cụ thể, chúng ta sẽ xem xét khái niệm *không gian thiết kế mạng* (network design spaces), vốn đã dẫn đến các mô hình RegNetX/Y
:cite:`Radosavovic.Kosaraju.Girshick.ea.2020`.

## Bài tập

1. Những khác biệt chính giữa Inception block trong :numref:`fig_inception` và residual block là gì? Chúng khác nhau như thế nào về tính toán, độ chính xác và các lớp hàm mà chúng có thể biểu diễn?
1. Tham khảo Bảng 1 trong bài báo ResNet :cite:`He.Zhang.Ren.ea.2016` để triển khai các biến thể khác nhau của mạng. 
1. Với các mạng sâu hơn, ResNet giới thiệu kiến trúc “bottleneck” để giảm độ phức tạp của mô hình. Hãy thử triển khai kiến trúc này.
1. Trong các phiên bản ResNet sau này, các tác giả đã thay đổi cấu trúc “convolution, batch normalization, và activation” thành cấu trúc “batch normalization, activation, và convolution”. Hãy tự thực hiện cải tiến này. Xem Hình 1 trong :citet:`He.Zhang.Ren.ea.2016*1` để biết chi tiết.
1. Tại sao chúng ta không thể cứ tăng độ phức tạp của các hàm vô hạn, ngay cả khi các lớp hàm được lồng nhau?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/85)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/86)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/8737)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18006)
:end_tab: