```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Thiết kế Kiến trúc Mạng Tích chập
:label:`sec_cnn-design`

Các phần trước đã đưa chúng ta đi tham quan thiết kế mạng hiện đại cho thị giác máy tính. Điểm chung của tất cả các công trình mà chúng ta đã đề cập là chúng phụ thuộc rất nhiều vào trực giác của các nhà khoa học. Nhiều kiến trúc chịu ảnh hưởng lớn từ sự sáng tạo của con người và ở mức độ thấp hơn nhiều từ việc khám phá có hệ thống không gian thiết kế mà các mạng sâu mang lại. Tuy vậy, cách tiếp cận *network engineering* này đã thành công vượt bậc.

Kể từ khi AlexNet (:numref:`sec_alexnet`)
đánh bại các mô hình thị giác máy tính truyền thống trên ImageNet,
việc xây dựng các mạng rất sâu
bằng cách xếp chồng các khối tích chập, tất cả đều được thiết kế theo cùng một mẫu, đã trở nên phổ biến.
Đặc biệt, các phép tích chập $3 \times 3$ đã được
các mạng VGG phổ biến hóa (:numref:`sec_vgg`).
NiN (:numref:`sec_nin`) cho thấy rằng ngay cả các phép tích chập $1 \times 1$ cũng có thể
hữu ích nhờ bổ sung các phi tuyến tính cục bộ.
Hơn nữa, NiN đã giải quyết bài toán tổng hợp thông tin ở phần đầu của mạng
bằng cách tổng hợp trên mọi vị trí.
GoogLeNet (:numref:`sec_googlenet`) đã thêm nhiều nhánh với độ rộng tích chập khác nhau,
kết hợp các ưu điểm của VGG và NiN trong khối Inception của nó.
ResNet (:numref:`sec_resnet`)
đã thay đổi inductive bias theo hướng ánh xạ đồng nhất (từ $f(x) = 0$). Điều này cho phép tạo ra các mạng rất sâu. Gần một thập kỷ sau, thiết kế ResNet vẫn phổ biến, minh chứng cho chất lượng thiết kế của nó. Cuối cùng, ResNeXt (:numref:`subsec_resnext`) đã bổ sung các phép tích chập theo nhóm, mang lại sự đánh đổi tốt hơn giữa số lượng tham số và tính toán. Là tiền thân của Transformers cho thị giác, Squeeze-and-Excitation Networks (SENets) cho phép truyền thông tin hiệu quả giữa các vị trí
:cite:`Hu.Shen.Sun.2018`. Điều này được thực hiện bằng cách tính một hàm attention toàn cục cho mỗi kênh.

Cho đến nay, chúng ta đã bỏ qua các mạng được tạo ra thông qua *neural architecture search* (NAS) :cite:`zoph2016neural,liu2018darts`. Chúng ta chọn làm vậy vì chi phí của chúng thường rất lớn, dựa vào tìm kiếm vét cạn, thuật toán di truyền, reinforcement learning, hoặc một dạng tối ưu hóa hyperparameter nào khác. Với một không gian tìm kiếm cố định,
NAS sử dụng một chiến lược tìm kiếm để tự động chọn
một kiến trúc dựa trên ước lượng hiệu năng trả về.
Kết quả của NAS
là một thể hiện mạng duy nhất. EfficientNet là một kết quả đáng chú ý của quá trình tìm kiếm này :cite:`tan2019efficientnet`.

Tiếp theo, chúng ta sẽ thảo luận một ý tưởng khá khác với việc tìm kiếm *mạng đơn lẻ tốt nhất*. Ý tưởng này tương đối ít tốn kém về tính toán, đồng thời mang lại các hiểu biết khoa học, và khá hiệu quả xét về chất lượng kết quả. Hãy xem lại chiến lược của :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` nhằm *thiết kế không gian thiết kế mạng*. Chiến lược này kết hợp thế mạnh của thiết kế thủ công và NAS. Nó đạt được điều này bằng cách hoạt động trên *các phân phối mạng* và tối ưu hóa các phân phối để đạt hiệu năng tốt cho toàn bộ họ mạng. Kết quả của phương pháp này là *RegNets*, cụ thể là RegNetX và RegNetY, cùng một loạt nguyên tắc định hướng cho việc thiết kế các CNN hiệu năng cao.

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
```

## Không gian thiết kế AnyNet
:label:`subsec_the-anynet-design-space`

Mô tả dưới đây bám sát lập luận trong :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`, với một số rút gọn để phù hợp với phạm vi của cuốn sách.
Để bắt đầu, chúng ta cần một khuôn mẫu cho họ mạng cần khám phá. Một điểm chung trong các thiết kế của chương này là các mạng gồm một *stem*, một *body* và một *head*. Stem thực hiện xử lý ảnh ban đầu, thường thông qua các phép tích chập với kích thước cửa sổ lớn hơn. Body gồm nhiều khối, thực hiện phần lớn các phép biến đổi cần thiết để đi từ ảnh thô đến biểu diễn đối tượng. Cuối cùng, head chuyển đổi biểu diễn này thành các đầu ra mong muốn, chẳng hạn thông qua một softmax regressor cho phân loại đa lớp.
Đến lượt mình, body gồm nhiều stage, hoạt động trên ảnh ở các độ phân giải giảm dần. Trên thực tế, cả stem và mỗi stage tiếp theo đều giảm độ phân giải không gian xuống còn một phần tư. Cuối cùng, mỗi stage gồm một hoặc nhiều block. Mẫu này phổ biến trong mọi mạng, từ VGG đến ResNeXt. Thật vậy, để thiết kế các mạng AnyNet tổng quát, :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` đã sử dụng block ResNeXt trong :numref:`fig_resnext_block`.


![Không gian thiết kế AnyNet. Các số $(\mathit{c}, \mathit{r})$ dọc theo mỗi mũi tên biểu thị số kênh $c$ và độ phân giải $\mathit{r} \times \mathit{r}$ của ảnh tại điểm đó. Từ trái sang phải: cấu trúc mạng tổng quát gồm stem, body và head; body gồm bốn stage; cấu trúc chi tiết của một stage; hai cấu trúc thay thế cho block, một không có downsampling và một giảm một nửa độ phân giải theo mỗi chiều. Các lựa chọn thiết kế bao gồm độ sâu $\mathit{d_i}$, số kênh đầu ra $\mathit{c_i}$, số nhóm $\mathit{g_i}$, và tỷ lệ bottleneck $\mathit{k_i}$ cho bất kỳ stage $\mathit{i}$ nào.](../img/anynet.svg)
:label:`fig_anynet_full`

Hãy xem xét chi tiết cấu trúc được phác thảo trong :numref:`fig_anynet_full`. Như đã đề cập, một AnyNet gồm stem, body và head. Stem nhận ảnh RGB (3 kênh) làm đầu vào, sử dụng một phép tích chập $3 \times 3$ với stride là $2$, tiếp theo là batch norm, để giảm một nửa độ phân giải từ $r \times r$ xuống $r/2 \times r/2$. Hơn nữa, nó tạo ra $c_0$ kênh làm đầu vào cho body.

Do mạng được thiết kế để hoạt động tốt với ảnh ImageNet có kích thước $224 \times 224 \times 3$, body có nhiệm vụ giảm ảnh này xuống $7 \times 7 \times c_4$ qua 4 stage (nhớ rằng $224 / 2^{1+4} = 7$), mỗi stage cuối cùng có stride là $2$. Cuối cùng, head sử dụng thiết kế hoàn toàn tiêu chuẩn thông qua global average pooling, tương tự NiN (:numref:`sec_nin`), tiếp theo là một fully connected layer để xuất ra vector $n$ chiều cho phân loại $n$ lớp.

Hầu hết các quyết định thiết kế liên quan đều nằm trong body của mạng. Nó tiến hành theo các stage, trong đó mỗi stage gồm cùng loại block ResNeXt như chúng ta đã thảo luận trong :numref:`subsec_resnext`. Thiết kế ở đây một lần nữa hoàn toàn tổng quát: chúng ta bắt đầu bằng một block giảm một nửa độ phân giải bằng stride $2$ (block ngoài cùng bên phải trong :numref:`fig_anynet_full`). Để khớp với điều này, nhánh residual của block ResNeXt cần đi qua một phép tích chập $1 \times 1$. Sau block này là một số lượng biến thiên các block ResNeXt bổ sung, giữ nguyên cả độ phân giải lẫn số kênh. Lưu ý rằng một thực hành thiết kế phổ biến là thêm một bottleneck nhẹ trong thiết kế các block tích chập.
Do đó, với tỷ lệ bottleneck $k_i \geq 1$, chúng ta dành một số kênh, $c_i/k_i$, bên trong mỗi block cho stage $i$ (như các thí nghiệm cho thấy, điều này thực sự không hiệu quả và nên được bỏ qua). Cuối cùng, vì đang xử lý các block ResNeXt, chúng ta cũng cần chọn số nhóm $g_i$ cho các phép tích chập theo nhóm tại stage $i$.

Không gian thiết kế có vẻ tổng quát này vẫn cung cấp cho chúng ta nhiều tham số: chúng ta có thể đặt độ rộng block (số kênh) $c_0, \ldots c_4$, độ sâu (số block) mỗi stage $d_1, \ldots d_4$, các tỷ lệ bottleneck $k_1, \ldots k_4$, và độ rộng nhóm (số nhóm) $g_1, \ldots g_4$.
Tổng cộng có 17 tham số, dẫn đến số lượng cấu hình cần khám phá lớn đến mức không hợp lý. Chúng ta cần một số công cụ để thu hẹp hiệu quả không gian thiết kế khổng lồ này. Đây là lúc vẻ đẹp về mặt khái niệm của các không gian thiết kế phát huy tác dụng. Trước khi làm điều đó, trước hết hãy triển khai thiết kế tổng quát.

```{.python .input}
%%tab mxnet
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        net = nn.Sequential()
        net.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=2),
                nn.BatchNorm(), nn.Activation('relu'))
        return net
```

```{.python .input}
%%tab pytorch
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=3, stride=2, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU())
```

```{.python .input}
%%tab tensorflow
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(num_channels, kernel_size=3, strides=2,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')])
```

```{.python .input}
%%tab jax
class AnyNet(d2l.Classifier):
    arch: tuple
    stem_channels: int
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def stem(self, num_channels):
        return nn.Sequential([
            nn.Conv(num_channels, kernel_size=(3, 3), strides=(2, 2),
                    padding=(1, 1)),
            nn.BatchNorm(not self.training),
            nn.relu
        ])
```

Mỗi stage gồm `depth` block ResNeXt,
trong đó `num_channels` chỉ định độ rộng block.
Lưu ý rằng block đầu tiên giảm một nửa chiều cao và chiều rộng của ảnh đầu vào.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = nn.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(
                num_channels, groups, bot_mul, use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(
                num_channels, num_channels, groups, bot_mul))
    return net
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return nn.Sequential(*blk)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = tf.keras.models.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return net
```

```{.python .input}
%%tab jax
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=(2, 2), training=self.training))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                                        training=self.training))
    return nn.Sequential(blk)
```

Kết hợp stem, body và head của mạng,
chúng ta hoàn tất việc triển khai AnyNet.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(AnyNet)
def __init__(self, arch, stem_channels, lr=0.1, num_classes=10):
    super(AnyNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add_module(f'stage{i+1}', self.stage(*s))
        self.net.add_module('head', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(units=num_classes)]))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(AnyNet)
def create_net(self):
    net = nn.Sequential([self.stem(self.stem_channels)])
    for i, s in enumerate(self.arch):
        net.layers.extend([self.stage(*s)])
    net.layers.extend([nn.Sequential([
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                            strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)])])
    return net
```

## Phân phối và Tham số của Không gian Thiết kế

Như vừa thảo luận trong :numref:`subsec_the-anynet-design-space`, các tham số của một không gian thiết kế là các hyperparameter của mạng trong không gian thiết kế đó.
Xét bài toán xác định các tham số tốt trong không gian thiết kế AnyNet. Chúng ta có thể thử tìm lựa chọn tham số *đơn lẻ tốt nhất* cho một lượng tính toán nhất định (ví dụ: FLOPs và thời gian tính toán). Nếu chỉ cho phép *hai* lựa chọn khả dĩ cho mỗi tham số, chúng ta sẽ phải khám phá $2^{17} = 131072$ tổ hợp để tìm được lời giải tốt nhất. Điều này rõ ràng không khả thi vì chi phí quá cao. Tệ hơn nữa, từ bài tập này chúng ta thực sự không học được gì về cách nên thiết kế một mạng. Lần sau khi thêm, chẳng hạn, một X-stage, một phép toán shift, hoặc tương tự, chúng ta sẽ phải bắt đầu lại từ đầu. Tệ hơn nữa, do tính ngẫu nhiên trong huấn luyện (làm tròn, xáo trộn, lỗi bit), không có hai lần chạy nào có khả năng cho kết quả hoàn toàn giống nhau. Một chiến lược tốt hơn là cố gắng xác định các hướng dẫn tổng quát về cách các lựa chọn tham số nên liên hệ với nhau. Chẳng hạn, tỷ lệ bottleneck, số kênh, block, nhóm, hoặc sự thay đổi của chúng giữa các layer lý tưởng nhất nên được chi phối bởi một tập hợp các quy tắc đơn giản. Cách tiếp cận trong :citet:`radosavovic2019network` dựa trên bốn giả định sau:

1. Chúng tôi giả định rằng các nguyên tắc thiết kế tổng quát thực sự tồn tại, vì vậy nhiều mạng thỏa mãn các yêu cầu này sẽ cho hiệu năng tốt. Do đó, việc xác định một *phân phối* trên các mạng có thể là một chiến lược hợp lý. Nói cách khác, chúng tôi giả định rằng có nhiều cây kim tốt trong đống cỏ khô.
1. Chúng ta không cần huấn luyện mạng đến hội tụ trước khi có thể đánh giá liệu một mạng có tốt hay không. Thay vào đó, chỉ cần sử dụng các kết quả trung gian làm chỉ dẫn đáng tin cậy cho độ chính xác cuối cùng. Việc sử dụng các proxy (xấp xỉ) để tối ưu một mục tiêu được gọi là tối ưu hóa đa độ trung thực (multi-fidelity optimization) :cite:`forrester2007multi`. Do đó, quá trình tối ưu hóa thiết kế được thực hiện dựa trên độ chính xác đạt được chỉ sau vài lượt đi qua tập dữ liệu, giúp giảm đáng kể chi phí. 
1. Các kết quả thu được ở quy mô nhỏ hơn (đối với các mạng nhỏ hơn) có thể khái quát hóa sang các mạng lớn hơn. Do đó, việc tối ưu hóa được thực hiện cho các mạng có cấu trúc tương tự, nhưng có số block ít hơn, ít channel hơn, v.v. Chỉ đến cuối cùng chúng ta mới cần xác minh rằng các mạng được tìm thấy như vậy cũng mang lại hiệu năng tốt ở quy mô lớn. 
1. Các khía cạnh của thiết kế có thể được phân tích thành các nhân tố một cách xấp xỉ, sao cho có thể suy ra ảnh hưởng của chúng lên chất lượng đầu ra tương đối độc lập. Nói cách khác, bài toán tối ưu hóa ở mức độ vừa phải là dễ.

Các giả định này cho phép chúng ta kiểm tra nhiều mạng với chi phí thấp. Cụ thể, chúng ta có thể *lấy mẫu* đồng đều từ không gian cấu hình và đánh giá hiệu năng của chúng. Sau đó, chúng ta có thể đánh giá chất lượng của lựa chọn tham số bằng cách xem xét *phân phối* lỗi/độ chính xác có thể đạt được với các mạng nói trên. Ký hiệu $F(e)$ là hàm phân phối tích lũy (CDF) cho các lỗi do mạng trong một không gian thiết kế nhất định gây ra, được lấy theo phân phối xác suất $p$. Tức là, 

$$F(e, p) \stackrel{\textrm{def}}{=} P_{\textrm{net} \sim p} \{e(\textrm{net}) \leq e\}.$$

Mục tiêu của chúng ta lúc này là tìm một phân phối $p$ trên các *mạng* sao cho hầu hết các mạng có tỷ lệ lỗi rất thấp và miền hỗ trợ của $p$ là gọn. Dĩ nhiên, việc thực hiện điều này một cách chính xác là không khả thi về mặt tính toán. Chúng ta sử dụng một mẫu các mạng $\mathcal{Z} \stackrel{\textrm{def}}{=} \{\textrm{net}_1, \ldots \textrm{net}_n\}$ (với các lỗi tương ứng là $e_1, \ldots, e_n$) từ $p$ và thay vào đó dùng CDF thực nghiệm $\hat{F}(e, \mathcal{Z})$:

$$\hat{F}(e, \mathcal{Z}) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(e_i \leq e).$$

Bất cứ khi nào CDF cho một tập lựa chọn nào đó trội hơn (hoặc bằng) một CDF khác, thì lựa chọn tham số của nó là tốt hơn (hoặc không khác biệt). Theo đó, :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` đã thử nghiệm một tỷ lệ bottleneck dùng chung $k_i = k$ cho mọi stage $i$ của mạng. Điều này loại bỏ ba trong số bốn tham số chi phối tỷ lệ bottleneck. Để đánh giá liệu điều này có ảnh hưởng (tiêu cực) đến hiệu năng hay không, ta có thể lấy mẫu các mạng từ phân phối bị ràng buộc và không bị ràng buộc, rồi so sánh các CDF tương ứng. Hóa ra ràng buộc này hoàn toàn không ảnh hưởng đến độ chính xác của phân phối các mạng, như có thể thấy trong panel đầu tiên của :numref:`fig_regnet-fig`. 
Tương tự, chúng ta có thể chọn cùng một độ rộng group $g_i = g$ xuất hiện ở các stage khác nhau của mạng. Một lần nữa, điều này không ảnh hưởng đến hiệu năng, như có thể thấy trong panel thứ hai của :numref:`fig_regnet-fig`.
Kết hợp cả hai bước sẽ giảm số lượng tham số tự do đi sáu. 

![So sánh các hàm phân phối thực nghiệm của lỗi trong các không gian thiết kế. $\textrm{AnyNet}_\mathit{A}$ là không gian thiết kế gốc; $\textrm{AnyNet}_\mathit{B}$ ràng buộc các tỷ lệ bottleneck giống nhau, $\textrm{AnyNet}_\mathit{C}$ cũng ràng buộc các độ rộng group giống nhau, $\textrm{AnyNet}_\mathit{D}$ tăng độ sâu mạng qua các stage. Từ trái sang phải: (i) ràng buộc các tỷ lệ bottleneck giống nhau không ảnh hưởng đến hiệu năng; (ii) ràng buộc các độ rộng group giống nhau không ảnh hưởng đến hiệu năng; (iii) tăng độ rộng mạng (channel) qua các stage cải thiện hiệu năng; (iv) tăng độ sâu mạng qua các stage cải thiện hiệu năng. Hình do :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` cung cấp.](../img/regnet-fig.png)
:label:`fig_regnet-fig`

Tiếp theo, chúng ta tìm cách giảm số lượng lớn các lựa chọn tiềm năng về độ rộng và độ sâu của các stage. Một giả định hợp lý là khi đi sâu hơn, số lượng channel nên tăng lên, tức là $c_i \geq c_{i-1}$ ($w_{i+1} \geq w_i$ theo ký hiệu của họ trong :numref:`fig_regnet-fig`), tạo ra 
$\textrm{AnyNetX}_D$. Tương tự, cũng hợp lý khi giả định rằng khi các stage tiến triển, chúng nên sâu hơn, tức là $d_i \geq d_{i-1}$, tạo ra $\textrm{AnyNetX}_E$. Điều này có thể được xác minh bằng thực nghiệm lần lượt trong panel thứ ba và thứ tư của :numref:`fig_regnet-fig`.

## RegNet

Không gian thiết kế $\textrm{AnyNetX}_E$ thu được bao gồm các mạng đơn giản
tuân theo các nguyên tắc thiết kế dễ diễn giải:

* Dùng chung tỷ lệ bottleneck $k_i = k$ cho mọi stage $i$;
* Dùng chung độ rộng group $g_i = g$ cho mọi stage $i$;
* Tăng độ rộng mạng qua các stage: $c_{i} \leq c_{i+1}$;
* Tăng độ sâu mạng qua các stage: $d_{i} \leq d_{i+1}$.

Điều này để lại cho chúng ta tập lựa chọn cuối cùng: cách chọn các giá trị cụ thể cho các tham số trên của không gian thiết kế $\textrm{AnyNetX}_E$ cuối cùng. Bằng cách nghiên cứu các mạng có hiệu năng tốt nhất từ phân phối trong $\textrm{AnyNetX}_E$, ta có thể quan sát điều sau: lý tưởng nhất là độ rộng của mạng tăng tuyến tính theo chỉ số block trong toàn mạng, tức là $c_j \approx c_0 + c_a j$, trong đó $j$ là chỉ số block và độ dốc $c_a > 0$. Vì chúng ta chỉ có thể chọn độ rộng block khác nhau cho từng stage, ta thu được một hàm hằng từng đoạn, được thiết kế để khớp với sự phụ thuộc này. Hơn nữa, các thí nghiệm cũng cho thấy tỷ lệ bottleneck $k = 1$ cho hiệu năng tốt nhất, tức là chúng ta được khuyến nghị không sử dụng bottleneck nào cả. 

Chúng tôi khuyến nghị độc giả quan tâm xem thêm chi tiết về thiết kế các mạng cụ thể cho các mức tính toán khác nhau trong :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`. Ví dụ, một biến thể RegNetX 32 tầng hiệu quả được xác định bởi $k = 1$ (không có bottleneck), $g = 16$ (độ rộng group là 16), $c_1 = 32$ và $c_2 = 80$ channel lần lượt cho stage thứ nhất và thứ hai, với độ sâu được chọn là $d_1=4$ và $d_2=6$ block. Điểm đáng kinh ngạc từ thiết kế này là nó vẫn áp dụng được, ngay cả khi khảo sát các mạng ở quy mô lớn hơn. Hơn nữa, nó thậm chí còn đúng với các thiết kế mạng Squeeze-and-Excitation (SE) (RegNetY) có kích hoạt channel toàn cục :cite:`Hu.Shen.Sun.2018`.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RegNetX32(AnyNet):
    def __init__(self, lr=0.1, num_classes=10):
        stem_channels, groups, bot_mul = 32, 16, 1
        depths, channels = (4, 6), (32, 80)
        super().__init__(
            ((depths[0], channels[0], groups, bot_mul),
             (depths[1], channels[1], groups, bot_mul)),
            stem_channels, lr, num_classes)
```

```{.python .input}
%%tab jax
class RegNetX32(AnyNet):
    lr: float = 0.1
    num_classes: int = 10
    stem_channels: int = 32
    arch: tuple = ((4, 32, 16, 1), (6, 80, 16, 1))
```

Chúng ta có thể thấy rằng mỗi stage RegNetX dần giảm độ phân giải và tăng số channel đầu ra.

```{.python .input}
%%tab mxnet, pytorch
RegNetX32().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow
RegNetX32().layer_summary((1, 96, 96, 1))
```

```{.python .input}
%%tab jax
RegNetX32(training=False).layer_summary((1, 96, 96, 1))
```

## Huấn luyện

Việc huấn luyện RegNetX 32 tầng trên tập dữ liệu Fashion-MNIST cũng giống như trước.

```{.python .input}
%%tab mxnet, pytorch, jax
model = RegNetX32(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = RegNetX32(lr=0.01)
    trainer.fit(model, data)
```

## Thảo luận

Với các inductive bias (giả định hoặc ưu tiên) mong muốn như tính cục bộ và tính bất biến với phép tịnh tiến (:numref:`sec_why-conv`)
cho thị giác máy tính, CNN là kiến trúc chủ đạo trong lĩnh vực này. Điều này vẫn đúng từ LeNet cho đến khi Transformers (:numref:`sec_transformer`) :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,touvron2021training` bắt đầu vượt qua CNN về độ chính xác. Mặc dù phần lớn tiến bộ gần đây của vision Transformers *có thể* được áp dụng ngược vào CNN :cite:`liu2022convnet`, điều này chỉ khả thi với chi phí tính toán cao hơn. Quan trọng không kém, các tối ưu hóa phần cứng gần đây (NVIDIA Ampere và Hopper) chỉ càng nới rộng khoảng cách theo hướng có lợi cho Transformers. 

Đáng lưu ý rằng Transformers có mức độ inductive bias hướng tới tính cục bộ và tính bất biến với phép tịnh tiến thấp hơn đáng kể so với CNN. Việc các cấu trúc được học chiếm ưu thế một phần không nhỏ là nhờ sự sẵn có của các bộ sưu tập ảnh lớn, chẳng hạn như LAION-400m và LAION-5B :cite:`schuhmann2022laion` với tối đa 5 tỷ ảnh. Khá đáng ngạc nhiên, một số công trình phù hợp hơn trong bối cảnh này thậm chí còn bao gồm MLP :cite:`tolstikhin2021mlp`. 

Tóm lại, vision Transformers (:numref:`sec_vision-transformer`) hiện nay dẫn đầu về 
hiệu năng state-of-the-art trong phân loại ảnh quy mô lớn, 
cho thấy rằng *khả năng mở rộng vượt trội hơn inductive bias* :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
Điều này bao gồm việc pretraining các Transformers quy mô lớn (:numref:`sec_large-pretraining-transformers`) với self-attention đa đầu (:numref:`sec_multihead-attention`). Chúng tôi mời độc giả đi sâu vào các chương này để có thảo luận chi tiết hơn nhiều.

## Bài tập

1. Tăng số lượng stage lên bốn. Bạn có thể thiết kế một RegNetX sâu hơn với hiệu năng tốt hơn không?
1. Loại bỏ ResNeXt khỏi RegNet bằng cách thay block ResNeXt bằng block ResNet. Mô hình mới của bạn hoạt động như thế nào?
1. Triển khai nhiều biến thể của một họ "VioNet" bằng cách *vi phạm* các nguyên tắc thiết kế của RegNetX. Chúng hoạt động như thế nào? Trong số ($d_i$, $c_i$, $g_i$, $b_i$), yếu tố nào là quan trọng nhất?
1. Mục tiêu của bạn là thiết kế MLP "hoàn hảo". Bạn có thể sử dụng các nguyên tắc thiết kế được giới thiệu ở trên để tìm các kiến trúc tốt không? Có thể ngoại suy từ mạng nhỏ sang mạng lớn không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/7462)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/7463)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/8738)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18009)
:end_tab: