# Phát hiện đa hộp một lần
:label:`sec_ssd`

Trong :numref:`sec_bbox`--:numref:`sec_object-detection-dataset`,
chúng ta đã giới thiệu các hộp giới hạn, các hộp neo,
phát hiện đối tượng đa tỷ lệ và tập dữ liệu cho việc phát hiện đối tượng.
Bây giờ chúng ta đã sẵn sàng sử dụng những kiến thức nền tảng đó
để thiết kế một mô hình phát hiện đối tượng:
phát hiện đa hộp một lần
(SSD) :cite:`Liu.Anguelov.Erhan.ea.2016`.
Mô hình này đơn giản, nhanh và được sử dụng rộng rãi.
Mặc dù đây chỉ là một trong vô số
mô hình phát hiện đối tượng,
một số nguyên tắc thiết kế
và chi tiết triển khai
trong phần này
cũng có thể áp dụng cho các mô hình khác.


## Mô hình

:numref:`fig_ssd` cung cấp tổng quan về
thiết kế của mô hình phát hiện đa hộp một lần.
Mô hình này chủ yếu gồm
một mạng cơ sở
theo sau bởi
một số block feature map đa tỷ lệ.
Mạng cơ sở
dùng để trích xuất đặc trưng từ ảnh đầu vào,
vì vậy nó có thể sử dụng một CNN sâu.
Ví dụ,
bài báo gốc về phát hiện đa hộp một lần
sử dụng một mạng VGG được cắt bỏ trước
lớp phân loại :cite:`Liu.Anguelov.Erhan.ea.2016`,
trong khi ResNet cũng thường được sử dụng.
Thông qua thiết kế của chúng ta,
chúng ta có thể khiến mạng cơ sở tạo ra
các feature map lớn hơn
để sinh ra nhiều hộp neo hơn
nhằm phát hiện các đối tượng nhỏ hơn.
Sau đó,
mỗi block feature map đa tỷ lệ
giảm (ví dụ, giảm một nửa)
chiều cao và chiều rộng của các feature map
từ block trước đó,
đồng thời cho phép mỗi unit
của các feature map
tăng receptive field trên ảnh đầu vào.


Hãy nhớ lại thiết kế
của việc phát hiện đối tượng đa tỷ lệ
thông qua các biểu diễn ảnh theo từng lớp của
mạng nơ-ron sâu
trong :numref:`sec_multiscale-object-detection`.
Vì
các feature map đa tỷ lệ gần phía trên của :numref:`fig_ssd`
nhỏ hơn nhưng có receptive field lớn hơn,
chúng phù hợp để phát hiện
ít đối tượng hơn nhưng có kích thước lớn hơn.

Tóm lại,
thông qua mạng cơ sở và một số block feature map đa tỷ lệ,
mô hình phát hiện đa hộp một lần
sinh ra số lượng hộp neo khác nhau với các kích thước khác nhau,
và phát hiện các đối tượng có kích thước khác nhau
bằng cách dự đoán các lớp và offset
của những hộp neo này (do đó là các hộp giới hạn);
vì vậy, đây là một mô hình phát hiện đối tượng đa tỷ lệ.


![Là một mô hình phát hiện đối tượng đa tỷ lệ, mô hình phát hiện đa hộp một lần chủ yếu gồm một mạng cơ sở theo sau bởi một số block feature map đa tỷ lệ.](../img/ssd.svg)
:label:`fig_ssd`


Trong phần tiếp theo,
chúng ta sẽ mô tả các chi tiết triển khai
của những block khác nhau trong :numref:`fig_ssd`. Trước hết, chúng ta thảo luận
cách triển khai
việc dự đoán lớp và hộp giới hạn.



### [**Lớp dự đoán lớp**]

Gọi số lớp đối tượng là $q$.
Khi đó các hộp neo có $q+1$ lớp,
trong đó lớp 0 là nền.
Ở một tỷ lệ nào đó,
giả sử chiều cao và chiều rộng của các feature map
lần lượt là $h$ và $w$.
Khi có $a$ hộp neo
được sinh ra với
mỗi vị trí không gian của các feature map này làm tâm,
tổng cộng cần phân loại $hwa$ hộp neo.
Điều này thường khiến việc phân loại bằng các lớp fully connected không khả thi do có khả năng
phải trả chi phí tham số hóa rất lớn.
Hãy nhớ lại cách chúng ta sử dụng các channel của
các lớp convolution
để dự đoán các lớp trong :numref:`sec_nin`.
Mô hình phát hiện đa hộp một lần sử dụng
cùng kỹ thuật đó để giảm độ phức tạp của mô hình.

Cụ thể,
lớp dự đoán lớp sử dụng một lớp convolution
mà không thay đổi chiều rộng hoặc chiều cao của các feature map.
Theo cách này,
có thể tạo ra sự tương ứng một-một
giữa đầu ra và đầu vào
tại cùng các chiều không gian (chiều rộng và chiều cao)
của các feature map.
Cụ thể hơn,
các channel của feature map đầu ra
tại bất kỳ vị trí không gian nào ($x$, $y$)
biểu diễn các dự đoán lớp
cho tất cả các hộp neo có tâm tại
($x$, $y$) của các feature map đầu vào.
Để tạo ra các dự đoán hợp lệ,
phải có $a(q+1)$ channel đầu ra,
trong đó tại cùng một vị trí không gian, channel đầu ra có chỉ số $i(q+1) + j$
biểu diễn dự đoán của
lớp $j$ ($0 \leq j \leq q$)
cho hộp neo $i$ ($0 \leq i < a$).

Dưới đây, chúng ta định nghĩa một lớp dự đoán lớp như vậy,
xác định $a$ và $q$ lần lượt thông qua các đối số `num_anchors` và `num_classes`.
Lớp này sử dụng một lớp convolution $3\times3$ với
padding bằng 1.
Chiều rộng và chiều cao của đầu vào và đầu ra của
lớp convolution này vẫn không thay đổi.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
```

### (**Lớp dự đoán hộp giới hạn**)

Thiết kế của lớp dự đoán hộp giới hạn tương tự như lớp dự đoán lớp.
Điểm khác biệt duy nhất nằm ở số lượng đầu ra cho mỗi hộp neo:
ở đây, chúng ta cần dự đoán bốn offset thay vì $q+1$ lớp.

```{.python .input}
#@tab mxnet
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)
```

```{.python .input}
#@tab pytorch
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```

### [**Nối các dự đoán cho nhiều tỷ lệ**]

Như đã đề cập, mô hình phát hiện đa hộp một lần
sử dụng các feature map đa tỷ lệ để sinh ra các hộp neo và dự đoán lớp cùng offset của chúng.
Ở các tỷ lệ khác nhau,
shape của các feature map
hoặc số lượng hộp neo có tâm tại cùng một unit
có thể khác nhau.
Do đó,
shape của các đầu ra dự đoán
ở các tỷ lệ khác nhau cũng có thể khác nhau.

Trong ví dụ sau,
chúng ta xây dựng các feature map ở hai tỷ lệ khác nhau,
`Y1` và `Y2`,
cho cùng một minibatch,
trong đó chiều cao và chiều rộng của `Y2`
bằng một nửa chiều cao và chiều rộng của `Y1`.
Hãy lấy dự đoán lớp làm ví dụ.
Giả sử có
5 và 3 hộp neo
được sinh ra cho mỗi unit trong `Y1` và `Y2`, tương ứng.
Ngoài ra, giả sử
số lớp đối tượng là 10.
Đối với các feature map `Y1` và `Y2`,
số channel trong các đầu ra dự đoán lớp
lần lượt là $5\times(10+1)=55$ và $3\times(10+1)=33$,
trong đó mỗi shape đầu ra là
(batch size, số channel, chiều cao, chiều rộng).

```{.python .input}
#@tab mxnet
def forward(x, block):
    block.initialize()
    return block(x)

Y1 = forward(np.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(np.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
Y1.shape, Y2.shape
```

```{.python .input}
#@tab pytorch
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape
```

Như chúng ta có thể thấy, ngoại trừ chiều batch size,
ba chiều còn lại đều có kích thước khác nhau.
Để nối hai đầu ra dự đoán này nhằm tính toán hiệu quả hơn,
chúng ta sẽ chuyển đổi các tensor này sang một định dạng nhất quán hơn.

Lưu ý rằng
chiều channel chứa các dự đoán cho
các hộp neo có cùng tâm.
Trước tiên, chúng ta di chuyển chiều này vào trong cùng.
Vì batch size vẫn giữ nguyên ở các tỷ lệ khác nhau,
chúng ta có thể chuyển đổi đầu ra dự đoán
thành một tensor hai chiều
có shape (batch size, height $\times$ width $\times$ number of channels).
Sau đó, chúng ta có thể nối
các đầu ra như vậy ở những tỷ lệ khác nhau
dọc theo chiều 1.

```{.python .input}
#@tab mxnet
def flatten_pred(pred):
    return npx.batch_flatten(pred.transpose(0, 2, 3, 1))

def concat_preds(preds):
    return np.concatenate([flatten_pred(p) for p in preds], axis=1)
```

```{.python .input}
#@tab pytorch
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
```

Theo cách này,
mặc dù `Y1` và `Y2` có kích thước khác nhau
ở các chiều channel, chiều cao và chiều rộng,
chúng ta vẫn có thể nối hai đầu ra dự đoán này ở hai tỷ lệ khác nhau cho cùng một minibatch.

```{.python .input}
#@tab all
concat_preds([Y1, Y2]).shape
```

### [**Block downsampling**]

Để phát hiện các đối tượng ở nhiều tỷ lệ,
chúng ta định nghĩa block downsampling `down_sample_blk` sau đây, block này
giảm một nửa chiều cao và chiều rộng của các feature map đầu vào.
Trên thực tế,
block này áp dụng thiết kế của các block VGG
trong :numref:`subsec_vgg-blocks`.
Cụ thể hơn,
mỗi block downsampling gồm
hai lớp convolution $3\times3$ với padding bằng 1,
theo sau bởi một lớp max-pooling $2\times2$ với stride bằng 2.
Như chúng ta đã biết, các lớp convolution $3\times3$ với padding bằng 1 không thay đổi shape của các feature map.
Tuy nhiên, max-pooling $2\times2$ theo sau đó làm giảm chiều cao và chiều rộng của các feature map đầu vào xuống một nửa.
Đối với cả feature map đầu vào và đầu ra của block downsampling này,
vì $1\times 2+(3-1)+(3-1)=6$,
mỗi unit trong đầu ra
có receptive field $6\times6$ trên đầu vào.
Do đó, block downsampling mở rộng receptive field của mỗi unit trong các feature map đầu ra của nó.

```{.python .input}
#@tab mxnet
def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk
```

```{.python .input}
#@tab pytorch
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
```

Trong ví dụ sau, block downsampling do chúng ta xây dựng thay đổi số channel đầu vào và giảm một nửa chiều cao cùng chiều rộng của các feature map đầu vào.

```{.python .input}
#@tab mxnet
forward(np.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape
```

```{.python .input}
#@tab pytorch
forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
```

### [**Block mạng cơ sở**]

Block mạng cơ sở được dùng để trích xuất đặc trưng từ ảnh đầu vào.
Để đơn giản,
chúng ta xây dựng một mạng cơ sở nhỏ
gồm ba block downsampling,
mỗi block tăng gấp đôi số channel.
Với một ảnh đầu vào $256\times256$,
block mạng cơ sở này tạo ra các feature map $32 \times 32$ ($256/2^3=32$).

```{.python .input}
#@tab mxnet
def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

forward(np.zeros((2, 3, 256, 256)), base_net()).shape
```

```{.python .input}
#@tab pytorch
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape
```

### Mô hình hoàn chỉnh


[**Mô hình
phát hiện đa hộp một lần
hoàn chỉnh gồm năm block.**]
Các feature map do mỗi block tạo ra
được sử dụng cho cả
(i) việc sinh các hộp neo và (ii) việc dự đoán các lớp cùng offset của những hộp neo này.
Trong năm block này,
block thứ nhất
là block mạng cơ sở,
block thứ hai đến thứ tư là
các block downsampling,
và block cuối cùng
sử dụng global max-pooling
để giảm cả chiều cao và chiều rộng xuống 1.
Về mặt kỹ thuật,
block thứ hai đến thứ năm
đều là
những
block feature map đa tỷ lệ
trong :numref:`fig_ssd`.

```{.python .input}
#@tab mxnet
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk
```

```{.python .input}
#@tab pytorch
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk
```

Bây giờ chúng ta [**định nghĩa quá trình lan truyền tiến**]
cho mỗi block.
Khác với
các tác vụ phân loại ảnh,
các đầu ra ở đây bao gồm
(i) các feature map CNN `Y`,
(ii) các hộp neo được sinh ra bằng `Y` ở tỷ lệ hiện tại,
và (iii) các lớp cùng offset được dự đoán (dựa trên `Y`)
cho những hộp neo này.

```{.python .input}
#@tab mxnet
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

```{.python .input}
#@tab pytorch
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

Hãy nhớ rằng
trong :numref:`fig_ssd`,
một block feature map đa tỷ lệ
ở gần phía trên hơn
dùng để phát hiện các đối tượng lớn hơn;
do đó, nó cần tạo ra các anchor box lớn hơn.
Trong quá trình lan truyền tiến ở trên,
tại mỗi block feature map đa tỷ lệ,
chúng ta truyền vào một danh sách gồm hai giá trị tỷ lệ
thông qua đối số `sizes`
của hàm `multibox_prior` được gọi (mô tả trong :numref:`sec_anchor`).
Sau đây,
khoảng từ 0.2 đến 1.05
được chia đều
thành năm phần để xác định các
giá trị tỷ lệ nhỏ hơn tại năm block: 0.2, 0.37, 0.54, 0.71 và 0.88.
Sau đó, các giá trị tỷ lệ lớn hơn của chúng
được xác định bởi
$\sqrt{0.2 \times 0.37} = 0.272$, $\sqrt{0.37 \times 0.54} = 0.447$, và tương tự.

[~~Siêu tham số cho mỗi block~~]

```{.python .input}
#@tab all
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
```

Bây giờ chúng ta có thể [**định nghĩa mô hình hoàn chỉnh**] `TinySSD` như sau.

```{.python .input}
#@tab mxnet
class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = np.concatenate(anchors, axis=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

```{.python .input}
#@tab pytorch
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

Chúng ta [**tạo một instance của mô hình
và sử dụng nó để thực hiện lan truyền tiến**]
trên một minibatch gồm các ảnh kích thước $256 \times 256$ `X`.

Như đã trình bày ở phần trước,
block đầu tiên tạo ra các feature map kích thước $32 \times 32$.
Hãy nhớ rằng
các block downsampling thứ hai đến thứ tư
giảm một nửa chiều cao
và chiều rộng,
còn block thứ năm sử dụng global pooling.
Vì có 4 anchor box
được tạo cho mỗi đơn vị theo các chiều không gian
của feature map,
ở cả năm tỷ lệ
tổng cộng $(32^2 + 16^2 + 8^2 + 4^2 + 1)\times 4 = 5444$ anchor box được tạo cho mỗi ảnh.

```{.python .input}
#@tab mxnet
net = TinySSD(num_classes=1)
net.initialize()
X = np.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

```{.python .input}
#@tab pytorch
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

## Huấn luyện

Bây giờ chúng ta sẽ giải thích
cách huấn luyện mô hình single-shot multibox detection
để phát hiện đối tượng.


### Đọc Dataset và Khởi tạo Mô hình

Trước tiên,
hãy [**đọc
dataset phát hiện chuối**]
được mô tả trong :numref:`sec_object-detection-dataset`.

```{.python .input}
#@tab all
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
```

Dataset phát hiện chuối chỉ có một class. Sau khi định nghĩa mô hình,
chúng ta cần (**khởi tạo các tham số của mô hình và định nghĩa
thuật toán tối ưu hóa**).

```{.python .input}
#@tab mxnet
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=device)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'wd': 5e-4})
```

```{.python .input}
#@tab pytorch
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
```

### [**Định nghĩa các Hàm Loss và Đánh giá**]

Bài toán phát hiện đối tượng có hai loại loss.
Loss thứ nhất liên quan đến các class của anchor box:
việc tính loss này
có thể đơn giản sử dụng lại
hàm cross-entropy loss
mà chúng ta đã dùng cho bài toán phân loại ảnh.
Loss thứ hai
liên quan đến các offset của các anchor box dương (không phải background):
đây là một bài toán regression.
Tuy nhiên, đối với bài toán regression này,
ở đây chúng ta không sử dụng squared loss
được mô tả trong :numref:`subsec_normal_distribution_and_squared_loss`.
Thay vào đó,
chúng ta sử dụng loss theo chuẩn $\ell_1$,
là giá trị tuyệt đối của hiệu giữa
dự đoán và ground-truth.
Biến mask `bbox_masks` loại bỏ
các anchor box âm và các anchor box không hợp lệ (được đệm)
trong quá trình tính loss.
Cuối cùng, chúng ta cộng
class loss của anchor box
và offset loss của anchor box
để thu được hàm loss cho mô hình.

```{.python .input}
#@tab mxnet
cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
bbox_loss = gluon.loss.L1Loss()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox
```

```{.python .input}
#@tab pytorch
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox
```

Chúng ta có thể sử dụng accuracy để đánh giá kết quả phân loại.
Do sử dụng loss theo chuẩn $\ell_1$ cho các offset,
chúng ta sử dụng *mean absolute error* để đánh giá
các bounding box được dự đoán.
Các kết quả dự đoán này được thu được
từ các anchor box đã tạo và
các offset được dự đoán cho chúng.

```{.python .input}
#@tab mxnet
def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(axis=-1).astype(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((np.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

```{.python .input}
#@tab pytorch
def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

### [**Huấn luyện Mô hình**]

Khi huấn luyện mô hình,
chúng ta cần tạo các anchor box đa tỷ lệ (`anchors`)
và dự đoán class (`cls_preds`) cùng các offset (`bbox_preds`) của chúng trong quá trình lan truyền tiến.
Sau đó, chúng ta gán nhãn class (`cls_labels`) và offset (`bbox_labels`) cho các anchor box được tạo như vậy
dựa trên thông tin nhãn `Y`.
Cuối cùng, chúng ta tính hàm loss
bằng cách sử dụng các giá trị được dự đoán và được gán nhãn
của class và offset.
Để triển khai ngắn gọn,
phần đánh giá trên test dataset được lược bỏ ở đây.

```{.python .input}
#@tab mxnet
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = d2l.Accumulator(4)
    for features, target in train_iter:
        timer.start()
        X = features.as_in_ctx(device)
        Y = target.as_in_ctx(device)
        with autograd.record():
            # Generate multiscale anchor boxes and predict their classes and
            # offsets
            anchors, cls_preds, bbox_preds = net(X)
            # Label the classes and offsets of these anchor boxes
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors,
                                                                      Y)
            # Calculate the loss function using the predicted and labeled
            # values of the classes and offsets
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
        l.backward()
        trainer.step(batch_size)
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size,
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.size)
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter._dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

```{.python .input}
#@tab pytorch
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # Generate multiscale anchor boxes and predict their classes and
        # offsets
        anchors, cls_preds, bbox_preds = net(X)
        # Label the classes and offsets of these anchor boxes
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # Calculate the loss function using the predicted and labeled values
        # of the classes and offsets
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

## [**Dự đoán**]

Trong quá trình dự đoán,
mục tiêu là phát hiện tất cả các đối tượng cần quan tâm
trên ảnh.
Bên dưới,
chúng ta đọc và thay đổi kích thước một ảnh test,
chuyển ảnh này thành
một tensor bốn chiều
cần thiết cho các convolutional layer.

```{.python .input}
#@tab mxnet
img = image.imread('../img/banana.jpg')
feature = image.imresize(img, 256, 256).astype('float32')
X = np.expand_dims(feature.transpose(2, 0, 1), axis=0)
```

```{.python .input}
#@tab pytorch
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
```

Sử dụng hàm `multibox_detection` bên dưới,
các bounding box được dự đoán
được thu được
từ các anchor box và các offset được dự đoán của chúng.
Sau đó, non-maximum suppression được sử dụng
để loại bỏ các bounding box được dự đoán tương tự nhau.

```{.python .input}
#@tab mxnet
def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_ctx(device))
    cls_probs = npx.softmax(cls_preds).transpose(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

```{.python .input}
#@tab pytorch
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

Cuối cùng, chúng ta [**hiển thị
tất cả các bounding box được dự đoán có
độ tin cậy từ 0.9 trở lên**]
làm kết quả đầu ra.

```{.python .input}
#@tab mxnet
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[:2]
        bbox = [row[2:6] * np.array((w, h, w, h), ctx=row.ctx)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output, threshold=0.9)
```

```{.python .input}
#@tab pytorch
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
```

## Tóm tắt

* Single-shot multibox detection là một mô hình phát hiện đối tượng đa tỷ lệ. Thông qua base network và một số block feature map đa tỷ lệ, single-shot multibox detection tạo ra số lượng anchor box khác nhau với các kích thước khác nhau, đồng thời phát hiện các đối tượng có kích thước khác nhau bằng cách dự đoán class và offset của các anchor box này (từ đó thu được các bounding box).
* Khi huấn luyện mô hình single-shot multibox detection, hàm loss được tính dựa trên các giá trị được dự đoán và được gán nhãn của class và offset của anchor box.



## Bài tập

1. Bạn có thể cải thiện single-shot multibox detection bằng cách cải thiện hàm loss không? Ví dụ, thay loss theo chuẩn $\ell_1$ bằng loss theo chuẩn smooth $\ell_1$ cho các offset được dự đoán. Hàm loss này sử dụng một hàm bình phương quanh số 0 để tạo độ trơn, được điều khiển bởi siêu tham số $\sigma$:

$$
f(x) =
    \begin{cases}
    (\sigma x)^2/2,& \textrm{if }|x| < 1/\sigma^2\\
    |x|-0.5/\sigma^2,& \textrm{otherwise}
    \end{cases}
$$

Khi $\sigma$ rất lớn, loss này tương tự loss theo chuẩn $\ell_1$. Khi giá trị của nó nhỏ hơn, hàm loss sẽ trơn hơn.

```{.python .input}
#@tab mxnet
sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = np.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = npx.smooth_l1(x, scalar=s)
    d2l.plt.plot(x.asnumpy(), y.asnumpy(), l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)

sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = torch.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = smooth_l1(x, scalar=s)
    d2l.plt.plot(x, y, l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

Ngoài ra, trong thí nghiệm, chúng ta sử dụng cross-entropy loss cho việc dự đoán class:
gọi $p_j$ là xác suất được dự đoán cho class ground-truth $j$, cross-entropy loss là $-\log p_j$. Chúng ta cũng có thể sử dụng focal loss
:cite:`Lin.Goyal.Girshick.ea.2017`: với các siêu tham số $\gamma > 0$
và $\alpha > 0$, loss này được định nghĩa như sau:

$$ - \alpha (1-p_j)^{\gamma} \log p_j.$$

Như chúng ta có thể thấy, việc tăng $\gamma$
có thể làm giảm hiệu quả loss tương đối
đối với các mẫu được phân loại chính xác (ví dụ: $p_j > 0.5$)
để quá trình huấn luyện
có thể tập trung nhiều hơn vào những mẫu khó bị phân loại sai.

```{.python .input}
#@tab mxnet
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * np.log(x)

x = np.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x.asnumpy(), focal_loss(gamma, x).asnumpy(), l,
                     label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * torch.log(x)

x = torch.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x, focal_loss(gamma, x), l, label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

2. Do giới hạn về dung lượng, chúng ta đã lược bỏ một số chi tiết triển khai của mô hình single-shot multibox detection trong phần này. Bạn có thể tiếp tục cải thiện mô hình ở các khía cạnh sau không:
    1. Khi một đối tượng nhỏ hơn rất nhiều so với ảnh, mô hình có thể thay đổi kích thước ảnh đầu vào lớn hơn.
    1. Thông thường có rất nhiều anchor box âm. Để phân phối class cân bằng hơn, chúng ta có thể downsample các anchor box âm.
    1. Trong hàm loss, gán các siêu tham số trọng số khác nhau cho class loss và offset loss.
    1. Sử dụng các phương pháp khác để đánh giá mô hình phát hiện đối tượng, chẳng hạn như các phương pháp trong bài báo về single-shot multibox detection :cite:`Liu.Anguelov.Erhan.ea.2016`.



:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/373)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1604)
:end_tab: