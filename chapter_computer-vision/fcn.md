# Mạng tích chập hoàn toàn
:label:`sec_fcn`

Như đã thảo luận trong :numref:`sec_semantic_segmentation`,
phân đoạn ngữ nghĩa
phân loại hình ảnh ở cấp độ pixel.
Mạng tích chập hoàn toàn (FCN)
sử dụng mạng nơ-ron tích chập để
chuyển đổi các pixel hình ảnh thành các lớp pixel :cite:`Long.Shelhamer.Darrell.2015`.
Khác với các CNN mà chúng ta đã gặp trước đây
trong bài toán phân loại hình ảnh
hoặc phát hiện đối tượng,
mạng tích chập hoàn toàn
chuyển đổi
chiều cao và chiều rộng của các feature map trung gian
trở lại thành kích thước của hình ảnh đầu vào:
điều này đạt được nhờ
lớp tích chập chuyển vị
được giới thiệu trong :numref:`sec_transposed_conv`.
Do đó,
đầu ra phân loại
và hình ảnh đầu vào
có sự tương ứng một-một
ở cấp độ pixel:
chiều kênh tại mỗi pixel đầu ra
chứa các kết quả phân loại
cho pixel đầu vào ở cùng vị trí không gian.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
```

## Mô hình

Ở đây, chúng ta mô tả thiết kế cơ bản của mô hình mạng tích chập hoàn toàn.
Như minh họa trong :numref:`fig_fcn`,
mô hình này trước tiên sử dụng một CNN để trích xuất các đặc trưng hình ảnh,
sau đó chuyển đổi số kênh thành
số lớp
thông qua một lớp tích chập $1\times 1$,
và cuối cùng chuyển đổi chiều cao và chiều rộng của
các feature map
thành
chiều cao và chiều rộng
của hình ảnh đầu vào thông qua phép tích chập chuyển vị được giới thiệu trong :numref:`sec_transposed_conv`.
Do đó,
đầu ra của mô hình có cùng chiều cao và chiều rộng với hình ảnh đầu vào,
trong đó kênh đầu ra chứa các lớp được dự đoán
cho pixel đầu vào ở cùng vị trí không gian.


![Mạng tích chập hoàn toàn.](../img/fcn.svg)
:label:`fig_fcn`

Dưới đây, chúng ta [**sử dụng mô hình ResNet-18 được pretrained trên bộ dữ liệu ImageNet để trích xuất các đặc trưng hình ảnh**]
và ký hiệu instance của mô hình là `pretrained_net`.
Một vài lớp cuối của mô hình này
bao gồm một lớp global average pooling
và một lớp fully connected:
chúng không cần thiết
trong mạng tích chập hoàn toàn.

```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
pretrained_net.features[-3:], pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]
```

Tiếp theo, chúng ta [**tạo instance mạng tích chập hoàn toàn `net`**].
Mô hình này sao chép tất cả các lớp đã pretrained trong ResNet-18
ngoại trừ lớp global average pooling cuối cùng
và lớp fully connected gần
đầu ra nhất.

```{.python .input}
#@tab mxnet
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*list(pretrained_net.children())[:-2])
```

Với đầu vào có chiều cao và chiều rộng lần lượt là 320 và 480,
quá trình lan truyền xuôi của `net`
giảm chiều cao và chiều rộng đầu vào xuống còn 1/32 kích thước ban đầu, tức là 10 và 15.

```{.python .input}
#@tab mxnet
X = np.random.uniform(size=(1, 3, 320, 480))
net(X).shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 3, 320, 480))
net(X).shape
```

Tiếp theo, chúng ta [**sử dụng một lớp tích chập $1\times 1$ để chuyển đổi số kênh đầu ra thành số lớp (21) của bộ dữ liệu Pascal VOC2012.**]
Cuối cùng, chúng ta cần (**tăng chiều cao và chiều rộng của các feature map lên 32 lần**) để đưa chúng trở lại chiều cao và chiều rộng của hình ảnh đầu vào.
Hãy nhớ cách tính
shape đầu ra của một lớp tích chập trong :numref:`sec_padding`.
Vì $(320-64+16\times2+32)/32=10$ và $(480-64+16\times2+32)/32=15$, chúng ta xây dựng một lớp tích chập chuyển vị với stride bằng $32$,
đặt
chiều cao và chiều rộng của kernel
bằng $64$, padding bằng $16$.
Nói chung,
ta có thể thấy rằng
với stride $s$,
padding $s/2$ (giả sử $s/2$ là một số nguyên),
và chiều cao, chiều rộng của kernel bằng $2s$,
phép tích chập chuyển vị sẽ tăng
chiều cao và chiều rộng của đầu vào lên $s$ lần.

```{.python .input}
#@tab mxnet
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(
            num_classes, kernel_size=64, padding=16, strides=32))
```

```{.python .input}
#@tab pytorch
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```

## [**Khởi tạo các lớp tích chập chuyển vị**]


Chúng ta đã biết rằng
các lớp tích chập chuyển vị có thể tăng
chiều cao và chiều rộng của
các feature map.
Trong xử lý hình ảnh, chúng ta có thể cần phóng to
một hình ảnh, tức là *upsampling*.
*Nội suy song tuyến tính*
là một trong những kỹ thuật upsampling thường được sử dụng.
Nó cũng thường được dùng để khởi tạo các lớp tích chập chuyển vị.

Để giải thích phép nội suy song tuyến tính,
giả sử rằng
với một hình ảnh đầu vào,
chúng ta muốn
tính toán từng pixel
của hình ảnh đầu ra sau khi upsampling.
Để tính pixel của hình ảnh đầu ra
tại tọa độ $(x, y)$,
trước tiên ánh xạ $(x, y)$
sang tọa độ $(x', y')$ trên hình ảnh đầu vào, chẳng hạn dựa trên tỷ lệ giữa kích thước đầu vào và kích thước đầu ra.
Lưu ý rằng $x'$ và $y'$ sau khi ánh xạ là các số thực.
Sau đó, tìm bốn pixel gần nhất với tọa độ
$(x', y')$ trên hình ảnh đầu vào.
Cuối cùng, pixel của hình ảnh đầu ra tại tọa độ $(x, y)$ được tính dựa trên bốn pixel gần nhất này
trên hình ảnh đầu vào và khoảng cách tương đối của chúng so với $(x', y')$.

Upsampling bằng nội suy song tuyến tính
có thể được triển khai bằng lớp tích chập chuyển vị
với kernel được xây dựng bởi hàm `bilinear_kernel` sau đây.
Do giới hạn về dung lượng, chúng tôi chỉ cung cấp phần triển khai của hàm `bilinear_kernel` bên dưới
mà không thảo luận về thiết kế thuật toán của hàm này.

```{.python .input}
#@tab mxnet
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)
```

```{.python .input}
#@tab pytorch
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
```

Hãy [**thử nghiệm với upsampling bằng nội suy song tuyến tính**]
được triển khai bằng một lớp tích chập chuyển vị.
Chúng ta xây dựng một lớp tích chập chuyển vị
tăng gấp đôi chiều cao và chiều rộng,
và khởi tạo kernel của nó bằng hàm `bilinear_kernel`.

```{.python .input}
#@tab mxnet
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))
```

```{.python .input}
#@tab pytorch
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));
```

Đọc hình ảnh `X` và gán đầu ra upsampling cho `Y`. Để in hình ảnh, chúng ta cần điều chỉnh vị trí của chiều kênh.

```{.python .input}
#@tab mxnet
img = image.imread('../img/catdog.jpg')
X = np.expand_dims(img.astype('float32').transpose(2, 0, 1), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose(1, 2, 0)
```

```{.python .input}
#@tab pytorch
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
```

Như có thể thấy, lớp tích chập chuyển vị tăng cả chiều cao và chiều rộng của hình ảnh lên hai lần.
Ngoại trừ sự khác biệt về tỷ lệ tọa độ,
hình ảnh được phóng to bằng nội suy song tuyến tính và hình ảnh gốc được in trong :numref:`sec_bbox` trông giống nhau.

```{.python .input}
#@tab mxnet
d2l.set_figsize()
print('input image shape:', img.shape)
d2l.plt.imshow(img.asnumpy());
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0));
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img);
```

Trong mạng tích chập hoàn toàn, chúng ta [**khởi tạo lớp tích chập chuyển vị bằng upsampling nội suy song tuyến tính. Đối với lớp tích chập $1\times 1$, chúng ta sử dụng khởi tạo Xavier.**]

```{.python .input}
#@tab mxnet
W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())
```

```{.python .input}
#@tab pytorch
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);
```

## [**Đọc bộ dữ liệu**]

Chúng ta đọc
bộ dữ liệu phân đoạn ngữ nghĩa
như đã giới thiệu trong :numref:`sec_semantic_segmentation`.
Shape đầu ra của phép crop ngẫu nhiên được
chỉ định là $320\times 480$: cả chiều cao và chiều rộng đều chia hết cho $32$.

```{.python .input}
#@tab all
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)
```

## [**Huấn luyện**]


Bây giờ chúng ta có thể huấn luyện
mạng tích chập hoàn toàn đã xây dựng.
Hàm loss và cách tính accuracy ở đây
về cơ bản không khác với những phần trong bài toán phân loại hình ảnh ở các chương trước.
Vì chúng ta sử dụng kênh đầu ra của
lớp tích chập chuyển vị để
dự đoán lớp cho từng pixel,
chiều kênh được chỉ định trong phép tính loss.
Ngoài ra, accuracy được tính
dựa trên tính chính xác
của lớp được dự đoán cho tất cả các pixel.

```{.python .input}
#@tab mxnet
num_epochs, lr, wd, devices = 5, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(devices)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## [**Dự đoán**]


Khi dự đoán, chúng ta cần chuẩn hóa hình ảnh đầu vào
trên từng kênh và chuyển hình ảnh thành định dạng đầu vào bốn chiều mà CNN yêu cầu.

```{.python .input}
#@tab mxnet
def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = np.expand_dims(X.transpose(2, 0, 1), axis=0)
    pred = net(X.as_in_ctx(devices[0])).argmax(axis=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

```{.python .input}
#@tab pytorch
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

Để [**trực quan hóa lớp được dự đoán**] của từng pixel, chúng ta ánh xạ lớp được dự đoán trở lại màu nhãn tương ứng trong bộ dữ liệu.

```{.python .input}
#@tab mxnet
def label2image(pred):
    colormap = np.array(d2l.VOC_COLORMAP, ctx=devices[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]
```

```{.python .input}
#@tab pytorch
def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]
```

Các hình ảnh trong bộ dữ liệu kiểm tra có kích thước và hình dạng khác nhau.
Vì mô hình sử dụng một lớp tích chập chuyển vị với stride bằng 32,
khi chiều cao hoặc chiều rộng của hình ảnh đầu vào không chia hết cho 32,
chiều cao hoặc chiều rộng đầu ra của
lớp tích chập chuyển vị sẽ lệch khỏi hình dạng của hình ảnh đầu vào.
Để giải quyết vấn đề này,
chúng ta có thể crop nhiều vùng hình chữ nhật có chiều cao và chiều rộng là các bội số nguyên của 32 trong hình ảnh,
và thực hiện lan truyền xuôi
riêng trên các pixel trong những vùng này.
Lưu ý rằng
hợp của các vùng hình chữ nhật này cần bao phủ hoàn toàn hình ảnh đầu vào.
Khi một pixel được bao phủ bởi nhiều vùng hình chữ nhật,
trung bình của các đầu ra tích chập chuyển vị
tại các vùng riêng biệt đối với cùng pixel này
có thể được đưa vào
phép toán softmax
để dự đoán lớp.


Để đơn giản, chúng tôi chỉ đọc một vài hình ảnh kiểm tra lớn hơn,
và crop một vùng $320\times480$ để dự đoán, bắt đầu từ góc trên bên trái của hình ảnh.
Đối với các hình ảnh kiểm tra này, chúng tôi
in các vùng đã crop,
kết quả dự đoán,
và nhãn thực tế theo từng hàng.

```{.python .input}
#@tab mxnet
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

```{.python .input}
#@tab pytorch
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

## Tóm tắt

* Mạng tích chập hoàn toàn trước tiên sử dụng một CNN để trích xuất các đặc trưng hình ảnh, sau đó chuyển đổi số kênh thành số lớp thông qua một lớp tích chập $1\times 1$, và cuối cùng chuyển đổi chiều cao và chiều rộng của các feature map thành chiều cao và chiều rộng của hình ảnh đầu vào thông qua phép tích chập chuyển vị.
* Trong mạng tích chập hoàn toàn, chúng ta có thể sử dụng upsampling bằng nội suy song tuyến tính để khởi tạo lớp tích chập chuyển vị.


## Bài tập

1. Nếu sử dụng khởi tạo Xavier cho lớp tích chập chuyển vị trong thử nghiệm, kết quả sẽ thay đổi như thế nào?
1. Bạn có thể cải thiện hơn nữa accuracy của mô hình bằng cách tinh chỉnh các hyperparameter không?
1. Dự đoán lớp của tất cả các pixel trong các hình ảnh kiểm tra.
1. Bài báo gốc về mạng tích chập hoàn toàn cũng sử dụng đầu ra của một số lớp CNN trung gian :cite:`Long.Shelhamer.Darrell.2015`. Hãy thử triển khai ý tưởng này.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/377)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1582)
:end_tab: