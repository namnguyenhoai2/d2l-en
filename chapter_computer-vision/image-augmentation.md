# Tăng cường ảnh
:label:`sec_image_augmentation`

Trong :numref:`sec_alexnet`, 
chúng ta đã đề cập rằng các tập dữ liệu lớn
là điều kiện tiên quyết
cho sự thành công của
mạng neural sâu
trong nhiều ứng dụng khác nhau.
*Tăng cường ảnh* 
tạo ra các mẫu huấn luyện tương tự nhưng khác biệt
sau một loạt thay đổi ngẫu nhiên đối với ảnh huấn luyện, qua đó mở rộng kích thước của tập huấn luyện.
Một cách khác,
tăng cường ảnh có thể được lý giải
bởi thực tế rằng
những điều chỉnh ngẫu nhiên đối với các mẫu huấn luyện
cho phép mô hình ít phụ thuộc hơn vào
một số thuộc tính nhất định, qua đó cải thiện khả năng khái quát hóa.
Ví dụ, chúng ta có thể cắt ảnh theo nhiều cách khác nhau để đối tượng cần quan tâm xuất hiện ở những vị trí khác nhau, qua đó giảm sự phụ thuộc của mô hình vào vị trí của đối tượng.
Chúng ta cũng có thể điều chỉnh các yếu tố như độ sáng và màu sắc để giảm độ nhạy của mô hình đối với màu sắc.
Có lẽ đúng là
tăng cường ảnh đóng vai trò không thể thiếu
đối với sự thành công của AlexNet vào thời điểm đó.
Trong phần này, chúng ta sẽ thảo luận về kỹ thuật được sử dụng rộng rãi này trong thị giác máy tính.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
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
```

## Các phương pháp tăng cường ảnh phổ biến

Trong quá trình tìm hiểu các phương pháp tăng cường ảnh phổ biến, chúng ta sẽ sử dụng ảnh $400\times 500$ sau đây làm ví dụ.

```{.python .input}
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

Hầu hết các phương pháp tăng cường ảnh đều có một mức độ ngẫu nhiên nhất định. Để chúng ta dễ quan sát tác động của việc tăng cường ảnh hơn, tiếp theo chúng ta định nghĩa một hàm phụ `apply`. Hàm này chạy phương pháp tăng cường ảnh `aug` nhiều lần trên ảnh đầu vào `img` và hiển thị tất cả kết quả.

```{.python .input}
#@tab all
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```

### Lật và cắt ảnh

:begin_tab:`mxnet`
[**Lật ảnh từ trái sang phải**] thường không làm thay đổi phân loại của đối tượng.
Đây là một trong những phương pháp tăng cường ảnh sớm nhất và được sử dụng rộng rãi nhất.
Tiếp theo, chúng ta sử dụng module `transforms` để tạo instance `RandomFlipLeftRight`, instance này lật ảnh từ trái sang phải với xác suất 50%.
:end_tab:

:begin_tab:`pytorch`
[**Lật ảnh từ trái sang phải**] thường không làm thay đổi phân loại của đối tượng.
Đây là một trong những phương pháp tăng cường ảnh sớm nhất và được sử dụng rộng rãi nhất.
Tiếp theo, chúng ta sử dụng module `transforms` để tạo instance `RandomHorizontalFlip`, instance này lật ảnh từ trái sang phải với xác suất 50%.
:end_tab:

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomFlipLeftRight())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomHorizontalFlip())
```

:begin_tab:`mxnet`
[**Lật ảnh từ trên xuống dưới**] không phổ biến bằng lật từ trái sang phải. Tuy nhiên, ít nhất đối với ảnh ví dụ này, việc lật từ trên xuống dưới không cản trở khả năng nhận dạng.
Tiếp theo, chúng ta tạo một instance `RandomFlipTopBottom` để lật ảnh từ trên xuống dưới với xác suất 50%.
:end_tab:

:begin_tab:`pytorch`
[**Lật ảnh từ trên xuống dưới**] không phổ biến bằng lật từ trái sang phải. Tuy nhiên, ít nhất đối với ảnh ví dụ này, việc lật từ trên xuống dưới không cản trở khả năng nhận dạng.
Tiếp theo, chúng ta tạo một instance `RandomVerticalFlip` để lật ảnh từ trên xuống dưới với xác suất 50%.
:end_tab:

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomVerticalFlip())
```

Trong ảnh ví dụ mà chúng ta sử dụng, con mèo nằm ở giữa ảnh, nhưng nhìn chung không phải lúc nào cũng như vậy.
Trong :numref:`sec_pooling`, chúng ta đã giải thích rằng lớp pooling có thể làm giảm độ nhạy của lớp tích chập đối với vị trí của đối tượng.
Ngoài ra, chúng ta cũng có thể cắt ảnh ngẫu nhiên để làm cho các đối tượng xuất hiện ở những vị trí khác nhau và với các tỷ lệ khác nhau trong ảnh, điều này cũng có thể làm giảm độ nhạy của mô hình đối với vị trí của đối tượng.

Trong đoạn code dưới đây, mỗi lần chúng ta [**cắt ngẫu nhiên**] một vùng có diện tích bằng $10\% \sim 100\%$ diện tích ban đầu, đồng thời tỷ lệ giữa chiều rộng và chiều cao của vùng này được chọn ngẫu nhiên trong khoảng $0.5 \sim 2$. Sau đó, cả chiều rộng và chiều cao của vùng đều được scale thành 200 pixel.
Trừ khi có quy định khác, số ngẫu nhiên giữa $a$ và $b$ trong phần này là một giá trị liên tục thu được bằng cách lấy mẫu ngẫu nhiên và đồng đều từ khoảng $[a, b]$.

```{.python .input}
#@tab mxnet
shape_aug = gluon.data.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input}
#@tab pytorch
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

### Thay đổi màu sắc

Một phương pháp tăng cường khác là thay đổi màu sắc. Chúng ta có thể thay đổi bốn khía cạnh của màu ảnh: độ sáng, độ tương phản, độ bão hòa và sắc độ. Trong ví dụ dưới đây, chúng ta [**thay đổi ngẫu nhiên độ sáng**] của ảnh thành một giá trị trong khoảng từ 50% ($1-0.5$) đến 150% ($1+0.5$) so với ảnh gốc.

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

Tương tự, chúng ta có thể [**thay đổi ngẫu nhiên sắc độ**] của ảnh.

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomHue(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

Chúng ta cũng có thể tạo một instance `RandomColorJitter` và thiết lập cách [**thay đổi ngẫu nhiên `brightness`, `contrast`, `saturation` và `hue` của ảnh cùng lúc**].

```{.python .input}
#@tab mxnet
color_aug = gluon.data.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

```{.python .input}
#@tab pytorch
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

### Kết hợp nhiều phương pháp tăng cường ảnh

Trong thực tế, chúng ta sẽ [**kết hợp nhiều phương pháp tăng cường ảnh**].
Ví dụ,
chúng ta có thể kết hợp các phương pháp tăng cường ảnh khác nhau được định nghĩa ở trên và áp dụng chúng cho từng ảnh thông qua một instance `Compose`.

```{.python .input}
#@tab mxnet
augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input}
#@tab pytorch
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

## [**Huấn luyện với tăng cường ảnh**]

Hãy huấn luyện một mô hình với tăng cường ảnh.
Ở đây, chúng ta sử dụng tập dữ liệu CIFAR-10 thay vì tập dữ liệu Fashion-MNIST đã sử dụng trước đó.
Lý do là vị trí và kích thước của các đối tượng trong tập dữ liệu Fashion-MNIST đã được chuẩn hóa, trong khi màu sắc và kích thước của các đối tượng trong tập dữ liệu CIFAR-10 có sự khác biệt đáng kể hơn.
32 ảnh huấn luyện đầu tiên trong tập dữ liệu CIFAR-10 được hiển thị bên dưới.

```{.python .input}
#@tab mxnet
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[:32][0], 4, 8, scale=0.8);
```

```{.python .input}
#@tab pytorch
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

Để thu được kết quả nhất quán trong quá trình dự đoán, chúng ta thường chỉ áp dụng tăng cường ảnh cho các mẫu huấn luyện và không sử dụng tăng cường ảnh với các phép toán ngẫu nhiên trong quá trình dự đoán.
[**Ở đây, chúng ta chỉ sử dụng phương pháp lật ngẫu nhiên từ trái sang phải đơn giản nhất**]. Ngoài ra, chúng ta sử dụng một instance `ToTensor` để chuyển một minibatch ảnh sang định dạng mà deep learning framework yêu cầu, tức là các số dấu phẩy động 32-bit trong khoảng từ 0 đến 1 với shape là (batch size, number of channels, height, width).

```{.python .input}
#@tab mxnet
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])
```

```{.python .input}
#@tab pytorch
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])
```

:begin_tab:`mxnet`
Tiếp theo, chúng ta định nghĩa một hàm phụ để thuận tiện cho việc đọc ảnh và
áp dụng tăng cường ảnh.
Hàm `transform_first` do Gluon cung cấp
áp dụng tăng cường ảnh cho phần tử đầu tiên của mỗi mẫu huấn luyện
(ảnh và nhãn), tức là ảnh.
Để
biết phần giới thiệu chi tiết về `DataLoader`, vui lòng tham khảo :numref:`sec_fashion_mnist`.
:end_tab:

:begin_tab:`pytorch`
Tiếp theo, chúng ta [**định nghĩa một hàm phụ để thuận tiện cho việc đọc ảnh và
áp dụng tăng cường ảnh**].
Đối số `transform` do dataset của PyTorch cung cấp
áp dụng augmentation để biến đổi ảnh.
Để
biết phần giới thiệu chi tiết về `DataLoader`, vui lòng tham khảo :numref:`sec_fashion_mnist`.
:end_tab:

```{.python .input}
#@tab mxnet
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```

### Huấn luyện Multi-GPU

Chúng ta huấn luyện mô hình ResNet-18 từ
:numref:`sec_resnet` trên
tập dữ liệu CIFAR-10.
Hãy nhớ lại phần giới thiệu về
huấn luyện multi-GPU trong :numref:`sec_multi_gpu_concise`.
Trong phần tiếp theo,
[**chúng ta định nghĩa một hàm để huấn luyện và đánh giá mô hình bằng nhiều GPU**].

```{.python .input}
#@tab mxnet
#@save
def train_batch_ch13(net, features, labels, loss, trainer, devices,
                     split_f=d2l.split_batch):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13)."""
    X_shards, y_shards = split_f(features, labels, devices)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    # The `True` flag allows parameters with stale gradients, which is useful
    # later (e.g., in fine-tuning BERT)
    trainer.step(labels.shape[0], ignore_stale_grad=True)
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab pytorch
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13)."""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab mxnet
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus(), split_f=d2l.split_batch):
    """Train a model with multiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Train a model with multiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

Bây giờ chúng ta có thể [**định nghĩa hàm `train_with_data_aug` để huấn luyện mô hình với tăng cường ảnh**].
Hàm này lấy tất cả GPU khả dụng,
sử dụng Adam làm thuật toán tối ưu,
áp dụng tăng cường ảnh cho tập dữ liệu huấn luyện,
và cuối cùng gọi hàm `train_ch13` vừa định nghĩa để huấn luyện và đánh giá mô hình.

```{.python .input}
#@tab mxnet
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)
net.initialize(init=init.Xavier(), ctx=devices)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

```{.python .input}
#@tab pytorch
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)
net.apply(d2l.init_cnn)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    net(next(iter(train_iter))[0])
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

Hãy [**huấn luyện mô hình**] bằng cách sử dụng tăng cường ảnh dựa trên phép lật ngẫu nhiên từ trái sang phải.

```{.python .input}
#@tab all
train_with_data_aug(train_augs, test_augs, net)
```

## Tóm tắt

* Tăng cường ảnh tạo ra các ảnh ngẫu nhiên dựa trên dữ liệu huấn luyện hiện có để cải thiện khả năng khái quát hóa của mô hình.
* Để thu được kết quả nhất quán trong quá trình dự đoán, chúng ta thường chỉ áp dụng tăng cường ảnh cho các mẫu huấn luyện và không sử dụng tăng cường ảnh với các phép toán ngẫu nhiên trong quá trình dự đoán.
* Các deep learning framework cung cấp nhiều phương pháp tăng cường ảnh khác nhau, có thể được áp dụng đồng thời.


## Bài tập

1. Huấn luyện mô hình mà không sử dụng tăng cường ảnh: `train_with_data_aug(test_augs, test_augs)`. So sánh độ chính xác huấn luyện và kiểm tra khi sử dụng và không sử dụng tăng cường ảnh. Thí nghiệm so sánh này có thể củng cố lập luận rằng tăng cường ảnh có thể giảm overfitting không? Tại sao?
1. Kết hợp nhiều phương pháp tăng cường ảnh khác nhau trong quá trình huấn luyện mô hình trên tập dữ liệu CIFAR-10. Điều này có cải thiện độ chính xác kiểm tra không?
1. Tham khảo tài liệu trực tuyến của deep learning framework. Framework đó còn cung cấp những phương pháp tăng cường ảnh nào khác?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/367)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1404)
:end_tab: