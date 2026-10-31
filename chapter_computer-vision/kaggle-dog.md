# Nhận diện giống chó (ImageNet Dogs) trên Kaggle

Trong phần này, chúng ta sẽ thực hành
bài toán nhận diện giống chó trên
Kaggle. (**Địa chỉ web của cuộc thi này là https://www.kaggle.com/c/dog-breed-identification**)

Trong cuộc thi này,
120 giống chó khác nhau sẽ được nhận diện.
Trên thực tế,
bộ dữ liệu cho cuộc thi này là
một tập con của bộ dữ liệu ImageNet.
Không giống như các hình ảnh trong bộ dữ liệu CIFAR-10 ở :numref:`sec_kaggle_cifar10`,
các hình ảnh trong bộ dữ liệu ImageNet có chiều cao và chiều rộng lớn hơn, với kích thước thay đổi.
:numref:`fig_kaggle_dog` hiển thị thông tin trên trang web của cuộc thi. Bạn cần có tài khoản Kaggle
để gửi kết quả.


![Trang web của cuộc thi nhận diện giống chó. Có thể lấy bộ dữ liệu của cuộc thi bằng cách nhấp vào tab "Data".](../img/kaggle-dog.jpg)
:width:`400px`
:label:`fig_kaggle_dog`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import os
```

## Lấy và tổ chức bộ dữ liệu

Bộ dữ liệu của cuộc thi được chia thành tập huấn luyện và tập kiểm tra, lần lượt chứa 10222 và 10357 ảnh JPEG
với ba kênh RGB (màu).
Trong bộ dữ liệu huấn luyện,
có 120 giống chó
như Labrador, Poodle, Dachshund, Samoyed, Husky, Chihuahua và Yorkshire Terrier.


### Tải xuống bộ dữ liệu

Sau khi đăng nhập vào Kaggle,
bạn có thể nhấp vào tab "Data" trên
trang web của cuộc thi được hiển thị trong :numref:`fig_kaggle_dog` và tải xuống bộ dữ liệu bằng cách nhấp vào nút "Download All".
Sau khi giải nén tệp đã tải xuống vào `../data`, bạn sẽ tìm thấy toàn bộ bộ dữ liệu tại các đường dẫn sau:

* ../data/dog-breed-identification/labels.csv
* ../data/dog-breed-identification/sample_submission.csv
* ../data/dog-breed-identification/train
* ../data/dog-breed-identification/test

Có thể bạn đã nhận thấy rằng cấu trúc trên
tương tự như cấu trúc của cuộc thi CIFAR-10 trong :numref:`sec_kaggle_cifar10`, trong đó các thư mục `train/` và `test/` lần lượt chứa ảnh chó dùng cho huấn luyện và kiểm tra, còn `labels.csv` chứa
nhãn của các ảnh huấn luyện.
Tương tự, để giúp bạn bắt đầu dễ dàng hơn, [**chúng tôi cung cấp một mẫu nhỏ của bộ dữ liệu**] nêu trên: `train_valid_test_tiny.zip`.
Nếu bạn định sử dụng toàn bộ bộ dữ liệu cho cuộc thi Kaggle, bạn cần thay đổi biến `demo` bên dưới thành `False`.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# If you use the full dataset downloaded for the Kaggle competition, change
# the variable below to `False`
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')
```

### [**Tổ chức bộ dữ liệu**]

Chúng ta có thể tổ chức bộ dữ liệu tương tự như cách đã thực hiện trong :numref:`sec_kaggle_cifar10`, cụ thể là tách
một tập validation khỏi tập huấn luyện ban đầu và chuyển các ảnh vào những thư mục con được nhóm theo nhãn.

Hàm `reorg_dog_data` bên dưới đọc
nhãn của dữ liệu huấn luyện, tách tập validation và tổ chức tập huấn luyện.

```{.python .input}
#@tab all
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)


batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)
```

## [**Tăng cường ảnh**]

Hãy nhớ rằng bộ dữ liệu giống chó này
là một tập con của bộ dữ liệu ImageNet,
với các hình ảnh
lớn hơn những hình ảnh trong bộ dữ liệu CIFAR-10
ở :numref:`sec_kaggle_cifar10`.
Dưới đây
liệt kê một số phép toán tăng cường ảnh
có thể hữu ích đối với các ảnh tương đối lớn.

```{.python .input}
#@tab mxnet
transform_train = gluon.data.vision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height-to-width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new 224 x 224 image
    gluon.data.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                   ratio=(3.0/4.0, 4.0/3.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    # Randomly change the brightness, contrast, and saturation
    gluon.data.vision.transforms.RandomColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4),
    # Add random noise
    gluon.data.vision.transforms.RandomLighting(0.1),
    gluon.data.vision.transforms.ToTensor(),
    # Standardize each channel of the image
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height-to-width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new 224 x 224 image
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # Randomly change the brightness, contrast, and saturation
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # Add random noise
    torchvision.transforms.ToTensor(),
    # Standardize each channel of the image
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

Trong quá trình dự đoán,
chúng ta chỉ sử dụng các phép tiền xử lý ảnh
không có tính ngẫu nhiên.

```{.python .input}
#@tab mxnet
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    # Crop a 224 x 224 square area from the center of the image
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # Crop a 224 x 224 square area from the center of the image
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

## [**Đọc bộ dữ liệu**]

Tương tự như trong :numref:`sec_kaggle_cifar10`,
chúng ta có thể đọc bộ dữ liệu đã được tổ chức
gồm các tệp ảnh thô.

```{.python .input}
#@tab mxnet
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ('train', 'valid', 'train_valid', 'test')]
```

```{.python .input}
#@tab pytorch
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]
```

Bên dưới, chúng ta tạo các instance của data iterator
theo cách giống
như trong :numref:`sec_kaggle_cifar10`.

```{.python .input}
#@tab mxnet
train_iter, train_valid_iter = [gluon.data.DataLoader(
    dataset.transform_first(transform_train), batch_size, shuffle=True,
    last_batch='discard') for dataset in (train_ds, train_valid_ds)]

valid_iter = gluon.data.DataLoader(
    valid_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='discard')

test_iter = gluon.data.DataLoader(
    test_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='keep')
```

```{.python .input}
#@tab pytorch
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)
```

## [**Fine-tuning mô hình pretrained**]

Một lần nữa,
bộ dữ liệu cho cuộc thi này là một tập con của bộ dữ liệu ImageNet.
Do đó, chúng ta có thể sử dụng phương pháp được thảo luận trong
:numref:`sec_fine_tuning`
để chọn một mô hình được pretrained trên
toàn bộ bộ dữ liệu ImageNet và sử dụng mô hình đó để trích xuất các đặc trưng ảnh, rồi đưa chúng vào một
mạng output tùy chỉnh quy mô nhỏ.
Các API cấp cao của các framework deep learning
cung cấp nhiều mô hình
được pretrained trên bộ dữ liệu ImageNet.
Ở đây, chúng ta chọn
một mô hình ResNet-34 pretrained,
trong đó chúng ta chỉ đơn giản tái sử dụng
đầu vào của output layer của mô hình này
(tức là các
đặc trưng đã trích xuất).
Sau đó, chúng ta có thể thay output layer ban đầu bằng một mạng output tùy chỉnh nhỏ
có thể được huấn luyện,
chẳng hạn như xếp chồng hai
fully connected layer.
Khác với thí nghiệm trong
:numref:`sec_fine_tuning`,
phần sau không
retrain mô hình pretrained được dùng để trích xuất
đặc trưng. Điều này làm giảm thời gian huấn luyện và
dung lượng bộ nhớ cần để lưu gradient.

Hãy nhớ rằng chúng ta
chuẩn hóa các ảnh bằng cách sử dụng
giá trị trung bình và độ lệch chuẩn của ba kênh RGB trên toàn bộ bộ dữ liệu ImageNet.
Trên thực tế,
điều này cũng nhất quán với phép chuẩn hóa
của mô hình pretrained trên ImageNet.

```{.python .input}
#@tab mxnet
def get_net(devices):
    finetune_net = gluon.model_zoo.vision.resnet34_v2(pretrained=True)
    # Define a new output network
    finetune_net.output_new = nn.HybridSequential(prefix='')
    finetune_net.output_new.add(nn.Dense(256, activation='relu'))
    # There are 120 output categories
    finetune_net.output_new.add(nn.Dense(120))
    # Initialize the output network
    finetune_net.output_new.initialize(init.Xavier(), ctx=devices)
    # Distribute the model parameters to the CPUs or GPUs used for computation
    finetune_net.collect_params().reset_ctx(devices)
    return finetune_net
```

```{.python .input}
#@tab pytorch
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # Define a new output network (there are 120 output categories)
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # Move the model to devices
    finetune_net = finetune_net.to(devices[0])
    # Freeze parameters of feature layers
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

Trước khi [**tính loss**],
trước tiên chúng ta lấy đầu vào của output layer của mô hình pretrained, tức là đặc trưng đã trích xuất.
Sau đó, chúng ta sử dụng đặc trưng này làm đầu vào cho mạng output tùy chỉnh nhỏ để tính loss.

```{.python .input}
#@tab mxnet
loss = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        X_shards, y_shards = d2l.split_batch(features, labels, devices)
        output_features = [net.features(X_shard) for X_shard in X_shards]
        outputs = [net.output_new(feature) for feature in output_features]
        ls = [loss(output, y_shard).sum() for output, y_shard
              in zip(outputs, y_shards)]
        l_sum += sum([float(l.sum()) for l in ls])
        n += labels.size
    return l_sum / n
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return l_sum / n
```

## Định nghĩa [**hàm huấn luyện**]

Chúng ta sẽ chọn mô hình và điều chỉnh hyperparameter dựa trên hiệu suất của mô hình trên tập validation. Hàm huấn luyện mô hình `train` chỉ
lặp qua các parameter của mạng output tùy chỉnh nhỏ.

```{.python .input}
#@tab mxnet
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Only train the small custom output network
    trainer = gluon.Trainer(net.output_new.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            X_shards, y_shards = d2l.split_batch(features, labels, devices)
            output_features = [net.features(X_shard) for X_shard in X_shards]
            with autograd.record():
                outputs = [net.output_new(feature)
                           for feature in output_features]
                ls = [loss(output, y_shard).sum() for output, y_shard
                      in zip(outputs, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            metric.add(sum([float(l.sum()) for l in ls]), labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss))
    measures = f'train loss {metric[0] / metric[1]:.3f}'
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Only train the small custom output network
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**Huấn luyện và validation mô hình**]

Bây giờ chúng ta có thể huấn luyện và thực hiện validation cho mô hình.
Tất cả hyperparameter sau đây đều có thể điều chỉnh.
Ví dụ, có thể tăng số epoch. Vì `lr_period` và `lr_decay` lần lượt được đặt thành 2 và 0.9, learning rate của thuật toán tối ưu hóa sẽ được nhân với 0.9 sau mỗi 2 epoch.

```{.python .input}
#@tab mxnet
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 5e-3, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## [**Phân loại tập kiểm tra**] và gửi kết quả lên Kaggle


Tương tự như bước cuối cùng trong :numref:`sec_kaggle_cifar10`,
cuối cùng, toàn bộ dữ liệu có nhãn (bao gồm cả tập validation) được sử dụng để huấn luyện mô hình và phân loại tập kiểm tra.
Chúng ta sẽ sử dụng mạng output tùy chỉnh đã huấn luyện
để phân loại.

```{.python .input}
#@tab mxnet
net = get_net(devices)
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output_features = net.features(data.as_in_ctx(devices[0]))
    output = npx.softmax(net.output_new(output_features))
    preds.extend(output.asnumpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

```{.python .input}
#@tab pytorch
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

Đoạn code trên
sẽ tạo một tệp `submission.csv`
để gửi
lên Kaggle theo cách được mô tả trong :numref:`sec_kaggle_house`.


## Tóm tắt


* Các ảnh trong bộ dữ liệu ImageNet lớn hơn (với kích thước thay đổi) so với ảnh CIFAR-10. Chúng ta có thể điều chỉnh các phép toán tăng cường ảnh cho các tác vụ trên những bộ dữ liệu khác nhau.
* Để phân loại một tập con của bộ dữ liệu ImageNet, chúng ta có thể tận dụng các mô hình pretrained trên toàn bộ bộ dữ liệu ImageNet để trích xuất đặc trưng và chỉ huấn luyện một mạng output tùy chỉnh quy mô nhỏ. Điều này sẽ giảm thời gian tính toán và chi phí bộ nhớ.


## Bài tập

1. Khi sử dụng toàn bộ bộ dữ liệu của cuộc thi Kaggle, bạn có thể đạt được kết quả nào khi tăng `batch_size` (batch size) và `num_epochs` (số epoch), đồng thời đặt một số hyperparameter khác thành `lr = 0.01`, `lr_period = 10` và `lr_decay = 0.1`?
1. Bạn có đạt được kết quả tốt hơn nếu sử dụng một mô hình pretrained sâu hơn không? Bạn điều chỉnh hyperparameter như thế nào? Bạn có thể tiếp tục cải thiện kết quả không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/380)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1481)
:end_tab: