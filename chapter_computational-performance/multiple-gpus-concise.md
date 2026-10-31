# Triển khai súc tích cho nhiều GPU
:label:`sec_multi_gpu_concise`

Việc tự triển khai tính song song từ đầu cho mỗi model mới không hề thú vị. Hơn nữa, việc tối ưu các công cụ đồng bộ hóa để đạt hiệu năng cao mang lại lợi ích đáng kể. Trong phần sau, chúng ta sẽ trình bày cách thực hiện điều này bằng các API cấp cao của các framework deep learning.
Toán học và các thuật toán giống như trong :numref:`sec_multi_gpu`.
Không có gì đáng ngạc nhiên, bạn sẽ cần ít nhất hai GPU để chạy mã trong phần này.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**Một mạng minh họa**]

Hãy sử dụng một mạng có ý nghĩa hơn LeNet trong :numref:`sec_multi_gpu` nhưng vẫn đủ đơn giản và nhanh để train.
Chúng ta chọn một biến thể ResNet-18 :cite:`He.Zhang.Ren.ea.2016`. Vì các ảnh đầu vào rất nhỏ, chúng ta điều chỉnh mạng một chút. Cụ thể, điểm khác biệt so với :numref:`sec_resnet` là chúng ta sử dụng convolution kernel, stride và padding nhỏ hơn ở phần đầu.
Ngoài ra, chúng ta loại bỏ lớp max-pooling.

```{.python .input}
#@tab mxnet
#@save
def resnet18(num_classes):
    """A slightly modified ResNet-18 model."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

```{.python .input}
#@tab pytorch
#@save
def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model."""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(out_channels, use_1x1conv=True, 
                                        strides=2))
            else:
                blk.append(d2l.Residual(out_channels))
        return nn.Sequential(*blk)

    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

## Khởi tạo mạng

:begin_tab:`mxnet`
Hàm `initialize` cho phép chúng ta khởi tạo các parameter trên device tùy chọn.
Để ôn lại các phương pháp khởi tạo, hãy xem :numref:`sec_numerical_stability`. Điều đặc biệt tiện lợi là hàm này cũng cho phép chúng ta khởi tạo mạng đồng thời trên *nhiều* device. Hãy thử xem cách này hoạt động trong thực tế.
:end_tab:

:begin_tab:`pytorch`
Chúng ta sẽ khởi tạo mạng bên trong training loop.
Để ôn lại các phương pháp khởi tạo, hãy xem :numref:`sec_numerical_stability`.
:end_tab:

```{.python .input}
#@tab mxnet
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# Initialize all the parameters of the network
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input}
#@tab pytorch
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# We will initialize the network inside the training loop
```

:begin_tab:`mxnet`
Sử dụng hàm `split_and_load` được giới thiệu trong :numref:`sec_multi_gpu`, chúng ta có thể chia một minibatch dữ liệu và sao chép các phần dữ liệu đến danh sách device do biến `devices` cung cấp. Instance của mạng *tự động* sử dụng GPU thích hợp để tính giá trị của forward propagation. Ở đây, chúng ta tạo 4 observation và chia chúng cho các GPU.
:end_tab:

```{.python .input}
#@tab mxnet
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
Sau khi dữ liệu đi qua mạng, các parameter tương ứng được khởi tạo *trên device mà dữ liệu đã đi qua*.
Điều này có nghĩa là việc khởi tạo diễn ra theo từng device. Vì chúng ta chọn GPU 0 và GPU 1 để khởi tạo, mạng chỉ được khởi tạo trên các device đó, không phải trên CPU. Trên thực tế, các parameter thậm chí không tồn tại trên CPU. Chúng ta có thể kiểm chứng điều này bằng cách in các parameter và quan sát bất kỳ lỗi nào có thể phát sinh.
:end_tab:

```{.python .input}
#@tab mxnet
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

:begin_tab:`mxnet`
Tiếp theo, hãy thay thế mã để [**đánh giá độ chính xác**] bằng mã hoạt động (**song song trên nhiều device**). Mã này thay thế cho hàm `evaluate_accuracy_gpu` từ :numref:`sec_lenet`. Điểm khác biệt chính là chúng ta chia minibatch trước khi gọi mạng. Mọi thứ khác về cơ bản đều giống nhau.
:end_tab:

```{.python .input}
#@tab mxnet
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Compute the accuracy for a model on a dataset using multiple GPUs."""
    # Query the list of devices
    devices = list(net.collect_params().values())[0].list_ctx()
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Run in parallel
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## [**Training**]

Như trước đây, mã training cần thực hiện một số chức năng cơ bản để đạt tính song song hiệu quả:

* Các parameter của mạng cần được khởi tạo trên tất cả device.
* Trong khi duyệt qua dataset, các minibatch cần được chia cho tất cả device.
* Chúng ta tính loss và gradient của nó song song trên các device.
* Các gradient được tổng hợp và các parameter được cập nhật tương ứng.

Cuối cùng, chúng ta tính accuracy (một lần nữa theo cách song song) để báo cáo hiệu năng cuối cùng của mạng. Training routine khá giống với các triển khai trong các chương trước, ngoại trừ việc chúng ta cần chia và tổng hợp dữ liệu.

```{.python .input}
#@tab mxnet
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```

```{.python .input}
#@tab pytorch
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(module):
        if type(module) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)
    # Set the model on multiple GPUs
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

Hãy xem cách này hoạt động trong thực tế. Để khởi động, chúng ta [**train mạng trên một GPU duy nhất.**]

```{.python .input}
#@tab mxnet
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

Tiếp theo, chúng ta [**sử dụng 2 GPU để training**]. So với LeNet
được đánh giá trong :numref:`sec_multi_gpu`,
model ResNet-18 phức tạp hơn đáng kể. Đây là lúc parallelization phát huy lợi thế. Thời gian tính toán lớn hơn đáng kể so với thời gian đồng bộ hóa các parameter. Điều này cải thiện khả năng mở rộng vì overhead của parallelization ít ảnh hưởng hơn.

```{.python .input}
#@tab mxnet
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## Tóm tắt

:begin_tab:`mxnet`
* Gluon cung cấp các primitive để khởi tạo model trên nhiều device bằng cách cung cấp một context list.
:end_tab:
* Dữ liệu được tự động tính toán trên các device nơi dữ liệu đó có thể được tìm thấy.
* Hãy đảm bảo khởi tạo mạng trên từng device trước khi cố gắng truy cập các parameter trên device đó. Nếu không, bạn sẽ gặp lỗi.
* Các thuật toán optimization tự động tổng hợp trên nhiều GPU.



## Bài tập

:begin_tab:`mxnet`
1. Phần này sử dụng ResNet-18. Hãy thử các epoch, batch size và learning rate khác nhau. Sử dụng nhiều GPU hơn để tính toán. Điều gì xảy ra nếu bạn thử với 16 GPU (ví dụ: trên một instance AWS p2.16xlarge)?
1. Đôi khi, các device khác nhau cung cấp năng lực tính toán khác nhau. Chúng ta có thể sử dụng GPU và CPU cùng lúc. Nên chia công việc như thế nào? Việc này có đáng công sức không? Tại sao? Tại sao không?
1. Điều gì xảy ra nếu chúng ta bỏ `npx.waitall()`? Bạn sẽ sửa training như thế nào để có độ chồng lấp tối đa hai bước cho parallelism?
:end_tab:

:begin_tab:`pytorch`
1. Phần này sử dụng ResNet-18. Hãy thử các epoch, batch size và learning rate khác nhau. Sử dụng nhiều GPU hơn để tính toán. Điều gì xảy ra nếu bạn thử với 16 GPU (ví dụ: trên một instance AWS p2.16xlarge)?
1. Đôi khi, các device khác nhau cung cấp năng lực tính toán khác nhau. Chúng ta có thể sử dụng GPU và CPU cùng lúc. Nên chia công việc như thế nào? Việc này có đáng công sức không? Tại sao? Tại sao không?
:end_tab:



:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1403)
:end_tab: