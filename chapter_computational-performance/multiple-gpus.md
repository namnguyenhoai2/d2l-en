# Huấn luyện trên nhiều GPU
:label:`sec_multi_gpu`

Cho đến nay, chúng ta đã thảo luận về cách huấn luyện mô hình hiệu quả trên CPU và GPU. Chúng ta thậm chí đã trình bày cách các framework deep learning cho phép tự động song song hóa việc tính toán và giao tiếp giữa chúng trong :numref:`sec_auto_para`. Chúng ta cũng đã trình bày trong :numref:`sec_use_gpu` cách liệt kê tất cả GPU khả dụng trên máy tính bằng lệnh `nvidia-smi`.
Điều chúng ta *chưa* thảo luận là cách thực sự song song hóa quá trình huấn luyện deep learning.
Thay vào đó, chúng ta ngầm giả định rằng bằng cách nào đó dữ liệu sẽ được chia trên nhiều thiết bị và mọi thứ sẽ hoạt động. Phần này trình bày chi tiết vấn đề đó và chỉ ra cách huấn luyện song song một mạng từ đầu. Chi tiết về cách tận dụng chức năng trong các API cấp cao được trình bày trong :numref:`sec_multi_gpu_concise`.
Chúng tôi giả định rằng bạn đã quen thuộc với các thuật toán stochastic gradient descent trên minibatch, chẳng hạn như những thuật toán được mô tả trong :numref:`sec_minibatch_sgd`.


## Chia nhỏ bài toán

Hãy bắt đầu với một bài toán thị giác máy tính đơn giản và một mạng hơi lỗi thời, chẳng hạn gồm nhiều lớp convolution, pooling và có thể thêm một vài lớp fully connected ở cuối.
Nói cách khác, hãy bắt đầu với một mạng khá giống LeNet :cite:`LeCun.Bottou.Bengio.ea.1998` hoặc AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012`.
Với nhiều GPU (2 nếu là máy chủ desktop, 4 trên một instance AWS g4dn.12xlarge, 8 trên p3.16xlarge hoặc 16 trên p2.16xlarge), chúng ta muốn phân chia quá trình huấn luyện sao cho đạt được mức tăng tốc tốt, đồng thời hưởng lợi từ các lựa chọn thiết kế đơn giản và có thể tái lập. Xét cho cùng, nhiều GPU làm tăng cả khả năng *bộ nhớ* lẫn *tính toán*. Tóm lại, với một minibatch dữ liệu huấn luyện mà chúng ta muốn phân loại, chúng ta có các lựa chọn sau.

Trước tiên, chúng ta có thể phân chia mạng trên nhiều GPU. Nghĩa là mỗi GPU nhận làm đầu vào dữ liệu truyền vào một lớp cụ thể, xử lý dữ liệu qua một số lớp tiếp theo rồi gửi dữ liệu đến GPU kế tiếp.
Cách này cho phép chúng ta xử lý các mạng lớn hơn so với khả năng của một GPU đơn lẻ.
Ngoài ra,
dung lượng bộ nhớ trên mỗi GPU có thể được kiểm soát tốt (đó là một phần của tổng dung lượng bộ nhớ của mạng).

Tuy nhiên, interface giữa các lớp (và do đó giữa các GPU) yêu cầu đồng bộ hóa chặt chẽ. Điều này có thể khó xử lý, đặc biệt khi khối lượng tính toán giữa các lớp không được phân bổ phù hợp. Vấn đề trở nên nghiêm trọng hơn khi có nhiều GPU.
Interface giữa các lớp cũng
yêu cầu truyền một lượng lớn dữ liệu,
chẳng hạn như activations và gradients.
Điều này có thể làm quá tải băng thông của các bus GPU.
Hơn nữa, việc phân chia các phép toán tuần tự nhưng cần nhiều tính toán là không hề đơn giản. Xem :citet:`Mirhoseini.Pham.Le.ea.2017` để biết một nỗ lực tối ưu theo hướng này. Đây vẫn là một bài toán khó và chưa rõ liệu có thể đạt được mức scaling tốt (tuyến tính) trên các bài toán không tầm thường hay không. Chúng tôi không khuyến nghị cách này trừ khi framework hoặc hệ điều hành có hỗ trợ tuyệt vời cho việc kết nối nhiều GPU.


Thứ hai, chúng ta có thể chia công việc theo từng lớp. Chẳng hạn, thay vì tính toán 64 channels trên một GPU duy nhất, chúng ta có thể chia bài toán cho 4 GPU, mỗi GPU tạo dữ liệu cho 16 channels.
Tương tự, đối với một lớp fully connected, chúng ta có thể chia số lượng output units. :numref:`fig_alexnet_original` (lấy từ :citet:`Krizhevsky.Sutskever.Hinton.2012`) minh họa thiết kế này, trong đó chiến lược được sử dụng để xử lý các GPU có dung lượng bộ nhớ rất nhỏ (2 GB vào thời điểm đó).
Cách này cho phép scaling tốt về mặt tính toán, miễn là số lượng channels (hoặc units) không quá nhỏ.
Ngoài ra,
nhiều GPU có thể xử lý các mạng ngày càng lớn hơn, vì bộ nhớ khả dụng tăng tuyến tính.

![Tính song song theo model trong thiết kế AlexNet ban đầu do bộ nhớ GPU bị giới hạn.](../img/alexnet-original.svg)
:label:`fig_alexnet_original`

Tuy nhiên,
chúng ta cần một *số lượng rất lớn* các thao tác đồng bộ hóa hoặc barrier, vì mỗi lớp phụ thuộc vào kết quả từ tất cả các lớp khác.
Hơn nữa, lượng dữ liệu cần truyền có thể còn lớn hơn so với việc phân phối các lớp trên nhiều GPU. Vì vậy, chúng tôi không khuyến nghị cách tiếp cận này do chi phí băng thông và độ phức tạp của nó.

Cuối cùng, chúng ta có thể phân chia dữ liệu trên nhiều GPU. Theo cách này, tất cả GPU thực hiện cùng một loại công việc, nhưng trên các mẫu dữ liệu khác nhau. Các gradients được tổng hợp trên các GPU sau mỗi minibatch dữ liệu huấn luyện.
Đây là cách tiếp cận đơn giản nhất và có thể áp dụng trong mọi tình huống.
Chúng ta chỉ cần đồng bộ hóa sau mỗi minibatch. Tuy nhiên, sẽ rất hữu ích nếu bắt đầu trao đổi các gradient parameters trong khi những gradient khác vẫn đang được tính toán.
Hơn nữa, số lượng GPU lớn hơn dẫn đến minibatch lớn hơn, qua đó nâng cao hiệu quả huấn luyện.
Tuy nhiên, việc bổ sung thêm GPU không cho phép chúng ta huấn luyện các mô hình lớn hơn.


![Tính song song trên nhiều GPU. Từ trái sang phải: bài toán ban đầu, phân chia mạng, phân chia theo lớp, tính song song dữ liệu.](../img/splitting.svg)
:label:`fig_splitting`


:numref:`fig_splitting` mô tả sự so sánh giữa các cách song song hóa khác nhau trên nhiều GPU.
Nhìn chung, tính song song dữ liệu là cách thuận tiện nhất, với điều kiện chúng ta có GPU sở hữu dung lượng bộ nhớ đủ lớn. Xem thêm :cite:`Li.Andersen.Park.ea.2014` để biết mô tả chi tiết về việc phân chia cho distributed training. Bộ nhớ GPU từng là một vấn đề trong những ngày đầu của deep learning. Đến nay, vấn đề này đã được giải quyết trong hầu hết các trường hợp, ngoại trừ những trường hợp bất thường nhất. Trong phần tiếp theo, chúng ta tập trung vào tính song song dữ liệu.

## Tính song song dữ liệu

Giả sử một máy có $k$ GPU. Với model cần huấn luyện, mỗi GPU sẽ duy trì độc lập một bộ đầy đủ các model parameters, mặc dù các giá trị parameters trên các GPU là giống nhau và được đồng bộ hóa.
Ví dụ,
:numref:`fig_data_parallel` minh họa
việc huấn luyện bằng
tính song song dữ liệu khi $k=2$.


![Tính toán stochastic gradient descent trên minibatch bằng tính song song dữ liệu trên hai GPU.](../img/data-parallel.svg)
:label:`fig_data_parallel`

Nhìn chung, quá trình huấn luyện diễn ra như sau:

* Trong mỗi iteration của quá trình huấn luyện, với một minibatch ngẫu nhiên, chúng ta chia các mẫu trong batch thành $k$ phần và phân phối đều chúng trên các GPU.
* Mỗi GPU tính loss và gradient của model parameters dựa trên phần minibatch được giao.
* Các gradients cục bộ từ mỗi GPU trong số $k$ GPU được tổng hợp để thu được stochastic gradient trên minibatch hiện tại.
* Gradient tổng hợp được phân phối lại đến từng GPU.
* Mỗi GPU sử dụng stochastic gradient trên minibatch này để cập nhật toàn bộ model parameters mà nó duy trì.




Lưu ý rằng trong thực tế, chúng ta *tăng* kích thước minibatch lên $k$ lần khi huấn luyện trên $k$ GPU, sao cho mỗi GPU có cùng lượng công việc phải thực hiện như khi chỉ huấn luyện trên một GPU. Trên một máy chủ 16 GPU, điều này có thể làm kích thước minibatch tăng đáng kể và chúng ta có thể phải tăng learning rate tương ứng.
Cũng lưu ý rằng batch normalization trong :numref:`sec_batch_norm` cần được điều chỉnh, chẳng hạn bằng cách duy trì một batch normalization coefficient riêng cho mỗi GPU.
Trong phần tiếp theo, chúng ta sẽ sử dụng một mạng toy để minh họa quá trình huấn luyện trên nhiều GPU.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

## [**Một mạng toy**]

Chúng ta sử dụng LeNet như được giới thiệu trong :numref:`sec_lenet` (với một vài sửa đổi nhỏ). Chúng ta định nghĩa mạng này từ đầu để minh họa chi tiết việc trao đổi parameters và đồng bộ hóa.

```{.python .input}
#@tab mxnet
# Initialize model parameters
scale = 0.01
W1 = np.random.normal(scale=scale, size=(20, 1, 3, 3))
b1 = np.zeros(20)
W2 = np.random.normal(scale=scale, size=(50, 20, 5, 5))
b2 = np.zeros(50)
W3 = np.random.normal(scale=scale, size=(800, 128))
b3 = np.zeros(128)
W4 = np.random.normal(scale=scale, size=(128, 10))
b4 = np.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Define the model
def lenet(X, params):
    h1_conv = npx.convolution(data=X, weight=params[0], bias=params[1],
                              kernel=(3, 3), num_filter=20)
    h1_activation = npx.relu(h1_conv)
    h1 = npx.pooling(data=h1_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2_conv = npx.convolution(data=h1, weight=params[2], bias=params[3],
                              kernel=(5, 5), num_filter=50)
    h2_activation = npx.relu(h2_conv)
    h2 = npx.pooling(data=h2_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = np.dot(h2, params[4]) + params[5]
    h3 = npx.relu(h3_linear)
    y_hat = np.dot(h3, params[6]) + params[7]
    return y_hat

# Cross-entropy loss function
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
# Initialize model parameters
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Define the model
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# Cross-entropy loss function
loss = nn.CrossEntropyLoss(reduction='none')
```

## Đồng bộ hóa dữ liệu

Để huấn luyện hiệu quả trên nhiều GPU, chúng ta cần hai thao tác cơ bản.
Trước tiên, chúng ta cần có khả năng [**phân phối một danh sách parameters đến nhiều thiết bị**] và gắn gradients (`get_params`). Nếu không có parameters, chúng ta không thể đánh giá mạng trên GPU.
Thứ hai, chúng ta cần có khả năng cộng các parameters trên nhiều thiết bị, tức là cần một hàm `allreduce`.

```{.python .input}
#@tab mxnet
def get_params(params, device):
    new_params = [p.copyto(device) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params
```

```{.python .input}
#@tab pytorch
def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params
```

Hãy thử bằng cách sao chép model parameters sang một GPU.

```{.python .input}
#@tab all
new_params = get_params(params, d2l.try_gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)
```

Vì chúng ta chưa thực hiện bất kỳ phép tính nào nên gradient liên quan đến bias parameter vẫn bằng không.
Bây giờ, giả sử chúng ta có một vector được phân phối trên nhiều GPU. Hàm [**`allreduce` cộng tất cả các vector và broadcast kết quả trở lại cho tất cả GPU**] sau đây thực hiện việc đó. Lưu ý rằng để thao tác này hoạt động, chúng ta cần sao chép dữ liệu đến thiết bị thực hiện việc tích lũy kết quả.

```{.python .input}
#@tab mxnet
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].ctx)
    for i in range(1, len(data)):
        data[0].copyto(data[i])
```

```{.python .input}
#@tab pytorch
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)
```

Hãy kiểm thử bằng cách tạo các vector có giá trị khác nhau trên các thiết bị khác nhau rồi tổng hợp chúng.

```{.python .input}
#@tab mxnet
data = [np.ones((1, 2), ctx=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

```{.python .input}
#@tab pytorch
data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

## Phân phối dữ liệu

Chúng ta cần một utility function đơn giản để [**phân phối đều một minibatch trên nhiều GPU**]. Chẳng hạn, với hai GPU, chúng ta muốn sao chép một nửa dữ liệu đến mỗi GPU.
Vì thuận tiện và ngắn gọn hơn, chúng ta sử dụng built-in function từ deep learning framework để thử nghiệm trên một ma trận $4 \times 5$.

```{.python .input}
#@tab mxnet
data = np.arange(20).reshape(4, 5)
devices = [npx.gpu(0), npx.gpu(1)]
split = gluon.utils.split_and_load(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

```{.python .input}
#@tab pytorch
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

Để tái sử dụng sau này, chúng ta định nghĩa một hàm `split_batch` để chia cả dữ liệu và labels.

```{.python .input}
#@tab mxnet
#@save
def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (gluon.utils.split_and_load(X, devices),
            gluon.utils.split_and_load(y, devices))
```

```{.python .input}
#@tab pytorch
#@save
def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
```

## Huấn luyện

Bây giờ chúng ta có thể triển khai [**huấn luyện multi-GPU trên một minibatch duy nhất**]. Việc triển khai này chủ yếu dựa trên phương pháp song song hóa dữ liệu được mô tả trong phần này. Chúng ta sẽ sử dụng các hàm phụ trợ vừa thảo luận, `allreduce` và `split_and_load`, để đồng bộ hóa dữ liệu giữa nhiều GPU. Lưu ý rằng chúng ta không cần viết bất kỳ mã cụ thể nào để đạt được tính song song. Vì computational graph không có bất kỳ dependency nào giữa các device trong một minibatch, nó được thực thi song song *tự động*.

```{.python .input}
#@tab mxnet
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    with autograd.record():  # Loss is calculated separately on each GPU
        ls = [loss(lenet(X_shard, device_W), y_shard)
              for X_shard, y_shard, device_W in zip(
                  X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    for i in range(len(device_params[0])):
        allreduce([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0])  # Here, we use a full-size batch
```

```{.python .input}
#@tab pytorch
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # Loss is calculated separately on each GPU
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # Here, we use a full-size batch
```

Bây giờ, chúng ta có thể định nghĩa [**hàm training**]. Hàm này hơi khác so với các hàm được sử dụng trong những chương trước: chúng ta cần cấp phát các GPU và sao chép tất cả tham số của model lên mọi device.
Rõ ràng, mỗi batch được xử lý bằng hàm `train_batch` để làm việc với nhiều GPU. Để thuận tiện (và giúp mã ngắn gọn hơn), chúng ta tính accuracy trên một GPU duy nhất, mặc dù cách này *không hiệu quả* vì các GPU khác không hoạt động.

```{.python .input}
#@tab mxnet
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            train_batch(X, y, device_params, devices, lr)
            npx.waitall()
        timer.stop()
        # Evaluate the model on GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        # Evaluate the model on GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

Hãy xem cách này hoạt động tốt đến đâu [**trên một GPU duy nhất**].
Trước tiên, chúng ta sử dụng batch size là 256 và learning rate là 0.2.

```{.python .input}
#@tab all
train(num_gpus=1, batch_size=256, lr=0.2)
```

Bằng cách giữ nguyên batch size và learning rate, đồng thời [**tăng số GPU lên 2**], chúng ta có thể thấy rằng test accuracy nhìn chung vẫn gần như không đổi so với
thí nghiệm trước.
Xét về các thuật toán optimization, chúng giống hệt nhau. Đáng tiếc là ở đây không đạt được mức tăng tốc đáng kể nào: model đơn giản là quá nhỏ; hơn nữa, chúng ta chỉ có một dataset nhỏ, trong đó cách triển khai multi-GPU training hơi thiếu tinh gọn của chúng ta phải chịu overhead đáng kể từ Python. Trong các phần tiếp theo, chúng ta sẽ gặp những model phức tạp hơn và các phương pháp parallelization tinh vi hơn.
Dù vậy, hãy xem điều gì xảy ra với Fashion-MNIST.

```{.python .input}
#@tab all
train(num_gpus=2, batch_size=256, lr=0.2)
```

## Tóm tắt

* Có nhiều cách để phân chia quá trình training deep network trên nhiều GPU. Chúng ta có thể phân chia giữa các layer, trên các layer hoặc theo data. Hai cách đầu tiên yêu cầu việc truyền data được phối hợp chặt chẽ. Data parallelism là chiến lược đơn giản nhất.
* Data parallel training khá đơn giản. Tuy nhiên, để đạt hiệu quả, nó làm tăng minibatch size thực tế.
* Trong data parallelism, data được phân chia trên nhiều GPU, mỗi GPU thực hiện forward operation và backward operation riêng, sau đó các gradient được tổng hợp và kết quả được broadcast trở lại các GPU.
* Chúng ta có thể sử dụng learning rate cao hơn một chút cho các minibatch lớn hơn.

## Bài tập

1. Khi training trên $k$ GPU, thay đổi minibatch size từ $b$ thành $k \cdot b$, tức là tăng nó theo số lượng GPU.
1. So sánh accuracy với các learning rate khác nhau. Nó thay đổi như thế nào theo số lượng GPU?
1. Triển khai hàm `allreduce` hiệu quả hơn để tổng hợp các parameter khác nhau trên các GPU khác nhau? Tại sao cách này hiệu quả hơn?
1. Triển khai việc tính test accuracy trên nhiều GPU.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/364)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1669)
:end_tab: