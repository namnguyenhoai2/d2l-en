```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# GPU
:label:`sec_use_gpu`

Trong :numref:`tab_intro_decade`, chúng tôi đã minh họa sự tăng trưởng nhanh chóng
của năng lực tính toán trong hai thập kỷ qua.
Tóm lại, hiệu năng GPU đã tăng
gấp 1000 lần mỗi thập kỷ kể từ năm 2000.
Điều này mang lại những cơ hội to lớn, nhưng cũng cho thấy
đã có nhu cầu đáng kể đối với hiệu năng như vậy.


Trong phần này, chúng ta bắt đầu thảo luận cách khai thác
hiệu năng tính toán này cho nghiên cứu của bạn.
Trước tiên là sử dụng một GPU đơn lẻ và sau đó,
cách sử dụng nhiều GPU và nhiều máy chủ (với nhiều GPU).

Cụ thể, chúng ta sẽ thảo luận cách
sử dụng một GPU NVIDIA đơn lẻ để tính toán.
Trước hết, hãy bảo đảm bạn đã cài đặt ít nhất một GPU NVIDIA.
Sau đó, tải xuống [trình điều khiển NVIDIA và CUDA](https://developer.nvidia.com/cuda-downloads)
và làm theo các lời nhắc để thiết lập đường dẫn phù hợp.
Sau khi hoàn tất các bước chuẩn bị này,
có thể sử dụng lệnh `nvidia-smi`
để (**xem thông tin card đồ họa**).

:begin_tab:`mxnet`
Bạn có thể đã nhận thấy rằng một tensor MXNet
trông gần như giống hệt một `ndarray` NumPy.
Nhưng có một vài khác biệt quan trọng.
Một trong những tính năng then chốt giúp phân biệt MXNet
với NumPy là khả năng hỗ trợ nhiều thiết bị phần cứng khác nhau.

Trong MXNet, mỗi mảng đều có một context.
Cho đến nay, theo mặc định, tất cả biến
và các phép tính liên quan
đều được gán cho CPU.
Thông thường, các context khác có thể là nhiều GPU khác nhau.
Mọi thứ có thể còn phức tạp hơn khi
chúng ta deploy công việc trên nhiều máy chủ.
Bằng cách gán mảng cho các context một cách hợp lý,
chúng ta có thể giảm thiểu thời gian dành cho việc
truyền dữ liệu giữa các thiết bị.
Ví dụ, khi huấn luyện mạng nơ-ron trên máy chủ có GPU,
chúng ta thường muốn các tham số của mô hình nằm trên GPU.

Tiếp theo, chúng ta cần xác nhận rằng
đã cài đặt phiên bản MXNet dành cho GPU.
Nếu đã cài đặt phiên bản MXNet dành cho CPU,
trước tiên chúng ta cần gỡ cài đặt nó.
Ví dụ, sử dụng lệnh `pip uninstall mxnet`,
sau đó cài đặt phiên bản MXNet tương ứng
với phiên bản CUDA của bạn.
Giả sử bạn đã cài đặt CUDA 10.0,
bạn có thể cài đặt phiên bản MXNet
hỗ trợ CUDA 10.0 thông qua `pip install mxnet-cu100`.
:end_tab:

:begin_tab:`pytorch`
Trong PyTorch, mỗi mảng đều có một device; chúng ta thường gọi nó là một *context*.
Cho đến nay, theo mặc định, tất cả biến
và các phép tính liên quan
đều được gán cho CPU.
Thông thường, các context khác có thể là nhiều GPU khác nhau.
Mọi thứ có thể còn phức tạp hơn khi
chúng ta deploy công việc trên nhiều máy chủ.
Bằng cách gán mảng cho các context một cách hợp lý,
chúng ta có thể giảm thiểu thời gian dành cho việc
truyền dữ liệu giữa các thiết bị.
Ví dụ, khi huấn luyện mạng nơ-ron trên máy chủ có GPU,
chúng ta thường muốn các tham số của mô hình nằm trên GPU.
:end_tab:

Để chạy các chương trình trong phần này,
bạn cần ít nhất hai GPU.
Lưu ý rằng điều này có thể là quá mức đối với hầu hết máy tính để bàn,
nhưng lại dễ dàng có được trên đám mây, ví dụ,
bằng cách sử dụng các instance AWS EC2 nhiều GPU.
Hầu như tất cả các phần khác đều *không* yêu cầu nhiều GPU, nhưng ở đây chúng ta chỉ muốn minh họa luồng dữ liệu giữa các thiết bị khác nhau.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
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
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## [**Thiết bị tính toán**]

Chúng ta có thể chỉ định các thiết bị, chẳng hạn như CPU và GPU,
để lưu trữ và tính toán.
Theo mặc định, tensor được tạo trong bộ nhớ chính
và sau đó CPU được sử dụng để tính toán.

:begin_tab:`mxnet`
Trong MXNet, CPU và GPU có thể được biểu thị bằng `cpu()` và `gpu()`.
Cần lưu ý rằng `cpu()`
(hoặc bất kỳ số nguyên nào trong dấu ngoặc đơn)
có nghĩa là tất cả CPU vật lý và bộ nhớ.
Điều này có nghĩa là các phép tính của MXNet
sẽ cố gắng sử dụng tất cả lõi CPU.
Tuy nhiên, `gpu()` chỉ biểu thị một card
và bộ nhớ tương ứng.
Nếu có nhiều GPU, chúng ta dùng `gpu(i)`
để biểu thị GPU thứ $i^\textrm{th}$ ($i$ bắt đầu từ 0).
Ngoài ra, `gpu(0)` và `gpu()` là tương đương.
:end_tab:

:begin_tab:`pytorch`
Trong PyTorch, CPU và GPU có thể được biểu thị bằng `torch.device('cpu')` và `torch.device('cuda')`.
Cần lưu ý rằng thiết bị `cpu`
có nghĩa là tất cả CPU vật lý và bộ nhớ.
Điều này có nghĩa là các phép tính của PyTorch
sẽ cố gắng sử dụng tất cả lõi CPU.
Tuy nhiên, một thiết bị `gpu` chỉ biểu thị một card
và bộ nhớ tương ứng.
Nếu có nhiều GPU, chúng ta dùng `torch.device(f'cuda:{i}')`
để biểu thị GPU thứ $i^\textrm{th}$ ($i$ bắt đầu từ 0).
Ngoài ra, `gpu:0` và `gpu` là tương đương.
:end_tab:

```{.python .input}
%%tab pytorch
def cpu():  #@save
    """Get the CPU device."""
    return torch.device('cpu')

def gpu(i=0):  #@save
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')

cpu(), gpu(), gpu(1)
```

```{.python .input}
%%tab mxnet, tensorflow, jax
def cpu():  #@save
    """Get the CPU device."""
    if tab.selected('mxnet'):
        return npx.cpu()
    if tab.selected('tensorflow'):
        return tf.device('/CPU:0')
    if tab.selected('jax'):
        return jax.devices('cpu')[0]

def gpu(i=0):  #@save
    """Get a GPU device."""
    if tab.selected('mxnet'):
        return npx.gpu(i)
    if tab.selected('tensorflow'):
        return tf.device(f'/GPU:{i}')
    if tab.selected('jax'):
        return jax.devices('gpu')[i]

cpu(), gpu(), gpu(1)
```

Chúng ta có thể (**truy vấn số lượng GPU khả dụng.**)

```{.python .input}
%%tab pytorch
def num_gpus():  #@save
    """Get the number of available GPUs."""
    return torch.cuda.device_count()

num_gpus()
```

```{.python .input}
%%tab mxnet, tensorflow, jax
def num_gpus():  #@save
    """Get the number of available GPUs."""
    if tab.selected('mxnet'):
        return npx.num_gpus()
    if tab.selected('tensorflow'):
        return len(tf.config.experimental.list_physical_devices('GPU'))
    if tab.selected('jax'):
        try:
            return jax.device_count('gpu')
        except:
            return 0  # No GPU backend found

num_gpus()
```

Bây giờ chúng ta [**định nghĩa hai hàm tiện dụng cho phép chạy mã
ngay cả khi các GPU được yêu cầu không tồn tại.**]

```{.python .input}
%%tab all
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]

try_gpu(), try_gpu(10), try_all_gpus()
```

## Tensor và GPU

:begin_tab:`pytorch`
Theo mặc định, tensor được tạo trên CPU.
Chúng ta có thể [**truy vấn thiết bị nơi tensor được đặt.**]
:end_tab:

:begin_tab:`mxnet`
Theo mặc định, tensor được tạo trên CPU.
Chúng ta có thể [**truy vấn thiết bị nơi tensor được đặt.**]
:end_tab:

:begin_tab:`tensorflow, jax`
Theo mặc định, tensor được tạo trên GPU/TPU nếu chúng khả dụng,
nếu không thì CPU sẽ được sử dụng.
Chúng ta có thể [**truy vấn thiết bị nơi tensor được đặt.**]
:end_tab:

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

```{.python .input}
%%tab jax
x = jnp.array([1, 2, 3])
x.device()
```

Điều quan trọng cần lưu ý là bất cứ khi nào chúng ta muốn
thao tác trên nhiều hạng tử,
chúng cần nằm trên cùng một thiết bị.
Ví dụ, nếu cộng hai tensor,
chúng ta cần bảo đảm rằng cả hai đối số
đều nằm trên cùng một thiết bị---nếu không framework
sẽ không biết lưu kết quả ở đâu
hoặc thậm chí quyết định thực hiện phép tính ở đâu.

### Lưu trữ trên GPU

Có một số cách để [**lưu tensor trên GPU.**]
Ví dụ, chúng ta có thể chỉ định thiết bị lưu trữ khi tạo tensor.
Tiếp theo, chúng ta tạo biến tensor `X` trên `gpu` đầu tiên.
Tensor được tạo trên GPU chỉ sử dụng bộ nhớ của GPU này.
Chúng ta có thể dùng lệnh `nvidia-smi` để xem mức sử dụng bộ nhớ GPU.
Nói chung, chúng ta cần bảo đảm không tạo dữ liệu vượt quá giới hạn bộ nhớ GPU.

```{.python .input}
%%tab mxnet
X = np.ones((2, 3), ctx=try_gpu())
X
```

```{.python .input}
%%tab pytorch
X = torch.ones(2, 3, device=try_gpu())
X
```

```{.python .input}
%%tab tensorflow
with try_gpu():
    X = tf.ones((2, 3))
X
```

```{.python .input}
%%tab jax
# By default JAX puts arrays to GPUs or TPUs if available
X = jax.device_put(jnp.ones((2, 3)), try_gpu())
X
```

Giả sử bạn có ít nhất hai GPU, đoạn mã sau sẽ (**tạo một tensor ngẫu nhiên, `Y`, trên GPU thứ hai.**)

```{.python .input}
%%tab mxnet
Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
Y
```

```{.python .input}
%%tab pytorch
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y
```

```{.python .input}
%%tab jax
Y = jax.device_put(jax.random.uniform(jax.random.PRNGKey(0), (2, 3)),
                   try_gpu(1))
Y
```

### Sao chép

[**Nếu muốn tính `X + Y`,
chúng ta cần quyết định thực hiện phép toán này ở đâu.**]
Ví dụ, như minh họa trong :numref:`fig_copyto`,
chúng ta có thể chuyển `X` sang GPU thứ hai
và thực hiện phép toán ở đó.
*Đừng* chỉ cộng `X` và `Y`,
vì điều này sẽ gây ra ngoại lệ.
Runtime engine sẽ không biết phải làm gì:
nó không thể tìm thấy dữ liệu trên cùng một thiết bị và thất bại.
Vì `Y` nằm trên GPU thứ hai,
chúng ta cần chuyển `X` đến đó trước khi có thể cộng hai tensor.

![Sao chép dữ liệu để thực hiện phép toán trên cùng một thiết bị.](../img/copyto.svg)
:label:`fig_copyto`

```{.python .input}
%%tab mxnet
Z = X.copyto(try_gpu(1))
print(X)
print(Z)
```

```{.python .input}
%%tab pytorch
Z = X.cuda(1)
print(X)
print(Z)
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z = X
print(X)
print(Z)
```

```{.python .input}
%%tab jax
Z = jax.device_put(X, try_gpu(1))
print(X)
print(Z)
```

Bây giờ [**dữ liệu (cả `Z` và `Y`) đều nằm trên cùng một GPU, chúng ta có thể cộng chúng lại.**]

```{.python .input}
%%tab all
Y + Z
```

:begin_tab:`mxnet`
Hãy tưởng tượng biến `Z` của bạn đã nằm trên GPU thứ hai.
Điều gì xảy ra nếu chúng ta vẫn gọi `Z.copyto(gpu(1))`?
Nó sẽ tạo một bản sao và cấp phát bộ nhớ mới,
mặc dù biến đó đã nằm trên thiết bị mong muốn.
Đôi khi, tùy thuộc vào môi trường mà mã của chúng ta đang chạy,
hai biến có thể đã nằm trên cùng một thiết bị.
Vì vậy, chúng ta chỉ muốn tạo bản sao nếu các biến
hiện đang nằm trên các thiết bị khác nhau.
Trong các trường hợp này, chúng ta có thể gọi `as_in_ctx`.
Nếu biến đã nằm trên thiết bị được chỉ định
thì đây là một no-op.
Trừ khi bạn đặc biệt muốn tạo bản sao,
`as_in_ctx` là phương thức nên dùng.
:end_tab:

:begin_tab:`pytorch`
Nhưng nếu biến `Z` của bạn đã nằm trên GPU thứ hai thì sao?
Điều gì xảy ra nếu chúng ta vẫn gọi `Z.cuda(1)`?
Nó sẽ trả về `Z` thay vì tạo một bản sao và cấp phát bộ nhớ mới.
:end_tab:

:begin_tab:`tensorflow`
Hãy tưởng tượng biến `Z` của bạn đã nằm trên GPU thứ hai.
Điều gì xảy ra nếu chúng ta vẫn gọi `Z2 = Z` trong cùng device scope?
Nó sẽ trả về `Z` thay vì tạo một bản sao và cấp phát bộ nhớ mới.
:end_tab:

:begin_tab:`jax`
Hãy tưởng tượng biến `Z` của bạn đã nằm trên GPU thứ hai.
Điều gì xảy ra nếu chúng ta vẫn gọi `Z2 = Z` trong cùng device scope?
Nó sẽ trả về `Z` thay vì tạo một bản sao và cấp phát bộ nhớ mới.
:end_tab:

```{.python .input}
%%tab mxnet
Z.as_in_ctx(try_gpu(1)) is Z
```

```{.python .input}
%%tab pytorch
Z.cuda(1) is Z
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z2 = Z
Z2 is Z
```

```{.python .input}
%%tab jax
Z2 = jax.device_put(Z, try_gpu(1))
Z2 is Z
```

### Ghi chú thêm

Mọi người sử dụng GPU để làm machine learning
vì họ kỳ vọng chúng sẽ nhanh.
Nhưng việc truyền biến giữa các thiết bị lại chậm: chậm hơn nhiều so với tính toán.
Vì vậy, chúng tôi muốn bạn chắc chắn 100%
rằng bạn muốn làm một việc chậm trước khi cho phép bạn thực hiện.
Nếu framework deep learning tự động sao chép
mà không gặp lỗi thì bạn có thể không nhận ra
rằng mình đã viết một đoạn mã chậm.

Việc truyền dữ liệu không chỉ chậm mà còn khiến việc song song hóa trở nên khó khăn hơn nhiều,
vì chúng ta phải chờ dữ liệu được gửi đi (hay chính xác hơn là được nhận)
trước khi có thể tiếp tục thực hiện thêm các thao tác.
Đây là lý do các thao tác sao chép cần được thực hiện hết sức cẩn thận.
Theo một quy tắc kinh nghiệm, nhiều thao tác nhỏ
tệ hơn rất nhiều so với một thao tác lớn.
Hơn nữa, thực hiện nhiều thao tác cùng lúc
tốt hơn nhiều so với nhiều thao tác đơn lẻ xen kẽ trong mã
trừ khi bạn biết rõ mình đang làm gì.
Điều này xảy ra vì các thao tác như vậy có thể bị chặn nếu một thiết bị
phải chờ thiết bị kia trước khi có thể làm việc khác.
Điều này hơi giống như xếp hàng gọi cà phê
thay vì đặt trước qua điện thoại
và biết rằng cà phê đã sẵn sàng khi bạn đến.

Cuối cùng, khi chúng ta in tensor hoặc chuyển đổi tensor sang định dạng NumPy,
nếu dữ liệu không nằm trong bộ nhớ chính,
framework trước tiên sẽ sao chép dữ liệu vào bộ nhớ chính,
dẫn đến thêm overhead truyền dữ liệu.
Tệ hơn nữa, lúc này dữ liệu chịu ảnh hưởng của global interpreter lock đáng sợ,
khiến mọi thứ phải chờ Python hoàn tất.


## [**Neural Networks và GPUs**]

Tương tự, một mô hình neural network có thể chỉ định thiết bị.
Đoạn mã sau đặt các tham số mô hình lên GPU.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(1))
net = net.to(device=try_gpu())
```

```{.python .input}
%%tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(1)])

key1, key2 = jax.random.split(jax.random.PRNGKey(0))
x = jax.random.normal(key1, (10,))  # Dummy input
params = net.init(key2, x)  # Initialization call
```

Chúng ta sẽ thấy thêm nhiều ví dụ
về cách chạy mô hình trên GPU trong các chương sau,
đơn giản vì các mô hình sẽ trở nên chuyên sâu về tính toán hơn đôi chút.

Ví dụ, khi đầu vào là một tensor trên GPU, mô hình sẽ tính kết quả trên chính GPU đó.

```{.python .input}
%%tab mxnet, pytorch, tensorflow
net(X)
```

```{.python .input}
%%tab jax
net.apply(params, x)
```

Hãy (**xác nhận rằng các tham số mô hình được lưu trữ trên cùng một GPU.**)

```{.python .input}
%%tab mxnet
net[0].weight.data().ctx
```

```{.python .input}
%%tab pytorch
net[0].weight.data.device
```

```{.python .input}
%%tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

```{.python .input}
%%tab jax
print(jax.tree_util.tree_map(lambda x: x.device(), params))
```

Hãy để trainer hỗ trợ GPU.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def set_scratch_params_device(self, device):
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            with autograd.record():
                setattr(self, attr, a.as_in_ctx(device))
            getattr(self, attr).attach_grad()
        if isinstance(a, d2l.Module):
            a.set_scratch_params_device(device)
        if isinstance(a, list):
            for elem in a:
                elem.set_scratch_params_device(device)
```

```{.python .input}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [d2l.to(a, self.gpus[0]) for a in batch]
    return batch

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    if self.gpus:
        if tab.selected('mxnet'):
            model.collect_params().reset_ctx(self.gpus[0])
            model.set_scratch_params_device(self.gpus[0])
        if tab.selected('pytorch'):
            model.to(self.gpus[0])
    self.model = model
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [d2l.to(a, self.gpus[0]) for a in batch]
    return batch
```

Tóm lại, miễn là tất cả dữ liệu và tham số nằm trên cùng một thiết bị, chúng ta có thể huấn luyện mô hình hiệu quả. Trong các chương tiếp theo, chúng ta sẽ thấy một số ví dụ như vậy.

## Tóm tắt

Chúng ta có thể chỉ định thiết bị để lưu trữ và tính toán, chẳng hạn như CPU hoặc GPU.
  Theo mặc định, dữ liệu được tạo trong bộ nhớ chính
  rồi sử dụng CPU để tính toán.
Framework deep learning yêu cầu tất cả dữ liệu đầu vào cho tính toán
  phải nằm trên cùng một thiết bị,
  dù là CPU hay cùng một GPU.
Bạn có thể mất hiệu năng đáng kể nếu di chuyển dữ liệu thiếu cẩn thận.
  Một lỗi điển hình như sau: tính loss
  cho mọi minibatch trên GPU rồi báo cáo lại
  cho người dùng trên command line (hoặc ghi log vào một NumPy `ndarray`)
  sẽ kích hoạt global interpreter lock, làm tất cả GPU bị đình trệ.
  Tốt hơn nhiều là cấp phát bộ nhớ
  để ghi log bên trong GPU và chỉ di chuyển các log lớn hơn.

## Bài tập

1. Hãy thử một tác vụ tính toán lớn hơn, chẳng hạn như nhân các ma trận lớn,
   và xem sự khác biệt về tốc độ giữa CPU và GPU.
   Còn một tác vụ có số lượng phép tính nhỏ thì sao?
1. Chúng ta nên đọc và ghi các tham số mô hình trên GPU như thế nào?
1. Đo thời gian cần thiết để tính 1000
   phép nhân ma trận--ma trận của các ma trận $100 \times 100$
   và ghi log chuẩn Frobenius của ma trận đầu ra từng kết quả một. So sánh với việc giữ log trên GPU và chỉ truyền kết quả cuối cùng.
1. Đo thời gian cần thiết để thực hiện hai phép nhân ma trận--ma trận
   trên hai GPU cùng lúc. So sánh với việc tính tuần tự
   trên một GPU. Gợi ý: bạn sẽ thấy khả năng mở rộng gần như tuyến tính.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/62)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/63)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/270)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17995)
:end_tab: