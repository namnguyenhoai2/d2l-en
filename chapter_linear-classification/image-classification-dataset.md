```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Bộ dữ liệu Phân loại Hình ảnh
:label:`sec_fashion_mnist`

(~~Bộ dữ liệu MNIST là một trong những bộ dữ liệu được sử dụng rộng rãi cho phân loại hình ảnh, nhưng nó quá đơn giản để làm bộ dữ liệu benchmark. Chúng ta sẽ sử dụng bộ dữ liệu Fashion-MNIST tương tự nhưng phức tạp hơn ~~)

Một bộ dữ liệu được sử dụng rộng rãi cho phân loại hình ảnh là bộ dữ liệu [MNIST](https://en.wikipedia.org/wiki/MNIST_database) :cite:`LeCun.Bottou.Bengio.ea.1998` gồm các chữ số viết tay. Vào thời điểm được phát hành trong những năm 1990, nó là một thách thức đáng gờm đối với hầu hết các thuật toán machine learning, bao gồm 60.000 hình ảnh có độ phân giải $28 \times 28$ pixel (cùng một bộ dữ liệu kiểm tra gồm 10.000 hình ảnh). Để dễ hình dung, vào năm 1995, một chiếc Sun SPARCStation 5 với 64MB RAM khổng lồ và tốc độ 5 MFLOPs cực nhanh được xem là thiết bị hiện đại nhất cho machine learning tại AT&T Bell Laboratories. Đạt độ chính xác cao trong nhận dạng chữ số là một thành phần then chốt để tự động hóa việc phân loại thư tín cho USPS vào những năm 1990. Các mạng sâu như LeNet-5 :cite:`LeCun.Jackel.Bottou.ea.1995`, máy vector hỗ trợ với tính bất biến :cite:`Scholkopf.Burges.Vapnik.1996`, và bộ phân loại khoảng cách tiếp tuyến :cite:`Simard.LeCun.Denker.ea.1998` đều có thể đạt tỷ lệ lỗi dưới 1%.

Trong hơn một thập kỷ, MNIST là *điểm* tham chiếu để so sánh các thuật toán machine learning.
Dù từng là một bộ dữ liệu benchmark tốt,
ngay cả các mô hình đơn giản theo tiêu chuẩn ngày nay cũng đạt độ chính xác phân loại trên 95%,
khiến nó không phù hợp để phân biệt giữa các mô hình mạnh và yếu. Hơn nữa, bộ dữ liệu này cho phép đạt mức độ chính xác *rất* cao, điều thường không thấy trong nhiều bài toán phân loại. Điều này làm lệch sự phát triển thuật toán về phía các họ thuật toán cụ thể có thể tận dụng bộ dữ liệu sạch, chẳng hạn như các phương pháp active set và thuật toán active set tìm kiếm biên.
Ngày nay, MNIST đóng vai trò kiểm tra tính hợp lý nhiều hơn là một benchmark. ImageNet :cite:`Deng.Dong.Socher.ea.2009` đặt ra một thách thức phù hợp hơn nhiều. Đáng tiếc là ImageNet quá lớn cho nhiều ví dụ và minh họa trong cuốn sách này, vì việc huấn luyện sẽ mất quá nhiều thời gian để các ví dụ có tính tương tác. Thay vào đó, trong các phần tiếp theo, chúng ta sẽ tập trung thảo luận về bộ dữ liệu Fashion-MNIST :cite:`Xiao.Rasul.Vollgraf.2017` tương tự về mặt định tính nhưng nhỏ hơn nhiều, được phát hành vào năm 2017. Bộ dữ liệu này chứa hình ảnh của 10 loại trang phục ở độ phân giải $28 \times 28$ pixel.

```{.python .input}
%%tab mxnet
%matplotlib inline
import time
from d2l import mxnet as d2l
from mxnet import gluon, npx
from mxnet.gluon.data.vision import transforms
npx.set_np()

d2l.use_svg_display()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
import time
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms

d2l.use_svg_display()
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
import time
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import time
import tensorflow as tf
import tensorflow_datasets as tfds

d2l.use_svg_display()
```

## Tải Bộ dữ liệu

Vì bộ dữ liệu Fashion-MNIST rất hữu ích, tất cả các framework lớn đều cung cấp các phiên bản đã được tiền xử lý của nó. Chúng ta có thể [**tải xuống và đọc nó vào bộ nhớ bằng các tiện ích tích hợp sẵn của framework.**]

```{.python .input}
%%tab mxnet
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = gluon.data.vision.FashionMNIST(
            train=True).transform_first(trans)
        self.val = gluon.data.vision.FashionMNIST(
            train=False).transform_first(trans)
```

```{.python .input}
%%tab pytorch
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
```

```{.python .input}
%%tab tensorflow, jax
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()
```

Fashion-MNIST gồm hình ảnh từ 10 loại, mỗi loại được biểu diễn
bởi 6000 hình ảnh trong bộ dữ liệu huấn luyện và 1000 hình ảnh trong bộ dữ liệu kiểm tra.
Một *bộ dữ liệu kiểm tra* được dùng để đánh giá hiệu năng mô hình (không được dùng để huấn luyện).
Do đó, tập huấn luyện và tập kiểm tra
lần lượt chứa 60.000 và 10.000 hình ảnh.

```{.python .input}
%%tab mxnet, pytorch
data = FashionMNIST(resize=(32, 32))
len(data.train), len(data.val)
```

```{.python .input}
%%tab tensorflow, jax
data = FashionMNIST(resize=(32, 32))
len(data.train[0]), len(data.val[0])
```

Các hình ảnh là thang độ xám và được nâng cấp lên độ phân giải $32 \times 32$ pixel ở trên. Điều này tương tự bộ dữ liệu MNIST gốc, vốn gồm các hình ảnh đen trắng (nhị phân). Tuy nhiên, lưu ý rằng hầu hết dữ liệu hình ảnh hiện đại có ba kênh (đỏ, lục, lam), và hình ảnh siêu phổ có thể có hơn 100 kênh (cảm biến HyMap có 126 kênh).
Theo quy ước, chúng ta lưu một hình ảnh dưới dạng tensor $c \times h \times w$, trong đó $c$ là số kênh màu, $h$ là chiều cao và $w$ là chiều rộng.

```{.python .input}
%%tab all
data.train[0][0].shape
```

[~~Hai hàm tiện ích để trực quan hóa bộ dữ liệu~~]

Các loại trong Fashion-MNIST có tên mà con người có thể hiểu được.
Phương thức tiện lợi sau đây chuyển đổi giữa nhãn số và tên của chúng.

```{.python .input}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

## Đọc một Minibatch

Để việc đọc từ tập huấn luyện và tập kiểm tra dễ dàng hơn,
chúng ta sử dụng data iterator tích hợp sẵn thay vì tạo mới từ đầu.
Hãy nhớ rằng ở mỗi lần lặp, một data iterator
[**đọc một minibatch dữ liệu có kích thước `batch_size`.**]
Chúng ta cũng xáo trộn ngẫu nhiên các ví dụ cho data iterator của dữ liệu huấn luyện.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return gluon.data.DataLoader(data, self.batch_size, shuffle=train,
                                 num_workers=self.num_workers)
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
```

```{.python .input}
%%tab tensorflow, jax
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
    shuffle_buf = len(data[0]) if train else 1
    if tab.selected('tensorflow'):
        return tf.data.Dataset.from_tensor_slices(process(*data)).batch(
            self.batch_size).map(resize_fn).shuffle(shuffle_buf)
    if tab.selected('jax'):
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(process(*data)).batch(
                self.batch_size).map(resize_fn).shuffle(shuffle_buf))
```

Để xem cách hoạt động, hãy tải một minibatch hình ảnh bằng cách gọi phương thức `train_dataloader`. Nó chứa 64 hình ảnh.

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
```

Hãy xem thời gian cần để đọc các hình ảnh. Dù là một loader tích hợp sẵn, nó không nhanh vượt trội. Tuy vậy, điều này là đủ vì xử lý hình ảnh bằng một mạng sâu mất nhiều thời gian hơn đáng kể. Do đó, nó đủ tốt để việc huấn luyện mạng không bị giới hạn bởi I/O.

```{.python .input}
%%tab all
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
```

## Trực quan hóa

Chúng ta sẽ thường xuyên sử dụng bộ dữ liệu Fashion-MNIST. Có thể dùng hàm tiện lợi `show_images` để trực quan hóa các hình ảnh và nhãn liên quan.
Bỏ qua chi tiết triển khai, chúng ta chỉ hiển thị interface bên dưới: với các hàm tiện ích như vậy, chúng ta chỉ cần biết cách gọi `d2l.show_images` thay vì biết cách nó hoạt động.

```{.python .input}
%%tab all
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    raise NotImplementedError
```

Hãy tận dụng nó. Nhìn chung, trực quan hóa và kiểm tra dữ liệu mà bạn đang huấn luyện là một ý tưởng tốt.
Con người rất giỏi phát hiện những điểm bất thường, và vì vậy, trực quan hóa đóng vai trò như một biện pháp bảo vệ bổ sung trước các sai sót và lỗi trong thiết kế thí nghiệm. Dưới đây là [**các hình ảnh và nhãn tương ứng của chúng**] (dạng văn bản)
cho một vài ví dụ đầu tiên trong bộ dữ liệu huấn luyện.

```{.python .input}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    if tab.selected('mxnet', 'pytorch'):
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
    if tab.selected('tensorflow'):
        d2l.show_images(tf.squeeze(X), nrows, ncols, titles=labels)
    if tab.selected('jax'):
        d2l.show_images(jnp.squeeze(X), nrows, ncols, titles=labels)

batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```

Giờ đây, chúng ta đã sẵn sàng làm việc với bộ dữ liệu Fashion-MNIST trong các phần tiếp theo.

## Tóm tắt

Bây giờ, chúng ta có một bộ dữ liệu thực tế hơn một chút để sử dụng cho phân loại. Fashion-MNIST là bộ dữ liệu phân loại trang phục gồm các hình ảnh biểu diễn 10 loại. Chúng ta sẽ sử dụng bộ dữ liệu này trong các phần và chương tiếp theo để đánh giá nhiều thiết kế mạng khác nhau, từ mô hình tuyến tính đơn giản đến các residual network tiên tiến. Như thường làm với hình ảnh, chúng ta đọc chúng dưới dạng tensor có shape (batch size, number of channels, height, width). Hiện tại, chúng ta chỉ có một kênh vì các hình ảnh là thang độ xám (phần trực quan hóa ở trên sử dụng bảng màu giả để tăng khả năng hiển thị).

Cuối cùng, data iterator là một thành phần then chốt để đạt hiệu năng hiệu quả. Chẳng hạn, chúng ta có thể dùng GPU để giải nén hình ảnh hiệu quả, chuyển mã video, hoặc tiền xử lý khác. Bất cứ khi nào có thể, bạn nên dựa vào các data iterator được triển khai tốt, tận dụng tính toán hiệu năng cao để tránh làm chậm training loop của mình.


## Bài tập

1. Việc giảm `batch_size` (chẳng hạn xuống 1) có ảnh hưởng đến hiệu năng đọc không?
1. Hiệu năng của data iterator rất quan trọng. Bạn có nghĩ rằng cách triển khai hiện tại đủ nhanh không? Hãy khám phá các lựa chọn khác nhau để cải thiện nó. Sử dụng system profiler để tìm vị trí các nút thắt cổ chai.
1. Hãy xem tài liệu API trực tuyến của framework. Có những bộ dữ liệu nào khác?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/224)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17980)
:end_tab: