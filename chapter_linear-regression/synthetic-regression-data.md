```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Dữ liệu hồi quy tổng hợp
:label:`sec_synthetic-regression-data`


Machine learning chủ yếu là việc trích xuất thông tin từ dữ liệu.
Vì vậy, bạn có thể tự hỏi: chúng ta có thể học được gì từ dữ liệu tổng hợp?
Mặc dù bản thân chúng ta có thể không quan tâm đến các mẫu
mà chính chúng ta đã đưa vào một mô hình tạo dữ liệu nhân tạo,
nhưng các tập dữ liệu như vậy vẫn hữu ích cho mục đích giảng dạy,
giúp chúng ta đánh giá các thuộc tính của các
thuật toán học và xác nhận rằng các triển khai của mình hoạt động như mong đợi.
Ví dụ, nếu chúng ta tạo dữ liệu mà các tham số đúng đã được biết *a priori*,
thì chúng ta có thể kiểm tra xem mô hình của mình có thực sự khôi phục được chúng hay không.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx, gluon
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import random
import tensorflow as tf
import tensorflow_datasets as tfds
```

## Tạo tập dữ liệu

Trong ví dụ này, để ngắn gọn,
chúng ta sẽ làm việc trong không gian có số chiều thấp.
Đoạn mã sau tạo ra 1000 ví dụ
với các đặc trưng 2 chiều được lấy mẫu
từ một phân phối chuẩn tắc.
Ma trận thiết kế thu được $\mathbf{X}$
thuộc $\mathbb{R}^{1000 \times 2}$. 
Chúng ta tạo mỗi nhãn bằng cách áp dụng
một hàm tuyến tính *đúng theo thực tế*,
làm nhiễu chúng bằng nhiễu cộng $\boldsymbol{\epsilon}$,
được lấy mẫu độc lập và đồng nhất cho mỗi ví dụ:

(**$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \boldsymbol{\epsilon}.$$**)

Để thuận tiện, chúng ta giả sử rằng $\boldsymbol{\epsilon}$ được lấy mẫu
từ một phân phối chuẩn có trung bình $\mu= 0$
và độ lệch chuẩn $\sigma = 0.01$.
Lưu ý rằng đối với thiết kế hướng đối tượng,
chúng ta thêm mã vào phương thức `__init__` của một lớp con của `d2l.DataModule` (được giới thiệu trong :numref:`oo-design-data`). 
Cho phép thiết lập mọi siêu tham số bổ sung là một thực hành tốt.
Chúng ta thực hiện điều này bằng `save_hyperparameters()`. 
`batch_size` sẽ được xác định sau.

```{.python .input}
%%tab all
class SyntheticRegressionData(d2l.DataModule):  #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, 
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        if tab.selected('pytorch') or tab.selected('mxnet'):                
            self.X = d2l.randn(n, len(w))
            noise = d2l.randn(n, 1) * noise
        if tab.selected('tensorflow'):
            self.X = tf.random.normal((n, w.shape[0]))
            noise = tf.random.normal((n, 1)) * noise
        if tab.selected('jax'):
            key = jax.random.PRNGKey(0)
            key1, key2 = jax.random.split(key)
            self.X = jax.random.normal(key1, (n, w.shape[0]))
            noise = jax.random.normal(key2, (n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```

Dưới đây, chúng ta đặt các tham số đúng thành $\mathbf{w} = [2, -3.4]^\top$ và $b = 4.2$.
Sau đó, chúng ta có thể đối chiếu các tham số ước lượng với các giá trị *đúng theo thực tế* này.

```{.python .input}
%%tab all
data = SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
```

[**Mỗi hàng trong `features` gồm một vector trong $\mathbb{R}^2$ và mỗi hàng trong `labels` là một vô hướng.**] Hãy xem phần tử đầu tiên.

```{.python .input}
%%tab all
print('features:', data.X[0],'\nlabel:', data.y[0])
```

## Đọc tập dữ liệu

Việc huấn luyện các mô hình machine learning thường yêu cầu lặp qua một tập dữ liệu nhiều lần,
mỗi lần lấy một minibatch các ví dụ.
Dữ liệu này sau đó được dùng để cập nhật mô hình.
Để minh họa cách hoạt động này, chúng ta
[**triển khai phương thức `get_dataloader`,**]
đăng ký nó trong lớp `SyntheticRegressionData` thông qua `add_to_class` (được giới thiệu trong :numref:`oo-design-utilities`).
Nó (**nhận vào kích thước batch, một ma trận các đặc trưng,
và một vector các nhãn, rồi tạo ra các minibatch có kích thước `batch_size`.**)
Do đó, mỗi minibatch gồm một tuple các đặc trưng và nhãn.
Lưu ý rằng chúng ta cần cân nhắc xem đang ở chế độ huấn luyện hay validation:
ở chế độ đầu tiên, chúng ta muốn đọc dữ liệu theo thứ tự ngẫu nhiên,
trong khi ở chế độ thứ hai, khả năng đọc dữ liệu theo một thứ tự được xác định trước
có thể rất quan trọng cho mục đích debug.

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        if tab.selected('mxnet', 'pytorch', 'jax'):
            batch_indices = d2l.tensor(indices[i: i+self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
        if tab.selected('tensorflow'):
            j = tf.constant(indices[i : i+self.batch_size])
            yield tf.gather(self.X, j), tf.gather(self.y, j)
```

Để hình thành trực giác, hãy kiểm tra minibatch dữ liệu đầu tiên.
Mỗi minibatch các đặc trưng cung cấp cả kích thước và số chiều của các đặc trưng đầu vào.
Tương tự, minibatch các nhãn sẽ có shape tương ứng được xác định bởi `batch_size`.

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

Mặc dù có vẻ không đáng kể, việc gọi
`iter(data.train_dataloader())`
minh họa sức mạnh của thiết kế hướng đối tượng trong Python.
Lưu ý rằng chúng ta đã thêm một phương thức vào lớp `SyntheticRegressionData`
*after* khi tạo đối tượng `data`.
Tuy nhiên, đối tượng vẫn được hưởng lợi từ
việc bổ sung chức năng cho lớp *ex post facto*.

Trong suốt quá trình lặp, chúng ta nhận được các minibatch khác nhau
cho đến khi toàn bộ tập dữ liệu đã được xử lý hết (hãy thử điều này).
Mặc dù cách lặp được triển khai ở trên phù hợp cho mục đích giảng dạy,
nó không hiệu quả theo những cách có thể gây rắc rối cho các bài toán thực tế.
Ví dụ, cách này yêu cầu chúng ta nạp toàn bộ dữ liệu vào bộ nhớ
và thực hiện rất nhiều truy cập bộ nhớ ngẫu nhiên.
Các iterator dựng sẵn được triển khai trong một framework deep learning
hiệu quả hơn đáng kể và có thể xử lý
các nguồn như dữ liệu được lưu trong tệp,
dữ liệu nhận được qua một stream,
và dữ liệu được tạo hoặc xử lý ngay tại thời điểm cần.
Tiếp theo, hãy thử triển khai cùng một phương thức bằng các iterator dựng sẵn.

## Triển khai ngắn gọn data loader

Thay vì tự viết iterator,
chúng ta có thể [**gọi API hiện có trong một framework để nạp dữ liệu.**]
Như trước, chúng ta cần một tập dữ liệu với các đặc trưng `X` và nhãn `y`. 
Ngoài ra, chúng ta đặt `batch_size` trong data loader dựng sẵn
và để nó tự xử lý việc xáo trộn các ví dụ một cách hiệu quả.

:begin_tab:`jax`
JAX tập trung vào API tương tự NumPy với khả năng tăng tốc trên thiết bị và các phép biến đổi hàm, vì vậy ít nhất trong phiên bản hiện tại, nó chưa bao gồm các phương thức nạp dữ liệu. Với các thư viện khác, chúng ta đã có những data loader rất tốt, và JAX khuyến nghị sử dụng chúng thay thế. Ở đây, chúng ta sẽ lấy data loader của TensorFlow và sửa đổi một chút để nó hoạt động với JAX.
:end_tab:

```{.python .input}
%%tab all
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    if tab.selected('mxnet'):
        dataset = gluon.data.ArrayDataset(*tensors)
        return gluon.data.DataLoader(dataset, self.batch_size,
                                     shuffle=train)
    if tab.selected('pytorch'):
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
    if tab.selected('jax'):
        # Use Tensorflow Datasets & Dataloader. JAX or Flax do not provide
        # any dataloading functionality
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(tensors).shuffle(
                buffer_size=shuffle_buffer).batch(self.batch_size))

    if tab.selected('tensorflow'):
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)
```

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```

Data loader mới hoạt động giống như data loader trước đó, ngoại trừ việc nó hiệu quả hơn và có thêm một số chức năng.

```{.python .input  n=4}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

Ví dụ, data loader do API của framework cung cấp
hỗ trợ phương thức dựng sẵn `__len__`,
vì vậy chúng ta có thể truy vấn độ dài của nó,
tức là số lượng batch.

```{.python .input}
%%tab all
len(data.train_dataloader())
```

## Tóm tắt

Data loader là một cách thuận tiện để trừu tượng hóa
quá trình nạp và thao tác với dữ liệu.
Nhờ đó, cùng một *thuật toán* machine learning
có khả năng xử lý nhiều loại và nguồn dữ liệu khác nhau
mà không cần sửa đổi.
Một trong những ưu điểm của data loader
là chúng có thể được kết hợp.
Ví dụ, chúng ta có thể nạp các hình ảnh
rồi áp dụng một bộ lọc hậu xử lý
để cắt hoặc sửa đổi chúng theo những cách khác.
Do đó, data loader có thể được dùng
để mô tả toàn bộ một pipeline xử lý dữ liệu.

Về bản thân mô hình, mô hình tuyến tính hai chiều
gần như là mô hình đơn giản nhất mà chúng ta có thể gặp.
Nó cho phép chúng ta kiểm tra độ chính xác của các mô hình hồi quy
mà không phải lo lắng về việc có quá ít dữ liệu
hoặc một hệ phương trình không xác định.
Chúng ta sẽ tận dụng điều này trong phần tiếp theo.


## Bài tập

1. Điều gì sẽ xảy ra nếu số lượng ví dụ không thể chia hết cho kích thước batch? Làm thế nào bạn thay đổi hành vi này bằng cách chỉ định một đối số khác thông qua API của framework?
1. Giả sử chúng ta muốn tạo một tập dữ liệu khổng lồ, trong đó cả kích thước của vector tham số `w` và số lượng ví dụ `num_examples` đều lớn.
    1. Điều gì xảy ra nếu chúng ta không thể lưu toàn bộ dữ liệu trong bộ nhớ?
    1. Bạn sẽ xáo trộn dữ liệu như thế nào nếu dữ liệu được lưu trên đĩa? Nhiệm vụ của bạn là thiết kế một thuật toán *hiệu quả*, không yêu cầu quá nhiều thao tác đọc hoặc ghi ngẫu nhiên. Gợi ý: Các bộ sinh hoán vị giả ngẫu nhiên [pseudorandom permutation generators](https://en.wikipedia.org/wiki/Pseudorandom_permutation) cho phép bạn thiết kế lại thứ tự mà không cần lưu trữ bảng hoán vị một cách tường minh :cite:`Naor.Reingold.1999`.
1. Triển khai một trình sinh dữ liệu tạo ra dữ liệu mới ngay tại thời điểm cần, mỗi khi iterator được gọi.
1. Bạn sẽ thiết kế một trình sinh dữ liệu ngẫu nhiên như thế nào để nó tạo ra *cùng một* dữ liệu mỗi lần được gọi?


:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/6662)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/6663)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/6664)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17975)
:end_tab: