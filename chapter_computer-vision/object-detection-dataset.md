# Bộ dữ liệu phát hiện đối tượng
:label:`sec_object-detection-dataset`

Không có bộ dữ liệu nhỏ nào như MNIST và Fashion-MNIST trong lĩnh vực phát hiện đối tượng.
Để nhanh chóng minh họa các mô hình phát hiện đối tượng,
[**chúng tôi đã thu thập và gắn nhãn một bộ dữ liệu nhỏ**].
Đầu tiên, chúng tôi chụp ảnh những quả chuối miễn phí tại văn phòng
và tạo ra
1000 ảnh chuối với các góc xoay và kích thước khác nhau.
Sau đó, chúng tôi đặt từng ảnh chuối
ở một vị trí ngẫu nhiên trên một số ảnh nền.
Cuối cùng, chúng tôi gắn nhãn các bounding box cho những quả chuối đó trên ảnh.


## [**Tải xuống bộ dữ liệu**]

Bộ dữ liệu phát hiện chuối cùng với tất cả các tệp ảnh và
tệp nhãn csv có thể được tải xuống trực tiếp từ Internet.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## Đọc bộ dữ liệu

Chúng ta sẽ [**đọc bộ dữ liệu phát hiện chuối**] trong hàm `read_data_bananas`
bên dưới.
Bộ dữ liệu bao gồm một tệp csv chứa
nhãn lớp đối tượng và
tọa độ bounding box ground-truth
tại các góc trên bên trái và dưới bên phải.

```{.python .input}
#@tab mxnet
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```{.python .input}
#@tab pytorch
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

Bằng cách sử dụng hàm `read_data_bananas` để đọc ảnh và nhãn,
lớp `BananasDataset` sau đây
sẽ cho phép chúng ta [**tạo một instance `Dataset` tùy chỉnh**]
để tải bộ dữ liệu phát hiện chuối.

```{.python .input}
#@tab mxnet
#@save
class BananasDataset(gluon.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

Cuối cùng, chúng ta định nghĩa
hàm `load_data_bananas` để [**trả về hai
instance data iterator tương ứng với tập huấn luyện và tập kiểm tra.**]
Đối với tập dữ liệu kiểm tra,
không cần đọc dữ liệu theo thứ tự ngẫu nhiên.

```{.python .input}
#@tab mxnet
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

Hãy [**đọc một minibatch và in shape của
cả ảnh và nhãn**] trong minibatch này.
Shape của minibatch ảnh,
(kích thước batch, số kênh, chiều cao, chiều rộng),
trông khá quen thuộc:
nó giống với trong các tác vụ phân loại ảnh trước đây.
Shape của minibatch nhãn là
(kích thước batch, $m$, 5),
trong đó $m$ là số lượng bounding box lớn nhất
mà bất kỳ ảnh nào trong bộ dữ liệu có thể chứa.

Mặc dù việc tính toán trên các minibatch hiệu quả hơn,
nó yêu cầu tất cả các mẫu ảnh
phải chứa cùng số lượng bounding box để tạo minibatch thông qua phép nối.
Nói chung,
các ảnh có thể có số lượng bounding box khác nhau;
do đó,
các ảnh có ít hơn $m$ bounding box
sẽ được đệm bằng các bounding box không hợp lệ
cho đến khi đạt $m$.
Sau đó,
nhãn của mỗi bounding box được biểu diễn bằng một mảng có độ dài 5.
Phần tử đầu tiên trong mảng là lớp của đối tượng trong bounding box,
trong đó -1 biểu thị một bounding box không hợp lệ dùng để đệm.
Bốn phần tử còn lại của mảng là
các giá trị tọa độ ($x$, $y$)
của góc trên bên trái và góc dưới bên phải
của bounding box (phạm vi nằm trong khoảng từ 0 đến 1).
Đối với bộ dữ liệu chuối,
vì mỗi ảnh chỉ có một bounding box,
nên ta có $m=1$.

```{.python .input}
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## [**Minh họa**]

Hãy minh họa mười ảnh cùng với các bounding box ground-truth đã được gắn nhãn.
Chúng ta có thể thấy rằng góc xoay, kích thước và vị trí của các quả chuối khác nhau giữa tất cả những ảnh này.
Tất nhiên, đây chỉ là một bộ dữ liệu nhân tạo đơn giản.
Trong thực tế, các bộ dữ liệu thế giới thực thường phức tạp hơn nhiều.

```{.python .input}
#@tab mxnet
imgs = (batch[0][:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab pytorch
imgs = (batch[0][:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## Tóm tắt

* Bộ dữ liệu phát hiện chuối mà chúng tôi thu thập có thể được dùng để minh họa các mô hình phát hiện đối tượng.
* Việc tải dữ liệu cho phát hiện đối tượng tương tự như đối với phân loại ảnh. Tuy nhiên, trong phát hiện đối tượng, các nhãn còn chứa thông tin về các bounding box ground-truth, vốn không có trong phân loại ảnh.


## Bài tập

1. Minh họa các ảnh khác có bounding box ground-truth trong bộ dữ liệu phát hiện chuối. Chúng khác nhau như thế nào xét về bounding box và đối tượng?
1. Giả sử chúng ta muốn áp dụng data augmentation, chẳng hạn như random cropping, cho bài toán phát hiện đối tượng. Điều này có thể khác với trong phân loại ảnh như thế nào? Gợi ý: điều gì xảy ra nếu ảnh được crop chỉ chứa một phần nhỏ của đối tượng?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/372)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1608)
:end_tab: