# Phân đoạn ngữ nghĩa và tập dữ liệu
:label:`sec_semantic_segmentation`

Khi thảo luận về các tác vụ phát hiện đối tượng
trong :numref:`sec_bbox`--:numref:`sec_rcnn`,
các hộp giới hạn hình chữ nhật
được sử dụng để gán nhãn và dự đoán các đối tượng trong ảnh.
Phần này sẽ thảo luận về bài toán *phân đoạn ngữ nghĩa*,
tập trung vào cách chia một ảnh thành các vùng thuộc những lớp ngữ nghĩa khác nhau.
Khác với phát hiện đối tượng,
phân đoạn ngữ nghĩa
nhận diện và hiểu
những gì có trong ảnh ở cấp độ pixel:
việc gán nhãn và dự đoán các vùng ngữ nghĩa của nó đều được thực hiện
ở cấp độ pixel.
:numref:`fig_segmentation` cho thấy các nhãn
của chó, mèo và nền trong ảnh khi phân đoạn ngữ nghĩa.
So với phát hiện đối tượng,
các đường biên ở cấp độ pixel được gán nhãn
trong phân đoạn ngữ nghĩa rõ ràng có độ chi tiết cao hơn.


![Nhãn của chó, mèo và nền trong ảnh khi phân đoạn ngữ nghĩa.](../img/segmentation.svg)
:label:`fig_segmentation`


## Phân đoạn ảnh và phân đoạn thể hiện

Ngoài ra còn có hai tác vụ quan trọng
trong lĩnh vực thị giác máy tính tương tự như phân đoạn ngữ nghĩa,
đó là phân đoạn ảnh và phân đoạn thể hiện.
Sau đây, chúng ta sẽ
phân biệt ngắn gọn chúng với phân đoạn ngữ nghĩa.

* *Phân đoạn ảnh* chia một ảnh thành một số vùng cấu thành. Các phương pháp cho loại bài toán này thường tận dụng mối tương quan giữa các pixel trong ảnh. Trong quá trình huấn luyện, phương pháp này không cần thông tin nhãn về các pixel của ảnh, và không thể đảm bảo rằng các vùng được phân đoạn sẽ có ngữ nghĩa mà chúng ta mong muốn trong quá trình dự đoán. Lấy ảnh trong :numref:`fig_segmentation` làm đầu vào, phân đoạn ảnh có thể chia chú chó thành hai vùng: một vùng bao phủ miệng và mắt, chủ yếu có màu đen, còn vùng kia bao phủ phần còn lại của cơ thể, chủ yếu có màu vàng.
* *Phân đoạn thể hiện* còn được gọi là *phát hiện và phân đoạn đồng thời*. Phương pháp này nghiên cứu cách nhận diện các vùng ở cấp độ pixel của từng thể hiện đối tượng trong một ảnh. Khác với phân đoạn ngữ nghĩa, phân đoạn thể hiện không chỉ cần phân biệt các ngữ nghĩa mà còn phải phân biệt các thể hiện đối tượng khác nhau. Ví dụ, nếu trong ảnh có hai chú chó, phân đoạn thể hiện cần phân biệt một pixel thuộc về chú chó nào trong hai chú chó đó.



## Tập dữ liệu phân đoạn ngữ nghĩa Pascal VOC2012

[**Một trong những tập dữ liệu phân đoạn ngữ nghĩa quan trọng nhất
là [Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).**]
Sau đây,
chúng ta sẽ tìm hiểu tập dữ liệu này.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
```

Tệp tar của tập dữ liệu có dung lượng khoảng 2 GB,
vì vậy có thể sẽ mất một khoảng thời gian để tải tệp xuống.
Tập dữ liệu sau khi giải nén được đặt tại `../data/VOCdevkit/VOC2012`.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
```

Sau khi truy cập đường dẫn `../data/VOCdevkit/VOC2012`,
chúng ta có thể thấy các thành phần khác nhau của tập dữ liệu.
Đường dẫn `ImageSets/Segmentation` chứa các tệp văn bản
chỉ định các mẫu huấn luyện và kiểm tra,
trong khi các đường dẫn `JPEGImages` và `SegmentationClass`
lần lượt lưu trữ ảnh đầu vào và nhãn cho từng ví dụ.
Nhãn ở đây cũng ở định dạng ảnh,
với cùng kích thước
như ảnh đầu vào được gán nhãn.
Ngoài ra,
các pixel có cùng màu trong bất kỳ ảnh nhãn nào đều thuộc cùng một lớp ngữ nghĩa.
Sau đây định nghĩa hàm `read_voc_images` để [**đọc tất cả ảnh đầu vào và nhãn vào bộ nhớ**].

```{.python .input}
#@tab mxnet
#@save
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(image.imread(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(image.imread(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png')))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

```{.python .input}
#@tab pytorch
#@save
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

Chúng ta [**vẽ năm ảnh đầu vào đầu tiên cùng với các nhãn của chúng**].
Trong các ảnh nhãn, màu trắng và đen lần lượt biểu thị đường biên và nền, còn các màu khác tương ứng với những lớp khác nhau.

```{.python .input}
#@tab mxnet
n = 5
imgs = train_features[:n] + train_labels[:n]
d2l.show_images(imgs, 2, n);
```

```{.python .input}
#@tab pytorch
n = 5
imgs = train_features[:n] + train_labels[:n]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, n);
```

Tiếp theo, chúng ta [**liệt kê
các giá trị màu RGB và tên lớp**]
cho tất cả các nhãn trong tập dữ liệu này.

```{.python .input}
#@tab all
#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
```

Với hai hằng số được định nghĩa ở trên,
chúng ta có thể thuận tiện
[**tìm chỉ số lớp cho từng pixel trong một nhãn**].
Chúng ta định nghĩa hàm `voc_colormap2label`
để xây dựng ánh xạ từ các giá trị màu RGB ở trên
sang các chỉ số lớp,
và hàm `voc_label_indices`
để ánh xạ bất kỳ giá trị RGB nào
sang chỉ số lớp tương ứng trong tập dữ liệu Pascal VOC2012 này.

```{.python .input}
#@tab mxnet
#@save
def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = np.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.astype(np.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

```{.python .input}
#@tab pytorch
#@save
def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

[**Ví dụ**], trong ảnh ví dụ đầu tiên,
chỉ số lớp của phần trước của máy bay là 1,
trong khi chỉ số của nền là 0.

```{.python .input}
#@tab all
y = voc_label_indices(train_labels[0], voc_colormap2label())
y[105:115, 130:140], VOC_CLASSES[1]
```

### Tiền xử lý dữ liệu

Trong các thí nghiệm trước đây,
chẳng hạn như trong :numref:`sec_alexnet`--:numref:`sec_googlenet`,
ảnh được thay đổi tỷ lệ
để phù hợp với hình dạng đầu vào mà mô hình yêu cầu.
Tuy nhiên, trong phân đoạn ngữ nghĩa,
việc này
đòi hỏi phải thay đổi tỷ lệ các lớp pixel được dự đoán
trở lại hình dạng ban đầu của ảnh đầu vào.
Việc thay đổi tỷ lệ như vậy có thể không chính xác,
đặc biệt đối với các vùng được phân đoạn thuộc những lớp khác nhau. Để tránh vấn đề này,
chúng ta cắt ảnh về một hình dạng *cố định* thay vì thay đổi tỷ lệ. Cụ thể, [**sử dụng phép cắt ngẫu nhiên trong tăng cường ảnh, chúng ta cắt cùng một vùng của
ảnh đầu vào và nhãn**].

```{.python .input}
#@tab mxnet
#@save
def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images."""
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label
```

```{.python .input}
#@tab pytorch
#@save
def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images."""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label
```

```{.python .input}
#@tab mxnet
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

```{.python .input}
#@tab pytorch
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

imgs = [img.permute(1, 2, 0) for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

### [**Lớp tập dữ liệu phân đoạn ngữ nghĩa tùy chỉnh**]

Chúng ta định nghĩa một lớp tập dữ liệu phân đoạn ngữ nghĩa tùy chỉnh `VOCSegDataset` bằng cách kế thừa lớp `Dataset` do các API cấp cao cung cấp.
Bằng cách triển khai hàm `__getitem__`,
chúng ta có thể tùy ý truy cập ảnh đầu vào có chỉ số `idx` trong tập dữ liệu và chỉ số lớp của từng pixel trong ảnh này.
Do một số ảnh trong tập dữ liệu
có kích thước nhỏ hơn
kích thước đầu ra của phép cắt ngẫu nhiên,
các ví dụ này được lọc bỏ
bằng hàm `filter` tùy chỉnh.
Ngoài ra, chúng ta cũng
định nghĩa hàm `normalize_image` để
chuẩn hóa giá trị của ba kênh RGB của ảnh đầu vào.

```{.python .input}
#@tab mxnet
#@save
class VOCSegDataset(gluon.data.Dataset):
    """A customized dataset to load the VOC dataset."""
    def __init__(self, is_train, crop_size, voc_dir):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature.transpose(2, 0, 1),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class VOCSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the VOC dataset."""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

### [**Đọc tập dữ liệu**]

Chúng ta sử dụng lớp `VOCSegDatase`t tùy chỉnh để
tạo các thể hiện lần lượt cho tập huấn luyện và tập kiểm tra.
Giả sử rằng
chúng ta chỉ định hình dạng đầu ra của các ảnh được cắt ngẫu nhiên là $320\times 480$.
Bên dưới, chúng ta có thể xem số lượng ví dụ
được giữ lại trong tập huấn luyện và tập kiểm tra.

```{.python .input}
#@tab all
crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)
```

Đặt kích thước batch là 64,
chúng ta định nghĩa iterator dữ liệu cho tập huấn luyện.
Hãy in hình dạng của minibatch đầu tiên.
Khác với phân loại ảnh hoặc phát hiện đối tượng, các nhãn ở đây là các tensor ba chiều.

```{.python .input}
#@tab mxnet
batch_size = 64
train_iter = gluon.data.DataLoader(voc_train, batch_size, shuffle=True,
                                   last_batch='discard',
                                   num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

```{.python .input}
#@tab pytorch
batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

### [**Kết hợp tất cả**]

Cuối cùng, chúng ta định nghĩa hàm `load_data_voc` sau đây
để tải xuống và đọc tập dữ liệu phân đoạn ngữ nghĩa Pascal VOC2012.
Hàm này trả về các iterator dữ liệu cho cả tập huấn luyện và tập kiểm tra.

```{.python .input}
#@tab mxnet
#@save
def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = gluon.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, last_batch='discard', num_workers=num_workers)
    test_iter = gluon.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        last_batch='discard', num_workers=num_workers)
    return train_iter, test_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter
```

## Tóm tắt

* Phân đoạn ngữ nghĩa nhận diện và hiểu những gì có trong một ảnh ở cấp độ pixel bằng cách chia ảnh thành các vùng thuộc những lớp ngữ nghĩa khác nhau.
* Một trong những tập dữ liệu phân đoạn ngữ nghĩa quan trọng nhất là Pascal VOC2012.
* Trong phân đoạn ngữ nghĩa, vì ảnh đầu vào và nhãn tương ứng một-một ở cấp độ pixel, ảnh đầu vào được cắt ngẫu nhiên về một hình dạng cố định thay vì thay đổi tỷ lệ.


## Bài tập

1. Phân đoạn ngữ nghĩa có thể được áp dụng như thế nào trong xe tự hành và chẩn đoán hình ảnh y tế? Bạn có thể nghĩ đến những ứng dụng nào khác?
1. Hãy nhớ lại phần mô tả về tăng cường dữ liệu trong :numref:`sec_image_augmentation`. Phương pháp tăng cường ảnh nào được sử dụng trong phân loại ảnh sẽ không khả thi khi áp dụng cho phân đoạn ngữ nghĩa?


:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/375)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1480)
:end_tab: