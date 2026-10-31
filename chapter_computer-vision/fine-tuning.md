# Tinh chỉnh
:label:`sec_fine_tuning`

Trong các chương trước, chúng ta đã thảo luận cách huấn luyện mô hình trên tập dữ liệu huấn luyện Fashion-MNIST chỉ gồm 60000 hình ảnh. Chúng ta cũng đã giới thiệu ImageNet, tập dữ liệu hình ảnh quy mô lớn được sử dụng rộng rãi nhất trong giới học thuật, với hơn 10 triệu hình ảnh và 1000 đối tượng. Tuy nhiên, quy mô của tập dữ liệu mà chúng ta thường gặp nằm giữa hai tập dữ liệu này.


Giả sử chúng ta muốn nhận dạng các loại ghế khác nhau từ hình ảnh, sau đó đề xuất các liên kết mua hàng cho người dùng. 
Một phương pháp khả thi là trước tiên xác định
100 loại ghế phổ biến,
chụp 1000 hình ảnh từ các góc độ khác nhau cho mỗi loại ghế,
sau đó huấn luyện một mô hình classification trên tập dữ liệu hình ảnh đã thu thập.
Mặc dù tập dữ liệu về ghế này có thể lớn hơn tập dữ liệu Fashion-MNIST,
số lượng mẫu vẫn ít hơn một phần mười
số lượng mẫu trong ImageNet.
Điều này có thể dẫn đến overfitting đối với các mô hình phức tạp
vốn phù hợp với ImageNet nhưng được áp dụng cho tập dữ liệu về ghế này.
Ngoài ra, do số lượng mẫu huấn luyện hạn chế,
độ chính xác của mô hình đã huấn luyện
có thể không đáp ứng các yêu cầu thực tế.


Để giải quyết các vấn đề trên,
một giải pháp rõ ràng là thu thập thêm dữ liệu.
Tuy nhiên, việc thu thập và gán nhãn dữ liệu có thể tốn rất nhiều thời gian và tiền bạc.
Ví dụ, để thu thập tập dữ liệu ImageNet, các nhà nghiên cứu đã chi hàng triệu đô la từ nguồn kinh phí nghiên cứu.
Mặc dù chi phí thu thập dữ liệu hiện nay đã giảm đáng kể, chi phí này vẫn không thể xem nhẹ.


Một giải pháp khác là áp dụng *transfer learning* để chuyển giao kiến thức đã học từ *source dataset* sang *target dataset*.
Ví dụ, mặc dù phần lớn hình ảnh trong tập dữ liệu ImageNet không liên quan đến ghế, mô hình được huấn luyện trên tập dữ liệu này có thể trích xuất các đặc trưng hình ảnh tổng quát hơn, qua đó giúp nhận dạng các cạnh, kết cấu, hình dạng và thành phần của đối tượng.
Các đặc trưng tương tự này
cũng có thể hiệu quả trong việc nhận dạng ghế.


## Các bước


Trong phần này, chúng ta sẽ giới thiệu một kỹ thuật phổ biến trong transfer learning: *fine-tuning*. Như minh họa trong :numref:`fig_finetune`, fine-tuning gồm bốn bước sau:


1. Pretrain một mô hình mạng neural, tức *source model*, trên một source dataset (ví dụ: tập dữ liệu ImageNet).
1. Tạo một mô hình mạng neural mới, tức *target model*. Mô hình này sao chép toàn bộ thiết kế mô hình và các tham số của source model, ngoại trừ output layer. Chúng ta giả định rằng các tham số mô hình này chứa kiến thức đã học từ source dataset và kiến thức này cũng có thể áp dụng cho target dataset. Chúng ta cũng giả định rằng output layer của source model có liên quan chặt chẽ đến các nhãn của source dataset; do đó, output layer này không được sử dụng trong target model.
1. Thêm một output layer vào target model, trong đó số lượng output bằng số danh mục trong target dataset. Sau đó, khởi tạo ngẫu nhiên các tham số mô hình của layer này.
1. Huấn luyện target model trên target dataset, chẳng hạn như tập dữ liệu về ghế. Output layer sẽ được huấn luyện từ đầu, trong khi các tham số của tất cả layer khác được fine-tune dựa trên các tham số của source model.

![Tinh chỉnh.](../img/finetune.svg)
:label:`fig_finetune`

Khi target dataset nhỏ hơn nhiều so với source dataset, fine-tuning giúp cải thiện khả năng khái quát hóa của mô hình.


## Nhận dạng bánh mì kẹp xúc xích

Hãy minh họa fine-tuning bằng một trường hợp cụ thể:
nhận dạng bánh mì kẹp xúc xích.
Chúng ta sẽ fine-tune một mô hình ResNet trên một tập dữ liệu nhỏ,
mô hình này đã được pretrain trên tập dữ liệu ImageNet.
Tập dữ liệu nhỏ này gồm
hàng nghìn hình ảnh có và không có bánh mì kẹp xúc xích.
Chúng ta sẽ sử dụng mô hình đã fine-tune để nhận dạng
bánh mì kẹp xúc xích từ hình ảnh.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from torch import nn
import torch
import torchvision
import os
```

### Đọc tập dữ liệu

[**Tập dữ liệu bánh mì kẹp xúc xích mà chúng ta sử dụng được lấy từ các hình ảnh trực tuyến**].
Tập dữ liệu này gồm
1400 hình ảnh thuộc lớp dương có chứa bánh mì kẹp xúc xích,
và số lượng tương đương các hình ảnh thuộc lớp âm có chứa những loại thức ăn khác.
1000 hình ảnh của mỗi lớp được sử dụng để huấn luyện và số còn lại được dùng để kiểm thử.


Sau khi giải nén tập dữ liệu đã tải xuống,
chúng ta nhận được hai thư mục `hotdog/train` và `hotdog/test`. Cả hai thư mục đều có `hotdog` và `not-hotdog` thư mục con, mỗi thư mục con chứa các hình ảnh của
lớp tương ứng.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')
```

Chúng ta tạo hai instance để lần lượt đọc tất cả tệp hình ảnh trong tập dữ liệu huấn luyện và kiểm thử.

```{.python .input}
#@tab mxnet
train_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'train'))
test_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'test'))
```

```{.python .input}
#@tab pytorch
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
```

8 mẫu dương đầu tiên và 8 hình ảnh âm cuối cùng được hiển thị bên dưới. Như bạn có thể thấy, [**các hình ảnh có kích thước và tỷ lệ khung hình khác nhau**].

```{.python .input}
#@tab all
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);
```

Trong quá trình huấn luyện, trước tiên chúng ta cắt một vùng ngẫu nhiên có kích thước và tỷ lệ khung hình ngẫu nhiên từ hình ảnh,
sau đó scale vùng này
thành một hình ảnh đầu vào có kích thước $224 \times 224$.
Trong quá trình kiểm thử, chúng ta scale cả chiều cao và chiều rộng của hình ảnh thành 256 pixel, sau đó cắt một vùng trung tâm có kích thước $224 \times 224$ làm đầu vào.
Ngoài ra,
đối với ba kênh màu RGB (đỏ, lục và lam),
chúng ta *standardize* các giá trị của từng kênh.
Cụ thể,
giá trị trung bình của một kênh được trừ khỏi từng giá trị của kênh đó, sau đó kết quả được chia cho độ lệch chuẩn của kênh.

[~~Tăng cường dữ liệu~~]

```{.python .input}
#@tab mxnet
# Specify the means and standard deviations of the three RGB channels to
# standardize each channel
normalize = gluon.data.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomResizedCrop(224),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    normalize])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    normalize])
```

```{.python .input}
#@tab pytorch
# Specify the means and standard deviations of the three RGB channels to
# standardize each channel
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])
```

### [**Định nghĩa và khởi tạo mô hình**]

Chúng ta sử dụng ResNet-18, đã được pretrain trên tập dữ liệu ImageNet, làm source model. Ở đây, chúng ta chỉ định `pretrained=True` để tự động tải xuống các tham số của mô hình đã pretrain.
Nếu đây là lần đầu tiên sử dụng mô hình này,
cần có kết nối Internet để tải xuống.

```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
```

:begin_tab:`mxnet`
Instance source model đã pretrain chứa hai biến thành viên: `features` và `output`. Biến đầu tiên chứa tất cả layer của mô hình ngoại trừ output layer, còn biến thứ hai là output layer của mô hình.
Mục đích chính của việc phân chia này là tạo điều kiện thuận lợi cho việc fine-tune các tham số mô hình của tất cả layer, ngoại trừ output layer. Biến thành viên `output` của source model được hiển thị bên dưới.
:end_tab:

:begin_tab:`pytorch`
Instance source model đã pretrain chứa một số feature layer và một output layer `fc`.
Mục đích chính của việc phân chia này là tạo điều kiện thuận lợi cho việc fine-tune các tham số mô hình của tất cả layer, ngoại trừ output layer. Biến thành viên `fc` của source model được trình bày bên dưới.
:end_tab:

```{.python .input}
#@tab mxnet
pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net.fc
```

Là một fully connected layer, layer này chuyển đổi các output của global average pooling cuối cùng của ResNet thành 1000 output lớp của tập dữ liệu ImageNet.
Sau đó, chúng ta xây dựng một mạng neural mới làm target model. Mô hình này được định nghĩa giống source model đã pretrain, ngoại trừ
số lượng output ở layer cuối
được đặt thành
số lượng lớp trong target dataset (thay vì 1000).

Trong đoạn code bên dưới, các tham số mô hình trước output layer của instance target model `finetune_net` được khởi tạo bằng các tham số mô hình của những layer tương ứng từ source model.
Vì các tham số mô hình này thu được thông qua quá trình pretrain trên ImageNet,
chúng có hiệu quả.
Do đó, chúng ta chỉ có thể sử dụng
learning rate nhỏ để *fine-tune* các tham số đã pretrain này.
Ngược lại, các tham số mô hình trong output layer được khởi tạo ngẫu nhiên và thường cần learning rate lớn hơn để học từ đầu.
Với learning rate cơ sở là $\eta$, learning rate $10\eta$ sẽ được sử dụng để lặp qua các tham số mô hình trong output layer.

```{.python .input}
#@tab mxnet
finetune_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
# The model parameters in the output layer will be iterated using a learning
# rate ten times greater
finetune_net.output.collect_params().setattr('lr_mult', 10)
```

```{.python .input}
#@tab pytorch
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight);
```

### [**Fine-tune mô hình**]

Trước tiên, chúng ta định nghĩa một hàm huấn luyện `train_fine_tuning` sử dụng fine-tuning để có thể gọi nhiều lần.

```{.python .input}
#@tab mxnet
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gluon.data.DataLoader(
        train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)
    devices = d2l.try_all_gpus()
    net.collect_params().reset_ctx(devices)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

```{.python .input}
#@tab pytorch
# If `param_group=True`, the model parameters in the output layer will be
# updated using a learning rate ten times greater
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)    
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

Chúng ta [**đặt learning rate cơ sở ở một giá trị nhỏ**]
để *fine-tune* các tham số mô hình thu được qua quá trình pretrain. Dựa trên các thiết lập trước đó, chúng ta sẽ huấn luyện các tham số output layer của target model từ đầu bằng learning rate lớn gấp mười lần.

```{.python .input}
#@tab mxnet
train_fine_tuning(finetune_net, 0.01)
```

```{.python .input}
#@tab pytorch
train_fine_tuning(finetune_net, 5e-5)
```

[**Để so sánh,**] chúng ta định nghĩa một mô hình giống hệt, nhưng (**khởi tạo tất cả tham số mô hình bằng các giá trị ngẫu nhiên**). Vì toàn bộ mô hình cần được huấn luyện từ đầu, chúng ta có thể sử dụng learning rate lớn hơn.

```{.python .input}
#@tab mxnet
scratch_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train_fine_tuning(scratch_net, 0.1)
```

```{.python .input}
#@tab pytorch
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
```

Như có thể thấy, mô hình đã fine-tune có xu hướng hoạt động tốt hơn ở cùng một epoch
vì các giá trị tham số ban đầu của nó hiệu quả hơn.


## Tóm tắt

* Transfer learning chuyển giao kiến thức đã học từ source dataset sang target dataset. Fine-tuning là một kỹ thuật phổ biến của transfer learning.
* Target model sao chép toàn bộ thiết kế mô hình cùng các tham số từ source model, ngoại trừ output layer, rồi fine-tune các tham số này dựa trên target dataset. Ngược lại, output layer của target model cần được huấn luyện từ đầu.
* Nhìn chung, việc fine-tune các tham số sử dụng learning rate nhỏ hơn, trong khi huấn luyện output layer từ đầu có thể sử dụng learning rate lớn hơn.


## Bài tập

1. Tiếp tục tăng learning rate của `finetune_net`. Độ chính xác của mô hình thay đổi như thế nào?
2. Tiếp tục điều chỉnh các hyperparameter của `finetune_net` và `scratch_net` trong thí nghiệm so sánh. Độ chính xác của chúng còn khác nhau không?
3. Đặt các tham số trước output layer của `finetune_net` bằng các tham số của source model và *không* cập nhật chúng trong quá trình huấn luyện. Độ chính xác của mô hình thay đổi như thế nào? Bạn có thể sử dụng đoạn code sau.

```{.python .input}
#@tab mxnet
finetune_net.features.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
for param in finetune_net.parameters():
    param.requires_grad = False
```

4. Trên thực tế, có một lớp "hotdog" trong tập dữ liệu `ImageNet`. Tham số trọng số tương ứng của lớp này trong output layer có thể được lấy bằng đoạn code sau. Làm thế nào chúng ta có thể tận dụng tham số trọng số này?

```{.python .input}
#@tab mxnet
weight = pretrained_net.output.weight
hotdog_w = np.split(weight.data(), 1000, axis=0)[713]
hotdog_w.shape
```

```{.python .input}
#@tab pytorch
weight = pretrained_net.fc.weight
hotdog_w = torch.split(weight.data, 1, dim=0)[934]
hotdog_w.shape
```

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/368)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1439)
:end_tab: