# Chuyển giao phong cách bằng mạng nơ-ron

Nếu bạn là người yêu thích nhiếp ảnh,
có thể bạn đã quen thuộc với các bộ lọc.
Chúng có thể thay đổi phong cách màu sắc của ảnh
để ảnh phong cảnh trở nên sắc nét hơn
hoặc làm da trong ảnh chân dung trắng hơn.
Tuy nhiên,
một bộ lọc thường chỉ thay đổi
một khía cạnh của ảnh.
Để áp dụng một phong cách lý tưởng
cho một bức ảnh,
có lẽ bạn cần
thử nhiều tổ hợp bộ lọc khác nhau.
Quá trình này
phức tạp không kém việc tinh chỉnh các hyperparameter của một mô hình.



Trong phần này, chúng ta sẽ
tận dụng các biểu diễn theo từng layer của một CNN
để tự động áp dụng phong cách của một ảnh
lên một ảnh khác, tức là *chuyển giao phong cách* :cite:`Gatys.Ecker.Bethge.2016`.
Tác vụ này cần hai ảnh đầu vào:
một là *ảnh nội dung*
và ảnh còn lại là *ảnh phong cách*.
Chúng ta sẽ sử dụng mạng nơ-ron
để biến đổi ảnh nội dung
sao cho phong cách của nó gần với ảnh phong cách.
Ví dụ,
ảnh nội dung trong :numref:`fig_style_transfer` là một ảnh phong cảnh do chúng tôi chụp
tại Vườn quốc gia Mount Rainier ở ngoại ô Seattle, trong khi ảnh phong cách là một bức tranh sơn dầu
với chủ đề những cây sồi mùa thu.
Trong ảnh tổng hợp đầu ra,
các nét cọ dầu của ảnh phong cách
được áp dụng, tạo ra màu sắc sống động hơn,
đồng thời giữ nguyên hình dạng chính của các đối tượng
trong ảnh nội dung.

![Với ảnh nội dung và ảnh phong cách cho trước, chuyển giao phong cách tạo ra một ảnh tổng hợp.](../img/style-transfer.svg)
:label:`fig_style_transfer`

## Phương pháp

:numref:`fig_style_transfer_model` minh họa
phương pháp chuyển giao phong cách dựa trên CNN bằng một ví dụ đơn giản hóa.
Trước tiên, chúng ta khởi tạo ảnh tổng hợp,
chẳng hạn bằng ảnh nội dung.
Ảnh tổng hợp này là biến duy nhất cần được cập nhật trong quá trình chuyển giao phong cách,
tức là các tham số mô hình cần được cập nhật trong quá trình huấn luyện.
Sau đó, chúng ta chọn một CNN đã được pretrained
để trích xuất các đặc trưng của ảnh và đóng băng
các tham số mô hình của nó trong quá trình huấn luyện.
CNN sâu này sử dụng nhiều layer
để trích xuất
các đặc trưng phân cấp của ảnh.
Chúng ta có thể chọn đầu ra của một số layer này làm đặc trưng nội dung hoặc đặc trưng phong cách.
Lấy :numref:`fig_style_transfer_model` làm ví dụ.
Mạng nơ-ron pretrained ở đây có 3 layer convolution,
trong đó layer thứ hai tạo ra các đặc trưng nội dung,
còn layer thứ nhất và thứ ba tạo ra các đặc trưng phong cách.

![Quy trình chuyển giao phong cách dựa trên CNN. Các đường liền nét biểu thị hướng lan truyền thuận và các đường chấm biểu thị lan truyền ngược. ](../img/neural-style.svg)
:label:`fig_style_transfer_model`

Tiếp theo, chúng ta tính hàm loss của quá trình chuyển giao phong cách thông qua lan truyền thuận (hướng của các mũi tên liền nét), rồi cập nhật các tham số mô hình (ảnh tổng hợp đầu ra) thông qua lan truyền ngược (hướng của các mũi tên nét đứt).
Hàm loss thường được sử dụng trong chuyển giao phong cách gồm ba phần:
(i) *content loss* làm cho ảnh tổng hợp và ảnh nội dung gần nhau về các đặc trưng nội dung;
(ii) *style loss* làm cho ảnh tổng hợp và ảnh phong cách gần nhau về các đặc trưng phong cách;
và (iii) *total variation loss* giúp giảm nhiễu trong ảnh tổng hợp.
Cuối cùng, khi quá trình huấn luyện mô hình kết thúc, chúng ta xuất các tham số mô hình của quá trình chuyển giao phong cách để tạo ra
ảnh tổng hợp cuối cùng.



Trong phần tiếp theo,
chúng ta sẽ giải thích các chi tiết kỹ thuật của chuyển giao phong cách thông qua một thí nghiệm cụ thể.


## [**Đọc ảnh nội dung và ảnh phong cách**]

Trước tiên, chúng ta đọc ảnh nội dung và ảnh phong cách.
Từ các trục tọa độ được in ra,
có thể thấy rằng các ảnh này có kích thước khác nhau.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

d2l.set_figsize()
content_img = image.imread('../img/rainier.jpg')
d2l.plt.imshow(content_img.asnumpy());
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn

d2l.set_figsize()
content_img = d2l.Image.open('../img/rainier.jpg')
d2l.plt.imshow(content_img);
```

```{.python .input}
#@tab mxnet
style_img = image.imread('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img.asnumpy());
```

```{.python .input}
#@tab pytorch
style_img = d2l.Image.open('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img);
```

## [**Tiền xử lý và hậu xử lý**]

Dưới đây, chúng ta định nghĩa hai hàm để tiền xử lý và hậu xử lý ảnh.
Hàm `preprocess` chuẩn hóa
từng kênh trong ba kênh RGB của ảnh đầu vào và biến đổi kết quả thành định dạng đầu vào của CNN.
Hàm `postprocess` khôi phục các giá trị pixel trong ảnh đầu ra về giá trị ban đầu trước khi chuẩn hóa.
Vì hàm in ảnh yêu cầu mỗi pixel có giá trị dấu phẩy động từ 0 đến 1,
chúng ta thay thế mọi giá trị nhỏ hơn 0 hoặc lớn hơn 1 lần lượt bằng 0 hoặc 1.

```{.python .input}
#@tab mxnet
rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)

def postprocess(img):
    img = img[0].as_in_ctx(rgb_std.ctx)
    return (img.transpose(1, 2, 0) * rgb_std + rgb_mean).clip(0, 1)
```

```{.python .input}
#@tab pytorch
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))
```

## [**Trích xuất đặc trưng**]

Chúng ta sử dụng mô hình VGG-19 được pretrained trên tập dữ liệu ImageNet để trích xuất các đặc trưng của ảnh :cite:`Gatys.Ecker.Bethge.2016`.

```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.vgg19(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.vgg19(pretrained=True)
```

Để trích xuất các đặc trưng nội dung và đặc trưng phong cách của ảnh, chúng ta có thể chọn đầu ra của một số layer nhất định trong mạng VGG.
Nói chung, layer càng gần layer đầu vào thì càng dễ trích xuất các chi tiết của ảnh, và ngược lại, càng dễ trích xuất thông tin toàn cục của ảnh. Để tránh
giữ lại quá nhiều chi tiết của ảnh nội dung trong ảnh tổng hợp,
chúng ta chọn một layer VGG gần đầu ra hơn làm *content layer* để tạo ra các đặc trưng nội dung của ảnh.
Chúng ta cũng chọn đầu ra của các layer VGG khác nhau để trích xuất các đặc trưng phong cách cục bộ và toàn cục.
Các layer này còn được gọi là *style layer*.
Như đã đề cập trong :numref:`sec_vgg`,
mạng VGG sử dụng 5 convolutional block.
Trong thí nghiệm, chúng ta chọn layer convolution cuối cùng của convolutional block thứ tư làm content layer, và layer convolution đầu tiên của mỗi convolutional block làm style layer.
Có thể lấy chỉ số của các layer này bằng cách in instance `pretrained_net`.

```{.python .input}
#@tab all
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
```

Khi trích xuất đặc trưng bằng các layer VGG,
chúng ta chỉ cần sử dụng tất cả các layer
từ layer đầu vào đến content layer hoặc style layer gần layer đầu ra nhất.
Hãy xây dựng một instance mạng mới `net`, chỉ giữ lại tất cả các layer VGG sẽ được
sử dụng để trích xuất đặc trưng.

```{.python .input}
#@tab mxnet
net = nn.Sequential()
for i in range(max(content_layers + style_layers) + 1):
    net.add(pretrained_net.features[i])
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])
```

Với đầu vào `X`, nếu chỉ gọi
lan truyền thuận `net(X)`, chúng ta chỉ có thể nhận được đầu ra của layer cuối cùng.
Vì cũng cần các đầu ra của những layer trung gian,
chúng ta cần thực hiện phép tính theo từng layer và lưu lại
đầu ra của content layer và style layer.

```{.python .input}
#@tab all
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles
```

Dưới đây định nghĩa hai hàm:
hàm `get_contents` trích xuất các đặc trưng nội dung từ ảnh nội dung,
còn hàm `get_styles` trích xuất các đặc trưng phong cách từ ảnh phong cách.
Vì không cần cập nhật các tham số mô hình của VGG pretrained trong quá trình huấn luyện,
chúng ta có thể trích xuất các đặc trưng nội dung và phong cách
ngay cả trước khi bắt đầu huấn luyện.
Vì ảnh tổng hợp
là một tập hợp các tham số mô hình cần được cập nhật
cho quá trình chuyển giao phong cách,
chúng ta chỉ có thể trích xuất các đặc trưng nội dung và phong cách của ảnh tổng hợp bằng cách gọi hàm `extract_features` trong quá trình huấn luyện.

```{.python .input}
#@tab mxnet
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).copyto(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).copyto(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

```{.python .input}
#@tab pytorch
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

## [**Định nghĩa hàm loss**]

Bây giờ chúng ta sẽ mô tả hàm loss cho quá trình chuyển giao phong cách. Hàm loss gồm
content loss, style loss và total variation loss.

### Content Loss

Tương tự như hàm loss trong hồi quy tuyến tính,
content loss đo lường sự khác biệt
về các đặc trưng nội dung
giữa ảnh tổng hợp và ảnh nội dung thông qua
hàm loss bình phương.
Hai đầu vào của hàm loss bình phương
đều là
đầu ra của content layer được tính bởi hàm `extract_features`.

```{.python .input}
#@tab mxnet
def content_loss(Y_hat, Y):
    return np.square(Y_hat - Y).mean()
```

```{.python .input}
#@tab pytorch
def content_loss(Y_hat, Y):
    # We detach the target content from the tree used to dynamically compute
    # the gradient: this is a stated value, not a variable. Otherwise the loss
    # will throw an error.
    return torch.square(Y_hat - Y.detach()).mean()
```

### Style Loss

Style loss, tương tự như content loss,
cũng sử dụng hàm loss bình phương để đo lường sự khác biệt về phong cách giữa ảnh tổng hợp và ảnh phong cách.
Để biểu diễn đầu ra phong cách của bất kỳ style layer nào,
trước tiên chúng ta sử dụng hàm `extract_features` để
tính đầu ra của style layer.
Giả sử đầu ra có
1 mẫu, $c$ kênh,
chiều cao $h$ và chiều rộng $w$,
chúng ta có thể biến đổi đầu ra này thành
ma trận $\mathbf{X}$ có $c$ hàng và $hw$ cột.
Ma trận này có thể được xem là
phép nối của
$c$ vector $\mathbf{x}_1, \ldots, \mathbf{x}_c$,
mỗi vector có độ dài $hw$.
Ở đây, vector $\mathbf{x}_i$ biểu diễn đặc trưng phong cách của kênh $i$.

Trong *ma trận Gram* của các vector này $\mathbf{X}\mathbf{X}^\top \in \mathbb{R}^{c \times c}$, phần tử $x_{ij}$ ở hàng $i$ và cột $j$ là tích vô hướng của các vector $\mathbf{x}_i$ và $\mathbf{x}_j$.
Nó biểu diễn mối tương quan giữa các đặc trưng phong cách của kênh $i$ và $j$.
Chúng ta sử dụng ma trận Gram này để biểu diễn đầu ra phong cách của bất kỳ style layer nào.
Lưu ý rằng khi giá trị của $hw$ lớn hơn,
nó có khả năng dẫn đến các giá trị lớn hơn trong ma trận Gram.
Cũng lưu ý rằng chiều cao và chiều rộng của ma trận Gram đều bằng số kênh $c$.
Để style loss không bị ảnh hưởng
bởi các giá trị này,
hàm `gram` dưới đây chia
ma trận Gram cho số phần tử của nó, tức là $chw$.

```{.python .input}
#@tab all
def gram(X):
    num_channels, n = X.shape[1], d2l.size(X) // X.shape[1]
    X = d2l.reshape(X, (num_channels, n))
    return d2l.matmul(X, X.T) / (num_channels * n)
```

Rõ ràng,
hai đầu vào ma trận Gram của hàm loss bình phương cho style loss đều dựa trên
đầu ra của style layer đối với
ảnh tổng hợp và ảnh phong cách.
Ở đây giả định rằng ma trận Gram `gram_Y` dựa trên ảnh phong cách đã được tính trước.

```{.python .input}
#@tab mxnet
def style_loss(Y_hat, gram_Y):
    return np.square(gram(Y_hat) - gram_Y).mean()
```

```{.python .input}
#@tab pytorch
def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()
```

### Total Variation Loss

Đôi khi, ảnh tổng hợp đã học
có nhiều nhiễu tần số cao,
tức là các pixel đặc biệt sáng hoặc tối.
Một phương pháp giảm nhiễu phổ biến là
*total variation denoising*.
Ký hiệu $x_{i, j}$ là giá trị pixel tại tọa độ $(i, j)$.
Việc giảm total variation loss

$$\sum_{i, j} \left|x_{i, j} - x_{i+1, j}\right| + \left|x_{i, j} - x_{i, j+1}\right|$$

làm cho giá trị của các pixel lân cận trên ảnh tổng hợp gần nhau hơn.

```{.python .input}
#@tab all
def tv_loss(Y_hat):
    return 0.5 * (d2l.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  d2l.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())
```

### Hàm loss

[**Hàm mất mát của chuyển giao phong cách là tổng có trọng số của hàm mất mát nội dung, hàm mất mát phong cách và hàm mất mát biến thiên toàn phần**].
Bằng cách điều chỉnh các hyperparameter trọng số này,
chúng ta có thể cân bằng giữa
việc giữ lại nội dung,
chuyển giao phong cách
và giảm nhiễu trên ảnh được tổng hợp.

```{.python .input}
#@tab all
content_weight, style_weight, tv_weight = 1, 1e4, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # Calculate the content, style, and total variance losses respectively
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # Add up all the losses
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l
```

## [**Khởi tạo ảnh được tổng hợp**]

Trong chuyển giao phong cách,
ảnh được tổng hợp là biến duy nhất cần được cập nhật trong quá trình huấn luyện.
Do đó, chúng ta có thể định nghĩa một model đơn giản, `SynthesizedImage`, và coi ảnh được tổng hợp là các tham số của model.
Trong model này, quá trình lan truyền xuôi chỉ trả về các tham số của model.

```{.python .input}
#@tab mxnet
class SynthesizedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()
```

```{.python .input}
#@tab pytorch
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight
```

Tiếp theo, chúng ta định nghĩa hàm `get_inits`.
Hàm này tạo một instance của model ảnh được tổng hợp và khởi tạo nó bằng ảnh `X`.
Các ma trận Gram của ảnh phong cách tại nhiều layer phong cách khác nhau, `styles_Y_gram`, được tính trước khi huấn luyện.

```{.python .input}
#@tab mxnet
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape)
    gen_img.initialize(init.Constant(X), ctx=device, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam',
                            {'learning_rate': lr})
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

```{.python .input}
#@tab pytorch
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

## [**Huấn luyện**]


Khi huấn luyện model cho chuyển giao phong cách,
chúng ta liên tục trích xuất
các đặc trưng nội dung và đặc trưng phong cách của ảnh được tổng hợp, rồi tính hàm mất mát.
Phần dưới đây định nghĩa vòng lặp huấn luyện.

```{.python .input}
#@tab mxnet
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs], ylim=[0, 20],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        with autograd.record():
            contents_Y_hat, styles_Y_hat = extract_features(
                X, content_layers, style_layers)
            contents_l, styles_l, tv_l, l = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step(1)
        if (epoch + 1) % lr_decay_epoch == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.8)
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X).asnumpy())
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

```{.python .input}
#@tab pytorch
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

Bây giờ chúng ta [**bắt đầu huấn luyện model**].
Chúng ta thay đổi tỷ lệ chiều cao và chiều rộng của ảnh nội dung và ảnh phong cách thành 300 x 450 pixel.
Chúng ta sử dụng ảnh nội dung để khởi tạo ảnh được tổng hợp.

```{.python .input}
#@tab mxnet
device, image_shape = d2l.try_gpu(), (450, 300)
net.collect_params().reset_ctx(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.9, 500, 50)
```

```{.python .input}
#@tab pytorch
device, image_shape = d2l.try_gpu(), (300, 450)  # PIL Image (h, w)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
```

Chúng ta có thể thấy rằng ảnh được tổng hợp
giữ lại phong cảnh và các đối tượng của ảnh nội dung,
đồng thời chuyển màu sắc của ảnh phong cách
sang ảnh đó.
Ví dụ,
ảnh được tổng hợp có các mảng màu giống như
những mảng màu trong ảnh phong cách.
Một số mảng thậm chí còn có kết cấu tinh tế của các nét cọ.




## Tóm tắt

* Hàm mất mát thường được sử dụng trong chuyển giao phong cách gồm ba phần: (i) hàm mất mát nội dung làm cho ảnh được tổng hợp và ảnh nội dung gần nhau về các đặc trưng nội dung; (ii) hàm mất mát phong cách làm cho ảnh được tổng hợp và ảnh phong cách gần nhau về các đặc trưng phong cách; và (iii) hàm mất mát biến thiên toàn phần giúp giảm nhiễu trong ảnh được tổng hợp.
* Chúng ta có thể sử dụng một CNN pretrained để trích xuất các đặc trưng ảnh và tối thiểu hóa hàm mất mát nhằm liên tục cập nhật ảnh được tổng hợp dưới dạng các tham số của model trong quá trình huấn luyện.
* Chúng ta sử dụng các ma trận Gram để biểu diễn đầu ra phong cách từ các layer phong cách.


## Bài tập

1. Đầu ra thay đổi như thế nào khi bạn chọn các layer nội dung và layer phong cách khác nhau?
1. Điều chỉnh các hyperparameter trọng số trong hàm mất mát. Đầu ra có giữ lại nhiều nội dung hơn hoặc ít nhiễu hơn không?
1. Sử dụng các ảnh nội dung và ảnh phong cách khác nhau. Bạn có thể tạo ra những ảnh được tổng hợp thú vị hơn không?
1. Chúng ta có thể áp dụng chuyển giao phong cách cho văn bản không? Gợi ý: bạn có thể tham khảo bài khảo sát của :citet:`10.1145/3544903.3544906`.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/378)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1476)
:end_tab: