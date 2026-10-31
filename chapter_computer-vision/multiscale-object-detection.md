# Phát hiện đối tượng đa tỷ lệ
:label:`sec_multiscale-object-detection`


Trong :numref:`sec_anchor`,
chúng ta đã tạo nhiều anchor box có tâm nằm trên mỗi pixel của ảnh đầu vào.
Về cơ bản, các anchor box này
đại diện cho các mẫu của
những vùng khác nhau trong ảnh.
Tuy nhiên,
chúng ta có thể phải tính toán quá nhiều anchor box
nếu tạo chúng cho *mọi* pixel.
Hãy xét một ảnh đầu vào có kích thước $561 \times 728$.
Nếu tạo năm anchor box
có các hình dạng khác nhau
cho mỗi pixel làm tâm,
thì cần gán nhãn và dự đoán hơn hai triệu anchor box ($561 \times 728 \times 5$) trên ảnh.

## Anchor box đa tỷ lệ
:label:`subsec_multiscale-anchor-boxes`

Bạn có thể nhận thấy rằng
việc giảm số lượng anchor box trên một ảnh không khó.
Chẳng hạn, chúng ta chỉ cần
lấy mẫu đồng đều một phần nhỏ các pixel
từ ảnh đầu vào
để tạo các anchor box có tâm nằm trên chúng.
Ngoài ra,
ở các tỷ lệ khác nhau,
chúng ta có thể tạo số lượng anchor box khác nhau
với các kích thước khác nhau.
Theo trực giác,
các đối tượng nhỏ
có khả năng xuất hiện trên ảnh cao hơn các đối tượng lớn.
Ví dụ,
các đối tượng $1 \times 1$, $1 \times 2$ và $2 \times 2$
có thể xuất hiện trên một ảnh $2 \times 2$
theo lần lượt 4, 2 và 1 cách khác nhau.
Do đó, khi sử dụng các anchor box nhỏ hơn để phát hiện các đối tượng nhỏ hơn, chúng ta có thể lấy mẫu nhiều vùng hơn,
còn đối với các đối tượng lớn hơn, chúng ta có thể lấy mẫu ít vùng hơn.

Để minh họa cách tạo anchor box
ở nhiều tỷ lệ, hãy đọc một ảnh.
Chiều cao và chiều rộng của ảnh lần lượt là 561 và 728 pixel.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, np, npx

npx.set_np()

img = image.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

Hãy nhớ rằng trong :numref:`sec_conv_layer`,
chúng ta gọi một mảng hai chiều là đầu ra của
một convolutional layer là một feature map.
Bằng cách xác định hình dạng của feature map,
chúng ta có thể xác định tâm của các anchor box được lấy mẫu đồng đều trên bất kỳ ảnh nào.


Hàm `display_anchors` được định nghĩa bên dưới.
[**Chúng ta tạo các anchor box (`anchors`) trên feature map (`fmap`) với mỗi unit (pixel) làm tâm của anchor box.**]
Vì các giá trị tọa độ trên trục $(x, y)$
trong các anchor box (`anchors`) đã được chia cho chiều rộng và chiều cao của feature map (`fmap`),
các giá trị này nằm trong khoảng từ 0 đến 1,
cho biết vị trí tương đối của
các anchor box trên feature map.

Vì tâm của các anchor box (`anchors`)
được phân bố trên tất cả các unit của feature map (`fmap`),
các tâm này phải được *phân bố đồng đều*
trên mọi ảnh đầu vào
xét theo vị trí không gian tương đối của chúng.
Cụ thể hơn,
với chiều rộng và chiều cao của feature map `fmap_w` và `fmap_h` lần lượt,
hàm sau sẽ *lấy mẫu đồng đều*
các pixel trên `fmap_h` hàng và `fmap_w` cột
trên mọi ảnh đầu vào.
Lấy các pixel được lấy mẫu đồng đều này làm tâm,
các anchor box có tỷ lệ `s` (giả sử độ dài của danh sách `s` là 1) và các aspect ratio khác nhau (`ratios`)
sẽ được tạo.

```{.python .input}
#@tab mxnet
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = np.zeros((1, 10, fmap_h, fmap_w))
    anchors = npx.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
```

```{.python .input}
#@tab pytorch
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = d2l.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = d2l.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

Trước tiên, hãy [**xét
việc phát hiện các đối tượng nhỏ**].
Để dễ phân biệt khi hiển thị, các anchor box có các tâm khác nhau ở đây không chồng lấp:
tỷ lệ của anchor box được đặt là 0.15
và chiều cao, chiều rộng của feature map được đặt là 4. Chúng ta có thể thấy
các tâm của anchor box trong 4 hàng và 4 cột trên ảnh được phân bố đồng đều.

```{.python .input}
#@tab all
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

Tiếp theo, chúng ta [**giảm một nửa chiều cao và chiều rộng của feature map, đồng thời sử dụng các anchor box lớn hơn để phát hiện các đối tượng lớn hơn**]. Khi tỷ lệ được đặt là 0.4,
một số anchor box sẽ chồng lấp lên nhau.

```{.python .input}
#@tab all
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

Cuối cùng, chúng ta [**tiếp tục giảm một nửa chiều cao và chiều rộng của feature map, đồng thời tăng tỷ lệ của anchor box lên 0.8**]. Lúc này, tâm của anchor box là tâm của ảnh.

```{.python .input}
#@tab all
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

## Phát hiện đa tỷ lệ


Vì đã tạo các anchor box đa tỷ lệ,
chúng ta sẽ sử dụng chúng để phát hiện các đối tượng có kích thước khác nhau
ở các tỷ lệ khác nhau.
Sau đây,
chúng ta giới thiệu một phương pháp phát hiện đối tượng đa tỷ lệ dựa trên CNN
mà chúng ta sẽ triển khai
trong :numref:`sec_ssd`.

Ở một tỷ lệ nào đó,
giả sử chúng ta có $c$ feature map có hình dạng $h \times w$.
Sử dụng phương pháp trong :numref:`subsec_multiscale-anchor-boxes`,
chúng ta tạo $hw$ tập anchor box,
trong đó mỗi tập có $a$ anchor box với cùng một tâm.
Ví dụ,
ở tỷ lệ đầu tiên trong các thí nghiệm tại :numref:`subsec_multiscale-anchor-boxes`,
với mười feature map (số kênh) có kích thước $4 \times 4$,
chúng ta đã tạo 16 tập anchor box,
mỗi tập chứa 3 anchor box có cùng một tâm.
Tiếp theo, mỗi anchor box được gán nhãn
về class và offset dựa trên các bounding box ground-truth. Ở tỷ lệ hiện tại, mô hình object detection cần dự đoán class và offset của $hw$ tập anchor box trên ảnh đầu vào, trong đó các tập khác nhau có các tâm khác nhau.


Giả sử $c$ feature map ở đây
là các đầu ra trung gian thu được
bằng quá trình forward propagation của CNN dựa trên ảnh đầu vào. Vì có $hw$ vị trí không gian khác nhau trên mỗi feature map,
ta có thể
coi mỗi vị trí không gian là có $c$ unit.
Theo
định nghĩa về receptive field trong :numref:`sec_conv_layer`,
$c$ unit này tại cùng một vị trí không gian
của các feature map
có cùng receptive field trên ảnh đầu vào:
chúng biểu diễn thông tin của ảnh đầu vào
trong cùng một receptive field.
Do đó, chúng ta có thể biến đổi $c$ unit
của các feature map tại cùng một vị trí không gian
thành
class và offset của $a$ anchor box
được tạo bằng vị trí không gian này.
Về bản chất,
chúng ta sử dụng thông tin của ảnh đầu vào trong một receptive field nhất định
để dự đoán class và offset của các anchor box
nằm
gần receptive field đó
trên ảnh đầu vào.


Khi các feature map ở các layer khác nhau
có các receptive field với kích thước khác nhau trên ảnh đầu vào, chúng có thể được dùng để phát hiện các đối tượng có kích thước khác nhau.
Ví dụ, chúng ta có thể thiết kế một neural network trong đó
các unit của feature map gần output layer hơn
có receptive field rộng hơn,
do đó có thể phát hiện các đối tượng lớn hơn trong ảnh đầu vào.

Tóm lại, chúng ta có thể tận dụng
các biểu diễn ảnh theo từng layer ở nhiều cấp độ
bằng deep neural network
để phát hiện đối tượng đa tỷ lệ.
Chúng ta sẽ trình bày cách thức hoạt động này qua một ví dụ cụ thể
trong :numref:`sec_ssd`.




## Tóm tắt

* Ở nhiều tỷ lệ, chúng ta có thể tạo các anchor box với kích thước khác nhau để phát hiện các đối tượng có kích thước khác nhau.
* Bằng cách xác định hình dạng của feature map, chúng ta có thể xác định tâm của các anchor box được lấy mẫu đồng đều trên bất kỳ ảnh nào.
* Chúng ta sử dụng thông tin của ảnh đầu vào trong một receptive field nhất định để dự đoán class và offset của các anchor box nằm gần receptive field đó trên ảnh đầu vào.
* Thông qua deep learning, chúng ta có thể tận dụng các biểu diễn ảnh theo từng layer ở nhiều cấp độ để phát hiện đối tượng đa tỷ lệ.


## Bài tập

1. Theo các thảo luận trong :numref:`sec_alexnet`, deep neural network học các đặc trưng phân cấp với mức độ trừu tượng ngày càng tăng cho ảnh. Trong phát hiện đối tượng đa tỷ lệ, các feature map ở các tỷ lệ khác nhau có tương ứng với các mức độ trừu tượng khác nhau không? Tại sao có hoặc tại sao không?
1. Ở tỷ lệ đầu tiên (`fmap_w=4, fmap_h=4`) trong các thí nghiệm tại :numref:`subsec_multiscale-anchor-boxes`, hãy tạo các anchor box được phân bố đồng đều và có thể chồng lấp lên nhau.
1. Cho một biến feature map có hình dạng $1 \times c \times h \times w$, trong đó $c$, $h$ và $w$ lần lượt là số kênh, chiều cao và chiều rộng của các feature map. Làm thế nào bạn có thể biến đổi biến này thành class và offset của các anchor box? Hình dạng của đầu ra là gì?


:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/371)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1607)
:end_tab: