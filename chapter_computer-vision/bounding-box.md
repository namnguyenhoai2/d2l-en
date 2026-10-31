# Phát hiện đối tượng và các hộp giới hạn
:label:`sec_bbox`


Trong các phần trước (ví dụ: :numref:`sec_alexnet`--:numref:`sec_googlenet`),
chúng ta đã giới thiệu nhiều mô hình khác nhau cho bài toán phân loại hình ảnh.
Trong các tác vụ phân loại hình ảnh,
chúng ta giả định rằng chỉ có *một*
đối tượng chính
trong hình ảnh và chỉ tập trung vào việc
nhận biết danh mục của đối tượng đó.
Tuy nhiên, thường có *nhiều* đối tượng
đáng quan tâm trong hình ảnh.
Chúng ta không chỉ muốn biết danh mục của chúng mà còn muốn biết vị trí cụ thể của chúng trong hình ảnh.
Trong thị giác máy tính, chúng ta gọi những tác vụ như vậy là *phát hiện đối tượng* (hoặc *nhận dạng đối tượng*).

Phát hiện đối tượng đã được
áp dụng rộng rãi trong nhiều lĩnh vực.
Ví dụ, xe tự lái cần lập kế hoạch
các tuyến đường di chuyển
bằng cách phát hiện vị trí
của phương tiện, người đi bộ, đường sá và chướng ngại vật trong các hình ảnh video được ghi lại.
Ngoài ra,
robot có thể sử dụng kỹ thuật này
để phát hiện và xác định vị trí các đối tượng đáng quan tâm
trong suốt quá trình điều hướng môi trường.
Hơn nữa,
các hệ thống an ninh
có thể cần phát hiện những đối tượng bất thường, chẳng hạn như kẻ xâm nhập hoặc bom.

Trong một vài phần tiếp theo, chúng ta sẽ giới thiệu
một số phương pháp deep learning để phát hiện đối tượng.
Chúng ta sẽ bắt đầu bằng phần giới thiệu
về *vị trí* (hoặc *tọa độ*) của các đối tượng.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

Chúng ta sẽ tải hình ảnh mẫu được sử dụng trong phần này. Có thể thấy một con chó ở phía bên trái hình ảnh và một con mèo ở phía bên phải.
Đó là hai đối tượng chính trong hình ảnh này.

```{.python .input}
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```{.python .input}
#@tab pytorch, tensorflow
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## Các hộp giới hạn


Trong phát hiện đối tượng,
chúng ta thường sử dụng một *hộp giới hạn* để mô tả vị trí không gian của một đối tượng.
Hộp giới hạn có dạng hình chữ nhật, được xác định bởi tọa độ $x$ và $y$ của góc trên bên trái của hình chữ nhật, cùng các tọa độ tương ứng của góc dưới bên phải.
Một cách biểu diễn hộp giới hạn thường được sử dụng khác là tọa độ $(x, y)$ của tâm hộp giới hạn, cùng với chiều rộng và chiều cao của hộp.

[**Tại đây, chúng ta định nghĩa các hàm để chuyển đổi giữa**] hai (**cách biểu diễn**):
`box_corner_to_center` chuyển đổi từ cách biểu diễn bằng hai góc
sang cách biểu diễn tâm–chiều rộng–chiều cao,
còn `box_center_to_corner` thực hiện chuyển đổi ngược lại.
Đối số đầu vào `boxes` phải là một tensor hai chiều có
shape ($n$, 4), trong đó $n$ là số lượng hộp giới hạn.

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

Chúng ta sẽ [**xác định các hộp giới hạn của con chó và con mèo trong hình ảnh**] dựa
trên thông tin tọa độ.
Gốc tọa độ trong hình ảnh
là góc trên bên trái của hình ảnh, còn bên phải và bên dưới lần lượt là
các hướng dương của trục $x$ và $y$.

```{.python .input}
#@tab all
# Here `bbox` is the abbreviation for bounding box
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

Chúng ta có thể kiểm tra tính đúng đắn của hai
hàm chuyển đổi hộp giới hạn bằng cách thực hiện chuyển đổi hai lần.

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

Hãy [**vẽ các hộp giới hạn lên hình ảnh**] để kiểm tra xem chúng có chính xác hay không.
Trước khi vẽ, chúng ta sẽ định nghĩa một hàm hỗ trợ `bbox_to_rect`. Hàm này biểu diễn hộp giới hạn theo định dạng hộp giới hạn của gói `matplotlib`.

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

Sau khi thêm các hộp giới hạn vào hình ảnh,
chúng ta có thể thấy rằng đường viền chính của hai đối tượng về cơ bản nằm bên trong hai hộp.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## Tóm tắt

* Phát hiện đối tượng không chỉ nhận dạng tất cả các đối tượng đáng quan tâm trong hình ảnh mà còn xác định vị trí của chúng. Vị trí thường được biểu diễn bằng một hộp giới hạn hình chữ nhật.
* Chúng ta có thể chuyển đổi giữa hai cách biểu diễn hộp giới hạn thường được sử dụng.

## Bài tập

1. Tìm một hình ảnh khác và thử gắn nhãn một hộp giới hạn bao quanh đối tượng. Hãy so sánh việc gắn nhãn hộp giới hạn với việc gắn nhãn danh mục: việc nào thường mất nhiều thời gian hơn?
1. Tại sao chiều trong cùng của đối số đầu vào `boxes` của `box_corner_to_center` và `box_center_to_corner` luôn là 4?


:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/369)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1527)
:end_tab: