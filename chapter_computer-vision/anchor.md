# Anchor Boxes
:label:`sec_anchor`


Các thuật toán phát hiện đối tượng thường
lấy mẫu một số lượng lớn vùng trong ảnh đầu vào, xác định xem các vùng này có chứa
đối tượng cần quan tâm hay không, và điều chỉnh ranh giới
của các vùng để dự đoán
*bounding box thực tế*
của các đối tượng chính xác hơn.
Các mô hình khác nhau có thể áp dụng
những phương pháp lấy mẫu vùng khác nhau.
Ở đây, chúng ta giới thiệu một trong những phương pháp đó:
phương pháp này tạo ra nhiều bounding box với các tỷ lệ và tỉ lệ khung hình khác nhau, được căn giữa tại mỗi pixel.
Các bounding box này được gọi là *anchor box*.
Chúng ta sẽ thiết kế một mô hình phát hiện đối tượng
dựa trên anchor box trong :numref:`sec_ssd`.

Trước tiên, hãy điều chỉnh độ chính xác khi in
chỉ để kết quả đầu ra ngắn gọn hơn.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx

np.set_printoptions(2)  # Simplify printing accuracy
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

torch.set_printoptions(2)  # Simplify printing accuracy
```

## Tạo nhiều Anchor Box

Giả sử ảnh đầu vào có chiều cao là $h$ và chiều rộng là $w$.
Chúng ta tạo các anchor box có hình dạng khác nhau, được căn giữa tại mỗi pixel của ảnh.
Gọi *scale* là $s\in (0, 1]$ và
*aspect ratio* (tỉ lệ giữa chiều rộng và chiều cao) là $r > 0$.
Khi đó [**chiều rộng và chiều cao của anchor box lần lượt là $ws\sqrt{r}$ và $hs/\sqrt{r}$.**]
Lưu ý rằng khi đã biết vị trí tâm, một anchor box có chiều rộng và chiều cao xác định sẽ được xác định hoàn toàn.

Để tạo nhiều anchor box có hình dạng khác nhau,
hãy đặt một dãy scale
$s_1,\ldots, s_n$ và
một dãy aspect ratio $r_1,\ldots, r_m$.
Khi sử dụng mọi tổ hợp của các scale và aspect ratio này với mỗi pixel làm tâm,
ảnh đầu vào sẽ có tổng cộng $whnm$ anchor box. Mặc dù các anchor box này có thể bao phủ tất cả
bounding box thực tế, độ phức tạp tính toán dễ dàng trở nên quá cao.
Trong thực tế,
chúng ta chỉ có thể (**xét những tổ hợp
chứa**) $s_1$ hoặc $r_1$:

(**$$(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$$**)

Nói cách khác, số lượng anchor box được căn giữa trên cùng một pixel là $n+m-1$. Đối với toàn bộ ảnh đầu vào, chúng ta sẽ tạo tổng cộng $wh(n+m-1)$ anchor box.

Phương pháp tạo anchor box nêu trên được triển khai trong hàm `multibox_prior` sau đây. Chúng ta chỉ định ảnh đầu vào, một danh sách scale và một danh sách aspect ratio, sau đó hàm này sẽ trả về tất cả anchor box.

```{.python .input}
#@tab mxnet
#@save
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.ctx, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, ctx=device)
    ratio_tensor = d2l.tensor(ratios, ctx=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y-axis
    steps_w = 1.0 / in_width  # Scaled steps in x-axis

    # Generate all center points for the anchor boxes
    center_h = (d2l.arange(in_height, ctx=device) + offset_h) * steps_h
    center_w = (d2l.arange(in_width, ctx=device) + offset_w) * steps_w
    shift_x, shift_y = d2l.meshgrid(center_w, center_h)
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]),
                        sizes[0] * np.sqrt(ratio_tensor[1:]))) \
                        * in_height / in_width  # Handle rectangular inputs
    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]),
                        sizes[0] / np.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = np.tile(np.stack((-w, -h, w, h)).T,
                                   (in_height * in_width, 1)) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = d2l.stack([shift_x, shift_y, shift_x, shift_y],
                         axis=1).repeat(boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulations
    return np.expand_dims(output, axis=0)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, device=device)
    ratio_tensor = d2l.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
```

Ta có thể thấy rằng [**hình dạng của biến anchor box được trả về `Y`**] là
(kích thước batch, số lượng anchor box, 4).

```{.python .input}
#@tab mxnet
img = image.imread('../img/catdog.jpg').asnumpy()
h, w = img.shape[:2]

print(h, w)
X = np.random.uniform(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

```{.python .input}
#@tab pytorch
img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

Sau khi thay đổi hình dạng của biến anchor box `Y` thành (chiều cao ảnh, chiều rộng ảnh, số lượng anchor box được căn giữa trên cùng một pixel, 4),
chúng ta có thể lấy tất cả anchor box được căn giữa tại một vị trí pixel cụ thể.
Sau đây,
chúng ta [**truy cập anchor box đầu tiên được căn giữa tại (250, 250)**]. Nó có bốn phần tử: tọa độ trên các trục $(x, y)$ tại góc trên bên trái và tọa độ trên các trục $(x, y)$ tại góc dưới bên phải của anchor box.
Các giá trị tọa độ của cả hai trục
được chia lần lượt cho chiều rộng và chiều cao của ảnh.

```{.python .input}
#@tab all
boxes = Y.reshape(h, w, 5, 4)
boxes[250, 250, 0, :]
```

Để [**hiển thị tất cả anchor box được căn giữa trên một pixel trong ảnh**],
chúng ta định nghĩa hàm `show_bboxes` sau đây để vẽ nhiều bounding box trên ảnh.

```{.python .input}
#@tab all
#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```

Như chúng ta vừa thấy, các giá trị tọa độ của các trục $x$ và $y$ trong biến `boxes` lần lượt đã được chia cho chiều rộng và chiều cao của ảnh.
Khi vẽ anchor box,
chúng ta cần khôi phục các giá trị tọa độ ban đầu của chúng;
do đó, bên dưới chúng ta định nghĩa biến `bbox_scale`.
Bây giờ, chúng ta có thể vẽ tất cả anchor box được căn giữa tại (250, 250) trong ảnh.
Như bạn có thể thấy, anchor box màu xanh lam có scale 0.75 và aspect ratio bằng 1 bao quanh tốt
con chó trong ảnh.

```{.python .input}
#@tab all
d2l.set_figsize()
bbox_scale = d2l.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
```

## [**Intersection over Union (IoU)**]

Chúng ta vừa đề cập rằng một anchor box bao quanh con chó trong ảnh một cách “tốt”.
Nếu đã biết bounding box thực tế của đối tượng, làm thế nào để định lượng mức độ “tốt” này?
Theo trực giác, chúng ta có thể đo độ tương đồng giữa
anchor box và bounding box thực tế.
Chúng ta biết rằng *chỉ số Jaccard* có thể đo độ tương đồng giữa hai tập hợp. Với hai tập hợp $\mathcal{A}$ và $\mathcal{B}$, chỉ số Jaccard của chúng là kích thước phần giao chia cho kích thước phần hợp:

$$J(\mathcal{A},\mathcal{B}) = \frac{\left|\mathcal{A} \cap \mathcal{B}\right|}{\left| \mathcal{A} \cup \mathcal{B}\right|}.$$


Trên thực tế, chúng ta có thể xem diện tích pixel của bất kỳ bounding box nào là một tập hợp các pixel.
Theo cách này, chúng ta có thể đo độ tương đồng của hai bounding box bằng chỉ số Jaccard của các tập pixel của chúng. Đối với hai bounding box, chúng ta thường gọi chỉ số Jaccard của chúng là *intersection over union* (*IoU*), tức là tỉ lệ giữa diện tích phần giao và diện tích phần hợp của chúng, như minh họa trong :numref:`fig_iou`.
Giá trị IoU nằm trong khoảng từ 0 đến 1:
0 nghĩa là hai bounding box hoàn toàn không chồng lấp,
trong khi 1 cho biết hai bounding box giống hệt nhau.

![IoU là tỉ lệ giữa diện tích phần giao và diện tích phần hợp của hai bounding box.](../img/iou.svg)
:label:`fig_iou`

Trong phần còn lại của mục này, chúng ta sẽ sử dụng IoU để đo độ tương đồng giữa các anchor box và bounding box thực tế, cũng như giữa các anchor box khác nhau.
Với hai danh sách anchor box hoặc bounding box,
`box_iou` sau đây tính IoU theo từng cặp
giữa hai danh sách này.

```{.python .input}
#@tab mxnet
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

```{.python .input}
#@tab pytorch
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

## Gán nhãn Anchor Box trong Dữ liệu Huấn luyện
:label:`subsec_labeling-anchor-boxes`


Trong một tập dữ liệu huấn luyện,
chúng ta xem mỗi anchor box là một mẫu huấn luyện.
Để huấn luyện mô hình phát hiện đối tượng,
chúng ta cần nhãn *class* và *offset* cho mỗi anchor box,
trong đó nhãn đầu tiên là
class của đối tượng liên quan đến anchor box,
còn nhãn thứ hai là offset
của bounding box thực tế so với anchor box.
Trong quá trình dự đoán,
đối với mỗi ảnh
chúng ta tạo nhiều anchor box,
dự đoán class và offset cho tất cả anchor box,
điều chỉnh vị trí của chúng theo các offset đã dự đoán để thu được các bounding box dự đoán,
và cuối cùng chỉ xuất ra những
bounding box dự đoán thỏa mãn các tiêu chí nhất định.


Như chúng ta đã biết, một tập huấn luyện phát hiện đối tượng
đi kèm với các nhãn cho
vị trí của *bounding box thực tế*
và class của các đối tượng mà chúng bao quanh.
Để gán nhãn cho bất kỳ *anchor box* nào được tạo ra,
chúng ta tham chiếu đến
vị trí và class đã được gán nhãn của bounding box thực tế *được gán* cho nó và gần anchor box nhất.
Sau đây,
chúng ta mô tả một thuật toán để gán
bounding box thực tế gần nhất cho các anchor box.

### [**Gán Bounding Box Thực tế cho Anchor Box**]

Với một ảnh,
giả sử các anchor box là $A_1, A_2, \ldots, A_{n_a}$ và các bounding box thực tế là $B_1, B_2, \ldots, B_{n_b}$, trong đó $n_a \geq n_b$.
Hãy định nghĩa một ma trận $\mathbf{X} \in \mathbb{R}^{n_a \times n_b}$, trong đó phần tử $x_{ij}$ tại hàng thứ $i^\textrm{th}$ và cột thứ $j^\textrm{th}$ là IoU của anchor box $A_i$ và bounding box thực tế $B_j$. Thuật toán gồm các bước sau:

1. Tìm phần tử lớn nhất trong ma trận $\mathbf{X}$ và ký hiệu chỉ số hàng và cột của nó lần lượt là $i_1$ và $j_1$. Khi đó, bounding box thực tế $B_{j_1}$ được gán cho anchor box $A_{i_1}$. Điều này khá trực quan vì $A_{i_1}$ và $B_{j_1}$ là cặp gần nhau nhất trong tất cả các cặp anchor box và bounding box thực tế. Sau lần gán đầu tiên, loại bỏ tất cả phần tử trong hàng thứ ${i_1}^\textrm{th}$ và cột thứ ${j_1}^\textrm{th}$ trong ma trận $\mathbf{X}$.
1. Tìm phần tử lớn nhất trong số các phần tử còn lại của ma trận $\mathbf{X}$ và ký hiệu chỉ số hàng và cột của nó lần lượt là $i_2$ và $j_2$. Chúng ta gán bounding box thực tế $B_{j_2}$ cho anchor box $A_{i_2}$ và loại bỏ tất cả phần tử trong hàng thứ ${i_2}^\textrm{th}$ và cột thứ ${j_2}^\textrm{th}$ trong ma trận $\mathbf{X}$.
1. Lúc này, các phần tử trong hai hàng và hai cột của ma trận $\mathbf{X}$ đã bị loại bỏ. Chúng ta tiếp tục cho đến khi tất cả phần tử trong $n_b$ cột của ma trận $\mathbf{X}$ bị loại bỏ. Khi đó, chúng ta đã gán một bounding box thực tế cho mỗi trong số $n_b$ anchor box.
1. Chỉ duyệt qua $n_a - n_b$ anchor box còn lại. Ví dụ, với bất kỳ anchor box $A_i$ nào, hãy tìm bounding box thực tế $B_j$ có IoU lớn nhất với $A_i$ trên toàn bộ hàng thứ $i^\textrm{th}$ của ma trận $\mathbf{X}$, và chỉ gán $B_j$ cho $A_i$ nếu IoU này lớn hơn một ngưỡng được định trước.

Hãy minh họa thuật toán trên bằng một
ví dụ cụ thể.
Như được trình bày trong :numref:`fig_anchor_label` (bên trái), giả sử giá trị lớn nhất trong ma trận $\mathbf{X}$ là $x_{23}$, ta gán bounding box ground-truth $B_3$ cho anchor box $A_2$.
Sau đó, ta loại bỏ tất cả các phần tử trong hàng 2 và cột 3 của ma trận, tìm $x_{71}$ lớn nhất trong các phần tử còn lại (vùng được tô bóng), rồi gán bounding box ground-truth $B_1$ cho anchor box $A_7$. 
Tiếp theo, như được trình bày trong :numref:`fig_anchor_label` (ở giữa), ta loại bỏ tất cả các phần tử trong hàng 7 và cột 1 của ma trận, tìm $x_{54}$ lớn nhất trong các phần tử còn lại (vùng được tô bóng), rồi gán bounding box ground-truth $B_4$ cho anchor box $A_5$. 
Cuối cùng, như được trình bày trong :numref:`fig_anchor_label` (bên phải), ta loại bỏ tất cả các phần tử trong hàng 5 và cột 4 của ma trận, tìm $x_{92}$ lớn nhất trong các phần tử còn lại (vùng được tô bóng), rồi gán bounding box ground-truth $B_2$ cho anchor box $A_9$.
Sau đó, ta chỉ cần duyệt qua
các anchor box còn lại $A_1, A_3, A_4, A_6, A_8$ và xác định có gán bounding box ground-truth cho chúng hay không dựa trên threshold.

![Gán bounding box ground-truth cho anchor box.](../img/anchor-label.svg)
:label:`fig_anchor_label`

Thuật toán này được triển khai trong hàm `assign_anchor_to_bbox` sau đây.

```{.python .input}
#@tab mxnet
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = np.full((num_anchors,), -1, dtype=np.int32, ctx=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = np.max(jaccard, axis=1), np.argmax(jaccard, axis=1)
    anc_i = np.nonzero(max_ious >= iou_threshold)[0]
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = np.full((num_anchors,), -1)
    row_discard = np.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = np.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).astype('int32')
        anc_idx = (max_idx / num_gt_boxes).astype('int32')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

```{.python .input}
#@tab pytorch
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

### Gán nhãn Class và Offset

Bây giờ ta có thể gán nhãn class và offset cho mỗi anchor box. Giả sử một anchor box $A$ được gán
một bounding box ground-truth $B$. 
Một mặt,
class của anchor box $A$ sẽ được
gán nhãn giống class của $B$.
Mặt khác,
offset của anchor box $A$ 
sẽ được gán nhãn dựa trên 
vị trí tương đối giữa
các tọa độ tâm của $B$ và $A$
cùng với kích thước tương đối giữa
hai box này.
Vì các box khác nhau trong dataset có
vị trí và kích thước khác nhau,
ta có thể áp dụng các phép biến đổi
cho những vị trí và kích thước tương đối đó
để tạo ra các offset
có phân phối đồng đều hơn
và dễ khớp hơn.
Sau đây, ta mô tả một phép biến đổi phổ biến.
[**Cho tọa độ tâm của $A$ và $B$ lần lượt là $(x_a, y_a)$ và $(x_b, y_b)$, 
chiều rộng lần lượt là $w_a$ và $w_b$, 
và chiều cao lần lượt là $h_a$ và $h_b$. 
Ta có thể gán nhãn offset của $A$ là

$$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
\frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
\frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
\frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right),$$
**]
trong đó giá trị mặc định của các hằng số là $\mu_x = \mu_y = \mu_w = \mu_h = 0, \sigma_x=\sigma_y=0.1$, và $\sigma_w=\sigma_h=0.2$.
Phép biến đổi này được triển khai bên dưới trong hàm `offset_boxes`.

```{.python .input}
#@tab all
#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets."""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * d2l.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = d2l.concat([offset_xy, offset_wh], axis=1)
    return offset
```

Nếu một anchor box không được gán bounding box ground-truth, ta chỉ cần gán nhãn class của anchor box là "background".
Các anchor box có class là background thường được gọi là anchor box *negative*,
còn các anchor box còn lại được gọi là anchor box *positive*.
Ta triển khai hàm `multibox_target` sau đây
để [**gán nhãn class và offset cho anchor box**] (đối số `anchors`) bằng cách sử dụng các bounding box ground-truth (đối số `labels`).
Hàm này đặt class background thành 0 và tăng chỉ số nguyên của một class mới lên một đơn vị.

```{.python .input}
#@tab mxnet
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.ctx, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = np.tile((np.expand_dims((anchors_bbox_map >= 0),
                                            axis=-1)), (1, 4)).astype('int32')
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = d2l.zeros(num_anchors, dtype=np.int32, ctx=device)
        assigned_bb = d2l.zeros((num_anchors, 4), dtype=np.float32,
                                ctx=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = np.nonzero(anchors_bbox_map >= 0)[0]
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].astype('int32') + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = d2l.stack(batch_offset)
    bbox_mask = d2l.stack(batch_mask)
    class_labels = d2l.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

### Một ví dụ

Hãy minh họa việc gán nhãn cho anchor box
bằng một ví dụ cụ thể.
Ta định nghĩa các bounding box ground-truth cho chó và mèo trong ảnh đã tải,
trong đó phần tử đầu tiên là class (0 đối với chó và 1 đối với mèo), còn bốn phần tử còn lại là
tọa độ trên các trục $(x, y)$
tại góc trên bên trái và góc dưới bên phải
(giá trị nằm trong khoảng từ 0 đến 1). 
Ta cũng tạo năm anchor box để gán nhãn
bằng cách sử dụng tọa độ của
góc trên bên trái và góc dưới bên phải:
$A_0, \ldots, A_4$ (chỉ số bắt đầu từ 0).
Sau đó, ta [**vẽ các bounding box ground-truth 
và anchor box 
trên ảnh.**]

```{.python .input}
#@tab all
ground_truth = d2l.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = d2l.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
```

Sử dụng hàm `multibox_target` được định nghĩa ở trên,
ta có thể [**gán nhãn class và offset
cho các anchor box này dựa trên
các bounding box ground-truth**] của chó và mèo.
Trong ví dụ này, chỉ số của
các class background, chó và mèo
lần lượt là 0, 1 và 2. 
Bên dưới, ta thêm một chiều cho các ví dụ của anchor box và bounding box ground-truth.

```{.python .input}
#@tab mxnet
labels = multibox_target(np.expand_dims(anchors, axis=0),
                         np.expand_dims(ground_truth, axis=0))
```

```{.python .input}
#@tab pytorch
labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
```

Kết quả trả về có ba phần tử, tất cả đều ở định dạng tensor.
Phần tử thứ ba chứa các class đã gán nhãn của các anchor box đầu vào.

Hãy phân tích các class label được trả về bên dưới dựa trên
vị trí của anchor box và bounding box ground-truth trong ảnh.
Trước tiên, trong tất cả các cặp anchor box
và bounding box ground-truth,
IoU của anchor box $A_4$ và bounding box ground-truth của mèo là lớn nhất. 
Do đó, class của $A_4$ được gán nhãn là mèo.
Sau khi loại bỏ
các cặp chứa $A_4$ hoặc bounding box ground-truth của mèo, trong số các cặp còn lại,
cặp gồm anchor box $A_1$ và bounding box ground-truth của chó có IoU lớn nhất.
Vì vậy, class của $A_1$ được gán nhãn là chó.
Tiếp theo, ta cần duyệt qua ba anchor box chưa được gán nhãn còn lại: $A_0$, $A_2$ và $A_3$.
Đối với $A_0$,
class của bounding box ground-truth có IoU lớn nhất là chó,
nhưng IoU thấp hơn threshold được định trước (0.5),
nên class được gán nhãn là background;
đối với $A_2$,
class của bounding box ground-truth có IoU lớn nhất là mèo và IoU vượt threshold, nên class được gán nhãn là mèo;
đối với $A_3$,
class của bounding box ground-truth có IoU lớn nhất là mèo, nhưng giá trị này thấp hơn threshold, nên class được gán nhãn là background.

```{.python .input}
#@tab all
labels[2]
```

Phần tử thứ hai được trả về là một biến mask có shape (batch size, bốn lần số lượng anchor box).
Mỗi bốn phần tử trong biến mask 
tương ứng với bốn giá trị offset của mỗi anchor box.
Vì ta không quan tâm đến việc phát hiện background,
các offset của class negative này không được ảnh hưởng đến hàm mục tiêu.
Thông qua phép nhân theo từng phần tử, các số 0 trong biến mask sẽ lọc bỏ các offset của class negative trước khi tính hàm mục tiêu.

```{.python .input}
#@tab all
labels[1]
```

Phần tử đầu tiên được trả về chứa bốn giá trị offset được gán nhãn cho mỗi anchor box.
Lưu ý rằng các offset của anchor box thuộc class negative được gán nhãn là 0.

```{.python .input}
#@tab all
labels[0]
```

## Dự đoán Bounding Box bằng Non-Maximum Suppression
:label:`subsec_predicting-bounding-boxes-nms`

Trong quá trình dự đoán,
ta tạo nhiều anchor box cho ảnh và dự đoán class cùng offset cho mỗi anchor box.
Do đó, một *bounding box được dự đoán*
được tạo ra dựa trên 
một anchor box cùng offset được dự đoán của nó.
Bên dưới, ta triển khai hàm `offset_inverse`
nhận anchor và
các dự đoán offset làm đầu vào, đồng thời [**áp dụng các phép biến đổi offset nghịch đảo để
trả về tọa độ của bounding box được dự đoán**].

```{.python .input}
#@tab all
#@save
def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets."""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = d2l.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = d2l.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox
```

Khi có nhiều anchor box,
nhiều bounding box được dự đoán tương tự nhau (có độ chồng lấp đáng kể)
có thể được tạo ra để bao quanh cùng một đối tượng.
Để đơn giản hóa đầu ra,
ta có thể gộp các bounding box được dự đoán tương tự nhau
và thuộc cùng một đối tượng
bằng cách sử dụng *non-maximum suppression* (NMS).

Sau đây là cách non-maximum suppression hoạt động.
Với một bounding box được dự đoán $B$,
mô hình object detection tính
likelihood được dự đoán cho mỗi class.
Gọi $p$ là likelihood được dự đoán lớn nhất,
class tương ứng với xác suất này là class được dự đoán của $B$.
Cụ thể, ta gọi $p$ là *confidence* (score) của bounding box được dự đoán $B$.
Trong cùng một ảnh,
tất cả các bounding box được dự đoán không phải background
được sắp xếp theo confidence giảm dần
để tạo thành một danh sách $L$.
Sau đó, ta xử lý danh sách đã sắp xếp $L$ theo các bước sau:

1. Chọn bounding box được dự đoán $B_1$ có confidence cao nhất trong $L$ làm cơ sở và loại khỏi $L$ tất cả các bounding box được dự đoán không phải cơ sở có IoU với $B_1$ vượt threshold $\epsilon$ được định trước. Tại thời điểm này, $L$ giữ lại bounding box được dự đoán có confidence cao nhất nhưng loại bỏ các bounding box khác quá tương tự với nó. Nói ngắn gọn, những bounding box có score *non-maximum* sẽ bị *suppressed*.
1. Chọn bounding box được dự đoán $B_2$ có confidence cao thứ hai trong $L$ làm một cơ sở khác và loại khỏi $L$ tất cả các bounding box được dự đoán không phải cơ sở có IoU với $B_2$ vượt $\epsilon$.
1. Lặp lại quy trình trên cho đến khi tất cả các bounding box được dự đoán trong $L$ đều đã được sử dụng làm cơ sở. Khi đó, IoU của mọi cặp bounding box được dự đoán trong $L$ đều thấp hơn threshold $\epsilon$; do đó, không có cặp nào quá tương tự nhau. 
1. Xuất tất cả các bounding box được dự đoán trong danh sách $L$.

[**Hàm `nms` sau đây sắp xếp các confidence score theo thứ tự giảm dần và trả về các chỉ số tương ứng.**]

```{.python .input}
#@tab mxnet
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = scores.argsort()[::-1]
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.size > 0:
        i = B[0]
        keep.append(i)
        if B.size == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = np.nonzero(iou <= iou_threshold)[0]
        B = B[inds + 1]
    return np.array(keep, dtype=np.int32, ctx=boxes.ctx)
```

```{.python .input}
#@tab pytorch
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return d2l.tensor(keep, device=boxes.device)
```

Ta định nghĩa `multibox_detection` sau đây
để [**áp dụng non-maximum suppression
cho việc dự đoán bounding box**].
Đừng lo lắng nếu bạn thấy phần triển khai
hơi phức tạp: ngay sau phần triển khai,
ta sẽ minh họa cách hoạt động của nó
bằng một ví dụ cụ thể.

```{.python .input}
#@tab mxnet
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.ctx, cls_probs.shape[0]
    anchors = np.squeeze(anchors, axis=0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = np.max(cls_prob[1:], 0), np.argmax(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = np.arange(num_anchors, dtype=np.int32, ctx=device)
        combined = d2l.concat((keep, all_idx))
        unique, counts = np.unique(combined, return_counts=True)
        non_keep = unique[counts == 1]
        all_id_sorted = d2l.concat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted].astype('float32')
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = d2l.concat((np.expand_dims(class_id, axis=1),
                                np.expand_dims(conf, axis=1),
                                predicted_bb), axis=1)
        out.append(pred_info)
    return d2l.stack(out)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return d2l.stack(out)
```

Bây giờ hãy [**áp dụng các phần triển khai trên
cho một ví dụ cụ thể với bốn anchor box**].
Để đơn giản, ta giả sử các
offset được dự đoán đều bằng 0.
Điều này có nghĩa là các bounding box được dự đoán chính là các anchor box. 
Đối với mỗi class trong số background, chó và mèo,
ta cũng định nghĩa likelihood được dự đoán của class đó.

```{.python .input}
#@tab all
anchors = d2l.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = d2l.tensor([0] * d2l.size(anchors))
cls_probs = d2l.tensor([[0] * 4,  # Predicted background likelihood 
                      [0.9, 0.8, 0.7, 0.1],  # Predicted dog likelihood 
                      [0.1, 0.2, 0.3, 0.9]])  # Predicted cat likelihood
```

Ta có thể [**vẽ các bounding box được dự đoán cùng confidence của chúng trên ảnh.**]

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
```

Bây giờ chúng ta có thể gọi hàm `multibox_detection`
để thực hiện non-maximum suppression,
trong đó ngưỡng được đặt là 0.5.
Lưu ý rằng chúng ta thêm
một chiều cho các mẫu trong tensor đầu vào.

Ta có thể thấy rằng [**shape của kết quả được trả về**] là
(kích thước batch, số lượng anchor box, 6).
Sáu phần tử trong chiều trong cùng
cung cấp thông tin đầu ra cho cùng một bounding box được dự đoán.
Phần tử đầu tiên là chỉ số lớp được dự đoán, bắt đầu từ 0 (0 là chó và 1 là mèo). Giá trị -1 biểu thị nền hoặc việc loại bỏ trong non-maximum suppression.
Phần tử thứ hai là độ tin cậy của bounding box được dự đoán.
Bốn phần tử còn lại lần lượt là tọa độ theo các trục $(x, y)$ của góc trên bên trái và
góc dưới bên phải của bounding box được dự đoán (giá trị nằm trong khoảng từ 0 đến 1).

```{.python .input}
#@tab mxnet
output = multibox_detection(np.expand_dims(cls_probs, axis=0),
                            np.expand_dims(offset_preds, axis=0),
                            np.expand_dims(anchors, axis=0),
                            nms_threshold=0.5)
output
```

```{.python .input}
#@tab pytorch
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
output
```

Sau khi loại bỏ các bounding box được dự đoán
có lớp -1,
chúng ta có thể [**xuất bounding box được dự đoán cuối cùng
được non-maximum suppression giữ lại**].

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
for i in d2l.numpy(output[0]):
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [d2l.tensor(i[2:]) * bbox_scale], label)
```

Trong thực tế, chúng ta có thể loại bỏ các bounding box được dự đoán có độ tin cậy thấp hơn ngay cả trước khi thực hiện non-maximum suppression, qua đó giảm lượng tính toán trong thuật toán này.
Chúng ta cũng có thể hậu xử lý đầu ra của non-maximum suppression, chẳng hạn chỉ giữ lại
các kết quả có độ tin cậy cao hơn
trong đầu ra cuối cùng.


## Tóm tắt

* Chúng ta tạo các anchor box có hình dạng khác nhau, được căn giữa tại mỗi pixel của ảnh.
* Intersection over union (IoU), còn được gọi là chỉ số Jaccard, đo độ tương đồng của hai bounding box. Đây là tỷ lệ giữa diện tích phần giao và diện tích phần hợp của chúng.
* Trong một tập huấn luyện, chúng ta cần hai loại nhãn cho mỗi anchor box. Một loại là lớp của đối tượng tương ứng với anchor box, loại còn lại là độ lệch của bounding box ground truth so với anchor box.
* Trong quá trình dự đoán, chúng ta có thể sử dụng non-maximum suppression (NMS) để loại bỏ các bounding box được dự đoán tương tự nhau, qua đó đơn giản hóa đầu ra.


## Bài tập

1. Thay đổi các giá trị của `sizes` và `ratios` trong hàm `multibox_prior`. Các anchor box được tạo ra thay đổi như thế nào?
1. Tạo và trực quan hóa hai bounding box có IoU bằng 0.5. Chúng chồng lấp với nhau như thế nào?
1. Sửa đổi biến `anchors` trong :numref:`subsec_labeling-anchor-boxes` và :numref:`subsec_predicting-bounding-boxes-nms`. Kết quả thay đổi như thế nào?
1. Non-maximum suppression là một thuật toán tham lam, trong đó các bounding box được dự đoán bị loại bỏ. Có khả năng một số bounding box bị loại bỏ này thực sự hữu ích không? Làm thế nào có thể sửa đổi thuật toán này để thực hiện suppression một cách *mềm*? Bạn có thể tham khảo Soft-NMS :cite:`Bodla.Singh.Chellappa.ea.2017`.
1. Thay vì được thiết kế thủ công, liệu non-maximum suppression có thể được học không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/370)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1603)
:end_tab: