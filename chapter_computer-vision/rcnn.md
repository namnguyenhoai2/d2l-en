# CNN dựa trên vùng (R-CNN)
:label:`sec_rcnn`

Bên cạnh phương pháp phát hiện đa hộp một lần
được mô tả trong :numref:`sec_ssd`,
CNN dựa trên vùng hoặc các vùng với đặc trưng CNN (R-CNN)
cũng nằm trong số nhiều
phương pháp tiên phong
áp dụng
deep learning cho bài toán phát hiện đối tượng
:cite:`Girshick.Donahue.Darrell.ea.2014`.
Trong phần này, chúng ta sẽ giới thiệu
R-CNN và chuỗi cải tiến của nó: fast R-CNN
:cite:`Girshick.2015`, faster R-CNN :cite:`Ren.He.Girshick.ea.2015`, và mask R-CNN
:cite:`He.Gkioxari.Dollar.ea.2017`.
Do giới hạn về dung lượng, chúng ta sẽ chỉ
tập trung vào thiết kế của các mô hình này.



## R-CNN


*R-CNN* trước tiên trích xuất
nhiều *đề xuất vùng*
(ví dụ: 2000) từ ảnh đầu vào
(ví dụ: các anchor box cũng có thể được xem
là các đề xuất vùng),
gán nhãn lớp và bounding box (ví dụ: các độ lệch) cho chúng.
:cite:`Girshick.Donahue.Darrell.ea.2014`
Sau đó, một CNN được sử dụng để
thực hiện forward propagation trên từng đề xuất vùng
nhằm trích xuất các đặc trưng của nó.
Tiếp theo, các đặc trưng của từng đề xuất vùng
được sử dụng để
dự đoán lớp và bounding box
của đề xuất vùng đó.


![Mô hình R-CNN.](../img/r-cnn.svg)
:label:`fig_r-cnn`

:numref:`fig_r-cnn` minh họa mô hình R-CNN. Cụ thể hơn, R-CNN gồm bốn bước sau:

1. Thực hiện *selective search* để trích xuất nhiều đề xuất vùng chất lượng cao trên ảnh đầu vào :cite:`Uijlings.Van-De-Sande.Gevers.ea.2013`. Các vùng được đề xuất này thường được chọn ở nhiều scale với các hình dạng và kích thước khác nhau. Mỗi đề xuất vùng sẽ được gán một lớp và một bounding box ground-truth.
1. Chọn một CNN đã được pretrained và cắt bỏ nó trước lớp đầu ra. Resize từng đề xuất vùng về kích thước đầu vào mà mạng yêu cầu, rồi xuất các đặc trưng đã trích xuất của đề xuất vùng đó thông qua forward propagation.
1. Lấy các đặc trưng đã trích xuất và lớp đã gán nhãn của từng đề xuất vùng làm một mẫu. Huấn luyện nhiều support vector machine để phân loại đối tượng, trong đó mỗi support vector machine xác định riêng liệu mẫu có chứa một lớp cụ thể hay không.
1. Lấy các đặc trưng đã trích xuất và bounding box đã gán nhãn của từng đề xuất vùng làm một mẫu. Huấn luyện một mô hình hồi quy tuyến tính để dự đoán bounding box ground-truth.


Mặc dù mô hình R-CNN sử dụng các CNN đã pretrained để trích xuất đặc trưng ảnh một cách hiệu quả,
nó lại chậm.
Hãy hình dung chúng ta chọn
hàng nghìn đề xuất vùng từ một ảnh đầu vào duy nhất:
điều này yêu cầu hàng nghìn lần
forward propagation qua CNN để thực hiện phát hiện đối tượng.
Khối lượng tính toán khổng lồ này khiến việc
sử dụng rộng rãi R-CNN trong các ứng dụng thực tế
trở nên không khả thi.

## Fast R-CNN

Nút thắt hiệu năng chính của
R-CNN nằm ở
việc thực hiện forward propagation qua CNN độc lập
cho từng đề xuất vùng,
mà không chia sẻ phép tính.
Vì các vùng này thường
chồng lấn lên nhau,
việc trích xuất đặc trưng độc lập dẫn đến
rất nhiều phép tính lặp lại.
Một trong những cải tiến lớn của
*fast R-CNN* so với
R-CNN là
forward propagation qua CNN
chỉ được thực hiện trên
toàn bộ ảnh :cite:`Girshick.2015`.

![Mô hình fast R-CNN.](../img/fast-rcnn.svg)
:label:`fig_fast_r-cnn`

:numref:`fig_fast_r-cnn` mô tả mô hình fast R-CNN. Các phép tính chính của mô hình gồm:

1. So với R-CNN, trong fast R-CNN, đầu vào của CNN để trích xuất đặc trưng là toàn bộ ảnh thay vì từng đề xuất vùng riêng lẻ. Ngoài ra, CNN này có thể được train. Với một ảnh đầu vào, giả sử đầu ra của CNN có shape là $1 \times c \times h_1  \times w_1$.
1. Giả sử selective search tạo ra $n$ đề xuất vùng. Các đề xuất vùng này (có hình dạng khác nhau) đánh dấu các vùng quan tâm (có hình dạng khác nhau) trên đầu ra của CNN. Sau đó, các vùng quan tâm này tiếp tục trích xuất các đặc trưng có cùng hình dạng (giả sử chiều cao $h_2$ và chiều rộng $w_2$ được chỉ định) để dễ dàng nối với nhau. Để thực hiện điều này, fast R-CNN giới thiệu lớp *region of interest (RoI) pooling*: đầu ra của CNN và các đề xuất vùng được đưa vào lớp này, tạo ra các đặc trưng đã nối có shape $n \times c \times h_2 \times w_2$, sau đó được dùng để trích xuất thêm đặc trưng cho tất cả các đề xuất vùng.
1. Sử dụng một lớp fully connected, biến đổi các đặc trưng đã nối thành đầu ra có shape $n \times d$, trong đó $d$ phụ thuộc vào thiết kế của mô hình.
1. Dự đoán lớp và bounding box cho từng trong số $n$ đề xuất vùng. Cụ thể hơn, trong phép dự đoán lớp và bounding box, biến đổi đầu ra của lớp fully connected thành một đầu ra có shape $n \times q$ ($q$ là số lớp) và một đầu ra có shape $n \times 4$, tương ứng. Phép dự đoán lớp sử dụng hồi quy softmax.


Lớp region of interest pooling được đề xuất trong fast R-CNN khác với lớp pooling được giới thiệu trong :numref:`sec_pooling`.
Trong lớp pooling,
chúng ta gián tiếp kiểm soát shape đầu ra
bằng cách chỉ định kích thước
của cửa sổ pooling, padding và stride.
Ngược lại,
chúng ta có thể chỉ định trực tiếp shape đầu ra
trong lớp region of interest pooling.

Ví dụ, hãy chỉ định
chiều cao và chiều rộng đầu ra
cho mỗi vùng lần lượt là $h_2$ và $w_2$.
Với bất kỳ cửa sổ vùng quan tâm nào
có shape $h \times w$,
cửa sổ này được chia thành một lưới $h_2 \times w_2$
gồm các cửa sổ con,
trong đó shape của mỗi cửa sổ con xấp xỉ
$(h/h_2) \times (w/w_2)$.
Trong thực tế,
chiều cao và chiều rộng của mỗi cửa sổ con sẽ được làm tròn lên, và phần tử lớn nhất sẽ được sử dụng làm đầu ra của cửa sổ con.
Do đó,
lớp region of interest pooling có thể trích xuất các đặc trưng có cùng shape
ngay cả khi các vùng quan tâm có hình dạng khác nhau.


Lấy một ví dụ minh họa,
trong :numref:`fig_roi`,
vùng quan tâm $3\times 3$ ở góc trên bên trái
được chọn trên đầu vào $4 \times 4$.
Với vùng quan tâm này,
chúng ta sử dụng lớp region of interest pooling $2\times 2$ để thu được
đầu ra $2\times 2$.
Lưu ý rằng
mỗi trong bốn cửa sổ con được chia
chứa các phần tử
0, 1, 4 và 5 (5 là lớn nhất);
2 và 6 (6 là lớn nhất);
8 và 9 (9 là lớn nhất);
và 10.

![Một lớp region of interest pooling $2\times 2$.](../img/roi.svg)
:label:`fig_roi`

Dưới đây, chúng ta minh họa phép tính của lớp region of interest pooling. Giả sử chiều cao và chiều rộng của các đặc trưng do CNN trích xuất `X` đều bằng 4, và chỉ có một channel.

```{.python .input}
#@tab mxnet
from mxnet import np, npx

npx.set_np()

X = np.arange(16).reshape(1, 1, 4, 4)
X
```

```{.python .input}
#@tab pytorch
import torch
import torchvision

X = torch.arange(16.).reshape(1, 1, 4, 4)
X
```

Giả sử thêm
rằng chiều cao và chiều rộng của ảnh đầu vào đều là 40 pixel và selective search tạo ra hai đề xuất vùng trên ảnh này.
Mỗi đề xuất vùng
được biểu diễn bằng năm phần tử:
lớp đối tượng của nó, theo sau là các tọa độ $(x, y)$
của góc trên bên trái và góc dưới bên phải.

```{.python .input}
#@tab mxnet
rois = np.array([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
```

```{.python .input}
#@tab pytorch
rois = torch.Tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
```

Vì chiều cao và chiều rộng của `X` bằng $1/10$ chiều cao và chiều rộng của ảnh đầu vào,
các tọa độ của hai đề xuất vùng
được nhân với 0.1 theo đối số `spatial_scale` đã chỉ định.
Sau đó, hai vùng quan tâm được đánh dấu trên `X` lần lượt là `X[:, :, 0:3, 0:3]` và `X[:, :, 1:4, 0:4]`.
Cuối cùng, trong lớp region of interest pooling $2\times 2$,
mỗi vùng quan tâm được chia
thành một lưới các cửa sổ con để
tiếp tục trích xuất các đặc trưng có cùng shape $2\times 2$.

```{.python .input}
#@tab mxnet
npx.roi_pooling(X, rois, pooled_size=(2, 2), spatial_scale=0.1)
```

```{.python .input}
#@tab pytorch
torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1)
```

## Faster R-CNN

Để phát hiện đối tượng chính xác hơn,
mô hình fast R-CNN
thường phải tạo ra
rất nhiều đề xuất vùng trong selective search.
Để giảm số lượng đề xuất vùng
mà không làm giảm độ chính xác,
*faster R-CNN*
đề xuất thay thế selective search bằng một *region proposal network* :cite:`Ren.He.Girshick.ea.2015`.



![Mô hình faster R-CNN.](../img/faster-rcnn.svg)
:label:`fig_faster_r-cnn`


:numref:`fig_faster_r-cnn` minh họa mô hình faster R-CNN. So với fast R-CNN,
faster R-CNN chỉ thay đổi
phương pháp tạo đề xuất vùng
từ selective search thành một region proposal network.
Phần còn lại của mô hình
giữ nguyên.
Region proposal network
hoạt động theo các bước sau:

1. Sử dụng một lớp convolution $3\times 3$ với padding bằng 1 để biến đổi đầu ra của CNN thành một đầu ra mới có $c$ channel. Theo cách này, mỗi unit dọc theo các chiều không gian của feature map do CNN trích xuất sẽ nhận được một vector đặc trưng mới có độ dài $c$.
1. Lấy mỗi pixel trên feature map làm tâm, tạo ra nhiều anchor box có scale và aspect ratio khác nhau rồi gán nhãn cho chúng.
1. Sử dụng vector đặc trưng độ dài $c$ tại tâm của mỗi anchor box, dự đoán lớp nhị phân (background hoặc đối tượng) và bounding box cho anchor box đó.
1. Xét các bounding box được dự đoán có lớp dự đoán là đối tượng. Loại bỏ các kết quả chồng lấn bằng non-maximum suppression. Các bounding box đối tượng được dự đoán còn lại là những đề xuất vùng mà lớp region of interest pooling yêu cầu.



Cần lưu ý rằng,
với tư cách là một phần của mô hình faster R-CNN,
region proposal network
được train joint
với phần còn lại của mô hình.
Nói cách khác, hàm mục tiêu của
faster R-CNN bao gồm
không chỉ phép dự đoán lớp và bounding box
trong phát hiện đối tượng,
mà còn cả phép dự đoán lớp nhị phân và bounding box
của các anchor box trong region proposal network.
Nhờ quá trình train end-to-end,
region proposal network học cách
tạo ra các đề xuất vùng chất lượng cao,
nhờ đó duy trì độ chính xác trong phát hiện đối tượng
với số lượng đề xuất vùng giảm xuống,
được học từ dữ liệu.




## Mask R-CNN

Trong tập dữ liệu huấn luyện,
nếu vị trí của đối tượng ở cấp pixel
cũng được gán nhãn trên ảnh,
*mask R-CNN* có thể tận dụng hiệu quả
các nhãn chi tiết như vậy
để tiếp tục cải thiện độ chính xác của việc phát hiện đối tượng :cite:`He.Gkioxari.Dollar.ea.2017`.


![Mô hình mask R-CNN.](../img/mask-rcnn.svg)
:label:`fig_mask_r-cnn`

Như được minh họa trong :numref:`fig_mask_r-cnn`,
mask R-CNN
được sửa đổi dựa trên faster R-CNN.
Cụ thể,
mask R-CNN thay thế
lớp region of interest pooling bằng
lớp *region of interest (RoI) alignment*.
Lớp region of interest alignment này
sử dụng phép nội suy song tuyến tính
để bảo toàn thông tin không gian trên các feature map, phù hợp hơn với việc dự đoán ở cấp độ pixel.
Đầu ra của lớp này
chứa các feature map có cùng hình dạng
cho tất cả các vùng quan tâm.
Chúng được sử dụng
để dự đoán
không chỉ lớp và bounding box cho mỗi vùng quan tâm,
mà còn cả vị trí của đối tượng ở cấp độ pixel thông qua một mạng tích chập đầy đủ bổ sung.
Các phần tiếp theo của chương này sẽ cung cấp
thêm chi tiết về việc sử dụng mạng tích chập đầy đủ để dự đoán ngữ nghĩa ở cấp độ pixel của một ảnh.




## Tóm tắt


* R-CNN trích xuất nhiều region proposal từ ảnh đầu vào, sử dụng CNN để thực hiện forward propagation trên từng region proposal nhằm trích xuất các đặc trưng của chúng, sau đó sử dụng các đặc trưng này để dự đoán lớp và bounding box của region proposal đó.
* Một trong những cải tiến chính của fast R-CNN so với R-CNN là CNN chỉ thực hiện forward propagation trên toàn bộ ảnh. Mô hình này cũng giới thiệu lớp region of interest pooling, nhờ đó có thể tiếp tục trích xuất các đặc trưng có cùng hình dạng cho những vùng quan tâm có hình dạng khác nhau.
* faster R-CNN thay thế selective search được sử dụng trong fast R-CNN bằng một region proposal network được huấn luyện đồng thời, nhờ đó phương pháp này vẫn duy trì độ chính xác trong việc phát hiện đối tượng với số lượng region proposal giảm xuống.
* Dựa trên faster R-CNN, mask R-CNN bổ sung một mạng tích chập đầy đủ nhằm tận dụng các nhãn ở cấp độ pixel để tiếp tục cải thiện độ chính xác của việc phát hiện đối tượng.


## Bài tập

1. Có thể xây dựng bài toán phát hiện đối tượng dưới dạng một bài toán hồi quy đơn, chẳng hạn như dự đoán bounding box và xác suất lớp, hay không? Bạn có thể tham khảo thiết kế của mô hình YOLO :cite:`Redmon.Divvala.Girshick.ea.2016`.
1. Hãy so sánh single shot multibox detection với các phương pháp được giới thiệu trong phần này. Những điểm khác biệt chính của chúng là gì? Bạn có thể tham khảo Hình 2 trong :citet:`Zhao.Zheng.Xu.ea.2019`.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/374)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1409)
:end_tab: