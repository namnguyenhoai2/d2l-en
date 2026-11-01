# Factorization Machines

Factorization machine (FM), được đề xuất bởi :citet:`Rendle.2010`, là một thuật toán có giám sát có thể được sử dụng cho các tác vụ phân loại, hồi quy và xếp hạng. Thuật toán này nhanh chóng thu hút sự chú ý và trở thành một phương pháp phổ biến, có ảnh hưởng trong việc đưa ra dự đoán và khuyến nghị. Cụ thể, đây là sự tổng quát hóa của mô hình hồi quy tuyến tính và mô hình matrix factorization. Ngoài ra, nó còn gợi nhớ đến support vector machine với polynomial kernel. Những ưu điểm của factorization machine so với hồi quy tuyến tính và matrix factorization là: (1) nó có thể mô hình hóa các tương tác biến $\chi$-chiều, trong đó $\chi$ là số bậc của đa thức và thường được đặt bằng hai. (2) Một thuật toán tối ưu nhanh đi kèm với factorization machine có thể giảm thời gian tính toán đa thức xuống độ phức tạp tuyến tính, khiến thuật toán này đặc biệt hiệu quả đối với các đầu vào thưa có số chiều cao. Vì những lý do này, factorization machine được sử dụng rộng rãi trong quảng cáo hiện đại và hệ thống khuyến nghị sản phẩm. Các chi tiết kỹ thuật và phần triển khai được mô tả dưới đây.


## Factorization Machine 2 chiều

Về hình thức, giả sử $x \in \mathbb{R}^d$ là vector đặc trưng của một mẫu, và $y$ là nhãn tương ứng, có thể là nhãn có giá trị thực hoặc nhãn lớp, chẳng hạn như lớp nhị phân "click/không click". Mô hình factorization machine bậc hai được định nghĩa như sau:

$$
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$

trong đó $\mathbf{w}_0 \in \mathbb{R}$ là bias tổng thể; $\mathbf{w} \in \mathbb{R}^d$ biểu diễn các trọng số của biến thứ i; $\mathbf{V} \in \mathbb{R}^{d\times k}$ biểu diễn các embedding của đặc trưng; $\mathbf{v}_i$ biểu diễn hàng $i^\textrm{th}$ của $\mathbf{V}$; $k$ là số chiều của các nhân tố tiềm ẩn; $\langle\cdot, \cdot \rangle$ là tích vô hướng của hai vector. $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ mô hình hóa tương tác giữa đặc trưng thứ $i^\textrm{th}$ và đặc trưng thứ $j^\textrm{th}$. Một số tương tác đặc trưng có thể dễ dàng được hiểu, vì vậy các chuyên gia có thể thiết kế chúng. Tuy nhiên, hầu hết các tương tác đặc trưng khác đều ẩn trong dữ liệu và khó xác định. Do đó, việc tự động mô hình hóa các tương tác đặc trưng có thể giảm đáng kể công sức trong feature engineering. Rõ ràng, hai hạng đầu tương ứng với mô hình hồi quy tuyến tính, còn hạng cuối là phần mở rộng của mô hình matrix factorization. Nếu đặc trưng $i$ biểu diễn một item và đặc trưng $j$ biểu diễn một user, hạng thứ ba chính là tích vô hướng giữa embedding của user và item. Cần lưu ý rằng FM cũng có thể được tổng quát hóa cho các bậc cao hơn (bậc > 2). Tuy nhiên, tính ổn định số có thể làm suy yếu khả năng tổng quát hóa.


## Tiêu chí Tối ưu Hiệu quả

Việc tối ưu factorization machine theo phương pháp trực tiếp dẫn đến độ phức tạp $\mathcal{O}(kd^2)$ vì cần tính tất cả các tương tác theo từng cặp. Để giải quyết vấn đề kém hiệu quả này, chúng ta có thể sắp xếp lại hạng thứ ba của FM, qua đó giảm đáng kể chi phí tính toán và đạt độ phức tạp thời gian tuyến tính ($\mathcal{O}(kd)$). Phép biến đổi của hạng tương tác theo từng cặp như sau:

$$
\begin{aligned}
&\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \\
 &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \\
 &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\\
 &=  \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \\
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
 \end{aligned}
$$

Với phép biến đổi này, độ phức tạp của mô hình được giảm đáng kể. Ngoài ra, đối với các đặc trưng thưa, chỉ cần tính các phần tử khác không, vì vậy độ phức tạp tổng thể tuyến tính theo số lượng đặc trưng khác không.

Để học mô hình FM, chúng ta có thể sử dụng hàm mất mát MSE cho tác vụ hồi quy, hàm mất mát cross-entropy cho các tác vụ phân loại và hàm mất mát BPR cho tác vụ xếp hạng. Các optimizer tiêu chuẩn như stochastic gradient descent và Adam đều phù hợp để tối ưu hóa.

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## Triển khai Mô hình
Đoạn mã sau triển khai factorization machine. Có thể thấy rõ rằng FM gồm một khối hồi quy tuyến tính và một khối tương tác đặc trưng hiệu quả. Chúng ta áp dụng hàm sigmoid lên điểm số cuối cùng vì coi việc dự đoán CTR là một tác vụ phân loại.

```{.python .input  n=2}
#@tab mxnet
class FM(nn.Block):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)

    def forward(self, x):
        square_of_sum = np.sum(self.embedding(x), axis=1) ** 2
        sum_of_square = np.sum(self.embedding(x) ** 2, axis=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        x = npx.sigmoid(x)
        return x
```

## Tải Tập dữ liệu Quảng cáo
Chúng ta sử dụng data wrapper CTR từ phần trước để tải tập dữ liệu quảng cáo trực tuyến.

```{.python .input  n=3}
#@tab mxnet
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
```

## Huấn luyện Mô hình
Sau đó, chúng ta huấn luyện mô hình. Learning rate được đặt mặc định là 0.02 và kích thước embedding là 20. Optimizer `Adam` và hàm mất mát `SigmoidBinaryCrossEntropyLoss` được sử dụng để huấn luyện mô hình.

```{.python .input  n=5}
#@tab mxnet
devices = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## Tóm tắt

* FM là một framework tổng quát có thể được áp dụng cho nhiều loại tác vụ như hồi quy, phân loại và xếp hạng.
* Tương tác/crossing giữa các đặc trưng rất quan trọng đối với các tác vụ dự đoán, và tương tác 2 chiều có thể được mô hình hóa hiệu quả bằng FM.

## Bài tập

* Bạn có thể kiểm thử FM trên các tập dữ liệu khác như Avazu, MovieLens và Criteo không?
* Hãy thay đổi kích thước embedding để kiểm tra ảnh hưởng của nó đến hiệu năng. Bạn có quan sát thấy một mô hình tương tự như mô hình của matrix factorization không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/406)
:end_tab: