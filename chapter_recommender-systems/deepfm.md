# Deep Factorization Machines

Việc học các tổ hợp đặc trưng hiệu quả đóng vai trò then chốt đối với sự thành công của bài toán dự đoán tỷ lệ nhấp. Factorization machines mô hình hóa các tương tác đặc trưng theo một paradigm tuyến tính (ví dụ: các tương tác song tuyến tính). Điều này thường không đủ đối với dữ liệu thực tế, nơi các cấu trúc giao cắt đặc trưng vốn có thường rất phức tạp và phi tuyến. Tệ hơn nữa, trong thực tế, factorization machines thường chỉ sử dụng các tương tác đặc trưng bậc hai. Về mặt lý thuyết, có thể mô hình hóa các tổ hợp đặc trưng bậc cao hơn bằng factorization machines, nhưng cách này thường không được áp dụng do tính bất ổn số và độ phức tạp tính toán cao.

Một giải pháp hiệu quả là sử dụng mạng neural sâu. Mạng neural sâu có khả năng mạnh mẽ trong việc học biểu diễn đặc trưng và có tiềm năng học các tương tác đặc trưng tinh vi. Do đó, việc tích hợp mạng neural sâu vào factorization machines là một hướng đi tự nhiên. Việc thêm các lớp biến đổi phi tuyến vào factorization machines giúp mô hình có khả năng mô hình hóa cả các tổ hợp đặc trưng bậc thấp và bậc cao. Ngoài ra, mạng neural sâu cũng có thể nắm bắt các cấu trúc phi tuyến vốn có từ đầu vào. Trong phần này, chúng ta sẽ giới thiệu một mô hình tiêu biểu có tên deep factorization machines (DeepFM) :cite:`Guo.Tang.Ye.ea.2017` kết hợp FM và mạng neural sâu.


## Kiến trúc mô hình

DeepFM bao gồm một thành phần FM và một thành phần deep được tích hợp theo cấu trúc song song. Thành phần FM giống với factorization machines 2-way, được dùng để mô hình hóa các tương tác đặc trưng bậc thấp. Thành phần deep là một MLP, được dùng để nắm bắt các tương tác đặc trưng bậc cao và các tính phi tuyến. Hai thành phần này chia sẻ cùng đầu vào/embeddings, và đầu ra của chúng được cộng lại để tạo thành dự đoán cuối cùng. Cần lưu ý rằng tinh thần của DeepFM tương tự kiến trúc Wide \& Deep, vốn có thể nắm bắt cả memorization và generalization. Ưu điểm của DeepFM so với mô hình Wide \& Deep là giảm công sức feature engineering thủ công bằng cách tự động xác định các tổ hợp đặc trưng.

Để ngắn gọn, chúng tôi bỏ qua phần mô tả thành phần FM và ký hiệu đầu ra là $\hat{y}^{(FM)}$. Người đọc có thể tham khảo phần cuối để biết thêm chi tiết. Gọi $\mathbf{e}_i \in \mathbb{R}^{k}$ là vector đặc trưng latent của trường $i^\textrm{th}$. Đầu vào của thành phần deep là phép nối các embedding dense của tất cả các trường được tra cứu bằng đầu vào đặc trưng phân loại sparse, được ký hiệu như sau:

$$
\mathbf{z}^{(0)}  = [\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_f],
$$

trong đó $f$ là số lượng trường. Sau đó, đầu vào này được đưa vào mạng neural sau:

$$
\mathbf{z}^{(l)}  = \alpha(\mathbf{W}^{(l)}\mathbf{z}^{(l-1)} + \mathbf{b}^{(l)}),
$$

trong đó $\alpha$ là hàm activation. $\mathbf{W}_{l}$ và $\mathbf{b}_{l}$ lần lượt là weight và bias tại layer $l^\textrm{th}$. Gọi $y_{DNN}$ là đầu ra của dự đoán. Dự đoán cuối cùng của DeepFM là tổng đầu ra từ FM và DNN. Do đó, ta có:

$$
\hat{y} = \sigma(\hat{y}^{(FM)} + \hat{y}^{(DNN)}),
$$

trong đó $\sigma$ là hàm sigmoid. Kiến trúc của DeepFM được minh họa bên dưới.
![Minh họa mô hình DeepFM](../img/rec-deepfm.svg)

Cần lưu ý rằng DeepFM không phải là cách duy nhất để kết hợp mạng neural sâu với FM. Chúng ta cũng có thể thêm các layer phi tuyến lên trên các tương tác đặc trưng :cite:`He.Chua.2017`.

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## Triển khai DeepFM
Việc triển khai DeepFM tương tự như triển khai FM. Chúng ta giữ nguyên phần FM và sử dụng một block MLP với `relu` làm hàm activation. Dropout cũng được sử dụng để regularize mô hình. Số lượng neuron của MLP có thể được điều chỉnh bằng hyperparameter `mlp_dims`.

```{.python .input  n=2}
#@tab mxnet
class DeepFM(nn.Block):
    def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)
        input_dim = self.embed_output_dim = len(field_dims) * num_factors
        self.mlp = nn.Sequential()
        for dim in mlp_dims:
            self.mlp.add(nn.Dense(dim, 'relu', True, in_units=input_dim))
            self.mlp.add(nn.Dropout(rate=drop_rate))
            input_dim = dim
        self.mlp.add(nn.Dense(in_units=input_dim, units=1))

    def forward(self, x):
        embed_x = self.embedding(x)
        square_of_sum = np.sum(embed_x, axis=1) ** 2
        sum_of_square = np.sum(embed_x ** 2, axis=1)
        inputs = np.reshape(embed_x, (-1, self.embed_output_dim))
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True) \
            + self.mlp(inputs)
        x = npx.sigmoid(x)
        return x
```

## Huấn luyện và đánh giá mô hình
Quy trình tải dữ liệu giống với FM. Chúng ta thiết lập thành phần MLP của DeepFM thành một mạng dense ba layer có cấu trúc hình kim tự tháp (30-20-10). Tất cả hyperparameter khác vẫn giữ nguyên như FM.

```{.python .input  n=4}
#@tab mxnet
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
field_dims = train_data.field_dims
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
devices = d2l.try_all_gpus()
net = DeepFM(field_dims, num_factors=10, mlp_dims=[30, 20, 10])
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.01, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

So với FM, DeepFM hội tụ nhanh hơn và đạt hiệu năng tốt hơn.

## Tóm tắt

* Việc tích hợp mạng neural vào FM cho phép mô hình hóa các tương tác phức tạp và bậc cao.
* DeepFM vượt trội FM gốc trên tập dữ liệu quảng cáo.

## Bài tập

* Thay đổi cấu trúc của MLP để kiểm tra ảnh hưởng của nó đến hiệu năng mô hình.
* Thay đổi tập dữ liệu thành Criteo và so sánh với mô hình FM gốc.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/407)
:end_tab: