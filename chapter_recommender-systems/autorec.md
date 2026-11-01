# AutoRec: Dự đoán đánh giá bằng Autoencoder

Mặc dù mô hình phân rã ma trận đạt hiệu năng khá tốt trong tác vụ dự đoán đánh giá, về bản chất đây vẫn là một mô hình tuyến tính. Do đó, các mô hình này không có khả năng nắm bắt những mối quan hệ phức tạp, phi tuyến và tinh vi có thể dự đoán sở thích của người dùng. Trong phần này, chúng ta giới thiệu một mô hình neural network phi tuyến cho collaborative filtering, AutoRec :cite:`Sedhain.Menon.Sanner.ea.2015`. Mô hình này xác định collaborative filtering (CF) bằng kiến trúc autoencoder và hướng tới việc tích hợp các phép biến đổi phi tuyến vào CF dựa trên explicit feedback. Neural network đã được chứng minh là có khả năng xấp xỉ mọi hàm liên tục, khiến chúng phù hợp để khắc phục hạn chế của matrix factorization và tăng cường khả năng biểu đạt của matrix factorization.

Một mặt, AutoRec có cùng cấu trúc với một autoencoder, gồm một input layer, một hidden layer và một reconstruction layer (output layer). Autoencoder là một neural network học cách sao chép input sang output để mã hóa input thành các biểu diễn trong hidden layer (thường có số chiều thấp). Trong AutoRec, thay vì embedding người dùng/item một cách rõ ràng vào không gian có số chiều thấp, mô hình sử dụng cột/hàng của interaction matrix làm input, sau đó tái tạo interaction matrix tại output layer.

Mặt khác, AutoRec khác với autoencoder truyền thống: thay vì học các biểu diễn ẩn, AutoRec tập trung vào việc học/tái tạo output layer. Mô hình sử dụng một interaction matrix được quan sát một phần làm input, nhằm tái tạo một rating matrix hoàn chỉnh. Đồng thời, các phần tử bị thiếu trong input được điền vào output layer thông qua quá trình tái tạo để phục vụ mục đích recommendation.

Có hai biến thể của AutoRec: user-based và item-based. Để ngắn gọn, ở đây chúng ta chỉ giới thiệu item-based AutoRec. User-based AutoRec có thể được suy ra tương tự.


## Mô hình

Gọi $\mathbf{R}_{*i}$ là cột thứ $i^\textrm{th}$ của rating matrix, trong đó các đánh giá chưa biết được mặc định đặt thành 0. Kiến trúc neural được định nghĩa như sau:

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

trong đó $f(\cdot)$ và $g(\cdot)$ biểu diễn các activation function, $\mathbf{W}$ và $\mathbf{V}$ là các weight matrix, còn $\mu$ và $b$ là các bias. Gọi $h( \cdot )$ là toàn bộ network của AutoRec. Output $h(\mathbf{R}_{*i})$ là phần tái tạo của cột thứ $i^\textrm{th}$ trong rating matrix.

Hàm mục tiêu sau đây nhằm tối thiểu hóa sai số tái tạo:

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$

trong đó $\| \cdot \|_{\mathcal{O}}$ nghĩa là chỉ tính phần đóng góp của các đánh giá đã quan sát, tức là chỉ các trọng số liên kết với các input đã quan sát mới được cập nhật trong quá trình back-propagation.

```{.python .input  n=3}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()
```

## Triển khai mô hình

Một autoencoder điển hình gồm encoder và decoder. Encoder ánh xạ input thành các biểu diễn ẩn, còn decoder ánh xạ hidden layer tới reconstruction layer. Chúng ta tuân theo cách làm này và tạo encoder, decoder bằng các fully connected layer. Activation của encoder mặc định được đặt là `sigmoid` và decoder không áp dụng activation nào. Dropout được thêm vào sau phép biến đổi encoding để giảm over-fitting. Gradient của các input chưa được quan sát sẽ bị mask để đảm bảo rằng chỉ các đánh giá đã quan sát mới đóng góp vào quá trình học của mô hình.

```{.python .input  n=2}
#@tab mxnet
class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation='sigmoid',
                                use_bias=True)
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # Mask the gradient during training
            return pred * np.sign(input)
        else:
            return pred
```

## Triển khai lại Evaluator

Do input và output đã thay đổi, chúng ta cần triển khai lại hàm đánh giá, trong khi vẫn sử dụng RMSE làm thước đo độ chính xác.

```{.python .input  n=3}
#@tab mxnet
def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, devices, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## Huấn luyện và đánh giá mô hình

Bây giờ, hãy huấn luyện và đánh giá AutoRec trên tập dữ liệu MovieLens. Chúng ta có thể thấy rõ rằng RMSE trên tập kiểm tra thấp hơn so với mô hình matrix factorization, qua đó xác nhận hiệu quả của neural network trong tác vụ dự đoán đánh giá.

```{.python .input  n=4}
#@tab mxnet
devices = d2l.try_all_gpus()
# Load the MovieLens 100K dataset
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users,
                                                num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users,
                                               num_items)
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch="rollover", batch_size=256,
                                   num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                  last_batch="keep", batch_size=1024,
                                  num_workers=d2l.get_dataloader_workers())
# Model initialization, training, and evaluation
net = AutoRec(500, num_users)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
d2l.train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices, evaluator, inter_mat=test_inter_mat)
```

## Tóm tắt

* Chúng ta có thể biểu diễn thuật toán matrix factorization bằng autoencoder, đồng thời tích hợp các layer phi tuyến và regularization bằng dropout.
* Các thử nghiệm trên tập dữ liệu MovieLens 100K cho thấy AutoRec đạt hiệu năng vượt trội hơn matrix factorization.



## Bài tập

* Thay đổi hidden dimension của AutoRec để xem ảnh hưởng của nó tới hiệu năng mô hình.
* Hãy thử thêm nhiều hidden layer hơn. Điều này có giúp cải thiện hiệu năng mô hình không?
* Bạn có thể tìm được sự kết hợp tốt hơn giữa activation function của decoder và encoder không?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/401)
:end_tab: