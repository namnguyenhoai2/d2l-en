```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Dự đoán Giá Nhà trên Kaggle
:label:`sec_kaggle_house`

Giờ đây, khi chúng ta đã giới thiệu một số công cụ cơ bản
để xây dựng và huấn luyện các mạng sâu
cũng như regularize chúng bằng các kỹ thuật bao gồm
weight decay và dropout,
chúng ta đã sẵn sàng áp dụng toàn bộ kiến thức này vào thực tế
bằng cách tham gia một cuộc thi Kaggle.
Cuộc thi dự đoán giá nhà
là một nơi tuyệt vời để bắt đầu.
Dữ liệu khá phổ quát và không có cấu trúc đặc biệt
có thể yêu cầu các mô hình chuyên biệt (như âm thanh hoặc video có thể cần).
Bộ dữ liệu này, do :citet:`De-Cock.2011` thu thập,
bao gồm giá nhà tại Ames, Iowa trong giai đoạn 2006--2010.
Nó lớn hơn đáng kể so với bộ dữ liệu nhà ở [Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) nổi tiếng của Harrison và Rubinfeld (1978),
với cả nhiều mẫu hơn lẫn nhiều đặc trưng hơn.


Trong phần này, chúng tôi sẽ hướng dẫn chi tiết
về tiền xử lý dữ liệu, thiết kế mô hình và lựa chọn hyperparameter.
Chúng tôi hy vọng rằng thông qua cách tiếp cận thực hành,
bạn sẽ có được những trực giác giúp định hướng
cho sự nghiệp của mình với tư cách là một nhà khoa học dữ liệu.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd

npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import pandas as pd
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd
```

## Tải xuống Dữ liệu

Xuyên suốt cuốn sách, chúng ta sẽ huấn luyện và kiểm thử mô hình
trên nhiều bộ dữ liệu đã tải xuống khác nhau.
Tại đây, chúng ta (**triển khai hai hàm tiện ích**)
để tải xuống và giải nén các tệp zip hoặc tar.
Một lần nữa, chúng ta bỏ qua các chi tiết triển khai
của những hàm tiện ích như vậy.

```{.python .input  n=2}
%%tab all
def download(url, folder, sha1_hash=None):
    """Download a file to folder and return the local filepath."""

def extract(filename, folder):
    """Extract a zip/tar file into folder."""
```

## Kaggle

[Kaggle](https://www.kaggle.com) là một nền tảng phổ biến
tổ chức các cuộc thi machine learning.
Mỗi cuộc thi xoay quanh một bộ dữ liệu và nhiều cuộc thi
được tài trợ bởi các bên liên quan trao giải thưởng
cho những giải pháp chiến thắng.
Nền tảng này giúp người dùng tương tác
thông qua diễn đàn và mã nguồn được chia sẻ,
thúc đẩy cả cộng tác lẫn cạnh tranh.
Mặc dù việc chạy theo bảng xếp hạng thường trở nên mất kiểm soát,
khi các nhà nghiên cứu chỉ tập trung thiển cận vào các bước tiền xử lý
thay vì đặt ra những câu hỏi nền tảng,
nền tảng này vẫn mang lại giá trị to lớn nhờ tính khách quan
trong việc hỗ trợ so sánh định lượng trực tiếp
giữa các phương pháp cạnh tranh, cũng như chia sẻ mã nguồn
để mọi người có thể học được điều gì hiệu quả và điều gì không.
Nếu muốn tham gia một cuộc thi Kaggle,
trước tiên bạn cần đăng ký tài khoản
(xem :numref:`fig_kaggle`).

![Trang web Kaggle.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

Trên trang cuộc thi dự đoán giá nhà, như minh họa
trong :numref:`fig_house_pricing`,
bạn có thể tìm bộ dữ liệu (trong tab "Data"),
gửi dự đoán và xem thứ hạng của mình.
URL ở ngay đây:

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![Trang cuộc thi dự đoán giá nhà.](../img/house-pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## Truy cập và Đọc Bộ dữ liệu

Lưu ý rằng dữ liệu cuộc thi được chia thành
tập huấn luyện và tập kiểm thử.
Mỗi bản ghi bao gồm giá trị bất động sản của ngôi nhà
và các thuộc tính như loại đường, năm xây dựng,
loại mái, tình trạng tầng hầm, v.v.
Các đặc trưng gồm nhiều kiểu dữ liệu khác nhau.
Ví dụ, năm xây dựng
được biểu diễn bằng một số nguyên,
loại mái bằng các nhãn phân loại rời rạc,
và những đặc trưng khác bằng số dấu phẩy động.
Và đây là lúc thực tế làm mọi thứ phức tạp hơn:
đối với một số mẫu, một số dữ liệu bị thiếu hoàn toàn,
trong đó giá trị thiếu được đánh dấu đơn giản là "na".
Giá của mỗi ngôi nhà chỉ được bao gồm
trong tập huấn luyện
(dù sao đây cũng là một cuộc thi).
Chúng ta sẽ cần phân chia tập huấn luyện
để tạo tập validation,
nhưng chỉ có thể đánh giá các mô hình trên tập kiểm thử chính thức
sau khi tải dự đoán lên Kaggle.
Tab "Data" trên trang cuộc thi
trong :numref:`fig_house_pricing`
có các liên kết để tải dữ liệu xuống.

Để bắt đầu, chúng ta sẽ [**đọc và xử lý dữ liệu
bằng `pandas`**], công cụ mà chúng ta đã giới thiệu trong :numref:`sec_pandas`.
Để thuận tiện, chúng ta có thể tải xuống và lưu cache
bộ dữ liệu nhà ở Kaggle.
Nếu một tệp tương ứng với bộ dữ liệu này đã tồn tại trong thư mục cache và SHA-1 của nó khớp với `sha1_hash`, mã của chúng ta sẽ sử dụng tệp đã cache để tránh làm nghẽn Internet của bạn bằng các lượt tải xuống dư thừa.

```{.python .input  n=30}
%%tab all
class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            self.raw_train = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_train.csv', self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_test.csv', self.root,
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))
```

Bộ dữ liệu huấn luyện gồm 1460 mẫu,
80 đặc trưng và một nhãn, trong khi dữ liệu validation
chứa 1459 mẫu và 80 đặc trưng.

```{.python .input  n=31}
%%tab all
data = KaggleHouse(batch_size=64)
print(data.raw_train.shape)
print(data.raw_val.shape)
```

## Tiền xử lý Dữ liệu

Hãy [**xem bốn đặc trưng đầu tiên và hai đặc trưng cuối cùng
cùng với nhãn (SalePrice)**] từ bốn mẫu đầu tiên.

```{.python .input  n=10}
%%tab all
print(data.raw_train.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])
```

Chúng ta có thể thấy rằng trong mỗi mẫu, đặc trưng đầu tiên là mã định danh.
Điều này giúp mô hình xác định từng mẫu huấn luyện.
Dù thuận tiện, nó không mang
bất kỳ thông tin nào phục vụ mục đích dự đoán.
Do đó, chúng ta sẽ loại bỏ nó khỏi bộ dữ liệu
trước khi đưa dữ liệu vào mô hình.
Hơn nữa, do có nhiều kiểu dữ liệu khác nhau,
chúng ta cần tiền xử lý dữ liệu trước khi bắt đầu xây dựng mô hình.


Hãy bắt đầu với các đặc trưng số.
Trước tiên, chúng ta áp dụng một heuristic,
[**thay thế mọi giá trị thiếu
bằng giá trị trung bình của đặc trưng tương ứng.**]
Sau đó, để đưa mọi đặc trưng về một thang đo chung,
chúng ta (***chuẩn hóa* dữ liệu bằng cách
rescale các đặc trưng về trung bình bằng không và phương sai đơn vị**):

$$x \leftarrow \frac{x - \mu}{\sigma},$$

trong đó $\mu$ và $\sigma$ lần lượt biểu thị trung bình và độ lệch chuẩn.
Để xác minh rằng điều này thực sự biến đổi
đặc trưng (biến) của chúng ta sao cho có trung bình bằng không và phương sai đơn vị,
lưu ý rằng $E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$
và $E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$.
Theo trực giác, chúng ta chuẩn hóa dữ liệu
vì hai lý do.
Thứ nhất, điều này thuận tiện cho việc optimization.
Thứ hai, bởi vì chúng ta không biết *a priori*
đặc trưng nào sẽ liên quan,
chúng ta không muốn phạt các hệ số
được gán cho một đặc trưng nhiều hơn bất kỳ đặc trưng nào khác.

[**Tiếp theo, chúng ta xử lý các giá trị rời rạc.**]
Chúng bao gồm các đặc trưng như "MSZoning".
(**Chúng ta thay thế chúng bằng one-hot encoding**)
theo cùng cách mà trước đó chúng ta đã biến đổi
các nhãn đa lớp thành vector (xem :numref:`subsec_classification-problem`).
Ví dụ, "MSZoning" nhận các giá trị "RL" và "RM".
Sau khi loại bỏ đặc trưng "MSZoning",
hai đặc trưng chỉ báo mới
"MSZoning_RL" và "MSZoning_RM" được tạo ra với các giá trị là 0 hoặc 1.
Theo one-hot encoding,
nếu giá trị gốc của "MSZoning" là "RL",
thì "MSZoning_RL" là 1 và "MSZoning_RM" là 0.
Package `pandas` tự động thực hiện việc này cho chúng ta.

```{.python .input  n=32}
%%tab all
@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    # Remove the ID and label columns
    label = 'SalePrice'
    features = pd.concat(
        (self.raw_train.drop(columns=['Id', label]),
         self.raw_val.drop(columns=['Id'])))
    # Standardize numerical columns
    numeric_features = features.dtypes[features.dtypes!='object'].index
    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # Replace NAN numerical features by 0
    features[numeric_features] = features[numeric_features].fillna(0)
    # Replace discrete features by one-hot encoding
    features = pd.get_dummies(features, dummy_na=True)
    # Save preprocessed features
    self.train = features[:self.raw_train.shape[0]].copy()
    self.train[label] = self.raw_train[label]
    self.val = features[self.raw_train.shape[0]:].copy()
```

Bạn có thể thấy rằng phép chuyển đổi này làm tăng
số lượng đặc trưng từ 79 lên 331 (không tính các cột ID và nhãn).

```{.python .input  n=33}
%%tab all
data.preprocess()
data.train.shape
```

## Thước đo Sai số

Để bắt đầu, chúng ta sẽ huấn luyện một mô hình tuyến tính với squared loss. Không có gì ngạc nhiên, mô hình tuyến tính của chúng ta sẽ không tạo ra một bài nộp chiến thắng trong cuộc thi, nhưng nó cung cấp một kiểm tra hợp lý để xem liệu dữ liệu có chứa thông tin có ý nghĩa hay không. Nếu ở đây chúng ta không thể làm tốt hơn việc đoán ngẫu nhiên, thì rất có thể đã có lỗi trong quá trình xử lý dữ liệu. Và nếu mọi thứ hoạt động, mô hình tuyến tính sẽ đóng vai trò baseline, cho chúng ta trực giác về mức độ gần của mô hình đơn giản này so với các mô hình tốt nhất được báo cáo, từ đó biết được mức cải thiện có thể kỳ vọng từ các mô hình phức tạp hơn.

Với giá nhà, cũng như giá cổ phiếu,
chúng ta quan tâm đến các đại lượng tương đối
hơn là các đại lượng tuyệt đối.
Do đó, [**chúng ta thường quan tâm nhiều hơn đến
sai số tương đối $\frac{y - \hat{y}}{y}$**]
so với sai số tuyệt đối $y - \hat{y}$.
Ví dụ, nếu dự đoán của chúng ta sai lệch \$100,000
khi ước tính giá một ngôi nhà ở vùng nông thôn Ohio,
nơi giá trị của một ngôi nhà điển hình là \$125,000,
thì có lẽ chúng ta đang làm rất tệ.
Mặt khác, nếu sai lệch cùng số tiền này
tại Los Altos Hills, California,
đó có thể là một dự đoán chính xác đáng kinh ngạc
(ở đó, giá nhà trung vị vượt quá \$4 triệu).

(**Một cách để giải quyết vấn đề này là
đo chênh lệch trong logarit của các ước lượng giá.**)
Trên thực tế, đây cũng là thước đo sai số chính thức
mà cuộc thi sử dụng để đánh giá chất lượng các bài nộp.
Xét cho cùng, một giá trị nhỏ $\delta$ với $|\log y - \log \hat{y}| \leq \delta$
tương ứng với $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$.
Điều này dẫn đến root-mean-squared-error sau đây giữa logarit của giá dự đoán và logarit của giá nhãn:

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input  n=60}
%%tab all
@d2l.add_to_class(KaggleHouse)
def get_dataloader(self, train):
    label = 'SalePrice'
    data = self.train if train else self.val
    if label not in data: return
    get_tensor = lambda x: d2l.tensor(x.values.astype(float),
                                      dtype=d2l.float32)
    # Logarithm of prices 
    tensors = (get_tensor(data.drop(columns=[label])),  # X
               d2l.reshape(d2l.log(get_tensor(data[label])), (-1, 1)))  # Y
    return self.get_tensorloader(tensors, train)
```

## Cross-Validation $K$-Fold

Bạn có thể nhớ rằng chúng ta đã giới thiệu [**cross-validation**]
trong :numref:`subsec_generalization-model-selection`, nơi chúng ta đã thảo luận
cách xử lý việc lựa chọn mô hình.
Chúng ta sẽ tận dụng kỹ thuật này để chọn thiết kế mô hình
và điều chỉnh các hyperparameter.
Trước tiên, chúng ta cần một hàm trả về
fold thứ $i^\textrm{th}$ của dữ liệu
trong quy trình cross-validation $K$-fold.
Hàm này tiến hành bằng cách tách đoạn thứ $i^\textrm{th}$
làm dữ liệu validation và trả về phần còn lại làm dữ liệu huấn luyện.
Lưu ý rằng đây không phải là cách xử lý dữ liệu hiệu quả nhất
và chắc chắn chúng ta sẽ làm điều gì đó thông minh hơn nhiều
nếu bộ dữ liệu lớn hơn đáng kể.
Nhưng sự phức tạp bổ sung này có thể làm mã nguồn của chúng ta khó hiểu không cần thiết,
vì vậy chúng ta có thể bỏ qua nó một cách an toàn ở đây nhờ tính đơn giản của bài toán.

```{.python .input}
%%tab all
def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k
    for j in range(k):
        idx = range(j * fold_size, (j+1) * fold_size)
        rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx),  
                                data.train.loc[idx]))    
    return rets
```

[**Sai số validation trung bình được trả về**]
khi chúng ta huấn luyện $K$ lần trong cross-validation $K$-fold.

```{.python .input}
%%tab all
def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = d2l.LinearRegression(lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models
```

## [**Lựa chọn Mô hình**]

Trong ví dụ này, chúng ta chọn một tập hyperparameter chưa được tinh chỉnh
và để người đọc tự cải thiện mô hình.
Việc tìm ra một lựa chọn tốt có thể mất thời gian,
tùy thuộc vào số lượng biến mà ta tối ưu.
Với một tập dữ liệu đủ lớn,
và các loại hyperparameter thông thường,
cross-validation $K$-fold thường
khá vững vàng trước multiple testing.
Tuy nhiên, nếu thử một số lượng lựa chọn lớn một cách không hợp lý,
ta có thể thấy rằng hiệu năng validation
không còn đại diện cho lỗi thực tế.

```{.python .input}
%%tab all
trainer = d2l.Trainer(max_epochs=10)
models = k_fold(trainer, data, k=5, lr=0.01)
```

Lưu ý rằng đôi khi số lỗi training
đối với một tập hyperparameter có thể rất thấp,
ngay cả khi số lỗi trên cross-validation $K$-fold
tăng cao đáng kể.
Điều này cho thấy chúng ta đang overfitting.
Trong suốt quá trình training, bạn nên theo dõi cả hai con số.
Ít overfitting hơn có thể cho thấy dữ liệu của chúng ta có thể hỗ trợ một mô hình mạnh hơn.
Overfitting nghiêm trọng có thể gợi ý rằng chúng ta có thể cải thiện
bằng cách áp dụng các kỹ thuật regularization.

##  [**Gửi Dự đoán lên Kaggle**]

Bây giờ khi đã biết một lựa chọn hyperparameter tốt nên như thế nào,
chúng ta có thể 
tính trung bình các dự đoán
trên tập test
từ toàn bộ $K$ mô hình.
Lưu các dự đoán trong một tệp csv
sẽ đơn giản hóa việc tải kết quả lên Kaggle.
Đoạn mã sau sẽ tạo một tệp có tên `submission.csv`.

```{.python .input}
%%tab all
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = [model(d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
             for model in models]
if tab.selected('jax'):
    preds = [model.apply({'params': trainer.state.params},
             d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
             for model in models]
# Taking exponentiation of predictions in the logarithm scale
ensemble_preds = d2l.reduce_mean(d2l.exp(d2l.concat(preds, 1)), 1)
submission = pd.DataFrame({'Id':data.raw_val.Id,
                           'SalePrice':d2l.numpy(ensemble_preds)})
submission.to_csv('submission.csv', index=False)
```

Tiếp theo, như được minh họa trong :numref:`fig_kaggle_submit2`,
chúng ta có thể gửi các dự đoán lên Kaggle
và xem chúng so sánh như thế nào với giá nhà thực tế (nhãn)
trong tập test.
Các bước khá đơn giản:

* Đăng nhập vào trang web Kaggle và truy cập trang cuộc thi dự đoán giá nhà.
* Nhấp vào nút “Submit Predictions” hoặc “Late Submission”.
* Nhấp vào nút “Upload Submission File” trong khung nét đứt ở cuối trang và chọn tệp dự đoán bạn muốn tải lên.
* Nhấp vào nút “Make Submission” ở cuối trang để xem kết quả.

![Gửi dữ liệu lên Kaggle.](../img/kaggle-submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## Tóm tắt và Thảo luận

Dữ liệu thực tế thường chứa sự kết hợp của nhiều kiểu dữ liệu khác nhau và cần được tiền xử lý.
Chuẩn hóa lại dữ liệu có giá trị thực về trung bình bằng không và phương sai đơn vị là một lựa chọn mặc định tốt. Thay thế các giá trị thiếu bằng giá trị trung bình của chúng cũng vậy.
Hơn nữa, chuyển đổi các đặc trưng categorical thành các đặc trưng indicator cho phép chúng ta xử lý chúng như các vector one-hot.
Khi chúng ta có xu hướng quan tâm nhiều hơn đến
lỗi tương đối thay vì lỗi tuyệt đối,
chúng ta có thể 
đo độ chênh lệch trong logarit của dự đoán.
Để chọn mô hình và điều chỉnh hyperparameter,
chúng ta có thể sử dụng cross-validation $K$-fold .



## Bài tập

1. Gửi các dự đoán của bạn trong phần này lên Kaggle. Chúng tốt đến mức nào?
1. Có phải lúc nào thay thế các giá trị thiếu bằng giá trị trung bình cũng là một ý tưởng tốt không? Gợi ý: bạn có thể xây dựng một tình huống trong đó các giá trị không bị thiếu ngẫu nhiên không?
1. Cải thiện điểm số bằng cách tinh chỉnh hyperparameter thông qua cross-validation $K$-fold.
1. Cải thiện điểm số bằng cách cải thiện mô hình (ví dụ: layers, weight decay và dropout).
1. Điều gì xảy ra nếu chúng ta không chuẩn hóa các đặc trưng số liên tục như đã làm trong phần này?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/106)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/107)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/237)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17988)
:end_tab: