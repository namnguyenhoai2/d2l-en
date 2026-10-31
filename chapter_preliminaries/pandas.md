```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Tiền xử lý dữ liệu
:label:`sec_pandas`

Cho đến nay, chúng ta đã làm việc với dữ liệu tổng hợp
được cung cấp dưới dạng các tensor dựng sẵn.
Tuy nhiên, để áp dụng deep learning trong thực tế,
chúng ta phải trích xuất dữ liệu lộn xộn
được lưu trữ ở các định dạng tùy ý,
sau đó tiền xử lý dữ liệu để phù hợp với nhu cầu.
May mắn là thư viện *pandas* [library](https://pandas.pydata.org/)
có thể đảm nhiệm phần lớn công việc nặng nhọc.
Phần này không thể thay thế
một *pandas* [tutorial](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html) đầy đủ,
nhưng sẽ cung cấp cho bạn một khóa học cấp tốc
về một số quy trình thường dùng nhất.

## Đọc tập dữ liệu

Các tệp giá trị được phân tách bằng dấu phẩy (CSV) xuất hiện phổ biến
khi lưu trữ dữ liệu dạng bảng (giống bảng tính).
Trong các tệp này, mỗi dòng tương ứng với một bản ghi
và gồm một số trường (được phân tách bằng dấu phẩy), ví dụ:
"Albert Einstein,March 14 1879,Ulm,Federal polytechnic school,field of gravitational physics".
Để minh họa cách tải tệp CSV bằng `pandas`,
chúng ta (**tạo một tệp CSV bên dưới**) `../data/house_tiny.csv`.
Tệp này biểu diễn một tập dữ liệu về các ngôi nhà,
trong đó mỗi hàng tương ứng với một ngôi nhà riêng biệt
và các cột tương ứng với số phòng (`NumRooms`),
loại mái (`RoofType`) và giá (`Price`).

```{.python .input}
%%tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
```

Bây giờ hãy import `pandas` và tải tập dữ liệu bằng `read_csv`.

```{.python .input}
%%tab all
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## Chuẩn bị dữ liệu

Trong học có giám sát, chúng ta huấn luyện các mô hình
để dự đoán một giá trị *target* được chỉ định,
dựa trên một tập các giá trị *input*.
Bước đầu tiên khi xử lý tập dữ liệu
là tách các cột tương ứng
với giá trị đầu vào và giá trị mục tiêu.
Chúng ta có thể chọn các cột theo tên hoặc
thông qua lập chỉ mục dựa trên vị trí số nguyên (`iloc`).

Có thể bạn đã nhận thấy rằng `pandas` đã thay thế
tất cả các mục trong CSV có giá trị `NA`
bằng một giá trị đặc biệt `NaN` (*not a number*).
Điều này cũng có thể xảy ra khi một mục bị bỏ trống,
ví dụ: "3,,,270000".
Chúng được gọi là *missing values*
và là những "con rệp" của khoa học dữ liệu,
một mối đe dọa dai dẳng mà bạn sẽ phải đối mặt
trong suốt sự nghiệp của mình.
Tùy theo ngữ cảnh,
các giá trị thiếu có thể được xử lý
bằng *imputation* hoặc *deletion*.
Imputation thay thế các giá trị thiếu
bằng các ước lượng của chúng,
còn deletion đơn giản là loại bỏ
các hàng hoặc cột
chứa giá trị thiếu.

Dưới đây là một số heuristic imputation phổ biến.
[**Đối với các trường input phân loại,
chúng ta có thể coi `NaN` là một category.**]
Vì cột `RoofType` nhận các giá trị `Slate` và `NaN`,
`pandas` có thể chuyển đổi cột này
thành hai cột `RoofType_Slate` và `RoofType_nan`.
Một hàng có loại mái là `Slate` sẽ đặt các giá trị
của `RoofType_Slate` và `RoofType_nan` lần lượt là 1 và 0.
Điều ngược lại cũng đúng với một hàng có giá trị `RoofType` bị thiếu.

```{.python .input}
%%tab all
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

Đối với các giá trị số bị thiếu,
một heuristic phổ biến là
[**thay thế các mục `NaN` bằng
giá trị trung bình của cột tương ứng**].

```{.python .input}
%%tab all
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

## Chuyển đổi sang định dạng Tensor

Bây giờ, [**tất cả các mục trong `inputs` và `targets` đều là số,
chúng ta có thể tải chúng vào một tensor**] (hãy nhớ :numref:`sec_ndarray`).

```{.python .input}
%%tab mxnet
from mxnet import np

X, y = np.array(inputs.to_numpy(dtype=float)), np.array(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab pytorch
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

X = tf.constant(inputs.to_numpy(dtype=float))
y = tf.constant(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab jax
from jax import numpy as jnp

X = jnp.array(inputs.to_numpy(dtype=float))
y = jnp.array(targets.to_numpy(dtype=float))
X, y
```

## Thảo luận

Giờ đây bạn đã biết cách phân vùng các cột dữ liệu,
impute các biến bị thiếu
và tải dữ liệu `pandas` vào các tensor.
Trong :numref:`sec_kaggle_house`, bạn sẽ
học thêm một số kỹ năng xử lý dữ liệu.
Mặc dù khóa học cấp tốc này giữ mọi thứ đơn giản,
việc xử lý dữ liệu có thể trở nên phức tạp.
Ví dụ, thay vì được cung cấp trong một tệp CSV duy nhất,
tập dữ liệu của chúng ta có thể được phân tán
trong nhiều tệp được trích xuất từ một cơ sở dữ liệu quan hệ.
Chẳng hạn, trong một ứng dụng thương mại điện tử,
địa chỉ khách hàng có thể nằm trong một bảng
còn dữ liệu mua hàng nằm trong một bảng khác.
Hơn nữa, các chuyên gia phải xử lý vô số kiểu dữ liệu
ngoài dữ liệu phân loại và dữ liệu số, chẳng hạn như
chuỗi văn bản, hình ảnh,
dữ liệu âm thanh và point cloud.
Thông thường, cần có các công cụ nâng cao và thuật toán hiệu quả
để ngăn việc xử lý dữ liệu trở thành
nút thắt cổ chai lớn nhất trong pipeline machine learning.
Những vấn đề này sẽ xuất hiện khi chúng ta chuyển sang
computer vision và natural language processing.
Cuối cùng, chúng ta phải chú ý đến chất lượng dữ liệu.
Các tập dữ liệu trong thực tế thường bị ảnh hưởng
bởi các giá trị ngoại lai, phép đo sai từ cảm biến và lỗi ghi chép,
những vấn đề cần được xử lý trước khi
đưa dữ liệu vào bất kỳ mô hình nào.
Các công cụ trực quan hóa dữ liệu như [seaborn](https://seaborn.pydata.org/),
[Bokeh](https://docs.bokeh.org/) hoặc [matplotlib](https://matplotlib.org/)
có thể giúp bạn kiểm tra dữ liệu thủ công
và hình thành trực giác về
loại vấn đề mà bạn có thể cần xử lý.


## Bài tập

1. Hãy thử tải các tập dữ liệu, chẳng hạn như Abalone từ [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets) và kiểm tra các thuộc tính của chúng. Tỷ lệ bao nhiêu trong số chúng có giá trị thiếu? Tỷ lệ bao nhiêu biến là số, phân loại hoặc văn bản?
1. Hãy thử lập chỉ mục và chọn các cột dữ liệu theo tên thay vì theo số cột. Tài liệu pandas về [indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html) có thêm chi tiết về cách thực hiện việc này.
1. Bạn nghĩ mình có thể tải một tập dữ liệu lớn đến mức nào theo cách này? Những giới hạn nào có thể xảy ra? Gợi ý: hãy xem xét thời gian đọc dữ liệu, cách biểu diễn, việc xử lý và dung lượng bộ nhớ. Hãy thử trên máy tính xách tay của bạn. Điều gì xảy ra nếu bạn thử trên một server?
1. Bạn sẽ xử lý dữ liệu có số lượng category rất lớn như thế nào? Nếu các nhãn category đều là duy nhất thì sao? Bạn có nên đưa các nhãn đó vào không?
1. Bạn có thể nghĩ đến những lựa chọn thay thế nào cho pandas? Còn [loading NumPy tensors from a file](https://numpy.org/doc/stable/reference/generated/numpy.load.html) thì sao? Hãy xem [Pillow](https://python-pillow.org/), thư viện hình ảnh Python.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/195)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17967)
:end_tab: