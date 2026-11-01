# Các phép toán hình học và đại số tuyến tính
:label:`sec_geometry-linear-algebraic-ops`

Trong :numref:`sec_linear-algebra`, chúng ta đã tìm hiểu những kiến thức cơ bản về đại số tuyến tính
và thấy cách sử dụng nó để biểu diễn các phép toán phổ biến nhằm biến đổi dữ liệu.
Đại số tuyến tính là một trong những nền tảng toán học quan trọng
làm cơ sở cho phần lớn công việc chúng ta thực hiện trong deep learning
và machine learning nói chung.
Mặc dù :numref:`sec_linear-algebra` đã cung cấp đủ công cụ
để trình bày cơ chế của các mô hình deep learning hiện đại,
chủ đề này còn nhiều nội dung hơn thế.
Trong phần này, chúng ta sẽ tìm hiểu sâu hơn,
làm nổi bật một số cách diễn giải hình học của các phép toán đại số tuyến tính
và giới thiệu một vài khái niệm nền tảng, bao gồm trị riêng và vector riêng.

## Hình học của vector
Trước hết, chúng ta cần thảo luận về hai cách diễn giải hình học phổ biến của vector,
đó là xem chúng như các điểm hoặc các hướng trong không gian.
Về bản chất, một vector là một danh sách các số, chẳng hạn như danh sách Python dưới đây.

```{.python .input}
#@tab all
v = [1, 7, 0, 1]
```

Các nhà toán học thường viết nó dưới dạng vector *cột* hoặc vector *hàng*, tức là một trong hai dạng

$$
\mathbf{x} = \begin{bmatrix}1\\7\\0\\1\end{bmatrix},
$$

hoặc

$$
\mathbf{x}^\top = \begin{bmatrix}1 & 7 & 0 & 1\end{bmatrix}.
$$

Các dạng này thường có những cách diễn giải khác nhau,
trong đó các mẫu dữ liệu là vector cột
còn các trọng số được dùng để tạo tổng có trọng số là vector hàng.
Tuy nhiên, việc linh hoạt cũng có thể mang lại lợi ích.
Như đã mô tả trong :numref:`sec_linear-algebra`,
mặc dù hướng mặc định của một vector là vector cột,
đối với bất kỳ ma trận nào biểu diễn một tập dữ liệu dạng bảng,
việc xem mỗi mẫu dữ liệu là một vector hàng
trong ma trận
sẽ phổ biến hơn.

Với một vector, cách diễn giải đầu tiên
mà chúng ta nên nghĩ đến là xem nó như một điểm trong không gian.
Trong hai hoặc ba chiều, chúng ta có thể trực quan hóa các điểm này
bằng cách sử dụng các thành phần của vector để xác định
vị trí của các điểm trong không gian so với
một mốc cố định gọi là *gốc tọa độ*. Điều này được minh họa trong :numref:`fig_grid`.

![Minh họa việc trực quan hóa các vector như những điểm trên mặt phẳng. Thành phần thứ nhất của vector cho biết tọa độ $\mathit{x}$, thành phần thứ hai cho biết tọa độ $\mathit{y}$. Các chiều cao hơn cũng tương tự, mặc dù khó trực quan hóa hơn nhiều.](../img/grid-points.svg)
:label:`fig_grid`

Cách nhìn hình học này cho phép chúng ta xem xét bài toán ở mức độ trừu tượng hơn.
Thay vì phải đối mặt với một bài toán có vẻ không thể vượt qua
như phân loại hình ảnh thành mèo hoặc chó,
chúng ta có thể bắt đầu xem xét các nhiệm vụ một cách trừu tượng
như những tập hợp điểm trong không gian, và hình dung nhiệm vụ đó
là việc tìm ra cách tách biệt hai cụm điểm khác nhau.

Song song với đó, có một cách nhìn thứ hai
mà mọi người thường áp dụng cho vector: xem chúng như các hướng trong không gian.
Chúng ta không chỉ có thể xem vector $\mathbf{v} = [3,2]^\top$
là vị trí cách gốc tọa độ $3$ đơn vị về bên phải và $2$ đơn vị lên trên,
mà còn có thể xem nó là chính hướng đi
cần thực hiện để đi $3$ bước sang phải và $2$ bước lên trên.
Theo cách này, chúng ta xem tất cả các vector trong hình :numref:`fig_arrow` là giống nhau.

![Bất kỳ vector nào cũng có thể được trực quan hóa dưới dạng một mũi tên trên mặt phẳng. Trong trường hợp này, mọi vector được vẽ đều là biểu diễn của vector $(3,2)^\top$.](../img/par-vec.svg)
:label:`fig_arrow`

Một trong những lợi ích của sự chuyển đổi cách nhìn này là
chúng ta có thể trực quan hóa ý nghĩa của phép cộng vector.
Cụ thể, chúng ta đi theo hướng do một vector chỉ ra,
sau đó đi theo hướng do vector còn lại chỉ ra, như trong :numref:`fig_add-vec`.

![Chúng ta có thể trực quan hóa phép cộng vector bằng cách đi theo một vector trước, rồi đến vector còn lại.](../img/vec-add.svg)
:label:`fig_add-vec`

Phép trừ vector cũng có cách diễn giải tương tự.
Từ đẳng thức $\mathbf{u} = \mathbf{v} + (\mathbf{u}-\mathbf{v})$,
chúng ta thấy rằng vector $\mathbf{u}-\mathbf{v}$ là hướng
đưa chúng ta từ điểm $\mathbf{v}$ đến điểm $\mathbf{u}$.


## Tích vô hướng và góc
Như đã thấy trong :numref:`sec_linear-algebra`,
nếu lấy hai vector cột $\mathbf{u}$ và $\mathbf{v}$,
chúng ta có thể tính tích vô hướng của chúng bằng cách:

$$\mathbf{u}^\top\mathbf{v} = \sum_i u_i\cdot v_i.$$
:eqlabel:`eq_dot_def`

Vì :eqref:`eq_dot_def` có tính đối xứng, chúng ta sẽ sử dụng ký hiệu
tương tự như phép nhân cổ điển và viết

$$
\mathbf{u}\cdot\mathbf{v} = \mathbf{u}^\top\mathbf{v} = \mathbf{v}^\top\mathbf{u},
$$

để nhấn mạnh rằng việc đổi thứ tự các vector sẽ cho cùng một kết quả.

Tích vô hướng :eqref:`eq_dot_def` cũng có cách diễn giải hình học: nó liên hệ chặt chẽ với góc giữa hai vector. Hãy xét góc được minh họa trong :numref:`fig_angle`.

![Giữa mọi cặp vector trên mặt phẳng đều có một góc $\theta$ xác định rõ. Chúng ta sẽ thấy góc này liên hệ mật thiết với tích vô hướng.](../img/vec-angle.svg)
:label:`fig_angle`

Trước hết, hãy xét hai vector cụ thể:

$$
\mathbf{v} = (r,0) \; \textrm{and} \; \mathbf{w} = (s\cos(\theta), s \sin(\theta)).
$$

Vector $\mathbf{v}$ có độ dài $r$ và song song với trục $x$,
còn vector $\mathbf{w}$ có độ dài $s$ và tạo với trục $x$ một góc $\theta$.
Nếu tính tích vô hướng của hai vector này, chúng ta có

$$
\mathbf{v}\cdot\mathbf{w} = rs\cos(\theta) = \|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta).
$$

Sau một vài biến đổi đại số đơn giản, chúng ta có thể sắp xếp lại các số hạng để thu được

$$
\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).
$$

Tóm lại, đối với hai vector cụ thể này,
tích vô hướng kết hợp với chuẩn cho chúng ta biết góc giữa hai vector. Kết quả tương tự cũng đúng trong trường hợp tổng quát. Mặc dù không suy ra biểu thức này ở đây,
nếu xét $\|\mathbf{v} - \mathbf{w}\|^2$ theo hai cách:
một cách sử dụng tích vô hướng, cách còn lại sử dụng định lý cosin về mặt hình học,
chúng ta có thể thu được mối quan hệ đầy đủ.
Thật vậy, với hai vector bất kỳ $\mathbf{v}$ và $\mathbf{w}$,
góc giữa hai vector là

$$\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).$$
:eqlabel:`eq_angle_forumla`

Đây là một kết quả hữu ích vì không có gì trong phép tính phụ thuộc vào không gian hai chiều.
Thật vậy, chúng ta có thể áp dụng nó trong không gian ba chiều hoặc ba triệu chiều mà không gặp vấn đề gì.

Hãy xem cách tính góc giữa một cặp vector qua một ví dụ đơn giản:

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import gluon, np, npx
npx.set_np()

def angle(v, w):
    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))

angle(np.array([0, 1, 2]), np.array([2, 3, 4]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
from torchvision import transforms
import torchvision

def angle(v, w):
    return torch.acos(v.dot(w) / (torch.norm(v) * torch.norm(w)))

angle(torch.tensor([0, 1, 2], dtype=torch.float32), torch.tensor([2.0, 3, 4]))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

def angle(v, w):
    return tf.acos(tf.tensordot(v, w, axes=1) / (tf.norm(v) * tf.norm(w)))

angle(tf.constant([0, 1, 2], dtype=tf.float32), tf.constant([2.0, 3, 4]))
```

Hiện tại chúng ta chưa sử dụng khái niệm này, nhưng cần biết rằng
chúng ta gọi các vector có góc bằng $\pi/2$
(hoặc tương đương là $90^{\circ}$) là *trực giao*.
Bằng cách xem xét phương trình trên, chúng ta thấy điều này xảy ra khi $\theta = \pi/2$,
tương đương với $\cos(\theta) = 0$.
Điều này chỉ có thể xảy ra khi bản thân tích vô hướng bằng không,
và hai vector trực giao khi và chỉ khi $\mathbf{v}\cdot\mathbf{w} = 0$.
Đây sẽ là một công thức hữu ích khi tìm hiểu các đối tượng theo góc nhìn hình học.

Có thể đặt câu hỏi: tại sao việc tính góc lại hữu ích?
Câu trả lời nằm ở dạng tính bất biến mà chúng ta kỳ vọng dữ liệu có.
Hãy xét một hình ảnh và một bản sao của hình ảnh đó,
trong đó giá trị của mọi pixel đều giống nhau nhưng độ sáng chỉ bằng $10\%$.
Nhìn chung, giá trị của từng pixel khác xa giá trị ban đầu.
Vì vậy, nếu tính khoảng cách giữa hình ảnh gốc và hình ảnh tối hơn,
khoảng cách có thể rất lớn.
Tuy nhiên, đối với hầu hết các ứng dụng ML, *nội dung* vẫn giống nhau---đối với một bộ phân loại mèo/chó,
đó vẫn là hình ảnh của một con mèo.
Tuy nhiên, nếu xét góc, không khó để thấy
rằng với mọi vector $\mathbf{v}$, góc
giữa $\mathbf{v}$ và $0.1\cdot\mathbf{v}$ bằng không.
Điều này tương ứng với thực tế là phép co giãn vector
giữ nguyên hướng và chỉ thay đổi độ dài.
Góc xem hình ảnh tối hơn là giống hệt hình ảnh ban đầu.

Những ví dụ như vậy xuất hiện ở khắp nơi.
Trong văn bản, chúng ta có thể muốn chủ đề được thảo luận
không thay đổi nếu viết một tài liệu dài gấp đôi nhưng vẫn nói cùng một nội dung.
Với một số cách mã hóa (chẳng hạn như đếm số lần xuất hiện của các từ trong một vocabulary), điều này tương ứng với việc nhân đôi vector mã hóa tài liệu,
vì vậy một lần nữa chúng ta có thể sử dụng góc.

### Độ tương đồng cosine
Trong các ngữ cảnh ML, khi góc được dùng
để đo mức độ gần nhau của hai vector,
các chuyên gia sử dụng thuật ngữ *độ tương đồng cosine*
để chỉ phần
$$
\cos(\theta) = \frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}.
$$

Cosine đạt giá trị lớn nhất là $1$
khi hai vector cùng hướng,
giá trị nhỏ nhất là $-1$ khi chúng ngược hướng,
và giá trị bằng $0$ khi hai vector trực giao.
Lưu ý rằng nếu các thành phần của các vector nhiều chiều
được lấy mẫu ngẫu nhiên với giá trị trung bình bằng $0$,
cosine của chúng gần như luôn gần bằng $0$.


## Siêu phẳng

Ngoài việc làm việc với các vector, một đối tượng quan trọng khác mà bạn phải hiểu rõ nếu muốn tiến xa trong đại số tuyến tính là *siêu phẳng* (hyperplane), một khái quát hóa trong các số chiều cao hơn của đường thẳng (hai chiều) hoặc mặt phẳng (ba chiều). Trong một không gian vector $d$ chiều, một siêu phẳng có $d-1$ chiều và chia không gian thành hai nửa không gian.

Hãy bắt đầu với một ví dụ. Giả sử ta có một vector cột $\mathbf{w}=[2,1]^\top$. Ta muốn biết: “các điểm $\mathbf{v}$ nào thỏa mãn $\mathbf{w}\cdot\mathbf{v} = 1$?”  
Bằng cách nhớ lại mối liên hệ giữa tích vô hướng và góc ở trên :eqref:`eq_angle_forumla`, ta có thể thấy rằng điều này tương đương với
$$
\|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta) = 1 \; \iff \; \|\mathbf{v}\|\cos(\theta) = \frac{1}{\|\mathbf{w}\|} = \frac{1}{\sqrt{5}}.
$$

![Nhắc lại lượng giác, ta thấy công thức $\|\mathbf{v}\|\cos(\theta)$ là độ dài hình chiếu của vector $\mathbf{v}$ lên hướng của $\mathbf{w}$](../img/proj-vec.svg)
:label:`fig_vector-project`

Nếu xét ý nghĩa hình học của biểu thức này, ta thấy điều đó tương đương với việc độ dài hình chiếu của $\mathbf{v}$ lên hướng của $\mathbf{w}$ chính xác bằng $1/\|\mathbf{w}\|$, như được minh họa trong :numref:`fig_vector-project`. Tập hợp tất cả các điểm thỏa mãn điều này là một đường thẳng vuông góc với vector $\mathbf{w}$. Nếu muốn, ta có thể tìm phương trình của đường thẳng này và thấy rằng đó là $2x + y = 1$, hoặc tương đương với $y = 1 - 2x$.

Nếu bây giờ xét điều xảy ra khi ta hỏi về tập hợp các điểm thỏa mãn
$\mathbf{w}\cdot\mathbf{v} > 1$ hoặc $\mathbf{w}\cdot\mathbf{v} < 1$,
ta có thể thấy rằng đây là các trường hợp mà các hình chiếu lần lượt dài hơn hoặc ngắn hơn $1/\|\mathbf{w}\|$. Do đó, hai bất đẳng thức này xác định hai phía của đường thẳng. Theo cách này, ta đã tìm ra một cách chia không gian thành hai nửa, trong đó tất cả các điểm ở một phía có tích vô hướng thấp hơn một ngưỡng, còn các điểm ở phía kia có tích vô hướng cao hơn ngưỡng đó, như minh họa trong :numref:`fig_space-division`.

![Nếu bây giờ xét dạng bất đẳng thức của biểu thức này, ta thấy siêu phẳng của mình (trong trường hợp này chỉ là một đường thẳng) chia không gian thành hai nửa.](../img/space-division.svg)
:label:`fig_space-division`

Câu chuyện trong số chiều cao hơn cũng tương tự. Nếu bây giờ lấy $\mathbf{w} = [1,2,3]^\top$ và xét các điểm trong không gian ba chiều thỏa mãn $\mathbf{w}\cdot\mathbf{v} = 1$, ta thu được một mặt phẳng vuông góc với vector đã cho $\mathbf{w}$. Hai bất đẳng thức một lần nữa xác định hai phía của mặt phẳng, như minh họa trong :numref:`fig_higher-division`.

![Siêu phẳng trong mọi số chiều đều chia không gian thành hai nửa.](../img/space-division-3d.svg)
:label:`fig_higher-division`

Mặc dù khả năng hình dung của chúng ta dừng lại ở đây, không có gì ngăn cản ta thực hiện điều này trong không gian có hàng chục, hàng trăm hoặc hàng tỷ chiều. Điều này thường xảy ra khi suy nghĩ về các mô hình machine learning. Chẳng hạn, ta có thể hiểu các mô hình phân loại tuyến tính như những mô hình trong :numref:`sec_softmax`, là các phương pháp tìm những siêu phẳng phân tách các lớp đích khác nhau. Trong ngữ cảnh này, những siêu phẳng như vậy thường được gọi là *mặt phẳng quyết định* (decision plane). Phần lớn các mô hình phân loại deep learning kết thúc bằng một layer tuyến tính được đưa vào softmax, vì vậy ta có thể diễn giải vai trò của mạng neural sâu là tìm một phép nhúng phi tuyến sao cho các lớp đích có thể được phân tách rõ ràng bằng các siêu phẳng.

Để đưa ra một ví dụ được xây dựng thủ công, hãy lưu ý rằng ta có thể tạo ra một mô hình hợp lý để phân loại các ảnh nhỏ của áo phông và quần dài từ dataset Fashion-MNIST (được trình bày trong :numref:`sec_fashion_mnist`) chỉ bằng cách lấy vector giữa các giá trị trung bình của chúng để xác định mặt phẳng quyết định, rồi ước lượng sơ bộ một ngưỡng. Trước tiên, ta sẽ tải dữ liệu và tính các giá trị trung bình.

```{.python .input}
#@tab mxnet
# Load in the dataset
train = gluon.data.vision.FashionMNIST(train=True)
test = gluon.data.vision.FashionMNIST(train=False)

X_train_0 = np.stack([x[0] for x in train if x[1] == 0]).astype(float)
X_train_1 = np.stack([x[0] for x in train if x[1] == 1]).astype(float)
X_test = np.stack(
    [x[0] for x in test if x[1] == 0 or x[1] == 1]).astype(float)
y_test = np.stack(
    [x[1] for x in test if x[1] == 0 or x[1] == 1]).astype(float)

# Compute averages
ave_0 = np.mean(X_train_0, axis=0)
ave_1 = np.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab pytorch
# Load in the dataset
trans = []
trans.append(transforms.ToTensor())
trans = transforms.Compose(trans)
train = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                          train=True, download=True)
test = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                         train=False, download=True)

X_train_0 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 0]).type(torch.float32)
X_train_1 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 1]).type(torch.float32)
X_test = torch.stack(
    [x[0] * 256 for x in test if x[1] == 0 or x[1] == 1]).type(torch.float32)
y_test = torch.stack([torch.tensor(x[1]) for x in test
                      if x[1] == 0 or x[1] == 1]).type(torch.float32)

# Compute averages
ave_0 = torch.mean(X_train_0, axis=0)
ave_1 = torch.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab tensorflow
# Load in the dataset
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.fashion_mnist.load_data()


X_train_0 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 0]] * 256), dtype=tf.float32)
X_train_1 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 1]] * 256), dtype=tf.float32)
X_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 0]] * 256), dtype=tf.float32)
y_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 1]] * 256), dtype=tf.float32)

# Compute averages
ave_0 = tf.reduce_mean(X_train_0, axis=0)
ave_1 = tf.reduce_mean(X_train_1, axis=0)
```

Việc xem xét chi tiết các giá trị trung bình này có thể hữu ích, vì vậy hãy vẽ hình để xem chúng trông như thế nào. Trong trường hợp này, ta thấy giá trị trung bình thực sự giống một ảnh áo phông bị mờ.

```{.python .input}
#@tab mxnet, pytorch
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(ave_0.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(tf.reshape(ave_0, (28, 28)), cmap='Greys')
d2l.plt.show()
```

Trong trường hợp thứ hai, ta lại thấy giá trị trung bình giống một ảnh quần dài bị mờ.

```{.python .input}
#@tab mxnet, pytorch
# Plot average trousers
d2l.plt.imshow(ave_1.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot average trousers
d2l.plt.imshow(tf.reshape(ave_1, (28, 28)), cmap='Greys')
d2l.plt.show()
```

Trong một giải pháp được machine learning hoàn toàn, ta sẽ học ngưỡng từ dataset. Trong trường hợp này, tôi chỉ tự ước lượng một ngưỡng có vẻ hoạt động tốt trên dữ liệu huấn luyện.

```{.python .input}
#@tab mxnet
# Print test set accuracy with eyeballed threshold
w = (ave_1 - ave_0).T
predictions = X_test.reshape(2000, -1).dot(w.flatten()) > -1500000

# Accuracy
np.mean(predictions.astype(y_test.dtype) == y_test, dtype=np.float64)
```

```{.python .input}
#@tab pytorch
# Print test set accuracy with eyeballed threshold
w = (ave_1 - ave_0).T
# '@' is Matrix Multiplication operator in pytorch.
predictions = X_test.reshape(2000, -1) @ (w.flatten()) > -1500000

# Accuracy
torch.mean((predictions.type(y_test.dtype) == y_test).float(), dtype=torch.float64)
```

```{.python .input}
#@tab tensorflow
# Print test set accuracy with eyeballed threshold
w = tf.transpose(ave_1 - ave_0)
predictions = tf.reduce_sum(X_test * tf.nest.flatten(w), axis=0) > -1500000

# Accuracy
tf.reduce_mean(
    tf.cast(tf.cast(predictions, y_test.dtype) == y_test, tf.float32))
```

## Hình học của các phép biến đổi tuyến tính

Thông qua :numref:`sec_linear-algebra` và các thảo luận ở trên, ta đã hiểu khá chắc chắn về hình học của các vector, độ dài và góc. Tuy nhiên, có một đối tượng quan trọng mà ta chưa thảo luận, đó là cách hiểu hình học về các phép biến đổi tuyến tính được biểu diễn bằng các ma trận. Việc hoàn toàn thấu hiểu cách ma trận có thể biến đổi dữ liệu giữa hai không gian có số chiều cao và có thể khác nhau đòi hỏi nhiều thời gian luyện tập, nằm ngoài phạm vi của phụ lục này. Tuy nhiên, ta có thể bắt đầu xây dựng trực giác trong hai chiều.

Giả sử ta có một ma trận:

$$
\mathbf{A} = \begin{bmatrix}
a & b \\ c & d
\end{bmatrix}.
$$

Nếu muốn áp dụng ma trận này cho một vector bất kỳ
$\mathbf{v} = [x, y]^\top$,
ta nhân hai đối tượng và thấy rằng

$$
\begin{aligned}
\mathbf{A}\mathbf{v} & = \begin{bmatrix}a & b \\ c & d\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} \\
& = \begin{bmatrix}ax+by\\ cx+dy\end{bmatrix} \\
& = x\begin{bmatrix}a \\ c\end{bmatrix} + y\begin{bmatrix}b \\d\end{bmatrix} \\
& = x\left\{\mathbf{A}\begin{bmatrix}1\\0\end{bmatrix}\right\} + y\left\{\mathbf{A}\begin{bmatrix}0\\1\end{bmatrix}\right\}.
\end{aligned}
$$

Phép tính này có vẻ hơi kỳ lạ, khi một điều rõ ràng trở nên tương đối khó hiểu. Tuy nhiên, nó cho ta biết rằng có thể viết cách một ma trận biến đổi *bất kỳ* vector nào dựa trên cách nó biến đổi *hai vector cụ thể*: $[1,0]^\top$ và $[0,1]^\top$. Điều này đáng để suy ngẫm một chút. Về cơ bản, ta đã biến một bài toán vô hạn (điều gì xảy ra với mọi cặp số thực) thành một bài toán hữu hạn (điều gì xảy ra với những vector cụ thể này). Các vector này là một ví dụ về một *cơ sở* (basis), trong đó ta có thể viết mọi vector trong không gian của mình dưới dạng một tổng có trọng số của các *vector cơ sở* này.

Hãy vẽ điều xảy ra khi ta sử dụng ma trận cụ thể

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix}.
$$

Nếu xét vector cụ thể $\mathbf{v} = [2, -1]^\top$, ta thấy vector này là $2\cdot[1,0]^\top + -1\cdot[0,1]^\top$, và do đó biết rằng ma trận $A$ sẽ biến đổi nó thành
$2(\mathbf{A}[1,0]^\top) + -1(\mathbf{A}[0,1])^\top = 2[1, -1]^\top - [2,3]^\top = [0, -5]^\top$.
Nếu cẩn thận theo dõi lập luận này, chẳng hạn bằng cách xét lưới gồm mọi cặp điểm nguyên, ta thấy phép nhân ma trận có thể làm lệch, xoay và co giãn lưới, nhưng cấu trúc lưới vẫn phải được giữ nguyên như trong :numref:`fig_grid-transform`.

![Ma trận $\mathbf{A}$ tác động lên các vector cơ sở đã cho. Hãy chú ý rằng toàn bộ lưới cũng được biến đổi theo.](../img/grid-transform.svg)
:label:`fig_grid-transform`

Đây là điểm trực giác quan trọng nhất cần ghi nhớ về các phép biến đổi tuyến tính được biểu diễn bằng ma trận. Ma trận không thể làm biến dạng một số phần của không gian khác với những phần khác. Tất cả những gì chúng có thể làm là lấy các tọa độ ban đầu trên không gian của ta rồi làm lệch, xoay và co giãn chúng.

Một số phép biến dạng có thể rất nghiêm trọng. Chẳng hạn, ma trận

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix},
$$

nén toàn bộ mặt phẳng hai chiều xuống thành một đường thẳng duy nhất. Việc xác định và làm việc với những phép biến đổi như vậy là chủ đề của một phần sau, nhưng về mặt hình học, ta có thể thấy rằng điều này về cơ bản khác với các loại phép biến đổi đã xét ở trên. Chẳng hạn, kết quả từ ma trận $\mathbf{A}$ có thể được “bẻ ngược” về lưới ban đầu. Kết quả từ ma trận $\mathbf{B}$ thì không thể, vì ta sẽ không bao giờ biết vector $[1,2]^\top$ xuất phát từ đâu—nó là $[1,1]^\top$ hay $[0, -1]^\top$?

Mặc dù hình minh họa này dành cho ma trận $2\times2$, không có gì ngăn cản ta áp dụng những bài học đã rút ra vào các số chiều cao hơn. Nếu lấy các vector cơ sở tương tự như $[1,0, \ldots,0]$ và xem ma trận biến đổi chúng như thế nào, ta có thể bắt đầu hình thành cảm nhận về cách phép nhân ma trận làm biến dạng toàn bộ không gian trong bất kỳ số chiều nào mà ta đang xét.

## Phụ thuộc tuyến tính

Một lần nữa, hãy xét ma trận

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix}.
$$

Điều này nén toàn bộ mặt phẳng xuống chỉ còn nằm trên đường thẳng $y = 2x$.
Bây giờ đặt ra câu hỏi: liệu có cách nào để phát hiện điều này
chỉ bằng cách nhìn vào chính ma trận hay không?
Câu trả lời là có.
Hãy lấy $\mathbf{b}_1 = [2,4]^\top$ và $\mathbf{b}_2 = [-1, -2]^\top$
làm hai cột của $\mathbf{B}$.
Hãy nhớ rằng ta có thể viết mọi thứ được biến đổi bởi ma trận $\mathbf{B}$
dưới dạng tổng có trọng số của các cột của ma trận:
chẳng hạn như $a_1\mathbf{b}_1 + a_2\mathbf{b}_2$.
Ta gọi đó là một *tổ hợp tuyến tính*.
Việc $\mathbf{b}_1 = -2\cdot\mathbf{b}_2$
có nghĩa là ta có thể viết mọi tổ hợp tuyến tính của hai cột đó
hoàn toàn theo, chẳng hạn, $\mathbf{b}_2$ vì

$$
a_1\mathbf{b}_1 + a_2\mathbf{b}_2 = -2a_1\mathbf{b}_2 + a_2\mathbf{b}_2 = (a_2-2a_1)\mathbf{b}_2.
$$

Điều này có nghĩa là một trong các cột, theo một nghĩa nào đó, là dư thừa
vì nó không xác định một hướng duy nhất trong không gian.
Điều này không nên khiến ta quá ngạc nhiên
vì ta đã thấy ma trận này
nén toàn bộ mặt phẳng xuống một đường thẳng.
Hơn nữa, ta thấy rằng sự phụ thuộc tuyến tính
$\mathbf{b}_1 = -2\cdot\mathbf{b}_2$ thể hiện điều này.
Để làm cho biểu thức đối xứng hơn giữa hai vector, ta sẽ viết nó như sau:

$$
\mathbf{b}_1  + 2\cdot\mathbf{b}_2 = 0.
$$

Nói chung, ta sẽ nói một tập hợp các vector
$\mathbf{v}_1, \ldots, \mathbf{v}_k$ là *phụ thuộc tuyến tính*
nếu tồn tại các hệ số $a_1, \ldots, a_k$ *không đồng thời bằng không* sao cho

$$
\sum_{i=1}^k a_i\mathbf{v_i} = 0.
$$

Trong trường hợp này, ta có thể biểu diễn một trong các vector
theo một tổ hợp nào đó của các vector còn lại,
và biến nó thành phần dư thừa.
Do đó, sự phụ thuộc tuyến tính trong các cột của một ma trận
là bằng chứng cho thấy ma trận của ta
đang nén không gian xuống một số chiều thấp hơn.
Nếu không có sự phụ thuộc tuyến tính, ta nói các vector là *độc lập tuyến tính*.
Nếu các cột của một ma trận độc lập tuyến tính,
thì không xảy ra sự nén và phép biến đổi có thể được đảo ngược.

## Hạng

Nếu ta có một ma trận tổng quát $n\times m$,
thì việc hỏi ma trận ánh xạ vào không gian có số chiều bao nhiêu là hợp lý.
Một khái niệm được gọi là *hạng* sẽ là câu trả lời cho chúng ta.
Trong phần trước, ta đã nhận thấy rằng sự phụ thuộc tuyến tính
là bằng chứng cho việc nén không gian xuống một số chiều thấp hơn
và vì vậy ta có thể dùng điều này để định nghĩa khái niệm hạng.
Cụ thể, hạng của một ma trận $\mathbf{A}$
là số lượng lớn nhất các cột độc lập tuyến tính
trong tất cả các tập con của các cột. Ví dụ, ma trận

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix},
$$

có $\textrm{rank}(B)=1$, vì hai cột phụ thuộc tuyến tính,
nhưng mỗi cột riêng lẻ không phụ thuộc tuyến tính.
Để xem một ví dụ khó hơn, ta có thể xét

$$
\mathbf{C} = \begin{bmatrix}
1& 3 & 0 & -1 & 0 \\
-1 & 0 & 1 & 1 & -1 \\
0 & 3 & 1 & 0 & -1 \\
2 & 3 & -1 & -2 & 1
\end{bmatrix},
$$

và chứng minh rằng $\mathbf{C}$ có hạng bằng hai vì, chẳng hạn,
hai cột đầu tiên độc lập tuyến tính,
tuy nhiên mỗi trong bốn tập hợp gồm ba cột đều phụ thuộc tuyến tính.

Quy trình này, như đã mô tả, rất kém hiệu quả.
Nó yêu cầu xem xét mọi tập con của các cột trong ma trận đã cho,
và do đó có thể tăng theo cấp số mũ theo số lượng cột.
Về sau ta sẽ thấy một cách hiệu quả hơn về mặt tính toán
để tính hạng của một ma trận, nhưng hiện tại,
điều này đủ để thấy rằng khái niệm này
được định nghĩa rõ ràng và hiểu được ý nghĩa của nó.

## Tính khả nghịch

Ở trên, ta đã thấy rằng phép nhân với một ma trận có các cột phụ thuộc tuyến tính
không thể được đảo ngược, tức là không có phép toán nghịch đảo nào luôn có thể khôi phục đầu vào. Tuy nhiên, phép nhân với một ma trận full-rank
(tức là một $\mathbf{A}$ nào đó là ma trận $n \times n$ có hạng $n$),
thì luôn có thể được đảo ngược. Hãy xét ma trận

$$
\mathbf{I} = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}.
$$

đây là ma trận có các số một trên đường chéo và các số không ở những vị trí còn lại.
Ta gọi nó là ma trận *identity*.
Đây là ma trận giữ nguyên dữ liệu của ta khi được áp dụng.
Để tìm một ma trận đảo ngược tác động mà ma trận $\mathbf{A}$ đã thực hiện,
ta muốn tìm một ma trận $\mathbf{A}^{-1}$ sao cho

$$
\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} =  \mathbf{I}.
$$

Nếu xem đây là một hệ phương trình, ta có $n \times n$ ẩn số
(các phần tử của $\mathbf{A}^{-1}$) và $n \times n$ phương trình
(đẳng thức cần đúng giữa mọi phần tử của tích $\mathbf{A}^{-1}\mathbf{A}$ và mọi phần tử của $\mathbf{I}$)
nên nhìn chung ta có thể kỳ vọng một nghiệm tồn tại.
Quả thực, trong phần tiếp theo ta sẽ thấy một đại lượng được gọi là *determinant*,
có tính chất là chừng nào determinant khác không, ta có thể tìm được một nghiệm. Ta gọi một ma trận $\mathbf{A}^{-1}$ như vậy là ma trận *nghịch đảo*. Ví dụ, nếu $\mathbf{A}$ là ma trận $2 \times 2$ tổng quát

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

thì ta có thể thấy ma trận nghịch đảo là

$$
 \frac{1}{ad-bc}  \begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}.
$$

Ta có thể kiểm chứng điều này bằng cách thấy rằng phép nhân
với ma trận nghịch đảo được cho bởi công thức trên thực sự hoạt động.

```{.python .input}
#@tab mxnet
M = np.array([[1, 2], [1, 4]])
M_inv = np.array([[2, -1], [-0.5, 0.5]])
M_inv.dot(M)
```

```{.python .input}
#@tab pytorch
M = torch.tensor([[1, 2], [1, 4]], dtype=torch.float32)
M_inv = torch.tensor([[2, -1], [-0.5, 0.5]])
M_inv @ M
```

```{.python .input}
#@tab tensorflow
M = tf.constant([[1, 2], [1, 4]], dtype=tf.float32)
M_inv = tf.constant([[2, -1], [-0.5, 0.5]])
tf.matmul(M_inv, M)
```

### Các vấn đề số học
Mặc dù ma trận nghịch đảo hữu ích về mặt lý thuyết,
ta phải nói rằng phần lớn thời gian chúng ta không muốn
*sử dụng* ma trận nghịch đảo để giải một bài toán trong thực tế.
Nói chung, có những thuật toán ổn định hơn nhiều về mặt số học
để giải các phương trình tuyến tính như

$$
\mathbf{A}\mathbf{x} = \mathbf{b},
$$

so với việc tính ma trận nghịch đảo rồi nhân để thu được

$$
\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}.
$$

Cũng như phép chia cho một số nhỏ có thể dẫn đến sự bất ổn định về mặt số học,
việc lấy nghịch đảo của một ma trận gần có hạng thấp cũng vậy.

Hơn nữa, thường thì ma trận $\mathbf{A}$ là *thưa*,
nghĩa là nó chỉ chứa một số lượng nhỏ các giá trị khác không.
Nếu xem xét các ví dụ, ta sẽ thấy
điều này không có nghĩa là ma trận nghịch đảo cũng thưa.
Ngay cả khi $\mathbf{A}$ là một ma trận $1$ triệu x $1$ triệu
chỉ có $5$ triệu phần tử khác không
(và vì vậy ta chỉ cần lưu trữ $5$ triệu phần tử đó),
ma trận nghịch đảo thường sẽ có hầu như mọi phần tử không âm,
buộc ta phải lưu trữ toàn bộ $1\textrm{M}^2$ phần tử---tức là $1$ nghìn tỷ phần tử!

Mặc dù không có thời gian để đi sâu hoàn toàn vào những vấn đề số học phức tạp
thường gặp khi làm việc với đại số tuyến tính,
chúng tôi muốn cung cấp cho bạn một số trực giác về những lúc cần thận trọng,
và nhìn chung, tránh lấy nghịch đảo trong thực tế là một nguyên tắc kinh nghiệm tốt.

## Định thức
Góc nhìn hình học về đại số tuyến tính cung cấp một cách trực quan
để diễn giải một đại lượng cơ bản được gọi là *determinant*.
Hãy xét hình ảnh lưới từ trước, nhưng lần này có một vùng được tô nổi bật (:numref:`fig_grid-filled`).

![Ma trận $\mathbf{A}$ lại làm biến dạng lưới. Lần này, tôi muốn đặc biệt chú ý đến điều xảy ra với hình vuông được tô nổi bật.](../img/grid-transform-filled.svg)
:label:`fig_grid-filled`

Hãy nhìn vào hình vuông được tô nổi bật. Đây là một hình vuông có các cạnh được xác định
bởi $(0, 1)$ và $(1, 0)$ nên nó có diện tích bằng một.
Sau khi $\mathbf{A}$ biến đổi hình vuông này,
ta thấy nó trở thành một hình bình hành.
Không có lý do gì để hình bình hành này phải có cùng diện tích
như hình ban đầu, và quả thực trong trường hợp cụ thể được minh họa ở đây là

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix},
$$

ta có thể dùng hình học tọa độ để tính diện tích của hình bình hành này
và thu được diện tích bằng $5$.

Nói chung, nếu ta có một ma trận

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

ta có thể thấy bằng một số phép tính rằng diện tích
của hình bình hành thu được là $ad-bc$.
Diện tích này được gọi là *determinant*.

Hãy nhanh chóng kiểm tra điều này bằng một đoạn mã ví dụ.

```{.python .input}
#@tab mxnet
import numpy as np
np.linalg.det(np.array([[1, -1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
torch.det(torch.tensor([[1, -1], [2, 3]], dtype=torch.float32))
```

```{.python .input}
#@tab tensorflow
tf.linalg.det(tf.constant([[1, -1], [2, 3]], dtype=tf.float32))
```

Những người tinh ý sẽ nhận thấy
rằng biểu thức này có thể bằng không hoặc thậm chí âm.
Đối với số hạng âm, đây là một quy ước
được sử dụng phổ biến trong toán học:
nếu ma trận lật hình,
ta nói diện tích bị đổi dấu.
Bây giờ hãy xem khi determinant bằng không thì ta có thể biết thêm điều gì.

Hãy xét

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix}.
$$

Nếu tính định thức của ma trận này,
ta được $2\cdot(-2 ) - 4\cdot(-1) = 0$.
Dựa trên những gì đã hiểu ở trên, điều này là hợp lý.
$\mathbf{B}$ nén hình vuông từ hình ảnh ban đầu
thành một đoạn thẳng, có diện tích bằng không.
Và quả thực, bị nén vào một không gian có số chiều thấp hơn
là cách duy nhất để có diện tích bằng không sau phép biến đổi.
Do đó, ta thấy kết quả sau là đúng:
ma trận $A$ khả nghịch khi và chỉ khi
định thức khác không.

Để kết luận, hãy tưởng tượng ta có một hình bất kỳ được vẽ trên mặt phẳng.
Suy nghĩ như các nhà khoa học máy tính, ta có thể phân rã
hình đó thành một tập hợp các hình vuông nhỏ
sao cho diện tích của hình về cơ bản
chính là số lượng hình vuông trong phép phân rã.
Nếu giờ ta biến đổi hình đó bằng một ma trận,
ta biến đổi mỗi hình vuông này thành một hình bình hành,
mỗi hình có diện tích được xác định bởi định thức.
Ta thấy rằng với mọi hình, định thức cho biết con số (có dấu)
mà một ma trận dùng để co giãn diện tích của hình đó.

Việc tính định thức cho các ma trận lớn hơn có thể tốn nhiều công sức,
nhưng trực giác vẫn giống nhau.
Định thức vẫn là hệ số
mà các ma trận $n\times n$ dùng để co giãn các thể tích $n$ chiều.

## Tensor và các phép toán Đại số tuyến tính Thông dụng

Trong :numref:`sec_linear-algebra`, khái niệm tensor đã được giới thiệu.
Trong phần này, chúng ta sẽ tìm hiểu sâu hơn về các phép co tensor
(tương đương tensor của phép nhân ma trận),
và xem cách nó cung cấp một góc nhìn thống nhất
cho một số phép toán ma trận và vector.

Với ma trận và vector, chúng ta biết cách nhân chúng để biến đổi dữ liệu.
Chúng ta cần một định nghĩa tương tự cho tensor nếu muốn chúng hữu ích.
Hãy nghĩ về phép nhân ma trận:

$$
\mathbf{C} = \mathbf{A}\mathbf{B},
$$

hay tương đương

$$ c_{i, j} = \sum_{k} a_{i, k}b_{k, j}.$$

Mẫu này có thể được lặp lại cho tensor.
Đối với tensor, không có một trường hợp duy nhất về việc
cần lấy tổng theo chỉ số nào có thể được chọn một cách phổ quát,
vì vậy chúng ta cần chỉ rõ chính xác những chỉ số nào muốn lấy tổng.
Chẳng hạn, ta có thể xét

$$
y_{il} = \sum_{jk} x_{ijkl}a_{jk}.
$$

Một phép biến đổi như vậy được gọi là *phép co tensor*.
Nó có thể biểu diễn một họ phép biến đổi linh hoạt hơn nhiều
so với chỉ phép nhân ma trận.

Để đơn giản hóa ký hiệu thường được sử dụng,
ta có thể nhận thấy rằng tổng được lấy theo đúng những chỉ số
xuất hiện nhiều hơn một lần trong biểu thức,
do đó người ta thường dùng *ký hiệu Einstein*,
trong đó phép lấy tổng được ngầm hiểu là thực hiện trên mọi chỉ số lặp lại.
Điều này cho ta biểu thức ngắn gọn:

$$
y_{il} = x_{ijkl}a_{jk}.
$$

### Các ví dụ Thông dụng từ Đại số tuyến tính

Hãy xem có bao nhiêu định nghĩa đại số tuyến tính
mà chúng ta đã gặp trước đây có thể được biểu diễn bằng ký hiệu tensor rút gọn này:

* $\mathbf{v} \cdot \mathbf{w} = \sum_i v_iw_i$
* $\|\mathbf{v}\|_2^{2} = \sum_i v_iv_i$
* $(\mathbf{A}\mathbf{v})_i = \sum_j a_{ij}v_j$
* $(\mathbf{A}\mathbf{B})_{ik} = \sum_j a_{ij}b_{jk}$
* $\textrm{tr}(\mathbf{A}) = \sum_i a_{ii}$

Bằng cách này, ta có thể thay thế vô số ký hiệu chuyên biệt
bằng các biểu thức tensor ngắn gọn.

### Biểu diễn trong Code
Tensor cũng có thể được thao tác một cách linh hoạt trong code.
Như đã thấy trong :numref:`sec_linear-algebra`,
ta có thể tạo tensor như dưới đây.

```{.python .input}
#@tab mxnet
# Define tensors
B = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab pytorch
# Define tensors
B = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = torch.tensor([[1, 2], [3, 4]])
v = torch.tensor([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab tensorflow
# Define tensors
B = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = tf.constant([[1, 2], [3, 4]])
v = tf.constant([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

Phép lấy tổng Einstein đã được triển khai trực tiếp.
Các chỉ số xuất hiện trong phép lấy tổng Einstein có thể được truyền dưới dạng một chuỗi,
theo sau là các tensor được tác động.
Chẳng hạn, để triển khai phép nhân ma trận,
ta có thể xét phép lấy tổng Einstein ở trên
($\mathbf{A}\mathbf{v} = a_{ij}v_j$)
và loại bỏ chính các chỉ số để có được phần triển khai:

```{.python .input}
#@tab mxnet
# Reimplement matrix multiplication
np.einsum("ij, j -> i", A, v), A.dot(v)
```

```{.python .input}
#@tab pytorch
# Reimplement matrix multiplication
torch.einsum("ij, j -> i", A, v), A@v
```

```{.python .input}
#@tab tensorflow
# Reimplement matrix multiplication
tf.einsum("ij, j -> i", A, v), tf.matmul(A, tf.reshape(v, (2, 1)))
```

Đây là một ký hiệu rất linh hoạt.
Chẳng hạn, nếu muốn tính
biểu thức theo cách viết truyền thống là

$$
c_{kl} = \sum_{ij} \mathbf{b}_{ijk}\mathbf{a}_{il}v_j.
$$

ta có thể triển khai bằng phép lấy tổng Einstein như sau:

```{.python .input}
#@tab mxnet
np.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab pytorch
torch.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab tensorflow
tf.einsum("ijk, il, j -> kl", B, A, v)
```

Ký hiệu này dễ đọc và hiệu quả đối với con người,
tuy nhiên sẽ cồng kềnh nếu vì bất kỳ lý do nào
ta cần tạo một chương trình co tensor theo cách lập trình.
Vì lý do này, `einsum` cung cấp một ký hiệu thay thế
bằng cách cung cấp các chỉ số nguyên cho mỗi tensor.
Ví dụ, cùng phép co tensor đó cũng có thể được viết như sau:

```{.python .input}
#@tab mxnet
np.einsum(B, [0, 1, 2], A, [0, 3], v, [1], [2, 3])
```

```{.python .input}
#@tab pytorch
# PyTorch does not support this type of notation.
```

```{.python .input}
#@tab tensorflow
# TensorFlow does not support this type of notation.
```

Cả hai ký hiệu đều cho phép biểu diễn các phép co tensor
ngắn gọn và hiệu quả trong code.

## Tóm tắt
* Vector có thể được diễn giải về mặt hình học như các điểm hoặc các hướng trong không gian.
* Tích vô hướng xác định khái niệm góc trong các không gian có số chiều tùy ý, kể cả số chiều rất cao.
* Siêu phẳng là sự tổng quát hóa các đường thẳng và mặt phẳng lên không gian nhiều chiều. Chúng có thể được dùng để xác định các mặt phẳng quyết định, thường được sử dụng ở bước cuối của một tác vụ phân loại.
* Phép nhân ma trận có thể được diễn giải về mặt hình học như những biến dạng đồng nhất của các tọa độ cơ sở. Chúng biểu diễn một cách rất hạn chế nhưng rõ ràng về mặt toán học để biến đổi vector.
* Phụ thuộc tuyến tính là cách xác định khi nào một tập hợp vector nằm trong một không gian có số chiều thấp hơn dự kiến (chẳng hạn, bạn có $3$ vector nằm trong một không gian $2$ chiều). Hạng của một ma trận là kích thước của tập con lớn nhất gồm các cột độc lập tuyến tính.
* Khi nghịch đảo của một ma trận được xác định, phép nghịch đảo ma trận cho phép ta tìm một ma trận khác để đảo ngược tác động của ma trận đầu tiên. Phép nghịch đảo ma trận hữu ích về mặt lý thuyết, nhưng trong thực tế cần thận trọng do tính không ổn định về số.
* Định thức cho phép ta đo mức độ một ma trận mở rộng hoặc co lại một không gian. Định thức khác không cho thấy ma trận khả nghịch (không suy biến), còn định thức bằng không nghĩa là ma trận không khả nghịch (suy biến).
* Phép co tensor và phép lấy tổng Einstein cung cấp một ký hiệu gọn gàng, rõ ràng để biểu diễn nhiều phép tính thường gặp trong machine learning.

## Bài tập
1. Góc giữa
$$
\vec v_1 = \begin{bmatrix}
1 \\ 0 \\ -1 \\ 2
\end{bmatrix}, \qquad \vec v_2 = \begin{bmatrix}
3 \\ 1 \\ 0 \\ 1
\end{bmatrix}?
$$
là bao nhiêu?
2. Đúng hay sai: $\begin{bmatrix}1 & 2\\0&1\end{bmatrix}$ và $\begin{bmatrix}1 & -2\\0&1\end{bmatrix}$ là nghịch đảo của nhau?
3. Giả sử ta vẽ một hình trên mặt phẳng có diện tích $100\textrm{m}^2$. Diện tích sau khi biến đổi hình bằng ma trận
$$
\begin{bmatrix}
2 & 3\\
1 & 2
\end{bmatrix}.
$$
là bao nhiêu?
4. Những tập hợp vector nào sau đây độc lập tuyến tính?
 * $\left\{\begin{pmatrix}1\\0\\-1\end{pmatrix}, \begin{pmatrix}2\\1\\-1\end{pmatrix}, \begin{pmatrix}3\\1\\1\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}3\\1\\1\end{pmatrix}, \begin{pmatrix}1\\1\\1\end{pmatrix}, \begin{pmatrix}0\\0\\0\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}1\\1\\0\end{pmatrix}, \begin{pmatrix}0\\1\\-1\end{pmatrix}, \begin{pmatrix}1\\0\\1\end{pmatrix}\right\}$
5. Giả sử bạn có một ma trận được viết dưới dạng $A = \begin{bmatrix}c\\d\end{bmatrix}\cdot\begin{bmatrix}a & b\end{bmatrix}$ với một số giá trị $a, b, c$ và $d$. Đúng hay sai: định thức của một ma trận như vậy luôn bằng $0$?
6. Các vector $e_1 = \begin{bmatrix}1\\0\end{bmatrix}$ và $e_2 = \begin{bmatrix}0\\1\end{bmatrix}$ trực giao. Điều kiện nào đối với ma trận $A$ để $Ae_1$ và $Ae_2$ trực giao?
7. Làm thế nào để viết $\textrm{tr}(\mathbf{A}^4)$ bằng ký hiệu Einstein cho một ma trận $A$ bất kỳ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/410)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1084)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1085)
:end_tab: