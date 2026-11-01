# Phân rã riêng
:label:`sec_eigendecompositions`

Trị riêng thường là một trong những khái niệm hữu ích nhất
mà chúng ta sẽ gặp khi học đại số tuyến tính,
tuy nhiên, với người mới bắt đầu, ta rất dễ bỏ qua tầm quan trọng của chúng.
Dưới đây, chúng ta giới thiệu phép phân rã riêng và
cố gắng giúp bạn hiểu được phần nào tại sao nó lại quan trọng đến vậy.

Giả sử ta có một ma trận $A$ với các phần tử như sau:

$$
\mathbf{A} = \begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix}.
$$

Nếu áp dụng $A$ lên bất kỳ vector $\mathbf{v} = [x, y]^\top$ nào,
ta thu được vector $\mathbf{A}\mathbf{v} = [2x, -y]^\top$.
Điều này có một cách diễn giải trực quan:
kéo giãn vector để nó rộng gấp đôi theo hướng $x$,
sau đó lật nó theo hướng $y$.

Tuy nhiên, có *một số* vector mà một điều gì đó vẫn không thay đổi.
Cụ thể, $[1, 0]^\top$ được biến thành $[2, 0]^\top$
và $[0, 1]^\top$ được biến thành $[0, -1]^\top$.
Các vector này vẫn nằm trên cùng một đường thẳng,
và thay đổi duy nhất là ma trận kéo giãn chúng
lần lượt theo hệ số $2$ và $-1$.
Ta gọi những vector như vậy là *vector riêng*
và hệ số mà chúng được kéo giãn là *trị riêng*.

Nói chung, nếu ta có thể tìm được một số $\lambda$
và một vector $\mathbf{v}$ sao cho

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$

Ta nói rằng $\mathbf{v}$ là một vector riêng của $A$ và $\lambda$ là một trị riêng.

## Tìm trị riêng
Hãy tìm hiểu cách tìm chúng. Bằng cách chuyển $\lambda \mathbf{v}$ từ cả hai vế sang vế kia,
sau đó phân tích vector ra ngoài,
ta thấy biểu thức trên tương đương với:

$$(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0.$$
:eqlabel:`eq_eigvalue_der`

Để :eqref:`eq_eigvalue_der` xảy ra, ta thấy rằng $(\mathbf{A} - \lambda \mathbf{I})$
phải nén một hướng nào đó về không,
do đó nó không khả nghịch, và vì vậy định thức của nó bằng không.
Vì thế, ta có thể tìm các *trị riêng*
bằng cách tìm $\lambda$ sao cho $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$.
Sau khi tìm được các trị riêng, ta có thể giải
$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$
để tìm *vector riêng* tương ứng.

### Một ví dụ
Hãy xét điều này với một ma trận phức tạp hơn:

$$
\mathbf{A} = \begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}.
$$

Nếu xét $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$,
ta thấy điều này tương đương với phương trình đa thức
$0 = (2-\lambda)(3-\lambda)-2 = (4-\lambda)(1-\lambda)$.
Do đó, hai trị riêng là $4$ và $1$.
Để tìm các vector tương ứng, ta cần giải

$$
\begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}x \\ y\end{bmatrix}  \; \textrm{and} \;
\begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix}  = \begin{bmatrix}4x \\ 4y\end{bmatrix} .
$$

Ta có thể giải bài toán này bằng các vector $[1, -1]^\top$ và $[1, 2]^\top$ tương ứng.

Ta có thể kiểm tra điều này trong code bằng routine `numpy.linalg.eig` tích hợp sẵn.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
import numpy as np

np.linalg.eig(np.array([[2, 1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch

torch.linalg.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

tf.linalg.eig(tf.constant([[2, 1], [2, 3]], dtype=tf.float64))
```

Lưu ý rằng `numpy` chuẩn hóa các vector riêng để có độ dài bằng một,
trong khi ta đã chọn các vector của mình có độ dài tùy ý.
Ngoài ra, việc chọn dấu là tùy ý.
Tuy nhiên, các vector được tính toán song song
với những vector ta tìm được bằng tay với cùng các trị riêng.

## Phân rã ma trận
Hãy tiếp tục ví dụ trước thêm một bước. Đặt

$$
\mathbf{W} = \begin{bmatrix}
1 & 1 \\
-1 & 2
\end{bmatrix},
$$

là ma trận mà các cột của nó là các vector riêng của ma trận $\mathbf{A}$. Đặt

$$
\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix},
$$

là ma trận có các trị riêng tương ứng trên đường chéo.
Khi đó, định nghĩa về trị riêng và vector riêng cho ta biết rằng

$$
\mathbf{A}\mathbf{W} =\mathbf{W} \boldsymbol{\Sigma} .
$$

Ma trận $W$ khả nghịch, vì vậy ta có thể nhân cả hai vế với $W^{-1}$ ở bên phải,
từ đó thấy rằng ta có thể viết

$$\mathbf{A} = \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^{-1}.$$
:eqlabel:`eq_eig_decomp`

Trong phần tiếp theo, ta sẽ thấy một số hệ quả thú vị của điều này,
nhưng hiện tại, ta chỉ cần biết rằng phép phân rã như vậy
sẽ tồn tại miễn là ta có thể tìm được một tập đầy đủ
các vector riêng độc lập tuyến tính (để $W$ khả nghịch).

## Các phép toán trên phép phân rã riêng
Một ưu điểm của phép phân rã riêng :eqref:`eq_eig_decomp` là
ta có thể biểu diễn rõ ràng nhiều phép toán thường gặp
theo phép phân rã riêng. Trước hết, hãy xét:

$$
\mathbf{A}^n = \overbrace{\mathbf{A}\cdots \mathbf{A}}^{\textrm{$n$ times}} = \overbrace{(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})\cdots(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})}^{\textrm{$n$ times}} =  \mathbf{W}\overbrace{\boldsymbol{\Sigma}\cdots\boldsymbol{\Sigma}}^{\textrm{$n$ times}}\mathbf{W}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^n \mathbf{W}^{-1}.
$$

Điều này cho ta biết rằng với mọi lũy thừa dương của một ma trận,
phép phân rã riêng được thu được chỉ bằng cách nâng các trị riêng lên cùng số mũ.
Điều tương tự cũng có thể chứng minh cho các lũy thừa âm,
vì vậy nếu muốn lấy nghịch đảo của một ma trận, ta chỉ cần xét

$$
\mathbf{A}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^{-1} \mathbf{W}^{-1},
$$

hay nói cách khác, chỉ cần lấy nghịch đảo của từng trị riêng.
Điều này đúng miễn là mỗi trị riêng đều khác không,
vì vậy ta thấy khả nghịch tương đương với việc không có trị riêng bằng không.

Thực tế, với các phép biến đổi bổ sung, ta có thể chứng minh rằng nếu $\lambda_1, \ldots, \lambda_n$
là các trị riêng của một ma trận, thì định thức của ma trận đó là

$$
\det(\mathbf{A}) = \lambda_1 \cdots \lambda_n,
$$

hay tích của tất cả các trị riêng.
Điều này có ý nghĩa trực quan vì bất kể $\mathbf{W}$ thực hiện phép kéo giãn nào,
$W^{-1}$ sẽ hoàn tác phép kéo giãn đó, nên cuối cùng phép kéo giãn duy nhất xảy ra là
phép nhân với ma trận đường chéo $\boldsymbol{\Sigma}$,
ma trận này kéo giãn thể tích theo tích của các phần tử trên đường chéo.

Cuối cùng, hãy nhớ rằng hạng là số lượng lớn nhất
các cột độc lập tuyến tính của ma trận.
Bằng cách xem xét kỹ phép phân rã riêng,
ta có thể thấy rằng hạng bằng
số lượng trị riêng khác không của $\mathbf{A}$.

Có thể tiếp tục đưa ra các ví dụ, nhưng hy vọng điểm chính đã rõ:
phép phân rã riêng có thể đơn giản hóa nhiều phép tính đại số tuyến tính
và là một phép toán nền tảng đứng sau nhiều thuật toán số
cũng như phần lớn các phép phân tích mà ta thực hiện trong đại số tuyến tính.

## Phân rã riêng của ma trận đối xứng
Không phải lúc nào cũng có thể tìm đủ các vector riêng độc lập tuyến tính
để quy trình trên hoạt động. Chẳng hạn, ma trận

$$
\mathbf{A} = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},
$$

chỉ có một vector riêng duy nhất, cụ thể là $(1, 0)^\top$.
Để xử lý các ma trận như vậy, ta cần những kỹ thuật nâng cao hơn
những gì có thể trình bày ở đây (chẳng hạn như Jordan Normal Form hoặc Singular Value Decomposition).
Ta thường cần giới hạn sự chú ý vào những ma trận
mà ta có thể đảm bảo sự tồn tại của một tập đầy đủ các vector riêng.

Họ ma trận thường gặp nhất là *ma trận đối xứng*,
tức là những ma trận thỏa mãn $\mathbf{A} = \mathbf{A}^\top$.
Trong trường hợp này, ta có thể chọn $W$ là một *ma trận trực giao*—một ma trận mà các cột của nó đều là các vector độ dài một và vuông góc với nhau, trong đó
$\mathbf{W}^\top = \mathbf{W}^{-1}$—và tất cả các trị riêng đều là số thực.
Do đó, trong trường hợp đặc biệt này, ta có thể viết :eqref:`eq_eig_decomp` dưới dạng

$$
\mathbf{A} = \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top .
$$

## Định lý vòng tròn Gershgorin
Các trị riêng thường khó diễn giải một cách trực quan.
Nếu được cho một ma trận bất kỳ, ta hầu như không thể nói gì
về các trị riêng nếu chưa tính toán chúng.
Tuy nhiên, có một định lý giúp ta dễ dàng xấp xỉ tốt
nếu các giá trị lớn nhất nằm trên đường chéo.

Cho $\mathbf{A} = (a_{ij})$ là một ma trận vuông bất kỳ ($n\times n$).
Ta định nghĩa $r_i = \sum_{j \neq i} |a_{ij}|$.
Gọi $\mathcal{D}_i$ là đĩa trong mặt phẳng phức
có tâm tại $a_{ii}$ và bán kính $r_i$.
Khi đó, mọi trị riêng của $\mathbf{A}$ đều nằm trong một trong các $\mathcal{D}_i$.

Điều này có thể hơi khó hiểu, vì vậy hãy xem một ví dụ.
Xét ma trận:

$$
\mathbf{A} = \begin{bmatrix}
1.0 & 0.1 & 0.1 & 0.1 \\
0.1 & 3.0 & 0.2 & 0.3 \\
0.1 & 0.2 & 5.0 & 0.5 \\
0.1 & 0.3 & 0.5 & 9.0
\end{bmatrix}.
$$

Ta có $r_1 = 0.3$, $r_2 = 0.6$, $r_3 = 0.8$ và $r_4 = 0.9$.
Ma trận đối xứng, vì vậy mọi trị riêng đều là số thực.
Điều này có nghĩa là tất cả các trị riêng của chúng ta sẽ nằm trong một trong các khoảng

$$[a_{11}-r_1, a_{11}+r_1] = [0.7, 1.3], $$

$$[a_{22}-r_2, a_{22}+r_2] = [2.4, 3.6], $$

$$[a_{33}-r_3, a_{33}+r_3] = [4.2, 5.8], $$

$$[a_{44}-r_4, a_{44}+r_4] = [8.1, 9.9]. $$


Tính toán số cho thấy
các trị riêng xấp xỉ bằng $0.99$, $2.97$, $4.95$, $9.08$,
tất cả đều nằm khá thoải mái trong các khoảng đã cho.

```{.python .input}
#@tab mxnet
A = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = np.linalg.eig(A)
v
```

```{.python .input}
#@tab pytorch
A = torch.tensor([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = torch.linalg.eig(A)
v
```

```{.python .input}
#@tab tensorflow
A = tf.constant([[1.0, 0.1, 0.1, 0.1],
                [0.1, 3.0, 0.2, 0.3],
                [0.1, 0.2, 5.0, 0.5],
                [0.1, 0.3, 0.5, 9.0]])

v, _ = tf.linalg.eigh(A)
v
```

Bằng cách này, ta có thể xấp xỉ các trị riêng,
và các giá trị xấp xỉ sẽ khá chính xác
trong trường hợp các phần tử trên đường chéo
lớn hơn đáng kể so với tất cả các phần tử khác.

Đây là một điều nhỏ, nhưng với một chủ đề
phức tạp và tinh tế như phân rã riêng,
việc có được bất kỳ trực giác nào cũng đều hữu ích.

## Một ứng dụng hữu ích: Sự tăng trưởng của các ánh xạ lặp

Giờ đây, khi đã hiểu về nguyên tắc các vector riêng là gì,
hãy xem chúng có thể được sử dụng như thế nào để giúp hiểu sâu
một vấn đề cốt lõi trong hoạt động của mạng neural: khởi tạo trọng số đúng cách.

### Vector riêng biểu diễn hành vi dài hạn

Việc khảo sát toán học đầy đủ về cách khởi tạo
các mạng neural sâu nằm ngoài phạm vi của tài liệu này,
nhưng ở đây ta có thể xem một phiên bản đơn giản hóa để hiểu
các trị riêng có thể giúp ta thấy cách những mô hình này hoạt động như thế nào.
Như đã biết, mạng neural hoạt động bằng cách xen kẽ các lớp
biến đổi tuyến tính với các phép toán phi tuyến.
Để đơn giản, ở đây ta sẽ giả sử không có tính phi tuyến,
và phép biến đổi là một phép nhân ma trận lặp lại duy nhất $A$,
sao cho đầu ra của mô hình là

$$
\mathbf{v}_{out} = \mathbf{A}\cdot \mathbf{A}\cdots \mathbf{A} \mathbf{v}_{in} = \mathbf{A}^N \mathbf{v}_{in}.
$$

Khi các mô hình này được khởi tạo, $A$ được chọn là
một ma trận ngẫu nhiên với các phần tử Gaussian, vì vậy hãy tạo một ma trận như vậy.
Cụ thể, ta bắt đầu với một ma trận $5 \times 5$ có phân phối Gaussian
với trung bình bằng không và phương sai bằng một.

```{.python .input}
#@tab mxnet
np.random.seed(8675309)

k = 5
A = np.random.randn(k, k)
A
```

```{.python .input}
#@tab pytorch
torch.manual_seed(42)

k = 5
A = torch.randn(k, k, dtype=torch.float64)
A
```

```{.python .input}
#@tab tensorflow
k = 5
A = tf.random.normal((k, k), dtype=tf.float64)
A
```

### Hành vi trên dữ liệu ngẫu nhiên
Để đơn giản trong mô hình minh họa này,
ta sẽ giả sử vector dữ liệu đầu vào $\mathbf{v}_{in}$
là một vector Gaussian ngẫu nhiên năm chiều.
Hãy suy nghĩ về điều ta muốn xảy ra.
Để có bối cảnh, hãy xét một bài toán ML tổng quát,
trong đó ta cố gắng biến dữ liệu đầu vào, chẳng hạn như một hình ảnh, thành một dự đoán,
chẳng hạn như xác suất hình ảnh đó là ảnh một con mèo.
Nếu việc áp dụng lặp lại $\mathbf{A}$
kéo giãn một vector ngẫu nhiên thành một vector rất dài,
thì những thay đổi nhỏ ở đầu vào sẽ bị khuếch đại
thành những thay đổi lớn ở đầu ra---những sửa đổi rất nhỏ trên ảnh đầu vào
sẽ dẫn đến các dự đoán khác biệt hoàn toàn.
Điều này có vẻ không đúng!

Ngược lại, nếu $\mathbf{A}$ thu nhỏ các vector ngẫu nhiên,
thì sau khi đi qua nhiều lớp, vector về cơ bản sẽ co lại thành không,
và đầu ra sẽ không phụ thuộc vào đầu vào. Điều này rõ ràng cũng không đúng!

Ta cần đi trên ranh giới hẹp giữa tăng trưởng và suy giảm
để bảo đảm đầu ra thay đổi theo đầu vào, nhưng không thay đổi quá nhiều!

Hãy xem điều gì xảy ra khi ta liên tục nhân ma trận $\mathbf{A}$
với một vector đầu vào ngẫu nhiên và theo dõi chuẩn của vector.

```{.python .input}
#@tab mxnet
# Calculate the sequence of norms after repeatedly applying `A`
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Calculate the sequence of norms after repeatedly applying `A`
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Calculate the sequence of norms after repeatedly applying `A`
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

Chuẩn đang tăng một cách mất kiểm soát!
Quả thực, nếu lấy danh sách các thương số, ta sẽ thấy một quy luật.

```{.python .input}
#@tab mxnet
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

Nếu nhìn vào phần cuối của phép tính ở trên,
ta thấy vector ngẫu nhiên được kéo giãn theo hệ số `1.974459321485[...]`,
trong đó phần ở cuối chỉ dịch chuyển một chút,
nhưng hệ số kéo giãn vẫn ổn định.

### Liên hệ trở lại với các vector riêng

Ta đã thấy các vector riêng và trị riêng tương ứng
với mức độ một thứ gì đó bị kéo giãn,
nhưng điều đó chỉ đúng với những vector cụ thể và những độ kéo giãn cụ thể.
Hãy xem chúng là gì đối với $\mathbf{A}$.
Có một lưu ý ở đây: hóa ra để thấy được tất cả chúng,
ta sẽ cần chuyển sang các số phức.
Bạn có thể hình dung chúng là các phép kéo giãn và phép quay.
Bằng cách lấy chuẩn của số phức
(căn bậc hai của tổng bình phương phần thực và phần ảo),
ta có thể đo hệ số kéo giãn đó. Hãy đồng thời sắp xếp chúng.

```{.python .input}
#@tab mxnet
# Compute the eigenvalues
eigs = np.linalg.eigvals(A).tolist()
norm_eigs = [np.absolute(x) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab pytorch
# Compute the eigenvalues
eigs = torch.linalg.eig(A).eigenvalues.tolist()
norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab tensorflow
# Compute the eigenvalues
eigs = tf.linalg.eigh(A)[0].numpy().tolist()
norm_eigs = [tf.abs(tf.constant(x, dtype=tf.float64)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

### Một nhận xét

Ta thấy một điều hơi bất ngờ đang xảy ra ở đây:
con số mà trước đó ta đã xác định cho
độ kéo giãn dài hạn của ma trận $\mathbf{A}$
khi áp dụng lên một vector ngẫu nhiên chính là *đúng bằng*
(chính xác đến mười ba chữ số thập phân!)
trị riêng lớn nhất của $\mathbf{A}$.
Rõ ràng đây không phải là sự trùng hợp!

Tuy nhiên, nếu giờ ta suy nghĩ về điều đang xảy ra xét về mặt hình học,
điều này bắt đầu trở nên hợp lý. Hãy xét một vector ngẫu nhiên.
Vector ngẫu nhiên này hướng một phần về mọi hướng,
vì vậy đặc biệt là nó hướng ít nhất một phần
theo cùng hướng với vector riêng của $\mathbf{A}$
ứng với trị riêng lớn nhất.
Điều này quan trọng đến mức nó được gọi là
*trị riêng chính* và *vector riêng chính*.
Sau khi áp dụng $\mathbf{A}$, vector ngẫu nhiên của ta
được kéo giãn theo mọi hướng có thể,
tương ứng với mọi vector riêng có thể,
nhưng nó được kéo giãn nhiều nhất theo hướng
ứng với vector riêng chính này.
Điều đó có nghĩa là sau khi áp dụng $A$,
vector ngẫu nhiên của ta dài hơn và hướng theo một hướng
gần với sự thẳng hàng cùng vector riêng chính hơn.
Sau khi áp dụng ma trận nhiều lần,
sự thẳng hàng với vector riêng chính ngày càng tăng cho đến khi,
vì mọi mục đích thực tế, vector ngẫu nhiên của ta đã được biến đổi
thành vector riêng chính!
Quả thực, thuật toán này là cơ sở
của phương pháp được gọi là *lặp lũy thừa*
để tìm trị riêng và vector riêng lớn nhất của một ma trận. Để biết chi tiết, hãy xem chẳng hạn :cite:`Golub.Van-Loan.1996`.

### Điều chỉnh chuẩn hóa

Từ những thảo luận ở trên, ta kết luận
rằng ta không muốn một vector ngẫu nhiên bị kéo giãn hay nén lại,
mà muốn các vector ngẫu nhiên duy trì kích thước gần như không đổi trong suốt quá trình.
Để làm vậy, giờ ta chuẩn hóa lại ma trận bằng trị riêng chính này
sao cho trị riêng lớn nhất giờ chỉ còn bằng một.
Hãy xem điều gì xảy ra trong trường hợp này.

```{.python .input}
#@tab mxnet
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

Ta cũng có thể vẽ tỷ số giữa các chuẩn liên tiếp như trước
và thấy rằng quả thực tỷ số này ổn định.

```{.python .input}
#@tab mxnet
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

## Thảo luận

Giờ đây ta thấy chính xác điều mình mong đợi!
Sau khi chuẩn hóa các ma trận bằng trị riêng chính,
ta thấy dữ liệu ngẫu nhiên không bùng nổ như trước,
mà cuối cùng ổn định quanh một giá trị cụ thể.
Sẽ thật hữu ích nếu có thể thực hiện những điều này từ các nguyên lý đầu tiên,
và hóa ra khi nghiên cứu sâu vào phần toán học,
ta có thể thấy rằng trị riêng lớn nhất
của một ma trận ngẫu nhiên lớn với các phần tử Gaussian độc lập,
có trung bình bằng không và phương sai bằng một, trung bình vào khoảng $\sqrt{n}$,
hay trong trường hợp của chúng ta là $\sqrt{5} \approx 2.2$,
do một sự thật thú vị được gọi là *định luật hình tròn* :cite:`Ginibre.1965`.
Mối quan hệ giữa các trị riêng (và một đối tượng liên quan gọi là các giá trị kỳ dị)
của các ma trận ngẫu nhiên đã được chứng minh là có mối liên hệ sâu sắc
với việc khởi tạo đúng cách các mạng neural, như đã thảo luận trong :citet:`Pennington.Schoenholz.Ganguli.2017` và các công trình tiếp theo.

## Tóm tắt
* Vector riêng là các vector được ma trận kéo giãn mà không thay đổi hướng.
* Trị riêng là mức độ mà các vector riêng bị kéo giãn khi áp dụng ma trận.
* Phân rã riêng của một ma trận có thể cho phép rút gọn nhiều phép toán thành các phép toán trên các trị riêng.
* Định lý vòng tròn Gershgorin có thể cung cấp các giá trị xấp xỉ cho các trị riêng của một ma trận.
* Hành vi của các lũy thừa ma trận lặp lại phụ thuộc chủ yếu vào độ lớn của trị riêng lớn nhất. Hiểu biết này có nhiều ứng dụng trong lý thuyết khởi tạo mạng neural.

## Bài tập
1. Các trị riêng và vector riêng của
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}?
$$
là gì?
1.  Các trị riêng và vector riêng của ma trận sau là gì, và điều gì kỳ lạ ở ví dụ này so với ví dụ trước?
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
0 & 2
\end{bmatrix}.
$$
1. Không cần tính các trị riêng, liệu có khả năng trị riêng nhỏ nhất của ma trận sau nhỏ hơn $0.5$ không? *Gợi ý*: có thể giải bài này trong đầu.
$$
\mathbf{A} = \begin{bmatrix}
3.0 & 0.1 & 0.3 & 1.0 \\
0.1 & 1.0 & 0.1 & 0.2 \\
0.3 & 0.1 & 5.0 & 0.0 \\
1.0 & 0.2 & 0.0 & 1.8
\end{bmatrix}.
$$

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/411)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1086)
:end_tab:


:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1087)
:end_tab: