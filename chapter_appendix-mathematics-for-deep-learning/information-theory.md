# Lý thuyết thông tin
:label:`sec_information_theory`

Vũ trụ tràn ngập thông tin. Thông tin cung cấp một ngôn ngữ chung vượt qua những ranh giới giữa các lĩnh vực: từ Sonnet của Shakespeare đến bài báo của các nhà nghiên cứu trên Cornell ArXiv, từ bức tranh Starry Night của Van Gogh đến bản giao hưởng Symphony No. 5 của Beethoven, từ ngôn ngữ lập trình đầu tiên Plankalkül đến các thuật toán machine learning hiện đại nhất. Mọi thứ đều phải tuân theo các quy tắc của lý thuyết thông tin, bất kể định dạng. Với lý thuyết thông tin, chúng ta có thể đo lường và so sánh lượng thông tin hiện diện trong các tín hiệu khác nhau. Trong phần này, chúng ta sẽ tìm hiểu các khái niệm nền tảng của lý thuyết thông tin và những ứng dụng của lý thuyết thông tin trong machine learning.

Trước khi bắt đầu, hãy phác thảo mối quan hệ giữa machine learning và lý thuyết thông tin. Machine learning nhằm trích xuất các tín hiệu thú vị từ dữ liệu và đưa ra những dự đoán quan trọng. Mặt khác, lý thuyết thông tin nghiên cứu việc mã hóa, giải mã, truyền và thao tác với thông tin. Vì vậy, lý thuyết thông tin cung cấp ngôn ngữ nền tảng để thảo luận về quá trình xử lý thông tin trong các hệ thống machine learning. Ví dụ, nhiều ứng dụng machine learning sử dụng cross-entropy loss như được mô tả trong :numref:`sec_softmax`. Loss này có thể được suy ra trực tiếp từ các xét trên lý thuyết thông tin.

## Thông tin

Hãy bắt đầu với “linh hồn” của lý thuyết thông tin: thông tin. *Thông tin* có thể được mã hóa trong bất kỳ thứ gì bằng một chuỗi cụ thể gồm một hoặc nhiều định dạng mã hóa. Giả sử chúng ta tự đặt cho mình nhiệm vụ cố gắng định nghĩa một khái niệm về thông tin. Điểm khởi đầu của chúng ta có thể là gì?

Hãy xem xét thí nghiệm tưởng tượng sau. Chúng ta có một người bạn sở hữu một bộ bài. Người bạn đó sẽ xáo trộn bộ bài, lật một số lá bài và nói cho chúng ta biết các nhận định về những lá bài đó. Chúng ta sẽ cố gắng đánh giá lượng thông tin trong từng nhận định.

Đầu tiên, người bạn lật một lá bài và nói: “Tôi thấy một lá bài.” Điều này hoàn toàn không cung cấp thông tin nào cho chúng ta. Chúng ta vốn đã chắc chắn rằng điều này xảy ra, vì vậy chúng ta kỳ vọng lượng thông tin phải bằng không.

Tiếp theo, người bạn lật một lá bài và nói: “Tôi thấy một lá cơ.” Điều này cung cấp cho chúng ta một ít thông tin, nhưng thực tế chỉ có $4$ chất bài khác nhau có thể xuất hiện, mỗi chất có khả năng như nhau, nên chúng ta không ngạc nhiên trước kết quả này. Chúng ta kỳ vọng rằng, bất kể thước đo thông tin là gì, sự kiện này phải có lượng thông tin thấp.

Tiếp theo, người bạn lật một lá bài và nói: “Đây là lá $3$ bích.” Đây là nhiều thông tin hơn. Quả thực, có $52$ kết quả có khả năng xảy ra như nhau, và người bạn đã cho chúng ta biết kết quả nào đã xảy ra. Đây phải là một lượng thông tin trung bình.

Hãy đưa điều này đến cực hạn logic. Giả sử cuối cùng người bạn lật mọi lá bài trong bộ bài và đọc toàn bộ thứ tự của bộ bài đã được xáo trộn. Có $52!$ thứ tự khác nhau của bộ bài, một lần nữa tất cả đều có khả năng như nhau, vì vậy chúng ta cần rất nhiều thông tin để biết đó là thứ tự nào.

Bất kỳ khái niệm nào về thông tin mà chúng ta xây dựng cũng phải phù hợp với trực giác này. Quả thực, trong các phần tiếp theo, chúng ta sẽ học cách tính rằng những sự kiện này lần lượt có lượng thông tin bằng $0\textrm{ bits}$, $2\textrm{ bits}$, $~5.7\textrm{ bits}$ và $~225.6\textrm{ bits}$.

Nếu đọc qua các thí nghiệm tưởng tượng này, chúng ta nhận thấy một ý tưởng tự nhiên. Là điểm khởi đầu, thay vì quan tâm đến tri thức, chúng ta có thể dựa trên ý tưởng rằng thông tin biểu thị mức độ bất ngờ hoặc khả năng trừu tượng của sự kiện. Ví dụ, nếu muốn mô tả một sự kiện bất thường, chúng ta cần rất nhiều thông tin. Với một sự kiện phổ biến, chúng ta có thể không cần nhiều thông tin.

Năm 1948, Claude E. Shannon công bố *A Mathematical Theory of Communication* :cite:`Shannon.1948`, đặt nền móng cho lý thuyết thông tin. Trong bài viết của mình, Shannon lần đầu tiên giới thiệu khái niệm entropy thông tin. Chúng ta sẽ bắt đầu hành trình từ đây.

### Thông tin tự thân

Vì thông tin thể hiện khả năng trừu tượng của một sự kiện, làm thế nào chúng ta ánh xạ khả năng đó sang số bit? Shannon đã giới thiệu thuật ngữ *bit* làm đơn vị thông tin; thuật ngữ này vốn do John Tukey tạo ra. Vậy “bit” là gì và tại sao chúng ta sử dụng nó để đo thông tin? Trong lịch sử, một thiết bị truyền tín hiệu cổ chỉ có thể gửi hoặc nhận hai loại mã: $0$ và $1$. Quả thực, mã nhị phân vẫn được sử dụng phổ biến trên mọi máy tính kỹ thuật số hiện đại. Theo cách này, mọi thông tin đều được mã hóa bằng một chuỗi $0$ và $1$. Do đó, một chuỗi chữ số nhị phân có độ dài $n$ chứa $n$ bit thông tin.

Bây giờ, giả sử rằng trong bất kỳ chuỗi mã nào, mỗi $0$ hoặc $1$ xuất hiện với xác suất $\frac{1}{2}$. Vì vậy, một sự kiện $X$ với chuỗi mã có độ dài $n$ xảy ra với xác suất $\frac{1}{2^n}$. Đồng thời, như đã đề cập trước đó, chuỗi này chứa $n$ bit thông tin. Vậy chúng ta có thể khái quát thành một hàm toán học chuyển xác suất $p$ thành số bit hay không? Shannon đã đưa ra câu trả lời bằng cách định nghĩa *thông tin tự thân*

$$I(X) = - \log_2 (p),$$

là số *bit* thông tin mà chúng ta nhận được từ sự kiện $X$ này. Lưu ý rằng trong phần này, chúng ta luôn sử dụng logarithm cơ số 2. Để đơn giản, phần còn lại của phần này sẽ bỏ chỉ số 2 trong ký hiệu logarithm, tức là $\log(.)$ luôn có nghĩa là $\log_2(.)$. Ví dụ, mã “0010” có thông tin tự thân

$$I(\textrm{"0010"}) = - \log (p(\textrm{"0010"})) = - \log \left( \frac{1}{2^4} \right) = 4 \textrm{ bits}.$$

Chúng ta có thể tính thông tin tự thân như bên dưới. Trước hết, hãy import tất cả các package cần thiết trong phần này.

```{.python .input}
#@tab mxnet
from mxnet import np
from mxnet.metric import NegativeLogLikelihood
from mxnet.ndarray import nansum
import random

def self_information(p):
    return -np.log2(p)

self_information(1 / 64)
```

```{.python .input}
#@tab pytorch
import torch
from torch.nn import NLLLoss

def nansum(x):
    # Define nansum, as pytorch does not offer it inbuilt.
    return x[~torch.isnan(x)].sum()

def self_information(p):
    return -torch.log2(torch.tensor(p)).item()

self_information(1 / 64)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

def log2(x):
    return tf.math.log(x) / tf.math.log(2.)

def nansum(x):
    return tf.reduce_sum(tf.where(tf.math.is_nan(
        x), tf.zeros_like(x), x), axis=-1)

def self_information(p):
    return -log2(tf.constant(p)).numpy()

self_information(1 / 64)
```

## Entropy

Vì thông tin tự thân chỉ đo thông tin của một sự kiện rời rạc đơn lẻ, chúng ta cần một thước đo tổng quát hơn cho mọi biến ngẫu nhiên có phân phối rời rạc hoặc liên tục.

### Động lực hình thành entropy

Hãy cố gắng nói cụ thể hơn về điều chúng ta mong muốn. Đây sẽ là một phát biểu không chính thức về những điều được gọi là *tiên đề của entropy Shannon*. Hóa ra tập hợp các phát biểu hợp lẽ thường sau đây buộc chúng ta phải đi đến một định nghĩa duy nhất về thông tin. Một phiên bản chính thức của các tiên đề này, cùng với một số tiên đề khác, có thể được tìm thấy trong :citet:`Csiszar.2008`.

1. Thông tin chúng ta thu được khi quan sát một biến ngẫu nhiên không phụ thuộc vào cách chúng ta gọi tên các phần tử, cũng như không phụ thuộc vào sự hiện diện của các phần tử bổ sung có xác suất bằng không.
2. Thông tin chúng ta thu được khi quan sát hai biến ngẫu nhiên không lớn hơn tổng thông tin thu được khi quan sát riêng từng biến. Nếu chúng độc lập, thì lượng thông tin này chính xác bằng tổng đó.
3. Thông tin thu được khi quan sát các sự kiện (gần như) chắc chắn là (gần như) bằng không.

Mặc dù việc chứng minh điều này nằm ngoài phạm vi của tài liệu, điều quan trọng là phải biết rằng điều này xác định duy nhất dạng mà entropy phải có. Điểm không xác định duy nhất mà các tiên đề này cho phép nằm ở việc lựa chọn các đơn vị cơ bản; lựa chọn này thường được chuẩn hóa bằng cách sử dụng quy ước đã thấy trước đó, theo đó thông tin do một lần tung đồng xu công bằng cung cấp bằng một bit.

### Định nghĩa

Với mọi biến ngẫu nhiên $X$ tuân theo phân phối xác suất $P$ có hàm mật độ xác suất (p.d.f.) hoặc hàm khối lượng xác suất (p.m.f.) $p(x)$, chúng ta đo lượng thông tin kỳ vọng thông qua *entropy* (hay *entropy Shannon*)

$$H(X) = - E_{x \sim P} [\log p(x)].$$
:eqlabel:`eq_ent_def`

Cụ thể, nếu $X$ là biến rời rạc, $$H(X) = - \sum_i p_i \log p_i \textrm{, where } p_i = P(X_i).$$

Ngược lại, nếu $X$ là biến liên tục, chúng ta cũng gọi entropy là *entropy vi phân*

$$H(X) = - \int_x p(x) \log p(x) \; dx.$$

Chúng ta có thể định nghĩa entropy như bên dưới.

```{.python .input}
#@tab mxnet
def entropy(p):
    entropy = - p * np.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(entropy.as_nd_ndarray())
    return out

entropy(np.array([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input}
#@tab pytorch
def entropy(p):
    entropy = - p * torch.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(entropy)
    return out

entropy(torch.tensor([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input}
#@tab tensorflow
def entropy(p):
    return nansum(- p * log2(p))

entropy(tf.constant([0.1, 0.5, 0.1, 0.3]))
```

### Diễn giải

Bạn có thể thắc mắc: trong định nghĩa entropy :eqref:`eq_ent_def`, tại sao chúng ta sử dụng kỳ vọng của một logarithm âm? Dưới đây là một số trực giác.

Đầu tiên, tại sao chúng ta sử dụng hàm *logarithm* $\log$? Giả sử rằng $p(x) = f_1(x) f_2(x) \ldots, f_n(x)$, trong đó mỗi hàm thành phần $f_i(x)$ độc lập với các hàm khác. Điều này có nghĩa là mỗi $f_i(x)$ đóng góp độc lập vào tổng thông tin thu được từ $p(x)$. Như đã thảo luận ở trên, chúng ta muốn công thức entropy có tính cộng đối với các biến ngẫu nhiên độc lập. May mắn là $\log$ có thể tự nhiên chuyển tích của các phân phối xác suất thành tổng của các số hạng riêng lẻ.

Tiếp theo, tại sao chúng ta sử dụng $\log$ *âm*? Theo trực giác, các sự kiện xảy ra thường xuyên hơn phải chứa ít thông tin hơn các sự kiện ít phổ biến, vì chúng ta thường thu được nhiều thông tin hơn từ một trường hợp bất thường so với một trường hợp thông thường. Tuy nhiên, $\log$ là hàm đơn điệu tăng theo xác suất và thực tế là âm với mọi giá trị trong $[0, 1]$. Chúng ta cần xây dựng một mối quan hệ đơn điệu giảm giữa xác suất của các sự kiện và entropy của chúng; lý tưởng nhất, mối quan hệ này luôn dương (không có điều gì chúng ta quan sát được buộc chúng ta phải quên những gì mình đã biết). Vì vậy, chúng ta thêm dấu âm trước hàm $\log$.

Cuối cùng, hàm *kỳ vọng* xuất hiện từ đâu? Hãy xét một biến ngẫu nhiên $X$. Chúng ta có thể diễn giải thông tin tự thân ($-\log(p)$) là mức độ *bất ngờ* khi thấy một kết quả cụ thể. Quả thực, khi xác suất tiến gần đến 0, sự bất ngờ trở nên vô hạn. Tương tự, chúng ta có thể diễn giải entropy là mức độ bất ngờ trung bình khi quan sát $X$. Ví dụ, hãy tưởng tượng một hệ thống máy đánh bạc phát ra các ký hiệu ${s_1, \ldots, s_k}$ một cách độc lập về mặt thống kê với các xác suất lần lượt là ${p_1, \ldots, p_k}$. Khi đó, entropy của hệ thống này bằng thông tin tự thân trung bình khi quan sát từng đầu ra, tức là

$$H(S) = \sum_i {p_i \cdot I(s_i)} = - \sum_i {p_i \cdot \log p_i}.$$



### Các tính chất của Entropy

Dựa trên các ví dụ và cách diễn giải trên, ta có thể suy ra các tính chất sau của entropy :eqref:`eq_ent_def`. Ở đây, ta gọi $X$ là một biến cố và $P$ là phân phối xác suất của $X$.

* $H(X) \geq 0$ với mọi $X$ rời rạc (entropy có thể âm đối với $X$ liên tục).

* Nếu $X \sim P$ với hàm mật độ xác suất hoặc hàm khối xác suất $p(x)$, và ta cố gắng ước lượng $P$ bằng một phân phối xác suất mới $Q$ với hàm mật độ xác suất hoặc hàm khối xác suất $q(x)$, thì $$H(X) = - E_{x \sim P} [\log p(x)] \leq  - E_{x \sim P} [\log q(x)], \textrm{ with equality if and only if } P = Q.$$ Nói cách khác, $H(X)$ cho một cận dưới của số bit trung bình cần thiết để mã hóa các ký hiệu được lấy từ $P$.

* Nếu $X \sim P$, thì $x$ truyền đạt lượng thông tin tối đa khi nó được phân bố đồng đều giữa tất cả các kết quả có thể xảy ra. Cụ thể, nếu phân phối xác suất $P$ là rời rạc với $\{p_1, \ldots, p_k \}$ lớp, thì $$H(X) \leq \log(k), \textrm{ with equality if and only if } p_i = \frac{1}{k}, \forall i.$$ Nếu $P$ là một biến ngẫu nhiên liên tục, vấn đề sẽ phức tạp hơn nhiều. Tuy nhiên, nếu ta áp đặt thêm rằng $P$ có miền xác định là một khoảng hữu hạn (với mọi giá trị nằm giữa $0$ và $1$), thì $P$ có entropy cao nhất khi là phân phối đều trên khoảng đó.


## Thông tin tương hỗ

Trước đây, ta đã định nghĩa entropy của một biến ngẫu nhiên đơn $X$; vậy entropy của một cặp biến ngẫu nhiên $(X, Y)$ thì sao? Ta có thể xem các kỹ thuật này như một nỗ lực nhằm trả lời dạng câu hỏi sau: “Thông tin nào được chứa trong $X$ và $Y$ khi xét cùng nhau so với khi xét riêng từng biến? Có thông tin dư thừa hay tất cả đều là thông tin riêng biệt?”

Trong phần thảo luận sau, ta luôn sử dụng $(X, Y)$ làm một cặp biến ngẫu nhiên tuân theo một phân phối xác suất kết hợp $P$ với hàm mật độ xác suất hoặc hàm khối xác suất $p_{X, Y}(x, y)$, trong khi $X$ và $Y$ lần lượt tuân theo các phân phối xác suất $p_X(x)$ và $p_Y(y)$.


### Entropy kết hợp

Tương tự như entropy của một biến ngẫu nhiên đơn :eqref:`eq_ent_def`, ta định nghĩa *entropy kết hợp* $H(X, Y)$ của một cặp biến ngẫu nhiên $(X, Y)$ như sau:

$$H(X, Y) = -E_{(x, y) \sim P} [\log p_{X, Y}(x, y)]. $$
:eqlabel:`eq_joint_ent_def`

Cụ thể, một mặt, nếu $(X, Y)$ là một cặp biến ngẫu nhiên rời rạc, thì

$$H(X, Y) = - \sum_{x} \sum_{y} p_{X, Y}(x, y) \log p_{X, Y}(x, y).$$

Mặt khác, nếu $(X, Y)$ là một cặp biến ngẫu nhiên liên tục, ta định nghĩa *entropy kết hợp vi phân* như sau:

$$H(X, Y) = - \int_{x, y} p_{X, Y}(x, y) \ \log p_{X, Y}(x, y) \;dx \;dy.$$

Ta có thể hiểu :eqref:`eq_joint_ent_def` là đại lượng cho biết tổng mức độ ngẫu nhiên trong cặp biến ngẫu nhiên. Xét hai trường hợp cực đoan: nếu $X = Y$ là hai biến ngẫu nhiên giống hệt nhau, thì thông tin trong cặp này chính xác bằng thông tin trong một biến và ta có $H(X, Y) = H(X) = H(Y)$. Ở thái cực ngược lại, nếu $X$ và $Y$ độc lập thì $H(X, Y) = H(X) + H(Y)$. Thực tế, ta luôn có thông tin chứa trong một cặp biến ngẫu nhiên không nhỏ hơn entropy của mỗi biến riêng lẻ và không lớn hơn tổng entropy của cả hai biến.

$$
H(X), H(Y) \le H(X, Y) \le H(X) + H(Y).
$$

Bây giờ, hãy triển khai entropy kết hợp từ đầu.

```{.python .input}
#@tab mxnet
def joint_entropy(p_xy):
    joint_ent = -p_xy * np.log2(p_xy)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(joint_ent.as_nd_ndarray())
    return out

joint_entropy(np.array([[0.1, 0.5], [0.1, 0.3]]))
```

```{.python .input}
#@tab pytorch
def joint_entropy(p_xy):
    joint_ent = -p_xy * torch.log2(p_xy)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(joint_ent)
    return out

joint_entropy(torch.tensor([[0.1, 0.5], [0.1, 0.3]]))
```

```{.python .input}
#@tab tensorflow
def joint_entropy(p_xy):
    joint_ent = -p_xy * log2(p_xy)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(joint_ent)
    return out

joint_entropy(tf.constant([[0.1, 0.5], [0.1, 0.3]]))
```

Lưu ý rằng đây là cùng một *đoạn code* như trước, nhưng giờ ta diễn giải nó theo cách khác: đoạn code này hoạt động trên phân phối kết hợp của hai biến ngẫu nhiên.


### Entropy có điều kiện

Entropy kết hợp được định nghĩa ở trên biểu thị lượng thông tin chứa trong một cặp biến ngẫu nhiên. Điều này hữu ích, nhưng thường không phải là điều ta quan tâm. Hãy xét bối cảnh machine learning. Ta coi $X$ là biến ngẫu nhiên (hoặc vector gồm các biến ngẫu nhiên) mô tả các giá trị pixel của một hình ảnh, và $Y$ là biến ngẫu nhiên biểu thị nhãn lớp. $X$ chứa một lượng thông tin đáng kể---một hình ảnh tự nhiên là một đối tượng phức tạp. Tuy nhiên, thông tin chứa trong $Y$ sau khi hình ảnh đã được hiển thị sẽ thấp. Thực tế, hình ảnh của một chữ số vốn đã chứa thông tin về chữ số đó, trừ khi chữ số không thể đọc được. Vì vậy, để tiếp tục mở rộng vốn từ của lý thuyết thông tin, ta cần có khả năng suy luận về lượng thông tin trong một biến ngẫu nhiên khi biết một biến khác.

Trong lý thuyết xác suất, ta đã thấy định nghĩa của *xác suất có điều kiện* để đo lường mối quan hệ giữa các biến. Tương tự, giờ ta muốn định nghĩa *entropy có điều kiện* $H(Y \mid X)$. Ta có thể viết như sau:

$$ H(Y \mid X) = - E_{(x, y) \sim P} [\log p(y \mid x)],$$
:eqlabel:`eq_cond_ent_def`

trong đó $p(y \mid x) = \frac{p_{X, Y}(x, y)}{p_X(x)}$ là xác suất có điều kiện. Cụ thể, nếu $(X, Y)$ là một cặp biến ngẫu nhiên rời rạc, thì

$$H(Y \mid X) = - \sum_{x} \sum_{y} p(x, y) \log p(y \mid x).$$

Nếu $(X, Y)$ là một cặp biến ngẫu nhiên liên tục, thì *entropy có điều kiện vi phân* được định nghĩa tương tự như sau:

$$H(Y \mid X) = - \int_x \int_y p(x, y) \ \log p(y \mid x) \;dx \;dy.$$


Giờ đây, câu hỏi tự nhiên là entropy có điều kiện $H(Y \mid X)$ liên hệ thế nào với entropy $H(X)$ và entropy kết hợp $H(X, Y)$? Dựa trên các định nghĩa ở trên, ta có thể biểu diễn mối quan hệ này một cách rõ ràng:

$$H(Y \mid X) = H(X, Y) - H(X).$$

Điều này có cách diễn giải trực quan: thông tin trong $Y$ khi đã biết $X$ ($H(Y \mid X)$) bằng thông tin trong cả $X$ và $Y$ khi xét cùng nhau ($H(X, Y)$) trừ đi thông tin đã có trong $X$. Như vậy, ta thu được phần thông tin trong $Y$ không được biểu diễn đồng thời trong $X$.

Bây giờ, hãy triển khai entropy có điều kiện :eqref:`eq_cond_ent_def` từ đầu.

```{.python .input}
#@tab mxnet
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * np.log2(p_y_given_x)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(cond_ent.as_nd_ndarray())
    return out

conditional_entropy(np.array([[0.1, 0.5], [0.2, 0.3]]), np.array([0.2, 0.8]))
```

```{.python .input}
#@tab pytorch
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * torch.log2(p_y_given_x)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(cond_ent)
    return out

conditional_entropy(torch.tensor([[0.1, 0.5], [0.2, 0.3]]),
                    torch.tensor([0.2, 0.8]))
```

```{.python .input}
#@tab tensorflow
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * log2(p_y_given_x)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(cond_ent)
    return out

conditional_entropy(tf.constant([[0.1, 0.5], [0.2, 0.3]]),
                    tf.constant([0.2, 0.8]))
```

### Thông tin tương hỗ

Với bối cảnh các biến ngẫu nhiên $(X, Y)$ như trên, có thể bạn sẽ thắc mắc: “Giờ ta đã biết lượng thông tin chứa trong $Y$ nhưng không có trong $X$, liệu ta có thể hỏi tương tự rằng có bao nhiêu thông tin được chia sẻ giữa $X$ và $Y$ không?” Câu trả lời là *thông tin tương hỗ* của $(X, Y)$, được ký hiệu là $I(X, Y)$.

Thay vì đi thẳng vào định nghĩa hình thức, trước hết hãy rèn luyện trực giác bằng cách cố gắng suy ra một biểu thức cho thông tin tương hỗ hoàn toàn dựa trên các thành phần mà ta đã xây dựng trước đó. Ta muốn tìm thông tin được chia sẻ giữa hai biến ngẫu nhiên. Một cách để thực hiện là bắt đầu với toàn bộ thông tin chứa trong cả $X$ và $Y$, sau đó loại bỏ những phần không được chia sẻ. Thông tin chứa trong cả $X$ và $Y$ được viết là $H(X, Y)$. Ta muốn trừ khỏi đại lượng này thông tin chứa trong $X$ nhưng không có trong $Y$, và thông tin chứa trong $Y$ nhưng không có trong $X$. Như đã thấy trong phần trước, lần lượt đó là $H(X \mid Y)$ và $H(Y \mid X)$. Do đó, thông tin tương hỗ phải là

$$
I(X, Y) = H(X, Y) - H(Y \mid X) - H(X \mid Y).
$$

Đây thực sự là một định nghĩa hợp lệ của thông tin tương hỗ. Nếu khai triển các định nghĩa của những thành phần này và kết hợp chúng, một chút biến đổi đại số cho thấy biểu thức trên cũng chính là

$$I(X, Y) = E_{x} E_{y} \left\{ p_{X, Y}(x, y) \log\frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)} \right\}. $$
:eqlabel:`eq_mut_ent_def`


Ta có thể tóm tắt tất cả các mối quan hệ này trong image :numref:`fig_mutual_information`. Đây là một cách kiểm tra trực giác rất tốt: hãy xem vì sao các phát biểu sau cũng tương đương với $I(X, Y)$.

* $H(X) - H(X \mid Y)$
* $H(Y) - H(Y \mid X)$
* $H(X) + H(Y) - H(X, Y)$

![Mối quan hệ giữa thông tin tương hỗ với entropy kết hợp và entropy có điều kiện.](../img/mutual-information.svg)
:label:`fig_mutual_information`


Theo nhiều cách, ta có thể xem thông tin tương hỗ :eqref:`eq_mut_ent_def` như một mở rộng có cơ sở lý thuyết của hệ số tương quan mà ta đã thấy trong :numref:`sec_random_variables`. Điều này cho phép ta không chỉ tìm hiểu các mối quan hệ tuyến tính giữa các biến, mà còn tìm hiểu lượng thông tin tối đa được chia sẻ giữa hai biến ngẫu nhiên dưới bất kỳ dạng quan hệ nào.

Bây giờ, hãy triển khai thông tin tương hỗ từ đầu.

```{.python .input}
#@tab mxnet
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * np.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(mutual.as_nd_ndarray())
    return out

mutual_information(np.array([[0.1, 0.5], [0.1, 0.3]]),
                   np.array([0.2, 0.8]), np.array([[0.75, 0.25]]))
```

```{.python .input}
#@tab pytorch
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * torch.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(mutual)
    return out

mutual_information(torch.tensor([[0.1, 0.5], [0.1, 0.3]]),
                   torch.tensor([0.2, 0.8]), torch.tensor([[0.75, 0.25]]))
```

```{.python .input}
#@tab tensorflow
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(mutual)
    return out

mutual_information(tf.constant([[0.1, 0.5], [0.1, 0.3]]),
                   tf.constant([0.2, 0.8]), tf.constant([[0.75, 0.25]]))
```

### Các tính chất của thông tin tương hỗ

Thay vì ghi nhớ định nghĩa của thông tin tương hỗ :eqref:`eq_mut_ent_def`, bạn chỉ cần ghi nhớ các tính chất đáng chú ý sau:

* Thông tin tương hỗ có tính đối xứng, tức là $I(X, Y) = I(Y, X)$.
* Thông tin tương hỗ không âm, tức là $I(X, Y) \geq 0$.
* $I(X, Y) = 0$ khi và chỉ khi $X$ và $Y$ độc lập. Ví dụ, nếu $X$ và $Y$ độc lập, thì việc biết $Y$ không cung cấp thông tin nào về $X$ và ngược lại, nên thông tin tương hỗ của chúng bằng không.
* Nói cách khác, nếu $X$ là một hàm khả nghịch của $Y$, thì $Y$ và $X$ chia sẻ toàn bộ thông tin và $$I(X, Y) = H(Y) = H(X).$$

### Thông tin tương hỗ điểm

Khi làm việc với entropy ở phần đầu chương này, ta đã có thể diễn giải $-\log(p_X(x))$ là mức độ *bất ngờ* của ta trước một kết quả cụ thể. Ta có thể đưa ra cách diễn giải tương tự cho số hạng logarithm trong thông tin tương hỗ; số hạng này thường được gọi là *thông tin tương hỗ điểm*:

$$\textrm{pmi}(x, y) = \log\frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)}.$$
:eqlabel:`eq_pmi_def`

Ta có thể hiểu :eqref:`eq_pmi_def` là phép đo mức độ kết hợp cụ thể của các kết quả $x$ và $y$ có khả năng xảy ra nhiều hơn hoặc ít hơn bao nhiêu so với kỳ vọng đối với các kết quả ngẫu nhiên độc lập. Nếu giá trị này lớn và dương, thì hai kết quả cụ thể này xảy ra thường xuyên hơn nhiều so với khi xảy ra một cách ngẫu nhiên (*lưu ý*: mẫu số là $p_X(x) p_Y(y)$, tức xác suất để hai kết quả xảy ra độc lập), còn nếu giá trị này lớn và âm, thì nó biểu thị rằng hai kết quả xảy ra ít hơn nhiều so với kỳ vọng do ngẫu nhiên.

Điều này cho phép ta diễn giải thông tin tương hỗ :eqref:`eq_mut_ent_def` là mức độ trung bình mà ta ngạc nhiên khi thấy hai kết quả xảy ra cùng nhau, so với mức ta kỳ vọng nếu chúng độc lập.

### Các ứng dụng của thông tin tương hỗ

Định nghĩa thuần túy của thông tin tương hỗ có thể hơi trừu tượng, vậy nó liên quan thế nào đến machine learning? Trong xử lý ngôn ngữ tự nhiên, một trong những bài toán khó nhất là *giải quyết nhập nhằng* (ambiguity resolution), tức vấn đề không thể xác định rõ nghĩa của một từ chỉ dựa vào ngữ cảnh. Ví dụ, gần đây một tiêu đề trên báo đưa tin rằng "Amazon is on fire". Bạn có thể tự hỏi liệu công ty Amazon có một tòa nhà đang cháy, hay rừng Amazon đang cháy.

Trong trường hợp này, thông tin tương hỗ có thể giúp ta giải quyết sự nhập nhằng. Trước tiên, ta tìm nhóm các từ mà mỗi từ có thông tin tương hỗ tương đối lớn với công ty Amazon, chẳng hạn như e-commerce, technology và online. Tiếp theo, ta tìm một nhóm khác gồm các từ mà mỗi từ có thông tin tương hỗ tương đối lớn với rừng Amazon, chẳng hạn như rain, forest và tropical. Khi cần phân biệt nghĩa của "Amazon", ta có thể so sánh nhóm nào xuất hiện nhiều hơn trong ngữ cảnh của từ Amazon. Trong trường hợp này, bài viết sẽ tiếp tục mô tả khu rừng, qua đó làm rõ ngữ cảnh.

## Độ phân kỳ Kullback–Leibler

Như đã thảo luận trong :numref:`sec_linear-algebra`, ta có thể sử dụng các chuẩn để đo khoảng cách giữa hai điểm trong không gian có số chiều bất kỳ. Ta cũng muốn thực hiện một nhiệm vụ tương tự với các phân phối xác suất. Có nhiều cách để thực hiện việc này, nhưng lý thuyết thông tin cung cấp một trong những cách tốt nhất. Bây giờ, ta sẽ tìm hiểu về *độ phân kỳ Kullback–Leibler (KL)*, một phương pháp đo xem hai phân phối có gần nhau hay không.

### Định nghĩa

Cho một biến ngẫu nhiên $X$ tuân theo phân phối xác suất $P$ với p.d.f. hoặc p.m.f. $p(x)$, và ta ước lượng $P$ bằng một phân phối xác suất khác $Q$ với p.d.f. hoặc p.m.f. $q(x)$. Khi đó, *độ phân kỳ Kullback–Leibler (KL)* (hay *entropy tương đối*) giữa $P$ và $Q$ là

$$D_{\textrm{KL}}(P\|Q) = E_{x \sim P} \left[ \log \frac{p(x)}{q(x)} \right].$$
:eqlabel:`eq_kl_def`

Tương tự như thông tin tương hỗ từng điểm :eqref:`eq_pmi_def`, ta có thể diễn giải thêm về số hạng logarit: $-\log \frac{q(x)}{p(x)} = -\log(q(x)) - (-\log(p(x)))$ sẽ lớn và dương nếu ta quan sát thấy $x$ thường xuyên hơn nhiều dưới $P$ so với kỳ vọng dưới $Q$, và lớn và âm nếu ta quan sát thấy kết quả này ít hơn nhiều so với kỳ vọng. Theo cách này, ta có thể diễn giải nó là mức độ ngạc nhiên *tương đối* của ta khi quan sát kết quả, so với mức độ ngạc nhiên nếu quan sát kết quả đó từ phân phối tham chiếu.

Hãy triển khai độ phân kỳ KL từ đầu.

```{.python .input}
#@tab mxnet
def kl_divergence(p, q):
    kl = p * np.log2(p / q)
    out = nansum(kl.as_nd_ndarray())
    return out.abs().asscalar()
```

```{.python .input}
#@tab pytorch
def kl_divergence(p, q):
    kl = p * torch.log2(p / q)
    out = nansum(kl)
    return out.abs().item()
```

```{.python .input}
#@tab tensorflow
def kl_divergence(p, q):
    kl = p * log2(p / q)
    out = nansum(kl)
    return tf.abs(out).numpy()
```

### Các tính chất của độ phân kỳ KL

Hãy cùng xem xét một số tính chất của độ phân kỳ KL :eqref:`eq_kl_def`.

* Độ phân kỳ KL không đối xứng, tức là tồn tại $P,Q$ sao cho $$D_{\textrm{KL}}(P\|Q) \neq D_{\textrm{KL}}(Q\|P).$$
* Độ phân kỳ KL không âm, tức là $$D_{\textrm{KL}}(P\|Q) \geq 0.$$ Lưu ý rằng đẳng thức chỉ xảy ra khi $P = Q$.
* Nếu tồn tại một $x$ sao cho $p(x) > 0$ và $q(x) = 0$, thì $D_{\textrm{KL}}(P\|Q) = \infty$.
* Có mối quan hệ chặt chẽ giữa độ phân kỳ KL và thông tin tương hỗ. Ngoài mối quan hệ được thể hiện trong :numref:`fig_mutual_information`, $I(X, Y)$ cũng tương đương về mặt số học với các số hạng sau:
    1. $D_{\textrm{KL}}(P(X, Y)  \ \| \ P(X)P(Y))$;
    1. $E_Y \{ D_{\textrm{KL}}(P(X \mid Y) \ \| \ P(X)) \}$;
    1. $E_X \{ D_{\textrm{KL}}(P(Y \mid X) \ \| \ P(Y)) \}$.

  Với số hạng thứ nhất, ta diễn giải thông tin tương hỗ là độ phân kỳ KL giữa $P(X, Y)$ và tích của $P(X)$ với $P(Y)$, do đó đây là phép đo mức độ khác biệt giữa phân phối đồng thời và phân phối trong trường hợp chúng độc lập. Với số hạng thứ hai, thông tin tương hỗ cho biết mức giảm trung bình của độ bất định về $Y$ sau khi biết giá trị của phân phối $X$. Tương tự đối với số hạng thứ ba.


### Ví dụ

Hãy xem qua một ví dụ đơn giản để thấy rõ tính không đối xứng.

Trước tiên, hãy tạo và sắp xếp ba tensor có độ dài $10,000$: một tensor mục tiêu $p$ tuân theo phân phối chuẩn $N(0, 1)$, và hai tensor ứng viên $q_1$ và $q_2$ tuân theo các phân phối chuẩn lần lượt là $N(-1, 1)$ và $N(1, 1)$.

```{.python .input}
#@tab mxnet
random.seed(1)

nd_len = 10000
p = np.random.normal(loc=0, scale=1, size=(nd_len, ))
q1 = np.random.normal(loc=-1, scale=1, size=(nd_len, ))
q2 = np.random.normal(loc=1, scale=1, size=(nd_len, ))

p = np.array(sorted(p.asnumpy()))
q1 = np.array(sorted(q1.asnumpy()))
q2 = np.array(sorted(q2.asnumpy()))
```

```{.python .input}
#@tab pytorch
torch.manual_seed(1)

tensor_len = 10000
p = torch.normal(0, 1, (tensor_len, ))
q1 = torch.normal(-1, 1, (tensor_len, ))
q2 = torch.normal(1, 1, (tensor_len, ))

p = torch.sort(p)[0]
q1 = torch.sort(q1)[0]
q2 = torch.sort(q2)[0]
```

```{.python .input}
#@tab tensorflow
tensor_len = 10000
p = tf.random.normal((tensor_len, ), 0, 1)
q1 = tf.random.normal((tensor_len, ), -1, 1)
q2 = tf.random.normal((tensor_len, ), 1, 1)

p = tf.sort(p)
q1 = tf.sort(q1)
q2 = tf.sort(q2)
```

Vì $q_1$ và $q_2$ đối xứng qua trục y (tức là $x=0$), ta kỳ vọng giá trị độ phân kỳ KL giữa $D_{\textrm{KL}}(p\|q_1)$ và $D_{\textrm{KL}}(p\|q_2)$ sẽ tương tự nhau. Như bạn có thể thấy bên dưới, độ chênh lệch giữa $D_{\textrm{KL}}(p\|q_1)$ và $D_{\textrm{KL}}(p\|q_2)$ chưa đến 3%.

```{.python .input}
#@tab all
kl_pq1 = kl_divergence(p, q1)
kl_pq2 = kl_divergence(p, q2)
similar_percentage = abs(kl_pq1 - kl_pq2) / ((kl_pq1 + kl_pq2) / 2) * 100

kl_pq1, kl_pq2, similar_percentage
```

Ngược lại, bạn có thể thấy rằng $D_{\textrm{KL}}(q_2 \|p)$ và $D_{\textrm{KL}}(p \| q_2)$ chênh lệch khá nhiều, khoảng 40% như minh họa bên dưới.

```{.python .input}
#@tab all
kl_q2p = kl_divergence(q2, p)
differ_percentage = abs(kl_q2p - kl_pq2) / ((kl_q2p + kl_pq2) / 2) * 100

kl_q2p, differ_percentage
```

## Cross-Entropy

Nếu bạn quan tâm đến các ứng dụng của lý thuyết thông tin trong deep learning, sau đây là một ví dụ ngắn. Ta định nghĩa phân phối thực $P$ với phân phối xác suất $p(x)$ và phân phối ước lượng $Q$ với phân phối xác suất $q(x)$, rồi sử dụng chúng trong phần còn lại của mục này.

Giả sử ta cần giải một bài toán phân loại nhị phân dựa trên $n$ ví dụ dữ liệu đã cho {$x_1, \ldots, x_n$}. Giả sử ta mã hóa $1$ và $0$ lần lượt là nhãn lớp dương và âm $y_i$, còn neural network của ta được tham số hóa bởi $\theta$. Nếu mục tiêu là tìm $\theta$ tốt nhất sao cho $\hat{y}_i= p_{\theta}(y_i \mid x_i)$, thì việc áp dụng phương pháp maximum log-likelihood như đã trình bày trong :numref:`sec_maximum_likelihood` là tự nhiên. Cụ thể, với các nhãn thực $y_i$ và các dự đoán $\hat{y}_i= p_{\theta}(y_i \mid x_i)$, xác suất được phân loại là lớp dương là $\pi_i= p_{\theta}(y_i = 1 \mid x_i)$. Do đó, hàm log-likelihood sẽ là

$$
\begin{aligned}
l(\theta) &= \log L(\theta) \\
  &= \log \prod_{i=1}^n \pi_i^{y_i} (1 - \pi_i)^{1 - y_i} \\
  &= \sum_{i=1}^n y_i \log(\pi_i) + (1 - y_i) \log (1 - \pi_i). \\
\end{aligned}
$$

Tối đa hóa hàm log-likelihood $l(\theta)$ tương đương với tối thiểu hóa $- l(\theta)$, vì vậy từ đây ta có thể tìm được $\theta$ tốt nhất. Để tổng quát hóa loss ở trên cho mọi phân phối, ta cũng gọi $-l(\theta)$ là *cross-entropy loss* $\textrm{CE}(y, \hat{y})$, trong đó $y$ tuân theo phân phối thực $P$ và $\hat{y}$ tuân theo phân phối ước lượng $Q$.

Tất cả điều này được suy ra từ góc nhìn maximum likelihood. Tuy nhiên, nếu xem xét kỹ, ta có thể thấy các số hạng như $\log(\pi_i)$ đã xuất hiện trong phép tính, đây là dấu hiệu rõ ràng cho thấy ta có thể hiểu biểu thức này từ góc nhìn lý thuyết thông tin.


### Định nghĩa hình thức

Tương tự độ phân kỳ KL, với một biến ngẫu nhiên $X$, ta cũng có thể đo độ phân kỳ giữa phân phối ước lượng $Q$ và phân phối thực $P$ thông qua *cross-entropy*,

$$\textrm{CE}(P, Q) = - E_{x \sim P} [\log(q(x))].$$
:eqlabel:`eq_ce_def`

Bằng cách sử dụng các tính chất của entropy đã thảo luận ở trên, ta cũng có thể diễn giải nó là tổng của entropy $H(P)$ và độ phân kỳ KL giữa $P$ và $Q$, tức là

$$\textrm{CE} (P, Q) = H(P) + D_{\textrm{KL}}(P\|Q).$$


Ta có thể triển khai cross-entropy loss như bên dưới.

```{.python .input}
#@tab mxnet
def cross_entropy(y_hat, y):
    ce = -np.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

```{.python .input}
#@tab pytorch
def cross_entropy(y_hat, y):
    ce = -torch.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    # `tf.gather_nd` is used to select specific indices of a tensor.
    ce = -tf.math.log(tf.gather_nd(y_hat, indices = [[i, j] for i, j in zip(
        range(len(y_hat)), y)]))
    return tf.reduce_mean(ce).numpy()
```

Bây giờ, hãy định nghĩa hai tensor cho các nhãn và dự đoán, rồi tính cross-entropy loss của chúng.

```{.python .input}
#@tab mxnet
labels = np.array([0, 2])
preds = np.array([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input}
#@tab pytorch
labels = torch.tensor([0, 2])
preds = torch.tensor([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input}
#@tab tensorflow
labels = tf.constant([0, 2])
preds = tf.constant([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

### Các tính chất

Như đã đề cập ở phần đầu của mục này, cross-entropy :eqref:`eq_ce_def` có thể được dùng để định nghĩa một hàm loss trong bài toán tối ưu hóa. Kết quả là các mục tiêu sau tương đương:

1. Tối đa hóa xác suất dự đoán của $Q$ đối với phân phối $P$, (tức là $E_{x
\sim P} [\log (q(x))]$);
1. Tối thiểu hóa cross-entropy $\textrm{CE} (P, Q)$;
1. Tối thiểu hóa độ phân kỳ KL $D_{\textrm{KL}}(P\|Q)$.

Định nghĩa cross-entropy gián tiếp chứng minh mối quan hệ tương đương giữa mục tiêu 2 và mục tiêu 3, miễn là entropy của dữ liệu thực $H(P)$ không đổi.


### Cross-Entropy với vai trò Hàm Mục tiêu của Phân loại Đa lớp

Nếu đi sâu vào hàm mục tiêu phân loại với cross-entropy loss $\textrm{CE}$, ta sẽ thấy việc tối thiểu hóa $\textrm{CE}$ tương đương với việc tối đa hóa hàm log-likelihood $L$.

Trước hết, giả sử chúng ta có một dataset gồm $n$ mẫu và dataset đó có thể được phân loại thành $k$ lớp. Với mỗi mẫu dữ liệu $i$, chúng ta biểu diễn mọi nhãn thuộc $k$ lớp $\mathbf{y}_i = (y_{i1}, \ldots, y_{ik})$ bằng phương pháp *one-hot encoding*. Cụ thể, nếu mẫu $i$ thuộc lớp $j$, chúng ta đặt phần tử thứ $j$ bằng $1$ và tất cả các thành phần còn lại bằng $0$, tức là

$$ y_{ij} = \begin{cases}1 & j \in J; \\ 0 &\textrm{otherwise.}\end{cases}$$

Ví dụ, nếu một bài toán phân loại đa lớp gồm ba lớp $A$, $B$ và $C$, thì các nhãn $\mathbf{y}_i$ có thể được mã hóa trong {$A: (1, 0, 0); B: (0, 1, 0); C: (0, 0, 1)$}.


Giả sử mạng neural của chúng ta được tham số hóa bởi $\theta$. Với các vector nhãn thực $\mathbf{y}_i$ và các dự đoán $$\hat{\mathbf{y}}_i= p_{\theta}(\mathbf{y}_i \mid \mathbf{x}_i) = \sum_{j=1}^k y_{ij} p_{\theta} (y_{ij}  \mid  \mathbf{x}_i).$$

Do đó, *hàm mất mát cross-entropy* sẽ là

$$
\textrm{CE}(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{i=1}^n \mathbf{y}_i \log \hat{\mathbf{y}}_i
 = - \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{p_{\theta} (y_{ij}  \mid  \mathbf{x}_i)}.\\
$$

Mặt khác, chúng ta cũng có thể tiếp cận bài toán thông qua phương pháp ước lượng hợp lý cực đại. Trước hết, hãy cùng giới thiệu nhanh về phân phối multinoulli $k$ lớp. Đây là phần mở rộng của phân phối Bernoulli từ lớp nhị phân sang nhiều lớp. Nếu một biến ngẫu nhiên $\mathbf{z} = (z_{1}, \ldots, z_{k})$ tuân theo một *phân phối multinoulli* $k$ lớp với các xác suất $\mathbf{p} =$ ($p_{1}, \ldots, p_{k}$), tức là $$p(\mathbf{z}) = p(z_1, \ldots, z_k) = \textrm{Multi} (p_1, \ldots, p_k), \textrm{ where } \sum_{i=1}^k p_i = 1,$$ thì hàm khối xác suất chung (p.m.f.) của $\mathbf{z}$ là
$$\mathbf{p}^\mathbf{z} = \prod_{j=1}^k p_{j}^{z_{j}}.$$


Có thể thấy rằng nhãn của mỗi mẫu dữ liệu, $\mathbf{y}_i$, tuân theo một phân phối multinoulli $k$ lớp với các xác suất $\boldsymbol{\pi} =$ ($\pi_{1}, \ldots, \pi_{k}$). Do đó, p.m.f. chung của mỗi mẫu dữ liệu $\mathbf{y}_i$ là $\mathbf{\pi}^{\mathbf{y}_i} = \prod_{j=1}^k \pi_{j}^{y_{ij}}.$
Do đó, hàm log-likelihood sẽ là

$$
\begin{aligned}
l(\theta)
 = \log L(\theta)
 = \log \prod_{i=1}^n \boldsymbol{\pi}^{\mathbf{y}_i}
 = \log \prod_{i=1}^n \prod_{j=1}^k \pi_{j}^{y_{ij}}
 = \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{\pi_{j}}.\\
\end{aligned}
$$

Trong phương pháp ước lượng hợp lý cực đại, chúng ta cực đại hóa hàm mục tiêu $l(\theta)$ bằng cách đặt $\pi_{j} = p_{\theta} (y_{ij}  \mid  \mathbf{x}_i)$. Vì vậy, đối với mọi bài toán phân loại đa lớp, việc cực đại hóa hàm log-likelihood $l(\theta)$ ở trên tương đương với việc tối thiểu hóa hàm mất mát CE $\textrm{CE}(y, \hat{y})$.


Để kiểm chứng chứng minh trên, hãy áp dụng measure tích hợp sẵn `NegativeLogLikelihood`. Sử dụng cùng `labels` và `preds` như trong ví dụ trước, chúng ta sẽ nhận được giá trị loss bằng với ví dụ trước, với độ chính xác đến 5 chữ số thập phân.

```{.python .input}
#@tab mxnet
nll_loss = NegativeLogLikelihood()
nll_loss.update(labels.as_nd_ndarray(), preds.as_nd_ndarray())
nll_loss.get()
```

```{.python .input}
#@tab pytorch
# Implementation of cross-entropy loss in PyTorch combines `nn.LogSoftmax()`
# and `nn.NLLLoss()`
nll_loss = NLLLoss()
loss = nll_loss(torch.log(preds), labels)
loss
```

```{.python .input}
#@tab tensorflow
def nll_loss(y_hat, y):
    # Convert labels to one-hot vectors.
    y = tf.keras.utils.to_categorical(y, num_classes= y_hat.shape[1])
    # We will not calculate negative log-likelihood from the definition.
    # Rather, we will follow a circular argument. Because NLL is same as
    # `cross_entropy`, if we calculate cross_entropy that would give us NLL
    cross_entropy = tf.keras.losses.CategoricalCrossentropy(
        from_logits = True, reduction = tf.keras.losses.Reduction.NONE)
    return tf.reduce_mean(cross_entropy(y, y_hat)).numpy()

loss = nll_loss(tf.math.log(preds), labels)
loss
```

## Tóm tắt

* Information theory là một lĩnh vực nghiên cứu về mã hóa, giải mã, truyền tải và thao tác với thông tin.
* Entropy là đơn vị dùng để đo lượng thông tin được biểu diễn trong các tín hiệu khác nhau.
* KL divergence cũng có thể đo độ phân kỳ giữa hai phân phối.
* Cross-entropy có thể được xem là một hàm mục tiêu của bài toán phân loại đa lớp. Việc tối thiểu hóa hàm mất mát cross-entropy tương đương với việc cực đại hóa hàm log-likelihood.


## Bài tập

1. Xác minh rằng các ví dụ về những lá bài trong phần đầu tiên thực sự có entropy như đã nêu.
1. Chứng minh rằng KL divergence $D(p\|q)$ không âm với mọi phân phối $p$ và $q$. Gợi ý: sử dụng bất đẳng thức Jensen, tức là sử dụng thực tế rằng $-\log x$ là một hàm lồi.
1. Hãy tính entropy từ một vài nguồn dữ liệu:
    * Giả sử bạn đang quan sát đầu ra do một con khỉ tạo ra trên máy đánh chữ. Con khỉ nhấn ngẫu nhiên một trong $44$ phím của máy đánh chữ (bạn có thể giả sử rằng nó vẫn chưa phát hiện ra các phím đặc biệt hoặc phím shift). Bạn quan sát được bao nhiêu bit ngẫu nhiên trên mỗi ký tự?
    * Vì không hài lòng với con khỉ, bạn thay nó bằng một người sắp chữ say rượu. Người này có thể tạo ra các từ, dù không mạch lạc. Thay vào đó, người này chọn ngẫu nhiên một từ trong từ vựng gồm $2,000$ từ. Giả sử độ dài trung bình của một từ là $4.5$ chữ cái trong tiếng Anh. Bây giờ bạn quan sát được bao nhiêu bit ngẫu nhiên trên mỗi ký tự?
    * Vì vẫn không hài lòng với kết quả, bạn thay người sắp chữ bằng một language model chất lượng cao. Hiện tại, language model có thể đạt perplexity thấp tới $15$ điểm trên mỗi từ. *Perplexity* theo ký tự của một language model được định nghĩa là nghịch đảo của trung bình nhân của một tập hợp các xác suất, trong đó mỗi xác suất tương ứng với một ký tự trong từ. Cụ thể, nếu độ dài của một từ cho trước là $l$, thì  $\textrm{PPL}(\textrm{word}) = \left[\prod_i p(\textrm{character}_i)\right]^{ -\frac{1}{l}} = \exp \left[ - \frac{1}{l} \sum_i{\log p(\textrm{character}_i)} \right].$  Giả sử từ kiểm thử có 4.5 chữ cái, bây giờ bạn quan sát được bao nhiêu bit ngẫu nhiên trên mỗi ký tự?
1. Giải thích một cách trực quan tại sao $I(X, Y) = H(X) - H(X \mid Y)$. Sau đó, chứng minh điều này bằng cách biểu diễn cả hai vế dưới dạng một kỳ vọng theo phân phối chung.
1. KL Divergence giữa hai phân phối Gaussian $\mathcal{N}(\mu_1, \sigma_1^2)$ và $\mathcal{N}(\mu_2, \sigma_2^2)$ là bao nhiêu?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/420)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1104)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1105)
:end_tab: