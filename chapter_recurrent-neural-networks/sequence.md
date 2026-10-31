# Làm việc với Dãy
:label:`sec_sequence`

Cho đến nay, chúng ta đã tập trung vào các mô hình có đầu vào
bao gồm một vector đặc trưng duy nhất $\mathbf{x} \in \mathbb{R}^d$.
Sự thay đổi chính về góc nhìn khi phát triển các mô hình
có khả năng xử lý dãy là giờ đây chúng ta
tập trung vào các đầu vào bao gồm một danh sách có thứ tự
các vector đặc trưng $\mathbf{x}_1, \dots, \mathbf{x}_T$,
trong đó mỗi vector đặc trưng $\mathbf{x}_t$ được
đánh chỉ mục bởi một bước thời gian $t \in \mathbb{Z}^+$
và thuộc $\mathbb{R}^d$.

Một số tập dữ liệu bao gồm một dãy đơn khổng lồ.
Ví dụ, hãy xem xét các luồng dữ liệu cảm biến cực dài
có thể उपलब्ध cho các nhà khoa học khí hậu.
Trong những trường hợp như vậy, chúng ta có thể tạo tập dữ liệu huấn luyện
bằng cách lấy mẫu ngẫu nhiên các dãy con có độ dài được xác định trước.
Tuy nhiên, thường thì dữ liệu của chúng ta đến dưới dạng một tập hợp các dãy.
Hãy xem xét các ví dụ sau:
(i) một tập hợp tài liệu,
mỗi tài liệu được biểu diễn bằng dãy từ riêng,
và mỗi tài liệu có độ dài $T_i$ riêng;
(ii) biểu diễn dạng dãy của
các đợt điều trị nội trú của bệnh nhân,
trong đó mỗi đợt điều trị bao gồm một số sự kiện
và độ dài dãy phụ thuộc gần đúng
vào thời gian điều trị.


Trước đây, khi xử lý các đầu vào riêng lẻ,
chúng ta giả định rằng chúng được lấy mẫu độc lập
từ cùng một phân phối nền tảng $P(X)$.
Mặc dù chúng ta vẫn giả định rằng toàn bộ các dãy
(ví dụ, toàn bộ tài liệu hoặc lộ trình của bệnh nhân)
được lấy mẫu độc lập, chúng ta không thể giả định
rằng dữ liệu đến tại mỗi bước thời gian
độc lập với nhau.
Ví dụ, các từ có khả năng xuất hiện về sau trong một tài liệu
phụ thuộc rất nhiều vào các từ xuất hiện trước đó trong tài liệu.
Loại thuốc mà một bệnh nhân có khả năng nhận
vào ngày thứ 10 của một đợt điều trị nội trú
phụ thuộc rất nhiều vào những gì đã xảy ra
trong chín ngày trước đó.

Điều này không có gì đáng ngạc nhiên.
Nếu chúng ta không tin rằng các phần tử trong một dãy có liên hệ với nhau,
ngay từ đầu chúng ta đã không cần mô hình hóa chúng thành một dãy.
Hãy xem xét tính hữu ích của các tính năng tự động điền
phổ biến trong các công cụ tìm kiếm và ứng dụng email hiện đại.
Chúng hữu ích chính bởi vì thường có thể
dự đoán (không hoàn hảo, nhưng tốt hơn việc đoán ngẫu nhiên)
các phần tiếp theo có khả năng xảy ra của một dãy,
khi biết một tiền tố ban đầu.
Đối với hầu hết các mô hình dãy,
chúng ta không yêu cầu tính độc lập,
hay thậm chí tính dừng, của các dãy.
Thay vào đó, chúng ta chỉ yêu cầu rằng
bản thân các dãy được lấy mẫu
từ một phân phối nền tảng cố định nào đó
trên toàn bộ các dãy.

Cách tiếp cận linh hoạt này cho phép các hiện tượng như
(i) tài liệu có phần đầu và phần cuối khác biệt đáng kể;
hoặc (ii) trạng thái bệnh nhân diễn biến
theo hướng hồi phục hoặc tử vong
trong suốt thời gian điều trị nội trú;
hoặc (iii) sở thích của khách hàng biến đổi theo những cách có thể dự đoán
trong quá trình tiếp tục tương tác với một hệ thống gợi ý.


Đôi khi chúng ta muốn dự đoán một mục tiêu cố định $y$
khi biết đầu vào có cấu trúc tuần tự
(ví dụ: phân loại cảm xúc dựa trên một bài đánh giá phim).
Vào những lúc khác, chúng ta muốn dự đoán một mục tiêu có cấu trúc tuần tự
($y_1, \ldots, y_T$)
khi biết một đầu vào cố định (ví dụ: tạo chú thích ảnh).
Trong những trường hợp khác nữa, mục tiêu của chúng ta là dự đoán các mục tiêu có cấu trúc tuần tự
dựa trên các đầu vào có cấu trúc tuần tự
(ví dụ: dịch máy hoặc tạo chú thích video).
Các tác vụ sequence-to-sequence như vậy có hai dạng:
(i) *căn chỉnh*: đầu vào ở mỗi bước thời gian
căn chỉnh với một mục tiêu tương ứng (ví dụ: gán nhãn từ loại);
(ii) *không căn chỉnh*: đầu vào và mục tiêu
không nhất thiết có sự tương ứng từng bước
(ví dụ: dịch máy).

Trước khi quan tâm đến việc xử lý các mục tiêu thuộc bất kỳ loại nào,
chúng ta có thể giải quyết bài toán đơn giản nhất:
mô hình hóa mật độ không giám sát (còn gọi là *mô hình hóa dãy*).
Ở đây, với một tập hợp các dãy,
mục tiêu của chúng ta là ước lượng hàm khối xác suất
cho biết khả năng chúng ta quan sát được bất kỳ dãy nào,
tức là $p(\mathbf{x}_1, \ldots, \mathbf{x}_T)$.

```{.python .input  n=6}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input  n=7}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=8}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=9}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input  n=9}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
```

## Mô hình Tự hồi quy


Trước khi giới thiệu các mạng nơ-ron chuyên biệt
được thiết kế để xử lý dữ liệu có cấu trúc tuần tự,
hãy xem xét một số dữ liệu dãy thực tế
và xây dựng một vài trực giác cùng các công cụ thống kê cơ bản.
Cụ thể, chúng ta sẽ tập trung vào dữ liệu giá cổ phiếu
từ chỉ số FTSE 100 (:numref:`fig_ftse100`).
Tại mỗi *bước thời gian* $t \in \mathbb{Z}^+$, chúng ta quan sát
giá, $x_t$, của chỉ số tại thời điểm đó.


![Chỉ số FTSE 100 trong khoảng 30 năm.](../img/ftse100.png)
:width:`400px`
:label:`fig_ftse100`


Bây giờ, giả sử một nhà giao dịch muốn thực hiện các giao dịch ngắn hạn,
vào hoặc thoát khỏi chỉ số một cách chiến lược,
tùy theo việc họ tin rằng
chỉ số sẽ tăng hay giảm
ở bước thời gian tiếp theo.
Khi không có bất kỳ đặc trưng nào khác
(tin tức, dữ liệu báo cáo tài chính, v.v.),
tín hiệu duy nhất có sẵn để dự đoán
giá trị tiếp theo là lịch sử giá cho đến hiện tại.
Do đó, nhà giao dịch quan tâm đến phân phối xác suất

$$P(x_t \mid x_{t-1}, \ldots, x_1)$$

của các mức giá mà chỉ số có thể nhận
ở bước thời gian tiếp theo.
Mặc dù việc ước lượng toàn bộ phân phối
trên một biến ngẫu nhiên có giá trị liên tục
có thể khó khăn, nhà giao dịch sẽ hài lòng
khi tập trung vào một vài thống kê chính của phân phối,
đặc biệt là giá trị kỳ vọng và phương sai.
Một chiến lược đơn giản để ước lượng kỳ vọng có điều kiện

$$\mathbb{E}[(x_t \mid x_{t-1}, \ldots, x_1)],$$

là áp dụng một mô hình hồi quy tuyến tính
(xem lại :numref:`sec_linear_regression`).
Những mô hình hồi quy giá trị của một tín hiệu
theo các giá trị trước đó của chính tín hiệu đó
được gọi một cách tự nhiên là *mô hình tự hồi quy*.
Chỉ có một vấn đề lớn: số lượng đầu vào,
$x_{t-1}, \ldots, x_1$ thay đổi tùy thuộc vào $t$.
Nói cách khác, số lượng đầu vào tăng lên
theo lượng dữ liệu mà chúng ta gặp phải.
Do đó, nếu muốn coi dữ liệu lịch sử
là một tập huấn luyện, chúng ta gặp phải vấn đề
rằng mỗi mẫu có một số lượng đặc trưng khác nhau.
Phần lớn nội dung tiếp theo trong chương này
sẽ xoay quanh các kỹ thuật
để vượt qua những thách thức này
khi xử lý các bài toán mô hình hóa *tự hồi quy* như vậy,
trong đó đối tượng cần quan tâm là
$P(x_t \mid x_{t-1}, \ldots, x_1)$
hoặc một hay nhiều thống kê của phân phối này.

Một vài chiến lược thường xuyên xuất hiện.
Trước hết,
chúng ta có thể tin rằng mặc dù các dãy dài
$x_{t-1}, \ldots, x_1$ có sẵn,
có thể không cần thiết
phải nhìn quá xa vào lịch sử
khi dự đoán tương lai gần.
Trong trường hợp này, chúng ta có thể chỉ cần
điều kiện hóa trên một cửa sổ có độ dài $\tau$
và chỉ sử dụng các quan sát $x_{t-1}, \ldots, x_{t-\tau}$.
Lợi ích tức thì là giờ đây số lượng đối số
luôn giống nhau, ít nhất đối với $t > \tau$.
Điều này cho phép chúng ta huấn luyện bất kỳ mô hình tuyến tính hoặc mạng sâu nào
yêu cầu các vector có độ dài cố định làm đầu vào.
Thứ hai, chúng ta có thể phát triển các mô hình duy trì
một bản tóm tắt $h_t$ nào đó của các quan sát trong quá khứ
(xem :numref:`fig_sequence-model`)
và đồng thời cập nhật $h_t$
bên cạnh dự đoán $\hat{x}_t$.
Điều này dẫn đến các mô hình không chỉ ước lượng $x_t$
bằng $\hat{x}_t = P(x_t \mid h_{t})$
mà còn cập nhật theo dạng
$h_t = g(h_{t-1}, x_{t-1})$.
Vì $h_t$ không bao giờ được quan sát,
các mô hình này còn được gọi là
*mô hình tự hồi quy tiềm ẩn*.

![Một mô hình tự hồi quy tiềm ẩn.](../img/sequence-model.svg)
:label:`fig_sequence-model`

Để xây dựng dữ liệu huấn luyện từ dữ liệu lịch sử, người ta
thường tạo các mẫu bằng cách lấy mẫu ngẫu nhiên các cửa sổ.
Nói chung, chúng ta không kỳ vọng thời gian đứng yên.
Tuy nhiên, chúng ta thường giả định rằng mặc dù
các giá trị cụ thể của $x_t$ có thể thay đổi,
động lực theo đó mỗi quan sát tiếp theo
được tạo ra khi biết các quan sát trước đó thì không thay đổi.
Các nhà thống kê gọi các động lực không thay đổi là *dừng*.



## Mô hình Dãy

Đôi khi, đặc biệt khi làm việc với ngôn ngữ,
chúng ta muốn ước lượng xác suất đồng thời
của toàn bộ một dãy.
Đây là một tác vụ phổ biến khi làm việc với các dãy
được cấu thành từ các *token* rời rạc, chẳng hạn như từ.
Thông thường, các hàm được ước lượng này được gọi là *mô hình dãy*
và đối với dữ liệu ngôn ngữ tự nhiên, chúng được gọi là *mô hình ngôn ngữ*.
Lĩnh vực mô hình hóa dãy được thúc đẩy mạnh mẽ bởi xử lý ngôn ngữ tự nhiên,
đến mức chúng ta thường mô tả các mô hình dãy là "mô hình ngôn ngữ",
ngay cả khi xử lý dữ liệu không phải ngôn ngữ.
Mô hình ngôn ngữ hữu ích vì đủ loại lý do.
Đôi khi chúng ta muốn đánh giá khả năng xảy ra của các câu.
Ví dụ, chúng ta có thể muốn so sánh
tính tự nhiên của hai đầu ra ứng viên
được tạo bởi một hệ thống dịch máy
hoặc bởi một hệ thống nhận dạng giọng nói.
Nhưng mô hình hóa ngôn ngữ không chỉ cho chúng ta
khả năng *đánh giá* likelihood,
mà còn khả năng *lấy mẫu* các dãy,
và thậm chí tối ưu hóa để tìm ra các dãy có khả năng xảy ra cao nhất.

Mặc dù thoạt nhìn mô hình hóa ngôn ngữ có thể không giống
một bài toán tự hồi quy,
chúng ta có thể quy mô hình hóa ngôn ngữ về dự đoán tự hồi quy
bằng cách phân rã mật độ đồng thời của một dãy $p(x_1, \ldots, x_T)$
thành tích của các mật độ có điều kiện
theo hướng từ trái sang phải
bằng cách áp dụng quy tắc chuỗi của xác suất:

$$P(x_1, \ldots, x_T) = P(x_1) \prod_{t=2}^T P(x_t \mid x_{t-1}, \ldots, x_1).$$

Lưu ý rằng nếu chúng ta làm việc với các tín hiệu rời rạc như từ,
thì mô hình tự hồi quy phải là một bộ phân loại xác suất,
xuất ra một phân phối xác suất đầy đủ
trên từ vựng cho bất kỳ từ nào sẽ xuất hiện tiếp theo,
với ngữ cảnh phía bên trái đã cho.



### Mô hình Markov
:label:`subsec_markov-models`


Bây giờ, giả sử rằng chúng ta muốn sử dụng chiến lược đã đề cập ở trên,
trong đó chúng ta chỉ điều kiện hóa trên $\tau$ bước thời gian trước đó,
tức là $x_{t-1}, \ldots, x_{t-\tau}$, thay vì
toàn bộ lịch sử chuỗi $x_{t-1}, \ldots, x_1$.
Bất cứ khi nào chúng ta có thể bỏ đi lịch sử
ngoài $\tau$ bước trước đó
mà không làm mất bất kỳ khả năng dự đoán nào,
chúng ta nói rằng chuỗi thỏa mãn một *điều kiện Markov*,
tức là *tương lai độc lập có điều kiện với quá khứ,
khi biết lịch sử gần đây*.
Khi $\tau = 1$, chúng ta nói dữ liệu được đặc trưng
bởi một *mô hình Markov bậc một*,
và khi $\tau = k$, chúng ta nói dữ liệu được đặc trưng
bởi một mô hình Markov bậc $k^{\textrm{th}}$.
Khi điều kiện Markov bậc một được thỏa mãn ($\tau = 1$),
phân tích nhân tử của xác suất đồng thời trở thành tích
của xác suất mỗi từ khi biết *từ* trước đó:

$$P(x_1, \ldots, x_T) = P(x_1) \prod_{t=2}^T P(x_t \mid x_{t-1}).$$

Chúng ta thường thấy hữu ích khi làm việc với các mô hình
hoạt động như thể một điều kiện Markov được thỏa mãn,
ngay cả khi chúng ta biết điều này chỉ đúng *xấp xỉ*.
Với các tài liệu văn bản thực tế, chúng ta vẫn tiếp tục thu được thông tin
khi đưa ngày càng nhiều ngữ cảnh phía bên trái vào.
Nhưng những lợi ích này giảm nhanh chóng.
Do đó, đôi khi chúng ta thỏa hiệp, tránh các khó khăn về tính toán và thống kê
bằng cách huấn luyện các mô hình mà tính hợp lệ của chúng phụ thuộc
vào điều kiện Markov bậc $k^{\textrm{th}}$.
Ngay cả các mô hình ngôn ngữ khổng lồ dựa trên RNN và Transformer hiện nay
cũng hiếm khi kết hợp quá vài nghìn từ ngữ cảnh.


Với dữ liệu rời rạc, một mô hình Markov thực thụ
chỉ đơn giản đếm số lần
mỗi từ xuất hiện trong từng ngữ cảnh, từ đó tạo ra
ước lượng tần suất tương đối của $P(x_t \mid x_{t-1})$.
Bất cứ khi nào dữ liệu chỉ nhận các giá trị rời rạc
(như trong ngôn ngữ),
chuỗi từ có xác suất cao nhất có thể được tính hiệu quả
bằng dynamic programming.


### Thứ tự giải mã

Bạn có thể đang tự hỏi tại sao chúng ta biểu diễn
phân tích nhân tử của một chuỗi văn bản $P(x_1, \ldots, x_T)$
dưới dạng một chuỗi xác suất có điều kiện từ trái sang phải.
Tại sao không từ phải sang trái hoặc theo một thứ tự ngẫu nhiên có vẻ khác?
Về nguyên tắc, không có gì sai khi khai triển
$P(x_1, \ldots, x_T)$ theo thứ tự ngược lại.
Kết quả là một phân tích nhân tử hợp lệ:

$$P(x_1, \ldots, x_T) = P(x_T) \prod_{t=T-1}^1 P(x_t \mid x_{t+1}, \ldots, x_T).$$


Tuy nhiên, có nhiều lý do khiến việc phân tích nhân tử văn bản
theo cùng hướng mà chúng ta đọc nó
(từ trái sang phải đối với hầu hết ngôn ngữ,
nhưng từ phải sang trái đối với tiếng Ả Rập và tiếng Do Thái)
được ưu tiên cho nhiệm vụ mô hình hóa ngôn ngữ.
Trước hết, đây đơn giản là hướng tự nhiên hơn để chúng ta suy nghĩ.
Dù sao, tất cả chúng ta đều đọc văn bản mỗi ngày,
và quá trình này được dẫn dắt bởi khả năng
dự đoán những từ và cụm từ nào
có khả năng xuất hiện tiếp theo.
Hãy nghĩ xem bạn đã bao nhiêu lần hoàn thành
câu nói của người khác.
Vì vậy, ngay cả khi chúng ta không có lý do nào khác để ưu tiên các cách giải mã theo thứ tự như vậy,
chúng vẫn hữu ích chỉ vì chúng ta có trực giác tốt hơn
về điều gì có khả năng xảy ra khi dự đoán theo thứ tự này.

Thứ hai, bằng cách phân tích nhân tử theo thứ tự,
chúng ta có thể gán xác suất cho các chuỗi dài tùy ý
bằng cùng một mô hình ngôn ngữ.
Để chuyển một xác suất trên các bước từ $1$ đến $t$
thành một xác suất mở rộng đến từ $t+1$, chúng ta chỉ cần
nhân với xác suất có điều kiện
của token bổ sung khi biết các token trước đó:
$P(x_{t+1}, \ldots, x_1) = P(x_{t}, \ldots, x_1) \cdot P(x_{t+1} \mid x_{t}, \ldots, x_1)$.

Thứ ba, chúng ta có các mô hình dự đoán mạnh hơn
để dự đoán các từ liền kề so với
các từ ở những vị trí khác tùy ý.
Mặc dù mọi thứ tự phân tích nhân tử đều hợp lệ,
chúng không nhất thiết đều biểu diễn các bài toán
mô hình hóa dự đoán dễ như nhau.
Điều này không chỉ đúng với ngôn ngữ,
mà còn với các loại dữ liệu khác,
ví dụ, khi dữ liệu có cấu trúc nhân quả.
Chẳng hạn, chúng ta tin rằng các sự kiện trong tương lai không thể ảnh hưởng đến quá khứ.
Do đó, nếu thay đổi $x_t$, chúng ta có thể tác động
đến những gì xảy ra với $x_{t+1}$ về sau nhưng không thể theo chiều ngược lại.
Tức là, nếu thay đổi $x_t$, phân phối trên các sự kiện trong quá khứ sẽ không thay đổi.
Trong một số ngữ cảnh, điều này giúp dự đoán $P(x_{t+1} \mid x_t)$
dễ hơn so với dự đoán $P(x_t \mid x_{t+1})$.
Ví dụ, trong một số trường hợp, chúng ta có thể tìm được $x_{t+1} = f(x_t) + \epsilon$
với một nhiễu cộng $\epsilon$,
trong khi điều ngược lại không đúng :cite:`Hoyer.Janzing.Mooij.ea.2009`.
Đây là tin tốt, vì thông thường hướng thuận
là hướng mà chúng ta quan tâm đến việc ước lượng.
Cuốn sách của :citet:`Peters.Janzing.Scholkopf.2017` có thêm nội dung về chủ đề này.
Chúng ta mới chỉ đề cập sơ lược về nó.


## Huấn luyện

Trước khi tập trung vào dữ liệu văn bản,
trước tiên hãy thử điều này với một số
dữ liệu tổng hợp có giá trị liên tục.

(**Ở đây, 1000 dữ liệu tổng hợp của chúng ta sẽ tuân theo
hàm lượng giác `sin`,
được áp dụng cho 0.01 lần bước thời gian.
Để bài toán thú vị hơn một chút,
chúng ta làm nhiễu mỗi mẫu bằng nhiễu cộng.**)
Từ chuỗi này, chúng ta trích xuất các ví dụ huấn luyện,
mỗi ví dụ gồm đặc trưng và nhãn.

```{.python .input  n=10}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, batch_size=16, T=1000, num_train=600, tau=4):
        self.save_hyperparameters()
        self.time = d2l.arange(1, T + 1, dtype=d2l.float32)
        if tab.selected('mxnet', 'pytorch'):
            self.x = d2l.sin(0.01 * self.time) + d2l.randn(T) * 0.2
        if tab.selected('tensorflow'):
            self.x = d2l.sin(0.01 * self.time) + d2l.normal([T]) * 0.2
        if tab.selected('jax'):
            key = d2l.get_key()
            self.x = d2l.sin(0.01 * self.time) + jax.random.normal(key,
                                                                   [T]) * 0.2
```

```{.python .input}
%%tab all
data = Data()
d2l.plot(data.time, data.x, 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

Để bắt đầu, chúng ta thử một mô hình hoạt động như thể
dữ liệu thỏa mãn điều kiện Markov bậc $\tau^{\textrm{th}}$,
và do đó dự đoán $x_t$ chỉ bằng $\tau$ quan sát trong quá khứ.
[**Vì vậy, với mỗi bước thời gian, chúng ta có một ví dụ
với nhãn $y  = x_t$ và các đặc trưng
$\mathbf{x}_t = [x_{t-\tau}, \ldots, x_{t-1}]$.**]
Người đọc tinh ý có thể nhận ra rằng
điều này tạo ra $1000-\tau$ ví dụ,
vì chúng ta không có đủ lịch sử cho $y_1, \ldots, y_\tau$.
Mặc dù chúng ta có thể đệm các chuỗi $\tau$ đầu tiên bằng số không,
để đơn giản, hiện tại chúng ta loại bỏ chúng.
Dataset kết quả chứa $T - \tau$ ví dụ,
trong đó mỗi đầu vào của mô hình có độ dài chuỗi là $\tau$.
Chúng ta (**tạo một data iterator trên 600 ví dụ đầu tiên**),
bao phủ một giai đoạn của hàm sin.

```{.python .input}
%%tab all
@d2l.add_to_class(Data)
def get_dataloader(self, train):
    features = [self.x[i : self.T-self.tau+i] for i in range(self.tau)]
    self.features = d2l.stack(features, 1)
    self.labels = d2l.reshape(self.x[self.tau:], (-1, 1))
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader([self.features, self.labels], train, i)
```

Trong ví dụ này, mô hình của chúng ta sẽ là linear regression tiêu chuẩn.

```{.python .input}
%%tab all
model = d2l.LinearRegression(lr=0.01)
trainer = d2l.Trainer(max_epochs=5)
trainer.fit(model, data)
```

## Dự đoán

[**Để đánh giá mô hình, trước tiên chúng ta kiểm tra
mô hình hoạt động tốt đến đâu trong dự đoán trước một bước**].

```{.python .input}
%%tab pytorch, mxnet, tensorflow
onestep_preds = d2l.numpy(model(data.features))
d2l.plot(data.time[data.tau:], [data.labels, onestep_preds], 'time', 'x',
         legend=['labels', '1-step preds'], figsize=(6, 3))
```

```{.python .input}
%%tab jax
onestep_preds = model.apply({'params': trainer.state.params}, data.features)
d2l.plot(data.time[data.tau:], [data.labels, onestep_preds], 'time', 'x',
         legend=['labels', '1-step preds'], figsize=(6, 3))
```

Các dự đoán này trông tốt,
ngay cả gần cuối tại $t=1000$.

Nhưng điều gì xảy ra nếu chúng ta chỉ quan sát dữ liệu chuỗi
cho đến bước thời gian 604 (`n_train + tau`)
và muốn dự đoán nhiều bước
trong tương lai?
Thật không may, chúng ta không thể trực tiếp tính
dự đoán trước một bước cho bước thời gian 609,
vì chúng ta không biết các đầu vào tương ứng,
do chỉ quan sát đến $x_{604}$.
Chúng ta có thể giải quyết vấn đề này bằng cách đưa vào
các dự đoán trước đó của mình làm đầu vào cho mô hình
để tạo các dự đoán tiếp theo,
chiếu về phía trước, từng bước một,
cho đến khi đạt bước thời gian mong muốn:

$$\begin{aligned}
\hat{x}_{605} &= f(x_{601}, x_{602}, x_{603}, x_{604}), \\
\hat{x}_{606} &= f(x_{602}, x_{603}, x_{604}, \hat{x}_{605}), \\
\hat{x}_{607} &= f(x_{603}, x_{604}, \hat{x}_{605}, \hat{x}_{606}),\\
\hat{x}_{608} &= f(x_{604}, \hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}),\\
\hat{x}_{609} &= f(\hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}, \hat{x}_{608}),\\
&\vdots\end{aligned}$$

Nói chung, với một chuỗi đã quan sát $x_1, \ldots, x_t$,
đầu ra dự đoán $\hat{x}_{t+k}$ của nó tại bước thời gian $t+k$
được gọi là *dự đoán trước $k$ bước*.
Vì chúng ta đã quan sát đến $x_{604}$,
dự đoán trước $k$ bước của nó là $\hat{x}_{604+k}$.
Nói cách khác, chúng ta sẽ phải
tiếp tục sử dụng các dự đoán của chính mình
để tạo các dự đoán nhiều bước phía trước.
Hãy xem điều này diễn ra tốt đến đâu.

```{.python .input}
%%tab mxnet, pytorch
multistep_preds = d2l.zeros(data.T)
multistep_preds[:] = data.x
for i in range(data.num_train + data.tau, data.T):
    multistep_preds[i] = model(
        d2l.reshape(multistep_preds[i-data.tau : i], (1, -1)))
multistep_preds = d2l.numpy(multistep_preds)
```

```{.python .input}
%%tab tensorflow
multistep_preds = tf.Variable(d2l.zeros(data.T))
multistep_preds[:].assign(data.x)
for i in range(data.num_train + data.tau, data.T):
    multistep_preds[i].assign(d2l.reshape(model(
        d2l.reshape(multistep_preds[i-data.tau : i], (1, -1))), ()))
```

```{.python .input}
%%tab jax
multistep_preds = d2l.zeros(data.T)
multistep_preds = multistep_preds.at[:].set(data.x)
for i in range(data.num_train + data.tau, data.T):
    pred = model.apply({'params': trainer.state.params},
                       d2l.reshape(multistep_preds[i-data.tau : i], (1, -1)))
    multistep_preds = multistep_preds.at[i].set(pred.item())
```

```{.python .input}
%%tab all
d2l.plot([data.time[data.tau:], data.time[data.num_train+data.tau:]],
         [onestep_preds, multistep_preds[data.num_train+data.tau:]], 'time',
         'x', legend=['1-step preds', 'multistep preds'], figsize=(6, 3))
```

Thật không may, trong trường hợp này chúng ta thất bại thảm hại.
Các dự đoán nhanh chóng suy giảm về một hằng số
sau vài bước.
Tại sao thuật toán lại hoạt động kém hơn nhiều
khi dự đoán xa hơn trong tương lai?
Cuối cùng, nguyên nhân là do
lỗi tích lũy.
Giả sử sau bước 1 chúng ta có một lỗi $\epsilon_1 = \bar\epsilon$.
Bây giờ *đầu vào* cho bước 2 bị nhiễu bởi $\epsilon_1$,
do đó chúng ta gặp một lỗi cỡ
$\epsilon_2 = \bar\epsilon + c \epsilon_1$
với một hằng số $c$, và cứ tiếp tục như vậy.
Các dự đoán có thể nhanh chóng lệch xa
khỏi các quan sát thực.
Có thể bạn đã quen thuộc
với hiện tượng phổ biến này.
Chẳng hạn, dự báo thời tiết cho 24 giờ tiếp theo
thường khá chính xác, nhưng sau đó
độ chính xác giảm nhanh chóng.
Chúng ta sẽ thảo luận các phương pháp để cải thiện điều này
trong toàn bộ chương này và các chương sau.

Hãy [**xem xét kỹ hơn những khó khăn trong dự đoán trước $k$ bước**]
bằng cách tính dự đoán trên toàn bộ chuỗi với $k = 1, 4, 16, 64$.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
def k_step_pred(k):
    features = []
    for i in range(data.tau):
        features.append(data.x[i : i+data.T-data.tau-k+1])
    # The (i+tau)-th element stores the (i+1)-step-ahead predictions
    for i in range(k):
        preds = model(d2l.stack(features[i : i+data.tau], 1))
        features.append(d2l.reshape(preds, -1))
    return features[data.tau:]
```

```{.python .input}
%%tab jax
def k_step_pred(k):
    features = []
    for i in range(data.tau):
        features.append(data.x[i : i+data.T-data.tau-k+1])
    # The (i+tau)-th element stores the (i+1)-step-ahead predictions
    for i in range(k):
        preds = model.apply({'params': trainer.state.params},
                            d2l.stack(features[i : i+data.tau], 1))
        features.append(d2l.reshape(preds, -1))
    return features[data.tau:]
```

```{.python .input}
%%tab all
steps = (1, 4, 16, 64)
preds = k_step_pred(steps[-1])
d2l.plot(data.time[data.tau+steps[-1]-1:],
         [d2l.numpy(preds[k-1]) for k in steps], 'time', 'x',
         legend=[f'{k}-step preds' for k in steps], figsize=(6, 3))
```

Điều này minh họa rõ ràng cách chất lượng dự đoán thay đổi
khi chúng ta cố dự đoán xa hơn vào tương lai.
Mặc dù các dự đoán trước 4 bước vẫn có vẻ tốt,
mọi dự đoán xa hơn thế gần như vô dụng.

## Tóm tắt

Có sự khác biệt đáng kể về độ khó
giữa nội suy và ngoại suy.
Do đó, nếu bạn có một chuỗi, hãy luôn tôn trọng
thứ tự thời gian của dữ liệu khi huấn luyện,
tức là không bao giờ huấn luyện trên dữ liệu tương lai.
Với loại dữ liệu này,
các mô hình chuỗi cần những công cụ thống kê chuyên biệt để ước lượng.
Hai lựa chọn phổ biến là mô hình tự hồi quy
và mô hình tự hồi quy biến tiềm ẩn.
Đối với các mô hình nhân quả (ví dụ: thời gian tiến về phía trước),
việc ước lượng theo chiều thuận thường
dễ hơn rất nhiều so với chiều ngược.
Với một chuỗi được quan sát đến bước thời gian $t$,
đầu ra dự đoán của nó tại bước thời gian $t+k$
là *dự đoán trước $k$ bước*.
Khi dự đoán xa hơn theo thời gian bằng cách tăng $k$,
các lỗi sẽ tích lũy và chất lượng dự đoán suy giảm,
thường là rất đáng kể.

## Bài tập

1. Cải thiện mô hình trong thí nghiệm của phần này.
    1. Kết hợp nhiều hơn bốn quan sát trong quá khứ? Bạn thực sự cần bao nhiêu?
    1. Bạn sẽ cần bao nhiêu quan sát trong quá khứ nếu không có nhiễu? Gợi ý: bạn có thể viết $\sin$ và $\cos$ dưới dạng một phương trình vi phân.
    1. Bạn có thể kết hợp các quan sát cũ hơn trong khi giữ tổng số đặc trưng không đổi không? Điều này có cải thiện độ chính xác không? Tại sao?
    1. Thay đổi kiến trúc mạng nơ-ron và đánh giá hiệu năng. Bạn có thể huấn luyện mô hình mới với nhiều epoch hơn. Bạn quan sát được gì?
1. Một nhà đầu tư muốn tìm một chứng khoán tốt để mua.
   Họ xem xét lợi nhuận trong quá khứ để quyết định chứng khoán nào có khả năng hoạt động tốt.
   Điều gì có thể xảy ra sai với chiến lược này?
1. Tính nhân quả có áp dụng cho văn bản không? Ở mức độ nào?
1. Hãy đưa ra một ví dụ về trường hợp cần một mô hình tự hồi quy biến tiềm ẩn
   để nắm bắt động lực của dữ liệu.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/113)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/114)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1048)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18010)
:end_tab: