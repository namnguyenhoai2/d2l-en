```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Xác suất và Thống kê
:label:`sec_prob`

Dù xét theo cách nào,
machine learning đều liên quan đến sự không chắc chắn.
Trong học có giám sát, chúng ta muốn dự đoán
một điều chưa biết (gọi là *target*)
dựa trên một điều đã biết (gọi là *features*).
Tùy thuộc vào mục tiêu,
chúng ta có thể cố gắng dự đoán
giá trị có khả năng xảy ra nhất của target.
Hoặc chúng ta có thể dự đoán giá trị có
khoảng cách kỳ vọng nhỏ nhất so với target.
Và đôi khi, chúng ta không chỉ muốn
dự đoán một giá trị cụ thể
mà còn muốn *định lượng mức độ không chắc chắn*.
Ví dụ, với một số features
mô tả một bệnh nhân,
chúng ta có thể muốn biết *khả năng*
người đó bị đau tim trong năm tới là bao nhiêu.
Trong học không giám sát,
chúng ta thường quan tâm đến sự không chắc chắn.
Để xác định một tập hợp các phép đo có bất thường hay không,
việc biết khả năng quan sát được các giá trị
trong một quần thể đáng quan tâm là bao nhiêu
sẽ rất hữu ích.
Hơn nữa, trong học tăng cường,
chúng ta muốn phát triển các agent
có thể hành động thông minh trong nhiều môi trường khác nhau.
Điều này đòi hỏi phải suy luận về
cách một môi trường có thể được kỳ vọng là sẽ thay đổi
và những phần thưởng nào có thể được kỳ vọng nhận được
để đáp lại mỗi hành động hiện có.

*Xác suất* là lĩnh vực toán học
liên quan đến việc suy luận trong điều kiện không chắc chắn.
Với một mô hình xác suất của một quá trình nào đó,
chúng ta có thể suy luận về khả năng xảy ra của nhiều sự kiện khác nhau.
Việc sử dụng xác suất để mô tả
tần suất của các sự kiện có thể lặp lại
(như tung đồng xu)
khá ít gây tranh cãi.
Trên thực tế, các học giả theo trường phái *frequentist*
tuân theo cách diễn giải xác suất
chỉ áp dụng *cho* những sự kiện có thể lặp lại như vậy.
Ngược lại, các học giả theo trường phái *Bayesian*
sử dụng ngôn ngữ xác suất rộng hơn
để hình thức hóa việc suy luận trong điều kiện không chắc chắn.
Xác suất Bayesian được đặc trưng
bởi hai tính chất riêng:
(i) gán mức độ tin tưởng
cho những sự kiện không thể lặp lại,
ví dụ, *xác suất*
một con đập bị vỡ là bao nhiêu?;
và (ii) tính chủ quan. Mặc dù xác suất Bayesian
đưa ra các quy tắc rõ ràng
về cách một người nên cập nhật niềm tin của mình
dựa trên bằng chứng mới,
nó cho phép những cá nhân khác nhau
bắt đầu với những niềm tin *prior* khác nhau.
*Thống kê* giúp chúng ta suy luận ngược,
bắt đầu từ việc thu thập và tổ chức dữ liệu
rồi suy ra những kết luận
mà chúng ta có thể rút ra về quá trình
đã tạo ra dữ liệu đó.
Bất cứ khi nào phân tích một dataset, tìm kiếm các mẫu
mà chúng ta hy vọng có thể đặc trưng cho một quần thể rộng hơn,
chúng ta đang sử dụng tư duy thống kê.
Nhiều khóa học, chuyên ngành, luận văn, nghề nghiệp,
khoa, công ty và tổ chức đã được dành riêng
cho việc nghiên cứu xác suất và thống kê.
Mặc dù phần này chỉ mới tìm hiểu sơ lược,
chúng ta sẽ cung cấp nền tảng
cần thiết để bắt đầu xây dựng các mô hình.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.numpy.random import multinomial
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import random
import torch
from torch.distributions.multinomial import Multinomial
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import random
import tensorflow as tf
from tensorflow_probability import distributions as tfd
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import random
import jax
from jax import numpy as jnp
import numpy as np
```

## Một ví dụ đơn giản: Tung đồng xu

Hãy tưởng tượng chúng ta dự định tung một đồng xu
và muốn định lượng khả năng
mặt ngửa xuất hiện (so với mặt sấp).
Nếu đồng xu là *công bằng*,
thì cả hai kết quả
(mặt ngửa và mặt sấp)
đều có khả năng xảy ra như nhau.
Hơn nữa, nếu dự định tung đồng xu $n$ lần
thì tỷ lệ mặt ngửa
mà chúng ta *kỳ vọng* nhìn thấy
sẽ khớp chính xác
với *tỷ lệ* mặt sấp kỳ vọng.
Một cách trực quan để thấy điều này
là dựa trên tính đối xứng:
với mọi kết quả có thể xảy ra
gồm $n_\textrm{h}$ mặt ngửa và $n_\textrm{t} = (n - n_\textrm{h})$ mặt sấp,
sẽ có một kết quả có khả năng xảy ra tương đương
gồm $n_\textrm{t}$ mặt ngửa và $n_\textrm{h}$ mặt sấp.
Lưu ý rằng điều này chỉ có thể xảy ra
nếu trung bình chúng ta kỳ vọng
$1/2$ số lần tung cho kết quả mặt ngửa
và $1/2$ cho kết quả mặt sấp.
Tất nhiên, nếu tiến hành thí nghiệm này
nhiều lần với mỗi lần tung $n=1000000$ lần,
có thể bạn sẽ không bao giờ thấy một lần thử
mà $n_\textrm{h} = n_\textrm{t}$ một cách chính xác.


Về mặt hình thức, đại lượng $1/2$ được gọi là một *xác suất*
và ở đây nó thể hiện mức độ chắc chắn rằng
bất kỳ lần tung nào cũng cho kết quả mặt ngửa.
Xác suất gán các điểm số từ $0$ đến $1$
cho những kết quả quan tâm, được gọi là *sự kiện*.
Ở đây, sự kiện quan tâm là $\textrm{heads}$
và chúng ta ký hiệu xác suất tương ứng là $P(\textrm{heads})$.
Xác suất bằng $1$ biểu thị sự chắc chắn tuyệt đối
(hãy hình dung một đồng xu đặc biệt mà cả hai mặt đều là mặt ngửa)
và xác suất bằng $0$ biểu thị điều không thể xảy ra
(ví dụ, nếu cả hai mặt đều là mặt sấp).
Các tần suất $n_\textrm{h}/n$ và $n_\textrm{t}/n$ không phải là xác suất
mà là *các thống kê*.
Xác suất là những đại lượng *lý thuyết*
làm nền tảng cho quá trình tạo dữ liệu.
Ở đây, xác suất $1/2$
là một thuộc tính của chính đồng xu.
Ngược lại, thống kê là những đại lượng *thực nghiệm*
được tính dưới dạng các hàm của dữ liệu đã quan sát.
Mối quan tâm của chúng ta đối với các đại lượng xác suất và thống kê
gắn kết chặt chẽ với nhau.
Chúng ta thường thiết kế những thống kê đặc biệt gọi là *estimator*
mà khi được cung cấp một dataset sẽ tạo ra các *estimate*
của các tham số mô hình như xác suất.
Hơn nữa, khi các estimator đó thỏa mãn
một tính chất tốt gọi là *tính nhất quán*,
các estimate của chúng ta sẽ hội tụ
về xác suất tương ứng.
Đổi lại, những xác suất được suy ra này
cho biết các tính chất thống kê có khả năng xảy ra
của dữ liệu từ cùng một quần thể
mà chúng ta có thể gặp trong tương lai.

Giả sử chúng ta tình cờ có được một đồng xu thật
mà không biết
$P(\textrm{heads})$ thực sự bằng bao nhiêu.
Để khảo sát đại lượng này
bằng các phương pháp thống kê,
chúng ta cần (i) thu thập một số dữ liệu;
và (ii) thiết kế một estimator.
Việc thu thập dữ liệu ở đây rất đơn giản;
chúng ta có thể tung đồng xu nhiều lần
và ghi lại tất cả các kết quả.
Về mặt hình thức, việc lấy các hiện thực
từ một quá trình ngẫu nhiên nền tảng nào đó
được gọi là *sampling*.
Như bạn có thể đoán,
một estimator tự nhiên
là tỷ số giữa
số lần quan sát được *mặt ngửa*
và tổng số lần tung.

Bây giờ, giả sử đồng xu thực sự công bằng,
tức là $P(\textrm{heads}) = 0.5$.
Để mô phỏng việc tung một đồng xu công bằng,
chúng ta có thể gọi bất kỳ trình tạo số ngẫu nhiên nào.
Có một số cách đơn giản để lấy mẫu
một sự kiện có xác suất $0.5$.
Ví dụ, `random.random` của Python
cho ra các số trong khoảng $[0,1]$
trong đó xác suất nằm
trong bất kỳ khoảng con $[a, b] \subset [0,1]$ nào
bằng $b-a$.
Do đó, chúng ta có thể nhận được `0` và `1` với xác suất `0.5` cho mỗi kết quả
bằng cách kiểm tra xem số thực được trả về có lớn hơn `0.5` hay không:

```{.python .input}
%%tab all
num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])
```

Tổng quát hơn, chúng ta có thể mô phỏng nhiều lần lấy mẫu
từ bất kỳ biến nào có số lượng hữu hạn
các kết quả có thể xảy ra
(như tung đồng xu hoặc gieo xúc xắc)
bằng cách gọi hàm multinomial,
đặt đối số đầu tiên
là số lần lấy mẫu
và đối số thứ hai là một danh sách các xác suất
tương ứng với từng kết quả có thể xảy ra.
Để mô phỏng mười lần tung một đồng xu công bằng,
chúng ta gán vector xác suất `[0.5, 0.5]`,
diễn giải index 0 là mặt ngửa
và index 1 là mặt sấp.
Hàm trả về một vector
có độ dài bằng số lượng
các kết quả có thể xảy ra (ở đây là 2),
trong đó thành phần đầu tiên cho biết
số lần xuất hiện mặt ngửa
và thành phần thứ hai cho biết
số lần xuất hiện mặt sấp.

```{.python .input}
%%tab mxnet
fair_probs = [0.5, 0.5]
multinomial(100, fair_probs)
```

```{.python .input}
%%tab pytorch
fair_probs = torch.tensor([0.5, 0.5])
Multinomial(100, fair_probs).sample()
```

```{.python .input}
%%tab tensorflow
fair_probs = tf.ones(2) / 2
tfd.Multinomial(100, fair_probs).sample()
```

```{.python .input}
%%tab jax
fair_probs = [0.5, 0.5]
# jax.random does not have multinomial distribution implemented
np.random.multinomial(100, fair_probs)
```

Mỗi lần chạy quy trình lấy mẫu này,
bạn sẽ nhận được một giá trị ngẫu nhiên mới
có thể khác với kết quả trước đó.
Chia cho số lần tung
sẽ cho chúng ta *tần suất*
của mỗi kết quả trong dữ liệu.
Lưu ý rằng các tần suất này,
giống như các xác suất
mà chúng được dùng để
ước lượng, có tổng bằng $1$.

```{.python .input}
%%tab mxnet
multinomial(100, fair_probs) / 100
```

```{.python .input}
%%tab pytorch
Multinomial(100, fair_probs).sample() / 100
```

```{.python .input}
%%tab tensorflow
tfd.Multinomial(100, fair_probs).sample() / 100
```

```{.python .input}
%%tab jax
np.random.multinomial(100, fair_probs) / 100
```

Ở đây, mặc dù đồng xu được mô phỏng là công bằng
(chính chúng ta đã đặt các xác suất `[0.5, 0.5]`),
số lần xuất hiện mặt ngửa và mặt sấp có thể không giống nhau.
Đó là vì chúng ta chỉ lấy một số lượng mẫu tương đối nhỏ.
Nếu chúng ta không tự triển khai mô phỏng
mà chỉ nhìn thấy kết quả,
làm thế nào biết được đồng xu hơi không công bằng
hay độ lệch có thể có khỏi $1/2$
chỉ là hệ quả của kích thước mẫu nhỏ?
Hãy xem điều gì xảy ra khi mô phỏng 10.000 lần tung.

```{.python .input}
%%tab mxnet
counts = multinomial(10000, fair_probs).astype(np.float32)
counts / 10000
```

```{.python .input}
%%tab pytorch
counts = Multinomial(10000, fair_probs).sample()
counts / 10000
```

```{.python .input}
%%tab tensorflow
counts = tfd.Multinomial(10000, fair_probs).sample()
counts / 10000
```

```{.python .input}
%%tab jax
counts = np.random.multinomial(10000, fair_probs).astype(np.float32)
counts / 10000
```

Nói chung, đối với các giá trị trung bình của những sự kiện lặp lại (như tung đồng xu),
khi số lần lặp tăng lên,
các estimate của chúng ta được đảm bảo hội tụ
về các xác suất nền tảng thực sự.
Cách diễn đạt toán học của hiện tượng này
được gọi là *luật số lớn*
và *định lý giới hạn trung tâm*
cho chúng ta biết rằng trong nhiều tình huống,
khi kích thước mẫu $n$ tăng,
những sai số này sẽ giảm
với tốc độ $(1/\sqrt{n})$.
Hãy tìm hiểu thêm một chút trực giác bằng cách nghiên cứu
estimate của chúng ta thay đổi như thế nào khi tăng
số lần tung từ 1 lên 10.000.

```{.python .input}
%%tab pytorch
counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()

d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
%%tab mxnet
counts = multinomial(1, fair_probs, size=10000)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)
```

```{.python .input}
%%tab tensorflow
counts = tfd.Multinomial(1, fair_probs).sample(10000)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)
estimates = estimates.numpy()
```

```{.python .input}
%%tab jax
counts = np.random.multinomial(1, fair_probs, size=10000).astype(np.float32)
cum_counts = counts.cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)
```

```{.python .input}
%%tab mxnet, tensorflow, jax
d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

Mỗi đường cong liền nét tương ứng với một trong hai giá trị của đồng xu
và cho biết xác suất ước lượng rằng đồng xu cho ra giá trị đó
sau mỗi nhóm thí nghiệm.
Đường màu đen nét đứt biểu thị xác suất nền tảng thực sự.
Khi có thêm dữ liệu nhờ tiến hành nhiều thí nghiệm hơn,
các đường cong hội tụ về xác suất thực.
Có thể bạn đã bắt đầu nhận ra hình dạng
của một số câu hỏi nâng cao hơn
mà các nhà thống kê thường trăn trở:
Sự hội tụ này diễn ra nhanh đến mức nào?
Nếu chúng ta đã kiểm tra nhiều đồng xu
được sản xuất tại cùng một nhà máy,
chúng ta có thể đưa thông tin này vào như thế nào?

##  Một cách xử lý hình thức hơn

Chúng ta đã tiến khá xa: xây dựng
một mô hình xác suất,
tạo dữ liệu tổng hợp,
chạy một estimator thống kê,
đánh giá thực nghiệm sự hội tụ,
và báo cáo các metrics về sai số (kiểm tra độ lệch).
Tuy nhiên, để tiến xa hơn nữa,
chúng ta sẽ cần chính xác hơn.


Khi xử lý tính ngẫu nhiên,
ta ký hiệu tập hợp các kết quả có thể xảy ra là $\mathcal{S}$
và gọi nó là *không gian mẫu* hoặc *không gian kết quả*.
Ở đây, mỗi phần tử là một *kết quả* có thể xảy ra và phân biệt.
Trong trường hợp tung một đồng xu,
$\mathcal{S} = \{\textrm{heads}, \textrm{tails}\}$.
Đối với một con xúc xắc, $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$.
Khi tung hai đồng xu, các kết quả có thể xảy ra là
$\{(\textrm{heads}, \textrm{heads}), (\textrm{heads}, \textrm{tails}), (\textrm{tails}, \textrm{heads}),  (\textrm{tails}, \textrm{tails})\}$.
*Biến cố* là các tập con của không gian mẫu.
Ví dụ, biến cố “lần tung đồng xu thứ nhất cho mặt ngửa”
tương ứng với tập hợp $\{(\textrm{heads}, \textrm{heads}), (\textrm{heads}, \textrm{tails})\}$.
Bất cứ khi nào kết quả $z$ của một phép thử ngẫu nhiên thỏa mãn
$z \in \mathcal{A}$, thì biến cố $\mathcal{A}$ đã xảy ra.
Với một lần gieo xúc xắc, ta có thể định nghĩa các biến cố
“xuất hiện $5$” ($\mathcal{A} = \{5\}$)
và “xuất hiện một số lẻ”  ($\mathcal{B} = \{1, 3, 5\}$).
Trong trường hợp này, nếu xúc xắc cho kết quả $5$,
ta sẽ nói rằng cả $\mathcal{A}$ và $\mathcal{B}$ đều đã xảy ra.
Mặt khác, nếu $z = 3$,
thì $\mathcal{A}$ đã không xảy ra,
nhưng $\mathcal{B}$ đã xảy ra.


Một hàm *xác suất* ánh xạ các biến cố
sang các giá trị thực ${P: \mathcal{A} \subseteq \mathcal{S} \rightarrow [0,1]}$.
Xác suất, được ký hiệu là $P(\mathcal{A})$, của một biến cố $\mathcal{A}$
trong không gian mẫu đã cho $\mathcal{S}$,
có các tính chất sau:

* Xác suất của mọi biến cố $\mathcal{A}$ là một số thực không âm, tức là $P(\mathcal{A}) \geq 0$;
* Xác suất của toàn bộ không gian mẫu là $1$, tức là $P(\mathcal{S}) = 1$;
* Với mọi dãy đếm được các biến cố $\mathcal{A}_1, \mathcal{A}_2, \ldots$ *loại trừ lẫn nhau* (tức là $\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ với mọi $i \neq j$), xác suất để bất kỳ biến cố nào trong số đó xảy ra bằng tổng các xác suất riêng lẻ của chúng, tức là $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.

Các tiên đề này của lý thuyết xác suất,
do :citet:`Kolmogorov.1933` đề xuất,
có thể được áp dụng để nhanh chóng suy ra nhiều hệ quả quan trọng.
Chẳng hạn, ngay lập tức ta có thể suy ra
rằng xác suất để một biến cố bất kỳ $\mathcal{A}$
*hoặc* phần bù $\mathcal{A}'$ của nó xảy ra là 1
(vì $\mathcal{A} \cup \mathcal{A}' = \mathcal{S}$).
Ta cũng có thể chứng minh rằng $P(\emptyset) = 0$
vì $1 = P(\mathcal{S} \cup \mathcal{S}') = P(\mathcal{S} \cup \emptyset) = P(\mathcal{S}) + P(\emptyset) = 1 + P(\emptyset)$.
Do đó, xác suất để một biến cố bất kỳ $\mathcal{A}$
*và* phần bù $\mathcal{A}'$ của nó đồng thời xảy ra
là $P(\mathcal{A} \cap \mathcal{A}') = 0$.
Nói một cách không chính thức, điều này cho chúng ta biết rằng các biến cố bất khả thi
có xác suất xảy ra bằng không.



## Biến ngẫu nhiên

Khi đề cập đến các biến cố như việc gieo xúc xắc cho ra số lẻ
hoặc lần tung đồng xu thứ nhất cho mặt ngửa,
chúng ta đang vận dụng khái niệm *biến ngẫu nhiên*.
Về mặt hình thức, biến ngẫu nhiên là các ánh xạ
từ một không gian mẫu cơ sở
sang một tập hợp gồm một hoặc nhiều giá trị.
Bạn có thể thắc mắc biến ngẫu nhiên
khác với không gian mẫu như thế nào,
vì cả hai đều là các tập hợp kết quả.
Điểm quan trọng là biến ngẫu nhiên có thể thô hơn nhiều
so với không gian mẫu ban đầu.
Ta có thể định nghĩa một biến ngẫu nhiên nhị phân như “lớn hơn 0.5”
ngay cả khi không gian mẫu cơ sở là vô hạn,
chẳng hạn các điểm trên đoạn thẳng nối từ $0$ đến $1$.
Ngoài ra, nhiều biến ngẫu nhiên
có thể dùng chung một không gian mẫu cơ sở.
Ví dụ, “chuông báo động ở nhà tôi có kêu hay không”
và “nhà tôi có bị trộm hay không”
đều là các biến ngẫu nhiên nhị phân
dùng chung một không gian mẫu cơ sở.
Do đó, biết giá trị của một biến ngẫu nhiên
có thể cho chúng ta biết điều gì đó về giá trị có khả năng xảy ra của một biến ngẫu nhiên khác.
Biết rằng chuông báo động đã kêu,
ta có thể nghi ngờ rằng ngôi nhà có khả năng đã bị trộm.


Mỗi giá trị mà một biến ngẫu nhiên nhận được tương ứng
với một tập con của không gian mẫu cơ sở.
Do đó, biến cố trong đó biến ngẫu nhiên $X$
nhận giá trị $v$, được ký hiệu là $X=v$, là một *biến cố*,
và $P(X=v)$ biểu thị xác suất của nó.
Đôi khi ký hiệu này có thể trở nên cồng kềnh,
và ta có thể lạm dụng ký hiệu khi ngữ cảnh đã rõ ràng.
Ví dụ, ta có thể dùng $P(X)$ để chỉ một cách khái quát
*phân phối* của $X$, tức là
hàm cho biết xác suất
để $X$ nhận một giá trị bất kỳ đã cho.
Trong những trường hợp khác, ta viết các biểu thức
như $P(X,Y) = P(X) P(Y)$,
như một cách viết tắt cho một mệnh đề
đúng với mọi giá trị
mà các biến ngẫu nhiên $X$ và $Y$ có thể nhận, tức là,
với mọi $i,j$, ta có $P(X=i \textrm{ and } Y=j) = P(X=i)P(Y=j)$.
Trong những trường hợp khác, ta lạm dụng ký hiệu bằng cách viết
$P(v)$ khi biến ngẫu nhiên đã rõ từ ngữ cảnh.
Vì một biến cố trong lý thuyết xác suất là một tập hợp các kết quả từ không gian mẫu,
ta có thể chỉ định một khoảng giá trị mà một biến ngẫu nhiên nhận.
Ví dụ, $P(1 \leq X \leq 3)$ biểu thị xác suất của biến cố $\{1 \leq X \leq 3\}$.


Lưu ý rằng có một khác biệt tinh tế
giữa các biến ngẫu nhiên *rời rạc*,
như kết quả tung đồng xu hoặc gieo xúc xắc,
và các biến ngẫu nhiên *liên tục*,
như cân nặng và chiều cao của một người
được lấy mẫu ngẫu nhiên từ quần thể.
Trong trường hợp này, chúng ta hiếm khi thực sự quan tâm
đến chiều cao chính xác của một người.
Hơn nữa, nếu đo đủ chính xác,
ta sẽ thấy rằng không có hai người nào trên hành tinh
có cùng chiều cao chính xác.
Trên thực tế, với các phép đo đủ tinh vi,
bạn sẽ không bao giờ có cùng một chiều cao
khi thức dậy và khi đi ngủ.
Việc hỏi xác suất chính xác để một người
cao 1.801392782910287192 mét
hầu như không có ý nghĩa.
Thay vào đó, chúng ta thường quan tâm hơn đến việc có thể nói
chiều cao của một người có nằm trong một khoảng nhất định hay không,
chẳng hạn từ 1.79 đến 1.81 mét.
Trong những trường hợp này, chúng ta làm việc với *mật độ* xác suất.
Chiều cao chính xác 1.80 mét
không có xác suất, nhưng có mật độ khác không.
Để tính xác suất được gán cho một khoảng,
ta phải lấy *tích phân* của mật độ
trên khoảng đó.

## Nhiều biến ngẫu nhiên

Có thể bạn đã nhận thấy rằng chúng ta thậm chí không thể
đi qua phần trước
mà không đưa ra các mệnh đề liên quan đến sự tương tác
giữa nhiều biến ngẫu nhiên
(nhớ rằng $P(X,Y) = P(X) P(Y)$).
Phần lớn machine learning
liên quan đến những mối quan hệ như vậy.
Ở đây, không gian mẫu sẽ là
quần thể được quan tâm,
chẳng hạn khách hàng giao dịch với một doanh nghiệp,
ảnh trên Internet,
hoặc các protein được các nhà sinh học biết đến.
Mỗi biến ngẫu nhiên sẽ đại diện cho
giá trị (chưa biết) của một thuộc tính khác nhau.
Mỗi khi lấy mẫu một cá thể từ quần thể,
chúng ta quan sát một hiện thực hóa của từng biến ngẫu nhiên.
Vì các giá trị mà các biến ngẫu nhiên nhận được
tương ứng với các tập con của không gian mẫu
có thể chồng lấn, chồng lấn một phần,
hoặc hoàn toàn rời nhau,
biết giá trị mà một biến ngẫu nhiên nhận được
có thể khiến chúng ta cập nhật niềm tin
về những giá trị nào của một biến ngẫu nhiên khác có khả năng xảy ra.
Nếu một bệnh nhân bước vào bệnh viện
và chúng ta quan sát thấy họ
gặp khó khăn khi thở
và mất khứu giác,
thì chúng ta tin rằng họ có nhiều khả năng
mắc COVID-19 hơn so với trường hợp
họ không gặp khó khăn khi thở
và có khứu giác hoàn toàn bình thường.


Khi làm việc với nhiều biến ngẫu nhiên,
ta có thể xây dựng các biến cố tương ứng
với mọi tổ hợp giá trị
mà các biến có thể đồng thời nhận.
Hàm xác suất gán
xác suất cho từng tổ hợp này
(ví dụ: $A=a$ và $B=b$)
được gọi là hàm *xác suất đồng thời*
và chỉ đơn giản trả về xác suất được gán
cho giao của các tập con tương ứng
của không gian mẫu.
*Xác suất đồng thời* được gán cho biến cố
trong đó các biến ngẫu nhiên $A$ và $B$
lần lượt nhận các giá trị $a$ và $b$,
được ký hiệu là $P(A = a, B = b)$,
trong đó dấu phẩy biểu thị “và”.
Lưu ý rằng với mọi giá trị $a$ và $b$,
ta có

$$P(A=a, B=b) \leq P(A=a) \textrm{ and } P(A=a, B=b) \leq P(B = b),$$

vì để $A=a$ và $B=b$ xảy ra,
$A=a$ phải xảy ra *và* $B=b$ cũng phải xảy ra.
Điều thú vị là xác suất đồng thời
cho chúng ta biết tất cả những gì có thể biết về các
biến ngẫu nhiên này theo nghĩa xác suất,
và có thể được dùng để suy ra nhiều đại lượng hữu ích khác,
bao gồm việc khôi phục các phân phối riêng lẻ $P(A)$ và $P(B)$.
Để khôi phục $P(A=a)$, ta chỉ cần tính tổng
$P(A=a, B=v)$ trên mọi giá trị $v$
mà biến ngẫu nhiên $B$ có thể nhận:
$P(A=a) = \sum_v P(A=a, B=v)$.


Tỷ số $\frac{P(A=a, B=b)}{P(A=a)} \leq 1$
hóa ra cực kỳ quan trọng.
Nó được gọi là *xác suất có điều kiện*,
và được ký hiệu bằng biểu tượng “$\mid$”:

$$P(B=b \mid A=a) = P(A=a,B=b)/P(A=a).$$

Điều này cho chúng ta biết xác suất mới
gắn với biến cố $B=b$,
sau khi điều kiện hóa theo thực tế rằng $A=a$ đã xảy ra.
Ta có thể xem xác suất có điều kiện này
là việc chỉ xét đến tập con
của không gian mẫu tương ứng với $A=a$
và sau đó chuẩn hóa lại sao cho
tổng tất cả các xác suất bằng 1.
Các xác suất có điều kiện
thực ra chỉ là các xác suất thông thường
và do đó tuân theo tất cả các tiên đề,
miễn là ta điều kiện hóa tất cả các số hạng
theo cùng một biến cố và vì thế
chỉ xét cùng một không gian mẫu.
Ví dụ, với các biến cố rời nhau
$\mathcal{B}$ và $\mathcal{B}'$, ta có
$P(\mathcal{B} \cup \mathcal{B}' \mid A = a) = P(\mathcal{B} \mid A = a) + P(\mathcal{B}' \mid A = a)$.


Sử dụng định nghĩa của các xác suất có điều kiện,
ta có thể suy ra kết quả nổi tiếng gọi là *định lý Bayes*.
Theo cách xây dựng, ta có $P(A, B) = P(B\mid A) P(A)$
và $P(A, B) = P(A\mid B) P(B)$.
Kết hợp hai phương trình, ta được
$P(B\mid A) P(A) = P(A\mid B) P(B)$ và do đó

$$P(A \mid B) = \frac{P(B\mid A) P(A)}{P(B)}.$$






Phương trình đơn giản này có những hàm ý sâu sắc vì
nó cho phép ta đảo ngược thứ tự điều kiện hóa.
Nếu biết cách ước lượng $P(B\mid A)$, $P(A)$ và $P(B)$,
ta có thể ước lượng $P(A\mid B)$.
Ta thường thấy việc ước lượng trực tiếp một số hạng
dễ hơn số hạng còn lại, và định lý Bayes có thể giúp giải quyết vấn đề này.
Ví dụ, nếu biết tỷ lệ phổ biến của các triệu chứng đối với một bệnh nhất định,
cũng như tỷ lệ phổ biến tổng thể của bệnh và các triệu chứng,
ta có thể xác định khả năng một người
mắc bệnh dựa trên các triệu chứng của họ.
Trong một số trường hợp, ta có thể không truy cập trực tiếp được $P(B)$,
chẳng hạn như tỷ lệ phổ biến của các triệu chứng.
Trong trường hợp này, một phiên bản đơn giản hóa của định lý Bayes sẽ hữu ích:

$$P(A \mid B) \propto P(B \mid A) P(A).$$

Vì biết rằng $P(A \mid B)$ phải được chuẩn hóa về $1$, tức là $\sum_a P(A=a \mid B) = 1$,
ta có thể dùng nó để tính

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{\sum_a P(B \mid A=a) P(A = a)}.$$

Trong thống kê Bayes, ta xem một người quan sát
là người có một số niềm tin tiên nghiệm (mang tính chủ quan)
về mức độ khả dĩ của các giả thuyết hiện có,
được mã hóa trong *prior* $P(H)$,
cùng với một *hàm likelihood* cho biết
khả năng quan sát được mỗi giá trị của bằng chứng thu thập được
đối với từng giả thuyết trong lớp $P(E \mid H)$.
Khi đó, định lý Bayes được diễn giải là cho biết cách
cập nhật *prior* ban đầu $P(H)$
dựa trên bằng chứng hiện có $E$
để tạo ra các niềm tin *posterior*
$P(H \mid E) = \frac{P(E \mid H) P(H)}{P(E)}$.
Nói một cách không chính thức, điều này có thể được phát biểu là
"posterior bằng prior nhân với likelihood, chia cho bằng chứng".
Bây giờ, vì bằng chứng $P(E)$ giống nhau đối với mọi giả thuyết,
ta có thể chỉ cần chuẩn hóa trên các giả thuyết.

Lưu ý rằng $\sum_a P(A=a \mid B) = 1$ cũng cho phép ta *marginalize* trên các biến ngẫu nhiên. Nghĩa là, ta có thể loại bỏ các biến khỏi một phân phối kết hợp chẳng hạn như $P(A, B)$. Sau cùng, ta có

$$\sum_a P(B \mid A=a) P(A=a) = \sum_a P(B, A=a) = P(B).$$

Tính độc lập là một khái niệm đặc biệt quan trọng khác,
tạo nền tảng cho
nhiều ý tưởng quan trọng trong thống kê.
Nói ngắn gọn, hai biến là *độc lập*
nếu việc điều kiện hóa theo giá trị của $A$ không
làm thay đổi bất kỳ điều gì đối với phân phối xác suất
gắn với $B$, và ngược lại.
Chính thức hơn, tính độc lập, được ký hiệu là $A \perp B$,
yêu cầu $P(A \mid B) = P(A)$ và do đó,
$P(A,B) = P(A \mid B) P(B) = P(A) P(B)$.
Tính độc lập thường là một giả định phù hợp.
Ví dụ, nếu biến ngẫu nhiên $A$
biểu diễn kết quả của việc tung một đồng xu công bằng
và biến ngẫu nhiên $B$
biểu diễn kết quả của việc tung một đồng xu khác,
thì việc biết $A$ ra mặt ngửa
không nên ảnh hưởng đến xác suất
$B$ ra mặt ngửa.


Tính độc lập đặc biệt hữu ích khi nó tồn tại giữa các
lần lấy dữ liệu liên tiếp từ một phân phối nền nào đó
(cho phép ta đưa ra những kết luận thống kê mạnh)
hoặc khi nó tồn tại giữa nhiều biến khác nhau trong dữ liệu,
cho phép ta làm việc với các mô hình đơn giản hơn
mà vẫn mã hóa cấu trúc độc lập này.
Mặt khác, việc ước lượng các mối phụ thuộc
giữa các biến ngẫu nhiên thường chính là mục tiêu của quá trình học.
Ta quan tâm đến việc ước lượng xác suất mắc bệnh khi biết các triệu chứng
chính vì tin rằng
bệnh và triệu chứng là *không độc lập*.


Lưu ý rằng vì các xác suất có điều kiện là những xác suất hợp lệ,
các khái niệm độc lập và phụ thuộc cũng áp dụng cho chúng.
Hai biến ngẫu nhiên $A$ và $B$ là *độc lập có điều kiện*
khi biết biến thứ ba $C$ khi và chỉ khi $P(A, B \mid C) = P(A \mid C)P(B \mid C)$.
Điều thú vị là hai biến có thể độc lập nói chung
nhưng trở nên phụ thuộc khi điều kiện hóa theo biến thứ ba.
Điều này thường xảy ra khi hai biến ngẫu nhiên $A$ và $B$
tương ứng với các nguyên nhân của một biến thứ ba $C$.
Ví dụ, gãy xương và ung thư phổi có thể độc lập
trong dân số nói chung, nhưng nếu ta điều kiện hóa theo việc đang ở bệnh viện
thì có thể phát hiện rằng gãy xương có tương quan âm với ung thư phổi.
Đó là vì việc bị gãy xương *giải thích thay thế* lý do một người đang ở bệnh viện
và do đó làm giảm xác suất người đó nhập viện vì mắc ung thư phổi.


Ngược lại, hai biến ngẫu nhiên phụ thuộc
có thể trở nên độc lập khi điều kiện hóa theo biến thứ ba.
Điều này thường xảy ra khi hai biến cố vốn không liên quan
có chung một nguyên nhân.
Kích thước giày và trình độ đọc có tương quan rất cao
ở học sinh tiểu học,
nhưng mối tương quan này biến mất nếu ta điều kiện hóa theo tuổi.



## Một ví dụ
:label:`subsec_probability_hiv_app`

Hãy thử áp dụng các kỹ năng của chúng ta.
Giả sử một bác sĩ tiến hành xét nghiệm HIV cho một bệnh nhân.
Xét nghiệm này khá chính xác và chỉ cho kết quả sai với xác suất 1%
nếu bệnh nhân khỏe mạnh nhưng bị báo cáo là mắc bệnh,
tức là bệnh nhân khỏe mạnh cho kết quả dương tính trong 1% trường hợp.
Hơn nữa, xét nghiệm không bao giờ không phát hiện được HIV nếu bệnh nhân thực sự mắc bệnh.
Ta dùng $D_1 \in \{0, 1\}$ để biểu thị chẩn đoán
($0$ nếu âm tính và $1$ nếu dương tính)
và $H \in \{0, 1\}$ để biểu thị tình trạng HIV.

| Xác suất có điều kiện | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_1 = 1 \mid H)$        |     1 |  0.01 |
| $P(D_1 = 0 \mid H)$        |     0 |  0.99 |

Lưu ý rằng tổng các cột đều bằng 1 (nhưng tổng các hàng thì không),
vì đây là các xác suất có điều kiện.
Hãy tính xác suất bệnh nhân mắc HIV
nếu xét nghiệm cho kết quả dương tính, tức là $P(H = 1 \mid D_1 = 1)$.
Theo trực giác, xác suất này sẽ phụ thuộc vào mức độ phổ biến của bệnh,
vì điều đó ảnh hưởng đến số lượng kết quả dương tính giả.
Giả sử quần thể tương đối ít mắc bệnh, chẳng hạn $P(H=1) = 0.0015$.
Để áp dụng định lý Bayes, ta cần thực hiện marginalization
để xác định

$$\begin{aligned}
P(D_1 = 1)
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

Từ đó ta có

$$P(H = 1 \mid D_1 = 1) = \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} = 0.1306.$$

Nói cách khác, chỉ có 13.06% khả năng
bệnh nhân thực sự mắc HIV,
mặc dù xét nghiệm khá chính xác.
Như ta có thể thấy, xác suất đôi khi trái với trực giác.
Bệnh nhân nên làm gì sau khi nhận được tin đáng sợ như vậy?
Có lẽ bệnh nhân sẽ yêu cầu bác sĩ
tiến hành một xét nghiệm khác để có được kết luận rõ ràng hơn.
Xét nghiệm thứ hai có các đặc điểm khác
và không tốt bằng xét nghiệm đầu tiên.

| Xác suất có điều kiện | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_2 = 1 \mid H)$          |  0.98 |  0.03 |
| $P(D_2 = 0 \mid H)$          |  0.02 |  0.97 |

Đáng tiếc là xét nghiệm thứ hai cũng cho kết quả dương tính.
Hãy tính các xác suất cần thiết để áp dụng định lý Bayes
bằng cách giả định tính độc lập có điều kiện:

$$\begin{aligned}
P(D_1 = 1, D_2 = 1 \mid H = 0)
& = P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0)
=& 0.0003, \\
P(D_1 = 1, D_2 = 1 \mid H = 1)
& = P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1)
=& 0.98.
\end{aligned}
$$

Bây giờ ta có thể áp dụng marginalization để thu được xác suất
rằng cả hai xét nghiệm đều cho kết quả dương tính:

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1)\\
&= P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
&= P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
&= 0.00176955.
\end{aligned}
$$

Cuối cùng, xác suất bệnh nhân mắc HIV khi cả hai xét nghiệm đều dương tính là

$$P(H = 1 \mid D_1 = 1, D_2 = 1)
= \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)}
= 0.8307.$$

Nói cách khác, xét nghiệm thứ hai giúp chúng ta tin tưởng hơn nhiều rằng mọi việc không ổn.
Mặc dù xét nghiệm thứ hai kém chính xác hơn đáng kể so với xét nghiệm đầu tiên,
nó vẫn cải thiện đáng kể ước lượng của chúng ta.
Giả định rằng hai xét nghiệm độc lập có điều kiện với nhau
có vai trò then chốt trong khả năng tạo ra một ước lượng chính xác hơn.
Hãy xét trường hợp cực đoan khi ta thực hiện cùng một xét nghiệm hai lần.
Trong tình huống này, ta sẽ kỳ vọng nhận được cùng một kết quả ở cả hai lần,
vì vậy việc thực hiện lại cùng một xét nghiệm không cung cấp thêm thông tin nào.
Một độc giả tinh ý có thể đã nhận ra rằng chẩn đoán này hoạt động
giống như một classifier ẩn hiện rõ ràng
trong đó khả năng quyết định bệnh nhân có khỏe mạnh hay không
tăng lên khi ta thu thập thêm nhiều feature (kết quả xét nghiệm).


## Kỳ vọng

Thông thường, việc ra quyết định không chỉ đòi hỏi xem xét
các xác suất được gán cho từng sự kiện riêng lẻ
mà còn phải kết hợp chúng thành những đại lượng tổng hợp hữu ích
có thể cung cấp cho chúng ta sự định hướng.
Ví dụ, khi các biến ngẫu nhiên nhận các giá trị vô hướng liên tục,
chúng ta thường quan tâm đến việc biết giá trị nào có thể được kỳ vọng *trung bình*.
Đại lượng này được gọi chính thức là *kỳ vọng*.
Nếu chúng ta đang thực hiện các khoản đầu tư,
đại lượng được quan tâm đầu tiên
có thể là lợi nhuận mà chúng ta có thể kỳ vọng,
lấy trung bình trên tất cả các kết quả có thể xảy ra
(và tính trọng số theo các xác suất thích hợp).
Chẳng hạn, giả sử với xác suất 50%,
một khoản đầu tư có thể thất bại hoàn toàn,
với xác suất 40% khoản đầu tư có thể mang lại lợi nhuận 2$\times$,
và với xác suất 10% khoản đầu tư có thể mang lại lợi nhuận 10$\times$ 10$\times$.
Để tính lợi nhuận kỳ vọng,
chúng ta cộng trên tất cả các mức lợi nhuận, nhân mỗi mức
với xác suất nó xảy ra.
Kết quả là kỳ vọng
$0.5 \cdot 0 + 0.4 \cdot 2 + 0.1 \cdot 10 = 1.8$.
Do đó, lợi nhuận kỳ vọng là 1.8$\times$.


Nói chung, *kỳ vọng* (hay giá trị trung bình)
của biến ngẫu nhiên $X$ được định nghĩa là

$$E[X] = E_{x \sim P}[x] = \sum_{x} x P(X = x).$$

Tương tự, đối với các mật độ, ta có $E[X] = \int x \;dp(x)$.
Đôi khi chúng ta quan tâm đến giá trị kỳ vọng
của một hàm nào đó của $x$.
Chúng ta có thể tính các kỳ vọng này như sau:

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x) \textrm{ and } E_{x \sim P}[f(x)] = \int f(x) p(x) \;dx$$

lần lượt đối với các xác suất rời rạc và các mật độ.
Quay lại ví dụ đầu tư ở trên,
$f$ có thể là *utility* (mức độ hạnh phúc)
gắn với lợi nhuận.
Các nhà kinh tế học hành vi từ lâu đã nhận thấy
rằng con người gắn mức độ khó chịu lớn hơn
với việc mất tiền so với mức utility đạt được
khi kiếm thêm một đô la so với mức cơ sở của họ.
Hơn nữa, giá trị của tiền có xu hướng tăng dưới tuyến tính.
Việc sở hữu 100 nghìn đô la so với không có đồng nào
có thể tạo ra khác biệt giữa việc trả được tiền thuê nhà,
ăn uống đầy đủ và được hưởng dịch vụ chăm sóc sức khỏe chất lượng
với việc phải chịu cảnh vô gia cư.
Mặt khác, lợi ích do sở hữu
200 nghìn so với 100 nghìn đô la mang lại ít đáng kể hơn.
Cách lập luận này dẫn đến câu nói sáo rỗng
rằng “utility của tiền là logarithmic”.


Nếu utility gắn với việc mất trắng toàn bộ là $-1$,
và utility gắn với các mức lợi nhuận $1$, $2$ và $10$
lần lượt là $1$, $2$ và $4$,
thì mức hạnh phúc kỳ vọng khi đầu tư
sẽ là $0.5 \cdot (-1) + 0.4 \cdot 2 + 0.1 \cdot 4 = 0.7$
(mức utility kỳ vọng giảm 30%).
Nếu đây thực sự là hàm utility của bạn,
có lẽ lựa chọn tốt nhất là giữ tiền trong ngân hàng.

Đối với các quyết định tài chính,
chúng ta cũng có thể muốn đo lường
một khoản đầu tư *rủi ro* đến mức nào.
Ở đây, chúng ta không chỉ quan tâm đến giá trị kỳ vọng
mà còn quan tâm đến mức độ các giá trị thực tế có xu hướng *biến thiên*
so với giá trị này.
Lưu ý rằng chúng ta không thể chỉ lấy
kỳ vọng của hiệu giữa các giá trị thực tế và giá trị kỳ vọng.
Đó là vì kỳ vọng của một hiệu
bằng hiệu của các kỳ vọng,
tức là $E[X - E[X]] = E[X] - E[E[X]] = 0$.
Tuy nhiên, chúng ta có thể xét kỳ vọng
của bất kỳ hàm không âm nào của hiệu này.
*Phương sai* của một biến ngẫu nhiên được tính bằng cách xét
giá trị kỳ vọng của các hiệu *bình phương*:

$$\textrm{Var}[X] = E\left[(X - E[X])^2\right] = E[X^2] - E[X]^2.$$

Ở đây, đẳng thức được suy ra bằng cách khai triển
$(X - E[X])^2 = X^2 - 2 X E[X] + E[X]^2$
và lấy kỳ vọng của từng hạng tử.
Căn bậc hai của phương sai là một
đại lượng hữu ích khác được gọi là *độ lệch chuẩn*.
Mặc dù độ lệch chuẩn và phương sai
truyền tải cùng một thông tin (một đại lượng có thể được tính từ đại lượng kia),
độ lệch chuẩn có tính chất thuận tiện
là được biểu diễn bằng cùng đơn vị
với đại lượng gốc được biểu diễn
bởi biến ngẫu nhiên.

Cuối cùng, phương sai của một hàm
của một biến ngẫu nhiên
được định nghĩa tương tự như sau:

$$\textrm{Var}_{x \sim P}[f(x)] = E_{x \sim P}[f^2(x)] - E_{x \sim P}[f(x)]^2.$$

Quay lại ví dụ đầu tư,
giờ đây chúng ta có thể tính phương sai của khoản đầu tư.
Nó được cho bởi $0.5 \cdot 0 + 0.4 \cdot 2^2 + 0.1 \cdot 10^2 - 1.8^2 = 8.36$.
Xét trên mọi phương diện, đây là một khoản đầu tư rủi ro.
Lưu ý rằng theo quy ước toán học, giá trị trung bình và phương sai
thường được ký hiệu lần lượt là $\mu$ và $\sigma^2$.
Điều này đặc biệt đúng khi chúng ta sử dụng chúng
để tham số hóa một phân phối Gaussian.

Tương tự như cách chúng ta đưa ra các khái niệm kỳ vọng
và phương sai cho các biến ngẫu nhiên *vô hướng*,
chúng ta cũng có thể làm như vậy với các biến ngẫu nhiên dạng vector.
Kỳ vọng rất đơn giản, vì chúng ta có thể áp dụng chúng theo từng phần tử.
Chẳng hạn, $\boldsymbol{\mu} \stackrel{\textrm{def}}{=} E_{\mathbf{x} \sim P}[\mathbf{x}]$
có các tọa độ $\mu_i = E_{\mathbf{x} \sim P}[x_i]$.
*Hiệp phương sai* phức tạp hơn.
Chúng ta định nghĩa chúng bằng cách lấy kỳ vọng của *tích ngoài*
của hiệu giữa các biến ngẫu nhiên và giá trị trung bình của chúng:

$$\boldsymbol{\Sigma} \stackrel{\textrm{def}}{=} \textrm{Cov}_{\mathbf{x} \sim P}[\mathbf{x}] = E_{\mathbf{x} \sim P}\left[(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top\right].$$

Ma trận $\boldsymbol{\Sigma}$ này được gọi là ma trận hiệp phương sai.
Một cách dễ dàng để thấy tác động của nó là xét một vector $\mathbf{v}$
có cùng kích thước với $\mathbf{x}$.
Ta có

$$\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v} = E_{\mathbf{x} \sim P}\left[\mathbf{v}^\top(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{v}\right] = \textrm{Var}_{x \sim P}[\mathbf{v}^\top \mathbf{x}].$$

Như vậy, $\boldsymbol{\Sigma}$ cho phép chúng ta tính phương sai
của bất kỳ hàm tuyến tính nào của $\mathbf{x}$
chỉ bằng một phép nhân ma trận đơn giản.
Các phần tử ngoài đường chéo cho biết các tọa độ tương quan với nhau như thế nào:
giá trị 0 có nghĩa là không có tương quan,
trong khi giá trị dương lớn hơn
cho biết chúng tương quan mạnh hơn.



## Thảo luận

Trong machine learning, có rất nhiều điều không chắc chắn!
Chúng ta có thể không chắc chắn về giá trị của một nhãn khi biết một đầu vào.
Chúng ta có thể không chắc chắn về giá trị ước lượng của một tham số.
Thậm chí chúng ta có thể không chắc chắn liệu dữ liệu đến trong quá trình deploy
có thực sự đến từ cùng một phân phối với dữ liệu huấn luyện hay không.

*Aleatoric uncertainty* là sự không chắc chắn
mang tính nội tại của bài toán,
và bắt nguồn từ tính ngẫu nhiên thực sự
không được giải thích bởi các biến quan sát.
*Epistemic uncertainty* là sự không chắc chắn
về các tham số của một mô hình, tức dạng không chắc chắn
mà chúng ta có thể hy vọng giảm bớt bằng cách thu thập thêm dữ liệu.
Chúng ta có thể có epistemic uncertainty
về xác suất
một đồng xu cho mặt ngửa,
nhưng ngay cả khi đã biết xác suất này,
chúng ta vẫn còn aleatoric uncertainty
về kết quả của bất kỳ lần tung nào trong tương lai.
Dù quan sát ai đó tung một đồng xu công bằng trong bao lâu,
chúng ta sẽ không bao giờ chắc chắn hơn hoặc kém hơn 50%
rằng lần tung tiếp theo sẽ cho mặt ngửa.
Các thuật ngữ này bắt nguồn từ mô hình hóa cơ học,
(xem ví dụ :citet:`Der-Kiureghian.Ditlevsen.2009` để có phần tổng quan về khía cạnh này của [uncertainty quantification](https://en.wikipedia.org/wiki/Uncertainty_quantification)).
Tuy nhiên, cần lưu ý rằng các thuật ngữ này là một cách dùng ngôn ngữ hơi không chính xác.
Thuật ngữ *epistemic* chỉ bất kỳ điều gì liên quan đến *tri thức*
và do đó, theo nghĩa triết học, mọi sự không chắc chắn đều mang tính epistemic.


Chúng ta đã thấy rằng việc lấy mẫu dữ liệu từ một phân phối xác suất chưa biết
có thể cung cấp cho chúng ta thông tin dùng để ước lượng
các tham số của phân phối sinh dữ liệu.
Tuy vậy, tốc độ thực hiện được điều này có thể khá chậm.
Trong ví dụ tung đồng xu (và nhiều ví dụ khác),
chúng ta không thể làm tốt hơn việc thiết kế các estimator
hội tụ với tốc độ $1/\sqrt{n}$,
trong đó $n$ là kích thước mẫu (ví dụ: số lần tung).
Điều này có nghĩa là khi tăng từ 10 lên 1000 quan sát (thường là một việc hoàn toàn khả thi),
chúng ta thấy mức độ không chắc chắn giảm đi mười lần,
trong khi 1000 quan sát tiếp theo chỉ giúp ích tương đối ít,
chỉ đem lại mức giảm 1.41 lần.
Đây là một đặc điểm dai dẳng của machine learning:
mặc dù thường có những cải thiện dễ đạt được,
để đạt thêm các cải thiện tiếp theo cần một lượng dữ liệu rất lớn,
và thường cùng với đó là lượng tính toán khổng lồ.
Để xem một tổng quan thực nghiệm về thực tế này đối với các mô hình ngôn ngữ quy mô lớn, hãy xem :citet:`Revels.Lubin.Papamarkou.2016`.

Chúng ta cũng đã làm rõ hơn ngôn ngữ và các công cụ
dùng cho mô hình hóa thống kê.
Trong quá trình đó, chúng ta đã tìm hiểu về xác suất có điều kiện
và một trong những phương trình quan trọng nhất trong thống kê---định lý Bayes.
Đây là một công cụ hiệu quả để tách biệt thông tin do dữ liệu truyền tải
thông qua một hạng likelihood $P(B \mid A)$, hạng này mô tả
mức độ các quan sát $B$ phù hợp với một lựa chọn tham số $A$,
và một xác suất tiên nghiệm $P(A)$, xác suất chi phối mức độ hợp lý
của một lựa chọn cụ thể của $A$ ngay từ đầu.
Cụ thể, chúng ta đã thấy quy tắc này có thể được áp dụng
để gán xác suất cho các chẩn đoán,
dựa trên hiệu quả của xét nghiệm *và*
mức độ phổ biến của chính căn bệnh đó (tức là prior của chúng ta).

Cuối cùng, chúng ta đã đưa ra tập hợp đầu tiên gồm các câu hỏi không tầm thường về ảnh hưởng của một phân phối xác suất cụ thể, cụ thể là kỳ vọng và phương sai.  
Mặc dù một phân phối xác suất có nhiều dạng kỳ vọng khác ngoài kỳ vọng tuyến tính và bậc hai, hai đại lượng này đã cung cấp khá nhiều thông tin về hành vi có thể xảy ra của phân phối.  
Chẳng hạn, [bất đẳng thức Chebyshev](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality) phát biểu rằng $P(|X - \mu| \geq k \sigma) \leq 1/k^2$, trong đó $\mu$ là kỳ vọng, $\sigma^2$ là phương sai của phân phối, và $k > 1$ là tham số độ tin cậy do chúng ta lựa chọn.  
Điều này cho biết các giá trị lấy mẫu từ một phân phối nằm trong khoảng $[-\sqrt{2} \sigma, \sqrt{2} \sigma]$ có tâm là kỳ vọng với xác suất ít nhất 50%.




## Bài tập

1. Hãy đưa ra một ví dụ trong đó việc quan sát thêm dữ liệu có thể làm giảm mức độ không chắc chắn về kết quả xuống một mức thấp tùy ý.
1. Hãy đưa ra một ví dụ trong đó việc quan sát thêm dữ liệu chỉ làm giảm mức độ không chắc chắn đến một mức nhất định, sau đó không giảm thêm nữa. Giải thích tại sao lại như vậy và bạn dự đoán mức này sẽ xuất hiện ở đâu.
1. Chúng ta đã thực nghiệm chứng minh sự hội tụ về giá trị trung bình khi tung một đồng xu. Hãy tính phương sai của ước lượng xác suất xuất hiện mặt ngửa sau khi lấy $n$ mẫu.
    1. Phương sai thay đổi như thế nào theo số lượng quan sát?
    1. Sử dụng bất đẳng thức Chebyshev để chặn độ lệch khỏi kỳ vọng.
    1. Điều này liên hệ như thế nào với định lý giới hạn trung tâm?
1. Giả sử chúng ta lấy $m$ mẫu $x_i$ từ một phân phối xác suất có trung bình bằng không và phương sai bằng một. Hãy tính các giá trị trung bình $z_m \stackrel{\textrm{def}}{=} m^{-1} \sum_{i=1}^m x_i$. Chúng ta có thể áp dụng bất đẳng thức Chebyshev cho từng $z_m$ một cách độc lập không? Tại sao không?
1. Với hai biến cố có xác suất $P(\mathcal{A})$ và $P(\mathcal{B})$, hãy tính các cận trên và cận dưới của $P(\mathcal{A} \cup \mathcal{B})$ và $P(\mathcal{A} \cap \mathcal{B})$. Gợi ý: hãy biểu diễn tình huống bằng một [sơ đồ Venn](https://en.wikipedia.org/wiki/Venn_diagram).
1. Giả sử chúng ta có một chuỗi biến ngẫu nhiên, chẳng hạn $A$, $B$ và $C$, trong đó $B$ chỉ phụ thuộc vào $A$, còn $C$ chỉ phụ thuộc vào $B$. Bạn có thể đơn giản hóa xác suất đồng thời $P(A, B, C)$ không? Gợi ý: đây là một [chuỗi Markov](https://en.wikipedia.org/wiki/Markov_chain).
1. Trong :numref:`subsec_probability_hiv_app`, giả sử kết quả của hai xét nghiệm không độc lập. Cụ thể, giả sử mỗi xét nghiệm riêng lẻ có tỷ lệ dương tính giả là 10% và tỷ lệ âm tính giả là 1%. Nghĩa là, giả sử $P(D =1 \mid H=0) = 0.1$ và $P(D = 0 \mid H=1) = 0.01$. Ngoài ra, giả sử rằng với $H = 1$ (bị nhiễm), kết quả của các xét nghiệm là độc lập có điều kiện, tức là $P(D_1, D_2 \mid H=1) = P(D_1 \mid H=1) P(D_2 \mid H=1)$, nhưng với các bệnh nhân khỏe mạnh, kết quả xét nghiệm phụ thuộc lẫn nhau thông qua $P(D_1 = D_2 = 1 \mid H=0) = 0.02$.
    1. Dựa trên những thông tin hiện có, hãy lập bảng xác suất đồng thời cho $D_1$ và $D_2$, với điều kiện $H=0$.
    1. Hãy suy ra xác suất bệnh nhân mắc bệnh ($H=1$) sau khi một xét nghiệm cho kết quả dương tính. Bạn có thể giả sử xác suất cơ sở như trước đây là $P(H=1) = 0.0015$.
    1. Hãy suy ra xác suất bệnh nhân mắc bệnh ($H=1$) sau khi cả hai xét nghiệm đều cho kết quả dương tính.
1. Giả sử bạn là một nhà quản lý tài sản cho một ngân hàng đầu tư và có một số cổ phiếu $s_i$ để lựa chọn đầu tư. Danh mục đầu tư của bạn cần có tổng trọng số bằng $1$, với trọng số $\alpha_i$ cho mỗi cổ phiếu. Các cổ phiếu có lợi suất trung bình $\boldsymbol{\mu} = E_{\mathbf{s} \sim P}[\mathbf{s}]$ và hiệp phương sai $\boldsymbol{\Sigma} = \textrm{Cov}_{\mathbf{s} \sim P}[\mathbf{s}]$.
    1. Hãy tính lợi suất kỳ vọng của một danh mục đầu tư $\boldsymbol{\alpha}$ nhất định.
    1. Nếu muốn tối đa hóa lợi suất của danh mục đầu tư, bạn nên lựa chọn khoản đầu tư như thế nào?
    1. Hãy tính *phương sai* của danh mục đầu tư.
    1. Hãy xây dựng một bài toán tối ưu hóa nhằm tối đa hóa lợi suất trong khi giữ phương sai không vượt quá một cận trên. Đây là bài toán [danh mục đầu tư Markovitz](https://en.wikipedia.org/wiki/Markowitz_model) :cite:`Mangram.2013` đã giành giải Nobel. Để giải bài toán này, bạn sẽ cần một trình giải quadratic programming, vượt xa phạm vi của cuốn sách này.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/36)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/37)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/198)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17971)
:end_tab: