```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

# Truy vấn, Khóa và Giá trị
:label:`sec_queries-keys-values`

Cho đến nay, tất cả các mạng mà chúng ta đã xem xét đều phụ thuộc thiết yếu vào việc đầu vào có kích thước được xác định rõ ràng. Ví dụ, các ảnh trong ImageNet có kích thước $224 \times 224$ pixel và CNN được tinh chỉnh riêng cho kích thước này. Ngay cả trong xử lý ngôn ngữ tự nhiên, kích thước đầu vào của RNN cũng được xác định rõ và cố định. Kích thước biến thiên được xử lý bằng cách tuần tự xử lý từng token một, hoặc bằng các convolution kernel được thiết kế đặc biệt :cite:`Kalchbrenner.Grefenstette.Blunsom.2014`. Cách tiếp cận này có thể dẫn đến các vấn đề đáng kể khi đầu vào thực sự có kích thước và lượng thông tin biến thiên, như trong :numref:`sec_seq2seq` khi biến đổi văn bản :cite:`Sutskever.Vinyals.Le.2014`. Đặc biệt, với các chuỗi dài, việc theo dõi mọi thứ đã được tạo ra hoặc thậm chí đã được mạng xem qua trở nên khá khó khăn. Ngay cả các heuristic theo dõi tường minh như được đề xuất bởi :citet:`yang2016neural` cũng chỉ mang lại lợi ích hạn chế.

Hãy so sánh điều này với cơ sở dữ liệu. Ở dạng đơn giản nhất, chúng là tập hợp các khóa ($k$) và giá trị ($v$). Ví dụ, cơ sở dữ liệu $\mathcal{D}$ của chúng ta có thể gồm các bộ \{("Zhang", "Aston"), ("Lipton", "Zachary"), ("Li", "Mu"), ("Smola", "Alex"), ("Hu", "Rachel"), ("Werness", "Brent")\}, trong đó họ là khóa và tên là giá trị. Chúng ta có thể thao tác trên $\mathcal{D}$, chẳng hạn bằng truy vấn chính xác ($q$) cho "Li", sẽ trả về giá trị "Mu". Nếu ("Li", "Mu") không phải là một bản ghi trong $\mathcal{D}$, sẽ không có câu trả lời hợp lệ. Nếu chúng ta cũng cho phép các khớp gần đúng, thay vào đó chúng ta sẽ truy xuất ("Lipton", "Zachary"). Tuy đơn giản và hiển nhiên, ví dụ này vẫn cho chúng ta thấy một số điều hữu ích:

* Chúng ta có thể thiết kế các truy vấn $q$ hoạt động trên các cặp ($k$,$v$) sao cho vẫn hợp lệ bất kể kích thước cơ sở dữ liệu.
* Cùng một truy vấn có thể nhận được các câu trả lời khác nhau, tùy theo nội dung của cơ sở dữ liệu.
* "Mã" được thực thi để thao tác trên một không gian trạng thái lớn (cơ sở dữ liệu) có thể khá đơn giản (ví dụ: khớp chính xác, khớp gần đúng, top-$k$).
* Không cần nén hoặc đơn giản hóa cơ sở dữ liệu để các thao tác trở nên hiệu quả.

Rõ ràng, chúng ta sẽ không giới thiệu một cơ sở dữ liệu đơn giản ở đây nếu không nhằm mục đích giải thích deep learning. Thật vậy, điều này dẫn đến một trong những khái niệm thú vị nhất được giới thiệu trong deep learning trong thập kỷ qua: *cơ chế attention* :cite:`Bahdanau.Cho.Bengio.2014`. Chúng ta sẽ trình bày chi tiết về ứng dụng của nó cho dịch máy ở phần sau. Còn bây giờ, chỉ cần xét điều sau: ký hiệu $\mathcal{D} \stackrel{\textrm{def}}{=} \{(\mathbf{k}_1, \mathbf{v}_1), \ldots (\mathbf{k}_m, \mathbf{v}_m)\}$ là một cơ sở dữ liệu gồm $m$ bộ *khóa* và *giá trị*. Hơn nữa, ký hiệu $\mathbf{q}$ là một *truy vấn*. Khi đó, ta có thể định nghĩa *attention* trên $\mathcal{D}$ là

$$\textrm{Attention}(\mathbf{q}, \mathcal{D}) \stackrel{\textrm{def}}{=} \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i,$$
:eqlabel:`eq_attention_pooling`

trong đó $\alpha(\mathbf{q}, \mathbf{k}_i) \in \mathbb{R}$ ($i = 1, \ldots, m$) là các trọng số attention vô hướng. Bản thân phép toán này thường được gọi là *attention pooling*. Tên gọi *attention* xuất phát từ thực tế rằng phép toán này đặc biệt chú ý đến các hạng có trọng số $\alpha$ đáng kể (tức là lớn). Do đó, attention trên $\mathcal{D}$ tạo ra một tổ hợp tuyến tính của các giá trị có trong cơ sở dữ liệu. Trên thực tế, điều này bao gồm ví dụ trên như một trường hợp đặc biệt, trong đó tất cả trừ một trọng số đều bằng không. Ta có một số trường hợp đặc biệt:

* Các trọng số $\alpha(\mathbf{q}, \mathbf{k}_i)$ không âm. Trong trường hợp này, đầu ra của cơ chế attention nằm trong nón lồi được sinh bởi các giá trị $\mathbf{v}_i$.
* Các trọng số $\alpha(\mathbf{q}, \mathbf{k}_i)$ tạo thành một tổ hợp lồi, tức là $\sum_i \alpha(\mathbf{q}, \mathbf{k}_i) = 1$ và $\alpha(\mathbf{q}, \mathbf{k}_i) \geq 0$ với mọi $i$. Đây là thiết lập phổ biến nhất trong deep learning.
* Chính xác một trong các trọng số $\alpha(\mathbf{q}, \mathbf{k}_i)$ bằng $1$, trong khi tất cả các trọng số khác bằng $0$. Điều này tương tự một truy vấn cơ sở dữ liệu truyền thống.
* Mọi trọng số đều bằng nhau, tức là $\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{1}{m}$ với mọi $i$. Điều này tương ứng với việc lấy trung bình trên toàn bộ cơ sở dữ liệu, còn được gọi là average pooling trong deep learning.

Một chiến lược phổ biến để đảm bảo tổng các trọng số bằng $1$ là chuẩn hóa chúng thông qua

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{{\sum_j} \alpha(\mathbf{q}, \mathbf{k}_j)}.$$

Đặc biệt, để bảo đảm các trọng số cũng không âm, ta có thể dùng phép lũy thừa. Điều này có nghĩa là giờ đây ta có thể chọn *bất kỳ* hàm nào $a(\mathbf{q}, \mathbf{k})$ rồi áp dụng phép toán softmax dùng cho các mô hình multinomial lên nó thông qua

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_j \exp(a(\mathbf{q}, \mathbf{k}_j))}. $$
:eqlabel:`eq_softmax_attention`

Phép toán này có sẵn trong mọi framework deep learning. Nó khả vi và gradient của nó không bao giờ triệt tiêu, tất cả đều là những tính chất mong muốn trong một mô hình. Tuy nhiên, lưu ý rằng cơ chế attention được giới thiệu ở trên không phải là lựa chọn duy nhất. Ví dụ, ta có thể thiết kế một mô hình attention không khả vi, có thể được huấn luyện bằng các phương pháp reinforcement learning :cite:`Mnih.Heess.Graves.ea.2014`. Như có thể dự đoán, việc huấn luyện một mô hình như vậy khá phức tạp. Do đó, phần lớn nghiên cứu attention hiện đại tuân theo framework được phác thảo trong :numref:`fig_qkv`. Vì vậy, chúng ta tập trung trình bày về họ cơ chế khả vi này.

![Cơ chế attention tính một tổ hợp tuyến tính trên các giá trị $\mathbf{v}_\mathit{i}$ thông qua attention pooling,
trong đó các trọng số được suy ra dựa trên độ tương thích giữa một truy vấn $\mathbf{q}$ và các khóa $\mathbf{k}_\mathit{i}$.](../img/qkv.svg)
:label:`fig_qkv`

Điều khá đáng chú ý là "mã" thực tế để thực thi trên tập khóa và giá trị, tức truy vấn, có thể rất ngắn gọn, dù không gian cần thao tác là đáng kể. Đây là một tính chất mong muốn đối với một network layer vì nó không đòi hỏi phải học quá nhiều tham số. Cũng thuận tiện không kém là attention có thể hoạt động trên các cơ sở dữ liệu lớn tùy ý mà không cần thay đổi cách thực hiện phép toán attention pooling.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input  n=2}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
```

## Trực quan hóa

Một trong những lợi ích của cơ chế attention là nó có thể khá trực quan, đặc biệt khi các trọng số không âm và có tổng bằng $1$. Trong trường hợp này, ta có thể *diễn giải* các trọng số lớn như một cách để mô hình chọn ra các thành phần liên quan. Mặc dù đây là một trực giác tốt, điều quan trọng cần nhớ là nó chỉ là một *trực giác*. Dù vậy, ta có thể muốn trực quan hóa tác động của nó lên tập khóa đã cho khi áp dụng nhiều truy vấn khác nhau. Hàm này sẽ hữu ích ở phần sau.

Vì vậy, chúng ta định nghĩa hàm `show_heatmaps`. Lưu ý rằng hàm này không nhận một ma trận (các trọng số attention) làm đầu vào, mà nhận một tensor có bốn trục, cho phép chứa một mảng gồm các truy vấn và trọng số khác nhau. Do đó, đầu vào `matrices` có hình dạng (số hàng để hiển thị, số cột để hiển thị, số truy vấn, số khóa). Điều này sẽ hữu ích ở phần sau khi chúng ta muốn trực quan hóa các cơ chế dùng để thiết kế Transformers.

```{.python .input  n=17}
%%tab all
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices."""
    d2l.use_svg_display()
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            if tab.selected('pytorch', 'mxnet', 'tensorflow'):
                pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if tab.selected('jax'):
                pcm = ax.imshow(matrix, cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
```

Để kiểm tra nhanh tính hợp lý, hãy trực quan hóa ma trận đơn vị, biểu diễn trường hợp
trọng số attention bằng $1$ chỉ khi truy vấn và khóa giống nhau.

```{.python .input  n=20}
%%tab all
attention_weights = d2l.reshape(d2l.eye(10), (1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

## Tóm tắt

Cơ chế attention cho phép chúng ta tổng hợp dữ liệu từ nhiều cặp (khóa, giá trị). Cho đến nay, phần thảo luận của chúng ta khá trừu tượng, chỉ mô tả một cách để pooling dữ liệu. Chúng ta vẫn chưa giải thích các truy vấn, khóa và giá trị bí ẩn đó có thể xuất phát từ đâu. Một chút trực giác có thể hữu ích ở đây: ví dụ, trong bối cảnh hồi quy, truy vấn có thể tương ứng với vị trí cần thực hiện hồi quy. Các khóa là những vị trí mà dữ liệu quá khứ được quan sát, còn các giá trị là chính các giá trị (hồi quy). Đây là cái gọi là bộ ước lượng Nadaraya--Watson :cite:`Nadaraya.1964,Watson.1964` mà chúng ta sẽ nghiên cứu trong phần tiếp theo.

Theo thiết kế, cơ chế attention cung cấp một phương thức kiểm soát *khả vi* để mạng nơ-ron có thể chọn các phần tử từ một tập hợp và xây dựng một tổng có trọng số tương ứng trên các biểu diễn.

## Bài tập

1. Giả sử bạn muốn triển khai lại các phép khớp (key, query) xấp xỉ như được dùng trong các cơ sở dữ liệu cổ điển, bạn sẽ chọn hàm attention nào? 
1. Giả sử rằng hàm attention được cho bởi $a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i$ và $\mathbf{k}_i = \mathbf{v}_i$ với $i = 1, \ldots, m$. Ký hiệu $p(\mathbf{k}_i; \mathbf{q})$ là phân phối xác suất trên các key khi sử dụng chuẩn hóa softmax trong :eqref:`eq_softmax_attention`. Hãy chứng minh rằng $\nabla_{\mathbf{q}} \mathop{\textrm{Attention}}(\mathbf{q}, \mathcal{D}) = \textrm{Cov}_{p(\mathbf{k}_i; \mathbf{q})}[\mathbf{k}_i]$.
1. Thiết kế một search engine khả vi bằng cơ chế attention. 
1. Xem lại thiết kế của Squeeze and Excitation Networks :cite:`Hu.Shen.Sun.2018` và diễn giải chúng dưới góc nhìn của cơ chế attention. 

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/1596)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/1592)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/1710)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18024)
:end_tab: