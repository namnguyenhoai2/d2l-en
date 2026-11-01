# Tìm kiếm beam
:label:`sec_beam-search`

Trong :numref:`sec_seq2seq`, 
chúng ta đã giới thiệu kiến trúc encoder--decoder
và các kỹ thuật tiêu chuẩn để huấn luyện chúng theo phương thức end-to-end. Tuy nhiên, khi đề cập đến việc dự đoán trong lúc kiểm thử,
chúng ta chỉ nói đến chiến lược *greedy*,
trong đó tại mỗi bước thời gian,
chúng ta chọn token có
xác suất dự đoán cao nhất
cho token tiếp theo,
cho đến khi tại một bước thời gian nào đó,
chúng ta nhận thấy mình đã dự đoán
token đặc biệt kết thúc chuỗi "&lt;eos&gt;".
Trong phần này, trước tiên
chúng ta sẽ hình thức hóa chiến lược *greedy search* này
và xác định một số vấn đề
mà người thực hành thường gặp phải.
Sau đó, chúng ta sẽ so sánh chiến lược này
với hai phương án khác:
*tìm kiếm vét cạn* (mang tính minh họa nhưng không thực tế)
và *beam search* (phương pháp tiêu chuẩn trong thực tế).

Hãy bắt đầu bằng cách thiết lập ký hiệu toán học,
mượn các quy ước từ :numref:`sec_seq2seq`.
Tại bất kỳ bước thời gian $t'$ nào, decoder xuất ra
các dự đoán biểu diễn xác suất
của mỗi token trong vocabulary
sẽ xuất hiện tiếp theo trong chuỗi
(giá trị có khả năng là $y_{t'+1}$),
với điều kiện là các token trước đó
$y_1, \ldots, y_{t'}$ và
biến ngữ cảnh $\mathbf{c}$,
được encoder tạo ra
để biểu diễn chuỗi đầu vào.
Để định lượng chi phí tính toán,
ký hiệu $\mathcal{Y}$
là vocabulary đầu ra
(bao gồm token đặc biệt kết thúc chuỗi "&lt;eos&gt;").
Ta cũng quy định số token tối đa
của một chuỗi đầu ra là $T'$.
Mục tiêu của chúng ta là tìm kiếm một đầu ra lý tưởng trong tất cả
$\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$
chuỗi đầu ra khả dĩ.
Lưu ý rằng con số này hơi lớn hơn
số lượng đầu ra khác nhau
vì không có token nào tiếp theo
sau khi token "&lt;eos&gt;" xuất hiện.
Tuy nhiên, với mục đích của chúng ta,
con số này gần đúng với
quy mô của không gian tìm kiếm.

## Tìm kiếm greedy

Xét chiến lược *greedy search* đơn giản từ :numref:`sec_seq2seq`.
Tại đây, ở bất kỳ bước thời gian $t'$ nào,
chúng ta chỉ cần chọn token
có xác suất có điều kiện cao nhất
từ $\mathcal{Y}$, tức là,

$$y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c}).$$

Khi model xuất ra "&lt;eos&gt;"
(hoặc chúng ta đạt đến độ dài tối đa $T'$)
thì chuỗi đầu ra hoàn tất.

Chiến lược này có vẻ hợp lý,
và thực tế thì nó cũng không tệ!
Xét đến chi phí tính toán rất thấp của nó,
bạn sẽ khó tìm được phương án nào hiệu quả hơn với cùng mức tài nguyên.
Tuy nhiên, nếu tạm gác hiệu quả sang một bên,
có vẻ hợp lý hơn khi tìm kiếm
*chuỗi có khả năng cao nhất*,
thay vì chuỗi gồm các *token có khả năng cao nhất*
(được chọn một cách greedy).
Hóa ra hai đối tượng này có thể rất khác nhau.
Chuỗi có khả năng cao nhất là chuỗi tối đa hóa biểu thức
$\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$.
Trong ví dụ dịch máy của chúng ta,
nếu decoder thực sự khôi phục được các xác suất
của quá trình sinh nền tảng,
thì đây sẽ là bản dịch có khả năng cao nhất.
Đáng tiếc là không có gì đảm bảo
greedy search sẽ tạo ra chuỗi này.

Hãy minh họa bằng một ví dụ.
Giả sử từ điển đầu ra có bốn token
"A", "B", "C" và "&lt;eos&gt;".
Trong :numref:`fig_s2s-prob1`,
bốn con số dưới mỗi bước thời gian biểu diễn
xác suất có điều kiện để sinh ra "A", "B", "C"
và "&lt;eos&gt;" tương ứng tại bước thời gian đó.

![Tại mỗi bước thời gian, greedy search chọn token có xác suất có điều kiện cao nhất.](../img/s2s-prob1.svg)
:label:`fig_s2s-prob1`

Tại mỗi bước thời gian, greedy search chọn
token có xác suất có điều kiện cao nhất.
Do đó, chuỗi đầu ra "A", "B", "C" và "&lt;eos&gt;"
sẽ được dự đoán (:numref:`fig_s2s-prob1`).
Xác suất có điều kiện của chuỗi đầu ra này
là $0.5\times0.4\times0.4\times0.6 = 0.048$.


Tiếp theo, hãy xem một ví dụ khác trong :numref:`fig_s2s-prob2`.
Khác với :numref:`fig_s2s-prob1`,
tại bước thời gian 2, chúng ta chọn token "C",
token có xác suất có điều kiện cao thứ *hai*.

![Bốn con số dưới mỗi bước thời gian biểu diễn
xác suất có điều kiện để sinh ra "A", "B", "C" và "&lt;eos&gt;" tại bước thời gian đó.
Tại bước thời gian 2, token "C", có xác suất có điều kiện cao thứ hai,
được chọn.](../img/s2s-prob2.svg)
:label:`fig_s2s-prob2`

Vì các chuỗi con đầu ra tại bước thời gian 1 và 2,
mà bước thời gian 3 dựa vào,
đã thay đổi từ "A" và "B" trong :numref:`fig_s2s-prob1`
thành "A" và "C" trong :numref:`fig_s2s-prob2`,
nên xác suất có điều kiện của mỗi token
tại bước thời gian 3 cũng đã thay đổi trong :numref:`fig_s2s-prob2`.
Giả sử chúng ta chọn token "B" tại bước thời gian 3.
Khi đó, bước thời gian 4 phụ thuộc vào
chuỗi con đầu ra tại ba bước thời gian đầu tiên
"A", "C" và "B",
đã thay đổi so với "A", "B" và "C" trong :numref:`fig_s2s-prob1`.
Do đó, xác suất có điều kiện để sinh ra
mỗi token tại bước thời gian 4 trong :numref:`fig_s2s-prob2`
cũng khác với xác suất trong :numref:`fig_s2s-prob1`.
Kết quả là, xác suất có điều kiện của chuỗi đầu ra
"A", "C", "B" và "&lt;eos&gt;" trong :numref:`fig_s2s-prob2`
là $0.5\times0.3 \times0.6\times0.6=0.054$,
lớn hơn xác suất của greedy search trong :numref:`fig_s2s-prob1`.
Trong ví dụ này, chuỗi đầu ra "A", "B", "C" và "&lt;eos&gt;"
thu được bằng greedy search không phải là chuỗi tối ưu.





## Tìm kiếm vét cạn

Nếu mục tiêu là thu được chuỗi có khả năng cao nhất,
chúng ta có thể cân nhắc sử dụng *tìm kiếm vét cạn*:
liệt kê tất cả các chuỗi đầu ra khả dĩ
cùng với xác suất có điều kiện của chúng,
sau đó xuất ra chuỗi có
xác suất dự đoán cao nhất.


Mặc dù cách này chắc chắn sẽ cho chúng ta kết quả mong muốn,
nó phải trả giá bằng chi phí tính toán quá lớn
là $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$,
tăng theo cấp số mũ với độ dài chuỗi và có
cơ số rất lớn do kích thước vocabulary.
Ví dụ, khi $|\mathcal{Y}|=10000$ và $T'=10$,
cả hai đều là những con số nhỏ khi so với các ứng dụng thực tế, chúng ta sẽ phải đánh giá $10000^{10} = 10^{40}$ chuỗi, con số này đã vượt quá khả năng của mọi máy tính có thể dự đoán trong tương lai gần.
Mặt khác, chi phí tính toán của greedy search là
$\mathcal{O}(\left|\mathcal{Y}\right|T')$:
rẻ một cách đáng kinh ngạc nhưng khác xa tối ưu.
Ví dụ, khi $|\mathcal{Y}|=10000$ và $T'=10$,
chúng ta chỉ cần đánh giá $10000\times10=10^5$ chuỗi.


## Beam search

Bạn có thể hình dung các chiến lược giải mã chuỗi nằm trên một dải liên tục,
trong đó *beam search* tạo ra sự cân bằng
giữa hiệu quả của greedy search
và tính tối ưu của tìm kiếm vét cạn.
Phiên bản đơn giản nhất của beam search
được đặc trưng bởi một hyperparameter duy nhất,
gọi là *beam size*, $k$.
Hãy giải thích thuật ngữ này.
Tại bước thời gian 1, chúng ta chọn $k$ token
có xác suất dự đoán cao nhất.
Mỗi token trong số đó sẽ lần lượt là token đầu tiên của
$k$ chuỗi đầu ra ứng viên.
Tại mỗi bước thời gian tiếp theo,
dựa trên $k$ chuỗi đầu ra ứng viên
ở bước thời gian trước,
chúng ta tiếp tục chọn $k$ chuỗi đầu ra ứng viên
có xác suất dự đoán cao nhất
từ $k\left|\mathcal{Y}\right|$ lựa chọn khả dĩ.

![Quy trình beam search (beam size $=2$; độ dài tối đa của chuỗi đầu ra $=3$). Các chuỗi đầu ra ứng viên là $\mathit{A}$, $\mathit{C}$, $\mathit{AB}$, $\mathit{CE}$, $\mathit{ABD}$ và $\mathit{CED}$.](../img/beam-search.svg)
:label:`fig_beam-search`


:numref:`fig_beam-search` minh họa
quy trình beam search bằng một ví dụ.
Giả sử vocabulary đầu ra
chỉ chứa năm phần tử:
$\mathcal{Y} = \{A, B, C, D, E\}$,
trong đó một phần tử là “&lt;eos&gt;”.
Giả sử beam size là hai và
độ dài tối đa của chuỗi đầu ra là ba.
Tại bước thời gian 1,
giả sử các token có xác suất có điều kiện cao nhất
$P(y_1 \mid \mathbf{c})$ là $A$ và $C$.
Tại bước thời gian 2, với mọi $y_2 \in \mathcal{Y}$,
chúng ta tính

$$\begin{aligned}P(A, y_2 \mid \mathbf{c}) = P(A \mid \mathbf{c})P(y_2 \mid A, \mathbf{c}),\\ P(C, y_2 \mid \mathbf{c}) = P(C \mid \mathbf{c})P(y_2 \mid C, \mathbf{c}),\end{aligned}$$  

và chọn hai giá trị lớn nhất trong mười giá trị này, chẳng hạn
$P(A, B \mid \mathbf{c})$ và $P(C, E \mid \mathbf{c})$.
Sau đó, tại bước thời gian 3, với mọi $y_3 \in \mathcal{Y}$, chúng ta tính

$$\begin{aligned}P(A, B, y_3 \mid \mathbf{c}) = P(A, B \mid \mathbf{c})P(y_3 \mid A, B, \mathbf{c}),\\P(C, E, y_3 \mid \mathbf{c}) = P(C, E \mid \mathbf{c})P(y_3 \mid C, E, \mathbf{c}),\end{aligned}$$ 

và chọn hai giá trị lớn nhất trong số mười giá trị này, chẳng hạn là  
$P(A, B, D \mid \mathbf{c})$   và  $P(C, E, D \mid  \mathbf{c}).$
Kết quả là, chúng ta nhận được sáu chuỗi đầu ra ứng viên:  
(i) $A$; (ii) $C$; (iii) $A$, $B$; (iv) $C$, $E$; (v) $A$, $B$, $D$; và (vi) $C$, $E$, $D$. 


Cuối cùng, chúng ta thu được tập hợp các chuỗi đầu ra ứng viên cuối cùng
dựa trên sáu chuỗi này (ví dụ: loại bỏ các phần bao gồm và nằm sau “&lt;eos&gt;”).
Sau đó, chúng ta chọn chuỗi đầu ra tối đa hóa điểm số sau:

$$ \frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}\mid \mathbf{c}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c});$$
:eqlabel:`eq_beam-search-score`

trong đó $L$ là độ dài của chuỗi ứng viên cuối cùng
và $\alpha$ thường được đặt bằng 0.75. 
Vì một chuỗi dài hơn có nhiều số hạng logarit hơn
trong tổng của :eqref:`eq_beam-search-score`,
số hạng $L^\alpha$ ở mẫu số sẽ phạt
các chuỗi dài.

Chi phí tính toán của beam search là $\mathcal{O}(k\left|\mathcal{Y}\right|T')$. 
Kết quả này nằm giữa kết quả của greedy search và exhaustive search.
Greedy search có thể được xem là một trường hợp đặc biệt của beam search
khi kích thước beam được đặt bằng 1.




## Tóm tắt

Các chiến lược tìm kiếm chuỗi bao gồm
greedy search, exhaustive search và beam search.
Beam search tạo ra sự cân bằng giữa độ chính xác và
chi phí tính toán thông qua việc lựa chọn linh hoạt kích thước beam.


## Bài tập

1. Chúng ta có thể xem exhaustive search là một dạng đặc biệt của beam search không? Tại sao hoặc tại sao không?
1. Áp dụng beam search vào bài toán dịch máy trong :numref:`sec_seq2seq`. Kích thước beam ảnh hưởng như thế nào đến kết quả dịch và tốc độ dự đoán?
1. Chúng ta đã sử dụng language modeling để sinh văn bản tiếp nối các tiền tố do người dùng cung cấp trong :numref:`sec_rnn-scratch`. Nó sử dụng loại chiến lược tìm kiếm nào? Bạn có thể cải thiện nó không?

[Thảo luận](https://discuss.d2l.ai/t/338)