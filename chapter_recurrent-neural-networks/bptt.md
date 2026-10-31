# Lan truyền ngược theo thời gian
:label:`sec_bptt`

Nếu bạn đã hoàn thành các bài tập trong :numref:`sec_rnn-scratch`,
bạn sẽ thấy rằng cắt gradient là thiết yếu
để ngăn các gradient cực lớn thỉnh thoảng xuất hiện
làm mất ổn định quá trình huấn luyện.
Chúng tôi đã gợi ý rằng gradient bùng nổ
bắt nguồn từ việc lan truyền ngược qua các chuỗi dài.
Trước khi giới thiệu hàng loạt kiến trúc RNN hiện đại,
hãy xem xét kỹ hơn cách *lan truyền ngược*
hoạt động trong các mô hình chuỗi ở mức độ toán học chi tiết.
Hy vọng rằng phần thảo luận này sẽ làm rõ hơn
khái niệm về gradient *tiêu biến* và *bùng nổ*.
Nếu bạn nhớ phần thảo luận của chúng ta về lan truyền xuôi và lan truyền ngược
qua các đồ thị tính toán
khi giới thiệu MLP trong :numref:`sec_backprop`,
thì lan truyền xuôi trong RNN
sẽ tương đối dễ hiểu.
Việc áp dụng lan truyền ngược trong RNN
được gọi là *lan truyền ngược theo thời gian* :cite:`Werbos.1990`.
Quy trình này yêu cầu chúng ta mở rộng (hoặc trải phẳng)
đồ thị tính toán của một RNN
từng bước thời gian một.
RNN đã được trải phẳng về cơ bản
là một mạng nơ-ron truyền xuôi
với đặc tính đặc biệt
là cùng các tham số
được lặp lại xuyên suốt mạng đã trải phẳng,
xuất hiện ở mỗi bước thời gian.
Sau đó, cũng như trong mọi mạng nơ-ron truyền xuôi,
chúng ta có thể áp dụng quy tắc dây chuyền,
lan truyền ngược gradient qua mạng đã trải phẳng.
Gradient đối với mỗi tham số
phải được cộng dồn qua mọi vị trí
mà tham số đó xuất hiện trong mạng đã trải phẳng.
Việc xử lý kiểu chia sẻ trọng số này hẳn đã quen thuộc
từ các chương về mạng nơ-ron tích chập của chúng ta.


Các vấn đề phức tạp nảy sinh vì chuỗi
có thể khá dài.
Không hiếm khi làm việc với các chuỗi văn bản
gồm hơn một nghìn token.
Lưu ý rằng điều này gây ra vấn đề cả về
tính toán (quá nhiều bộ nhớ)
và tối ưu hóa (bất ổn định số học).
Đầu vào từ bước đầu tiên đi qua
hơn 1000 phép nhân ma trận trước khi đến đầu ra,
và cần thêm 1000 phép nhân ma trận nữa
để tính gradient.
Bây giờ, chúng ta sẽ phân tích những điều có thể xảy ra sai sót và
cách xử lý chúng trong thực tế.


## Phân tích Gradient trong RNN
:label:`subsec_bptt_analysis`

Chúng ta bắt đầu với một mô hình đơn giản hóa về cách RNN hoạt động.
Mô hình này bỏ qua các chi tiết cụ thể
về trạng thái ẩn và cách nó được cập nhật.
Ký hiệu toán học ở đây
không phân biệt rõ ràng
giữa vô hướng, vector và ma trận.
Chúng ta chỉ đang cố gắng xây dựng một số trực giác.
Trong mô hình đơn giản hóa này,
chúng ta ký hiệu $h_t$ là trạng thái ẩn,
$x_t$ là đầu vào, và $o_t$ là đầu ra
tại bước thời gian $t$.
Hãy nhớ lại các thảo luận của chúng ta trong
:numref:`subsec_rnn_w_hidden_states`
rằng đầu vào và trạng thái ẩn
có thể được ghép nối trước khi nhân
với một biến trọng số trong lớp ẩn.
Do đó, chúng ta dùng $w_\textrm{h}$ và $w_\textrm{o}$ để chỉ các trọng số
của lớp ẩn và lớp đầu ra tương ứng.
Kết quả là, các trạng thái ẩn và đầu ra
tại mỗi bước thời gian là

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_\textrm{h}),\\o_t &= g(h_t, w_\textrm{o}),\end{aligned}$$
:eqlabel:`eq_bptt_ht_ot`

trong đó $f$ và $g$ lần lượt là các phép biến đổi
của lớp ẩn và lớp đầu ra.
Do đó, chúng ta có một chuỗi các giá trị
$\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$ 
phụ thuộc lẫn nhau thông qua phép tính hồi quy.
Lan truyền xuôi khá đơn giản.
Tất cả những gì cần làm là lặp qua các bộ ba $(x_t, h_t, o_t)$ theo từng bước thời gian.
Sai khác giữa đầu ra $o_t$ và mục tiêu mong muốn $y_t$
sau đó được đánh giá bởi một hàm mục tiêu
trên toàn bộ $T$ bước thời gian như sau

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_\textrm{h}, w_\textrm{o}) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$



Đối với lan truyền ngược, mọi việc phức tạp hơn một chút,
đặc biệt khi chúng ta tính gradient
theo các tham số $w_\textrm{h}$ của hàm mục tiêu $L$.
Cụ thể, theo quy tắc dây chuyền,

$$\begin{aligned}\frac{\partial L}{\partial w_\textrm{h}}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_\textrm{h}}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_\textrm{o})}{\partial h_t}  \frac{\partial h_t}{\partial w_\textrm{h}}.\end{aligned}$$
:eqlabel:`eq_bptt_partial_L_wh`

Thừa số thứ nhất và thứ hai trong
tích ở :eqref:`eq_bptt_partial_L_wh`
dễ tính.
Thừa số thứ ba $\partial h_t/\partial w_\textrm{h}$ là nơi mọi việc trở nên phức tạp,
vì chúng ta cần tính hồi quy ảnh hưởng của tham số $w_\textrm{h}$ lên $h_t$.
Theo phép tính hồi quy
trong :eqref:`eq_bptt_ht_ot`,
$h_t$ phụ thuộc vào cả $h_{t-1}$ lẫn $w_\textrm{h}$,
trong đó việc tính $h_{t-1}$
cũng phụ thuộc vào $w_\textrm{h}$.
Vì vậy, việc đánh giá đạo hàm toàn phần của $h_t$
theo $w_\textrm{h}$ bằng quy tắc dây chuyền cho ta

$$\frac{\partial h_t}{\partial w_\textrm{h}}= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}} +\frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.$$
:eqlabel:`eq_bptt_partial_ht_wh_recur`


Để suy ra gradient ở trên, giả sử rằng chúng ta có
ba chuỗi $\{a_{t}\},\{b_{t}\},\{c_{t}\}$
thỏa mãn $a_{0}=0$ và $a_{t}=b_{t}+c_{t}a_{t-1}$ với $t=1, 2,\ldots$.
Khi đó, với $t\geq 1$, dễ dàng chứng minh rằng

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$
:eqlabel:`eq_bptt_at`

Bằng cách thay $a_t$, $b_t$, và $c_t$ theo

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_\textrm{h}},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}},\end{aligned}$$

phép tính gradient trong :eqref:`eq_bptt_partial_ht_wh_recur` thỏa mãn
$a_{t}=b_{t}+c_{t}a_{t-1}$.
Do đó, theo :eqref:`eq_bptt_at`,
chúng ta có thể loại bỏ phép tính hồi quy
trong :eqref:`eq_bptt_partial_ht_wh_recur` bằng

$$\frac{\partial h_t}{\partial w_\textrm{h}}=\frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_\textrm{h})}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_\textrm{h})}{\partial w_\textrm{h}}.$$
:eqlabel:`eq_bptt_partial_ht_wh_gen`

Mặc dù có thể dùng quy tắc dây chuyền để tính $\partial h_t/\partial w_\textrm{h}$ một cách đệ quy,
chuỗi này có thể trở nên rất dài khi $t$ lớn.
Hãy thảo luận một số chiến lược để xử lý vấn đề này.

### Tính toán đầy đủ ### 

Một ý tưởng có thể là tính toàn bộ tổng trong :eqref:`eq_bptt_partial_ht_wh_gen`.
Tuy nhiên, cách này rất chậm và gradient có thể bùng nổ,
vì những thay đổi nhỏ trong điều kiện ban đầu
có khả năng ảnh hưởng rất lớn đến kết quả.
Nói cách khác, chúng ta có thể thấy hiện tượng tương tự hiệu ứng cánh bướm,
trong đó những thay đổi tối thiểu ở điều kiện ban đầu
dẫn đến những thay đổi không tương xứng ở kết quả.
Điều này thường không mong muốn.
Xét cho cùng, chúng ta đang tìm kiếm các bộ ước lượng vững chắc có khả năng khái quát tốt.
Do đó, chiến lược này gần như không bao giờ được sử dụng trong thực tế.

### Cắt ngắn các bước thời gian###

Một cách khác,
chúng ta có thể cắt ngắn tổng trong
:eqref:`eq_bptt_partial_ht_wh_gen`
sau $\tau$ bước.
Đây là điều chúng ta đã thảo luận cho đến nay.
Điều này dẫn đến một *xấp xỉ* của gradient thực,
đơn giản bằng cách kết thúc tổng tại $\partial h_{t-\tau}/\partial w_\textrm{h}$.
Trong thực tế, cách này hoạt động khá tốt.
Đây là phương pháp thường được gọi là
lan truyền ngược rút gọn theo thời gian :cite:`Jaeger.2002`.
Một trong các hệ quả là mô hình
chủ yếu tập trung vào ảnh hưởng ngắn hạn
thay vì các hệ quả dài hạn.
Điều này thực ra là *mong muốn*, vì nó làm thiên lệch ước lượng
về phía các mô hình đơn giản và ổn định hơn.


### Cắt ngắn ngẫu nhiên ### 

Cuối cùng, chúng ta có thể thay $\partial h_t/\partial w_\textrm{h}$
bằng một biến ngẫu nhiên đúng theo kỳ vọng
nhưng cắt ngắn chuỗi.
Điều này đạt được bằng cách sử dụng một chuỗi $\xi_t$
với $0 \leq \pi_t \leq 1$ được xác định trước,
trong đó $P(\xi_t = 0) = 1-\pi_t$ và
$P(\xi_t = \pi_t^{-1}) = \pi_t$, do đó $E[\xi_t] = 1$.
Chúng ta dùng điều này để thay gradient
$\partial h_t/\partial w_\textrm{h}$
trong :eqref:`eq_bptt_partial_ht_wh_recur`
bằng

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.$$


Từ định nghĩa của $\xi_t$
suy ra rằng $E[z_t] = \partial h_t/\partial w_\textrm{h}$.
Bất cứ khi nào $\xi_t = 0$, phép tính hồi quy
sẽ kết thúc tại bước thời gian $t$ đó.
Điều này dẫn đến một tổng có trọng số của các chuỗi với độ dài khác nhau,
trong đó các chuỗi dài hiếm gặp nhưng được gán trọng số lớn hơn một cách phù hợp.
Ý tưởng này được đề xuất bởi
:citet:`Tallec.Ollivier.2017`.

### So sánh các chiến lược

![So sánh các chiến lược tính gradient trong RNN. Từ trên xuống dưới: cắt ngắn ngẫu nhiên, cắt ngắn thông thường và tính toán đầy đủ.](../img/truncated-bptt.svg)
:label:`fig_truncated_bptt`


:numref:`fig_truncated_bptt` minh họa ba chiến lược này 
khi phân tích vài ký tự đầu tiên của *The Time Machine* 
bằng phương pháp lan truyền ngược theo thời gian cho RNN:

* Hàng đầu tiên là cắt ngắn ngẫu nhiên, phân chia văn bản thành các đoạn có độ dài khác nhau.
* Hàng thứ hai là cắt ngắn thông thường, chia văn bản thành các dãy con có cùng độ dài. Đây là phương pháp chúng ta đã sử dụng trong các thí nghiệm RNN.
* Hàng thứ ba là lan truyền ngược theo thời gian đầy đủ, dẫn đến một biểu thức không khả thi về mặt tính toán.


Đáng tiếc là, mặc dù hấp dẫn về mặt lý thuyết, 
cắt ngắn ngẫu nhiên không hoạt động 
tốt hơn nhiều so với cắt ngắn thông thường, 
nhiều khả năng do một số yếu tố.
Thứ nhất, ảnh hưởng của một quan sát
sau một số bước lan truyền ngược 
về quá khứ là khá đủ 
để nắm bắt các phụ thuộc trong thực tế. 
Thứ hai, phương sai tăng lên triệt tiêu lợi ích 
của việc gradient chính xác hơn khi có nhiều bước hơn. 
Thứ ba, chúng ta thực sự *muốn* các mô hình chỉ có 
phạm vi tương tác ngắn. 
Do đó, lan truyền ngược theo thời gian được cắt ngắn thường xuyên 
có tác dụng regularization nhẹ, điều này có thể là mong muốn.

## Lan truyền ngược theo thời gian chi tiết

Sau khi thảo luận về nguyên lý chung,
hãy cùng tìm hiểu chi tiết về lan truyền ngược theo thời gian.
Khác với phần phân tích trong :numref:`subsec_bptt_analysis`,
sau đây chúng ta sẽ trình bày cách tính
gradient của hàm mục tiêu
đối với tất cả các tham số mô hình đã phân tách.
Để đơn giản, chúng ta xét 
một RNN không có tham số bias,
trong đó hàm kích hoạt ở lớp ẩn
sử dụng ánh xạ đồng nhất ($\phi(x)=x$).
Tại bước thời gian $t$, đầu vào của một mẫu đơn 
và mục tiêu lần lượt là $\mathbf{x}_t \in \mathbb{R}^d$ và $y_t$. 
Trạng thái ẩn $\mathbf{h}_t \in \mathbb{R}^h$ 
và đầu ra $\mathbf{o}_t \in \mathbb{R}^q$
được tính như sau

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_\textrm{hx} \mathbf{x}_t + \mathbf{W}_\textrm{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_\textrm{qh} \mathbf{h}_{t},\end{aligned}$$

trong đó $\mathbf{W}_\textrm{hx} \in \mathbb{R}^{h \times d}$, $\mathbf{W}_\textrm{hh} \in \mathbb{R}^{h \times h}$ và
$\mathbf{W}_\textrm{qh} \in \mathbb{R}^{q \times h}$
là các tham số trọng số.
Ký hiệu $l(\mathbf{o}_t, y_t)$
là loss tại bước thời gian $t$. 
Hàm mục tiêu của chúng ta,
tức loss trên $T$ bước thời gian
tính từ đầu chuỗi, do đó là

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$


Để trực quan hóa các phụ thuộc giữa
các biến và tham số mô hình trong quá trình tính toán
của RNN,
chúng ta có thể vẽ computational graph cho mô hình,
như trong :numref:`fig_rnn_bptt`.
Ví dụ, việc tính trạng thái ẩn ở bước thời gian 3,
$\mathbf{h}_3$, phụ thuộc vào các tham số mô hình
$\mathbf{W}_\textrm{hx}$ và $\mathbf{W}_\textrm{hh}$,
trạng thái ẩn của bước thời gian trước đó $\mathbf{h}_2$,
và đầu vào tại bước thời gian hiện tại $\mathbf{x}_3$.

![Computational graph cho thấy các phụ thuộc của một mô hình RNN với ba bước thời gian. Các hộp biểu diễn biến (không tô bóng) hoặc tham số (tô bóng), còn các hình tròn biểu diễn toán tử.](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt`

Như vừa đề cập, các tham số mô hình trong :numref:`fig_rnn_bptt` 
là $\mathbf{W}_\textrm{hx}$, $\mathbf{W}_\textrm{hh}$ và $\mathbf{W}_\textrm{qh}$. 
Nói chung, việc huấn luyện mô hình này đòi hỏi 
tính gradient đối với các tham số này,
$\partial L/\partial \mathbf{W}_\textrm{hx}$, $\partial L/\partial \mathbf{W}_\textrm{hh}$ và $\partial L/\partial \mathbf{W}_\textrm{qh}$.
Theo các phụ thuộc trong :numref:`fig_rnn_bptt`,
chúng ta có thể duyệt theo chiều ngược lại với các mũi tên
để lần lượt tính và lưu các gradient.
Để biểu diễn linh hoạt phép nhân của 
ma trận, vector và vô hướng có các hình dạng khác nhau
trong quy tắc dây chuyền,
chúng ta tiếp tục sử dụng toán tử $\textrm{prod}$ 
như mô tả trong :numref:`sec_backprop`.


Trước hết, việc lấy đạo hàm của hàm mục tiêu
đối với đầu ra mô hình tại bất kỳ bước thời gian $t$ nào
khá đơn giản:

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$
:eqlabel:`eq_bptt_partial_L_ot`

Bây giờ chúng ta có thể tính gradient của hàm mục tiêu 
đối với tham số $\mathbf{W}_\textrm{qh}$
ở lớp đầu ra:
$\partial L/\partial \mathbf{W}_\textrm{qh} \in \mathbb{R}^{q \times h}$. 
Dựa trên :numref:`fig_rnn_bptt`, 
mục tiêu $L$ phụ thuộc vào $\mathbf{W}_\textrm{qh}$ 
thông qua $\mathbf{o}_1, \ldots, \mathbf{o}_T$. 
Áp dụng quy tắc dây chuyền, ta có

$$
\frac{\partial L}{\partial \mathbf{W}_\textrm{qh}}
= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_\textrm{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

trong đó $\partial L/\partial \mathbf{o}_t$
được cho bởi :eqref:`eq_bptt_partial_L_ot`.

Tiếp theo, như minh họa trong :numref:`fig_rnn_bptt`,
tại bước thời gian cuối cùng $T$,
hàm mục tiêu
$L$ chỉ phụ thuộc vào trạng thái ẩn $\mathbf{h}_T$ 
thông qua $\mathbf{o}_T$.
Do đó, chúng ta có thể dễ dàng tìm gradient 
$\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$
bằng quy tắc dây chuyền:

$$\frac{\partial L}{\partial \mathbf{h}_T} = \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$
:eqlabel:`eq_bptt_partial_L_hT_final_step`

Mọi thứ trở nên phức tạp hơn tại bất kỳ bước thời gian nào $t < T$,
khi hàm mục tiêu $L$ phụ thuộc vào 
$\mathbf{h}_t$ thông qua $\mathbf{h}_{t+1}$ và $\mathbf{o}_t$.
Theo quy tắc dây chuyền,
gradient của trạng thái ẩn
$\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$
tại bất kỳ bước thời gian $t < T$ nào có thể được tính đệ quy như sau:


$$\frac{\partial L}{\partial \mathbf{h}_t} = \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_\textrm{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$
:eqlabel:`eq_bptt_partial_L_ht_recur`

Để phân tích, khai triển phép tính đệ quy
cho bất kỳ bước thời gian nào $1 \leq t \leq T$ cho ta

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_\textrm{hh}^\top\right)}^{T-i} \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$
:eqlabel:`eq_bptt_partial_L_ht`

Từ :eqref:`eq_bptt_partial_L_ht`, chúng ta có thể thấy 
rằng ví dụ tuyến tính đơn giản này đã
bộc lộ một số vấn đề then chốt của mô hình chuỗi dài:
nó liên quan đến các lũy thừa có thể rất lớn của $\mathbf{W}_\textrm{hh}^\top$.
Trong đó, các giá trị riêng nhỏ hơn 1 sẽ tiêu biến
và các giá trị riêng lớn hơn 1 sẽ phân kỳ.
Điều này không ổn định về mặt số học,
biểu hiện dưới dạng gradient tiêu biến 
và bùng nổ.
Một cách để giải quyết vấn đề này là cắt ngắn các bước thời gian
ở một kích thước thuận tiện về mặt tính toán 
như đã thảo luận trong :numref:`subsec_bptt_analysis`. 
Trên thực tế, việc cắt ngắn này cũng có thể được thực hiện 
bằng cách detach gradient sau một số bước thời gian nhất định.
Sau đây, chúng ta sẽ thấy cách các mô hình chuỗi tinh vi hơn, 
chẳng hạn như long short-term memory, có thể giảm nhẹ vấn đề này hơn nữa. 

Cuối cùng, :numref:`fig_rnn_bptt` cho thấy
rằng hàm mục tiêu $L$
phụ thuộc vào các tham số mô hình $\mathbf{W}_\textrm{hx}$ và $\mathbf{W}_\textrm{hh}$
trong hidden layer thông qua các hidden state
$\mathbf{h}_1, \ldots, \mathbf{h}_T$.
Để tính gradient theo các tham số đó
$\partial L / \partial \mathbf{W}_\textrm{hx} \in \mathbb{R}^{h \times d}$ và $\partial L / \partial \mathbf{W}_\textrm{hh} \in \mathbb{R}^{h \times h}$,
chúng ta áp dụng quy tắc dây chuyền, thu được

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_\textrm{hx}}
&= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_\textrm{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_\textrm{hh}}
&= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_\textrm{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$

trong đó $\partial L/\partial \mathbf{h}_t$,
được tính đệ quy bởi
:eqref:`eq_bptt_partial_L_hT_final_step`
và :eqref:`eq_bptt_partial_L_ht_recur`,
là đại lượng then chốt ảnh hưởng đến tính ổn định số.

Vì backpropagation through time là việc áp dụng backpropagation trong RNN,
như chúng ta đã giải thích trong :numref:`sec_backprop`,
việc huấn luyện RNN xen kẽ forward propagation với
backpropagation through time.
Hơn nữa, backpropagation through time
lần lượt tính toán và lưu trữ các gradient nêu trên.
Cụ thể, các giá trị trung gian đã lưu trữ
được tái sử dụng để tránh các phép tính trùng lặp,
chẳng hạn như lưu $\partial L/\partial \mathbf{h}_t$
để dùng khi tính cả $\partial L / \partial \mathbf{W}_\textrm{hx}$
và $\partial L / \partial \mathbf{W}_\textrm{hh}$.


## Tóm tắt

Backpropagation through time chỉ đơn thuần là việc áp dụng backpropagation cho các mô hình tuần tự có hidden state.
Cắt ngắn, chẳng hạn như theo cách đều đặn hoặc ngẫu nhiên, là cần thiết để thuận tiện cho tính toán và đảm bảo tính ổn định số.
Lũy thừa bậc cao của ma trận có thể dẫn đến các eigenvalue phân kỳ hoặc triệt tiêu. Điều này biểu hiện dưới dạng gradient bùng nổ hoặc triệt tiêu.
Để tính toán hiệu quả, các giá trị trung gian được cache trong quá trình backpropagation through time.



## Bài tập

1. Giả sử chúng ta có một ma trận đối xứng $\mathbf{M} \in \mathbb{R}^{n \times n}$ với các eigenvalue $\lambda_i$ có các eigenvector tương ứng là $\mathbf{v}_i$ ($i = 1, \ldots, n$). Không mất tính tổng quát, giả sử rằng chúng được sắp xếp theo thứ tự $|\lambda_i| \geq |\lambda_{i+1}|$. 
   1. Chứng minh rằng $\mathbf{M}^k$ có các eigenvalue $\lambda_i^k$.
   1. Chứng minh rằng với một vector ngẫu nhiên $\mathbf{x} \in \mathbb{R}^n$, với xác suất cao $\mathbf{M}^k \mathbf{x}$ sẽ gần như thẳng hàng với eigenvector $\mathbf{v}_1$
của $\mathbf{M}$. Hãy phát biểu kết quả này một cách chính xác.
   1. Kết quả trên có ý nghĩa gì đối với gradient trong RNN?
1. Ngoài gradient clipping, bạn có thể nghĩ đến phương pháp nào khác để xử lý gradient bùng nổ trong recurrent neural network không?

[Thảo luận](https://discuss.d2l.ai/t/334)