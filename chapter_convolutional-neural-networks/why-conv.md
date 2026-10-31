# Từ các lớp Fully Connected đến Convolution
:label:`sec_why-conv`

Cho đến ngày nay,
các mô hình mà chúng ta đã thảo luận cho đến nay
vẫn là những lựa chọn phù hợp
khi làm việc với dữ liệu dạng bảng.
Theo dạng bảng, chúng tôi muốn nói dữ liệu bao gồm
các hàng tương ứng với các ví dụ
và các cột tương ứng với các đặc trưng.
Với dữ liệu dạng bảng, chúng ta có thể dự đoán
rằng các mẫu cần tìm có thể liên quan đến
sự tương tác giữa các đặc trưng,
nhưng không giả định bất kỳ cấu trúc *a priori* nào
về cách các đặc trưng tương tác.

Đôi khi, chúng ta thực sự thiếu kiến thức để có thể định hướng việc xây dựng các kiến trúc phức tạp hơn.
Trong những trường hợp này, một MLP
có thể là điều tốt nhất mà chúng ta có thể làm.
Tuy nhiên, đối với dữ liệu cảm nhận có số chiều cao,
các mạng không có cấu trúc như vậy có thể trở nên cồng kềnh.

Ví dụ, hãy quay lại ví dụ xuyên suốt của chúng ta
về việc phân biệt mèo và chó.
Giả sử chúng ta thực hiện kỹ lưỡng việc thu thập dữ liệu,
thu thập một bộ dữ liệu ảnh có chú thích với độ phân giải một megapixel.
Điều này có nghĩa là mỗi đầu vào của mạng có một triệu chiều.
Ngay cả khi giảm mạnh xuống còn một nghìn chiều ẩn
cũng sẽ cần một lớp fully connected
được đặc trưng bởi $10^6 \times 10^3 = 10^9$ tham số.
Trừ khi chúng ta có rất nhiều GPU, có năng khiếu
về tối ưu hóa phân tán,
và một lượng kiên nhẫn phi thường,
việc học các tham số của mạng này
có thể trở nên bất khả thi.

Một độc giả cẩn thận có thể phản đối lập luận này
dựa trên việc độ phân giải một megapixel có thể không cần thiết.
Tuy nhiên, mặc dù chúng ta có thể
tạm chấp nhận một trăm nghìn pixel,
lớp ẩn có kích thước 1000 của chúng ta đã đánh giá quá thấp nghiêm trọng
số lượng đơn vị ẩn cần thiết
để học các biểu diễn tốt của ảnh,
vì vậy một hệ thống thực tế vẫn sẽ cần hàng tỷ tham số.
Hơn nữa, việc học một bộ phân loại bằng cách khớp nhiều tham số như vậy
có thể đòi hỏi phải thu thập một bộ dữ liệu khổng lồ.
Thế nhưng ngày nay, cả con người lẫn máy tính đều có thể
phân biệt mèo với chó khá tốt,
dường như mâu thuẫn với những trực giác này.
Đó là bởi vì ảnh thể hiện cấu trúc phong phú
mà cả con người
và các mô hình machine learning đều có thể khai thác.
Mạng nơ-ron tích chập (convolutional neural networks, CNNs) là một cách sáng tạo
mà machine learning đã áp dụng để khai thác
một phần cấu trúc đã biết trong các ảnh tự nhiên.


## Tính bất biến

Hãy tưởng tượng rằng chúng ta muốn phát hiện một đối tượng trong ảnh.
Có vẻ hợp lý khi bất kỳ phương pháp nào
chúng ta dùng để nhận diện đối tượng cũng không nên quá quan tâm
đến vị trí chính xác của đối tượng trong ảnh.
Lý tưởng nhất, hệ thống của chúng ta nên khai thác kiến thức này.
Lợn thường không bay và máy bay thường không bơi.
Tuy vậy, chúng ta vẫn nên nhận ra
một con lợn nếu nó xuất hiện ở phía trên ảnh.
Ở đây, chúng ta có thể lấy cảm hứng
từ trò chơi trẻ em "Where's Waldo"
(bản thân trò chơi này đã truyền cảm hứng cho nhiều sản phẩm mô phỏng ngoài đời thực, chẳng hạn như sản phẩm được minh họa trong :numref:`img_waldo`).
Trò chơi gồm một số khung cảnh hỗn loạn
tràn ngập hoạt động.
Waldo xuất hiện ở đâu đó trong mỗi khung cảnh,
thường ẩn nấp tại một vị trí khó ngờ.
Mục tiêu của người đọc là tìm ra anh ấy.
Dù có trang phục đặc trưng,
điều này có thể khó đến bất ngờ,
do có rất nhiều yếu tố gây xao nhãng.
Tuy nhiên, *Waldo trông như thế nào*
không phụ thuộc vào *Waldo đang ở đâu*.
Chúng ta có thể quét ảnh bằng một bộ phát hiện Waldo
có thể gán điểm cho mỗi mảng ảnh,
cho biết khả năng mảng đó chứa Waldo. 
Thực tế, nhiều thuật toán phát hiện đối tượng và phân đoạn 
dựa trên cách tiếp cận này :cite:`Long.Shelhamer.Darrell.2015`. 
CNN hệ thống hóa ý tưởng về *tính bất biến không gian* này,
khai thác nó để học các biểu diễn hữu ích
với ít tham số hơn.

![Bạn có thể tìm thấy Waldo không (ảnh do William Murphy (Infomatique) cung cấp)?](../img/waldo-football.jpg)
:width:`400px`
:label:`img_waldo`

Giờ đây, chúng ta có thể cụ thể hóa hơn các trực giác này
bằng cách liệt kê một vài yêu cầu mong muốn để định hướng thiết kế
một kiến trúc mạng nơ-ron phù hợp cho computer vision:

1. Ở các lớp đầu tiên, mạng của chúng ta
   nên phản hồi tương tự với cùng một mảng ảnh,
   bất kể nó xuất hiện ở đâu trong ảnh. Nguyên tắc này được gọi là *tính bất biến tịnh tiến* (hoặc *tính equivariance theo tịnh tiến*).
1. Các lớp đầu tiên của mạng nên tập trung vào các vùng cục bộ,
   không quan tâm đến nội dung của ảnh ở các vùng xa. Đây là nguyên tắc *tính cục bộ*.
   Cuối cùng, các biểu diễn cục bộ này có thể được tổng hợp
   để đưa ra dự đoán ở cấp độ toàn bộ ảnh.
1. Khi đi sâu hơn, các lớp sâu hơn phải có khả năng nắm bắt các đặc trưng phạm vi xa hơn của 
   ảnh, theo cách tương tự như thị giác cấp cao trong tự nhiên. 

Hãy xem điều này được chuyển thành toán học như thế nào.


## Ràng buộc MLP

Để bắt đầu, chúng ta có thể xét một MLP
với ảnh hai chiều $\mathbf{X}$ làm đầu vào
và các biểu diễn ẩn trực tiếp của chúng
$\mathbf{H}$ cũng được biểu diễn tương tự dưới dạng ma trận (trong code, chúng là các tensor hai chiều), trong đó cả $\mathbf{X}$ và $\mathbf{H}$ có cùng hình dạng.
Hãy suy ngẫm về điều đó.
Giờ đây, chúng ta hình dung rằng không chỉ đầu vào mà
cả các biểu diễn ẩn cũng sở hữu cấu trúc không gian.

Gọi $[\mathbf{X}]_{i, j}$ và $[\mathbf{H}]_{i, j}$ lần lượt là pixel
tại vị trí $(i,j)$
trong ảnh đầu vào và biểu diễn ẩn.
Do đó, để mỗi đơn vị ẩn
nhận đầu vào từ mỗi pixel đầu vào,
chúng ta sẽ chuyển từ việc dùng ma trận trọng số
(như đã làm trước đây trong các MLP)
sang biểu diễn các tham số
dưới dạng tensor trọng số bậc bốn $\mathsf{W}$.
Giả sử $\mathbf{U}$ chứa các bias,
ta có thể biểu diễn chính thức lớp fully connected như sau

$$\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned}$$

Việc chuyển từ $\mathsf{W}$ sang $\mathsf{V}$ hiện tại hoàn toàn chỉ mang tính hình thức
vì có sự tương ứng một-một
giữa các hệ số trong cả hai tensor bậc bốn.
Chúng ta chỉ đơn giản đánh chỉ mục lại các chỉ số dưới $(k, l)$
sao cho $k = i+a$ và $l = j+b$.
Nói cách khác, ta đặt $[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$.
Các chỉ số $a$ và $b$ chạy qua cả các độ lệch dương lẫn âm,
bao phủ toàn bộ ảnh.
Với bất kỳ vị trí ($i$, $j$) nào trong biểu diễn ẩn $[\mathbf{H}]_{i, j}$,
chúng ta tính giá trị của nó bằng cách cộng tổng trên các pixel trong $x$,
lấy tâm tại $(i, j)$ và được gán trọng số bởi $[\mathsf{V}]_{i, j, a, b}$. Trước khi tiếp tục, hãy xem xét tổng số tham số cần thiết cho *một* lớp duy nhất trong cách tham số hóa này: một ảnh $1000 \times 1000$ (1 megapixel) được ánh xạ tới một biểu diễn ẩn $1000 \times 1000$. Điều này cần $10^{12}$ tham số, vượt xa khả năng xử lý hiện tại của máy tính.  

### Tính bất biến tịnh tiến

Bây giờ, hãy áp dụng nguyên tắc đầu tiên
đã được thiết lập ở trên: tính bất biến tịnh tiến :cite:`Zhang.ea.1988`.
Điều này ngụ ý rằng một phép dịch chuyển trong đầu vào $\mathbf{X}$
chỉ đơn giản dẫn đến một phép dịch chuyển trong biểu diễn ẩn $\mathbf{H}$.
Điều này chỉ có thể xảy ra nếu $\mathsf{V}$ và $\mathbf{U}$ thực sự không phụ thuộc vào $(i, j)$. Do đó,
ta có $[\mathsf{V}]_{i, j, a, b} = [\mathsf{V}]_{a, b}$ và $\mathbf{U}$ là một hằng số, chẳng hạn $u$.
Kết quả là, chúng ta có thể đơn giản hóa định nghĩa của $\mathbf{H}$:

$$[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$


Đây là một phép *convolution*!
Về thực chất, chúng ta đang gán trọng số cho các pixel tại $(i+a, j+b)$
trong vùng lân cận của vị trí $(i, j)$ bằng các hệ số $[\mathbf{V}]_{a, b}$
để thu được giá trị $[\mathbf{H}]_{i, j}$.
Lưu ý rằng $[\mathbf{V}]_{a, b}$ cần ít hệ số hơn nhiều so với $[\mathsf{V}]_{i, j, a, b}$ vì nó
không còn phụ thuộc vào vị trí trong ảnh. Do đó, số lượng tham số cần thiết không còn là $10^{12}$ mà là $4 \times 10^6$ hợp lý hơn nhiều: chúng ta vẫn có sự phụ thuộc vào $a, b \in (-1000, 1000)$. Tóm lại, chúng ta đã đạt được tiến bộ đáng kể. Mạng nơ-ron trễ thời gian (time-delay neural networks, TDNNs) là một trong những ví dụ đầu tiên khai thác ý tưởng này :cite:`Waibel.Hanazawa.Hinton.ea.1989`.

###  Tính cục bộ

Bây giờ, hãy áp dụng nguyên tắc thứ hai: tính cục bộ.
Như đã nêu động cơ ở trên, chúng ta tin rằng không cần
phải nhìn quá xa khỏi vị trí $(i, j)$
để thu thập thông tin liên quan
nhằm đánh giá điều gì đang diễn ra tại $[\mathbf{H}]_{i, j}$.
Điều này có nghĩa là bên ngoài một phạm vi nào đó $|a|> \Delta$ hoặc $|b| > \Delta$,
chúng ta nên đặt $[\mathbf{V}]_{a, b} = 0$.
Tương đương, chúng ta có thể viết lại $[\mathbf{H}]_{i, j}$ thành

$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$
:eqlabel:`eq_conv-layer`

Điều này giảm số lượng tham số từ $4 \times 10^6$ xuống còn $4 \Delta^2$, trong đó $\Delta$ thường nhỏ hơn $10$. Như vậy, chúng ta đã giảm số lượng tham số thêm bốn bậc độ lớn nữa. Lưu ý rằng :eqref:`eq_conv-layer` về cơ bản chính là thứ được gọi là một *convolutional layer*. *Convolutional neural networks* (CNN) là một họ đặc biệt của mạng nơ-ron có chứa các convolutional layer. Trong cộng đồng nghiên cứu deep learning, $\mathbf{V}$ được gọi là một *convolution kernel*, một *filter*, hoặc đơn giản là *weights* của layer, tức các tham số có thể học được.

Trong khi trước đây chúng ta có thể cần hàng tỷ tham số để biểu diễn chỉ một layer duy nhất trong mạng xử lý ảnh, giờ đây ta thường chỉ cần vài trăm tham số mà không làm thay đổi số chiều của cả đầu vào lẫn các biểu diễn ẩn. Cái giá phải trả cho việc giảm mạnh số lượng tham số này là các đặc trưng của chúng ta giờ đây bất biến theo phép tịnh tiến, và layer của chúng ta chỉ có thể kết hợp thông tin cục bộ khi xác định giá trị của mỗi kích hoạt ẩn. Mọi quá trình học đều phụ thuộc vào việc áp đặt inductive bias. Khi bias đó phù hợp với thực tế, chúng ta có được các mô hình sử dụng mẫu hiệu quả và khái quát hóa tốt sang dữ liệu chưa thấy. Nhưng dĩ nhiên, nếu các bias đó không phù hợp với thực tế, chẳng hạn nếu ảnh hóa ra không bất biến theo phép tịnh tiến, các mô hình của chúng ta có thể gặp khó khăn ngay cả khi khớp với dữ liệu huấn luyện.

Việc giảm mạnh số lượng tham số này đưa chúng ta đến yêu cầu cuối cùng, đó là các layer sâu hơn nên biểu diễn những khía cạnh lớn hơn và phức tạp hơn của một ảnh. Điều này có thể đạt được bằng cách lặp lại xen kẽ các phi tuyến tính và convolutional layer.

## Phép tích chập

Hãy cùng xem nhanh vì sao :eqref:`eq_conv-layer` được gọi là một phép tích chập. Trong toán học, *phép tích chập* giữa hai hàm :cite:`Rudin.1973`, chẳng hạn $f, g: \mathbb{R}^d \to \mathbb{R}$, được định nghĩa là

$$(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}.$$

Tức là, chúng ta đo phần chồng lấp giữa $f$ và $g$ khi một hàm được "lật" và dịch chuyển theo $\mathbf{x}$. Mỗi khi có các đối tượng rời rạc, tích phân sẽ trở thành tổng. Chẳng hạn, với các vector thuộc tập các vector vô hạn chiều có tổng bình phương hữu hạn với chỉ số chạy trên $\mathbb{Z}$, ta có định nghĩa sau:

$$(f * g)(i) = \sum_a f(a) g(i-a).$$

Đối với tensor hai chiều, ta có tổng tương ứng với các chỉ số $(a, b)$ cho $f$ và lần lượt $(i-a, j-b)$ cho $g$:

$$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b).$$
:eqlabel:`eq_2d-conv-discrete`

Điều này trông giống :eqref:`eq_conv-layer`, nhưng có một khác biệt lớn. Thay vì dùng $(i+a, j+b)$, chúng ta dùng hiệu. Tuy nhiên, lưu ý rằng sự khác biệt này phần lớn chỉ mang tính hình thức vì ta luôn có thể đối chiếu ký hiệu giữa :eqref:`eq_conv-layer` và :eqref:`eq_2d-conv-discrete`. Định nghĩa ban đầu của chúng ta trong :eqref:`eq_conv-layer` chính xác hơn mô tả một *cross-correlation*. Chúng ta sẽ quay lại vấn đề này trong phần tiếp theo.


## Kênh
:label:`subsec_why-conv-channels`

Quay lại bộ phát hiện Waldo, hãy xem điều này trông như thế nào. Convolutional layer chọn các cửa sổ có kích thước cho trước và gán trọng số cho cường độ theo filter $\mathsf{V}$, như minh họa trong :numref:`fig_waldo_mask`. Chúng ta có thể muốn học một mô hình sao cho ở bất kỳ nơi nào "mức độ Waldo" cao nhất, ta sẽ tìm thấy một đỉnh trong các biểu diễn hidden layer.

![Phát hiện Waldo (hình ảnh do William Murphy (Infomatique) cung cấp).](../img/waldo-mask.jpg)
:width:`400px`
:label:`fig_waldo_mask`

Cách tiếp cận này chỉ có một vấn đề. Cho đến nay, chúng ta đã vô tư bỏ qua thực tế rằng ảnh gồm ba kênh: đỏ, lục và lam. Tóm lại, ảnh không phải là các đối tượng hai chiều mà là các tensor bậc ba, được đặc trưng bởi chiều cao, chiều rộng và kênh, chẳng hạn có hình dạng $1024 \times 1024 \times 3$ pixel. Trong khi hai trục đầu tiên liên quan đến các quan hệ không gian, trục thứ ba có thể được xem là gán một biểu diễn đa chiều cho mỗi vị trí pixel. Vì vậy, ta lập chỉ mục $\mathsf{X}$ là $[\mathsf{X}]_{i, j, k}$. Convolutional filter cũng phải thích ứng tương ứng. Thay vì $[\mathbf{V}]_{a,b}$, giờ đây ta có $[\mathsf{V}]_{a,b,c}$.

Hơn nữa, cũng như đầu vào của chúng ta gồm một tensor bậc ba, hóa ra việc biểu diễn các hidden representations dưới dạng tensor bậc ba $\mathsf{H}$ cũng là một ý tưởng hay. Nói cách khác, thay vì chỉ có một hidden representation tương ứng với mỗi vị trí không gian, chúng ta muốn có cả một vector hidden representations tương ứng với mỗi vị trí không gian. Ta có thể xem các hidden representations như gồm một số lưới hai chiều được xếp chồng lên nhau. Tương tự như ở đầu vào, chúng đôi khi được gọi là *channels*. Chúng cũng đôi khi được gọi là *feature maps*, vì mỗi cái cung cấp một tập đặc trưng đã học được không gian hóa cho layer tiếp theo. Theo trực giác, bạn có thể hình dung rằng ở các layer thấp gần đầu vào hơn, một số channel có thể chuyên nhận diện cạnh trong khi các channel khác có thể nhận diện kết cấu.

Để hỗ trợ nhiều channel ở cả đầu vào ($\mathsf{X}$) lẫn hidden representations ($\mathsf{H}$), ta có thể thêm tọa độ thứ tư vào $\mathsf{V}$: $[\mathsf{V}]_{a, b, c, d}$. Tổng hợp tất cả lại, ta có:

$$[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c},$$
:eqlabel:`eq_conv-layer-channels`

trong đó $d$ lập chỉ mục các output channel trong hidden representations $\mathsf{H}$. Convolutional layer tiếp theo sẽ nhận một tensor bậc ba, $\mathsf{H}$, làm đầu vào. Do tính tổng quát của nó, chúng ta lấy :eqref:`eq_conv-layer-channels` làm định nghĩa của một convolutional layer cho nhiều channel, trong đó $\mathsf{V}$ là kernel hoặc filter của layer.

Vẫn còn nhiều phép toán cần được giải quyết. Chẳng hạn, chúng ta cần tìm cách kết hợp tất cả hidden representations thành một output duy nhất, ví dụ liệu có Waldo *ở bất kỳ đâu* trong ảnh hay không. Chúng ta cũng cần quyết định cách tính toán hiệu quả, cách kết hợp nhiều layer, các activation function phù hợp, và cách đưa ra các lựa chọn thiết kế hợp lý để tạo ra những mạng hiệu quả trong thực tế. Chúng ta sẽ xem xét các vấn đề này trong phần còn lại của chương.

## Tóm tắt và Thảo luận

Trong phần này, chúng ta đã suy ra cấu trúc của convolutional neural networks từ các nguyên lý đầu tiên. Mặc dù không rõ liệu đây có phải là con đường dẫn đến việc phát minh ra CNN hay không, thật thỏa đáng khi biết rằng chúng là lựa chọn *đúng* khi áp dụng các nguyên lý hợp lý về cách các thuật toán xử lý ảnh và computer vision nên vận hành, ít nhất ở các mức thấp hơn. Cụ thể, tính bất biến theo phép tịnh tiến trong ảnh hàm ý rằng mọi mảng ảnh sẽ được xử lý theo cùng một cách. Tính cục bộ có nghĩa là chỉ một vùng lân cận nhỏ các pixel sẽ được dùng để tính các hidden representations tương ứng. Một số tài liệu tham khảo sớm nhất về CNN có dạng Neocognitron :cite:`Fukushima.1982`.

Nguyên lý thứ hai mà chúng ta gặp trong quá trình lập luận là cách giảm số lượng tham số trong một lớp hàm mà không hạn chế khả năng biểu đạt của nó, ít nhất là khi một số giả định nhất định về mô hình được thỏa mãn. Chúng ta đã thấy sự giảm mạnh về độ phức tạp do hạn chế này, biến những bài toán không khả thi về mặt tính toán và thống kê thành các mô hình có thể xử lý được.

Việc thêm các channel cho phép chúng ta khôi phục một phần độ phức tạp đã mất do các hạn chế mà tính cục bộ và tính bất biến theo phép tịnh tiến áp đặt lên convolution kernel. Lưu ý rằng việc thêm các channel khác ngoài đỏ, lục và lam là hoàn toàn tự nhiên. Nhiều ảnh vệ tinh, đặc biệt trong nông nghiệp và khí tượng học, có từ hàng chục đến hàng trăm channel, từ đó tạo ra các ảnh siêu phổ. Chúng báo cáo dữ liệu ở nhiều bước sóng khác nhau. Trong phần tiếp theo, chúng ta sẽ thấy cách dùng phép tích chập hiệu quả để thao tác số chiều của các ảnh mà chúng xử lý, cách chuyển từ biểu diễn dựa trên vị trí sang biểu diễn dựa trên channel, và cách xử lý hiệu quả số lượng lớn category.

## Bài tập

1. Giả sử kích thước của kernel tích chập là $\Delta = 0$.
   Hãy chỉ ra rằng trong trường hợp này, kernel tích chập
   triển khai một MLP độc lập cho từng tập kênh. Điều này dẫn đến các kiến trúc Network in Network
   :cite:`Lin.Chen.Yan.2013`.
1. Dữ liệu âm thanh thường được biểu diễn dưới dạng một chuỗi một chiều.
    1. Khi nào bạn muốn áp đặt tính cục bộ và tính bất biến theo phép tịnh tiến cho âm thanh?
    1. Hãy suy ra các phép toán tích chập cho âm thanh.
    1. Bạn có thể xử lý âm thanh bằng cùng các công cụ như computer vision không? Gợi ý: sử dụng spectrogram.
1. Tại sao tính bất biến theo phép tịnh tiến rốt cuộc có thể không phải là một ý hay? Hãy đưa ra một ví dụ.
1. Bạn có nghĩ rằng các lớp tích chập cũng có thể áp dụng cho dữ liệu văn bản không?
   Bạn có thể gặp phải những vấn đề nào với ngôn ngữ?
1. Điều gì xảy ra với các phép tích chập khi một đối tượng nằm ở ranh giới của ảnh?
1. Chứng minh rằng phép tích chập là đối xứng, tức là $f * g = g * f$.

[Thảo luận](https://discuss.d2l.ai/t/64)