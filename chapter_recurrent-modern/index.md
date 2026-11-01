# Mạng nơ-ron hồi quy hiện đại
:label:`chap_modern_rnn`

Chương trước đã giới thiệu những ý tưởng then chốt 
đằng sau mạng nơ-ron hồi quy (RNN). 
Tuy nhiên, cũng như mạng nơ-ron tích chập,
đã có một lượng đổi mới đáng kể
trong các kiến trúc RNN, dẫn đến một số thiết kế phức tạp
đã chứng minh được tính hiệu quả trong thực tế. 
Cụ thể, những thiết kế phổ biến nhất 
có các cơ chế giảm thiểu
sự bất ổn định số tai tiếng mà RNN phải đối mặt,
điển hình là gradient tiêu biến và gradient bùng nổ.
Hãy nhớ rằng trong :numref:`chap_rnn`, chúng ta đã xử lý 
gradient bùng nổ bằng cách áp dụng một heuristic cắt gradient
khá thô. 
Mặc dù thủ thuật này có hiệu quả,
nó vẫn để ngỏ vấn đề gradient tiêu biến. 

Trong chương này, chúng ta giới thiệu những ý tưởng then chốt đằng sau 
các kiến trúc RNN thành công nhất dành cho chuỗi,
bắt nguồn từ hai bài báo.
Bài báo đầu tiên, *Long Short-Term Memory* :cite:`Hochreiter.Schmidhuber.1997`,
giới thiệu *tế bào bộ nhớ (memory cell)*, một đơn vị tính toán thay thế 
các node truyền thống trong lớp ẩn của một mạng.
Với các tế bào bộ nhớ này, mạng có thể 
vượt qua những khó khăn trong quá trình huấn luyện 
mà các mạng hồi quy trước đây gặp phải.
Trực giác là tế bào bộ nhớ tránh 
vấn đề gradient tiêu biến
bằng cách duy trì các giá trị trong trạng thái bên trong của mỗi tế bào bộ nhớ
truyền dọc theo một cạnh hồi quy có trọng số bằng 1 
qua nhiều bước thời gian liên tiếp. 
Một tập các cổng nhân giúp mạng
xác định không chỉ những đầu vào nào được phép 
đi vào trạng thái bộ nhớ, 
mà còn xác định khi nào nội dung của trạng thái bộ nhớ 
nên ảnh hưởng đến đầu ra của mô hình. 

Bài báo thứ hai, *Bidirectional Recurrent Neural Networks* :cite:`Schuster.Paliwal.1997`,
giới thiệu một kiến trúc trong đó thông tin 
từ cả tương lai (các bước thời gian tiếp theo) 
và quá khứ (các bước thời gian trước đó)
được sử dụng để xác định đầu ra 
tại bất kỳ điểm nào trong chuỗi.
Điều này trái ngược với các mạng trước đây, 
trong đó chỉ đầu vào quá khứ mới có thể ảnh hưởng đến đầu ra.
RNN hai chiều đã trở thành một thành phần chủ chốt 
trong các tác vụ gán nhãn chuỗi thuộc lĩnh vực xử lý ngôn ngữ tự nhiên,
cùng vô số tác vụ khác. 
May mắn là hai đổi mới này không loại trừ lẫn nhau, 
và đã được kết hợp thành công để phân loại âm vị
:cite:`Graves.Schmidhuber.2005` và nhận dạng chữ viết tay :cite:`graves2008novel`.


Các phần đầu tiên trong chương này sẽ giải thích kiến trúc LSTM,
một phiên bản nhẹ hơn có tên gated recurrent unit (GRU),
những ý tưởng then chốt đằng sau RNN hai chiều 
và phần giải thích ngắn gọn về cách các lớp RNN 
được xếp chồng lên nhau để tạo thành các RNN sâu. 
Sau đó, chúng ta sẽ tìm hiểu việc áp dụng RNN
trong các tác vụ sequence-to-sequence, 
giới thiệu machine translation
cùng với những ý tưởng then chốt như kiến trúc *encoder--decoder* và *beam search*.

```toc
:maxdepth: 2

lstm
gru
deep-rnn
bi-rnn
machine-translation-and-dataset
encoder-decoder
seq2seq
beam-search
```
