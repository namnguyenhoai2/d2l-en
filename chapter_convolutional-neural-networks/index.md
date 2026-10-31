# Mạng nơ-ron tích chập
:label:`chap_cnn`

Dữ liệu hình ảnh được biểu diễn dưới dạng một lưới pixel hai chiều, bất kể hình ảnh
là đơn sắc hay có màu. Theo đó, mỗi pixel tương ứng với một
hoặc nhiều giá trị số. Cho đến nay, chúng ta đã bỏ qua cấu trúc phong phú này
và xem hình ảnh như các vector số bằng cách *làm phẳng* chúng, không xét đến
mối quan hệ không gian giữa các pixel. Cách tiếp cận hết sức không thỏa đáng này
là cần thiết để đưa các vector một chiều thu được
qua một MLP kết nối đầy đủ.

Vì các mạng này không phụ thuộc vào thứ tự của các đặc trưng, chúng ta
có thể nhận được kết quả tương tự dù giữ nguyên thứ tự tương ứng với
cấu trúc không gian của các pixel hay hoán vị các cột của ma trận thiết kế
trước khi khớp các tham số của MLP. Lý tưởng nhất là chúng ta sẽ tận dụng
kiến thức có sẵn rằng các pixel lân cận thường có liên hệ với nhau,
để xây dựng các mô hình hiệu quả nhằm học từ dữ liệu hình ảnh.

Chương này giới thiệu *mạng nơ-ron tích chập* (convolutional neural networks, CNNs)
:cite:`LeCun.Jackel.Bottou.ea.1995`, một họ mạng nơ-ron mạnh mẽ được
thiết kế chính xác cho mục đích này.
Các kiến trúc dựa trên CNN hiện nay
phổ biến rộng rãi trong lĩnh vực thị giác máy tính.
Ví dụ, trên tập dữ liệu Imagnet
:cite:`Deng.Dong.Socher.ea.2009`, chính việc sử dụng mạng nơ-ron tích chập,
hay gọi tắt là Convnets, đã mang lại những cải thiện hiệu năng đáng kể
:cite:`Krizhevsky.Sutskever.Hinton.2012`.

CNN hiện đại, như cách chúng thường được gọi, có thiết kế bắt nguồn từ
cảm hứng sinh học, lý thuyết nhóm và rất nhiều thử nghiệm điều chỉnh.
Ngoài hiệu quả về số lượng mẫu trong việc đạt được các mô hình chính xác,
CNN thường hiệu quả về mặt tính toán, vừa vì chúng cần ít tham số hơn
các kiến trúc kết nối đầy đủ, vừa vì các phép tích chập dễ được song song hóa
trên các lõi GPU :cite:`Chetlur.Woolley.Vandermersch.ea.2014`. Do đó, các practitioner thường
áp dụng CNN bất cứ khi nào có thể, và ngày càng nhiều CNN trở thành những đối thủ
đáng tin cậy ngay cả trong các tác vụ có cấu trúc chuỗi một chiều,
chẳng hạn như âm thanh :cite:`Abdel-Hamid.Mohamed.Jiang.ea.2014`, văn bản
:cite:`Kalchbrenner.Grefenstette.Blunsom.2014`, và phân tích chuỗi thời gian
:cite:`LeCun.Bengio.ea.1995`, nơi mạng nơ-ron hồi quy thường được sử dụng
theo truyền thống. Một số cải tiến thông minh của CNN cũng đã giúp chúng được áp dụng
cho dữ liệu có cấu trúc đồ thị :cite:`Kipf.Welling.2016` và
trong các hệ thống gợi ý.

Trước hết, chúng ta sẽ tìm hiểu sâu hơn về động lực của mạng nơ-ron
tích chập. Tiếp theo là phần trình bày các phép toán cơ bản tạo nên
xương sống của mọi mạng tích chập.
Chúng bao gồm chính các lớp tích chập,
những chi tiết cụ thể như padding và stride,
các lớp pooling dùng để tổng hợp thông tin
qua các vùng không gian lân cận,
việc sử dụng nhiều channel tại mỗi lớp,
và thảo luận kỹ lưỡng về cấu trúc của các kiến trúc hiện đại.
Chúng ta sẽ kết thúc chương bằng một ví dụ hoàn chỉnh, có thể chạy được về LeNet,
mạng tích chập đầu tiên được triển khai thành công,
từ rất lâu trước khi deep learning hiện đại phát triển mạnh.
Trong chương tiếp theo, chúng ta sẽ đi sâu vào phần triển khai đầy đủ
của một số kiến trúc CNN phổ biến và tương đối mới
có thiết kế đại diện cho hầu hết các kỹ thuật
thường được các practitioner hiện đại sử dụng.

```toc
:maxdepth: 2

why-conv
conv-layer
padding-and-strides
channels
pooling
lenet
```
