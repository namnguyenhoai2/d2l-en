# Thuật toán tối ưu hóa
:label:`chap_optimization`

Nếu bạn đã đọc cuốn sách theo trình tự đến thời điểm này, bạn đã sử dụng một số thuật toán tối ưu hóa để huấn luyện các mô hình deep learning.
Chúng là những công cụ cho phép chúng ta tiếp tục cập nhật các tham số của mô hình và tối thiểu hóa giá trị của hàm mất mát, được đánh giá trên tập huấn luyện. Thật vậy, bất kỳ ai hài lòng với việc xem tối ưu hóa như một thiết bị hộp đen để tối thiểu hóa các hàm mục tiêu trong một bối cảnh đơn giản có lẽ cũng sẽ hài lòng với hiểu biết rằng tồn tại một loạt các câu thần chú cho quy trình như vậy (với các tên như "SGD" và "Adam").

Tuy nhiên, để làm tốt, cần có một số kiến thức sâu hơn.
Các thuật toán tối ưu hóa rất quan trọng đối với deep learning.
Một mặt, việc huấn luyện một mô hình deep learning phức tạp có thể mất hàng giờ, hàng ngày, hoặc thậm chí hàng tuần.
Hiệu năng của thuật toán tối ưu hóa ảnh hưởng trực tiếp đến hiệu quả huấn luyện của mô hình.
Mặt khác, việc hiểu các nguyên lý của những thuật toán tối ưu hóa khác nhau và vai trò của các hyperparameter của chúng
sẽ cho phép chúng ta điều chỉnh hyperparameter một cách có mục tiêu nhằm cải thiện hiệu năng của các mô hình deep learning.

Trong chương này, chúng ta sẽ khám phá sâu các thuật toán tối ưu hóa phổ biến trong deep learning.
Hầu như mọi bài toán tối ưu hóa phát sinh trong deep learning đều *không lồi*.
Tuy vậy, việc thiết kế và phân tích các thuật toán trong bối cảnh các bài toán *lồi* đã chứng tỏ là rất bổ ích.
Vì lý do đó, chương này bao gồm phần nhập môn về tối ưu hóa lồi và chứng minh cho một thuật toán stochastic gradient descent rất đơn giản trên một hàm mục tiêu lồi.

```toc
:maxdepth: 2

optimization-intro
convexity
gd
sgd
minibatch-sgd
momentum
adagrad
rmsprop
adadelta
adam
lr-scheduler
```
