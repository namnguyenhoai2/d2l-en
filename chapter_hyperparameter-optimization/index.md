# Tối ưu hóa siêu tham số
:label:`chap_hyperopt`

**Aaron Klein** (*Amazon*), **Matthias Seeger** (*Amazon*) và **Cedric Archambeau** (*Amazon*)

Hiệu năng của mọi mô hình machine learning đều phụ thuộc vào các siêu tham số.
Chúng kiểm soát thuật toán học hoặc cấu trúc của mô hình thống kê cơ sở.
Tuy nhiên, trên thực tế không có cách tổng quát nào để lựa chọn các siêu tham số.
Thay vào đó, các practitioner thường thiết lập siêu tham số theo cách thử và sai,
hoặc đôi khi giữ nguyên các giá trị mặc định của chúng, dẫn đến khả năng
tổng quát hóa không tối ưu.

Tối ưu hóa siêu tham số cung cấp một cách tiếp cận có hệ thống cho vấn đề này bằng
cách chuyển nó thành một bài toán tối ưu hóa: một tập hợp siêu tham số tốt ít nhất
phải giảm thiểu sai số validation. So với hầu hết các bài toán tối ưu hóa khác
phát sinh trong machine learning, tối ưu hóa siêu tham số là một bài toán lồng nhau,
trong đó mỗi lần lặp đều yêu cầu huấn luyện và validation một mô hình machine learning.

Trong chương này, trước tiên chúng ta sẽ giới thiệu những kiến thức cơ bản về tối ưu hóa
siêu tham số. Chúng ta cũng sẽ trình bày một số tiến bộ gần đây giúp cải thiện hiệu quả
tổng thể của quá trình tối ưu hóa siêu tham số bằng cách tận dụng các đại diện
(proxy) có chi phí đánh giá thấp của hàm mục tiêu ban đầu. Khi kết thúc chương này, bạn
sẽ có thể áp dụng các kỹ thuật tối ưu hóa siêu tham số hiện đại nhất để tối ưu hóa
siêu tham số của thuật toán machine learning của riêng mình.

```toc
:maxdepth: 2

hyperopt-intro
hyperopt-api
rs-async.md
sh-intro
sh-async
```
