# Gaussian Processes
:label:`chap_gp`

**Andrew Gordon Wilson** (*New York University và Amazon*)


Gaussian processes (GP) xuất hiện ở khắp nơi. Bạn đã bắt gặp nhiều ví dụ về GP mà không nhận ra. Bất kỳ mô hình nào tuyến tính theo các tham số và có phân phối Gaussian trên các tham số đều là một Gaussian process. Lớp mô hình này bao gồm các mô hình rời rạc, chẳng hạn như random walk và các quá trình tự hồi quy, cũng như các mô hình liên tục, bao gồm các mô hình hồi quy tuyến tính Bayesian, đa thức, chuỗi Fourier, các hàm cơ sở xuyên tâm, và thậm chí cả các mạng neural với vô hạn đơn vị ẩn. Có một câu nói đùa phổ biến rằng “mọi thứ đều là một trường hợp đặc biệt của Gaussian process”.

Việc tìm hiểu về Gaussian processes quan trọng vì ba lý do: (1) chúng cung cấp một góc nhìn theo _không gian hàm_ về việc mô hình hóa, giúp việc tìm hiểu nhiều lớp mô hình, bao gồm cả mạng neural sâu, trở nên dễ tiếp cận hơn nhiều; (2) chúng có phạm vi ứng dụng đặc biệt rộng, trong đó chúng đạt hiệu năng hiện đại, bao gồm active learning, học hyperparameter, auto-ML và hồi quy không-thời gian; (3) trong vài năm qua, những tiến bộ về thuật toán đã giúp Gaussian processes ngày càng có khả năng mở rộng và phù hợp hơn, đồng thời hài hòa với deep learning thông qua các framework như [GPyTorch](https://gpytorch.ai) :cite:`Gardner.Pleiss.Weinberger.Bindel.Wilson.2018`. Thực tế, GPs và các mạng neural sâu không phải là những phương pháp cạnh tranh với nhau, mà bổ trợ mạnh mẽ cho nhau và có thể được kết hợp để đạt hiệu quả cao. Những tiến bộ về thuật toán này không chỉ hữu ích cho Gaussian processes, mà còn cung cấp nền tảng về các phương pháp số có giá trị rộng rãi trong deep learning.

Trong chương này, chúng ta giới thiệu Gaussian processes. Trong notebook nhập môn, chúng ta bắt đầu bằng cách suy luận trực quan về Gaussian processes là gì và cách chúng trực tiếp mô hình hóa các hàm. Trong notebook về priors, chúng ta tập trung vào cách xác định các Gaussian process priors. Chúng ta kết nối trực tiếp phương pháp mô hình hóa truyền thống theo không gian trọng số với không gian hàm, qua đó giúp chúng ta suy luận về việc xây dựng và tìm hiểu các mô hình machine learning, bao gồm cả mạng neural sâu. Sau đó, chúng ta giới thiệu các hàm hiệp phương sai phổ biến, còn được gọi là _kernels_, dùng để kiểm soát các thuộc tính khái quát hóa của một Gaussian process. Một GP với kernel cho trước xác định một prior trên các hàm. Trong notebook về inference, chúng ta sẽ chỉ ra cách sử dụng dữ liệu để suy ra một _posterior_, nhằm đưa ra dự đoán. Notebook này chứa mã được viết từ đầu để đưa ra dự đoán bằng Gaussian process, đồng thời giới thiệu về GPyTorch. Trong các notebook sắp tới, chúng ta sẽ giới thiệu các phép tính số đằng sau Gaussian processes, vốn hữu ích cho việc mở rộng quy mô Gaussian processes, đồng thời cũng là một nền tảng tổng quát mạnh mẽ cho deep learning, cũng như các trường hợp sử dụng nâng cao như tinh chỉnh hyperparameter trong deep learning. Các ví dụ của chúng ta sẽ sử dụng GPyTorch, giúp Gaussian processes có khả năng mở rộng và được tích hợp chặt chẽ với chức năng deep learning và PyTorch.

```toc
:maxdepth: 2

gp-intro
gp-priors
gp-inference
```
