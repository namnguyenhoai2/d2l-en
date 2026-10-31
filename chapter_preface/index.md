# Lời nói đầu

Chỉ vài năm trước, chưa có những đội ngũ đông đảo các nhà khoa học deep learning phát triển các sản phẩm và dịch vụ thông minh tại những công ty lớn và startup. Khi chúng tôi bước vào lĩnh vực này, machine learning chưa xuất hiện trên các trang nhất của những tờ báo hằng ngày. Cha mẹ chúng tôi không biết machine learning là gì, chứ chưa nói đến việc hiểu tại sao chúng tôi có thể thích nó hơn một nghề nghiệp trong y khoa hoặc luật. Machine learning là một ngành học thuật mang tính lý tưởng, với ý nghĩa công nghiệp chỉ giới hạn trong một tập hợp hẹp các ứng dụng thực tế, bao gồm nhận dạng giọng nói và thị giác máy tính. Hơn nữa, nhiều ứng dụng trong số này đòi hỏi quá nhiều kiến thức chuyên ngành đến mức chúng thường được xem là những lĩnh vực hoàn toàn riêng biệt, trong đó machine learning chỉ là một thành phần nhỏ. Vào thời điểm đó, neural network---những tiền thân của các phương pháp deep learning mà chúng tôi tập trung trình bày trong cuốn sách này---thường bị xem là đã lỗi thời.


Thế nhưng chỉ trong vài năm, deep learning đã khiến cả thế giới kinh ngạc, thúc đẩy những tiến bộ nhanh chóng trong các lĩnh vực đa dạng như thị giác máy tính, xử lý ngôn ngữ tự nhiên, nhận dạng giọng nói tự động, reinforcement learning và tin-sinh học. Hơn nữa, thành công của deep learning trong rất nhiều tác vụ được quan tâm trên thực tế thậm chí còn thúc đẩy những phát triển trong machine learning lý thuyết và thống kê. Với những tiến bộ này, giờ đây chúng ta có thể xây dựng những chiếc xe tự lái với mức độ tự chủ cao hơn bao giờ hết (dù vẫn kém tự chủ hơn mức một số công ty có thể khiến bạn tin), các hệ thống đối thoại có thể debug code bằng cách đặt câu hỏi làm rõ, và các software agent đánh bại những kỳ thủ con người giỏi nhất thế giới trong các trò chơi bàn cờ như cờ vây—một thành tựu từng được cho là còn cách xa hàng thập kỷ. Những công cụ này đã ngày càng có ảnh hưởng rộng hơn đến công nghiệp và xã hội, thay đổi cách làm phim, chẩn đoán bệnh, đồng thời đóng vai trò ngày càng lớn trong các ngành khoa học cơ bản—từ vật lý thiên văn, mô hình hóa khí hậu, dự báo thời tiết đến y sinh học.



## Về cuốn sách này

Cuốn sách này là nỗ lực của chúng tôi nhằm giúp deep learning trở nên dễ tiếp cận, bằng cách dạy bạn về *các khái niệm*, *bối cảnh* và *code*.

### Một phương tiện kết hợp Code, Toán học và HTML

Để bất kỳ công nghệ máy tính nào phát huy được toàn bộ tác động, công nghệ đó phải được hiểu rõ, được ghi chép đầy đủ và được hỗ trợ bởi các công cụ trưởng thành, được duy trì tốt. Những ý tưởng cốt lõi cần được chắt lọc một cách rõ ràng, giảm thiểu thời gian onboarding cần thiết để đưa những người mới thực hành bắt kịp kiến thức. Các thư viện trưởng thành nên tự động hóa những tác vụ phổ biến, còn code mẫu nên giúp người thực hành dễ dàng sửa đổi, áp dụng và mở rộng các ứng dụng phổ biến cho phù hợp với nhu cầu của họ.


Lấy các ứng dụng web động làm ví dụ. Mặc dù một số lượng lớn công ty, chẳng hạn như Amazon, đã phát triển thành công các ứng dụng web dựa trên cơ sở dữ liệu vào những năm 1990, tiềm năng của công nghệ này trong việc hỗ trợ các doanh nhân sáng tạo chỉ được nhận ra ở mức độ lớn hơn nhiều trong mười năm qua, một phần nhờ sự phát triển của các framework mạnh mẽ và được ghi chép đầy đủ.


Việc khảo sát tiềm năng của deep learning đặt ra những thách thức đặc biệt, vì mỗi ứng dụng đơn lẻ đều kết hợp nhiều lĩnh vực khác nhau. Việc áp dụng deep learning đòi hỏi đồng thời phải hiểu (i) động cơ của việc biểu diễn một bài toán theo một cách cụ thể; (ii) dạng toán học của một model nhất định; (iii) các thuật toán tối ưu hóa để khớp các model với dữ liệu; (iv) các nguyên lý thống kê cho chúng ta biết khi nào có thể kỳ vọng các model của mình tổng quát hóa được sang dữ liệu chưa từng thấy, cùng những phương pháp thực tiễn để xác nhận rằng chúng thực sự đã tổng quát hóa; và (v) các kỹ thuật engineering cần thiết để train model một cách hiệu quả, vượt qua những cạm bẫy của tính toán số và tận dụng tối đa phần cứng hiện có. Việc giảng dạy các kỹ năng tư duy phản biện cần thiết để xây dựng bài toán, toán học để giải quyết chúng và các công cụ phần mềm để triển khai những lời giải đó trong cùng một nơi là một thách thức rất lớn. Mục tiêu của chúng tôi trong cuốn sách này là cung cấp một tài nguyên thống nhất để giúp những người mong muốn trở thành người thực hành bắt kịp kiến thức.

Khi bắt đầu dự án viết cuốn sách này, không có tài nguyên nào đồng thời (i) luôn được cập nhật; (ii) bao quát bề rộng của các thực hành machine learning hiện đại với độ sâu kỹ thuật đầy đủ; và (iii) kết hợp phần trình bày có chất lượng mà người ta kỳ vọng ở một giáo trình với code sạch, có thể chạy được, như người ta kỳ vọng ở một tutorial thực hành. Chúng tôi tìm thấy rất nhiều ví dụ code minh họa cách sử dụng một deep learning framework nhất định (ví dụ: cách thực hiện tính toán số cơ bản với ma trận trong TensorFlow) hoặc triển khai các kỹ thuật cụ thể (ví dụ: các đoạn code cho LeNet, AlexNet, ResNet, v.v.) rải rác trên nhiều bài đăng blog và repository GitHub khác nhau. Tuy nhiên, những ví dụ này thường tập trung vào *cách* triển khai một phương pháp nhất định nhưng bỏ qua thảo luận về *lý do* đưa ra những quyết định thuật toán cụ thể. Mặc dù một số tài nguyên tương tác đã xuất hiện rải rác để đề cập đến một chủ đề cụ thể, chẳng hạn như các bài đăng blog hấp dẫn được xuất bản trên website [Distill](http://distill.pub), hoặc các blog cá nhân, chúng chỉ bao quát một số chủ đề được chọn trong deep learning và thường thiếu code đi kèm. Mặt khác, mặc dù đã xuất hiện một số giáo trình deep learning—ví dụ :citet:`Goodfellow.Bengio.Courville.2016`, cung cấp một khảo sát toàn diện về những nền tảng của deep learning—những tài nguyên này không kết hợp phần mô tả với việc hiện thực hóa các khái niệm bằng code, đôi khi khiến độc giả không biết phải triển khai chúng như thế nào. Hơn nữa, quá nhiều tài nguyên bị ẩn sau các paywall của những nhà cung cấp khóa học thương mại.

Chúng tôi bắt tay xây dựng một tài nguyên có thể (i) được cung cấp miễn phí cho tất cả mọi người; (ii) có đủ độ sâu kỹ thuật để trở thành điểm khởi đầu trên con đường thực sự trở thành một nhà khoa học machine learning ứng dụng; (iii) bao gồm code có thể chạy được, cho độc giả thấy *cách* giải quyết các bài toán trong thực tế; (iv) cho phép cập nhật nhanh chóng, cả bởi chúng tôi lẫn cộng đồng nói chung; và (v) được bổ trợ bởi một [forum](https://discuss.d2l.ai/c/5) để thảo luận tương tác về các chi tiết kỹ thuật và trả lời câu hỏi.

Những mục tiêu này thường mâu thuẫn với nhau. Các phương trình, định lý và trích dẫn phù hợp nhất khi được quản lý và trình bày bằng LaTeX. Code phù hợp nhất khi được mô tả bằng Python. Còn các trang web có bản chất là HTML và JavaScript. Hơn nữa, chúng tôi muốn nội dung có thể được tiếp cận dưới dạng code thực thi, sách in, PDF có thể tải xuống và website trên Internet. Dường như không có workflow nào phù hợp với những yêu cầu này, vì vậy chúng tôi quyết định tự xây dựng hệ thống của mình (:numref:`sec_how_to_contribute`). Chúng tôi chọn GitHub để chia sẻ source và tạo điều kiện cho cộng đồng đóng góp; Jupyter notebook để kết hợp code, phương trình và văn bản; Sphinx làm rendering engine; và Discourse làm nền tảng thảo luận. Mặc dù hệ thống của chúng tôi chưa hoàn hảo, những lựa chọn này tạo ra một sự thỏa hiệp giữa các mối quan tâm cạnh tranh. Chúng tôi tin rằng *Dive into Deep Learning* có thể là cuốn sách đầu tiên được xuất bản bằng một workflow tích hợp như vậy.


### Học thông qua thực hành

Nhiều giáo trình trình bày các khái niệm theo trình tự, bao quát từng khái niệm với đầy đủ chi tiết. Ví dụ, giáo trình xuất sắc của :citet:`Bishop.2006` giảng dạy từng chủ đề kỹ lưỡng đến mức để đi đến chương về hồi quy tuyến tính đòi hỏi một lượng công sức đáng kể. Những chuyên gia yêu thích cuốn sách này chính vì sự đầy đủ của nó, nhưng đối với những người mới bắt đầu thực sự, đặc điểm này lại hạn chế tính hữu ích của sách như một tài liệu nhập môn.

Trong cuốn sách này, chúng tôi giảng dạy hầu hết các khái niệm *đúng lúc cần thiết*. Nói cách khác, bạn sẽ học các khái niệm ngay tại thời điểm chúng cần thiết để đạt được một mục tiêu thực tiễn nào đó. Mặc dù ngay từ đầu chúng tôi dành thời gian giảng dạy những kiến thức nền tảng như đại số tuyến tính và xác suất, chúng tôi muốn bạn được trải nghiệm niềm vui khi train model đầu tiên trước khi lo lắng về những khái niệm chuyên sâu hơn.

Ngoài một vài notebook nhập môn cung cấp khóa học cấp tốc về nền tảng toán học cơ bản, mỗi chương tiếp theo vừa giới thiệu một số lượng hợp lý các khái niệm mới, vừa cung cấp nhiều ví dụ hoạt động độc lập sử dụng các dataset thực tế. Điều này tạo ra một thách thức về tổ chức. Một số model có thể được nhóm lại một cách hợp lý trong cùng một notebook. Một số ý tưởng có thể được giảng dạy tốt nhất bằng cách thực thi liên tiếp nhiều model. Ngược lại, việc tuân thủ nguyên tắc *một ví dụ hoạt động, một notebook* mang lại một lợi thế lớn: điều này giúp bạn dễ dàng nhất có thể bắt đầu các dự án nghiên cứu của riêng mình bằng cách tận dụng code của chúng tôi. Chỉ cần sao chép một notebook và bắt đầu sửa đổi nó.

Trong suốt cuốn sách, chúng tôi đan xen code có thể chạy với tài liệu nền tảng khi cần thiết. Nhìn chung, chúng tôi có xu hướng cung cấp các công cụ trước khi giải thích đầy đủ về chúng (thường bổ sung phần nền tảng sau). Chẳng hạn, chúng tôi có thể sử dụng *stochastic gradient descent* trước khi giải thích tại sao nó hữu ích hoặc đưa ra trực giác về lý do nó hoạt động. Cách này giúp cung cấp cho người thực hành những công cụ cần thiết để giải quyết vấn đề nhanh chóng, đổi lại độc giả phải tin tưởng chúng tôi trong một số quyết định tuyển chọn nội dung.

Cuốn sách này giảng dạy các khái niệm deep learning từ đầu. Đôi khi, chúng tôi đi sâu vào các chi tiết tinh tế về những model mà các deep learning framework hiện đại thường che giấu khỏi người dùng. Điều này đặc biệt xuất hiện trong các tutorial cơ bản, nơi chúng tôi muốn bạn hiểu mọi thứ diễn ra bên trong một layer hoặc optimizer nhất định. Trong những trường hợp này, chúng tôi thường trình bày hai phiên bản của ví dụ: một phiên bản trong đó chúng tôi triển khai mọi thứ từ đầu, chỉ dựa vào chức năng giống NumPy và automatic differentiation, cùng một ví dụ thực tiễn hơn, trong đó chúng tôi viết code súc tích bằng các high-level API của deep learning framework. Sau khi giải thích cách một thành phần hoạt động, chúng tôi sử dụng high-level API trong các tutorial tiếp theo.


### Nội dung và cấu trúc

Cuốn sách có thể được chia thành khoảng ba phần, lần lượt đề cập đến các kiến thức nhập môn, những kỹ thuật deep learning và các chủ đề nâng cao tập trung vào những hệ thống thực tế và ứng dụng (:numref:`fig_book_org`).

![Cấu trúc sách.](../img/book-org.svg)
:label:`fig_book_org`


* **Phần 1: Kiến thức cơ bản và phần nhập môn**.
:numref:`chap_introduction` là phần giới thiệu về deep learning.
Sau đó, trong :numref:`chap_preliminaries`,
chúng tôi nhanh chóng cung cấp cho bạn
các kiến thức tiên quyết cần thiết
để thực hành deep learning,
chẳng hạn như cách lưu trữ và thao tác với dữ liệu,
cũng như cách áp dụng nhiều phép toán số khác nhau
dựa trên các khái niệm cơ bản của đại số tuyến tính,
giải tích và xác suất.
:numref:`chap_regression` và :numref:`chap_perceptrons`
trình bày những khái niệm và kỹ thuật nền tảng nhất trong deep learning,
bao gồm hồi quy và phân loại;
các mô hình tuyến tính; perceptron đa lớp;
cũng như hiện tượng overfitting và regularization.

* **Phần 2: Các kỹ thuật deep learning hiện đại**.
:numref:`chap_computation` mô tả
các thành phần tính toán then chốt
của những hệ thống deep learning
và đặt nền tảng
cho việc triển khai các mô hình phức tạp hơn
trong các phần tiếp theo.
Tiếp đó, :numref:`chap_cnn` và :numref:`chap_modern_cnn`
trình bày các mạng neural tích chập (CNN),
những công cụ mạnh mẽ tạo nên nền tảng
cho hầu hết các hệ thống thị giác máy tính hiện đại.
Tương tự, :numref:`chap_rnn` và :numref:`chap_modern_rnn`
giới thiệu các mạng neural hồi quy (RNN),
những mô hình khai thác cấu trúc tuần tự (ví dụ: theo thời gian)
trong dữ liệu và thường được sử dụng
cho xử lý ngôn ngữ tự nhiên
và dự đoán chuỗi thời gian.
Trong :numref:`chap_attention-and-transformers`,
chúng tôi mô tả một lớp mô hình tương đối mới,
dựa trên cái gọi là *cơ chế attention*,
đã thay thế RNN để trở thành kiến trúc chủ đạo
cho hầu hết các tác vụ xử lý ngôn ngữ tự nhiên.
Các phần này sẽ giúp bạn nắm được
những công cụ mạnh mẽ và tổng quát nhất
được các chuyên gia deep learning sử dụng rộng rãi.

* **Phần 3: Khả năng mở rộng, hiệu quả và ứng dụng** (có sẵn [trực tuyến](https://d2l.ai)).
Trong Chương 12,
chúng tôi thảo luận về một số thuật toán tối ưu hóa phổ biến
được sử dụng để huấn luyện các mô hình deep learning.
Tiếp theo, trong Chương 13,
chúng tôi xem xét một số yếu tố then chốt
ảnh hưởng đến hiệu năng tính toán
của mã deep learning.
Sau đó, trong Chương 14,
chúng tôi minh họa những ứng dụng chính
của deep learning trong thị giác máy tính.
Cuối cùng, trong Chương 15 và Chương 16,
chúng tôi trình bày cách pretrain các mô hình biểu diễn ngôn ngữ
và áp dụng chúng vào các tác vụ xử lý ngôn ngữ tự nhiên.


### Mã
:label:`sec_code`

Hầu hết các phần trong sách này đều có mã có thể thực thi.
Chúng tôi tin rằng một số trực giác được hình thành tốt nhất
thông qua việc thử và sai,
điều chỉnh mã theo những cách nhỏ và quan sát kết quả.
Lý tưởng nhất là một lý thuyết toán học tao nhã có thể cho chúng ta biết
chính xác cách điều chỉnh mã để đạt được kết quả mong muốn.
Tuy nhiên, các chuyên gia deep learning ngày nay
thường phải tiến vào những lĩnh vực mà chưa có lý thuyết vững chắc nào đưa ra hướng dẫn.
Dù đã cố gắng hết sức, các lời giải thích mang tính hình thức
về tính hiệu quả của nhiều kỹ thuật khác nhau
vẫn còn thiếu, vì nhiều lý do: toán học dùng để đặc trưng hóa các mô hình này
có thể cực kỳ khó;
lời giải thích có thể phụ thuộc vào những tính chất
của dữ liệu hiện chưa có định nghĩa rõ ràng;
và việc nghiên cứu nghiêm túc về các chủ đề này
chỉ mới thực sự được đẩy mạnh trong thời gian gần đây.
Chúng tôi hy vọng rằng khi lý thuyết deep learning tiến bộ,
mỗi ấn bản tương lai của cuốn sách này sẽ cung cấp những hiểu biết
vượt qua những gì hiện đang có.

Để tránh lặp lại không cần thiết, chúng tôi tập hợp
một số hàm và lớp được import và sử dụng thường xuyên nhất
trong package `d2l`.
Trong toàn bộ cuốn sách, chúng tôi đánh dấu các khối mã
(chẳng hạn như hàm, lớp
hoặc tập hợp các câu lệnh import) bằng `#@save`
để cho biết rằng chúng sẽ được truy cập sau này
thông qua package `d2l`.
Chúng tôi cung cấp phần tổng quan chi tiết
về các lớp và hàm này trong :numref:`sec_d2l`.
Package `d2l` nhẹ và chỉ yêu cầu
các dependency sau:

```{.python .input}
#@tab all
#@save
import inspect
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`mxnet`
Hầu hết mã trong cuốn sách này dựa trên Apache MXNet,
một framework mã nguồn mở dành cho deep learning
được AWS (Amazon Web Services),
cũng như nhiều trường đại học và công ty, ưu tiên lựa chọn.
Toàn bộ mã trong cuốn sách này đã vượt qua các bài kiểm thử
trên phiên bản MXNet mới nhất.
Tuy nhiên, do deep learning phát triển nhanh chóng,
một số mã *trong ấn bản in*
có thể không hoạt động đúng trong các phiên bản MXNet tương lai.
Chúng tôi dự định cập nhật phiên bản trực tuyến.
Nếu gặp bất kỳ vấn đề nào,
vui lòng tham khảo :ref:`chap_installation`
để cập nhật mã và môi trường runtime của bạn.
Dưới đây là các dependency trong phần triển khai MXNet của chúng tôi.
:end_tab:

:begin_tab:`pytorch`
Hầu hết mã trong cuốn sách này dựa trên PyTorch,
một framework mã nguồn mở phổ biến
được cộng đồng nghiên cứu deep learning nhiệt tình đón nhận.
Toàn bộ mã trong cuốn sách này đã vượt qua các bài kiểm thử
trên phiên bản ổn định mới nhất của PyTorch.
Tuy nhiên, do deep learning phát triển nhanh chóng,
một số mã *trong ấn bản in*
có thể không hoạt động đúng trong các phiên bản PyTorch tương lai.
Chúng tôi dự định cập nhật phiên bản trực tuyến.
Nếu gặp bất kỳ vấn đề nào,
vui lòng tham khảo :ref:`chap_installation`
để cập nhật mã và môi trường runtime của bạn.
Dưới đây là các dependency trong phần triển khai PyTorch của chúng tôi.
:end_tab:

:begin_tab:`tensorflow`
Hầu hết mã trong cuốn sách này dựa trên TensorFlow,
một framework mã nguồn mở dành cho deep learning
được áp dụng rộng rãi trong ngành
và phổ biến trong giới nghiên cứu.
Toàn bộ mã trong cuốn sách này đã vượt qua các bài kiểm thử
trên phiên bản ổn định mới nhất của TensorFlow.
Tuy nhiên, do deep learning phát triển nhanh chóng,
một số mã *trong ấn bản in*
có thể không hoạt động đúng trong các phiên bản TensorFlow tương lai.
Chúng tôi dự định cập nhật phiên bản trực tuyến.
Nếu gặp bất kỳ vấn đề nào,
vui lòng tham khảo :ref:`chap_installation`
để cập nhật mã và môi trường runtime của bạn.
Dưới đây là các dependency trong phần triển khai TensorFlow của chúng tôi.
:end_tab:

:begin_tab:`jax`
Hầu hết mã trong cuốn sách này dựa trên Jax,
một framework mã nguồn mở cho phép
biến đổi các hàm có thể kết hợp, chẳng hạn như phép vi phân
của các hàm Python và NumPy tùy ý, cũng như biên dịch JIT,
vectorization và nhiều khả năng khác! Framework này đang trở nên phổ biến trong
lĩnh vực nghiên cứu machine learning và có
API tương tự NumPy, dễ học. Trên thực tế, JAX cố gắng
đạt được mức tương thích 1:1 với NumPy, vì vậy việc chuyển đổi
mã của bạn có thể đơn giản như thay đổi một câu lệnh import duy nhất!
Tuy nhiên, do deep learning phát triển nhanh chóng,
một số mã *trong ấn bản in*
có thể không hoạt động đúng trong các phiên bản Jax tương lai.
Chúng tôi dự định cập nhật phiên bản trực tuyến.
Nếu gặp bất kỳ vấn đề nào,
vui lòng tham khảo :ref:`chap_installation`
để cập nhật mã và môi trường runtime của bạn.
Dưới đây là các dependency trong phần triển khai JAX của chúng tôi.
:end_tab:

```{.python .input}
#@tab mxnet
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from scipy.spatial import distance_matrix
```

```{.python .input}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab jax
#@save
from dataclasses import field
from functools import partial
import flax
from flax import linen as nn
from flax.training import train_state
import jax
from jax import numpy as jnp
from jax import grad, vmap
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from types import FunctionType
from typing import Any
```

### Đối tượng độc giả

Cuốn sách này dành cho sinh viên (bậc đại học hoặc sau đại học),
kỹ sư và nhà nghiên cứu muốn nắm vững
các kỹ thuật thực tiễn của deep learning.
Vì chúng tôi giải thích mọi khái niệm từ đầu,
bạn không cần có nền tảng trước đó về deep learning hoặc machine learning.
Việc giải thích đầy đủ các phương pháp deep learning
đòi hỏi một số kiến thức toán học và lập trình,
nhưng chúng tôi chỉ giả định rằng bạn đã có một số kiến thức cơ bản,
bao gồm một lượng kiến thức vừa phải về đại số tuyến tính,
giải tích, xác suất và lập trình Python.
Để đề phòng trường hợp bạn đã quên điều gì đó,
[Phụ lục trực tuyến](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html) cung cấp phần ôn tập
về hầu hết kiến thức toán học
mà bạn sẽ gặp trong cuốn sách này.
Thông thường, chúng tôi sẽ ưu tiên
trực giác và ý tưởng
hơn tính chặt chẽ về mặt toán học.
Nếu bạn muốn mở rộng những nền tảng này
vượt ra ngoài các kiến thức tiên quyết để hiểu cuốn sách,
chúng tôi rất vui được giới thiệu một số tài liệu tuyệt vời khác:
*Linear Analysis* của :citet:`Bollobas.1999`
trình bày đại số tuyến tính và giải tích hàm một cách chuyên sâu.
*All of Statistics* :cite:`Wasserman.2013`
cung cấp phần nhập môn tuyệt vời về thống kê.
Các [cuốn sách](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1138369918)
và [khóa học](https://projects.iq.harvard.edu/stat110/home) của Joe Blitzstein
về xác suất và suy luận là những tài liệu sư phạm quý giá.
Và nếu trước đây bạn chưa từng sử dụng Python,
bạn có thể muốn xem qua [hướng dẫn Python](http://learnpython.org/) này.


### Notebook, Website, GitHub và Forum

Bạn có thể tải xuống toàn bộ notebook của chúng tôi
từ [website D2L.ai](https://d2l.ai)
và từ [GitHub](https://github.com/d2l-ai/d2l-en).
Cùng với cuốn sách này, chúng tôi đã ra mắt một forum thảo luận
tại [discuss.d2l.ai](https://discuss.d2l.ai/c/5).
Bất cứ khi nào có câu hỏi về bất kỳ phần nào của cuốn sách,
bạn có thể tìm thấy liên kết đến trang thảo luận tương ứng
ở cuối mỗi notebook.



## Lời cảm ơn

Chúng tôi mang ơn hàng trăm cộng tác viên đã đóng góp cho cả bản thảo tiếng Anh và tiếng Trung.
Họ đã giúp cải thiện nội dung và đưa ra những phản hồi quý báu.
Cuốn sách này ban đầu được xây dựng với MXNet là framework chính.
Chúng tôi cảm ơn Anirudh Dagar và Yuan Tang đã lần lượt chuyển phần lớn mã MXNet trước đây sang các bản triển khai bằng PyTorch và TensorFlow.
Kể từ tháng 7 năm 2021, chúng tôi đã thiết kế lại và triển khai lại cuốn sách này bằng PyTorch, MXNet và TensorFlow, đồng thời chọn PyTorch làm framework chính.
Chúng tôi cảm ơn Anirudh Dagar đã chuyển phần lớn mã PyTorch mới hơn sang các bản triển khai bằng JAX.
Chúng tôi cảm ơn Gaosheng Wu, Liujun Hu, Ge Zhang và Jiehang Xie từ Baidu đã chuyển phần lớn mã PyTorch mới hơn sang các bản triển khai bằng PaddlePaddle trong bản thảo tiếng Trung.
Chúng tôi cảm ơn Shuai Zhang đã tích hợp style LaTeX từ nhà xuất bản vào quá trình xây dựng PDF.

Trên GitHub, chúng tôi cảm ơn mọi cộng tác viên của bản thảo tiếng Anh này
vì đã làm cho bản thảo trở nên tốt hơn đối với tất cả mọi người.
GitHub ID hoặc tên của họ là (không theo thứ tự cụ thể):
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU,
Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder,
mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD,
TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, Yue Ying, tayfununal,
steinsag, charleybeller, Andrew Lumsdaine, Jiekui Zhang, Deepak Pathak, Florian Donhauser, Tim Gates,
Adriaan Tijsseling, Ron Medina, Gaurav Saha, Murat Semerci, Lei Mao, Levi McClenny, Joshua Broyde,
jake221, jonbally, zyhazwraith, Brian Pulfer, Nick Tomasino, Lefan Zhang, Hongshen Yang, Vinney Cavallo,
yuntai, Yuanxiang Zhu, amarazov, pasricha, Ben Greenawald, Shivam Upadhyay, Quanshangze Du, Biswajit Sahoo,
Parthe Pandit, Ishan Kumar, HomunculusK, Lane Schwartz, varadgunjal, Jason Wiener, Armin Gholampoor,
Shreshtha13, eigen-arnav, Hyeonggyu Kim, EmilyOng, Bálint Mucsányi, Chase DuBois, Juntian Tao,
Wenxiang Xu, Lifu Huang, filevich, quake2005, nils-werner, Yiming Li, Marsel Khisamutdinov,
Francesco "Fuma" Fumagalli, Peilin Sun, Vincent Gurgul, qingfengtommy, Janmey Shukla, Mo Shan,
Kaan Sancak, regob, AlexSauer, Gopalakrishna Ramachandra, Tobias Uelwer, Chao Wang, Tian Cao,
Nicolas Corthorn, akash5474, kxxt, zxydi1992, Jacob Britton, Shuangchi He, zhmou, krahets, Jie-Han Chen,
Atishay Garg, Marcel Flygare, adtygan, Nik Vaessen, bolded, Louis Schlessinger, Balaji Varatharajan,
atgctg, Kaixin Li, Victor Barbaros, Riccardo Musto, Elizabeth Ho, azimjonn, Guilherme Miotto, Alessandro Finamore,
Joji Joseph, Anthony Biel, Zeming Zhao, shjustinbaek, gab-chen, nantekoto, Yutaro Nishiyama, Oren Amsalem,
Tian-MaoMao, Amin Allahyar, Gijs van Tulder, Mikhail Berkov, iamorphen, Matthew Caseres, Andrew Walsh,
pggPL, RohanKarthikeyan, Ryan Choi và Likun Lei.

Chúng tôi cảm ơn Amazon Web Services, đặc biệt là Wen-Ming Ye, George Karypis, Swami Sivasubramanian, Peter DeSantis, Adam Selipsky
và Andrew Jassy vì sự hỗ trợ hào phóng của họ trong quá trình viết cuốn sách này.
Nếu không có thời gian, nguồn lực, những cuộc thảo luận với đồng nghiệp
và sự động viên liên tục, cuốn sách này đã không thể ra đời.
Trong quá trình chuẩn bị xuất bản cuốn sách,
Cambridge University Press đã hỗ trợ chúng tôi rất tận tình.
Chúng tôi cảm ơn commissioning editor David Tranah
vì sự giúp đỡ và tính chuyên nghiệp của ông.


## Tóm tắt

Deep learning đã cách mạng hóa việc nhận dạng mẫu,
đưa vào những công nghệ hiện đang vận hành nhiều loại công nghệ,
trong các lĩnh vực đa dạng như thị giác máy tính,
xử lý ngôn ngữ tự nhiên
và nhận dạng tiếng nói tự động.
Để áp dụng deep learning thành công,
bạn phải hiểu cách phát biểu một bài toán,
toán học cơ bản của việc lập mô hình,
các thuật toán để điều chỉnh mô hình theo dữ liệu
và các kỹ thuật engineering để triển khai toàn bộ quy trình.
Cuốn sách này cung cấp một tài liệu toàn diện,
bao gồm văn bản, hình, toán học và code, tất cả trong cùng một nơi.



## Bài tập

1. Đăng ký tài khoản trên diễn đàn thảo luận của cuốn sách này [discuss.d2l.ai](https://discuss.d2l.ai/).
1. Cài đặt Python trên máy tính của bạn.
1. Theo các liên kết ở cuối phần này đến diễn đàn, nơi bạn có thể tìm kiếm sự trợ giúp, thảo luận về cuốn sách và tìm câu trả lời cho các câu hỏi của mình bằng cách trao đổi với các tác giả và cộng đồng rộng lớn hơn.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/186)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/17963)
:end_tab: