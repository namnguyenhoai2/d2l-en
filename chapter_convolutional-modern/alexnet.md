```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Mạng nơ-ron tích chập sâu (AlexNet)
:label:`sec_alexnet`


Mặc dù CNN đã được biết đến rộng rãi
trong cộng đồng thị giác máy tính và machine learning
sau khi LeNet được giới thiệu :cite:`LeCun.Jackel.Bottou.ea.1995`,
chúng không ngay lập tức chiếm ưu thế trong lĩnh vực này.
Mặc dù LeNet đạt kết quả tốt trên các bộ dữ liệu nhỏ thời kỳ đầu,
hiệu năng và tính khả thi của việc huấn luyện CNN
trên các bộ dữ liệu lớn hơn, thực tế hơn vẫn chưa được xác lập.
Trên thực tế, trong phần lớn khoảng thời gian từ đầu những năm 1990
đến các kết quả mang tính bước ngoặt năm 2012 :cite:`Krizhevsky.Sutskever.Hinton.2012`,
mạng nơ-ron thường bị vượt qua bởi các phương pháp machine learning khác,
chẳng hạn như phương pháp kernel :cite:`Scholkopf.Smola.2002`, phương pháp ensemble :cite:`Freund.Schapire.ea.1996`,
và ước lượng có cấu trúc :cite:`Taskar.Guestrin.Koller.2004`.

Đối với thị giác máy tính, sự so sánh này có lẽ không hoàn toàn chính xác.
Cụ thể, mặc dù đầu vào của mạng tích chập
gồm các giá trị pixel thô hoặc được xử lý nhẹ (ví dụ: căn giữa), các nhà thực hành sẽ không bao giờ đưa pixel thô vào các mô hình truyền thống.
Thay vào đó, các pipeline thị giác máy tính điển hình
bao gồm các pipeline trích xuất đặc trưng được thiết kế thủ công, như SIFT :cite:`Lowe.2004`, SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`, và túi từ trực quan :cite:`Sivic.Zisserman.2003`.
Thay vì *học* các đặc trưng, các đặc trưng được *chế tác*.
Phần lớn tiến bộ đến từ các ý tưởng thông minh hơn về trích xuất đặc trưng ở một mặt và hiểu biết sâu sắc về hình học :cite:`Hartley.Zisserman.2000` ở mặt khác. Thuật toán học thường được xem là yếu tố phụ.

Mặc dù một số bộ tăng tốc mạng nơ-ron đã có mặt vào những năm 1990,
chúng vẫn chưa đủ mạnh để xây dựng
các CNN sâu, đa kênh, nhiều tầng
với số lượng lớn tham số. Ví dụ, NVIDIA GeForce 256 từ năm 1999
chỉ có thể xử lý tối đa 480 triệu phép toán dấu phẩy động, như phép cộng và phép nhân, mỗi giây (MFLOPS), mà không có bất kỳ
framework lập trình có ý nghĩa nào cho các thao tác ngoài game. Các bộ tăng tốc ngày nay có thể thực hiện hơn 1000 TFLOPs trên mỗi thiết bị.
Hơn nữa, các bộ dữ liệu khi đó vẫn tương đối nhỏ: OCR trên 60.000 ảnh pixel độ phân giải thấp $28 \times 28$ được xem là một nhiệm vụ cực kỳ thách thức.
Bên cạnh các trở ngại này, những thủ thuật then chốt để huấn luyện mạng nơ-ron
bao gồm heuristic khởi tạo tham số :cite:`Glorot.Bengio.2010`,
các biến thể thông minh của stochastic gradient descent :cite:`Kingma.Ba.2014`,
các hàm kích hoạt không bão hòa :cite:`Nair.Hinton.2010`,
và các kỹ thuật regularization hiệu quả :cite:`Srivastava.Hinton.Krizhevsky.ea.2014` vẫn còn thiếu.

Vì vậy, thay vì huấn luyện các hệ thống *end-to-end* (từ pixel đến phân loại),
các pipeline cổ điển trông giống như sau:

1. Thu thập một bộ dữ liệu thú vị. Trong những ngày đầu, các bộ dữ liệu này đòi hỏi cảm biến đắt tiền. Ví dụ, [Apple QuickTake 100](https://en.wikipedia.org/wiki/Apple_QuickTake) năm 1994 sở hữu độ phân giải đáng kinh ngạc 0,3 megapixel (VGA), có thể lưu trữ tối đa 8 ảnh, với mức giá \$1000.
1. Tiền xử lý bộ dữ liệu bằng các đặc trưng được thiết kế thủ công dựa trên một số kiến thức về quang học, hình học, các công cụ phân tích khác, và đôi khi là những khám phá tình cờ của các nghiên cứu sinh may mắn.
1. Đưa dữ liệu qua một tập hợp chuẩn các bộ trích xuất đặc trưng như SIFT (scale-invariant feature transform) :cite:`Lowe.2004`, SURF (speeded up robust features) :cite:`Bay.Tuytelaars.Van-Gool.2006`, hoặc bất kỳ số lượng pipeline tinh chỉnh thủ công nào khác. OpenCV đến nay vẫn cung cấp các bộ trích xuất SIFT!
1. Đưa các biểu diễn thu được vào classifier yêu thích của bạn, có thể là một mô hình tuyến tính hoặc phương pháp kernel, để huấn luyện một classifier.

Nếu bạn nói chuyện với các nhà nghiên cứu machine learning,
họ sẽ trả lời rằng machine learning vừa quan trọng vừa đẹp đẽ.
Các lý thuyết thanh lịch chứng minh những tính chất của nhiều classifier khác nhau :cite:`boucheron2005theory` và tối ưu hóa lồi
:cite:`Boyd.Vandenberghe.2004` đã trở thành nền tảng chính để thu được chúng.
Lĩnh vực machine learning đang phát triển mạnh, chặt chẽ, và cực kỳ hữu ích. Tuy nhiên,
nếu bạn nói chuyện với một nhà nghiên cứu thị giác máy tính,
bạn sẽ nghe một câu chuyện rất khác.
Sự thật trần trụi của nhận dạng ảnh, theo họ,
là các đặc trưng, hình học :cite:`Hartley.Zisserman.2000,hartley2009global`, và kỹ thuật,
chứ không phải các thuật toán học mới lạ, đã thúc đẩy tiến bộ.
Các nhà nghiên cứu thị giác máy tính có lý khi tin rằng
một bộ dữ liệu lớn hơn hoặc sạch hơn đôi chút
hoặc một pipeline trích xuất đặc trưng được cải thiện đôi chút
quan trọng hơn rất nhiều đối với độ chính xác cuối cùng so với bất kỳ thuật toán học nào.

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Học biểu diễn

Một cách khác để mô tả tình hình này là
phần quan trọng nhất của pipeline chính là biểu diễn.
Và cho đến năm 2012, biểu diễn chủ yếu được tính toán một cách cơ học.
Trên thực tế, việc thiết kế một tập hàm đặc trưng mới, cải thiện kết quả, và viết phương pháp
đều là những nội dung nổi bật trong các bài báo.
SIFT :cite:`Lowe.2004`,
SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`,
HOG (histograms of oriented gradient) :cite:`Dalal.Triggs.2005`,
túi từ trực quan :cite:`Sivic.Zisserman.2003`,
và các bộ trích xuất đặc trưng tương tự đã thống trị lĩnh vực này.

Một nhóm nhà nghiên cứu khác,
bao gồm Yann LeCun, Geoff Hinton, Yoshua Bengio,
Andrew Ng, Shun-ichi Amari, và Juergen Schmidhuber,
có những kế hoạch khác.
Họ tin rằng bản thân các đặc trưng nên được học.
Hơn nữa, họ tin rằng để có độ phức tạp hợp lý,
các đặc trưng nên được cấu thành theo cấu trúc phân cấp
với nhiều tầng được học đồng thời, mỗi tầng có các tham số có thể học.
Trong trường hợp ảnh, các tầng thấp nhất có thể học
cách phát hiện cạnh, màu sắc và kết cấu, tương tự cách hệ thị giác của động vật
xử lý đầu vào. Đặc biệt, việc tự động thiết kế các đặc trưng trực quan như những đặc trưng thu được
bằng sparse coding :cite:`olshausen1996emergence` vẫn là một thách thức mở cho đến khi CNN hiện đại xuất hiện.
Mãi đến :citet:`Dean.Corrado.Monga.ea.2012,le2013building` thì ý tưởng tự động sinh đặc trưng
từ dữ liệu ảnh mới thu hút được sự quan tâm đáng kể.

CNN hiện đại đầu tiên :cite:`Krizhevsky.Sutskever.Hinton.2012`, mang tên
*AlexNet* theo tên một trong các nhà phát minh của nó, Alex Krizhevsky, về cơ bản là một cải tiến tiến hóa
so với LeNet. Nó đạt hiệu năng xuất sắc trong thử thách ImageNet năm 2012.

![Các bộ lọc ảnh được học bởi tầng đầu tiên của AlexNet. Tái bản với sự cho phép của :citet:`Krizhevsky.Sutskever.Hinton.2012`.](../img/filters.png)
:width:`400px`
:label:`fig_filters`

Thú vị là, ở các tầng thấp nhất của mạng,
mô hình đã học các bộ trích xuất đặc trưng tương tự một số bộ lọc truyền thống.
:numref:`fig_filters`
minh họa các bộ mô tả ảnh cấp thấp.
Các tầng cao hơn trong mạng có thể xây dựng trên các biểu diễn này
để biểu diễn những cấu trúc lớn hơn, như mắt, mũi, ngọn cỏ, v.v.
Các tầng còn cao hơn có thể biểu diễn toàn bộ đối tượng
như con người, máy bay, chó, hoặc đĩa frisbee.
Cuối cùng, trạng thái ẩn cuối cùng học một biểu diễn cô đọng
của ảnh, tóm tắt nội dung của nó
sao cho dữ liệu thuộc các hạng mục khác nhau có thể được phân tách dễ dàng.

AlexNet (2012) và tiền thân của nó là LeNet (1995) chia sẻ nhiều thành phần kiến trúc. Điều này đặt ra câu hỏi: tại sao lại mất nhiều thời gian đến vậy?
Một khác biệt then chốt là, trong hai thập kỷ trước đó, lượng dữ liệu và năng lực tính toán sẵn có đã tăng lên đáng kể. Do đó AlexNet lớn hơn nhiều: nó được huấn luyện trên nhiều dữ liệu hơn và trên các GPU nhanh hơn rất nhiều so với CPU có sẵn vào năm 1995.

### Thành phần còn thiếu: Dữ liệu

Các mô hình sâu với nhiều tầng đòi hỏi lượng dữ liệu lớn
để đạt đến chế độ
mà chúng vượt trội đáng kể so với các phương pháp truyền thống
dựa trên tối ưu hóa lồi (ví dụ: phương pháp tuyến tính và kernel).
Tuy nhiên, do dung lượng lưu trữ hạn chế của máy tính,
chi phí tương đối cao của cảm biến (chụp ảnh),
và ngân sách nghiên cứu tương đối eo hẹp hơn vào những năm 1990,
phần lớn nghiên cứu dựa vào các bộ dữ liệu rất nhỏ.
Nhiều bài báo dựa vào bộ sưu tập bộ dữ liệu UCI,
trong đó nhiều bộ chỉ chứa hàng trăm hoặc (vài) nghìn ảnh
được chụp ở độ phân giải thấp và thường có nền sạch nhân tạo.

Năm 2009, bộ dữ liệu ImageNet được phát hành :cite:`Deng.Dong.Socher.ea.2009`,
thách thức các nhà nghiên cứu học mô hình từ 1 triệu ví dụ,
mỗi hạng mục gồm 1000 ví dụ thuộc 1000 hạng mục đối tượng riêng biệt. Bản thân các hạng mục
dựa trên các node danh từ phổ biến nhất trong WordNet :cite:`Miller.1995`.
Nhóm ImageNet đã sử dụng Google Image Search để tiền lọc các tập ứng viên lớn
cho mỗi hạng mục và sử dụng
pipeline crowdsourcing Amazon Mechanical Turk
để xác nhận với từng ảnh xem nó có thuộc hạng mục liên quan hay không.
Quy mô này chưa từng có tiền lệ, vượt các bộ khác hơn một bậc độ lớn
(ví dụ: CIFAR-100 có 60.000 ảnh). Một khía cạnh khác là ảnh có
độ phân giải tương đối cao $224 \times 224$ pixel, không giống bộ dữ liệu
TinyImages với 80 triệu ảnh :cite:`Torralba.Fergus.Freeman.2008`, gồm các ảnh thumbnail $32 \times 32$ pixel.
Điều này cho phép hình thành các đặc trưng cấp cao hơn.
Cuộc thi liên quan, được gọi là ImageNet Large Scale Visual Recognition
Challenge :cite:`russakovsky2015imagenet`,
đã thúc đẩy nghiên cứu thị giác máy tính và machine learning tiến lên,
thách thức các nhà nghiên cứu xác định mô hình nào có hiệu năng tốt nhất
ở quy mô lớn hơn những gì giới học thuật từng xem xét. Các bộ dữ liệu thị giác lớn nhất, như LAION-5B
:cite:`schuhmann2022laion` chứa hàng tỷ ảnh cùng metadata bổ sung.

### Thành phần còn thiếu: Phần cứng

Các mô hình deep learning tiêu thụ chu kỳ tính toán rất lớn.
Việc huấn luyện có thể mất hàng trăm epoch, và mỗi lần lặp
đòi hỏi đưa dữ liệu đi qua nhiều tầng gồm các phép toán đại số tuyến tính
tốn kém về mặt tính toán.
Đây là một trong các lý do chính khiến trong những năm 1990 và đầu những năm 2000,
các thuật toán đơn giản dựa trên những hàm mục tiêu lồi (convex objectives)
được tối ưu hiệu quả hơn đã được ưa chuộng.

*Graphics processing units* (GPU) đã chứng minh là một yếu tố thay đổi cuộc chơi
giúp deep learning trở nên khả thi.
Những chip này trước đó được phát triển để tăng tốc
xử lý đồ họa nhằm phục vụ trò chơi điện tử.
Cụ thể, chúng được tối ưu cho các phép nhân ma trận--vector $4 \times 4$
có throughput cao, vốn cần thiết cho nhiều tác vụ đồ họa máy tính.
May mắn thay, phép toán này rất tương đồng
với phép toán cần thiết để tính các tầng convolution.
Vào khoảng thời gian đó, NVIDIA và ATI đã bắt đầu tối ưu GPU
cho các phép toán tính toán tổng quát :cite:`Fernando.2004`,
thậm chí tiếp thị chúng dưới tên gọi *general-purpose GPUs* (GPGPU).

Để có trực giác, hãy xét các lõi của một vi xử lý hiện đại
(CPU).
Mỗi lõi khá mạnh, chạy ở tần số xung nhịp cao
và có cache lớn (lên đến vài megabyte L3).
Mỗi lõi phù hợp để thực thi nhiều loại lệnh,
với bộ dự đoán nhánh, pipeline sâu, các đơn vị thực thi chuyên dụng,
thực thi suy đoán,
và nhiều tính năng bổ sung khác
cho phép nó chạy nhiều loại chương trình có control flow phức tạp.
Tuy nhiên, sức mạnh rõ ràng này cũng chính là gót chân Achilles của nó:
các lõi đa dụng rất đắt để chế tạo. Chúng xuất sắc với mã đa dụng
có nhiều control flow.
Điều này đòi hỏi nhiều diện tích chip, không chỉ dành cho
ALU (arithmetic logical unit) thực hiện tính toán, mà còn cho
toàn bộ các tính năng bổ sung đã đề cập, cùng với
giao diện bộ nhớ, logic caching giữa các lõi,
kết nối liên thông tốc độ cao, v.v. CPU
tương đối kém trong bất kỳ tác vụ đơn lẻ nào khi so sánh với phần cứng chuyên dụng.
Laptop hiện đại có 4--8 lõi,
và ngay cả các server cao cấp cũng hiếm khi vượt quá 64 lõi trên mỗi socket,
đơn giản vì điều đó không hiệu quả về chi phí.

Ngược lại, GPU có thể gồm hàng nghìn phần tử xử lý nhỏ (các chip Ampere mới nhất của NIVIDA có tới 6912 lõi CUDA), thường được nhóm thành các nhóm lớn hơn (NVIDIA gọi chúng là warp).
Chi tiết có khác biệt đôi chút giữa NVIDIA, AMD, ARM và các nhà cung cấp chip khác. Dù mỗi lõi tương đối yếu,
chạy ở tần số xung nhịp khoảng 1GHz,
chính tổng số lượng các lõi này khiến GPU nhanh hơn CPU nhiều bậc độ lớn.
Chẳng hạn, GPU Ampere A100 gần đây của NVIDIA cung cấp hơn 300 TFLOPs mỗi chip cho các phép nhân ma trận-ma trận chuyên dụng với độ chính xác 16-bit (BFLOAT16), và đến 20 TFLOPs cho các phép toán số thực đa dụng hơn (FP32).
Trong khi đó, hiệu năng số thực của CPU hiếm khi vượt quá 1 TFLOPs. Chẳng hạn, Graviton 3 của Amazon đạt hiệu năng đỉnh 2 TFLOPs cho các phép toán độ chính xác 16-bit, một con số tương tự hiệu năng GPU của bộ xử lý M1 của Apple.

Có nhiều lý do khiến GPU nhanh hơn CPU rất nhiều xét theo FLOPs.
Thứ nhất, mức tiêu thụ điện năng có xu hướng tăng *theo bậc hai* cùng tần số xung nhịp.
Do đó, với ngân sách điện năng của một lõi CPU chạy nhanh hơn bốn lần (một con số điển hình),
bạn có thể dùng 16 lõi GPU ở $\frac{1}{4}$ tốc độ,
mang lại $16 \times \frac{1}{4} = 4$ lần hiệu năng.
Thứ hai, các lõi GPU đơn giản hơn nhiều
(thực tế, trong một thời gian dài chúng thậm chí không *có khả năng*
thực thi mã đa dụng),
điều này khiến chúng tiết kiệm năng lượng hơn. Chẳng hạn, (i) chúng thường không hỗ trợ thực thi suy đoán, (ii) thường không thể lập trình riêng từng phần tử xử lý, và (iii) cache trên mỗi lõi thường nhỏ hơn nhiều.
Cuối cùng, nhiều phép toán trong deep learning đòi hỏi băng thông bộ nhớ cao.
Một lần nữa, GPU nổi bật ở điểm này với các bus rộng ít nhất gấp 10 lần nhiều CPU.

Quay lại năm 2012. Một đột phá lớn đã xuất hiện
khi Alex Krizhevsky và Ilya Sutskever
triển khai một CNN sâu
có thể chạy trên GPU.
Họ nhận ra rằng các nút thắt tính toán trong CNN,
convolution và phép nhân ma trận,
đều là các phép toán có thể được song song hóa bằng phần cứng.
Sử dụng hai NVIDIA GTX 580 với bộ nhớ 3GB, mỗi chiếc có khả năng đạt 1.5 TFLOPs (vẫn là thách thức với hầu hết CPU một thập kỷ sau đó),
họ đã triển khai các phép convolution nhanh.
Mã [cuda-convnet](https://code.google.com/archive/p/cuda-convnet/)
đủ tốt để trong vài năm
trở thành tiêu chuẩn ngành và thúc đẩy
những năm đầu tiên của làn sóng bùng nổ deep learning.

## AlexNet

AlexNet, sử dụng một CNN 8 tầng,
đã chiến thắng ImageNet Large Scale Visual Recognition Challenge 2012
với cách biệt lớn :cite:`Russakovsky.Deng.Huang.ea.2013`.
Mạng này lần đầu tiên cho thấy
các đặc trưng thu được từ việc học có thể vượt qua các đặc trưng được thiết kế thủ công, phá vỡ mô hình trước đây trong computer vision.

Kiến trúc của AlexNet và LeNet rất giống nhau,
như :numref:`fig_alexnet` minh họa.
Lưu ý rằng chúng tôi cung cấp một phiên bản AlexNet được tinh giản đôi chút,
loại bỏ một số điểm đặc thù trong thiết kế vốn cần thiết vào năm 2012
để mô hình vừa với hai GPU nhỏ.

![Từ LeNet (bên trái) đến AlexNet (bên phải).](../img/alexnet.svg)
:label:`fig_alexnet`

Cũng có những khác biệt đáng kể giữa AlexNet và LeNet.
Thứ nhất, AlexNet sâu hơn nhiều so với LeNet-5 tương đối nhỏ.
AlexNet gồm tám tầng: năm tầng convolution,
hai tầng ẩn fully connected và một tầng đầu ra fully connected.
Thứ hai, AlexNet sử dụng ReLU thay cho sigmoid
làm hàm kích hoạt. Hãy đi sâu vào các chi tiết dưới đây.

### Kiến trúc

Ở tầng đầu tiên của AlexNet, hình dạng cửa sổ convolution là $11\times11$.
Vì ảnh trong ImageNet cao và rộng gấp tám lần
so với ảnh MNIST,
các đối tượng trong dữ liệu ImageNet thường chiếm nhiều pixel hơn và có nhiều chi tiết trực quan hơn.
Do đó, cần một cửa sổ convolution lớn hơn để nắm bắt đối tượng.
Hình dạng cửa sổ convolution ở tầng thứ hai
được giảm xuống $5\times5$, sau đó là $3\times3$.
Ngoài ra, sau tầng convolution thứ nhất, thứ hai và thứ năm,
mạng thêm các tầng max-pooling
với hình dạng cửa sổ $3\times3$ và stride là 2.
Hơn nữa, AlexNet có số kênh convolution nhiều gấp mười lần LeNet.

Sau tầng convolution cuối cùng là hai tầng fully connected rất lớn
với 4096 đầu ra.
Các tầng này cần gần 1GB tham số mô hình.
Do bộ nhớ hạn chế của GPU đời đầu,
AlexNet gốc sử dụng thiết kế luồng dữ liệu kép,
để mỗi trong hai GPU của chúng chỉ cần chịu trách nhiệm
lưu trữ và tính toán một nửa mô hình.
May mắn là bộ nhớ GPU hiện nay tương đối dồi dào,
nên ngày nay chúng ta hiếm khi cần chia nhỏ mô hình trên các GPU
(phiên bản mô hình AlexNet của chúng tôi khác
so với bài báo gốc ở khía cạnh này).

### Hàm kích hoạt

Hơn nữa, AlexNet đã thay đổi hàm kích hoạt sigmoid thành hàm kích hoạt ReLU đơn giản hơn. Một mặt, việc tính toán hàm kích hoạt ReLU đơn giản hơn. Chẳng hạn, nó không có phép lũy thừa như trong hàm kích hoạt sigmoid.
 Mặt khác, hàm kích hoạt ReLU giúp việc huấn luyện mô hình dễ dàng hơn khi sử dụng các phương pháp khởi tạo tham số khác nhau. Điều này là do khi đầu ra của hàm kích hoạt sigmoid rất gần 0 hoặc 1, gradient tại các vùng này gần như bằng 0, vì vậy backpropagation không thể tiếp tục cập nhật một số tham số của mô hình. Ngược lại, gradient của hàm kích hoạt ReLU trong khoảng dương luôn bằng 1 (:numref:`subsec_activation-functions`). Do đó, nếu các tham số mô hình không được khởi tạo đúng cách, hàm sigmoid có thể nhận được gradient gần bằng 0 trong khoảng dương, nghĩa là mô hình không thể được huấn luyện hiệu quả.

### Kiểm soát dung lượng và tiền xử lý

AlexNet kiểm soát độ phức tạp mô hình của tầng fully connected
bằng dropout (:numref:`sec_dropout`),
trong khi LeNet chỉ sử dụng weight decay.
Để tăng cường dữ liệu hơn nữa, vòng lặp huấn luyện của AlexNet
đã thêm rất nhiều phép image augmentation,
chẳng hạn như lật, cắt và thay đổi màu sắc.
Điều này làm mô hình mạnh mẽ hơn và kích thước mẫu lớn hơn giúp giảm overfitting một cách hiệu quả.
Xem :citet:`Buslaev.Iglovikov.Khvedchenya.ea.2020` để có bài đánh giá chuyên sâu về các bước tiền xử lý như vậy.

```{.python .input  n=5}
%%tab pytorch, mxnet, tensorflow
class AlexNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
                nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
                nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
                nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
                nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
                nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
                nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5),
                nn.LazyLinear(num_classes))
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class AlexNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(features=96, kernel_size=(11, 11), strides=4, padding=1),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            nn.Conv(features=256, kernel_size=(5, 5)),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            nn.Conv(features=384, kernel_size=(3, 3)), nn.relu,
            nn.Conv(features=384, kernel_size=(3, 3)), nn.relu,
            nn.Conv(features=256, kernel_size=(3, 3)), nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(features=4096),
            nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(features=4096),
            nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(features=self.num_classes)
        ])
```

Chúng tôi [**xây dựng một ví dụ dữ liệu một kênh**] có cả chiều cao lẫn chiều rộng là 224 (**để quan sát hình dạng đầu ra của từng tầng**). Nó khớp với kiến trúc AlexNet trong :numref:`fig_alexnet`.

```{.python .input  n=6}
%%tab pytorch, mxnet
AlexNet().layer_summary((1, 1, 224, 224))
```

```{.python .input  n=7}
%%tab tensorflow
AlexNet().layer_summary((1, 224, 224, 1))
```

```{.python .input}
%%tab jax
AlexNet(training=False).layer_summary((1, 224, 224, 1))
```

## Huấn luyện

Mặc dù AlexNet đã được huấn luyện trên ImageNet trong :citet:`Krizhevsky.Sutskever.Hinton.2012`,
ở đây chúng tôi sử dụng Fashion-MNIST
vì việc huấn luyện một mô hình ImageNet đến hội tụ có thể mất hàng giờ hoặc hàng ngày
ngay cả trên một GPU hiện đại.
Một trong các vấn đề khi áp dụng trực tiếp AlexNet lên [**Fashion-MNIST**]
là (**ảnh của nó có độ phân giải thấp hơn**) ($28 \times 28$ pixel)
(**so với ảnh ImageNet.**)
Để mọi thứ hoạt động, (**chúng tôi nội suy tăng kích thước chúng lên $224 \times 224$**).
Đây nhìn chung không phải là một thực hành thông minh, vì nó chỉ làm tăng độ phức tạp
tính toán mà không bổ sung thông tin. Tuy nhiên, chúng tôi làm điều này ở đây để trung thành với kiến trúc AlexNet.
Chúng tôi thực hiện việc thay đổi kích thước này bằng đối số `resize` trong constructor `d2l.FashionMNIST`.

Bây giờ, chúng ta có thể [**bắt đầu huấn luyện AlexNet.**]
So với LeNet trong :numref:`sec_lenet`,
thay đổi chính ở đây là sử dụng learning rate nhỏ hơn
và tốc độ huấn luyện chậm hơn nhiều do mạng sâu và rộng hơn,
độ phân giải ảnh cao hơn và các phép convolution tốn kém hơn.

```{.python .input  n=8}
%%tab pytorch, mxnet, jax
model = AlexNet(lr=0.01)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)
```

```{.python .input  n=9}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = AlexNet(lr=0.01)
    trainer.fit(model, data)
```

## Thảo luận

Cấu trúc của AlexNet có sự tương đồng đáng kể với LeNet, với một số cải tiến then chốt, cả về độ chính xác (dropout) lẫn tính dễ huấn luyện (ReLU). Điều cũng đáng chú ý không kém là mức độ tiến bộ đã đạt được về công cụ deep learning. Công việc từng mất vài tháng vào năm 2012 nay có thể được hoàn thành trong hàng chục dòng code bằng bất kỳ framework hiện đại nào.

Xem xét kiến trúc này, ta thấy AlexNet có một điểm yếu Achilles về hiệu quả: hai hidden layer cuối yêu cầu các ma trận có kích thước lần lượt là $6400 \times 4096$ và $4096 \times 4096$. Điều này tương ứng với 164 MB bộ nhớ và 81 MFLOPs tính toán, cả hai đều là chi phí đáng kể, đặc biệt trên các thiết bị nhỏ hơn như điện thoại di động. Đây là một trong những lý do AlexNet đã bị vượt qua bởi các kiến trúc hiệu quả hơn nhiều mà chúng ta sẽ đề cập trong các phần tiếp theo. Tuy vậy, đây là một bước chuyển quan trọng từ các mạng nông sang các mạng sâu được sử dụng ngày nay. Lưu ý rằng mặc dù số lượng tham số vượt xa lượng dữ liệu huấn luyện trong các thí nghiệm của chúng ta (hai layer cuối có hơn 40 triệu tham số, được huấn luyện trên bộ dữ liệu gồm 60 nghìn ảnh), hầu như không có overfitting: training loss và validation loss gần như giống hệt nhau trong suốt quá trình huấn luyện. Điều này là nhờ regularization được cải thiện, chẳng hạn như dropout, vốn có trong các thiết kế deep network hiện đại.

Mặc dù có vẻ như phần triển khai AlexNet chỉ nhiều hơn LeNet vài dòng, cộng đồng học thuật đã mất nhiều năm để đón nhận thay đổi về mặt khái niệm này và tận dụng các kết quả thực nghiệm xuất sắc của nó. Điều này cũng do thiếu các công cụ tính toán hiệu quả. Vào thời điểm đó, cả DistBelief :cite:`Dean.Corrado.Monga.ea.2012` lẫn Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014` đều chưa tồn tại, còn Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010` vẫn thiếu nhiều tính năng nổi bật. Chính sự xuất hiện của TensorFlow :cite:`Abadi.Barham.Chen.ea.2016` đã thay đổi tình hình một cách mạnh mẽ.

## Bài tập

1. Dựa trên phần thảo luận ở trên, hãy phân tích các đặc tính tính toán của AlexNet.
    1. Tính memory footprint cho các convolution và fully connected layer, tương ứng. Loại nào chiếm ưu thế?
    1. Tính chi phí tính toán cho các convolution và các fully connected layer.
    1. Bộ nhớ (băng thông đọc và ghi, độ trễ, kích thước) ảnh hưởng đến việc tính toán như thế nào? Tác động của nó có khác nhau giữa huấn luyện và suy luận không?
1. Bạn là một nhà thiết kế chip và cần đánh đổi giữa năng lực tính toán và băng thông bộ nhớ. Ví dụ, chip nhanh hơn cần nhiều điện năng hơn và có thể cần diện tích chip lớn hơn. Băng thông bộ nhớ lớn hơn cần nhiều chân cắm và logic điều khiển hơn, do đó cũng cần nhiều diện tích hơn. Bạn tối ưu như thế nào?
1. Tại sao các kỹ sư không còn báo cáo benchmark hiệu năng trên AlexNet?
1. Hãy thử tăng số epoch khi huấn luyện AlexNet. So với LeNet, kết quả khác nhau như thế nào? Tại sao?
1. AlexNet có thể quá phức tạp đối với bộ dữ liệu Fashion-MNIST, đặc biệt do độ phân giải thấp của các ảnh ban đầu.
    1. Hãy thử đơn giản hóa mô hình để huấn luyện nhanh hơn, đồng thời bảo đảm độ chính xác không giảm đáng kể.
    1. Thiết kế một mô hình tốt hơn hoạt động trực tiếp trên ảnh $28 \times 28$.
1. Thay đổi batch size và quan sát các thay đổi về throughput (ảnh/giây), độ chính xác và bộ nhớ GPU.
1. Áp dụng dropout và ReLU cho LeNet-5. Chúng có cải thiện không? Bạn có thể cải thiện hơn nữa bằng cách tiền xử lý để tận dụng các tính bất biến vốn có trong ảnh không?
1. Bạn có thể làm AlexNet overfit không? Bạn cần loại bỏ hoặc thay đổi đặc điểm nào để làm quá trình huấn luyện thất bại?

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/75)
:end_tab:

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/76)
:end_tab:

:begin_tab:`tensorflow`
[Thảo luận](https://discuss.d2l.ai/t/276)
:end_tab:

:begin_tab:`jax`
[Thảo luận](https://discuss.d2l.ai/t/18001)
:end_tab: