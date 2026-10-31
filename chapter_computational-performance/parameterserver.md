# Máy chủ tham số
:label:`sec_parameterserver`

Khi chuyển từ một GPU sang nhiều GPU, rồi sang nhiều máy chủ chứa nhiều GPU, có thể được phân bố trên nhiều rack và switch mạng, các thuật toán dùng cho việc huấn luyện phân tán và song song của chúng ta cần trở nên tinh vi hơn nhiều. Các chi tiết rất quan trọng vì những interconnect khác nhau có băng thông rất khác nhau (ví dụ: NVLink có thể cung cấp tới 100 GB/s qua 6 liên kết trong một số cấu hình phù hợp, PCIe 4.0 (16-lane) cung cấp 32 GB/s, trong khi ngay cả Ethernet 100GbE tốc độ cao cũng chỉ đạt 10 GB/s). Đồng thời, thật vô lý khi kỳ vọng một nhà mô hình hóa thống kê phải là chuyên gia về mạng và hệ thống.

Ý tưởng cốt lõi của parameter server được giới thiệu trong :citet:`Smola.Narayanamurthy.2010` trong bối cảnh các mô hình biến ẩn phân tán. Mô tả về semantics push và pull sau đó được trình bày trong :citet:`Ahmed.Aly.Gonzalez.ea.2012`, còn mô tả về hệ thống và một thư viện nguồn mở được trình bày trong :citet:`Li.Andersen.Park.ea.2014`. Trong phần tiếp theo, chúng ta sẽ giải thích động lực cần có các thành phần này để đạt hiệu quả cao.

## Huấn luyện song song dữ liệu

Hãy cùng ôn lại phương pháp huấn luyện song song dữ liệu trong huấn luyện phân tán. Trong phần này, chúng ta sẽ chỉ sử dụng phương pháp này và không dùng các phương pháp khác, vì trên thực tế phương pháp này đơn giản hơn đáng kể để triển khai. Hiện nay hầu như không có trường hợp sử dụng nào (ngoài deep learning trên graph) mà một chiến lược song song khác được ưu tiên, vì GPU ngày nay có rất nhiều bộ nhớ. :numref:`fig_parameterserver` mô tả biến thể của phương pháp song song dữ liệu mà chúng ta đã triển khai trong :numref:`sec_multi_gpu`. Điểm then chốt là việc tổng hợp gradient diễn ra trên một GPU duy nhất (GPU 0), trước khi các parameter đã cập nhật được phát lại tới tất cả GPU.

![Bên trái: huấn luyện trên một GPU. Bên phải: một biến thể của huấn luyện trên nhiều GPU: (1) chúng ta tính loss và gradient, (2) tất cả gradient được tổng hợp trên một GPU, (3) parameter được cập nhật và phân phối lại tới tất cả GPU.](../img/ps.svg)
:label:`fig_parameterserver`

Nhìn lại, quyết định tổng hợp trên GPU 0 có vẻ khá tùy tiện. Xét cho cùng, chúng ta hoàn toàn có thể tổng hợp trên CPU. Thậm chí, chúng ta có thể quyết định tổng hợp một số parameter trên GPU này và một số parameter khác trên GPU khác. Miễn là thuật toán optimization hỗ trợ điều đó, không có lý do thực sự nào khiến chúng ta không thể làm như vậy. Chẳng hạn, nếu có bốn vector parameter cùng các gradient tương ứng $\mathbf{g}_1, \ldots, \mathbf{g}_4$, chúng ta có thể tổng hợp gradient trên một GPU cho mỗi $\mathbf{g}_i$ ($i = 1, \ldots, 4$).


Lập luận này có vẻ tùy tiện và thiếu cơ sở. Xét cho cùng, về mặt toán học thì mọi thứ đều giống nhau. Tuy nhiên, chúng ta đang làm việc với phần cứng vật lý thực, trong đó các bus khác nhau có băng thông khác nhau như đã thảo luận trong :numref:`sec_hardware`.
Hãy xem xét một máy chủ 4-way GPU thực tế như được mô tả trong :numref:`fig_bw_hierarchy`. Nếu có kết nối đặc biệt tốt, máy chủ này có thể có card mạng 100 GbE. Các con số phổ biến hơn nằm trong khoảng 1--10 GbE, với băng thông hiệu dụng từ 100 MB/s đến 1 GB/s.
Vì CPU có quá ít lane PCIe để kết nối trực tiếp với tất cả GPU (ví dụ: CPU Intel dành cho người dùng phổ thông có 24 lane), chúng ta cần một [bộ ghép kênh](https://www.broadcom.com/products/pcie-switches-bridges/pcie-switches). Băng thông từ CPU trên liên kết 16x Gen3 là 16 GB/s. Đây cũng là tốc độ kết nối của *mỗi* GPU với switch. Điều này có nghĩa là việc giao tiếp giữa các thiết bị sẽ hiệu quả hơn.

![Một máy chủ 4-way GPU.](../img/bw-hierarchy.svg)
:label:`fig_bw_hierarchy`

Để minh họa, hãy giả sử rằng các gradient có kích thước 160 MB. Trong trường hợp này, mất 30 ms để gửi gradient từ 3 GPU còn lại tới GPU thứ tư (mỗi lần truyền mất 10 ms = 160 MB / 16 GB/s). Cộng thêm 30 ms để truyền các vector trọng số trở lại, ta có tổng cộng 60 ms.
Nếu gửi toàn bộ dữ liệu tới CPU, chúng ta chịu thêm 40 ms vì *mỗi* trong bốn GPU đều phải gửi dữ liệu tới CPU, dẫn đến tổng cộng 80 ms. Cuối cùng, giả sử chúng ta có thể chia gradient thành 4 phần, mỗi phần 40 MB. Khi đó, chúng ta có thể tổng hợp từng phần trên một GPU khác nhau *đồng thời*, vì PCIe switch hỗ trợ hoạt động toàn băng thông giữa tất cả các liên kết. Thay vì mất 30 ms, quá trình này chỉ mất 7,5 ms, dẫn đến tổng cộng 15 ms cho một thao tác synchronization. Tóm lại, tùy thuộc vào cách chúng ta synchronization parameter, cùng một thao tác có thể mất từ 15 ms đến 80 ms. :numref:`fig_ps_distributed` minh họa các chiến lược trao đổi parameter khác nhau.

![Các chiến lược synchronization parameter.](../img/ps-distributed.svg)
:label:`fig_ps_distributed`

Lưu ý rằng chúng ta còn có một công cụ khác để cải thiện hiệu năng: trong một deep network, việc tính tất cả gradient từ trên xuống dưới cần một khoảng thời gian. Chúng ta có thể bắt đầu synchronization gradient cho một số nhóm parameter ngay cả khi vẫn đang bận tính gradient cho các nhóm khác. Xem chẳng hạn :citet:`Sergeev.Del-Balso.2018` để biết chi tiết về cách thực hiện điều này trong [Horovod](https://github.com/horovod/horovod).

## Ring Synchronization

Khi synchronization trên phần cứng deep learning hiện đại, chúng ta thường gặp cấu trúc kết nối mạng được thiết kế khá đặc thù. Chẳng hạn, các instance AWS p3.16xlarge và NVIDIA DGX-2 có chung cấu trúc kết nối như trong :numref:`fig_nvlink`. Mỗi GPU kết nối với một CPU máy chủ qua liên kết PCIe, với tốc độ tối đa là 16 GB/s. Ngoài ra, mỗi GPU còn có 6 kết nối NVLink, mỗi kết nối có khả năng truyền hai chiều ở tốc độ 300 Gbit/s. Con số này tương đương khoảng 18 GB/s cho mỗi liên kết theo mỗi hướng. Tóm lại, băng thông NVLink tổng hợp cao hơn đáng kể so với băng thông PCIe. Câu hỏi đặt ra là làm thế nào để sử dụng nó hiệu quả nhất.

![Kết nối NVLink trên 8 máy chủ GPU V100 (hình ảnh do NVIDIA cung cấp).](../img/nvlink.svg)
:label:`fig_nvlink`

Hóa ra chiến lược synchronization tối ưu là phân rã mạng thành hai vòng và sử dụng chúng để synchronization dữ liệu trực tiếp :cite:`Wang.Li.Liberty.ea.2018`. :numref:`fig_nvlink_twoloop` minh họa rằng mạng có thể được phân rã thành một vòng (1-2-3-4-5-6-7-8-1) với băng thông NVLink gấp đôi và một vòng khác (1-4-6-3-5-8-2-7-1) với băng thông thông thường. Việc thiết kế một protocol synchronization hiệu quả trong trường hợp này không hề đơn giản.

![Phân rã mạng NVLink thành hai vòng.](../img/nvlink-twoloop.svg)
:label:`fig_nvlink_twoloop`


Hãy xem xét thí nghiệm tư duy sau: với một vòng gồm $n$ node tính toán (hoặc GPU), chúng ta có thể gửi gradient từ node thứ nhất tới node thứ hai. Tại đó, gradient được cộng vào gradient cục bộ rồi gửi tiếp tới node thứ ba, và cứ tiếp tục như vậy. Sau $n-1$ bước, gradient tổng hợp có thể được tìm thấy tại node được ghé thăm cuối cùng. Nghĩa là, thời gian tổng hợp gradient tăng tuyến tính theo số node. Tuy nhiên, nếu làm như vậy thì thuật toán khá kém hiệu quả. Xét cho cùng, tại mỗi thời điểm chỉ có một node đang giao tiếp. Điều gì xảy ra nếu chúng ta chia gradient thành $n$ chunk và bắt đầu synchronization chunk $i$ tại node $i`?
Vì mỗi chunk có kích thước $1/n$, tổng thời gian lúc này là $(n-1)/n \approx 1$. Nói cách khác, thời gian dùng để tổng hợp gradient *không tăng* khi chúng ta tăng kích thước của vòng. Đây là một kết quả khá đáng kinh ngạc. :numref:`fig_ringsync` minh họa chuỗi bước trên $n=4$ node.

![Ring synchronization trên 4 node. Mỗi node bắt đầu truyền các phần của gradient tới node lân cận bên trái cho đến khi gradient hoàn chỉnh có thể được tìm thấy tại node lân cận bên phải của nó.](../img/ringsync.svg)
:label:`fig_ringsync`

Nếu sử dụng cùng ví dụ synchronization 160 MB trên 8 GPU V100, chúng ta có khoảng $2 \cdot 160 \textrm{MB} / (3 \cdot 18 \textrm{GB/s}) \approx 6 \textrm{ms}$. Kết quả này tốt hơn việc sử dụng bus PCIe, dù lúc này chúng ta đang sử dụng 8 GPU. Lưu ý rằng trên thực tế, các con số này hơi kém hơn, vì các framework deep learning thường không thể gộp hoạt động giao tiếp thành các lần truyền burst lớn.

Lưu ý rằng có một quan niệm sai lầm phổ biến cho rằng ring synchronization về bản chất khác với các thuật toán synchronization khác. Điểm khác biệt duy nhất là đường dẫn synchronization phức tạp hơn đôi chút so với một tree đơn giản.

## Huấn luyện đa máy

Huấn luyện phân tán trên nhiều máy tạo thêm một thách thức: chúng ta cần giao tiếp với các server chỉ được kết nối qua một fabric có băng thông tương đối thấp, trong một số trường hợp có thể chậm hơn hơn một bậc độ lớn.
Synchronization giữa các thiết bị rất phức tạp. Xét cho cùng, các máy khác nhau chạy code huấn luyện sẽ có tốc độ hơi khác nhau. Vì vậy, chúng ta cần *synchronization* chúng nếu muốn sử dụng distributed optimization đồng bộ. :numref:`fig_ps_multimachine` minh họa cách diễn ra quá trình huấn luyện song song phân tán.

1. Một lô dữ liệu (khác) được đọc trên mỗi máy, chia thành nhiều GPU và truyền vào bộ nhớ GPU. Tại đó, các dự đoán và gradient được tính riêng trên từng lô GPU.
2. Các gradient từ tất cả GPU cục bộ được tổng hợp trên một GPU (hoặc các phần của chúng được tổng hợp trên nhiều GPU khác nhau).
3. Các gradient được gửi đến CPU.
4. CPU gửi các gradient đến một parameter server trung tâm, nơi tổng hợp tất cả gradient.
5. Các gradient đã tổng hợp sau đó được dùng để cập nhật các parameter, và các parameter đã cập nhật được broadcast trở lại từng CPU.
6. Thông tin được gửi đến một (hoặc nhiều) GPU.
7. Các parameter đã cập nhật được phân bổ trên tất cả GPU.

![Huấn luyện song song phân tán đa máy, đa GPU.](../img/ps-multimachine.svg)
:label:`fig_ps_multimachine`

Mỗi thao tác này có vẻ khá đơn giản. Và trên thực tế, chúng có thể được thực hiện hiệu quả *trong* một máy duy nhất. Tuy nhiên, khi xem xét nhiều máy, chúng ta có thể thấy rằng parameter server trung tâm trở thành nút thắt cổ chai. Xét cho cùng, băng thông trên mỗi server là có giới hạn, do đó với $m$ worker, thời gian cần để gửi tất cả gradient đến server là $\mathcal{O}(m)$. Chúng ta có thể vượt qua giới hạn này bằng cách tăng số lượng server lên $n$. Khi đó, mỗi server chỉ cần lưu trữ $\mathcal{O}(1/n)$ số parameter, do đó tổng thời gian cho việc cập nhật và tối ưu hóa trở thành $\mathcal{O}(m/n)$.
Đặt hai đại lượng này bằng nhau sẽ cho mức scaling không đổi bất kể chúng ta đang xử lý bao nhiêu worker. Trong thực tế, chúng ta sử dụng *cùng* các máy vừa làm worker vừa làm server. :numref:`fig_ps_multips` minh họa thiết kế này (xem thêm :cite:`Li.Andersen.Park.ea.2014` để biết chi tiết).
Đặc biệt, việc đảm bảo nhiều máy hoạt động mà không có độ trễ quá lớn là điều không hề đơn giản.

![Trên: một parameter server duy nhất là nút thắt cổ chai vì băng thông của nó có giới hạn. Dưới: nhiều parameter server lưu trữ các phần của parameter với băng thông tổng hợp.](../img/ps-multips.svg)
:label:`fig_ps_multips`

## Kho key--value

Việc triển khai trên thực tế các bước cần thiết để huấn luyện phân tán đa GPU là điều không hề đơn giản.
Vì vậy, việc sử dụng một abstraction phổ biến, cụ thể là *key--value store* với semantics cập nhật được định nghĩa lại, sẽ mang lại nhiều lợi ích.

Trên nhiều worker và nhiều GPU, phép tính cho gradient $i$ có thể được định nghĩa như sau:

$$\mathbf{g}_{i} = \sum_{k \in \textrm{workers}} \sum_{j \in \textrm{GPUs}} \mathbf{g}_{ijk},$$

trong đó $\mathbf{g}_{ijk}$ là một phần của gradient $i$ được chia trên GPU $j$ của worker $k$.
Điểm cốt yếu trong thao tác này là đây là một *phép reduction giao hoán*, tức là nó biến nhiều vector thành một vector và thứ tự áp dụng thao tác không ảnh hưởng đến kết quả. Điều này rất phù hợp với mục đích của chúng ta, vì chúng ta không có (hoặc không cần) quyền kiểm soát chi tiết về thời điểm nhận từng gradient. Ngoài ra, hãy lưu ý rằng thao tác này độc lập giữa các $i$ khác nhau.

Điều này cho phép chúng ta định nghĩa hai thao tác sau: *push*, dùng để tích lũy các gradient, và *pull*, dùng để truy xuất các gradient đã tổng hợp. Vì có nhiều tập gradient khác nhau (suy cho cùng, chúng ta có nhiều layer), chúng ta cần lập chỉ mục cho các gradient bằng một key $i$. Sự tương đồng này với các key--value store, chẳng hạn như store được giới thiệu trong Dynamo
:cite:`DeCandia.Hastorun.Jampani.ea.2007` không phải là ngẫu nhiên. Chúng cũng có nhiều đặc điểm tương tự, đặc biệt là khi phân phối các parameter trên nhiều server.

Các thao tác push và pull đối với key-value store được mô tả như sau:

* **push(key, value)** gửi một gradient cụ thể (value) từ một worker đến bộ lưu trữ chung. Tại đó, value được tổng hợp, chẳng hạn bằng cách cộng lại.
* **pull(key, value)** truy xuất một value đã tổng hợp từ bộ lưu trữ chung, chẳng hạn sau khi kết hợp các gradient từ tất cả worker.

Bằng cách ẩn toàn bộ độ phức tạp liên quan đến synchronization phía sau một thao tác push và pull đơn giản, chúng ta có thể tách biệt mối quan tâm của các nhà thống kê mô hình, những người muốn biểu diễn việc tối ưu hóa bằng các thuật ngữ đơn giản, với mối quan tâm của các kỹ sư hệ thống, những người phải xử lý sự phức tạp vốn có trong synchronization phân tán.

## Tóm tắt

* Synchronization cần có khả năng thích ứng cao với hạ tầng mạng cụ thể và khả năng kết nối bên trong server. Điều này có thể tạo ra khác biệt đáng kể đối với thời gian cần để synchronization.
* Ring-synchronization có thể là lựa chọn tối ưu cho server p3 và DGX-2. Với các server khác thì có thể không tối ưu bằng.
* Chiến lược synchronization phân cấp hoạt động hiệu quả khi bổ sung nhiều parameter server để tăng băng thông.

## Bài tập

1. Bạn có thể cải thiện ring synchronization hơn nữa không? Gợi ý: bạn có thể gửi message theo cả hai hướng.
1. Có thể cho phép giao tiếp bất đồng bộ (trong khi computation vẫn đang diễn ra) không? Điều đó ảnh hưởng thế nào đến performance?
1. Điều gì xảy ra nếu chúng ta mất một server trong một computation kéo dài? Làm thế nào để thiết kế một cơ chế *fault tolerance* nhằm tránh phải khởi động lại toàn bộ computation?


[Thảo luận](https://discuss.d2l.ai/t/366)