# Thị giác máy tính
:label:`chap_cv`

Dù là chẩn đoán y khoa, phương tiện tự lái, giám sát bằng camera hay các bộ lọc thông minh, nhiều ứng dụng trong lĩnh vực thị giác máy tính đều gắn bó mật thiết với cuộc sống hiện tại và tương lai của chúng ta.  
Trong những năm gần đây, deep learning đã trở thành động lực mang tính chuyển đổi, thúc đẩy hiệu năng của các hệ thống thị giác máy tính.  
Có thể nói rằng những ứng dụng thị giác máy tính tiên tiến nhất hầu như không thể tách rời khỏi deep learning.  
Vì vậy, chương này sẽ tập trung vào lĩnh vực thị giác máy tính, đồng thời tìm hiểu các phương pháp và ứng dụng gần đây có ảnh hưởng trong giới học thuật và công nghiệp.


Trong :numref:`chap_cnn` và :numref:`chap_modern_cnn`, chúng ta đã nghiên cứu nhiều mạng neural tích chập thường được sử dụng trong thị giác máy tính và áp dụng chúng vào các tác vụ phân loại hình ảnh đơn giản.  
Ở phần đầu chương này, chúng ta sẽ trình bày hai phương pháp có thể cải thiện khả năng khái quát hóa của mô hình, đó là *image augmentation* và *fine-tuning*, đồng thời áp dụng chúng vào việc phân loại hình ảnh.  
Vì các mạng neural sâu có thể biểu diễn hình ảnh ở nhiều cấp độ, những biểu diễn theo từng lớp như vậy đã được sử dụng thành công trong nhiều tác vụ thị giác máy tính khác nhau, chẳng hạn như *object detection*, *semantic segmentation* và *style transfer*.  
Tiếp nối ý tưởng cốt lõi về việc tận dụng các biểu diễn theo từng lớp trong thị giác máy tính, trước tiên chúng ta sẽ bắt đầu với các thành phần và kỹ thuật chính của *object detection*. Tiếp theo, chúng ta sẽ trình bày cách sử dụng *fully convolutional networks* để thực hiện *semantic segmentation* hình ảnh. Sau đó, chúng ta sẽ giải thích cách sử dụng các kỹ thuật *style transfer* để tạo ra những hình ảnh giống như bìa của cuốn sách này.  
Cuối cùng, chúng ta sẽ khép lại chương này bằng cách áp dụng nội dung của chương này và một số chương trước vào hai bộ dữ liệu benchmark phổ biến trong lĩnh vực thị giác máy tính.

```toc
:maxdepth: 2

image-augmentation
fine-tuning
bounding-box
anchor
multiscale-object-detection
object-detection-dataset
ssd
rcnn
semantic-segmentation-and-dataset
transposed-conv
fcn
neural-style
kaggle-cifar10
kaggle-dog
```
