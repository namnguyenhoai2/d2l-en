# Hệ thống gợi ý giàu đặc trưng

Dữ liệu tương tác là chỉ báo cơ bản nhất về sở thích và mối quan tâm của người dùng. Nó đóng vai trò quan trọng trong các mô hình được giới thiệu trước đó. Tuy nhiên, dữ liệu tương tác thường cực kỳ thưa và đôi khi có thể bị nhiễu. Để giải quyết vấn đề này, chúng ta có thể tích hợp thông tin bổ sung, chẳng hạn như các đặc trưng của sản phẩm, hồ sơ người dùng, và thậm chí cả ngữ cảnh diễn ra tương tác, vào mô hình gợi ý. Việc sử dụng các đặc trưng này rất hữu ích cho việc đưa ra gợi ý, vì chúng có thể là yếu tố dự đoán hiệu quả về mối quan tâm của người dùng, đặc biệt khi dữ liệu tương tác còn thiếu. Do đó, điều thiết yếu là các mô hình gợi ý cũng phải có khả năng xử lý những đặc trưng này và cung cấp cho mô hình khả năng nhận biết nội dung/ngữ cảnh. Để minh họa cho loại mô hình gợi ý này, chúng ta giới thiệu một tác vụ khác về tỷ lệ nhấp (click-through rate, CTR) cho việc gợi ý quảng cáo trực tuyến :cite:`McMahan.Holt.Sculley.ea.2013` và trình bày một bộ dữ liệu quảng cáo ẩn danh. Các dịch vụ quảng cáo nhắm mục tiêu đã thu hút sự chú ý rộng rãi và thường được xem như các hệ thống gợi ý. Việc gợi ý những quảng cáo phù hợp với sở thích và mối quan tâm cá nhân của người dùng rất quan trọng để cải thiện tỷ lệ nhấp.


Các nhà tiếp thị kỹ thuật số sử dụng quảng cáo trực tuyến để hiển thị quảng cáo cho khách hàng. Tỷ lệ nhấp là một chỉ số đo lường số lượt nhấp mà nhà quảng cáo nhận được trên quảng cáo của họ trên mỗi lượt hiển thị, và được biểu thị dưới dạng phần trăm, tính theo công thức:

$$ \textrm{CTR} = \frac{\#\textrm{Clicks}} {\#\textrm{Impressions}} \times 100 \% .$$

Tỷ lệ nhấp là một tín hiệu quan trọng cho biết mức độ hiệu quả của các thuật toán dự đoán. Dự đoán tỷ lệ nhấp là tác vụ dự đoán khả năng một nội dung nào đó trên website được nhấp vào. Các mô hình dự đoán CTR không chỉ có thể được sử dụng trong các hệ thống quảng cáo nhắm mục tiêu mà còn trong các hệ thống gợi ý sản phẩm nói chung (ví dụ: phim, tin tức, sản phẩm), chiến dịch email và thậm chí cả công cụ tìm kiếm. Nó cũng liên quan chặt chẽ đến mức độ hài lòng của người dùng và tỷ lệ chuyển đổi, đồng thời có thể hữu ích trong việc thiết lập mục tiêu chiến dịch vì giúp nhà quảng cáo đưa ra những kỳ vọng thực tế.

```{.python .input}
#@tab mxnet
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## Một bộ dữ liệu quảng cáo trực tuyến

Cùng với những tiến bộ đáng kể của Internet và công nghệ di động, quảng cáo trực tuyến đã trở thành một nguồn thu nhập quan trọng và tạo ra phần lớn doanh thu trong ngành Internet. Việc hiển thị các quảng cáo có liên quan hoặc thu hút sự quan tâm của người dùng là rất quan trọng để những khách truy cập ngẫu nhiên có thể trở thành khách hàng trả phí. Bộ dữ liệu mà chúng ta giới thiệu là một bộ dữ liệu quảng cáo trực tuyến. Bộ dữ liệu này gồm 34 trường, trong đó cột đầu tiên biểu diễn biến mục tiêu cho biết một quảng cáo có được nhấp vào hay không (1 là có, 0 là không). Tất cả các cột còn lại là các đặc trưng phân loại. Các cột có thể biểu diễn id quảng cáo, id website hoặc ứng dụng, id thiết bị, thời gian, hồ sơ người dùng, v.v. Ý nghĩa thực tế của các đặc trưng không được công bố do yêu cầu ẩn danh hóa và những lo ngại về quyền riêng tư.

Đoạn mã sau tải bộ dữ liệu từ máy chủ của chúng ta và lưu vào thư mục dữ liệu cục bộ.

```{.python .input  n=15}
#@tab mxnet
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

Có một tập huấn luyện và một tập kiểm tra, lần lượt gồm 15000 và 3000 mẫu/dòng.

## Dataset Wrapper

Để thuận tiện cho việc tải dữ liệu, chúng ta triển khai một `CTRDataset` để tải bộ dữ liệu quảng cáo từ tệp CSV và có thể được sử dụng bởi `DataLoader`.

```{.python .input  n=13}
#@tab mxnet
#@save
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy()
                                 [:-1]))
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']
```

Ví dụ sau tải dữ liệu huấn luyện và in bản ghi đầu tiên.

```{.python .input  n=16}
#@tab mxnet
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

Có thể thấy, cả 34 trường đều là các đặc trưng phân loại. Mỗi giá trị biểu diễn chỉ mục one-hot của mục tương ứng. Nhãn $0$ có nghĩa là quảng cáo không được nhấp vào. `CTRDataset` này cũng có thể được sử dụng để tải các bộ dữ liệu khác như bộ dữ liệu Criteo display advertising challenge [dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) và bộ dữ liệu Avazu click-through rate prediction [dataset](https://www.kaggle.com/c/avazu-ctr-prediction).

## Tóm tắt
* Tỷ lệ nhấp là một chỉ số quan trọng được sử dụng để đo lường mức độ hiệu quả của các hệ thống quảng cáo và hệ thống gợi ý.
* Dự đoán tỷ lệ nhấp thường được chuyển thành bài toán phân loại nhị phân. Mục tiêu là dự đoán liệu một quảng cáo/sản phẩm có được nhấp vào hay không dựa trên các đặc trưng đã cho.

## Bài tập

* Bạn có thể tải bộ dữ liệu Criteo và Avazu bằng `CTRDataset` được cung cấp không? Cần lưu ý rằng bộ dữ liệu Criteo gồm các đặc trưng có giá trị thực, vì vậy bạn có thể phải sửa đổi một chút đoạn mã.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/405)
:end_tab: