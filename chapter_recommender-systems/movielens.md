#  Bộ dữ liệu MovieLens

Có một số bộ dữ liệu được cung cấp cho nghiên cứu về recommendation. Trong số đó, bộ dữ liệu [MovieLens](https://movielens.org/) có lẽ là một trong những bộ phổ biến hơn. MovieLens là một hệ thống recommendation phim trên nền web, phi thương mại. Hệ thống được tạo ra vào năm 1997 và do GroupLens, một phòng thí nghiệm nghiên cứu tại University of Minnesota, vận hành nhằm thu thập dữ liệu đánh giá phim cho mục đích nghiên cứu. Dữ liệu MovieLens đóng vai trò quan trọng trong một số nghiên cứu, bao gồm recommendation được cá nhân hóa và tâm lý học xã hội.


## Lấy dữ liệu


Bộ dữ liệu MovieLens được lưu trữ trên website [GroupLens](https://grouplens.org/datasets/movielens/). Có một số phiên bản được cung cấp. Chúng ta sẽ sử dụng bộ dữ liệu MovieLens 100K :cite:`Herlocker.Konstan.Borchers.ea.1999`. Bộ dữ liệu này bao gồm $100,000$ lượt đánh giá, từ 1 đến 5 sao, của 943 người dùng dành cho 1682 bộ phim. Dữ liệu đã được làm sạch để mỗi người dùng đánh giá ít nhất 20 bộ phim. Một số thông tin nhân khẩu học đơn giản như tuổi, giới tính và thể loại của người dùng và item cũng được cung cấp. Chúng ta có thể tải xuống [ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip) và giải nén file `u.data`, trong đó chứa toàn bộ $100,000$ lượt đánh giá ở định dạng csv. Có nhiều file khác trong thư mục; mô tả chi tiết cho từng file có thể được tìm thấy trong file [README](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt) của bộ dữ liệu.

Trước tiên, hãy import các package cần thiết để chạy các thí nghiệm trong phần này.

```{.python .input  n=1}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
import pandas as pd
```

Tiếp theo, chúng ta tải xuống bộ dữ liệu MovieLens 100k và nạp các interaction dưới dạng `DataFrame`.

```{.python .input  n=2}
#@tab mxnet
#@save
d2l.DATA_HUB['ml-100k'] = (
    'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

#@save
def read_data_ml100k():
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t',
                       names=names, engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items
```

## Thống kê của bộ dữ liệu

Hãy nạp dữ liệu và kiểm tra thủ công năm bản ghi đầu tiên. Đây là một cách hiệu quả để tìm hiểu cấu trúc dữ liệu và xác minh rằng dữ liệu đã được nạp đúng cách.

```{.python .input  n=3}
#@tab mxnet
data, num_users, num_items = read_data_ml100k()
sparsity = 1 - len(data) / (num_users * num_items)
print(f'number of users: {num_users}, number of items: {num_items}')
print(f'matrix sparsity: {sparsity:f}')
print(data.head(5))
```

Có thể thấy mỗi dòng gồm bốn cột, bao gồm "user id" từ 1-943, "item id" từ 1-1682, "rating" từ 1-5 và "timestamp". Chúng ta có thể xây dựng một ma trận interaction có kích thước $n \times m$, trong đó $n$ và $m$ lần lượt là số người dùng và số item. Bộ dữ liệu này chỉ ghi lại các rating hiện có, vì vậy chúng ta cũng có thể gọi nó là ma trận rating; hai thuật ngữ ma trận interaction và ma trận rating sẽ được sử dụng thay thế cho nhau khi các giá trị của ma trận này biểu thị rating chính xác. Phần lớn các giá trị trong ma trận rating là không xác định vì người dùng chưa đánh giá phần lớn các bộ phim. Chúng ta cũng hiển thị độ thưa của bộ dữ liệu này. Độ thưa được định nghĩa là `1 - number of nonzero entries / ( number of users * number of items)`. Rõ ràng, ma trận interaction cực kỳ thưa (tức là sparsity = 93.695%). Các bộ dữ liệu trong thế giới thực có thể gặp mức độ thưa lớn hơn, và đây từ lâu đã là một thách thức trong việc xây dựng các hệ thống recommendation. Một giải pháp khả thi là sử dụng thêm side information như các feature của user/item để giảm thiểu độ thưa.

Sau đó, chúng ta vẽ phân phối số lượng của các rating khác nhau. Đúng như dự đoán, phân phối này có vẻ là phân phối chuẩn, trong đó phần lớn rating tập trung ở mức 3-4.

```{.python .input  n=4}
#@tab mxnet
d2l.plt.hist(data['rating'], bins=5, ec='black')
d2l.plt.xlabel('Rating')
d2l.plt.ylabel('Count')
d2l.plt.title('Distribution of Ratings in MovieLens 100K')
d2l.plt.show()
```

## Chia bộ dữ liệu

Chúng ta chia bộ dữ liệu thành tập training và tập test. Hàm sau đây cung cấp hai chế độ chia, bao gồm `random` và `seq-aware`. Ở chế độ `random`, hàm chia ngẫu nhiên 100k interaction mà không xét đến timestamp, mặc định sử dụng 90% dữ liệu làm mẫu training và 10% còn lại làm mẫu test. Ở chế độ `seq-aware`, chúng ta để lại item mà người dùng đánh giá gần đây nhất làm test, còn các interaction trước đây của người dùng được dùng làm tập training. Các interaction trước đây của người dùng được sắp xếp từ cũ nhất đến mới nhất dựa trên timestamp. Chế độ này sẽ được sử dụng trong phần recommendation có nhận biết thứ tự.

```{.python .input  n=5}
#@tab mxnet
#@save
def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """Split the dataset in random mode or seq-aware mode."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data
```

Lưu ý rằng trong thực tế, việc sử dụng một tập validation bên cạnh tập test duy nhất là một thông lệ tốt. Tuy nhiên, để ngắn gọn, chúng ta sẽ bỏ qua tập này. Trong trường hợp đó, tập test của chúng ta có thể được xem là tập validation được giữ lại.

## Nạp dữ liệu

Sau khi chia bộ dữ liệu, chúng ta sẽ chuyển tập training và tập test thành các list và dictionary/matrix để thuận tiện sử dụng. Hàm sau đây đọc dataframe theo từng dòng và đánh số index của user/item bắt đầu từ zero. Sau đó, hàm trả về các list user, item, rating và một dictionary/matrix ghi lại các interaction. Chúng ta có thể chỉ định loại feedback là `explicit` hoặc `implicit`.

```{.python .input  n=6}
#@tab mxnet
#@save
def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter
```

Sau đó, chúng ta kết hợp các bước trên; phần này sẽ được sử dụng trong section tiếp theo. Kết quả được đóng gói bằng `Dataset` và `DataLoader`. Lưu ý rằng `last_batch` của `DataLoader` cho dữ liệu training được đặt ở chế độ `rollover` (Các mẫu còn lại sẽ được chuyển sang epoch tiếp theo.) và thứ tự được shuffle.

```{.python .input  n=7}
#@tab mxnet
#@save
def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    train_set = gluon.data.ArrayDataset(
        np.array(train_u), np.array(train_i), np.array(train_r))
    test_set = gluon.data.ArrayDataset(
        np.array(test_u), np.array(test_i), np.array(test_r))
    train_iter = gluon.data.DataLoader(
        train_set, shuffle=True, last_batch='rollover',
        batch_size=batch_size)
    test_iter = gluon.data.DataLoader(
        test_set, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter
```

## Tóm tắt

* Các bộ dữ liệu MovieLens được sử dụng rộng rãi cho nghiên cứu về recommendation. Chúng được công khai và miễn phí sử dụng.
* Chúng ta định nghĩa các hàm để tải xuống và tiền xử lý bộ dữ liệu MovieLens 100k nhằm sử dụng trong các section tiếp theo.


## Bài tập

* Bạn có thể tìm thấy những bộ dữ liệu recommendation tương tự nào khác?
* Truy cập trang [https://movielens.org/](https://movielens.org/) để biết thêm thông tin về MovieLens.

:begin_tab:`mxnet`
[Thảo luận](https://discuss.d2l.ai/t/399)
:end_tab: