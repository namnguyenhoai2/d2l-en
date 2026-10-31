# Word Embedding với Global Vectors (GloVe)
:label:`sec_glove`


Các lần đồng xuất hiện của từ
trong các cửa sổ ngữ cảnh
có thể chứa thông tin ngữ nghĩa phong phú.
Ví dụ,
trong một corpus lớn,
từ "solid" có
khả năng đồng xuất hiện
với "ice" cao hơn "steam",
nhưng từ "gas"
có lẽ đồng xuất hiện với "steam"
thường xuyên hơn "ice".
Ngoài ra,
các thống kê corpus toàn cục
về những lần đồng xuất hiện như vậy
có thể được tính toán trước:
điều này có thể giúp quá trình huấn luyện hiệu quả hơn.
Để tận dụng thông tin thống kê
trong toàn bộ corpus
cho word embedding,
trước tiên hãy xem lại
mô hình skip-gram trong :numref:`subsec_skip-gram`,
nhưng diễn giải mô hình
bằng các thống kê corpus toàn cục
chẳng hạn như số lần đồng xuất hiện.

## Skip-Gram với Thống kê Corpus Toàn cục
:label:`subsec_skipgram-global`

Ký hiệu $q_{ij}$
là xác suất có điều kiện
$P(w_j\mid w_i)$
của từ $w_j$ khi biết từ $w_i$
trong mô hình skip-gram,
ta có

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \exp(\mathbf{u}_k^\top \mathbf{v}_i)},$$

trong đó,
với mọi chỉ số $i$,
các vector $\mathbf{v}_i$ và $\mathbf{u}_i$
lần lượt biểu diễn từ $w_i$
với vai trò từ trung tâm và từ ngữ cảnh, còn $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$
là tập chỉ số của vocabulary.

Xét từ $w_i$
có thể xuất hiện nhiều lần
trong corpus.
Trong toàn bộ corpus,
tất cả các từ ngữ cảnh
ở mọi vị trí mà $w_i$ được chọn làm từ trung tâm
tạo thành một *multiset* (đa tập) $\mathcal{C}_i$
của các chỉ số từ,
cho phép *nhiều thể hiện của cùng một phần tử*.
Với mỗi phần tử,
số lượng thể hiện của phần tử đó được gọi là *multiplicity* (bội số).
Để minh họa bằng một ví dụ,
giả sử từ $w_i$ xuất hiện hai lần trong corpus
và các chỉ số của những từ ngữ cảnh
có $w_i$ làm từ trung tâm
trong hai cửa sổ ngữ cảnh
là
$k, j, m, k$ và $k, l, k, j$.
Do đó, multiset $\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$, trong đó
multiplicity của các phần tử $j, k, l, m$
lần lượt là 2, 4, 1, 1.

Bây giờ, hãy ký hiệu multiplicity của phần tử $j$ trong
multiset $\mathcal{C}_i$ là $x_{ij}$.
Đây là số lần đồng xuất hiện toàn cục
của từ $w_j$ (với vai trò từ ngữ cảnh)
và từ $w_i$ (với vai trò từ trung tâm)
trong cùng một cửa sổ ngữ cảnh
trên toàn bộ corpus.
Sử dụng các thống kê corpus toàn cục như vậy,
hàm mất mát của mô hình skip-gram
tương đương với

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-x_ij`

Ta tiếp tục ký hiệu
$x_i$
là số lượng tất cả các từ ngữ cảnh
trong những cửa sổ ngữ cảnh
mà $w_i$ xuất hiện với vai trò từ trung tâm,
tương đương với $|\mathcal{C}_i|$.
Gọi $p_{ij}$
là xác suất có điều kiện
$x_{ij}/x_i$ để sinh ra
từ ngữ cảnh $w_j$ khi biết từ trung tâm $w_i$,
:eqref:`eq_skipgram-x_ij`
có thể được viết lại thành

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-p_ij`

Trong :eqref:`eq_skipgram-p_ij`, $-\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$ tính
cross-entropy
giữa
phân phối có điều kiện $p_{ij}$
của các thống kê corpus toàn cục
và
phân phối
có điều kiện $q_{ij}$
của các dự đoán từ mô hình.
Như đã giải thích ở trên,
hàm mất mát này
cũng được tính trọng số bởi $x_i$.
Việc cực tiểu hóa hàm mất mát trong
:eqref:`eq_skipgram-p_ij`
sẽ cho phép
phân phối có điều kiện được dự đoán
tiến gần đến
phân phối có điều kiện
từ các thống kê corpus toàn cục.


Mặc dù thường được sử dụng
để đo khoảng cách
giữa các phân phối xác suất,
hàm mất mát cross-entropy có thể không phải là lựa chọn phù hợp ở đây.
Một mặt, như đã đề cập trong :numref:`sec_approx_train`,
chi phí để chuẩn hóa đúng $q_{ij}$
dẫn đến việc tính tổng trên toàn bộ vocabulary,
có thể tốn kém về mặt tính toán.
Mặt khác,
một số lượng lớn các sự kiện hiếm
từ một corpus lớn
thường được mô hình hóa bằng hàm mất mát cross-entropy
và bị gán trọng số quá lớn.

## Mô hình GloVe

Xét những điều này,
mô hình *GloVe* thực hiện ba thay đổi
đối với mô hình skip-gram dựa trên squared loss :cite:`Pennington.Socher.Manning.2014`:

1. Sử dụng các biến $p'_{ij}=x_{ij}$ và $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$
không phải là các phân phối xác suất
và lấy logarithm của cả hai, để hạng tử squared loss là $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$.
2. Thêm hai tham số vô hướng của mô hình cho mỗi từ $w_i$: bias của từ trung tâm $b_i$ và bias của từ ngữ cảnh $c_i$.
3. Thay trọng số của mỗi hạng tử mất mát bằng hàm trọng số $h(x_{ij})$, trong đó $h(x)$ tăng trên khoảng $[0, 1]$.

Kết hợp tất cả lại, việc huấn luyện GloVe là cực tiểu hóa hàm mất mát sau:

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$
:eqlabel:`eq_glove-loss`

Đối với hàm trọng số, một lựa chọn được đề xuất là:
$h(x) = (x/c) ^\alpha$ (ví dụ $\alpha = 0.75$) nếu $x < c$ (ví dụ, $c = 100$); nếu không thì $h(x) = 1$.
Trong trường hợp này,
vì $h(0)=0$,
hạng tử squared loss với mọi $x_{ij}=0$ có thể được bỏ qua
để tăng hiệu quả tính toán.
Ví dụ,
khi sử dụng stochastic gradient descent theo minibatch để huấn luyện,
ở mỗi iteration,
ta lấy ngẫu nhiên một minibatch gồm các $x_{ij}$ *khác không*
để tính gradient
và cập nhật các tham số của mô hình.
Lưu ý rằng các $x_{ij}$ khác không này là
các thống kê corpus toàn cục được tính trước;
do đó, mô hình được gọi là GloVe,
viết tắt của *Global Vectors*.

Cần nhấn mạnh rằng
nếu từ $w_i$ xuất hiện trong cửa sổ ngữ cảnh của
từ $w_j$, thì *ngược lại cũng đúng*.
Do đó, $x_{ij}=x_{ji}$.
Không giống word2vec
vốn khớp với xác suất có điều kiện bất đối xứng
$p_{ij}$,
GloVe khớp với $\log \, x_{ij}$ đối xứng.
Do đó, vector từ trung tâm và
vector từ ngữ cảnh của bất kỳ từ nào về mặt toán học là tương đương trong mô hình GloVe.
Tuy nhiên trên thực tế, do các giá trị khởi tạo khác nhau,
cùng một từ vẫn có thể nhận các giá trị khác nhau
trong hai vector này sau khi huấn luyện:
GloVe cộng chúng lại để tạo thành vector đầu ra.



## Diễn giải GloVe từ Tỷ lệ Xác suất Đồng xuất hiện


Ta cũng có thể diễn giải mô hình GloVe từ một góc nhìn khác.
Sử dụng cùng ký hiệu trong
:numref:`subsec_skipgram-global`,
gọi $p_{ij} \stackrel{\textrm{def}}{=} P(w_j \mid w_i)$ là xác suất có điều kiện sinh ra từ ngữ cảnh $w_j$ khi biết $w_i$ là từ trung tâm trong corpus.
:numref:`tab_glove`
liệt kê một số xác suất đồng xuất hiện
với các từ "ice" và "steam"
cùng các tỷ lệ của chúng dựa trên thống kê từ một corpus lớn.


:Xác suất đồng xuất hiện giữa các từ và tỷ lệ của chúng từ một corpus lớn (phỏng theo Bảng 1 trong :citet:`Pennington.Socher.Manning.2014`)
:label:`tab_glove`

|$w_k$=|solid|gas|water|fashion|
|:--|:-|:-|:-|:-|
|$p_1=P(w_k\mid \textrm{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k\mid \textrm{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|



Ta có thể quan sát những điều sau từ :numref:`tab_glove`:

* Với một từ $w_k$ có liên quan đến "ice" nhưng không liên quan đến "steam", chẳng hạn như $w_k=\textrm{solid}$, ta kỳ vọng tỷ lệ xác suất đồng xuất hiện lớn hơn, chẳng hạn 8.9.
* Với một từ $w_k$ có liên quan đến "steam" nhưng không liên quan đến "ice", chẳng hạn như $w_k=\textrm{gas}$, ta kỳ vọng tỷ lệ xác suất đồng xuất hiện nhỏ hơn, chẳng hạn 0.085.
* Với một từ $w_k$ có liên quan đến cả "ice" và "steam", chẳng hạn như $w_k=\textrm{water}$, ta kỳ vọng tỷ lệ xác suất đồng xuất hiện gần bằng 1, chẳng hạn 1.36.
* Với một từ $w_k$ không liên quan đến cả "ice" lẫn "steam", chẳng hạn như $w_k=\textrm{fashion}$, ta kỳ vọng tỷ lệ xác suất đồng xuất hiện gần bằng 1, chẳng hạn 0.96.




Có thể thấy rằng tỷ lệ
của các xác suất đồng xuất hiện
có thể trực quan thể hiện
mối quan hệ giữa các từ.
Do đó, ta có thể thiết kế một hàm
của ba vector từ
để khớp với tỷ lệ này.
Đối với tỷ lệ xác suất đồng xuất hiện
${p_{ij}}/{p_{ik}}$
với $w_i$ là từ trung tâm
còn $w_j$ và $w_k$ là các từ ngữ cảnh,
ta muốn khớp tỷ lệ này
bằng một hàm $f$ nào đó:

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$
:eqlabel:`eq_glove-f`

Trong số nhiều thiết kế khả dĩ cho $f$,
ta chỉ chọn một lựa chọn hợp lý như sau.
Vì tỷ lệ xác suất đồng xuất hiện
là một vô hướng,
ta yêu cầu
$f$ là một hàm vô hướng, chẳng hạn như
$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$.
Hoán đổi các chỉ số từ
$j$ và $k$ trong :eqref:`eq_glove-f`,
ta phải có
$f(x)f(-x)=1$,
vì vậy một khả năng là $f(x)=\exp(x)$,
tức là,

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

Bây giờ, hãy chọn
$\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$,
trong đó $\alpha$ là một hằng số.
Vì $p_{ij}=x_{ij}/x_i$, sau khi lấy logarithm hai vế, ta được $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$.
Ta có thể sử dụng thêm các hạng tử bias để khớp với $- \log\, \alpha + \log\, x_i$, chẳng hạn như bias của từ trung tâm $b_i$ và bias của từ ngữ cảnh $c_j$:

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.$$
:eqlabel:`eq_glove-square`

Đo sai số bình phương của
:eqref:`eq_glove-square` với các trọng số,
ta thu được hàm mất mát GloVe trong
:eqref:`eq_glove-loss`.



## Tóm tắt

* Mô hình skip-gram có thể được diễn giải bằng cách sử dụng các thống kê toàn cục của corpus, chẳng hạn như số lần đồng xuất hiện giữa các từ.
* Hàm mất mát cross-entropy có thể không phải là lựa chọn phù hợp để đo sự khác biệt giữa hai phân phối xác suất, đặc biệt là đối với một corpus lớn. GloVe sử dụng sai số bình phương để khớp với các thống kê toàn cục của corpus đã được tính trước.
* Vector của từ trung tâm và vector của từ ngữ cảnh tương đương về mặt toán học đối với mọi từ trong GloVe.
* GloVe có thể được diễn giải dựa trên tỷ lệ giữa các xác suất đồng xuất hiện của các từ.


## Bài tập

1. Nếu các từ $w_i$ và $w_j$ đồng xuất hiện trong cùng một cửa sổ ngữ cảnh, làm thế nào chúng ta có thể sử dụng khoảng cách giữa chúng trong chuỗi văn bản để thiết kế lại phương pháp tính xác suất có điều kiện $p_{ij}$? Gợi ý: xem Mục 4.2 của bài báo GloVe :cite:`Pennington.Socher.Manning.2014`.
1. Đối với bất kỳ từ nào, bias của từ trung tâm và bias của từ ngữ cảnh có tương đương về mặt toán học trong GloVe không? Tại sao?


[Discussions](https://discuss.d2l.ai/t/385)