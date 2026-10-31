# Quy trình quyết định Markov (MDP)
:label:`sec_mdp`
Trong phần này, chúng ta sẽ thảo luận về cách xây dựng các bài toán học tăng cường bằng cách sử dụng các quy trình quyết định Markov (MDP) và mô tả chi tiết các thành phần khác nhau của MDP.

## Định nghĩa MDP

Quy trình quyết định Markov (MDP) :cite:`BellmanMDP` là một mô hình mô tả cách trạng thái của một hệ thống tiến triển khi các hành động khác nhau được áp dụng lên hệ thống. Một số đại lượng khác nhau kết hợp với nhau để tạo thành một MDP.

![Một tác vụ điều hướng gridworld đơn giản, trong đó robot không chỉ phải tìm đường đến vị trí mục tiêu (được biểu thị bằng ngôi nhà màu xanh lá cây) mà còn phải tránh các vị trí bẫy (được biểu thị bằng các dấu chéo màu đỏ).](../img/mdp.png)
:width:`250px`
:label:`fig_mdp`

* Gọi $\mathcal{S}$ là tập các trạng thái trong MDP. Xét ví dụ cụ thể trong :numref:`fig_mdp`, về một robot đang điều hướng trong gridworld. Trong trường hợp này, $\mathcal{S}$ tương ứng với tập các vị trí mà robot có thể ở tại bất kỳ timestep nào.
* Gọi $\mathcal{A}$ là tập các hành động mà robot có thể thực hiện tại mỗi trạng thái, chẳng hạn như “đi thẳng”, “rẽ phải”, “rẽ trái”, “ở nguyên vị trí”, v.v. Các hành động có thể thay đổi trạng thái hiện tại của robot thành một trạng thái khác trong tập $\mathcal{S}$.
* Có thể chúng ta không biết chính xác robot di chuyển như thế nào mà chỉ biết gần đúng. Trong học tăng cường, chúng ta mô hình hóa tình huống này như sau: nếu robot thực hiện hành động “đi thẳng”, có thể có một xác suất nhỏ là nó vẫn ở trạng thái hiện tại, một xác suất nhỏ khác là nó “rẽ trái”, v.v. Về mặt toán học, điều này tương đương với việc định nghĩa một “hàm chuyển trạng thái” $T: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0,1]$ sao cho $T(s, a, s') = P(s' \mid s, a)$, sử dụng xác suất có điều kiện đạt đến trạng thái $s'$ với điều kiện robot đang ở trạng thái $s$ và thực hiện hành động $a$. Hàm chuyển trạng thái là một phân phối xác suất, do đó ta có $\sum_{s' \in \mathcal{S}} T(s, a, s') = 1$ với mọi $s \in \mathcal{S}$ và $a \in \mathcal{A}$, tức là robot phải chuyển đến một trạng thái nào đó nếu nó thực hiện một hành động.
* Bây giờ, chúng ta xây dựng khái niệm về hành động nào hữu ích và hành động nào không hữu ích bằng cách sử dụng khái niệm “phần thưởng” $r: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$. Ta nói robot nhận được phần thưởng $r(s,a)$ nếu robot thực hiện hành động $a$ tại trạng thái $s$. Nếu phần thưởng $r(s, a)$ lớn, điều đó cho biết việc thực hiện hành động $a$ tại trạng thái $s$ hữu ích hơn cho việc đạt được mục tiêu của robot, tức là đi đến ngôi nhà màu xanh lá cây. Nếu phần thưởng $r(s, a)$ nhỏ, thì hành động $a$ kém hữu ích hơn để đạt mục tiêu này. Điều quan trọng cần lưu ý là phần thưởng được người dùng (người tạo ra thuật toán học tăng cường) thiết kế với mục tiêu cụ thể trong mind.

## Return và hệ số chiết khấu

Các thành phần khác nhau ở trên kết hợp lại tạo thành một quy trình quyết định Markov (MDP)
$$\textrm{MDP}: (\mathcal{S}, \mathcal{A}, T, r).$$

Bây giờ, hãy xét tình huống robot bắt đầu tại một trạng thái cụ thể $s_0 \in \mathcal{S}$ và tiếp tục thực hiện các hành động để tạo ra một trajectory
$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, s_2, a_2, r_2, \ldots).$$

Tại mỗi timestep $t$, robot ở trạng thái $s_t$ và thực hiện hành động $a_t$, từ đó nhận được phần thưởng $r_t = r(s_t, a_t)$. *Return* của một trajectory là tổng phần thưởng mà robot nhận được dọc theo trajectory đó:
$$R(\tau) = r_0 + r_1 + r_2 + \cdots.$$

Mục tiêu của học tăng cường là tìm một trajectory có *return* lớn nhất.

Hãy xét tình huống robot tiếp tục di chuyển trong gridworld mà không bao giờ đến vị trí mục tiêu. Trong trường hợp này, chuỗi trạng thái và hành động trong một trajectory có thể dài vô hạn, và *return* của bất kỳ trajectory vô hạn nào như vậy sẽ là vô hạn. Để việc xây dựng bài toán học tăng cường vẫn có ý nghĩa ngay cả với những trajectory như vậy, chúng ta đưa vào khái niệm hệ số chiết khấu $\gamma < 1$. Ta viết *return* chiết khấu như sau:
$$R(\tau) = r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots = \sum_{t=0}^\infty \gamma^t r_t.$$

Lưu ý rằng nếu $\gamma$ rất nhỏ, các phần thưởng mà robot nhận được trong tương lai xa, chẳng hạn tại $t = 1000$, sẽ bị chiết khấu mạnh bởi hệ số $\gamma^{1000}$. Điều này khuyến khích robot chọn các trajectory ngắn để đạt được mục tiêu, cụ thể là đi đến ngôi nhà màu xanh lá cây trong ví dụ gridwold (xem :numref:`fig_mdp`). Với các giá trị lớn của hệ số chiết khấu, chẳng hạn $\gamma = 0.99$, robot được khuyến khích *khám phá* rồi tìm trajectory tốt nhất để đi đến vị trí mục tiêu.

## Thảo luận về giả định Markov

Hãy xét một robot mới, trong đó trạng thái $s_t$ vẫn là vị trí như trên nhưng hành động $a_t$ là gia tốc mà robot áp dụng lên các bánh xe thay vì một lệnh trừu tượng như “đi thẳng”. Nếu robot này có vận tốc khác không tại trạng thái $s_t$, thì vị trí tiếp theo $s_{t+1}$ là một hàm của vị trí trước đó $s_t$, gia tốc $a_t$, cũng như vận tốc của robot tại thời điểm $t$, đại lượng này tỷ lệ với $s_t - s_{t-1}$. Điều này cho thấy ta nên có

$$s_{t+1} = \textrm{some function}(s_t, a_t, s_{t-1});$$

“một hàm nào đó” trong trường hợp này sẽ là định luật chuyển động của Newton. Điều này khá khác với hàm chuyển trạng thái của chúng ta, vốn chỉ phụ thuộc vào $s_t$ và $a_t$.

Các hệ thống Markov là tất cả những hệ thống mà trạng thái tiếp theo $s_{t+1}$ chỉ là một hàm của trạng thái hiện tại $s_t$ và hành động $a_t$ được thực hiện tại trạng thái hiện tại. Trong các hệ thống Markov, trạng thái tiếp theo không phụ thuộc vào những hành động đã thực hiện trong quá khứ hay những trạng thái mà robot từng ở trong quá khứ. Ví dụ, robot mới có gia tốc làm hành động như trên không có tính Markov vì vị trí tiếp theo $s_{t+1}$ phụ thuộc vào trạng thái trước đó $s_{t-1}$ thông qua vận tốc. Có vẻ như tính Markov của một hệ thống là một giả định hạn chế, nhưng không phải vậy. Các quy trình quyết định Markov vẫn có khả năng mô hình hóa một lớp rất lớn các hệ thống thực tế. Ví dụ, với robot mới, nếu ta chọn trạng thái $s_t$ là bộ đôi $(\textrm{location}, \textrm{velocity})$ thì hệ thống là Markov vì trạng thái tiếp theo $(\textrm{location}_{t+1}, \textrm{velocity}_{t+1})$ của nó chỉ phụ thuộc vào trạng thái hiện tại $(\textrm{location}_t, \textrm{velocity}_t)$ và hành động tại trạng thái hiện tại $a_t$.

## Tóm tắt
Bài toán học tăng cường thường được mô hình hóa bằng các quy trình quyết định Markov. Một quy trình quyết định Markov (MDP) được định nghĩa bởi bộ bốn thực thể $(\mathcal{S}, \mathcal{A}, T, r)$, trong đó $\mathcal{S}$ là không gian trạng thái, $\mathcal{A}$ là không gian hành động, $T$ là hàm chuyển trạng thái mã hóa các xác suất chuyển trạng thái của MDP, còn $r$ là phần thưởng tức thời nhận được khi thực hiện hành động tại một trạng thái cụ thể.


## Bài tập

1. Giả sử chúng ta muốn thiết kế một MDP để mô hình hóa bài toán [MountainCar](https://www.gymlibrary.dev/environments/classic_control/mountain_car/).
    1. Tập các trạng thái sẽ là gì?
    2. Tập các hành động sẽ là gì?
    3. Các hàm phần thưởng khả dĩ sẽ là gì?
2. Bạn sẽ thiết kế một MDP cho một trò chơi Atari như [Pong game](https://www.gymlibrary.dev/environments/atari/pong/) như thế nào?

:begin_tab:`pytorch`
[Thảo luận](https://discuss.d2l.ai/t/12084)
:end_tab: