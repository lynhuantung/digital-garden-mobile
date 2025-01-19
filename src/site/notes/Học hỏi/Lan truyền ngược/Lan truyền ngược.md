---
{"dg-publish":true,"dg-home":false,"permalink":"/hoc-hoi/lan-truyen-nguoc/lan-truyen-nguoc/","dgPassFrontmatter":true,"noteIcon":"","updated":"2025-01-14T22:28:15.451+07:00"}
---

Lan truyền ngược (Backpropagation) 

Hãy tưởng tượng bạn đang học ném bóng rổ vào rổ. Ban đầu, bạn ném trượt, nhưng sau mỗi lần ném, bạn điều chỉnh lại lực ném, góc ném để lần sau chính xác hơn.

Trong mạng thần kinh nhân tạo:

Lan truyền tiến (forward propagation) giống như bạn ném quả bóng. Dữ liệu được "ném" qua các lớp của mạng, từ đầu vào (input) đến đầu ra (output).

Lan truyền ngược (backpropagation) là quá trình bạn "nghĩ lại" sau mỗi lần ném sai. Sau khi ném trượt (kết quả dự đoán sai), mạng thần kinh sẽ điều chỉnh "cách ném" (hay trọng số của từng kết nối) để lần sau "ném" chính xác hơn.


Cụ thể hơn:

Lan truyền tiến: Bạn đưa dữ liệu (ví dụ hình ảnh con mèo) vào mạng thần kinh. Mạng sẽ đi qua từng lớp ẩn, tính toán và đưa ra dự đoán (con chó chẳng hạn).

Sai số: Sau khi biết dự đoán sai (nó phải là con mèo chứ không phải con chó), mạng thần kinh sẽ tính toán sự khác biệt giữa dự đoán và kết quả mong muốn.

Lan truyền ngược: Mạng sẽ "đi ngược" từ kết quả sai về các lớp trước đó, tính toán xem từng kết nối (trọng số) cần điều chỉnh bao nhiêu để cải thiện dự đoán.


Tóm lại, lan truyền ngược giúp mạng thần kinh tự học từ sai lầm, giống như cách bạn điều chỉnh lực và góc ném bóng để lần sau ném vào rổ chính xác hơn.

Thuyết lan truyền kích hoạt (Spread Activation Theory) [[Xoa Thuyet lan truyền kích hoạt\|Xoa Thuyet lan truyền kích hoạt]]

Giờ mình lấy ví dụ đơn giản hơn về trí nhớ nhé! Hãy tưởng tượng bộ não của bạn là một mạng lưới khổng lồ gồm nhiều "node" (nút) thông tin. Khi bạn nghĩ về một từ hay một ý tưởng, nó sẽ kích hoạt những nút liên quan khác trong mạng lưới trí nhớ của bạn.

Ví dụ:

Bạn nghĩ về "cà phê". Ngay lập tức, bạn có thể liên tưởng đến "quán cà phê", "bạn bè", "trà sữa", hoặc thậm chí "buổi sáng". Đó là vì những khái niệm này có sự kết nối với nhau trong trí nhớ của bạn.

Khi một nút (ví dụ: "cà phê") được kích hoạt, nó sẽ làm lan truyền sự kích hoạt sang các nút liên quan khác. Đây chính là thuyết lan truyền kích hoạt.


Thuyết này không chỉ áp dụng cho trí nhớ, mà còn có thể dùng để tổ chức thông tin, đặc biệt là trong ghi chú như Obsidian:

Bạn có thể liên kết các ghi chú với nhau giống như cách các nút trong não bạn kết nối với nhau.

Khi bạn mở một ghi chú về "nhân sự", nó có thể kích hoạt (liên kết) đến các ghi chú khác như "phần mềm quản lý", "tuyển dụng", hay "bảo hiểm".


Sự khác biệt giữa hai khái niệm:

Lan truyền ngược là quá trình điều chỉnh trong mạng thần kinh để giúp nó dự đoán chính xác hơn sau mỗi lần sai.

Lan truyền kích hoạt là quá trình một ý tưởng hay khái niệm kích hoạt các khái niệm liên quan khác, như cách bạn liên tưởng từ này sang từ khác trong trí nhớ.


Hy vọng cách giải thích này giúp bạn dễ hiểu hơn! Nếu cần mình làm rõ thêm điểm nào nữa, cứ hỏi nhé! 😄

