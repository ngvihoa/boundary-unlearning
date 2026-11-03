# Visualization cho flow CIFAR-10/CIFAR-100

## Nguồn và phạm vi

Được tách và điều chỉnh từ ba cell visualization trong `da-tgmt-v2.ipynb`:
Grad-CAM, t-SNE và log entropy. Bổ sung hai biểu đồ accuracy và chi phí cho flow
5 biến thể unlearning + Original/Retrain. File nguồn của bạn vẫn được giữ nguyên.

Code dùng chung nằm ở [visualization.py](../boundary_unlearning/visualization.py).
Sau khi tách CIFAR-100: Phase 1 chỉ vẽ 4 model gốc; Phase 2 vẽ 4 model được
restore cùng 3 biến thể. CIFAR-10 tiếp tục dùng flow 7 model hiện tại.
Phần training, chuẩn hóa dữ liệu và thuật toán unlearning không thay đổi.
Các hình minh họa khi kiểm chứng dùng fixture và model nhỏ, chưa phải kết quả
thí nghiệm đầy đủ để đưa vào báo cáo.

## Chạy trong một phiên Kaggle đã có model

1. Cập nhật code bằng cell **Small code update**. Nếu chỉ đổi visualization,
   không chạy lại training hay các phương pháp unlearning.
2. Nếu trọng số đã thay đổi, chạy lại cell evaluation để cập nhật `rows`.
3. Chạy cell **VISUALIZATION SETUP** để reload module, chọn mẫu và lấy lại tham chiếu
   tới Original, Retrain cùng các model trong `METHODS_TO_RUN`.
4. Chạy riêng một hoặc nhiều cell vẽ. t-SNE là tùy chọn, có thể bỏ qua khi cần nhanh.
5. Chạy cell **EXPORT RESULTS** cuối notebook để cập nhật ZIP.

Riêng cell accuracy/chi phí có thể chạy độc lập, không cần Visualization setup
hay model object; chỉ cần `rows`, `OUTPUT_ROOT`, `run_tag`, `NUM_CLASSES`. Có thể
nạp `rows` từ metrics CSV để vẽ lại phần tổng quan mà không forward model.

Cần Matplotlib và scikit-learn (riêng t-SNE, API có `max_iter`). Import module
không tự tải dataset, train, vẽ hoặc tạo file. Các hàm nhận `device` và model rõ
ràng, hỗ trợ AllCNN và model Sequential đã prune của Boundary Expanding.

## Lấy mẫu và tính nhất quán

Setup có các tham số riêng, không tham gia hash cấu hình training:

| Tham số | Mặc định |
| --- | ---: |
| `VIS_SEED` | `SEED`, mặc định 2022 |
| `VIS_GRADCAM_SAMPLES` | 4 ảnh train forget |
| `VIS_ENTROPY_SAMPLES` | Tối đa 2.000 ảnh cho mỗi split |
| `VIS_TSNE_SAMPLES` | Tối đa 1.500 ảnh test tổng cộng |
| `VIS_TSNE_PER_CLASS` | Tối đa 120 ảnh/lớp |
| `VIS_TSNE_PERPLEXITY` | 30, giảm xuống `n_samples - 1` nếu cần |
| `VIS_TSNE_ITERATIONS` | 1.000 |
| `VIS_BATCH_SIZE` | 128 |

Mẫu được chọn ngẫu nhiên có phân tầng theo lớp, dùng RNG riêng và không reset
seed toàn cục. Mọi model dùng cùng ảnh, theo cùng thứ tự. Giới hạn áp dụng trước
forward, không làm tròn lên theo kích thước batch. Với CIFAR-10 đầy đủ, t-SNE
mặc định lấy 1.200 ảnh; CIFAR-100 lấy 1.500 ảnh. Không dùng test competition
không nhãn của CIFAR-10. Dataset phải có `targets`, `labels` hoặc là TensorDataset;
hỗ trợ cả Subset lồng nhau.

Chỉ số mẫu là chỉ số tương đối trong dataset truyền vào: Grad-CAM dùng train
forget Subset; entropy có chỉ số riêng cho từng split; t-SNE dùng test dataset.
Setup ghi split, chỉ số, seed, class names, input normalization và run tag trong
`visualization_config.json`.

## Cách đọc từng hình

**Accuracy:** hai panel chênh lệch test retain/forget so với Retrain, đơn vị điểm
phần trăm, và một panel giá trị accuracy. Đường 0 là bằng Retrain; dấu dương chỉ
có nghĩa accuracy cao hơn, không tự có nghĩa unlearning tốt hơn.

**Chi phí:** thời gian giai đoạn và tổng chi phí gồm prerequisite, cùng số bước
stage/student/prerequisite. Không tính train Original. Biến thể Boundary KL có
hai giai đoạn cập nhật student và cần cả teacher Shrink; model có sẵn vẫn có chi
phí chuẩn bị. Thời gian thiếu do nạp checkpoint được giữ là thiếu, không gán bằng 0.

**Grad-CAM:** mọi model giải thích logit của cùng lớp forget. Caption tách confidence
của prediction khỏi `P(forget)`. Heatmap được chuẩn hóa riêng từng ảnh/model nên
không so sánh độ đỏ như cường độ tuyệt đối giữa các model. Tối đa 4 model/trang;
hai trang mặc định dùng cùng ảnh. Gradient chỉ tính với activation phục vụ CAM;
trọng số, gradient tham số, hooks và trạng thái train/eval không bị thay đổi sau khi vẽ.

**Entropy:** trục ngang là `ln(H)`, với Shannon entropy `H` tính bằng natural log
(nats), chưa chia `ln(num_classes)`. ECDF thể hiện tỷ lệ mẫu có log entropy không
vượt quá mức trên trục ngang. Trục chung giữa các model; đường có màu/kiểu riêng
cho train retain, train forget, test. Không tự thay train split bằng test/validation.
Không so sánh trực tiếp thang entropy CIFAR-10/CIFAR-100 như cùng một phân phối.

**t-SNE:** trích features trước classifier, fit embedding riêng cho từng model.
Seed và mẫu cố định giúp lặp lại cách vẽ, nhưng các trục không được căn chỉnh;
không đọc khoảng dịch chuyển giữa hai panel như khoảng cách định lượng. Với
CIFAR-100, retain có màu xám để tránh legend 100 lớp; forget là vòng đỏ.

Các hình Grad-CAM/entropy/t-SNE dùng để giải thích kết quả, không thay thế accuracy,
so sánh với Retrain hoặc kiểm tra privacy. Tiêu đề không khẳng định tái lập chính
xác hình trong bài báo.

## Các file xuất ra

```text
result/figures/<run_tag>/
├── visualization_config.json
├── accuracy.{png,pdf,json}
├── cost.{png,pdf,json}
├── gradcam_1.{png,pdf,json,npz}
├── gradcam_2.{png,pdf,json,npz}
├── entropy.{png,pdf,json,npz}
└── tsne.{png,pdf,json,npz}
```

NPZ lưu chỉ số và dữ liệu tương ứng: CAM/probabilities, log entropy, hoặc
features/embedding của từng model. JSON của accuracy/chi phí giữ nguyên các hàng
metrics để vẽ lại mà không forward model. Các file cùng tên được cập nhật khi
rerun. Lần export tiếp theo vẫn giữ hình đã tạo trước đó, kể cả khi bỏ qua một
cell vẽ; cần kiểm tra run tag và metadata khi sử dụng lại hình.

Cell export chỉ đóng gói các hình đã có, không chạy visualization lại.

## Kiểm chứng

- Test tích hợp hai phase thực thi đủ các cell accuracy/chi phí, Grad-CAM,
  entropy và t-SNE trên AllCNN 100 lớp, gồm Expanding đã prune và model nạp từ ZIP.
  Phase 1 dùng 4 model, Phase 2 dùng 7; test giảm entropy còn 12 mẫu/split,
  t-SNE còn 100 mẫu/250 iteration để kiểm chứng tại local.
- Sau mỗi cell vẽ, so sánh trọng số/BN buffers, gradient, train/eval mode và
  forward hooks để xác nhận model không bị thay đổi. Kiểm tra mapping model,
  chỉ số mẫu, shape/giá trị hữu hạn trong NPZ và các file hình/dữ liệu có trong ZIP
  của cả hai phase.
- Test CPU cho sampling, gradient/BatchNorm/state/hook cleanup, entropy đối chiếu
  công thức, giới hạn mẫu, t-SNE 100 lớp, model prune, đơn vị accuracy/chi phí và ZIP.
- Chạy thử hai notebook với AllCNN và fixture nhỏ, đủ 7 model và toàn bộ cell vẽ.
- Xem ảnh PNG xuất ra để kiểm tra caption, legend, bố cục nhiều panel/trang.
- Chưa chạy visualization trên đầy đủ các model đã huấn luyện ở Kaggle.

```bash
python -m unittest discover -s tests -v
```
