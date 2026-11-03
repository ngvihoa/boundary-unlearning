# Thí nghiệm 03 — Phase 2: Distance weighting và retain distillation

## Nguồn và trạng thái

- Ngày tích hợp: 03/11/2026.
- Nguồn thuật toán: ba cell cuối của `no-refactor-new.ipynb` (nhãn CELL 18–20).
- Flow: [CIFAR-10](boundary-unlearning-cifar-10-experiment.ipynb) và
  [CIFAR-100 Phase 2](boundary-unlearning-cifar-100-phase2.ipynb).
- CIFAR-100 được tách khỏi [Phase 1](02-cifar-100.md): Phase 1 chỉ có 4 model gốc;
  Phase 2 nạp các snapshot đã đánh giá, không train lại Original/Retrain/Shrink/Expanding.
- Đã có kết quả các lần chạy flow cũ do người dùng cung cấp; flow hai notebook mới
  được kiểm chứng bằng dữ liệu nhỏ tại local, chưa có lượt chạy Kaggle đầy đủ mới.
- Không thay đổi cấu hình train/retrain: 30 epoch, SGD LR 0.1, momentum 0.9,
  batch 64; mỗi giai đoạn unlearning 10 epoch, LR 1e-5 với profile `paper`.
- Giữ lớp quên: CIFAR-10 `4 = deer`; CIFAR-100 `4 = beaver`.

## Các model được so sánh

| ID phương pháp | Student khởi tạo từ | Teacher pseudo-label | Teacher retain KL |
| --- | --- | --- | --- |
| `boundary_shrink` | Original | Original | Không |
| `boundary_expanding` | Original, mở classifier | Không dùng pseudo-label FGSM | Không |
| `boundary_shrink_weight_assign` | Original | Original | Không |
| `boundary_shrink_weight_assign_distill` | Original | Original | Original |
| `boundary_shrink_weight_assign_boundary_distill` | Distance weighting | Original | Boundary Shrink |

Cộng Original và Retrain thành **7 model**. Ba biến thể mới là phần mở rộng của
Boundary Shrink, không phải các baseline Finetune/Random Labels/Fisher… của bài báo.
`EXTRA_EXP` tiếp tục dành riêng cho enhanced loss của tác giả; mặc định `None`.
Khi chạy biến thể Boundary KL, teacher phải là Boundary Shrink thuần (`EXTRA_EXP=None`).

## Thuật toán mới

1. Tính dấu gradient CE của Original một lần trên ảnh forget sạch.
2. Dò epsilon theo bước `1/255` đến `FGSM_BOUND=0.1`, trong miền pixel `[0,1]`,
   làm tròn pixel về mức 8-bit trước khi forward teacher.
3. Ghi epsilon và prediction tại lần đầu prediction khác nhãn thật. Ảnh đã sai
   từ đầu có epsilon 0. Ảnh không vượt biên dùng epsilon tối đa và prediction cuối.
   Đây là khoảng cách ước lượng theo một hướng FGSM, không phải khoảng cách biên tối ưu.
4. Weight thô là `clamp(1 - epsilon / bound, 0.1, 1)` rồi chia trung bình batch.
   Weight cuối có trung bình 1 và có thể lớn hơn 1. Loss forget là weighted CE
   trên ảnh sạch với pseudo-label tìm được.
5. Hai biến thể distillation thêm `0.1 * T² * KL(teacher || student)`, `T=2`, trên
   một batch retain mỗi bước forget; quay vòng loader retain khi hết dữ liệu.
   BatchNorm cập nhật khi forward forget và dùng eval khi forward retain.

Distance search dùng input mean/std `(0.5, 0.5, 0.5)` (`matched`), đúng notebook nguồn.
Baseline CIFAR-10 giữ `repo_exact`; baseline CIFAR-100 dùng `matched`. Vì vậy,
so với Shrink mặc định trên CIFAR-10, các biến thể mới còn khác cả chuẩn hóa attack
và cách chọn pseudo-label. Cần xét các khác biệt đó khi quy kết hiệu quả cho weighting.
KL từ Boundary Shrink thay đổi cả initializer lẫn teacher so với KL từ Original;
không thể coi đây là thí nghiệm chỉ thay teacher.

## Ngân sách và thời gian

| Student | Epoch/student | Bước CIFAR-10 | Bước CIFAR-100 |
| --- | ---: | ---: | ---: |
| Distance weighting | 10 | 790 | 80 |
| Distance + Original KL | 10 | 790 | 80 |
| Distance + Shrink KL | 20 | 1.580 | 160 |

Biến thể cuối cần cả model distance weighting lẫn teacher Boundary Shrink.
`seconds`/`update_steps` là giai đoạn đang chạy; `student_update_steps` bao gồm
giai đoạn distance weighting khởi tạo student; `total_seconds`/`total_update_steps`
bao gồm cả hai model phụ thuộc, không bao gồm train Original. Với cấu hình đầy đủ,
tổng số bước để tạo cả ba model là 2.370 trên CIFAR-10 và 240 trên CIFAR-100.
Chi phí chuẩn bị model được tính đầy đủ dù model đã có sẵn và được tái sử dụng.

## Chạy từng cell và xuất kết quả

**CIFAR-100:** chạy xong Phase 1, tải `result.zip`, rồi gắn gói đó hoặc Output đã
lưu của Phase 1 vào notebook Phase 2, cùng dataset CIFAR-100. Cell cấu hình tự tìm
một bundle hoặc dùng đường dẫn `PHASE1_SOURCE` chỉ định. Nó kiểm tra SHA-256 và
khôi phục đúng 4 model đã đánh giá, kể cả Expanding dạng Sequential đã prune.
Pixel/label, thứ tự lớp và normalization phải khớp Phase 1; không có fallback train.

Chạy lần lượt ba cell biến thể và evaluation để so sánh 7 model. BN update là mặc
định để khớp Phase 1; các cell BN/weighting ablation cuối notebook mặc định tắt.
Chạy lại cell runner chỉ định nghĩa lại hàm, không thay các baseline đã nạp.
Phase 2 ghi source run tag, SHA manifest/checkpoints và sao chép bundle tham chiếu
vào `result/phase1_source/`; cell cuối tạo ZIP gồm tham chiếu và kết quả mới.

**CIFAR-10 và các phiên cũ đang có model:** không cần chạy lại các cell train:

1. Cập nhật code bằng cell **Small code update**: bỏ comment hai lệnh Git rồi chạy
   để fetch/checkout và reload module. Nếu đã fetch bằng cell khác, chỉ cần reload.
2. Cập nhật cell cấu hình và chạy cell `baseline_tag_settings` để tạo run tag mới.
   Cell tính tag đã tách khỏi cell train Original. Giữ nguyên dữ liệu và cấu hình
   training nếu dùng tiếp các model cũ.
3. Chạy cell **Incremental execution** để định nghĩa lại runner. Dictionary model
   đang có được giữ lại. Có thể bỏ qua cell hai baseline nếu đã chạy chúng.
4. Chạy các cell biến thể muốn thêm. Boundary KL yêu cầu đã có Boundary Shrink và
   distance weighting; không yêu cầu đã chạy Original KL. Rerun một cell chỉ train
   phương pháp đó. Rerun Shrink/distance weighting sẽ xóa kết quả Boundary KL trong
   bộ nhớ để yêu cầu tính lại model phụ thuộc trước khi tổng hợp.
5. Chạy evaluation và export. Evaluation kiểm tra đủ các phương pháp đã chọn,
   tạo bảng accuracy và bảng chi phí riêng, CSV/JSON cùng checkpoint trong `result/`.
   Cell export tạo hai HTML và `result.zip`.

Run tag mới chứa epsilon step, chuẩn hóa distance, lambda/T và quan hệ initializer/
teacher. Config JSON lưu stats của tất cả phương pháp. Hash đặt tên checkpoint
Original/Retrain giữ tương thích với baseline mặc định trước khi thêm ba biến thể;
đường dẫn checkpoint baseline thực tế được ghi riêng trong config. CIFAR-10 hiện
lưu trực tiếp vào `result/checkpoints/`; nếu checkpoint cũ còn ở `working/checkpoints/`,
cần chép vào đường dẫn mới hoặc giữ `CHECKPOINT_ROOT` cũ khi tiếp tục phiên.

Khi nạp lại baseline từ checkpoint, lưu ý code train vẫn lưu model tốt nhất theo
test accuracy và trả về model cuối epoch. Để so sánh trong cùng phiên, dùng cùng
Original đang có cho mọi phương pháp; không trộn model cuối với checkpoint tốt nhất.

## Kiểm chứng

- 14 kiểm tra tự động, gồm loader CIFAR-100 và 8 kiểm tra mới: epsilon đầu tiên,
  ảnh đã sai/không vượt biên, điểm cuối epsilon grid, weight, KL/gradient/BatchNorm,
  initializer, số bước, checkpoint, quan hệ phụ thuộc và bảng hiển thị.
- Chạy thử hai notebook trên fixture nhỏ với AllCNN 10/100 đầu ra: train/retrain,
  đủ 5 biến thể, đánh giá 7 model, CSV/JSON, hai HTML và ZIP.
- Kiểm tra rerun riêng Original KL giữ nguyên Original và model distance weighting;
  rerun distance weighting làm mất hiệu lực Boundary KL phụ thuộc.
- Kết quả kiểm chứng không thay thế thí nghiệm đầy đủ hoặc đánh giá nhiều seed.

## Visualization sau tích hợp

Hai notebook có thêm [các cell visualization dùng chung](visualization.md) trước
export: accuracy/chi phí, Grad-CAM, entropy và t-SNE. Dùng đủ 7 model theo
`METHODS_TO_RUN`, không cập nhật trọng số. Hình và dữ liệu vẽ được lưu ở
`result/figures/<run_tag>/` và được đưa vào ZIP. Cấu hình train/unlearning ở trên
không thay đổi.

```bash
python -m unittest discover -s tests -v
```
