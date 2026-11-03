# Thí nghiệm 05 — Distance weights so với weights = 1 trên CIFAR-100

## Mục tiêu và căn cứ

Tách ảnh hưởng của **sample weighting** khỏi cách tìm pseudo-label. Trạng thái:
đã chuẩn bị code và kiểm chứng bằng dữ liệu nhỏ tại local; chưa chạy cặp đối chứng
này trên CIFAR-100 thật ở Kaggle.

Theo kết quả BN ablation do người dùng cung cấp, distance weighting + BN freeze
đạt test retain 51,80%, test forget 6%, train forget 50,60%. Thêm Original KL cho
cùng accuracy ở độ chính xác hiển thị; mean scaled KL là 0,003273 so với CE 3,214009.
Teacher audit một lượt có 95,4% mẫu vượt biên tại epsilon đầu tiên (1/255),
weight std khoảng 0,008514, min/max khoảng 0,99746/1,04015.

Các weights này gần 1. So sánh với Shrink thuần chưa cô lập được weighting, vì
Shrink dùng pseudo-label tại fixed bound, còn distance dùng tìm epsilon vượt biên.
Audit một lượt cũng chưa mô tả toàn bộ các epoch unlearning.

## Thiết kế

Hai student đều khởi tạo độc lập từ **cùng Original đang nằm trong bộ nhớ**:

| Candidate | Pseudo-label | Weight | BatchNorm | Retain KL |
| --- | --- | --- | --- | --- |
| Distance weights | Distance boundary search | Công thức distance hiện tại, chuẩn hóa theo batch | Freeze | Không |
| Uniform control | Cùng distance boundary search | Luôn bằng 1 | Freeze | Không |

Giữ epoch/LR/momentum/batch size/seed/bound/epsilon step và chuẩn hóa đầu vào.
Không sửa rounding ảnh hay thuật toán boundary search. BN dùng running statistics
của Original; BN affine parameters vẫn được học.

Entry script lấy `UNLEARN_EPOCHS`, `UNLEARN_LR`, `MOMENTUM`, `make_loader`, `SEED`,
`FGSM_BOUND` từ phiên hiện tại. Epsilon step mặc định `1/255`, có thể đặt qua
`WEIGHT_EPSILON_STEP` nếu lần chạy cần đối chứng đã dùng giá trị khác.
Cấu hình CIFAR-100 hiện tại là 10 epoch, LR `1e-5`, batch 64, 500 ảnh forget:
**80 update/candidate**, tổng 160 update. Nếu đang bật giới hạn batch cho smoke
run, entry tiếp tục dùng giới hạn đó và ghi vào config.

Original/Retrain chỉ được đánh giá, không train lại. Bảng có 4 hàng: hai baseline
và hai candidate. Kết quả mới nằm trong biến `weighting_ablation`; `rows`,
`unlearned_models`, `unlearn_stats` và `bn_ablation` hiện có được giữ nguyên.

## Chạy riêng trên Kaggle

Sau khi push code này lên GitHub, dùng cell cập nhật nhỏ hoặc chạy:

```python
!git -C /kaggle/working/boundary-unlearning fetch --depth 1 origin main
!git -C /kaggle/working/boundary-unlearning checkout --detach FETCH_HEAD
```

Trong phiên còn Original/Retrain, dataset và loaders, thêm một code cell:

```python
from pathlib import Path
exec((Path(CODE_ROOT) / "experiments" / "run-weighting-ablation.py").read_text())
```

Entry tự reload module, chạy hai candidate, đánh giá, hiển thị bảng/hình và tạo
`/kaggle/working/result.zip` kèm link tải. Không cần Run All hay chạy lại các cell
train, evaluation hoặc export của flow chính.

## Đầu ra và cách đọc

Thư mục `result/weighting_ablation/<source_run_tag>_<hash>/` gồm:

- `config.json`: cấu hình, fingerprint Original/Retrain, stats và kiểm tra cặp.
- `metrics.csv`, `results.html`: accuracy và chi phí của 4 model.
- `checkpoints/`: hai student; thư mục riêng từng student có `losses.csv` và
  `boundary_samples.npz` ghi epsilon/weight/true label/pseudo-label/epoch của
  **toàn bộ lượt mẫu thực sự dùng khi unlearning**.
- `pair_validation.json`: true labels, pseudo-labels, epsilon và epoch phải khớp
  hoàn toàn giữa hai nhánh; uniform weights phải luôn bằng 1. Nếu sai, dừng so sánh
  bằng lỗi rõ ràng. Loader factory phải tạo loader mới với cùng seed cho mỗi nhánh.
- `weighting_comparison.csv`: chênh lệch **Uniform − Distance**, accuracy tính
  bằng điểm phần trăm. Retain delta dương nghĩa là uniform giữ accuracy tốt hơn;
  forget delta dương nghĩa là uniform còn nhận diện lớp forget nhiều hơn.
- `applied_weights.png/.pdf`: phân phối weights thực dùng qua tất cả epoch.
  `losses.png/.pdf`: loss theo từng update; cả hai nhánh đều có KL bằng 0.

CE của nhánh distance có weighting, CE của uniform không có weighting; không
dùng riêng chênh lệch hai loss để kết luận hiệu quả quên. Đọc đồng thời train/test
retain/forget và chênh lệch so với Retrain. Một seed là đối chứng cho lần chạy này;
kết quả gần nhau cần được xác nhận bằng nhiều seed trước khi khẳng định tổng quát.

Gửi lại bảng accuracy, bảng chênh lệch và `Pair validation` để đánh giá bước tiếp.

## Kiểm chứng code

Tests kiểm tra BN buffers giữ nguyên, không train baseline, chỉ có hai student,
uniform CE đúng với unweighted CE và weights luôn bằng 1; khi distance weights
cũng bằng 1 thì cả hai student phải có state_dict giống hệt. Kiểm tra thêm việc
từ chối loader không khớp, giữ hành vi distance mặc định, chạy entry độc lập,
giữ kết quả cũ và export đầy đủ ZIP.

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/boundary-mpl python -m unittest discover -s tests -v
```
