# Thí nghiệm 04 — BatchNorm ablation trên CIFAR-100

## Vị trí trong flow hai phase

Thí nghiệm này thuộc **Phase 2**. Chạy trong
[notebook Phase 2](boundary-unlearning-cifar-100-phase2.ipynb) sau khi nạp bundle
Phase 1; cell BN ablation đã được chuyển ra khỏi notebook baseline.
Các hướng dẫn dùng model trong phiên bên dưới cũng áp dụng cho model vừa restore.
Kết quả do người dùng cung cấp cho Shrink update/freeze có test retain 43,42%/51,15%,
train forget 0%/41,20%, test forget 0%/4%; đây là các lượt chạy trước khi tách phase,
không phải kết quả mới của notebook Phase 1 vừa sửa.

## Mục tiêu và trạng thái

Kiểm tra việc BatchNorm dùng/cập nhật statistics theo riêng forget set có liên
quan đến suy giảm retain accuracy hay không. Đã có bảng kết quả Kaggle do người
dùng cung cấp; flow khôi phục model qua hai phase mới được kiểm tra tại local.

Notebook CIFAR-100 đã có run `cifar100_paper_forget4_1084d69d`, quên lớp
`4 = beaver`, với test retain/forget: Original 52,65%/26%; Retrain 52,27%/0%;
Boundary Shrink 43,42%/0%. Đây là output lần chạy trước, không phải kết quả BN freeze.

## Thiết kế đối chứng

- Cùng một Original trong bộ nhớ làm điểm khởi tạo cho cả hai candidate.
- Mặc định chỉ chạy **Boundary Shrink thuần**, mỗi candidate 10 epoch, LR `1e-5`,
  momentum `0.9`, batch size 64, seed 2022, bound 0.1, normalization `matched`.
  Các giá trị được lấy từ cấu hình đang có trong notebook.
- `update`: hành vi cũ, BN dùng statistics theo batch forget và cập nhật running
  mean/variance. `freeze`: gọi BN.eval(), dùng running statistics của Original
  trong forward và giữ nguyên chúng; affine weight/bias vẫn được học.
- Đây là ablation chế độ BN: cả thống kê dùng khi forward và việc cập nhật buffer
  đều khác nhau giữa train/eval. Không phải phép thử chỉ thay việc ghi buffer.
- Cả hai candidate dùng loader mới với cùng seed; RNG Python/NumPy/Torch được
  đặt lại theo cặp và khôi phục sau khi chạy. Original và Retrain chỉ được đánh giá,
  không train lại và không nạp checkpoint tự động.
- 500 ảnh forget, 8 batch/epoch, **80 bước/candidate**, 160 bước cho cặp.
- Các dictionary `unlearned_models`, `unlearn_stats`, `rows` của flow chính được
  giữ nguyên. Kết quả đối chứng được lưu trong biến `bn_ablation` riêng.

## Cách chạy trên Kaggle, không Run All

Sau khi push code mới lên GitHub:

1. Trong phiên đang có Original/Retrain, chạy hai lệnh Git ở cell **Small code update**
   để fetch/checkout code. Không chạy lại cell train.
2. Tìm section **Optional: paired BatchNorm ablation** gần cuối notebook.
   Đổi `RUN_BN_ABLATION = True`; giữ `BN_ABLATION_METHODS = ["boundary_shrink"]`.
3. Chạy đúng cell này. Nó tự reload module, train hai candidate unlearning,
   đánh giá cả train/test retain/forget, tạo bảng và log, rồi export `result.zip`.
   Không cần chạy evaluation/visualization/export của flow chính lại.

Nếu notebook trên Kaggle chưa có cell mới, sau khi fetch/checkout có thể thêm
một code cell và chạy:

```python
from pathlib import Path
exec((Path(CODE_ROOT) / "experiments" / "run-bn-ablation.py").read_text())
```

Entry script mặc định chọn Shrink, không cần đặt cờ khi gọi trực tiếp. Có thể chạy
lại riêng cell để lặp lại cặp. Các model/dataset/loader của phiên phải đang tồn tại.
Nếu kernel đã restart, cần nạp lại các object trước; checkpoint tốt nhất có thể
khác Original cuối epoch đã dùng trong run cũ. Đối chứng mới luôn dùng cùng model
cho hai mode và ghi fingerprint để nhận diện điểm khởi tạo thực tế.

## Chẩn đoán và đầu ra

```text
result/bn_ablation/<source_run_tag>_<hash>/
├── config.json
├── metrics.csv
├── results.html
├── checkpoints/
│   ├── boundary_shrink_bn_update.pt
│   └── boundary_shrink_bn_freeze.pt
├── boundary_shrink_bn_update/losses.csv
├── boundary_shrink_bn_freeze/losses.csv
├── losses.png / losses.pdf
├── boundary_search_audit.npz / .json
└── boundary_search_audit.png / .pdf
```

- `config.json`: settings, SHA-256 state_dict của Original/Retrain, nguồn model
  trong bộ nhớ, stats của từng candidate, audit; hash thư mục phụ thuộc các thông tin này.
- `metrics.csv`: accuracy cùng độ thay đổi lớn nhất của BN running mean/variance
  và số lần cập nhật BN so với Original. Candidate freeze phải có mức thay đổi 0.
- Loss CSV: forget CE, KL đã nhân T², lambda × KL và total loss theo từng update.
  Shrink thuần không dùng KL nên hai cột KL bằng 0.
- Boundary-search audit là **một lượt riêng qua Original teacher**, dùng thuật toán
  epsilon search của distance weighting và input normalization. Ghi epsilon,
  batch-normalized weight, true/pseudo-label, batch index và histogram; không
  thay pseudo-label hoặc loss của Shrink đang kiểm chứng.
- Audit chỉ mô tả một epoch seed-shuffled; không phải phân phối gộp 10 epoch
  của training. `first_step_fraction` và độ lệch chuẩn weight giúp kiểm tra
  giả thuyết hầu hết epsilon ở bước đầu nên weighting gần đồng đều.
- Export tạo lại `result.zip`, bao gồm cả kết quả cũ và thư mục đối chứng mới.
  File có cùng hash được cập nhật khi chạy lại đúng thiết lập và cùng model.

Sau khi kiểm tra cặp Shrink, có thể mở rộng rõ ràng:

```python
BN_ABLATION_METHODS = [
    "boundary_shrink",
    "boundary_shrink_weight_assign",
    "boundary_shrink_weight_assign_distill",
]
```

Mỗi phương pháp sẽ có cặp update/freeze độc lập từ Original. Hai biến thể distance
ghi thêm `boundary_samples.npz` cho toàn bộ lượt mẫu đã train, kèm epoch, epsilon
và weights. Original-KL có log KL thực để đối chiếu với CE. Không thêm biến thể
warm start trong ablation này vì nó đổi initializer và có teacher phụ thuộc.

## Cách diễn giải

- Retain tăng, forget vẫn thấp: hướng BN freeze có triển vọng, cần kiểm tra nhiều seed.
- Retain tăng nhưng forget quay về gần Original: mức 0% trước đó có thể liên quan
  đến thay đổi BN hơn là quá trình học quên đủ mạnh; freeze chưa phải giải pháp hoàn chỉnh.
- Ít thay đổi: tiếp tục kiểm tra pseudo-label, epsilon/weight và độ lớn loss.

Không thay đổi cấu hình train/retrain hay khẳng định BN là nguyên nhân trước khi
có kết quả đối chứng thật.

## Kiểm chứng code

- Kiểm tra mặc định và `bn_stats_mode="update"` cho ra cùng state_dict.
- Freeze giữ nguyên toàn bộ BN buffers nhưng cập nhật classifier và BN affine;
  kiểm tra Shrink 10/100 lớp và distance + Original KL.
- Kiểm tra loss CSV/NPZ, paired RNG, model modes/gradient, fingerprint và checkpoint.
- Thực thi cell notebook độc lập trên fixture và kiểm tra ZIP, giữ nguyên kết quả
  flow chính; không gọi train từ đầu của Original/Retrain.
- Chạy suite bằng `python -m unittest discover -s tests -v`.
