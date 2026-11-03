# Thí nghiệm 06 — So sánh ngân sách unlearning trong Phase 1

## Mục tiêu và căn cứ

So sánh Shrink/Expanding gốc tại các mốc **1, 3, 5, 10, 20 epoch unlearning**,
giữ Original/Retrain đã train 30 epoch và toàn bộ cấu hình còn lại.
Đây là khảo sát ngân sách của phương pháp gốc trong **Phase 1**, không thêm
distance weights, retain KL hay BN freeze.

Kết quả run `cifar100_phase1_paper_forget4_4905e906` trong `test-phase-1-100/`
có test retain: Retrain 52,27%, Shrink 43,42%, Expanding 43,39%; cả hai phương
pháp unlearning đều có train/test forget accuracy 0% tại 10 epoch/80 update.
Vì vậy cần kiểm tra mức quên và retain ở các mốc sớm hơn. Mốc 20 kiểm tra hướng
tăng ngân sách; không mặc định ít hay nhiều epoch sẽ tốt hơn.

Trạng thái: đã chuẩn bị runner và kiểm tra bằng dữ liệu nhỏ tại local;
chưa có kết quả Kaggle của phép so sánh nhiều mốc này.

## Chạy một cell riêng

Trong [notebook Phase 1](boundary-unlearning-cifar-100-experiment.ipynb), tìm
section **Optional: Phase 1 epoch comparison** sau visualization, trước export.
Chạy sau evaluation; không cần chạy các cell hình trước. Đặt:

```python
RUN_EPOCH_SWEEP = True
UNLEARN_EPOCHS_TO_TEST = [1, 3, 5, 10, 20]
EPOCH_SWEEP_SOURCE = globals().get("phase1_manifest")
```

Chạy riêng cell đó. Không sửa `ORIGINAL_EPOCHS`, `UNLEARN_EPOCHS` ở cấu hình chính
và không chạy lại các cell train/tag/evaluation của flow chính.

Nếu notebook Kaggle đang mở chưa có cell mới: commit/push code, fetch/checkout
trong cell cập nhật nhỏ rồi thêm một code cell:

```python
from pathlib import Path
UNLEARN_EPOCHS_TO_TEST = [1, 3, 5, 10, 20]
exec((Path(CODE_ROOT) / "experiments" / "run-epoch-sweep.py").read_text())
```

Nếu kernel đã restart, gắn lại bundle Phase 1 và chạy các cell bootstrap,
cấu hình/dataset của Phase 1 (không chạy cell train). Chỉ định nguồn rõ ràng:

```python
EPOCH_SWEEP_SOURCE = Path("/kaggle/input/my-phase1-output/result.zip")
```

Nguồn có thể là ZIP chỉ chứa một manifest, thư mục chứa một manifest hoặc chính
file `phase1_manifest.json`. Dữ liệu gốc phải được gắn và khớp fingerprint.
Không nạp legacy best checkpoints: ở lượt chạy đã cung cấp, cả Original và
Retrain best checkpoint đều khác snapshot được đánh giá trong bundle.

## Kiểm soát phép so sánh

- Original/Retrain được khôi phục từ evaluated snapshots; không train lại.
  Bảng baseline lấy đúng metrics đã export, sau khi xác nhận dataset giống hệt.
- Mỗi candidate khởi tạo độc lập từ cùng Original, optimizer SGD mới, loader mới
  cùng seed và thứ tự mẫu. Khôi phục RNG Python/NumPy/Torch của phiên sau khi chạy.
- Cả hai phương pháp dùng cùng ngân sách. Với 500 ảnh forget/batch 64:
  1/3/5/10/20 epoch tương ứng 8/24/40/80/160 update mỗi phương pháp.
- LR/batch size/seed/FGSM/chuẩn hóa/giới hạn batch lấy từ config nguồn;
  Shrink giữ `EXTRA_EXP=None`, BN update; Expanding giữ nguyên code tác giả.
- Chạy lại cả mốc 10 trong cùng phép so sánh để có điều kiện seed đồng nhất.
  Expanding khởi tạo lớp phụ ngẫu nhiên, nên mốc 10 mới có thể khác lượt cũ khi
  global RNG của lượt cũ đã đi qua training/các cell khác.
- Không đổi `rows`, `config`, `unlearned_models`, `unlearn_stats` hay bundle
  của bảng Phase 1 chính. Kết quả riêng nằm trong biến `epoch_sweep`.

## Đầu ra và Phase 2

```text
result/epoch_sweep/<source_run>_<hash>/
├── config.json                 # Nguồn snapshot, hash code, seed và danh sách epoch
├── metrics.csv                 # Hai baseline + kết quả của từng phương pháp/mốc
├── epochs.png / epochs.pdf     # Train/test retain/forget theo epoch
└── runs/cifar100_phase1_epochs<E>_<hash>/
    ├── phase1_manifest.json
    ├── config.json / rows.json / metrics.csv / results.html
    ├── checkpoints/            # Đúng 4 model được đánh giá ở mốc E
    └── boundary_shrink/losses.csv
```

Cell tạo `result.zip` gồm toàn bộ so sánh và các kết quả cũ. Đồng thời hiện link
ZIP **riêng từng mốc**, đặt ngoài `result/` để không lồng ZIP và nhân đôi archive.
Gắn ZIP của mốc được chọn vào Phase 2; nó có đúng một manifest nên loader hiện tại
đọc trực tiếp được. ZIP tổng có nhiều manifest: nếu dùng nó cần giải nén và chỉ
định chính xác manifest/thư mục của mốc muốn dùng, không truyền cả ZIP tổng.

Đọc trade-off retain/forget và chi phí cùng với Retrain. Mọi mốc được xác định
trước; nếu chọn cấu hình dựa trên test thì kết quả chọn đó là thăm dò, cần
validation hoặc đánh giá độc lập trước khi đưa ra kết luận tổng quát.

## Kiểm chứng

Tests xác nhận mỗi mốc có số update đúng và khởi tạo lại từ Original, giữ nguyên
baseline/RNG, trùng state_dict với chạy Shrink độc lập cùng seed, tạo được bundle
Phase 2 cho từng mốc, từ chối sai dữ liệu/ngân sách và chạy entry độc lập/export ZIP.

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/boundary-mpl python -m unittest discover -s tests -v
```
