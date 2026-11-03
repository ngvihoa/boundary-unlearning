# Boundary Unlearning trên Kaggle

[Notebook CIFAR-10](experiments/boundary-unlearning-cifar-10-experiment.ipynb)
và hai notebook CIFAR-100 nằm trong `experiments/`:

| Notebook CIFAR-100 | Phạm vi |
| --- | --- |
| [Phase 1](experiments/boundary-unlearning-cifar-100-experiment.ipynb) | Train và đánh giá Original, Retrain, Shrink, Expanding; export model/kết quả |
| [Phase 2](experiments/boundary-unlearning-cifar-100-phase2.ipynb) | Nạp đúng 4 model Phase 1; thử distance/KL và các ablation, không train lại baseline |

Các hàm và class nằm trong package `boundary_unlearning/`, có thể import độc lập
sau khi clone repo.

Thông tin từng thí nghiệm:

- [Thí nghiệm 01 — CIFAR-10, quên lớp deer](experiments/01-cifar-10.md): cấu hình
  và kết quả đã lưu từ lần chạy Kaggle ngày 17/10/2026.
- [Thí nghiệm 02 — CIFAR-100 Phase 1, quên lớp beaver](experiments/02-cifar-100.md):
  cấu hình phương pháp gốc, đánh giá 4 model và đóng gói bàn giao sang Phase 2.
- [Thí nghiệm 03 — Ba biến thể distance weighting/KL](experiments/03-distance-weighted-variants.md):
  luồng tích hợp vào cả hai notebook, ngân sách cập nhật và cách chạy từng cell.
- [Thí nghiệm 04 — BatchNorm ablation CIFAR-100](experiments/04-cifar100-batchnorm-ablation.md):
  chạy cặp Shrink update/freeze từ cùng Original, không train lại baseline.
- [Thí nghiệm 05 — Distance weights so với weights = 1](experiments/05-cifar100-weighting-ablation.md):
  cùng distance pseudo-label, BN freeze và initializer; chỉ thay sample weights.

| File | Nội dung |
| --- | --- |
| `data.py` | Đọc NPY/CSV, xử lý nhãn, dataset và DataLoader |
| `cifar100.py` | Đọc pickle train/test/meta và nhãn fine của CIFAR-100 |
| `models.py` | AllCNN và khởi tạo classifier |
| `attacks.py` | FGSM của tác giả và tìm epsilon vượt biên theo hướng FGSM |
| `enhanced.py` | Curvature, probability/distance weighting và retain KL |
| `training.py` | Train/retrain, checkpoint và đánh giá accuracy |
| `unlearning.py` | Hai phương pháp gốc và ba biến thể distance weighting/KL |
| `reporting.py` | Bảng so sánh kết quả trong notebook |
| `visualization.py` | Accuracy/chi phí, Grad-CAM, entropy và t-SNE cho các model đã có |
| `diagnostics.py` | Đối chứng BatchNorm, loss và audit epsilon/weight từ model đang có |
| `artifacts.py` | Thu gom kết quả và tạo file ZIP để tải về |
| `phases.py` | Snapshot model đã đánh giá, manifest/SHA-256 và khôi phục Phase 1 cho Phase 2 |
| `constants.py` | Tên lớp và các hằng số chuẩn hóa |

## Chạy từ GitHub trên Kaggle

1. Push `boundary_unlearning/`, `experiments/`, `README.md` và `.gitignore` lên
   GitHub. `ref-code/` chỉ dùng để đối chiếu và được gitignore; code chạy không phụ
   thuộc thư mục này.
2. Import `experiments/boundary-unlearning-cifar-10-experiment.ipynb` vào Kaggle, gắn dataset
   `petitbonney/cifar10-image-recognition`, bật **Internet** và **GPU**.
3. Điền URL repo trong cell đầu:

   ```python
   GITHUB_REPO = "https://github.com/YOUR_USERNAME/boundary-unlearning.git"
   GITHUB_REF = "main"
   ```

4. Chạy **Run All**. Cell đầu clone code vào
   `/kaggle/working/boundary-unlearning`, checkout `GITHUB_REF` và thêm repo vào
   Python import path. Những cell sau import các module `.py`.

`GITHUB_REF` nhận branch, tag hoặc commit SHA. Để lặp lại cùng phiên bản code, dùng
commit SHA. URL HTTPS trong ví dụ dùng cho repo public. Với cập nhật nhỏ trong
phiên đang chạy, dùng cell **Small code update** để fetch/checkout và reload module,
rồi chạy lại cell định nghĩa runner cùng các cell phương pháp bị ảnh hưởng.
Không cần chạy lại cell train Original/Retrain khi cấu hình và dữ liệu không đổi.

Notebook vẫn giữ cấu hình mặc định: profile `paper`, train/retrain 30 epoch với LR
`0.1`, batch 64; unlearning 10 epoch với LR `1e-5`. Train dùng đủ 50.000 ảnh, đánh
giá bằng test CIFAR-10 chính thức có nhãn. Lần chạy đầu tải tập test qua torchvision.

Các đầu ra nằm trong `/kaggle/working/result/`:

```text
result/
├── checkpoints/              # Original, retrain và các model unlearning
├── figures/<run_tag>/        # PNG/PDF, cấu hình, dữ liệu dùng để vẽ
├── <run_tag>_metrics.csv
├── <run_tag>_config.json
├── <run_tag>_results.html     # Bảng so sánh kết quả
└── <run_tag>_unlearning_details.html # Chi phí, số bước, initializer và teacher
```

Chạy cell **Export results** ở cuối notebook để tạo
`/kaggle/working/result.zip` và hiện link tải. File ZIP chứa cả thư mục `result/`,
không chứa dataset tải về hoặc code repo.

Có thể chạy lại riêng cell export sau khi cập nhật code. Cell này dùng các kết quả
đã có trong phiên hiện tại, đồng thời sao chép checkpoint/metrics/config từ đường
dẫn cũ nếu cần; không train hoặc đánh giá model lại. Các lần chạy trước trong
`result/` cũng được giữ và đưa vào ZIP.

## Import trực tiếp

Sau khi chạy cell tải code, có thể dùng hàm từ notebook khác:

```python
from pathlib import Path
from boundary_unlearning.data import load_concatenated_npy_images
from boundary_unlearning.models import AllCNN
from boundary_unlearning.unlearning import run_boundary_shrink

images, backing_array = load_concatenated_npy_images(
    Path("/kaggle/input/datasets/petitbonney/cifar10-image-recognition/train.npy")
)

# original_model và forget_loader là model đã train và loader của forget set.
unlearned_model, stats = run_boundary_shrink(
    original_model,
    forget_loader,
    Path("/kaggle/working/boundary_shrink.pt"),
    device="cuda",
    epochs=10,
    learning_rate=1e-5,
    extra_exp=None,  # Hoặc "curv", "weight_assign".
)
```

## Chạy tại máy local

Cần Python, NumPy, PyTorch, torchvision, pandas, Jinja2 và IPython. Phần visualization
cần thêm Matplotlib; t-SNE cần scikit-learn hỗ trợ tham số `max_iter`. Mở notebook từ thư mục gốc repo, để
`GITHUB_REPO = ""`, rồi sửa `DATA_ROOT`, `INPUT_ROOT` và `OUTPUT_ROOT` theo máy.
Các module chỉ định nghĩa hàm/class; import không tải dữ liệu hoặc tự train model.

## Thí nghiệm CIFAR-100

Import `experiments/boundary-unlearning-cifar-100-experiment.ipynb` vào Kaggle và gắn dataset
[alincijov/cifar-100](https://www.kaggle.com/datasets/alincijov/cifar-100) bằng
**Add Input**. Dataset cung cấp `train`, `test`, `meta` dưới dạng pickle; notebook
đọc trực tiếp cả train và test có nhãn, không tải CIFAR từ nguồn khác.

`DATA_ROOT` mặc định là `/kaggle/input/datasets/alincijov/cifar-100`, có hỗ trợ đường
dẫn mount cũ `/kaggle/input/cifar-100` và thư mục con `cifar-100-python/`.

- Dùng `fine_labels` cho **100 lớp**, tên lớp lấy từ `meta`.
- AllCNN có 100 đầu ra; Boundary Expanding mở thành 101 rồi bỏ lớp phụ.
- Mặc định `ATTACK_NORMALIZATION = "matched"`, dữ liệu và FGSM dùng cùng mean/std
  `(0.5, 0.5, 0.5)`. Notebook CIFAR-10 tiếp tục giữ chế độ `repo_exact`.
- Giữ cấu hình ban đầu: 30 epoch train/retrain, LR `0.1`, batch 64; 10 epoch
  unlearning, LR `1e-5`, BN update cho Shrink thuần. Đây là cấu hình chuyển từ
  CIFAR-10, không phải tuyên bố tái lập toàn bộ paper trên CIFAR-100.
  Một lớp có 500 ảnh train, tương ứng khoảng 80 bước unlearning.
- Run tag bắt đầu bằng `cifar100_`; config ghi dataset, loại nhãn, danh sách lớp
  và chuẩn hóa. Checkpoint, CSV/JSON/HTML nằm trong `result/`; cell cuối tạo
  `result.zip` giống notebook CIFAR-10.

**Chạy Phase 1 trong phiên mới**, kiểm tra bảng 4 model rồi tải `result.zip`.
Ngoài các checkpoint tốt nhất được lưu khi train, gói có:

```text
result/phase1/<run_tag>/
├── phase1_manifest.json      # Run tag, SHA-256, chính sách snapshot
├── config.json              # Cấu hình, tên lớp, fingerprint pixels/labels
├── rows.json / metrics.csv / results.html
└── checkpoints/             # Đúng 4 state_dict đã dùng trong bảng đánh giá
    ├── original_evaluated.pt
    ├── retrain_evaluated.pt
    ├── boundary_shrink_evaluated.pt
    └── boundary_expanding_evaluated.pt
```

Sau đó import notebook **Phase 2**, gắn lại CIFAR-100 và gắn Output đã lưu của
notebook Phase 1, hoặc upload `result.zip` thành Kaggle Dataset rồi **Add Input**.
`PHASE1_SOURCE=None` tự tìm một gói duy nhất; nếu có nhiều gói, điền đường dẫn ZIP,
thư mục chứa một manifest hoặc chính file `phase1_manifest.json` cần sử dụng.
Phase 2 kiểm tra SHA-256, khôi phục cả model Expanding đã prune, xác nhận dữ liệu
và dùng lại cấu hình Phase 1. Thiếu/sai gói sẽ báo lỗi, không tự train baseline.
Các bản tham chiếu được sao chép vào `result/phase1_source/` để export Phase 2 tự đủ.

Bundle lưu **model đang được đánh giá**, không tự thay bằng checkpoint tốt nhất
ở epoch khác. Không cần trạng thái optimizer: các biến thể Phase 2 khởi tạo
optimizer SGD mới như flow hiện tại. Hai phase dùng run tag riêng chứa `phase1`/`phase2`.

Kiểm tra loader, bàn giao giữa hai phase và smoke workflow bằng dữ liệu giả lập:

```bash
python -m unittest discover -s tests -v
```

## Ba biến thể bổ sung

Notebook CIFAR-10 tiếp tục chạy **5 biến thể unlearning + Original và Retrain**.
CIFAR-100 **Phase 2** nạp 4 model Phase 1, rồi chỉ train ba biến thể:
distance weighting, distance weighting + KL từ Original và distance weighting + KL
từ Boundary Shrink. Mỗi biến thể mới có cell riêng; cell cuối đánh giá đủ 7 model
và export cùng một CSV/JSON/ZIP.

Biến thể KL từ Boundary Shrink khởi tạo từ model distance weighting, nên student
được cập nhật tổng cộng **20 epoch**, gồm hai giai đoạn 10 epoch. Bảng chi tiết
tách thời gian giai đoạn cuối khỏi tổng thời gian tạo cả model đầu vào và teacher.
Hai biến thể đầu khởi tạo riêng từ Original. Cấu hình train/retrain vẫn giữ nguyên.

Distance search dùng chuẩn hóa đầu vào (`matched`) đúng theo notebook nguồn;
Boundary Shrink gốc của CIFAR-10 vẫn dùng `repo_exact`. Đây cũng là một khác biệt
khi diễn giải kết quả, bên cạnh loss và pseudo-label. `EXTRA_EXP="weight_assign"`
cũ là probability weighting của tác giả, không phải distance weighting mới.

Trên CIFAR-10, tên checkpoint Original/Retrain tiếp tục dùng hash baseline cũ; hash
kết quả mới chứa thêm epsilon step và cấu hình KL. Khi tái sử dụng checkpoint,
bật `REUSE_CHECKPOINTS=True` và kiểm tra đường dẫn được in ra. Checkpoint baseline
lưu model tốt nhất, khác với model cuối epoch đang nằm trong phiên train trước đó.

## Visualization

Các notebook có phần visualization trước cell export: CIFAR-100 Phase 1 dùng
4 model; Phase 2 và CIFAR-10 dùng 7 model.
Chạy **Visualization setup** rồi chọn cell vẽ: accuracy/chi phí, Grad-CAM,
entropy hoặc t-SNE. Các cell dùng model đang có, không train hoặc nạp checkpoint lại.
Sau khi cập nhật riêng module visualization, chỉ cần reload, chạy lại setup và
cell vẽ bị ảnh hưởng, rồi export ZIP. Nếu model thay đổi, chạy lại evaluation trước.

- Accuracy: giá trị test retain/forget và chênh lệch so với Retrain theo điểm phần trăm.
- Chi phí: thời gian giai đoạn/tổng chi phí và số bước stage/student/prerequisite.
- Grad-CAM: cùng ảnh, cùng lớp mục tiêu; chia tối đa 4 model/trang, hiện prediction
  confidence và `P(forget)` riêng. Không cập nhật trọng số hay gradient tham số.
- Entropy: ECDF của log Shannon entropy trên các mẫu train retain, train forget
  và test, được chọn cố định và dùng chung giữa mọi model.
- t-SNE: cùng mẫu test, tối đa 1.500 ảnh và 120 ảnh/lớp. CIFAR-100 gom các lớp
  retain bằng màu xám, lớp forget là vòng đỏ. Có thể bỏ qua cell này để export nhanh.

Hình lưu ở `result/figures/<run_tag>/` dưới dạng PNG/PDF, kèm JSON metadata và NPZ
chứa CAM/probabilities, entropy hoặc features/embedding để vẽ lại. Cell export cuối
tự đưa toàn bộ thư mục hình vào ZIP. Xem [hướng dẫn chi tiết](experiments/visualization.md).

## Kiểm tra BatchNorm trên CIFAR-100

Trên CIFAR-100, chạy ablation trong **Phase 2** sau khi nạp bundle. Bật
`RUN_BN_ABLATION=True` ở cell **Optional: paired BatchNorm ablation**, rồi chạy riêng
cell đó. Mặc định chỉ chạy Boundary Shrink với BN update/freeze, giữ nguyên epoch/LR
và batch size, lưu kết quả trong `result/bn_ablation/` và tạo ZIP ngay.

Nếu notebook Kaggle chưa có cell mới, có thể gọi trực tiếp sau khi fetch code:

```python
exec((Path(CODE_ROOT) / "experiments" / "run-bn-ablation.py").read_text())
```

Mặc định của các hàm unlearning vẫn là `bn_stats_mode="update"`. Cặp đối chứng
ghi riêng checkpoint và metrics; các kết quả 7 model hiện tại được giữ nguyên.
Xem [thiết kế, log và cách đọc kết quả](experiments/04-cifar100-batchnorm-ablation.md).

## Đối chứng riêng distance weighting

Trong phiên **Phase 2** đã nạp các model, bật cell weighting ablation hoặc chạy:

```python
from pathlib import Path
exec((Path(CODE_ROOT) / "experiments" / "run-weighting-ablation.py").read_text())
```

Chạy hai student với cùng distance search, seed, cấu hình unlearning và BN freeze,
so sánh distance weights với weights luôn bằng 1, không dùng retain KL.
Entry tự kiểm tra pseudo-label/epsilon của cặp, đánh giá và export ZIP; không cần
Run All. Đầu ra riêng ở `result/weighting_ablation/`, biến `weighting_ablation`.
Xem [thiết kế và cách đọc chênh lệch](experiments/05-cifar100-weighting-ablation.md).
