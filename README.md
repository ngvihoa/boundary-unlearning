# Boundary Unlearning trên Kaggle

[Notebook CIFAR-10](experiments/boundary-unlearning-cifar-10-experiment.ipynb)
và [notebook CIFAR-100](experiments/experiment-cifar-100.ipynb) nằm trong `experiments/`.
Các hàm và class nằm trong package `boundary_unlearning/`, có thể import độc lập
sau khi clone repo.

Thông tin từng thí nghiệm:

- [Thí nghiệm 01 — CIFAR-10, quên lớp deer](experiments/01-cifar-10.md): cấu hình
  và kết quả đã lưu từ lần chạy Kaggle ngày 17/10/2026.
- [Thí nghiệm 02 — CIFAR-100, quên lớp beaver](experiments/02-cifar-100.md): cấu
  hình dự kiến và trạng thái kiểm chứng, chưa có kết quả train đầy đủ.

| File | Nội dung |
| --- | --- |
| `data.py` | Đọc NPY/CSV, xử lý nhãn, dataset và DataLoader |
| `cifar100.py` | Đọc pickle train/test/meta và nhãn fine của CIFAR-100 |
| `models.py` | AllCNN và khởi tạo classifier |
| `attacks.py` | FGSM với chuẩn hóa của tác giả |
| `enhanced.py` | Curvature regularization và weight assignment |
| `training.py` | Train/retrain, checkpoint và đánh giá accuracy |
| `unlearning.py` | Boundary Shrink và Boundary Expanding |
| `reporting.py` | Bảng so sánh kết quả trong notebook |
| `artifacts.py` | Thu gom kết quả và tạo file ZIP để tải về |
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
commit SHA. Khi cập nhật code trên branch hoặc đổi ref, **restart kernel rồi Run All**
để Python nạp lại module. URL HTTPS trong ví dụ dùng cho repo public.

Notebook vẫn giữ cấu hình mặc định: profile `paper`, train/retrain 30 epoch với LR
`0.1`, batch 64; unlearning 10 epoch với LR `1e-5`. Train dùng đủ 50.000 ảnh, đánh
giá bằng test CIFAR-10 chính thức có nhãn. Lần chạy đầu tải tập test qua torchvision.

Các đầu ra nằm trong `/kaggle/working/result/`:

```text
result/
├── checkpoints/              # Original, retrain và các model unlearning
├── <run_tag>_metrics.csv
├── <run_tag>_config.json
└── <run_tag>_results.html     # Bảng so sánh kết quả
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

Cần Python, NumPy, PyTorch, torchvision, pandas, Jinja2 và IPython. Mở notebook từ thư mục gốc repo, để
`GITHUB_REPO = ""`, rồi sửa `DATA_ROOT`, `INPUT_ROOT` và `OUTPUT_ROOT` theo máy.
Các module chỉ định nghĩa hàm/class; import không tải dữ liệu hoặc tự train model.

## Thí nghiệm CIFAR-100

Import `experiments/experiment-cifar-100.ipynb` vào Kaggle và gắn dataset
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
  unlearning, LR `1e-5`. Đây là cấu hình khởi đầu, chưa phải kết quả đã kiểm chứng
  cho CIFAR-100. Một lớp có 500 ảnh train, tương ứng khoảng 80 bước unlearning.
- Run tag bắt đầu bằng `cifar100_`; config ghi dataset, loại nhãn, danh sách lớp
  và chuẩn hóa. Checkpoint, CSV/JSON/HTML nằm trong `result/`; cell cuối tạo
  `result.zip` giống notebook CIFAR-10.

Kiểm tra loader và nhãn bằng dữ liệu giả lập tại local:

```bash
python -m unittest discover -s tests -v
```
