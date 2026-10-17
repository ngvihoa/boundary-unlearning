# Boundary Unlearning trên Kaggle

`experiment.ipynb` chứa cấu hình và các bước chạy thí nghiệm. Các hàm và class nằm
trong package `boundary_unlearning/`, có thể import độc lập sau khi clone repo.

| File | Nội dung |
| --- | --- |
| `data.py` | Đọc NPY/CSV, xử lý nhãn, dataset và DataLoader |
| `models.py` | AllCNN và khởi tạo classifier |
| `attacks.py` | FGSM với chuẩn hóa của tác giả |
| `enhanced.py` | Curvature regularization và weight assignment |
| `training.py` | Train/retrain, checkpoint và đánh giá accuracy |
| `unlearning.py` | Boundary Shrink và Boundary Expanding |
| `reporting.py` | Bảng so sánh kết quả trong notebook |
| `constants.py` | Tên lớp và các hằng số chuẩn hóa |

## Chạy từ GitHub trên Kaggle

1. Push `boundary_unlearning/`, `experiment.ipynb`, `README.md` và `.gitignore` lên
   GitHub. `ref-code/` chỉ dùng để đối chiếu và được gitignore; code chạy không phụ
   thuộc thư mục này.
2. Import `experiment.ipynb` vào Kaggle, gắn dataset
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

Checkpoint và các file metrics/config nằm trong `/kaggle/working`.

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
