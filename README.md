# Boundary Unlearning trên CIFAR-10 và CIFAR-100

Repository thực nghiệm **Machine Unlearning ở cấp độ lớp**: điều chỉnh một mô hình
đã huấn luyện để quên một lớp dữ liệu, đồng thời giữ khả năng nhận diện các lớp còn lại.
Dự án triển khai Boundary Shrink và Boundary Expanding dựa trên công trình
*Boundary Unlearning: Rapid Forgetting of Deep Networks via Shifting the Decision
Boundary* của Chen và cộng sự (CVPR 2023), cùng các biến thể distance weighting
và chưng cất tri thức bằng KL divergence.

Code được tổ chức thành các module Python dùng chung, notebook chạy trên Kaggle,
kết quả thực nghiệm và báo cáo LaTeX. Phạm vi là đánh giá những phương pháp được
triển khai trên CIFAR-10 rồi chuyển quy trình sang CIFAR-100; đây không phải bản
tái lập toàn bộ thí nghiệm và baseline của bài báo.

## Mục tiêu thực nghiệm

Các phương pháp được so sánh theo ba khía cạnh:

- **Khả năng quên:** accuracy trên dữ liệu thuộc lớp cần quên (*forget set*).
- **Khả năng giữ kiến thức:** accuracy trên các lớp còn lại (*retain set*).
- **Chi phí:** thời gian unlearning và số bước cập nhật, tính cả các giai đoạn phụ thuộc.

**Original** là mô hình được huấn luyện trên toàn bộ dữ liệu. **Retrain** là mô hình
được huấn luyện lại từ đầu sau khi loại bỏ lớp cần quên, dùng làm mốc tham chiếu.
Các kết quả được đọc tương quan với hai mốc này; forget accuracy thấp chưa đủ để
kết luận đã loại bỏ hoàn toàn ảnh hưởng của dữ liệu.

## Các phương pháp

| Phương pháp | Ý tưởng |
| --- | --- |
| Boundary Shrink | Dùng FGSM tìm nhãn giả từ mô hình gốc, rồi cập nhật trên ảnh forget để dịch chuyển ranh giới quyết định. |
| Boundary Expanding | Thêm một lớp phụ, chuyển ảnh forget sang lớp đó rồi loại bỏ đầu ra phụ. |
| Distance weighting (DW) | Tìm ngưỡng vượt biên để tạo nhãn giả và gán trọng số cho từng mẫu forget. |
| DW + KL từ Original | Bổ sung loss KL trên dữ liệu retain, dùng Original làm teacher. |
| DW + KL từ Boundary Shrink | Tiếp tục cập nhật mô hình DW, dùng Boundary Shrink làm teacher trên dữ liệu retain. |

Biến thể cuối có hai giai đoạn cập nhật student, nên ngân sách lớn hơn hai biến thể
DW còn lại. Các khảo sát BatchNorm, trọng số đồng đều và số epoch được tổ chức
thành thí nghiệm phụ, tách khỏi bảng so sánh chính.

## Notebook và dữ liệu

| Notebook | Phạm vi | Lớp cần quên mặc định |
| --- | --- | --- |
| [CIFAR-10](experiments/boundary-unlearning-cifar-10-experiment.ipynb) | Original, Retrain và năm phương pháp unlearning trong cùng notebook. | `4` — deer |
| [CIFAR-100 · Phase 1](experiments/boundary-unlearning-cifar-100-experiment.ipynb) | Original, Retrain, Shrink và Expanding; lưu model đã đánh giá để dùng ở Phase 2. | `4` — beaver |
| [CIFAR-100 · Phase 2](experiments/boundary-unlearning-cifar-100-phase2.ipynb) | Nạp bốn model Phase 1, chạy ba biến thể DW/KL và các ablation tùy chọn. | Theo bundle Phase 1 |

- **CIFAR-10:** train từ dataset Kaggle
  [petitbonney/cifar10-image-recognition](https://www.kaggle.com/datasets/petitbonney/cifar10-image-recognition);
  đánh giá bằng test CIFAR-10 chính thức có nhãn qua `torchvision`.
- **CIFAR-100:** đọc trực tiếp `train`, `test`, `meta` của dataset
  [alincijov/cifar-100](https://www.kaggle.com/datasets/alincijov/cifar-100), sử dụng
  `fine_labels` cho 100 lớp.

Cấu hình mặc định: AllCNN có BatchNorm, seed `2022`, batch size `64`;
Original/Retrain được huấn luyện 30 epoch với LR `0.1`; mỗi giai đoạn unlearning
chạy 10 epoch với LR `1e-5`. CIFAR-10 giữ attack normalization `repo_exact`,
CIFAR-100 dùng `matched`. Cùng số epoch nhưng số bước cập nhật khác nhau do kích
thước forget set: 5.000 ảnh trên CIFAR-10 và 500 ảnh trên CIFAR-100.

## Bắt đầu trên Kaggle

1. Import notebook tương ứng từ `experiments/` vào Kaggle.
2. Gắn dataset phù hợp, bật **GPU** và **Internet**.
3. Kiểm tra `GITHUB_REPO` và `GITHUB_REF` trong cell đầu; dùng commit SHA nếu muốn
   cố định phiên bản code. Cell này clone hoặc cập nhật repo và thiết lập import.
4. Chạy các cell cấu hình, dữ liệu và phương pháp theo thứ tự. Với CIFAR-100,
   hoàn thành Phase 1 trước khi chạy Phase 2, hoặc dùng bundle Phase 1 đã lưu.
5. Chạy các cell đánh giá, visualization và **Export results** để tải `result.zip`.

Phase 2 mặc định dùng bundle 30 epoch train / 10 epoch unlearning đã có trong repo:

```python
PHASE1_SOURCE = Path(CODE_ROOT) / "results" / "boundary-unlearning-cifar-100-result"
```

Không cần upload kết quả riêng lên Kaggle khi nhánh/commit đã clone chứa bundle này.
Có thể đổi `PHASE1_SOURCE` thành đường dẫn ZIP, thư mục hoặc manifest của Output /
Dataset được gắn qua **Add Input**. Đặt `None` để tự tìm một bundle duy nhất trong
`/kaggle/input`. Phase 2 kiểm tra SHA-256 và tính nhất quán dữ liệu, rồi khôi phục
đúng bốn model đã đánh giá; không huấn luyện lại baseline.

Với cập nhật code nhỏ trong phiên đang chạy, dùng cell **Small code update**, rồi
chạy lại các cell bị ảnh hưởng. Không cần train lại Original/Retrain khi dữ liệu,
cấu hình và các model nền vẫn được giữ nguyên.

## Kết quả và trực quan hóa

Notebook tạo bảng accuracy train/test theo retain/forget, bảng chi phí, biểu đồ
accuracy và thời gian, Grad-CAM, phân phối entropy và t-SNE. Các hình dùng chung
mẫu giữa các model để hỗ trợ đối chiếu.

```text
/kaggle/working/
├── result/
│   ├── checkpoints/         # Trọng số các model
│   ├── figures/<run_tag>/   # PNG/PDF, metadata JSON và dữ liệu NPZ khi có
│   ├── <run_tag>_metrics.csv
│   ├── <run_tag>_config.json
│   └── ...                 # Bảng HTML và bundle bàn giao giữa hai phase
└── result.zip              # Gói tải về, nằm ngoài result/
```

Các kết quả đã lưu nằm trong [results/](results/), gồm gói CIFAR-10, bundle
[CIFAR-100 Phase 1](results/boundary-unlearning-cifar-100-result/) và
[kết quả Phase 2](results/boundary-unlearning-cifar-100-result-p2/).
Grad-CAM, entropy và t-SNE là các công cụ mô tả hành vi model, không thay thế
đánh giá privacy bằng membership inference attack.

## Cấu trúc repository

| Thư mục | Nội dung |
| --- | --- |
| [boundary_unlearning/](boundary_unlearning/) | Module dữ liệu, AllCNN, FGSM, training, unlearning, đánh giá, visualization và export. |
| [experiments/](experiments/) | Notebook Kaggle, script chạy thí nghiệm phụ và ghi chú thiết lập. |
| [results/](results/) | Kết quả, cấu hình, hình và checkpoint đã lưu từ các lượt chạy. |
| [tests/](tests/) | Kiểm tra loader, phương pháp, bàn giao hai phase, ablation và visualization. |
| [latex-report/](latex-report/) | Báo cáo tiếng Việt/tiếng Anh, nguồn LaTeX và PDF. |
| [origin-reference/](origin-reference/) | Bài báo Boundary Unlearning gốc để tham khảo. |

## Tài liệu thực nghiệm và báo cáo

- [CIFAR-10: dữ liệu, cấu hình và quy trình](experiments/01-cifar-10.md).
- [CIFAR-100 Phase 1: các phương pháp gốc](experiments/02-cifar-100.md).
- [Ba biến thể distance weighting và KL](experiments/03-distance-weighted-variants.md).
- [Đối chứng BatchNorm update/freeze](experiments/04-cifar100-batchnorm-ablation.md).
- [Đối chứng distance weights và weights = 1](experiments/05-cifar100-weighting-ablation.md).
- [So sánh số epoch unlearning](experiments/06-cifar100-phase1-epoch-comparison.md).
- [Hướng dẫn visualization](experiments/visualization.md).
- Báo cáo: [tiếng Việt](latex-report/main.pdf) · [English](latex-report/main_en.pdf).

Ghi chú từng thí nghiệm có thể mô tả một lượt chạy cũ; đối chiếu run tag và
`config.json` / `metrics.csv` trong gói kết quả khi sử dụng số liệu.

## Chạy và kiểm tra tại local

Môi trường sử dụng Python, PyTorch, torchvision, NumPy, pandas, Jinja2, IPython,
Matplotlib, scikit-learn và threadpoolctl. Khi mở notebook từ thư mục gốc repo,
đặt `GITHUB_REPO = ""` và điều chỉnh các đường dẫn input/output cho máy local.

```bash
python -m unittest discover -s tests -v
```

Các module trong `boundary_unlearning/` có thể import độc lập; import không tự tải
dữ liệu hoặc huấn luyện model.
