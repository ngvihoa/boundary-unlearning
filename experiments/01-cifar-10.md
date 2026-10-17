# Thí nghiệm 01 — Boundary Unlearning trên CIFAR-10

## Nguồn và trạng thái

- Notebook: [boundary-unlearning-cifar-10-experiment.ipynb](boundary-unlearning-cifar-10-experiment.ipynb).
- Trạng thái: **có output chạy trên Kaggle ngày 17/10/2026**.
- Run tag: `paper_forget4_7b3dd1f4`.
- Commit được cell bootstrap in ra: `e8d90cdf1eb3103838b99d0d18ac5cfd84c6604c`.
- Ghi chú được lập từ source và output đã lưu trong notebook; không chạy lại thí nghiệm.
  Notebook không ghi riêng SHA của những lần cập nhật module giữa phiên.

## Mục tiêu và dữ liệu

So sánh Original, Retrain, Boundary Shrink và Boundary Expanding khi yêu cầu quên
toàn bộ lớp **`deer` (hươu), `FORGET_CLASS = 4`**. Chỉ số lớp bắt đầu từ 0.

Nguồn train là [petitbonney/cifar10-image-recognition](https://www.kaggle.com/datasets/petitbonney/cifar10-image-recognition).
Ảnh được đọc từ các record NPY nối tiếp, giữ đúng layout `(32, 32, 3)` trong header;
nhãn lấy từ `trainLabels.csv`. Tập đánh giá là test CIFAR-10 chính thức có nhãn,
được tải qua `torchvision.datasets.CIFAR10(train=False)`.

| Tập | Tất cả | Forget: deer | Retain: 9 lớp còn lại |
| --- | ---: | ---: | ---: |
| Train | 50.000 | 5.000 | 45.000 |
| Test chính thức | 10.000 | 1.000 | 9.000 |

Kaggle competition còn có 300.000 ảnh test không có nhãn thật. Notebook tạo loader
inference cho chúng nhưng **không dùng để tính accuracy**; nhãn ví dụ trong
`sampleSubmission.csv` không phải ground truth. Không tách validation từ train.

## Cấu hình của lần chạy

| Tham số | Giá trị |
| --- | --- |
| Profile | `paper` |
| Seed | `2022` |
| Model | AllCNN, `filters_percentage=0.5`, BatchNorm, không dropout |
| Đầu ra / số tham số | 10 lớp / 406.810 tham số |
| Batch size / workers | `64` / `2` |
| Train Original và Retrain | Mỗi model 30 epoch, SGD, LR `0.1`, momentum `0.9` |
| Loss | Cross-entropy |
| Unlearning | 10 epoch, SGD, LR `1e-5`, momentum `0.9` |
| Giới hạn batch | Không giới hạn |
| FGSM bound | `0.1`, không random start |
| Enhanced Boundary Shrink | `EXTRA_EXP = None` |
| Tái sử dụng checkpoint | `REUSE_CHECKPOINTS = False` |

Chuẩn hóa ảnh đầu vào: mean/std đều `(0.5, 0.5, 0.5)`.
Attack dùng `ATTACK_NORMALIZATION = "repo_exact"`, giữ hằng số từ code tác giả:

```python
mean = (0.4914, 0.4822, 0.2265)
std = (0.2023, 0.1994, 0.2010)
```

Như vậy, chuẩn hóa attack khác chuẩn hóa ảnh đầu vào trong thí nghiệm này.
Tên profile `paper` là tên cấu hình trong repo, không tự xác nhận tái lập đầy đủ
thiết lập hoặc kết quả của bài báo.

## Quy trình

1. Train Original trên đủ 50.000 ảnh, giữ classifier 10 lớp.
2. Train Retrain từ đầu trên 45.000 ảnh retain, vẫn giữ classifier 10 lớp.
3. Boundary Shrink: copy Original làm teacher và student; tạo ảnh FGSM từ forget
   set, lấy pseudo-label của teacher rồi cập nhật student trên ảnh gốc.
4. Boundary Expanding: copy Original, mở classifier từ 10 thành 11 lớp; train
   forget set với nhãn phụ `10`, sau đó bỏ đầu ra phụ để còn 10 lớp.
5. Đánh giá accuracy trên train/test, chia theo all/retain/forget khi có tương ứng.

Mỗi phương pháp unlearning thực hiện **790 bước cập nhật**:
`10 × ceil(5.000 / 64)`. Boundary Shrink ghi nhận tổng cộng 50.000 lượt mẫu qua
10 epoch; đây là lượt xử lý lặp lại, không phải số ảnh forget khác nhau.

## Kết quả lưu trong notebook

Accuracy dưới đây được chép từ bảng output, đơn vị **%**. Thời gian lấy từ log của
từng giai đoạn, đơn vị **giây**; không bao gồm thời gian tải dataset.

| Model | Train retain | Train forget | Test all | Test retain | Test forget | Train (s) | Unlearn (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 99.43 | 99.64 | 72.91 | 73.46 | 68.00 | 443.00 | — |
| Retrain | 99.70 | 0.00 | 68.05 | 75.61 | 0.00 | 400.55 | — |
| Boundary Shrink | 94.86 | 11.54 | 56.09 | 62.07 | 2.30 | — | 22.87 |
| Boundary Expanding | 94.95 | 11.34 | 56.53 | 62.57 | 2.20 | — | 12.34 |

FGSM crossing rate của Boundary Shrink: **89.68%**. Đây là tỷ lệ pseudo-label
khác nhãn gốc trên các lượt xử lý forget set, không phải accuracy đánh giá model.

Trong lần chạy này, cả hai phương pháp giảm mạnh test accuracy của lớp deer,
nhưng test retain accuracy cũng thấp hơn Retrain: 62.07% / 62.57% so với 75.61%.
Đây là kết quả của một lần chạy với seed 2022, chưa phải trung bình nhiều seed.

## Checkpoint, export và cách đọc kết quả

Original/retrain lưu **checkpoint tốt nhất theo test accuracy**, còn hàm train
trả về **model ở epoch cuối**. Với `REUSE_CHECKPOINTS = False`, bảng trên và model
đầu vào của unlearning dùng model cuối, không tự nạp lại checkpoint tốt nhất.

| Model | Accuracy tốt nhất của checkpoint | Accuracy model cuối ở cùng tập |
| --- | ---: | ---: |
| Original — test all | 74.65% | 72.91% |
| Retrain — test retain | 76.29% | 75.61% |

Các file checkpoint là `state_dict`, cần dựng đúng kiến trúc khi nạp lại:

```text
result/
├── checkpoints/
│   ├── original_paper_forget4_7b3dd1f4.pt
│   ├── retrain_paper_forget4_7b3dd1f4.pt
│   ├── boundary_shrink_paper_forget4_7b3dd1f4.pt
│   ├── boundary_expanding_paper_forget4_7b3dd1f4.pt
│   └── boundary_expanding_paper_forget4_7b3dd1f4_widened.pt
├── paper_forget4_7b3dd1f4_metrics.csv
├── paper_forget4_7b3dd1f4_config.json
└── paper_forget4_7b3dd1f4_results.html
```

Bản notebook lưu lần chạy này còn ghi checkpoint ở `/kaggle/working/checkpoints/`
và CSV/JSON ở `/kaggle/working/`. Cell export sao chép chúng vào `result/` rồi tạo
`/kaggle/working/result.zip`; output ghi kích thước **7.26 MiB**. Có thể chạy riêng
cell export bằng kết quả sẵn có, không cần train hoặc đánh giá lại.

Môi trường được output ghi nhận: Tesla T4, Python `3.13.15`, PyTorch
`2.11.0+cu128`, torchvision `0.26.0+cu128`, NumPy `2.1.3`. Repo hiện có notebook
và output hiển thị; chưa có các file checkpoint/CSV/JSON của lần chạy để kiểm tra
trực tiếp nội dung nhị phân hoặc số liệu chưa làm tròn.
