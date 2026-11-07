# Báo cáo Boundary Unlearning

Nguồn chính là `main_1.tex`, theo định dạng IEEE hai cột. Bản đọc Markdown tại
`BaoCao/BaoCao_Boundary_Unlearning.md` được đồng bộ từ nguồn LaTeX.

## Phạm vi

- So sánh Original, Retrain, Boundary Shrink và Boundary Expanding trên CIFAR-10
  và CIFAR-100 trong cùng nhóm bảng kết quả cơ sở.
- Khảo sát riêng DW, DW+KL-O và DW+KL-S trên CIFAR-10.
- Không đưa thí nghiệm BN freeze vào báo cáo.
- Grad-CAM, entropy và t-SNE CIFAR-10 nằm trong phụ lục. Chúng chỉ bổ trợ mô tả;
  báo cáo không tuyên bố đã kiểm chứng MIA hoặc bảo đảm xóa ảnh hưởng dữ liệu.

## Nguồn số liệu

- CIFAR-10: bảng accuracy và bảng chi phí trong output lưu của
  `../experiments/boundary-unlearning-cifar-10-experiment.ipynb`, lượt
  `paper_forget4_07782060`. Thời gian DW+KL-S gồm DW và Shrink chuẩn bị teacher:
  226,97 giây; 1.580 bước student và 2.370 bước toàn quy trình.
- CIFAR-100: `../results/boundary-unlearning-cifar-100-result/phase1/`
  `cifar100_phase1_paper_forget4_4905e906/metrics.csv` cùng `config.json`.
  Thời gian Retrain 424,53 giây, Shrink 3,18 giây, Expanding 1,44 giây.
  Không dùng lại thời gian của các lượt cũ trong bản thảo trước.

Đây là các kết quả đã lưu, không phải một lần huấn luyện mới khi sửa báo cáo.
Các điểm cần công khai về seed, chuẩn hóa attack, checkpoint, validation và phạm vi
tính thời gian được ghi trong phần thiết lập và giới hạn của paper.

## Biên dịch

Chạy từ thư mục này, với TeX Live có IEEEtran, vntex, amsmath, booktabs và hyperref:

```bash
mkdir -p build
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -pdflatex='pdflatex -no-shell-escape %O %S' -outdir=build main_1.tex
```

PDF nằm tại `build/main_1.pdf`. Phần chính giữ hai cột; phụ lục dùng một cột để
các hình nhiều panel đọc được. Bản IEEE template và PDF bài CVPR gốc là tài liệu
tham khảo, không phải PDF báo cáo đã biên dịch. Khối tên tác giả hiện giữ thông tin
nhóm có sẵn; cần điền tên thành viên thật theo yêu cầu môn học trước khi nộp.
