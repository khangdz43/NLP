# Vietnamese Named Entity Recognition (NER)

Dự án xây dựng hệ thống nhận diện thực thể có tên trong văn bản tiếng Việt. Bài toán được xử lý theo hướng **sequence labeling**: mỗi token trong câu được gán một nhãn BIO để xác định token đó có thuộc thực thể hay không.

Hai hướng mô hình trong dự án:

- **CRF**: mô hình xác suất cho gán nhãn chuỗi, dùng feature thủ công.
- **BiLSTM / BiLSTM-CRF**: mô hình học sâu đọc ngữ cảnh hai chiều, được huấn luyện trong notebook.

## 1. Nhãn thực thể

| Nhãn | Ý nghĩa | Ví dụ |
| --- | --- | --- |
| `PER` | Người | Nguyễn Văn A, Donald Trump |
| `LOC` | Địa điểm | Hà Nội, Washington, Việt Nam |
| `ORG` | Tổ chức | Bộ Công an, Google, Điện Kremlin |
| `MISC` | Thực thể khác | Các thực thể ngoài 3 nhóm chính nếu có trong dữ liệu |

## 2. Chuẩn BIO

Dữ liệu được biểu diễn bằng chuẩn BIO:

| Tag | Ý nghĩa |
| --- | --- |
| `B-*` | Token bắt đầu một thực thể |
| `I-*` | Token tiếp theo bên trong thực thể |
| `O` | Token không thuộc thực thể |

Ví dụ:

```text
Input : Ông Donald Trump gặp lãnh đạo Mỹ tại Washington
Output: O   B-PER  I-PER O   O        B-LOC O   B-LOC
```

## 3. Cấu trúc dự án

```text
.
├── README.md
├── CRF/
│   ├── crf_ner_model.ipynb
│   ├── retrain_crf_improved.py
│   ├── predict_crf.py
│   ├── requirements.txt
│   ├── pipeline.md
│   ├── data/
│   │   ├── vlsp_train_raw.csv
│   │   └── vlsp_valid_raw.csv
│   └── models/
│       └── crf_vlsp2016_notebook.joblib
└── BILSTM/
    ├── Vietnamese_NER_Project.ipynb
    ├── BiLSTM_Pure_NER.ipynb
    ├── test_module.py
    ├── data/
    │   ├── merged_train.csv
    │   ├── merged_valid.csv
    │   ├── vlsp_train_raw.csv
    │   ├── vlsp_valid_raw.csv
    │   ├── vlsp_Covid19_train_raw.csv
    │   └── vlsp_Covid19_validation_raw.csv
    └── models/
        ├── best_bilstm_pure.pt
        ├── best_model_notebook.pt
        ├── last_checkpoint.pt
        └── training_history*.json/png
```

## 4. Dữ liệu

Dự án dùng dữ liệu NER tiếng Việt đã được tách từ và gán nhãn:

- VLSP 2016 NER.
- Bộ dữ liệu Covid19 NER trong thư mục `BILSTM/data/`.
- File `BILSTM/data/merge_vlsp_and_Covid19.py` dùng để gộp dữ liệu VLSP và Covid19 thành `merged_train.csv` và `merged_valid.csv`.

Định dạng chính trong các file CSV:

- `tokens`: danh sách token của câu.
- `ner_tags`: danh sách nhãn tương ứng với từng token.

Label mapping được dùng trong code:

| ID | Label |
| --- | --- |
| 0 | `O` |
| 1 | `B-PER` |
| 2 | `I-PER` |
| 3 | `B-ORG` |
| 4 | `I-ORG` |
| 5 | `B-LOC` |
| 6 | `I-LOC` |
| 7 | `B-MISC` |
| 8 | `I-MISC` |

## 5. Mô hình CRF

CRF nằm trong thư mục `CRF/`. Mô hình này dự đoán cả chuỗi nhãn BIO thay vì dự đoán từng token độc lập. Khi huấn luyện, mỗi token được chuyển thành một tập feature, sau đó CRF học quan hệ giữa feature và nhãn, đồng thời học luật chuyển trạng thái giữa các nhãn.

Feature chính:

- Token viết thường.
- Token có viết hoa chữ cái đầu hay không.
- Token có viết hoa toàn bộ hay không.
- Token có phải số hay không.
- Độ dài token.
- Prefix, suffix.
- Word shape.
- Token trước và token sau.
- Bigram với token trước/sau.
- Các prefix gợi ý người, địa điểm, tổ chức.

Pipeline CRF:

```text
text
-> underthesea word_tokenize
-> trích xuất feature cho từng token
-> CRF dự đoán BIO tags
-> hậu xử lý bằng gazetteer/rule
-> gom BIO tags thành entity
-> lọc entity trùng
```

File chính:

| Mục đích | File |
| --- | --- |
| Notebook xây dựng mô hình | `CRF/crf_ner_model.ipynb` |
| Train lại CRF cải tiến | `CRF/retrain_crf_improved.py` |
| Dự đoán bằng CRF | `CRF/predict_crf.py` |
| Model đã lưu | `CRF/models/crf_vlsp2016_notebook.joblib` |

Chạy CRF:

```powershell
cd CRF
pip install -r requirements.txt
python predict_crf.py --text "Ông Donald Trump gặp lãnh đạo Mỹ tại Washington."
```

Chạy kèm debug token/tag:

```powershell
python predict_crf.py --debug --text "Ông Donald Trump gặp lãnh đạo Mỹ tại Washington."
```

Train lại model CRF:

```powershell
cd CRF
python retrain_crf_improved.py
```

Model mới được lưu tại:

```text
CRF/models/crf_vlsp2016_improved.joblib
```

Script cũng backup model cũ và cập nhật lại model chính nếu `crf_vlsp2016_notebook.joblib` tồn tại.

## 6. Mô hình BiLSTM / BiLSTM-CRF

Phần BiLSTM nằm trong thư mục `BILSTM/` và được triển khai chủ yếu bằng notebook.

Có 2 notebook chính:

| Notebook | Nội dung |
| --- | --- |
| `BILSTM/BiLSTM_Pure_NER.ipynb` | BiLSTM thuần cho NER |
| `BILSTM/Vietnamese_NER_Project.ipynb` | Hệ thống NER dùng BiLSTM-CRF |

Ý tưởng của BiLSTM:

- Embedding biến token thành vector.
- LSTM chiều xuôi đọc câu từ trái sang phải.
- LSTM chiều ngược đọc câu từ phải sang trái.
- Vector ngữ cảnh hai chiều được dùng để dự đoán nhãn BIO.

Với BiLSTM-CRF, tầng CRF ở cuối giúp mô hình học thêm quan hệ chuyển tiếp giữa các nhãn, ví dụ `B-PER -> I-PER` hợp lý hơn `O -> I-PER`.

Pipeline BiLSTM:

```text
text
-> tokenize
-> token id
-> embedding
-> BiLSTM
-> Linear/CRF decode
-> BIO tags
-> gom entity
```

Model và kết quả đã lưu:

| File | Ý nghĩa |
| --- | --- |
| `BILSTM/models/best_bilstm_pure.pt` | Model BiLSTM thuần tốt nhất |
| `BILSTM/models/best_model_notebook.pt` | Model từ notebook BiLSTM-CRF |
| `BILSTM/models/last_checkpoint.pt` | Checkpoint cuối |
| `BILSTM/models/best_result.json` | Kết quả tốt nhất được lưu |
| `BILSTM/models/training_history*.json` | Lịch sử loss/F1 khi train |
| `BILSTM/models/training_history*.png` | Biểu đồ quá trình train |

Theo file kết quả hiện có, mô hình notebook ghi nhận `best_entity_f1` khoảng `0.8635`. File history của BiLSTM thuần ghi nhận F1 cao nhất khoảng `0.8931`.

## 7. So sánh hai hướng mô hình

| Tiêu chí | CRF | BiLSTM / BiLSTM-CRF |
| --- | --- | --- |
| Cách học | Dựa vào feature thủ công | Tự học đặc trưng từ dữ liệu |
| Ngữ cảnh | Token trước/sau, bigram, transition | Ngữ cảnh hai chiều rộng hơn |
| BIO constraint | CRF học transition giữa nhãn | BiLSTM-CRF học transition ở tầng CRF |
| Ưu điểm | Dễ giải thích, chạy nhẹ | Mạnh hơn khi có nhiều dữ liệu |
| Nhược điểm | Phụ thuộc feature engineering | Cần nhiều dữ liệu, dễ gặp OOV nếu tokenize lệch |

## 8. Cài đặt môi trường

Tạo môi trường Python mới nếu cần:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Cài thư viện:

```powershell
cd CRF
pip install -r requirements.txt
```

Các thư viện chính:

- `pandas`
- `numpy`
- `scikit-learn`
- `scikit-learn-crfsuite`
- `underthesea`
- `joblib`
- `torch`
- `datasets`
- `tqdm`

## 9. Ghi chú khi chạy

- Các lệnh CRF nên chạy bên trong thư mục `CRF/` vì script dùng path tương đối như `data/...` và `models/...`.
- Phần BiLSTM hiện được lưu chủ yếu trong notebook; nếu muốn chạy suy luận bằng script riêng, cần đảm bảo các file module như `config.py`, `dataset.py`, `model.py`, `utils/persistence.py` tồn tại đúng như import trong `BILSTM/test_module.py`.
- `BILSTM/test_module.py` đang load mặc định `models/best_model.pt`, trong khi thư mục model hiện có `best_model_notebook.pt` và `best_bilstm_pure.pt`. Khi dùng script này cần đổi path model cho khớp.
- Một số file cũ có nội dung bị lỗi encoding; README này dùng UTF-8.

## 10. Hướng phát triển

- Tách code trong notebook BiLSTM thành các file `.py` rõ ràng: `dataset.py`, `model.py`, `train.py`, `predict.py`.
- Chuẩn hóa lại encoding toàn bộ source code sang UTF-8.
- Thống nhất đường dẫn model giữa notebook và script predict.
- Thêm báo cáo đánh giá theo entity-level precision, recall, F1.
- Thử nghiệm thêm PhoBERT hoặc XLM-R cho NER tiếng Việt.
