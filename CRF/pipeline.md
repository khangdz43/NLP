# Pipeline NER: CRF va BiLSTM

## Muc tieu chung

Du an nay dung de nhan dien thuc the trong van ban tieng Viet.

Loai entity dang xu ly:

| Nhan | Y nghia |
| --- | --- |
| `PER` | Ten nguoi |
| `ORG` | To chuc |
| `LOC` | Dia diem |
| `MISC` | Thuc the khac, neu co trong du lieu train |

Du lieu dung dinh dang `BIO`:

| Nhan | Y nghia |
| --- | --- |
| `B-PER` | Token bat dau ten nguoi |
| `I-PER` | Token tiep theo trong ten nguoi |
| `B-ORG` / `I-ORG` | To chuc |
| `B-LOC` / `I-LOC` | Dia diem |
| `O` | Token khong phai entity |

Vi du:

```python
[
    ("Ong", "O"),
    ("Nam", "B-PER"),
    ("o", "O"),
    ("Ha_Noi", "B-LOC"),
]
```

---

## 1. Pipeline mo hinh CRF

### File lien quan

| Muc dich | File |
| --- | --- |
| Train | `retrain_crf_improved.py` hoac notebook CRF |
| Predict | `predict_crf.py` |
| Model da luu | `models/crf_vlsp2016_notebook.joblib` |

### CRF lam gi?

CRF la mo hinh gan nhan chuoi. No khong du doan tung tu rieng le mot cach doc lap, ma du doan ca chuoi nhan `BIO` cho ca cau.

Vi du cau:

```text
Ong Nam o Ha_Noi
```

CRF se tim chuoi nhan co diem cao nhat:

```text
O B-PER O B-LOC
B-PER I-PER O B-LOC
O O O O
...
```

Chuoi nao co tong diem cao nhat se duoc chon.

### Buoc 1: Tokenize van ban

`predict_crf.py` dung `underthesea`:

```python
word_tokenize(text, format="text")
```

Ket qua la danh sach token. Cac cum tu tieng Viet co the duoc noi bang dau `_` de giong du lieu VLSP.

Vi du:

```text
"Ha Noi" -> "Ha_Noi"
"Dien Kremlin" -> "Dien_Kremlin"
```

### Buoc 2: Tao feature cho tung token

Moi token duoc chuyen thanh dictionary feature bang ham `word2features()`.

Feature cua token gom:

- `word.lower()`: chu thuong cua tu
- `word.istitle()`: tu co viet hoa chu cai dau khong
- `word.isupper()`: tu co viet hoa toan bo khong
- `word.isdigit()`: co phai so khong
- `word_len`: do dai token
- `prefix_2`, `prefix_3`: 2/3 ky tu dau
- `suffix_2`, `suffix_3`: 2/3 ky tu cuoi
- `has_digit`: token co chua so khong
- `word_shape`: dang chu hoa/chu thuong/so cua token
- `BOS` / `EOS`: token o dau/cuoi cau
- Feature cua token ben trai va ben phai
- Bigram voi token truoc/sau
- Token truoc co phai prefix cua nguoi/dia diem/to chuc khong

Vi du feature cua token `Nam`:

```python
{
    "word.lower()": "nam",
    "word.istitle()": True,
    "-1:word.lower()": "ong",
    "+1:word.lower()": "o",
}
```

### Buoc 3: Train CRF

Khi train, du lieu duoc chuyen thanh:

```python
X_train = [
    [feature_token_1, feature_token_2, ...]
]

y_train = [
    ["O", "B-PER", "O", "B-LOC"]
]
```

CRF hoc 2 loai diem.

**1. Diem feature -> label**

Vi du:

- `word.istitle() = True` co the tang diem cho `PER`, `LOC`, `ORG`.
- Token sau `tai` co the tang diem cho `LOC`.
- Token sau `ong`, `ba`, `chu_tich` co the tang diem cho `PER`.

**2. Diem chuyen trang thai label -> label**

Vi du:

- `B-PER -> I-PER`: hop ly
- `B-ORG -> I-ORG`: hop ly
- `O -> I-PER`: thuong khong hop ly
- `B-PER -> I-LOC`: kem hop ly

### Buoc 4: Predict bang Viterbi

Khi du doan, CRF tinh diem cho nhieu chuoi nhan co the co. Thuat toan Viterbi chon chuoi nhan co tong diem cao nhat.

```text
Tong diem = diem feature + diem chuyen trang thai giua cac nhan
```

### Buoc 5: Hau xu ly ket qua CRF

Sau khi model tra ve `BIO tags`, `predict_crf.py` chay `postprocess_tags()`.

Muc dich:

- Bo cac tu xung ho/prefix nhu `ong`, `ba`, `chu_tich` neu bi gan nham thanh `PER`.
- Sua mot so dia danh bang `LOCATION_GAZETTEER`.
- Sua mot so to chuc bang `ORG_GAZETTEER`.
- Nhan dien them ten nguoi co dang viet hoa nhieu thanh phan.
- Giu `ORG` trong mot so truong hop nhu `Hoi dong An ninh Nga`, `Dien Kremlin`.

### Buoc 6: Gom BIO tag thanh entity

Ham `tags_to_entities()` gom cac tag lien tiep thanh entity hoan chinh.

```text
["B-PER", "I-PER"] + ["Donald", "Trump"]
=> PER: "Donald Trump"
```

```text
["B-ORG", "I-ORG", "I-ORG"] + ["Hoi_dong", "An_ninh", "Nga"]
=> ORG: "Hoi dong An ninh Nga"
```

### Buoc 7: Loc trung va in ket qua

Ham `dedupe_entities()` chi giu entity dau tien neu bi lap lai cung `type` va `text`.

Vi du neu `My` xuat hien 3 lan voi type `LOC`, ket qua chi in:

```text
LOC: "My"
```

Thong ke cung tinh theo danh sach da loc trung.

---

## 2. Pipeline mo hinh BiLSTM

### File lien quan

| Muc dich | File |
| --- | --- |
| Kien truc dung chung | `ner_bilstm.py` |
| Train | `bilstm_ner_model.ipynb` |
| Predict | `predict_bilstm.py` |
| Model da luu | `models/bilstm_vlsp2016_notebook.pt` |

### BiLSTM lam gi?

BiLSTM la mo hinh neural network doc cau theo 2 chieu:

- Trai sang phai
- Phai sang trai

Nho vay, moi token co vector bieu dien dua tren ca ngu canh truoc va sau.

Vi du:

```text
Ong Nam o Ha_Noi
```

BiLSTM doc:

```text
Ong -> Nam -> o -> Ha_Noi
Ha_Noi -> o -> Nam -> Ong
```

Sau do model du doan nhan `BIO` cho tung token.

### Buoc 1: Tokenize va tach cau

`predict_bilstm.py` goi `tokenize_lines_for_vocab()` trong `ner_bilstm.py`.

Ham nay lam cac viec:

- Tach van ban thanh tung cau/doan ngan.
- Dung `underthesea` de tokenize.
- Tach lai token neu `underthesea` tra ve token co khoang trang bat thuong.
- Ghep nhieu token lien tiep bang dau `_` neu cum do co trong vocabulary.

Ly do phai ghep theo vocabulary:

Du lieu train VLSP thuong luu cum tu bang dau `_`, vi du:

```text
Ha_Noi
Dien_Kremlin
Hoi_dong
```

Neu predict tokenize khac train, model se gap nhieu token OOV va du doan kem hon.

### Buoc 2: Chuyen token thanh id

Moi token duoc doi sang so bang `vocab` trong artifact model.

```text
["Ong", "Nam", "o", "Ha_Noi"]
=> [45, 120, 8, 300]
```

Neu token khong co trong `vocab`, model dung id cua `<UNK>`.

### Buoc 3: Dua vao embedding

Embedding bien moi token id thanh vector so thuc.

```text
45 -> [0.12, -0.04, 0.88, ...]
```

Vector nay la dau vao cho BiLSTM.

### Buoc 4: Chay BiLSTM 2 chieu

BiLSTM tao vector ngu canh cho moi token.

Vector moi cua token chua thong tin:

- Token hien tai
- Cac token phia truoc
- Cac token phia sau

Dieu nay giup model hieu ngu canh hon so voi viec nhin tung tu rieng le.

### Buoc 5: Linear layer du doan label

Output cua BiLSTM di qua layer Linear:

```text
hidden_vector -> diem cho tung label BIO
```

Sau do `argmax` chon label co diem cao nhat.

Vi du token `Nam`:

```python
{
    "O": 0.1,
    "B-PER": 3.2,
    "B-LOC": 0.4,
}
```

Ket qua:

```text
B-PER
```

### Buoc 6: Fallback dia danh

`predict_bilstm.py` co ham `apply_gazetteer_fallback()`.

Neu token nam trong `LOCATION_GAZETTEER` thi gan lai thanh `B-LOC`.

Muc dich:

- Sua mot so dia danh pho bien ma model neural co the doan sai.
- Tang kha nang nhan dien `LOC` voi cac nuoc/thanh pho da biet.

### Buoc 7: Gom BIO tag thanh entity

Giong CRF, ham `tags_to_entities()` gom cac `BIO tag` thanh entity hoan chinh.

```text
["B-LOC"] + ["Washington"]
=> LOC: "Washington"
```

```text
["B-PER", "I-PER"] + ["Donald", "Trump"]
=> PER: "Donald Trump"
```

### Buoc 8: Loc trung va in ket qua

`predict_bilstm.py` cung dung `dedupe_entities()`.

Neu cung mot entity lap lai nhieu lan, chi in mot lan. Thong ke `LOC`, `ORG`, `PER` cung tinh theo danh sach da loc trung.

---

## 3. So sanh nhanh CRF va BiLSTM

| Tieu chi | CRF | BiLSTM |
| --- | --- | --- |
| Cach hoc | Dua vao feature thu cong | Tu hoc pattern tu du lieu |
| Ngu canh | Dung feature token truoc/sau va transition | Doc 2 chieu nen hieu ngu canh rong hon |
| BIO rule | Hoc transition giua cac nhan rat ro | Du doan moi token bang neural output |
| Giai thich | De giai thich hon | Kho giai thich hon |
| Du lieu | On dinh khi du lieu khong qua lon | Manh hon khi co nhieu du lieu train |
| Diem yeu | Phu thuoc feature engineering | De bi anh huong boi token ngoai vocab |

---

## 4. Luong predict tong quat

```text
Input text
  -> tokenize
  -> dua ve dinh dang giong du lieu train
  -> model du doan BIO tags
  -> hau xu ly neu co
  -> gom BIO tags thanh entity
  -> loc trung entity
  -> in danh sach entity va thong ke
```

---

## 5. Lenh chay

### Chay CRF

```powershell
python predict_crf.py
```

### Chay BiLSTM

```powershell
python predict_bilstm.py
```

### Debug CRF

```powershell
python predict_crf.py --debug
```

### Debug BiLSTM

```powershell
python predict_bilstm.py --debug
```

### Truyen text rieng

```powershell
python predict_crf.py --text "Ong Donald Trump gap lanh dao My tai Washington."
python predict_bilstm.py --text "Ong Donald Trump gap lanh dao My tai Washington."
```
