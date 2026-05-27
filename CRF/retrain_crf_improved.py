"""
Script train lại CRF model với cấu hình cải thiện
- Features đầy đủ (prefix, suffix, bigram, word_shape)
- Prefix lists mở rộng
- Hyperparameters tối ưu hơn
"""

import ast
from pathlib import Path
import joblib
import pandas as pd
import sklearn_crfsuite
from sklearn.metrics import classification_report, precision_recall_fscore_support
from typing import List, Tuple, Dict, Any


# =========================================================
# Label Mapping
# =========================================================
LABEL_MAP = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}


# =========================================================
# Feature Engineering - PHIÊN BẢN CẢI THIỆN
# =========================================================
def word2features(sent: List[Tuple[str, str]], i: int) -> Dict[str, Any]:
    """
    Trích xuất features cho một từ trong câu - PHIÊN BẢN CẢI THIỆN
    """
    word = sent[i][0]

    # ====== FEATURE CƠ BẢN ======
    features: Dict[str, Any] = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word.istitle()": word.istitle(),
        "word.isupper()": word.isupper(),
        "word.isdigit()": word.isdigit(),
        "word_len": len(word),

        # Prefix và suffix - QUAN TRỌNG cho nhận diện entity
        "prefix_2": word[:2],
        "prefix_3": word[:3],
        "suffix_2": word[-2:],
        "suffix_3": word[-3:],

        # Chứa số không
        "has_digit": any(c.isdigit() for c in word),

        # Shape của từ: "Hà_Nội" -> "Xx_Xxxx"
        "word_shape": "".join(
            "X" if c.isupper() else
            "x" if c.islower() else
            "d" if c.isdigit() else c
            for c in word
        ),
    }

    # ====== PREFIX ENTITY - MỞ RỘNG ======
    person_prefixes = {
        "ông", "bà", "anh", "chị", "em", "ngài", "thông", "nạn_nhân", "thợ", "cụ",
        "bác", "cô", "dì", "chú", "giáo_sư", "tiến_sĩ", "thạc_sĩ", "kỹ_sư", 
        "bác_sĩ", "luật_sư", "họa_sĩ", "nhà_thơ", "nhà_văn", "ca_sĩ"
    }
    
    loc_prefixes = {
        "tại", "ở", "xã", "huyện", "tỉnh", "thành_phố", "quận", "khu_vực", "biển", "đảo", "phường",
        "đường", "làng", "thôn", "khu", "vùng", "miền", "thị_trấn", "thị_xã", "ấp", "bản",
        "núi", "sông", "hồ", "cảng", "sân_bay", "ga", "bến", "chợ"
    }
    
    org_prefixes = {
        "công_ty", "tập_đoàn", "bộ", "ban", "ngành", "ngân_hàng", "đại_học", "trường",
        "viện", "sở", "cục", "tòa", "ủy_ban", "hội", "liên_đoàn", "chi_nhánh", "phòng",
        "trung_tâm", "học_viện", "nhà_máy", "xí_nghiệp", "tổng_công_ty", "văn_phòng",
        "đội", "đoàn", "tổ", "nhóm", "câu_lạc_bộ", "hiệp_hội", "liên_hiệp"
    }

    # ====== CONTEXT TRÁI ======
    if i > 0:
        word_prev = sent[i - 1][0].lower()

        features.update({
            "-1:word.lower()": word_prev,
            "-1:word.istitle()": sent[i - 1][0].istitle(),

            "-1:is_person_prefix": word_prev in person_prefixes,
            "-1:is_loc_prefix": word_prev in loc_prefixes,
            "-1:is_org_prefix": word_prev in org_prefixes,

            # Bigram trái - QUAN TRỌNG
            "-1:bigram": word_prev + "_" + word.lower()
        })
    else:
        features["BOS"] = True

    # ====== CONTEXT PHẢI ======
    if i < len(sent) - 1:
        word_next = sent[i + 1][0].lower()

        features.update({
            "+1:word.lower()": word_next,
            "+1:word.istitle()": sent[i + 1][0].istitle(),

            # Bigram phải - QUAN TRỌNG
            "+1:bigram": word.lower() + "_" + word_next
        })
    else:
        features["EOS"] = True

    return features


def sent2features(sent: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Trích xuất features cho mỗi từ trong câu"""
    return [word2features(sent, i) for i in range(len(sent))]


# =========================================================
# Load Data
# =========================================================
def prepare_data_from_csv(csv_path: Path) -> List[List[Tuple[str, str]]]:
    """
    Đọc dữ liệu từ CSV và chuyển đổi sang định dạng phù hợp cho CRF
    Trả về: sentence = [(token, BIO_TAG), ...]
    """
    df = pd.read_csv(csv_path)
    sentences = []

    for _, row in df.iterrows():
        # Parse tokens và tags từ string
        tokens = ast.literal_eval(row["tokens"]) if isinstance(row["tokens"], str) else row["tokens"]
        tags = ast.literal_eval(row["ner_tags"]) if isinstance(row["ner_tags"], str) else row["ner_tags"]

        # Chuyển đổi từ ID sang BIO tag
        words_and_tags = [(w, LABEL_MAP[tid]) for w, tid in zip(tokens, tags)]
        sentences.append(words_and_tags)

    return sentences


# =========================================================
# Main Training
# =========================================================
def main():
    print("=" * 70)
    print("🔄 TRAIN LẠI CRF MODEL VỚI CẤU HÌNH CẢI THIỆN")
    print("=" * 70)

    # Load dữ liệu
    print("\n📂 Đang load dữ liệu...")
    train_csv = Path("data/vlsp_train_raw.csv")
    valid_csv = Path("data/vlsp_valid_raw.csv")

    train_sentences = prepare_data_from_csv(train_csv)
    valid_sentences = prepare_data_from_csv(valid_csv)

    print(f"✅ Số câu train: {len(train_sentences)}")
    print(f"✅ Số câu validation: {len(valid_sentences)}")

    # Trích xuất features
    print("\n🔧 Đang trích xuất features...")
    X_train = [sent2features(s) for s in train_sentences]
    y_train = [[label for _, label in s] for s in train_sentences]

    X_valid = [sent2features(s) for s in valid_sentences]
    y_valid = [[label for _, label in s] for s in valid_sentences]

    print(f"✅ Train: {len(X_train)} câu")
    print(f"✅ Valid: {len(X_valid)} câu")

    # Khởi tạo và huấn luyện CRF với hyperparameters tối ưu
    print("\n🚀 Đang huấn luyện CRF với cấu hình cải thiện...")
    print("   • c1=0.15 (L1 regularization)")
    print("   • c2=0.15 (L2 regularization)")
    print("   • max_iterations=150")
    
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.15,  # Tăng từ 0.1 -> 0.15
        c2=0.15,  # Tăng từ 0.1 -> 0.15
        max_iterations=150,  # Tăng từ 100 -> 150
        all_possible_transitions=True,
        verbose=True,  # Hiển thị quá trình training
    )

    crf.fit(X_train, y_train)
    print("\n✅ Huấn luyện hoàn tất!")

    # Đánh giá mô hình
    print("\n" + "=" * 70)
    print("📊 ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 70)

    y_pred = crf.predict(X_valid)

    # Chuyển đổi danh sách labels phẳng (flat)
    y_true_flat = [label for sent in y_valid for label in sent]
    y_pred_flat = [label for sent in y_pred for label in sent]

    # Tính Precision, Recall, F1-Score cho từng nhãn
    labels = list(LABEL_MAP.values())
    labels.remove('O')  # Loại bỏ nhãn O

    print("\n📊 Báo cáo chi tiết từng loại entity:")
    print("-" * 70)
    print(classification_report(y_true_flat, y_pred_flat, labels=labels, zero_division=0))

    # Tính cho từng loại entity
    entity_types = ['PER', 'ORG', 'LOC', 'MISC']
    print("\n📊 Chi tiết từng loại Entity:")
    print("-" * 70)

    for ent_type in entity_types:
        b_label = f"B-{ent_type}"
        i_label = f"I-{ent_type}"
        
        # Lọc các vị trí có entity thực sự
        true_entities = [1 if y in [b_label, i_label] else 0 for y in y_true_flat]
        pred_entities = [1 if y in [b_label, i_label] else 0 for y in y_pred_flat]
        
        if sum(true_entities) > 0:
            p, r, f, _ = precision_recall_fscore_support(
                true_entities, pred_entities, average='binary', zero_division=0
            )
            print(f"  {ent_type:6s} - Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")

    # Lưu model
    print("\n" + "=" * 70)
    print("💾 ĐANG LƯU MÔ HÌNH")
    print("=" * 70)

    model_out = Path("models/crf_vlsp2016_improved.joblib")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": crf}, model_out)
    print(f"✅ Đã lưu model vào: {model_out}")

    # Backup model cũ và thay thế
    old_model = Path("models/crf_vlsp2016_notebook.joblib")
    if old_model.exists():
        backup = Path("models/crf_vlsp2016_notebook_backup.joblib")
        import shutil
        shutil.copy(old_model, backup)
        print(f"✅ Đã backup model cũ vào: {backup}")
        
        # Copy model mới thành model chính
        shutil.copy(model_out, old_model)
        print(f"✅ Đã cập nhật model chính: {old_model}")

    print("\n" + "=" * 70)
    print("🎉 HOÀN TẤT!")
    print("=" * 70)
    print("\n📝 Bước tiếp theo:")
    print("   1. Chạy: python predict_ner.py")
    print("   2. Kiểm tra kết quả nhận diện")
    print("   3. So sánh với model cũ")


if __name__ == "__main__":
    main()
