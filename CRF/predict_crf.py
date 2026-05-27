import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
from underthesea import word_tokenize


DEFAULT_TEXT = """
Phó chủ tịch Hội đồng An ninh Nga cho rằng Mỹ khó lòng đóng vai trò trung gian hiệu quả trong các cuộc xung đột trên thế giới.

"Một quốc gia có hành động bắt cóc nguyên thủ nước khác và khơi mào xung đột một cách tùy tiện khó lòng được coi là bên trung gian hiệu quả trong mọi tình huống", Phó chủ tịch Hội đồng An ninh Nga Dmitry Medvedev phát biểu tại một diễn đàn giáo dục ở Moskva hôm nay.

Ông Medvedev dường như đề cập chiến dịch do Mỹ - Israel phát động nhằm vào Iran, cũng như cuộc đột kích bắt giữ vợ chồng Tổng thống Venezuela Nicolas Maduro do Washington tiến hành hồi tháng 1.

Phát biểu này trái ngược với quan điểm của giới lãnh đạo Nga, khi Điện Kremlin luôn nói rằng Mỹ đang đóng vai trò tích cực trong tìm kiếm giải pháp hòa bình cho xung đột Ukraine. Tuy vậy, ông Medvedev cũng đánh giá chính quyền Tổng thống Donald Trump đang nỗ lực giải quyết chiến sự, khác với người tiền nhiệm Joe Biden.
"""


PERSON_PREFIXES = {
    "ông", "bà", "anh", "chị", "em", "ngài", "nạn_nhân", "thợ", "cụ",
    "bác", "cô", "dì", "chú", "giáo_sư", "tiến_sĩ", "thạc_sĩ", "kỹ_sư",
    "bác_sĩ", "luật_sư", "họa_sĩ", "nhà_thơ", "nhà_văn", "ca_sĩ",
    "tổng_thống", "thủ_tướng", "chủ_tịch", "phó_chủ_tịch", "bộ_trưởng",
}

LOC_PREFIXES = {
    "tại", "ở", "xã", "huyện", "tỉnh", "thành_phố", "quận", "khu_vực", "biển", "đảo", "phường",
    "đường", "làng", "thôn", "khu", "vùng", "miền", "thị_trấn", "thị_xã", "ấp", "bản",
    "núi", "sông", "hồ", "cảng", "sân_bay", "ga", "bến", "chợ",
}

ORG_PREFIXES = {
    "công_ty", "tập_đoàn", "bộ", "ban", "ngành", "ngân_hàng", "đại_học", "trường",
    "viện", "sở", "cục", "tòa", "ủy_ban", "hội", "liên_đoàn", "chi_nhánh", "phòng",
    "trung_tâm", "học_viện", "nhà_máy", "xí_nghiệp", "tổng_công_ty", "văn_phòng",
    "đội", "đoàn", "tổ", "nhóm", "câu_lạc_bộ", "hiệp_hội", "liên_hiệp",
}

LOCATION_GAZETTEER = {
    "mỹ", "nga", "israel", "iran", "ukraine", "venezuela", "washington", "moskva",
    "việt_nam", "trung_quốc", "nhật_bản", "hàn_quốc", "pháp", "đức", "anh",
}

ORG_GAZETTEER = {
    "điện_kremlin",
}

ORG_NAME_HINTS = {
    "hội_đồng", "an_ninh", "điện", "kremlin", "bộ", "ủy_ban", "công_ty",
}


def normalize_token(token: str) -> str:
    return token.replace("_", " ")


def tokenize_for_crf(text: str) -> List[str]:
    """Tokenize giống dữ liệu train VLSP: multi-word token được nối bằng '_'."""
    tokenized_text = word_tokenize(text, format="text")
    return tokenized_text.split()


def word2features(sent: List[Tuple[str, str]], i: int) -> Dict[str, Any]:
    word = sent[i][0]

    features: Dict[str, Any] = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word.istitle()": word.istitle(),
        "word.isupper()": word.isupper(),
        "word.isdigit()": word.isdigit(),
        "word_len": len(word),
        "prefix_2": word[:2],
        "prefix_3": word[:3],
        "suffix_2": word[-2:],
        "suffix_3": word[-3:],
        "has_digit": any(c.isdigit() for c in word),
        "word_shape": "".join(
            "X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c
            for c in word
        ),
    }

    if i > 0:
        word_prev = sent[i - 1][0].lower()
        features.update(
            {
                "-1:word.lower()": word_prev,
                "-1:word.istitle()": sent[i - 1][0].istitle(),
                "-1:is_person_prefix": word_prev in PERSON_PREFIXES,
                "-1:is_loc_prefix": word_prev in LOC_PREFIXES,
                "-1:is_org_prefix": word_prev in ORG_PREFIXES,
                "-1:bigram": word_prev + "_" + word.lower(),
            }
        )
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        word_next = sent[i + 1][0].lower()
        features.update(
            {
                "+1:word.lower()": word_next,
                "+1:word.istitle()": sent[i + 1][0].istitle(),
                "+1:bigram": word.lower() + "_" + word_next,
            }
        )
    else:
        features["EOS"] = True

    return features


def tags_to_entities(tokens: List[str], tags: List[str]) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    current = None

    for i, tag in enumerate(tags):
        if tag == "O":
            if current is not None:
                entities.append(current)
                current = None
            continue

        if "-" not in tag:
            continue
        prefix, ent_type = tag.split("-", 1)

        if prefix == "B" or current is None or current["type"] != ent_type:
            if current is not None:
                entities.append(current)
            current = {"type": ent_type, "start": i, "end": i}
        else:
            current["end"] = i

    if current is not None:
        entities.append(current)

    for ent in entities:
        ent["text"] = " ".join(normalize_token(t) for t in tokens[ent["start"] : ent["end"] + 1])

    return entities


def is_probable_person_name(token: str, prev_token: str | None = None) -> bool:
    lower = token.lower()
    if lower in LOCATION_GAZETTEER or lower in ORG_GAZETTEER or lower in ORG_NAME_HINTS:
        return False

    parts = token.split("_")
    if len(parts) >= 2 and all(part and part[0].isupper() for part in parts):
        return True

    if prev_token and prev_token.lower() in PERSON_PREFIXES and token and token[0].isupper():
        return True

    return False


def is_inside_org_name(tokens: List[str], index: int) -> bool:
    start = index
    while start > 0:
        prev = tokens[start - 1].lower()
        if prev in ORG_NAME_HINTS or prev in LOCATION_GAZETTEER:
            start -= 1
            continue
        break

    window = {token.lower() for token in tokens[start : index + 1]}
    return bool(window & ORG_NAME_HINTS)


def postprocess_tags(tokens: List[str], tags: List[str]) -> List[str]:
    fixed = list(tags)

    for i, token in enumerate(tokens):
        lower = token.lower()
        prev_token = tokens[i - 1] if i > 0 else None

        if lower in PERSON_PREFIXES:
            fixed[i] = "O"
            continue

        if lower in ORG_GAZETTEER:
            fixed[i] = "B-ORG"
            continue

        if is_probable_person_name(token, prev_token):
            fixed[i] = "B-PER"
            continue

        if lower in LOCATION_GAZETTEER:
            if fixed[i] == "I-ORG" and is_inside_org_name(tokens, i):
                continue
            fixed[i] = "B-LOC"

    return fixed


def predict_crf(text: str, model) -> List[Dict[str, Any]]:
    tokens = tokenize_for_crf(text)
    temp_sent = [(w, "O") for w in tokens]
    tags = model.predict_single([word2features(temp_sent, i) for i in range(len(temp_sent))])
    tags = postprocess_tags(tokens, tags)
    return tags_to_entities(tokens, tags)


def print_debug(text: str, model) -> None:
    tokens = tokenize_for_crf(text)
    temp_sent = [(w, "O") for w in tokens]
    tags = model.predict_single([word2features(temp_sent, i) for i in range(len(temp_sent))])
    fixed_tags = postprocess_tags(tokens, tags)

    print("\nDebug token/tag:")
    for token, tag, fixed_tag in zip(tokens, tags, fixed_tags):
        suffix = "" if tag == fixed_tag else f" -> {fixed_tag}"
        print(f"    {token:<28} {tag}{suffix}")


def dedupe_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique_entities: List[Dict[str, Any]] = []
    seen = set()

    for ent in entities:
        key = (ent["type"], ent["text"].casefold())
        if key in seen:
            continue
        seen.add(key)
        unique_entities.append(ent)

    return unique_entities


def print_entities(entities: List[Dict[str, Any]]) -> None:
    entities = dedupe_entities(entities)

    print("\nEntities phát hiện được:")
    if not entities:
        print("  (Không phát hiện thực thể nào)")
        return

    for ent in entities:
        print(f"    - {ent['type']}: \"{ent['text']}\"")

    stats: Dict[str, int] = {}
    for ent in entities:
        stats[ent["type"]] = stats.get(ent["type"], 0) + 1

    print("\nThống kê:")
    for etype, count in sorted(stats.items()):
        print(f"    - {etype}: {count}")
    print(f"\nTổng entities: {len(entities)}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="NER Predict - CRF")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to run NER on")
    parser.add_argument("--model", default="models/crf_vlsp2016_notebook.joblib", help="Path to CRF joblib model")
    parser.add_argument("--debug", action="store_true", help="Print CRF tokens and BIO tags")
    args = parser.parse_args()

    crf_path = Path(args.model)
    print(f"Loading CRF model: {crf_path}")

    crf_model_obj = joblib.load(crf_path)
    crf_model = crf_model_obj["model"] if isinstance(crf_model_obj, dict) else crf_model_obj

    if args.debug:
        print_debug(args.text, crf_model)

    entities = predict_crf(args.text, crf_model)
    print_entities(entities)


if __name__ == "__main__":
    main()
