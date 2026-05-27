from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

import torch

from config import ModelConfig, UNK_TOKEN
from dataset import tokenize_text
from model import BiLSTMCRF
from utils.persistence import load_checkpoint


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# GAZETTEER
# =========================
PERSON_PREFIXES = {
    "ông", "bà", "anh", "chị", "em", "ngài", "tổng_thống",
    "thủ_tướng", "chủ_tịch", "phó_chủ_tịch", "bộ_trưởng",
}

LOCATION_GAZETTEER = {
    "mỹ", "nga", "israel", "iran", "ukraine",
    "venezuela", "washington", "moskva",
    "việt_nam", "trung_quốc"
}

ORG_GAZETTEER = {
    "điện_kremlin",
}


# =========================
# LOAD MODEL
# =========================
def load_model_checkpoint(path: str):
    ckpt = load_checkpoint(Path(path), DEVICE.type)

    token2id = ckpt["token2id"]
    id2tag = ckpt["id2tag"]
    cfg = ckpt["config"]

    config = ModelConfig(
        vocab_size=int(cfg["vocab_size"]),
        tag_size=int(cfg["tag_size"]),
        embedding_dim=int(cfg["embedding_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
        pad_idx=int(cfg["pad_idx"]),
    )

    model = BiLSTMCRF(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    lowercase = bool(ckpt.get("training_config", {}).get("lowercase", False))

    return model, token2id, id2tag, lowercase


# =========================
# TOKENIZE
# =========================
def tokenize(sentence: str, lowercase: bool):
    return tokenize_text(sentence, lowercase=lowercase)


# =========================
# GAZETTEER FEATURES (IMPORTANT FIX)
# =========================
def build_gazetteer_feats(tokens: List[str]) -> torch.Tensor:
    feats = []

    for t in tokens:
        tl = t.lower()

        feats.append([
            float(tl in PERSON_PREFIXES),
            float(tl in LOCATION_GAZETTEER),
            float(tl in ORG_GAZETTEER),
        ])

    return torch.tensor(feats, dtype=torch.float)


# =========================
# INPUT PREP (FIXED)
# =========================
def prepare_input(tokens, token2id, lowercase):
    unk = token2id.get(UNK_TOKEN, 1)

    norm = [t.lower() if lowercase else t for t in tokens]
    ids = [token2id.get(t, unk) for t in norm]

    x = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    lengths = torch.tensor([len(ids)], dtype=torch.long).to(DEVICE)
    mask = torch.arange(len(ids), device=DEVICE).unsqueeze(0) < lengths.unsqueeze(1)

    gaz = build_gazetteer_feats(tokens).unsqueeze(0).to(DEVICE)

    return x, gaz, lengths, mask


# =========================
# PREDICT (NO OVERWRITE TAGS)
# =========================
def predict(model, token2id, id2tag, tokens, lowercase):
    x, gaz, lengths, mask = prepare_input(tokens, token2id, lowercase)

    with torch.no_grad():
        pred_ids = model.decode(x, lengths, mask)[0]

    return [id2tag[int(i)] for i in pred_ids]


# =========================
# DEBUG VIEW
# =========================
def print_debug(tokens, tags):
    print("\nDEBUG TOKEN/TAG:")
    for t, tag in zip(tokens, tags):
        print(f"{t:<25} {tag}")


# =========================
# BIO -> ENTITIES (CLEAN)
# =========================
def normalize(t): return t.replace("_", " ")


def tags_to_entities(tokens, tags):
    entities = []
    cur = None

    for i, tag in enumerate(tags):
        if tag == "O":
            if cur:
                entities.append(cur)
                cur = None
            continue

        if "-" not in tag:
            continue

        pref, typ = tag.split("-", 1)

        if pref == "B" or cur is None or cur["type"] != typ:
            if cur:
                entities.append(cur)
            cur = {"type": typ, "start": i, "end": i}
        else:
            cur["end"] = i

    if cur:
        entities.append(cur)

    for e in entities:
        e["text"] = " ".join(normalize(t) for t in tokens[e["start"]:e["end"]+1])

    return entities


def dedupe(entities):
    seen = set()
    out = []
    for e in entities:
        k = (e["type"], e["text"].lower())
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def print_entities(entities):
    entities = dedupe(entities)

    print("\nENTITIES:")
    if not entities:
        print("  (none)")
        return

    stats = {}
    for e in entities:
        print(f"  - {e['type']}: {e['text']}")
        stats[e["type"]] = stats.get(e["type"], 0) + 1

    print("\nSTATS:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    model, token2id, id2tag, lowercase = load_model_checkpoint(
        "models/best_model.pt"
    )

    text = """
Hôm qua, Bộ Công an Việt Nam đã tổ chức cuộc họp tại Hà Nội để thảo luận về hợp tác an ninh mạng với Tập đoàn Google và Microsoft.

Thượng tướng Nguyễn Văn Minh cho biết Việt Nam sẽ tăng cường phối hợp với Interpol trong việc truy bắt tội phạm xuyên biên giới.

Trong chuyến công tác tại Tokyo, Thủ tướng Phạm Minh Chính đã gặp Tổng thống Hàn Quốc Yoon Suk-yeol để bàn về dự án đường sắt cao tốc Hà Nội - Hải Phòng.

Phía Mỹ và Liên minh châu Âu (EU) cũng cam kết hỗ trợ kỹ thuật cho Việt Nam trong lĩnh vực trí tuệ nhân tạo.

Trong khi đó, Tập đoàn Viettel đang mở rộng hợp tác với Samsung tại thị trường Ấn Độ và Brazil.

Ông Elon Musk cho biết SpaceX sẽ tiếp tục phóng vệ tinh Starlink phục vụ khu vực Đông Nam Á, bao gồm cả Philippines và Indonesia.
"""

    tokens = tokenize(text, lowercase)

    tags = predict(model, token2id, id2tag, tokens, lowercase)

    print_debug(tokens, tags)

    entities = tags_to_entities(tokens, tags)
    print_entities(entities)