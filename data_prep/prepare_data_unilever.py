"""
Chuẩn bị bộ dữ liệu thứ hai (file Uniliverdataset.csv) cho aspect classification.

File này KHÔNG phải dữ liệu Unilever — nội dung là review khách sạn/homestay
tiếng Việt, cùng miền với dataset_luan. Vì vậy dùng lại đúng 9 nhãn aspect của
dataset_luan thay vì bịa ra bộ nhãn mới: hai bộ so sánh trực tiếp được, và mở ra
thí nghiệm chuyển miền (train bộ này, test bộ kia).

Quy trình:
  1. Làm sạch, tách cột `content` (đã tách từ bằng dấu _) thành 2 dạng:
       content_seg  giữ dấu _   -> dùng cho PhoBERT
       sentence     bỏ dấu _    -> dùng cho mBERT / XLM-R / mT5
  2. Gán NHÃN BẠC bằng bộ phân loại huấn luyện trên phần tiếng Việt của
     dataset_luan (12.497 câu có nhãn thật), kèm độ tin cậy.
  3. Rút ra phiếu kiểm tay (verify sheet) để người gán nhãn xác nhận -> nhãn VÀNG.
  4. Chia train/val/test theo CÂU (không rò rỉ). Test chỉ dùng nhãn vàng.

    python src/prepare_data_unilever.py --raw data_unilever/Uniliverdataset.csv
    # sau khi kiểm tay xong:
    python src/prepare_data_unilever.py --gold data_unilever/verify_sheet_done.csv
"""
import argparse
import json
import os
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data_unilever", "Uniliverdataset.csv")
OUT = os.path.join(HERE, "data_unilever", "splits")
LUAN = os.path.join(HERE, "dataset_luan")

ASPECTS = ["ambiance", "amenities", "cleanliness", "food", "location",
           "overall", "price", "room", "service"]


def clean(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_silver_model(min_conf_report=True):
    """Huấn luyện bộ gán nhãn bạc trên phần tiếng Việt của dataset_luan."""
    tr = pd.read_csv(os.path.join(LUAN, "B_train.csv"))
    te = pd.read_csv(os.path.join(LUAN, "B_test.csv"))
    trv = tr[tr.language == "vi"]
    tev = te[te.language == "vi"]
    feat = make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    )
    pipe = make_pipeline(feat, CalibratedClassifierCV(LinearSVC(C=0.5), cv=3))
    pipe.fit(trv.sentence, trv.aspect)
    if min_conf_report:
        pr = pipe.predict(tev.sentence)
        acc = accuracy_score(tev.aspect, pr)
        f1 = f1_score(tev.aspect, pr, average="macro", zero_division=0)
        print(f"[bộ gán nhãn bạc] huấn luyện trên {len(trv):,} câu vi có nhãn thật")
        print(f"    kiểm trên test vi của dataset_luan: acc={acc:.4f} macroF1={f1:.4f}")
        print(f"    -> đây là TRẦN chất lượng nhãn bạc. Khoảng {1-acc:.0%} nhãn train sẽ sai.")
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--gold", default=None,
                    help="file verify sheet đã kiểm tay (có cột gold_aspect)")
    ap.add_argument("--n_verify", type=int, default=900,
                    help="số câu rút ra để kiểm tay (test 600 + val 300)")
    ap.add_argument("--min_words", type=int, default=2,
                    help="bỏ câu quá ngắn, không đủ thông tin để gán aspect")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    print(f"[đọc] {len(df):,} dòng, cột: {list(df.columns)}")

    # ---- làm sạch
    df["content_seg"] = df["content"].map(clean)
    df["sentence"] = df["content_seg"].str.replace("_", " ", regex=False)
    df["n_words"] = df["sentence"].str.split().str.len()

    n0 = len(df)
    df = df[df.n_words >= args.min_words].reset_index(drop=True)
    print(f"[lọc câu ngắn] bỏ {n0-len(df):,} câu dưới {args.min_words} từ -> còn {len(df):,}")

    n1 = len(df)
    df = df.drop_duplicates(subset=["sentence"]).reset_index(drop=True)
    print(f"[khử trùng lặp] {n1:,} -> {len(df):,} câu khác nhau")

    # `max_length` chỉ là số từ (tương quan 0.998) -> không phải đặc trưng, bỏ
    if "max_length" in df.columns:
        print("[ghi chú] cột 'max_length' chính là số từ của câu (r=0.998), không dùng làm đặc trưng")

    # ---- nhãn bạc
    model = build_silver_model()
    prob = model.predict_proba(df.sentence)
    cls = model.classes_
    df["silver_aspect"] = cls[prob.argmax(1)]
    df["confidence"] = prob.max(1).round(4)
    df["silver_2nd"] = cls[np.argsort(-prob, axis=1)[:, 1]]

    print(f"\n[nhãn bạc] phân bố dự đoán (%)")
    print((df.silver_aspect.value_counts(normalize=True) * 100).round(1).to_string())
    print(f"    độ tin cậy trung bình {df.confidence.mean():.3f}; "
          f">=0.6: {(df.confidence >= .6).mean():.1%}; >=0.8: {(df.confidence >= .8).mean():.1%}")

    os.makedirs(args.out, exist_ok=True)

    # ---- chưa có nhãn vàng: xuất phiếu kiểm tay rồi dừng
    if not args.gold:
        rng = np.random.default_rng(args.seed)
        # lấy mẫu phân tầng theo nhãn bạc, ưu tiên phủ đều cả vùng tin cậy thấp
        per = max(1, args.n_verify // len(ASPECTS))
        picks = []
        for a, g in df.groupby("silver_aspect"):
            k = min(per, len(g))
            picks.append(g.sample(k, random_state=args.seed))
        sheet = pd.concat(picks).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        sheet["gold_aspect"] = ""      # người gán điền vào đây
        sheet["ghi_chu"] = ""
        cols = ["sentence", "silver_aspect", "silver_2nd", "confidence",
                "gold_aspect", "ghi_chu", "score", "content_seg"]
        path = os.path.join(os.path.dirname(args.out), "verify_sheet.csv")
        sheet[[c for c in cols if c in sheet.columns]].to_csv(path, index=False)

        guide = os.path.join(os.path.dirname(args.out), "labeling_guide.md")
        with open(guide, "w", encoding="utf-8") as f:
            f.write(LABELING_GUIDE)

        df.to_csv(os.path.join(args.out, "all_silver.csv"), index=False)
        print(f"\n[bước tiếp theo] Chưa có nhãn vàng nên dừng ở đây.")
        print(f"    1. Mở {path} ({len(sheet):,} câu)")
        print(f"    2. Điền cột 'gold_aspect' theo {guide}")
        print(f"    3. Lưu thành verify_sheet_done.csv rồi chạy lại với --gold")
        print(f"\n[đã lưu] {os.path.join(args.out, 'all_silver.csv')} (toàn bộ nhãn bạc)")
        return

    # ---- đã có nhãn vàng
    gold = pd.read_csv(args.gold)
    gold["gold_aspect"] = gold.gold_aspect.astype(str).str.strip().str.lower()
    gold = gold[gold.gold_aspect.isin(ASPECTS)].reset_index(drop=True)
    print(f"\n[nhãn vàng] {len(gold):,} câu đã kiểm tay")

    m = gold.merge(df[["sentence", "silver_aspect"]], on="sentence",
                   how="left", suffixes=("", "_df"))
    sv = m.silver_aspect_df.fillna(m.silver_aspect)
    agree = (sv == m.gold_aspect).mean()
    kappa = cohen_kappa_score(sv, m.gold_aspect)
    print(f"    ĐỘ ĐỒNG THUẬN bạc vs vàng: {agree:.1%}   Cohen's kappa = {kappa:.3f}")
    print(f"    -> con số này PHẢI đưa vào paper: nó là ước lượng chất lượng nhãn train.")

    gset = set(gold.sentence)
    rest = df[~df.sentence.isin(gset)].copy()
    rest["aspect"] = rest.silver_aspect
    rest["label_source"] = "silver"

    g = gold[["sentence", "gold_aspect"]].rename(columns={"gold_aspect": "aspect"})
    g = g.merge(df.drop(columns=["silver_aspect"]), on="sentence", how="left")
    g["label_source"] = "gold"

    # test + val lấy TOÀN BỘ từ nhãn vàng; train là phần còn lại (nhãn bạc)
    te_g, va_g = train_test_split(g, test_size=0.35, random_state=args.seed,
                                  stratify=g.aspect if g.aspect.value_counts().min() >= 2 else None)
    written = {"train": rest, "validation": va_g, "test": te_g}

    for name, d in written.items():
        d = d.copy()
        d["language"] = "vi"
        d["place_type"] = "hotel"
        d["source"] = "unilever_file"
        d["review_id"] = [f"{name}_{i}" for i in range(len(d))]
        keep = ["sentence", "content_seg", "aspect", "label_source", "confidence",
                "score", "language", "place_type", "source", "review_id"]
        d[[c for c in keep if c in d.columns]].to_csv(
            os.path.join(args.out, f"{name}.csv"), index=False)
    with open(os.path.join(args.out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(ASPECTS, f, indent=2, ensure_ascii=False)

    print(f"\n[đã lưu] {args.out}")
    for k, v in written.items():
        src = v.get("label_source", pd.Series(["?"])).iloc[0] if len(v) else "-"
        print(f"    {k:11s} {len(v):6,d} dòng   nhãn: {src}")
    print("\n    LƯU Ý: train dùng nhãn bạc (có sai số), val/test dùng nhãn vàng.")
    print("    Sai ở train chỉ làm mô hình yếu đi; sai ở test thì mọi con số đều vô nghĩa.")


LABELING_GUIDE = """# Hướng dẫn gán nhãn aspect — bộ review khách sạn tiếng Việt

Điền cột `gold_aspect` bằng **đúng một** trong 9 giá trị dưới đây (chữ thường).
Cột `silver_aspect` là gợi ý của máy — sai khoảng 25%, đừng tin sẵn.

| Nhãn | Nói về | Ví dụ |
|---|---|---|
| `room` | bản thân căn phòng: rộng/chật, đẹp/cũ, giường, cách âm | "phòng rộng rãi", "phòng cách âm kém" |
| `cleanliness` | vệ sinh, sạch/bẩn, mùi, côn trùng | "sạch sẽ", "chăn ga có mùi" |
| `service` | nhân viên, thái độ, chủ nhà, hỗ trợ, check-in | "nhân viên nhiệt tình", "chủ dễ thương" |
| `amenities` | tiện nghi, hồ bơi, wifi, điều hoà, thang máy, đồ dùng | "có hồ bơi", "wifi yếu" |
| `location` | vị trí, gần biển/trung tâm, đi lại, chỗ đỗ xe | "vị trí gần biển", "khó tìm" |
| `food` | đồ ăn, bữa sáng, nhà hàng, đồ uống | "đồ ăn sáng không ngon" |
| `price` | giá, đáng tiền, phụ phí | "giá rẻ", "giá hơi cao so với chất lượng" |
| `ambiance` | không gian, view, khung cảnh, yên tĩnh, trang trí | "không gian đẹp", "view thung lũng" |
| `overall` | nhận xét tổng thể, không nhắm vào khía cạnh cụ thể nào | "mọi thứ đều ổn", "sẽ quay lại" |

## Quy tắc xử lý ca khó

1. **Câu nói nhiều khía cạnh** → chọn khía cạnh được nói **nhiều nhất / là ý chính**.
   "phòng sạch sẽ nhân viên thân thiện" → nếu cân bằng, ưu tiên khía cạnh đứng đầu câu.
   Ghi vào cột `ghi_chu` là "đa nhãn" để sau này thống kê được.
2. **Câu chung chung, khen/chê cả trải nghiệm** → `overall`, không phải `service`.
   "tuyệt vời", "rất hài lòng", "sẽ quay lại" → `overall`.
3. **`room` vs `amenities`**: thứ gắn liền với căn phòng (giường, cách âm, diện tích) là
   `room`; thứ là dịch vụ/thiết bị dùng chung hoặc thêm vào (hồ bơi, wifi, thang máy,
   máy sấy) là `amenities`.
4. **`ambiance` vs `room`**: cảm nhận không gian, view, sự yên tĩnh là `ambiance`;
   mô tả vật lý căn phòng là `room`.
5. **`service` vs `overall`**: có nhắc tới con người (nhân viên, chủ, lễ tân) → `service`.
6. **Câu sai chính tả / thiếu dấu**: vẫn gán bình thường nếu đoán được ý.
   Không đoán được → để trống `gold_aspect` và ghi "không rõ" vào `ghi_chu`.
7. **Câu không nói gì về khách sạn** (spam, quảng cáo, ký tự rác) → để trống và
   ghi "loại" vào `ghi_chu`.

## Cách làm để số liệu dùng được cho paper

- Nếu có **2 người gán độc lập** trên cùng 200 câu đầu, ta tính được Cohen's kappa
  giữa hai người — đây là con số reviewer sẽ hỏi. Kappa > 0.6 là chấp nhận được,
  > 0.8 là tốt.
- Gán xong 200 câu đầu thì dừng lại, đối chiếu, thống nhất các ca bất đồng, cập nhật
  hướng dẫn này, rồi mới gán tiếp phần còn lại.
- Đừng nhìn cột `silver_aspect` trước khi tự quyết định — nhìn trước sẽ bị neo theo máy
  và độ đồng thuận báo cáo sẽ bị thổi phồng.
"""


if __name__ == "__main__":
    main()
