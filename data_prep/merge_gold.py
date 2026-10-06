"""
Gộp hai phiếu gán nhãn tay -> tập kiểm tra chuyển miền (đa nhãn 9 khía cạnh),
kèm Cohen's kappa giữa hai người gán trên 200 câu chung.

    python src/merge_gold.py \
        --a data_unilever/gold/gan_nhan_A.xlsx \
        --b data_unilever/gold/gan_nhan_B.xlsx \
        --out data_unilever/splits/transfer_test.csv

Quy ước:
  - Ô khía cạnh: 1 = câu có nói tới khía cạnh đó, để trống = không.
  - ghi_chu chứa "loại" hoặc "không rõ"  -> bỏ câu khỏi tập kiểm tra.
  - 200 câu CHUNG: sau khi tính kappa, lấy hợp/giao theo --resolve.
"""
import argparse, sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ASP = ["room","cleanliness","service","amenities","location","food","price","ambiance","overall"]
OUT_ORDER = ["overall","amenities","food","service","location","ambiance","room","price","cleanliness"]
DROP_NOTES = ("loại","loai","không rõ","khong ro")


def read_sheet(path, who):
    d = pd.read_excel(path, sheet_name="gan_nhan", skiprows=1)
    d = d[d["phan"].isin(["CHUNG", "RIENG"])].copy()      # bỏ dòng ví dụ
    for a in ASP:
        d[a] = pd.to_numeric(d[a], errors="coerce").fillna(0).astype(int).clip(0, 1)
    d["ghi_chu"] = d.get("ghi_chu", "").fillna("").astype(str).str.strip().str.lower()
    d["annotator"] = who
    n_lab = d[ASP].sum(axis=1)
    d["_trong"] = (n_lab == 0) & (d.ghi_chu == "")
    if d["_trong"].any():
        print(f"  [cảnh báo] {who}: {int(d._trong.sum())} câu chưa gán (không nhãn, không ghi chú)")
    return d[["id", "phan", "cau", *ASP, "ghi_chu", "annotator", "_trong"]]


def kappa_report(A, B):
    ca = A[A.phan == "CHUNG"].set_index("id").sort_index()
    cb = B[B.phan == "CHUNG"].set_index("id").sort_index()
    common = ca.index.intersection(cb.index)
    if len(common) == 0:
        sys.exit("Hai phiếu không có câu CHUNG nào trùng id — kiểm tra lại file.")
    ca, cb = ca.loc[common], cb.loc[common]
    done = ~(ca._trong | cb._trong)
    ca, cb = ca[done], cb[done]
    print(f"\n=== ĐỘ ĐỒNG THUẬN GIỮA HAI NGƯỜI GÁN ({len(ca)} câu chung đã gán xong) ===")
    rows, kappas = [], []
    for a in ASP:
        x, y = ca[a].values, cb[a].values
        if x.sum() == 0 and y.sum() == 0:
            k, note = np.nan, "không ai gán nhãn này"
        else:
            k, note = cohen_kappa_score(x, y), ""
            kappas.append(k)
        agree = float((x == y).mean())
        rows.append({"aspect": a, "n_A": int(x.sum()), "n_B": int(y.sum()),
                     "agreement": round(agree, 4),
                     "kappa": round(k, 4) if k == k else None, "ghi_chu": note})
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))
    mean_k = float(np.mean(kappas)) if kappas else float("nan")
    exact = float((ca[ASP].values == cb[ASP].values).all(axis=1).mean())
    inter = (ca[ASP].values & cb[ASP].values).sum()
    union = (ca[ASP].values | cb[ASP].values).sum()
    print(f"\n  Kappa trung bình trên 9 khía cạnh : {mean_k:.4f}")
    print(f"  Trùng khớp hoàn toàn cả 9 ô        : {exact:.1%}")
    print(f"  Jaccard trên tập nhãn              : {inter/union if union else float('nan'):.4f}")
    print("  Mốc tham chiếu: kappa > 0.6 chấp nhận được, > 0.8 tốt.")
    print("  -> ĐƯA CON SỐ NÀY VÀO PAPER, kèm số câu và số người gán.")
    return rep, mean_k, ca, cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default="transfer_test.csv")
    ap.add_argument("--resolve", choices=["intersection", "union", "drop"], default="intersection",
                    help="cách xử lý 200 câu chung khi hai người bất đồng "
                         "(mặc định: chỉ giữ nhãn cả hai cùng đánh)")
    ap.add_argument("--report", default=None, help="lưu bảng kappa ra CSV")
    args = ap.parse_args()

    A, B = read_sheet(args.a, "A"), read_sheet(args.b, "B")
    rep, mean_k, ca, cb = kappa_report(A, B)
    if args.report:
        rep.to_csv(args.report, index=False, encoding="utf-8-sig")

    # --- phần chung: hoà giải
    if args.resolve == "drop":
        dis = (ca[ASP].values != cb[ASP].values).any(axis=1)
        shared = ca[~dis].copy()
        print(f"\n[hoà giải] bỏ {int(dis.sum())} câu chung còn bất đồng")
    else:
        op = np.logical_and if args.resolve == "intersection" else np.logical_or
        shared = ca.copy()
        shared[ASP] = op(ca[ASP].values, cb[ASP].values).astype(int)
        print(f"\n[hoà giải] phần chung lấy {args.resolve} của hai người")
    shared = shared.reset_index()

    own = pd.concat([A[A.phan == "RIENG"], B[B.phan == "RIENG"]], ignore_index=True)
    allrows = pd.concat([shared, own], ignore_index=True)

    n0 = len(allrows)
    allrows = allrows[~allrows._trong]
    bad = allrows.ghi_chu.str.contains("|".join(DROP_NOTES), na=False)
    print(f"[lọc] bỏ {n0-len(allrows)} câu chưa gán, {int(bad.sum())} câu đánh dấu loại/không rõ")
    allrows = allrows[~bad]
    empty = allrows[ASP].sum(axis=1) == 0
    if empty.any():
        print(f"[lọc] bỏ thêm {int(empty.sum())} câu không có nhãn nào")
        allrows = allrows[~empty]

    out = pd.DataFrame({"sentence": allrows.cau.astype(str), "language": "vi"})
    for a in OUT_ORDER:
        out[a] = allrows[a].values
    out["label_source"] = "gold"
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    card = out[OUT_ORDER].sum(axis=1)
    print(f"\n[đã lưu] {args.out}  {len(out):,} câu")
    print("  số nhãn/câu:", card.value_counts().sort_index().to_dict())
    print("  theo khía cạnh:", out[OUT_ORDER].sum().to_dict())
    print(f"\nCâu để đưa vào paper: “Two annotators independently labelled {len(ca)} shared "
          f"sentences; mean Cohen's kappa across the nine aspects was {mean_k:.2f}.”")


if __name__ == "__main__":
    main()
