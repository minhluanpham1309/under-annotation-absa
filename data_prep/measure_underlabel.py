"""
Đo mức GÁN THIẾU của bộ vn_hotel: so nhãn gốc với nhãn vàng do hai người gán lại
trên cùng 200 câu tiếng Việt, theo đúng hướng dẫn đa nhãn.

    python src/measure_underlabel.py \
        --a data_unilever/gold/vong2_vnhotel_A.xlsx \
        --b data_unilever/gold/vong2_vnhotel_B.xlsx \
        --goc data_unilever/gold/_vnhotel200_goc.csv \
        --out results/underlabel_report.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

ASP_SHEET = ["room","cleanliness","service","amenities","location","food","price","ambiance","overall"]
ASP = ["overall","amenities","food","service","location","ambiance","room","price","cleanliness"]


def read_sheet(path, who):
    d = pd.read_excel(path, sheet_name="gan_nhan", skiprows=1)
    d = d[d["phan"].isin(["CHUNG", "RIENG"])].copy()
    for a in ASP_SHEET:
        d[a] = pd.to_numeric(d[a], errors="coerce").fillna(0).astype(int).clip(0, 1)
    d["ghi_chu"] = d.get("ghi_chu", "").fillna("").astype(str).str.strip().str.lower()
    d["_trong"] = (d[ASP_SHEET].sum(axis=1) == 0) & (d.ghi_chu == "")
    if d["_trong"].any():
        print(f"  [cảnh báo] {who}: {int(d._trong.sum())} câu chưa gán")
    return d.set_index("id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--goc", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resolve", choices=["intersection","union","drop"], default="intersection")
    args = ap.parse_args()

    A, B = read_sheet(args.a, "A"), read_sheet(args.b, "B")
    goc = pd.read_csv(args.goc).set_index("id")
    ids = A.index.intersection(B.index).intersection(goc.index)
    A, B, goc = A.loc[ids], B.loc[ids], goc.loc[ids]
    ok = ~(A._trong | B._trong)
    A, B, goc = A[ok], B[ok], goc[ok]
    print(f"\n{len(A)} câu dùng được\n")

    # --- kappa vòng 2
    print("=== ĐỘ ĐỒNG THUẬN GIỮA HAI NGƯỜI (vòng 2) ===")
    ks = []
    for a in ASP_SHEET:
        x, y = A[a].values, B[a].values
        if x.sum() or y.sum():
            k = cohen_kappa_score(x, y); ks.append(k)
            print(f"  {a:12s} kappa {k:.4f}   (A gán {x.sum():3d}, B gán {y.sum():3d})")
    print(f"  kappa trung bình: {np.mean(ks):.4f}\n")

    # --- hoà giải thành nhãn vàng
    if args.resolve == "drop":
        keep = ~(A[ASP_SHEET].values != B[ASP_SHEET].values).any(axis=1)
        gold = A[keep][ASP_SHEET].values; goc2 = goc[keep]
        print(f"[hoà giải] bỏ {int((~keep).sum())} câu bất đồng")
    else:
        op = np.logical_and if args.resolve == "intersection" else np.logical_or
        gold = op(A[ASP_SHEET].values, B[ASP_SHEET].values).astype(int); goc2 = goc
        print(f"[hoà giải] {args.resolve}")
    G = pd.DataFrame(gold, columns=ASP_SHEET, index=goc2.index)[ASP].values
    Ogoc = goc2[ASP].values

    cg, co = G.sum(1), Ogoc.sum(1)
    print("\n=== MỨC GÁN THIẾU CỦA vn_hotel ===")
    print(f"  nhãn/câu — nhãn gốc {co.mean():.2f}   nhãn vàng mới {cg.mean():.2f}   "
          f"(gấp {cg.mean()/co.mean():.2f} lần)")
    print(f"  câu chỉ 1 nhãn — gốc {(co==1).mean():.1%}   mới {(cg==1).mean():.1%}")
    miss = ((G == 1) & (Ogoc == 0)).sum()
    extra = ((G == 0) & (Ogoc == 1)).sum()
    print(f"  nhãn bị BỎ SÓT trong bộ gốc : {miss:,} / {int(G.sum()):,} nhãn vàng ({miss/G.sum():.1%})")
    print(f"  nhãn THỪA trong bộ gốc      : {extra:,}")
    sub = ((G == 1) | (Ogoc == 0)).all(axis=1).mean()
    print(f"  câu mà nhãn gốc là tập con của nhãn vàng: {sub:.1%}")

    print("\n  Coi nhãn gốc như 'dự đoán', nhãn vàng mới là chuẩn:")
    print(f"    micro-P {precision_score(G,Ogoc,average='micro',zero_division=0):.4f}  "
          f"micro-R {recall_score(G,Ogoc,average='micro',zero_division=0):.4f}  "
          f"micro-F1 {f1_score(G,Ogoc,average='micro',zero_division=0):.4f}")

    print("\n=== THEO ĐỘ DÀI CÂU ===")
    w = goc2.sentence.astype(str).str.split().str.len().values
    rows = []
    for lo, hi in zip([0,5,10,15,25],[5,10,15,25,10**6]):
        m = (w > lo) & (w <= hi)
        if m.sum() >= 5:
            rows.append({"do_dai": f"{lo+1}-{hi if hi<10**6 else '+'}", "n": int(m.sum()),
                         "nhan_goc": round(co[m].mean(),2), "nhan_vang": round(cg[m].mean(),2),
                         "bo_sot_%": round(100*((G[m]==1)&(Ogoc[m]==0)).sum()/max(G[m].sum(),1),1)})
    dd = pd.DataFrame(rows); print(dd.to_string(index=False))

    print("\n=== THEO KHÍA CẠNH ===")
    rows = []
    for j, a in enumerate(ASP):
        rows.append({"aspect": a, "goc": int(Ogoc[:,j].sum()), "vang": int(G[:,j].sum()),
                     "bo_sot": int(((G[:,j]==1)&(Ogoc[:,j]==0)).sum()),
                     "thua": int(((G[:,j]==0)&(Ogoc[:,j]==1)).sum())})
    ka = pd.DataFrame(rows).sort_values("bo_sot", ascending=False)
    print(ka.to_string(index=False))

    if args.out:
        ka.to_csv(args.out, index=False, encoding="utf-8-sig")
        pd.DataFrame({"sentence": goc2.sentence.values,
                      **{f"goc_{a}": Ogoc[:,j] for j,a in enumerate(ASP)},
                      **{f"vang_{a}": G[:,j] for j,a in enumerate(ASP)}}
                     ).to_csv(args.out.replace(".csv","_chi_tiet.csv"), index=False, encoding="utf-8-sig")
        print(f"\n[đã lưu] {args.out} và bản chi tiết từng câu")

    print(f"\nCâu cho paper: “Re-annotating {len(G)} Vietnamese sentences from the corpus under a "
          f"strict multi-aspect guideline raised the mean number of aspects per sentence from "
          f"{co.mean():.2f} to {cg.mean():.2f}; {miss/G.sum():.0%} of gold aspect mentions are absent "
          f"from the original annotation.”")


if __name__ == "__main__":
    main()
