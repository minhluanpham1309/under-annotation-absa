# Data

The datasets are **not** distributed in this repository. They are available from the corresponding
author upon reasonable request. The scripts expect the following layout at the repository root:

```
vn_hotel_res_aspect/            source corpus (EN/VI/ZH/JA), one sentence per row
    train.csv  validation.csv  test.csv
    columns: sentence, language, overall, amenities, food, service, location,
             ambiance, room, price, cleanliness   (binary 0/1)
data_unilever/splits/
    transfer_test.csv           human-verified Vietnamese target set (892 sentences, same columns)
    train.csv  validation.csv   target-domain silver labels (ablation only)
dataset_luan/                   review-level release of the source corpus (review_id, source,
    B_train.csv ...             place_type, sentence, aspect); only needed to rebuild
                                data/test_review_overlap.csv for the review-disjoint analysis
data_external/                  SemEval test sets mapped to the nine aspects (see below)
```

## SemEval external test sets

The SemEval ABSA data are distributed by the task organisers (SemEval-2015 Task 12, SemEval-2016 Task 5).
Download the following files into `data_external/raw/` and run the mapping script:

| File | Set |
|---|---|
| `ABSA15_Hotels_Test.xml` | SemEval-2015 hotel test set (English) |
| `EN_REST_SB1_TEST.xml.gold` | SemEval-2016 SB1 restaurant test set, gold (English) |
| `restaurants_dutch_training.xml` | SemEval-2016 SB1 restaurant training set (Dutch) |

```
python data_prep/prepare_semeval.py --src data_external/raw --out data_external
```

The category-to-aspect mapping is documented at the top of `data_prep/prepare_semeval.py`.
