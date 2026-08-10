# TRKH 5-Class Internal SSL + Surface Pairwise Audit 2026-06-26

## Muc tieu

- Split dung cho gate: `D:\DataAI\AIEx\newdataset\class_f_groupclean_v1`.
- Khong dung test de tune. Tat ca so lieu ben duoi la validation group-clean.
- Gate full train hien tai: class-1 validation F1 `>= 0.70`.

## Thay doi da them

- Them `foreground_surface_pairwise_head` trong `trkh.models.model`.
- Head moi dung 129 foreground surface stats, sinh score cho cac cap `0-1,1-2,1-4,2-3`.
- Adjustment duoc route theo top-2 logits va probability margin de tranh day class 1 toan cuc.
- Them CLI/config/launcher flags:
  - `--foreground-surface-pairwise-head`
  - `--foreground-surface-pairwise-pairs`
  - `--foreground-surface-pairwise-logit-scale`
  - `--foreground-surface-pairwise-dropout`
  - `--disable-foreground-surface-pairwise-routing`
  - `--foreground-surface-pairwise-route-max-probability-margin`
- Them trace metadata:
  - `foreground_surface_pairwise_stats_shape`
  - `foreground_surface_pairwise_logits`
  - `foreground_surface_pairwise_route_weights`
- Them test `tests/test_foreground_surface_pairwise.py`.

## Kiem chung

- `python -m py_compile trkh\models\model.py trkh\training\train.py trkh\core\config.py trkh\tools\trace_architecture.py`: pass.
- `pytest tests\test_foreground_surface_pairwise.py tests\test_surface_detail_amplification.py -q`: `6 passed`.
- Smoke V38 pass, trace co 5 sample, route weights duoc ghi trong `architecture_trace`.

## Ket qua probe

| Run | Huong | Best val macro F1 | Class-1 F1 | P1 | R1 | Ket luan |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| V35 | Barlow train-only V34 + c1 loss 0.80 | 0.7107 | 0.4096 | 0.3453 | 0.5033 | Best class-1 trong nhom nay |
| V38 | V35 + foreground surface fusion + surface pairwise head | 0.7052 | 0.3936 | 0.3318 | 0.4837 | Reject |
| V39 | Resume Barlow thuan den epoch 6, fine-tune V35 config | 0.7042 | 0.3696 | 0.3017 | 0.4771 | Reject |
| V40 | V35 + angular-margin + pairwise-confusion nhe | 0.7142 | 0.4011 | 0.3363 | 0.4967 | Macro tang, class-1 chua tang |

## Audit loi

- V38 confusion class 1: TP `74`, FP `149`, FN `79`.
  - FP chinh: `0->1=58`, `2->1=62`, `4->1=19`, `3->1=10`.
- V35 confusion class 1: TP `77`, FP `146`, FN `76`.
- Ensemble V35/V37/V38 voi bias class 1 am nhe chi dat class-1 F1 `0.4304`, macro `0.7199`.
- Confidence/PR cua V35 khong cho thay threshold don gian co the vuot gate; one-vs-rest class-1 best F1 chi khoang `0.3909`.

## Ket luan

- Surface stats va surface-pairwise khong du de tach class 1 tren group-clean split.
- Barlow thuan hoc them representation (SSL loss `84.06 -> 70.27`) nhung fine-tune lai giam class-1, nen khong nen keo Barlow dai hon mot cach mu.
- Angular/pairwise confusion giup macro nhung khong giai quyet class-1 precision.
- Diem nghen hien tai la label-boundary/representation cua class 1 voi class 0/2/4, khong phai chi background hay threshold.

## Huong tiep theo

- Uu tien data-cartography train-only cho mau boundary: tach `easy/ambiguous/hard/noisy`.
- Tao quality-group manifest train-only theo lighting/dirty/partial/background-mask va thu group-aware loss nhe.
- Thu mot probe tong hop nho voi class-1 loss thap hon trong V40 de xem precision co phuc hoi khong; neu khong tang class-1 F1 ro ret thi dung huong loss knob.
- Neu data-cartography xac nhan nhieu label boundary mo ho, chuyen sang ambiguous/soft label hoac label review thay vi them head.
