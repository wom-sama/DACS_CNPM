# TRKH class_f/yolo_f valid-mask audit 2026-06-30

## Muc tieu

- Kiem tra gia thuyet ket hop `class_f` va `yolo_f`: dung anh/crop dung nhan de phan loai, dung bbox/ngu canh YOLO de giup model phan biet nen va doi tuong.
- Khong sua raw data. Moi tin hieu tu `yolo_f` chi di qua metadata, bbox prior, source-context auxiliary hoac valid padding mask.
- Sau moi smoke/probe, chay XAI audit nho de xac nhan model co that su tap trung vao qua hay chi an theo nen/border.

## Cap nhat 2026-07-02

- Calibration/reroute hau xu ly khong mo gate: `class1_reroute` train-fit tren train predictions lam val class1 giam xuong `0.6383`; logit-bias train-fit lam val class1 giam xuong `0.6503`; val-fit diagnostic cao nhat chi `0.6879`.
- AIDT non-target/DKD gated da duoc trien khai va probe: `probe_v8_yolof_aidtnckd008_conf70_boundarydrop_bboxprior_120b_2e_20260702` dat best val macro/class1 `0.8818/0.6763`, thap hon best no-pretrain `0.8847/0.6860`. XAI selected20: attention/grad-rollout/Grad-CAM foreground `0.9762/0.9754/0.9869`, background blur/gray drop gan `0`, object desaturate drop `0.0499`; loi van la `3->2`, `0->1`, `1->0`, khong phai nen rong.
- Cumulative ordinal head smoke tren best teacher-focus-binary dat val macro/class1 `0.8831/0.6784`; XAI selected12 foreground tot (`attention/Grad-CAM fg~0.982`) nhung class1 thap hon best va `1->0` tang, nen khong probe 2e voi tham so nay.
- Soft probability ensemble AIDT+TRKH la artifact dau tien vuot AIDT single tren validation: `runs\softensemble_trkh0775_aidt0225_val_20260702` voi weights `TRKH=0.775`, `AIDT=0.225` dat macro/class1 `0.9139/0.7410`, so voi AIDT val `0.9088/0.7169` va TRKH val `0.8847/0.6860`. Confusion ensemble: `0->1=38`, `1->0=13`, `1->2=13`, `3->2=31`, `4->1=3`. Day la validation-selected frozen ensemble, khong phai single no-pretrain TRKH.
- Final test audit sau khi freeze weight khong xac nhan ensemble la artifact tot nhat: TRKH single `runs\eval_yolof_teacherfocusbinary015_best_test_final_20260702` dat test macro/class1 `0.8865/0.6667`; AIDT mapped test `runs\aidt_yolof_mapped_test_metrics_20260702` dat `0.8575/0.5806`; frozen ensemble `runs\softensemble_trkh0775_aidt0225_test_final_20260702` dat `0.8785/0.6494`. Vi vay, best final artifact trong nhom nay van la TRKH single; ensemble chi la diagnostic complementarity/validation baseline.
- Ket luan cap nhat: `yolo_f`/bbox/context da giup foreground focus sach, nhung XAI lap lai rang background perturbation gan nhu vo hai. Loi class 1 chu yeu nam tren surface/illumination/border va ranh label. TRKH single hien tai vuot AIDT tren test, nhung chua dat muc ly tuong class1 `>0.98`; de vuot manh tren validation/final can representation signal moi thay vi lap lai source-context/background/calibration.

## Ket qua probe

| Run | Data/input | Thay doi chinh | Val macro F1 | Val class-1 F1 | Ket luan |
| --- | --- | --- | ---: | ---: | --- |
| `probe_v8_yolof_bboxprior_120b_2e_20260630` | `yolo_f` object crop | bbox spatial prior | `0.8800` | `0.6724` | Best no-pretrain YOLO probe hien tai, van duoi gate `0.70`. |
| `probe_v8_classf_yolopair_bboxprior_120b_2e_20260630` | `class_f` + paired `yolo_f` | paired bbox prior | `0.8799` | `0.6628` | Pairing khop nhung khong hon object-crop YOLO. |
| `probe_v8_classf_yolopair_bboxprior_sourceaux_120b_2e_20260630` | `class_f` + paired `yolo_f` | train-time source-context bbox attention aux | `0.8763` | `0.6494` | Ep attention vao bbox tu anh goc lam giam F1. |
| `probe_v8_classf_yolopair_validmask_bboxprior_120b_2e_20260630` | `class_f` + paired `yolo_f` | valid padding mask + bbox prior | `0.8799` | `0.6628` | Valid-mask khong doi metric so voi paired bbox prior. |
| `probe_v8_yolof_validmask_bboxprior_120b_2e_20260630` | `yolo_f` object crop | valid padding mask + bbox prior | `0.8784` | `0.6647` | XAI sach hon nhung F1 thap hon bbox prior cu. |
| `probe_v8_yolof_bboxprior_boundarydrop_120b_2e_20260630` | `yolo_f` object crop | bbox prior + bbox boundary-band dropout | `0.8821` | `0.6725` | Macro tot hon va class1 nhich rat nho, van duoi gate `0.70`. |
| `probe_v8_yolof_bboxprior_interiordrop_120b_2e_20260630` | `yolo_f` object crop | bbox prior + bbox interior dropout | `0.8794` | `0.6667` | Che loi be mat lam class1 giam; reject. |
| `probe_v8_yolof_parttoken_bboxprior_120b_2e_20260630` | `yolo_f` object crop | bbox prior + full 5-class learned part-token residual | `0.8777` | `0.6648` | Part attention bam foreground nhung qua loang; reject. |
| `probe_v8_yolof_parttokenpair_bboxprior_120b_2e_20260630` | `yolo_f` object crop | bbox prior + part-token pairwise margin head | `0.8770` | `0.6571` | Pairwise BCE gan nhu khong hoc sau probe; reject. |
| `probe_v8_yolof_surfacepair_boundarydrop_bboxprior_120b_2e_20260630` | `yolo_f` object crop | bbox prior + boundary-band dropout + foreground-surface pairwise head | `0.8804` | `0.6628` | Surface pairwise loss/logits gan nhu khong hoc nhanh; reject. |
| `probe_v8_yolof_mixstyle015_boundarydrop_bboxprior_120b_2e_20260630` | `yolo_f` object crop | bbox prior + boundary-band dropout + MixStyle nhe `p=0.15, alpha=0.05` | `0.8789` | `0.6628` | Style regularization nhe khong tang class 1; reject. |
| `probe_v8_yolof_highfreq_boundarydrop_bboxprior_120b_2e_20260630` | `yolo_f` object crop | bbox prior + boundary-band dropout + high-frequency texture expert | `0.8796` | `0.6648` | Texture loss co hoc nhe nhung class 1 van thap; reject. |
| `probe_v8_yolof_boundarycenter_boundarydrop_bboxprior_120b_2e_20260630` | `yolo_f` object crop | bbox prior + boundary-band dropout + boundary-center margin | `0.8776` | `0.6628` | Center loss co terms nhung metric giam; reject. |
| `probe_v8_yolof_pairroute_boundarydrop_bboxprior_120b_2e_20260630` | `yolo_f` object crop | bbox prior + boundary-band dropout + pairwise-margin routing | `0.8841` | `0.6822` | Best YOLO no-pretrain hien tai, chua qua gate `0.70`. |
| `probe_v8_yolof_pairroute_boundarydrop_bboxprior_120b_4e_20260630` | `yolo_f` object crop | cung config pairwise-routing, keo dai 4 epoch/120 batch | `0.8831` | `0.6802` | Best nam o epoch 2; 4e khong hon 2e, reject full train. |
| `route_margin_sweep_val` tren checkpoint 2e | validation-only | sweep route margin `0.08-0.30` | `0.8841` | `0.6822` | Margin `0.16/0.20` tot nhat raw; calibrated class1 cao nhat `0.6933` nhung van duoi gate. |
| `probe_v8_yolof_pairroute_ordboundary_boundarydrop_bboxprior_120b_2e_20260630` | `yolo_f` object crop | pairwise-routing + ordinal-boundary nhe `0.02` | `0.8844` | `0.6822` | Macro tang nhe nho class 2/4, class1 khong doi; reject full train. |
| `pairwise_scale_sweep_val` tren checkpoint ordinal | validation-only | sweep pairwise logit scale `0.15-0.40` | `0.8839` | `0.6839` | Scale `0.35` tang recall class1 nhung precision giam; chua du gate. |
| `probe_v8_yolof_pairroute035_boundarydrop_bboxprior_120b_2e_20260630` | `yolo_f` object crop | train lai pairwise-routing voi logit scale `0.35` | `0.8833` | `0.6819` | Recall class1 tang, precision giam; khong hon scale `0.25`. |
| `probe_v8_yolof_pairroute_localzoom_c1fp_boundarydrop_bboxprior_120b_2e_20260630` | `yolo_f` object crop | pairwise-routing scale `0.25` + local-zoom branch route `0-1,4-1` | `0.8824` | `0.6766` | Local detail branch hoc aux loss nhung lam giam recall class1; reject. |
| `probe_v8_yolof_pairroute_interiorboundary_objgate_c1fp_boundarydrop_bboxprior_120b_2e_20260630` | `yolo_f` object crop | pairwise-routing + objectness-gated interior/boundary pairwise head route `0-1,4-1` | `0.8813` | `0.6706` | Foreground gate giam bbox block trong smoke nhung probe van thap; aux loss dung o `0.6914`, trace bam crop/bbox border; reject. |
| `probe_v8_yolof_pairroute_objecterase_boundarydrop_bboxprior_120b_2e_20260630` | `yolo_f` object crop | pairwise-routing + bbox object-erasure negative loss | `0.8799` | `0.6760` | Tang recall class1 nhung lam precision giam; reject. |
| `probe_v8_yolof_pairroute_mixedclassf05_boundarydrop_bboxprior_120b_2e_20260630` | `yolo_f` primary + `class_f` aux | mixed train concat, aux weight `0.5` | `0.8810` | `0.6628` | Tron them class_f crop lam class1 giam; reject. |
| `probe_v8_yolof_pairroute_pairviewclassf_boundarydrop_bboxprior_120b_2e_20260630` | `yolo_f` primary + paired `class_f` crop | pair-view supervised consistency | `0.8813` | `0.6705` | Pair-view khong hon YOLO object crop; reject. |
| `probe_v8_yolof_pairroute_c1fpsmooth025_boundarydrop_bboxprior_120b_2e_20260630` | `yolo_f` object crop | smooth false-positive margin cho class `0,4 -> 1` | `0.8841` | `0.6822` | Hoa best pairwise-routing, khong giam `0->1/4->1`; khong mo gate. |
| `probe_v8_yolof_pairroute_surfaceamp_boundarydrop_bboxprior_120b_2e_20260630` | `yolo_f` object crop | pairwise-routing + surface-amplified supervised view | `0.8835` | `0.6746` | Surface loss kich hoat nhung lam class1 giam; reject. |
| `probe_v8_yolof_pairroute035_objecterase_surfaceamp_boundarydrop_bboxprior_120b_2e_20260701` | `yolo_f` object crop | stacking nhe tu complementarity: pairroute scale `0.35` + object erasure + surfaceamp | `0.8812` | `0.6686` | Complementarity khong chuyen thanh single-model train signal; reject. |
| `probe_v8_yolof_pairroute_fgbg015_boundarydrop_bboxprior_120b_2e_20260701` | `yolo_f` object crop | foreground-background mix p=`0.15` + pairwise-routing/bbox prior | `0.8838` | `0.6836` | Nhich class1 nho nhat nho recall `0.8013`, nhung van duoi gate `0.70`; chua full train. |
| `probe_v8_yolof_pairroute_fgbg025_boundarydrop_bboxprior_120b_2e_20260701` | `yolo_f` object crop | foreground-background mix p=`0.25` | `0.8808` | `0.6744` | Mix qua manh lam class1 giam; reject va khong sweep mix them neu khong co signal moi. |
| `probe_v8_yolof_pairroute_paired_classf_boundarydrop_bboxprior_120b_2e_20260701` | `yolo_f` primary + paired `class_f` crop | paired-view CE/KL/feature consistency: YOLO object/bbox view + class_f crop view | `0.8824` | `0.6744` | Mapping 9215/9215 dung, loss active, nhung class1 giam so voi pairroute; reject. |

Best single no-pretrain YOLO probe trong nhanh hien tai la `probe_v8_yolof_pairroute_boundarydrop_bboxprior_120b_2e_20260630`: val macro/class1 `0.8841/0.6822`. Chua du dieu kien full train theo gate class-1 validation F1 `>=0.70`, nhung day la huong dau tien vuot ro boundary-band/bbox-prior trong cum probe ngay 2026-06-30.

Keo dai cung config len 4 epoch khong giup: best epoch van la 2 voi val macro/class1 `0.8831/0.6802`; epoch 3-4 tang recall class 1 len `0.7881` nhung precision giam ve `0.5920 -> 0.5862`, lam F1 di xuong. Sweep routing threshold cung khong du: raw class1 F1 cao nhat `0.6822`, calibrated class1 F1 cao nhat `0.6933` voi coverage `0.9800`, van chua qua gate `0.70`.

Ordinal-boundary nhe va pairwise-scale manh hon deu khong mo gate. Ordinal-boundary chi tang macro len `0.8844` nhung class1 giu `0.6822`; scale `0.35` tang class1 recall `0.7748 -> 0.7881` nhung precision giam `0.6094 -> 0.6010`, F1 thap hon best. Ket luan: can them signal/route phan biet false-positive vao class 1, khong chi day manh boundary hien co.

Local-zoom route chi nham `0-1,4-1` cung khong mo gate: best epoch `1`, val macro/class1 `0.8824/0.6766`, class-1 P/R `0.6129/0.7550`. So voi pairwise-routing best, precision class 1 tang nhe `0.6094 -> 0.6129` nhung recall giam `0.7748 -> 0.7550`, nen F1 giam. Aux loss local-zoom giam `4.7023 -> 3.0814`, tuc branch co hoc nhung tin hieu crop/detail khong dung vung phan biet class 1 trong probe ngan.

Interior-boundary objectness-gated branch cung khong mo gate. Run `probe_v8_yolof_pairroute_interiorboundary_objgate_c1fp_boundarydrop_bboxprior_120b_2e_20260630` dat best epoch `2`, val macro/class1 `0.8813/0.6706`, class-1 P/R `0.6032/0.7550`. Confusion van giu mau cu: `3->2=52`, `0->1=47`, `1->0=20`, `4->1=13`, `1->2=12`, `2->1=11`. `train_interior_boundary_pairwise_loss` khong giam (`0.69140625` ca 2 epoch), logits trace class 1 chi khoang `[0.0015, 0.0002]`, tuc head gan nhu chua hoc signal phan biet. Visual trace class 1 cho thay foreground mask da bot bbox thuan so voi smoke dau tien nhung boundary weight van bam vien crop/bbox va nen sat trai/duoi, khong phai ranh surface/maturity tren qua.

Boundary-band dropout cap nhat best macro nhe thanh `0.8821`, nhung class-1 F1 chi `0.6725`, gan nhu hoa voi bbox-prior cu va van khong mo gate.

Object-erasure negative view tren best pairwise-routing khong mo gate. Run `probe_v8_yolof_pairroute_objecterase_boundarydrop_bboxprior_120b_2e_20260630` dat val macro/class1 `0.8799/0.6760`, class-1 P/R `0.5845/0.8013`. Recall class 1 tang nhung false-positive vao class 1 tang manh, nen F1 van thap hon pairwise-routing best. XAI da chay sau probe; attention/Grad-CAM van nam tren qua va background perturbation gan `0`, nen day khong phai huong nen/tach-object can lap lai.

Ket hop them `class_f` vao best YOLO route khong giup. Mixed train concat `probe_v8_yolof_pairroute_mixedclassf05_boundarydrop_bboxprior_120b_2e_20260630` dat val macro/class1 `0.8810/0.6628`; pair-view `class_f` crop `probe_v8_yolof_pairroute_pairviewclassf_boundarydrop_bboxprior_120b_2e_20260630` dat `0.8813/0.6705`. Ca hai deu thap hon YOLO object-crop pairwise-routing, xac nhan `class_f` crop khong tao supervision bo sung huu ich trong cau hinh nay.

Focused false-positive smooth margin cho `0,4 -> 1` chi hoa best, khong cai thien. Run `probe_v8_yolof_pairroute_c1fpsmooth025_boundarydrop_bboxprior_120b_2e_20260630` dat val macro/class1 `0.8841/0.6822`, class-1 P/R `0.6094/0.7748`, voi confusion `0->1=49`, `4->1=12`, gan nhu trung best pairwise-routing. Loss co kich hoat (`train_focused_false_positive_margin_loss` khoang `0.10-0.11`) nhung khong dich duoc false-positive pattern, nen khong nen tang tiep cung loss truoc khi co signal moi.

Surface-amplified supervised view theo huong HERBS/WS-DAN cung khong mo gate. Smoke pass, trace completed, XAI smoke foreground/background tot (`attention 0.9554/0.0446`, `Grad-CAM 0.9580/0.0420`). Probe `probe_v8_yolof_pairroute_surfaceamp_boundarydrop_bboxprior_120b_2e_20260630` dat best epoch `1`, val macro/class1 `0.8835/0.6746`, class-1 P/R `0.6096/0.7550`; epoch `2` giam ve `0.8816/0.6725`. `train_surface_amplified_supervised_loss` co kich hoat `1.16 -> 1.20`, fraction khoang `0.352`, boundary terms khoang `11.26`, nhung recall/precision class1 khong vuot pairwise-routing best. XAI probe foreground/background: attention `0.9426/0.0574`, grad-rollout `0.9613/0.0387`, Grad-CAM `0.9603/0.0397`; robustness background blur/gray gan `0`, object desaturate `0.0455`. Confusion van `3->2=51`, `0->1=47`, `1->0=21`, `4->1=13`, `1->2=12`. Ket luan: surface amplification dang lam regularizer be mat hop ly ve XAI, nhung khong tao boundary signal moi cho class 1.

## XAI sau probe

### `class_f` + paired `yolo_f` valid-mask

- `xai_audit_val_paired_small`: attention foreground/background/border mass `0.8521/0.1479/0.1466`.
- Grad-CAM foreground/background/border mass `0.7721/0.2279/0.2545`.
- Robustness drop trung binh: background blur `0.0018`, background gray `0.0003`, center occlusion `0.0249`, object desaturate `0.1489`.
- Flags: near-tie `6/20`, object-color-sensitive `12/20`, register divergent `20/20`.
- Confusion chinh van la `3->2`, `0->1`, `1->0`, `1->2`, `4->1`.

### `yolo_f` valid-mask

- `xai_audit_val_yolo_small`: attention foreground/background/border mass `0.9357/0.0643/0.1556`.
- Grad-CAM foreground/background/border mass `0.9609/0.0391/0.1840`.
- Robustness drop trung binh: background blur `0.0004`, background gray `-0.0006`, center occlusion `0.0127`, object desaturate `0.0518`.
- Flags giam ro: attention background `2/20`, gradcam background `1/20`, object-color-sensitive `3/20`.
- Metric van khong tang: class-1 F1 `0.6647`, duoi `yolo_f` bbox-prior khong valid-mask `0.6724`.

### `yolo_f` boundary-band bbox dropout

- Them mode `bbox_foreground_dropout_mode=random|interior|boundary_band`; default `random` giu hanh vi cu.
- Probe `boundary_band` dung cung tham so bboxdrop cu: loss `0.08`, consistency `0.03`, probability `0.45`, area `0.04-0.14`, temperature `1.2`.
- Val macro/class1 `0.8821/0.6725`, class-1 P/R `0.5979/0.7682`, confusion chinh: `3->2=55`, `0->1=49`, `1->0=18`, `4->1=14`, `1->2=11`.
- XAI: attention foreground/background/border `0.9805/0.0195/0.1031`, Grad-CAM `0.9285/0.0715/0.1928`.
- Robustness: background blur `0.0005`, background gray `-0.0002`, center occlusion `0.0180`, object desaturate `0.0912`.
- Truc quan case `0->1`, `1->0`, `4->1`: heatmap chu yeu nam tren qua/vet/sang, nen tiep tuc cho thay loi la boundary/surface/label chua khong phai nen.

### `yolo_f` interior bbox dropout

- Probe `interior` dung cung tham so voi boundary-band de tach vai tro loi be mat va bien bbox.
- Val macro/class1 `0.8794/0.6667`, class-1 P/R `0.5888/0.7682`, thap hon boundary-band va bbox-prior cu.
- XAI gan nhu tuong tu boundary-band: attention foreground/background/border `0.9802/0.0198/0.1044`, Grad-CAM `0.9269/0.0731/0.2234`.
- Ket luan: che loi be mat khong giup; model can surface/interior signal de phan biet class 1. Boundary-band it hai hon va co the regularize nhe, nhung khong du dot pha.

### `yolo_f` learned part-token residual

- Them `AdaptivePartTokenLearner` theo huong TransFG/TokenLearner: hoc 4 part token tu patch foreground/bbox, zero-init residual, route tren cac cap `0-1,1-2,4-1,2-3`.
- Probe full residual 5-class: val macro/class1 `0.8777/0.6648`, thap hon bbox-prior va boundary-band.
- XAI probe: attention foreground/background/border `0.9500/0.0500/0.1243`, Grad-CAM `0.9170/0.0830/0.2098`; background blur/gray gan `0`, object desaturate `0.0897`.
- Trace part-token cho thay foreground mass `~0.79-0.85` nhung `max_weight` chi `~0.01-0.015` tren 256 patch, tuc attention phu ca qua thay vi tim duoc part/chi tiet sac net.
- Ket luan: them token residual toan lop lam nhiu phan phoi va khong giai quyet cac cap `3->2`, `0->1`, `1->0`, `4->1`, `1->2`.

### `yolo_f` part-token pairwise margin

- Them `part_token_pairwise_head`: dung cung foreground/bbox part token nhung chi sinh margin cho cac cap route, khong residual 5-class toan cuc.
- Probe val macro/class1 `0.8770/0.6571`, class-1 P/R `0.5816/0.7550`; thap hon ca full part-token.
- Loss phu `train_part_token_pairwise_loss` giu gan `0.6914` trong 2 epoch, cho thay binary head chua hoc duoc signal huu ich trong probe ngan.
- XAI probe: attention foreground/background/border `0.9761/0.0239/0.1185`, Grad-CAM `0.9723/0.0277/0.2606`; background blur/gray gan `0`, object desaturate `0.0858`.
- Trace route vector hoat dong dung shape, nhung chi kich hoat tren mau gan `0-1`; attention van loang va khong tao cai thien precision class 1.
- Ket luan: reject huong part-token hien tai. Neu tiep tuc part-level, can co train signal lam sac attention/part contrastive manh hon, khong chi them residual/route head.

### `yolo_f` foreground-surface pairwise + boundary-band dropout

- Probe dung bbox spatial prior, boundary-band foreground dropout va `foreground_surface_pairwise_head` tren cac cap `0-1,1-2,1-4,2-3`.
- Val macro/class1 `0.8804/0.6628`, class-1 P/R `0.5907/0.7550`; thap hon bbox-prior/boundary-band best `0.6725`.
- Confusion chinh van giu mau hinh cu: `3->2=54`, `0->1=51`, `1->0=19`, `4->1=14`, `1->2=12`.
- XAI probe: attention foreground/background/border `0.9798/0.0202/0.1053`, Grad-CAM `0.9223/0.0777/0.2436`.
- Robustness: background blur `0.0001`, background gray `0.0001`, center occlusion `0.0211`, object desaturate `0.0939`.
- `train_foreground_surface_pairwise_loss` giu gan `0.6914`, trace logits pairwise chi khoang `0.002`, nen head nay khong tao du tin hieu trong probe ngan.
- Ket luan: reject. Khong nen chi tang scale/loss cua cung head khi logits/loss chua cho thay hoc nhanh va metric class 1 giam.

### `yolo_f` MixStyle nhe + boundary-band dropout

- Doi chieu MixStyle/domain-generalization feature statistics; chon muc nhe de tranh pha mau maturity: `probability=0.15`, `alpha=0.05`.
- Probe val macro/class1 `0.8789/0.6628`, class-1 P/R `0.5867/0.7616`; thap hon bbox-prior/boundary-band.
- Confusion: `3->2=55`, `0->1=48`, `1->0=19`, `4->1=16`, `2->1=13`, `1->2=10`.
- XAI probe: attention foreground/background/border `0.9810/0.0190/0.0970`, Grad-CAM `0.9252/0.0748/0.2094`.
- Robustness: background blur `0.0010`, background gray `0.0003`, center occlusion `0.0212`, object desaturate `0.0931`.
- Ket luan: reject. Style-stat regularization nhe khong lam giam confusion boundary class 1; neu dung nua can signal surface/texture ro hon, khong chi tron statistics.

### `yolo_f` high-frequency texture expert + boundary-band dropout

- Them expert texture/tan so cao da co san, route tren cac cap `0-1,1-2,2-3,1-4,4-rest`, voi CE phu `0.03` va pairwise phu `0.03`.
- Probe val macro/class1 `0.8796/0.6648`, class-1 P/R `0.5859/0.7682`; thap hon bbox-prior/boundary-band best.
- Loss phu co hoc nhe trong 2 epoch: high-frequency aux `5.1654 -> 4.3039`, pairwise `0.6881 -> 0.6687`.
- Confusion: `3->2=53`, `0->1=51`, `1->0=17`, `4->1=15`, `1->2=12`, `2->1=12`.
- XAI probe: attention foreground/background/border `0.9773/0.0227/0.1222`, Grad-CAM `0.9722/0.0278/0.2665`.
- Robustness: background blur `0.0003`, background gray `0.0001`, center occlusion `0.0162`, object desaturate `0.0876`.
- Ket luan: reject. Texture expert khong lam giam confusion class 1; huong surface/detail don le tiep tuc bi gioi han boi ranh nhan/maturity mo.

### `yolo_f` boundary-center margin + boundary-band dropout

- Thu boundary-center objective tren cac cap `0-1,1-2,2-3,1-4` voi source `head,patch`, margin `0.10`, temperature `0.20`, compactness `0.10`.
- Probe val macro/class1 `0.8776/0.6628`, class-1 P/R `0.5867/0.7616`; thap hon bbox-prior/boundary-band best.
- Loss co terms on dinh (`~102.4`) nhung khong dich boundary class 1 theo huong tot; confusion chinh van `3->2`, `0->1`, `1->0`, `4->1`.
- XAI probe: attention foreground/background/border `0.9815/0.0185/0.0986`, Grad-CAM `0.9747/0.0253/0.2386`; background blur/gray gan `0`, object desaturate `0.1040`.
- Ket luan: reject. Keo center feature khong giai quyet duoc ranh maturity/surface va co the lam phang chi tiet can cho class 1.

### `yolo_f` pairwise-margin routing + boundary-band dropout

- V8 da co pairwise-margin head tren `0-1,1-2,2-3,4-rest`; bien the nay bat routing de head cap chi can thiep khi top-2 prediction dang mo ho.
- Probe val macro/class1 `0.8841/0.6822`, class-1 P/R `0.6094/0.7748`; tot nhat trong cac probe YOLO no-pretrain hien tai.
- Confusion: `3->2=52`, `0->1=49`, `1->0=18`, `4->1=12`, `1->2=11`; giam `3->2` va `4->1`, nhung `0->1` van la nut lon.
- Pairwise-margin loss giam nhe `0.5802 -> 0.5684`; class1 tang tu epoch 1 `0.6726` len epoch 2 `0.6822`.
- XAI probe: attention foreground/background/border `0.9812/0.0188/0.1086`, Grad-CAM `0.9269/0.0731/0.2111`; background blur/gray gan `0`, object desaturate `0.0801`.
- Probe 4e/120b khong cai thien: best epoch `2`, val macro/class1 `0.8831/0.6802`, class-1 P/R `0.6062/0.7748`. Epoch 3-4 day recall len `0.7881` nhung precision tut, nen F1 giam.
- XAI 4e: attention foreground/background/border `0.9833/0.0167/0.1111`, Grad-CAM `0.9253/0.0747/0.2146`; background blur `0.0006`, background gray `-0.0004`, center occlusion `0.0141`, object desaturate `0.0806`.
- Sweep route margin validation-only tren checkpoint 2e: margin `0.16/0.20` tot nhat raw voi class1 `0.6822`; calibrated best class1 `0.6933` voi coverage `0.9800`, van duoi gate `0.70`.
- Ordinal-boundary nhe tren `0,1,2,3` (`loss=0.02`, weights `1.5,1.3,1.0`) cho val macro/class1 `0.8844/0.6822`: macro nhich nho, class1 khong doi. XAI: attention foreground/background/border `0.9801/0.0199/0.1266`, Grad-CAM `0.9798/0.0202/0.2328`; background perturbation gan `0`, object desaturate `0.0439`.
- Pairwise scale sweep tren checkpoint ordinal: scale `0.35` co raw class1 cao nhat `0.6839` trong eval-only, nhung train lai scale `0.35` cho val macro/class1 `0.8833/0.6819`, class-1 P/R `0.6010/0.7881`.
- XAI scale `0.35`: attention foreground/background/border `0.9770/0.0230/0.1233`, Grad-CAM `0.9823/0.0177/0.2019`; background blur/gray gan `0`, object desaturate `0.0397`.
- Ket luan: chua mo full-train gate, nhung routing la thanh phan nen giu. Cac bien the chi day manh boundary hien co lam trade precision/recall, chua giam du false-positive `0->1` va `4->1`.

### `yolo_f` local-zoom false-positive class1 route

- Them `LocalZoomImageExpert`: crop detail tu anh da normalize, residual logits zero-init, route gioi han tren `0-1,4-1`, logit scale `0.12`, aux loss `0.03`.
- Focused tests pass; preflight va smoke pass. Smoke XAI: attention foreground/background/border `0.9429/0.0571/0.1851`, Grad-CAM `0.9568/0.0432/0.1315`; background blur/gray gan `0`.
- Probe best epoch `1`, val macro/class1 `0.8824/0.6766`, class-1 P/R `0.6129/0.7550`, confusion `0->1=47`, `1->0=20`, `1->2=12`, `4->1=12`.
- Epoch 2 khong cai thien: macro `0.8814`; local-zoom aux loss giam `4.7023 -> 3.0814`, pairwise loss giam `0.5812 -> 0.5733`, nhung recall class 1 thap hon pairwise-routing best.
- XAI probe: attention foreground/background/border `0.9426/0.0574/0.1799`, Grad-CAM `0.9648/0.0352/0.1893`; robustness background blur `0.0002`, background gray `-0.0000`, center occlusion `0.0041`, object desaturate `0.0409`.
- Trace `09j/09k_local_zoom_*` cho thay score/crop con loang va co luc chon vung goc/da day qua; case `0->1` co attention roi vao cuong/day va nen sat qua, case `4->1` Grad-CAM co diem nong tren qua nhung cung nhieu response o vat the lan can.
- Ket luan: reject. Local-zoom tu do co xu huong tang precision nhe bang cach si class 1, nhung bo them true class 1. Neu tiep tuc detail branch, phai rang buoc bang interior/object mask hoac anti-class1-FP route dua tren interior-vs-boundary score, khong crop tu do.

### `yolo_f` interior-boundary objectness gate

- Them `InteriorBoundaryPairwiseHead`: tach foreground/interior/boundary tu RGB/HSV/local contrast, co bbox mask neu co YOLO metadata, route chi tren `0-1,4-1`, residual zero-init va loss BCE nhe `0.03`.
- Smoke objectness-gate cai thien trace so voi bbox mask thuan: foreground fraction class 1 giam tu khoang `0.436` xuong `0.371`, khong con full rectangle, nhung van co block nen/border.
- Probe best epoch `2`: val macro/class1 `0.8813/0.6706`, class-1 P/R `0.6032/0.7550`; thap hon best pairwise-routing `0.8841/0.6822`.
- XAI probe: attention foreground/background/border `0.9672/0.0328/0.1966`, Grad-CAM `0.9686/0.0314/0.2212`. Robustness: background blur `-0.0000`, background gray `0.0005`, center occlusion `0.0044`, object desaturate `0.0395`.
- Review flags: attention border `6/20`, Grad-CAM border `9/20`, near-tie `8/20`, register/CLS divergent `20/20`. Top confusions van la `3->2`, `0->1`, `1->0`, `4->1`, `1->2`.
- Trace class 1: `interior_boundary_foreground_fraction=0.3708`, `interior=0.1438`, `boundary=0.2271`, `bbox=0.5049`, route weight `0.3729` cho pair `0-1`; logits rat nho va loss khong giam.
- Ket luan: reject. Huong nay khong nen tiep tuc bang cach tang loss/scale, vi loi la mask/feature dang bam crop/bbox border va branch khong hoc nhanh, khong phai thieu do manh residual.

### Complementarity validation cac checkpoint no-pretrain gan best

- Export validation probabilities cho 6 probe gan best: `pairroute`, `c1fpsmooth`, `objecterase`, `localzoom`, `pairroute035`, `surfaceamp`. Tat ca chi dung split validation de phan tich, khong dung test de tune.
- Single-checkpoint class-1 F1: `pairroute=0.6822`, `c1fpsmooth=0.6822`, `objecterase=0.6723`, `localzoom=0.6766`, `pairroute035=0.6819`, `surfaceamp=0.6746`.
- Best weighted ensemble buoc `0.1` theo class1 dung weights `c1fpsmooth=0.2`, `objecterase=0.4`, `pairroute035=0.3`, `surfaceamp=0.1`: val macro/class1 `0.8854/0.6934`, class-1 P/R `0.6111/0.8013`.
- Focused grid buoc `0.05` tot nhat gan tuong tu: `pairroute=0.1`, `objecterase=0.4`, `pairroute035=0.4`, `surfaceamp=0.1`, val macro/class1 `0.8851/0.6934`.
- Bias sweep class 1 khong giup; bias tot nhat la `0.0`. Tang bias chi tang recall doi lay precision va lam class1 F1 giam.
- Oracle any-model tren 6 checkpoint dat val macro/class1 `0.9002/0.7273`, tuc co loi sai duoc model khac cuu, nhung weighted/router don gian chua trich duoc signal de qua gate.
- Overlap voi `pairroute`: co `34` false negative class1, trong do `objecterase` cuu `6`, `pairroute035` cuu `2`, `localzoom` cuu `1`, `surfaceamp` cuu `1`; voi `75` false-positive vao class1, `objecterase/localzoom/surfaceamp` moi cai thien moi loai `6`.
- Top confusion cua best focus ensemble van la `0->1=54`, `3->2=52`, `1->0=14`, `0->4=12`, `1->2=11`.
- Ket luan: complementarity co that nhung chua du gate. Huong thu tiep hop ly nhat la mot smoke/probe stacking nhe cua cac thanh phan co tin hieu khac nhau (`objecterase` tang recall, `pairroute035` tang recall boundary, `surfaceamp` giu foreground/XAI tot) voi weight thap. Neu stacking khong qua `0.70`, dung nhom regularizer nay va chuyen sang signal khac.

### `yolo_f` complementarity stacking nhe

- Preflight, compile/focused pytest, smoke, trace va XAI smoke deu pass cho combo `pairroute035 + objecterase + surfaceamp` voi weight thap.
- Smoke history xac nhan cac loss kich hoat: object-erasure fraction `0.3125`, surface-amplified fraction `0.3125`, pairwise margin loss `0.5820`; smoke XAI attention/Grad-CAM foreground/background `0.9236/0.0764` va `0.9495/0.0505`.
- Probe `probe_v8_yolof_pairroute035_objecterase_surfaceamp_boundarydrop_bboxprior_120b_2e_20260701` chon best epoch `1`, val macro/class1 `0.8812/0.6686`, class-1 P/R `0.6000/0.7550`; epoch `2` giam macro `0.8790`.
- Confusion best: `0->1=51`, `3->2=48`, `1->0=19`, `1->2=13`, `0->4=13`, `4->1=11`. So voi best pairroute, `0->1` tang va class1 recall giam, nen khong co loi ich thuc.
- XAI probe van sach: attention foreground/background/border `0.9424/0.0576/0.1648`, grad-rollout `0.9615/0.0385/0.1816`, Grad-CAM `0.9667/0.0333/0.1894`; background blur/gray drop gan `0`, object desaturate `0.0444`.
- Ket luan: complementarity cua checkpoint khong tu dong chuyen thanh train-time stacking trong mot student. Khong lap lai stacking regularizer surface/object-erasure/pairwise scale nay; can chuyen sang signal khac co kha nang thay doi representation, khong chi day precision-recall tren logits.

### `yolo_f` foreground-background mix

- Thu train-time foreground-background recombination vi user dat gia thuyet context rong giup model hoc tach nen/doi tuong. Khong sua raw data; mix dung bbox/mask metadata trong collate.
- Smoke p=`0.15` pass, trace completed, XAI smoke foreground/background tot: attention `0.9554/0.0446`, Grad-CAM `0.9579/0.0421`.
- Probe p=`0.15` dat best epoch `2`, val macro/class1 `0.8838/0.6836`, class-1 P/R `0.5961/0.8013`. Day la bump class1 nho so voi pairroute `0.6822`, nhung precision giam va van duoi full-train gate `0.70`.
- XAI p=`0.15`: attention/Grad-CAM foreground/background `0.9605/0.0395` va `0.9809/0.0191`; background blur/gray drop gan `0`, object desaturate `0.0442`; top confusions van `3->2=52`, `0->1=51`, `1->0=15`, `4->1=15`, `2->1=11`.
- Probe p=`0.25` dat val macro/class1 `0.8808/0.6744`, class-1 P/R `0.5969/0.7748`; giam ro so voi p=`0.15` va pairroute.
- XAI p=`0.25`: attention/Grad-CAM foreground/background `0.9797/0.0203` va `0.9883/0.0117`; nen duoc tach sach hon nhung metric class1 giam. Background perturbation van gan `0`, object desaturate `0.0419`.
- Ket luan: foreground-background mix co tac dong nhung loi chinh khong phai nen. p=`0.15` co the giu nhu ung vien nho neu sau nay can recall class1, nhung khong mo gate; khong sweep xac suat mix cao hon neu khong them signal moi.

### `yolo_f` primary + paired `class_f` crop view

- Thu dung `yolo_f` lam primary validation/training protocol va attach `class_f` crop nhu paired view train-only qua `--auxiliary-train-classification-folder-yolo-data yolo_f`, dung y tuong "YOLO view hoc doi tuong/bbox, class_f view hoc crop phan loai".
- Mapping that dung: `primary_base_samples=9215`, `auxiliary_base_samples=9215`, `effective_samples=9215`; validation/test van la primary `yolo_f`, khong tune test.
- Smoke pass, trace completed, XAI smoke pass. Paired loss active: supervised CE `0.7715`, KL consistency `0.0036`, feature consistency `0.0507`, paired fraction `1.0`.
- Probe `probe_v8_yolof_pairroute_paired_classf_boundarydrop_bboxprior_120b_2e_20260701` dat best epoch `2`, val macro/class1 `0.8824/0.6744`, class-1 P/R `0.6010/0.7682`; thap hon pairroute baseline `0.8841/0.6822` va fg-bg p=`0.15`.
- Confusion: `3->2=51`, `0->1=50`, `1->0=18`, `4->1=15`, `1->2=12`, `2->1=8`. So voi pairroute, `0->1` va `4->1` khong giam du.
- XAI probe: attention/Grad-CAM foreground/background `0.9492/0.0508` va `0.9686/0.0314`; background blur/gray gan `0`, object desaturate `0.0431`, Grad-CAM border flags `11/20`.
- Ket luan: ket hop `class_f` theo paired-view da duoc xac nhan ve mat pipeline nhung khong them signal huu ich cho class 1 trong cau hinh nhe nay. Khong tang paired-view weight tiep; neu quay lai ket hop data, can la objective moi nhu cross-view masked reconstruction/teacher-free consistency co gate XAI rieng.

### `yolo_f` supervised masked reconstruction train-time

- Them MIM train-time nhe: mask patch foreground/detail/bbox-biased, decoder chi dung train loss, khong sua raw data va khong pretrain offline. Smoke pass, trace completed, XAI smoke pass; loss kich hoat voi mask fraction khoang `0.35`, patch count `256`, bbox prior mean khoang `0.70`.
- Probe `probe_v8_yolof_pairroute_mim003_boundarydrop_bboxprior_120b_2e_20260701` dat best epoch `2`, val macro/class1 `0.8795/0.6705`, class-1 P/R `0.5949/0.7682`; thap hon pairroute best `0.8841/0.6822`, fg-bg p=`0.15` va duoi gate `0.70`.
- Confusion probe: `3->2=55`, `0->1=49`, `1->0=21`, `4->1=15`, `4->0=13`, `1->2=12`, `2->1=11`. MIM khong giam du false-positive vao class 1 va lam macro giam.
- XAI probe: attention foreground/background/border `0.9794/0.0206/0.1166`, Grad-CAM `0.9781/0.0219/0.2166`, rollout `0.9636/0.0364/0.2532`; background blur/gray drop gan `0`, object desaturate drop `0.0452`.
- Review nhanh case `0->1` cho thay Grad-CAM co the van bat diem nong o vung sang/nen sat qua, nhung trung binh robustness khong ung ho ket luan model dang dua vao nen rong. Loi chinh van la ranh surface/maturity `0/1/2` va expert complementarity.
- Ket luan: reject full train. Khong nen tang MIM weight/mask ratio trong cau hinh nay; neu quay lai MIM thi can cross-view/teacher-free objective co bang chung validation moi, khong chi reconstruction RGB.

### `yolo_f` object-level ensemble/router protocol fix

- Khi tao teacher/ensemble tu `yolo_f`, phat hien multi-object image lam nhieu sample co cung `image_path` nhung target/bbox khac nhau. Path-only cache lam collapse object va co the inflate metric.
- Da sua `TeacherProbabilityDataset` va offline distillation loader de uu tien `sample_index` neu CSV co cot nay; path-only van giu cho dataset khong duplicate path. Focused test `tests/test_teacher_probability_dataset.py` pass, kem compile `trkh/data/dataset.py trkh/training/train.py`.
- Grid weighted ensemble dung path-collapsed tung cho class1 `0.7035` khong con duoc xem la gate hop le. Chay lai theo object-level `sample_index` tren 2606 val objects: best focus weights `c1fpsmooth=0.35`, `objecterase=0.25`, `surfaceamp=0.25`, `fgbg015=0.15`, `pairroute=0`, `pairroute035=0` dat val macro/class1 `0.8864/0.6939`, class-1 P/R `0.6198/0.7881`.
- Mean/frozen teacher train cache moi `teacher_probs_train_sampleindex_focus005.csv`: train macro/class1 `0.9404/0.8183`, val macro/class1 `0.8864/0.6939`; test chua dung.
- Ket luan: complementarity co that nhung object-level validation van duoi gate `0.70`; moi route/router/teacher sau nay tren `yolo_f` phai dung `sample_index`, khong duoc map chi bang path.

### `yolo_f` sample-index ensemble distillation student

- Tra cuu/doi chieu Knowledge Distillation: Hinton et al. ung ho nen ensemble -> soft-label teacher; Noisy Student ung ho student co noise/augmentation, nhung trong TRKH chi dung train split va khong them unlabeled/raw data.
- Patch launcher V8 expose `-DistillationTeacherCsv`, `-DistillationWeight`, `-DistillationTemperature`, `-DistillationFocusClassIndex`, `-DistillationFocusClassWeight`; dry-run xac nhan tham so va teacher CSV.
- Smoke `smoke_v8_yolof_pairroute_distillens_sampleindex_boundarydrop_bboxprior_20260701` pass: offline teacher `key_mode=sample_index`, `sample_index_samples=9215`, `duplicate_rows=1151`, `teacher_label_agreement=0.9621`, `train_distillation_loss=0.0614`; trace completed.
- XAI smoke: attention/Grad-CAM foreground/background `0.9429/0.0571` va `0.9564/0.0436`; background blur/gray drop gan `0`, object desaturate drop `0.0238`. Case `0->1` van co hotspot tren qua kem vung nen sang goc duoi phai.
- Probe `probe_v8_yolof_pairroute_distillens_sampleindex_boundarydrop_bboxprior_120b_2e_20260701` best epoch `2`, val macro/class1 `0.8840/0.6860`, class-1 P/R `0.6114/0.7815`. So voi pairroute best `0.8841/0.6822`, class1 tang nhe nhung van duoi gate; so voi teacher object-level `0.6939` student chua bat kip.
- XAI probe: attention/Grad-CAM foreground/background `0.9775/0.0225` va `0.9765/0.0235`; robustness background blur `0.00027`, background gray `-0.00011`, object desaturate `0.0455`. Grad-CAM border flags `11/20`, near-tie `7/20`, top confusions `3->2=53`, `0->1=49`, `1->0=17`, `4->1=12`, `0->4=11`.
- Review overlay `0->1`: foreground chinh nam tren qua nhung van co diem nong lon o nen/vat the sang sat goc duoi phai; loi nay khong hien ro trong robustness trung binh. Distillation soft-label khong loai duoc false-positive class 1 va khong mo gate.
- Thu bien the gioi han them foreground-background mix p=`0.15`: smoke pass, XAI smoke gan trung voi distillation-only. Probe `probe_v8_yolof_pairroute_distillens_fgbg015_sampleindex_boundarydrop_bboxprior_120b_2e_20260701` best epoch `2`, val macro/class1 `0.8838/0.6799`, class-1 P/R `0.5941/0.7947`; thap hon distillation-only `0.6860` va pairroute best `0.6822` ve class1 F1.
- XAI distill+fgbg: attention/Grad-CAM foreground/background `0.9753/0.0247` va `0.9839/0.0161`, background blur/gray drop gan `0`, object desaturate `0.0500`; top confusions xau hon cho class 1 false-positive: `0->1=52`, `4->1=15`, `2->1=11`.
- Ket luan: reject full train va dung huong soft-label ensemble student hien tai. Fg-bg mix tang recall nhung lam precision class 1 tut; distillation khong du sua false-positive `0/4 -> 1`. Huong tiep theo phai la anti-background-hotspot/part-boundary signal co route cu the, khong tang tiep distillation weight hoac mix probability.

### `yolo_f` balanced-softmax va patch objectness

- Balanced-softmax chi duoc dung nhu diagnostic nho tren best teacher-focus-binary route. Smoke + XAI pass, nhung full-val smoke cho thay class-1 FN tang (`1->0=21`, `1->2=14`) va khong giam du false-positive vao class 1; khong chay probe 2e vi khong co signal mo gate.
- Them `PatchObjectnessGuidedHead`: token-level objectness head hoc bbox prior YOLO va residual descriptor gom weighted/mean/max token feature. Muc tieu la dung `yolo_f` nhu signal detect-object, khong sua raw data.
- Smoke pass, trace completed, XAI smoke sach. Probe `probe_v8_yolof_patchobj02_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` dat best epoch `2`, val macro/class1 `0.8851/0.6839`, class-1 P/R `0.6041/0.7881`.
- Confusion probe: `0->1=51`, `3->2=50`, `1->0=16`, `4->1=12`, `1->2=11`, `2->1=11`. So voi best teacher-focus-binary, recall class 1 tang nho nhung precision giam, nen F1 van thap hon `0.6860`.
- Loss audit run dau: `train_patch_objectness_loss` quanh `0.6938 -> 0.6948`, tuc objectness BCE gan nhu chua hoc bbox patch.
- XAI selected20: attention/Grad-CAM foreground/background `0.9832/0.0168` va `0.9854/0.0146`; background blur/gray drop gan `0`, object desaturate drop `0.0345`. Border/uncertainty van con: near-tie `9/20`, Grad-CAM border `10/20`, rollout border `11/20`.
- Thu them mot probe head-only/high-objectness tu checkpoint patch-objectness dau de kiem tra xem loss co hoc khi freeze backbone/head cu hay khong. Run `probe_v8_yolof_patchobj_headonly10_from_patchobj02_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` dat val macro/class1 `0.8805/0.6686`, class-1 P/R `0.5842/0.7815`; `train_patch_objectness_loss` van cao `0.8327 -> 0.8351`.
- XAI head-only selected20: foreground mass van cao nhung border/near-tie xau hon (`Grad-CAM border 12/20`, `rollout border 13/20`, near-tie `8/20`); background blur/gray van gan `0`, object desaturate `0.0392`.
- Ket luan: patch objectness theo weight `0.02` hoac head-only weight `0.10` khong mo gate va khong hoc bbox-token objectness du nhanh. Bbox/objectness lam foreground focus sach hon, nhung loi class 1 tiep tuc la boundary/surface/illumination tren qua, khong phai nen rong; dung huong patch-objectness voi tham so hien tai.

### `yolo_f` bbox-prior patch context head

- Thu `BBoxPriorPatchContextHead` de tao residual classifier tu 4 pooled descriptor: token trong bbox, token ngoai bbox, sai khac object-background va max/object-global token. Khac source-context full-frame, head nay chi so sanh object/background token trong crop YOLO va khong sua raw data.
- Preflight compile + focused tests pass (`10 passed`). Smoke `smoke_v8_yolof_bboxctx_teacherfocusbinary015_boundarydrop_bboxprior_20260702` pass, trace completed, missing keys chi thuoc head moi va duoc allowlist. Smoke XAI full-val selected12 khong co background robustness bat thuong, nen da chay probe.
- Probe `probe_v8_yolof_bboxctx_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` reject: best epoch `2`, val macro/class1 `0.8815/0.6686`, class-1 P/R `0.5959/0.7616`. So voi best teacher-focus-binary `0.8847/0.6860`, ca macro va class 1 deu giam.
- Confusion best: `0->1=49`, `3->2=50`, `1->0=20`, `4->1=13`, `2->1=11`, `1->2=11`; head khong giam false-positive vao class 1 va lam class-1 FN tang.
- XAI selected20: attention/grad-rollout/Grad-CAM foreground-background `0.9367/0.0633`, `0.9543/0.0457`, `0.9631/0.0369`; background blur/gray drop gan `0`, object desaturate drop `0.0436`. Border/uncertainty van con: Grad-CAM border `9/20`, rollout border `10/20`, near-tie `8/20`.
- Ket luan: object-vs-background token pooling trong crop khong tao signal boundary moi va con lam foreground focus kem hon best no-pretrain. Khong lap lai bbox-context scale `0.08`/temperature `0.45` neu khong co loss/route moi ro rang.

### `yolo_f` internal VICReg object-crop SSL warm-start

- Mo rong `trkh.tools.pretrain_internal_barlow` de dung `yolo_f` object crop train-only cho Barlow/VICReg/DINO-style internal SSL. Dry-run xac nhan 9,215 object train, class counts `[1941, 541, 1920, 2520, 2293]`, balanced sampler exposure deu va khong doc val/test.
- Chay `ssl_vicreg_yolof_objectcrop_120b2e_20260702` voi VICReg + SupCon nhe tren train-only object crop. SSL loss giam `32.68 -> 29.80`, best epoch `2`, checkpoint hop le ve artifact nhung chua la metric phan loai.
- Fine-tune tu checkpoint SSL-only theo dung recipe best teacher-focus-binary/bboxprior/boundarydrop. Smoke resume/trace pass, partial load chi thieu `bbox_spatial_fusion_head` va duoc allowlist. XAI smoke cho thay attention foreground cao, background perturb gan `0`, nhung full-val sau 2 batch con rat yeu vi classifier supervised gan nhu phai hoc lai tu dau.
- Probe `probe_v8_yolof_sslvicreg_warmstart_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` reject: raw val macro/class1 o checkpoint best khoang `0.7476/0.3959`; epoch 2 macro/class1 `0.7510/0.3884`. Confusion best: `0->1=77`, `4->1=58`, `1->0=44`, `1->2=24`, ngoai ra `3->2=111`.
- XAI selected20 sau probe: attention/grad-rollout/Grad-CAM foreground-background `0.9658/0.0342`, `0.9753/0.0247`, `0.9772/0.0228`; background blur/gray drop gan `0`, object desaturate drop `0.0632`. Ket luan: SSL-only warm-start khong sai do nen, ma lam mat neo supervised/head class 1; khong lap lai cach thay checkpoint supervised bang SSL-only random-head neu khong co head-preserving merge/adapter hoac train-time SSL consistency tren checkpoint supervised.

### `yolo_f` DINO train-only voi supervised init

- Them `--init-checkpoint` cho `trkh.tools.pretrain_internal_barlow`: load `model_state` tu checkpoint supervised/internal de khoi tao SSL, co guard class-name, khong load optimizer/epoch. Neu `ssl-method=dino`, teacher EMA duoc sync lai sau khi init. Compile pass va `tests/test_internal_ssl_mae.py` pass (`8 passed`).
- Dry-run DINO tren `yolo_f` object-crop tu checkpoint best `teacherfocusbinary015` pass: 9,215 train objects, class counts `[1941, 541, 1920, 2520, 2293]`, balanced exposure hop le, init missing `0`, unexpected `6` key `bbox_spatial_fusion_head` vi SSL model khong bat bbox-spatial head.
- Run `ssl_dino_yolof_objectcrop_supervisedinit_60b1e_20260702`: `60` train-only batches, batch `24`, DINO loss `3.4983`, teacher entropy `1.1496`, center norm `6.3971`, khong doc val/test.
- Fine-tune smoke tu checkpoint DINO nay theo recipe teacher-focus-binary/bboxprior/boundarydrop pass va trace completed, nhung XAI full-val selected12 cho confusion xau: `0->1=52`, `4->1=22`, `2->1=20`, `1->0=19`. So voi best teacher-focus-binary, class 1 bi mo rong sai sang class 4/2.
- XAI smoke DINO: foreground/background van sach (`attention 0.9727/0.0273`, `Grad-CAM 0.9862/0.0138`), background blur/gray gan `0`, object_desaturate `0.0582`. Ket luan: DINO ngan co the lam representation object-focused hon nhung pha boundary class 1; khong chay probe 2e tu DINO 60b/entropy thap nay.

### `yolo_f` surface-counterfactual consistency tu checkpoint supervised

- Thu bien the giu checkpoint supervised `teacherfocusbinary015` lam neo: them train-time surface-counterfactual consistency nhe (`weight=0.02`, probability `0.35`, `foreground_luma`, strength `0.18`, temperature `1.2`) thay vi thay backbone/head bang SSL-only checkpoint.
- Smoke `smoke_v8_yolof_surfacecons02_from_teacherfocusbinary015_boundarydrop_bboxprior_20260702` pass, trace completed va XAI smoke foreground sach. Loss consistency active nhe (`train_surface_counterfactual_consistency_loss` khoang `0.0029`, fraction khoang `0.36`).
- Probe `probe_v8_yolof_surfacecons02_from_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` reject: best epoch `1`, val macro/class1 `0.8828/0.6783`, class-1 P/R `0.6031/0.7748`; epoch 2 giam ve macro/class1 `0.8793/0.6686`.
- Confusion best: `0->1=51`, `3->2=51`, `1->0=18`, `4->1=12`, `1->2=11`, `2->1=10`. So voi best teacher-focus-binary `0.8847/0.6860`, surface-consistency lam class1 giam va khong giam false-positive vao class 1.
- XAI selected20: attention/Grad-CAM foreground-background `0.9809/0.0191` va `0.9836/0.0164`; background blur/gray drop `0.0008/0.0013`, object desaturate drop `0.0439`. Review flags van con border/uncertainty: Grad-CAM border `9/20`, rollout border `12/20`, near-tie `7/20`.
- Ket luan: day la cach SSL/consistency giu neo supervised hop le hon SSL-only, nhung van khong mo gate va khong sua mau loi `0/4 -> 1` hoac `3 -> 2`. Khong full train va khong sweep surface-consistency nhe neu khong co signal boundary/part moi.

### `yolo_f` background-focus suppression

- Them loss background-focus suppression: tao counterfactual nen trong batch, chi phat khi mau am `0,4` van day class 1 va probability class 1 giam qua nhieu sau khi trung hoa nen.
- Smoke pass, loss active nhe; XAI smoke co foreground/background attention `0.9431/0.0569`, Grad-CAM `0.9411/0.0589`, background blur/gray gan `0`, nhung case `0->1` van co hotspot vat the sang sat goc duoi phai.
- Probe `probe_v8_yolof_pairroute_bgfocus035_boundarydrop_bboxprior_120b_2e_20260701`: val macro/class1 `0.8809/0.6686`, class-1 P/R `0.6043/0.7483`; thap hon pairroute best.
- XAI probe: attention/Grad-CAM foreground/background `0.9317/0.0683` va `0.9666/0.0334`; background blur/gray `-0.0006/-0.0025`, object desaturate `0.0372`.
- Ket luan: reject. Suppress theo nen lam metric class 1 giam; background counterfactual tiep tuc cho thay nen khong phai nut that chinh.

### `yolo_f` pairwise-routing explicit `1-4`

- Expose `PairwiseMarginPairs` trong launcher de thu thay `4-rest` bang cap cu the `1-4`: `0-1,1-2,2-3,1-4`.
- Smoke + XAI pass; XAI smoke foreground/background `0.9588/0.0412` attention va `0.9620/0.0380` Grad-CAM, background perturbation gan `0`.
- Probe `probe_v8_yolof_pairroute14_boundarydrop_bboxprior_120b_2e_20260701`: val macro/class1 `0.8814/0.6763`, class-1 P/R `0.6000/0.7748`.
- XAI probe: top confusions `3->2=52`, `0->1=48`, `1->0=17`, `0->4=15`, `4->1=15`; attention/Grad-CAM foreground/background `0.9429/0.0571` va `0.9714/0.0286`.
- Ket luan: reject. `1-4` cu the khong thay the duoc `4-rest`; `4->1` tang len `15`, class1 thap hon best.

### `yolo_f` teacher-gated focus margin

- Them `teacher_focus_margin_loss`: dung `teacher_probs` object-level theo `sample_index`, khong soft-distill toan cuc; chi phat mau hard-label am `0,2,4` khi teacher co `p(class1)<=0.20` va teacher argmax trung hard label.
- Focused tests pass; dry-run xac nhan `distillation_weight=0` va teacher CSV sample-index hop le.
- Smoke pass, loss active: `train_teacher_focus_margin_loss=0.0231`, fraction `0.5469`, `train_distillation_loss=0`.
- XAI smoke: attention/Grad-CAM foreground/background `0.9483/0.0517` va `0.9564/0.0436`; background blur/gray gan `0`.
- Probe `probe_v8_yolof_pairroute_teacherfocus035_boundarydrop_bboxprior_120b_2e_20260701`: val macro/class1 `0.8837/0.6822`, class-1 P/R `0.6094/0.7748`; hoa class1 best pairroute nhung macro thap hon `0.8841`.
- XAI probe sach hon pairroute: attention/Grad-CAM foreground/background `0.9836/0.0164` va `0.9829/0.0171`, background blur/gray gan `0`, object desaturate `0.0395`; top confusions van `0->1=49`, `4->1=12`, `1->0=18`, `1->2=11`.
- Ket luan: chua mo gate. Teacher-gated margin lam focus sach hon nhung khong dich du decision boundary; neu tiep tuc teacher, chi thu combo rat nhe voi distillation hoac focus-binary objective, khong tang margin don le.

### `yolo_f` distillation nhe + teacher-gated focus margin

- Thu dung combo rat nhe vi distillation-only tung dat class1 `0.6860`, con teacher-gated margin lam XAI sach hon: `DistillationWeight=0.05`, `DistillationFocusClassWeight=1.2`, `TeacherFocusMarginLossWeight=0.02`, teacher CSV object-level theo `sample_index`.
- Probe `probe_v8_yolof_pairroute_distill005_teacherfocus02_boundarydrop_bboxprior_120b_2e_20260701` best epoch `2`, val macro/class1 `0.8834/0.6802`, class-1 P/R `0.6062/0.7748`; thap hon distillation-only `0.6860`, teacher-margin-only `0.6822` va pairroute best `0.6822`.
- Loss active nhung khong tao loi ich: epoch 2 `train_distillation_loss=0.0544`, `train_teacher_focus_margin_loss=0.0267`, teacher-focus fraction `0.5323`; pairwise loss van quanh `0.5687`.
- Confusion best van lap lai nut that cu: `3->2=52`, `0->1=50`, `1->0=18`, `1->2=11`, `4->1=12`, `2->1=10`.
- XAI probe: attention/Grad-CAM foreground/background `0.9822/0.0178` va `0.9806/0.0194`; background blur/gray drop gan `0`, object desaturate drop `0.0404`; Grad-CAM border flags `8/20`, near-tie `9/20`. Overlay `0->1` co hotspot chinh tren dau qua va mot vat sang sat mep duoi, khong ung ho gia thuyet nen rong la nut that chinh.
- Ket luan: reject full train. Khong tang them distillation weight hoac teacher-focus margin; neu tiep tuc teacher thi can objective hep hon tren xac suat binary `class1 vs neighbor` de tranh keo toan bo distribution sai huong.

### `yolo_f` teacher focus-binary soft distillation

- Them `teacher_focus_binary_loss`: chuan hoa teacher tren tap neighbor `0,1,2,4`, dung binary logit `class1 vs non-class1-neighbor` va `binary_cross_entropy_with_logits` de an toan voi bf16 AMP. Focused tests va compile pass; smoke ban dau bat duoc loi autocast cua BCE probability, da sua sang BCE-with-logits.
- Smoke `smoke_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_20260701` pass, trace completed; loss active `train_teacher_focus_binary_loss=0.5371`, fraction `0.7969`, student/teacher focus probability `0.2393/0.2441`. XAI smoke foreground/background: attention `0.9417/0.0583`, Grad-CAM `0.9537/0.0463`, background blur/gray gan `0`.
- Probe `probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701` best epoch `2`, val macro/class1 `0.8847/0.6860`, class-1 P/R `0.6114/0.7815`; confusion `3->2=52`, `0->1=49`, `1->0=17`, `4->1=12`, `1->2=11`, `2->1=10`.
- XAI probe: attention/Grad-CAM foreground/background `0.9783/0.0217` va `0.9817/0.0183`; background blur/gray drop gan `0`, object desaturate drop `0.0416`; Grad-CAM border flags `9/20`, near-tie `9/20`. Overlay `0->1` van giong cac run truoc: hotspot dau qua va vat sang sat mep duoi.
- Phan tich teacher object-level: teacher binary target tren train rat mem, class1 mean `0.3838`, class0/2/4 mean `0.2029/0.1906/0.1875`; tren val class1 mean `0.3413`, class0/2/4 mean `0.2127/0.1901/0.1870`. Teacher confidence trung binh chi khoang `0.30-0.33`, nen gate confidence cao se lam mat nhieu mau.
- Ket luan: focus-binary soft ngang distillation-only ve class1 nhung van duoi gate `0.70`; khong full train. Neu tiep tuc, can blend nhe hard binary label voi teacher binary target de tang separability, khong chi match soft teacher mem.

### `yolo_f` hard-blend, top-k reassessment va c1 false-positive combo

- Hard-label blend tren focus-binary khong giup. Probe `probe_v8_yolof_pairroute_teacherfocusbinary02_hardblend025_boundarydrop_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8838/0.6802`, thap hon focus-binary soft `0.8847/0.6860`; reject full train.
- Foreground-surface + teacher focus-binary cung giam metric. Probe `probe_v8_yolof_pairroute_fgsurf_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8805/0.6686`; XAI foreground/background van tot (`attention 0.9218/0.0782`, `Grad-CAM 0.9064/0.0936`) nhung class 1 tut, nen surface-stat branch khong tao signal moi cho teacher-binary.
- Trien khai `TopKReassessmentHead` theo huong confusion-aware/router: residual nhan head embedding + base logits/probs + top-k logits/probs, co routing theo cap `0-1,1-2,2-3,4-rest`, trace va aux CE rieng tren adjustment. Compile va focused pytest pass.
- Probe top-k khong aux `probe_v8_yolof_pairroute_topk3_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8817/0.6725`; XAI foreground tot (`Grad-CAM 0.9593/0.0407`) nhung adjustment gan zero (`~1e-4`) nen khong sua boundary.
- Them top-k aux loss de ep residual hoc nhanh hon; smoke pass va loss active (`train_topk_reassessment_aux_loss=1.2344`, fraction `0.7969`). Probe `probe_v8_yolof_pairroute_topk3aux_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8805/0.6667`, P/R `0.5928/0.7616`; reject. XAI probe: attention/Grad-CAM foreground/background `0.9216/0.0784` va `0.9613/0.0387`, background blur/gray gan `0`, object desaturate `0.0439`. Ket luan: top-k residual/aux khong mo gate voi 2e/120b, khong keo dai full train.
- Thu combo hep hon de chong false-positive class 1: teacher focus-binary `0.015` + focused false-positive margin weight `0.025`, negatives `0,2,4`, margin `0.08`. Smoke pass, loss active `train_focused_false_positive_margin_loss=0.0302`, fraction `0.5938`; XAI smoke cho background gan nhu vo hai.
- Probe `probe_v8_yolof_pairroute_teacherfocusbinary015_c1fp025024_boundarydrop_bboxprior_120b_2e_20260701` best epoch `2`, val macro/class1 `0.8819/0.6726`, class-1 P/R `0.6064/0.7550`; confusion `0->1=47`, `1->0=20`, `1->2=12`, `2->1=11`, `4->1=12`. So voi focus-binary soft, false-positive `0->1` giam nho `49->47` nhung recall class 1 giam `118->114`, F1 tut manh.
- XAI probe c1fp combo: attention/Grad-CAM foreground/background `0.9323/0.0677` va `0.9668/0.0332`; background blur/gray drop `0.0004/0.0005`, object desaturate `0.0484`, near-tie `8/20`, border flags cao (`attention_border=18`, `gradcam_border=9`). Ket luan: margin false-positive dang lam boundary qua bao thu, khong phai loi nen rong; khong lap lai c1fp/teacher-binary combo voi tham so nay.

### `yolo_f` refined object-level teacher va distillation lai

- Export them object-level probabilities theo `sample_index` cho `teacherfocusbinary`, `distillens` va `c1fpcombo`, roi grid lai cung cac expert gan best cu (`c1fpsmooth`, `objecterase`, `surfaceamp`, `fgbg015`). Ket qua coarse 9-expert buoc `0.10` van khong vuot best cu; refined 7-expert buoc `0.05` dat val macro/class1 `0.8864/0.6954`, class-1 P/R `0.6142/0.8013`.
- Weights refined tot nhat: `c1fpsmooth=0.35`, `objecterase=0.35`, `surfaceamp=0.05`, `fgbg015=0.15`, `teacherfocusbinary=0.05`, `distillens=0.05`, `c1fpcombo=0`. Confusion: `0->1=52`, `1->0=14`, `1->2=11`, `2->1=9`, `4->1=11`.
- Random/local search quanh refined weights khong vuot `0.6954`; validation-only class-1 threshold sweep cung giam class1 xuong best `0.6555`, nen khong co simple threshold/reroute sau logits de mo gate.
- Tao teacher CSV refined `teacher_probs_train_sampleindex_refine005.csv` va `teacher_probs_val_sampleindex_refine005.csv`. Teacher train macro/class1 `0.9400/0.8173`, class-1 P/R `0.7011/0.9797`; teacher val macro/class1 `0.8864/0.6954`, van duoi gate `0.70`.
- Binary teacher refined mem hon teacher cu: train mean p1 overall/class1/class0/class2/class4 `0.1708/0.3276/0.1700/0.1565/0.1577`, val class1 mean `0.2894`. Vi teacher focus signal qua mem, khong dung tiep focus-binary tu teacher nay; chi thu soft distillation de kiem tra transfer.
- Smoke refined distillation pass: `key_mode=sample_index`, `teacher_label_agreement=0.9618`, `train_distillation_loss=0.0632`, trace completed. XAI smoke foreground/background attention/Grad-CAM `0.8951/0.1049` va `0.9443/0.0557`; background perturbation gan `0`.
- Probe `probe_v8_yolof_pairroute_distillens_refine005_boundarydrop_bboxprior_120b_2e_20260701` dat best epoch `2`, val macro/class1 `0.8827/0.6765`, class-1 P/R `0.6085/0.7616`; confusion `0->1=47`, `1->0=20`, `1->2=11`, `2->1=11`, `4->1=12`. Student refined teacher thap hon ca teacherfocusbinary/distillens cu, reject full train.
- XAI refined distillation probe da xem: attention/Grad-CAM foreground/background `0.9318/0.0682` va `0.9617/0.0383`; grad-rollout foreground/background `0.9757/0.0243`; background blur/gray drop `-0.0003/-0.0001`, center occlusion `0.0113`, object desaturate `0.0464`; flags `attention_border=17`, `gradcam_border=9`, `near_tie=7`. Ket luan: refined weighted teacher co complementarity nhung khong du manh de distill vao student scratch trong 2e/120b.

### Oracle complementarity, color/defect fusion va `class_f+yolo_f` fixed teacher

- Them tool `trkh.tools.analyze_expert_complementarity` de xem sample-level rescue/block thay vi chi xem weighted ensemble. Oracle 9 expert tren refined teacher dat val macro/class1 `0.9011/0.7294`; oracle 11 expert tang nhe len `0.9026/0.7353`. So voi refined teacher class1 `0.6954`, chi co `3` false-negative class 1 duoc expert nao do cuu va `11` false-positive vao class 1 duoc expert nao do chan trong tap case dang can sua. Ket luan: ensemble hien tai co complementarity nhung tran oracle van rat xa `0.98`, nen khong nen hy vong routing lai cung cac expert nay se mo gate lon.
- XAI target vao cac case oracle/refined sai da chay tren `probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\xai_audit_oracle_cases_refine005`. Background perturbation van nho, heatmap chinh nam tren qua/surface/illumination/border; day tiep tuc ung ho loi boundary fine-grained hon la loi nen rong.
- MIM cap nhat 2026-07-02 tren route teacher-focus-binary/bbox/boundarydrop voi weight `0.002` tiep tuc fail gate: `probe_v8_yolof_mim002_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` dat val macro/class1 `0.8790/0.6705`, class-1 P/R `0.5909/0.7748`; XAI selected20 co foreground cao nhung near-tie/border con, background perturbation gan `0`, va false-positive `0->1=50`, `4->1=15`. Ket luan: supervised RGB reconstruction train-time khong them du signal boundary class 1; khong keo full train.
- Source-context focus suppression 2026-07-02 da them vao train-time loss: dung full-frame YOLO context chi de phat hien khi `p(class1)` tren context cao hon crop object voi negative classes `0/2/4`, khong doi inference. Smoke pass va loss active; probe `probe_v8_yolof_sourcectxfocus015_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` dat val macro/class1 `0.8776/0.6648`, class-1 P/R `0.5784/0.7815`, confusion chinh `0->1=58`, `3->2=55`, `1->0=15`, `4->1=15`. XAI selected20: attention/Grad-CAM foreground/background `0.9464/0.0536` va `0.9171/0.0829`, background blur/gray drop gan `0`, object desaturate drop lon hon (`0.0419`), Grad-CAM background/border flags tang. Ket luan: loss co giam delta context nhung lam precision class 1 xau hon; reject va khong lap lai context-rong/source-context loss voi tham so hien tai.
- Thu color-stat + defect-stat fusion tren best teacher-focus-binary route: `probe_v8_yolof_pairroute_teacherfocusbinary015_colordefect_boundarydrop_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8845/0.6825`, class-1 P/R `0.6183/0.7616`. XAI selected `20` case, `object_color_sensitive=9`, `rollout_background_attention=13`, Grad-CAM foreground/background `0.7918/0.2082`. Ket luan: color/defect statistics tang precision nhe nhung giam recall, khong vuot focus-binary `0.6860` va khong qua gate.
- Phat hien run `class_f+yolo_f sourceaux` dung truc tiep teacher CSV object-level la sai protocol: teacher agreement chi `0.129`, vi `classification_folder` sample index khac `yolo_f` object index. Da them `trkh.tools.remap_yolo_teacher_to_classification_folder` de map teacher theo crop filename `Image_x_boxNNN` ve order cua `class_f`.
- Remap refined teacher: train/val map du `9215/9215` va `2606/2606`, teacher-label agreement `0.9618/0.9202`. Remap focus005: train/val agreement `0.9621/0.9206`. Smoke fixed-map loader xac nhan offline distillation agreement `0.9621`, nen protocol sau do hop le.
- Probe `class_f+yolo_f` source-context auxiliary khong teacher: `probe_v8_classf_yolopair_sourceaux_noteacher_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8824/0.6706`, confusion class 1 van `1->0=16`, `1->2=15`, false-positive vao class 1 tu `0/2/3/4 = 52/9/3/13`.
- Probe fixed-map teacher-focus + source-context auxiliary: `probe_v8_classf_yolopair_sourceaux_teacherfocus015_fixmap_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8818/0.6706`, class-1 P/R `0.5990/0.7616`, thap hon YOLO object-crop teacher-focus-binary va thap hon no-teacher sourceaux nhe. XAI selected `20` case: flags `attention_background=8`, `gradcam_background=8`, `rollout_background=13`, `object_color_sensitive=9`; attention foreground/background `0.8567/0.1433`, Grad-CAM `0.7918/0.2082`; robustness background blur/gray drop `0.0026/0.0008`, object desaturate drop `0.1305`.
- Ket luan cap nhat cho gia thuyet "dung `yolo_f` lam ngu canh de hoc dau la nen, dung `class_f` de phan loai": pipeline da dung va teacher mapping da sua, nhung context aux lam tang background/color flags va khong cai thien class 1. Khong chay full train vi moi probe deu duoi gate `0.70`.

### GCE robust loss tren best teacher-focus-binary route

- Doi chieu GCE/noisy-label robust loss vi label boundary class 1 co nhieu case mo ho; repo da co `GeneralizedCrossEntropyLoss`, nen chi doi `ClassificationLoss=gce` tren cau hinh best `yolo_f + bbox prior + boundary-band + pairwise routing + teacher_focus_binary015`.
- Smoke `smoke_v8_yolof_pairroute_teacherfocusbinary015_gce_boundarydrop_bboxprior_20260701` pass va trace completed. XAI smoke foreground/background tot: attention `0.9657/0.0343`, Grad-CAM `0.9727/0.0273`; background blur/gray drop gan `0`, object desaturate drop `0.0289`.
- Probe `probe_v8_yolof_pairroute_teacherfocusbinary015_gce_boundarydrop_bboxprior_120b_2e_20260701` best epoch `1`, val macro/class1 `0.8762/0.6609`, class-1 P/R `0.5876/0.7550`. Confusion: `0->1=51`, `1->0=20`, `1->2=12`, `2->1=12`, `4->1=14`; thap hon focus-binary soft `0.8847/0.6860` va thap hon best pairroute.
- XAI probe GCE selected `20`: attention/Grad-CAM foreground/background `0.9454/0.0546` va `0.9646/0.0354`, background perturbation gan `0`, object desaturate drop `0.0686`; flags `gradcam_border=8`, `rollout_border=12`, `near_tie=6`, `object_color_sensitive=4`. Ket luan: robust loss lam representation foreground sach nhung lam boundary qua mem/giam separability, reject full train va khong sweep `q` tiep neu khong co signal moi.

### Train-only focus-neighbor binary hard-case route

- Doi chieu lai voi Knowledge Distillation specialist idea: thay vi soft-distill toan bo distribution, tao objective hep `class1` vs cac neighbor `0/2/4` tu train predictions cua checkpoint gan best `teacher-focus-binary`. Tool moi `trkh.tools.build_focus_neighbor_binary_manifest` chi doc train predictions va co leakage guard loai val/test path.
- Train predictions cua checkpoint `probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701` co train macro/class1 `0.9399/0.8183`; loi lon nhat tren train la `0->1=158`, `2->1=54`, `1->0=12`, `4->1=10`. Manifest anchor sau cung co `700` rows, by-reason gom `focus_positive_anchor=274`, `focus_low_margin_negative=176`, `focus_false_positive=178`, `focus_low_margin_positive=62`, `focus_false_negative=10`; do duplicate object-level paths, dataset matched thanh `878` train samples.
- Patch model/training gom `FocusNeighborBinaryDataset`, config/CLI/launcher, loss BCE tren logit `class1 - neighbor`, CSV history va focused tests `tests/test_focus_neighbor_binary_manifest.py` pass. Loss chi ap dung row trong manifest, khong sua raw dataset.
- Smoke `smoke_v8_yolof_focusneighborbinary_anchor002_boundarydrop_bboxprior_20260701` pass; loss active va XAI small sach: attention foreground/background khoang `0.978/0.022`, Grad-CAM `0.979/0.021`, background perturbation gan `0`, object desaturate drop nho.
- Probe `probe_v8_yolof_focusneighborbinary_anchor002_boundarydrop_bboxprior_120b_2e_20260701` best epoch `2`: val macro/class1 `0.8818/0.6780`, class-1 P/R `0.5911/0.7947`. Confusion: `0->1=56`, `1->0=14`, `1->2=12`, `2->1=11`, `4->1=12`; so voi focus-binary soft, false-positive `0->1` tang `49->56`, precision class 1 giam.
- XAI probe selected `20`: top confusions `0->1=56`, `3->2=54`, `1->0=14`, `1->2=12`, `4->1=12`; attention/Grad-CAM foreground/background `0.9857/0.0143` va `0.9859/0.0141`; background blur/gray drop `0.00017/0.00012`, object desaturate drop `0.0594`; flags chinh van border/near-tie/object-color. Ket luan: hard-case binary route khong bi background shortcut, nhung day boundary qua class 1 lam precision giam. Reject full train va khong lap lai tham so nay.

### Frozen embedding prototype diagnostic va two-view complementarity

- Vi train-time hard routing/loss tiep tuc lam false-positive class 1 tang, da kiem tra gia thuyet "head hien tai khong khai thac het embedding" bang prototype/kNN/logistic fit train-only. Truoc khi chay, patch `trkh.tools.probe_embedding_prototypes` de ho tro dung `yolo_f` object-crop, `bbox` va `image_mask` metadata nhu `evaluate.py`, tranh diagnostic sai do bo bbox spatial prior.
- `yolo_f` diagnostic tren checkpoint `probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701`: base head val macro/class1 `0.8847/0.6860`; centroid cosine/euclidean chi `0.8426/0.5637` va `0.8294/0.5235`; kNN tot nhat `0.8767/0.6438`; logistic balanced `0.8775/0.6579`. Ket luan tren `yolo_f`, prototype/metric classifier khong vuot head hien tai, nen khong them centroid/cosine-prototype head neu khong co embedding moi.
- `class_f` diagnostic cung checkpoint: base head `0.8804/0.6686`; logistic balanced `0.8858/0.6824`, precision/recall class1 `0.6966/0.6689`. No cao hon base `class_f` nhung van thap hon `yolo_f` base class1 `0.6860`.
- Map object-level `class_f` crop key sang `yolo_f` val thanh cong `2606/2606`. Rule tho dung `class_f` logistic balanced de chan yolo class-1 false-positive chi dat macro/class1 `0.8869/0.6873`, precision/recall class1 `0.7143/0.6623`; union/recover giam class1 `0.6819`. Oracle chon dung neu mot trong hai view dung dat macro/class1 `0.9124/0.7556`, cho thay complementarity co that nhung can router train-only co feature ro, khong phai rule tay dua val.
- Ket luan: `class_f` co tin hieu calibration/balanced surface khac `yolo_f`, nhung route paired-view CE/KL truoc day va rule thohien tai chua khai thac duoc. Neu quay lai two-view, phai train router tren train-only logits/embeddings/metadata va audit leakage; khong dung val threshold/rule de chon.

### Bilinear patch fusion + multi-granularity auxiliary

- Doi chieu B-CNN va progressive multi-granularity FGVC: bilinear pooling co co so bat local feature/texture interactions; multi-granularity auxiliary head ep intermediate layers phan biet o nhieu muc hat. Day la representation signal moi hon prototype/logit loss va phu hop loi surface/illumination/border, nen thu mot route hep tren best `yolo_f + bbox prior + boundary-band + pairwise-routing + teacher-focus-binary`.
- Preflight: compile `trkh.models.model`, `trkh.training.train`, `trkh.tools.probe_embedding_prototypes`; PowerShell launcher parse ok; focused pytest `tests/test_multi_granularity_aux_heads.py tests/test_probe_embedding_prototypes.py` pass. Smoke 1e/8b pass, trace completed, loss active (`train_multi_granularity_aux_loss~8.98`, weight `0.02`); XAI small foreground/background sach: attention `0.9794/0.0206`, Grad-CAM `0.9825/0.0175`, background blur/gray gan `0`.
- Probe `probe_v8_yolof_bilinear_multigran_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701` best epoch `2`: val macro/class1 `0.8816/0.6780`, class-1 P/R `0.5911/0.7947`. Confusion: `0->1=54`, `3->2=54`, `1->0=15`, `4->1=15`, `1->2=11`; so voi teacher-focus-binary soft, false-positive class 1 tang va macro/class1 giam.
- XAI probe selected `20`: attention/Grad-CAM foreground/background `0.9748/0.0252` va `0.9734/0.0266`; background blur/gray drop `0.0003/0.0005`, object desaturate drop `0.0444`; flags `near_tie=8`, `gradcam_border=9`, `rollout_border=9`, `object_color_sensitive=2`. Ket luan: bilinear/multi-gran khong tao shortcut nen, nhung khong tang separability class 1 trong 2e/120b va lam boundary nghieng ve class 1. Reject full train va khong lap lai tham so nay.

### Two-view train-only router va TTA bbox-aware

- Tra cuu them multi-view stacking, Trusted Multi-View Classification va MV-HFMD truoc khi tiep tuc: cac tai lieu nay ung ho ket hop view bang meta-learner/evidence/router hoac hybrid CNN-Transformer fusion, nhung can view-quality estimation va train signal khong leak. Voi TRKH, paired-view CE/KL truoc do chi regularize single-view, chua co fused inference.
- Sua `trkh.tools.remap_yolo_teacher_to_classification_folder` de doc duoc cots `prob_0_ClassName` va ghi metadata router (`path`, `true_name`, `pred_name`, `confidence`, `prob_*`). Remap `teacherfocusbinary` sang `class_f` map du train/val `9215/9215` va `2606/2606`, teacher label agreement `0.9620/0.9194`.
- Khi chay router lan dau, phat hien bug trong `trkh.tools.train_trainval_expert_router`: `_source_class_names` uu tien `metrics.json` hon suffix cot probability, lam `class_f` probabilities bi doc sai order. Da sua de uu tien suffix `prob_{i}_{class}` va them `tests/test_trainval_expert_router.py`; compile + focused pytest pass.
- Sau fix, sanity average hai view dung hon nhung khong mo gate: average val macro/class1 `0.8816/0.6725`, thap hon `yolo_f` base `0.8847/0.6860`. Router chon train-OOF `logreg_c0p03` bi overfit train predictions in-sample: val macro/class1 `0.8785/0.6505`, class-1 P/R `0.6812/0.6225`. Logistic balanced/ExtraTrees cung chi quanh class1 `0.6646-0.6687`.
- Static fusion sweep diagnostic tren val (khong dung lam gate) cho thay weight tot nhat van la `yolo_weight=1.0`: macro/class1 `0.8841/0.6841`; cac weight co them `class_f` deu giam class 1. Ket luan: complementarity `class_f/yolo_f` co oracle nhung probability/router in-sample hien tai khong khai thac duoc; can OOF base predictions that hoac fused representation train/eval that, khong lap lai post-hoc router tu train predictions in-sample.
- Sua `TestTimeAugmentation.forward` va `evaluate.py` de TTA photometric truyen dung `image_valid_mask` va `bbox_token_prior` qua callback `forward_features + forward_heads`; TTA cu bo qua bbox prior nen do sai cho `yolo_f`. Bbox-aware TTA val `runs/tta_eval_yolof_teacherfocusbinary015_bboxaware_val_20260701` dat macro/class1 `0.8777/0.6648`, class-1 P/R `0.5859/0.7682`, confusion `0->1=53`, `1->0=17`, `1->2=13`, `4->1=13`; reject vi thap hon base.

### Teacher pairwise, model soup va router object-level 3 expert

- Thu teacher pairwise-margin loss tren best teacher-focus-binary route de day truc tiep cac pair logits theo offline teacher. Smoke va XAI small deu sach, nhung probe `probe_v8_yolof_pairroute_teacherpairwise02_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8816/0.6763`, class-1 P/R `0.6000/0.7748`; epoch 2 giam nhe `0.8811/0.6761`. XAI selected20: attention/Grad-CAM foreground/background `0.9725/0.0275` va `0.9812/0.0188`, background perturbation gan `0`, nhung near-tie `10/20`, Grad-CAM border `8/20`, rollout border `11/20`. Ket luan: teacher pairwise khong sai do nen, ma khong them du signal cho boundary class 1; reject.
- Doi chieu Model Soups va SWA nen them `trkh.tools.average_checkpoints` de average `model_state`/`ema_model_state` co guard key/shape va sidecar summary. Focused test `tests/test_average_checkpoints.py` pass. Top-2 soup `teacherfocusbinary + distillens` dat val macro/class1 `0.8850/0.6860`, chi tang macro nho do sua mot mau class 2, class 1 khong doi. Top3/top4/top5 family soups giam class1 (`0.6801` hoac `0.6783`) hoac hoa best; SWA `best+last` cung run chi dat macro/class1 `0.8818/0.6778`, trong khi `last.pt` rieng roi xuong `0.8715/0.6458`. Ket luan: checkpoint averaging khong giai quyet class-1 boundary; khong lap lai soup tren cung tap checkpoint.
- Export object-level val predictions cho 3 expert co trade-off khac nhau: `teacherfocusbinary` (base), `teacherfocusbinary_colordefect` va `fgbg015`. Val export doc lap cho thay base `0.8841/0.6841`, color `0.8851/0.6845` voi precision cao hon/recall thap hon, fgbg `0.8844/0.6856` voi recall cao hon/precision thap hon. Oracle 3 expert dat macro/class1 `0.8940/0.7097`, tuc vuot gate neu co router ly tuong; nhung grid weighted best chi `0.8859/0.6882` voi weights `base=0.05,color=0.70,fgbg=0.25`.
- Vi `yolo_f` co multi-object image, phat hien `train_trainval_expert_router` can object-level key. Da sua `_read_csv` de uu tien `sample_index:<id>`, reject duplicate path neu CSV cu khong co sample index, va them test duplicate guard. Focused pytest `tests/test_trainval_expert_router.py` pass.
- Export train predictions cho `colordefect` va `fgbg015`, sau do chay router train-only object-key 3 expert. Ban khong dung image features overfit manh: selected train-OOF `logreg_c0p03` co OOF macro/focus `0.9477/0.8426` nhung val chi `0.8757/0.6411`; equal average val `0.8821/0.6763`; ExtraTrees balanced val `0.8806/0.6727`.
- Ban co image/background features cung khong mo gate: selected ExtraTrees val macro/class1 `0.8708/0.6254`; best candidate theo val trong log chi khoang `0.8797/0.6687`. Ket luan: context/brightness/background features khong du lam router tong quat; loi con lai la representation/label-boundary, khong phai thieu feature nen.
- Ket luan cap nhat: oracle 3 expert cho thay co complementarity dung cho class 1, nhung moi router train bang in-sample predictions deu overfit va weighted/probability fusion khong qua `0.70`. Neu tiep tuc route/fusion, can OOF base predictions thuc hoac fused representation/router duoc train end-to-end voi supervision sach; khong dung path-collapsed hay train-in-sample router lam bang chung.

### Learned paired-view feature fusion `yolo_f + class_f`

- Doi chieu lai gia thuyet hai view theo MV-HFMD/TMC/CrossViT: static probability sweep van chon yolo-only, nen da trien khai fused inference that su thay vi chi CE/KL paired-view. Model co `paired_view_fusion_head`, train-time fused CE/KL, evaluate learned fusion va launcher flags rieng.
- Truoc khi tin ket qua head-only, phat hien va sua bug protocol: optimizer duoc tao truoc khi freeze prefixes, BatchNorm frozen bi `model.train()` bat lai train mode, va EMA update ca frozen states. Da them regression tests va xac nhan tensor diff sau smoke/probe: non-paired `ema_model` va train model khong doi (`changed_nonpaired=0`, `max_diff=0.0`).
- Smoke `smoke_v8_yolof_classf_pairfusion_headonly_bnfreeze_emafix_teacherfocusbinary015_20260702` pass: fusion loss active, `paired_view_fusion_fraction=1.0`, gate mean khoang `0.119`, XAI selected12 foreground sach (`attention/Grad-CAM fg~0.983/0.982`) va background blur/gray gan `0`; learned fusion full-val bang yolo-only `0.8829/0.6783`.
- Probe `probe_v8_yolof_classf_pairfusion_headonly_bnfreeze_emafix_teacherfocusbinary015_120b_2e_20260702` reject: best epoch 1 val macro/class1 `0.8837/0.6822`, class-1 P/R `0.6094/0.7748`, duoi best `0.8847/0.6860` va duoi gate `0.70`. Learned fusion tren `best.pt` van bang yolo-only, con `last.pt` giam `0.8775/0.6667`; XAI top confusions khong doi (`3->2=52`, `0->1=49`, `1->0=18`, `4->1=12`).
- Ket luan: paired-view residual/logit fusion da duoc trien khai va audit dung, nhung `class_f` khong cung cap signal train sach cho head nho tren frozen embedding. Khong lap lai two-view residual/static/router voi tham so hien tai; neu quay lai two-view phai co objective moi nhu uncertainty/evidence/cross-attention hoac OOF base predictions that.

### AIDT pretrained teacher diagnostic va distillation

- Export AIDT local `ResNet50+ViT-B/16` pretrained tren `class_f` bang TTA horizontal flip va remap class order ve TRKH. Val AIDT dat macro/class1 `0.9088/0.7169`, class-1 P/R `0.6575/0.7881`, cao hon best no-pretrain TRKH class1 `0.6860`.
- Align theo `(source_stem, object_index)` giua AIDT `class_f` va TRKH `yolo_f` tren val map du `2606/2606`. Oracle "TRKH hoac AIDT dung" dat macro/class1 `0.9426/0.8375`; rieng class 1 AIDT cuu `17` FN cua TRKH nhung cung pha `15` TP, va sua `42` FP1 cua TRKH nhung tao them `23` FP1 tu mau TRKH dung. Tin hieu pretrained co complementarity that nhung khong phai teacher hoan hao.
- Them `trkh.tools.remap_classification_teacher_to_yolo` de map probability CSV tu classification-folder teacher sang `yolo_f` sample order bang crop filename/object index; compile va `tests/test_remap_classification_teacher_to_yolo.py` pass. Remap AIDT train sang `yolo_f` map du `9215/9215`, teacher-label agreement `0.9861`, teacher doan class1 `627` mau so voi hard label class1 `541`, nen co rui ro keo false-positive neu distill toan cuc.
- Smoke `smoke_v8_yolof_aidtsoft003_tfbinary010_boundarydrop_bboxprior_20260702` pass: offline teacher dung `sample_index`, distillation/focus-binary loss active, trace completed. XAI selected12 foreground sach (`attention/grad_rollout/Grad-CAM fg~0.952/0.956/0.963`), background blur/gray gan `0`, nen protocol khong hoc nen ro rang.
- Probe `probe_v8_yolof_aidtsoft003_tfbinary010_boundarydrop_bboxprior_120b_2e_20260702` reject: best epoch 2 val macro/class1 `0.8844/0.6804`, class-1 P/R `0.6105/0.7682`, thap hon current best `0.8847/0.6860` va duoi gate `0.70`. Confusion best: `0->1=47`, `1->0=19`, `1->2=11`, `2->1=11`, `3->2=50`, `4->1=12`.
- XAI probe selected20: attention/grad-rollout/Grad-CAM foreground `0.9755/0.9769/0.9829`, background blur/gray drop `0.00015/0.00087`, object desaturate drop `0.0410`; flags van chu yeu near-tie/border (`near_tie=9`, `Grad-CAM border=8`, rollout border=11). Loi con lai van la boundary/object-surface, khong phai nen rong.
- Ket luan: AIDT/pretrained giai thich vi sao cac model kia vuot TRKH scratch: representation ngoai va pretrained hybrid CNN+ViT da tach tot hon class 1. Tuy nhien soft KD toan cuc `DistillationWeight=0.03` + focus-binary `0.010` khong transfer loi the do vao student no-pretrain, co kha nang vi teacher AIDT cung tao them FP1. Khong lap lai AIDT global KD voi tham so nay; neu dung AIDT tiep, chi thu focus-only/targeted rescue objective train-only, co agreement/confidence gate va audit sau smoke.
- AIDT focus-only khong soft KD toan cuc cung khong transfer duoc loi the teacher: `probe_v8_yolof_aidtfocusonly015_conf65_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` best epoch 1 dat val macro/class1 `0.8815/0.6763`, class-1 P/R `0.6000/0.7748`; thap hon best `0.6860` va duoi gate. XAI selected20 foreground sach (`attention/grad-rollout/Grad-CAM fg~0.978/0.974/0.980`) nhung top confusion van `0->1=52`, `3->2=52`, `1->0=18`, `1->2=11`, `2->1=11`; reject.
- De tranh keo ca distribution AIDT sai huong, da them targeted-margin sample-index support va tool `trkh.tools.build_aidt_rescue_margin_manifest`. Manifest train-only hien tai chi giu sample baseline sai nhung AIDT dung/confident, co `121` rows: `10` class-1 false-negative rescue va `111` false-positive class-1 suppression (`0->1=77`, `2->1=26`, `4->1=8`, `1->0=9`, `1->4=1`). Smoke 24 batch cho thay loss active (`train_targeted_margin_fraction~0.0122`) va XAI foreground sach (`attention/grad-rollout/Grad-CAM fg~0.983/0.973/0.984`), background blur/gray gan `0`.
- Probe `probe_v8_yolof_aidtrescuemargin015_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` reject: best epoch 2 val macro/class1 `0.8817/0.6780`, class-1 P/R `0.5911/0.7947`, thap hon best `0.6860` va duoi gate `0.70`. Confusion: `0->1=56`, `1->0=14`, `1->2=12`, `2->1=12`, `3->2=54`, `4->1=11`; targeted suppression lam recall class 1 tang nhe nhung precision giam manh.
- XAI probe selected20: attention/grad-rollout/Grad-CAM foreground `0.9883/0.9789/0.9879`, background blur/gray drop `0.00022/0.00022`, object desaturate drop `0.0562`; flags `near_tie=6`, `Grad-CAM border=8`, rollout border `11`. Ket luan: AIDT targeted margin khong giai quyet boundary class 1; khong lap lai margin/suppression train-only dua tren AIDT rescue manifest voi tham so nay. Huong tiep theo neu dung AIDT la non-target/DKD gated distillation: chi lay quan he giua cac lop khong phai hard target tren sample teacher dong y/confident, khong distill target mass class 1.

## Dien giai

- Gia thuyet "nen rong va vi tri san cua YOLO giup ViT phan biet nen/doi tuong" da duoc thu bang full source-context, crop+context inset, source-context aux, bbox prior va valid padding mask.
- XAI cho thay valid-mask va YOLO crop co the lam foreground mass sach hon, nhung background perturbation gan nhu khong lam doi du doan. Nut that con lai la tin hieu tren qua: mau/surface/illumination/border va ranh nhan `0/1/2`, khong phai nen rong.
- Them supervision ep attention vao bbox co the tang "inside mass" nhung lam giam class-1 F1, nen khong nen lap lai bbox/source-context loss don thuan.
- Foreground-background mix va paired-view `class_f`/`yolo_f` da kiem tra truc tiep gia thuyet context/kep data. Ca hai co pipeline dung va XAI khong phat hien hoc nen; learned paired-view fusion cung khong vuot yolo-only, nen class1 van bi gioi han boi ranh surface/maturity hon la nen rong.
- Supervised MIM train-time cung khong mo gate: representation foreground sach hon nhung boundary class 1 khong du tach. Buoc tiep hop ly la khai thac complementarity giua cac checkpoint no-pretrain bang router train-only truoc khi them loss moi.
- Cac probe ngay 2026-07-01 xac nhan teacher/object-level complementarity co ich hon viec tang margin don le: focus-binary soft dat `0.6860`, refined object-level weighted teacher dat `0.6954`, con hard-blend/top-k/c1fp/distill-refined deu giam. Vi teacher/refined ensemble van duoi gate va distill khong transfer duoc, huong tiep theo phai la router/specialist hep dua tren oracle complementarity thay vi them loss margin/toan cuc.
- Model soup/SWA va router object-level 3 expert cung xac nhan complementarity hien tai chua chuyen thanh artifact deploy duoc: oracle co the vuot gate, nhung weighted average, soup va train-only router deu khong qua `0.70`.
- AIDT/pretrained co loi the chu yeu tu representation ngoai va da dang expert. Val AIDT single `ResNet50+ViT-B/16` tren `class_f` class-1 F1 `0.7169` va oracle voi TRKH len `0.8375`, nhung soft KD/focus-binary toan cuc khong transfer duoc va con keo class1 xuong `0.6804`; top-5 pretrained TTA van la artifact pretrained manh hon hien tai.

## Tai lieu nghien cuu da doi chieu

- [TransFG](https://arxiv.org/abs/2103.07976): fine-grained transformer voi part/token selection, phu hop y tuong chon vung tren doi tuong thay vi toan anh.
- [CoAtNet](https://arxiv.org/abs/2106.04803) va [MaxViT](https://arxiv.org/abs/2204.01697): hybrid convolution + attention local/global co co so cho kien truc CNN/Transformer, nhung can train signal dung hon thay vi chi them context.
- [TokenLearner](https://arxiv.org/abs/2106.11297): hoc token quan trong co the la huong tiep theo neu can token policy adaptive hon bbox-prune cung.
- [Random Erasing](https://arxiv.org/abs/1708.04896): co ich cho robustness, nhung trong audit nay object desaturate/surface moi la perturbation anh huong lon; background erasing/blur khong phai nut that chinh.
- [MixStyle](https://arxiv.org/abs/2104.02008): feature-statistic style mixing co chi phi thap, nhung probe nhe tren `yolo_f` khong cai thien class 1.
- [HERBS](https://arxiv.org/abs/2303.06442) va [WS-DAN](https://arxiv.org/abs/1901.09891): ung ho object/part attention, background suppression va attention-guided augmentation cho FGVC; voi TRKH, XAI cho thay nen xa khong phai nut that lon, nen chi nen thu regularizer foreground/background nhe hoac surface/part consistency co kiem soat.
- [CORAL ordinal regression](https://arxiv.org/abs/1901.07884) va [CORN](https://arxiv.org/abs/2111.08851): khai thac thu tu tu nhien cua label bang cac binary boundary/rank task; phu hop voi nhom maturity `0/1/2/3`, nhung can weight nhe de khong ep class 4 vao ordinal chain.
- [ArcFace/angular margin](https://arxiv.org/abs/1801.07698): margin tren embedding giup tang separability, nhung voi label noise/maturity mo co rui ro lam class 1 precision giam neu ap dung manh.
- [LDAM-DRW](https://arxiv.org/abs/1906.07413): margin cho class imbalance co co so, nhung cac probe sample-weight/reweight truoc day da khong mo gate; neu dung lai can ket hop signal boundary ro hon.
- [Supervised Masked Autoencoders](https://arxiv.org/abs/2205.03892): MIM co supervision co the tao representation moi khi train tu dau, nhung hai probe TRKH 2e/120b (`mim003` va `mim002`) deu duoi gate class 1; khong lap lai RGB reconstruction don thuan neu khong co objective moi nhu cross-view/part/teacher-free boundary signal.
- [ForAug](https://arxiv.org/html/2503.09399v1): foreground/background recombination co lien quan den background-bias, nhung voi TRKH phai co mask tam thoi dang tin va khong sua raw data; chi nen xem nhu train-time augmentation neu cac probe stacking khong mo gate.
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531): co so cho nen ensemble/specialist -> soft-label student; voi TRKH can map object-level dung `sample_index` de tranh teacher sai target.
- [Decoupled Knowledge Distillation](https://arxiv.org/abs/2203.08679): tach target-class KD va non-target-class KD; phu hop huong tiep theo cho TRKH vi AIDT co signal quan he lop nhung target mass class 1 co the tao FP1.
- [Self-training with Noisy Student](https://arxiv.org/abs/1911.04252): ung ho student co noise/augmentation hoc tu teacher, nhung thiet lap nay khong them unlabeled data nen chi dung nhu regularized distillation train-only.
- [Teacher-Student Architecture for Knowledge Distillation survey](https://arxiv.org/abs/2308.04268): tong quan cac dang knowledge/output/feature distillation; trong phase nay chi dung output probability distillation de giu sua doi hep va de audit.
- [Pairwise Confusion](https://arxiv.org/abs/1705.08016): ung ho regularization/decision boundary theo cap lop de giam overfit cho FGVC; voi TRKH, pairwise-routing co ich nho nhung cac bien the margin/false-positive manh hon lam giam recall class 1.
- [Adaptive Token Sampling](https://arxiv.org/abs/2111.15667): ung ho chon token adaptive de tiet kiem/tap trung vao vung quan trong; top-k residual cua TRKH can train signal manh hon, vi probe 2e cho thay adjustment con qua yeu.
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) specialist section: co so cho generalist + specialist theo nhom lop de model hoa cac lop hay bi nham; voi TRKH chi nen dung sau khi oracle/disagreement chi ra target hep va train-only.
- [Mixture of Experts in Image Classification: What's the Sweet Spot?](https://arxiv.org/html/2411.18322v2): MoE trong vision phu thuoc routing va data scale; voi dataset nho/nhieu ambiguity, router phai regularize manh va khong duoc hoc val/test.
- [Fast Fine-grained Image Classification via Weakly Supervised Discriminative Localization](https://arxiv.org/abs/1710.01168): ung ho localization/attention da muc cho fine-grained khi khong co part annotation; TRKH da co bbox/object mask nen nen khai thac de tao specialist theo confusion thay vi ep bbox toan cuc.
- [Prototypical Networks](https://arxiv.org/abs/1703.05175): classifier theo prototype/centroid trong embedding co inductive bias don gian cho du lieu it va co the dung lam diagnostic train-only xem embedding co tach class 1 tot hon head hien tai khong.
- [CosFace/large-margin cosine loss](https://arxiv.org/abs/1801.09414) va [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362): deu nham tang inter-class variance/giam intra-class variance trong embedding; voi TRKH chi nen ap dung nhe sau khi diagnostic prototype cho thay embedding/head la nut that, vi margin qua manh truoc day da lam class-1 precision/recall trade-off xau.
- [Bilinear CNNs for Fine-grained Visual Recognition](https://arxiv.org/abs/1504.07889) va [Hierarchical Bilinear Pooling](https://arxiv.org/abs/1807.09915): ung ho second-order/local feature interaction cho FGVC; probe TRKH voi bilinear patch fusion khong cai thien class 1 khi resume tu checkpoint hien tai.
- [Progressive Multi-Granularity Training of Jigsaw Patches](https://arxiv.org/abs/2003.03836): ung ho hoc/fuse nhieu muc granularity; multi-granularity aux nhe da thu nhung khong mo gate, nen can signal/router khac thay vi chi them intermediate CE.
- [View selection in multi-view stacking](https://arxiv.org/abs/2010.16271): ung ho base learner tung view + meta-learner, nhung ket qua TRKH cho thay train predictions in-sample lam router OOF qua lac quan va val giam.
- [Trusted Multi-View Classification](https://arxiv.org/abs/2102.02051): nhan manh dynamic view-quality/evidence-level fusion; voi TRKH, static probability fusion va router don gian chua du, can uncertainty/router co train signal sach.
- [Multi-View Classification Using Hybrid Fusion and Mutual Distillation](https://openaccess.thecvf.com/content/WACV2024/papers/Black_Multi-View_Classification_Using_Hybrid_Fusion_and_Mutual_Distillation_WACV_2024_paper.pdf): ung ho hybrid CNN-Transformer fusion va mutual distillation cho multi-view; paired-view CE/KL hien tai cua TRKH moi regularize single-view, chua phai fused inference.
- [CrossViT](https://rpand002.github.io/data/ICCV_2021_crossvit.pdf): co so cross-attention multi-scale/token fusion, nhung static two-view probability sweep khong co signal du manh de uu tien trien khai fusion sau neu khong co OOF/representation moi.
- [StackingClassifier sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) va [cross_val_predict sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html): meta learner/stacking dung cross-validated predictions, moi sample duoc du doan khi no nam trong held-out fold; day la ly do router TRKH khong duoc hoc tu train in-sample predictions.
- [StratifiedGroupKFold sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html): split co gang giu ti le class trong khi group khong overlap; voi `yolo_f`, group bat buoc la source image de tranh cung anh xuat hien o fold/train va fold/val.
- [Model Soups](https://arxiv.org/abs/2203.05482): weight averaging co the tang accuracy/robustness ma khong tang inference cost khi checkpoint nam trong cung basin; TRKH da thu top2/top3/top5 soups nhung class 1 khong vuot gate.
- [Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407): average cac diem tren trajectory SGD co the cho optima rong hon; voi run TRKH ngan hien tai, `best+last` soup giam class1 nen khong tiep tuc SWA neu khong co checkpoint trajectory dai hon.

## Todo tiep theo

- Khong lap lai `source_context`, `bbox-prior`, `valid-mask`, `YOLO wide-context` neu khong co signal moi.
- Source-context feature fusion 2026-07-02 da duoc trien khai dung fused inference cho crop + full-frame context, nhung ca full-update (`0.8762/0.6552`) va head-only (`0.8760/0.6591`) deu thap hon baseline `0.8847/0.6860`; XAI cho thay foreground focus van cao va background blur/gray gan nhu khong lam doi prediction, nen khong lap lai context-rong voi scale/gate hien tai.
- `--trainable-module-prefixes` da co de freeze backbone/head cu va chi train module moi; smoke head-only source-context xac nhan `268318` tham so trainable. Cong cu nay nen dung cho cac dau moi co rui ro cao truoc khi cho full-update.
- Da co OOF infrastructure cho `yolo_f`: `build_yolo_oof_folds` tao 5 fold train-only grouped by image (`8064` anh, `9215` object, hardlink only), `evaluate.py` export object metadata vao `predictions_detailed.csv`, va `stitch_yolo_oof_predictions` stitch theo `(source_stem, object_index)` de tranh collapse multi-object image. Luu y: checkpoint resume tu full train chi hop le de smoke pipeline, khong hop le de tao OOF.
- `scripts\run_trkh_yolof_oof_fold_experts.ps1` dong goi train/export fold expert va mac dinh cam unsafe resume. Dry-run fold_00 pass; guard da chan resume tu full-train checkpoint neu khong co `-AllowUnsafeResume`.
- OOF fold_00 scratch probe 2026-07-02 (`2e/120b`, no resume, skip test) khong du manh: best/eval macro khoang `0.751` nhung class1 chi `0.331`; `last.pt` export sach hon ve OOF cung chi class1 `0.317`. XAI selected12 foreground cao, background perturbation gan `0`, object desaturate anh huong lon hon. Ket luan: khong mo rong 5 folds voi schedule nay; neu quay lai OOF phai co warm-start hop le khong dung label fold-val hoac schedule moi.
- OOF launcher da sua sang hashtable splatting va mac dinh export `last.pt` de tranh chon checkpoint bang fold-val; smoke 1 batch da xac nhan launcher chay duoc.
- Neu tiep tuc tren `yolo_f`, khong lap lai bbox/objectness head theo tham so hien tai; cac bien the objectness/interior/patch-objectness deu khong mo gate. Huong moi phai giu checkpoint supervised lam neo hoac tao signal representation khac, roi bat buoc smoke -> XAI -> probe.
- Khong lap lai `BBoxPriorPatchContextHead` voi scale `0.08`/temperature `0.45`: object/background token pooling trong crop lam class1 giam va foreground focus kem hon best.
- Dung them `interior` bbox dropout voi tham so hien tai; no lam class1 giam.
- Khong tiep tuc full part-token residual hoac part-token pairwise voi tham so hien tai; ca hai deu thap hon bbox-prior/boundary-band.
- Khong tiep tuc foreground-surface pairwise head voi tham so hien tai; no khong hoc nhanh va lam class 1 giam.
- Khong lap lai MixStyle nhe `p=0.15/alpha=0.05`; khong cai thien boundary class 1.
- Khong lap lai high-frequency texture expert don le voi loss `0.03/0.03`; loss co hoc nhe nhung metric class 1 khong tang.
- Khong lap lai boundary-center margin voi tham so hien tai; XAI sach nhung metric class 1 giam.
- Khong keo dai tiep pairwise-margin routing don thuan: 4e/120b va route-margin sweep deu khong qua gate.
- Khong lap lai ordinal-boundary nhe `0.02` hoac train pairwise scale `0.35` voi cung config: ca hai khong cai thien class1 F1.
- Khong lap lai local-zoom route `0-1,4-1` voi crop tu do/logit scale `0.12`: aux co hoc nhung class1 recall giam.
- Khong tiep tuc interior-boundary objectness-gated pairwise head voi tham so hien tai; loss khong giam va trace bam crop/bbox border.
- Khong lap lai object-erasure negative loss voi tham so hien tai; no tang recall class1 nhung lam precision giam.
- Khong lap lai mixed concat/pair-view `class_f` voi weight/cau truc hien tai, bao gom paired-view CE/KL/feature consistency ngay 2026-07-01; cac bien the nay thap hon YOLO object-crop pairwise-routing.
- Khong tang tiep focused false-positive smooth margin don le; no hoa best nhung khong giam `0->1/4->1`.
- Khong lap lai surface-amplified supervised view weight `0.03/0.02`, probability `0.35`, strength `0.22`; loss kich hoat nhung class1 giam.
- Complementarity validation da xac nhan oracle 6 checkpoint vuot `0.70` nhung weighted ensemble chi dat class1 `0.6934`; combo stacking nhe `objecterase + pairroute035 + surfaceamp` da fail voi class1 `0.6686`, nen dung nhom regularizer nay.
- Foreground-background mix p=`0.15` co class1 `0.6836` nhung p=`0.25` giam `0.6744`; khong sweep tiep mix xac suat cao hon neu khong co objective moi.
- Combo stacking/fg-bg/paired-view, supervised MIM don thuan, SSL-only random-head warm-start, DINO 60b supervised-init va surface-counterfactual consistency nhe deu da fail gate. Neu quay lai reconstruction/SSL thi phai la objective khac co signal boundary/cross-view ro, hoac adapter/head-preserving merge co bang chung smoke/XAI tot hon; van phai smoke -> XAI -> probe truoc full train.
- Can doc trace/audit sau moi probe; neu class-1 validation F1 khong vuot `0.70`, khong full train.
- Khong lap lai hard-label blend focus-binary, top-k reassessment residual/aux, hoac teacher focus-binary + c1fp margin voi tham so ngay 2026-07-01; ca ba thap hon focus-binary soft.
- Object-level refined teacher da build va van duoi gate `0.70`; refined distillation student giam xuong class1 `0.6765`. Dung huong teacher/distill/focus-binary mem hien tai.
- Chuyen sang phan tich oracle complementarity theo sample: tim cac mau class 1/refined ensemble sai nhung expert khac dung, roi thiet ke router/specialist train-only co target hep. Khong dung path-collapsed, khong tune tren test.
- Khong lap lai focus-neighbor binary hard-case manifest voi loss weight `0.02`; XAI sach nhung false-positive class 1 tang. Buoc co gia tri tiep theo la diagnostic frozen embedding bang prototype/kNN/logistic train-only: neu head khong khai thac het embedding thi them prototype/cosine-margin head, neu embedding cung khong tach duoc thi can representation signal moi.
- Diagnostic frozen embedding da xac nhan `yolo_f` base head tot hon prototype/kNN/logistic; khong them prototype head neu khong co embedding moi. `class_f` logistic balanced co complementarity nho nhung rule tho khong du; neu dung tiep phai la router train-only hai-view, khong threshold val.
- Khong lap lai bilinear patch fusion + multi-granularity aux `0.02` voi teacher-focus-binary route hien tai; XAI sach nhung class-1 false-positive tang va F1 giam.
- Khong lap lai two-view router tu in-sample train predictions cua cung checkpoint; sau fix class-order, average/router deu duoi base `yolo_f`, va OOF train bi overfit.
- Khong dung TTA photometric mac dinh cho checkpoint `teacherfocusbinary015`; bbox-aware TTA lam class1 giam `0.6860 -> 0.6648`.
- Neu quay lai two-view, can fused representation co val/eval paired view that su hoac OOF base predictions train-only; khong chi average/logistic/ExtraTrees tren probabilities hien tai. Da thu learned residual paired-view fusion voi fixed head-only protocol, nhung no khong vuot yolo-only; lan tiep theo phai la objective moi nhu uncertainty/evidence/cross-attention co train signal ro.
