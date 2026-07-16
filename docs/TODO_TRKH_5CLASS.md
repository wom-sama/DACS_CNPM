# TODO TRKH 5-Class

## Cap nhat 2026-07-02

- [x] Tra cuu/doi chieu DKD/NCKD, TransFG/CrossViT va multi-view mutual distillation; ket luan tiep tuc uu tien signal hep tren class-boundary/object-level, khong lap lai background/source-context loss don thuan.
- [x] Fit `class1_reroute` va logit-bias calibration tren checkpoint `teacherfocusbinary015`: train-fit khong tong quat sang val (`class1 0.6841 -> 0.6383/0.6503`), val-fit diagnostic chi len `0.6879`; khong tich hop reroute/calibration lam inference path.
- [x] Chay probe AIDT gated non-target/DKD `probe_v8_yolof_aidtnckd008_conf70_boundarydrop_bboxprior_120b_2e_20260702`: best val macro/class1 `0.8818/0.6763`, epoch 2 `0.8762/0.6611`; XAI selected20 foreground sach nhung `0->1=52`, `1->0=18`, `3->2=53`; reject full train.
- [x] Thu ELR nhe tren current `yolo_f` teacher-focus-binary anchor: smoke `smoke_v8_yolof_elr003_teacherfocusbinary015_boundarydrop_bboxprior_20260702` pass voi ELR state `[9215,5]`; probe `probe_v8_yolof_elr003_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` dat val macro/class1 `0.8848/0.6856`, P/R class1 `0.5990/0.8013`, confusion `0->1=53`, `1->0=14`, `2->1=12`, `4->1=12`; XAI foreground sach, background perturb gan `0`, object_desaturate drop `0.0642`; reject full train vi class1 thap hon anchor `0.6860` va duoi gate `0.70`.
- [x] Them va thu teacher-guided supervised contrastive tren current `yolo_f` teacher-focus-binary anchor: active smoke co loss that (`loss=1.4202`, fraction `0.3542`); probe `probe_v8_yolof_tgcontrast002_c032_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` dat val macro/class1 `0.8854/0.6857`, P/R class1 `0.6030/0.7947`, confusion `0->1=52`, `1->0=14`, `1->2=12`, `2->1=11`, `4->1=12`; XAI `0->1=53`, `4->1=13`, object_desaturate drop `0.0734`; reject full train vi class1 khong vuot anchor `0.6860` va duoi gate `0.70`.
- [x] Thu Pairwise Confusion FGVC regularizer tren current `yolo_f` teacher-focus-binary anchor sau khi sua launcher de actual `TrainArgs` truyen `--pairwise-confusion-*`: smoke active `train_pairwise_confusion_loss=1.0342`; probe `probe_v8_yolof_pc005_all_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` dat val macro/class1 `0.8806/0.6685`, P/R class1 `0.5769/0.7947`, confusion `0->1=57`, `4->1=15`; XAI foreground sach nhung false positive class1 tang; reject full train va khong lap lai `weight=0.005`, sources `head,patch,registers,logits`.
- [x] Tra cuu va thu Mutual-Channel Loss tren current `yolo_f` teacher-focus-binary anchor (`weight=0.02`, `top_k=8`, diversity `0.20`, start epoch `1`, hard-repeat off): smoke active `train_mutual_channel_loss=1.7025`, XAI foreground sach; probe `probe_v8_yolof_mcl002_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` dat val macro/class1 `0.8816/0.6763`, P/R class1 `0.6000/0.7748`, confusion `0->1=51`, `1->0=18`, `4->1=12`; XAI background perturb gan `0` nhung near-tie/border con cao; reject full train va khong lap lai cau hinh nay.
- [x] Don them cac artifact reject khong phai keeper da co ket luan trong doc: xoa `ssl_dino_yolof_objectcrop_supervisedinit_60b1e_20260702`, `ssl_vicreg_yolof_objectcrop_120b2e_20260702`, va `probe_oof_yolof_v8_scratch_120b2e_20260702_fold_00`; manifest `runs\cleanup_manifest_20260702_ssl_oof_reject.json`, reclaimed `624.81 MB`, keeper best/final van ton tai.
- [x] Tra cuu va thu Complement Objective Training / complement entropy tren current `yolo_f` teacher-focus-binary anchor (`weight=0.02`, classes `0,1,2,4`, start epoch `1`, hard-repeat off): smoke active `train_complement_entropy_loss=0.0063`, XAI foreground sach; probe `probe_v8_yolof_cot002_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` dat val macro/class1 `0.8821/0.6780`, P/R class1 `0.5911/0.7947`, confusion `0->1=55`, `1->0=14`, `1->2=12`, `2->1=12`, `4->1=12`; XAI background perturb gan `0` nhung object_desaturate drop `0.0704` va near-tie `6/20`; reject full train, xoa run reject qua manifest `runs\cleanup_manifest_20260702_cot002_reject.json` (`548.15 MB`), khong lap lai cau hinh nay.
- [x] Thu cumulative ordinal head/CORN-style nhe tren best teacher-focus-binary: smoke `smoke_v8_yolof_cumordinal015_teacherfocusbinary015_boundarydrop_bboxprior_24b_20260702` dat val macro/class1 `0.8831/0.6784`; XAI selected12 foreground sach nhung class1 thap hon best va `1->0=19`; khong probe 2e voi tham so nay.
- [x] Them `trkh.tools.soft_ensemble_predictions` + `tests/test_soft_ensemble_predictions.py` de blend probability CSV theo `sample_index`; focused test pass.
- [x] Tao artifact validation `runs\softensemble_trkh0775_aidt0225_val_20260702`: soft ensemble frozen weights `TRKH=0.775`, `AIDT=0.225` dat val macro/class1 `0.9139/0.7410`, vuot AIDT single val `0.9088/0.7169` va vuot gate class1 `0.70`. Day la ensemble validation-selected, khong phai single no-pretrain TRKH.
- [x] Chay final test audit sau khi freeze ensemble weight: TRKH single `runs\eval_yolof_teacherfocusbinary015_best_test_final_20260702` dat test macro/class1 `0.8865/0.6667`; AIDT mapped test `runs\aidt_yolof_mapped_test_metrics_20260702` dat `0.8575/0.5806`; frozen ensemble `runs\softensemble_trkh0775_aidt0225_test_final_20260702` dat `0.8785/0.6494`. Ket luan: TRKH single hien tai vuot AIDT tren test, con validation ensemble khong tong quat tot hon TRKH.
- [ ] Neu viet paper, trinh bay soft ensemble nhu diagnostic/complementarity baseline validation-only, khong chon lam final test artifact; final artifact hien tai trong nhom nay van la TRKH single best test.
- [ ] Neu muc tieu bat buoc la single no-pretrain TRKH, ensemble tren chi la upper-bound/deploy baseline; khong xem la hoan tat single-model, tiep tuc can representation signal khac hoac OOF router hop le.
- [ ] Huong tiep theo can thay doi representation/decision boundary that su; khong lap lai regularizer lam tang recall class 1 nhung giam precision hoac khong tach duoc bien (`ELR`, `tgcontrast`, `Pairwise Confusion`, `Mutual-Channel Loss`, `Complement Entropy`) neu khong co co che kiem soat false positive `0->1/4->1`.

## Da hoan thanh

- [x] Audit kien truc cu: chua co token gating/pruning thuc.
- [x] Them attention + foreground-prior hard token pruning.
- [x] Them patch detail enhancer cho color/edge/texture cuc bo.
- [x] Ket noi color/edge branch tokens va CNN residual logits.
- [x] Sua color-stat logit path de tham gia train/evaluate.
- [x] Them balanced epoch sampler chi dung train split.
- [x] Uu tien `splits.train.classes` trong `canbang.yaml`.
- [x] Tao trace 1 anh moi class qua tung block.
- [x] Them unit test va benchmark tensor.
- [x] Run chinh trong gioi han 30 epoch; completed 30 epoch, best epoch 27.
- [x] Luu checkpoint best epoch 27 va ket qua test.
- [x] Chay architecture trace bang checkpoint 5-class da train.
- [x] Audit lai foreground/token-pruning bang XAI va trace checkpoint.
- [x] Sua pseudo foreground mask de khong coi center padding la foreground.
- [x] Sua token foreground prior trong model va foreground-consistency loss trong train.
- [x] Tao hard-mining manifest train-only cho cac bien 0/1, 1/2, 2/3 va 4/rest.
- [x] Sua wiring de hard-sample repeat van hoat dong khi strict balanced sampler bat.
- [x] Smoke test hard-repeat + balanced sampler: effective samples 9524, exposure `[1906, 1906, 1906, 1906, 1906]`.
- [x] Ghi audit toi uu tai `docs/TRKH_5CLASS_OPTIMIZATION_AUDIT_20260608.md`.
- [x] Chay full train `mango_cls_256_5class_hardneg_maskfix_v4_30e`; test macro F1 `0.8823`, class 1 F1 `0.6545`.
- [x] Chay XAI audit v4; foreground mass tot hon, background blur gan nhu khong anh huong prediction.
- [x] Tao architecture trace v4 tai `runs/mango_cls_256_5class_hardneg_maskfix_v4_30e/architecture_trace_v4_checkpoint`.
- [x] Them tool `trkh.tools.ensemble_predictions` de tao majority/weighted vote tu prediction CSV.
- [x] Tao top-5 pretrained vote ensemble; macro F1 `0.9226`, accuracy `0.9526`.
- [x] Export val/test probabilities cho top-5 pretrained experts.
- [x] Them validation-only logistic selector; ket qua thap hon majority vote nen khong chon lam system chinh.
- [x] Them AIDT class_f train wrapper co `-UsePretrained true/false`.
- [x] Xac nhan architecture trace v4 da co du 1 mau moi class va anh qua tung block.
- [x] Tich hop `--trace-architecture` vao TRKH train de tu tao `run_dir/architecture_trace` sau run.
- [x] Them block preflight khong train vao `image_baseline_experiments/Train.md`.
- [x] Ghi chan doan run bi dung tai `docs/TRKH_TRAIN_STOP_DIAGNOSIS_20260610.md`.
- [x] Them `--pretrained/--no-pretrained` vao TRKH cho backbone torchvision; custom TRKH van mac dinh scratch va reject pretrained ro rang.
- [x] Ghi resume audit tai `docs/TRKH_RESUME_AUDIT_20260610.md`.
- [x] Chay smoke TRKH 1 train/1 val batch; trace sau train completed.
- [x] Tao `trkh.tools.audit_class_confusions`.
- [x] Tao class 1 audit va copy review images tai `runs/mango_cls_256_5class_hardneg_maskfix_v4_30e/class1_confusion_audit_20260611`.
- [x] Chot khuyen nghi RTX 4060 8 GB: batch `64`, workers `4/2`; fallback batch `48`.
- [x] Ghi audit dot pha tai `docs/TRKH_5CLASS_BREAKTHROUGH_ENSEMBLE_AUDIT_20260608.md`.
- [x] Them ordinal maturity head chung cho class `0<1<2<3`; class 4 giu binary `4-rest`.
- [x] Them `--train-scale-crop-probability` de random crop khong con bat buoc tren moi train sample.
- [x] Mo rong architecture trace voi anh truoc preprocess, illumination-normalized va foreground-mask overlay.
- [x] Chay ordinal probe train-only/val-only: test macro F1 `0.8823 -> 0.8907`, class 1 `0.6545 -> 0.6800`.
- [x] Chay smoke v5 2 train/2 val batch voi workers `4/2`; ordinal loss finite, trace completed.
- [x] Dat early stopping v5 `patience=3`; dung sau 3 epoch lien tiep khong cai thien `fair_macro_f1`.
- [x] Hoan thanh full run `mango_cls_256_5class_ordinal_maskaudit_v5_30e`; early-stop epoch 8, best epoch 5, test macro F1 `0.8095`, class 1 F1 `0.5376`; reject.
- [x] Them validation-only logit bias calibration; chi tang nhe test macro F1 `0.8823 -> 0.8866`, class 1 `0.6545 -> 0.6582`; reject lam huong chinh.
- [x] Them CV prediction selector voi image quality features; OOF tot nhung test thap hon top-5 pretrained vote; reject.
- [x] Them focus-class specialist cho class 1; test thap hon majority vote; reject.
- [x] Them pretrained teacher distillation cho TRKH voi flag `--pretrained-distillation/--no-pretrained-distillation`.
- [x] Them launcher `scripts/run_trkh_5class_distill_v6.ps1` co toggle distillation, batch size, workers, smoke, max batches va trace architecture.
- [x] Smoke launcher v6: resume v4 best, MobileNetV3 teacher mapping `[4,3,1,0,2]`, batch `48`, workers `4/2`, trace completed.
- [x] Them offline distillation CSV teacher cache cho TRKH; support `--distillation-teacher-csv` / `--offline-distillation-csv`.
- [x] Tao `trkh.tools.build_ensemble_teacher_cache` de map probability experts ve class order cua TRKH.
- [x] Tao top-5 teacher cache train/val/test va temperature-2 cache.
- [x] Chay full v7 `mango_cls_256_5class_v4_top5_offline_distill_v7_30e`; early-stop epoch 4, best epoch 1, test macro F1 `0.8801`, class 1 F1 `0.6467`; reject.
- [x] Xac nhan v7 khong treo; tong thoi gian `860.9s`, `launcher_status.json` exit code `0`.
- [x] Export val/test probabilities cho 5 pretrained experts bo sung: EfficientNet-B0, ResNet50, MobileNetV2, InceptionV3, VGG16.
- [x] Thu top-10 average teacher cache; test macro F1 `0.9117`; reject.
- [x] Thu top-10 focus-class specialist; test macro/class1 khong doi `0.9183/0.7597`; reject.
- [x] Them TTA flags vao `trkh.tools.export_timm_predictions`: horizontal flip, brightness deltas, contrast scales.
- [x] Tao top-5 TTA teacher cache; test macro F1 `0.9248`, class 1 F1 `0.7786`.
- [x] Mo rong `trkh.tools.audit_class_confusions` de doc CSV teacher probability voi `--data`.
- [x] Tao class 1 audit cho top-5 TTA tai `runs/ensemble_top5_tta_teacher_cache_5class_v1/class1_confusion_audit_test`.
- [x] Chay XAI audit nho cho v7 tai `runs/mango_cls_256_5class_v4_top5_offline_distill_v7_30e/xai_audit_test_small`.
- [x] Ghi audit v7/TTA/fusion tai `docs/TRKH_5CLASS_V7_TTA_FUSION_AUDIT_20260611.md`.
- [x] Nghien cuu ma chinh thuc WS-DAN, Pairwise Confusion, ELR, PMG va small-data ViT scratch.
- [x] Ghi quyet dinh tai `docs/TRKH_NO_PRETRAIN_RESEARCH_DECISION_20260612.md`.
- [x] Them attention-guided crop/drop classification training, mac dinh tat va cau hinh bang CLI.
- [x] Tai su dung fine-grained patch attention, ho tro anh xa score sau token pruning.
- [x] Them history audit cho attention view loss va crop/drop fraction.
- [x] Compile va 12 unit/regression test attention/token/foreground/pairwise/ordinal pass.
- [x] Ghi context recovery tai `docs/TRKH_CONTEXT_CHECKPOINT_20260612.md`.
- [x] Them bounded attention dropping voi dien tich cau hinh `6%-16%`.
- [x] Ghi `train_attention_drop_area_fraction` va dung train config trong architecture trace.
- [x] Chay full test suite sau bounded drop: `101 passed`.
- [x] Chay preflight attention v8 voi batch `32`, grad accumulation `2`, workers `4/2`.
- [x] Chay smoke 10 train/4 val batch; exit code `0`, peak reserved VRAM `5136 MiB`.
- [x] Theo doi GPU moi giay: train sau warm-up dat `74%-91%`, khong co deadlock.
- [x] Xuat score/crop/drop 1 mau moi class tai smoke v8 architecture trace.
- [x] Ghi audit tai `docs/TRKH_5CLASS_ATTENTION_V8_SMOKE_AUDIT_20260612.md`.
- [x] Hoan thanh full v8; early-stop epoch 16, best epoch 13, exit code `0`.
- [x] V8 test macro F1 `0.8869`, class 1 F1 `0.6626`; tot hon v4 nhe nhung recall class 1 khong doi.
- [x] Xuat `final_test_detailed`, class-1 confusion audit va review images cho v8.
- [x] Sua `audit_class_confusions` de uu tien class-name mapping, tranh trao class khi CSV co thu tu baseline khac `data.yaml`.
- [x] Chay XAI/robustness v8 tren 36 case; background blur/gray drop gan `0`, object desaturate drop `0.1157`.
- [x] Review thu cong mau v8 `0->1`, `2->1`, `1->0`, `1->2`; xac nhan nhieu ranh gioi label/maturity mo ho.
- [x] Ghi full audit tai `docs/TRKH_5CLASS_ATTENTION_V8_FULL_AUDIT_20260612.md`.
- [x] Cai Repomix `1.14.1` va them `repomix.config.json` de snapshot codebase khong gom runs/checkpoints/anh nang.
- [x] Them personal skill `trkh-5class` cho cac thread sau.
- [x] Ghi agent rules/gate tai `docs/TRKH_AGENT_RULES_20260612.md`.
- [x] Tra cuu WS-DAN, PMG, ELR, background masking va tooling; ghi tai `docs/TRKH_RESEARCH_AND_TOOLING_UPDATE_20260612.md`.

## Can tiep tuc

- [x] Chay full test suite: `100 passed` voi pytest 8.4.2.
- [x] Them `pytest==8.4.2` vao requirements de moi truong moi chay duoc test suite.
- [x] Chay preflight dataset/config/leak cho attention-view run.
- [x] Chay smoke attention crop/drop gioi han va smoke throughput 10 batch.
- [x] Xuat anh audit original/score-map/crop/drop cho 1 mau moi class.
- [x] Chi chay full 30 epoch neu smoke throughput/GPU utilization on dinh; patience `3`.
- [x] Sau full run, xuat class-1 confusion va XAI/background audit roi so sanh v4/top-5 TTA.
- [x] Them score source `surface_detail`/`hybrid` cho attention views va expose trong trace/config.
- [x] Smoke v9 de xac nhan score/crop/drop nam tren be mat qua, khong bam padding/vien; trace class 1 tap trung vao vet lom/dot tren be mat, con mot diem nong mep trai.
- [x] Chay probe v9 80 train batch x 4 epoch, full val: best val macro F1 `0.8857`, class 1 F1 `0.6783`; test macro F1 `0.8894`, class 1 F1 `0.6708`.
- [x] Khong full-train v9 vi chua dat gate class-1 val F1 `>=0.70`; ghi audit tai `docs/TRKH_5CLASS_SURFACE_DETAIL_V9_AUDIT_20260612.md`.
- [x] Them ELR optional cho classification-only voi `IndexedSampleDataset`, CLI flags `--elr-loss-weight/--elr-beta/--elr-start-epoch`, mac dinh tat.
- [x] Them `train_elr_loss` vao history va test unit cho sample index + ELR target history.
- [x] Them launcher `scripts/run_trkh_5class_elr_v10.ps1` va tham so ELR vao attention-view launcher v8.
- [x] Chay smoke v10: exit `0`, ELR state `[9524,5]`, architecture trace completed, `train_elr_loss=-0.2485`.
- [x] Chay probe v10 120 train batch x 4 epoch, full val: best val macro F1 `0.8862`, class 1 F1 `0.6784`; test macro F1 `0.8923`, class 1 F1 `0.6792`.
- [x] Khong full-train v10 vi chua dat gate class-1 val F1 `>=0.70`; ghi audit tai `docs/TRKH_5CLASS_ELR_V10_AUDIT_20260612.md`.
- [x] Xuat v10 test detailed predictions va class-1 confusion audit tai `runs/probe_mango_cls_256_5class_surface_detail_elr_v10_120b_6e_20260612/class1_confusion_audit_test`.
- [x] Them train-only boundary sample weighting v11 voi `SampleWeightDataset`, per-sample loss va launcher `scripts/run_trkh_5class_boundary_weight_v11.ps1`.
- [x] Tao manifest v11 tu train-only predictions: `459` weighted samples, `skipped_non_train_rows=0`, pairs `0-1=233`, `1-2=70`, `2-3=98`, `4-rest=58`.
- [x] Chay probe v11 120 train batch x 4 epoch, full val: best val macro F1 `0.8856`, class 1 F1 `0.6745`; test macro F1 `0.8923`, class 1 F1 `0.6792`; reject full train.
- [x] Them in-batch boundary contrastive v12 voi max-pairs gioi han, khong tao offline pair dataset; flags `--boundary-contrastive-*`.
- [x] Chay smoke v12: `train_boundary_contrastive_loss=0.8235`, `terms=143.0`, trace completed.
- [x] Chay probe v12 120 train batch x 6 epoch, full val: best val macro F1 `0.8866`, class 1 F1 `0.6864`; test macro F1 `0.8943`, class 1 F1 `0.6879`; reject full train vi chua qua gate `0.70`.
- [x] Xuat v12 test detailed predictions va class-1 confusion audit tai `runs/probe_mango_cls_256_5class_boundary_contrastive_v12_120b_6e_20260612/class1_confusion_audit_test`.
- [x] Ghi audit v11/v12 tai `docs/TRKH_5CLASS_BOUNDARY_CONTRASTIVE_V12_AUDIT_20260612.md`.
- [x] Them optional foreground surface statistic fusion V13: pseudo foreground, gray-world RGB/Lab/HSV, exposure, damage/detail va center-border stats `[B,129]`.
- [x] Sua resume checkpoint extension cho ca model va EMA; missing key ngoai allowlist van fail.
- [x] Chay full regression test V13: `102 passed`.
- [x] Chay final smoke V13; exit `0`, partial model/EMA load thanh cong, trace du 1 mau moi class.
- [x] Chay probe V13 120 train batch x toi da 6 epoch: best val macro/class1 `0.8868/0.6822`, test `0.8938/0.6835`.
- [x] Xuat detailed predictions, class-1 confusion audit va `09a-09f` surface trace cho V13.
- [x] Review close-margin `0->1`, `1->0`, `1->2`: boundary maturity/illumination mo ho; mask van thu mot phan nen co mau gan xoai.
- [x] Khong full-train V13 vi val class-1 F1 `0.6822 < 0.70` va thap hon V12.
- [x] Ghi audit V13 tai `docs/TRKH_5CLASS_FOREGROUND_SURFACE_V13_AUDIT_20260612.md`.
- [x] Nghien cuu Bilinear CNN, Compact Bilinear Pooling va Low-rank Bilinear Pooling cho fine-grained recognition.
- [x] Them optional compact bilinear patch fusion V14 tren patch sau pruning, residual zero-init va padding-aware attention.
- [x] Them CLI/config/launcher V14 va checkpoint extension allowlist cho model + EMA.
- [x] Them `09g_bilinear_patch_attention.png` va descriptor/attention shapes vao architecture trace.
- [x] Chay full regression V14: `105 passed`; preflight va smoke exit `0`.
- [x] Chay probe V14 120 batch x toi da 6 epoch: best val macro/class1 `0.8859/0.6802`; test `0.8912/0.6750`.
- [x] Khong full-train V14 vi class-1 validation F1 thap hon V12 va khong qua gate `0.70`.
- [x] Ghi audit V14 tai `docs/TRKH_5CLASS_BILINEAR_PATCH_V14_AUDIT_20260612.md`.
- [x] Audit exact/near duplicate/source sequence cua `class_f`: khong co exact SHA1, nhung co source-sequence leakage tren cung qua/phien chup giua train/val/test; ghi tai `docs/TRKH_5CLASS_CLASS_F_SPLIT_AUDIT_20260612.md`.
- [x] Doi chieu baseline ViT `1.0`: khong co trong bang local tren cung protocol; khong du thong tin de so sanh truc tiep.
- [x] Chay validation-only boundary calibration tren V12: val tang nhe `0.8863/0.6864 -> 0.8929/0.6925`, nhung test macro giam `0.8943 -> 0.8877`, class 1 khong doi; reject.
- [x] Nghien cuu va them frequency-selective patch pooling V15 theo LaSt-ViT, co foreground gating, residual blend, CLI, launcher va trace vote.
- [x] Full regression V15: `116 passed`; preflight/smoke/probe exit `0`.
- [x] Probe V15: val macro/class1 `0.8872/0.6788`, test `0.8975/0.6923`; khong full train vi class-1 val thap hon V12 va chua qua `0.70`.
- [x] Xuat detailed predictions, class-1 audit va 1 trace moi class cho V15.
- [x] Them routed pairwise specialist V16: pairwise margin head chi kich hoat khi top-2 logits khop boundary va probability margin du nho.
- [x] Them `-SkipFinalTest` cho launcher V8/V12/V16 de probe khong bi timeout o final audit; test/trace co the chay rieng.
- [x] V16 preflight/smoke/regression pass; probe 120 batch x 6 epoch chon epoch 3, val macro/class1 `0.8874/0.6866`.
- [x] Xuat V16 test detailed, class-1 confusion audit va architecture trace thu cong sau automation timeout; test macro/class1 `0.8949/0.6923`.
- [x] Khong full-train V16 vi class-1 validation F1 van duoi gate `0.70`; trace cho thay pseudo-mask van lay nen co mau gan qua.
- [x] Nghien cuu GrabCut/Cutout/Random Erasing/AugMix cho background/occlusion/robustness; codebase da co erasing/local exposure/obstacle nen uu tien mask.
- [x] Them GrabCut background suppression modes `grabcut_*`, audit tool va preprocessing cache builder co `--dry-run`, `--max-samples-per-class`, `--workers`.
- [x] Them `-DataYaml` cho launcher V8/V12/V16/V17 de co the tro toi preprocessing cache dataset.
- [x] V17 mask audit: border foreground class 1 giam `0.3675 -> 0.0729`, nhung van giu mot phan nen xanh sat qua.
- [x] V17 smoke exit `0`, trace completed; online GrabCut cham nhung khong treo.
- [x] V17 probe 80 batch x 4 epoch: best val macro/class1 `0.8789/0.6405`; reject, khong full-train.
- [ ] Tao group-clean split theo qua goc/phien chup truoc khi dung metric de khang dinh generalization.
- [ ] Tao train-only label-boundary review set cho 0/1/2, ghi label dung/sai/ambiguous va quality condition.
- [ ] Thu learned object-tight localization/mask co supervision; khong tiep tuc siet pseudo-mask mau vi co/la xanh co the dinh vao qua.
- [x] Thu pairwise specialist/router chi kich hoat tren top-2 boundary, khong adjustment toan cuc; cai thien nho nhung chua qua gate.
- [ ] Neu tiep tuc ELR, chi thu ablation nho hon/lon hon (`0.03`, `0.20`) sau label audit; khong full train neu class-1 val F1 chua qua `0.70`.
- [x] Chay preflight khong train truoc smoke 2026-06-11.
- [x] Chay smoke TRKH voi `--max-train-batches 1 --max-val-batches 1 --skip-final-test`.
- [x] Tao audit list anh sai/confusion class 1 tu `final_test_detailed/predictions_detailed.csv`.
- [ ] Review thu cong anh `0->1`, `2->1`, `1->0`, `1->2` trong class 1 audit; danh dau label dung/sai/ambiguous.
- [x] Thiet ke run v5 tap trung ordinal boundary 0/1 va 1/2, khong tang oversampling class 1 vo dieu kien.
- [ ] Review thu cong 22 false negative va 7 false positive class 1 trong `runs/ensemble_top5_tta_teacher_cache_5class_v1/class1_confusion_audit_test/review_images`.
- [ ] Tao `ambiguous_boundary.csv` cho cac mau class 0/1/2 co label khong ro; khong dung test de train lai.
- [ ] Neu label audit xac nhan nhan dung, train them expert pretrained manh hon (ViT/DeiT/Swin/ConvNeXt) hoac feature-level branch pretrained; khong lap lai distillation student v7.
- [ ] Neu label audit xac nhan loi do nen/che khuat, thu object-tight segmentation/crop preprocessing roi export lai expert/TTA.
- [ ] Neu can artifact inference manh tam thoi, dung `runs/ensemble_top5_tta_teacher_cache_5class_v1` lam best current teacher-level baseline.
- [ ] Chay AIDT finalize/export chi khi chap nhan thoi gian dai; lan export probability inline da timeout sau 15 phut.
- [ ] Neu van chua dat macro F1 `0.99` va class 1 F1 `0.96`, bao cao ro rang oracle hien tai: test class 1 any-expert recall `58/73`, oracle class 1 F1 `0.8722`.
- [ ] So sanh confusion 0/1, 1/2, 2/3 va 4/rest voi v3 maskfix.
- [ ] Neu class 1 precision van thap, audit label boundary cho mau 0->1 va 1->2 co confidence cao.
- [ ] Chay ablation `token_keep_rates=0.75,0.50` vs `0.85,0.65` neu v4 cham nhung khong tang F1.
- [ ] Can nhac ablation pretrained/frozen ViT chi khi chap nhan so sanh ngang voi MobileNetV3/ViT/AIDT pretrained.
- [ ] Export ONNX/TensorRT sau khi chot checkpoint.

## Cap nhat 2026-06-25

- [x] Doc `D:\deep-research-report-11-clean.md` va ghi audit hanh dong tai `docs/TRKH_DEEP_RESEARCH_11_ACTION_AUDIT_20260625.md`.
- [x] Xac nhan huong bao cao huu ich nhat la forensic/decision-layer, counterfactual consistency va ambiguity handling; khong phai them head/loss chung chung.
- [x] Doi chieu voi V16 `class_f`: cap loi uu tien hien tai la `0-1`, `2-3`, phu `1-2`; khong phai `3-4`.
- [x] Chay validation-only pair calibration audit tren V16: `0-1` khong cai thien class 1; `1-2` chi tang class-1 F1 `0.6866 -> 0.6905`; chua qua gate `0.70`.
- [x] Loai huong simple pairwise logit bias/calibration lam ung vien full train.
- [x] Tich hop forensic/data-boundary workflow bang tool rieng `trkh.tools.build_boundary_review_manifest`; foreground stats chi tinh sau khi gioi han selected rows de tranh timeout.
- [x] Tao train-only ambiguous-boundary manifest tu V16 train predictions tai `runs/boundary_review_v16_train_20260625`; uu tien cap `0-1`, `2-3`, `1-2`; khong dung val/test.
- [x] Thu V25 background-neutralized prediction consistency nhe + ambiguous soft target train-only; probe best val macro/class1 `0.8858/0.6787`; reject, khong full train.
- [x] Them V26 targeted directional margin train-only cho hard false-positive/false-negative quanh class 1; smoke trace completed.
- [x] V26 probe 120 batch x 6 epoch: best val macro/class1 `0.8864/0.6805`; forensic `TP=115`, `FP=72`, `FN=36`; reject, khong full train.
- [x] Xac nhan V19 ordinal boundary va V20 pairwise confusion probe da co ket qua thap hon V16; khong lap lai.
- [x] Ghi audit V25/V26 tai `docs/TRKH_5CLASS_V25_V26_AUDIT_20260625.md`.
- [x] Them data-centric sample-weight builder V27 co guard train-only, dry-run, max issue/pair, copy review images va unit test.
- [x] V27 probe tu manifest V3 stale: best val macro/class1 `0.8842/0.6746`; reject, khong full train.
- [x] Export V16 train detailed predictions de mining moi, nhung dry-run cho thay low-self-confidence khong on dinh vi V16 under-confident; chua dung lam run chinh.
- [x] Them `ImageSize` cho launcher V8/V12/V16/V27; V29 high-res 384 smoke/probe pass nhung best val macro/class1 `0.8788/0.6524`; reject.
- [x] Chay V29 validation forensic va review anh; xac nhan loi class 1 chu yeu do ranh label/illumination/object surface, khong phai nen don thuan.
- [x] Them foreground object crop V30 (`foreground_crop_mode none|pseudo|grabcut`) vao transform/config/CLI/eval/XAI/trace va launcher rieng.
- [x] V30 smoke trace completed, probe best val macro/class1 `0.8851/0.6786`; reject, khong full train.
- [x] Ghi audit V27/V29/V30 tai `docs/TRKH_5CLASS_V27_V29_V30_AUDIT_20260625.md`.
- [ ] Gate full train van la class-1 validation F1 `>=0.70`; V16 van la best no-pretrain probe hien tai (`0.6866`) tren split cu.
- [x] Tao boundary review manifest train/val voi cot `correct/ambiguous/wrong/lighting/dirty/partial` tai `runs/boundary_review_v16_train_20260625` va `runs/boundary_review_v16_val_20260625`.
- [x] Tao group-clean split theo source sequence tai `D:\DataAI\AIEx\newdataset\class_f_groupclean_v1` va leak audit tai `runs/class_f_groupclean_v1_leak_audit_20260625`.
- [x] Sua launcher preflight khong hard-code `class_f` khi dem split va them toggle `HardSampleManifest/HardSampleRepeatFactor` de tat manifest split cu.
- [x] Chay preflight + smoke group-clean V16; trace completed tai `runs/smoke_groupclean_v16_20260625/architecture_trace`.
- [x] Chay probe group-clean V16/V26 warm-start va xac nhan chi dung diagnostic do checkpoint cu overlap source-sequence voi group-clean val/test.
- [x] Chay scratch group-clean V26 baseline: class1 val F1 `0.3360`; khong full train.
- [x] Them `--class-loss-multipliers`, reroute calibration tool va unit test; class1 loss `0.65` tang scratch group-clean len macro/class1 `0.7278/0.3780` nhung chua qua gate.
- [x] Thu balanced softmax tren group-clean scratch; class1 FP tang, reject.
- [x] Thu bundle ket hop MixStyle + foreground surface fusion + background-counterfactual consistency + RandAugment/local exposure/obstacle; macro/class1 `0.7081/0.3379`, reject.
- [x] Thu probe dai hon c1loss065 180 batch x 12e; early stop epoch 6, best macro/class1 `0.7131/0.3557`, reject.
- [x] Them `ImageSize` cho V26 va thu high-res 320 c1loss065; macro/class1 `0.7136/0.3478`, reject.
- [x] Chay train->val reroute/logit-bias/focus-specialist diagnostic; deu khong generalize du, reject.
- [x] Tao forensic audit val group-clean tai `runs/forensics_val_groupclean_c1loss065_20260625`; loi chinh la low-margin boundary, khong phai nen don thuan.
- [x] Ghi audit tong hop tai `docs/TRKH_5CLASS_GROUPCLEAN_SCRATCH_BUNDLE_AUDIT_20260625.md`.
- [ ] Gate full train group-clean van la class-1 validation F1 `>=0.70`; best clean scratch hien tai chi `0.3780`, nen khong full train.
- [ ] Them training-dynamics/data-cartography logger cho train/val boundary samples de tach easy/ambiguous/hard-label-error.
- [ ] Tao train-only quality-group manifest tu lighting/background/partial buckets va thu group-aware/worst-group loss nhe.
- [ ] Nghien cuu va thu self-supervised pretrain noi bo tren `class_f_groupclean_v1/train` neu data-cartography xac nhan representation thieu, khong dung external pretrained.
- [ ] Review thu cong `runs/boundary_review_v16_train_20260625/boundary_review_manifest.csv`, dien `manual_label_status`, `quality_lighting`, `quality_dirty_obstacle`, `quality_partial_fruit`, `quality_background_mask`.
- [ ] Sau manual review train-only, tao clean/ambiguous/ignore/soft-target manifest; khong dung val/test de tune.

## Cap nhat 2026-06-26

- [x] Them `foreground_surface_pairwise_head`: foreground surface stats `[B,129]` -> pairwise logits cho `0-1,1-2,1-4,2-3`, route theo top-2 probability margin.
- [x] Expose CLI/config/launcher cho surface-pairwise head va them trace metadata `foreground_surface_pairwise_logits/route_weights`.
- [x] Them test `tests/test_foreground_surface_pairwise.py`; compile + focused pytest pass.
- [x] Smoke V38 pass; architecture trace co 5 class va surface-pairwise route weights.
- [x] Probe V38 surface-pairwise: best val macro/class1 `0.7052/0.3936`; reject.
- [x] Continue Barlow thuan train-only V39 tu V34: SSL loss `84.06 -> 70.27`, nhung fine-tune val macro/class1 `0.7042/0.3696`; reject.
- [x] Ensemble diagnostic V35/V37/V38/V39/V36 tren val: best class1 `0.4304`, macro `0.7199`; chua qua gate.
- [x] Probe V40 ket hop angular-margin + pairwise-confusion: val macro/class1 `0.7142/0.4011`; macro tang nhung class1 thap hon V35; reject full train.
- [x] Ghi audit tai `docs/TRKH_5CLASS_INTERNAL_SSL_SURFACE_PAIRWISE_AUDIT_20260626.md`.
- [ ] Gate full train group-clean van la class-1 validation F1 `>=0.70`; best class1 probe hien tai trong nhom SSL la V35 `0.4096`, ensemble diagnostic `0.4304`.
- [ ] Chay probe V40 voi class-1 loss multiplier thap hon (`0.65` hoac `0.50`) de xem precision class 1 co phuc hoi khong.
- [ ] Neu class-1 van quanh `0.40-0.43`, dung huong loss/head va uu tien data-cartography + quality-group manifest train-only.
- [x] Them `HighFrequencyTextureExpert`: high-pass/gradient/laplacian/foreground-detail maps -> residual logits, co route theo low-margin boundary.
- [x] Chay V64 high-frequency texture probe tren group-clean: best val macro/class1 `0.7088/0.4151`; day la best clean single probe hien tai.
- [x] Thu V65 high-frequency pairwise aux: val macro/class1 `0.7090/0.4140`; reject vi khong vuot V64.
- [x] Thu V66 margin-sharpen high-frequency: val macro/class1 `0.7052/0.4053`; reject.
- [x] Thu V67 slow fine-tune + class1 precision bias: macro peak `0.7146` nhung class1 peak chi `0.3869`; reject.
- [x] Tao quality/cartography manifest train-only V64 va thu GroupDRO V68: val macro/class1 `0.7067/0.3938`; reject.
- [x] Them train-only `surface_amplified_supervised` + boundary margin loss de khuech dai chi tiet be mat co kiem soat.
- [x] V69 smoke xac nhan loss moi active va history ghi dung; probe best macro/class1 `0.7063/0.3700`, class1 peak `0.3821`; reject.
- [x] V69 class1 audit: TP `74`, FP `173`, FN `79`; FP chinh `0->1=89`, `2->1=49`, `4->1=31`. Ket luan: surface amplification dang khuech dai ca dau hieu gay nham class 1.
- [x] Them `trkh.tools.build_quality_sample_weights` de tao sample-weight train-only tu quality/cartography manifest, co dry-run va guard val/test.
- [x] Tao V70 sample weights tu `runs/quality_group_v64_train_20260626/quality_groups_train_only.csv`: `9161` rows train-only, weight mean normalized `1.0`, min/max `0.6045/1.3739`.
- [x] V70 probe quality sample weighting: best val macro/class1 `0.6962/0.3912`; reject vi precision class1 thap va macro giam.
- [x] Ghi audit V64-V70 tai `docs/TRKH_5CLASS_V64_V70_HIGHFREQ_SURFACEAMP_QUALITY_AUDIT_20260626.md`.
- [x] Mo rong `trkh.tools.pretrain_internal_barlow` thanh internal SSL `--ssl-method barlow|vicreg`, them VICReg loss va checkpoint kind `internal_vicreg_pretrain`.
- [x] Them test `tests/test_internal_ssl_vicreg.py`; compile + focused pytest pass.
- [x] Dry-run/smoke VICReg pass; dataloader train-only, checkpoint save duoc.
- [x] Pretrain V71 VICReg+SupCon train-only 80 batch x 3 epoch: SSL loss `34.56 -> 31.44`, checkpoint best epoch 3.
- [x] Fine-tune V71 bang cau hinh V64: best val macro/class1 `0.6972/0.3356`; reject.
- [x] V71 detailed val/class1 audit: TP `72`, FP `209`, FN `81`; FP chinh `0->1=85`, `2->1=66`, `4->1=47`. Ket luan: VICReg ngan lam class 1 rong hon, khong phai dot pha.
- [x] Ghi audit V71 tai `docs/TRKH_5CLASS_V71_INTERNAL_VICREG_AUDIT_20260626.md`.
- [x] Xuat V64 detailed val tai `runs/probe_groupclean_v64_highfreq_texture_80b_6e_20260626/val_detailed_eval`.
- [x] Tao boundary review manifest train-only V64 tai `runs/boundary_review_groupclean_v64_train_20260626` va val diagnostic tai `runs/boundary_review_groupclean_v64_val_20260626`.
- [x] Them `trkh.tools.render_boundary_review_html` de render contact-sheet HTML tu boundary review CSV; them unit test.
- [x] Render HTML review reports: `boundary_review_report.html` cho train-only `300` rows va val diagnostic `240` rows.
- [x] Chay XAI audit nho V64 val tai `runs/xai_audit_groupclean_v64_val_small_20260626`; heatmap foreground mass cao, background robustness drop gan 0, object desaturate moi anh huong ro.
- [x] Review bang mat mau `1->0` va `0->1`: model chu yeu nhin qua/surface; FP class1 co cham/vet nhe tren vo, FN class1 rat giong class0. Ket luan: foreground/background khong phai nut that chinh.
- [x] Thu V64 eval TTA nhe: macro/class1 `0.7014/0.3969`, thap hon non-TTA `0.7088/0.4151`; reject.
- [x] Ghi audit forensic tai `docs/TRKH_5CLASS_V64_BOUNDARY_FORENSIC_REVIEW_AUDIT_20260626.md`.
- [ ] Gate full train group-clean van la class-1 validation F1 `>=0.70`; best clean single probe hien tai la V64 `0.4151`, chua du dieu kien full train.
- [ ] Dung huong khuech dai/weighting toan cuc neu khong co bang chung moi; uu tien representation learning noi bo manh hon va boundary review train-only.
- [ ] Khong chay lai VICReg ngan neu chi doi he so nho; neu quay lai SSL, can DINO/MAE hoac pretext co object/part-level evidence ro hon.
- [ ] Dien manual label/status cho `runs/boundary_review_groupclean_v64_train_20260626/boundary_review_manifest.csv`; chi train-only moi duoc dung cho clean/ambiguous manifest.
- [x] Tao V72 ambiguous soft-target tu V64 train predictions (`700` train-only rows); probe macro/class1 `0.7088/0.4142`, thap hon V64 `0.4151`; reject.
- [x] Thu V73 boundary-center + high-frequency bundle; smoke loss active, probe macro/class1 `0.6974/0.3895`; reject vi class-1 precision giam manh.
- [x] Tao V74 data-centric downweight manifest tu V64 train predictions (`412` train-only rows, weights `0.5-0.8`, skipped_non_train `0`); probe macro/class1 `0.7025/0.4021`; reject.
- [x] Ghi audit V72-V74 tai `docs/TRKH_5CLASS_V72_V74_DATA_POLICY_BOUNDARY_AUDIT_20260626.md`.
- [ ] Gate full train group-clean van la class-1 validation F1 `>=0.70`; best clean single probe hien tai van la V64 `0.4151`, chua du dieu kien full train.
- [ ] Khong lap lai soft-target/downweight tu dong dua tren cung V64 error neu chua co manual review; no chi quay quanh cung boundary noise.
- [ ] Huong tiep theo co kha nang dot pha hon: manual train-only boundary review de tao `clean/ambiguous/ignore/soft-target`, hoac thiet ke MAE/DINO patch-level noi bo thay vi global SSL/head/loss.
- [x] Them MAE train-only vao `trkh.tools.pretrain_internal_barlow` voi foreground/detail weighted patch masking; focused tests pass.
- [x] V75 MAE pretrain 80b x 3e + V64 fine-tune: val macro/class1 `0.7134/0.3596`; reject vi class 1 giam.
- [x] V76b local zoom + micro-detail + high-frequency: val macro/class1 `0.7037/0.3929`; reject vi class-1 precision giam.
- [x] V77 LDAM+GCE: val macro/class1 `0.4812/0.2776`, recall class1 `0.9608` nhung precision `0.1623`; reject.
- [x] V78 cumulative ordinal: val macro/class1 `0.7046/0.3720`; reject.
- [x] Them DINO train-only vao internal SSL tool: student/teacher EMA, center update, entropy logging va test `tests/test_internal_ssl_dino.py`.
- [x] V79b DINO soft-temp pretrain 80b x 3e + V64 fine-tune: val macro/class1 `0.7256/0.3922`; macro tang nhung class 1 giam.
- [x] Diagnostic average/router V64+V79b: macro toi da `0.7236`, class1 toi da chi `0.4160`; khong dat gate.
- [x] Export V79b train detailed predictions: train macro `0.8271`, calibrated macro `0.8435`.
- [x] Tao V79b train-only boundary review tai `runs/boundary_review_groupclean_v79b_train_20260626`: 300 rows, pairs `0-1=120`, `1-2=89`, `4-rest=76`, `2-3=15`, buckets `1->0=95`, `4->1=50`, `2->1=29`, `0->1=20`, over-bright/glare `62`.
- [x] Review bang mat 4 mau V79b: loi class1 den tu ranh maturity/quality va vung sang/toi/vet be mat; khong phai nen don thuan.
- [x] Ghi audit tai `docs/TRKH_5CLASS_V75_V79_SSL_AMPLIFICATION_AUDIT_20260626.md`.
- [ ] Gate full train group-clean van la class-1 validation F1 `>=0.70`; best clean single van la V64 `0.4151`, ensemble diagnostic chi `0.4160`.
- [ ] Dung lap them amplification/SSL ngan neu khong co supervision moi; can train-only boundary review/label-policy de tao manifest `clean/ambiguous/ignore/soft-target`.
- [ ] Neu tiep tuc model-side truoc manual review, chi thu phuong phap co annotation/routing gioi han tren boundary da xac nhan, khong dung val/test de tao train signal.

## Cap nhat 2026-06-27

- [x] Cai `cleanlab==2.7.1` va them tool `trkh.tools.audit_label_issues_cleanlab`.
- [x] Them `trkh.tools.evaluate_embedding_retrieval` de thu TIMM/DINO/CLIP/fine-tuned feature kNN, prototype, logistic co cache va smoke caps.
- [x] Sua `trkh.tools.train_prediction_selector` de doc duoc `teacher_pred_name`, `selector_pred_name`, `teacher_pred_index`.
- [x] Chay DINOv2 small retrieval: val/test macro/class1 `0.8918/0.6715` va `0.8751/0.6047`; reject.
- [x] Chay MobileNetV3 fine-tuned feature retrieval: val/test `0.9022/0.7042` va `0.8985/0.6720`; reject.
- [x] Chay EfficientNetV2-S fine-tuned feature retrieval: val/test `0.9105/0.7273` va `0.9009/0.6870`; reject.
- [x] Thu blend top-5 TTA + EfficientNetV2-S kNN: validation tang nhe `0.9143/0.7400`, test giam `0.9169/0.7538`; reject.
- [x] Thu CLIP ConvNeXt raw retrieval: val macro/class1 `0.8828/0.6308`; reject.
- [x] Chay selector top-5 TTA + EffV2 kNN: val/test macro `0.9136/0.9107`; reject.
- [x] Chay cleanlab train audit tren top-5 teacher in-sample; `cleanlab_issue_count=0`, ket luan can OOF train probabilities moi dung dung cach.
- [x] Chay Swin-T pretrained baseline-like probe: best val macro/class1 `0.8343/0.4519`; reject gate.
- [x] Thu neighbor/source refinement prototype bang train labels + Image ID + perceptual hash + RGB/exposure stats; validation `0.9209/0.7609` nhung test `0.9230/0.7634`, thap hon top-5 TTA; reject.
- [x] Ghi audit tong hop tai `docs/TRKH_5CLASS_PRETRAINED_RETRIEVAL_NEIGHBOR_AUDIT_20260627.md`.
- [x] Them safe-probe flags vao `D:\DataAI\AIEx\image_baseline_experiments\scripts\02_train_timm_classifier.py`: max batches, patience, skip-test, best metric, focus-class-name.
- [x] Chay ConvNeXtV2-Tiny pretrained probe 120b x 6e: val macro/focus-class F1 `0.8570/0.5360`; reject gate.
- [x] Chay ViT-B/16 augreg2 pretrained probe 120b x 8e: val macro/focus-class F1 `0.7791/0.3303`; reject gate.
- [ ] Gate full train tren `class_f` van khong duoc mo neu probe khong vuot top-5 TTA validation mot cach on dinh va khong chi overfit validation. Best current artifact van la `runs/ensemble_top5_tta_teacher_cache_5class_v1`.
- [x] Tao OOF train probabilities MobileNetV3 5-fold train-only: `runs\oof_mobilenetv3_5fold_2e120b_20260627`, OOF train macro/class1 `0.8599/0.5403`.
- [ ] Neu uu tien benchmark split cu va chap moi loi the: train/export them expert pretrained da dang hon, chon bang validation, khong tune test.
- [ ] Neu uu tien honest generalization: tiep tuc `class_f_groupclean_v1`, nhung can train-only boundary label-policy/manual review hoac data moi; V64/V79 cho thay model-side amplification/SSL ngan khong du.
- [x] Them AIDT probability exporter dung pipeline train-time; top5+AIDT val tang `0.9188/0.7566` nhung test giam `0.9209/0.7669`, reject.
- [x] Audit oracle 16 experts: test class1 FN cua top-5 TTA la `22`, chi `7` duoc expert hien co bat; con `15` mau can expert/representation moi.
- [x] Them `--train-include-class-name`, `--balanced-train-sampler`, `--eval-only-checkpoint` cho binary focus specialist.
- [x] Them `trkh.tools.fuse_focus_pairwise_specialists` de sweep rule tren val va apply y nguyen sang test.
- [x] Pairwise external specialist `0/1` + `1/2`: val tang `0.9182/0.7557` nhung apply test giam `0.9223/0.7704`; reject, khong full train.
- [x] Probe CaFormer-S18, MobileNetV4 Conv Medium, RepViT-M2 pretrained: best val class1 chi `0.6493`, `0.5704`, `0.6523`; deu duoi top-5 TTA, reject.
- [x] Them `--tta-spatial-crop-fractions` cho TIMM exporter va thu crop fraction `0.86` tren top-5; spatial-only/blend deu thap hon top-5 TTA, reject.
- [x] Tao audit 15 unseen class1 false-negative tai `runs/unseen_class1_fn_audit_20260627`; review nhanh cho thay boundary/quality ambiguity ro, khong phai loi nen don thuan.
- [x] Ghi audit tong hop tai `docs/TRKH_5CLASS_EXPERT_DIVERSITY_AND_TTA_AUDIT_20260627.md`.
- [ ] Khong tiep tuc nem backbone pretrained ngan neu khong co gia thuyet bat nhom `15/22` unseen FN; uu tien OOF/data-cartography hoac expert boundary co train signal moi tu train-only.
- [x] Them robust train-time augmentation flags cho baseline TIMM (`color_jitter`, `random_erasing`, `randaugment`); MobileNetV3 robust probe best val macro/focus `0.8832/0.6515`, reject va khong export test.
- [x] Them percentile-stretch TTA (`channel/luma stretch`) cho `trkh.tools.export_timm_predictions`; MobileNetV3 val giam `0.9040/0.7059 -> 0.9018/0.7048`, reject.
- [x] Sua `trkh.tools.calibrate_classification_logits` de doc teacher-cache cu chi co `true_name`; top-5 logit bias tang val `0.9181/0.7500` nhung giam test `0.9225/0.7727`, reject.
- [x] Them `configs/class_f_5class.yaml` cho cac tool TRKH doc classification-folder dataset `class_f`.
- [x] Them `--train-splits` va `--final-fit` cho baseline TIMM, ghi `optimizer_steps`, ha AMP `init_scale=1024`; smoke final-fit train+val pass voi `optimizer_steps=1`.
- [x] Luu lenh final-fit top-5 train+val tai `docs/TRKH_5CLASS_FINALFIT_COMMANDS_20260627.md`; chua chay full trong phien nay theo quy tac gui lenh truoc full train.
- [ ] Neu nguoi dung chay final-fit top-5, buoc tiep theo la export TTA test, build ensemble bang `configs/class_f_5class.yaml`, audit confusion class focus va so voi top-5 TTA cu.
- [x] Them data-cartography recorder vao TRKH train va launcher V8; probe cartography-only tren `class_f` dat val macro/class1 `0.8809/0.6705`, chua qua gate `0.70`.
- [x] Tao cartography policy manifest train-only va probe soft-target + targeted-margin; val class1 `0.6610`, reject.
- [x] Tao AI-reviewed boundary manifest train-only tu contact sheets; reviewed wide policy val class1 `0.6571`, c1-only targeted-margin `0.6553`, deu reject.
- [x] Export train/val predictions tu cartography checkpoint va thu train->val class-1 reroute; precision tang nhung class1 F1 giam `0.6705 -> 0.6549`, reject.
- [x] Thu diagnostic val-fit reroute upper-bound; class1 F1 chi `0.6790`, duoi gate.
- [x] Sua `trkh.tools.audit_label_issues_cleanlab` de doc `evaluate.py` detailed schema va them test schema; cleanlab train-only tren cartography probe tim `29` issue review candidates.
- [x] Thu blend validation-only top-5 TTA + TRKH cartography alpha `0.46`: val tang `0.9139/0.7383 -> 0.9176/0.7533`, nhung frozen test giam macro `0.9248 -> 0.9235` va class1 khong doi `0.7786`; reject.
- [x] Chay cleanlab tren OOF MobileNetV3: `442` issue train-only, top pairs `0->1=164`, `1->0=112`, `1->2=35`, `3->2=30`, `2->3=26`.
- [x] Them `trkh.tools.build_classification_oof_folds`, `trkh.tools.stitch_timm_oof_predictions`, `trkh.tools.build_oof_neighbor_label_policy`; focused tests pass.
- [x] Tao OOF+cartography consensus `88` rows va probe policy sample-weight + soft-target + targeted-margin: val macro/class1 `0.8803/0.6590`; reject, khong full train.
- [x] Sua `trkh.tools.build_data_centric_sample_weights` de doc cleanlab OOF aliases; tao OOF cleanlab downweight `240` rows; probe val macro/class1 `0.8804/0.6570`; reject.
- [x] Tat hard-sample repeat stale trong V8 cartography probe: best val macro/class1 `0.8805/0.6608`; reject, stale hard-repeat khong phai nut that chinh.
- [x] Them OOF class-1 boundary review package/tool (`203` rows) va strict soft-target tool (`93` rows, alpha `0.85`); probe val macro/class1 `0.8794/0.6566`; reject.
- [x] Ablate metric learning/SCL trong V8 cartography (`MetricLearningLossWeight=0`): best val macro/class1 `0.8801/0.6609`; reject, contrastive noise khong phai nut that chinh.
- [x] Thu focus-class auxiliary head cho class 1 (`loss=0.05`, positive weight `1.25`): best val macro/class1 `0.8817/0.6667`; reject, chua vuot cartography-only `0.6705` va gate `0.70`.
- [x] Tao quality-group manifest train-only (`9215` rows, `26` nhom `class_quality`) va probe GroupDRO `0.05`: best val macro/class1 `0.8796/0.6609`; reject, quality weighting khong mo gate.
- [x] Audit `class_f` vs `yolo_f`: `yolo_f` la YOLO source dataset co object counts trung `class_f`, train co `691` anh multi-object, val co `29`; khong thay missing/malformed/out-of-range label.
- [x] Sua `trkh.tools.trace_architecture` de doc dung YOLO-format bang `MangoYOLOCropDataset`; direct trace retry tren smoke YOLO completed voi `5` samples.
- [x] Probe `yolo_f` object-crop dong khong hard-repeat: best val macro/class1 `0.8797/0.6647`; reject, khong vuot cartography-only `0.6705` va gate `0.70`.
- [x] Them `trkh.tools.build_clean_anchor_manifest` va `tests/test_clean_anchor_manifest.py`; build train-only clean-anchor `800` rows (`160`/class) tu cartography.
- [x] Probe clean-anchor balanced repeat (`repeat=1.6`): best val macro/class1 `0.8788/0.6569`; reject, recall class 1 giam.
- [x] Them `trkh.tools.build_yolo_source_context_review` va test; tao source-context review train-only `runs\yolo_source_context_oof_class1_boundary_review_20260628` voi `160` rows, `0` missing mapping, `0` class_f-vs-YOLO label mismatch, co san cot `manual_expected_class/quality_*` de build reviewed training manifests.
- [x] Expose `CropMarginRatio/ClassCropMargin*` trong launcher V8 va smoke YOLO margin `0.20`; trace completed.
- [x] Probe YOLO global crop margin `0.20`: best val macro/class1 `0.8562/0.6113`; reject, context rong hon lam class 1 te hon.
- [x] Expose YOLO source-context input modes cho classification (`full`, `crop_inset`) va trace/evaluate/train path; focused tests pass.
- [x] Probe YOLO full source-context: best val macro/class1 `0.7573/0.4460`; reject, nen rong lam tin hieu nhiu hon tin hieu doi tuong.
- [x] Probe YOLO crop+context inset: best val macro/class1 `0.8688/0.6535`; reject, van thap hon YOLO object-crop baseline `0.6647`.
- [x] Probe YOLO focus+boundary head: best val macro/class1 `0.8803/0.6667`; reject, tang nhe so voi YOLO object-crop nhung khong vuot `class_f` cartography-only `0.6705`.
- [x] Them bbox spatial prior fusion khong sua data: YOLO dataset tra bbox xywh metadata, model them residual bbox MLP zero-init, eval/trace/train/launcher deu support.
- [x] Smoke bbox prior trace completed; probe `runs\probe_v8_yolof_bboxprior_120b_2e_20260630` best val macro/class1 `0.8825/0.6725`, P/R class1 `0.5979/0.7682`; reject full train vi van duoi gate `0.70`.
- [x] Ket luan: OOF cleanlab huu ich de review/audit, nhung auto soft-target/margin/downweight van khong qua gate `0.70`.
- [x] Giai thich tam thoi vi sao pretrained/AIDT/top-5 TTA vuot TRKH scratch: pretrained/distilled representation co loi the data-scale/inductive-bias lon; yolo_f context/bbox chi cho tin hieu nho nen khong giai quyet ambiguity class 1.
- [x] Them source-context auxiliary bbox-attention loss: YOLO full source view chi dung train-time de ep patch energy vao bbox, khong sua raw data.
- [x] Probe `runs\probe_v8_yolof_sourceaux_bboxattn_120b_2e_20260630`: best val macro/class1 `0.8761/0.6477`; XAI val small cho thay background blur/gray drop gan `0.001`, object desaturate drop `0.1278`; reject vi thap hon bbox-prior va gate `0.70`.
- [x] Them bbox-guided token-prune prior optional (`token_prune_bbox_weight`, `token_prune_bbox_margin_ratio`) va test `test_bbox_token_prior_changes_pruned_patch_selection`; focused pytest `6 passed`.
- [x] Probe ket hop bbox spatial prior + source-context aux bbox-prune `runs\probe_v8_yolof_bboxprior_sourceaux_bboxprune_120b_2e_20260630`: best val macro/class1 `0.8753/0.6480`; inside mass tang `~0.507` nhung class1 F1 giam, reject full train.
- [x] Chay XAI audit nho cho probe bbox-prune: top confusion van `0->1`, `1->0`, `1->2`, `4->1`; background perturbation van gan nhu vo hai, object desaturate moi lam doi du doan ro.
- [x] Don cac run loi/smoke tam: xoa 4 run sourceaux 180b fail va 3 smoke sourceaux; giu lai cac probe co audit.
- [ ] Khong lap lai reviewed soft-target/targeted-margin/reroute/downweight/quality-GroupDRO/YOLO-crop-only/YOLO-wide-margin/source-context/bbox-prior/clean-anchor-repeat don gian tren cung cartography/cleanlab neu khong co manual label-policy moi.
- [ ] Neu tiep tuc cleanlab/data-policy, dung OOF cleanlab de chon anh review train-only, khong auto-relabel/downweight hang loat.
- [ ] Khong tiep tuc full-source auxiliary classification/consistency; neu dung `yolo_f`, chi dung nhu bbox/objectness supervision train-only va phai qua probe class1 `>=0.70`.
- [ ] Huong tiep theo nen la model-side co train signal moi hon: object-aware token dropout/negative background invariance tren crop, hoac internal DINO/MAE dai hon co part-level audit; khong lap lai source-context/bbox-prior loss don thuan.

## Cap nhat 2026-06-30

- [x] Hoan tat pairing `class_f` + `yolo_f`: train/val/test khop object mapping, khong co missing label dang ke; pair co the tra `bbox`, `crop_bbox`, `source_context_image`, `source_context_bbox` va `image_mask` ma khong sua raw data.
- [x] Trien khai valid padding mask tu dataset -> collator -> train/evaluate/XAI cho classification-folder paired YOLO va YOLO crop; focused compile + pytest `tests/test_yolo_source_context_dataset.py tests/test_yolo_bbox_spatial_fusion.py` pass.
- [x] Smoke paired `class_f` + `yolo_f` valid-mask + bbox prior pass, trace completed, XAI small pass.
- [x] Probe `probe_v8_classf_yolopair_validmask_bboxprior_120b_2e_20260630`: val macro/class1 `0.8799/0.6628`, P/R class1 `0.5947/0.7483`; reject vi khong vuot gate `0.70`.
- [x] XAI paired valid-mask: attention foreground/background `0.8521/0.1479`, Grad-CAM foreground/background `0.7721/0.2279`; background blur/gray drop gan `0`, object desaturate drop `0.1489`. Ket luan van la tin hieu qua/surface, khong phai nen rong.
- [x] Mo rong `xai_audit.py` de doc truc tiep YOLO-format bang `MangoYOLOCropDataset`, kem bbox/mask-aware forward va robustness probes.
- [x] Smoke `yolo_f` valid-mask + bbox prior pass, trace completed, XAI small pass.
- [x] Probe `probe_v8_yolof_validmask_bboxprior_120b_2e_20260630`: val macro/class1 `0.8784/0.6647`, P/R class1 `0.5897/0.7616`; reject vi thap hon `probe_v8_yolof_bboxprior_120b_2e_20260630`.
- [x] XAI `yolo_f` valid-mask: attention foreground/background `0.9357/0.0643`, Grad-CAM foreground/background `0.9609/0.0391`; background blur/gray gan nhu vo hai, object desaturate drop `0.0518`. XAI sach hon nhung metric khong tang.
- [x] Ghi audit rieng tai `docs/TRKH_5CLASS_CLASSF_YOLOF_VALIDMASK_AUDIT_20260630.md`.
- [x] Ket luan sau khi tra cuu va doi chieu TransFG/CoAtNet/MaxViT/TokenLearner: hybrid conv-attention va part/token selection la dung huong, nhung trong dataset hien tai source-context/bbox/valid-mask don thuan khong mo gate; can train signal moi tren token/part/boundary.
- [x] Them `bbox_foreground_dropout_mode=random|interior|boundary_band` vao config/CLI/launcher; default `random` giu hanh vi cu. Focused compile + pytest pass.
- [x] Smoke `smoke_v8_yolof_bboxprior_boundarydrop_20260630` pass, trace completed, XAI small pass.
- [x] Probe `probe_v8_yolof_bboxprior_boundarydrop_120b_2e_20260630`: val macro/class1 `0.8821/0.6725`, P/R class1 `0.5979/0.7682`; macro tot hon, class1 chi nhich rat nho so voi bbox-prior cu, reject full train vi duoi gate `0.70`.
- [x] XAI boundary-band: attention foreground/background `0.9805/0.0195`, Grad-CAM `0.9285/0.0715`; background blur/gray gan `0`, object desaturate `0.0912`; top confusion van `3->2`, `0->1`, `1->0`, `4->1`, `1->2`.
- [x] Review nhanh overlay `0->1`, `1->0`, `4->1`: heatmap nam tren qua/vet/sang/ranh cuong do; khong co bang chung nen la nut that.
- [x] Smoke/probe `bbox_foreground_dropout_mode=interior` cung tham so boundary-band: val macro/class1 `0.8794/0.6667`, P/R class1 `0.5888/0.7682`; reject vi thap hon boundary-band va bbox-prior.
- [x] XAI interior: attention foreground/background `0.9802/0.0198`, Grad-CAM `0.9269/0.0731`; robustness giong boundary-band. Ket luan che loi/surface lam class1 giam, khong phai huong chinh.
- [x] Best no-pretrain class-1 probe hien tai la gan nhu hoa giua `runs\probe_v8_yolof_bboxprior_120b_2e_20260630` (`0.8800/0.6724`) va `runs\probe_v8_yolof_bboxprior_boundarydrop_120b_2e_20260630` (`0.8821/0.6725`); chua du full train.
- [x] Don cac smoke `smoke_v8_classf_yolopair_validmask_bboxprior_20260630`, `smoke_v8_yolof_validmask_bboxprior_20260630`, `smoke_v8_classf_yolopair_bboxprior_sourceaux_20260630`, `smoke_v8_yolof_bboxdrop_20260630`, `smoke_v8_yolof_bboxprior_focushead_20260630` sau khi audit da ghi.
- [x] Don smoke `smoke_v8_yolof_bboxprior_boundarydrop_20260630` va `smoke_v8_yolof_bboxprior_interiordrop_20260630` sau khi audit da ghi; giu lai probe.
- [x] Trien khai full learned part-token residual theo TransFG/TokenLearner: config/CLI/launcher/trace/tests pass, smoke + XAI pass, probe `probe_v8_yolof_parttoken_bboxprior_120b_2e_20260630` val macro/class1 `0.8777/0.6648`; reject vi thap hon bbox-prior/boundary-band.
- [x] Mo rong trace voi `09q_part_token_*`, shapes `part_token_foreground_mass/max_weight/route`; trace cho thay attention foreground nhung loang (`max_weight ~0.01-0.015`), khong thanh part sac net.
- [x] Trien khai part-token pairwise margin head: config/CLI/launcher/trace/tests pass, smoke + XAI pass, probe `probe_v8_yolof_parttokenpair_bboxprior_120b_2e_20260630` val macro/class1 `0.8770/0.6571`; reject, pairwise loss gan `0.6914` va khong cai thien class 1.
- [x] Thu foreground-surface pairwise head tren `yolo_f` object crop + bbox spatial prior + boundary-band dropout: probe val macro/class1 `0.8804/0.6628`, XAI foreground/background tot nhung pairwise loss/logits gan nhu khong hoc; reject.
- [x] Thu MixStyle nhe tren `yolo_f` object crop + bbox prior + boundary-band dropout (`p=0.15`, `alpha=0.05`): probe val macro/class1 `0.8789/0.6628`, XAI van sach nhung class 1 khong tang; reject.
- [x] Thu high-frequency/texture expert co route gioi han tren `yolo_f` object crop + bbox prior + boundary-band dropout: probe val macro/class1 `0.8796/0.6648`, high-frequency aux/pairwise loss co hoc nhe nhung khong vuot best; reject.
- [x] Thu boundary-center margin tren `yolo_f` object crop + bbox prior + boundary-band dropout: probe val macro/class1 `0.8776/0.6628`, center loss co terms nhung metric giam; reject.
- [x] Thu pairwise-margin routing tren `yolo_f` object crop + bbox prior + boundary-band dropout: probe val macro/class1 `0.8841/0.6822`, best YOLO no-pretrain hien tai nhung chua qua gate `0.70`.
- [x] Keo dai cung config `pairwise-margin routing + bbox prior + boundary-band` len 4 epoch/120 batch: best epoch van la 2, val macro/class1 `0.8831/0.6802`; XAI foreground sach nhung precision class 1 giam sau epoch 2, reject full train.
- [x] Sweep validation-only `pairwise_margin_route_max_probability_margin=0.08..0.30` tren checkpoint 2e: raw best margin `0.16/0.20` class1 `0.6822`, calibrated class1 cao nhat `0.6933` voi coverage `0.9800`; van duoi gate `0.70`.
- [x] Thu ordinal-boundary nhe tren `0,1,2,3` ket hop pairwise-routing: probe val macro/class1 `0.8844/0.6822`; macro nhich nho nhung class1 khong doi, reject full train.
- [x] Expose `PairwiseMarginLogitScale` trong launcher V8; preflight/smoke pass.
- [x] Sweep validation-only pairwise logit scale `0.15..0.40`: scale `0.35` cho class1 raw `0.6839` nhung chi la trade recall/precision.
- [x] Train probe pairwise logit scale `0.35`: val macro/class1 `0.8833/0.6819`, class1 P/R `0.6010/0.7881`; recall tang nhung false-positive vao class1 tang, reject.
- [x] Thu local-zoom image expert route `0-1,4-1` giu pairwise-routing scale `0.25`: focused tests, preflight, smoke + XAI, 2e/120b probe + XAI pass; val macro/class1 `0.8824/0.6766`, P/R class1 `0.6129/0.7550`; reject vi recall class1 giam so voi pairwise-routing best `0.6822`.
- [x] XAI local-zoom: attention/Grad-CAM foreground `0.9426/0.9648`, background perturb gan `0`, nhung trace `09j/09k` cho thay crop/score con loang va co case `0->1` roi vao cuong/day/nen sat qua; khong lap lai local-zoom crop tu do voi tham so nay.
- [x] Trien khai `InteriorBoundaryPairwiseHead` objectness-gated co bbox mask, trace `09s-09z`, CLI/launcher/config/train loss va focused tests; compile + pytest pass.
- [x] Smoke interior-boundary objectness-gate pass va XAI pass; trace bot bbox rectangle thuan nhung van co block nen/border.
- [x] Probe `probe_v8_yolof_pairroute_interiorboundary_objgate_c1fp_boundarydrop_bboxprior_120b_2e_20260630`: val macro/class1 `0.8813/0.6706`, P/R class1 `0.6032/0.7550`; reject vi thap hon best pairwise-routing `0.8841/0.6822`.
- [x] XAI interior-boundary objgate: attention/Grad-CAM foreground `0.9672/0.9686`, background perturb gan `0`, object desaturate drop `0.0395`; border flags cao (`6/20`, `9/20`) va trace cho thay boundary map bam vien crop/bbox. `train_interior_boundary_pairwise_loss` dung o `0.6914`, khong tang loss/scale tiep.
- [x] Thu bbox object-erasure negative loss tren best pairwise-routing: probe `probe_v8_yolof_pairroute_objecterase_boundarydrop_bboxprior_120b_2e_20260630` val macro/class1 `0.8799/0.6760`, P/R class1 `0.5845/0.8013`; reject vi tang recall nhung lam precision class 1 giam.
- [x] Thu mixed train concat `yolo_f` primary + `class_f` aux weight `0.5`: probe `probe_v8_yolof_pairroute_mixedclassf05_boundarydrop_bboxprior_120b_2e_20260630` val macro/class1 `0.8810/0.6628`; reject.
- [x] Thu paired-view `yolo_f` primary + paired `class_f` crop: probe `probe_v8_yolof_pairroute_pairviewclassf_boundarydrop_bboxprior_120b_2e_20260630` val macro/class1 `0.8813/0.6705`; reject, `class_f` crop khong bo sung signal huu ich.
- [x] Them focused false-positive smooth margin cho `0,4 -> 1`, compile/focused pytest pass; smoke/probe + XAI pass.
- [x] Probe `probe_v8_yolof_pairroute_c1fpsmooth025_boundarydrop_bboxprior_120b_2e_20260630`: val macro/class1 `0.8841/0.6822`, P/R class1 `0.6094/0.7748`, confusion `0->1=49`, `4->1=12`; hoa best pairwise-routing nhung khong giam false-positive, reject full train.
- [x] Thu surface-amplified supervised view nhe theo HERBS/WS-DAN tren best pairwise-routing; compile/focused pytest, preflight, smoke + XAI, probe + XAI deu pass.
- [x] Probe `probe_v8_yolof_pairroute_surfaceamp_boundarydrop_bboxprior_120b_2e_20260630`: best epoch `1`, val macro/class1 `0.8835/0.6746`, P/R class1 `0.6096/0.7550`; epoch `2` giam `0.8816/0.6725`; reject vi thap hon best pairwise-routing `0.8841/0.6822`.
- [x] XAI surface-amplified probe: attention/Grad-CAM foreground `0.9426/0.9603`, background perturb gan `0`, object desaturate drop `0.0455`; confusion van `3->2`, `0->1`, `1->0`, `4->1`, `1->2`, nen surface supervised view khong tao signal moi cho class 1.
- [x] Kiem tra complementarity cua cac checkpoint no-pretrain gan best bang validation probabilities/confusion overlap: best weighted validation ensemble dat macro/class1 `0.8854/0.6934`, oracle 6 model dat `0.9002/0.7273`; co complementarity nhung ensemble/route don gian van duoi gate `0.70`, chua full train.
- [x] Thu combo stacking nhe tu complementarity `pairroute035 + objecterase + surfaceamp`: smoke/XAI pass nhung probe val macro/class1 `0.8812/0.6686`, thap hon best pairroute `0.8841/0.6822`; reject, khong lap lai stacking regularizer cung loai.
- [x] Thu foreground-background recombination train-only p=`0.15` tren `yolo_f` + pairroute/bbox/boundarydrop: val macro/class1 `0.8838/0.6836`, class-1 P/R `0.5961/0.8013`; XAI foreground sach nhung precision giam, chua qua gate.
- [x] Thu foreground-background recombination p=`0.25`: val macro/class1 `0.8808/0.6744`; reject, mix manh hon lam class 1 giam.
- [x] Thu paired-view `yolo_f` primary + `class_f` crop view CE/KL/feature consistency: val macro/class1 `0.8824/0.6744`; pipeline dung nhung khong them signal huu ich.
- [x] Sau combo/fg-bg/paired-view fail, chuyen sang signal dai hoi tu hon thay vi lap lai regularizer surface/logit cung loai.
- [x] Trien khai supervised masked reconstruction/MIM branch train-time nhe tren `yolo_f` pairroute/bbox/boundarydrop: smoke + XAI + 2e probe pass ve pipeline, nhung val macro/class1 `0.8795/0.6705`, P/R class1 `0.5949/0.7682`; XAI foreground sach hon pairroute nhung Grad-CAM van co border/background case, reject full train vi duoi gate `0.70`.
- [x] Thu train-only expert router tu cac checkpoint no-pretrain bo sung nhau (`pairroute`, `c1fpsmooth`, `objecterase`, `pairroute035`, `surfaceamp`, `fgbg015`) bang train/val prediction CSV; khong dung test, khong sua data. Router train-OOF overfit: val macro/class1 `0.8801/0.6528`; binary focus specialist `0.8837/0.6776`; reject.
- [x] Sua offline distillation teacher cache de ho tro `sample_index` vi `yolo_f` co multi-object image trung `image_path`; path-only cache/ensemble co the collapse object va inflate metric.
- [x] Chay lai grid weighted ensemble theo object-level `sample_index`: best focus weights `c1fpsmooth=0.35`, `objecterase=0.25`, `surfaceamp=0.25`, `fgbg015=0.15`, val macro/class1 `0.8864/0.6939`, P/R class1 `0.6198/0.7881`; chua qua gate `0.70`. Ket qua path-collapsed `0.7035` bi xem la invalid cho gate.
- [x] Tao teacher CSV `teacher_probs_train_sampleindex_focus005.csv` va smoke/probe offline ensemble distillation tren best pairroute/bbox/boundarydrop. Probe `probe_v8_yolof_pairroute_distillens_sampleindex_boundarydrop_bboxprior_120b_2e_20260701`: val macro/class1 `0.8840/0.6860`, P/R class1 `0.6114/0.7815`; XAI foreground sach nhung Grad-CAM border flag cao va van co background hotspot trong `0->1`; reject full train.
- [x] Thu mot probe duy nhat ket hop sample-index ensemble distillation voi foreground-background mix p=`0.15`: `probe_v8_yolof_pairroute_distillens_fgbg015_sampleindex_boundarydrop_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8838/0.6799`, P/R class1 `0.5941/0.7947`; `0->1` va `4->1` tang, reject. Dung huong soft-label ensemble student hien tai.
- [x] Them background-focus suppression counterfactual loss va launcher/test; smoke/probe/XAI pass nhung `probe_v8_yolof_pairroute_bgfocus035_boundarydrop_bboxprior_120b_2e_20260701` chi dat val macro/class1 `0.8809/0.6686`, reject.
- [x] Expose `PairwiseMarginPairs` trong launcher va thu explicit `1-4` thay `4-rest`: `probe_v8_yolof_pairroute14_boundarydrop_bboxprior_120b_2e_20260701` dat `0.8814/0.6763`, reject.
- [x] Them teacher-gated focus margin dung teacher CSV sample-index nhung `DistillationWeight=0`: smoke/probe/XAI pass, loss active, XAI sach hon; `probe_v8_yolof_pairroute_teacherfocus035_boundarydrop_bboxprior_120b_2e_20260701` dat `0.8837/0.6822`, hoa class1 best nhung khong vuot gate.
- [x] Thu mot combo rat nhe `teacher_focus_margin + distillation_weight=0.05` vi distill rieng tang class1 len `0.6860` con margin rieng sach XAI; probe `probe_v8_yolof_pairroute_distill005_teacherfocus02_boundarydrop_bboxprior_120b_2e_20260701` chi dat val macro/class1 `0.8834/0.6802`, XAI nen gan nhu vo hai, reject.
- [x] Thu teacher focus-binary objective tren cac neighbor `0,1,2,4`: smoke/XAI/probe pass, `probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8847/0.6860`, P/R `0.6114/0.7815`; XAI foreground sach nhung van duoi gate `0.70`.
- [x] Thu teacher focus-binary + hard-label blend nhe vi teacher binary target qua mem: probe `probe_v8_yolof_pairroute_teacherfocusbinary02_hardblend025_boundarydrop_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8838/0.6802`; reject vi thap hon focus-binary soft `0.6860`.
- [x] Thu foreground-surface + teacher focus-binary: probe `probe_v8_yolof_pairroute_fgsurf_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8805/0.6686`; XAI foreground tot nhung class 1 giam, reject.
- [x] Thu top-k reassessment residual theo huong confusion-aware/router: probe khong aux `0.8817/0.6725`, probe co aux `0.8805/0.6667`; XAI foreground sach nhung adjustment qua yeu va khong giam confusion class 1, reject full train.
- [x] Thu teacher focus-binary + focused false-positive margin tren negatives `0,2,4`: smoke/XAI/probe pass, `probe_v8_yolof_pairroute_teacherfocusbinary015_c1fp025024_boundarydrop_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8819/0.6726`, P/R `0.6064/0.7550`; reject vi lam giam recall class 1.
- [x] Export object-level val/train probabilities cho checkpoint gan best moi (`teacherfocusbinary`, `distillens`, `c1fpcombo`) va grid lai theo `sample_index`; refined 7-expert teacher dat val macro/class1 `0.8864/0.6954`, P/R class1 `0.6142/0.8013`, van duoi gate `0.70`.
- [x] Tao teacher CSV refined `teacher_probs_train_sampleindex_refine005.csv`; train class1 `0.8173`, val class1 `0.6954`, binary p1 mem (`val class1 mean 0.2894`) nen khong dung tiep focus-binary tu teacher nay.
- [x] Thu refined teacher distillation: smoke/trace/XAI pass, probe `probe_v8_yolof_pairroute_distillens_refine005_boundarydrop_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8827/0.6765`; XAI foreground sach nhung border flags con cao, reject full train.
- [x] Phan tich sample-level refined ensemble bang `trkh.tools.analyze_expert_complementarity`: oracle 9 expert val macro/class1 `0.9011/0.7294`, oracle 11 expert `0.9026/0.7353`; chi `3` class-1 FN duoc expert cuu va `11` FP vao class 1 duoc expert chan, nen khong du de dat muc `0.98` bang routing lai cung tap expert hien tai.
- [x] Mo rong `xai_audit.py` voi `--sample-indices` va `--case-csv`, chay audit truc tiep tren cac case oracle/refined sai; XAI tiep tuc cho thay loi surface/illumination/border tren qua, khong phai nen rong.
- [x] Them color-stat/defect-stat fusion dropout vao launcher/train resume allowlist; smoke/test pass. Probe `probe_v8_yolof_pairroute_teacherfocusbinary015_colordefect_boundarydrop_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8845/0.6825`, thap hon focus-binary soft `0.6860`; XAI object-color-sensitive va rollout/background flags cao, reject.
- [x] Them `trkh.tools.remap_yolo_teacher_to_classification_folder` de map teacher object-level ve order `class_f` bang crop filename, tranh loi sample-index sai khi ket hop `class_f+yolo_f`.
- [x] Remap refined/focus teacher sang `class_f`: train/val map du `9215/9215` va `2606/2606`, agreement refined `0.9618/0.9202`, focus `0.9621/0.9206`; smoke fixed-map xac nhan loader agreement `0.9621`.
- [x] Thu `class_f+yolo_f` sourceaux khong teacher: `probe_v8_classf_yolopair_sourceaux_noteacher_bboxprior_120b_2e_20260701` val macro/class1 `0.8824/0.6706`; reject, context rong hon khong giup class 1.
- [x] Thu `class_f+yolo_f` fixed-map teacher-focus + sourceaux: `probe_v8_classf_yolopair_sourceaux_teacherfocus015_fixmap_bboxprior_120b_2e_20260701` val macro/class1 `0.8818/0.6706`; XAI attention/Grad-CAM foreground `0.8567/0.7918`, object desaturate drop `0.1305`, background flags con nhieu; reject full train.
- [x] Doi chieu GCE/noisy-label robust loss tren best teacher-focus-binary route: smoke + XAI sach, nhung probe `probe_v8_yolof_pairroute_teacherfocusbinary015_gce_boundarydrop_bboxprior_120b_2e_20260701` chi dat val macro/class1 `0.8762/0.6609`; reject, khong sweep `gce_q` tiep khi chua co signal moi.
- [x] Export train predictions tu best teacher-focus-binary checkpoint va tao train-only hard-case focus-neighbor binary manifest (`700` rows, matched `878` object samples sau duplicate path, positives/negatives `471/407`) cho cac cap `class1` vs `0/2/4`.
- [x] Them `FocusNeighborBinaryDataset`, loss `class1` vs neighbor logits, launcher/config/CSV history va tests; leakage guard chan val/test path trong manifest.
- [x] Smoke focus-neighbor binary pass, loss active va XAI small foreground/background rat sach (`attention fg~0.978`, `Grad-CAM fg~0.979`), nen da chay 2e/120b probe.
- [x] Probe `probe_v8_yolof_focusneighborbinary_anchor002_boundarydrop_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8818/0.6780`, P/R class1 `0.5911/0.7947`; reject vi thap hon teacher-focus-binary soft `0.8847/0.6860` va lam `0->1` tang `49->56`.
- [x] XAI focus-neighbor binary: foreground van sach (`attention/Grad-CAM fg~0.986`) va background blur/gray gan `0`, nhung object/color/border flags con; ket luan hard-case binary theo train predictions lam precision class 1 giam, khong nen keo full train.
- [x] Sua `trkh.tools.probe_embedding_prototypes` de ho tro dung `yolo_f` object-crop, bbox metadata va image mask nhu `evaluate.py`; compile + `tests/test_probe_embedding_prototypes.py` pass.
- [x] Diagnostic frozen embedding tren `yolo_f` checkpoint best: base head val macro/class1 `0.8847/0.6860`, centroid/kNN/logistic deu thap hon (`logistic_balanced 0.8775/0.6579`). Ket luan head/prototype khong phai nut that chinh cho `yolo_f`.
- [x] Diagnostic `class_f` cung checkpoint: logistic balanced dat `0.8858/0.6824`, cao hon base `class_f` `0.8804/0.6686` nhung van thap hon `yolo_f` base. Two-view rule tho chi nhich class1 len `0.6873`, oracle two-view `0.7556`; co complementarity nhung router phai hoc dung, khong dung rule tay/val-tuned.
- [x] Thu route bilinear patch fusion + multi-granularity auxiliary nhe tren best teacher-focus-binary: smoke + XAI pass, loss active, probe `probe_v8_yolof_bilinear_multigran_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701` dat val macro/class1 `0.8816/0.6780`; reject vi `0->1=54`, `4->1=15` va class1 precision giam.
- [x] XAI bilinear+multi-gran: foreground van sach (`attention/Grad-CAM fg~0.975/0.973`), background blur/gray gan `0`, nhung near-tie `8/20`, Grad-CAM/rollout border `9/20`; ket luan second-order/multi-layer aux khong tao signal separability class 1 voi probe 2e.
- [x] Tra cuu them multi-view stacking, Trusted Multi-View Classification, MV-HFMD va CrossViT; ket luan two-view can router/fusion co train signal sach, khong dung val-threshold/rule tay.
- [x] Sua `remap_yolo_teacher_to_classification_folder` de doc `prob_0_ClassName` va ghi metadata router; remap `teacherfocusbinary` sang `class_f` train/val du `9215/9215`, `2606/2606`.
- [x] Sua bug class-order trong `train_trainval_expert_router`: uu tien suffix cot probability truoc `metrics.json`; compile + `tests/test_trainval_expert_router.py` pass.
- [x] Chay router train-only hai-view `class_f/yolo_f` sau fix: average val macro/class1 `0.8816/0.6725`, selected `logreg_c0p03` val `0.8785/0.6505`; reject vi thap hon `yolo_f` base va train-OOF bi overfit do expert train predictions in-sample.
- [x] Sweep static probability/logit fusion hai-view diagnostic: weight tot nhat van la `yolo_weight=1.0`, class1 `0.6841`; them `class_f` lam giam class1, khong trien khai fusion sau neu khong co OOF/representation moi.
- [x] Sua TTA eval de truyen dung `image_valid_mask`/`bbox_token_prior` qua callback `forward_features`; bbox-aware TTA val `runs/tta_eval_yolof_teacherfocusbinary015_bboxaware_val_20260701` dat macro/class1 `0.8777/0.6648`, reject.
- [x] Thu teacher pairwise-margin objective tren best teacher-focus-binary route: smoke/XAI sach, nhung probe `probe_v8_yolof_pairroute_teacherpairwise02_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701` chi dat val macro/class1 `0.8816/0.6763`; XAI selected20 foreground sach nhung near-tie/border con cao, reject.
- [x] Tra cuu va trien khai checkpoint averaging/model soup theo SWA/Model Soups; them `trkh.tools.average_checkpoints` va `tests/test_average_checkpoints.py`, focused pytest pass.
- [x] Soup top-2 `teacherfocusbinary + distillens` dat val macro/class1 `0.8850/0.6860`: macro tang nhe nhung class1 khong doi; broader soups/top3/top4/top5 va best+last SWA deu thap hon hoặc hoa class1, reject.
- [x] Don 5 soup reject da ghi audit (`top3`, `weighted4`, `weighted5`, `focus_ordboundary`, `best_last`), giu `soup_v8_yolof_top2_teacherfocus_distill_20260701` lam moc diagnostic.
- [x] Export lai object-level val predictions cho `teacherfocusbinary`, `colordefect`, `fgbg015`. Oracle 3 expert dat val macro/class1 `0.8940/0.7097`, nhung best weighted grid chi `0.8859/0.6882`; co complementarity that nhung weighted average khong mo gate.
- [x] Sua `train_trainval_expert_router` de uu tien object-level `sample_index` key, tranh collapse multi-object `yolo_f`; them focused tests object-key duplicate guard (`3 passed`).
- [x] Export train predictions cho `colordefect` va `fgbg015`; chay router train-only object-key 3 expert. Ban no-image-features bi overfit train OOF va val selected chi `0.8757/0.6411`; equal average `0.8821/0.6763`.
- [x] Chay router object-key co image/background features; selected val chi `0.8708/0.6254`, best candidate theo val cung chi khoang `0.8797/0.6687`; context/background feature khong giai quyet duoc class 1.
- [x] Trien khai source-context feature fusion that su cho `yolo_f` crop + full-frame context: model co gated residual fusion head, train/evaluate/XAI deu dung fused logits khi co `source_context_aux`; focused compile va pytest `tests/test_source_context_feature_fusion.py tests/test_yolo_source_context_dataset.py` pass.
- [x] Probe full-update `probe_v8_yolof_pairroute_sourcectxfusion_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` fail gate: val macro/class1 `0.8762/0.6552`, class1 P/R `0.5787/0.7550`, top confusions `0->1=56`, `3->2=55`, `4->1=14`.
- [x] XAI source-context full-update: attention/grad-rollout/Grad-CAM van bam foreground cao (`fg~0.971/0.974/0.927`), background blur/gray gan `0`, object desaturate anh huong lon hon; ket luan full-frame context khong tao signal phan biet nen-vs-object cho class 1 trong cau hinh nay.
- [x] Them `--trainable-module-prefixes` de chi train head moi khi resume checkpoint; smoke head-only xac nhan chi `source_context_fusion_head` trainable (`268318` params), freeze backbone/head cu dung.
- [x] Probe source-context head-only `probe_v8_yolof_sourcectxfusion_headonly_lr02_scale1_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` cung fail: val macro/class1 `0.8760/0.6591`, class1 P/R `0.5771/0.7682`, `0->1=59`, `3->2=59`; reject.
- [x] Tra cuu stacking/OOF va grouped stratification: `cross_val_predict`/stacking can prediction tren held-out fold, `StratifiedGroupKFold` giu ti le class va khong overlap group; vi `yolo_f` multi-object nen group phai la source image, key object phai la `(source_stem, object_index)`.
- [x] Them `trkh.tools.build_yolo_oof_folds` + tests: build 5 train-only grouped-stratified folds cho `yolo_f` vao `runs\yolof_oof_folds_train5_20260702`, source `8064` images / `9215` objects, class counts `[1941,541,1920,2520,2293]`, materialize bang `96768` hardlinks va khong copy raw data.
- [x] Sua `evaluate.py` de `predictions_detailed.csv` co `source_stem`, `label_path`, `object_index`, `primary_label`, bbox metadata cho YOLO object-crop; them test chong mat object metadata.
- [x] Them `trkh.tools.stitch_yolo_oof_predictions` + tests: stitch fold-val predictions bang `(source_stem, object_index)`, detect duplicate/missing object, output `predictions_train_oof.csv` dung global train sample order.
- [x] Smoke fold pipeline: resume-baseline smoke tren `fold_00` chi de kiem data/evaluate/export, XAI small foreground sach va partial stitch map duoc `1858/9215` objects; khong tinh la OOF vi checkpoint resume da train tren full train.
- [x] Smoke OOF-safe scratch tren `fold_00` pass train/trace/XAI nhung 2 batch gan random (`val_macro_f1~0.0308`, model do vao class1); chi xac nhan fold expert co the train tu dau.
- [x] Them `scripts\run_trkh_yolof_oof_fold_experts.ps1`: launcher train/export fold experts tu `runs\yolof_oof_folds_train5_20260702`, default `ResumeCheckpoint=''`, co guard chan resume full-train neu khong co `-AllowUnsafeResume`; dry-run fold_00 va guard unsafe resume deu pass.
- [x] Chay OOF-safe fold_00 scratch probe `probe_oof_yolof_v8_scratch_120b2e_20260702_fold_00` (`2e`, `120` train batches/epoch, skip test): best/eval macro khoang `0.751`, class1 chi `0.331`, confusion `0->1=82`, `1->0=44`, `2->1=27`, `4->1=34`; reject va khong mo rong 5 folds.
- [x] XAI OOF fold_00 selected12: foreground van cao (`attention/grad_rollout/gradcam fg~0.953/0.952/0.983`), background blur/gray gan `0`, object desaturate drop lon hon (`~0.065`); van la boundary/object signal, khong phai context/background.
- [x] Sua OOF launcher: dung hashtable splatting cho PowerShell script params, smoke 1 batch pass; mac dinh export `last.pt` thay vi `best.pt` de tranh chon checkpoint bang fold-val. Eval `last.pt` van fail class1 (`0.317`), nen khong dung OOF scratch schedule hien tai.
- [ ] Dung complementarity oracle de thiet ke router/no-pretrain specialist co target hep hon nhung can signal moi hoac OOF base predictions that; khong chi route lai 9-11 expert hien tai, khong lap lai hard-blend, c1fp margin, top-k residual, foreground-surface head, color/defect stat, sourceaux, teacher/distill mem, two-view in-sample router, hoac TTA photometric voi tham so hien tai.
- [ ] Neu tiep tuc router/ensemble, can OOF predictions that cho tung expert hoac router duoc train bang fused representation trong model; khong dung train in-sample predictions/path-collapsed router lam bang chung gate.
- [x] Khong lap lai source-context feature fusion/full-frame context voi cac mode scale/gate da thu; neu quay lai `class_f+yolo_f`, can OOF predictions that hoac fused representation co supervision sach, khong chi dua vao context rong.
- [ ] OOF expert hop le khong duoc resume tu checkpoint da train tren full train split; neu train 5 fold expert thi dung scratch/no-pretrain hoac pretrain/SSL khong dung label fold-val, sau do export `predictions_detailed.csv` tung fold va stitch bang `trkh.tools.stitch_yolo_oof_predictions`.
- [ ] Khong mo rong OOF scratch fold experts voi schedule `2e/120b`, LR `8e-5`, strict balanced sampler hien tai; class1 qua thap. Chi quay lai OOF neu co warm-start hop le khong label-leakage hoac schedule fold_00 moi co class1 gap ro rang.
- [x] Huong tiep theo: smoke/probe masked reconstruction/MIM nhe tren baseline `yolo_f` de tao representation signal moi, vi context/source/fusion va regularizer surface/part da fail. Da chay probe MIM weight `0.002` tren route teacher-focus-binary/bbox/boundarydrop: val macro/class1 `0.8790/0.6705`, P/R class1 `0.5909/0.7748`; XAI foreground cao nhung near-tie/border va false-positive `0->1/4->1` con, reject full train.
- [x] Them train-time `source_context_focus_suppression` dung full-frame YOLO context chi trong train de phat hien khi context lam tang xac suat class 1 tren negative classes `0/2/4`. Smoke pass va loss active (`fraction~0.56`), nhung probe `probe_v8_yolof_sourcectxfocus015_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` chi dat val macro/class1 `0.8776/0.6648`, P/R class1 `0.5784/0.7815`; XAI selected20 cho thay background perturbation gan `0` nhung `0->1=58`, `4->1=15`, Grad-CAM background/border flags tang. Reject va khong tiep tuc loss context-rong/source-context theo tham so nay.
- [x] Thu balanced-softmax diagnostic tren best teacher-focus-binary route: smoke/XAI pass nhung top confusion va class-1 FN xau hon baseline (`1->0=21`, `1->2=14` tren smoke full-val); khong chay probe 2e vi khong co signal gate.
- [x] Them/thu `PatchObjectnessGuidedHead` dung bbox prior YOLO de ep token objectness va tao residual descriptor object-aware. Probe `probe_v8_yolof_patchobj02_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` dat val macro/class1 `0.8851/0.6839`, P/R class1 `0.6041/0.7881`; macro tang nhe nhung class1 thap hon best `0.6860` va duoi gate. XAI selected20 foreground rat sach (`attention/gradcam bg~0.0168/0.0146`), background blur/gray gan `0`, object_desaturate drop `0.0345`, nhung near-tie `9/20`, Grad-CAM border `10/20`, rollout border `11/20`; reject full train.
- [x] Thu dung mot probe objectness head-only/high-objectness sau run patch-objectness dau. `probe_v8_yolof_patchobj_headonly10_from_patchobj02_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` chi dat val macro/class1 `0.8805/0.6686`, P/R class1 `0.5842/0.7815`; `train_patch_objectness_loss` van quanh `0.833-0.835` va XAI border flags tang (`Grad-CAM border 12/20`, rollout border 13/20). Dung huong patch-objectness theo tham so hien tai.
- [x] Thu `BBoxPriorPatchContextHead` de so sanh pooled token trong bbox voi token ngoai bbox cua chinh crop YOLO. Smoke/trace/XAI pass, nhung probe `probe_v8_yolof_bboxctx_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` chi dat val macro/class1 `0.8815/0.6686`, P/R class1 `0.5959/0.7616`; XAI selected20 cho thay attention/rollout background cao hon best (`0.0633/0.0759`) va confusion `0->1=49`, `4->1=13`, `1->0=20`. Reject; khong lap lai object-vs-background token pooling trong crop voi scale `0.08`/temperature `0.45`.
- [x] Mo rong `trkh.tools.pretrain_internal_barlow` de pretrain SSL train-only tren `yolo_f` object crop; dry-run xac nhan 9,215 object train, balanced exposure hop le. VICReg+SupCon probe `ssl_vicreg_yolof_objectcrop_120b2e_20260702` giam SSL loss `32.68 -> 29.80` va luu best epoch 2, khong dung val/test.
- [x] Fine-tune tu SSL-only checkpoint theo recipe teacher-focus-binary/bboxprior/boundarydrop: smoke/resume/trace pass, XAI smoke pass ve foreground nhung full-val sau 2 batch rat yeu. Probe `probe_v8_yolof_sslvicreg_warmstart_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` chi dat raw val macro/class1 khoang `0.7476/0.3959` o checkpoint best (`epoch 1`), epoch 2 `0.7510/0.3884`; reject. XAI selected20 foreground cao va background perturb gan `0`, nhung top confusions `3->2=111`, `0->1=77`, `4->1=58`, `1->0=44`; ket luan SSL-only warm-start thay the supervised checkpoint lam classifier class 1 hong, khong phai loi nen.
- [x] Them `--init-checkpoint` cho `trkh.tools.pretrain_internal_barlow` de DINO/VICReg/MAE train-only co the giu checkpoint supervised lam neo ma khong resume optimizer/epoch; compile + `tests/test_internal_ssl_mae.py` pass (`8 passed`).
- [x] Chay DINO train-only object-crop tu checkpoint supervised `teacherfocusbinary015`: dry-run pass, run `ssl_dino_yolof_objectcrop_supervisedinit_60b1e_20260702` 60 batch/1e dat DINO loss `3.4983`, teacher entropy `1.1496`, khong doc val/test. Fine-tune smoke + XAI `smoke_v8_yolof_dino_supervisedinit60b_teacherfocusbinary015_boundarydrop_bboxprior_20260702` cho confusion xau (`0->1=52`, `4->1=22`, `2->1=20`) du foreground sach; reject, khong probe 2e tu DINO 60b nay.
- [x] Thu surface-counterfactual consistency nhe tu checkpoint supervised `teacherfocusbinary015` lam neo thay vi SSL-only random head. Smoke/XAI sach, nhung probe `probe_v8_yolof_surfacecons02_from_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` chi dat val macro/class1 `0.8828/0.6783`, P/R class1 `0.6031/0.7748`; XAI selected20 foreground cao (`attention/Grad-CAM bg~0.0191/0.0164`), background blur/gray drop gan `0`, nhung `0->1=51`, `3->2=51`, Grad-CAM border `9/20`, rollout border `12/20`. Reject, khong dung surface-consistency hien tai de full train.
- [x] Trien khai learned paired-view feature fusion that su cho inference `yolo_f + class_f`: them `paired_view_fusion_head`, train-time fused CE/KL, evaluate learned fusion, launcher flags, tests, va sua `--trainable-module-prefixes` de freeze optimizer/BatchNorm/EMA dung khi chi train head moi. Static two-view sweep truoc/sau van chon yolo-only.
- [x] Smoke `smoke_v8_yolof_classf_pairfusion_headonly_bnfreeze_emafix_teacherfocusbinary015_20260702` pass: non-paired state diff `0`, fusion loss active, XAI selected12 foreground sach (`attention/Grad-CAM fg~0.983/0.982`) va background blur/gray gan `0`; learned fusion tren full val bang yolo-only `0.8829/0.6783`.
- [x] Probe `probe_v8_yolof_classf_pairfusion_headonly_bnfreeze_emafix_teacherfocusbinary015_120b_2e_20260702` reject: best epoch 1 val macro/class1 `0.8837/0.6822`, P/R class1 `0.6094/0.7748`, duoi best `0.8847/0.6860` va duoi gate `0.70`. Paired learned fusion tren `best.pt` bang yolo-only `0.8829/0.6783`; `last.pt` learned fusion con thap hon `0.8775/0.6667`. XAI selected12 khong doi top confusions (`3->2=52`, `0->1=49`, `1->0=18`, `4->1=12`) va tiep tuc cho thay nen khong phai nut that.
- [x] Export/evaluate AIDT pretrained `ResNet50+ViT-B/16` tren `class_f` va align voi TRKH `yolo_f`: AIDT val macro/class1 `0.9088/0.7169`; oracle AIDT+TRKH val macro/class1 `0.9426/0.8375`, co complementarity that nhung teacher cung tao FP1.
- [x] Them `trkh.tools.remap_classification_teacher_to_yolo` + test de map AIDT classification-folder probabilities sang `yolo_f` sample_index. Remap train du `9215/9215`, teacher-label agreement `0.9861`, teacher du doan class1 `627` so voi hard label class1 `541`.
- [x] Smoke/probe AIDT soft KD + focus-binary reject: `probe_v8_yolof_aidtsoft003_tfbinary010_boundarydrop_bboxprior_120b_2e_20260702` chi dat val macro/class1 `0.8844/0.6804`, P/R `0.6105/0.7682`, XAI foreground sach nhung near-tie/border con; khong lap lai global AIDT KD `0.03` + focus-binary `0.010`.
- [x] Thu AIDT focus-only co confidence/agreement gate, `DistillationWeight=0`: probe `probe_v8_yolof_aidtfocusonly015_conf65_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` chi dat val macro/class1 `0.8815/0.6763`, P/R class1 `0.6000/0.7748`; XAI foreground sach nhung top confusion van `0->1=52`, `3->2=52`, `1->0=18`, reject.
- [x] Them sample-index support cho targeted-margin manifest de tranh collapse multi-object path trong `yolo_f`; focused tests `tests/test_targeted_margin_sample_index.py` va targeted-margin regression pass.
- [x] Them `trkh.tools.build_aidt_rescue_margin_manifest` + test: tao manifest train-only tu case baseline sai nhung AIDT dung/confident. Manifest hien tai co `121` rows, gom `10` class-1 false-negative rescue va `111` false-positive class-1 suppression, key bang `sample_index`.
- [x] Thu AIDT rescue targeted-margin nhe (`TargetedMarginLossWeight=0.015`) ket hop teacher focus-binary goc, khong soft KD AIDT: smoke 24 batch loss active, probe `probe_v8_yolof_aidtrescuemargin015_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260702` dat val macro/class1 `0.8817/0.6780`, P/R class1 `0.5911/0.7947`; XAI selected20 foreground sach nhung `0->1=56`, reject va khong full train.
- [x] Trien khai va thu teacher non-target/DKD gated distillation tu AIDT: `probe_v8_yolof_aidtnckd008_conf70_boundarydrop_bboxprior_120b_2e_20260702` dat val macro/class1 `0.8818/0.6763`, XAI foreground sach nhung van `0->1`, `1->0`, `3->2`; reject va khong lap lai AIDT DKD voi confidence/weight hien tai.
- [ ] Neu tiep tuc SSL/representation, phai co signal boundary/part moi that su va giu checkpoint supervised lam neo; probe surface-consistency nhe va DINO 60b supervised-init da fail. Khong lap lai SSL-only random-head warm-start, MIM RGB don thuan, DINO ngan 60b entropy thap, source-context fusion, source-context focus suppression, patch-objectness/bbox-context hien tai, surface-consistency weight `0.02`, hay router train-in-sample.
- [ ] Khong lap lai focus-neighbor binary objective dua tren hard-case manifest train-predictions voi weight `0.02`/neighbor `0,2,4`; no tang false-positive vao class 1.
- [ ] Khong tiep tuc prototype/cosine-centroid head rieng tren `yolo_f` neu khong co embedding moi; diagnostic da cho thay base head tot hon centroid/kNN/logistic.
- [ ] Khong lap lai bilinear patch fusion + multi-granularity aux loss `0.02` voi teacher-focus-binary route hien tai; no lam class-1 false-positive tang.
- [ ] Huong tiep theo nen tao representation signal moi that su hoac OOF base predictions train-only; khong lap lai two-view `class_f/yolo_f` residual/static/router theo tham so hien tai. Neu quay lai two-view thi can cross-attention/evidence objective moi co signal train sach, khong chi residual logits nho tren frozen embedding.
- [x] 2026-07-02 disk hygiene: tao `docs\TRKH_5CLASS_RESEARCH_JOURNAL.md`, ghi keeper artifacts/current baselines/do-not-repeat list, tao manifest `runs\cleanup_manifest_20260702_obsolete_smoke_probe.json`, xoa `175` obsolete `smoke_*`/rejected `probe_v8_yolof_*`/OOF fold materialization, tang D: free tu khoang `2.55 GB` len `35.94 GB`. Cac artifact best/final/AIDT comparison da kiem tra con ton tai.
- [x] 2026-07-02 thu ordinal-distribution/EMD label-distribution smoothing tren current teacher-focus-binary anchor sau khi tra cuu EMD/DLDL. Smoke loss active va trace pass, nhung XAI selected12 loang hon anchor (`attention/Grad-CAM fg 0.8405/0.7375`), `1->0=19`, `0->1=50`, background perturbation van gan `0`; reject truoc probe va khong lap lai `weight=0.02`, classes `0,1,2,3,4`, sigma `0.55` tren anchor nay.
- [x] 2026-07-02 thu SAM optimizer rho `0.015` tren current teacher-focus-binary anchor sau khi tra cuu SAM/ASAM. Sua bug SAM replay trong `train.py`, smoke loss active va trace pass, nhung XAI selected12 van loang (`attention/Grad-CAM fg 0.8397/0.7459`), `1->0=19`, `0->1=50`, Grad-CAM background `0.2541`; reject truoc probe va khong lap lai SAM rho `0.015` non-adaptive voi attention-view off tren anchor nay.
- [x] 2026-07-02 tra cuu SNSCL/Sel-CL cho noisy fine-grained labels, trien khai teacher-guided SupCon co sample-weight + stochastic feature embedding. Smoke `tgcontrast012 confidence_margin sstd0.04` loss active (`0.8812`) nhung selected signal thua (`fraction=0.1068`, ~`3.42` mau/batch), XAI selected12 loang (`attention/Grad-CAM fg 0.8401/0.7411`), `1->0=19`, `0->1=50`, Grad-CAM background `0.2589`; reject truoc probe va khong lap lai exact setting nay.
- [ ] Huong tiep theo nen tam dung cac loss smoothing/logit-distribution tong quat; uu tien phan tich case-level/representation signal co kha nang thay doi cue surface-boundary truc tiep ma khong lam heatmap loang them.
- [ ] Neu tiep tuc optimizer-level, can co ly do manh hon SAM rho `0.015`; uu tien phuong phap co the tac dong vao representation boundary/texture ma khong tat AMP hoac lam XAI loang.
- [ ] Neu tiep tuc contrastive/noisy-label, khong chi dung teacher agreement sparse tu teacher CSV hien tai; can co signal/reliability train-only dam bao du positive pairs class 1 ma khong lam heatmap loang.
- [x] 2026-07-02 sua loi artifact AIDT teacher CSV theo view: CSV `teacher_probs_yolof_train_sampleindex_aidtpretrained_ttaflip.csv` dung voi `yolo_f` order nhung smoke lai dung primary `class_f`, lam loader agreement chi `0.1310` va contrastive fraction `0.1016`. Them guard `TeacherProbabilityDataset` kiem tra sample-index/path overlap va test regression de chan sample_index cua view khac.
- [x] 2026-07-02 tao teacher CSV AIDT dung `class_f` order:
  `runs\aidt_classf_pretrained_train_20260702\teacher_probs_classf_train_sampleindex_aidtpretrained_ttaflip.csv`,
  mapped `9215/9215`, agreement `0.9861`, overlap ratio `1.0`.
- [x] 2026-07-02 smoke corrected AIDT teacher-guided SupCon `weight=0.008/temp=0.18/confidence/std=0.03/minrel=0.20`: signal khong con thua (`fraction=0.7734`, `24.75` mau/batch) nhung loss phu qua lon (`3.1536`) va XAI khong cai thien top confusion (`3->2=52`, `0->1=50`, `1->0=19`), Grad-CAM foreground van thap `0.7409`, background `0.2591`; reject truoc probe va khong lap lai exact setting nay.
- [x] 2026-07-02 tra cuu ArcFace/Sub-center ArcFace va thu angular-margin hep tren anchor `yolo_f` (`weight=0.01`, margin `0.06`, scale `12`, classes `0,1,2,4`). Smoke full-val co class1 tang nhe `0.6879`, nhung probe 2e/120b tut xuong macro/class1 `0.8824/0.6761`, P/R class1 `0.5882/0.7947`, `0->1=56`; XAI foreground rat sach nhung background khong phai nut that. Reject full train va khong lap lai plain cosine/angular-margin setting nay.
- [x] 2026-07-02 cleanup angular-margin reject: xoa run loi primary view, smoke hop le, eval smoke phu va probe reject; manifest `runs\cleanup_manifest_20260702_angmargin001_m006_reject.json`, reclaimed `501.57 MB`, D: free `36.51 GB`.
- [x] 2026-07-02 cleanup sourceaux reject: xoa 2 probe `class_f+yolo_f` sourceaux da audit reject (`noteacher`, `teacherfocus015_fixmap`), manifest `runs\cleanup_manifest_20260702_classf_yolopair_sourceaux_reject.json`, reclaimed `528.15 MB`.
- [x] 2026-07-02 da thu reliability-gated boundary-center/sub-center-lite tren anchor `yolo_f`: smoke full-val nhich class1 `0.6879` nhung probe tut `0.8817/0.6781`, P/R class1 `0.5950/0.7881`, `0->1=53`, `4->1=13`; XAI foreground sach va background perturb gan `0`, nen reject va khong lap lai `weight=0.01`, pairs `0-1,1-2,2-3,1-4`, margin `0.08`, teacher conf `0.30`, require agreement, `confidence` weighting.
- [x] 2026-07-02 tra cuu TransFG/FFVT/TokenLearner va thu lai micro-detail token selection tren current teacher-focus-binary anchor: smoke `topK=6`, aux `0.012`, scale `0.12`, route `0-1,1-2,2-3,1-4` dat val macro/class1 `0.8843/0.6879`, thap macro hon anchor va van `0->1=50`, `4->1=12`; XAI foreground sach nhung khong co tin hieu moi, route weight trace sparse. Reject truoc probe va khong lap lai setting nay.
- [x] 2026-07-02 diagnostic pairwise calibration guard: global class1 bias khong giup, nhung rule val-only reassign du doan class1 sang rival `0/3/4` khi margin nho dat current-code val macro/class1 `0.8892/0.6993` va original anchor CSV `0.8895/0.7036`. Train-in-sample threshold fit overfit: train `0.9454/0.8347` nhung val chi `0.8861/0.6859` hoac `0.8864/0.6901`. Final-test audit voi nguong da chot tu val fail: base test `0.8865/0.6667`, guard val-best `0.8863/0.6622`, trainfit `0.8829/0.6486`. Reject post-hoc val-only/train-in-sample class1 calibration guard.
- [x] 2026-07-02 phan tich AIDT vs TRKH val aligned: AIDT sua `112` loi TRKH nhung tao `48` loi moi; sua nhieu `3->2`, `0->1`, `4->1`, `1->0` nhung van tao FP class1. Ket luan AIDT manh do representation pretrained/complementarity, khong phai teacher target sach; khong lap lai global/focus KD AIDT neu khong co disagreement-aware gate OOF/fold-safe.
- [x] 2026-07-02 diagnostic AIDT disagreement gate: rule class1-aware fit tren val khong tune test dat val macro/class1 `0.9171/0.7541`; final test cung tang so voi TRKH va AIDT don le (`0.8888/0.6857` so voi TRKH `0.8865/0.6667`, AIDT `0.8575/0.5806`) nhung van duoi gate `0.70` va phu thuoc AIDT inference. Giu artifact `runs\aidt_disagreement_gate_audit_20260702`, khong xem la no-pretrain TRKH moi.
- [ ] Neu tiep tuc margin/representation, khong dung plain margin hoac center pressure tren toan boundary noisy nua; can signal local-part/texture co kha nang doi cue ben trong qua thay vi chi day embedding/logit class 1.
- [ ] Neu tiep tuc token/part selection, khong dung residual head top-K/TokenLearner/local-zoom/high-frequency theo setting da reject; can co supervision moi hon, vi du teacher/OOF-guided case routing hop le hoac local pair objective co bang chung giam `0->1` ngay o smoke.
- [ ] Neu tiep tuc calibration/router, bat buoc can OOF/fold-safe predictions hoac representation moi; khong lap lai threshold tay tune truc tiep tren val hay train-in-sample margin guard cho class1.
- [ ] Neu tiep tuc AIDT, uu tien compress disagreement-aware gate vao TRKH/no-pretrain hoac router fold-safe; khong lap lai global soft KD, focus-only KD, DKD, rescue margin, hay soft ensemble don gian.
- [x] 2026-07-02 cleanup historical rejects: giu keeper/final/AIDT audit, xoa `106` thu muc probe/smoke/eval cu da document reject (`probe_mango_cls_*`, `probe_groupclean_*`, cartography/cleanlab/v8 policy cu, paired-fusion eval reject, class_f+yolo_f probe cu), manifest `runs\cleanup_manifest_20260702_historical_reject_probes.json`, reclaimed `16.233 GB`, D: free ~`53.28 GB`.
- [x] 2026-07-02 phan tich compress AIDT disagreement gate: train gate switch `230` mau, train macro/class1 tang `0.9399/0.8154 -> 0.9740/0.9097` nhung gan nhu toan correction in-sample (`225` wrong->right), overlap `116/121` voi rescue margin da fail. TRKH-only logit/prob calibrator hoc tu train khong tai tao duoc gain tren val; best class1 chi bang/thap hon base `0.6783`. Ket luan: khong dung train-logit router/calibrator de compress AIDT gate hien tai.
- [x] 2026-07-02 sua sample-weight manifest ho tro `sample_index` va chan path fallback khi row co `sample_index`, tranh collapse multi-object `yolo_f`; focused tests sample-weight/teacher-probability/targeted-margin pass (`6 passed`).
- [x] 2026-07-02 thu AIDT reliability soft sample reweight train-only (`245` object rows, sample_index strict, max weight `1.25`): smoke loader hop le (`paths=0`, `sample_indices=245`, `matched=245`) nhung full-val macro/class1 chi `0.8837/0.6841`, P/R class1 `0.6082/0.7815`, `0->1=50`, `4->1=12`; XAI foreground sach va background khong phai nut that. Reject truoc probe, xoa smoke checkpoints, giu eval/XAI/manifest.
- [ ] Khong lap lai AIDT reliability sample reweight exact policy `245 rows/max1.25` hoac TRKH-only train-logit gate/calibrator. Neu tiep tuc AIDT, can signal khac AIDT representation-level/fold-safe hon, khong phai ep target, margin, sample-weight mem, hay logit calibration in-sample.
- [ ] Huong tiep theo nen danh gia representation-level distillation co cau truc hon (vi du feature relation/attention-map alignment co gate disagreement va confidence) hoac self-supervised/part signal moi, nhung phai co smoke evidence giam `0->1`/`4->1` truoc khi probe.
- [x] 2026-07-02 trien khai AIDT feature export/remap + `TeacherFeatureDataset` + RKD distance loss de thu representation-level distillation thay vi logit KD. Smoke `teacher_feature_rkd_loss_weight=0.02`, source `head`, distance-only, AIDT clean concat features active (`loss=0.08646`) nhung full-val chi dat macro/class1 `0.8792/0.6707`, P/R class1 `0.6120/0.7417`; XAI foreground sach va background perturb gan `0`, object desaturate drop `0.0682`. Reject truoc probe, khong lap lai exact global feature-RKD setting nay.
- [ ] Neu tiep tuc AIDT feature distillation, khong dung RKD toan batch/toan lop nua; can gate theo disagreement/confidence/OOF hoac class-pair cu the va phai co smoke evidence tang recall class 1 ma khong tang `0->1`/`4->1`.
- [x] 2026-07-02 them boundary-gated AIDT feature RKD (`teacher_feature_rkd_pair_mode=boundary`) va test mask pairwise. Smoke `weight=0.01`, source `head`, pairs `0-1,1-2,1-4,2-3` active (`loss=0.05643`, pair_count `328.25`) nhung full-val chi dat macro/class1 `0.8801/0.6727`, P/R `0.6154/0.7417`; `0->1` giam `47` nhung `1->0=21`, `1->2=13`; XAI foreground sach, background gan `0`, object_desaturate `0.0704`. Reject truoc probe, khong lap lai exact boundary-RKD setting nay.
- [ ] Neu tiep tuc feature distillation, can doi signal that su: source `patch/registers` hoac confidence/disagreement weighting theo sample/cap class. Khong chi giam weight global/boundary RKD tren `head` vi hai smoke lien tiep deu lam class-1 recall xau.
- [x] 2026-07-02 thu patch-source boundary AIDT feature RKD `weight=0.01`, pairs `0-1,1-2,1-4,2-3`: active (`loss=0.06189`, pair_count `328.25`) nhung full-val macro/class1 chi `0.8808/0.6766`, P/R `0.6175/0.7483`, confusion `0->1=47`, `1->0=20`, `1->2=12`; XAI foreground sach va background khong phai nut that. Reject truoc probe, khong lap lai exact patch-boundary RKD setting nay.
- [ ] Dung huong AIDT RKD source `head/patch` voi feature concat clean hien tai. Neu tiep tuc AIDT, can khai thac disagreement gate bang cach khac hoac tao teacher signal local/part moi, khong dung relation-distance loss cung cache/setting nay.
- [x] 2026-07-02 them residual MLP classification head checkpoint-compatible (`head.weight/head.bias` giu key cu, `head.residual_*` zero-init) sau khi tra cuu ViT/MLP-Mixer; compile + focused pytest pass (`7 passed`).
- [x] 2026-07-02 smoke residual MLP head full-update `hidden=512/dropout=0.08/scale=0.20` voi teacher-focus-binary anchor: full-val macro/class1 chi `0.8732/0.6477`, P/R class1 `0.5672/0.7550`, `0->1=55`, `4->1=16`; XAI foreground cao nhung border/object-color sensitivity con, reject truoc probe.
- [x] 2026-07-02 smoke residual-only MLP head freeze backbone/base linear (`397,318` trainable params, LR `5e-4`): full-val macro/class1 `0.8750/0.6551`, P/R `0.5825/0.7483`, `0->1=53`, `1->2=14`, `3->2=60`; XAI background gan `0`, object desaturate dominant, reject truoc probe.
- [x] 2026-07-02 cleanup residual-MLP reject smokes: xoa 2 thu muc smoke train, giu eval/XAI, manifest `runs\cleanup_manifest_20260702_residual_mlp_head_reject_smokes.json`, reclaimed `322.19 MB`, D: free ~`56.72 GB`.
- [ ] Khong lap lai deep/residual MLP classification head tren embedding hien tai theo full-update `hidden=512/dropout=0.08` hoac residual-only LR `5e-4`; diagnostic prototype/logistic va hai smoke nay deu noi rang head capacity khong phai nut that chinh.
- [x] 2026-07-02 cleanup legacy detection/4-class: giu keeper/final/AIDT audit, xoa `16` thu muc old `mango_hybrid_*`, `mango_detr_*`, object-crop/4-class non-keeper; manifest `runs\cleanup_manifest_20260702_obsolete_legacy_detection_4class.json`, reclaimed `4.496 GB`.
- [x] 2026-07-02 cleanup old probes/bench/rejected pretrain: xoa `66` thu muc `mango_cls_*probe*`, `worker_bench_*`, `audit_perf_*`, `audit_smoke_train_*`, `pretrain_groupclean_*`; manifest `runs\cleanup_manifest_20260702_legacy_probes_bench_pretrain_rejects.json`, reclaimed `9.030 GB`, D: free ~`70.26 GB`, keeper khong mat.
- [ ] Khong dung cac run legacy detection/full-frame/4-class/old probe lam gate cho current `class_f`/`yolo_f`; chi xem chung la ghi chu lich su neu metrics da nam trong docs. Gate hien tai van la no-pretrain TRKH `teacherfocusbinary015` va cac audit final/keeper da preserve.
- [x] 2026-07-02 trien khai self-adaptive target loss train-only theo `sample_index`, compile + focused pytest pass (`4 passed`). Smoke `weight=0.015/beta=0.90/hard_weight=0.10` nhich class1 len `0.6879` nhung probe 2e/120b fail: val macro/class1 `0.8798/0.6760`, P/R class1 `0.5845/0.8013`, `0->1=58`, `4->1=12`; XAI foreground rat sach nhung object desaturate van dominant va background gan `0`.
- [x] 2026-07-02 cleanup SAT reject: xoa 3 thu muc train SAT bi loai, giu XAI smoke/probe va keeper/final; manifest `runs\cleanup_manifest_20260702_sat015_reject_smoke_probe.json`, reclaimed `364.30 MB`, D: free ~`69.80 GB`.
- [ ] Khong lap lai SAT exact setting `weight=0.015/beta=0.90/hard_weight=0.10/start=2` tren current anchor. Vi SAT hard-agreement `1.0`, no dang reinforce hard labels hon la tao signal moi cho boundary/noisy class 1.
- [ ] Sau khi don SAT reject, huong tiep theo phai co signal moi truc tiep cho local surface/part/disagreement, khong them loss noisy-label tong quat chi dua tren hard label/EMA neu smoke khong giam `0->1` ro rang.
- [x] 2026-07-02 tra cuu mixup/manifold mixup/CP-Mix va trien khai confusion-pair manifold mixup tren head-input thay vi image-space mixup de khong pha bbox/object semantics. Preflight compile + focused pytest pass; guard cung bat duoc teacher CSV sai view (`class_f` path overlap `0`) truoc khi train hop le.
- [x] 2026-07-02 smoke CP-Mix hop le `confusion_pair_mixup_loss_weight=0.006/alpha=0.40/pairs=0-1,1-2,1-4,2-3` active (`loss=1.3871`, fraction `0.7695`, ~`24.625` pairs/batch) nhung val macro/class1 chi `0.8805/0.6741`, P/R class1 `0.5817/0.8013`; top confusion `0->1=57`, `4->1=15`.
- [x] 2026-07-02 XAI CP-Mix selected12 foreground van cao (`attention/gradcam fg 0.9797/0.9726`) va background gan nhu vo hai, nhung object_desaturate drop `0.1256`; ket luan hidden-state mix tren confusion pairs noisy lam vung class1 rong hon, reject truoc probe.
- [x] 2026-07-02 cleanup CP-Mix reject: xoa 2 run loi preflight va 1 smoke hop le bi reject, giu XAI va keeper/final/AIDT artifacts; manifest `runs\cleanup_manifest_20260702_cpmix006_reject_smokes.json`, reclaimed `182.04 MB`, D: free ~`69.74 GB`.
- [ ] Khong lap lai CP-Mix exact setting `weight=0.006/alpha=0.40/pairs=0-1,1-2,1-4,2-3` tren current anchor. Neu tiep tuc mixup-like thi can reliability/OOF/disagreement signal chon cap sach hon, khong mix truc tiep cac cap dang tao FP class 1.
- [x] 2026-07-02 tra cuu ConViT/CvT va thu architectural signal nhe: them `BranchCnnTokens=1` de dua pooled CNN-stem token vao transformer prefix stream. Da them resume adapter giu lai color/edge branch embeddings cu va chi cho phep missing `branch_token_fusion.cnn_branch.*`; compile + focused pytest pass (`3 passed`).
- [x] 2026-07-02 smoke branch CNN token `branch_cnn_tokens=1` tren current teacher-focus-binary anchor dat val macro/class1 `0.8841/0.6785`, P/R class1 `0.6117/0.7616`; `0->1` giam nhe `49->47` nhung `1->0=18`, `1->2=12`, `3->2=57`, nen recall class1 tut va khong probe.
- [x] 2026-07-02 XAI branch CNN token selected12 foreground sach hon (`attention/gradcam fg 0.9834/0.9821`, Grad-CAM border `2/12`) nhung background blur/gray van gan `0` va object_desaturate drop `0.1226`; ket luan CNN prefix token chi lam focus sach hon, khong them signal surface/boundary phan lop class1.
- [x] 2026-07-02 cleanup branch CNN token reject: xoa smoke train dir, giu XAI va keeper/final/AIDT artifacts; manifest `runs\cleanup_manifest_20260702_branchcnn1_reject_smoke.json`, reclaimed `182.99 MB`, D: free ~`69.68 GB`.
- [ ] Khong lap lai `BranchCnnTokens=1` don le tren anchor hien tai. Neu tiep tuc CNN/ViT hybrid, can supervision/auxiliary signal moi cho CNN token hoac thay doi representation tu dau; khong chi them prefix token roi fine-tune 24 batch.
- [x] 2026-07-02 export train predictions cua anchor vao `runs\eval_yolof_teacherfocusbinary015_best_train_20260702`: train macro/class1 `0.9400/0.8164`, cao hon val nhieu nen calibration tren train la diagnostic in-sample, khong duoc xem la gate.
- [x] 2026-07-02 thu grouped train-holdout class1 margin guard (fit tren 4/5 train groups, holdout theo `source_stem`, khong dung val de fit). Best average holdout threshold `0.045/min_p1=0.0` cho val macro/class1 `0.8851/0.6817`, thap hon anchor class1 `0.6860`; val changes co `16` sua `0->1->0` nhung lam hai `9` true class1 sang `0` va `2` sang `4`.
- [ ] Khong tiep tuc OOF/fold-safe class1 margin guard tu tin hieu threshold nay; train-holdout da khong giu duoc recall class1 tren val. Neu quay lai calibration, phai co representation/teacher signal moi, khong chi margin top2 cua TRKH.
- [x] 2026-07-02 tra cuu R-Drop va trien khai dropout-output consistency loss tuy chon (`rdrop_loss_weight`, `rdrop_temperature`) cho classification path, compile + focused pytest pass (`5 passed`).
- [x] 2026-07-02 smoke R-Drop `weight=0.10/temp=1.0` tren current teacher-focus-binary anchor: loss active nho (`train_rdrop_loss=0.0020`) nhung val macro/class1 chi `0.8781/0.6648`, P/R class1 `0.5681/0.8013`, `0->1=59`, `4->1=16`.
- [x] 2026-07-02 XAI R-Drop selected12: foreground van sach (`attention/gradcam fg 0.9796/0.9727`), background blur/gray gan `0`, object_desaturate drop `0.1262`; ket luan R-Drop lam vung class1 rong hon va tang FP, reject truoc probe.
- [x] 2026-07-02 cleanup R-Drop reject: xoa smoke train dir, giu XAI va keeper/final/AIDT artifacts; manifest `runs\cleanup_manifest_20260702_rdrop010_reject_smoke.json`, reclaimed `180.05 MB`, D: free ~`69.61 GB`.
- [ ] Khong lap lai R-Drop exact setting `weight=0.10/temp=1.0` tren current anchor. Neu thu consistency tiep, can signal bat doi xung theo class-pair/reliability hoac cue local moi, khong phai KL toan distribution giua hai dropout pass.
- [x] 2026-07-02 tra cuu multi-task/CO-TASK va trien khai semantic-attribute hierarchical logit loss khong them nhan moi: specs `maturity:0,1|2,3|4;transport:0,2|1|3,4;quality:0,1,2|3|4`, compile + focused pytest pass (`8 passed`).
- [x] 2026-07-02 smoke semantic-attribute `weight=0.02` tren current anchor: loss active (`0.7272`, 3 task) nhung val macro/class1 chi `0.8796/0.6685`, P/R class1 `0.5735/0.8013`, `0->1=57`, `4->1=16`.
- [x] 2026-07-02 XAI semantic-attribute selected12: foreground sach (`attention/gradcam fg 0.9795/0.9724`), background blur/gray gan `0`, object_desaturate drop `0.1249`; reject truoc probe vi grouping coarse khong sua duoc FP class1.
- [x] 2026-07-02 cleanup semantic-attribute reject: xoa smoke train dir, giu XAI va keeper/final/AIDT artifacts; manifest `runs\cleanup_manifest_20260702_semattr020_reject_smoke.json`, reclaimed `180.09 MB`, D: free ~`69.55 GB`.
- [ ] Khong lap lai semantic-attribute exact setting `weight=0.02/specs maturity+transport+quality` tren current anchor. Neu quay lai hierarchy, can co consistency/gate moi theo case-level reliability, khong chi CE tren group logsumexp.
- [x] 2026-07-02 tra cuu GrabCut/object-centric classification va smoke foreground crop `grabcut` tren `yolo_f` current anchor voi p=`1.0`, margin `0.04`, mask area `0.03-0.92`, max crop area `0.85`. Full-val macro/class1 chi `0.8792/0.6667`, P/R class1 `0.5742/0.7947`, confusion `0->1=56`, `4->1=16`.
- [x] 2026-07-02 XAI GrabCut crop selected12: foreground mass van kha cao (`attention/gradcam fg 0.9574/0.9146`) nhung Grad-CAM border `8/12`, background blur/gray gan nhu vo hai hoac am (`-0.0012/-0.0053`), object_desaturate drop `0.1219`; reject truoc probe vi object-crop chat hon khong giai quyet surface/boundary class 1.
- [x] 2026-07-02 cleanup GrabCut crop reject: xoa smoke train dir, giu XAI selected12 va keeper/final/AIDT artifacts; manifest `runs\cleanup_manifest_20260702_grabcutcrop_m04_reject_smoke.json`, reclaimed `180.06 MB`, D: free ~`69.48 GB`.
- [ ] Khong lap lai foreground crop `grabcut p=1.0/margin=0.04/max_crop=0.85` tren current anchor. Neu tiep tuc object-centric augmentation, can co signal local/texture moi hoac policy co bang chung smoke giam `0->1`/`4->1`, khong chi cat nen manh hon.
- [x] 2026-07-02 tra cuu AugMix/color constancy/illumination robustness va trien khai tensor-level illumination consistency loss (brightness/contrast/gamma, khong doi hue/saturation). Compile + focused pytest pass (`10 passed`).
- [x] 2026-07-02 smoke illumination consistency `weight=0.02/prob=0.50/b=c=g 0.08/0.08/0.12` tren current anchor: loss active (`0.0043`, fraction `0.5143`) nhung val macro/class1 chi `0.8796/0.6685`, P/R class1 `0.5735/0.8013`, `0->1=57`, `4->1=16`.
- [x] 2026-07-02 XAI illumination selected12: foreground sach (`attention/gradcam fg 0.9799/0.9733`), background blur/gray gan `0` (`0.0008/-0.0006`), object_desaturate van dominant `0.1256`; reject truoc probe vi consistency chieu sang khong tao cue surface/boundary moi.
- [x] 2026-07-02 cleanup illumination consistency reject: xoa smoke train dir, giu XAI selected12 va keeper/final/AIDT artifacts; manifest `runs\cleanup_manifest_20260702_illumcons020_p50_reject_smoke.json`, reclaimed `180.19 MB`, D: free ~`69.39 GB`.
- [ ] Khong lap lai illumination consistency exact setting `weight=0.02/prob=0.50/brightness=0.08/contrast=0.08/gamma=0.12` tren current anchor. Neu tiep tuc robustness/consistency, can khac ve signal case-level/part-level, khong them KL global tren photometric jitter nhe.
- [x] 2026-07-02 diagnostic HSV/Lab/texture hist tren `yolo_f` object crops: standalone ExtraTrees tot nhat val macro/class1 `0.8423/0.5498`, thap xa TRKH; Logistic/RF class1 chi `0.439-0.529`.
- [x] 2026-07-02 OOF color-texture guard theo `StratifiedGroupKFold` train-only: ExtraTrees threshold `0.05` doi 2 val mau dat `0.8841/0.6822`, RandomForest doi 3 val mau dat `0.8847/0.6842`; van duoi keeper class1 `0.6860`, khong du de trien khai branch/router.
- [ ] Khong mo rong hand-crafted HSV/Lab/texture router/branch tu artifact `runs\hsv_lab_texture_diagnostic_20260702` neu khong co feature set moi cho OOF class1 > keeper ro rang; signal hien tai chi sua vai FP1 va khong giup recall.
- [x] 2026-07-02 cleanup legacy large `mango_cls_*` runs: da preserve current keeper/final/AIDT/ensemble/disagreement artifacts, tao manifest `runs\cleanup_manifest_20260702_legacy_mango_cls_large_runs.json`, xoa `19` thu muc legacy khong con la current gate, reclaimed `6721.93 MB`, D: free ~`75.97 GB`; sau cleanup chi con current keeper tren `200 MB`.
- [x] 2026-07-02 tra cuu Cutout/Random Erasing/Hide-and-Seek/background-bias masking va smoke bbox object-erasure negative `weight=0.006/prob=0.50/fill=blur/margin=0.04` tren current anchor: loss active (`0.0472`, fraction `0.4909`, erased max prob `0.3172`) nhung val macro/class1 chi `0.8783/0.6667`, P/R class1 `0.5708/0.8013`, `0->1=58`, `4->1=16`; XAI selected12 foreground van kha (`attention/gradcam fg 0.9223/0.8863`), background blur/gray gan `0`, object_desaturate `0.0709`. Reject truoc probe.
- [x] 2026-07-02 cleanup object-erasure reject: xoa smoke train dir, giu XAI selected12 va keeper/final artifacts; manifest `runs\cleanup_manifest_20260702_objerase006_p50_blur_reject_smoke.json`, reclaimed `180.16 MB`, D: free ~`75.90 GB`.
- [ ] Khong lap lai bbox object-erasure negative `weight=0.006/prob=0.50/fill=blur/margin=0.04` tren current anchor; no lam class-1 FP rong hon, nen cac huong tiep theo can signal local boundary/surface moi hoac OOF/disagreement hop le, khong chi them object/background erasing.
- [x] 2026-07-02 sua quality-group loader de ho tro `sample_index` cho `yolo_f`: `_load_quality_group_manifest` tra ve path + sample-index maps, `QualityGroupDataset` uu tien sample_index, test regression duplicate path/two object pass; compile + focused pytest pass (`7 passed`).
- [x] 2026-07-02 tao train-only manifest `runs\quality_group_yolof_anchor_classcart_train_20260702\quality_groups_train_sampleindex.csv`: `9215/9215` sample, `23` group class+cartography, loader smoke xac nhan `sample_indices=9215`, `matched=9215`, `paths=0`, khong collapse multi-object.
- [x] 2026-07-02 tra cuu GroupDRO/Dataset Cartography va smoke GroupDRO `weight=0.015/temp=0.30/min_samples=1` tren manifest class-cartography: loss active (`5.5635`, ~`8.125` groups/batch) nhung val macro/class1 chi `0.8787/0.6648`, P/R class1 `0.5714/0.7947`, confusion `0->1=57`, `4->1=16`.
- [x] 2026-07-02 XAI GroupDRO selected12: foreground van kha (`attention/gradcam fg 0.9222/0.8836`), background blur/gray gan `0` (`0.00079/0.00071`), object_desaturate `0.0718`; reject truoc probe vi group-DRO tren train-pred cartography chi amplify noisy boundary groups.
- [x] 2026-07-02 cleanup GroupDRO reject: xoa smoke train dir, giu train-only quality manifest + XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260702_groupdro015_classcart_reject_smoke.json`, reclaimed `180.20 MB`, D: free ~`75.82 GB`.
- [ ] Khong lap lai GroupDRO class-cartography `weight=0.015/temp=0.30/min_samples=1` tren current anchor. Neu dung quality group lai, phai co OOF/dynamics that hoac label-free quality group co bang chung giam `0->1/4->1`; khong dung train-pred in-sample cartography lam main signal.
- [x] 2026-07-02 tra cuu SnapMix/CutMix/SaliencyMix va trien khai foreground SnapMix auxiliary train-time theo bbox, khong sua raw dataset. Smoke `ForegroundSnapmixLossWeight=0.015`, probability `0.50`, pairs `0-1,1-2,1-4,2-3`, bbox area `0.06-0.18` active (~`4.3-5.4` mau mix/batch, source weight ~`0.116`) nhung best val macro/class1 chi `0.8844/0.6838`, P/R class1 `0.6000/0.7947`, confusion `0->1=52`, `4->1=15`, `1->0=14`, `1->2=11`.
- [x] 2026-07-02 XAI foreground SnapMix selected12: foreground mass cao (`attention/gradcam fg 0.9849/0.9737`) nhung background blur/gray van gan `0` (`0.0005/-0.0001`), object_desaturate drop `0.1086`, object_color_sensitive `6/12`, Grad-CAM border `6/12`; reject truoc probe.
- [x] 2026-07-02 cleanup foreground SnapMix reject: xoa smoke train dir, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260702_fgsnapmix015_p50_reject_smoke.json`, reclaimed `182.15 MB`.
- [ ] Khong lap lai foreground SnapMix exact setting `weight=0.015/prob=0.50/pairs=0-1,1-2,1-4,2-3/area=0.06-0.18` tren current anchor. Neu quay lai patch mixing, can co reliability/OOF/saliency signal manh hon va smoke phai giam ro `0->1`/`4->1` ma khong tang object-color sensitivity.
- [x] 2026-07-02 smoke CE classification-loss ablation `ClassificationLoss=cross_entropy` tren current keeper voi teacher-focus-binary `0.015`, bbox spatial fusion, 24 batches/1e: val macro/class1 chi `0.8778/0.6611`, P/R class1 `0.5694/0.7881`, confusion `0->1=56`, `4->1=17`, `1->0=14`, `1->2=12`.
- [x] 2026-07-02 XAI CE selected12: foreground mass cao (`attention/gradcam fg 0.9794/0.9728`) nhung background blur/gray gan `0` (`0.0009/-0.0004`), object_desaturate drop `0.1285`, object_color_sensitive `7/12`, Grad-CAM border `6/12`; reject truoc probe vi CE lam class-1 precision xau hon.
- [x] 2026-07-02 cleanup CE reject: xoa smoke train dir, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260702_ce_loss_reject_smoke.json`, reclaimed `180.17 MB`, D: free ~`75.66 GB`.
- [ ] Khong lap lai CE loss ablation tren current anchor; no bo bot shaping huu ich va lam vung class1 FP rong hon. Neu sua classification loss tiep, can signal moi co bang chung giam FP class1, khong chi doi ve CE plain.
- [x] 2026-07-02 tra cuu Quantized Label Learning/CPU va LNL-FG noisy fine-grained labels, trien khai auxiliary quantized-label CPU/PU risk khong sua raw data: target class la positive, cac class boundary chon `0,1,2,4` la unlabeled thay vi hard negative. Compile + focused pytest pass (`8 passed`), launcher dry-run resolve dung flags.
- [x] 2026-07-02 smoke QL-CPU `weight=0.01/classes=0,1,2,4/prior=0.02-0.60/negative_weight=0.7` tren current keeper: loss active (`0.3140`, prior/positive fraction `0.1999`) nhung val macro/class1 chi `0.8818/0.6761`, P/R class1 `0.5882/0.7947`, confusion `0->1=52`, `4->1=16`, `1->0=14`, `1->2=11`, duoi keeper `0.8847/0.6860`.
- [x] 2026-07-02 XAI QL-CPU selected12: top confusion audit van `3->2=57`, `0->1=55`, `4->1=16`; foreground mass cao (`attention/gradcam fg 0.9793/0.9721`), background blur/gray gan `0` (`0.0006/-0.0006`), object_desaturate dominant `0.1249`, object_color_sensitive `7/12`. Reject truoc probe.
- [x] 2026-07-02 cleanup QL-CPU reject: xoa smoke train dir, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260702_qlcpu010_nw07_reject_smoke.json`, reclaimed `180.29 MB`, D: free ~`75.60 GB`.
- [ ] Khong lap lai QL-CPU exact setting `weight=0.01/classes=0,1,2,4/prior=0.02-0.60/negative_weight=0.7/start=1` tren current anchor; no chi lam mem hard-negative pressure ma khong tao cue surface/boundary moi.
- [x] 2026-07-02 tra cuu FFVT/TransFG va trien khai FFVT-lite intermediate layer-token fusion: chon top-k patch tokens tu layer transformer `2,4,6` bang CLS attention + bbox/foreground prior, blend vao classification head input, khong sua raw data va khong them tham so hoc moi. Compile + focused pytest pass (`5 passed`), dry-run xac nhan `hard_sample_manifest=""` va `skip_final_test=true`.
- [x] 2026-07-02 smoke FFVT-lite `layers=2,4,6/topK=4/blend=0.12/bbox=0.20/foreground=0.10` tren current keeper: val macro/class1 chi `0.8798/0.6740`, P/R class1 `0.5748/0.8146`, confusion `0->1=60`, `4->1=17`, `1->0=11`, `1->2=11`; duoi keeper va khong dat gate probe.
- [x] 2026-07-02 XAI FFVT-lite selected12: attention/Grad-CAM loang hon (`fg 0.8785/0.8741`, bg `0.1215/0.1259`), attention border `0.3065`, flags `attention_border_attention=11/12`, background blur/gray gan `0` (`0.0005/-0.0003`), object_desaturate van dominant `0.1125`. Reject truoc probe.
- [x] 2026-07-02 cleanup FFVT-lite reject: xoa smoke train dir, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260702_layerfusion_ffvtlite_reject_smoke.json`, reclaimed `180.19 MB`, D: free ~`75.53 GB`.
- [ ] Khong lap lai FFVT-lite exact setting `layers=2,4,6/topK=4/blend=0.12/bbox=0.20/foreground=0.10` tren current anchor; no lam class-1 FP rong hon va XAI loang. Neu tiep tuc layer-token fusion, can co gating/reliability khac va smoke phai giam `0->1/4->1` ro rang.
- [x] 2026-07-02 tra cuu color jitter/color constancy va mango color grading, trien khai foreground chroma consistency bbox-masked khong sua raw data. Compile + focused pytest pass (`4 passed`), dry-run launcher xac nhan flags va `hard_sample_manifest=""`.
- [x] 2026-07-02 smoke foreground chroma consistency `weight=0.012/prob=0.50/sat=0.12/hue=0.012/bbox_margin=0.02` tren current keeper: loss active (`0.00327`, fraction `0.4987`, mask fraction `0.6843`) nhung val macro/class1 chi `0.8786/0.6648`, P/R class1 `0.5749/0.7881`, confusion `0->1=56`, `4->1=16`.
- [x] 2026-07-02 XAI foreground chroma selected12: heatmap khong sach hon (`attention/gradcam fg 0.8785/0.8751`, bg `0.1215/0.1249`), background blur/gray van gan `0` (`0.0004/-0.0006`), object_desaturate van cao `0.1250`; reject truoc probe.
- [x] 2026-07-02 cleanup foreground chroma reject: xoa smoke train dir, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260702_fgchroma012_p50_reject_smoke.json`, reclaimed `180.36 MB`, D: free ~`75.46 GB`.
- [ ] Khong lap lai foreground chroma consistency exact setting `weight=0.012/prob=0.50/sat=0.12/hue=0.012/bbox_margin=0.02/classes=0,1,2,3,4` tren current anchor; no lam class-1 FP rong hon. Neu tiep tuc color/chroma thi can reliability/OOF/case-level gating, khong ep consistency toan foreground theo hard label hien tai.
- [x] 2026-07-02 diagnostic probability-router sample-index val: no-pretrain ensemble focus005 dat `0.8864/0.6939`, nhung grouped-CV router tot nhat tu base+ensemble probs chi `0.8845/0.6880` va doi `9` mau; khong du gate va khong nen trien khai router val-tuned moi.
- [x] 2026-07-02 diagnostic teacher-consensus train manifest: yeu cau AIDT + no-pretrain ensemble cung correct tren base train mistakes trong scope class1 rescue/suppress cho `0` rows, nen khong co targeted-margin manifest hep hon khac that su so voi AIDT rescue margin da fail.
- [ ] Khong tiep tuc probability-router/class1 margin guard/targeted-margin tu current base+ensemble+AIDT probabilities neu khong co representation moi hoac true OOF signal moi; val-only router/guard da lap lai mot mau that bai.
- [x] 2026-07-02 tra cuu MAE/SimMIM va smoke train-time masked reconstruction bbox/detail-weighted tren current keeper: preflight + focused pytest pass (`16 passed`), MIM loss active (`1.1148`, mask fraction `0.2461`, bbox prior mean `0.6607`) nhung val macro/class1 chi `0.8831/0.6784`, P/R class1 `0.6073/0.7682`; confusion `0->1=49`, `1->0=19`, `4->1=12`.
- [x] 2026-07-02 XAI MIM selected12: top confusion `3->2=54`, `0->1=49`, `1->0=19`; heatmap foreground kha tot (`attention/gradcam fg 0.9044/0.9557`) nhung border flags cao (`attention_border=11/12`, `gradcam_border=7/12`), background blur/gray gan `0`, object_desaturate `0.0731`; reject truoc probe.
- [x] 2026-07-02 cleanup MIM reject: xoa smoke train dir, giu XAI selected12 va keeper/final artifacts; manifest `runs\cleanup_manifest_20260702_mim010_mask35_reject_smoke.json`, reclaimed `185.21 MB`, D: free ~`75.38 GB`.
- [ ] Khong lap lai train-time masked reconstruction exact setting `weight=0.01/mask_ratio=0.35/foreground=0.85/detail=0.35/bbox=0.60/bbox_margin=0.02` tren current anchor. MIM nhe khong sua `0->1` va lam `1->0` tang; neu quay lai MIM/SSL can objective local/teacher/OOF moi hon, khong chi pixel reconstruction phu.
- [x] 2026-07-02 xem contact sheet/XAI boundary cases: loi chinh la near-boundary tren object surface, `4->1` co vet nhan/tham cuc bo nhung Grad-CAM bam vung sang/vien, `3->2` co dom den lon nhung Grad-CAM co case tranh dom va bam nen/vien; nen khong quay lai context rong/background-only.
- [x] 2026-07-02 sua `trkh.tools.probe_foreground_surface_stats` de chay dung object-level `yolo_f` bang `MangoYOLOCropDataset`, giu `sample_index/source_stem/object_index/label_path/bbox` trong CSV; compile pass va pytest `tests\test_probe_foreground_surface_stats.py` pass (`1 passed`).
- [x] 2026-07-02 diagnostic `runs\yolof_foreground_surface_stats_diagnostic_20260702`: foreground-surface stats ExtraTrees val macro/class1 chi `0.8384/0.5214`, threshold class1 tot nhat `0.6379`; yeu hon TRKH anchor.
- [x] 2026-07-02 surface-stat guard val-only co the tang class1 len `0.6930` bang cach doi `16` mau base-predicted class1, fix `13` FP1 va harm `3` true class1, nhung van duoi gate `0.70` va la threshold val-only.
- [x] 2026-07-02 surface-stat guard fold-safe bi reject: train OOF surface class1 chi `0.4367`, threshold train-OOF `0.02` ap len val chi doi `1` mau, val macro/class1 `0.8832/0.6802`, thap hon keeper `0.8847/0.6860`.
- [ ] Khong trien khai surface-stat guard/head/router tu deterministic HSV/Lab/dark-brown spot stats hien tai; neu tiep tuc defect/spot thi can supervision moi khac val-threshold va khac summary stat OOF yeu.
- [x] 2026-07-02 tra cuu TransFG/WS-DAN/NTS-Net va trien khai `LocalZoomImageExpert` score mode `defect_spot`, exposed qua train CLI/v8 launcher; preflight pass (`py_compile` + focused pytest `5 passed`) va dry-run xac nhan `hard_sample_manifest=""`, `skip_final_test=true`, keeper resume, `LocalZoomScoreMode=defect_spot`.
- [x] 2026-07-02 smoke local zoom defect-spot `logit_scale=0.14/aux=0.01/routes=0-1,1-2,2-3,1-4,4-rest` tren current keeper: val macro/class1 `0.8831/0.6784`, P/R class1 `0.6073/0.7682`, confusion `0->1=49`, `1->0=19`, `4->1=12`; duoi keeper `0.8847/0.6860`, khong probe.
- [x] 2026-07-02 architecture trace local zoom defect-spot: class-4 sample co dom hong nhung crop chon vung top-left/vien sach, score map bi border/background edge chi phoi; class0/1/3 trace cung co border/lower-edge attraction.
- [x] 2026-07-02 XAI local zoom defect-spot selected12: foreground van cao (`attention/gradcam fg 0.8982/0.9461`) nhung border flags rat cao (`attention_border=12/12`, `gradcam_border=7/12`), background blur/gray gan `0`, object_desaturate `0.0882`; reject truoc probe.
- [x] 2026-07-02 cleanup local zoom defect-spot reject: xoa smoke train dir, giu XAI selected12 + keeper/final/diagnostic artifacts; manifest `runs\cleanup_manifest_20260702_localzoom_defectspot_reject_smoke.json`, reclaimed `186.32 MB`, D: free ~`75.30 GB`.
- [ ] Khong lap lai local zoom `defect_spot` exact setting `score_mode=defect_spot/logit_scale=0.14/aux=0.01/routes=0-1,1-2,2-3,1-4,4-rest` tren current anchor. Neu tiep tuc local-part/defect zoom, phai fix score map de loai border bias bang supervision/OOF/case-level signal moi, khong chi dark-brown heuristic.
- [x] 2026-07-02 trien khai local zoom `defect_spot_interior` voi soft-erode foreground support + frame-margin suppression de sua loi crop bam vien; preflight pass (`py_compile` + focused pytest `5 passed`), dry-run xac nhan `hard_sample_manifest=""`, `distillation_weight=0`, `skip_final_test=true`.
- [x] 2026-07-02 smoke local zoom `defect_spot_interior` tren current keeper: val macro/class1 `0.8822/0.6764`, P/R class1 `0.6042/0.7682`, confusion `0->1=50`, `1->0=19`, `4->1=12`; duoi keeper va khong dat probe gate.
- [x] 2026-07-02 trace/XAI local zoom `defect_spot_interior`: synthetic border-band test pass nhung real class4 trace van chon crop top-left cham `x=0`, bo qua cum dom toi tren than qua; XAI selected12 van `attention_border=12/12`, `gradcam_border=7/12`, object_desaturate `0.1061`, background blur/gray gan `0`.
- [x] 2026-07-02 cleanup local zoom `defect_spot_interior` reject: xoa smoke train dir, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260702_localzoom_defectspotinterior_reject_smoke.json`, reclaimed `186.31 MB`, D: free ~`75.23 GB`.
- [ ] Khong lap lai local zoom `defect_spot_interior` exact setting `score_mode=defect_spot_interior/logit_scale=0.14/aux=0.01/routes=0-1,1-2,2-3,1-4,4-rest`. Neu tiep tuc part/defect zoom, can signal reliability/supervision manh hon hand-crafted dark-brown/interior heuristic.
- [x] 2026-07-02 smoke oracle-teacher KD `DistillationWeight=0.03/temp=2.0` tu train-only expert oracle `runs\oracle_teacher_base_color_fgbg_train_20260701\teacher_probs_train.csv`: sample-index overlap `1.0`, KD active (`train_distillation_loss=0.1085`) nhung val macro/class1 chi `0.8822/0.6764`, P/R class1 `0.6042/0.7682`, confusion `0->1=50`, `1->0=19`, `4->1=12`.
- [x] 2026-07-02 XAI oracle-teacher KD selected12: top confusions `3->2=52`, `0->1=50`, `1->0=19`, `4->1=12`; foreground mass van cao (`attention/gradcam fg 0.8982/0.9461`) nhung border flags cao (`attention_border=12/12`, `gradcam_border=7/12`), background blur/gray gan `0`, object_desaturate `0.0881`. Reject truoc probe vi oracle train-in-sample khong them cue surface/boundary va lam class1 FP/FN xau hon keeper.
- [x] 2026-07-02 cleanup oracle-teacher KD reject: xoa smoke train dir, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260702_oracleteacherkd003_reject_smoke.json`, reclaimed `180.30 MB`, D: free ~`75.15 GB`.
- [ ] Khong lap lai oracle-teacher KD exact setting `DistillationWeight=0.03/temp=2.0/teacher=oracle_teacher_base_color_fgbg_train_20260701` tren current anchor. Neu dung oracle/ensemble teacher lai, phai fold-safe/OOF hoac disagreement-confident, va smoke phai giam ro `0->1/4->1` ma khong tang `1->0`.
- [x] 2026-07-02 tra cuu Generalized Cross Entropy/robust loss va smoke `ClassificationLoss=ldam_gce`, `GceQ=0.7` tren current keeper voi teacher-focus-binary `0.015`, bbox spatial fusion, pairwise routing, hard-repeat tat. Preflight pass (`py_compile` + focused pytest `3 passed`), dry-run dung flags.
- [x] 2026-07-02 smoke `ldam_gce q=0.7`: val macro/class1 chi `0.8816/0.6744`, P/R class1 `0.6010/0.7682`, confusion `0->1=51`, `1->0=19`, `4->1=12`; duoi keeper va khong dat probe gate.
- [x] 2026-07-02 XAI `ldam_gce q=0.7` selected12: top confusions `3->2=52`, `0->1=51`, `1->0=18`, `4->1=12`; foreground mass van cao (`attention/gradcam fg 0.9044/0.9559`) nhung border flags cao (`attention_border=11/12`, `gradcam_border=7/12`), background blur/gray gan `0`, object_desaturate `0.0739`. Reject truoc probe.
- [x] 2026-07-02 cleanup `ldam_gce q=0.7` reject: xoa smoke train dir, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260702_ldamgce_q070_reject_smoke.json`, reclaimed `180.25 MB`, D: free ~`75.08 GB`.
- [ ] Khong lap lai robust-loss swap `ClassificationLoss=ldam_gce/GceQ=0.7` tren current anchor; no giam scale loss nhung khong sua surface/boundary va lam `0->1` xau hon. Neu quay lai robust loss, can sample-selection/OOF reliability moi, khong chi doi criterion.
- [x] 2026-07-02 tra cuu DSN/FFVT/Contrastive Deep Supervision va smoke multi-granularity auxiliary CE heads tren current keeper: preflight + focused pytest pass (`5 passed`), dry-run dung `yolo_f` primary sau khi sample-index guard bat loi launcher nham `class_f`.
- [x] 2026-07-02 smoke multi-granularity aux CE `layers=2,5,8/loss_weight=0.02/dropout=0.08`: aux loss active (`8.8385`), trace co logits `[1,5]` tai 3 layer, nhung val macro/class1 chi `0.8830/0.6822`, P/R class1 `0.6094/0.7748`, confusion `0->1=49`, `1->0=18`, `4->1=12`; duoi keeper va khong dat probe gate.
- [x] 2026-07-02 XAI multi-granularity aux selected12: top confusions `3->2=53`, `0->1=48`, `1->0=17`, `4->1=12`; heatmap foreground rat cao (`attention/gradcam fg 0.9830/0.9757`) nhung Grad-CAM border `8/12`, background blur/gray gan `0`, object_desaturate `0.0883`. Reject truoc probe.
- [x] 2026-07-02 cleanup multi-granularity aux CE reject: xoa smoke train dir, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260702_multigranaux020_reject_smoke.json`, reclaimed `180.43 MB`, D: free ~`75.01 GB`.
- [ ] Khong lap lai multi-granularity auxiliary CE exact setting `heads=true/layers=2,5,8/loss_weight=0.02/dropout=0.08` tren current anchor; CE layer trung gian khong tao cue moi. Neu quay lai deep supervision, thu representation-level/contrastive aux co reliability/confusion-pair gating thay vi CE logits phu.
- [x] 2026-07-02 trien khai multi-granularity contrastive deep supervision tren pooled layer features `2,5,8`, co teacher agreement/confidence-margin weighting va confusion pairs `0-1,1-2,1-4,2-3`; preflight pass (`py_compile` + focused pytest `7 passed`), dry-run dung `yolo_f`, keeper resume, `hard_sample_manifest=""`.
- [x] 2026-07-02 smoke multi-granularity contrastive `weight=0.006/temp=0.18/teacher_confidence_margin/power=0.5`: contrastive loss active (`1.5623`, fraction `0.9740`, selected count `15.58`, positive pairs `34.17`) nhung val macro/class1 chi `0.8843/0.6860`, P/R class1 `0.6114/0.7815`, confusion `0->1=49`, `1->0=17`, `4->1=12`, `3->2=53`; chi tie class1 keeper va macro thap hon nhe.
- [x] 2026-07-02 XAI multi-granularity contrastive selected12: top confusions `3->2=53`, `0->1=49`, `1->0=18`, `4->1=12`; foreground mass cao (`attention/gradcam fg 0.9830/0.9757`), background blur/gray gan `0`, object_desaturate `0.0883`, Grad-CAM border `8/12`; reject truoc probe vi khong tao cue surface/boundary moi.
- [x] Cleanup smoke multi-granularity contrastive reject: xoa smoke train dir, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260702_multigrancontrast006_reject_smoke.json`, reclaimed `180.69 MB`, D: free ~`74.94 GB`.
- [ ] Khong lap lai multi-granularity contrastive exact setting `heads=true/layers=2,5,8/contrastive_weight=0.006/temp=0.18/pairs=0-1,1-2,1-4,2-3/teacher=focus005/weight_mode=confidence_margin/power=0.5` tren current anchor; signal active nhung trung lap voi keeper va XAI khong doi. Huong tiep theo can signal khac ve target/representation, khong them weight cho cung objective.
- [x] 2026-07-02 tra cuu TTA aggregation va eval current keeper voi color TTA `--tta --tta-brightness-delta 0.08`: macro/class1 roi xuong `0.8763/0.6590`, P/R class1 `0.5808/0.7616`, confusion `0->1=52`, `1->0=18`, `1->2=13`, `2->1=13`, `4->1=14`; per-sample chi `10` corrections nhung `23` harms tren `37` changes. Reject.
- [x] Cleanup eval color TTA reject: xoa `runs\eval_yolof_teacherfocusbinary015_best_val_tta_b08_20260702`; manifest `runs\cleanup_manifest_20260702_color_tta_b08_reject_eval.json`, reclaimed `2.85 MB`, D: free ~`74.94 GB`.
- [ ] Khong bat color TTA/logit-average exact setting `tta_brightness_delta=0.08` cho current keeper; object color/surface la cue that, khong phai invariant an toan.
- [x] 2026-07-02 tra cuu TENT/SAR va trien khai tool diagnostic `trkh.tools.evaluate_test_time_adaptation`: entropy adaptation khong nhan, chi update LayerNorm affine + `head.bias`, co reliable selection theo confidence/entropy; compile + pytest pass (`2 passed`), smoke 4 batch end-to-end.
- [x] 2026-07-02 full-val reliable entropy TTA-adapt `layernorm_head_bias/lr=1e-5/selection_fraction=0.50/min_confidence=0.30`: selected `919/2606`, `41` optimizer steps, nhung val macro/class1 chi `0.8826/0.6764`, P/R class1 `0.6042/0.7682`, confusion `0->1=49`, `1->0=19`, `4->1=13`; per-sample doi `7` mau, correct `3`, harm `4`. Reject.
- [x] Cleanup TTA-adapt smoke/full reject: xoa `runs\ttaadapt_smoke_lnheadbias_lr1e5_frac50_conf30_4b_20260702` va `runs\ttaadapt_val_lnheadbias_lr1e5_frac50_conf30_20260702`; manifest `runs\cleanup_manifest_20260702_ttaadapt_entropy_reject.json`, reclaimed `3.67 MB`, D: free ~`74.94 GB`.
- [ ] Khong lap lai reliable entropy TTA-adapt exact setting `layernorm_head_bias/lr=1e-5/selection_fraction=0.50/min_confidence=0.30/one_step` tren current keeper; plain entropy pressure khong tao cue boundary moi va lam mat recall class1.
- [x] 2026-07-02 cleanup leftover legacy `mango_cls_*`, `bench_cls*`, va old groupclean XAI dirs con sot lai: xoa `23` thu muc khong phai current gate; manifest `runs\cleanup_manifest_20260702_legacy_mango_cls_bench_leftovers.json`, reclaimed `3441.33 MB`, D: free ~`78.30 GB`.
- [x] 2026-07-03 trien khai diagnostic `trkh.tools.audit_yolo_source_groups` va test de do same-source/multi-object structure cua `yolo_f` ma khong sua raw data. Preflight pass (`py_compile` + pytest `1 passed`), artifact `runs\yolof_source_group_consistency_audit_20260703`.
- [x] 2026-07-03 audit `yolo_f`: train co `691` multi-object source images nhung `144` mixed-label; class1 chi co `115` train multi-object images, trong do `13` same-label class1 va `102` mixed-label. Val chi co `29` multi-object, toan same-label class `0/3`; test hoan toan single-object. Ket luan global same-image consistency nguy hiem va khong co val/test class1 peer coverage.
- [ ] Khong trien khai global same-image/source-context consistency loss. Neu buoc phai thu source-peer, chi duoc same-label-only, weight rat nhe, xem nhu diagnostic phu va phai smoke giam `0->1/4->1` ma khong tang `1->0`; khong dung lam huong chinh.
- [x] 2026-07-03 expose `-Seed` trong launcher V8 de co the chay seed/schedule diagnostic; default van `42`. Seed-replay dung base cu bi chan vi checkpoint resume lich su da duoc cleanup, nen khong khoi phuc run rac.
- [x] 2026-07-03 smoke low-LR continuation tu keeper hien tai `LR=1e-5/Seed=7/24b/1e`: val macro/class1 chi `0.8825/0.6764`, P/R class1 `0.6042/0.7682`, confusion `0->1=50`, `1->0=19`, `4->1=12`. XAI foreground sach, background gan `0`, object_desaturate `0.0925`; reject truoc probe.
- [x] Cleanup low-LR continuation reject: xoa `runs\smoke_v8_yolof_keeper_lowlr1e5_seed7_24b_1e_20260703`, giu XAI `runs\xai_smoke_v8_yolof_keeper_lowlr1e5_seed7_val_selected12_20260703`; manifest `runs\cleanup_manifest_20260703_keeper_lowlr1e5_reject_smoke.json`, reclaimed `180.68 MB`.
- [ ] Khong lap lai low-LR keeper continuation exact setting `LR=1e-5/Seed=7/24b/1e` tren current keeper; no chi lam class1 recall xau hon va khong doi failure mode. Neu quay lai schedule/seed, can base checkpoint hop le hoac signal moi, khong chi fine-tune tiep keeper.
- [x] 2026-07-03 tra cuu FINE/spectral noisy-label filtering, LNL-FG, va PASS; trien khai diagnostic `trkh.tools.probe_embedding_spectral_noise` de do ambiguity tren embedding frozen cua keeper ma khong sua raw data. Preflight pass (`py_compile` + focused pytest, tong `4 passed` voi targeted-margin sample-index test).
- [x] 2026-07-03 full train-split spectral audit `runs\spectral_noise_yolof_keeper_train_20260703`: `9215` object rows, dim `256`, class1 mean same-label neighbor fraction chi `0.7012` (classes 0/2/3/4 lan luot `0.9211/0.9409/0.9731/0.9774`); top-risk class1 same-label neighbor chi `0.2525`, rival chu yeu `0/2`, xac nhan embedding hien tai chong lan class1 voi 0/2.
- [x] 2026-07-03 build spectral targeted-margin manifest sample-index `runs\spectral_margin_yolof_keeper_train_20260703\spectral_targeted_margin_train.csv`: `285` rows, `233` false-positive-risk va `52` false-negative-risk; pairs `0->1=152`, `2->1=66`, `4->1=8`, `3->1=7`, `1->0=33`, `1->2=19`.
- [x] 2026-07-03 smoke spectral targeted-margin `loss_weight=0.012/margin=0.08/max_weight=1.15` tren keeper voi teacher-focus-binary `0.015`, bbox spatial fusion, pairwise routing, hard-repeat off: loss active (`0.0989`, fraction `0.0417`) nhung val macro/class1 chi `0.8822/0.6764`, P/R class1 `0.6042/0.7682`, confusion `0->1=50`, `1->0=19`, `4->1=12`; reject truoc probe.
- [x] 2026-07-03 XAI spectral targeted-margin selected12: foreground van cao (`attention/gradcam fg 0.9828/0.9760`), background blur/gray gan `0` (`0.0001/-0.0005`), object_desaturate `0.0879`, Grad-CAM border `7/12`; failure mode khong doi.
- [x] Cleanup spectral-margin reject: xoa smoke train dir va diagnostic smoke128 tam, giu full spectral audit + manifest + XAI selected12; manifest `runs\cleanup_manifest_20260703_spectralmargin_reject_smoke.json`, reclaimed `180.60 MB`, D: free ~`78.17 GB`.
- [ ] Khong lap lai representation-derived targeted margin exact setting `loss_weight=0.012/margin=0.08/max_weight=1.15/min_ambiguity_nonfocus=0.78/min_ambiguity_focus=0.74` tren current keeper. Spectral/FINE signal chi dung lam audit hoac phai ket hop voi representation/OOF signal moi; hard-label margin tren cung embedding lam class1 recall xau hon.
- [x] 2026-07-03 tra cuu Tversky/Focal-Tversky va F-beta surrogate, trien khai focus-class soft Tversky loss tuy chon trong `TrainConfig`/train CLI/V8 launcher. Preflight pass (`py_compile` + focused pytest `4 passed`).
- [x] 2026-07-03 smoke focus-class Tversky `weight=0.010/alpha=0.70/beta=0.30/gamma=1.0/prob_power=1.0/class=1/start=1` tren keeper: loss active (`0.6772`, index `0.3226`) nhung val macro/class1 chi `0.8831/0.6784`, P/R class1 `0.6073/0.7682`; confusion `0->1=49`, `1->0=19`, `4->1=12`.
- [x] 2026-07-03 XAI focus-class Tversky selected12: top confusions `3->2=52`, `0->1=49`, `1->0=19`, `4->1=12`; foreground rat cao (`attention/gradcam fg 0.9780/0.9793`), background blur/gray gan `0`, object_desaturate `0.0730`, Grad-CAM border `7/12`. Reject truoc probe.
- [x] Cleanup focus-class Tversky reject: xoa smoke train dir va 2 failed background-launch logs, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_focustversky010_reject_smoke.json`, reclaimed `180.58 MB`, D: free ~`78.10 GB`.
- [ ] Khong lap lai focus-class Tversky exact setting tren current keeper; no khong giam FP class1 va lam recall class1 xau hon. Chi xem lai khi da co representation/teacher/OOF signal moi thay doi manifold, khong phai chi doi alpha/beta/weight.
- [x] 2026-07-03 tra cuu Class-Independent Regularization va LNL-FG/SNSCL, trien khai class-independent OVA auxiliary head zero-init, checkpoint-safe, khong doi inference logits; preflight pass (`py_compile` + focused pytest `6 passed`).
- [x] 2026-07-03 smoke CIR-lite/OVA `class_independent_head=true/loss_weight=0.020/positive_weight=4.0/dropout=0.05` tren current keeper: loss active (`1.1003`) nhung binary head gan nhu chua tach duoc positive/negative (`0.5042/0.4990`, margin `0.0054`); val macro/class1 chi `0.8825/0.6764`, P/R class1 `0.6042/0.7682`, confusion `0->1=50`, `1->0=19`, `4->1=12`.
- [x] 2026-07-03 XAI OVA selected12: top confusions `3->2=52`, `0->1=50`, `1->0=19`, `4->1=12`; foreground van cao (`attention/gradcam fg 0.9780/0.9793`), background blur/gray gan `0`, object_desaturate `0.0732`, Grad-CAM border `7/12`. Reject truoc probe.
- [x] Cleanup OVA reject: xoa `runs\smoke_v8_yolof_classindova020_pos4_teacherfocusbinary015_24b_1e_20260703`, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_classindova020_reject_smoke.json`, reclaimed `180.81 MB`, D: free ~`78.03 GB`.
- [ ] Khong lap lai CIR-lite/OVA exact setting `class_independent_head=true/loss_weight=0.020/positive_weight=4.0/dropout=0.05` tren current keeper; neu quay lai OVA/CIR can co co-teaching/OOF/reliability moi, khong chi BCE one-vs-all tren cung embedding.
- [x] 2026-07-03 tra cuu DCL va part-centric FGVC/PARTICLE, trien khai DCL-lite bbox region shuffle auxiliary CE khong sua raw data; preflight pass (`py_compile` + focused pytest `10 passed`).
- [x] 2026-07-03 dry-run DCL-lite xac nhan `yolo_f`, keeper resume, teacher CSV sample-index, distillation off, teacher-focus-binary `0.015`, bbox spatial fusion, pairwise routing, va hard/sample/quality/targeted manifests rong.
- [x] 2026-07-03 smoke DCL-lite `loss_weight=0.010/prob=0.50/grid=4/bbox_margin=0.02/classes=all`: loss active (`1.6413`, fraction `0.4948`, bbox area `0.6812`) nhung val macro/class1 chi `0.8835/0.6784`, P/R class1 `0.6073/0.7682`, confusion `0->1=49`, `1->0=19`, `4->1=12`, `3->2=54`.
- [x] 2026-07-03 XAI DCL-lite selected12: Grad-CAM foreground giam `0.9025`, background tang `0.0975`, co `gradcam_background_attention=1`, object_desaturate tang `0.0931`, object_color_sensitive `4/12`; reject truoc probe.
- [x] Cleanup DCL-lite reject: xoa `runs\smoke_v8_yolof_dclshuffle010_p50_g4_teacherfocusbinary015_24b_1e_20260703`, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_dclshuffle010_reject_smoke.json`, reclaimed `180.75 MB`, D: free ~`77.97 GB`.
- [ ] Khong lap lai DCL-lite exact setting `loss_weight=0.010/probability=0.50/grid=4/bbox_margin=0.02/classes=all` tren current keeper; destructive local view lam Grad-CAM loang hon va khong giam FP class1.
- [x] 2026-07-03 tra cuu CRD/RKD/feature distillation va trien khai boundary AIDT feature CRD tuy chon: align TRKH feature voi AIDT same-object feature, in-batch boundary negatives, khong dung AIDT luc inference, khong sua raw data. Preflight pass (`py_compile` + focused pytest `7 passed`), dry-run dung `yolo_f`, keeper resume, teacher CSV sample-index, va AIDT feature NPZ clean.
- [x] 2026-07-03 smoke AIDT CRD `weight=0.006/temp=0.20/proj=128/source=head/pairs=0-1,1-2,1-4,2-3`: loss active (`2.3438`) nhung alignment yeu (`top1=0.2279`, positive_similarity `-0.0205`, negative_similarity `-0.0253`); val macro/class1 chi `0.8828/0.6764`, P/R class1 `0.6042/0.7682`, confusion `0->1=50`, `1->0=19`, `4->1=12`.
- [x] 2026-07-03 XAI AIDT CRD selected12: top confusions `3->2=51`, `0->1=50`, `1->0=19`, `4->1=12`; foreground mass cao (`attention/gradcam fg 0.9829/0.9761`), background blur/gray gan `0`, object_desaturate `0.0881`, Grad-CAM border `7/12` va 1 background flag. Reject truoc probe.
- [x] Cleanup AIDT CRD reject: xoa `runs\smoke_v8_yolof_aidtcrd006_boundary_teacherfocusbinary015_24b_1e_20260703`, giu XAI selected12 + keeper/final/AIDT artifacts; manifest `runs\cleanup_manifest_20260703_aidtcrd006_reject_smoke.json`, reclaimed `180.92 MB`, D: free ~`77.90 GB`.
- [ ] Khong lap lai fixed-projection boundary AIDT CRD exact setting tren current keeper; neu feature-distillation quay lai, can learnable adapter hoac disagreement/OOF reliability moi, khong chi doi weight/temp/projection cua cung CRD.
- [x] 2026-07-03 tra cuu Co-teaching/JoCoR/SuperLoss va fine-grained noisy-label learning, trien khai self-paced small-loss reweighting tuy chon cho classification loss; preflight pass (`py_compile`, `tests\test_self_paced_loss.py`, teacher-feature RKD regression, sample-weight `-k` regression).
- [x] 2026-07-03 smoke self-paced `weight=0.25/percentile=0.80/gamma=0.75/min_weight=0.50/start=1` tren current keeper: loss active (`threshold=0.3029`, mean_weight `0.9267`, selected_fraction `0.7813`, high_loss_weight `0.6651`) nhung val macro/class1 chi `0.8831/0.6784`, P/R class1 `0.6073/0.7682`; confusion `0->1=49`, `1->0=19`, `1->2=11`, `4->1=12`.
- [x] 2026-07-03 XAI self-paced selected12: top confusions `3->2=52`, `0->1=49`, `1->0=19`, `4->1=12`; foreground mass cao (`attention/gradcam fg 0.9828/0.9760`), background blur/gray gan `0`, object_desaturate `0.0879`, Grad-CAM border `7/12` va 1 background flag. Reject truoc probe.
- [x] Cleanup self-paced reject: xoa `runs\smoke_v8_yolof_selfpaced025_p80_g075_min50_teacherfocusbinary015_24b_1e_20260703`, giu XAI selected12 + keeper/final/AIDT artifacts; manifest `runs\cleanup_manifest_20260703_selfpaced025_reject_smoke.json`, reclaimed `181.06 MB`, D: free ~`77.84 GB`.
- [ ] Khong lap lai self-paced small-loss exact setting `weight=0.25/percentile=0.80/gamma=0.75/min_weight=0.50/start=1` tren current keeper; batch-level small-loss reweighting khong tao cue surface/boundary moi. Neu quay lai sample selection, can OOF/peer-disagreement/reliability signal that, khong chi batch quantile theo hard-label loss.
- [x] 2026-07-03 trien khai learnable projection adapter cho AIDT feature CRD: `TeacherFeatureProjectionAdapter` map TRKH 256-dim va AIDT 2816-dim sang shared 128-dim, checkpoint-safe resume, launcher flags, va regression test gradients. Preflight pass (`py_compile`, `tests\test_teacher_feature_rkd.py` = `8 passed`, resume-prefix tests = `6 passed`).
- [x] 2026-07-03 smoke learnable-adapter AIDT CRD `weight=0.004/temp=0.20/proj=128/dropout=0.02/source=head/pairs=0-1,1-2,1-4,2-3`: adapter active, CRD top1 tang len `0.5169` nhung val macro/class1 chi `0.8825/0.6764`, P/R class1 `0.6042/0.7682`, confusion `0->1=50`, `1->0=19`, `1->2=11`, `4->1=12`.
- [x] 2026-07-03 XAI learnable-adapter AIDT CRD selected12: foreground cao (`attention/gradcam fg 0.9827/0.9760`), background blur/gray gan `0` (`0.0002/-0.0006`), object_desaturate `0.0880`, Grad-CAM border `7/12` va 1 background flag; manual overlays van la surface/boundary ambiguity.
- [x] Cleanup learnable-adapter AIDT CRD reject: xoa `runs\smoke_v8_yolof_aidtcrd_adapter004_boundary_teacherfocusbinary015_24b_1e_20260703`, giu XAI selected12 + keeper/final/AIDT artifacts; manifest `runs\cleanup_manifest_20260703_aidtcrd_adapter004_reject_smoke.json`, reclaimed `190.26 MB`, D: free ~`77.77 GB`.
- [ ] Khong lap lai learnable-adapter AIDT CRD exact setting tren current keeper; projector hoc duoc cai thien telemetry nhung khong giam FP/FN class1. Feature transfer tiep theo phai co fold-safe disagreement/reliability hoac case-level routing moi, khong chi doi weight/temp/dropout/projection.
- [x] 2026-07-03 tra cuu Balanced Softmax/logit adjustment/Seesaw va smoke loss co san `ClassificationLoss=balanced_softmax`, `BalancedSoftmaxTau=0.25`: smoke macro/class1 `0.8850/0.6860`, P/R class1 `0.6114/0.7815`, confusion `0->1=49`, `1->0=17`, `4->1=12`; XAI smoke khong xau hon nen cho phep probe ngan.
- [x] 2026-07-03 probe Balanced Softmax tau025 `120b/2e`: reject, best epoch macro/class1 chi `0.8747/0.6558`, P/R class1 `0.5550/0.8013`, confusion `0->1=58`, `2->1=16`, `4->1=19`; epoch 2 con roi `0.8621/0.6272`.
- [x] 2026-07-03 XAI probe Balanced Softmax tau025: top confusions `0->1=58`, `3->2=54`, `4->1=19`, `2->1=16`; foreground van cao (`attention/gradcam fg 0.9781/0.9681`) nhung object_desaturate tang `0.1391`, object_color_sensitive `6/12`; loss mo rong vung class1 chu khong tao cue moi.
- [x] Cleanup Balanced Softmax tau025 reject: xoa smoke/probe train dirs, giu XAI smoke+probe + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_balsoftmax_tau025_reject_smoke_probe.json`, reclaimed `363.26 MB`, D: free ~`77.64 GB`.
- [ ] Khong lap lai Balanced Softmax `ClassificationLoss=balanced_softmax/BalancedSoftmaxTau=0.25` tren current keeper; khong chi tune tau quanh day neu khong co OOF/reliability constraint vi probe lam class1 FP bung rong.
- [x] 2026-07-03 trien khai Seesaw-style CE `ClassificationLoss=seesaw` voi mitigation/compensation power; focused tests pass (`4 passed`) va dry-run dung `p=0/q=2`, `yolo_f`, keeper resume, teacher-focus-binary `0.015`, no stale manifests.
- [x] 2026-07-03 smoke Seesaw compensation-only `p=0/q=2`: val macro/class1 `0.8836/0.6842`, P/R class1 `0.6126/0.7748`, confusion `0->1=48`, `1->0=18`, `1->2=11`, `4->1=12`; duoi keeper va khong probe.
- [x] 2026-07-03 XAI Seesaw p0q2 selected12: top confusions `3->2=52`, `0->1=48`, `1->0=18`, `4->1=12`; foreground cao (`attention/gradcam fg 0.9829/0.9764`), background gan `0`, object_desaturate `0.0901`, Grad-CAM border `7/12`; khong tao cue surface moi.
- [x] Cleanup Seesaw p0q2 reject: xoa smoke train dir, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_seesaw_p0_q2_reject_smoke.json`, reclaimed `180.97 MB`, D: free ~`77.57 GB`.
- [ ] Khong lap lai Seesaw compensation-only `ClassificationLoss=seesaw/p=0/q=2` tren current keeper; full mitigation cung khong phai next step neu khong co signal moi vi co nguy co giam negative pressure voi tail class1.
- [x] 2026-07-03 expose launcher flag `-DisableBalancedEpochSampling` va smoke no-balanced default LDAM/focal tren keeper: val macro/class1 `0.8831/0.6804`, P/R class1 `0.6105/0.7682`, confusion `0->1=48`, `1->0=19`, `1->2=11`, `4->1=12`; duoi keeper.
- [x] 2026-07-03 XAI no-balanced selected12: top confusions `3->2=52`, `0->1=48`, `1->0=19`, `4->1=12`; foreground cao (`attention/gradcam fg 0.9825/0.9760`), background gan `0`, object_desaturate `0.0869`; failure mode khong doi.
- [x] Cleanup no-balanced reject: xoa smoke train dir, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_nobalanced_reject_smoke.json`, reclaimed `180.99 MB`, D: free ~`77.51 GB`.
- [ ] Khong lap lai `DisableBalancedEpochSampling=true` continuation tren current keeper; no chi doi FP/recall nhe va khong tao cue surface/boundary moi.
- [x] 2026-07-03 tra cuu WS-DAN va ADL/attention dropout, trien khai border attention suppression train-time tren patch-token energy: frame ring + bbox edge band, class filter `0,1,2,4`, khong doi raw data va khong doi inference.
- [x] 2026-07-03 preflight border attention pass: `py_compile` cho `train.py`/`config.py`, pytest `tests\test_border_attention_suppression.py` = `3 passed`; dry-run V8 xac nhan `yolo_f`, keeper resume, teacher CSV sample-index, no stale manifests, va actual TrainArgs co `--border-attention-suppression-*`.
- [x] 2026-07-03 smoke border attention `weight=0.006/frame=1/bbox_band=0.12/bbox_weight=0.50/temp=0.20/classes=0,1,2,4`: loss active (`0.2678`, frame mass `0.1181`, bbox-band mass `0.3454`, valid fraction `0.7986`) nhung val macro/class1 chi `0.8828/0.6764`, P/R class1 `0.6042/0.7682`, confusion `0->1=50`, `1->0=19`, `1->2=11`, `4->1=12`, `3->2=52`.
- [x] 2026-07-03 XAI border attention selected12: heatmap foreground sach (`attention/gradcam fg 0.9832/0.9836`) va border flags giam (`gradcam_border=4/12`), nhung background blur/gray van gan `0`, object_desaturate van dominant `0.0633`, overlays van bam vung sang/dom/near-object artifacts; reject truoc probe.
- [x] Cleanup border attention reject: xoa `runs\smoke_v8_yolof_borderattn006_fw1_bb12_bw50_teacherfocusbinary015_24b_1e_20260703` va background launch logs, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_borderattn006_reject_smoke.json`, reclaimed `181.27 MB`, D: free ~`77.44 GB`.
- [ ] Khong lap lai border attention suppression exact setting `weight=0.006/frame=1/bbox_band=0.12/bbox_weight=0.50/temp=0.20/classes=0,1,2,4` tren current keeper; no lam XAI sach hon nhung giam class1 recall va khong giam FP quyet dinh. Neu quay lai border/attention, can reliability/OOF/disagreement signal moi, khong chi global patch-energy penalty.
- [x] 2026-07-03 tra cuu Sub-center ArcFace / Proxy-NCA / Proxy Anchor va trien khai `SubCenterProxyHead` K proxy/class, init tu classifier weight, resume-safe, train CLI + V8 launcher flags; preflight pass (`py_compile` model/train/config + pytest `tests\test_subcenter_proxy_loss.py` = `2 passed`).
- [x] 2026-07-03 smoke sub-center proxy `weight=0.006/K=3/margin=0.08/scale=12/classes=0,1,2,4` tren current keeper: loss active (`0.9938`, valid fraction `0.7986`, pos-neg margin `0.1227`, selected subcenters `2.83`) nhung val macro/class1 chi `0.8822/0.6764`, P/R class1 `0.6042/0.7682`; confusion `0->1=50`, `1->0=19`, `1->2=11`, `4->1=12`, `3->2=53`.
- [x] 2026-07-03 XAI sub-center proxy selected12: foreground cao (`attention/gradcam fg 0.9833/0.9835`), background blur/gray gan `0` (`0.0003/-0.0007`), object_desaturate van dominant `0.0633`, Grad-CAM border `4/12`; failure mode khong doi, reject truoc probe.
- [x] Cleanup sub-center proxy reject: xoa `runs\smoke_v8_yolof_subcenterproxy006_k3_m08_s12_teacherfocusbinary015_24b_1e_20260703` va background launch logs, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_subcenterproxy006_reject_smoke.json`, reclaimed `181.76 MB`, D: free ~`77.37 GB`.
- [ ] Khong lap lai sub-center proxy exact setting `weight=0.006/K=3/margin=0.08/scale=12/classes=0,1,2,4` tren current keeper; multi-proxy active nhung khong tach duoc class1 tren embedding hien tai. Proxy/sub-center tiep theo can representation moi hoac OOF/reliability target, khong chi doi K/weight quanh recipe nay.
- [x] 2026-07-03 tra cuu Bilinear CNN / GSoP second-order pooling va smoke `CompactBilinearPatchFusion` co san thay vi viet module trung lap; preflight pass (`py_compile` model/train/config + 2 bilinear regression tests).
- [x] 2026-07-03 audit launcher: `-Smoke` cua V8 tu set `MaxValBatches=2`, nen metric smoke train chi co `64` val samples va khong duoc dung lam gate. Tu gio gate metric phai full-val support `2606`, bang cach khong dung `-Smoke` hoac chay eval full-val rieng.
- [x] 2026-07-03 full-val eval bilinear `BilinearPatchFusion=true/rank=24/dropout=0.05`: macro/class1 `0.8824/0.6841`, P/R class1 `0.6082/0.7815`, confusion `0->1=50`, `1->0=17`, `1->2=11`, `4->1=12`, `3->2=53`; duoi keeper va khong probe.
- [x] 2026-07-03 XAI bilinear selected12: foreground cao (`attention/gradcam fg 0.9802/0.9846`), background blur/gray gan `0` (`0.0002/-0.0004`), object_desaturate van dominant `0.0628`, Grad-CAM border `4/12`, rollout border `5/12`; failure mode khong doi.
- [x] Cleanup bilinear reject: xoa smoke train dir + logs + full-val eval artifact, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_bilinear_rank24_reject_smoke_eval.json`, reclaimed `186.66 MB`, D: free ~`77.31 GB`.
- [ ] Khong lap lai `BilinearPatchFusion=true/rank=24/dropout=0.05` tren current keeper; second-order residual gan tie keeper nhung khong giam FP/FN class1. Khong tune rank/dropout gan do neu khong co reliability/OOF/case-level signal moi.
- [x] 2026-07-03 tra cuu TransFG/NTS-Net va smoke part-token pairwise head co san, chi bat pairwise route `0-1,1-2,4-1,2-3` thay vi full class residual rong; focused pytest `tests\test_part_token_learner.py` pass (`7 passed`), dry-run full-val support dung.
- [x] 2026-07-03 smoke part-token pairwise `weight=0.02/part_count=4/fg_power=1.20/bbox_weight=0.80/logit_scale=0.16/dropout=0.05`: loss active (`0.6914`) nhung full-val macro/class1 chi `0.8828/0.6784`, P/R class1 `0.6073/0.7682`, confusion `0->1=49`, `1->0=19`, `1->2=11`, `4->1=12`, `3->2=52`.
- [x] 2026-07-03 XAI part-token pairwise selected12: foreground cao (`attention/gradcam fg 0.9796/0.9843`), background blur/gray gan `0` (`0.0003/-0.0003`), object_desaturate van dominant `0.0627`, Grad-CAM border `4/12`, rollout border `5/12`; failure mode khong doi.
- [x] Cleanup part-token pairwise reject: xoa smoke train dir + logs, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_parttokenpair020_reject_smoke.json`, reclaimed `186.49 MB`, D: free ~`77.24 GB`.
- [ ] Khong lap lai part-token pairwise exact setting `PartTokenPairwiseHead=true/weight=0.02/part_count=4/fg_power=1.20/bbox_weight=0.80/pairs=0-1,1-2,4-1,2-3/logit_scale=0.16/dropout=0.05` tren current keeper; hoc part cuc bo van khong tao cue class1 moi va lam recall giam.
- [x] 2026-07-03 tra cuu KD/DKD/RKD va selective noisy-label contrastive learning, roi smoke teacher-pairwise margin co san thay vi them code moi; preflight pass (`py_compile`, focused pytest + full `tests\test_focused_false_positive_margin.py` = `8 passed`), dry-run dung `yolo_f`, keeper resume, strict sample-index teacher CSV, `MaxValBatches=0`.
- [x] 2026-07-03 smoke teacher-pairwise margin `weight=0.006/mass_threshold=0.20/error_power=0.50/hard_target_blend=0.05/require_agreement=true`: loss active (`0.6517`, fraction `0.5358`) nhung full-val macro/class1 chi `0.8828/0.6784`, P/R class1 `0.6073/0.7682`, confusion `0->1=49`, `1->0=19`, `1->2=11`, `4->1=12`, `3->2=52`.
- [x] 2026-07-03 XAI teacher-pairwise selected12: foreground cao (`attention/gradcam fg 0.9833/0.9834`), background blur/gray gan `0` (`0.0005/-0.0004` original-pred drop), object_desaturate van dominant `0.0632`, Grad-CAM border `4/12`, rollout border `4/12`; failure mode khong doi.
- [x] Cleanup teacher-pairwise reject: xoa smoke train dir + logs, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_teacherpair006_reject_smoke.json`, reclaimed `181.31 MB`, D: free ~`77.17 GB`.
- [ ] Khong lap lai teacher-pairwise exact setting `weight=0.006/mass_threshold=0.20/error_power=0.50/hard_target_blend=0.05/require_agreement=true` tren current keeper; teacher margin active nhung khong giam FP/FN class1. Lan sau neu dung teacher-pairwise phai co fold-safe reliability/disagreement/case-level routing moi, khong chi doi weight/threshold gan day.
- [x] 2026-07-03 tra cuu foreground/background recombination, fine-grained domain generalization, GCViT context/local-global design; smoke ket hop `yolo_f` primary + `class_f` auxiliary train-only thay vi sua raw data.
- [x] 2026-07-03 them `trkh.tools.build_mixed_teacher_probability_cache` de tao teacher CSV strict theo sample-index cho MixedTrainDataset; preflight pass (`py_compile` + focused pytest mixed-cache/mixed-dataset = `2 passed`); cache sinh tai `runs\mixed_teacher_cache_yolof_auxclassf025_focus005_20260703` co `11504` rows (`9215` yolo_f + `2289` class_f).
- [x] 2026-07-03 smoke aux-classf `AuxiliaryTrainWeight=0.25` tren current keeper: mixed train active, teacher cache full coverage, full-val macro/class1 chi `0.8834/0.6804`, P/R class1 `0.6105/0.7682`, confusion `0->1=48`, `1->0=19`, `1->2=11`, `4->1=12`, `3->2=53`; duoi keeper va duoi gate.
- [x] 2026-07-03 XAI aux-classf selected12: foreground cao (`attention/gradcam fg 0.9779/0.9794`), background blur/gray gan `0` (`0.0004/-0.0002`), object_desaturate `0.0731`; Grad-CAM border tang `7/12`, rollout border `6/12`; manual overlays van la surface/boundary + bright-background hotspot cuc bo.
- [x] Cleanup aux-classf reject: xoa smoke train dir, giu XAI selected12 + mixed teacher cache + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_auxclassf025_reject_smoke.json`, reclaimed `181.16 MB`, D: free ~`77.11 GB`.
- [ ] Khong lap lai plain mixed `AuxiliaryTrainData=class_f/AuxiliaryTrainWeight=0.25` voi same teacher-focus-binary/bboxprior/boundarydrop anchor; no khong tao cue class1 moi va lam border saliency xau hon. Neu ket hop data tiep, can signal moi nhu OOF/reliability/foreground-background recombination co kiem soat, khong chi them class_f vao batch.
- [x] 2026-07-03 tra cuu LaST-ViT / "Vision Transformers Need More Than Registers" va retest frequency-selective pooling tren current yolo_f keeper thay vi dua vao ket qua V15 cu.
- [x] 2026-07-03 preflight frequency-selective pass: `py_compile` cho `model.py`/`train.py`/`config.py`, pytest `tests\test_detection_calibration.py -k frequency_selective_pooling` = `3 passed`; dry-run xac nhan `yolo_f`, keeper resume, `MaxValBatches=0`, no stale manifests, bbox spatial fusion, pairwise routing, boundarydrop, teacher-focus-binary `0.015`.
- [x] 2026-07-03 smoke frequency-selective `top_k=1/blend=0.12/threshold=0.50`: full-val macro/class1 `0.8856/0.6866`, P/R class1 `0.6250/0.7616`, confusion `0->1=45`, `1->0=19`, `1->2=12`, `4->1=11`, `3->2=46`; small precision/macro bump nhung class1 van duoi gate `0.70`.
- [x] 2026-07-03 XAI smoke frequency-selective selected12: foreground cao (`attention/gradcam fg 0.9828/0.9799`), background blur/gray gan `0`, object_desaturate `0.0810`, Grad-CAM border `7/12`, rollout border `5/12`; vi confusions co cai thien nhe nen cho probe ngan.
- [x] 2026-07-03 probe frequency-selective `120b/2e`: reject, selected best macro/class1 `0.8825/0.6800`, P/R class1 `0.5980/0.7881`, confusion `0->1=53`, `3->2=51`, `1->0=15`, `1->2=12`, `4->1=12`; smoke gain khong on dinh.
- [x] 2026-07-03 XAI probe frequency-selective selected12: foreground van cao (`attention/gradcam fg 0.9881/0.9774`) nhung object_desaturate tang `0.1112`, object_color_sensitive `5/12`; background van khong phai nut that.
- [x] Cleanup frequency-selective reject: xoa smoke/probe train dirs, giu smoke/probe XAI + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_freqsel012_thr50_reject_smoke_probe.json`, reclaimed `363.69 MiB`, D: free ~`76.98 GiB`.
- [ ] Khong lap lai `FrequencySelectivePooling=true/top_k=1/blend=0.12/threshold=0.50` tren current keeper; no chi co transient precision bump o smoke, probe tang surface/color sensitivity va class1 van duoi keeper. Neu quay lai frequency pooling, can foreground reliability/OOF mask hoac case-level signal moi.
- [x] 2026-07-03 preflight focus-class residual head pass: `py_compile` cho `model.py`/`train.py`/`config.py`, `tests\test_focus_class_head.py` = `2 passed`; dry-run da bat loi default `DistillationWeight=0.1` va bbox dropout off, sau do xac nhan lai `DistillationWeight=0`, boundarydrop active, full-val support `2606`, no stale hard manifest.
- [x] 2026-07-03 smoke focus-class head `scale=0.12/aux=0.010/pos=1.25/route_margin=0.28/route_min_p=0.06`: aux loss active (`0.7275`) nhung full-val macro/class1 chi `0.8834/0.6784`, P/R class1 `0.6073/0.7682`, confusion `0->1=49`, `1->0=19`, `1->2=11`, `4->1=12`, `3->2=51`; duoi keeper va duoi gate.
- [x] 2026-07-03 XAI focus-class head selected12: foreground van cao (`attention/gradcam fg 0.9781/0.9794`), background blur/gray gan `0` (`0.00027/-0.00070`), object_desaturate `0.0731`, Grad-CAM border `7/12`, rollout border `6/12`; overlay van la surface/boundary + local artifact, khong co signal moi.
- [x] Cleanup focus-class head reject: xoa smoke train dir, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_focushead012_aux010_pos125_reject_smoke.json`, reclaimed `181.34 MiB`, D: free ~`76.91 GiB`.
- [ ] Khong lap lai `FocusClassHead=true/FocusClassLogitScale=0.12/FocusClassAuxLossWeight=0.010/FocusClassAuxPositiveWeight=1.25/FocusClassRouteMaxProbabilityMargin=0.28/FocusClassRouteMinProbability=0.06` tren current keeper; residual logit class1 khong giam FP/FN va khong doi XAI failure mode.
- [x] 2026-07-03 trien khai diagnostic `trkh.tools.evaluate_counterfactual_color_guard` sau khi tra cuu TTA/confidence-estimation; preflight pass (`py_compile` + `tests\test_counterfactual_color_guard.py` = `3 passed`).
- [x] 2026-07-03 color-counterfactual val smoke `256` mau: object-desaturate doi `253/256` predictions, `0` corrections va `247` harms; train1024 cung doi `1019/1024`, `0` corrections va `999` harms, cho thay desaturation pha semantic cue that.
- [x] 2026-07-03 apply train1024-fit guard `drop=0.16/margin=0.08/conf=0.34` len full val: guarded macro/class1 chi `0.8840/0.6842`, chi giam `0->1` tu `49` xuong `48`, duoi keeper `0.6860` va duoi gate.
- [x] Cleanup counterfactual color diagnostic reject: xoa val smoke, aborted full-train logs/dir, train1024, va full-val train1024-fit artifacts; manifest `runs\cleanup_manifest_20260703_counterfactual_color_guard_reject_diagnostic.json`, reclaimed `1.72 MiB`.
- [ ] Khong lap lai object-desaturate counterfactual voting/guard config `0.16/0.08/0.34` hoac color-logit averaging tren current keeper. Neu quay lai counterfactual color, can perturbation label-preserving nhe hon va selection fold-safe/OOF, khong dung desaturation tho.
- [x] 2026-07-03 cleanup truoc khi tiep tuc: xoa `7` thu muc obsolete `eval_smoke_*` va hflip rong da loi, giu keeper/final/full-val/full-train eval va XAI audit; manifest `runs\cleanup_manifest_20260703_obsolete_eval_smoke_and_empty_hflip.json`, reclaimed `17.04 MiB`.
- [x] 2026-07-03 bat loi hflip diagnostic dung sai default data `D:\DataAI\AIEx\dataset`; xoa artifact sai nguon qua `runs\cleanup_manifest_20260703_hflip_wrong_dataset_reject.json`. Tu nay moi diagnostic gate phai truyen explicit `D:\DataAI\AIEx\newdataset\yolo_f\data.yaml` hoac `class_f_5class.yaml`, khong dua vao `default_data_yaml()`.
- [x] 2026-07-03 tra cuu TTA aggregation/geometric flip, chay hflip full-val no-AMP tren dung `yolo_f`: clean baseline khop evaluator `0.8829/0.6783`, flip hurt `0.8786/0.6553`, avg_prob/entropy_weighted chi `0.8830/0.6762`, entropy_select `0.8824/0.6779`; audit changed predictions cho thay sua `1->0` duoc `5` mau nhung tao them FP class1 tu `0/2/4`, nen reject.
- [x] Cleanup hflip duplicate: xoa ban AMP vi baseline khong khop evaluator, giu `runs\geometric_tta_hflip_keeper_yolof_val_full_noamp_20260703` lam artifact audit nho; manifest `runs\cleanup_manifest_20260703_hflip_amp_duplicate_reject.json`, reclaimed `0.57 MiB`.
- [ ] Khong lap lai hflip geometric TTA simple averaging/entropy weighting tren current keeper; no tang recall bang cach noi rong class1 va lam precision xau hon. Neu revisit geometric TTA, can reliability fold-safe/case-selective truoc, khong tune aggregation tren val.
- [x] 2026-07-03 head-pooling diagnostic tren keeper voi explicit `yolo_f`: current `cls_branch_register_mean` khop baseline `0.8829/0.6783`; `cls_register_mean` roi xuong `0.8616/0.6310` va pure `cls` roi `0.8589/0.6163`, voi harms nhieu hon corrections.
- [ ] Khong train lai head-pooling `cls_register_mean` hoac `cls` tren current keeper; branch/register aggregation dang giu signal bo sung can thiet. Kien truc moi nen them local evidence quanh aggregation hien tai thay vi thay the pooling.
- [x] 2026-07-03 trien khai `PatchMemoryAdapter` local depthwise/pointwise token adapter zero-init, checkpoint-safe, expose config/CLI/V8 launcher; preflight pass (`py_compile` + `tests\test_patch_memory_adapter.py` = `3 passed`). Sua resume allowlist `patch_memory_adapter.*` sau khi smoke dau tien fail truoc train; xoa artifact fail qua `runs\cleanup_manifest_20260703_patchmem_failed_resume_allowlist.json`.
- [x] 2026-07-03 smoke patch-memory tren keeper voi dung recipe `yolo_f` + bbox spatial fusion + boundarydrop + pairwise routing + teacher-focus-binary: full-val macro/class1 chi `0.8822/0.6764`, P/R class1 `0.6042/0.7682`, confusion `0->1=50`, `1->0=19`, `1->2=11`, `4->1=12`, duoi keeper va duoi gate.
- [x] 2026-07-03 XAI patch-memory selected12: top confusions van `3->2=52`, `0->1=50`, `1->0=19`, `4->1=12`; foreground mass cao (`attention/gradcam=0.9780/0.9794`), background blur/gray gan `0`, object_desaturate `0.0730`, Grad-CAM border `7/12`, rollout border `6/12`; overlay van bam edge/shadow/near-background artifact.
- [x] Cleanup patch-memory reject: xoa smoke train dir + launcher logs, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_patchmem000_reject_smoke.json`, reclaimed `191.28 MB`.
- [ ] Khong lap lai `PatchMemoryAdapter=true/dropout=0.0` tren current keeper; local conv adapter zero-init khong tao surface/boundary cue moi. Neu quay lai local-token memory, can reliability/OOF/case-level target moi, khong chi tune dropout/weight quanh adapter nay.
- [x] 2026-07-03 tra cuu LogitNorm/JoCoR/noisy-imbalanced label learning va trien khai `ClassificationLoss=logit_norm` voi temperature `0.04`; preflight pass (`py_compile` + `tests\test_logit_norm_loss.py` = `2 passed`), dry-run dung explicit `yolo_f`, keeper resume, full-val `MaxValBatches=0`, hard manifest rong.
- [x] 2026-07-03 smoke LogitNorm `t=0.04` tren current keeper: full-val macro/class1 chi `0.8818/0.6764`, P/R class1 `0.6042/0.7682`, confusion `0->1=50`, `1->0=19`, `1->2=11`, `4->1=12`, `3->2=53`; duoi keeper va duoi gate.
- [x] 2026-07-03 XAI LogitNorm selected12: top confusions `3->2=53`, `0->1=49`, `1->0=19`, `4->1=12`; foreground cao (`attention/gradcam fg 0.9780/0.9794`), background blur/gray gan `0`, object_desaturate `0.0728`, Grad-CAM border `7/12`, rollout border `6/12`; reject truoc probe.
- [x] Cleanup LogitNorm reject: xoa smoke train dir + launcher logs, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_logitnorm_t004_reject_smoke.json`, reclaimed `190.12 MB`.
- [ ] Khong lap lai `ClassificationLoss=logit_norm/LogitNormTemperature=0.04` tren current keeper; loss-level calibration khong tao cue surface/boundary moi. Neu quay lai LogitNorm, can representation/OOF/reliability target moi, khong chi tune temperature quanh `0.04`.
- [x] 2026-07-03 tra cuu ForAug/background-bias va smoke foreground/background recombination co san trong collator thay vi viet code moi; preflight pass (`py_compile` + `tests\test_detection_calibration.py -k foreground_background_mix` = `2 passed`), dry-run dung `yolo_f`, keeper resume, `classification_loss=ldam_focal`, hard manifest rong, full-val support.
- [x] 2026-07-03 smoke fgbg mix `prob=0.35/margin=0.08/min=0.06/max=0.88/softness=5`: full-val macro/class1 chi `0.8834/0.6784`, P/R class1 `0.6073/0.7682`, confusion `0->1=49`, `1->0=19`, `1->2=11`, `4->1=12`, `3->2=52`; duoi keeper va duoi gate.
- [x] 2026-07-03 XAI fgbg mix selected12: foreground cao (`attention/gradcam fg 0.9781/0.9794`), background blur/gray gan `0`, object_desaturate `0.0734`, Grad-CAM border `7/12`, rollout border `6/12`; overlays gan nhu giong cac smoke truoc. Reject truoc probe.
- [x] Cleanup fgbg mix reject: xoa smoke train dir + launcher logs, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_fgbgmix035_reject_smoke.json`, reclaimed `190.15 MB`.
- [ ] Khong lap lai foreground/background mix exact setting `ForegroundBackgroundMixProbability=0.35/Margin=0.08/Min=0.06/Max=0.88/Softness=5` tren current keeper; background recombination toan cuc khong doi bottleneck. Neu quay lai, can mask foreground dang tin hon hoac OOF/case-level target, khong chi tang/xuong probability.
- [x] 2026-07-03 trien khai crop-bbox foreground/background mix mask source de dung `yolo_f` object position thay pseudo mask; preflight pass (`py_compile` + `tests\test_detection_calibration.py -k foreground_background_mix` = `4 passed`), dry-run dung explicit `yolo_f`, keeper resume, crop-bbox token prior, hard manifest rong, full-val support.
- [x] 2026-07-03 smoke crop-bbox fgbg mix `prob=0.35/mask=crop_bbox/margin=0/min=0.05/max=0.95/softness=5`: full-val macro/class1 chi `0.8827/0.6784`, P/R class1 `0.6073/0.7682`, confusion `0->1=49`, `1->0=19`, `1->2=11`, `4->1=12`, `3->2=52`; duoi keeper va duoi gate.
- [x] 2026-07-03 XAI crop-bbox fgbg selected12: top confusions van `3->2=52`, `0->1=49`, `1->0=19`, `4->1=12`; foreground cao (`attention/gradcam fg 0.9828/0.9759`), background blur/gray gan `0`, object_desaturate `0.0881`, Grad-CAM border `7/12`, rollout border `5/12`, co background hotspot cuc bo. Reject truoc probe.
- [x] Cleanup crop-bbox fgbg reject: xoa smoke train dir + launch logs, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_fgbgmix_cropbbox035_reject_smoke.json`, reclaimed `190.13 MB`.
- [ ] Khong lap lai crop-bbox foreground/background mix exact setting `ForegroundBackgroundMixProbability=0.35/MaskSource=crop_bbox/Margin=0.0/Min=0.05/Max=0.95/Softness=5` tren current keeper; bbox mask dung hon nhung khong doi surface/boundary bottleneck. Neu revisit background/context, can OOF/case-level reliability hoac signal representation moi, khong chi mask source/probability.
- [x] 2026-07-03 tra cuu ForAug v4, AutoBackSwap va CVPR 2022 foreground/background sensitivity; retest nhe `background_focus_suppression` thay vi tiep tuc gia thuyet nen rong khi audit background perturbation gan 0.
- [x] 2026-07-03 preflight bgfocus006 pass: `py_compile train.py/config.py`, V8 launcher parse OK, pytest `tests\test_focused_false_positive_margin.py -k background_focus_suppression` = `1 passed`; dry-run dung explicit `yolo_f`, keeper resume, sample-index teacher CSV, hard manifest rong, full-val `MaxValBatches=0`.
- [x] 2026-07-03 smoke bgfocus006 `weight=0.006/prob=0.50/neg=0,4/margin=0.015/minp=0.05/power=1.5`: full-val macro/class1 chi `0.8816/0.6725`, P/R class1 `0.6021/0.7616`, confusion `0->1=50`, `1->0=19`, `1->2=12`, `4->1=12`, `3->2=53`; loss active nhung delta nen am nhe `-0.0007`, duoi keeper va duoi gate.
- [x] 2026-07-03 XAI bgfocus006 selected12: foreground van cao (`attention/gradcam fg 0.9779/0.9788`), background blur/gray gan `0` (`0.00046/-0.00043`), object_desaturate `0.0730`, Grad-CAM border `9/12`, rollout border `6/12`; manual overlays van la fruit top/edge-shadow/plate-object hotspots, khong co signal nen on dinh.
- [x] Cleanup bgfocus006 reject: xoa smoke train dir + launch logs, giu XAI selected12 + keeper/final artifacts; manifest `runs\cleanup_manifest_20260703_bgfocus006_p50_reject_smoke.json`, reclaimed `190.23 MB`.
- [ ] Khong lap lai `BackgroundFocusSuppressionLossWeight=0.006/Probability=0.50/FocusClass=1/NegativeClasses=0,4/Margin=0.015/MinProbability=0.05/Power=1.5` tren current keeper; background-focus suppression nhe van lam class1 giam va xac nhan background khong phai nut that. Neu quay lai background/context, can reliability/OOF/case-level signal moi.
- [x] 2026-07-03 trien khai standalone peer small-loss co-teaching diagnostic `trkh.tools.coteach_finetune` sau khi tra cuu Co-teaching/DivideMix/GCE/PRODEN; dung 2 peer keeper + soup, khong sua raw data, sample selection train-only.
- [x] 2026-07-03 preflight co-teaching pass: `py_compile trkh\tools\coteach_finetune.py tests\test_coteach_finetune.py` va `tests\test_coteach_finetune.py` = `2 passed`.
- [x] 2026-07-03 smoke co-teaching `anchor+soup/remember_rate=0.80/LR=1e-5/24b/1e`: full-val support `2606`, best peer A macro/class1 chi `0.8697/0.6407`, P/R class1 `0.5847/0.7086`; peer B `0.8691/0.6369`; confusion best `0->1=45`, `1->0=27`, `1->2=12`, `4->1=14`, `3->2=44`. Reject vi duoi keeper va duoi gate.
- [x] 2026-07-03 XAI co-teaching selected12: foreground van cao (`attention/gradcam fg 0.9628/0.9677`), background blur/gray gan `0`, object_desaturate `0.0778`, Grad-CAM border `7/12`, object_color_sensitive `4/12`; overlay `0->1/1->0/4->1` van la surface/boundary cue, khong phai nen rong.
- [x] Cleanup co-teaching reject: copy summary/history/peer metrics vao `runs\xai_smoke_coteach_yolof_peer_anchor_soup_r80_lr1e5_val_selected12_20260703\source_run_metrics`, xoa smoke train dir + smoke logs; manifest `runs\cleanup_manifest_20260703_coteach_peer_r80_reject_smoke.json`, reclaimed `37.17 MB`.
- [ ] Khong lap lai co-teaching exact setting `anchor+soup/remember_rate=0.80/LR=1e-5/24b/1e`; peer selected-overlap qua cao `0.9625`, nen no lap lai loss-rank/self-paced va lam class1 sup manh. Neu quay lai sample-selection, can OOF/fold-safe reliability hoac diverse peers co disagreement do duoc truoc smoke.
- [x] 2026-07-03 trien khai diagnostic frozen pairwise feature verifier `trkh.tools.probe_pairwise_feature_verifier` tren head embedding + probabilities, pairs `0-1,1-2,1-4,2-3`; preflight pass (`py_compile` + `tests\test_pairwise_feature_verifier.py` = `4 passed`).
- [x] 2026-07-03 verifier `t060/m040`: full train `9215`, full val `2606`, pair OOF local acc cao (`0-1=0.9222`, `1-2=0.9817`, `1-4=0.9954`, `2-3=0.9806`); val base macro/class1 `0.8841/0.6841`, verified `0.8855/0.6901`, P/R class1 `0.6667/0.7152`, changed `65` (`32` corrections, `27` harms). Gan gate nhung van duoi `0.70`.
- [x] 2026-07-03 verifier conservative `t065/m030`: val macro/class1 `0.8859/0.6899`, P/R class1 `0.6606/0.7219`, changed `55` (`27` corrections, `22` harms); macro nhe hon nhung class1 khong vuot `t060`, dung khong sweep val them.
- [x] 2026-07-03 XAI changed-case verifier selected12: near-tie `5/12`, foreground van la chinh (`attention/gradcam fg 0.9356/0.9429`), background blur/gray gan `0`, object_desaturate lon hon, rollout border `10/12`; overlay harm/correction deu la cung surface-boundary ambiguity, khong co cue moi.
- [x] Cleanup verifier variant: giu `runs\pairwise_feature_verifier_yolof_keeper_t060_m040_20260703` lam diagnostic gan-gate va XAI `runs\xai_pairwise_feature_verifier_yolof_keeper_t065_m030_changed12_20260703`, xoa run conservative kem hon + verifier logs; manifest `runs\cleanup_manifest_20260703_pairwise_feature_verifier_variant_reject.json`, reclaimed `2.20 MB`.
- [ ] Khong lap lai frozen pairwise verifier thresholds quanh `t=0.60-0.65/margin=0.30-0.40` tren current keeper; no chi dich bien class1 va trade precision/recall, chua qua gate. Neu quay lai verifier/pairwise, can representation moi hoac OOF confidence/in-model target giu recall class1 ma khong mo rong `0->1/4->1`.
- [x] 2026-07-03 tra cuu dense-readout/MIL, TransFG va FFVT; trien khai diagnostic `trkh.tools.probe_patch_evidence_mil` ap head len patch tokens, tom tat patch/bbox top-k evidence, fit train-only verifier. Preflight pass (`py_compile` + `tests\test_patch_evidence_mil.py` = `3 passed`).
- [x] 2026-07-03 all-pairs patch-evidence `pairs=0-1,1-2,1-4,2-3`: val macro/class1 giam xuong `0.8827/0.6792`, changed `72` (`33` corrections, `32` harms). Direct patch max/top-k rat te (class1 toi da ~`0.295`), nen khong dung naive dense aggregation.
- [x] 2026-07-03 patch-evidence `0-1` only: full train `9215`, full val `2606`, OOF local acc `0.9243`; val base `0.8841/0.6841` -> patch-verified `0.8892/0.7073`, P/R class1 `0.6554/0.7682`, changed `31` (`16` corrections, `9` harms, `6` neutral). Day la diagnostic legal dau tien vuot class1 gate `0.70`, nhung van la post-hoc verifier.
- [x] 2026-07-03 XAI patch-evidence `0-1` changed16: foreground cao (`attention/grad_rollout/gradcam/rollout 0.9510/0.9812/0.9934/0.9546`), background blur/gray gan `0`, object_desaturate lon hon, near-tie `10/16`, rollout_border `13/16`. Harm/correction deu la surface/boundary cues rat gan nhau; mot harm class1 co leaf/branch/background hotspot.
- [x] Cleanup patch-evidence all-pairs reject: copy summary/metrics vao `runs\xai_patch_evidence_mil_yolof_keeper_01only_changed16_20260703\source_patch_evidence_metrics`, xoa all-pairs run; manifest `runs\cleanup_manifest_20260703_patch_evidence_allpairs_reject.json`, reclaimed `2.38 MB`.
- [ ] Khong lap lai patch-evidence all-pairs routing voi `2-3`; no lam mat gain class1. Huong tiep theo co gia tri: bien signal `0-1` patch-evidence thanh in-model trainable MIL/router checkpoint-safe hoac OOF target, co reliability gating de giu recall class1 ma khong tang `0->1`.
- [x] 2026-07-03 trien khai train-time patch-evidence MIL loss `0-1`: config/CLI/resume override/history telemetry, ap current head len patch tokens va BCE tren top-k bbox patch margin; preflight pass (`py_compile` + focused pytest `11 passed`).
- [x] 2026-07-03 smoke patch-evidence MIL loss `weight=0.004/pair=0-1/top_k=4/bbox_threshold=0.05/positive_weight=1.0/LR=1e-5/24b/1e`: loss active (`0.6614`, selected fraction `0.3993`, pos-neg margin `0.2534`) nhung full-val macro/class1 chi `0.8835/0.6784`, P/R class1 `0.6073/0.7682`, confusion `0->1=49`, `1->0=19`, `1->2=11`, `4->1=12`, `3->2=52`; reject truoc probe.
- [x] 2026-07-03 XAI patch-evidence MIL loss selected12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9139/0.9733/0.9469/0.9529`), background blur/gray gan `0`, object desaturation lon hon, attention border `12/12`, Grad-CAM border `7/12`; overlays van la fruit surface/boundary voi mot so table/background-adjacent artifact.
- [x] Cleanup patch-evidence MIL loss reject: copy `history.csv`, metrics, config vao `runs\xai_smoke_patch_evidence_mil_yolof_keeper01_w004_top4_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_patch_evidence_mil_w004_reject_smoke.json`, reclaimed `185.03 MB`.
- [x] 2026-07-03 sua `plot_all_training_metrics` de chunk plot lich su qua nhieu cot thanh `all_training_metrics_part*.png`, tranh crash sau train khi telemetry tang.
- [ ] Khong lap lai train-time patch-evidence MIL top-k BCE exact setting `weight=0.004/pair=0-1/top_k=4/bbox_threshold=0.05/positive_weight=1.0/LR=1e-5/24b/1e` tren current keeper. Neu dung lai patch signal, phai co reliability-gated/OOF/learned selector target de chon local evidence dang tin, khong dung raw top-k BCE.
- [x] 2026-07-03 tra cuu gated attention MIL/CLAM/TransMIL va trien khai checkpoint-safe `PatchEvidenceRouterHead` zero-init tren patch tokens + pair patch margin + crop-bbox prior; preflight pass (`py_compile` + focused pytest `13 passed`).
- [x] 2026-07-03 smoke gated patch router `pair=0-1/hidden=128/top_k=4/bbox_weight=0.75/logit_scale=0.12/loss=0.020/router-only/LR=5e-4/24b/1e`: full-val macro/class1 bang keeper `0.8847/0.6860`, class1 P/R `0.6114/0.7815`; router loss active nhung logit margin chi `0.0057`, khong tao decision gain.
- [x] 2026-07-03 XAI gated patch router selected12: top confusions van `3->2=52`, `0->1=50`, `1->0=18`, `4->1=13`, `1->2=11`; background blur/gray gan `0`, object_desaturate `0.0665`, attention_border `11/12`, Grad-CAM border `6/12`. Manual overlays cho thay loi bam bright edge/shadow/stem/table line, can interior-surface reliability signal.
- [x] Cleanup gated patch router reject: copy metrics/config/history vao `runs\xai_smoke_patch_router_yolof_keeper01_w020_top4_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_patch_router_w020_reject_smoke.json`, reclaimed `131.76 MB`.
- [ ] Khong lap lai gated patch router exact setting `PatchEvidenceRouterHead=true/pair=0-1/hidden=128/top_k=4/bbox_weight=0.75/logit_scale=0.12/loss_weight=0.020/router-only/LR=5e-4/24b/1e` tren current keeper; zero-init residual router khong du gradient/target de di chuyen boundary. Huong tiep theo phai tao representation/interior-surface signal hoac fold-safe reliability target, khong chi route lai patch margin cu.
- [x] 2026-07-03 tra cuu stacked generalization/OOF confidence va mo rong `probe_patch_evidence_mil` de xuat `train\\pair_verifier_teacher_oof.csv`; preflight pass (`py_compile` + focused pytest `14 passed`). Artifact `runs\patch_evidence_mil_yolof_keeper_01only_t060_m040_top4_cropbbox_oofteacher_20260703` phu `9215` train samples, OOF local acc `0.9243`, teacher agreement `0.9796`.
- [x] 2026-07-03 smoke OOF patch-verifier teacher -> pairwise margin head `loss=0.020/mass=0.50/error_power=0.50/hard_blend=0/trainable=pairwise_margin_norm+head/LR=5e-4/24b/1e`: loss active (`0.5531`, fraction `0.5273`) nhung full-val macro/class1 chi `0.8834/0.6822`, P/R class1 `0.6094/0.7748`; duoi keeper.
- [x] 2026-07-03 XAI OOF teacher-pairwise selected12: top confusions van `3->2=52`, `0->1=49`, `1->0=19`, `4->1=13`; attention_border `12/12`, Grad-CAM border `6/12`, background blur/gray gan `0`, object_desaturate `0.0662`; overlay `0->1` van bam bright edge/shadow.
- [x] Cleanup OOF teacher-pairwise reject: copy metrics vao `runs\xai_smoke_oof_patch_teacher_pairwise_yolof_keeper01_w020_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_oof_patch_teacher_pairwise_reject_smoke.json`, reclaimed `127.71 MB`.
- [ ] Khong lap lai OOF patch-verifier teacher-pairwise exact setting `teacher_pairwise_margin_loss_weight=0.020/mass=0.50/error_power=0.50/hard_blend=0/trainable=pairwise_margin_norm+head/LR=5e-4/24b/1e` tren current keeper. OOF verifier van huu ich post-hoc, nhung target gan hard label khong tao representation/interior-surface signal moi khi distill vao pairwise head hien tai.
- [x] 2026-07-03 them auxiliary train support cho `probe_patch_evidence_mil`: `--auxiliary-train-data`, `--auxiliary-train-weight`, source-domain feature, va logistic sample weights; preflight pass (`py_compile` + focused pytest `10 passed`).
- [x] 2026-07-03 probe auxiliary `class_f/train` patch verifier `weight=0.25/source_domain_feature/pair=0-1/t=0.60/margin=0.40/top_k=4/crop_bbox`: OOF local acc tang `0.9363` nhung full-val `yolo_f` chi `0.8883/0.7030`, thap hon yolo-only verifier `0.8892/0.7073`, voi `31` changes (`16` corrections, `10` harms, `5` neutral).
- [x] 2026-07-03 XAI auxiliary patch verifier changed16 harm-first: foreground cao (`attention/grad_rollout/gradcam/rollout 0.9522/0.9802/0.9928/0.9542`), background blur/gray gan `0`, attention_border `9/16`, rollout_border `14/16`, near-tie `9/16`; overlays van la surface/edge + leaf/background-adjacent hotspots.
- [x] Cleanup auxiliary patch verifier reject: copy summary/metrics vao `runs\xai_patch_evidence_mil_yolof_keeper_01only_auxclassf_w025_domain_changed16_20260703\source_patch_evidence_metrics`, xoa probe dir; manifest `runs\cleanup_manifest_20260703_auxclassf_patch_evidence_reject_probe.json`, reclaimed `2.84 MB`.
- [ ] Khong lap lai `auxiliary_train_data=class_f/weight=0.25/source_domain_feature=true/pair=0-1/t=0.60/margin=0.40/top_k=4/crop_bbox` tren current keeper; class_f crop-domain tang OOF train nhung khong cai thien yolo_f val so voi yolo-only patch verifier. Neu dung lai class_f, can cross-view objective hoac OOF domain adaptation moi, khong chi them crop rows vao verifier.
- [x] 2026-07-03 them optional spatial interior/border features cho `probe_patch_evidence_mil` (`--spatial-evidence-features`, `--spatial-interior-erode`), tach token interior vs border/ring; preflight pass (`py_compile` + focused pytest `11 passed`).
- [x] 2026-07-03 probe spatial patch verifier `spatial=true/erode=1/pair=0-1/t=0.60/margin=0.40/top_k=4/crop_bbox`: feature dim `388`, OOF local acc giam `0.9226`, full-val chi `0.8883/0.7030`, thap hon yolo-only verifier `0.8892/0.7073`, changes `33` (`17` corrections, `11` harms, `5` neutral).
- [x] 2026-07-03 XAI spatial patch verifier changed16 harm-first: attention_border `11/16`, rollout_border `14/16`, background perturb gan `0`, object_desaturate van lon hon; harm moi `Image_6003` bi attention nen/la/vien keo manh va Grad-CAM object gan nhu trong.
- [x] Cleanup spatial patch verifier reject: copy summary/metrics vao `runs\xai_patch_evidence_mil_yolof_keeper_01only_spatial_erode1_changed16_20260703\source_patch_evidence_metrics`, xoa probe dir; manifest `runs\cleanup_manifest_20260703_spatial_patch_evidence_reject_probe.json`, reclaimed `4.72 MB`.
- [ ] Khong lap lai `spatial_evidence_features=true/spatial_interior_erode=1/pair=0-1/t=0.60/margin=0.40/top_k=4/crop_bbox` tren current keeper; interior/border feature quanh frozen logits khong giam harm va lam verifier kem hon yolo-only. Neu quay lai spatial reliability, can target supervision/representation moi, khong tune erode/ring quanh recipe nay.
- [x] 2026-07-03 trien khai OOF patch-verifier disagreement sample weighting: tool `trkh.tools.build_patch_oof_sample_weights`, train resume override cho `--sample-weight-manifest`, preflight `py_compile` + focused pytest `2 passed`.
- [x] 2026-07-03 manifest `runs\patch_oof_sample_weights_yolof_keeper01_t065_w035_20260703`: train-only OOF, sample-index strict, `9215` rows, selected `151` confident 0-1 disagreements (`0->1=117`, `1->0=34`), weight `0.35`.
- [x] 2026-07-03 smoke OOF patch sample-weight `threshold=0.65/weight=0.35/LR=1e-5/24b/1e`: full-val macro/class1 chi `0.8823/0.6744`, P/R class1 `0.6010/0.7682`, confusion `0->1=51`, `1->0=19`, `1->2=11`, `4->1=12`, `3->2=52`; duoi keeper va duoi gate.
- [x] 2026-07-03 XAI OOF patch sample-weight selected12: foreground cao (`attention/grad_rollout/gradcam/rollout 0.9454/0.9781/0.9683/0.9636`), background blur/gray gan `0` (`0.0001/-0.0008`), object_desaturate lon hon `0.0345`, attention_border `11/12`, Grad-CAM border `6/12`; overlay van surface/edge va table/background hotspot cuc bo.
- [x] Cleanup OOF patch sample-weight reject: copy metrics/config/history vao `runs\xai_smoke_patchoof_sampleweight_yolof_keeper01_t065_w035_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_patchoof_sampleweight_reject_smoke.json`, reclaimed `185.60 MB`.
- [ ] Khong lap lai OOF patch-verifier disagreement sample weighting exact setting `pair=0-1/threshold=0.65/weight=0.35/sample_weight_factor=1.0/sample_weight_max=1.0/LR=1e-5/24b/1e` tren current keeper; no giam class1 va khong tao cue interior-surface moi. Neu quay lai sample reliability, can OOF target khong don gian la deweight disagreement hard 0/1.
- [x] 2026-07-03 final-test diagnostic cho yolo-only patch-evidence 0-1 verifier voi val-selected fixed setting `t=0.60/margin=0.40/top_k=4/crop_bbox`: test base `0.8865/0.6667` -> patch-verified `0.8905/0.6795`, changes `11` (`7` corrections, `3` harms, `1` neutral).
- [x] 2026-07-03 XAI patch-evidence test changed11: foreground cao (`attention/grad_rollout/gradcam/rollout 0.9663/0.9818/0.9039/0.9738`), background blur/gray gan `0`, object_desaturate target drop `0.0489`, attention_border `9/11`, rollout_border `8/11`, near-tie `5/11`; corrections giam `0->1`, harms la class1 that bi keo ve 0.
- [ ] Giu `runs\patch_evidence_mil_yolof_keeper_01only_t060_m040_top4_cropbbox_test_final_20260703` va `runs\xai_patch_evidence_mil_yolof_keeper_01only_test_changed11_20260703` lam diagnostic post-hoc nho; khong tune threshold/margin/top_k bang test. Huong tiep theo phai hoc reliability target tren train/val de giu recall class1, khong chi hard-route local patch verifier.
- [x] 2026-07-03 them `trkh.tools.apply_patch_verifier_directional_gate` de chon gate theo huong tu train-OOF prediction CSV; preflight pass (`py_compile` + `tests\test_patch_verifier_directional_gate.py` = `1 passed`).
- [x] 2026-07-03 directional gate OOF-risk `0.60`: train OOF tu dong disable `0->1` vi `18` candidates co `0` corrections/`17` harms; chon `1->0` threshold `0.7273` voi train OOF `25` corrections/`13` harms.
- [x] 2026-07-03 val directional gate `runs\patch_verifier_directional_gate_yolof_keeper01_oofrisk060_20260703`: base `0.8841/0.6841` -> gated `0.8883/0.7015`, P/R class1 `0.6552/0.7550`, changes `20` (`11` corrections, `4` harms, `5` neutral); qua gate nhung thap hon fixed verifier `0.8892/0.7073`, nen khong test.
- [x] 2026-07-03 XAI directional gate changed20: foreground cao (`attention/grad_rollout/gradcam/rollout 0.9603/0.9778/0.9892/0.9621`), background gan `0`, object_desaturate target drop `0.0460`, near-tie `12/20`, rollout_border `16/20`; van la reliability cua cue surface/boundary, khong phai nen rong.
- [ ] Khong bat lai `0->1` patch-verifier routing tren current keeper neu khong co target moi; train OOF cho thay direction nay doc. Neu dung patch verifier tiep, uu tien conservative `1->0` false-positive suppressor + co che bao ve class1 recall.
- [x] 2026-07-03 them `trkh.tools.build_patch_directional_margin_manifest` de bien directional gate train changed-cases thanh targeted-margin manifest sample-index strict; preflight pass (`py_compile` + targeted-margin tests `2 passed`).
- [x] 2026-07-03 manifest patch-directional margin `runs\patch_directional_targeted_margin_yolof_keeper01_t0727_w15p08_20260703`: `40` train rows, gom `27` suppressor (`0/2 > 1`) va `13` protector (`1 > 0`), khong co leakage/path fallback.
- [x] 2026-07-03 smoke patch-directional targeted margin `loss=0.010/LR=1e-5/120b/1e`: targeted loss active nhung rat thua (`fraction=0.0070`, loss `0.0089`), full-val macro/class1 chi `0.8816/0.6763`, P/R class1 `0.6000/0.7748`, confusion `0->1=51`, `1->0=18`, `1->2=11`, `4->1=12`, `3->2=53`.
- [x] 2026-07-03 XAI patch-directional margin selected12: foreground kem hon keeper (`attention/gradcam fg 0.8927/0.9479`), background blur/gray van gan `0` (`0.0005/-0.0004`), object_desaturate `0.0886`, attention_background `5/12`, attention_border `11/12`, Grad-CAM border `6/12`; overlay bam background/stem/edge va surface rong.
- [x] Cleanup patch-directional targeted-margin reject: copy metrics/config/history vao `runs\xai_smoke_patchdir_targetedmargin_yolof_keeper01_t0727_w010_selected12_20260703\source_smoke_metrics`, xoa train dir + launch logs; manifest `runs\cleanup_manifest_20260703_patchdir_targetedmargin_reject_smoke.json`, reclaimed `181.67 MB`.
- [ ] Khong lap lai patch-directional targeted-margin exact policy `direction=1->0/threshold=0.7273/suppressor=0.06x1.5/protector=0.04x0.8/loss_weight=0.010/LR=1e-5/120b/1e` tren current keeper; OOF signal qua thua va khong tao representation/interior-surface cue moi.
- [x] 2026-07-03 cleanup obsolete train-to-val reroute/logit-bias/focus-specialist probes da reject trong dong group-clean cu; giu keeper/final/post-hoc diagnostic, manifest `runs\cleanup_manifest_20260703_obsolete_train_to_val_reroute_focus_probes.json`, reclaimed `127.48 MB`.
- [x] 2026-07-03 tao diagnostic `runs\patch_evidence_direction_filter_diagnostic_20260703`: val `1->0 only` nhin tot hon fixed verifier (`0.8904/0.7103` vs `0.8892/0.7073`) nhung final-test reference xau hon fixed verifier (`0.8884/0.6710` vs `0.8905/0.6795`), nen day la val trap.
- [ ] Khong implement/tune post-hoc patch-verifier `1->0 only` hoac "remove 0->1" bang val/test tren current keeper. Neu dung direction reliability tiep, phai tao target train-only/OOF moi va audit XAI truoc khi cham test.
- [x] 2026-07-03 tra cuu stacked generalization/selective classification/SelectiveNet, sua `probe_pairwise_feature_verifier` de export true train OOF `verifier_0_1_prob_*`, va them `trkh.tools.stack_patch_feature_verifiers`; preflight pass `py_compile` + focused pytest `8 passed`.
- [x] 2026-07-03 stacked patch+feature selective val `class1_benefit/t=0.5`: `runs\stacked_patch_feature_verifier_yolof_keeper01_class1benefit_t050_20260703`, val base `0.8841/0.6841` -> stacked `0.8911/0.7125`, P/R class1 `0.6746/0.7550`, changes `25` (`15` corrections, `4` harms, `6` neutral); qua gate val nhung chi la diagnostic post-hoc.
- [x] 2026-07-03 XAI stacked val changed25: foreground cao (`attention/gradcam fg 0.9489/0.9894`), background blur/gray gan `0`, object_desaturate target drop `0.0378`, near_tie `12/25`, rollout_border `21/25`; gain la suppress FP class1 tren surface/border, khong phai cue nen/context moi.
- [x] 2026-07-03 frozen final-test audit stacked selective: `runs\stacked_patch_feature_verifier_yolof_keeper01_class1benefit_t050_test_final_20260703`, test base `0.8865/0.6667`, fixed patch `0.8905/0.6795`, stacked roi xuong `0.8842/0.6536`; changes `12` (`6` corrections, `5` harms, `1` neutral), reject.
- [x] 2026-07-03 XAI stacked test changed12: foreground cao (`attention/grad_rollout/gradcam/rollout 0.9678/0.9898/0.9112/0.9777`), background gan `0`, object_desaturate target drop `0.0535`; accepted changes co `5` true class1 bi suppress ve 0 nen class1 recall sup.
- [ ] Khong lap lai `stack_patch_feature_verifiers target_mode=class1_benefit/threshold=0.5` hoac tune nearby post-hoc accept threshold tren current keeper. Huong tiep theo can recall-protection/representation cho true class1 surface defect, khong chi suppressor FP class1.
- [x] 2026-07-03 remap top-5 pretrained TTA teacher cache tu `class_f` sang `yolo_f` sample-index trong `runs\top5_tta_teacher_yolof_sampleindex_20260703`; strict coverage val/test `2606/2606` va `1267/1267`, no missing key, class order explicit tu `target_classes`.
- [x] 2026-07-03 top-5 recall-protector val diagnostic cho fixed patch suppressions: fixed patch val `0.8892/0.7073`; `24` suppressions base `1 -> non1` co target `{0:14,1:4,2:1,4:5}`; moi rule non-empty theo top-5 p1/pred1 deu giam metric, best la no-op high threshold. Summary: `runs\top5_tta_teacher_yolof_sampleindex_20260703\recall_protector_val_summary.json`.
- [ ] Khong lap lai simple top-5/AIDT teacher recall-protector cho patch suppressions bang confidence/pred1 threshold. Neu dung top-5 tiep, dung de phan tich error/representation va tao trainable/fold-safe signal, khong them post-hoc restore threshold.
- [x] 2026-07-03 diagnostic complementarity top-5 vs TRKH val: top-5 sua `105` loi TRKH nhung pha `34` ca TRKH dung; voi class1, top-5 sua `12` loi nhung pha `20` ca TRKH dung, cho thay top-5 conservative hon va khong phai recall-protector.
- [x] 2026-07-03 frozen soft ensemble val-class1-opt `TRKH=0.75/top5=0.25`: val `0.9166/0.7500`, final test `0.9196/0.7737`; vuot TRKH/AIDT-style gate nhung thap hon top-5 TTA test `0.9248/0.7786`.
- [x] 2026-07-03 frozen soft ensemble val-macro-opt `TRKH=0.48/top5=0.52`: val `0.9166/0.7492`, final test `0.9242/0.7786`; gan bang nhung van thap hon top-5 TTA macro. Summary: `runs\top5_tta_teacher_yolof_sampleindex_20260703\trkh_top5_soft_ensemble_diagnostic_summary.json`.
- [ ] Khong sweep/tune them TRKH+top5 soft-ensemble weights bang test. Huong nay chi la pretrained upper-bound; tiep theo phai distill/reproduce conservative FP control vao no-pretrain model ma khong lam mat recall class1.
- [x] 2026-07-03 export top-5 pretrained TTA train predictions cho 5 TIMM experts vao `runs\pretrained_expert_predictions_tta_5class_train_20260703`, build train ensemble `runs\ensemble_top5_tta_teacher_cache_5class_train_20260703`, va remap sang `yolo_f/train` sample-index full coverage `9215/9215`.
- [x] 2026-07-03 audit top-5 train teacher: agreement hard label `0.9998`, chi `2/9215` disagreements, mean confidence `0.9919`, median `0.99998`; direct KD/teacher-pairwise tu cache nay se gan nhu lap lai CE in-sample. Summary: `runs\top5_tta_teacher_yolof_sampleindex_20260703\train_teacher_reliability_audit.json`.
- [ ] Khong dung top-5 train teacher in-sample nay lam direct KD/teacher-pairwise/conservative FP-control target. Neu can top-5 transfer, phai co OOF/fold-safe teacher prediction hoac representation signal khac.
- [x] 2026-07-03 tao normalized sample-index router inputs `runs\top5_trkh_router_inputs_20260703` cho TRKH/top5 voi class name `0..4`; focused router tests pass (`tests\test_trainval_expert_router.py`, `3 passed`).
- [x] 2026-07-03 trainval router top5+TRKH fit tren val bang OOF selection, final test sau khi chon candidate: selected `avg_temp_0p5`, OOF `0.9166/0.7492`, test `0.9226/0.7727`, thap hon top-5 TTA `0.9248/0.7786`; reject final route. Artifact `runs\top5_trkh_trainval_router_valfit_testfinal_20260703`.
- [ ] Khong tune router candidates/selection weights sau khi xem test. `extra_trees_leaf10` nhin co test macro `0.9250` nhung khong duoc OOF chon, nen khong dung lam ket luan. Huong tiep theo phai la trainable no-pretrain representation cue, khong phai post-hoc top5/TRKH router.
- [x] 2026-07-03 tra cuu AugMix/JSD consistency va WS-DAN; trien khai AugMix/JSD-lite geometry-preserving photometric/detail consistency cho train loop, config, CLI, v8 launcher. Preflight pass: `py_compile trkh\training\train.py trkh\core\config.py` va `tests\test_illumination_consistency_loss.py` = `7 passed`.
- [x] 2026-07-03 smoke AugMix/JSD-lite `weight=0.008/prob=0.50/severity=0.18/width=2/depth=2/alpha=1.0/temp=1.0`: loss active (`0.00104`, fraction `0.4974`) nhung full-val macro/class1 chi `0.8831/0.6784`, P/R class1 `0.6073/0.7682`, confusion `0->1=49`, `1->0=19`, `1->2=11`, `4->1=12`, `3->2=53`; duoi keeper va duoi gate.
- [x] 2026-07-03 XAI AugMix/JSD selected12: foreground cao (`attention/grad_rollout/gradcam/rollout 0.9834/0.9733/0.9834/0.9529`), background blur/gray gan `0`, object_desaturate `0.0633`, border/object-color flags van xuat hien; reject truoc probe.
- [x] Cleanup AugMix/JSD reject: copy metrics/config vao `runs\xai_smoke_v8_yolof_augmixjsd008_p50_s18_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir va auto `final_test` khong dung; manifest `runs\cleanup_manifest_20260703_augmixjsd008_reject_smoke.json`, reclaimed `191.76 MB`.
- [ ] Khong lap lai AugMix/JSD-lite exact setting `weight=0.008/prob=0.50/severity=0.18/width=2/depth=2/alpha=1.0/temp=1.0` tren current keeper; generic photometric consistency khong tao cue surface/boundary moi. Neu revisit AugMix, can class-pair/OOF/case-level reliability target, khong chi tune severity/probability.
- [ ] Smoke/probe sau nay phai them `-SkipFinalTest` cho v8 launcher hoac `--skip-final-test` khi goi train truc tiep; neu auto tao `final_test` trong smoke thi phai xoa/ignore va ghi ro khong dung test de ra quyet dinh.
- [x] 2026-07-03 tra cuu SCOPE/FT-Former/WST ve subtle-cue/frequency detail; dung module san co `HighFrequencyTextureExpert` thay vi viet duplicate texture branch. Preflight pass: `py_compile model.py/train.py/config.py`, `tests\test_high_frequency_texture_expert.py` = `4 passed`; dry-run dung explicit `yolo_f`, `SkipFinalTest`, hard manifest rong.
- [x] 2026-07-03 smoke high-frequency texture `Expert=true/aux=0.010/pairwise=0.010/route_pairs=0-1,1-2,2-3,1-4/route_margin=0.25/logit_scale=0.16/analysis=96`: loss active (`aux=5.4336`, `pairwise=0.6914`) nhung full-val macro/class1 chi `0.8828/0.6764`, P/R class1 `0.6042/0.7682`, confusion `0->1=50`, `1->0=19`, `1->2=11`, `4->1=12`, `3->2=52`; duoi keeper va gate.
- [x] 2026-07-03 XAI high-frequency selected12: foreground cao (`attention/grad_rollout/gradcam/rollout 0.9834/0.9733/0.9834/0.9529`), background blur/gray gan `0`, object_desaturate `0.0632`, border/object-color flags khong giam; reject truoc probe.
- [x] Cleanup high-frequency reject: copy metrics/config vao `runs\xai_smoke_v8_yolof_hifreqtex_w010_pw010_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_hifreqtex_w010_reject_smoke.json`, reclaimed `196.69 MB`.
- [ ] Khong lap lai high-frequency texture exact setting `HighFrequencyTextureExpert=true/aux=0.010/pairwise=0.010/route_pairs=0-1,1-2,2-3,1-4/route_margin=0.25/logit_scale=0.16/analysis=96` tren current keeper; raw high-pass/gradient/laplacian expert khong tao cue class1 moi. Neu revisit frequency/detail, can reliability/OOF/interior-surface target moi, khong chi tang aux/pairwise weight.
- [x] 2026-07-03 sua ambiguous soft-target loader de uu tien `sample_index`/`dataset_index` tren `yolo_f`, tranh path-collapse voi multi-object image; preflight pass `py_compile train.py/dataset.py` va `tests\test_detection_calibration.py -k "ambiguous_soft_target or soft_target_class_weights"` = `2 passed`.
- [x] 2026-07-03 smoke direct OOF patch-verifier soft target tu `pair_verifier_teacher_oof.csv` voi `AmbiguousSoftTargetAlpha=0.0`: manifest matched `9215/9215` sample-index rows, no leakage, no final_test, nhung full-val macro/class1 chi `0.8828/0.6784`, P/R class1 `0.6073/0.7682`, confusion `0->1=49`, `1->0=19`, `1->2=11`, `4->1=12`, `3->2=52`; duoi keeper va duoi gate.
- [x] 2026-07-03 XAI OOF soft-target selected12: foreground cao (`attention/grad_rollout/gradcam/rollout 0.9797/0.9723/0.9846/0.9536`), background blur/gray gan `0`, object_desaturate `0.0627`, Grad-CAM border `4/12`, rollout border `5/12`; overlays van la surface/edge + bright background/object hotspots, khong co cue moi.
- [x] Cleanup OOF soft-target reject: copy metrics/config vao `runs\xai_smoke_v8_yolof_patchoof_softtarget_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_patchoof_softtarget_reject_smoke.json`, reclaimed `190.44 MB`.
- [ ] Khong lap lai direct full-prob OOF patch-verifier soft target `AmbiguousSoftTargetManifest=pair_verifier_teacher_oof.csv/AmbiguousSoftTargetAlpha=0.0` tren current keeper; loader sample-index fix giu lai, nhung recipe nay gan hard label va khong tao representation/interior-surface cue moi. Neu dung OOF verifier tiep, can target reliability/case-level moi, khong distill full probability vao CE.
- [x] 2026-07-03 Cleanlab diagnostic tren OOF patch-verifier teacher `runs\cleanlab_patchoof_teacher_yolof_train_20260703`: full train `9215`, issue count `81`, dominated `0->1=70` va `1->0=11`; low self-confidence class0/class1 `134/44`.
- [ ] Khong auto-relabel hoac train strict soft-target tu Cleanlab patch-teacher artifact hien tai; distribution `0->1` trung voi direction da duoc OOF gate chung minh doc (`0` corrections, `17` harms). Giu code schema/sample-index fix de review thu cong hoac diagnostic.
- [x] 2026-07-03 them `--verifier-model extra_trees` cho pairwise/patch verifier probes; preflight pass `py_compile` + focused pytest `18 passed`.
- [x] 2026-07-03 probe ExtraTrees patch-evidence `pair=0-1/top_k=4/t=0.60/margin=0.40/leaf=5/trees=300`: train OOF local acc tang `0.9331` nhung val base `0.8841/0.6841` -> verified `0.8812/0.6744`, changed `7` voi `1` correction, `5` harms, `1` neutral; reject.
- [x] 2026-07-03 XAI ExtraTrees changed7: `near_tie_top2=7/7`, register divergence `7/7`, foreground van cao (`attention/gradcam 0.9449/0.9710`), Grad-CAM border `0.3507`; nonlinear verifier chi phong dai ranh gioi 0-1 mong manh.
- [x] Cleanup ExtraTrees reject: copy summary/metrics/changed cases vao `runs\diagnostic_patch_evidence_extratrees_leaf5_reject_20260703`, xoa full probe dir; manifest `runs\cleanup_manifest_20260703_patch_evidence_extratrees_leaf5_reject_probe.json`, reclaimed `4.46 MB`.
- [ ] Khong lap lai `probe_patch_evidence_mil --verifier-model extra_trees --extra-trees-min-samples-leaf 5 --extra-trees-n-estimators 300` tren current keeper; OOF train dep hon nhung val harm class1. Neu quay lai nonlinear verifier, can representation/OOF reliability target moi va recall protection, khong chi doi classifier tren frozen feature.
- [x] 2026-07-03 tra cuu CYFLOD SmoothStep loss damping cho noisy-label fine-grained classification; trien khai helper/config/CLI/V8 launcher va preflight pass `py_compile train.py/config.py` + `tests\test_self_paced_loss.py` = `4 passed`.
- [x] 2026-07-03 smoke CYFLOD `weight=0.35/delta=0.25/cycle=2/min_weight=0.10/teacherfocusbinary015/BBoxSpatialFusion/crop_bbox/LR=1e-5/24b/1e`: damping active (`fraction=0.0833`, damped weight mean `0.3129`) nhung full-val macro/class1 chi `0.8795/0.6685`; class1 P/R `0.5769/0.7947`, voi `0->1=55`, `4->1=17`, `2->1=12`.
- [x] 2026-07-03 XAI CYFLOD selected12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.8964/0.9555/0.9487/0.9145`), background blur/gray gan `0`, object_desaturate lon hon, rollout border `8/12`, Grad-CAM border `6/12`, near-tie `6/12`; loi do mo rong class1 FP, khong phai do thieu context.
- [x] Cleanup CYFLOD reject: copy metrics/config/history vao `runs\xai_smoke_v8_yolof_cyflod_damp025_w035_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_cyflod_damp025_w035_rejected_smoke.json`, reclaimed `185.99 MB`.
- [ ] Khong lap lai CYFLOD global loss damping exact setting `weight=0.35/delta=0.25/cycle=2/min_weight=0.10/start=1` tren current keeper; no tang recall class1 bang cach them false positives. Neu revisit CYFLOD, can class-pair/OOF false-positive protection, khong global high-loss damping.
- [x] 2026-07-03 trien khai Register Attention Alignment loss tren last-block attention: align CLS/register patch attention + bbox foreground mass, co CLI/config/V8/history va focused tests.
- [x] 2026-07-03 smoke RegisterAttentionAlignment `weight=0.010/classes=0,1,2,4/bbox_margin=0/foreground=0.35/agreement=1.0/teacherfocusbinary015/BBoxSpatialFusion/crop_bbox/LR=1e-5/24b/1e`: loss active (`0.1108`) nhung full-val macro/class1 chi `0.8781/0.6648`; class1 P/R `0.5714/0.7947`, voi `0->1=55`, `4->1=17`, `2->1=13`.
- [x] 2026-07-03 XAI RegisterAttentionAlignment selected12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9083/0.9579/0.9586/0.9229`), background blur/gray gan `0`, object_desaturate lon hon, `near_tie=6/12`, rollout border `8/12`, va `cls_register_heatmap_similarity=0.9995`; CLS/register disagreement khong phai nut that hien tai.
- [x] Cleanup RegisterAttentionAlignment reject: copy metrics/config/history vao `runs\xai_smoke_v8_yolof_regattn_align_w010_fg035_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_regattn_align_w010_rejected_smoke.json`, reclaimed `177.54 MB`.
- [ ] Khong lap lai Register Attention Alignment exact setting `weight=0.010/classes=0,1,2,4/bbox_margin=0/foreground_weight=0.35/agreement_weight=1.0/start=1` tren current keeper; no ep them foreground/bbox attention va lam class1 false-positive rong hon. Neu revisit attention, can signal reliability/interior-surface moi, khong chi align CLS/register hoac day mass vao bbox.
- [x] 2026-07-03 them `trkh.tools.build_yolo_source_group_sample_weights`: train-only sample-index manifest tu YOLO source groups, tu choi non-train split; preflight pass `py_compile` + pytest source-group/sample-weight tests (`3 passed`).
- [x] 2026-07-03 manifest source-group `mixed_label_weight=0.65/mixed_class1=0.85/same_label_multi=0.85/same_label_class4=0.75`: `1842/9215` rows weighted, mean train weight `0.9524`, class weighted `0:234,1:139,2:271,3:37,4:1161`.
- [x] 2026-07-03 smoke source-group sample weights `60b/1e`: sample weights active/matched `1842`, nhung full-val macro/class1 chi `0.8779/0.6629`, class1 P/R `0.5756/0.7815`; confusion `0->1=54`, `4->1=16`, `2->1=13`, `1->0=15`, `3->2=58`.
- [x] 2026-07-03 XAI source-group selected12: foreground van cao (`attention/gradcam 0.9156/0.9538`), background blur/gray gan `0`, object_desaturate lon hon, near-tie `6/12`, rollout border `7/12`, Grad-CAM border `6/12`; downweight source multi/mixed khong tao cue moi.
- [x] 2026-07-03 tra cuu noisy-label FGVC + MIL/TransFG, sau do test mot truc khac: internal SSL VICReg warmup tren train-only `yolo_f` object crops thay vi tiep tuc attention/background forcing.
- [x] 2026-07-03 preflight internal SSL pass: `py_compile pretrain_internal_barlow.py train.py config.py model.py`, pytest internal SSL (`13 passed`), dry-run xac nhan explicit `yolo_f`, train-only `9215` objects, balanced sampler, CUDA, va class counts `[1941,541,1920,2520,2293]`.
- [x] 2026-07-03 pretrain `internal_vicreg_yolof_objectcrop_128b_1e`: 1 epoch / 128 train batches, best SSL loss `33.4361`; fine-tune smoke tu checkpoint nay voi V8 keeper recipe `120b/1e`, `BBoxSpatialFusion=true`, `crop_bbox`, teacher-focus-binary `0.015`, no hard manifest, full-val support `2606`, `SkipFinalTest`.
- [x] 2026-07-03 smoke internal VICReg reject: full-val macro/class1 chi `0.7435/0.3801`, class1 P/R `0.3403/0.4305`, confusion class1 `1->[53,65,22,0,11]`; qua thap nen khong probe.
- [x] 2026-07-03 XAI internal VICReg selected12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9662/0.9445/0.9569/0.9340`), background blur/gray gan `0`, object_desaturate lon hon (`0.0592`), rollout border `10/12`, register diffuse `12/12`; van la surface/boundary separability, khong phai context/localization.
- [x] Cleanup internal VICReg reject: copy SSL/fine-tune metrics/config/history vao `runs\xai_smoke_v8_yolof_internalvicreg128b1e_ft120b_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir + SSL pretrain dir + dry-run; manifest `runs\cleanup_manifest_20260703_internalvicreg_yolof_rejected_smoke.json`, reclaimed `365.84 MB`.
- [x] 2026-07-03 trien khai optional `focus_auc_rank` loss cho class1 vs negatives `0,2,4`: config/CLI/V8 launcher/history telemetry, pairwise softplus margin tren score `logit1-logsumexp(logits[0,2,4])`; preflight pass `py_compile train.py/config.py` + focused pytest (`12 passed`).
- [x] 2026-07-03 sua telemetry bug `focus_auc_rank_*`: `loss_details` da co key nhung `train_loss_components` chua accumulate nen history ve 0. Active-check 1 batch sau fix co `loss=0.00317`, `7` positives, `19` negatives, `133` pairs.
- [x] 2026-07-03 smoke `FocusAucRankLossWeight=0.004/class=1/negatives=0,2,4/margin=0.04/temp=0.12/hard_fraction=0.50/LR=1e-5/120b/1e`: loss active (`0.0107`, `122.8` pairs/batch) nhung full-val macro/class1 chi `0.8772/0.6611`, P/R class1 `0.5728/0.7815`, confusion `0->1=57`, `4->1=15`, `2->1=13`; reject truoc probe.
- [x] 2026-07-03 XAI focus_auc_rank selected12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9817/0.9739/0.9777/0.9539`), background blur/gray gan `0`, object_desaturate lon hon `0.0808`, Grad-CAM border `6/12`, rollout border `5/12`, `cls_register_heatmap_similarity=0.9997`; loss ranking lam rong class1 FP, khong giai quyet surface/boundary reliability.
- [x] Cleanup focus_auc_rank reject: copy metrics/config/history vao `runs\xai_smoke_v8_yolof_focusaucrank_w004_h50_teacherfocusbinary015_120b_1e_active_val_selected12_20260703\source_smoke_metrics`, xoa old inactive smoke + active-checks + active full smoke; manifest `runs\cleanup_manifest_20260703_focusaucrank_rejected_smokes.json`, reclaimed `727.80 MB`.
- [ ] Khong lap lai plain focus AUC-rank exact setting `weight=0.004/class=1/negatives=0,2,4/margin=0.04/temp=0.12/hard_fraction=0.50/teacherfocusbinary015/BBoxSpatialFusion/crop_bbox/LR=1e-5/120b/1e` tren current keeper; one-vs-rest ranking day class1 FP rong hon. Neu revisit ranking/AUC, phai co asymmetric FP protection hoac OOF/case-level reliability target.
- [x] 2026-07-03 preflight surface-detail attention view tren current keeper: explicit `yolo_f`, `SkipFinalTest=true`, hard manifest rong, `AttentionViewScoreSource=surface_detail`, loss `0.30`, crop/drop `0.40/0.20`, foreground weight `0.80`.
- [x] 2026-07-03 smoke surface-detail attention `60b/1e`: attention-view loss active (`1.1368`, crop/drop fraction `0.3901/0.2057`) nhung full-val macro/class1 chi `0.8786/0.6667`, P/R class1 `0.5777/0.7881`, confusion `0->1=54`, `4->1=16`, `2->1=13`; duoi keeper/gate.
- [x] 2026-07-03 XAI surface-detail attention selected12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9815/0.9733/0.9775/0.9534`), background blur/gray gan `0`, object_desaturate lon hon `0.0789`, Grad-CAM border `6/12`, rollout border `5/12`, object-color flags `4/12`; khong tao cue class1 moi.
- [x] Cleanup surface-detail attention reject: copy metrics/config/history vao `runs\xai_smoke_v8_yolof_surfaceattention_score_currentkeeper_60b_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_surfaceattention_currentkeeper_rejected_smoke.json`, reclaimed `182.21 MB`.
- [ ] Khong lap lai current-keeper surface-detail attention-view exact setting `AttentionViewScoreSource=surface_detail/loss=0.30/crop=0.40/drop=0.20/fg=0.80/teacherfocusbinary015/BBoxSpatialFusion/crop_bbox/LR=1e-5/60b-120b`; deterministic surface-detail crop/drop van tang FP class1 va khong giai quyet boundary reliability.
- [ ] Khong lap lai exact recipe `internal_vicreg_yolof_objectcrop 128b/1e + V8 fine-tune 120b/1e` tren current anchor; SSL tu dau qua lanh va lam sap class1. Neu revisit SSL, can lich pretrain/target khac co bang chung tao signal surface/boundary truoc khi probe dai.
- [x] Cleanup source-group reject: copy metrics/config/history vao `runs\xai_smoke_v8_yolof_sourcegroup_w_mixed065_c1p085_same085_c4same075_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_sourcegroup_sampleweight_rejected_smoke.json`, reclaimed `181.86 MB`.
- [ ] Khong lap lai source-group sample-weight exact policy `mixed=0.65/mixed_class1=0.85/same_multi=0.85/same_class4=0.75/SampleWeightFactor=1/Max=1` tren current keeper; blanket downweight multi/mixed source groups lam mat evidence va tang FP class1. Neu revisit source groups, can case-level reliability/group objective co recall-protection, khong chi hạ weight theo source layout.
- [x] 2026-07-03 tra cuu class-aware noisy-label selection/reweighting (IJCAI 2024 LT noisy sample selection + balanced loss, CBS, SED) va trien khai optional `self_paced_loss_class_balanced` cho self-paced loss: config/CLI/V8/history telemetry, per-class loss normalization trong batch; preflight pass `py_compile train.py/config.py` + `pytest tests\test_self_paced_loss.py -q` (`5 passed`).
- [x] 2026-07-03 smoke class-balanced self-paced `weight=0.18/percentile=0.70/gamma=0.75/min_weight=0.60/class_balanced=true/teacherfocusbinary015/BBoxSpatialFusion/crop_bbox/LR=1e-5/60b/1e`: loss active (`selected_fraction=0.6875`, `class_count=5`, `normalized_loss=0.3026`) nhung full-val macro/class1 chi `0.8779/0.6629`, P/R class1 `0.5756/0.7815`; confusion `0->1=54`, `4->1=16`, `2->1=13`, `1->0=15`, `1->2=12`.
- [x] 2026-07-03 XAI class-balanced self-paced selected12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9814/0.9731/0.9776/0.9534`), background blur/gray gan `0`, object_desaturate lon hon `0.0783`, Grad-CAM border `7/12`, rollout border `5/12`, object-color flags `4/12`; overlay `4->1` va `1->0` van la surface/edge/border ambiguity, khong phai thieu context.
- [x] Cleanup class-balanced self-paced reject: copy metrics/config/history/logs/trace vao `runs\xai_smoke_v8_yolof_cbselfpaced_w018_p70_g075_min60_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir + copied logs; manifest `runs\cleanup_manifest_20260703_cbselfpaced_rejected_smoke.json`, reclaimed `182.30 MB`.
- [ ] Khong lap lai class-balanced self-paced exact recipe `self_paced_loss_weight=0.18/percentile=0.70/gamma=0.75/min_weight=0.60/class_balanced=true/teacherfocusbinary015/BBoxSpatialFusion/crop_bbox/LR=1e-5/60b-120b` tren current keeper; class-wise normalization van lam rong FP class1. Neu revisit sample selection, can OOF/case-level reliability target va recall protection.
- [x] Huong tiep theo co gia tri: thiet ke pAUC/FPR-range false-positive guard that su khac plain `focus_auc_rank`, dua tren NeurIPS 2022/ICML 2022 partial AUC. Chi phat top-FPR negative `0/2/4 -> 1`, co positive recall-protection cho class1, smoke-gated va `SkipFinalTest`.
- [x] 2026-07-03 trien khai optional `focus_partial_auc` guard: config/CLI/V8 launcher/resume override/history telemetry + tests; preflight pass `py_compile train.py/config.py` va focused pytest `13 passed`.
- [x] 2026-07-03 smoke pAUC `weight=0.004/class=1/neg=0,2,4/margin=0.04/temp=0.12/nf=0.25/pf=0.30/pw=0.45/minp=0.05/teacherfocusbinary015/BBoxSpatialFusion/crop_bbox/LR=1e-5/60b/1e`: loss active (`0.0957`, neg/pos `0.0727/0.0510`) nhung full-val macro/class1 chi `0.8776/0.6629`, P/R class1 `0.5756/0.7815`; confusion `0->1=54`, `4->1=16`, `2->1=13`, `1->0=15`, `1->2=12`.
- [x] 2026-07-03 XAI pAUC selected12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9814/0.9731/0.9776/0.9534`), background blur/gray gan `0.0007`, object_desaturate lon hon `0.0783`, Grad-CAM border `7/12`, rollout border `5/12`, object-color flags `4/12`; van la surface/edge/severity ambiguity, khong phai thieu context nen rong.
- [x] Cleanup pAUC reject: copy metrics/config/history/logs/trace vao `runs\xai_smoke_v8_yolof_focuspartialauc_w004_nf25_pf30_pw045_min05_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_focuspartialauc_rejected_smoke.json`, reclaimed `182.56 MB`.
- [ ] Khong lap lai pAUC exact recipe `focus_partial_auc_loss_weight=0.004/class=1/negative_classes=0,2,4/margin=0.04/temp=0.12/negative_fraction=0.25/positive_fraction=0.30/positive_weight=0.45/min_negative_probability=0.05` tren current keeper; no van giam precision class1 nhu cac loss-level guard khac.
- [x] Huong tiep theo co gia tri: quay lai signal gan-gate `0-1` patch-evidence nhung khac cac attempt da reject. Da expose patch-router trong V8 va smoke coupled full-model update, khong freeze router-only.
- [x] 2026-07-03 V8 launcher expose `PatchEvidenceRouter*` flags va pass vao TrainArgs/resolved_config; preflight pass `py_compile train.py/config.py` va `pytest tests\test_patch_evidence_router.py tests\test_focus_partial_auc_loss.py -q` (`7 passed`).
- [x] 2026-07-03 smoke coupled patch-router `pair=0-1/hidden=128/top_k=4/bbox=0.75/dropout=0.05/logit_scale=0.12/routing=true/route_margin=0.30/min_pair_prob=0.04/loss=0.012/pos_weight=1.20/full-model/LR=1e-5/60b/1e`: router loss active nhung pos/neg logit margin gan `0` (`-1.1e-6`, prob `0.49997`), full-val macro/class1 chi `0.8788/0.6648`, P/R class1 `0.5784/0.7815`; confusion `0->1=53`, `4->1=16`, `2->1=13`, `1->0=15`, `1->2=12`.
- [x] 2026-07-03 XAI coupled patch-router selected12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9815/0.9732/0.9775/0.9534`), background blur/gray gan `0.0005`, object_desaturate lon hon `0.0783`, Grad-CAM border `7/12`, rollout border `5/12`, object-color flags `4/12`; overlay `4->1` va `1->0` van la surface/edge/severity cue, khong phai thieu context nen rong.
- [x] Cleanup coupled patch-router reject: copy metrics/config/history/logs/trace vao `runs\xai_smoke_v8_yolof_patchrouter_coupled_w012_pos12_top4_lr1e5_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_patchrouter_coupled_rejected_smoke.json`, reclaimed `186.31 MB`.
- [ ] Khong lap lai coupled patch-router exact recipe `PatchEvidenceRouterHead=true/pair=0-1/hidden=128/top_k=4/bbox_weight=0.75/dropout=0.05/logit_scale=0.12/routing=true/route_margin=0.30/min_pair_prob=0.04/loss_weight=0.012/positive_weight=1.20/full-model/LR=1e-5/60b-120b` tren current keeper; router zero-start khong tach pos/neg. Neu patch-evidence tiep, can non-zero reliability/selector init hoac fold-safe instance target manh hon, khong them BCE/router variant gan recipe nay.
- [x] Huong tiep theo co gia tri: inspect trace/router weights va thiet ke warm-start/reliability selector cho `0-1` patch evidence tu train-only/OOF artifacts, co recall protection cho true class1; smoke-gate bang full-val `2606` va XAI truoc khi probe.
- [x] 2026-07-03 them optional `PatchEvidenceRouterMarginPriorMode/Scale` (`none|weighted|max|topk_mean|weighted_minus_mean`) vao model/config/CLI/V8/trace; default off checkpoint-safe. Preflight pass `py_compile model.py/train.py/config.py` va `pytest tests\test_patch_evidence_router.py tests\test_trainable_module_prefixes.py -q` (`10 passed`).
- [x] 2026-07-03 smoke router max-prior `mode=max/scale=1.0/logit_scale=0.20/loss=0.006/pos_weight=1.10/full-model/LR=1e-5/60b/1e`: router non-zero hon (`pos/neg logits 0.3524/0.1110`, margin `0.2414`, prob `0.5558`) nhung full-val macro/class1 chi `0.8794/0.6685`, P/R class1 `0.5701/0.8079`; confusion `0->1=58`, `4->1=16`, `2->1=14`, `1->0=11`, `1->2=12`.
- [x] 2026-07-03 XAI router max-prior selected12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9789/0.9703/0.9757/0.9466`), background blur/gray gan `0.001`, object_desaturate tang `0.1016`, Grad-CAM border `5/12`, rollout border `5/12`, object-color flags `5/12`; top confusion doi sang `0->1=58`.
- [x] Cleanup router max-prior reject: copy metrics/config/history/logs/trace vao `runs\xai_smoke_v8_yolof_patchrouter_maxprior_s100_w006_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_patchrouter_maxprior_rejected_smoke.json`, reclaimed `186.34 MB`.
- [ ] Khong lap lai router max-prior exact recipe `PatchEvidenceRouterMarginPriorMode=max/Scale=1.0/logit_scale=0.20/loss_weight=0.006/positive_weight=1.10/full-model/LR=1e-5/60b-120b`; max-instance prior tang recall bang cach mo rong FP class1. Neu dung prior tiep, can conservative reliability/OOF suppression signal, khong raw max margin.
- [x] Huong tiep theo co gia tri: thay vi raw margin prior, tao train-only/OOF reliability target cho router theo huong conservative false-positive suppression co recall-protection; uu tien diagnostic khong train truoc, sau do smoke nho neu co bang chung giam `0->1/4->1` ma khong tang `1->0/1->2`.
- [x] 2026-07-03 trien khai OOF-reliability patch-router teacher target rieng cho `PatchEvidenceRouterHead`: doc `pair_verifier_teacher_oof.csv` vao metadata `patch_router_teacher_probs`, filter theo hard pair `0-1`, teacher agreement, pair mass/confidence, khong relabel disagreement cases; CLI/config/V8/history/tests default-off. Preflight pass `py_compile dataset.py/train.py/config.py`, PowerShell parse V8, va focused pytest `14 passed`.
- [x] 2026-07-03 smoke OOF-router teacher `min_conf=0.60/min_pair_mass=0.20/teacher_loss=0.010/positive_weight=2.5/hard_router_loss=0/full-model/LR=1e-5/60b/1e`: teacher loss active (`0.6914`, fraction `0.3656`, pair mass `0.9997`, confidence `0.9718`) nhung router prob van `0.49996`, full-val macro/class1 chi `0.8769/0.6592`, P/R class1 `0.5735/0.7748`; confusion `0->1=56`, `4->1=15`, `2->1=13`, `1->0=16`, `1->2=12`.
- [x] 2026-07-03 XAI OOF-router teacher selected12: foreground cao (`attention/grad_rollout/gradcam/rollout 0.9815/0.9766/0.9786/0.9549`), background blur/gray gan `0`, object_desaturate `0.0752`, Grad-CAM border `6/12`, rollout border `5/12`, object-color flags `4/12`; overlay van la surface/boundary cue, khong phai thieu context.
- [x] Cleanup OOF-router teacher reject: copy metrics/config/history/logs/trace vao `runs\xai_smoke_v8_yolof_patchrouter_oofteacher_conf60_pw25_w010_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_patchrouter_oofteacher_rejected_smoke.json`, reclaimed `186.43 MB`.
- [ ] Khong lap lai exact recipe `PatchEvidenceRouterTeacherCsv=pair_verifier_teacher_oof.csv/min_conf=0.60/min_pair_mass=0.20/teacher_loss_weight=0.010/teacher_positive_weight=2.5/hard_router_loss=0/full-model/LR=1e-5/60b-120b`; target OOF active nhung selector khong tach va class1 F1 giam. Neu tiep patch-router, can doi optimization/parameterization hoac case-level conservative signal, khong chi them teacher BCE.
- [x] Huong tiep theo co gia tri: test compression cua OOF teacher bang router-only/high-LR diagnostic rat ngan hoac thay parameterization selector sao cho non-zero signal hoc duoc ma khong lam doi backbone; chi probe neu full-val class1 tien sat/vuot `0.70` va XAI khong tang FP border/background.
- [x] 2026-07-03 smoke OOF-router teacher head-only `TrainableModulePrefixes=patch_evidence_router_head/LR=5e-4/teacher_loss=0.020/min_conf=0.60/min_pair_mass=0.20/positive_weight=2.5/24b/1e`: 166,920 params trainable, teacher loss active (`0.6999`, target `0.4546`, positive fraction `0.4682`), router prob moved to `0.4847` nhung full-val macro/class1 chi `0.8796/0.6685`, P/R class1 `0.5769/0.7947`; confusion `0->1=55`, `4->1=17`, `3->2=58`, `1->0=14`, `1->2=11`.
- [x] 2026-07-03 XAI OOF-router teacher head-only selected12: attention/rollout foreground giam (`attention/gradcam/rollout 0.9300/0.9638/0.9223`), background flags xuat hien, background perturb van gan `0`, object_desaturate `0.0763`, Grad-CAM/rollout border `6/12`; reject truoc probe.
- [x] Cleanup OOF-router head-only reject: copy metrics/config/history/logs/trace vao `runs\xai_smoke_v8_yolof_patchrouter_oofteacher_routeronly_lr5e4_w020_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_patchrouter_oofteacher_routeronly_rejected_smoke.json`, reclaimed `131.12 MB`.
- [ ] Khong lap lai exact recipe `PatchEvidenceRouterTeacherCsv=pair_verifier_teacher_oof.csv/min_conf=0.60/min_pair_mass=0.20/teacher_loss_weight=0.020/teacher_positive_weight=2.5/hard_router_loss=0/router-only/LR=5e-4/24b-120b`; selector co hoc nhung lam rong FP class1 va XAI loang nen hon. Tam dung patch-router neu khong co signal OOF/case-level moi.
- [ ] Huong tiep theo co gia tri: roi patch-router, khai thac complementary representation tu AIDT/top5 theo case-level conservative rule hoac feature-space diagnostic train-only/val-only khong tune test; uu tien tim signal giam `0/4->1` ma khong mat recall class1.
- [x] 2026-07-03 val-only AIDT/top-5/TRKH consensus diagnostic `runs\aidt_top5_consensus_val_diagnostic_20260703`: best `combined_consensus/sup_conf=0.70/sup_p1=0.15/res_conf=0.80/trkh_max_conf=0.30` reached val macro/class1 `0.8985/0.7319`, P/R class1 `0.6988/0.7682`, with `38` changes (`32` corrections, `6` harms), mostly suppressing non-1 consensus rows.
- [ ] Khong dung consensus AIDT/top-5/TRKH val-threshold router lam final/no-pretrain path va khong tune them threshold tren test. Diagnostic nay xac nhan pretrained complementarity that, nhung van duoi top-5 standalone va can pretrained inference; neu transfer tiep phai fold-safe/trainable, khong direct KD tu top-5 train cache vi train agreement hard-label `0.9998`.
- [x] 2026-07-03 tra cuu robust noisy-label losses SCE/EDL/bi-tempered va trien khai default-off `SymmetricCrossEntropyLoss` voi hard/soft targets, label smoothing, class multipliers, `per_sample_loss`; expose config/CLI/V8/resolved-config. Preflight pass `py_compile losses.py/train.py/config.py`, PowerShell parse V8, va focused pytest `14 passed`.
- [x] 2026-07-03 smoke SCE nhe `ClassificationLoss=symmetric_cross_entropy/SymmetricCeAlpha=0.10/SymmetricCeBeta=0.10/SymmetricCeEpsilon=1e-4/teacherfocusbinary015/BBoxSpatialFusion=true/LR=1e-5/60b/1e`: full-val macro/class1 chi `0.8785/0.6648`, P/R class1 `0.5749/0.7881`, confusion `0->1=56`, `4->1=16`, `2->1=13`, `1->0=14`, `1->2=12`; reject truoc probe.
- [x] 2026-07-03 XAI SCE selected12: foreground cao (`attention/grad_rollout/gradcam/rollout 0.9815/0.9731/0.9778/0.9534`), background blur/gray gan `0` (`0.0004/0.0006`), object_desaturate lon hon `0.0813`, Grad-CAM border `7/12`, rollout border `5/12`, object-color flags `4/12`; van la surface/border/severity ambiguity va loss-level robust term lam rong FP class1.
- [x] Cleanup SCE reject: copy metrics/config/history/logs vao `runs\xai_smoke_v8_yolof_sce_a010_b010_teacherfocusbinary015_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir + failed launch logs; manifest `runs\cleanup_manifest_20260703_sce_a010_b010_rejected_smoke.json`, reclaimed `182.49 MB`.
- [ ] Khong lap lai SCE exact setting `ClassificationLoss=symmetric_cross_entropy/alpha=0.10/beta=0.10/epsilon=1e-4` tren current keeper; no giong cac global robust/loss-damping recipe khac va khong tao cue surface/boundary moi. Neu revisit robust loss, can class-pair/OOF/case-level FP-protection, khong chi tang beta/alpha.
- [x] 2026-07-03 tra cuu FixRes/multi-scale FGVC va chay diagnostic override image-size `320` tren current keeper, explicit `yolo_f/val`, full support `2606`: macro/class1 chi `0.8805/0.6667`; `0->1` giam `50->45` nhung class1 recall giam `117->113`, nen reject high-res train/probe.
- [x] 2026-07-03 diagnostic probability blend 256+320: best macro `w320=0.70` dat `0.8859/0.6806`, best class1 `w320=0.80` dat `0.8858/0.6826`, van duoi keeper/gate; khong dung test.
- [x] 2026-07-03 multi-scale `w320=0.80` + frozen train-only patch verifier `0-1/t=0.60/margin=0.40` dat val `0.8895/0.7015`, changed `27` (`13` corrections, `9` harms, `5` neutral), nhung thap hon yolo-only patch verifier cu `0.8892/0.7073`; khong spend test budget.
- [x] Cleanup override-320 eval reject: compact metrics vao `runs\diagnostic_yolof_multiscale_256_320_val_20260703`, giu `runs\diagnostic_yolof_multiscale320_patchverifier_val_20260703`, xoa raw eval dir; manifest `runs\cleanup_manifest_20260703_override320_rejected_eval.json`, reclaimed `2.99 MB`.
- [ ] Khong lap lai high-res inference/train/probe `OverrideImageSize=320` hoac multi-scale 256+320 simple probability blend tren current keeper. Neu revisit resolution, can co train-time FixRes schedule/representation signal moi va smoke phai giam `1->0/1->2`, khong chi giam `0->1`.
- [x] 2026-07-03 smoke FixRes-style 320 adaptation chi train `head,pos_embed,cls_token,register_tokens,bbox_spatial_fusion_head` (`107,822` params), explicit `yolo_f`, `BBoxSpatialFusion=true`, `crop_bbox`, teacher-focus-binary `0.015`, boundary-band bbox dropout, `60b/1e`, full-val `2606`, `SkipFinalTest`, hard manifest rong: macro/class1 chi `0.8780/0.6667`, P/R class1 `0.5850/0.7748`; confusion `4->1=21`, `1->2=14`, `3->2=63`, duoi keeper/gate.
- [x] 2026-07-03 XAI FixRes 320 head/pos selected12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9173/0.9589/0.9663/0.9269`), background blur/gray gan `0`, object_desaturate `0.0937`, Grad-CAM border `6/12`, rollout border `5/12`, object-color flags `4/12`; resolution adaptation khong tao cue surface/boundary moi.
- [x] Cleanup FixRes 320 reject: copy metrics/config/history/logs/trace vao `runs\xai_smoke_v8_yolof_fixres320_headpos_bboxfusion_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_fixres320_headpos_bboxfusion_rejected_smoke.json`, reclaimed `130.41 MB`.
- [ ] Khong lap lai FixRes-style 320 head/pos/bbox-fusion adaptation exact recipe `ImageSize=320/TrainableModulePrefixes=head,pos_embed,cls_token,register_tokens,bbox_spatial_fusion_head/LR=5e-5/60b-120b` tren current keeper; no lam rong `4->1` va tang `1->2`. Neu revisit resolution, can representation/supervision moi hoac schedule khac co recall-protection ro rang.
- [x] 2026-07-03 tra cuu cRT/decoupled classifier re-training va LORT/logits-retargeting (`arXiv:1910.09217`, `arXiv:2403.00250`); smoke natural-prior classifier-only tren current keeper chi train `head,pairwise_margin_norm,pairwise_margin_head,bbox_spatial_fusion_head,cnn_fusion_norm,cnn_fusion_head` (`6199` params), explicit `yolo_f`, hard manifest rong, `SkipFinalTest`, `DisableBalancedEpochSampling=true`, `MetricLearningLossWeight=0`, `LR=5e-4`, `60b/1e`: macro/class1 `0.8815/0.6686`, P/R class1 `0.6000/0.7550`, confusion `0->1=49`, `1->0=19`, `1->2=12`, `4->1=14`; duoi keeper/gate.
- [x] 2026-07-03 XAI cRT natural-prior selected12: foreground cao (`attention/grad_rollout/gradcam/rollout 0.9774/0.9735/0.9776/0.9518`), background blur/gray gan `0`, object_desaturate `0.0777`, Grad-CAM border `7/12`, rollout border `6/12`, object-color flags `4/12`; van la surface/border/severity ambiguity.
- [x] Cleanup cRT natural-prior reject: copy metrics/config/history/logs/trace vao `runs\xai_smoke_v8_yolof_crt_natural_head_pair_bbox_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_crt_natural_head_pair_bbox_rejected_smoke.json`, reclaimed `127.26 MB`.
- [ ] Khong lap lai cRT natural-prior exact recipe `DisableBalancedEpochSampling=true/TrainableModulePrefixes=head,pairwise_margin_norm,pairwise_margin_head,bbox_spatial_fusion_head,cnn_fusion_norm,cnn_fusion_head/LR=5e-4/60b-120b` tren current keeper; head-only natural prior mat recall class1 va khong tao cue surface/boundary moi. Neu explore LORT/logits-retargeting, phai co objective khac voi recall-protection ro rang, smoke full-val va XAI truoc probe.
- [x] 2026-07-03 tra cuu official LOS/LORT implementation (`Thinklab-SJTU/LOS`) va xac nhan stage-2 la classifier-only CE voi label smoothing rat cao (`label_smooth=0.98`); dung path co san `ClassificationLoss=cross_entropy/LabelSmoothing=0.98` thay vi viet loss moi.
- [x] 2026-07-03 smoke LOS over-smooth cRT `LabelSmoothing=0.98/ClassificationLoss=cross_entropy/DisableBalancedEpochSampling=true/TrainableModulePrefixes=head,pairwise_margin_norm,pairwise_margin_head,bbox_spatial_fusion_head,cnn_fusion_norm,cnn_fusion_head/LR=5e-4/60b/1e`, explicit `yolo_f`, hard manifest rong, full-val `2606`, `SkipFinalTest`: macro/class1 chi `0.8800/0.6685`, P/R class1 `0.5805/0.7881`, confusion `0->1=55`, `2->1=12`, `4->1=15`, `1->0=15`, `1->2=11`; duoi keeper/gate.
- [x] 2026-07-03 XAI LOS selected12: foreground cao (`attention/grad_rollout/gradcam/rollout 0.9806/0.9744/0.9774/0.9519`), background blur/gray gan `0`, object_desaturate `0.0733`, Grad-CAM border `7/12`, rollout border `4/12`, object-color flags `3/12`, near-tie `2/12`; over-smoothing van lam rong FP class1 thay vi tao cue moi.
- [x] Cleanup LOS reject: copy metrics/config/history/logs/trace vao `runs\xai_smoke_v8_yolof_los098_crt_head_pair_bbox_val_selected12_20260703\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260703_los098_crt_head_pair_bbox_rejected_smoke.json`, reclaimed `127.27 MB`.
- [ ] Khong lap lai LOS/label-over-smooth exact recipe `ClassificationLoss=cross_entropy/LabelSmoothing=0.98/DisableBalancedEpochSampling=true/head+pair+bbox+cnn-fusion-only/LR=5e-4/60b-120b` tren current keeper; no giong cac global smoothing/loss-softening recipe va mo rong class1 false positives. Neu revisit LORT, can objective khac voi explicit FP-control va recall-protection, khong chi doi smoothing.
- [x] 2026-07-03 trien khai `trkh.tools.probe_relative_mahalanobis` cho diagnostic train-only/val-only theo DkNN/selective-classification/Mahalanobis nonconformity: fit class means/diag variance/Ledoit-Wolf covariance tren train embedding, dung guard threshold-free cho base-predicted class1, khong doc test. Preflight pass `py_compile` + focused pytest `7 passed`.
- [x] 2026-07-03 relative Mahalanobis probe current keeper: train best guard tang `0.9398/0.8157 -> 0.9459/0.8354` nhung val khong generalize. Best val `guard1_to0_diag_shared_std` chi doi `2` rows va giam nhe macro/class1 `0.8847/0.6860 -> 0.8843/0.6842`; one correction `0:1->0` va one harm `1:1->0`. Aggressive `guard1_any_ledoit_shared_l2std` doi `27` rows va roi xuong `0.8818/0.6688`.
- [x] 2026-07-03 XAI relative Mahalanobis changed2: foreground cao (`attention/grad_rollout/gradcam/rollout 0.9855/0.9903/0.9770/0.9846`), background blur/gray gan `0`, object_desaturate lon hon (`0.0327/0.0493`), Grad-CAM/rollout border flags `2/2`, near-tie `1/2`; distance uncertainty khong tach duoc true class1 boundary khoi false-positive class1 surface/border ambiguity.
- [x] Cleanup relative Mahalanobis reject: copy summary/changed-cases/predictions vao `runs\xai_relative_mahalanobis_yolof_keeper_guard1to0_diagshared_val_changed2_20260703\source_probe_metrics`, xoa probe dir; manifest `runs\cleanup_manifest_20260703_relative_mahalanobis_rejected_probe.json`, reclaimed `2.87 MB`.
- [ ] Khong lap lai exact relative-Mahalanobis class1 guard `diag/shared/class/Ledoit covariance on current frozen head embeddings + threshold-free base-pred1 switch-to-nearest/to0` tren current keeper; no overfit train va khong co recall-safe val gain. Neu dung lai uncertainty, phai ket hop signal moi co recall protection, khong chi distance-to-class manifold.
- [x] 2026-07-04 trien khai fixed-ETF/Neural-Collapse Ridge diagnostic trong `probe_embedding_prototypes`: simplex ETF targets + train-only Ridge projection, optional class-balanced weights; preflight pass `py_compile` + `tests\test_probe_embedding_prototypes.py` (`5 passed`).
- [x] 2026-07-04 ETF/Ridge probe current keeper: base val `0.8847/0.6860` van tot nhat; best ETF macro `etf_ridge_a100` chi `0.8749/0.6400` P/R class1 `0.7097/0.5828`; best balanced ETF class1 `etf_ridge_balanced_a10` chi `0.8707/0.6556`. `a100` doi `127` rows (`60` corrections, `61` harms), suppress `0:1->0=31` nhung mat true class1 `1:1->0=20`, `1:1->2=12`.
- [x] 2026-07-04 XAI ETF harm-first changed12: da so la true class1 base dung nhung ETF keo ra khoi class1; foreground cao (`attention/grad_rollout/gradcam/rollout 0.9651/0.9718/0.9499/0.9562`), background blur/gray gan `0`, Grad-CAM border `9/12`, rollout border `7/12`, near-tie `3/12`.
- [x] Cleanup ETF reject: copy summary/predictions/changed cases vao `runs\xai_embedding_etf_ridge_a100_yolof_keeper_val_changed12_20260704\source_probe_metrics`, xoa probe dir; manifest `runs\cleanup_manifest_20260704_embedding_etf_rejected_probe.json`, reclaimed `1.17 MB`.
- [ ] Khong lap lai fixed-ETF/Ridge/cosine classifier tren frozen current embedding hoac train-loop ETF head neu chua co representation/recall-protection signal moi; fixed geometry chi trade precision lay recall mat class1.
- [ ] Huong tiep theo co gia tri: tim signal representation/supervision moi that su, co kha nang bao ve true class1 boundary. Cac huong chi doi decision geometry/threshold/loss tren embedding/logit hien tai da het gia tri: Mahalanobis, ETF/Ridge, prototype/KNN/logistic, cRT/LOS, loss-level class1 guard, patch-router zero-start, va post-hoc pretrained threshold.
- [x] 2026-07-04 remap EfficientNetV2-S retrieval feature cache tu `class_f` sang `yolo_f` sample-index order: train `9215/9215`, val `2606/2606`, `feature_dim=1280`, no missing/duplicate sample-index rows; artifact `runs\efficientnetv2_s_features_yolof_train_20260704`.
- [x] 2026-07-04 smoke EfficientNetV2-S feature CRD adapter `weight=0.002/temp=0.16/proj=128/dropout=0.02/source=head/pair_mode=boundary_or_intra/pairs=0-1,1-2,1-4/60b/1e`: full-val macro/class1 chi `0.8837/0.6802`, P/R class1 `0.6062/0.7748`, confusion `3->2=52`, `0->1=50`, `1->0=18`, `1->2=12`, `4->1=12`; CRD telemetry active nhung xau (`pos_sim=0.0258`, `neg_sim=0.0413`, `top1=0.0844`).
- [x] 2026-07-04 XAI EfficientNetV2-S CRD selected12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9832/0.9685/0.9761/0.9464`), background blur/gray gan `0`, object_desaturate lon hon `0.0886`, Grad-CAM border `7/12`, rollout border `5/12`, object-color flags `4/12`; van la surface/border cue, khong phai thieu context nen rong.
- [x] Cleanup EfficientNetV2-S CRD reject: copy metrics/config/history/logs/trace vao `runs\xai_smoke_v8_yolof_effv2feature_crd_adapter_w002_val_selected12_20260704\source_smoke_metrics`, xoa smoke train dir va launch logs; manifest `runs\cleanup_manifest_20260704_effv2feature_crd_adapter_rejected_smoke.json`, reclaimed `187.14 MB`.
- [ ] Khong lap lai EfficientNetV2-S feature CRD adapter exact recipe `teacher_feature_contrastive_loss_weight=0.002/temp=0.16/proj=128/dropout=0.02/source=head/pair_mode=boundary_or_intra/pairs=0-1,1-2,1-4` tren current keeper; feature retrieval manh hon khong transfer qua adapter nhe nay. Neu feature-distill tiep, can fold-safe reliability/case-level recall protection hoac batch/objective khac, khong chi doi weight/temp/dropout.
- [ ] Huong tiep theo co gia tri: tra cuu va thiet ke signal representation/supervision moi ngoai logit/threshold/geometry, uu tien objective co explicit true-class1 recall protection va conservative false-positive control; smoke phai full-val `2606`, XAI selected cases, va cleanup sau moi reject.
- [x] 2026-07-04 tra cuu TENT/MEMO/T3A test-time adaptation va them `--save-adapted-checkpoint` vao `trkh.tools.evaluate_test_time_adaptation` de XAI duoc checkpoint sau adaptation; preflight pass `py_compile` va `tests\test_test_time_adaptation.py` (`3 passed`).
- [x] 2026-07-04 diagnostic TTA entropy `layernorm_head_bias/LR=1e-5/steps=1/selection_fraction=0.50/min_confidence=0.30`, explicit `yolo_f/val`, full support `2606`: selected `915/2606`, val macro/class1 chi `0.8841/0.6822`, P/R class1 `0.6094/0.7748`, duoi keeper `0.8847/0.6860`.
- [x] 2026-07-04 TTA changed-row audit: chi doi `3` rows (`2` corrections `0:1->0`, `0:4->0`, va `1` harm `1:1->0`), nen class1 recall bi mat va khong co gate.
- [x] 2026-07-04 XAI TTA selected12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9778/0.9697/0.9799/0.9440`), background blur/gray gan `0`, object_desaturate `0.0726`, Grad-CAM border `6/12`, rollout border `6/12`; entropy adaptation chi dich margin, khong tao cue surface/boundary moi.
- [x] Cleanup TTA reject: copy metrics/setup/predictions/changed rows/adapted checkpoint vao `runs\xai_tta_entropy_yolof_keeper_layernorm_headbias_lr1e5_conf30_frac50_val_selected12_20260704\source_tta_metrics`, xoa hai TTA diagnostic dirs; manifest `runs\cleanup_manifest_20260704_tta_entropy_rejected_diagnostic.json`, reclaimed `32.86 MB`.
- [ ] Khong lap lai TTA entropy exact config `layernorm_head_bias/LR=1e-5/steps=1/selection_fraction=0.50/min_confidence=0.30` tren current keeper; no chi doi rat it rows va harm true class1. Neu revisit TTA, can MEMO/T3A-style objective/selection moi co evidence bao ve class1, khong chi tune LR/confidence gan config nay.
- [x] 2026-07-04 trien khai T3A target-template diagnostic trong `probe_embedding_prototypes`: lay classifier templates tu checkpoint head, chon target supports entropy thap theo pseudo-label, them `--t3a-support-per-class`, `--t3a-source-weight`, `--t3a-min-confidence`; preflight pass `py_compile` va `tests\test_probe_embedding_prototypes.py` (`7 passed`).
- [x] 2026-07-04 T3A probe current keeper, explicit `yolo_f`, train `9215`, val `2606`, no test: base val `0.8847/0.6860` van tot nhat; best T3A `t3a_k32_sw32_c0` chi `0.8649/0.6322`, P/R class1 `0.5584/0.7285`. Changed `89` rows voi `20` corrections, `56` harms, `13` neutral; true class1 base-dung bi keo sang `0/2/4`, va non-class1 bi keo them sang class1 (`2:2->1=9`, `0:0->1=8`).
- [x] 2026-07-04 XAI T3A harm-first changed12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9823/0.9830/0.9821/0.9629`), background blur/gray gan `0`, object_desaturate `0.0403`, `near_tie=6/12`, `rollout_border=9/12`, `gradcam_border=4/12`; T3A la template drift tren frozen embedding, khong phai context/object separation moi.
- [x] Cleanup T3A reject: copy summary/predictions/changed cases vao `runs\xai_embedding_t3a_k32_sw32_yolof_keeper_val_changed12_20260704\source_probe_metrics`, xoa probe dir; manifest `runs\cleanup_manifest_20260704_embedding_t3a_rejected_probe.json`, reclaimed `1.21 MB`, D: free khoang `73.58 GB`.
- [ ] Khong lap lai T3A target-template sweep `support_per_class=16/32/64`, `source_weight=1/8/32`, `min_confidence=0.0` tren frozen current embedding; no harm true class1 va them class1 FP. Neu revisit TTA/prototype, phai co explicit recall protection hoac representation signal moi truoc.
- [x] 2026-07-04 tra cuu HERBS/high-temperature refinement va trien khai optional multi-granularity refinement loss: final logits detach lam teacher, KL high-temperature den aux logits o cac layer `2,5,8`; config/CLI/V8/history telemetry default-off. Preflight pass `py_compile trkh\training\train.py trkh\core\config.py`, `tests\test_multi_granularity_aux_heads.py` (`9 passed`), va dry-run xac nhan explicit `yolo_f`, keeper checkpoint, hard manifest rong, `SkipFinalTest`.
- [x] 2026-07-04 smoke HERBS-style multi-granularity high-temperature refinement `aux_layers=2,5,8/refine_weight=0.006/T=32/aux_CE=0/teacherfocusbinary015/BBoxSpatialFusion/crop_bbox/LR=1e-5/60b/1e`: refinement active (`loss=0.0798`, `count=3`, teacher entropy gan uniform `1.6094`) nhung full-val macro/class1 chi `0.8837/0.6802`, P/R class1 `0.6062/0.7748`; confusion `0->1=50`, `1->0=18`, `1->2=11`, `4->1=12`, `3->2=52`, duoi keeper `0.8847/0.6860`.
- [x] 2026-07-04 XAI HERBS-style refinement selected12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9820/0.9737/0.9865/0.9506`), background blur/gray gan `0`, object_desaturate `0.0775`, Grad-CAM border `6/12`, rollout border `6/12`, object-color flags `3/12`; refinement chi lam aux heads bat chuoc final uncertainty, khong tao cue surface/boundary/class1 moi.
- [x] Cleanup HERBS-style refinement reject: copy metrics/config/history/trace/logs vao `runs\xai_smoke_v8_yolof_multigran_htrefine_w006_t32_val_selected12_20260704\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260704_multigran_htrefine_rejected_smoke.json`, reclaimed `182.74 MB`.
- [ ] Khong lap lai HERBS-style multi-granularity refinement exact recipe `MultiGranularityAuxHeads=true/layers=2,5,8/dropout=0.08/aux_loss=0/refinement_weight=0.006/T=32/teacherfocusbinary015/BBoxSpatialFusion/crop_bbox/LR=1e-5/60b-120b` tren current keeper; teacher logits gan uniform nen chi distill uncertainty, khong cai thien class1. Neu revisit multi-granularity, can teacher/target co thong tin surface-boundary moi hoac recall-protecting class-pair signal, khong chi KL final-to-aux.
- [x] 2026-07-04 trien khai Self-Boosting CAM attention default-off theo huong self-boosting attention: tao hard-class token CAM pseudo-target tu head hien tai, train class-agnostic projection bang KL; checkpoint resume cho phep projection moi. Preflight pass `py_compile train.py/model.py/config.py`, launcher dry-run explicit `yolo_f`, va focused pytest `13 passed`.
- [x] 2026-07-04 smoke Self-Boosting CAM `loss=0.006/T=0.40/classes=all/teacherfocusbinary015/BBoxSpatialFusion/crop_bbox/attention-view=0.35/LR=1e-5/60b/1e`: loss active (`target_entropy=4.8091`, `target_peak=0.0259`, `projection_peak=0.0239`, similarity `0.5602`) nhung full-val macro/class1 chi `0.8834/0.6802`, P/R class1 `0.6062/0.7748`, confusion `3->2=52`, `0->1=50`, `1->0=18`, `4->1=12`, `1->2=11`; duoi keeper va gate.
- [x] 2026-07-04 XAI Self-Boosting CAM selected12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9832/0.9686/0.9762/0.9463`), background blur/gray gan `0`, object_desaturate `0.0887`, Grad-CAM border `7/12`, rollout border `5/12`, object-color flags `4/12`; overlay `case_012_t3_p2` nong o bright spot, fruit border va adjacent background, khong tao cue surface/boundary class1 moi.
- [x] Cleanup Self-Boosting CAM reject: copy metrics/config/history/logs/trace vao `runs\xai_smoke_v8_yolof_selfboostcam_w006_t040_val_selected12_20260704\source_smoke_metrics`, xoa smoke train dir va launch logs; manifest `runs\cleanup_manifest_20260704_selfboostcam_rejected_smoke.json`, reclaimed `182.80 MB`, D: free khoang `73.42 GB`.
- [ ] Khong lap lai Self-Boosting CAM exact recipe `SelfBoostingAttentionLossWeight=0.006/Temperature=0.40/classes=all` hoac nearby weight/temp tweaks tren current keeper; hard-label CAM target qua diffuse va lap lai current head boundary cue. Neu revisit self-boosting attention, phai dung target sparse/reliability/OOF hoac class-pair recall-protecting signal moi, khong chi KL vao CAM cua head hien tai.
- [x] 2026-07-04 trien khai CMT/LeFF-style `BlockLocalPatchMixer` default-off trong transformer blocks, expose V8/config/CLI, checkpoint resume-safe. Fix quan trong: token pruning lam patch count con `167`, nen mixer phai scatter sparse `patch_indices` ve full grid roi gather lai; pre-fix adapter-only run bi crash loss khong co grad, sau fix reproduction xac nhan logits/loss co grad va `15/15` trainable tensors co grad. Preflight pass `py_compile model.py/train.py/config.py` va `tests\test_block_local_patch_mixer.py` (`3 passed`).
- [x] 2026-07-04 smoke block-local sparsefix adapter-only `layers=6,7,8/dropout=0.02/scale=0.10/TrainableModulePrefixes=blocks.5-7.local_patch_mixer/LR=5e-4/60b/1e`, explicit `yolo_f`, full-val `2606`, `SkipFinalTest`: macro/class1 chi `0.8837/0.6822`, P/R class1 `0.6094/0.7748`, duoi keeper `0.8847/0.6860`; top confusion `3->2=52`, `0->1=49`, `1->0=18`, `4->1=12`, `1->2=11`.
- [x] 2026-07-04 XAI block-local sparsefix selected12: foreground van cao nhung attention/Grad-CAM van border/background (`attention/grad_rollout/gradcam/rollout fg 0.9105/0.9769/0.9558/0.9564`), background blur/gray gan `0`, object_desaturate `0.0919`, flags `attention_border=12/12`, `attention_background=4/12`, `gradcam_border=7/12`, `rollout_border=3/12`, `object_color_sensitive=4/12`; overlay `0->1` va `3->2` nong o mep/nen/off-object, khong tao cue surface/boundary moi.
- [x] Cleanup block-local sparsefix reject: copy metrics/config/history/logs va pre-fix crash evidence vao `runs\xai_smoke_v8_yolof_blocklocal678_sparsefix_val_selected12_20260704\source_smoke_metrics`, xoa sparsefix smoke train dir va crash run; manifest `runs\cleanup_manifest_20260704_blocklocal_sparsefix_rejected_smoke.json`, reclaimed `134.32 MB`, D: free khoang `73.35 GB`.
- [ ] Khong lap lai block-local adapter-only exact recipe `BlockLocalPatchMixer=true/layers=6,7,8/dropout=0.02/scale=0.10/TrainableModulePrefixes=blocks.5-7.local_patch_mixer/LR=5e-4/60b-120b` tren current keeper; sparse mixer da hoat dong nhung chi sua graph, khong them class1 surface/boundary signal. Neu revisit local conv, can noi vao head/patch pooling hoac co target/supervision moi co recall protection, khong chi adapter-only.
- [x] 2026-07-04 smoke block-local + part-token pairwise `BlockLocalPatchMixer layers=6,7,8/dropout=0.02/scale=0.10 + PartTokenPairwiseHead pairs=0-1,1-2,4-1,2-3/loss=0.02/part_count=4/temp=0.55/fg_power=1.2/bbox_weight=0.85/TrainableModulePrefixes=blocks.5-7.local_patch_mixer,part_token_pairwise_learner/LR=5e-4/60b/1e`, explicit `yolo_f`, full-val `2606`, `SkipFinalTest`: macro/class1 chi `0.8837/0.6822`, P/R class1 `0.6094/0.7748`, duoi keeper; confusion khong doi dang ke (`3->2=52`, `0->1=49`, `1->0=18`, `4->1=12`).
- [x] 2026-07-04 XAI block-local + part-token pairwise selected12: foreground cao nhung border/background van ro (`attention/grad_rollout/gradcam/rollout fg 0.9044/0.9722/0.9557/0.9498`, `attention_border=11/12`, `gradcam_border=7/12`, `attention_background=4/12`), background blur/gray gan `0`, object_desaturate `0.0727`; overlay `0->1` nong o surface/edge, `3->2` co Grad-CAM off-object va attention vao nen/edge.
- [x] Cleanup block-local + part-token pairwise reject: copy metrics/config/history/logs/best-val/trace vao `runs\xai_smoke_v8_yolof_blocklocal_partpair678_val_selected12_20260704\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260704_blocklocal_partpair_rejected_smoke.json`, reclaimed `136.88 MB`, D: free khoang `73.27 GB`.
- [ ] Khong lap lai combined block-local + part-token exact recipe tren current keeper; viec noi local conv vao part head van khong tao cue surface/boundary class1 moi. Neu revisit local/part token, phai co target/supervision moi co recall protection hoac selector khoi tao bang signal dang tin, khong chi ghep lai hai module da fail.
- [x] 2026-07-04 trien khai paired-view local patch-token alignment source `patch_tokens`: top-k local token soft nearest-neighbor cosine giua primary `yolo_f` va paired `class_f`, khong dung CE/KL/fusion cu; preflight pass `py_compile`, PowerShell parse, va focused pytest `11 passed`.
- [x] 2026-07-04 smoke paired token alignment `PairedViewTrain=true/yolo_f primary + class_f paired/FeatureConsistency=0.004/FeatureSource=patch_tokens/Temperature=0.12/teacherfocusbinary015/BBoxSpatialFusion/crop_bbox/60b/1e`, full-val `2606`, `SkipFinalTest`: macro/class1 `0.8850/0.6841`, P/R class1 `0.6082/0.7815`, macro nhich nho nhung class1 thap hon keeper `0.6860`; confusion `0->1=50`, `1->0=17`, `1->2=11`, `4->1=12`, `3->2=52`.
- [x] 2026-07-04 XAI paired token alignment selected12: foreground cao (`attention/grad_rollout/gradcam/rollout 0.9831/0.9680/0.9757/0.9465`) nhung `gradcam_border=9/12`, `rollout_border=5/12`, `object_color_sensitive=4/12`, background blur/gray gan `0`, object_desaturate `0.0895`; overlay cho thay `0->1` nong ca nen ben phai, `3->2` co Grad-CAM off-object/edge/background.
- [x] Cleanup paired token alignment reject: copy metrics/config/history/logs/best-val/trace vao `runs\xai_smoke_v8_yolof_pairtokenalign_classf_w004_t012_val_selected12_20260704\source_smoke_metrics`, xoa smoke train dir va launch logs; manifest `runs\cleanup_manifest_20260704_pairtokenalign_rejected_smoke.json`, reclaimed `182.70 MB`, D: free khoang `73.20 GB`.
- [ ] Khong lap lai exact paired token alignment recipe `PairedViewFeatureConsistencyWeight=0.004/FeatureSource=patch_tokens/Temperature=0.12/yolo_f primary + class_f paired/60b-120b`; local view-invariance lam tang macro nho nhung khong tao recall-safe class1 surface/boundary signal. Neu revisit paired data, can la objective co target reliability/OOF hoac cross-view attention/selector co gate class1, khong chi local token cosine.
- [x] 2026-07-04 trien khai `trkh.tools.probe_crop_margin_sensitivity`: validation-only crop-margin override cho `MangoYOLOCropDataset`, export full-val metrics/predictions/changed rows, refuse `--split test`; preflight pass `py_compile` va `tests\test_probe_crop_margin_sensitivity.py` (`3 passed`).
- [x] 2026-07-04 diagnostic crop-margin current keeper, explicit `yolo_f/val`, full support `2606`, margins `0,0.02,0.05,0.08,0.12,0.18,0.25`, baseline `0.05`: best macro/class1 van la baseline `0.8841/0.6841`; `m0.12` giam `0->1` `50->39` nhung mat recall class1 (`1->0=22`, `1->4=9`) va chi `0.6727`; `m0.18/m0.25` sap manh.
- [x] 2026-07-04 XAI margin-sensitive changed12: foreground tuong doi cao nhung border/background flags van ro (`gradcam_border=7/12`, `rollout_border=9/12`, `attention_background=3/12`), background blur/gray gan `0`, overlay nong o fruit edge/stem/hand/background; day la surface/boundary ambiguity, khong phai thieu context nen rong.
- [x] Cleanup crop-margin diagnostic duplicates: xoa `predictions_best_class1_m0p05.csv` va `predictions_best_macro_m0p05.csv`, giu summary/sweep/changed CSV/baseline predictions/XAI; manifest `runs\cleanup_manifest_20260704_cropmargin_duplicate_predictions.json`, reclaimed `2.93 MB`.
- [ ] Khong lap lai simple crop-margin inference/train/eval widening/narrowing tren current keeper. Neu context duoc revisit, phai la reliability-gated selector co explicit class1 recall protection va khong dung test de chon margin/threshold.
- [x] 2026-07-04 train-only crop-margin router diagnostic: tao full train/val predictions cho `m0.05,m0.12` tu keeper (`train=9215`, `val=2606`), chay `train_trainval_expert_router` voi `selection_source=train_oof`, `focus_class_weight=0.50`, `disable_image_features`. Train OOF chon `logreg_c0p1` (`0.9464/0.8376`) nhung validation roi xuong macro/class1 `0.8758/0.6458`; fixed average candidates on val chi khoang `0.8843/0.6845`, van duoi keeper class1 `0.6860`.
- [x] 2026-07-04 router changed-case audit: selected router doi `76` val rows (`33` corrections, `34` harms, `9` neutral), `58` rows lien quan class1; no sua `0:1->0=20` nhung lam hong true class1 base-dung `1:1->0=13` va `1:1->2=12`. Baseline class1 `0.6841` trong router order roi xuong `0.6458`.
- [x] 2026-07-04 XAI router class1 harm12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9737/0.9795/0.9750/0.9589`), background blur/gray gan `0`, `near_tie=7/12`, border flags con; overlay cho thay surface/boundary ambiguity va mot so off-object heat, khong co context cue dang tin.
- [x] Cleanup crop-margin router reject: xoa full prediction CSVs va rejected `router.pkl`, giu summary/sweep/router metrics/changed summary/XAI; manifest `runs\cleanup_manifest_20260704_cropmargin_router_rejected_intermediates.json`, reclaimed `28.39 MB`.
- [ ] Khong lap lai two-expert crop-margin router `m0.05/m0.12` bang probability/logit features tren current keeper, ke ca logistic/tree/router candidate sweep gan do; train OOF overfit va khong recall-safe cho class1. Neu context tiep tuc, can signal reliability moi ngoai hai margin probabilities va phai bao ve true class1 recall truoc khi dung validation gate.
- [x] 2026-07-04 trien khai `probe_bbox_token_region_separability`: train/val-only, fit decision layers tren descriptor `head`, `foreground_tokens`, `background_tokens`, `fg_minus_bg`, `fg_bg_concat`, dung `crop_bbox` token prior, refuse test; preflight pass `py_compile` va `tests\test_probe_bbox_token_region_separability.py` (`3 passed`).
- [x] 2026-07-04 diagnostic bbox-region current keeper, explicit `yolo_f`, train/val support `9215/2606`: base head val `0.8847/0.6860`; foreground-token logistic-balanced chi `0.8658/0.6427`, background-token logistic-balanced chi `0.8398/0.5833`, fg+bg concat chi `0.8438/0.5836`; background token co class signal nhung yeu va khong dang tin.
- [x] 2026-07-04 changed-case audit background-token logreg-balanced vs base: doi `245` val rows (`63` corrections, `157` harms, `25` neutral), `136` class1-related; sua `0:1->0=19` nhung gay `1:1->0=13`, `1:1->2=8`, `4:4->1=22`, `0:0->1=20`, `2:2->1=16`.
- [x] 2026-07-04 XAI bbox-region background harm12: base keeper van foreground-heavy (`attention/grad_rollout/gradcam/rollout 0.9598/0.9631/0.9446/0.9436`), background blur/gray gan `0.0005/-0.0001`, object_desaturate lon hon `0.0373`, border/background-adjacent flags con (`rollout_border=9/12`, `gradcam_background=3/12`); day khong ung ho background-token router hay background-uniform loss.
- [x] Cleanup bbox-region smoke64: xoa `runs\diagnostic_yolof_bbox_region_separability_keeper_smoke64_20260704`, giu full diagnostic va XAI; manifest `runs\cleanup_manifest_20260704_bbox_region_smoke64_deleted.json`, reclaimed `0.12 MB`.
- [ ] Khong lap lai background-token-only classifier, `fg_bg_concat` router, hoac background-region uniform/adversarial loss tren current keeper; no qua gan background-focus/background-counterfactual da reject va diagnostic cho thay background signal gay harm nhieu hon sua. Chi revisit neu mot representation moi lam background perturbation tro thanh causal ro rang.
- [ ] Huong tiep theo co gia tri: roi khoi context/background va tim signal moi co kha nang thay doi representation surface-boundary, uu tien cac target/diagnostic fold-safe OOF hoac self-supervised/part-supervision co recall-protection, khong them mot post-hoc selector tren logits hien tai.
- [x] 2026-07-04 trien khai `trkh.tools.probe_token_prune_sensitivity`: config-only diagnostic override `token_pruning/token_keep_rates`, export metrics/predictions/changed rows, refuse test by default; preflight pass `py_compile` va `tests\test_probe_token_prune_sensitivity.py` (`4 passed`).
- [x] 2026-07-04 diagnostic token-prune current keeper, explicit `yolo_f/val`, full support `2606`: baseline keep `0.85,0.65` giu `167` tokens va dat `0.8847/0.6860`; `keep95_85` giu `218` tokens chi `0.8778/0.6514`; `keep100_85` `0.8788/0.6534`; `no_prune` giu `256` tokens chi `0.8787/0.6554`.
- [x] 2026-07-04 changed-case token-prune: `no_prune` doi `49` rows (`20` corrections, `25` harms, `4` neutral), `24` lien quan class1; sua mot so `3:2->3=15` nhung gay `2:2->1=6`, `0:0->1=4`, `4:4->1=5`, va `1:1->0=3`, nen class1 precision/recall deu thua keeper.
- [x] 2026-07-04 XAI no-prune class1 harm12: foreground van cao (`attention/grad_rollout/gradcam/rollout 0.9601/0.9804/0.9383/0.9612`), background blur/gray gan `0` (`-0.0001/0.0041`), object_desaturate lon hon `0.0214`, flags `attention_border=6/12`, `rollout_border=8/12`, `gradcam_border=4/12`, `near_tie=6/12`; giu them token lam phong dai surface/border ambiguity, khong khoi phuc cue bi prune mat.
- [x] Cleanup token-prune diagnostic intermediates: xoa smoke2 va checkpoint tam `keeper_no_prune_config_only.pt`, giu full diagnostic + XAI + keeper; manifest `runs\cleanup_manifest_20260704_tokenprune_diagnostic_intermediates.json`, reclaimed `55.73 MB`.
- [ ] Khong lap lai no-prune hoac keep-rate cao hon (`0.95,0.85`, `1.0,0.85`) tren current keeper; no lam rong `0/2/4->1` false positives va khong vuot gate `0.70`. Neu token pruning duoc revisit, can target recall-safe/surface-boundary moi truoc, khong chi giu them patch tokens.
- [x] 2026-07-04 tao current boundary review cho keeper tu full `yolo_f/val` baseline: `runs\boundary_review_yolof_keeper_val_current_20260704`, `260` rows, HTML report, reason counts `FP1=75`, `FN1=33`, `high_conf_error=8`, `low_margin_error=70`, `low_margin_correct_boundary=74`; pair counts `0-1=110`, `1-2=56`, `2-3=63`, `4-rest=31`.
- [x] 2026-07-04 phat hien va sua XAI audit mismatch: old XAI render/robustness doc anh goc thay vi dataset object-crop tensor; `xai_audit` bay gio uu tien `dataset[sample_index][0]` va clean_recheck dung tensor goc. Preflight pass `py_compile` + `tests\test_xai_audit_dataset_tensor.py` (`3 passed`).
- [x] 2026-07-04 XAI corrected dataset-crop cases16: `runs\xai_boundary_review_yolof_keeper_val_current_cases16_datasetcrop_20260704`, clean recheck khop selected predictions; flags `rollout_background=10/16`, `gradcam_border=8/16`, `attention_background=7/16`, `object_color_sensitive=7/16`, `near_tie=4/16`. Background blur/gray drop chi `0.0077/0.0073`, object_desaturate drop `0.1344`, nen object surface/color van la cue chinh.
- [x] Cleanup pre-fix XAI artifact mismatch: xoa `runs\xai_boundary_review_yolof_keeper_val_current_cases16_20260704`, manifest `runs\cleanup_manifest_20260704_xai_boundary_review_mismatched_tensor_deleted.json`, reclaimed `86.34 MB`.
- [x] 2026-07-04 forensics basic full-val: `runs\forensics_yolof_keeper_val_current_basic_20260704`, class1 P/R/F1 `0.6114/0.7815/0.6860`, ECE `0.6052`, low-margin bucket `96/257` errors (`37.35%`), shadow-heavy `27/136` errors, highlight-heavy `24/344` errors; foreground-mode full forensics bi timeout va da dung process, khong giu artifact partial.
- [ ] Khong dung cac XAI artifact truoc fix dataset-crop de so sanh foreground/background mass tuyet doi neu dang ra quyet dinh moi; robustness/selection tu `xai_boundary_review_yolof_keeper_val_current_cases16_datasetcrop_20260704` tro di moi la chuan hon cho object-crop pipeline.
- [ ] Khong lap lai low-margin/logit calibration, color-stat guard, background/context, or token-prune route tu cac diagnostic moi; evidence moi chi xac nhan object-surface low-margin ambiguity. Huong tiep theo phai la train-only representation/target moi hoac fold-safe pretrained-disagreement compression, khong phai threshold/router quanh val.
- [x] 2026-07-04 trien khai train-only local-neighbor CMSF feature cache `trkh.tools.build_local_neighbor_feature_cache`: extract keeper train embedding tren explicit `yolo_f`, refuse non-train split, output NPZ theo `sample_index`, target = blend voi same-class local neighbors rank `4..24`; preflight pass `py_compile` va focused pytest `12 passed`.
- [x] 2026-07-04 sinh cache `runs\local_neighbor_feature_cache_yolof_keeper_cmsf_head_20260704`: cover `9215/9215`, feature_dim `256`, no fallback, mean cosine target-source `0.99937`, reasons `same_class=8451`, `focus_class1=541`, `boundary_predicted_focus=223`.
- [x] 2026-07-04 smoke local-neighbor CMSF CRD adapter `TeacherFeatureContrastiveLossWeight=0.002/temp=0.18/proj=128/dropout=0.02/source=head/pair_mode=boundary_or_intra/pairs=0-1,1-2,1-4,2-3`, explicit `yolo_f`, teacherfocusbinary015, BBoxSpatialFusion/crop_bbox, hard manifest rong, `60b/1e`, full-val `2606`, `SkipFinalTest`: macro/class1 chi `0.8774/0.6592`, P/R class1 `0.5700/0.7815`, confusion `0->1=57`, `4->1=16`, `1->0=15`, `1->2=12`, `3->2=57`; duoi keeper va gate.
- [x] 2026-07-04 XAI local-neighbor CMSF selected12: `object_desaturate=0.1887` trong khi background blur/gray chi `0.0106/0.0107`; flags `object_color_sensitive=10/12`, `gradcam_border=8/12`, `rollout_background=5/12`, `attention_background=4/12`; overlay cho thay FP class1 nong o surface/mep qua va FN class1 co heat o la/nen canh qua.
- [x] Cleanup local-neighbor CMSF CRD reject: copy metrics/config/history/transcript/trace vao `runs\xai_smoke_v8_yolof_localneighbor_cmsfcrd_w002_val_selected12_20260704\source_smoke_metrics`, giu eval/boundary/XAI/cache, xoa smoke train dir; manifest `runs\cleanup_manifest_20260704_localneighbor_cmsfcrd_rejected_smoke.json`, reclaimed `184.27 MB`.
- [ ] Khong lap lai exact local-neighbor CMSF CRD recipe `min_rank=4/max_rank=24/neighbors=6/blend=0.18/focus_blend=0.28/boundary_blend=0.22 + CRD weight=0.002/temp=0.18/proj=128/dropout=0.02/source=head/boundary_or_intra` tren current keeper; target qua gan current embedding nhung van lam rong false-positive class1.
- [ ] Huong tiep theo co gia tri: neu dung neighbor/spectral evidence nua, phai bien no thanh signal conservative co recall-protection ro rang (vi du chi ap dung cho train-side high-risk `0/2/4->1` FP suppressor co doi ung bao ve true class1), khong tiep tuc same-class smoothing/toan-bo-batch CRD. Uu tien diagnostic train-only truoc smoke, va XAI sau moi smoke/probe.
- [x] 2026-07-04 trien khai diagnostic train-only patch part-prototype `trkh.tools.probe_patch_part_prototypes` theo PARTICLE/CCT: cluster bbox/crop-bbox frozen patch tokens train-only, fit logistic readouts, refuse test by default. Preflight pass `py_compile` va `tests\test_probe_patch_part_prototypes.py` (`2 passed`).
- [x] 2026-07-04 full diagnostic part-prototype current keeper: `runs\diagnostic_yolof_part_prototypes_keeper_full_20260704`, train/val `9215/2606`, `32` prototypes, `120000` fit patch rows, `129010/36484` train/val selected patch rows. `part_hist_only` rat yeu (`val macro/class1 0.7302/0.3727`); train-OOF-selected `head_prob_part_hist C=0.05` giam xuong `0.8794/0.6667`, doi `93` rows (`41` corrections, `43` harms), sua `0:1->0=17` nhung lam hong true class1 `1:1->0=11`, `1:1->2=8`.
- [x] 2026-07-04 XAI part-prototype changed12: `runs\xai_part_prototype_headprob_changed12_yolof_keeper_20260704`; background blur/gray drop am/gan `0` (`-0.0049/-0.0090`), object_desaturate lon hon `0.0978`, flags `near_tie=6/12`, `rollout_border=9/12`, `gradcam_border=4/12`, `gradcam_background=4/12`; overlay van la object surface/edge/background-adjacent ambiguity.
- [x] Cleanup part-prototype smoke nho: xoa `runs\diagnostic_yolof_part_prototypes_keeper_smoke_20260704`, manifest `runs\cleanup_manifest_20260704_partprototype_smoke_deleted.json`.
- [ ] Khong lap lai frozen patch part-prototype readout exact route `num_prototypes=32/max_fit_patches=120000/max_patches_per_sample=14/bbox_threshold=0.05 + part_hist/head_prob_part_hist logistic` tren current keeper; no overfit train OOF va khong bao ve true class1 recall. Neu part route tiep, can selector/target duoc khoi tao bang signal moi hoac supervision khac, khong histogram/logistic tren frozen patch manifold.
- [ ] Huong tiep theo co gia tri: tam roi frozen readout/router tren current embeddings. Uu tien mot diagnostic fold-safe pretrained-disagreement/complementarity o feature-space hoac mot objective representation moi co signal khac that su; khong train them loss-level class1 guard neu no chi tac dong logits/frozen surface boundary.
- [x] 2026-07-04 trien khai `trkh.tools.build_top5_vote_disagreement_sample_weights`: doc tung expert pretrained TTA, remap strict class_f crop key sang `yolo_f` sample_index, train-only manifest, diagnostic-only cho val/non-train; preflight pass `py_compile` va `tests\test_build_top5_vote_disagreement_sample_weights.py` (`2 passed`).
- [x] 2026-07-04 manifest top5 vote-disagreement train: `runs\top5_vote_disagreement_sample_weights_yolof_train_20260704`, full train `9215`, disagreement `169`, write `109` non-class1 weighted rows, protect `60` true class1 rows; loader smoke matched all `109`, mean effective train weight `0.9980`.
- [x] 2026-07-04 val diagnostic-only top5 vote disagreement: `219` val disagreement rows, `86` keeper errors, co signal lien quan class1 nhung mixed risk (`fp1_suppress_candidate=14`, `class1_harm_risk=12`, `new_fp1_risk=9`), nen khong dung val de chon threshold/weight.
- [x] 2026-07-04 smoke top5 vote-disagreement sample weights `class1_vote=0.82/strong=0.72/majority_conflict=0.80/other=0.92 + teacherfocusbinary015/BBoxSpatialFusion/crop_bbox/60b/1e`: full-val macro/class1 roi xuong `0.8769/0.6574`, P/R class1 `0.5673/0.7815`, confusion xau hon `0->1=58`, `4->1=16`, `2->1=12`, `1->0=15`, `1->2=12`, `3->2=57`; reject truoc probe.
- [x] 2026-07-04 XAI top5 vote-disagreement selected12: background blur/gray drop rat nho `0.0070/0.0045`, object_desaturate lon `0.1471`, flags `attention_background=6/12`, `attention_border=7/12`, `gradcam_border=8/12`, `rollout_background=7/12`, `near_tie=6/12`; van la surface/border/near-tie, khong phai sample-weight uncertainty giai duoc.
- [x] Cleanup top5 vote-disagreement smoke reject: copy metrics/config/history/transcript/best-val/architecture_trace vao `runs\xai_smoke_v8_yolof_top5votedisagree_sampleweight_w082_val_selected12_20260704\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260704_top5votedisagree_sampleweight_rejected_smoke.json`, reclaimed `191.53 MB`.
- [ ] Khong lap lai exact top5 vote-disagreement sample-weight policy `class1_vote_weight=0.82/class1_vote_strong=0.72/majority_conflict=0.80/other=0.92/SampleWeightMax=1.0` tren current keeper; sparse in-sample expert disagreement van lam rong false-positive class1. Neu pretrained-disagreement tiep, can OOF/fold-safe case-level target hoac representation transfer manh hon, khong chi downweight nho cac disagreement row.
- [x] 2026-07-04 trien khai diagnostic `trkh.tools.probe_pretrained_feature_oof_readout`: train OOF readout tren pretrained feature cache, select bang train OOF, export selected train/val predictions, refuse test feature cache; preflight pass `py_compile` va `tests\test_probe_pretrained_feature_oof_readout.py` (`1 passed`).
- [x] 2026-07-04 diagnostic EfficientNetV2-S feature OOF readout `runs\pretrained_feature_oof_readout_effv2s_yolof_20260704`: train OOF selected `logreg_balanced_c0p3` macro/class1 `0.9960/0.9863` nhung validation chi `0.8876/0.6454`, P/R class1 `0.6947/0.6026`; best val class1 trong candidates chi `knn_cosine_k31=0.6691`, van duoi keeper `0.6860`.
- [ ] Khong train TRKH tu direct soft-target/KD/sample-weight dua tren selected frozen EfficientNetV2-S feature OOF classifier nay; no qua manh tren train OOF nhung khong generalize class1 val. Neu dung pretrained features tiep, can fold-held-out deep expert predictions that su hoac objective representation khac co bang chung val class1 truoc smoke.
- [x] 2026-07-04 tra cuu CORAL/CORN/cumulative-link ordinal regression va test mot route khac ordinal-distribution EMD: cumulative threshold head zero-init cho `classes=0,1,2,3`, khong dung class4, preflight pass `py_compile model.py/train.py/config.py`, V8 parse, va `tests\test_cumulative_ordinal_head.py` (`3 passed`).
- [x] 2026-07-04 smoke cumulative ordinal `CumulativeOrdinalHead=true/classes=0,1,2,3/logit_scale=0.18/dropout=0.02/loss_weight=0.006/threshold_weights=1.7,1.7,0.6/teacherfocusbinary015/BBoxSpatialFusion/crop_bbox/LR=1e-5/60b/1e`: trainer full-val macro/class1 chi `0.8762/0.6574`, P/R class1 `0.5673/0.7815`; standalone eval `0.8771/0.6592`. False-positive class1 van rong (`0->1` khoang `56-57`, `4->1=16`, `2->1=13`) nen reject truoc probe.
- [x] 2026-07-04 XAI cumulative ordinal selected12: background blur/gray drop nho `0.0121/0.0102`, object_desaturate lon hon `0.1357`, flags `gradcam_border=7/12`, `rollout_border=6/12`, `near_tie=6/12`, `object_color_sensitive=6/12`; overlay van la edge/surface/background-adjacent ambiguity, khong co cue ordinal moi.
- [x] Cleanup cumulative ordinal reject: copy metrics/config/history/transcript/best-val/architecture_trace vao `runs\xai_smoke_v8_yolof_cumulativeordinal_c0123_w006_s018_val_selected12_20260704\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260704_cumulativeordinal_rejected_smoke.json`, reclaimed `182.70 MB`.
- [ ] Khong lap lai exact cumulative ordinal route `CumulativeOrdinalHead=true/classes=0,1,2,3/logit_scale=0.18/dropout=0.02/loss_weight=0.006/threshold_weights=1.7,1.7,0.6/LR=1e-5/60b-120b` tren current keeper. Cac bien the ordinal/logit-shaping khac chi nen quay lai neu co signal OOF/reliability/representation moi, vi route nay giong cac loss-level guard va lam rong class1 FP.
- [ ] Huong tiep theo co gia tri: ngung them loss-level/logit-shaping tren keeper. Can tim signal moi that su, uu tien fold-safe pretrained disagreement, case-level conservative representation target, hoac neu chap nhan pretrained thi benchmark mot pretrained-hybrid moi tren val-only gate de xac dinh tran thuc te truoc khi tiep tuc no-pretrain compression.
- [x] 2026-07-04 tra cuu ConvNeXt (`https://arxiv.org/abs/2201.03545`) va benchmark `convnext_tiny.fb_in22k_ft_in1k` nhu diagnostic pretrained-CNN representation, khong dung test.
- [x] 2026-07-04 ConvNeXt Tiny warmup-heavy all-param smoke `80b/1e` bi underfit: full-val macro/class1 chi `0.3927/0.1911`, do warmup mac dinh lam LR qua thap trong 1 epoch; khong dung lam ket luan kien truc.
- [x] 2026-07-04 ConvNeXt Tiny head-only smoke `head/LR=1e-3/no-warmup/160b`, disable class-aware crop margin/aug, full `yolo_f/val=2606`: macro/class1 chi `0.7243/0.3261`, class1 P/R `0.2318/0.5497`, FP class1 rat lon (`0->1=122`, `4->1=101`, `2->1=47`), thua xa TRKH keeper.
- [x] 2026-07-04 sua XAI de ho tro timm-style model: `xai_audit` va `attention_viz` chi goi `forward_features(image_valid_mask/bbox_token_prior)` neu signature ho tro metadata TRKH; preflight pass `py_compile` va `tests\test_xai_audit_dataset_tensor.py` (`4 passed`).
- [x] 2026-07-04 XAI ConvNeXt head-only cases8: Grad-CAM foreground/background `0.6758/0.3242`, background blur/gray drop `0.0239/0.0077`, object_desaturate drop `0.4156`, flags `gradcam_background=4/8`, `object_color_sensitive=7/8`; overlay cho thay reliance vao pad/background/edge/surface, kem foreground hon keeper.
- [x] Cleanup ConvNeXt reject: copy metrics/config/history vao `runs\xai_smoke_timm_convnexttiny_fb22k_yolof_headonly_val_cases8_20260704\source_smoke_metrics`, xoa hai smoke train dir; manifest `runs\cleanup_manifest_20260704_convnexttiny_pretrained_rejected_smokes.json`, reclaimed `661.22 MB`.
- [ ] Khong lap lai exact ConvNeXt Tiny pretrained route `convnext_tiny.fb_in22k_ft_in1k/head-only/LR=1e-3/160b/no-warmup` hoac one-epoch all-param warmup smoke tren current data; generic pretrained CNN head khong giai duoc class1 boundary va con background/pad sensitive. Neu pretrained tiep, can stronger fold-safe expert disagreement/representation transfer hoac fine-tuning co explicit class1 FP control truoc khi dung de huong dan TRKH.
- [x] 2026-07-04 smoke ConvNeXt Tiny `stages.3+head` voi explicit class1 partial-AUC FP control `weight=0.02/margin=0.05/temp=0.12/neg_fraction=0.35/pos_fraction=0.50/pos_weight=0.70/LR=2e-4/160b`: active telemetry nhung val macro/class1 chi `0.8146/0.4595`, class1 P/R `0.3793/0.5828`, confusion `0->1=78`, `4->1=33`, `2->1=29`, `1->0=34`, `1->2=24`.
- [x] 2026-07-04 XAI ConvNeXt stage3+head cases8: attention fallback background `0.7500`, Grad-CAM foreground/background `0.8254/0.1746`, background blur/gray `0.0169/0.0056`, object_desaturate `0.5266`, flags `attention_background=6/8`, `object_color_sensitive=8/8`; overlay van nong o edge/defect/color patch va pad/background.
- [x] Cleanup ConvNeXt stage3+head reject: copy metrics vao `runs\xai_smoke_timm_convnexttiny_stage3head_pauc_val_cases8_20260704\source_smoke_metrics`, xoa smoke train dir; manifest `runs\cleanup_manifest_20260704_convnexttiny_stage3head_pauc_rejected_smoke.json`, reclaimed `342.64 MB`.
- [ ] Khong lap lai nearby `ConvNeXt Tiny stages.3+head + partial-AUC/FPR focus loss` smoke tren current data; light pretrained update co cai thien so voi head-only nhung van thua xa keeper va khong tao class1 surface cue on dinh. Neu pretrained tiep, can model/expert khac co evidence validation gan top-5/AIDT hoac fold-safe transfer, khong tiep tuc ConvNeXt Tiny local fine-tune nhe.
- [x] 2026-07-04 cap nhat `evaluate_embedding_retrieval` de doc `yolo_f` object-crop, cache `sample_index/source_stem/classes`, them adapter picklable cho Windows multiprocessing; preflight pass `py_compile` va focused pytest `tests\test_evaluate_embedding_retrieval_yolo.py tests\test_probe_pretrained_feature_oof_readout.py`.
- [x] 2026-07-04 DINOv2 direct `yolo_f` smoke `vit_small_patch14_dinov2/max_samples_per_class=3`: chi la pipeline check `15/15`, val macro/class1 `0.5603/0.5000`; full direct extraction bi tranh vi input `518x518` cham.
- [x] 2026-07-04 remap DINOv2-small `class_f -> yolo_f` strict full coverage: train `9215/9215`, val `2606/2606`, feature_dim `384`, missing `0`; artifact `runs\dinov2_small_features_yolof_remap_20260704`.
- [x] 2026-07-04 OOF readout DINOv2 remap `runs\pretrained_feature_oof_readout_dinov2small_yolof_20260704`: selected `logreg_balanced_c0p3`, train OOF macro/class1 `0.8470/0.5167`, validation chi `0.8369/0.5112`, P/R class1 `0.4439/0.6026`; confusion `0->1=80`, `1->0=26`, `1->2=27`, `2->1=21`, `4->1=9`.
- [x] 2026-07-04 complementarity DINOv2 vs keeper `runs\dinov2small_vs_keeper_complementarity_val_20260704`: DINOv2 cuu `13/34` keeper FN class1 va block dung `37/77` keeper FP class1, oracle val macro/class1 `0.9322/0.8100`, nhung DINOv2 predicted_support class1 `205` voi precision chi `0.4439`, nen unsafe/deploy khong duoc.
- [x] Cleanup DINOv2 direct smoke: copy `summary.json/metrics_val.json` vao `runs\pretrained_feature_oof_readout_dinov2small_yolof_20260704\source_yolof_smoke`, xoa `runs\embedding_retrieval_dinov2_small_yolof_smoke_20260704`; manifest `runs\cleanup_manifest_20260704_dinov2_yolof_smoke_deleted.json`, reclaimed `0.07 MB`.
- [ ] Khong lap lai DINOv2-small remapped `class_f -> yolo_f` feature OOF readout de lam direct soft-target/KD/sample-weight/router tren current keeper. Neu revisit DINO, can full `yolo_f` extraction hoac signal fold-safe moi co bang chung val class1 vuot keeper truoc smoke, khong dung validation-oracle threshold.
- [x] 2026-07-04 full direct DINOv2-small `yolo_f` extraction `runs\embedding_retrieval_dinov2_small_yolof_direct_full_20260704`: train/val `9215/2606`, feature_dim `384`, `vit_small_patch14_dinov2`, `--skip-test`. Val-selected retrieval macro/class1 `0.8867/0.6414`, van duoi keeper class1 `0.6860`.
- [x] 2026-07-04 OOF readout direct DINOv2 `runs\pretrained_feature_oof_readout_dinov2small_yolof_direct_full_20260704`: train OOF-selected `logreg_balanced_c0p3`, train OOF `0.8521/0.5263`, validation chi `0.8449/0.5260`, P/R class1 `0.4486/0.6358`, predicted_support class1 `214`, confusion `0->1=86`, `2->1=24`, `1->0=25`, `1->2=24`.
- [x] 2026-07-04 complementarity direct DINOv2 vs keeper `runs\dinov2small_direct_vs_keeper_complementarity_val_20260704`: oracle class1 `0.8254`, DINO cuu `13/34` FN1 va block dung `43/77` FP1, nhung rule diagnostic `base_pred=1 && dino_pred!=1 -> dino` sua `43` loi dong thoi lam hai `34` true-class1 dung, class1 F1 roi `0.6783 -> 0.6360`.
- [x] Cleanup DINOv2 direct cap diagnostics: xoa `runs\embedding_retrieval_dinov2_small_yolof_direct_cap10_20260704` va `runs\embedding_retrieval_dinov2_small_yolof_direct_cap50_20260704`; giu full extraction/OOF/complementarity, manifest `runs\cleanup_manifest_20260704_dinov2_direct_cap_diagnostics_deleted.json`, reclaimed `0.94 MB`.
- [ ] Khong lap lai DINOv2-small direct `yolo_f` feature readout de lam teacher/router/KD/sample-weight tren current keeper. No co oracle complementarity nhung class1 precision thap va simple FP1 suppressor khong recall-safe; neu dung DINO lai, can signal fold-safe moi bao ve true class1 truoc khi smoke TRKH.
- [x] 2026-07-04 built current `yolo_f/train` 5-fold OOF dataset artifact `runs\yolof_oof_folds_train5_20260704`: grouped by source image, `8064` images, `9215` objects, class counts `[1941,541,1920,2520,2293]`, materialized `hardlink=96768/copy=0`; raw dataset untouched.
- [x] 2026-07-04 MobileNetV3 command-check + fold00 pretrained probe: `mobilenetv3_large_100.ra_in1k`, fold00 full val `1858`, `3e`, CE/no oversampling/no class-aware crop margin. Best epoch 3 macro/class1 `0.8648/0.5389`, class1 P/R `0.6118/0.4815`; epoch2 class1 `0.5441` only via low precision `0.4512`. Far below keeper `0.8847/0.6860`, so no 5-fold expansion.
- [x] 2026-07-04 boundary/XAI MobileNetV3 fold00: boundary selected `86` rows (`FN1=30`, `FP1=26`, `high_conf_error=23`); XAI cases8 all high-confidence true-class1 misses predicted `0/2`, attention fallback all-background, Grad-CAM fg/bg/border `0.7556/0.2444/0.2878`, flags `gradcam_background=7/8`, `gradcam_border=8/8`.
- [x] 2026-07-04 fixed timm Grad-CAM hook failure by disabling `inplace=True` activation modules inside `attention_viz` before Grad-CAM/grad-rollout; preflight pass `py_compile attention_viz.py xai_audit.py` and `tests\test_xai_audit_dataset_tensor.py` (`5 passed`). Case metadata records disabled modules such as `bn1.act` and `blocks.*.act`.
- [x] Cleanup MobileNetV3 fold00 reject: copy train/eval/boundary/XAI metrics and process logs into `runs\xai_probe_timm_mobilenetv3_yolof_oof_fold00_plain_val_cases8_20260704\source_probe_metrics`; delete command-check/probe train dirs and copied logs; manifest `runs\cleanup_manifest_20260704_mobilenetv3_oof_fold00_rejected_probe.json`, reclaimed `154.97 MB`.
- [ ] Khong lap lai plain MobileNetV3-yolo OOF fold route `mobilenetv3_large_100.ra_in1k/3e/plain CE/no-oversampling/no-class-aware-crop` hoac mo full 5-fold tu fold00 nay; pretrained fold expert qua yeu va XAI kem foreground hon TRKH. Neu pretrained OOF tiep, can model/expert manh hon hoac representation/fold-safe signal co val class1 gan/vuot keeper truoc khi spend full 5-fold time.
- [x] 2026-07-04 EfficientNetV2-S command-check `tf_efficientnetv2_s.in21k_ft_in1k` fold00 pass va tai xong weights; smoke `60b/2e`, batch32, CE/no focal/no LDAM/no balanced/no rare-repeat/no class-aware aug, full fold00 val. Trainer history best macro/class1 `0.8205/0.3804`, class1 P/R `0.4605/0.3241`, da thua MobileNetV3 fold00 va keeper.
- [x] 2026-07-04 reload eval EfficientNetV2-S best checkpoint qua normal evaluator chi `0.6452/0.1120`, class1 P/R `0.4118/0.0648`, confusion class1 `1->0=44`, `1->2=26`, `1->4=16`, `1->3=15`; artifact reload khong on dinh nen khong dung trainer history lam gate.
- [x] 2026-07-04 XAI EfficientNetV2-S reload class1 FN cases8: attention fg/bg/border `0.5176/0.4824/0.6663`, Grad-CAM `0.7072/0.2928/0.4177`, flags `attention_border=8/8`, `attention_background=7/8`, `gradcam_border=8/8`, `gradcam_background=6/8`; border/background reliance te hon TRKH keeper.
- [x] Cleanup EfficientNetV2-S fold00 reject: copy train/eval/boundary/XAI metrics vao `runs\xai_smoke_timm_effv2s_yolof_oof_fold00_plain_val_cases8_20260704\source_smoke_metrics`, xoa micro/smoke train dirs; manifest `runs\cleanup_manifest_20260704_effv2s_oof_fold00_rejected_smoke.json`, reclaimed `644.44 MB`.
- [x] 2026-07-04 correction: old EfficientNetV2-S reload eval/XAI above used wrong evaluator normalization. Fixed checkpoint `input_mean/std` propagation in `evaluate.py`, `xai_audit.py`, and `attention_viz.py`; preflight `py_compile` pass and `tests\test_xai_audit_dataset_tensor.py` now `8 passed`.
- [x] 2026-07-04 rerun EfficientNetV2-S normfix smoke same recipe `smoke_timm_effv2s_yolof_oof_fold00_plain_normfix_60b_2e_20260704`: trainer macro/class1 `0.8125/0.3497`, class1 P/R `0.4267/0.2963`; reload eval now matches macro `0.8128`, so normalization bug is fixed but route is still much weaker than keeper. Confusion class1 `1->0=52`, `1->2=22`.
- [x] 2026-07-04 robust XAI EfficientNetV2-S normfix cases12: attention fg/bg/border `0.5433/0.4567/0.6339`, Grad-CAM `0.7112/0.2888/0.4234`, flags `attention_border=12/12`, `gradcam_border=12/12`, `object_color_sensitive=12/12`; object_desaturate drop `0.5381`, background_gray drop `0.0007`, overlays hot on pad/background/hand/border.
- [x] Cleanup EfficientNetV2-S normfix reject: copy metrics into `runs\xai_smoke_timm_effv2s_yolof_oof_fold00_plain_normfix_val_cases12_robust_20260704\source_smoke_metrics`, keep eval/boundary/robust XAI, delete smoke dir + duplicate non-robust XAI + logs; manifest `runs\cleanup_manifest_20260704_effv2s_normfix_rejected_smoke.json`, reclaimed `327.65 MB`.
- [ ] Khong lap lai EfficientNetV2-S quick fold00 route `tf_efficientnetv2_s.in21k_ft_in1k/60b/2e/LR=5e-4/plain CE` hoac mo full 5-fold OOF tu route nay. Old reload artifact la invalid do normalization mismatch; normfix artifact hop le nhung class1 chi `0.3497`, XAI border/background/pad sensitive. Neu EfficientNetV2 revisit, phai la recipe reload-stable va gan keeper truoc khi spend 5-fold.
- [x] 2026-07-04 tra cuu CoAtNet hybrid conv-attention (`https://arxiv.org/abs/2106.04803`) va command-check timm `coatnet_0_rw_224.sw_in1k`; model load/normalization `(0.5,0.5,0.5)` OK tren fold00 `yolo_f`.
- [x] 2026-07-04 smoke CoAtNet-0 fold00 `coatnet_0_rw_224.sw_in1k/60b/2e/LR=3e-4/plain CE/no focal/no LDAM/no balanced/no rare-repeat/no class-aware aug`: trainer macro/class1 `0.7892/0.2550`, class1 P/R `0.4634/0.1759`; reload eval match `0.7916/0.2649`, class1 P/R `0.4651/0.1852`. Main confusion class1 `1->0=67`, `1->2=17`, `1->4=3`, `1->3=1`.
- [x] 2026-07-04 robust XAI CoAtNet-0 cases12: all selected were class1 false negatives. Attention fg/bg/border `0.0833/0.9167/0.0000`, Grad-CAM `0.7480/0.2520/0.0129`, flags `attention_background=11/12`, `object_color_sensitive=10/12`, `register_cls_attention_divergent=12/12`; object_desaturate drop `0.4745`, background blur/gray gan zero. Overlay `1->4` nong o vung dot den cuc bo; `1->2` heat rat yeu.
- [x] Cleanup CoAtNet-0 reject: copy train/eval metrics vao `runs\xai_smoke_timm_coatnet0_yolof_oof_fold00_plain_val_cases12_robust_20260704\source_smoke_metrics`, keep eval/boundary/robust XAI, delete micro + smoke train dir + logs; manifest `runs\cleanup_manifest_20260704_coatnet0_fold00_rejected_smoke.json`, reclaimed `839.10 MB`.
- [ ] Khong lap lai plain CoAtNet-0 fold00 route `coatnet_0_rw_224.sw_in1k/60b/2e/LR=3e-4/plain CE` hoac mo 5-fold OOF tu recipe nay; hybrid backbone pretrained khong tu dong giai quyet class1, attention con all-background va recall class1 rat thap. Neu dung hybrid pretrained tiep, can recipe/teacher co val evidence gan keeper truoc khi spend full fold.
- [x] 2026-07-04 tra cuu MaxViT local-global hybrid attention (`https://arxiv.org/abs/2204.01697`) va command-check timm `maxvit_rmlp_tiny_rw_256.sw_in1k`; model load/normalization `(0.5,0.5,0.5)` OK tren fold00 `yolo_f`.
- [x] 2026-07-04 smoke MaxViT-256 fold00 `maxvit_rmlp_tiny_rw_256.sw_in1k/60b/2e/LR=2e-4/plain CE/no focal/no LDAM/no balanced/no rare-repeat/no class-aware aug`: trainer macro/class1 `0.7447/0.0339`, class1 P/R `0.2000/0.0185`; reload eval match `0.7452/0.0339`. Confusion class1 chi dung `2/108`, voi `1->0=76`, `1->2=25`, `1->4=4`, `1->3=1`.
- [x] 2026-07-04 robust XAI MaxViT-256 cases12: all selected were high-confidence class1 false negatives. Attention fg/bg/border `0.5661/0.4339/0.5104`, Grad-CAM `0.9872/0.0128/0.0899`, flags `attention_background=12/12`, `attention_border=12/12`, `object_color_sensitive=12/12`; object_desaturate drop `0.6510`, background blur/gray gan zero. Overlay cho thay model co nhin object nhung hoc sai surface/color cue.
- [x] Cleanup MaxViT-256 reject: copy train/eval/boundary metrics vao `runs\xai_smoke_timm_maxvit256_yolof_oof_fold00_plain_val_cases12_robust_20260704\source_smoke_metrics`, keep eval/boundary/robust XAI, delete micro + smoke train dir + logs; manifest `runs\cleanup_manifest_20260704_maxvit256_fold00_rejected_smoke.json`, reclaimed `900.41 MB`.
- [ ] Khong lap lai plain MaxViT-256 fold00 route `maxvit_rmlp_tiny_rw_256.sw_in1k/60b/2e/LR=2e-4/plain CE` hoac mo 5-fold OOF tu recipe nay; local-global pretrained hybrid con bo roi class1 nang hon CoAtNet. Ket luan cho cau hoi context/backbone: generic pretrained hybrid khong phai loi giai, quay ve TRKH-native/fold-safe signal co recall-protection.
- [x] 2026-07-04 trien khai `PatchEvidenceRouterSummaryStats` default-off: router descriptor them verifier-like pair-margin/bbox/top-k/positive-fraction/base-prob stats, checkpoint-safe va zero-init residual giu default unchanged. Preflight pass `py_compile model.py/train.py/config.py`, `tests\test_patch_evidence_router.py` (`7 passed`), V8 launcher parse.
- [x] 2026-07-04 smoke patch-summary router + OOF patch teacher head-only `60b/1e`, explicit `yolo_f`, full val `2606`: trainer macro/class1 chi `0.8777/0.6611`, P/R class1 `0.5728/0.7815`, confusion `0->1=55`, `4->1=17`, `2->1=12`, `1->0=16`, `1->2=11`; reload eval macro `0.8789`. Reject truoc probe vi duoi keeper `0.8847/0.6860` va duoi gate `0.70`.
- [x] 2026-07-04 boundary/XAI patch-summary router: boundary `runs\boundary_review_smoke_v8_yolof_patchsummaryrouter_summary_oofteacher_val_20260704` selected `153` rows (`FN1=33`, `FP1=40`, `low_margin_error=40`, pair `0-1=80`). XAI `runs\xai_smoke_v8_yolof_patchsummaryrouter_summary_oofteacher_val_cases12_robust_20260704` showed attention fg/bg `0.9318/0.0682`, Grad-CAM fg/bg `0.7934/0.2066`, object_desaturate drop `0.2082` vs background blur/gray `0.0084/0.0067`, flags `object_color_sensitive=11/12`, `gradcam_border=8/12`.
- [x] Cleanup patch-summary router reject: metrics/config/history/transcript/trace copied into `runs\xai_smoke_v8_yolof_patchsummaryrouter_summary_oofteacher_val_cases12_robust_20260704\source_smoke_metrics`; failed partial + trained smoke dirs deleted via `runs\cleanup_manifest_20260704_patchsummaryrouter_summary_rejected_smoke.json`, reclaimed `133.35 MB`.
- [ ] Khong lap lai exact patch-evidence router summary-stats recipe `pair=0-1/topK=4/bbox_weight=0.75/logit_scale=0.12/hard_loss=0.004/teacher_loss=0.020/OOF patch teacher` tren current keeper. Offline verifier co gia tri diagnostic nhung trainable summary router lam rong false-positive class1 va van bi surface/color cue chi phoi.
- [ ] Huong tiep theo co gia tri: dung patch verifier/MIL chi khi co initialization/fold-safe target bao ve true class1 recall ro rang. Neu khong, tam roi cac frozen-router/readout tren current embedding va uu tien signal representation moi hoac conservative two-stage verifier chi suppress FP1 sau khi chung minh khong lam mat recall class1 tren val gate.
- [x] 2026-07-04 check AIDT status: `D:\DataAI\AIDT\runs\resnet50_vit_b16_5class` la pretrained ResNet50+ViT-B/16, `112.7M` params, train tren `class_f`, test macro/class1 `0.8841/0.6225`; khong vuot TRKH single final `0.8865/0.6667`. Dung AIDT lam diagnostic/pretrained contrast, khong xem la muc tieu thay TRKH.
- [x] 2026-07-04 refresh source-group audit `runs\yolo_source_group_audit_current_20260704`: train `691` multi-object va `144` mixed-label images, class1 `115` multi-object trong do `102` mixed-label; val chi `29` multi-object same-label `0/3`, test single-object het. Khong dung global same-source consistency/source-group trick de giai class1.
- [x] 2026-07-04 da giai thich voi nguoi dung: cac model ngoai duoc test de chan doan pretraining/architecture/split, khong phai doi muc tieu. Gate van la TRKH tren valid `class_f`/`yolo_f`, test split khong cham lai neu khong freeze truoc.
- [x] 2026-07-04 MobileNetV3 fold00 no-warmup/fair-selection probe: `mobilenetv3_large_100.ra_in1k`, fold00 `yolo_f` OOF data, full val `1858`, `--warmup-epochs 0`, `--best-metric fair_macro_f1`, CE/sqrt-inverse weights, no class-aware aug/rare-repeat/balanced sampling, stopped after epoch3. Best/reload macro/class1 only `0.8635/0.5361`, class1 P/R `0.6047/0.4815`, confusion true class1 `[33,52,19,2,2]`; below old MobileNetV3 fold00 and far below TRKH keeper `0.8847/0.6860`.
- [x] 2026-07-04 XAI MobileNetV3 no-warmup/fair cases12: boundary review selected `51` rows (`FN1=21`, `FP1=6`, high-conf error `24`); XAI high-conf class1 misses showed attention fallback all-background, Grad-CAM fg/bg/border `0.7117/0.2883/0.2877`, flags `object_color_sensitive=12/12`, `gradcam_border=12/12`; background blur/gray drops `0.0005/0.0129`, object_desaturate drop `0.6318`. One case was an ultra-thin padded crop, and typical cases still used surface/color rather than useful wide context.
- [x] Cleanup MobileNetV3 no-warmup/fair reject: copied history/config/logs into `runs\xai_probe_timm_mobilenetv3_yolof_oof_fold00_nowarmup_fair_val_cases12_20260704\source_probe_metrics\probe_train`, kept eval/boundary/XAI, deleted checkpoint run via `runs\cleanup_manifest_20260704_rejected_mobilenetv3_fold00_nowarmup_probe.json`, reclaimed `71.46 MB`.
- [ ] Khong lap lai MobileNetV3 fold-safe teacher route `mobilenetv3_large_100.ra_in1k` on `yolo_f` fold00 by only changing warmup/checkpoint metric/short epoch count; no-warmup fair selection still class1 `0.5361`, so no 5-fold OOF expansion and no KD/sample-weight/router transfer from this expert. Next pretrained-teacher work needs a stronger fold-held-out expert near keeper class1 or a different representation signal with recall protection.
- [x] 2026-07-04 built `class_f/train` 5-fold OOF artifact `runs\classf_oof_folds_train5_20260704`: `9215` source samples, hardlink `55290`, copy `0`, raw dataset untouched. Fixed generated fold `data.yaml` roots to absolute paths and patched `build_classification_oof_folds` so future folds do not resolve as zero-sample TRKH eval roots; focused fold/stitch tests pass (`2 passed`).
- [x] 2026-07-05 compacted `runs\classf_oof_folds_train5_20260704` after the class_f MobileNetV3 fold route was rejected: deleted generated hardlink fold trees `fold_00..fold_04`, kept `summary.json` and `COMPACTED_README.md`, wrote `runs\cleanup_manifest_20260705_compact_classf_oof_fold_materialization.json`. Raw `class_f` and current `yolo_f` folds were verified present; measured hardlink targets were `4459.693 MB`, actual free-space delta `11.336 MB`.
- [x] 2026-07-04 patched timm baseline import/XAI infrastructure: `import_timm_baseline_checkpoint` now stores timm mean/std and `resize_mode=stretch`; `ClassificationTransform` supports stretch resize for imported timm audit. This is audit infrastructure only, not a TRKH training recipe.
- [x] 2026-07-04 `class_f` MobileNetV3 fold00 probe `mobilenetv3_large_100.ra_in1k`, baseline script, full fold00 val `1843`, stopped after epoch10: best epoch7 direct export macro/focus-class F1 `0.8822/0.5933`, P/R focus `0.6139/0.5741`, below TRKH keeper `0.8847/0.6860` and below teacher gate. Do not expand to 5-fold.
- [x] 2026-07-04 boundary/XAI `class_f` MobileNetV3 fold00: boundary selected `64` rows (`FN_focus=24`, `FP_focus=24`, high-conf error `16`); XAI cases12 standard-order top confusions `0->1=32`, `1->0=27`, `1->2=16`, `2->3=12`; Grad-CAM fg/bg/border `0.6601/0.3399/0.2395`, background blur/gray drops `0.0056/0.0031`, object_desaturate drop `0.5853`, flags `high_confidence_misclassification=12`, `object_color_sensitive=9`, `gradcam_border=10/12`.
- [x] Cleanup `class_f` MobileNetV3 fold00 reject: copied train logs/import summary/export metrics/boundary case CSV into `runs\xai_probe_timm_mobilenetv3_classf_oof_fold00_std_val_cases12_20260704\source_probe_metrics`; deleted raw/imported checkpoint dirs via `runs\cleanup_manifest_20260704_rejected_mobilenetv3_classf_fold00_probe.json`, reclaimed `32.54 MB`.
- [ ] Khong lap lai MobileNetV3 fold-safe teacher route tren `class_f` fold00 voi baseline/pretrained recipe nay; no cung khong dat class1 `0.60`, XAI van surface/color/border dominated va khong ung ho wide-context/background hypothesis. Pretrained teacher tiep theo phai co fold-held-out class1 gan keeper hoac signal recall-safe moi truoc khi spend 5-fold.
- [x] 2026-07-04 trien khai TRKH-native `LateClassAttentionPooling` default-off theo CaiT/cross-attention idea: head token query patch tokens truoc classifier, zero-init residual de checkpoint-safe; expose config/CLI/V8/resume extension va trace. Preflight pass `py_compile model.py config.py train.py`, V8 parser, va `tests\test_late_class_attention_pooling.py tests\test_trainable_module_prefixes.py -q` (`8 passed`).
- [x] 2026-07-04 smoke late class-attention pool-only, resume keeper, trainable chi `late_class_attention_pool`, LR `5e-4`, `60b/1e`, full `yolo_f/val=2606`, `SkipFinalTest`: validation macro/class1 chi `0.8825/0.6765`, P/R class1 `0.6085/0.7616`, duoi keeper `0.8847/0.6860`; confusion van `3->2=52`, `0->1=49`, `1->0=19`, `1->2=12`, `4->1=12`, `2->1=9`.
- [x] 2026-07-04 boundary/XAI late class-attention pool-only: boundary selected `79` rows (`FN1=24`, `FP1=19`, `low_margin_error=24`). XAI cases16 foreground cao (`attention/grad_rollout/gradcam/rollout 0.8999/0.9447/0.9478/0.8953`), background blur/gray near zero (`0.0010/0.0013`), object_desaturate lon hon (`0.0834`), rollout border/background `8/16`/`9/16`, Grad-CAM border `5/16`; overlay van surface/edge/stem/background-adjacent, khong tao context cue moi.
- [x] Cleanup late class-attention reject: copied smoke metrics/config/history/logs into `runs\xai_smoke_v8_yolof_lateclassattn_poolonly_val_cases16_20260704\source_smoke_metrics`; deleted rejected smoke train dir via `runs\cleanup_manifest_20260704_lateclassattn_poolonly_rejected_smoke.json`, reclaimed `135.08 MB`. Kept reload eval, boundary review, XAI package, and cleanup manifest.
- [ ] Khong lap lai exact late class-attention pool-only recipe `LateClassAttentionPooling=true/TrainableModulePrefixes=late_class_attention_pool/LR=5e-4/60b-120b` tren current keeper. Module co the giu default-off lam building block, nhung neu revisit phai co head/representation update hoac target fold-safe/recall-protecting moi; pool-only chi lap lai surface/boundary ambiguity va thua gate.
- [x] 2026-07-04 smoke late class-attention + head adaptation, `TrainableModulePrefixes=late_class_attention_pool,head`, LR `3e-4`, `60b/1e`, resume keeper, full `yolo_f/val=2606`, `SkipFinalTest`: validation macro/class1 chi `0.8828/0.6784`, P/R class1 `0.6073/0.7682`, van duoi keeper va gate; confusion `0->1=50`, `1->0=17-18`, `1->2=12`, `4->1=12`, `2->1=9`.
- [x] 2026-07-04 boundary/XAI late class-attention + head: boundary selected `78` rows (`FN1=24`, `FP1=19`, `low_margin_error=24`). XAI cases16 background blur/gray near zero (`0.0008/0.0008`) trong khi object_desaturate `0.0861`, rollout background/border `9/16`/`8/16`, Grad-CAM border `6/16`; overlay lap lai stem/edge/background-adjacent va surface-bright heat, khong tao cue moi.
- [x] Cleanup late class-attention + head reject: copied metrics/config/history/logs into `runs\xai_smoke_v8_yolof_lateclassattn_headpool_val_cases16_20260704\source_smoke_metrics`; deleted rejected smoke train dir via `runs\cleanup_manifest_20260704_lateclassattn_headpool_rejected_smoke.json`, reclaimed `135.08 MB`.
- [ ] Khong lap lai nearby late class-attention `pool+head` variants bang cach doi LR/batch count tren current keeper. Head adapt chi tang nhe class1 so voi pool-only (`0.6784` vs `0.6765`) nhung van thua keeper; neu dung lai late class-attention phai co target OOF/reliability/recall-protecting moi.
- [x] 2026-07-04 tao current-keeper train-only boundary review batch `runs\manual_boundary_review_current_keeper_train_batch01_20260704` tu `boundary_review_teacherfocusbinary015_train_20260703`: `31` mau gom `11` FN `1->0`, `12` FP `0->1`, va `8` low-margin `0/1`; sinh `review_batch_unfilled.csv`, `contact_sheet_01/02.jpg`, `summary.json`, `visual_review_notes.md`; raw data khong doi, val/test khong dung.
- [x] 2026-07-04 visual pass batch01: class1 FN nhin nhu xoai xanh/subtle risk cue kho phan biet trong crop; class0 FP co glare/shadow/speckle/stem/skin marks/crop context artifact; tiep tuc ung ho ket luan surface/label-boundary ambiguity, khong phai thieu generic context/backbone.
- [ ] Khong train tu `manual_boundary_review_current_keeper_train_batch01_20260704` khi `manual_label_status` con trong. Neu dung review route, phai dien policy train-only ro rang (`correct/ambiguous/wrong/needs_crop`), bao ve true class1 recall, khong auto-relabel bang validation/test.
- [ ] Huong tiep theo co gia tri: neu khong co signal representation fold-safe moi, tiep tuc bang current-keeper train-only boundary review policy thay vi them loss/router gan cac recipe da reject; review batch nho truoc, build manifest, smoke full-val/no-test chi khi policy co ly do recall-protecting ro.
- [x] 2026-07-04 diagnostic bbox-geometry/context selector `runs\diagnostic_yolof_bbox_geometry_selector_keeper_runval_20260704`: fit train-only tren geometry/source-count + keeper probability, eval full `yolo_f/val=2606`, test untouched. Geometry-only sap manh (`0.3619/0.1667` logreg, `0.3348/0.0252` ExtraTrees); baseprob+geometry van duoi base (`0.8772-0.8773/class1 0.6627-0.6667` vs base `0.8841/0.6841`).
- [x] Cleanup bbox-geometry superseded diagnostic: xoa `runs\diagnostic_yolof_bbox_geometry_selector_keeper_20260704`, giu `runs\diagnostic_yolof_bbox_geometry_selector_keeper_runval_20260704`; manifest `runs\cleanup_manifest_20260704_bbox_geometry_superseded_deleted.json`, reclaimed `0.0087 MB`.
- [ ] Khong lap lai bbox-position/source-count geometry router, background/context selector, hoac dung YOLO object position nhu signal chinh tren current keeper. `yolo_f` bbox chi nen giu lam object prior yeu; neu revisit context, phai co representation/target moi va recall-protection ro, khong dung standalone geometry.
- [x] 2026-07-04 audit AIDT/top-5/TRKH consensus train-signal `runs\diagnostic_aidt_top5_consensus_train_signal_20260704`: no train, no raw data edit, no test read. Phat hien old summary ghi `trkh_max_conf=0.30` nhung ket qua val `38` changes / `32` corrections / `6` harms duoc reproduce voi effective `trkh_max_conf=0.35`; artifact giu ca rule `0.30` va `0.35`.
- [x] 2026-07-04 consensus metrics: rule `0.30` train `0.9579/0.8812` voi `93` corrections `0` harms, val `0.8985/0.7329` voi `33` changes; rule `0.35` train `0.9616/0.8944` voi `111` corrections `0` harms, val `0.8985/0.7319` voi `38` changes. Day la teacher in-sample signal, khong du fold-safe de smoke.
- [x] 2026-07-04 tao train-only review queue `runs\manual_consensus_review_train_signal_20260704`: `111` rows, `102` both_non1_suppress + `9` both_1_rescue, 6 contact sheets, `review_batch_unfilled.csv`, raw data untouched. Visual audit cho thay green-fruit/surface-boundary/glare/dark-spot/stem/crop ambiguity, khong phai nen rong/context la loi giai.
- [ ] Khong train tu consensus review queue khi `manual_label_status` con trong, khong auto-relabel/sample-weight/KD tu AIDT/top-5 in-sample train consensus. Neu dung tiep, phai la manual train-only policy hoac OOF/fold-safe teacher signal co explicit class1 recall protection.
- [ ] Trang thai hien tai: van dang hoan thien TRKH. AIDT/top-5/timm/benchmark ngoai chi la diagnostic/teacher/upper-bound de boc tach loi va tim signal; khong doi muc tieu sang model ngoai, khong dung test de chon threshold.
- [x] 2026-07-04 sua review manifest tooling de giu `sample_index` cho `yolo_f`: `build_boundary_review_manifest` ghi `sample_index`, `build_reviewed_boundary_training_manifests` propagate `sample_index/source_row_index` vao sample-weight/targeted-margin/soft-target/relabel rows, tranh collapse multi-object path. Preflight pass `py_compile` va focused pytest boundary/reviewed manifests (`6 passed`).
- [x] 2026-07-04 tao train-only manual-review policy v1 `runs\manual_boundary_review_current_keeper_train_batch01_policy_v1_20260704`: `31` rows, `23` correct boundary rows thanh targeted-margin, `8` needs_crop downweight, raw data/test/val khong doi; loader matched `sample_weight=31`, `targeted_margin=23` theo `sample_index`.
- [x] 2026-07-04 smoke manual-review policy v1 full `yolo_f/val=2606`: reload eval macro/class1 `0.8797/0.6667`, P/R class1 `0.5813/0.7815`, duoi keeper `0.8847/0.6860` va gate `0.70`; confusion van `0->1=52`, `4->1=16`, `2->1=13`, `1->0=15`, `1->2=12`, `1->4=6`.
- [x] 2026-07-04 boundary/XAI manual-review policy v1: boundary `runs\boundary_review_smoke_v8_yolof_manualreview_policyv1_val_20260704` selected `153` rows (`FN1=33`, `FP1=40`, low-margin error `40`); XAI `runs\xai_smoke_v8_yolof_manualreview_policyv1_val_cases12_robust_20260704` showed background blur/gray drops `0.0083/0.0076` vs object_desaturate `0.1606`, flags `object_color_sensitive=8/12`, `gradcam_border=8/12`.
- [x] Cleanup manual-review policy v1 reject: copied metrics/config/history/eval predictions/trace into `runs\xai_smoke_v8_yolof_manualreview_policyv1_val_cases12_robust_20260704\source_smoke_metrics`; deleted rejected checkpoint run via `runs\cleanup_manifest_20260704_manualreview_policyv1_rejected_smoke.json`, reclaimed `185.53 MB`.
- [ ] Khong lap lai exact manual-review policy v1 `correct + needs_crop_downweight + TargetedMarginLossWeight=0.004` tren current keeper. Batch qua thua (`train_targeted_margin_fraction~0.0036`) va lam rong FP1; neu dung manual review tiep, can coverage lon hon, policy recall-protecting ro hon, hoac fold-safe reliability signal truoc smoke.
- [x] 2026-07-04 tao consensus hard-case policy v1 `runs\manual_consensus_review_train_signal_policy_v1_20260704`: tu train-only AIDT/top-5 consensus review queue, giu hard label, khong relabel/raw edit/test; `111` targeted-margin rows keyed by `sample_index`, pairs `0->1=68`, `2->1=22`, `4->1=9`, `1->0=8`, `3->1=3`, `1->4=1`.
- [x] 2026-07-04 smoke consensus hard-case policy v1 `80b/1e`, `TargetedMarginLossWeight=0.0025`, full `yolo_f/val=2606`: targeted margin active (`fraction=0.0113`, loss `0.0431`) nhung val macro/class1 chi `0.8783/0.6611`, P/R class1 `0.5728/0.7815`, reload macro `0.8786`; FP1 rong hon (`0->1=55`, `4->1=16`, `2->1=13`).
- [x] 2026-07-04 boundary/XAI consensus policy v1: boundary `runs\boundary_review_smoke_v8_yolof_consensus_policyv1_val_20260704` selected `153` rows (`FN1=33`, `FP1=40`, pair `0-1=80`); XAI `runs\xai_smoke_v8_yolof_consensus_policyv1_val_cases12_robust_20260704` showed background blur/gray `0.0067/0.0073` vs object_desaturate `0.1614`, flags `object_color_sensitive=8/12`, `gradcam_border=7/12`, `near_tie=4/12`.
- [x] Cleanup consensus policy v1 reject: copied metrics/config/history/eval predictions/trace into `runs\xai_smoke_v8_yolof_consensus_policyv1_val_cases12_robust_20260704\source_smoke_metrics`; deleted rejected checkpoint run via `runs\cleanup_manifest_20260704_consensus_policyv1_rejected_smoke.json`, reclaimed `185.53 MB`.
- [ ] Khong lap lai in-sample AIDT/top-5 consensus direct targeted-margin/sample-weight/KD policy tren current keeper. Consensus co gia tri review/diagnostic nhung khi train truc tiep van lam rong false-positive class1; neu dung lai can OOF/fold-safe hoac representation-level signal co recall protection, khong sparse margin nua.
- [x] 2026-07-05 internal SSL-DINO train-only phase `runs\ssl_dino_yolof_keeperinit_surfdetail_64b_1e_20260705`: keeper init, `yolo_f/train`, `64` batches, surface-detail amplification, raw data untouched. SSL loss `2.8367`, teacher entropy `0.8788`; caveat: SSL config did not include `bbox_spatial_fusion_head`, so V8 smoke initialized that head from scratch.
- [x] 2026-07-05 smoke V8 from SSL-DINO checkpoint, full `yolo_f/val=2606`, no test, no hard/sample-weight/targeted-margin manifests: macro/class1 only `0.8730/0.6446`, class1 P/R `0.5519/0.7748`, below keeper `0.8847/0.6860`. FP1 widened (`0->1=58`, `4->1=23`, `2->1=11`) while true class1 misses stayed (`1->0=17`, `1->2=14`, `1->4=3`).
- [x] 2026-07-05 boundary/XAI SSL-DINO reject: boundary selected `153` rows (`FN1=34`, `FP1=40`, low-margin error `40`, low-margin correct boundary `39`); XAI `runs\xai_smoke_v8_yolof_ssl_dino_keeperinit_surfdetail64b_val_cases12_robust_20260705` showed background blur/gray drops `0.0050/0.0043` vs object_desaturate `0.1457`, flags `rollout_background=8/12`, `rollout_border=6/12`, `object_color_sensitive=5/12`.
- [x] Cleanup SSL-DINO reject: copied evidence into `runs\evidence_ssl_dino_keeperinit_surfdetail64b_rejected_20260705`; deleted `runs\ssl_dino_yolof_keeperinit_surfdetail_64b_1e_20260705` and `runs\smoke_v8_yolof_ssl_dino_keeperinit_surfdetail64b_60b_1e_20260705`; manifest `runs\cleanup_manifest_20260705_ssl_dino_keeperinit_surfdetail64b_rejected.json`, reclaimed `357.66 MB`.
- [ ] Khong lap lai exact internal SSL-DINO route `keeper init + foreground_luma surface-detail + 64b/1e + V8 60b/1e` tren current keeper. Neu SSL/repr tiep, can bbox-fusion-compatible init hoac target recall-protecting moi; DINO surface pretrain hien tai chi lam rong FP1 va khong tao signal background/context huu ich.
- [x] 2026-07-05 tra cuu RandAugment (`https://arxiv.org/abs/1909.13719`) va chay smoke rat nhe tren current keeper: `RandAugmentNumOps=1/Magnitude=4`, V8 keeper resume, `60b/1e`, full `yolo_f/val=2606`, `SkipFinalTest`, khong hard/sample-weight/soft-target/targeted-margin manifest.
- [x] 2026-07-05 light RandAugment smoke bi reject: launcher val macro/class1 `0.8776/0.6629`, reload `0.8771/0.6629`, class1 P/R `0.5756/0.7815`, duoi keeper `0.8847/0.6860` va gate `0.70`; FP1 rong hon (`0->1=53-54`, `4->1=16`, `2->1=13`) trong khi FN1 van `1->0=15`, `1->2=12`, `1->4=6`.
- [x] 2026-07-05 boundary/XAI light RandAugment: boundary selected `153` rows (`FN1=33`, `FP1=40`, low-margin error `40`, pair `0-1=80`); XAI `runs\xai_smoke_v8_yolof_light_randaugment1m4_val_cases12_robust_20260705` showed background blur/gray drop gan zero `0.0010/-0.0001`, object_desaturate `0.1423`, flags `rollout_background=7/12`, `rollout_border=6/12`, `object_color_sensitive=5/12`.
- [x] Cleanup light RandAugment reject: copied metrics/config/history/trace vao `runs\xai_smoke_v8_yolof_light_randaugment1m4_val_cases12_robust_20260705\source_smoke_metrics`; deleted checkpoint smoke via `runs\cleanup_manifest_20260705_light_randaugment1m4_rejected_smoke.json`, reclaimed `182.69 MB`.
- [ ] Khong lap lai nearby light RandAugment `NumOps=1/Magnitude=4` hoac chi tang generic augmentation strength tren current keeper. Generic stochastic augmentation lam mo color/surface boundary va tang class1 FP; neu revisit augmentation phai co class-pair/case-level reliability va recall-protection ro, khong phai photometric/global robustness.
- [ ] Huong tiep theo co gia tri: tiep tuc TRKH-native, nhung tranh loss/router/sparse margin tren current embedding. Ung vien tiep theo nen la representation signal moi co guard recall class1 ro rang, hoac manual train-only review coverage lon hon voi policy bao ve true class1; khong dung test va khong chay 30e neu smoke+probe chua vuot class1 val `0.70`.
- [x] 2026-07-05 reviewed all 6 contact sheets / 111 rows from `runs\manual_consensus_review_train_signal_20260704` after the user asked whether the work was still TRKH-focused. Conclusion: queue is diagnostic only; suppressor rows and rescue rows are both visually plausible, but not clean enough for a safe auto policy.
- [x] 2026-07-05 updated `runs\manual_consensus_review_train_signal_20260704\visual_review_notes.md` with the no-train decision: dominant issue remains fruit surface/boundary/illumination ambiguity, not broad background.
- [ ] Do not create consensus policy v2 while `manual_label_status` is empty. Do not repeat direct in-sample consensus targeted-margin/sample-weight/KD from this queue. Revisit only with human-reviewed train-only labels or fold-safe/OOF teacher evidence that explicitly protects true class-1 recall.
- [ ] Next valuable step: choose a TRKH-native representation or reliability signal that is not another loss-level/logit/sparse-margin tweak on the current embedding; if no such signal is available, run a diagnostic-only analysis first rather than launching another smoke.
- [x] 2026-07-05 implemented verifier model export tooling: pairwise/patch-evidence verifier now exports train-fit logistic `raw_coef/raw_intercept`, standardizer params, and feature schema support; preflight `py_compile` plus focused pytest passed (`15 passed`).
- [x] 2026-07-05 full no-test patch-evidence export `runs\patch_evidence_mil_yolof_keeper_01only_exportparams_20260705`: train `9215`, val `2606`, pair `0-1`, feature_dim `318`, fit pair samples `2482`, OOF local accuracy `0.9243`; validation improved from base `0.8841/0.6841` to patch-verified `0.8892/0.7073`.
- [x] 2026-07-05 audit changed val cases for export: `31` changes = `16` corrections, `9` harms, `6` neutral; transitions mostly `1->0` suppressions (`24`) plus `0->1` rescues (`7`). Keep as recall-risk diagnostic, not final model.
- [x] 2026-07-05 cleanup obsolete export smoke dirs via `runs\cleanup_manifest_20260705_patch_evidence_exportparams_smokes_deleted.json`, reclaimed `0.394 MB`; preserved full export artifact.
- [x] 2026-07-05 implemented checkpoint-safe default-off in-model `PatchEvidenceLinearVerifier` and `evaluate.py` loader flags for the exported `318`-dim verifier JSON; focused preflight passed `py_compile` and verifier/router pytest (`25 passed`).
- [x] 2026-07-05 runtime verifier full-val/no-test on same evaluator: baseline `runs\eval_yolof_keeper_full_val_baseline_20260705` macro/class1 `0.8829/0.6783`; softboost verifier `runs\eval_patch_linear_verifier_softboost001_full_val_20260705` macro/class1 `0.8887/0.7030`, class1 P/R `0.6480/0.7682`, full support `2606`, `logit_boost=0.01`, loss `1.1858`.
- [x] 2026-07-05 runtime changed-case audit saved `runs\eval_patch_linear_verifier_softboost001_full_val_20260705\changed_cases_runtime_vs_baseline.csv`: `29` changes = `16` corrections, `8` harms, `5` neutral; transitions `1->0=22`, `0->1=7`.
- [x] 2026-07-05 recall-protection p1 diagnostic rejected: restoring `1->0` suppressions by base p1 threshold worsened class1 F1 (`0.6782-0.7009`) versus no restore `0.7030`; p1 distributions overlap between true-class1 harms and correct class0 suppressions.
- [x] 2026-07-05 XAI runtime-changed cases `runs\xai_eval_patch_linear_verifier_full_val_changed12_20260705`: selected `8` harms + `4` corrections; object_desaturate drop `0.0870` vs background blur/gray `0.0017/0.0013`, near-tie `7/12`, rollout border `10/12`. Added verifier-aware XAI loading and confirmed softboost predictions in `runs\xai_eval_patch_linear_verifier_softboost001_changed4_verifieraware_20260705`: routed confidence stays realistic (`0.272-0.284`) and heatmaps remain foreground-heavy.
- [x] Cleanup runtime verifier smoke eval superseded by full-val: deleted `runs\eval_patch_linear_verifier_smoke_20260705`; manifest `runs\cleanup_manifest_20260705_runtime_verifier_smoke_deleted.json`, reclaimed `1.128 MB`.
- [x] Cleanup hardboost verifier artifacts superseded by softboost: deleted `runs\eval_patch_linear_verifier_full_val_20260705` and `runs\xai_eval_patch_linear_verifier_full_val_changed4_verifieraware_20260705`; manifest `runs\cleanup_manifest_20260705_runtime_verifier_hardboost_superseded_deleted.json`, reclaimed `4.609 MB`.
- [x] 2026-07-05 train-OOF patch-verifier acceptance meta-gate diagnostic rejected: `runs\patch_verifier_acceptance_meta_gate_yolof_keeper01_20260705` selected thresholds from train OOF only, but val stayed below softboost verifier. Best mode `class1_benefit` reached only macro/class1 `0.8854/0.6930` vs softboost runtime `0.8887/0.7030` and post-hoc proposal `0.8892/0.7073`.
- [x] 2026-07-05 XAI for rejected meta-gate accepted cases `runs\xai_patch_verifier_acceptance_meta_gate_class1benefit_accepted8_20260705`: `8` accepted cases (`4` harms, `4` corrections), near-tie `7/8`, Grad-CAM border `4/8`, attention/Grad-CAM background `3/8`, still foreground surface/boundary ambiguity.
- [ ] Do not continue with nearby logistic/threshold accept-reject gates over the same patch-verifier CSV features on the current embeddings. They look better on train OOF but do not generalize to val and do not create a new reliability cue.
- [ ] Do not repeat rejected patch-summary zero-init/OFF teacher router, and do not sweep simple p1 recall-protection thresholds around the runtime verifier. Next valuable TRKH-native step is verifier-aware XAI/trace support plus a fold-safe recall/FP reliability target, or a new representation signal that can push the next class1 val milestone `0.75` without losing true class1 recall.
- [x] 2026-07-05 softboost verifier remaining-error atlas `runs\forensics_softboost_verifier_remaining_errors_20260705`: current runtime verifier leaves `203/2606` val errors; dominant pairs `3->2=52`, `0->1=41`, `1->0=19`, `4->0=12`, `1->2=11`, `0->4=10`. Class1 FP (`63`) is often near-top2/reroutable (`target_in_top2=48`, `patch_any_target=32`), but class1 FN (`35`) lacks class1 support more often (`target_in_top2=16`, `patch_any_target=13`).
- [x] 2026-07-05 verifier-aware robust XAI on remaining errors `runs\xai_softboost_verifier_remaining_errors16_20260705`: selected `16` cases, `near_tie_top2=13/16`, rollout border `10/16`, Grad-CAM border `8/16`, background blur/gray drops `0.0035/-0.0038`, object_desaturate `0.0627`; remaining failures are still object surface/boundary ambiguity, not wide-background reliance.
- [x] Current answer to model-scope question: yes, work remains TRKH-focused. External/AIDT/timm/top5 checks are diagnostic/teacher/upper-bound evidence only; the implementation target is TRKH-native. Do not switch to a generic pretrained outside model as the final solution unless the user explicitly changes the research target.
- [ ] Next TRKH-native loop should not be another selector over the same logits/patch-verifier probabilities. Prioritize a representation change that improves class1 FN top2 support while keeping softboost verifier FP control as a guard; gate on full `yolo_f/val=2606`, no test, and only advance toward full train if class1 val reaches the next milestone `0.75`.
- [x] 2026-07-05 patched internal SSL-DINO tooling so V8 SSL can preserve `bbox_spatial_fusion_head`; dry-run confirmed `missing_key_count=0`, `unexpected_key_count=0`, and synced DINO teacher. Preflight `py_compile` plus focused SSL pytest passed (`14 passed`).
- [x] 2026-07-05 train-only SSL-DINO bbox-compatible retest `runs\ssl_dino_yolof_keeperinit_bboxfusion_surfdetail32b_1e_20260705`: keeper init, bbox spatial fusion enabled, `yolo_f/train`, foreground-luma surface detail, `32` batches, no raw-data edit/test use. SSL loss `4.4120`, teacher entropy `1.2361`.
- [x] 2026-07-05 supervised smoke from bbox-compatible SSL checkpoint, full `yolo_f/val=2606`, `48b/1e`, `SkipFinalTest`: validation macro/class1 only `0.8647/0.6213`, class1 P/R `0.5278/0.7550`, below keeper `0.8847/0.6860`; FP1 widened badly (`0->1=60`, `4->1=22`, `2->1=17`) and FN1 remained (`1->0=16`, `1->2=15`, `1->4=5`).
- [x] 2026-07-05 boundary/XAI SSL-DINO bbox-compatible reject: boundary selected `153` rows from `530` candidates, `focus_false_positive=102`; XAI cases12 showed attention/rollout background flags `8/12`, rollout border `5/12`, object_color_sensitive `5/12`, background blur/gray drops `0.0025/0.0017`, object_desaturate `0.1392`. Visual overlays showed fruit surface/color/border focus, not useful wide-background context.
- [x] Cleanup SSL-DINO bbox-compatible reject: copied compact evidence into `runs\evidence_ssl_dino_bboxfusion_surfdetail32b_rejected_20260705`; deleted dryrun/SSL/smoke/eval/boundary/XAI raw dirs via `runs\cleanup_manifest_20260705_ssl_dino_bboxfusion_surfdetail32b_rejected.json`, reclaimed `391.11 MB`.
- [ ] Do not repeat nearby internal SSL-DINO keeper-init surface-detail variants by only preserving bbox fusion, changing short batch count, or rerunning V8 48b/60b smoke. The bbox-compatible fix made the test fair, but the route still widened class1 false positives and did not improve representation.
- [x] 2026-07-05 implemented TRKH-native `ComplementaryPatchSuppressionHead` default-off: zero-init residual head suppresses top salient patch tokens and pools complementary patch evidence with bbox/foreground priors. Exposed config/CLI/V8 launcher/resume extension and trace. Preflight passed `py_compile`, focused pytest (`11 passed`), and V8 dry-run argument plumbing.
- [x] 2026-07-05 smoke complementary patch suppression, resume keeper, trainable only `complementary_patch_suppression_head`, LR `5e-4`, `60b/1e`, full `yolo_f/val=2606`, `SkipFinalTest`: validation macro/class1 only `0.8777/0.6611`, class1 P/R `0.5728/0.7815`, below keeper `0.8847/0.6860`; widened FP1 (`0->1=55`, `4->1=17`, `2->1=12`) while FN1 stayed (`1->0=16`, `1->2=11`, `1->4=6`).
- [x] 2026-07-05 boundary/XAI complementary patch suppression reject: independent eval macro `0.8771`; boundary selected `153` rows from `357` (`FP1=89`, `FN1=33`). Robust XAI cases12 showed background blur/gray near zero (`0.0007/-0.0004`) versus object_desaturate `0.1402`, rollout background/border `7/12`/`6/12`, and visual overlays still emphasized stem/border/glare/adjacent background rather than a new complementary surface cue.
- [x] Cleanup complementary patch suppression reject: copied compact evidence into `runs\evidence_complementary_patch_suppression_rejected_20260705`; deleted raw smoke/eval/boundary/XAI dirs via `runs\cleanup_manifest_20260705_complementary_patch_suppression_rejected.json`, reclaimed `143.734 MB`.
- [ ] Do not repeat nearby complementary/peak-suppression residual-head variants on the current keeper by only changing top-k/suppression strength/bbox weight/logit scale/LR/short batch count. It is a useful default-off building block but current patch embeddings do not contain enough class1 recall signal; next route needs a materially different feature source or fold-safe recall/FP reliability signal.
- [x] 2026-07-05 implemented train-time soft `PatchEvidenceLinearVerifier`: default-off differentiable pair-gated logit delta from the exported full-vector verifier, non-persistent verifier buffers, config/train CLI/V8 launcher args, resume-prefix handling, and focused tests.
- [x] 2026-07-05 preflight for train-time verifier passed: `py_compile model.py/config.py/train.py/test_patch_evidence_router.py`; pytest `tests/test_patch_evidence_router.py tests/test_trainable_module_prefixes.py -q` -> `16 passed`; V8 dry-run confirmed explicit `yolo_f`, `crop_bbox`, verifier JSON, soft-train flags, and `SkipFinalTest`.
- [x] 2026-07-05 soft-train verifier smoke `60b/1e`, resume keeper, full `yolo_f/val=2606`, runtime verifier `logit_boost=0.01`, soft scale `0.05`: macro/class1 only `0.8808/0.6782`, class1 P/R `0.5990/0.7815`, below keeper `0.8847/0.6860` and below runtime verifier `0.8887/0.7030`; no test touched.
- [x] 2026-07-05 reload evals for soft-train verifier: no-verifier eval macro `0.8785`; verifier eval reproduced `0.8808`. Boundary selected `153` rows (`FP1=79`, `FN1=33`, buckets `0->1=51`, `1->0=15`, `1->2=12`, `2->1=12`, `4->1=12`).
- [x] 2026-07-05 verifier-aware robust XAI for soft-train verifier: selected 12 class1 boundary misses; background blur/gray drops only `0.0025/0.0012`, object_desaturate drop `0.1496`, flags include attention background `5/12`, rollout background `7/12`, rollout border `5/12`, object_color_sensitive `6/12`; failure remains fruit surface/boundary ambiguity, not broad background context.
- [x] Cleanup soft-train verifier reject: copied compact evidence into `runs\evidence_patchlinear_softtrain_rejected_20260705`; deleted raw smoke/eval/boundary/XAI dirs via `runs\cleanup_manifest_20260705_patchlinear_softtrain_rejected.json`, reclaimed `197.991 MB`.
- [ ] Do not repeat train-time soft patch-linear verifier over the current exported verifier/current embeddings by only changing soft scale, LR, batch count, or temperature. Keep the default-off implementation for future stronger/fold-safe verifier signals, but next TRKH loop should target a genuinely new representation or recall-protecting reliability signal.
- [x] 2026-07-05 implemented default-off bbox-interior token-label auxiliary loss: `bbox_token_label_loss` reuses the classifier head on `features["patches"]`, filters selected tokens by `patch_bbox_prior`/`memory_key_padding_mask`, exposes config/CLI/V8 flags, and records token coverage/accuracy/margin. Preflight passed `py_compile` and focused pytest (`tests/test_bbox_token_label_loss.py tests/test_trainable_module_prefixes.py -q` -> `8 passed`); V8 dry-run confirmed explicit `yolo_f`, `crop_bbox`, teacher focus-binary, and `SkipFinalTest`.
- [x] 2026-07-05 smoke bbox-token-label `weight=0.006/min_prior=0.45/focus_weight=1.35`, resume keeper, `60b/1e`, full `yolo_f/val=2606`, no test: validation macro/class1 only `0.8785/0.6648`, class1 P/R `0.5784/0.7815`, below keeper `0.8847/0.6860`; token telemetry active but weak (`fraction=0.7604`, `accuracy=0.3677`, `target_margin=-0.0890`).
- [x] 2026-07-05 reload eval/boundary/XAI bbox-token-label: reload macro/class1 reproduced `0.8785/0.6648`; boundary selected `153` from `356` (`FP1=77`, `FN1=33`, buckets `0->1=46`, `4->1=14`, `2->1=13`, `1->0=15`, `1->2=12`). XAI cases12 kept high foreground mass but background blur/gray drops stayed near zero (`0.0013/0.0002`) while object_desaturate was large (`0.1429`); visual overlays still hit edge/stem/glare/adjacent regions.
- [x] Cleanup bbox-token-label reject: compact evidence saved in `runs\evidence_bbox_token_label_rejected_20260705` (`20.945 MB`), excluding rejected checkpoints; raw smoke/eval/boundary/XAI dirs deleted via `runs\cleanup_manifest_20260705_bbox_token_label_rejected.json`, reclaimed `197.212 MB`.
- [ ] Do not repeat bbox-interior token-label loss on the current keeper by only changing loss weight, min_prior, prior_power, or focus_weight. Without reliable per-token labels, this forces image-level boundary labels onto all interior patches and widens class1 false positives; revisit only with a fold-safe token teacher, manual train-only part/token policy, or recall-protecting reliability target.
- [x] 2026-07-05 color-stat fusion head-only smoke stayed TRKH-native/no-pretrain, not an outside-model pivot: resumed keeper, `ColorStatFusion=true`, trained only `color_fusion_head` (`14,683` params), LR `5e-4`, `60b/1e`, full `yolo_f/val=2606`, `SkipFinalTest`; validation macro/class1 only `0.8777/0.6611`, reload `0.8771/0.6611`, below keeper `0.8847/0.6860` and runtime verifier `0.8887/0.7030`.
- [x] 2026-07-05 boundary/XAI color-stat reject: boundary selected `153` rows from `354` (`FP1=78`, `FN1=33`, buckets `0->1=46`, `4->1=16`, `2->1=12`, `1->0=16`, `1->2=11`); robust XAI cases12 showed background blur/gray near zero (`0.0007/-0.0004`) vs object_desaturate `0.1402`, rollout background/border `7/12`/`6/12`, and overlays still emphasized surface/border/stem/glare/adjacent context instead of a stable class1 cue.
- [x] Cleanup color-stat reject: compact evidence saved in `runs\evidence_colorstatfusion_headonly_rejected_20260705` (`0.931 MB`), excluding rejected checkpoints; raw smoke/eval/boundary/XAI dirs deleted via `runs\cleanup_manifest_20260705_colorstatfusion_headonly_rejected.json`, reclaimed `142.054 MB`.
- [ ] Do not repeat color-stat fusion head-only on the current keeper by only changing LR/dropout/short-batch count, and do not add another shallow HSV/Lab/color-stat branch without a new fold-safe feature set and explicit class1 recall protection. It widens class1 false positives and confirms color/surface is label-relevant but not separable by shallow stats on current embeddings.
- [x] 2026-07-05 patched `build_oof_neighbor_label_policy` to prefer `sample_index` feature-bank matching and propagate `sample_index` into strict sample-weight/soft-target outputs; duplicate-path regression protects `yolo_f` multi-object rows. Preflight passed `py_compile` and focused pytest (`tests/test_oof_neighbor_label_policy.py tests/test_data_centric_sample_weights.py -q` -> `4 passed`).
- [x] 2026-07-05 diagnostic current patch-OOF cleanlab + multi-bank feature-neighbor policy: `runs\oof_neighbor_label_policy_patchoof_yolof_current_multibank_20260705`, no test/val input, `188` candidate rows from current patch-OOF cleanlab, issue pairs `0->1=70`, `1->0=11`, feature banks TRKH local-neighbor + DINOv2-small + EfficientNetV2-S. Strict all-bank consensus selected `0` rows; sample-weight and soft-target manifests are empty.
- [ ] Do not smoke/train from current patch-OOF cleanlab + neighbor policy. Even relaxed 2-of-3 agreement has only `14` rows (`1->0=8`, `0->1=6`) with weak/min-zero vote fractions and the same dangerous 0/1 boundary where prior train-OOF `0->1` routing was toxic. Revisit only with human-reviewed train labels or a stronger fold-safe reliability signal with explicit true-class1 recall protection.
- [x] 2026-07-05 smoke TRKH-native `BlockLocalPatchMixer` layers `5,6,7,8` + head adapter, resume keeper, LR `3e-4`, `60b/1e`, full `yolo_f/val=2606`, `SkipFinalTest`: validation macro/class1 only `0.8797/0.6667`, class1 P/R `0.5850/0.7748`, below keeper `0.8847/0.6860` and runtime verifier `0.8887/0.7030`. Confusion stayed `3->2=58`, `0->1=51`, `1->0=16`, `4->1=16`, `1->2=12`, `2->1=12`.
- [x] 2026-07-05 boundary/XAI block-local head reject: boundary selected `153` rows from `354` (`FP1=76`, `FN1=34`, low-margin `43`, buckets `0->1=45`, `1->0=16`, `1->2=12`, `4->1=15`, `3->2=28`). Robust XAI cases12 showed background blur/gray near zero (`0.0018/0.0007`) vs object_desaturate `0.1430`, rollout background/border `6/12`/`5/12`, Grad-CAM border `4/12`, and overlays still heated object border, pad/background, stem/glare, or diffuse surface spots.
- [x] Cleanup block-local head reject: compact evidence saved in `runs\evidence_blocklocal_head_layers5to8_rejected_20260705` (`6.292 MB`), excluding rejected checkpoints; raw smoke/eval/boundary/XAI dirs plus logs deleted via `runs\cleanup_manifest_20260705_blocklocal_head_layers5to8_rejected.json`, reclaimed `148.079 MB`.
- [ ] Do not repeat nearby `BlockLocalPatchMixer` + head variants on the current keeper by only changing layers, LR, residual scale, dropout, or short-batch count. The sparse mixer is connected, but current embeddings still lack a recall-safe class1 surface/boundary cue; revisit local convolution only with a new fold-safe target, initialized local selector, or explicit class1 recall protection.
- [ ] Next TRKH-native loop should prioritize a genuinely new representation/reliability signal over another head/loss/router tweak on the current embedding. The strongest live hook remains runtime patch-verifier softboost as a conservative FP diagnostic, but the next trainable change must improve class1 FN top2 support without widening `0/4/2->1`.
- [x] 2026-07-05 implemented default-off teacher-guided contrastive memory queue: FIFO detached cross-batch embeddings per source, queue size/min-count config and V8 flags, replay-safe no-double-enqueue behavior, and history telemetry for memory loss/anchors/positives/size. Preflight passed `py_compile train.py/config.py`, launcher dry-run, and `tests\test_teacher_guided_contrastive.py -q` (`6 passed`).
- [x] 2026-07-05 micro functional check `runs\micro_v8_yolof_tgcmemory_head_q512_8b_1e_20260705`: queue telemetry active (`memory_anchor_count=3.0`, `memory_positive_count=23.0`, `memory_loss=0.3087`), so implementation works; micro val was only 2 batches and not used as a gate.
- [x] 2026-07-05 full-val smoke teacher-guided contrastive memory queue, resume keeper, source `head`, queue `512`, loss `0.004`, confidence-margin teacher agreement, `60b/1e`, full `yolo_f/val=2606`, `SkipFinalTest`: validation macro/class1 only `0.8777/0.6592`, class1 P/R `0.5735/0.7748`, below keeper `0.8847/0.6860` and runtime verifier `0.8887/0.7030`.
- [x] 2026-07-05 boundary/XAI teacher-memory reject: boundary selected `153` rows (`FP1=76`, `FN1=34`, buckets `0->1=45`, `1->0=16`, `1->2=12`, `4->1=15`, `3->2=27`); XAI cases12 background blur/gray drops `0.0022/0.0010` vs object_desaturate `0.1448`, with rollout background/border `6/12`/`5/12` and Grad-CAM border `4/12`.
- [x] Cleanup teacher-guided memory contrastive reject: compact evidence saved in `runs\evidence_teacher_guided_memory_contrastive_rejected_20260705` (`3.153 MB`), excluding rejected checkpoints; raw smoke/eval/boundary/XAI/micro/log dirs deleted via `runs\cleanup_manifest_20260705_teacher_guided_memory_contrastive_rejected.json`, reclaimed `378.432 MB`.
- [ ] Do not repeat nearby teacher-guided contrastive memory queue variants on the current keeper by only changing queue size, temperature, source=head weight, teacher confidence threshold, or short batch count. Queue is functional but selected anchors are sparse and it widens class1 false positives. Revisit contrastive only with a stronger fold-safe clean-anchor policy, explicit class1 recall protection, or train-only diagnostics proving better class1 positive support.
- [ ] Next TRKH-native loop: look for a representation signal that specifically improves class1 FN top2 support without widening `0/2/4->1`; avoid another selector or contrastive variant over the same current embeddings unless a new fold-safe reliability source changes the anchor set.
- [x] 2026-07-05 checked class-1 intra-class foreground part swap using existing `foreground_snapmix` in same-class mode `ForegroundSnapmixPairs=1-1`, motivated by intra-class part swapping but without raw-data edits or test use. Preflight passed `py_compile train.py/config.py`, `tests/test_foreground_snapmix.py -q`, and V8 dry-run with explicit `yolo_f`, `crop_bbox`, empty manifests, and `SkipFinalTest`.
- [x] 2026-07-05 smoke intra-class class1 SnapMix `ForegroundSnapmixLossWeight=0.006/Probability=0.65/Area=0.08-0.18/BboxMargin=0.01/60b/1e/full yolo_f val`: trainer macro/class1 only `0.8802/0.6648`, class1 P/R `0.5821/0.7748`; reload eval full support `2606` confirmed macro/class1 about `0.8799/0.6648`, below keeper and runtime verifier.
- [x] 2026-07-05 boundary/XAI intra-class class1 SnapMix reject: boundary selected `153` rows with `FP1=84`, `FN1=34`, `0->1=55`, `4->1=16`, `2->1=9`, `1->0=16`, `1->2=12`. FN XAI background blur/gray drops stayed near zero (`0.0019/0.0006`) vs object desaturate `0.1480`; FP1 XAI cases8 had `object_color_sensitive=8/8`, Grad-CAM border `6/8`, and object desaturate `0.2417`, confirming false-positive expansion from surface/border cues.
- [x] Cleanup intra-class class1 SnapMix reject: compact evidence saved in `runs\evidence_intraclass1_foreground_snapmix_rejected_20260705`; deleted raw smoke/eval/boundary/FN-XAI/FP1-XAI dirs via `runs\cleanup_manifest_20260705_intraclass1_foreground_snapmix_rejected.json`, reclaimed `201.591 MB`.
- [ ] Do not repeat nearby same-class foreground SnapMix / intra-class class1 part-swap variants on the current keeper by only changing `ForegroundSnapmixPairs=1-1`, loss weight, probability, patch area, bbox margin, or short-batch count. It preserves labels but expands class1 false positives; any foreground part-swap revisit needs explicit conservative FP control plus true-class1 recall protection.
- [ ] Next TRKH-native loop should avoid generic foreground-part augmentation and another current-embedding selector. Prefer a fold-safe/case-level reliability target or a representation source that directly increases class1 FN top2 support while measuring `0/2/4->1` FP risk before smoke.
- [x] 2026-07-05 implemented TRKH-native default-off `foreground_counterexample_mix`: source class `1` foreground patch pasted into target classes `0,2,4` while preserving target labels, with config/train CLI/V8 flags, SAM replay pass-through, telemetry, and focused tests in `tests/test_foreground_snapmix.py`.
- [x] 2026-07-05 preflight/dry-run for counterexample mix passed: `py_compile train.py/config.py`, focused pytest `4 passed`, V8 dry-run confirmed explicit `yolo_f`, `crop_bbox`, keeper resume, empty manifests, teacher-focus-binary, and `SkipFinalTest`.
- [x] 2026-07-05 smoke counterexample mix rejected: `runs\smoke_v8_yolof_fgcountermix_src1_neg024_w006_60b_1e_20260705`, loss `0.006`, prob `0.50`, area `0.04-0.10`, `60b/1e`, full `yolo_f/val=2606`; telemetry active (`count=9.68/batch`, fraction `0.303`), but validation macro/class1 only `0.8773/0.6534`, class1 P/R `0.5721/0.7616`, below keeper/runtime verifier.
- [x] 2026-07-05 reload/boundary/XAI counterexample mix: reload eval reproduced `0.8773/0.6534`; boundary selected `153` from `354` with `FP1=86`, `FN1=36`, `0->1=55`, `4->1=16`, `2->1=11`, `1->0=18`, `1->2=12`. XAI FN6/FP6 showed object_desaturate drop `0.2098` vs background blur/gray `0.0088/0.0073`, Grad-CAM border `8/12`, object_color_sensitive `11/12`.
- [x] Cleanup counterexample mix reject: compact evidence saved in `runs\evidence_fgcountermix_src1_neg024_w006_rejected_20260705`; raw smoke/eval/boundary/XAI/log dirs deleted via `runs\cleanup_manifest_20260705_fgcountermix_src1_neg024_rejected.json`, reclaimed `195.375 MB`.
- [ ] Do not repeat nearby foreground counterexample mix variants on the current keeper by only changing source/target probability, loss weight, patch area, bbox margin, or short-batch count. Label-safe hard-negative patches still widened class1 false positives and weakened recall; any revisit needs fold-safe FP/recalibration evidence plus explicit true-class1 recall protection.
- [x] 2026-07-05 exposed `RegisterDiversityLossWeight` in V8 launcher because it was hardcoded to `0.0`; preflight passed `py_compile`, focused register-diversity pytest, and dry-run confirmed `RegisterDiversityLossWeight=0.003`, explicit `yolo_f`, `crop_bbox`, keeper resume, teacher-focus-binary, and `SkipFinalTest`.
- [x] 2026-07-05 smoke register-diversity-only rejected: `60b/1e`, full `yolo_f/val=2606`, no test; validation macro/class1 `0.8770/0.6554`, class1 P/R `0.5714/0.7682`, reload macro `0.8764`, below keeper `0.8847/0.6860` and runtime verifier `0.8887/0.7030`.
- [x] 2026-07-05 boundary/XAI register-diversity reject: boundary selected `153` from `351` with `FP1=88`, `FN1=35`, buckets `0->1=57`, `4->1=16`, `2->1=11`, `1->0=17`, `1->2=12`; XAI FN6/FP6 had background blur/gray `0.0092/0.0077` vs object_desaturate `0.2105`, `object_color_sensitive=11/12`, Grad-CAM border `8/12`, register similarity `0.99945`.
- [x] Cleanup register-diversity reject: compact evidence saved in `runs\evidence_registerdiversity_w003_rejected_20260705`; raw smoke/eval/boundary/XAI/log dirs deleted; manifest `runs\cleanup_manifest_20260705_registerdiversity_w003_rejected.json` records reclaimed MB as `null` because the first manifest write failed after deletion.
- [ ] Do not repeat nearby register-diversity-only variants on the current keeper by only changing loss weight or short-batch count. It did not break register collapse usefully and widened class1 false positives; any revisit needs a different register target with explicit true-class1 recall protection.
- [ ] Next TRKH-native loop should avoid foreground patch augmentation and current-embedding selectors. A useful next step must either create new recall-supporting representation for class1 FN cases or use a fold-safe/manual reliability source that can be smoke-gated against both `0/2/4->1` FP and `1->0/2/4` FN.
- [x] 2026-07-05 answered scope concern while continuing TRKH: external/AIDT/top5/timm runs remain diagnostic/teacher/upper-bound only; implementation target is still TRKH-native/no-pretrain unless the research goal changes explicitly.
- [x] 2026-07-05 preflight for auxiliary `class_f` multi-pair patch-evidence diagnostic passed: `py_compile probe_patch_evidence_mil.py probe_pairwise_feature_verifier.py model.py` and focused pytest `tests/test_patch_evidence_mil.py tests/test_pairwise_verifier_model_export.py -q` (`8 passed`).
- [x] 2026-07-05 diagnostic `yolo_f/train + class_f/train(weight=0.25)` multi-pair patch verifier rejected: train OOF looked strong (`97` corrections, `3` harms), but full `yolo_f/val=2606` dropped from base macro/class1 `0.8841/0.6841` to `0.8836/0.6835`. Class1 precision rose to `0.6545` but recall fell to `0.7152`; val changes were `34` corrections, `32` harms, `7` neutral.
- [x] 2026-07-05 XAI for aux-classf multi-pair changed12 `runs\xai_patch_evidence_auxclassf_pairs_changed12_20260705`: foreground mass high, background blur/gray drops only `0.0082/0.0071`, object_desaturate `0.0748`, `near_tie=5/12`, rollout border `8/12`, register similarity `0.999792`; overlays stayed fruit surface/border/glare/crop-edge driven, not wide-background driven.
- [x] Cleanup aux-classf multi-pair patch-evidence reject: compact metrics/verifier params/logs copied under `runs\xai_patch_evidence_auxclassf_pairs_changed12_20260705\source_patch_evidence_metrics` and `source_logs`; raw probe dir deleted via `runs\cleanup_manifest_20260705_auxclassf_multipair_patch_evidence_rejected.json`, reclaimed `4.234 MB`.
- [ ] Do not repeat auxiliary `class_f/train` patch-evidence fitting on the current keeper by only changing pair list, auxiliary weight, source-domain flag, threshold, margin, or top-k. It overfits train/OOF and hurts val recall; any future class_f+yolo_f combination needs a new cross-view representation objective or fold-safe reliability target with explicit true-class1 recall protection.
- [ ] Next TRKH-native loop should avoid both foreground patch augmentation and current-embedding verifier selectors. Before another smoke, identify a representation change or fold-safe reliability source that can improve class1 FN top2 support while guarding `0/2/4->1` false positives on full `yolo_f/val`.
- [x] 2026-07-05 scope diagnostic `runs\diagnostic_external_scope_trkh_remaining_errors_20260705`: TRKH softboost base `0.8887/0.7030`, AIDT aligned val `0.9088/0.7169` corrected `110/203` softboost errors but harmed `54` correct rows; top5 aligned val `0.9139/0.7383` corrected `97` but harmed `33`. TRKH+top5 diagnostic val ensemble reaches `0.9166/0.7500`, but it is external/pretrained validation evidence, not the no-pretrain TRKH target.
- [ ] Keep external/AIDT/top5/timm models as diagnostic/teacher/upper-bound only. Do not pivot final target away from TRKH and do not run direct KD/sample-weight/router compression from their validation superiority unless a fold-safe/OOF reliability signal explicitly protects true class1 recall and suppresses `0/2/4->1` false positives.
- [x] 2026-07-05 cleanup compacted image-heavy case folders from `20` obsolete rejected `xai_smoke_v8_yolof_*` audits while preserving root `xai_audit_summary.json`, metrics, CSV/MD audit files, and source metrics. Manifest `runs\cleanup_manifest_20260705_compact_obsolete_xai_case_images.json` reclaimed `1445.987 MB`; keeper, runtime softboost verifier, latest forensics/XAI, external scope diagnostic, and aux-classf XAI were protected.
- [x] 2026-07-05 gradient feature/token-injection precheck `runs\diagnostic_gradient_feature_readout_yolof_keeper_20260705`: no test/raw-data edit; train-only GroupKFold Sobel/edge readouts over bbox crops. `gradient_only` val `0.5952/0.1972`; `baseprob_gradient` `0.8762/0.6540`; best `softprob_gradient` only `0.8799/0.6710`, below current softboost `0.8887/0.7030`, with `25` corrections vs `35` harms and damaging transitions `3->2=19`, `1->2=12`, `1->0=8`.
- [ ] Do not implement or repeat simple Sobel/gradient readout, gradient-token injection, or shallow edge-stat branch on the current keeper. Raw edge/texture summaries are not a safe complementary signal and hurt true class1 recall; revisit only with a materially different fold-safe target and explicit `0/2/4->1` FP plus true-class1 recall protection.
- [x] 2026-07-05 added and ran border/interior counterfactual TTA diagnostic `trkh.tools.probe_border_interior_tta` on full `yolo_f/val=2606` with runtime patch softboost verifier, no test/raw-data edit. Clean recheck `0.8889/0.7052`; `border_suppressed` `0.8653/0.6319`, `core_only` `0.8652/0.6295`, `crop_edge_suppressed` `0.8625/0.6070`, `interior_suppressed` `0.1489/0.0000`; fixed averages stayed below clean (`avg_clean_core_only` `0.8791/0.6624`).
- [x] 2026-07-05 XAI for border/interior TTA changed cases `runs\xai_border_interior_tta_avgcore_changed12_20260705`: foreground mass stayed high, rollout border `7/12`, near-tie `5/12`, background blur/gray drops `0.0228/0.0272` vs object-desaturate drop `0.1003`; overlays show fruit edge/stem/pad/crop-border/surface spots. Smoke64 artifact was deleted via `runs\cleanup_manifest_20260705_border_interior_tta_smoke64.json`.
- [ ] Do not repeat fixed border/core/crop-edge TTA, simple border-suppression augmentation, or train-time border erasure on the current keeper by only changing mask radius/blur strength/average weight. These masks remove true class1 evidence faster than they remove false-positive evidence; revisit only with a new fold-safe reliability target that protects true class1 recall while suppressing `0/2/4->1`.
- [x] 2026-07-05 fixed TRKH auxiliary surface forward consistency: `_surface_counterfactual_consistency_loss` and `_surface_amplified_supervised_losses` now select/pass `image_valid_mask`, `bbox_metadata`, and `bbox_token_prior` into auxiliary `_forward_model_outputs`; regression tests added. Preflight passed `py_compile` and `tests/test_surface_amplified_supervised_loss.py -q` (`4 passed`).
- [x] 2026-07-05 smoke bbox-compatible surface-amplified supervised/boundary retest: `SurfaceAmplifiedSupervisedLossWeight=0.004`, boundary `0.003`, probability `0.50`, strength `0.18`, resume keeper, crop-bbox prior, full `yolo_f/val=2606`, `SkipFinalTest`. Validation only macro/class1 `0.8770/0.6572`, class1 P/R `0.5743/0.7682`, below keeper `0.8847/0.6860` and runtime verifier `0.8887/0.7030`; class1 FP widened (`0->1=53`, `4->1=16`, `2->1=13`).
- [x] 2026-07-05 XAI/cleanup bbox-compatible surface-amplified reject: `runs\xai_smoke_v8_yolof_surfaceamp_bboxcompat_val_cases12_robust_20260705` selected 12 full-val cases; `object_color_sensitive=9/12`, `gradcam_border=8/12`, background blur/gray drops only `0.0148/0.0163` vs object_desaturate `0.1883`. Smoke checkpoint/logs deleted after copying source metrics; manifest `runs\cleanup_manifest_20260705_surfaceamp_bboxcompat_rejected_smoke.json` reclaimed `183.154 MB`.
- [ ] Do not repeat nearby surface-amplified or surface-counterfactual smokes on the current keeper by only changing loss weights, probability, blur, strength, boundary margin, or short-batch count. The bbox metadata bugfix is kept, but the corrected recipe still worsens class1 precision/recall balance; revisit only with a new fold-safe reliability target that protects true class1 recall and suppresses `0/2/4->1`.
- [x] 2026-07-05 compacted more obsolete V8 XAI smoke case images: manifest `runs\cleanup_manifest_20260705_compact_more_obsolete_xai_case_images.json` deleted only `case_*` subdirectories from 64 documented/rejected `xai_smoke_v8_yolof_*` folders dated 20260702-20260704, `776` case dirs total, reclaiming `3786.209 MB`. Root metrics/CSV/MD/source evidence stayed, current keeper/runtime softboost/remaining-error/current surfaceamp XAI were verified present.
- [x] 2026-07-05 added reproducible diagnostic `trkh.tools.analyze_remaining_error_external_support` plus regression test; preflight passed `py_compile` and `tests/test_analyze_remaining_error_external_support.py -q` (`1 passed`). The tool joins full-val support predictions by `sample_index`, not path, and writes `summary.json`, `remaining_error_external_support.csv`, README, and contact sheets.
- [x] 2026-07-05 diagnostic `runs\diagnostic_remaining_error_external_support_20260705`: base TRKH softboost `0.8887/0.7030`; external AIDT `0.9088/0.7169`, top5 `0.9139/0.7383`, EffV2 readout `0.8876/0.6454`, DINOv2 readout `0.8449/0.5260`. Of `203` remaining TRKH errors, external correct-count distribution is `0:56`, `1:39`, `2:26`, `3:21`, `4:61`.
- [x] 2026-07-05 class1 support audit: among `35` class1 FN, `13` have zero external correction support and only `9` have a correct >=2-vote external consensus; among `63` class1 FP, `50` have at least one external blocker and `22` are blocked correctly by all four. General external consensus route is val-only upper-bound `0.9131/0.7393` with `88` corrections vs `27` harms; focus-only consensus hurts class1 (`0.8901/0.6993`, `51` corrections vs `42` harms).
- [x] 2026-07-05 visual audit contact sheets in `runs\diagnostic_remaining_error_external_support_20260705`: no-support class1 FN look like hard green/yellow/defect label-boundary cases also missed by pretrained; all-external-correct class1 FP show surface spots, yellow lighting, green texture, stem/edge/crop context that TRKH overreads as class1. This reinforces representation/label-boundary bottleneck, not wide-background failure.
- [ ] Do not compress validation external consensus into router/KD/sample-weight/targeted-margin. It is useful upper-bound evidence but not a TRKH-native/fold-safe training signal, and the focus-only route already drops class1 below current softboost. Any future external transfer must be train-only/OOF and explicitly protect true class1 recall while suppressing `0/2/4->1`.
- [ ] Next TRKH-native loop should not be another validation-support selector, generic part/token/background repeat, or shallow surface/edge branch. Look for a fold-safe reliability source or a genuinely new representation objective that increases class1 FN top2 support; smoke only if it has a clear guard against widening class1 FP.
- [x] 2026-07-05 train-only OOF support check `runs\diagnostic_train_oof_support_cleanlab_patchqueue_20260705`: intersected patch-OOF cleanlab queue with EffV2-S and DINOv2-small train OOF readouts. Among `81` flagged issue rows (`70` `0->1`, `11` `1->0`), both OOF sources supported cleanlab suggested class only `4` times, both kept hard label `19` times, and disagreed `58` times; `0->1` support was only `3/70`, `1->0` only `1/11`.
- [ ] Do not create auto soft-target/sample-weight/targeted-margin policy by relaxing current patch-cleanlab + EffV2/DINO OOF thresholds. The only fold-safe support available here is too sparse and disagreement-heavy, matching the prior multi-bank `strict_rows=0` diagnostic.
- [x] 2026-07-05 generated train-only EffV2+DINO OOF reliability targeted-margin manifest `runs\oof_external_reliability_margin_policy_t07_20260705`: `166` sample-index rows (`55` FP1 suppressors, `110` near-boundary TP1 recall protectors, `1` FN1 rescue), no validation/test/raw-data edit.
- [x] 2026-07-05 smoke `smoke_v8_yolof_oofextrel_t07_marginrecall_60b_1e_20260705`, full `yolo_f/val=2606`, teacher-focus-binary `0.015`, targeted-margin weight `0.002`, `SkipFinalTest`: validation macro/class1 only `0.8779/0.6629`, class1 P/R `0.5756/0.7815`, below keeper `0.8847/0.6860` and runtime softboost `0.8887/0.7030`.
- [x] 2026-07-05 reload/boundary/XAI for OOF external margin recall-protect: reload macro/class1 about `0.8771/0.6629`; boundary selected `153` rows (`FP1=40`, `FN1=33`, `0-1=80`), confusions stayed `0->1=53-54`, `4->1=16`, `2->1=13`, `1->0=15`, `1->2=12`, `1->4=6`. XAI cases12 showed background blur/gray near zero (`0.0013/-0.0000`) vs object_desaturate `0.1425`, rollout background/border `7/12`/`6/12`.
- [x] Cleanup OOF external margin recall-protect reject: compact evidence saved in `runs\evidence_oofextrel_t07_marginrecall_rejected_20260705`; raw smoke/eval/boundary/XAI dirs deleted via `runs\cleanup_manifest_20260705_oofextrel_t07_marginrecall_rejected.json`, reclaimed about `193.542 MB`.
- [ ] Do not repeat nearby EffV2/DINO train-OOF targeted-margin policies on the current keeper by only changing threshold, FP/TP ratio, margin, loss weight, or short-batch count. Even explicit TP1 recall protection did not stop class1 FP widening; future external transfer needs a stronger fold-safe representation signal or manual train-only review, not another sparse margin over current embeddings.
- [x] 2026-07-05 TRKH-only internal ensemble + runtime softboost diagnostic `runs\diagnostic_trkh_internal_ensemble_softboost_blend_20260705`: full `yolo_f/val=2606`, no test/raw-data edit/no train manifest. Softboost baseline `0.8887/0.7030`, internal no-pretrain ensemble `0.8864/0.6939`; best validation log blend `alpha=0.50` only `0.8903/0.7091`, changing 6 rows (`4` corrections, `2` harms).
- [x] XAI for internal-blend changed6 `runs\xai_trkh_internal_ensemble_softboost_blend_changed6_20260705`: `near_tie_top2=4/6`, Grad-CAM border `3/6`, rollout border `3/6`, background blur/gray drops `0.0061/-0.0098` vs object_desaturate `0.0747`; still foreground surface/border ambiguity.
- [ ] Do not sweep more validation-selected internal TRKH ensemble + softboost weights or train a router/KD/sample-weight policy from the 6 changed validation rows. The gain is too small and near-tie-driven; next route must add a new representation or fold-safe recall/FP reliability signal rather than another validation blend.
- [x] 2026-07-05 compacted obsolete non-smoke XAI case images from 24 documented rejected/superseded roots: manifest `runs\cleanup_manifest_20260705_compact_obsolete_non_smoke_xai_case_images.json`, `311` `case_*` dirs deleted, `1801.301 MB` reclaimed. Root metrics/summaries/source evidence retained; current keeper, runtime softboost, remaining-error, boundary-review, and final-test XAI protected and verified present.
- [ ] Do not regenerate compacted obsolete XAI case images unless a new route explicitly needs side-by-side visual evidence. Prefer retained summaries/metrics/CSVs for historical decisions, and continue protecting current softboost/remaining-error/final-test evidence before deleting any run artifact.
- [x] 2026-07-05 scope/status clarification: active target is still TRKH-native/no-pretrain. AIDT/top-5/timm/pretrained checks remain diagnostic/teacher/upper-bound evidence only; do not pivot the final target away from TRKH unless explicitly requested.
- [x] 2026-07-05 literature cross-check before another smoke: recent noisy fine-grained/VLM-cleaning/co-transformer ideas map mostly to already-tested local families (sample selection, contrastive/co-teaching, external teacher support). Treat CLIP/VLM-style cleaning as a possible train-only/manual-review diagnostic, not an automatic policy or final pretrained route.
- [x] 2026-07-05 train-only current softboost eval `runs\eval_patch_linear_verifier_softboost001_train_20260705`: explicit `yolo_f/train=9215`, runtime patch linear verifier softboost, no test/raw-data edit. Train macro F1 `0.9478`; class1 P/R/F1 `0.7389/0.9834/0.8438`. Use only as in-sample review source.
- [x] 2026-07-05 train boundary review `runs\boundary_review_softboost_train_current_20260705`: selected `260/9215` rows from `1171` candidates, dominated by `focus_false_positive=120` versus `focus_false_negative=9`; major buckets `0->1=90`, `2->1=27`, `3->2=31`, `1->0=8`. Report HTML and 126 copied review images are preserved.
- [x] 2026-07-05 verifier-aware train-review XAI `runs\xai_softboost_train_focus_fpfn12_20260705`: 12 selected class1 FP/FN rows, foreground mass high (`attention/gradcam=0.9055/0.9321`), rollout border `9/12`, object-color sensitive `6/12`; background blur/gray drops `0.0341/0.0161` versus object_desaturate `0.1479`. Evidence again points to surface/border/label-boundary ambiguity, not a broad-background shortcut.
- [x] 2026-07-05 visual triage batch `runs\manual_softboost_train_triage_batch01_20260705`: selected `58` train-only rows from the softboost boundary queue (`9` class1 FN, `30` high-confidence `0->1`, `12` high-confidence `2->1`, `3` other non1-to-1 FP, `4` low-margin anchors). Contact sheets show true class1 recall protectors are pale/glare/shadow/subtle-spot boundary cases, while many `0/2->1` false positives contain real class1-like fruit-surface cues rather than broad background shortcuts.
- [ ] Keep `manual_softboost_train_triage_batch01_20260705` review-only. Its `manual_label_status` fields are intentionally empty (`0/58` filled); do not feed it to `build_reviewed_boundary_training_manifests.py`, auto-relabel, sample-weight, targeted-margin, or smoke from it until explicit reviewed status fields and recall/FP coverage are added and dry-run sample-index matching passes.
- [x] 2026-07-05 selfreview draft dry-run `runs\manual_softboost_train_triage_batch01_selfreview_draft_20260705`: generated a separate draft CSV from contact-sheet review and ran the reviewed-manifest builder with `dry_run=True`. It matched all `58` rows by `sample_index` and would produce `50` targeted-margin rows (`1->0=8`, `0->1=30`, `2->1=12`) plus `4` needs-crop downweights, but no trainable CSV outputs were written.
- [ ] Do not smoke from the selfreview draft. It proves the manual-policy path is reproducible, but the draft is too small and imbalanced (`8` true-class1 recall protectors vs `42` class1-FP suppressors), which repeats the known risk of targeted-margin policies suppressing true class 1.
- [x] 2026-07-05 batch02 balanced train review candidates `runs\manual_softboost_train_balanced_reliability_batch02_20260705`: generated `165` train-only rows from current softboost train predictions, with `69` class1 recall-side rows (`9` FN + `60` low-margin TP1), `72` class1-FP suppressor candidates (`36` `0->1`, `24` `2->1`, `12` complex `3/4->1`), and `24` low-margin non1 anchors. Manual fields remain empty and raw data is unchanged.
- [x] 2026-07-05 added optional `manual_sample_weight` / `review_sample_weight` support to `build_reviewed_boundary_training_manifests.py` so reviewed TP1 recall protectors can be represented as light sample weights instead of being inert `keep` rows. Existing manifests are unchanged when the column is absent; quality flags still cap overrides. Preflight passed `py_compile` and `tests\test_reviewed_boundary_training_manifests.py -q` (`4 passed`).
- [x] 2026-07-05 batch02 selfreview dry-run `runs\manual_softboost_train_balanced_reliability_batch02_selfreview_dryrun_20260705`: builder `dry_run=True`, no trainable CSV outputs. Hypothetical policy: `60` TP1 recall weights `1.05`, `8` FN1 `1->0` margins at weight `1.08`, `60` light FP suppressor margins (`0->1=36`, `2->1=24`, sample weight `0.97`, targeted-margin weight `0.0007`), `13` needs-crop rows, `24` anchors.
- [x] 2026-07-05 batch02 similarity-conflict diagnostic `runs\manual_softboost_train_batch02_similarity_conflicts_20260705`: train-only aHash/dHash/RGB clustering over the same `165` review rows found `133` clusters, `21` multi-item clusters, `10` mixed-target clusters, and `6` mixed recall/suppressor clusters covering `20` rows. Contact sheets confirm near-duplicate or presentation-similar true class1 and `0/2/4->1` suppressor cases.
- [x] 2026-07-05 strict batch02 review filter `runs\manual_softboost_train_batch02_strict_review_filter_20260705`: removed mixed recall/suppressor clusters, mixed-target clusters, complex `3/4->1` rows, and one ultra-low-margin uncertain FN. It keeps `125/165` rows (`56` recall protectors, `50` FP suppressors, `19` anchors) and rejects `40`; no trainable manifest is created.
- [ ] Do not smoke batch02 or its strict filter. It is more balanced than batch01 and proves the builder can encode recall protectors, but contact sheets plus similarity clusters still show severe visual overlap between TP1 and `0/2/4->1` rows. Require explicit reviewed manual fields, approval, and a new dry-run before generating non-dry-run sample-weight/targeted-margin manifests.
- [x] 2026-07-05 structural-label/reverse-kNN precheck `runs\diagnostic_structural_labels_batch02_strict_20260705`: train-only top-30 forward/reverse kNN over `9215` keeper train embeddings joined to `125` strict batch02 rows. Structural class1 support is inverted: recall protectors mean blend `0.3488`, FP suppressors `0.6592`; AUC recall>suppression only `0.0996`. Reject structural-label loss/manifest on current keeper before smoke.
- [ ] Do not implement a Structural Labels / reverse-kNN objective on the current keeper using current embeddings. It would likely pull class1-like `0/2->1` suppressors further toward class1 and does not protect true class1 recall. Revisit only with stronger features or confirmed manual train labels.
- [x] 2026-07-05 external-feature structural precheck `runs\diagnostic_structural_labels_external_features_batch02_strict_20260705`: same-source-excluded top-30 support over strict batch02 shows EffV2-S separates recall protectors from FP suppressors strongly (`AUC=0.9957`, recall mean `0.8812`, suppressor mean `0.1538`), while DINOv2 direct/remap remain weak or inverted (`AUC=0.2998/0.2875`). Treat EffV2 as diagnostic/review-priority only.
- [x] 2026-07-05 full-val EffV2 structural support diagnostic `runs\diagnostic_effv2_structural_support_softboost_val_20260705`: fixed thresholds `0.35/0.55`, full `yolo_f/val=2606`, no test/no threshold sweep. Overall true1-vs-non1 AUC `0.8976`, but fixed guard changes `89` rows with `44` corrections, `41` harms, class1 F1 drops `0.7030 -> 0.6990`, macro `0.8887 -> 0.8882`.
- [ ] Do not convert EffV2-S structural support into TRKH KD/router/sample-weight/targeted-margin/structural-label loss on the current keeper. It is useful for manual review prioritization and representation diagnostics, but the fixed val guard is not recall-safe enough.
- [x] 2026-07-05 EffV2 structural manual-priority worklist `runs\manual_softboost_train_batch02_effv2_structural_review_priority_20260705`: from `125` strict rows, keeps `95` review candidates (`55` recall protectors with EffV2 p1 `>=0.55`, `40` FP suppressors with p1 `<=0.35`), rejects `30`, and creates contact sheets only. Manual fields remain empty; no trainable manifest.
- [x] 2026-07-05 added generic renderer `trkh.tools.render_review_worklist_html` plus focused test so non-boundary review worklists can render arbitrary selected columns and summary JSON without ad hoc scripts or trainable manifests.
- [x] 2026-07-05 EffV2 priority HTML review report `runs\manual_softboost_train_batch02_effv2_structural_review_priority_20260705\effv2_structural_priority_review_report.html`: rendered with `trkh.tools.render_review_worklist_html`; includes all `95` priority rows with images, transition, softboost score, EffV2 structural p1, review order, suggestions, and empty manual fields. Summary confirms `manual_label_status_filled=0/95` and `missing_images_rendered=0`.
- [ ] Keep the EffV2 structural priority worklist review-only. Do not pass it to `build_reviewed_boundary_training_manifests.py` until `manual_label_status` is explicitly filled and a new dry-run proves recall-balanced sample weights/margins.
- [x] 2026-07-05 literature re-check: imbalanced noisy-data sample selection, LNL-FG, transformer/part-based FGVC, and VLM-assisted dataset renovation support cautious review/fold-safe reliability targets, but do not justify rerunning generic small-loss, part-attention, VLM/teacher auto-cleaning, or current-embedding selector smokes without a new recall-protecting signal.
- [x] 2026-07-05 disk hygiene: compacted rejected `class_f` OOF fold materialization, keeping metadata only. Do not look for `runs\classf_oof_folds_train5_20260704\fold_*` as live folders; regenerate from source `class_f` data YAML and seed 42 if needed.
- [x] 2026-07-05 preservation audit `runs\artifact_inventory_current_trkh_20260705`: verified current keeper/final/runtime-verifier artifacts, `yolo_f` OOF folds, EffV2 priority review worklist, journal, and TODO all exist after cleanup; verified compacted `class_f` fold tree is intentionally absent while `summary.json` remains.
- [x] 2026-07-05 added `trkh.tools.audit_review_worklist_readiness` plus focused tests. It audits manual-fill, sample-index safety, train-only paths, actionable recall/FP coverage, and mixed-side clusters before any reviewed worklist is passed to manifest building.
- [x] 2026-07-05 EffV2 priority readiness audit `runs\manual_softboost_train_batch02_effv2_structural_review_priority_20260705\readiness_audit`: `95` rows, sample indices/path train OK, no mixed-side clusters, but `manual_label_status_filled=0/95`, actionable recall `0<20`, actionable FP `0<20`; `ready_for_training_manifest=false`.
- [x] 2026-07-05 review queue readiness matrix `runs\review_worklist_readiness_matrix_20260705`: audited boundary softboost train, triage batch01, balanced batch02, strict batch02, and EffV2 priority. `ready_count=0/5`; all queues are blocked by empty manual fields/actionable counts. Strict batch02 has `2` suppressor+anchor warning clusters but `0` blocking mixed actionable-side clusters.
- [x] 2026-07-05 builder CLI readiness guard: `trkh.tools.build_reviewed_boundary_training_manifests` now runs readiness audit before non-dry-run manifest generation, aborting if readiness fails unless `--skip-readiness-check` is explicit. Focused tests cover non-dry-run refusal and dry-run warning behavior; EffV2 priority guard check wrote readiness evidence and no trainable CSVs.
- [x] 2026-07-05 cleanup compacted rejected verifier/XAI case images: manifest `runs\cleanup_manifest_20260705_compact_rejected_xai_test_extratrees_case_images.json` deleted only `case_*` subdirectories from `xai_stacked_patch_feature_verifier_yolof_keeper01_class1benefit_test_changed12_20260703` and `xai_patch_evidence_extratrees_leaf5_reject_changed7_20260703`. Both roots keep `xai_audit_summary.json`, `xai_metrics.json`, and `xai_cases.csv`; protected runtime softboost, remaining-error XAI, final-test XAI/audit, and raw `class_f`/`yolo_f` roots were verified present. Manifest records `19` removed case dirs, `97.376 MB` measured case material, `97.824 MB` free-space delta, and `raw_dataset_touched=false`.
- [x] 2026-07-05 added DaSC-style soft-centroid prototype precheck to `trkh.tools.probe_embedding_prototypes` (`--soft-centroid-temperature`, `--soft-centroid-min-confidence`) plus focused tests. Diagnostic `runs\diagnostic_soft_centroid_yolof_keeper_trainprob_20260705` used full `yolo_f/train=9215` and `val=2606`, no test/raw-data edit/train manifest. Best soft centroid reached only val macro/class1 `0.8450/0.5455` versus base `0.8829/0.6783`; it changed `171` rows with `54` corrections but `103` harms, including true-class1 harms `1:1->0=26` and `1:1->2=11`.
- [ ] Do not convert current-embedding soft centroids / distribution-aware prototype estimates into a prototype loss, router, sample selection rule, KD target, or targeted-margin manifest. It suppresses some `0->1` false positives but destroys true class1 recall; centroid/prototype work needs a new representation or stronger fold-safe reliability source first.
- [x] 2026-07-05 added `trkh.tools.audit_hflip_decision_instability` plus focused tests, then audited existing hflip full-val against current softboost full-val in `runs\diagnostic_hflip_softboost_instability_20260705`: support `2606`, target mismatches `0`, clean `0.8829/0.6783`, hflip `0.8786/0.6553`, softboost `0.8887/0.7030`.
- [x] 2026-07-05 hflip instability rejected as a training signal: hflip changed only `59/2606` rows, corrected `28` current softboost errors but broke `39` softboost-correct rows; class1 FN rescue was only `7/35`, true-class1 correct flip breaks were `8`, and non1-correct rows created `15` clean-non1 to flip-class1 FP risks.
- [x] 2026-07-05 verifier-aware XAI for hflip instability: FN audit `runs\xai_hflip_softboost_instability_cases16_20260705` object_desaturate `0.0764` vs background blur/gray `0.0056/0.0035`; FP audit `runs\xai_hflip_softboost_class1fp_cases16_20260705` object_desaturate `0.1343` vs background blur/gray `0.0170/0.0169`, rollout border `11/16`.
- [ ] Do not enable hflip TTA aggregation, hflip consistency loss, or hflip-instability sample-weight/targeted-margin/router policy on the current keeper. This signal is not recall-safe and can create `0/2/4->1` false positives; revisit geometric equivariance only with a stronger fold-safe reliability signal.
- [x] 2026-07-05 cleanup compacted rejected hflip XAI case images: manifest `runs\cleanup_manifest_20260705_compact_rejected_hflip_xai_case_images.json` deleted only `32` `case_*` dirs from the two hflip XAI roots, measured `25.367 MiB`, free delta `27.337 MB`; root `xai_audit_summary.json`, `xai_metrics.json`, `xai_cases.csv`, `xai_audit.md` remain and protected softboost/final/raw dataset artifacts were verified.
- [x] 2026-07-05 added offline INTR-style class-specific patch-query diagnostic `trkh.tools.probe_class_specific_patch_query` plus focused tests. It fits a tiny class-query readout on frozen `yolo_f/train` patch tokens and evaluates full `yolo_f/val`, refusing test and writing no trainable manifests.
- [x] 2026-07-05 class-query full diagnostic `runs\diagnostic_classquery_patchreadout_full_base1_res05_20260705`: train `9215`, val `2606`, `max_patches=24`, `base_logit_scale=1.0`, `residual_logit_scale=0.5`, `query_epochs=8`. Base path `0.8841/0.6841` macro/class1; query readout collapsed to `0.7960/0.4426`, with `306` val changes, `74` corrections, `212` harms, `20` neutral. Main harms include true class1 `1:1->0=44` and `1:1->2=22`.
- [x] 2026-07-05 XAI class-query changed16 `runs\xai_classquery_patchreadout_full_changed16_20260705`: `12` true-class1 harms + `4` corrections; background blur/gray drops `-0.0003/0.0034`, object_desaturate `0.1336`, Grad-CAM border `9/16`, rollout border `11/16`, `cls_register_heatmap_similarity=0.99995`. Failure remains surface/border/foreground ambiguity, not a useful new class-query cue.
- [x] 2026-07-05 cleanup class-query reject: manifest `runs\cleanup_manifest_20260705_classquery_patchreadout_rejected_aux.json` deleted two small smoke precheck dirs and `16` XAI `case_*` dirs, measured `12.753 MiB`, free delta `13.109 MB`; full diagnostic summary/predictions and root XAI metrics preserved; protected softboost/final/raw dataset artifacts verified.
- [ ] Do not integrate or repeat INTR-style class-specific patch-query readout/head/router on the current frozen keeper patch tokens by only changing base/residual scale, query epochs, LR, temperature, or selected-patch count. It destroys true class1 recall; revisit only with a new trainable representation target or fold-safe/manual reliability source that explicitly protects class1 recall.
- [x] 2026-07-05 visual pass over EffV2 priority worklist contact sheets wrote `runs\manual_softboost_train_batch02_effv2_structural_review_priority_20260705\CODEX_VISUAL_PASS_20260705.md`; decision remains review-only because recall protectors and FP suppressors visually overlap too much for auto-filled labels.
- [ ] Do not auto-fill `manual_label_status` from Codex visual pass or train from EffV2 priority worklist. Require explicit reviewed fields plus readiness pass with recall-balanced actionable rows before any non-dry-run reviewed manifest.
- [ ] Do not train from `boundary_review_softboost_train_current_20260705` while manual review fields are empty, and do not convert it into auto relabel/sample-weight/targeted-margin policy. It is a train-only hard-case queue for future manual or fold-safe reliability work.
- [ ] Next TRKH-native loop should not be another logit gate, crop-context/background suppression, shallow color/edge stat branch, current-embedding contrastive queue, or patch-verifier threshold tweak. Launch a smoke only after identifying a new surface/boundary representation signal or fold-safe/manual reliability target with explicit true-class1 recall protection and conservative `0/2/4->1` FP control.
- [x] 2026-07-06 compacted recent documented rejected XAI case images: manifest `runs\cleanup_manifest_20260706_compact_rejected_recent_xai_case_images.json` deleted only `166` `case_*` dirs from `16` rejected/superseded roots, measured `89.454 MiB`, free delta `92.605 MiB`. Root summaries/metrics/CSV/MD/source evidence remain, `raw_dataset_touched=false`, and protected softboost/final/current XAI artifacts all exist.
- [ ] Do not regenerate the compacted recent rejected XAI case images unless a new side-by-side visual comparison explicitly needs them. The intentionally retained `case_*` roots are current/final evidence: softboost remaining-error, current boundary review, softboost train-review, runtime changed-case, final patch-evidence test, and verifier-aware softboost.
- [x] 2026-07-06 literature route audit: DeFT/CLIPCleaner/NLPrompt/Jo-SNC/CCT/Early-Cutting style methods mainly support VLM/manual/fold-safe clean-sample detection, neighbor consistency, or co-training sample selection. They do not justify another automatic KD/router/sample-weight/loss smoke on current TRKH embeddings while manual fields are empty and OOF support is sparse.
- [ ] Next automatic TRKH smoke remains gated: require a genuinely new fold-safe reliability source or representation target that protects true class1 recall and controls `0/2/4->1` false positives before spending train time. Current best implementation target stays runtime softboost as diagnostic guard plus manual/fold-safe reliability work, not another current-embedding selector.
- [x] 2026-07-06 correction-budget diagnostic `runs\diagnostic_softboost_class1_correction_budget_20260706`: validation-only, no test/raw-data edit/train manifest. Current softboost class1 is `TP=116, FP=63, FN=35`; class1 F1 `0.7030`. Minimum arbitrary correction budgets: `0.75` needs `13` FN rescues or equivalent, `0.80` needs `27`, `0.85` needs all `35` FN plus `10` FP suppressions, `0.90` needs all `35` FN plus `30` FP suppressions, and `0.98` needs all `35` FN plus `57/63` FP suppressions.
- [ ] Do not turn transition correction-budget rows into a validation-tuned router. Use the diagnostic only to size the next signal: a real route must fix many `1->0/1->2/1->4` false negatives or suppress many `0/2/3/4->1` false positives while avoiding reciprocal harms.
- [x] 2026-07-06 external support scope check `runs\diagnostic_current_trkh_external_support_20260706`: current TRKH softboost remaining errors `203/2606`; EffV2-S/DINOv2 val consensus is upper-bound only, reaching macro/class1 `0.9034/0.7237` with `67` corrections, `28` harms, and `4` wrong-to-wrong changes. Focus-only consensus changes `50` rows with `30` corrections and `18` harms.
- [x] Transition support from EffV2/DINO explains why external models look better but does not solve TRKH: consensus helps class1 FP suppression (`0->1=9/41`, `2->1=7/9`, `4->1=5/9`, `3->1=3/4`) but barely protects class1 FN recall (`1->0=5/19`, `1->2=0/11`, `1->4=1/5`).
- [ ] Keep testing AIDT/top5/EffV2/DINO only as diagnostic/teacher/upper-bound or manual-review priority. Do not pivot final work away from TRKH, and do not compress current external consensus into KD/router/sample-weight/targeted-margin smoke until a fold-safe/manual signal covers `1->2`/`1->4` recall and controls `0/2/4->1` false positives.
- [x] 2026-07-06 signal-gap readiness crosswalk `runs\diagnostic_trkh_signal_gap_readiness_20260706`: no test/raw-data edit/train manifest. Joined current transition budget with EffV2/DINO val support, train-only OOF support, active review queues, readiness matrix, and patch-OOF cleanlab multibank summary. Smoke gate is `false`.
- [x] Crosswalk blocker: `1->2` has `11` val errors but `0` val consensus-correct rows, `0` train-OOF rows, and `0` current review rows; `1->4` has `5` val errors, only `1` val consensus-correct row, `1` train-OOF row, and `2` review rows with `0` manual-filled. Review readiness remains `0/5`; patch-OOF cleanlab multibank `strict_rows=0`; external consensus upper-bound class1 `0.7237` is below the next `0.75` milestone.
- [ ] Do not launch a new automatic TRKH smoke from existing EffV2/DINO support, current review queues, patch-OOF cleanlab multibank policy, or their threshold variants. The next valid smoke requires a new fold-safe/manual reliability source or representation target that covers `1->2`/`1->4` recall while keeping `0/2/4->1` FP suppression.
- [x] 2026-07-06 literature mapping after crosswalk: SNSCL/LNL-FG, DULL partial unlearning, class-independent margin dual-space, and GFT gradient focal patch selection were checked against current local evidence. They map to already-rejected or currently blocked families: contrastive queues need clean anchors, partial unlearning/suppression needs trustworthy wrong-feature localization, OVA/CIR-lite was rejected, and gradient/patch selection lacks a recall-safe target.
- [ ] Do not treat SNSCL/DULL/class-independent-margin/GFT-style ideas as smoke-ready on the current keeper. Revisit only after a new fold-safe/manual target covers `1->2`/`1->4` recall and protects against `0/2/4->1` false positives.
- [x] 2026-07-06 class1 rival2/4 train-only review worklist `runs\manual_softboost_train_class1_rival24_gap_review_20260706`: selected `155` rows from current softboost train predictions, with `65` recall protectors, `42` FP suppressors, and `48` anchors. It adds worklist coverage for risk transitions `1->2=32`, `1->4=33`, `2->1=32`, and `4->1=10`; HTML report rendered all rows with `missing_images_rendered=0`.
- [x] Readiness audit for rival2/4 worklist: no raw data/test/train manifest; `manual_label_status_filled=0/155`, actionable recall/FP counts are `0`, and one mixed actionable source cluster `Image_4098` blocks training. `ready_for_training_manifest=false`.
- [ ] Keep `manual_softboost_train_class1_rival24_gap_review_20260706` review-only. Do not use it for auto relabel, sample weights, soft targets, targeted margins, or a smoke until explicit manual fields are filled, the mixed-side cluster is resolved, and readiness passes.
- [x] 2026-07-06 full-val rival2/4 forensic diagnostic `runs\diagnostic_softboost_val_rival24_forensics_20260706`: full `yolo_f/val=2606`, no test/raw-data edit/train manifest. Current softboost remains macro/class1 `0.8887/0.7030`; class1 one-vs-rest `TP=116, FP=63, FN=35`, P/R `0.6480/0.7682`. Pair subsets are comparatively strong (`1-2=0.9065`, `1-4=0.9238`), so the remaining weakness is class1 one-vs-rest boundary control, not a single isolated pair route.
- [x] 2026-07-06 validation XAI `runs\xai_softboost_val_rival24_cases16_20260706`: 16 validation-only case indices, 4 each for `1->2`, `1->4`, `2->1`, `4->1`, using the same runtime patch-verifier softboost parameters. Mean foreground mass is high (`attention=0.9266`, `gradcam=0.8910`); background blur/gray drops are tiny (`0.0009/0.0011`) while object desaturation drop is large (`0.1043`), with Grad-CAM border attention `12/16`.
- [ ] Do not use the rival2/4 validation XAI case list for threshold tuning, routing, sample weights, targeted margins, or training. It is audit evidence only and argues against another background/context/patch-threshold smoke; next progress needs a new surface/boundary representation target or reviewed/fold-safe reliability source.
- [x] 2026-07-06 preservation snapshot `runs\artifact_inventory_current_trkh_20260706`: verified current best checkpoint, current softboost val metrics/predictions, final no-pretrain test metrics, patch-verifier params, valid `class_f`/`yolo_f` YAMLs, rival2/4 review/forensics/XAI artifacts, journal/TODO/skill; `missing_count=0`, `D:` free `79.33 GB`.
- [ ] Do not delete the new rival2/4 forensic or XAI artifacts; they are current evidence. Future cleanup should continue to target only documented rejected/superseded case directories after summaries/metrics are preserved.
- [x] 2026-07-06 added reusable XAI transition summarizer `trkh.tools.summarize_xai_transitions` plus `tests\test_summarize_xai_transitions.py`. Preflight passed `py_compile` and focused pytest (`1 passed`). The tool reads existing `xai_audit_summary.json` files and writes transition CSV/JSON/contact sheets only; no raw-data edit, no trainable manifest.
- [x] 2026-07-06 ran summarizer on rival2/4 XAI: `runs\xai_softboost_val_rival24_cases16_20260706\transition_summary_tool`, summarizing `16` validation cases across `4` transitions with `48` rendered crop/Grad-CAM/rollout images and `missing_images=0`.
- [ ] Use `trkh.tools.summarize_xai_transitions` for future XAI transition audits instead of inline scripts. Keep its outputs diagnostic-only; do not treat validation transition summaries as threshold selectors, routers, sample weights, targeted margins, or smoke permission.
- [x] 2026-07-06 added reusable review readiness matrix builder `trkh.tools.build_review_worklist_readiness_matrix` plus `tests\test_build_review_worklist_readiness_matrix.py`. It calls `audit_review_worklist_readiness` for each config item and writes matrix summaries only; no raw-data edit, no trainable manifest, no test usage. Preflight passed `py_compile`; focused pytest with readiness tests passed (`6 passed`).
- [x] 2026-07-06 readiness matrix `runs\review_worklist_readiness_matrix_20260706`: audits six active train-only review queues including new rival2/4 worklist. `ready_count=0/6`, `blocked_count=6`, all manual fields are empty (`0/260`, `0/58`, `0/165`, `0/125`, `0/95`, `0/155` filled), and rival2/4 still has one blocking mixed actionable-side cluster.
- [ ] Do not create non-dry-run reviewed manifests, relabels, sample weights, soft targets, targeted margins, or a smoke from any current review queue until explicit manual fields are filled and the 2026-07-06 matrix or a newer matrix reports readiness.
- [x] 2026-07-06 added reusable smoke-gate auditor `trkh.tools.audit_trkh_smoke_gate` plus `tests\test_audit_trkh_smoke_gate.py`. Preflight passed `py_compile` and focused pytest (`2 passed`). The tool reads existing diagnostic summaries and writes only gate `summary.json`/README; no raw-data edit, no trainable manifest, no test usage.
- [x] 2026-07-06 gate audit `runs\audit_trkh_smoke_gate_current_20260706`: `smoke_gate_ready=false`. Current class1 F1 is `0.7030`; next `0.75` milestone needs at least `13` effective corrections. Review readiness is `0/6` with `0/858` manual fields filled; external consensus class1 is only `0.7237`; `1->2` and `1->4` recall coverage remains insufficient; XAI is surface/boundary-dominant rather than background-context-driven.
- [ ] Do not launch automatic smoke/full train from current external consensus, current review queues, patch-threshold variants, or validation XAI case indices. Open the next TRKH-native smoke only after a new surface/boundary representation target or fold-safe/manual reliability target is available and the gate audit is rerun.
- [x] 2026-07-06 cleanup rejected eval-smoke leftovers: compacted `14` documented rejected/superseded `eval_smoke_*` dirs into `runs\evidence_eval_smoke_rejected_leftovers_20260706`, copying `metrics.json`/`metrics_detailed.json`, then deleted the original eval dirs. Manifest `runs\cleanup_manifest_20260706_eval_smoke_rejected_leftovers.json`; measured payload `39.238 MB`; no raw data/test tuning/trainable manifest touched.
- [x] Post-cleanup verification: no `runs\eval_smoke_*` directories remain; current best checkpoint, current softboost full-val, final no-pretrain test, fixed patch-evidence final summary, rival2/4 artifacts, gate audit, readiness matrix, and `yolof_oof_folds_train5_20260704` still exist.
- [ ] Do not look for the compacted `eval_smoke_*` directories as live evidence. Use `runs\evidence_eval_smoke_rejected_leftovers_20260706` and the journal/TODO metrics if a future comparison needs their rejected results.
- [x] 2026-07-06 cleanup legacy MobileNetV3 `class_f` OOF fold train dirs: compacted metrics/history/logs from `runs\oof_mobilenetv3_fold00_2e120b_20260627` through `fold_04` into `runs\evidence_mobilenetv3_classf_oof_fold_train_dirs_20260706`, then deleted the five per-fold checkpoint dirs. Manifest `runs\cleanup_manifest_20260706_mobilenetv3_classf_oof_fold_train_dirs.json`; measured payload `83.958 MB`.
- [x] Post-cleanup verification: aggregate `runs\oof_mobilenetv3_5fold_2e120b_20260627\summary.json` and `predictions_train_oof.csv`, old cleanlab outputs, current `yolof_oof_folds_train5_20260704`, best checkpoint, current softboost val, final no-pretrain test, and gate audit still exist.
- [ ] Do not use the deleted MobileNetV3 per-fold checkpoint dirs as live evidence. Historical comparison should use `runs\evidence_mobilenetv3_classf_oof_fold_train_dirs_20260706` plus aggregate `oof_mobilenetv3_5fold_2e120b_20260627`.
- [x] 2026-07-06 retention audit `runs\artifact_retention_audit_current_trkh_20260706`: scanned `500` run directories, deleted nothing, and classified current keepers/caches/history. Largest live artifact is `runs\yolof_oof_folds_train5_20260704` (`2547.804 MB`) and must be retained for fold-safe diagnostics; best no-pretrain anchor remains protected (`347.008 MB`).
- [x] Retention verification: protected current `yolo_f` OOF summary, best checkpoint, current softboost full-val metrics, final no-pretrain test metrics, gate audit, and review readiness matrix all exist; no `runs\eval_smoke_*` or `runs\oof_mobilenetv3_fold*` dirs remain.
- [ ] Treat `future_cleanup_review_candidates` in the retention audit as a review queue only. It flags `34` historical candidates (`484.105 MB`), but each needs a separate preservation/docs check before any deletion.
- [x] 2026-07-06 added manual-review minimum fill planner `trkh.tools.build_review_minimum_fill_plan` plus focused test. It reads the current readiness matrix and smoke gate, writes only review-planning artifacts, and records no raw-data edit, no test use, no trainable manifest, no auto-labeling, and no smoke permission.
- [x] Fill-plan artifact `runs\review_minimum_fill_plan_current_20260706`: selected `360` priority review rows across six active train-only queues (`184` unique sample indices). It confirms the older broad queues have only `9` recall-side rows each, while the rival2/4 queue is the only current source with direct critical-transition coverage (`1->2=32`, `1->4=33`).
- [ ] Do not treat `runs\review_minimum_fill_plan_current_20260706\priority_fill_plan.csv` as labels, sample weights, soft targets, targeted margins, or smoke permission. Fill explicit manual fields first, resolve the blocking mixed cluster `Image_4098`, rerun the readiness matrix, and rerun the smoke gate before any non-dry-run manifest or TRKH smoke.
- [x] 2026-07-06 rendered the fill plan to review HTML: `runs\review_minimum_fill_plan_current_20260706\priority_fill_plan_review.html` plus `.summary.json`. It renders `360/360` rows with `missing_images_rendered=0`, `manual_label_status_filled=0`, and sorted priority: `Image_4098` mixed cluster first, then critical `1->2`/`1->4` recall-gap rows. Focused renderer/fill-plan tests passed (`3 passed`).
- [ ] Keep the fill-plan HTML review-only. Manual decisions must be written back to the source train-only review queues and audited with readiness tooling; do not use the HTML/CSV as a trainable source or gate bypass.
- [x] 2026-07-06 cleanup legacy embedding-retrieval caches: compacted `10` old `embedding_retrieval_*_20260627` directories into `runs\evidence_embedding_retrieval_legacy_20260706`, preserving non-`.npz` summaries/metrics/predictions and deleting the original feature-cache dirs. Manifest `runs\cleanup_manifest_20260706_legacy_embedding_retrieval_compacted.json`; source `223.476 MB`, preserved `4.486 MB`, measured deleted feature payload `218.990 MB`.
- [x] Post-cleanup verification: all `10` original legacy embedding-retrieval dirs are absent; each evidence subdir has `summary.json`; protected current best checkpoint, softboost full-val, final no-pretrain test, smoke gate, readiness matrix, fill-plan summary/HTML, `yolof_oof_folds_train5_20260704`, and valid `class_f`/`yolo_f` YAMLs still exist.
- [ ] Do not look for the compacted legacy `embedding_retrieval_*_20260627` directories as live evidence. Use `runs\evidence_embedding_retrieval_legacy_20260706` plus recorded TODO/journal metrics; regenerate `.npz` feature caches only for a justified new diagnostic.
- [x] 2026-07-06 cleanup legacy quality/boundary review images: compacted `14` old quality-group and boundary-review directories into `runs\evidence_legacy_quality_boundary_reviews_20260706`, preserving CSV/JSON/HTML/README evidence and deleting original `review_images` payload dirs. Manifest `runs\cleanup_manifest_20260706_legacy_quality_boundary_reviews_compacted.json`; source `170.612 MB`, preserved `20.320 MB`, measured deleted rendered-image payload `150.292 MB`.
- [x] Post-cleanup verification: all `14` original legacy quality/boundary review dirs are absent; each evidence subdir has `summary.json`; protected current best checkpoint, softboost full-val, final no-pretrain test, smoke gate, readiness matrix, fill-plan summary/HTML, compact embedding evidence, `yolof_oof_folds_train5_20260704`, and valid dataset YAMLs still exist.
- [ ] Do not look for compacted legacy quality/boundary review dirs as live evidence. Use `runs\evidence_legacy_quality_boundary_reviews_20260706` plus recorded TODO/journal conclusions; regenerate rendered review images only for a justified future visual audit.
- [x] 2026-07-06 cleanup legacy selector/posthoc artifacts: compacted `10` old selector, focus-class1 specialist, and Q34 posthoc sweep directories into `runs\evidence_legacy_selectors_posthoc_20260706`, preserving CSV/JSON/MD/TXT/HTML evidence and deleting binary selector/specialist plus plot payloads. Manifest `runs\cleanup_manifest_20260706_legacy_selectors_posthoc_compacted.json`; source `90.017 MB`, preserved `34.802 MB`, measured deleted binary/plot payload `55.214 MB`.
- [x] Post-cleanup verification: all `10` original selector/posthoc dirs are absent, evidence file count is `79`, protected current best checkpoint, softboost full-val, final no-pretrain test, smoke gate, readiness matrix, fill-plan summary/HTML, compact embedding/quality evidence, `yolof_oof_folds_train5_20260704`, and valid dataset YAMLs still exist.
- [ ] Do not look for compacted legacy selector/posthoc dirs as live evidence. Use `runs\evidence_legacy_selectors_posthoc_20260706` plus recorded TODO/journal conclusions; old posthoc test sweeps are historical audit only and not current gate evidence or threshold-tuning permission.
- [x] 2026-07-06 added reusable signal-gap crosswalk `trkh.tools.build_trkh_signal_gap_readiness` plus focused test. It reads existing val/train-OOF/review summaries and named review CSVs, writes only diagnostic `summary.json`, `transition_gap_rows.csv`, and README; no raw-data edit, no test use, no trainable manifest.
- [x] Rerun signal-gap artifact `runs\diagnostic_trkh_signal_gap_readiness_current_20260706`: now includes rival2/4 review coverage, so `1->2` has `32` review rows and `1->4` has `36`, but both have `review_manual_filled_total=0`.
- [x] Rerun smoke gate `runs\audit_trkh_smoke_gate_current_rerun_20260706`: still `smoke_gate_ready=false`; review readiness `0/6`, manual fields `0/858`, one rival2/4 mixed actionable cluster, external consensus class1 `0.7237`, and true-class1 recall coverage still insufficient.
- [ ] Treat `runs\audit_trkh_smoke_gate_current_rerun_20260706` as the current gate audit. Do not launch smoke/full train from the updated coverage alone; fill explicit manual fields, resolve `Image_4098`, rerun readiness, then rerun gate.
- [x] 2026-07-06 wrote current best manual override command packet `docs\TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt` for VS Code PowerShell: full train current no-pretrain TRKH-native `yolo_f` recipe, raw final test, patch-linear softboost final test, verifier-aware XAI audit, and XAI transition summary. Training command uses `-SkipFinalTest`; I did not run these commands.
- [x] 2026-07-06 added reusable retention rerun tool `trkh.tools.audit_trkh_artifact_retention` plus focused test. It writes read-only inventory CSV/JSON/README only; no deletion/raw-data/test-tuning/trainable manifest.
- [x] Retention rerun `runs\artifact_retention_audit_current_trkh_rerun_20260706`: scanned `472` run dirs, `retention_audit_passed=true`, blockers `[]`, free `79.861 GB`, protected artifacts all present, no `eval_smoke_*`/`oof_mobilenetv3_fold*`, and `34/34` compacted original dirs absent.
- [ ] Use `runs\artifact_retention_audit_current_trkh_rerun_20260706` as the current inventory reference. The older `artifact_retention_audit_current_trkh_20260706` is stale after later compaction; only `7` small future cleanup candidates remain (`31.602 MB`) and require a separate preservation/docs check before deletion.
- [x] 2026-07-06 added diagnostic `trkh.tools.audit_patch_score_pib` plus focused tests. It measures cosine patch score Point-in-Box/top-k foreground mass from `patch_bbox_prior`, refuses test unless explicit, and writes no raw-data edits or trainable manifests.
- [x] Patch-score PiB full-val audit: raw `bbox` prior makes top1 PiB look low (`~0.34-0.36`), but transformed `crop_bbox` + `cls_register_mean` shows object focus is already high on full `yolo_f/val=2606`: all top1 PiB `0.9279`, incorrect top1 PiB `0.9286`, class1 top1 PiB `0.9338`, `background_shortcut_suspected=false`.
- [x] 2026-07-06 probe `runs\probe_v8_yolof_cropbbox_tokenprior_120b_1e_20260706`: switched only `-BboxTokenPriorSource crop_bbox`, 1e/120b/full-val/no-test. It dropped validation macro/class1 to `0.8822/0.6763` versus keeper `0.8847/0.6860`; XAI cases12 stayed foreground-high and object-desaturation/border driven.
- [x] 2026-07-06 cleanup rejected cropbbox token-prior payload: manifest `runs\cleanup_manifest_20260706_cropbbox_tokenprior_rejected_aux.json` deleted the rejected probe `checkpoints` dir plus 12 rendered XAI `case_*` dirs after preserving metrics/config/XAI summaries/transition summary, measured payload `175.766 MB`.
- [x] 2026-07-06 retention refresh after cropbbox cleanup: `runs\artifact_retention_audit_after_cropbbox_cleanup_20260706` scanned `480` run dirs, `retention_audit_passed=true`, blockers `[]`, compacted originals remaining `0`, free `79.838 GB`.
- [ ] Do not repeat simple `crop_bbox` token-prior switching on the current keeper by only changing LR, epochs, batch count, or query token. Use `crop_bbox` for transformed-frame diagnostics, keep the current best command explicit with `-BboxTokenPriorSource bbox`, and move next automatic work toward a new surface/boundary representation or fold-safe/manual reliability signal.
- [x] 2026-07-10 launcher failure audit: manual run `runs\full_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_30e_20260706` failed before training on 2026-07-08 because PowerShell converted a Python stderr INFO log into `NativeCommandError`; no checkpoint/metrics were produced, so it is not model evidence.
- [x] 2026-07-10 patched `scripts\run_trkh_5class_attention_views_v8.ps1` to invoke Python directly under a temporary `ErrorActionPreference="Continue"` guard instead of routing native stderr through a pipeline.
- [x] 2026-07-10 revalidated the full command packet with preflight and a one-train-batch/one-val-batch launcher smoke. Preflight resolved explicit `yolo_f`, `SkipFinalTest`, `BboxTokenPriorSource=bbox`, teacher-focus-binary `0.015`, bbox spatial fusion, pairwise routing, metric learning, and boundary-band dropout. Smoke `smoke_launcher_stderrfix_v8_yolof_1b_20260710` exited `0`; do not use its tiny-val metrics for model selection.
- [x] 2026-07-10 cleanup compacted the failed full run and launcher-validation smoke into `runs\evidence_launcher_stderrfix_validation_20260710`; manifest `runs\cleanup_manifest_20260710_launcher_stderrfix_validation.json`; observed free delta `178.082 MB`; raw dataset untouched.
- [x] 2026-07-10 retention refresh `runs\artifact_retention_audit_after_launcher_fix_20260710`: scanned `482` dirs, `retention_audit_passed=true`, blockers `[]`, protected artifacts present, compacted originals remaining `0`, free `79.992 GB`.
- [ ] Treat `docs\TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt` as updated for the 20260710 manual override run name. It is still not automatic gate permission; next automatic TRKH work still requires a new surface/boundary representation target or fold-safe/manual reliability signal with class-1 recall protection and conservative `0/2/4->1` FP control.
- [x] 2026-07-10 added train-only `trkh.tools.audit_self_neighbor_consistency` with explicit raw-vs-smoothed feature-key selection, same-source exclusion, review overlap deduplication, and hard `smoke_ready=false/training_permission=false` guardrails. Final focused preflight across neighbor audit and prototype-cache tooling passed (`19 passed`).
- [x] Current raw keeper train audit `runs\diagnostic_self_neighbor_consistency_sourceemb_combined_20260710`: `9215/9215` feature coverage, zero label mismatch, `280` review rows deduplicated to `248`, and no global recall-protector versus FP-suppressor separation (`best AUC=0.6317`; class1-neighbor label/probability AUC `0.4349/0.4091`).
- [x] Queue-specific check: rival2/4 looked moderately separable only in-sample (`best AUC=0.7207`), while strict batch02 inverted (`0.4511`). CMSF-smoothed targets inflated rival2/4 to `0.7714`, confirming that the feature key must be explicit and that train-only queue separation is not sufficient evidence.
- [x] Added reusable embedding-cache export to `probe_embedding_prototypes` and train-to-evaluation audit `trkh.tools.audit_neighbor_transfer_consistency`. Full-val kNN transfer artifact `runs\diagnostic_keeper_knn_transfer_k15_30_50_cache_20260710` kept base `0.8847/0.6860`, while k15/k30/k50 fell to `0.8663/0.6159`, `0.8651/0.6154`, and `0.8619/0.6063` macro/class1 F1.
- [x] Independent transfer artifact `runs\diagnostic_neighbor_transfer_consistency_keeper_val_20260710`: no threshold fitting, no test, no raw-data edit, and no trainable manifest. Stable recall-error versus FP-suppressor AUC was only `0.0582` for class1-neighbor label fraction and `0.0327-0.0426` for the other signals, so the train rival2/4 signal reverses on validation.
- [ ] Do not launch a Jo-SNC/SNSCL/current-embedding neighbor smoke, global or rival2/4-only. Do not tune k, neighbor thresholds, queues, weights, soft targets, or targeted margins around this cache. Continue only with a genuinely new surface/boundary representation or fold-safe/manual reliability source, then rerun the smoke gate.
- [x] 2026-07-10 cleanup compacted nine superseded CMSF-smoothed, k-sensitivity, and queue-only neighbor diagnostics into `runs\evidence_self_neighbor_consistency_sensitivity_20260710`, preserving per-run summaries/READMEs. Manifest `runs\cleanup_manifest_20260710_self_neighbor_sensitivity_compacted.json`; source `27.930 MB`, preserved `0.079 MB`, observed free delta `27.938 MB`; protected keeper/current-val/final-test/cache/transfer artifacts all present.
- [x] 2026-07-11 researched and implemented default-off, checkpoint-compatible stem pooling modes `soft` and `max_soft` from SoftPool/DPP/anti-aliasing/early-convolution literature. Added config/train/evaluate/V8 wiring and `tests\test_stem_pooling.py`; focused stem/resume/prefix suite passed (`10 passed`), with identical state-dict schema to legacy max pooling.
- [x] Direct eight-batch validation precheck rejected inference-only replacement: legacy max macro/class1 `0.7607/0.8462`; SoftPool `0.7253/0.7826` with 3 corrections versus 18 harms; `max_soft=0.15` `0.7492/0.8000` with two changes and both harmful.
- [x] New-representation smoke gate opened only for a short adaptation. `runs\smoke_v8_yolof_maxsoft015_stemadapt_60b_1e_20260710` used `60b/1e`, full `yolo_f/val=2606`, keeper resume, current core recipe, and no test; it fell to macro/class1 `0.8803/0.6627`, class1 P/R `0.5989/0.7417`.
- [x] Full-val runtime-verifier check `runs\eval_maxsoft015_stemadapt_smoke_softboost001_full_val_20260710` fell further to macro/class1 `0.8714/0.6187`, class1 recall `0.5695` (`1->0=47`). Balanced boundary review retained 12 rows each for class1 FN, class1 FP, low-margin errors, and low-margin correct boundaries.
- [x] Verifier-aware robust XAI `runs\xai_maxsoft015_stemadapt_smoke_val_cases12_20260710` audited 12 locked validation errors. Attention/grad-rollout foreground mass remained `0.8709/0.8982`, but Grad-CAM background/border was `0.3544/0.3313`; flags were background `6/12`, border `7/12`, object-color sensitive `8/12`. Object desaturation drop `0.1371` dominated background blur/gray `0.0242/0.0254`.
- [ ] Do not repeat direct SoftPool or nearby `max_soft` keeper adaptation by sweeping blend/LR/batches/epochs/query/prefixes. Keep the implementation default-off; the result lost true-class1 recall and did not create a transferable surface cue, so no probe/full train is permitted.
- [x] Cleanup manifest `runs\cleanup_manifest_20260711_softpool_stem_rejected.json` removed two rejected checkpoints, 12 rendered XAI case dirs, and 157 plot PNGs after preserving all compact evidence; deleted `145.954 MB`, observed free delta `146.512 MB`, raw dataset untouched.
- [x] Retention refresh `runs\artifact_retention_audit_after_softpool_reject_20260711`: `496` run dirs, `retention_audit_passed=true`, `blockers=[]`, protected keeper/current-val/final-test/OOF/review/YAML artifacts present, free `79.960 GB`.
- [ ] Next automatic route must not be another pooling blend or current-embedding selector. Require a materially new local surface/boundary representation with an independent pre-smoke signal and explicit class1 recall plus `0/2/4->1` FP guards.
- [x] 2026-07-11 researched SPT/LSA, SPD-Conv, and LIFE for scratch ViTs on small datasets; local no-repeat audit confirmed SPT/LSA/space-to-depth had not been implemented while paired context, local-token heads, high-pass experts, and frequency pooling had already failed.
- [x] Implemented default-off checkpoint-safe `ShiftedPatchTokenResidual`: four zero-padded diagonal post-stem views projected at patch stride, zero-init exact base-logit behavior, ModelConfig/train/V8 wiring, allowed resume prefix, trainable-prefix support, residual trace heatmap/norm, and focused tests.
- [x] Fixed V8 scheduler-horizon bug: launcher now exposes/validates/records `-SchedulerTotalEpochs` and forwards the effective value instead of hard-coding `$Epochs`; regression test added. History LR is the post-step/end-epoch LR, so do not describe one-epoch runs as having used min LR for every batch.
- [x] Three SPT `shift=1/scale=0.10/60b/full-val/no-test` schedules all stayed below keeper: base `2e-4` with one-epoch decay `0.8840/0.6842`; base `1e-4` with one-epoch decay `0.8834/0.6822`; fixed horizon `10`, base `1e-4`, final LR `9.76e-5` `0.8830/0.6822`, class1 P/R `0.6094/0.7748`.
- [x] Effective SPT trace proved activity (residual norm mean `0.0694-0.0997`, max `0.1449-0.3051`) but maps repeated patch/border/bright-end structure. Existing runtime verifier fell to macro/class1 `0.8772/0.6502`, recall `0.6093`, `1->0=43`.
- [x] SPT robust XAI on 12 locked validation errors: attention/grad-rollout foreground `0.9462/0.8979`; Grad-CAM background/border `0.2287/0.2218`; object-color sensitive `8/12`; object desaturation `0.1259` versus background blur/gray `0.0115/0.0125`; visual FN/FP maps still emphasized branches, padding, and object endpoints.
- [ ] Do not repeat current-keeper SPT residual by sweeping shift/scale/LR/scheduler/batches/head freezing. Keep it default-off for a future from-scratch architecture ablation only; no probe/full train is permitted from this result.
- [x] Compacted SPT evidence to `runs\evidence_shifted_patch_tokenization_rejected_20260711`; cleanup manifest preserved 84 files (`3.626 MB`) from `467.494 MB`, deleted payload `463.868 MB`, observed free delta `465.070 MB`, raw data untouched.
- [x] Retention `runs\artifact_retention_audit_after_spt_reject_20260711`: 499 dirs, pass, blockers `[]`, all six originals absent, protected artifacts present, free `79.955 GB`.
- [ ] Any future one-epoch smoke that needs a meaningful near-base LR must pass an explicit scheduler horizon (for example `-SchedulerTotalEpochs 10`) and verify trainer `Scheduler setup`; do not infer batchwise LR from the final `history.csv` value alone.
- [x] 2026-07-11 implemented default-off checkpoint-safe GPSA-style gated relative-position attention on patch-to-patch mass only, preserving prefix interactions and original pruned-token coordinates; added config/train/V8/resume/prefix/trace wiring and focused tests.
- [x] Keeper preflight proved exact identity (`max logit delta=0`), only eight allowed missing keys, nonzero gradients in all four enabled attention blocks, and `1,413` trainable parameters including the legacy head. Focused suite passed `12/12`; broad relevant suite passed `138` with four known unrelated stale dirty-worktree failures.
- [x] Fixed short-adapter EMA correctness: partial architecture resumes now reset mature keeper EMA updates so new keys receive a real warmup. Fixed clamp-dead gates with a straight-through projected training clamp; evaluation remains truly clamped. The first GPSA smoke before these fixes is infrastructure-invalid.
- [x] Valid GPSA `layers=1-4/max_mix=.25/LR=3e-4/scheduler10/60b/full-val/no-test` smoke reached only macro/class1 `0.87895/0.65455`, class1 P/R `0.60335/0.71523`; gate maxima stayed `0.00136-0.00430`, showing optimization mostly rejected the local-position mix.
- [x] Raw train-state evaluation remained weak at `0.87503/0.65193`; current patch-linear softboost fell to `0.87113/0.61702`, class1 recall `0.57616`, `1->0=38`, `1->2=19`. Balanced val boundary review retained 48 rows; no test was used.
- [x] GPSA XAI audited 12 locked validation cases: Grad-CAM background/border flags `7/9`, rollout `7/7`, object-desaturation drop `0.14755` versus background blur/gray `0.01544/0.01542`, CLS/register similarity `0.9997805`. Visual maps still emphasize border/stem/endpoints or broad color regions.
- [ ] Do not sweep GPSA LR, max mix, layers, batch count, epochs, or head freezing as a keeper adapter. Keep it default-off for a future from-scratch architecture ablation; it did not improve class1 surface separability or combine safely with the current verifier.
- [ ] For every future partial architecture extension resumed from a mature checkpoint, require log evidence that EMA updates were reset for new keys, then audit both EMA and raw train state before making a gate decision.
- [x] Compacted rejected GPSA evidence into `runs\evidence_gpsa_relative_position_rejected_20260711`: preserved 80 files (`7.240 MB`) from seven source roots (`328.130 MB`), including detailed full-val predictions, gate traces, boundary rows, transition XAI, and three visual groups; observed free delta `329.340 MB`.
- [x] Retention `runs\artifact_retention_audit_after_gpsa_reject_20260711`: 502 dirs, pass, blockers `[]`, `14/14` protected checks present, all seven originals absent, raw dataset untouched, free `79.947 GB`.
- [x] 2026-07-11 added validation-only runtime `probe_locality_self_attention`: patch-only post-softmax smoothing/sharpening/diagonal transforms preserve prefix rows, patch-to-prefix links, and patch mass while measuring per-layer self mass, entropy, top1 concentration, predictions, and class1 correction/harm budgets. Focused suite passed `17/17`.
- [x] Full `yolo_f/val=2606` LSA precheck reproduced keeper `0.884675/0.686047`; self fraction layers 1-4 was only `0.004446/0.002607/0.003150/0.004030`, so diagonal `0.50` changed zero predictions.
- [x] LSA matrix rejected: smoothing `0.882298/0.674419` (5 corrections/7 harms), sharpening `0.881644/0.676301` (4/9), sharp+diag50 `0.882578/0.680115` (4/8), sharp+hard-diag `0.883175/0.682081` (4/7). Hard LSA rescued/broke one class1 row each and created four versus removed two class1 FP.
- [ ] Do not implement or smoke a trainable LSA temperature/self-mask adapter, and do not sweep nearby layer subsets, temperature multipliers, or diagonal suppression on the current keeper. Baseline self mass is already negligible and both smoothing and sharpening fail recall/FP/correction-harm gates.
- [x] LSA cleanup manifest `runs\cleanup_manifest_20260711_lsa_precheck_compacted.json`: gzipped full row evidence, removed superseded 8-row dry-run, retained `6.154 MB`, observed free delta `12.820 MB`; no raw data/test/trainable manifest.
- [x] Retention `runs\artifact_retention_audit_after_lsa_precheck_20260711`: 504 dirs, pass, blockers `[]`, `14/14` protected checks present, compacted dry-run absent, free `79.940 GB`.
- [x] 2026-07-11 completed the missing best-recipe full continuation after the launcher stderr fix: `runs\full_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_30e_autorun_20260711`, max 30 epochs/patience 3/full `yolo_f/val=2606`/no test. It stopped at epoch 5, selected epoch 2, and reached raw macro/class1 `0.886808/0.681690`, class1 P/R `0.593137/0.801324`, below keeper class1 `0.686047` and selection metric `0.762949`.
- [x] Frozen softboost evaluation of the continuation fell to macro/class1 `0.877042/0.637288`, class1 P/R `0.652778/0.622517`, with `1->0=41`, `1->2=10`, `0->1=26`, and `4->1=12`; it does not rescue the new checkpoint.
- [x] Continuation boundary/XAI audit used 48 balanced validation rows and 12 locked cases, no test. Object-desaturation drop `0.15491` dominated background blur/gray `0.02342/0.02601`; object-color flag `8/12`, Grad-CAM background/border `5/12`/`6/12`, rollout `9/12`/`7/12`. Visual evidence still points to color plus local stem/endpoint/border cues, not a new interior-surface representation.
- [ ] Do not replace the keeper, run test, extend epochs, or tune LR/patience/verifier strength around the rejected 20260711 full continuation. Preserve it only as compact negative evidence that optimizing the existing recipe raises class1 recall by sacrificing precision.
- [x] Compacted rejected full-continuation evidence into `runs\evidence_full_continuation_rejected_20260711`: 58 files, `5.453 MB`, no checkpoint/test result. Cleanup manifest leaves unavailable source-size/free-delta fields `null` after a post-delete `DriveInfo.Refresh` measurement error; independent retention audit passed over 506 dirs with `blockers=[]` and all four originals absent.
- [x] Packaged the current deep-research brief, journal, method audits, architecture source, dataset metadata, competitor metadata, keeper/runtime/final-test summaries, compact negative evidence, and selected XAI visuals into `docs\TRKH_DEEP_RESEARCH_REFERENCE_PACK_20260711.zip`. Verified 191 ZIP entries, all 190 payload SHA-256 hashes, 17 required paths, zero checkpoint/raw-manifest entries; compressed size `7.634 MB`.
- [x] Wrote `docs\TRKH_DEEP_RESEARCH_REFERENCE_PACK_INDEX_20260711.md` and `docs\TRKH_DEEP_RESEARCH_REFERENCE_PACK_20260711.sha256.txt`; the pack explicitly warns that AIDT native metrics are not directly comparable until class order, sample alignment, split, and crop view are verified.
- [x] 2026-07-11 researched DropPos, LOCA, and Siamese Image Modeling, then implemented diagnostic-only `probe_paired_bbox_aligned_patch_correspondence` to test exact `yolo_f` bbox-to-`class_f` crop patch mapping after token pruning. It adds no loss/model/manifest and refuses test; focused paired/context suite passed `14/14`.
- [x] Full raw-context correspondence precheck `runs\diagnostic_paired_bboxalign_rawcontext_full_val_20260711` covered `2606/2606` validation objects. Geometry passed (`2605/2606` valid, matched/control cosine `0.60074/0.29195`, margin `0.30879`, top1 lift `0.07306`, MRR `0.16106`, mean distance `0.03575`).
- [x] Class safety gate rejected direct crop-to-context alignment: raw context macro/class1 `0.57793/0.19792`, paired crop `0.88065/0.67055`; crop rescued `97/132` raw-context class1 FN and broke `1/19` TP but created `67` class1 FP while suppressing only `12`. Error rows had no lower geometry margin than correct rows (`0.3168` vs `0.3048`).
- [ ] Do not implement or smoke direct bbox-aligned crop-to-context feature matching, and do not sweep its weight/radius/top-k/context alpha/layer on the current keeper. Geometry is real but the crop teacher fails the predeclared class1 FP budget; any location-aware follow-up must predict position without transferring crop class bias.
- [x] Preserved the 64-row checkpoint-crop control under the full artifact, removed both superseded dry-run roots, and wrote `runs\cleanup_manifest_20260711_paired_bboxalign_precheck_dryruns.json`. Retention `runs\artifact_retention_audit_after_paired_bboxalign_precheck_20260711` passed over 508 dirs with `blockers=[]`.
- [x] 2026-07-11 implemented diagnostic-only `probe_bbox_position_reconstruction_readiness`: it removes patch positional embeddings while preserving prefix positions, fits an in-memory train-only ridge decoder for bbox-relative token `(x,y)`, and audits the frozen keeper on full validation. Focused compile/tests passed `15/15`; no test/model/trainable manifest/raw-data edit.
- [x] Full position precheck `runs\diagnostic_bbox_position_readiness_poszero_train120_full_val_20260711` used `122868` train tokens and all `2606` validation rows/`83390` tokens. Position was decodable (mean R2/MAE `0.215319/0.205873`, 4x4 accuracy `0.154671` vs random `0.0625`); class1 was stronger, not weaker (mean R2 `0.281719`, MAE `0.197190`, 4x4 `0.164528`).
- [x] Removing all patch positional embeddings barely affected classification: macro/class1 `0.884675/0.686046 -> 0.884067/0.682216`, only `8/2606` predictions changed with `4` corrections and `4` harms; mean absolute logit/probability deltas were `0.003426/0.000659`. The predeclared `classification_position_sensitive` gate failed (`8 < 25`).
- [ ] Do not implement or smoke DropPos/bbox-relative position reconstruction from this exact signal, and do not sweep ablation scale, ridge strength, token cap, auxiliary weight, or DropPos probability on the keeper. Coordinate decodability is almost orthogonal to the current class decision; require a position signal that independently separates class1 recall errors from `0/2/4->1` false positives.
- [x] Removed the superseded 4-batch position-readiness dry run via `runs\cleanup_manifest_20260711_bbox_position_readiness_dryrun.json` after preserving full evidence. Retention `runs\artifact_retention_audit_after_bbox_position_precheck_20260711` passed over 510 dirs with `blockers=[]`, the original absent, protected artifacts present, and `79.925 GB` free.
- [x] 2026-07-11 researched CIConv-W, Multiscale Retinex, and intrinsic-image illumination invariance; local no-repeat audit confirmed TRKH had only tried RGB illumination consistency, HSV/Lab statistics, and raw high-frequency maps, not a physics-based reflection-invariant derivative view.
- [x] Added standalone `ColorInvariantW` plus diagnostic-only `probe_photometric_invariant_complementarity`. The fixed protocol keeps RGB, uses official-default W scale `0`, suppresses invalid-padding derivative halos, reuses the exact RGB cache, and applies one `C=0.3` five-fold source-grouped OOF readout with predeclared class1 recall/FP gates. Compile and focused tests passed `6/6`.
- [x] Full train/val precheck `runs\diagnostic_ciconv_w_complementarity_full_train_val_20260711` aligned all `9215/2606` rows and `8064` train source groups, no test. W maps visibly expose spots/wrinkles/contours and are not border dominated (`border mass=0.27434`, clip fraction `<0.001`). Effective rank is preserved rather than collapsed: RGB/W `6.063/6.820`, ratio `1.125`; rejection is due to absent class-separating and class1-safe signal.
- [x] CIConv-W failed representation and class-safety gates: W-only validation macro/class1 `0.64965/0.21445`; RGB readout `0.88106/0.66667`; RGB+W fell to `0.87421/0.63973` and also lost OOF macro/class1 (`0.93328/0.79934 -> 0.92939/0.78385`). Against direct keeper it changed `117` rows (`50` corrections, `55` harms), removed/created `32/8` class1 FP but broke/rescued `25/2` true-class1 TP/FN; error-direction AUC `0.55879`.
- [ ] Do not implement or smoke shared-trunk CIConv-W fusion, and do not sweep its scale, clipping, readout C, fusion weight, or a nearby zero-init adapter on this keeper. It behaves as another unsafe class1 suppressor. Revisit a physics view only after an independent dedicated-encoder or reflectance target proves both subtle class1 recall and conservative `0/2/4->1` FP control.
- [x] Removed the superseded `1024/256` CIConv-W dry run via `runs\cleanup_manifest_20260711_ciconv_w_precheck_dryrun.json` (`2.034 MB` source, `2.051 MB` observed free delta). Retention `runs\artifact_retention_audit_after_ciconv_w_precheck_20260711` passed over 512 dirs with `blockers=[]`, protected artifacts present, and the dry-run absent.
- [x] Preserved and audited user deep-research reports 14/15 under `docs\research_inputs`; verified SHA-256 `561BA5...94374D7F` and `793398...46A7D4B0`. Their useful scope correction is now explicit: adapter/current-embedding failures do not reject a concurrent hybrid trained from initialization, multi-prototype methods after a representation change, or teacher-as-uncertainty-critic.
- [x] No-repeat crosswalk found that generic/whole-patch bilinear pooling, bbox token means/labels, sub-centers, neighbors, soft targets, ordinal heads, and generic invariance have already failed. The report's genuinely new narrow gate is interior-core-only second-order pooling with boundary/context controls; hybrid-from-scratch and two-stage ambiguity-aware supervision remain later phases.
- [x] Implemented context-estimated Gray-Edge color constancy with target-bbox exclusion and a preserved RGB branch. Fixed protocol was order1/sigma2/p6/margin0.08/gain0.5-2.0 plus one five-fold grouped OOF `C=0.3`; compile and focused tests passed `15/15`, no test/model/raw-data edit.
- [x] Full Gray-Edge precheck `runs\diagnostic_context_grayedge_p6_s2_full_train_val_20260711` aligned all `9215/2606` rows. Support/gain/rank checks passed (corrected/RGB effective rank `6.551/6.063`), but direct macro/class1 collapsed `0.88468/0.68605 -> 0.79411/0.45326` with `61/232` corrections/harms and `10/48` FN-rescues/TP-breaks.
- [x] RGB+corrected readout also failed: OOF `0.93328/0.79934 -> 0.93035/0.79149`, validation `0.88106/0.66667 -> 0.87857/0.66000`, direct-keeper transitions `54` corrections vs `55` harms, only `1` class1 FN rescue vs `20` TP breaks, error-direction AUC `0.48040`. No smoke.
- [ ] Do not sweep Gray-Edge p/sigma/margin/gain, readout C, or fusion weight on this keeper. Revisit structured color constancy only within a new jointly trained interior representation with explicit class1 recall protection.
- [x] Removed the superseded Gray-Edge `1024/256` dry root and duplicate process logs via `runs\cleanup_manifest_20260711_context_grayedge_dryrun_logs.json` (`2.771 MB` source, `2.802 MB` observed free delta). Retention `runs\artifact_retention_audit_after_context_grayedge_precheck_20260711` passed over 514 dirs with `blockers=[]`; full evidence and protected keepers remain.
- [x] Built one fixed diagnostic-only interior-core compact covariance/second-order readout, compared it against head and all-patch controls with source-grouped OOF plus full `yolo_f/val=2606`, and refused smoke because class1 recall and class-safety gates failed.
- [x] Implemented diagnostic-only `probe_interior_second_order_readiness`: fixed QR projection rank24/seed20260711, `crop_bbox` core erode0.20, exact covariance matrix square root, head/all/core/ring/context controls, one grouped-OOF `C=0.3`, no test/model/raw-data edit. Compile and focused tests passed `4/4`.
- [x] Full artifact `runs\diagnostic_interior_secondorder_rank24_erode020_full_train_val_20260711` covered `9215/2606` rows and 8064 train source groups. Geometry passed (core mean train/val `48.924/49.384`, p01 `30/32`, fallback `0.000109/0.000384`) and core effective rank was `24.827`; overlay audit is correct.
- [x] Interior second-order failed class signal: head OOF/val macro-class1 `0.93304/0.79901`, `0.88072/0.66667`; core-only `0.84244/0.52125`, `0.82293/0.53261`; all-patch was still stronger than core. Head+core fell to OOF `0.91684/0.74561` and val `0.86914/0.63607`.
- [x] Candidate-vs-keeper changed `127` rows (`47` corrections, `67` harms), class1 FN-rescue/TP-break `3/24`, FP-remove/create `27/9`, and error-direction AUC `0.47717`. Reject trainable readout/smoke.
- [ ] Do not sweep rank, random seed, erode ratio, covariance normalization, or readout strength around this frozen-token route. It rejects interior second-order as a keeper adapter only; compact concurrent hybrid training from initialization remains the report-backed open architecture route.
- [x] Removed the biased dry descriptor root and four duplicate logs via `runs\cleanup_manifest_20260711_interior_secondorder_dryrun_logs.json` (`11.762 MB` source, `11.788 MB` observed free delta). Retention `runs\artifact_retention_audit_after_interior_secondorder_precheck_20260711` passed over 516 dirs with `blockers=[]`; the full 99.226 MB artifact and all keepers remain.
- [x] Added `scripts\run_trkh_5class_compact_hybrid_scratch.ps1` for no-pretrain MobileViT/EdgeNeXt probes with immutable `yolo_f`, strict balanced sampling, fixed mild augmentation, full-val gate, teacher-focus-binary `0.015`, EMA, and forced no-test behavior. Batch 64 was the measured MobileViT-S throughput/VRAM point; batch 80 thrashed.
- [x] MobileViT-S epoch-5 probe independently reproduced validation macro/class1 `0.81299/0.50000`; full boundary and 12-case robust XAI showed correct object localization but 155 class1 FP, 49 FN, and strong object-color dependence.
- [x] Added validated `-ResumeCheckpoint` support that preserves epoch/optimizer/scheduler/scaler/EMA and records the source. The bounded epoch-6-to-10 continuation restored all available states, retained scheduler horizon 15, completed cleanly, skipped test, and selected epoch 8.
- [x] MobileViT-S total-epoch-10 best independently reached only macro/class1 `0.82908/0.54595` (class1 P/R `0.46119/0.66887`), below keeper `0.88468/0.68605`; epochs 9/10 did not improve the fair selector. Same-12-case XAI corrected only 1/12 and left 11 wrong. Reject this exact recipe and do not extend/tune it to 15/30 epochs.
- [x] Compacted the rejected MobileViT route to `runs\evidence_compacthybrid_mobilevit_s_yolof_scratch_10e_reject_20260711` with 229 verified files and no checkpoints. Cleanup manifest `runs\cleanup_manifest_20260711_mobilevit_compacthybrid_rejected.json` removed 268 source files (`273.336 MB`) and observed `273.953 MB` free-space gain; datasets/test/keepers were untouched.
- [x] Ran one unchanged-protocol no-pretrain EdgeNeXt X-Small probe (2.145M parameters). Independent full-val best at epoch 5 was only macro/class1 `0.73157/0.38776`, class1 P/R `0.31535/0.50331`, below MobileViT epoch 5 `0.81299/0.50000`; reject before continuation.
- [x] EdgeNeXt X-Small full audit found 1,292 boundary candidates. Twelve-case XAI had attention/Grad-CAM foreground `0.99954/0.90663`, background blur/gray drops `0.00088/0.00406`, and object-desaturation drop `0.08546`; it localized the fruit but underfit internal class structure rather than depending on wide context.
- [x] Compacted X-Small evidence to `runs\evidence_compacthybrid_edgenext_xsmall_yolof_scratch_5e_reject_20260711`; `runs\cleanup_manifest_20260711_edgenext_xsmall_compacthybrid_rejected.json` removed 122 source files (`71.752 MB`) and observed `72.039 MB` free-space gain without touching test/datasets/keepers.
- [x] Ran the final parameter-matched `edgenext_small` check (5.283M). Independent epoch-5 macro/class1 was only `0.72898/0.40201`, class1 P/R `0.32389/0.52980`; it failed to beat X-Small macro, MobileViT epoch 5, or keeper. Close stock EdgeNeXt and do not sweep size/LR/loss/sampler/teacher/epochs.
- [x] EdgeNeXt-Small full audit found 1,341 boundary candidates. Grad-CAM foreground was `0.99967`; background blur/gray drops `0.00117/0.00462` versus object-desaturation `0.07966`. Generic attention fallback background mass was not treated as real SDTA attention because robustness and Grad-CAM contradicted it.
- [x] Compacted Small evidence to `runs\evidence_compacthybrid_edgenext_small_yolof_scratch_5e_reject_20260711`; cleanup manifest removed 122 source files (`143.372 MB`) and observed `143.656 MB` free-space gain without touching test/datasets/keepers.
- [x] Audited whether a train-only teacher/ensemble uncertainty signal separates current-keeper class1 false negatives from `0/2/4->1` false positives. Added strict no-test/sample-index tool `audit_teacher_uncertainty_critic_readiness`; compile and focused tests passed `2/2`.
- [x] Full `9215/2606` EffV2+DINO OOF/full-val audit is retained at `runs\diagnostic_teacher_uncertainty_critic_readiness_20260711`. Error detection was strong, but FP-vs-FN direction AUROC flipped `0.91763 -> 0.43900`; validation suppress/rescue precision was only `0.55882/0.30769`, and teacher-mean class1 F1 fell `0.70303 -> 0.66667`. Gate rejected before smoke; test stayed closed.
- [ ] Do not train an evidential/uncertainty critic, per-transition KD, soft-target, router, or targeted-margin variant from the current EffV2/DINO OOF caches. They can rank ambiguity for abstention but lack transferable correction direction and `1->2/1->4` support. Generic EDL remains open only with independently grounded ambiguity targets or a demonstrably new representation.
- [x] Completed the CoAtNet/CMT/Conformer/CCT no-repeat audit. Historical CMT/LeFF sparse block-local adaptation and CCT/PARTICLE-style part prototypes already failed; local timm has no stock CMT/Conformer/CCT implementation. The old pretrained CoAtNet-0 fold-00/plain-CE run was not equivalent to the current scratch recipe, so exactly one fixed `coatnet_nano_rw_224` stage-wise candidate was permitted.
- [x] Added fixed-native-image support to `scripts\run_trkh_5class_compact_hybrid_scratch.ps1` and a CoAtNet forward regression. `coatnet_nano_rw_224` has 14.631M parameters and alternates MBConv stages before relative-attention stages; batch 64/image 224 measured about 274 synthetic img/s and 4.96 GB peak allocation.
- [x] CoAtNet-Nano scratch improved full-validation macro/class1 from `0.83032/0.55340` at epoch 5 to `0.85530/0.61957` at epoch 10. A stateful continuation completed the cosine horizon at epoch 15 and selected epoch 13; independent reload reached only `0.86161/0.62637`, class1 P/R `0.53521/0.75497`, below raw keeper `0.88468/0.68605` and runtime keeper `0.88868/0.70303`. No test and no 30-epoch extension.
- [x] CoAtNet boundary candidates fell `584 -> 460 -> 422`, but class1 TP stayed fixed at `114` while FP only fell `147 -> 103 -> 99`; 37 FN remained. Same-12-case XAI stayed `6/12` correct. Final object-desaturation drop `0.10312` dominated background blur/gray `-0.00149/-0.00082`; diffuse Grad-CAM background mass is not accepted as causal background evidence because perturbations contradict it.
- [x] Compacted CoAtNet evidence to `runs\evidence_compacthybrid_coatnet_nano_yolof_scratch_15e_reject_20260711` with 320 files, SHA-256 manifest, no checkpoint/test output. Cleanup `runs\cleanup_manifest_20260711_coatnet_nano_compacthybrid_rejected.json` removed 12 roots plus 14 logs (`1072.518 MB` source; `1073.395 MB` observed free-space gain) without touching datasets or keepers.
- [ ] Do not extend or tune this CoAtNet recipe, and do not launch another stock MobileViT/EdgeNeXt/CoAtNet/CMT/Conformer/CCT catalogue backbone. The next trainable route must use the measured stage-wise result to justify a TRKH-native representation change and must first demonstrate fold-safe separation of class1 FN rescue from `0/2/4->1` FP suppression; another confidence blend/router is not sufficient.
- [x] Ran label-using validation diagnostic `runs\diagnostic_coatnet_nano_softboost_complementarity_val_20260711`: CoAtNet corrected `12/35` softboost class1 FN and `19/63` class1 FP, but still predicted class1 on `43/63` base FP. Even the non-deployable focus-error oracle reached only macro/class1 `0.91221/0.79257`; no blend/router/KD smoke was opened.
- [x] Implemented default-preserving `stem_architecture=coatnet_mbconv`: exact scratch timm CoAtNet-Nano stem plus its first two MBConv stages at stride 8, projected into unchanged TRKH patch/token/transformer layers. Added config/train/V8 wiring, fixed scratch wrapper `scripts\run_trkh_5class_stagewise_mbconv_scratch.ps1`, checkpoint roundtrip/default-schema tests; focused suite passed `11/11`.
- [x] Resource gate: full-config BF16 synthetic batch 32 measured legacy versus MBConv `221.96/140.27 img/s`, `2.53/4.00 GB` peak allocated, and `7.246M/8.243M` parameters. Real 20-batch smoke completed full validation and 132-file trace with `6.88/7.07 GB` allocated/reserved; XAI Grad-CAM foreground `0.9771`, background blur/gray drops `0.00623/0.00501`.
- [x] Interrupted the attention-view-enabled probe after epoch 1 because epoch-2 activation raised batch time from about `0.5-0.8s` to `6-8s` and reserved VRAM to `8.154 GB`. It is explicitly diagnostic-only and was not used for model selection.
- [x] Fixed no-attention-view architecture gate completed 5 epochs/full val/no test. Fair-best epoch 3 independently reproduced macro/class1 `0.79725/0.50495`, class1 P/R `0.40316/0.67550`; epochs 4/5 raised macro to `0.81322/0.81071` while class1 fell to `0.48675/0.48227`. It is below stock CoAtNet epoch 5 `0.83032/0.55340`, MobileViT/CoAtNet continuation evidence, and keeper.
- [x] Full audit found 709 boundary candidates and independent confusion `[[460,73,2,0,14],[37,102,11,0,1],[1,32,466,40,5],[0,2,86,623,1],[33,44,15,9,549]]`. Final Grad-CAM/grad-rollout/rollout foreground was `0.99183/0.94547/0.96235`; background blur/gray drops `0.00124/0.00131` versus object desaturation `0.14349`. Reject as class-unsafe surface/maturity underfit, not background shortcut.
- [x] Compacted 437 files (`32.888 MB`, no checkpoint/test) to `runs\evidence_stagewise_mbconv_stem_scratch_5e_reject_20260711`. Cleanup `runs\cleanup_manifest_20260711_stagewise_mbconv_stem_rejected.json` removed seven roots plus eight logs (`634.046 MB` source; `635.453 MB` observed free-space gain), preserving all keepers/datasets.
- [x] Retention `runs\artifact_retention_audit_after_stagewise_mbconv_cleanup_20260711` passed over 526 directories with `14/14` protected checks, compacted originals absent, and `blockers=[]`.
- [ ] Do not extend or sweep the stagewise MBConv stem route: no nearby batch/LR/loss/attention schedule, MBConv depth/width, or partial CoAtNet-stage transplant. The result indicates stock CoAtNet's later relative-attention representation, not its early stem alone, explains its advantage; require a distinct pre-smoke signal before any next architecture.
- [x] Added strict 17-model validation information-ceiling and three-space residual-neighbor audits. All-model exact oracle reached macro/class1 `0.98395/0.95765`; class1 binary oracle reached only `0.97351`, with 4 FN + 4 FP unanimous across every model. Test remained closed.
- [x] Visually and numerically audited all eight unanimous class1 residuals. They are foreground-dominant maturity/boundary cases with cross-label adjacent frames; 52/70 nearby sequence pairs conflict. Do not treat wide background or localization as the remaining explanation, and never use `Image_N` sequence IDs as model input.
- [x] Implemented default-off Deep Abstaining Classification end to end: stable DAC loss, exact q initialization, strict checkpoint extension, V8/wrapper wiring, telemetry, evaluator export, validation-only risk-coverage audit, and focused tests (`8/8`).
- [x] DAC smoke `w0.15/alpha1.30/q0.01/60b/1e` independently reached only macro/class1 `0.87749/0.65487`. q error AUROC was `0.42040`; selective risk worsened as coverage fell; 24-case XAI showed weaker foreground focus and object-desaturation drop `0.13602`. Reject before probe.
- [x] Compact DAC rejection evidence to `runs\evidence_deep_abstention_w015_a130_reject_20260711`, remove both source runs/checkpoints plus eval/risk/XAI source roots, then rerun protected-artifact retention.
- [ ] Do not sweep DAC weight, penalty, q initialization/dropout, epochs, batch cap, or q thresholds on the current keeper. A future uncertainty head needs independently grounded supervision or a representation change, and must improve both closed-set class1 F1 and risk-coverage without validation-label routing.
- [ ] Next method must be distinct from current-embedding soft targets, EDL/DAC confidence heads, prototype/neighbor regularization, stock backbone substitution, and background filtering. Gate it first with train-only/fold-safe evidence that separates true class1 recall protection from `0/2/4->1` suppression.
- [x] Researched LocalViT, CeiT/LeFF, and CvT, then implemented a default-off true LeFF inside TRKH transformer layers. Prefix tokens bypass the hidden-channel depthwise spatial convolution; dense/pruned patch grids are checkpoint-safe. Config/train/V8/trace plus `scripts\run_trkh_5class_leff_scratch.ps1` are wired, and focused compile/parse/dry-run/regression tests passed `13/13`.
- [x] LeFF resource gate passed: parameters `7.246M -> 7.291M`, synthetic BF16 batch-32 throughput `222.09 -> 206.92 img/s`, peak allocation `2.533 -> 2.756 GB`. The 20-batch smoke completed full validation/no test; XAI foreground was `0.945-0.960`, background perturbation near zero, and object-desaturation drop `0.2176`.
- [x] Fixed LeFF 120-batch/5-epoch/full-val/no-test probe progressed macro/class1 `0.73415/0.35829 -> 0.79170/0.47465`. Independent reload exactly matched class1 P/R `0.36396/0.68212`; class1 had 103 TP, 180 FP, and 48 FN. It is below stage-wise MBConv `0.79725/0.50495`, CoAtNet epoch 5 `0.83032/0.55340`, and keeper.
- [x] LeFF full audit found 784 boundary candidates. Final attention/grad-rollout/Grad-CAM/rollout foreground was `0.9640/0.9379/0.8347/0.8658`; background blur/gray drops `0.0060/0.0072` versus object desaturation `0.1262`. Visual `4->1`, `0->1`, `1->2`, and `1->0` cases confirm class-unsafe surface/maturity cues rather than a background shortcut.
- [x] Compacted LeFF evidence to `runs\evidence_trkh_leff1234_scratch_5e_reject_20260711` (591 files, `60.839 MB`, no checkpoint/test, SHA-256 manifest), deleted six source roots/592 files via `runs\cleanup_manifest_20260711_trkh_leff1234_scratch_rejected.json` (`396.520 MB` observed free gain), and passed retention over 531 directories with `blockers=[]`.
- [ ] Do not continue or sweep the exact layers-1-4/kernel-3 LeFF route: no nearby layer subset, kernel, expansion, LR, loss, sampler, teacher, normalization, or 10/15/30-epoch extension. It increases class1 recall without conservative FP control and fails the five-epoch architecture gate.
- [x] Researched Visual Conformer, CMT, and CoaT, then implemented default-off persistent concurrent local-global coupling in TRKH. A 64-channel 16x16 CNN state exchanges information with sparse/dense patch tokens after layers 1-8 while preserving prefix tokens; config/train/V8/DETR/trace/wrapper wiring and focused regressions passed `17/17`.
- [x] Concurrent-coupling resource and pipeline gates passed: parameters `7.246M -> 7.670M`, synthetic BF16 throughput `221.13 -> 198.05 img/s`, and real peak allocation/reserve about `4.808/5.412 GB`. Smoke XAI remained fruit-focused and coupling norms/scales proved the branch was active.
- [x] Fixed 120-batch/5-epoch/full-val/no-test coupling probe reached only selected macro/class1 `0.78454/0.44898`, class1 P/R `0.34138/0.65563`, with 99 TP, 191 FP, 52 FN, and 802 boundary candidates. Final background perturbations were near zero while object desaturation was `0.1539`; reject as class-unsafe maturity/surface discrimination, not a dead branch or context failure.
- [x] Compacted coupling evidence to `runs\evidence_trkh_concurrent_localglobal_scratch_5e_reject_20260711` (601 files, `58.949 MB`, no checkpoint/test/raw data; SHA-256 manifest), deleted six source roots/602 files via `runs\cleanup_manifest_20260711_trkh_concurrent_localglobal_rejected.json` (`413.340 MB` observed free gain), and passed retention over 533 directories with `blockers=[]`.
- [ ] Do not continue or sweep exact concurrent layers 1-8/dim 64/kernel 3: no nearby layer subset, local width/kernel/scale, LR, loss, sampler, teacher, normalization, or epoch extension. Keep the default-off implementation only for reproducible ablation.
- [ ] With stock hybrids, stage-wise MBConv, LeFF, and persistent Conformer-style coupling now closed, require the next route to change the supervision/problem formulation and pass a strict train-only or fold-safe class1 FN-versus-FP readiness audit before any GPU smoke. Do not revisit current-embedding soft targets, prototype/neighbor regularization, DAC/EDL confidence-only heads, logit routers, or background filtering under a new name.
- [x] 2026-07-11 paired `yolo_f`/`class_f` decision-complementarity audit: only 28 full-val disagreements; `class_f` rescued `0/33` context class1 FN and corrected `2/76` class1 FP; exact label oracle only macro/class1 `0.888270/0.688047`. Reject direct CrossViT/Dual-Cross-Attention fusion before GPU; do not sweep fusion layer/weight/router.
- [x] Implemented fixed ICML-2023 LDR-KL with hard/soft targets, smoothing, class weights, per-sample output, config/train/V8/wrapper wiring, and focused tests (`16/16`); compile, PowerShell parse, dry-run, and real preflight passed.
- [x] Keeper stage-2 LDR-KL margin `2.0`, temperature `1.0`, LR `8e-5`, 60b/1e/full-val/no-test reached macro/class1 `0.880233/0.672515`, P/R `0.602094/0.761589`, below raw/runtime keepers; no longer probe warranted.
- [x] Mandatory 12-case LDR XAI: foreground attention/grad-rollout/Grad-CAM/rollout `0.8804/0.9486/0.8438/0.9129`; background blur/gray `0.0133/0.0129` versus object desaturation `0.2058`; direct review confirms the same color/spot/lesion and border failure modes.
- [ ] Do not sweep fixed LDR-KL margin, temperature, LR, sampler, teacher, batch count, or epochs. Keep implementation default-off for ablation only.
- [x] Compacted paired-view and LDR evidence to `runs\evidence_ldrkl_m200_t100_stage2_smoke_reject_20260711` (38 payload files, ~2.03 MB, manifest SHA `17b4ad7...`); cleanup removed 293 source files and observed `193.090 MB` gain.
- [x] Retention `runs\artifact_retention_audit_after_ldrkl_cleanup_20260711`: 535 directories, pass, `blockers=[]`, protected keepers present, compacted originals absent.
- [ ] Build a no-train ALDR uncertainty-readiness audit from aligned validation probabilities. Require a predeclared error-ranking and class1 FN/FP directional gate before implementing or smoking adaptive LDR.
- [x] Implemented diagnostic-only `audit_adaptive_ldr_readiness` with the official detached-lambda formula, source-group bootstrap, class1 FN/FP ordering, target-gradient telemetry, locked-test refusal, and focused tests (`4/4`).
- [x] Full keeper ALDR audit `runs\diagnostic_adaptive_ldr_readiness_keeper_20260711` covered train/val `9215/2606`: validation overall-error/FP/FN AUROC `0.71463/0.62182/0.84691`; FP-vs-FN AUROC only `0.24637`, so true class1 FN receive systematically larger robust lambda than FP.
- [x] ALDR effect-size gate failed: validation lambda p90-p10 span `0.01619 < 0.05` and median target-gradient change `0.000715 < 0.01`; train values `0.01442/0.000705`. It is numerically the rejected fixed LDR route.
- [ ] Do not implement, smoke, or sweep adaptive LDR alpha/lambda/margin on the current keeper. Reopen only if a new representation produces both materially wider calibrated uncertainty and FP-safe class1 ordering.
- [x] Researched CVPR-2017 forward loss correction and the ICLR-2017 noise-adaptation layer, then implemented no-test `audit_class_noise_transition_readiness`; aligned OOF/validation proxy matrices, locked-test guards, and focused uncertainty/transition tests passed (`6/6`).
- [x] Full transition audit `runs\diagnostic_class_noise_transition_readiness_20260711`: EffV2 class1 diagonal changed `0.9764 -> 0.6947` with train-val class1-row L1 `0.5636`; DINO class1 diagonal only `0.4628/0.4486`; cross-expert class1-row L1 `1.0272` train and `0.5311` validation.
- [ ] Do not implement or sweep class-conditional forward/backward correction, noise-adaptation layers, or pseudo-clean transition matrices from current EffV2/DINO caches. The transition process is not identifiable and would encode expert mistakes as label corruption.
- [ ] Run one fixed native-224 EfficientFormerV2-S0 scratch gate after resource/smoke/XAI validation. Require epoch-5 macro/class1 at least CoAtNet-Nano's `0.8303/0.5534`; otherwise stop without image-size/LR/loss/backbone-neighbor sweeps.
- [x] Researched the EfficientFormerV2 ICCV-2023 paper/official code, added `efficientformerv2_s0` to the compact scratch launcher allowlist, verified native-224 BF16 throughput/resource use, and completed a no-test pipeline smoke with mandatory same-12-case XAI.
- [x] Fixed 120-batch/5-epoch/full-val EfficientFormerV2-S0 probe failed the CoAtNet gate: epoch-5 macro/class1 `0.80767/0.49890`; highest raw macro at epoch 4 was `0.80890/0.51364`; the fair selector retained epoch 3.
- [x] Independent selected-checkpoint reload reached only macro/class1 `0.79915/0.51016`, class1 P/R `0.38699/0.74834`, with 113 TP, 176 FP, 38 FN, and 853 boundary candidates versus CoAtNet's 584.
- [x] Final same-case XAI was correct on 3/12; attention/Grad-CAM foreground `0.88101/0.88577`, background blur/gray drops `-0.00376/-0.00481`, object-desaturation `0.12554`. Visual review retained stem/glove/padding/boundary shortcuts despite object focus.
- [x] Compacted EfficientFormerV2 evidence to `runs\evidence_compacthybrid_efficientformerv2_s0_yolof_scratch_5e_reject_20260711` (52 payload files, 4.657 MiB, SHA `01fc0cfa...`), deleted six source roots/238 files, observed `192.508 MiB` free-space gain, and passed retention over 539 directories with no blockers.
- [ ] Do not continue/sweep EfficientFormerV2-S0 or neighboring EfficientFormer variants, image size, LR, loss, sampler, teacher, normalization, or epochs. Stock scratch substitution is now closed; a future GPU route needs genuinely new supervision/problem formulation and a fold-safe/manual class1 FN-versus-FP readiness signal.
- [x] Replaced fragile multiline VS Code commands with `scripts\run_trkh_current_best_full_pipeline.ps1`: one-line absolute invocation, direct native stderr handling, strict `$LASTEXITCODE`, train-side `SkipFinalTest`, explicit post-validation final test, architecture trace, raw/softboost eval, forensics, confusion, boundary, robustness, XAI/transition, and retention audit.
- [x] Added separate `run_trkh_export_engine.ps1` and `run_trkh_test_video.ps1`; export preflight verified TensorRT `10.7.0`, CUDA, and `trtexec`, while commands default to `runs\latest_full_pipeline.json` or accept explicit checkpoint/run paths.
- [x] Fixed classification-only TensorRT video: no longer requires bbox engine output; shared top-k/confidence fallback, temporal smoothing, drift, and overlay logic with PyTorch. Compile plus command/video focused tests passed `6/6`.
- [x] PowerShell AST parse passed all three wrappers; exact full-pipeline `-PreflightOnly -RunFinalTest` passed against current keeper, verifier, immutable `yolo_f`, 30-epoch cap, and GPU without creating a train/test run.
- [x] Updated `docs\TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt` with six one-line VS Code commands: preflight, complete train-to-final-audit pipeline, latest engine export, explicit keeper export, TensorRT video, and PyTorch video fallback.
- [ ] Whenever a genuinely better selected checkpoint appears, update full-pipeline defaults and the command TXT together, then update journal/TODO/skill and rerun parser, full preflight, engine preflight, and focused tests. Do not update from smoke-only, validation-oracle, or test-tuned evidence.
- [x] Refreshed the deep-research reference-pack staging to 266 payloads (~16.82 MiB uncompressed), with 63 selected follow-up artifacts covering reports 14/15, validation ceiling, CoAtNet, EfficientFormerV2, LeFF, concurrent coupling, LDR/ALDR, transition-noise readiness, and current train/export/video infrastructure.
- [ ] Preserve the command-promotion invariant: only a checkpoint that passes independent locked-validation selection may replace the current full-train recipe. Promotion must atomically synchronize wrapper defaults, command TXT, metrics, journal/TODO/skill, parser/preflights/tests, and the deploy pointer.
- [x] Verified and atomically replaced the research ZIP: 266/266 payload hashes, 267 entries with manifest, zero prohibited/extra entries, 8.537 MiB compressed, SHA-256 `82be9ffd37b4d6361e5cddf11313a5f065dc64d524228324ebe260c48cd49385`.
- [x] Removed the verified pack staging tree and passed final retention over 540 run directories using all 22 cleanup manifests: 67/67 compacted originals absent, protected artifacts complete, `blockers=[]`.
- [x] Researched Knowledge Review, Feature Distillation Overhaul, Masked Generative Distillation, and fine-grained noisy-label learning, then added no-test `audit_multistage_teacher_feature_readiness` plus four focused tests.
- [x] Full generic EfficientNetV2-S multi-stage audit aligned all `9215/2606` `class_f -> yolo_f sample_index` rows with zero missing/mismatch. The signature was non-collapsed (effective rank `37.10`) and stage-energy maps visibly localized fruit/surface evidence without border domination.
- [x] The predeclared head+multi-stage readout reached only validation macro/class1 `0.85619/0.57658` versus direct keeper `0.88407/0.68406`; transitions were `102` corrections versus `135` harms, class1 FN-rescue/TP-break `11/33`, and FP-remove/create `52/62`.
- [ ] Do not implement or sweep generic multi-stage teacher feature distillation on the current keeper. Train-OOF versus validation FN/FP direction AUROC flipped `0.30735 -> 0.68820`, so `smoke_permission=false`; preserve `runs\diagnostic_multistage_generic_effv2s_readiness_20260711` as the no-repeat artifact.
- [x] Captured pre-cleanup worktree hash baseline at `runs\worktree_hygiene_baseline_20260711`: branch/upstream at the same commit, staged `0`, tracked modified `33` (`+45,782/-6,020`), untracked `269` (`12.336 MiB`), and `git diff --check` findings `0`.
- [ ] Follow `docs\TRKH_WORKTREE_HYGIENE_20260711.md`: classify ownership/dependencies first, verify one explicit-path batch at a time, inspect the cached diff, then commit. Never bulk-normalize line endings or revert/delete unclear `BaoCao`, root research reports, datasets, keepers, or generated evidence.
- [ ] Before the next GPU method, require another genuinely distinct supervision/problem formulation with train-only/fold-safe class1 FN-versus-FP direction. Multi-stage generic teacher transfer joins pooled RKD/CRD, teacher critic, label-noise correction, and current-embedding objectives on the closed list.
- [x] Collected the full worktree test surface (`620` tests), then ran it. Initial result was `616 passed / 4 failed`, all in accumulated shared contracts rather than the new teacher diagnostic.
- [x] Reconciled classification dataset tests with the intended hard-label plus pad-`image_mask` metadata contract. Synchronized the hybrid DETR constructor with bbox-spatial-fusion and teacher-feature-projection fields; automated signature comparison now finds zero unconsumed fields across all 270 `ModelConfig` entries.
- [x] Revalidated compileall, all 33 PowerShell scripts, focused model/dataset suites (`20/20`), EOF regression (`1/1`), `git diff --check`, and the complete suite (`620/620` in 61.63s).
- [x] Created explicit-path preservation commit `4c2c7cb` for the verified runtime/config/scripts/tests snapshot: 275 files, no dataset/model binary/secret signature, staged patch SHA `61000013...a9a6c`. Worktree entries dropped `302 -> 28`; index is clean.
- [x] Completed the separate 21-path research-document batch as `7548c1f`: UTF-8/conflict/link checks passed, ZIP companion and all 266 stream hashes matched, reports 14/15 matched archived bytes, and only 15 intentional report-source Markdown hard breaks were allowlisted.
- [x] Pushed runtime commit `4c2c7cb` and docs commit `7548c1f` to `origin/classification-only-research`; independent `ls-remote` matched local/remote head `7548c1f` with ahead/behind `0/0`.
- [ ] Keep all five `BaoCao/*` files plus root `deep-research-report (9|10).md` untouched until ownership/scope is established. These seven paths are the only remaining worktree entries and are deliberately not treated as cleanup candidates.
- [x] Researched invariant wavelet scattering, Kymatio, wavelet-integrated CNNs, and parametric scattering, then added train/validation-only `audit_wavelet_scattering_readiness`, pinned `kymatio==0.3.0`, and passed five focused tests plus a real `[2,6,61,16,16] -> [2,2196]` descriptor check.
- [x] Full fixed wavelet audit `runs\diagnostic_wavelet_scattering_rbf_c3_readiness_20260711` strictly aligned `class_f -> yolo_f` over `9215/2606` rows and `8064` train source groups. The predeclared PCA-128/RBF-SVC reached OOF macro/class1 `0.84388/0.51383` and validation `0.85632/0.63118`, below the direct keeper `0.88407/0.68406`.
- [x] Reviewed both prediction audits and the five-class first/second-order energy preview. Validation changed `248` rows (`88` corrections, `138` harms), removed/created class1 FP `56/9`, but rescued/broke class1 FN/TP `9/44`; train OOF was much worse at `2/270`. FN-vs-FP direction AUROC changed `0.38087 -> 0.64872`, and standardized projected effective rank was only `11.64` despite PCA-128 explaining `0.98808` variance.
- [ ] Do not implement or sweep a fixed Kymatio scattering branch, J/L/order, channel set, PCA, SVM kernel/C/gamma, fusion threshold, or wavelet post-hoc router on the current keeper. The validation binary oracle class1 F1 `0.85235` is label-assisted complementarity, not fold-safe permission; `smoke_permission=false`, no checkpoint was created, and current-best commands remain unchanged.
- [x] Wavelet closure verification passed compileall, full pytest `625/625`, `git diff --check`, and retention `runs\artifact_retention_audit_after_wavelet_scattering_precheck_20260711` over `544` run directories with `blockers=[]`. `pip check` still reports pre-existing shared-venv pin conflicts for MambaVision/OpenCV; do not alter those unrelated packages as part of this diagnostic.
- [x] Researched NTS-Net, ELoPE, compact joint localization/classification, Cross-X, and InsLoc, then implemented a state-reset, allowlist-audited classifier<->detector curriculum checkpoint builder plus focused tests.
- [x] Added a fail-closed runtime contract for generated detector checkpoints and propagated it through descendant best/last/interrupt checkpoints. Fresh curriculum payloads strip stale resume/calibration/stage/data-summary/EMA-selection state. The initial cropped benchmark ignored 1,151 secondary boxes and is invalid; corrected detector runs must include `--full-image-detection --skip-final-test` and log train `8064/9215`, validation `2577/2606`, `ignored_objects=0`.
- [x] Benchmarked corrected full-frame DETR at batch 8 and 32. Batch 32 used about `2,656.7 MiB` allocated / `3,098 MiB` reserved and was selected for the fixed two-epoch, 120-batch probe.
- [x] The detector reached matched-query macro/class1 `0.928815/0.769760`, bbox IoU `0.483235`, and detection F1@0.5 `0.414182`. Treat matched-query class scores as ground-truth-assisted diagnostics, never as a promotion gate.
- [x] Detector encoder round-trip preserved the original classifier head bit-exact but reduced pre-finetune macro/class1 to `0.782285/0.522167`. Two unchanged V8 epochs recovered only to independent raw `0.874127/0.659574`, failing keeper minima `0.882925/0.678261`; no test was run and no command/deploy update was made.
- [x] Reviewed all required audits: aligned candidate-vs-keeper rows (`28` corrections / `50` harms, class1 FP remove/create `8/33`), raw and softboost prediction audits, boundary evidence, architecture trace, 24-case raw/softboost XAI, and full corruption robustness.
- [x] Fixed robustness evaluation to use classification object crops and preserve bbox/crop-bbox/pad-mask metadata through transforms and the exact bbox-prior forward path. Added regression tests; final clean macro/class1 was `0.873288/0.657825`, with much larger dim/bright/contrast degradation than center occlusion.
- [ ] Do not repeat or sweep direct DETR-20/depth-2 localization loss, LR, batch, epochs, verifier thresholds, or the same unrestricted detector-encoder round-trip plus current V8 fine-tune. Localization is empirically adequate; the next route must target fold-safe photometric/surface separability and conservative class1 FP control.
- [x] Compacted the rejected curriculum into `runs\evidence_detcls_fullframe_curriculum_rejected_20260711` (89 payloads, no model/deploy binary; manifest SHA `32abb934...808f62`), deleted exactly seven guarded source roots, and observed `1,458,147,328` bytes reclaimed.
- [x] Closure passed compileall, full pytest `637/637`, `git diff --check`, and retention over `547` run directories with `blockers=[]`. Current-best full-train/export/video commands remain unchanged.
- [x] Screened same-frame matched-illumination supervision before implementation: train has only `306` cross-class object pairs, while validation has zero mixed-label pair and zero class1 multi-object frame. Close this route because independent FN/FP transfer cannot be measured.
- [x] Researched CLCC, learning-based simple-feature illuminant estimation, Convolutional Color Constancy, FC4, and CNN color constancy, then added fixed no-test `probe_context_illuminant_estimator_readiness` plus four focused tests.
- [x] Full source-grouped estimator diagnostic covered `8064/2577` train/validation sources and all `2606` validation objects. Fixed 67D context-outside-all-bboxes statistics plus 96-tree ExtraTrees recovered eight synthetic gains with min OOF/val R2 `0.98857/0.98282` and max log-MAE `0.00409/0.00583`.
- [x] Reviewed every raw/corrected condition, per-class F1, prediction transition, pixel reconstruction, and the five-class warm-cast preview. Near-neutral clean correction changed 52 rows (`8` corrections / `40` harms), rescued/broke class1 FN/TP `1/6`, and removed/created class1 FP `5/20`; macro/class1 fell `0.884675/0.686047 -> 0.867572/0.638418`.
- [x] Known-gain oracle ceiling proved the remaining failure is not estimator quality: exact inversion still reached only bright `0.764950/0.399142`, red `0.852863/0.596774`, green `0.824396/0.500000`, and warm `0.863249/0.628099` because clipping/sRGB nonlinear processing destroys maturity evidence.
- [ ] Do not sweep context-illuminant estimator family/tree count/descriptors, gain strengths, correction shrinkage/dead-zone, or implement an FC4/context-CNN branch on this sRGB route. Eight readiness gates failed and even the true-gain oracle is below the clean/class1 envelope; no GPU smoke is permitted.
- [x] Retained six-payload evidence at `runs\diagnostic_context_illuminant_estimator_full_20260711` (`6,140,666` bytes, manifest SHA `010c610d...58c76`, no model/deploy binary), removed the superseded 100/50 dry-run (`581,496` bytes), and passed py-compile, focused tests `4/4`, full pytest `641/641`, `git diff --check`, and retention over `549` directories with `blockers=[]`.
- [x] Researched API-Net, DCAL/PWCA, and Pairwise Confusion, then added fixed no-test `probe_api_pairwise_interaction_readiness` plus three focused tests. This is a training-only adaptive interaction precheck, not another deploy-time embedding selector.
- [x] Full cache audit covered `9215/2606` rows and five source-grouped folds with zero leakage. Natural-frequency batching visited every anchor exactly 20 times without anchor oversampling; all classes had nearest intra/inter support and every one of `368600` final pairs excluded the same source.
- [x] Pooled API interaction failed in all folds: matched-control OOF macro/class1 `0.911297/0.740431` became `0.907701/0.726531`; validation `0.848918/0.602041` became `0.846950/0.597590`, far below direct keeper `0.884675/0.686047`.
- [x] Reviewed pair plans, fold/final curves, per-class confusion, full OOF/validation prediction audits, and transition directions. Versus validation control, API made `15` corrections and `22` harms and removed/created class1 FP `2/19`; versus direct keeper it made `25/111` corrections/harms and `4/69` FP remove/create.
- [x] Probability-direction audit found class-unsafe smoothing: validation mean API-control class1 probability delta was `-0.069506` on true class1 and `+0.042589` on negatives, with `2303/2455` negatives raised. Lower pair-conditioned train loss therefore does not transfer to unloaded single-image inference.
- [x] Self-review found symmetric API partner-label amplification: class1 was `5.871%` of natural anchors but `14.535%` of selected partners (`53576/368600`). Added explicit `anchor_only|symmetric` loss scope, partner reuse telemetry, and a test proving anchor-only ignores partner labels.
- [x] Ran exactly one corrected anchor-only diagnostic with every other setting fixed. OOF control/API macro-class1 `0.929138/0.798137 -> 0.928312/0.795699`; validation `0.862903/0.628743 -> 0.863613/0.631268`. Only 2/5 folds were positive and gains remained below gate.
- [x] Anchor-only still learned the wrong probability direction: true-class1 mean p1 delta `-0.082786`, non-class1 `+0.030825`, with `2335/2455` negatives raised. Versus direct keeper it made `24/64` corrections/harms, broke/rescued class1 TP/FN `14/3`, and removed/created FP `10/16`.
- [ ] Do not integrate or sweep either pooled-head API loss scope, MLP width, optimizer/LR, class weighting, batch size, ranking margin/weight, or epochs on this keeper cache. `smoke_permission=false`; no test/model checkpoint/current-best command update is allowed.
- [x] Retained compact API evidence at `runs\diagnostic_api_pairwise_interaction_full_20260711` (7 payloads, `3464753` bytes, manifest SHA `e4c84711...770d0`, no model/test) and passed py-compile, focused pytest `3/3`, manifest verification, and `git diff --check`.
- [x] API closure passed compileall, full pytest `644/644`, unchanged command SHA `3c718130...57dd`, and retention `runs\artifact_retention_audit_after_api_pairwise_precheck_20260711` over 551 directories with `blockers=[]`.
- [x] Retained corrected anchor-only evidence at `runs\diagnostic_api_pairwise_anchoronly_full_20260711` (7 payloads, `3485228` bytes, manifest SHA `d65f25b1...9bc83`, no model/test); all hashes, row counts, source/pair audits, and empty stderr were verified.
- [x] Final pooled-API closure passed compileall, focused pytest `4/4`, full pytest `645/645`, `git diff --check`, unchanged current-best command hash, and retention `runs\artifact_retention_audit_after_api_anchoronly_precheck_20260711` over 553 directories with `blockers=[]`.
- [ ] Token-level DCAL/PWCA remains conceptually distinct from pooled API, but do not jump directly to a GPU implementation. First require a low-cost, source-safe token-interaction proxy or another new supervision formulation that demonstrates conservative class1 FP direction on validation without using test.
- [x] Re-read DCAL/PWCA and its supplement: natural random same-dataset distractors outperform intra-only, inter-only, 1:1, noise, and external-data pair policies. Implemented label-blind random different-source pairing rather than another nearest-pair API variant.
- [x] Added fixed PWCA token-cache extractor and shared-weight SA/PWCA readiness proxy plus five focused tests. Full cache covers `9215/2606` rows with zero source overlap; top-32 rank-64 tokens retain mean attention mass `0.886970/0.888785` and direct val remains exactly `0.884675/0.686047`.
- [x] Reviewed train/validation five-class top-token previews. Most selected tokens cover fruit surface, spots, lesions, and decay; occasional stem/background edges remain, and p05 bbox fraction is `0.46875`. Do not tune top-k/rank from this validation view.
- [x] Full five-fold natural-pair proxy used `184300` final pairs, zero same-source pair, no anchor oversampling/partner label loss, and natural inter-class fraction `0.773880`. Every row was anchor and partner exactly 20 times.
- [x] PWCA has positive but sub-gate transfer: OOF control/candidate `0.933468/0.805324 -> 0.934686/0.810855`; validation `0.874168/0.636656 -> 0.874874/0.647436`. Gains miss fixed macro/class1 thresholds and candidate remains below direct keeper and class1 `0.70`.
- [x] Reviewed all fold curves, per-class confusion, OOF/validation prediction audits, natural-pair telemetry, and PWCA attention. Validation versus control made `15/18` corrections/harms and FP remove/create `4/3`; versus keeper it made `38/47`, removed/created FP `22/7`, but broke/rescued class1 TP/FN `19/2`.
- [x] Distractor attention mean `0.329867` passed, but p95 `0.637411` failed the `0.55` tail gate; class1 FN/FP p95 remained above `0.60`. `smoke_permission=false`, so no image-model smoke/test/current-best command update is allowed.
- [ ] Do not sweep rank/top-k/heads/dropout/optimizer/LR/PWCA weight/pair policy/epochs on this frozen-token proxy. Revisit PWCA only through a genuinely keeper-preserving residual formulation with a new recall-protection gate, not by relaxing the current thresholds.
- [x] Compacted PWCA evidence to `runs\diagnostic_pwca_token_rank64_top32_full_20260711` (14 payloads, `4176398` bytes, manifest SHA `bbe9fea1...42b0b6`, no model/test), then deleted the 11-file temporary cache (`53500161` bytes; observed `53526528` free-byte gain) under `runs\cleanup_manifest_20260711_pwca_token_cache_rejected.json`.
- [x] PWCA closure passed py-compile, focused pytest `5/5`, `git diff --check`, unchanged current-best command hash, and retention `runs\artifact_retention_audit_after_pwca_token_precheck_20260711` over 555 directories with `blockers=[]`.
- [x] Final PWCA code/docs snapshot passed compileall and full pytest `650/650`.
- [x] Audited the exact keeper-relative scalar residual `log(p_keeper)+alpha*(log(p_pwca)-log(p_control))` before regenerating token caches or using GPU. Selection used only train OOF with fixed macro/class1/recall/FP preservation constraints; validation was transfer-only and test was forbidden.
- [x] Of 201 fixed coefficients in `[-1,1]`, only identity `alpha=0` was eligible. Train OOF stayed `0.940013/0.816358` and validation stayed `0.884675/0.686047`; therefore the retained PWCA/control logits provide no nonzero keeper-preserving action.
- [ ] Do not sweep scalar/class-specific PWCA residual coefficients, grids, thresholds, or post-hoc routers. A future trainable zero-init token adapter is distinct, but it now requires a different train-only supervision signal and must prove a nonzero recall-safe FN-versus-FP action before token-cache regeneration or GPU work.
- [x] Retained `runs\diagnostic_pwca_keeper_relative_residual_full_20260711` (5 payloads, `2525541` bytes, manifest SHA `6996ed28...974f`, no model/checkpoint/test). Current-best commands remain unchanged.
- [x] PWCA residual closure passed compileall, focused API/residual tests `7/7`, full pytest `653/653`, manifest hash verification, `git diff --check`, unchanged command SHA, and retention over 557 directories with `blockers=[]`.
- [x] Researched primary MaskFeat and implemented a no-test RGB-HOG target readiness audit: separate RGB histograms, 9 unsigned bins, 8x8 cells, local L2 normalization, strict `class_f -> yolo_f` alignment, five source-grouped folds, and fixed PCA-128 linear/RBF controls.
- [x] Self-review corrected the first pooled-8x8 proxy to preserve the complete 16x16 HOG cell grid (`6912` dimensions). Standardized projected rank rose to `61.822`, but primary RBF OOF/val macro-class1 remained only `0.733291/0.331096` and `0.656872/0.262911` versus keeper val `0.884073/0.684058`.
- [x] Reviewed full transitions and five-class RGB/channel-energy preview. HOG removed/created class1 FP `66/24` but rescued/broke FN/TP only `3/93`; FN-vs-FP AUROC inverted `0.309054 -> 0.736045`, and maps emphasize contours/stems/scratches/speckles/background edges rather than a recall-safe interior maturity cue.
- [ ] Do not implement or sweep MaskFeat-HOG pretraining/auxiliary heads, masks, bins, cells, channel spaces, decoder/target weights, schedules, or HOG routers on current data. Eight readiness checks failed and the paper's hundreds-of-epochs regime is incompatible with the project budget without a much stronger train-OOF signal.
- [x] Retained full-grid evidence at `runs\diagnostic_maskfeat_hog_rgb9c8_fullgrid16_readiness_20260712` (10 payloads, `3906228` bytes, manifest SHA `2782c403...5886a`, no model/checkpoint/test), preserved the pooled summary/manifest, deleted exactly nine superseded files (`3882869` bytes; observed `3903488` free bytes), and passed retention over 559 directories with `blockers=[]`.
- [x] MaskFeat-HOG closure passed py-compile, compileall, focused HOG/wavelet tests `8/8`, full pytest `656/656`, manifest hash verification, and `git diff --check`; current-best commands/deploy pointer remain unchanged.
- [x] Researched primary DINOv2/model-card/source and added a locked no-test dense-patch readiness audit plus four focused tests. The fixed 224px ViT-S/14 comparison uses CLS-only versus `CLS+patch mean/std+2x2 spatial means`, the same scaler/PCA-128/readouts, and all `9215/2606` strictly aligned rows.
- [x] Self-review rejected the first completed output because control/candidate used different grouped-fold seeds. Fixed the tool to lock one model and common seed, recorded `matched_source_folds=true`, reran all folds, and did not use the unmatched result for a decision.
- [x] Matched-fold dense patches improved RBF OOF macro/class1 `0.872683/0.581068 -> 0.880335/0.603738`, but validation regressed `0.887625/0.664384 -> 0.882850/0.636364`, below keeper `0.884073/0.684058`.
- [x] Reviewed both readouts, every fold, per-class confusion, complete prediction audits, transitions, probability direction, binary oracle, and five-class patch-energy preview. Versus keeper, dense patches removed/created 49/17 class1 FP but broke/rescued 31/4 true-class1 TP/FN; train/val direction AUROC shifted `0.592613 -> 0.771886`.
- [ ] Do not implement or sweep DINOv2 dense target distillation, input/model size, grid/moments/pooling, PCA/SVM, thresholds, adapters, scalar residuals, or routers. On keeper FN rows dense-minus-CLS p1 delta is negative on train/validation (`-0.035150/-0.088202`), so oracle class1 `0.813333` is not smoke permission.
- [x] Retained corrected DINOv2 dense evidence at `runs\diagnostic_dinov2_dense_patch_semantic_surface_matchedfolds_readiness_20260712` (11 payloads, `3866659` bytes, manifest SHA `7a44e788...bceb5f52`, no model/checkpoint/test), preserved the invalid summary/manifest, deleted exactly ten superseded files (`3823398` bytes; observed `3846144` free bytes), and passed retention over 561 directories with `blockers=[]`.
- [x] DINOv2 dense closure passed py-compile, compileall, focused tests `4/4`, full pytest `660/660`, evidence/cleanup/retention hash assertions, and `git diff --check`; current-best command SHA remains `3c718130...57dd`.
- [ ] Next candidate must be a genuinely different supervision/problem formulation and pass a source-safe keeper-FN-versus-FP gate. Do not infer permission from a representation that only suppresses class1 FP by sacrificing true class1 recall.
- [x] Cross-checked TEDL, Re-EDL, and the official Re-EDL source, then added a locked no-test two-stage evidential readiness audit plus five focused tests. The zero-init residual preserves keeper logits initially; CE 20e is shared, followed by matched CE/Re-EDL 10e branches under the same source folds, optimizer settings, and batch order.
- [x] Full cache audit covered `9215/2606` rows and `8064/2577` source groups with zero fold or train/validation source overlap. Natural-frequency sampling, 30-epoch cap, CPU-only execution, and no test/raw-data/checkpoint/trainable-manifest guards all held.
- [x] Re-EDL failed transfer: OOF CE/Re-EDL macro-class1 `0.939108/0.817204 -> 0.938132/0.821662`, but validation `0.877448/0.640288 -> 0.858718/0.590406`, below direct keeper `0.884675/0.686047`; candidate class1 P/R was `0.666667/0.529801`.
- [x] Reviewed every fold, loss curve, full prediction transitions, calibration, risk-coverage, and uncertainty plots. Versus keeper, Re-EDL removed/created class1 FP `38/3` but rescued/broke FN/TP `1/39`; val error AUROC/AURC `0.838151/0.022503` trailed CE `0.885434/0.013678`, and 80% coverage retained only `35/151` class1 rows.
- [ ] Do not implement or sweep generic EDL/Re-EDL/TEDL heads, prior weight, CE/evidential epoch split, LR/weight decay, residual capacity, class weights, calibration, uncertainty thresholds, or routers on the current keeper representation. Stable FN/FP delta direction is relative to a recall-damaging CE residual and is not smoke permission.
- [x] Retained `runs\diagnostic_two_stage_reedl_keeper_residual_readiness_20260712` (10 payloads, `5670155` bytes, manifest SHA `24a71f28...177da1b`, no model/checkpoint/test); `smoke_permission=false` and current-best commands remain unchanged.
- [x] Re-EDL closure passed py-compile, compileall, focused tests `5/5`, full pytest `665/665`, manifest verification, `git diff --check`, unchanged command SHA, and retention over 563 directories with `blockers=[]`.
- [ ] With current-embedding CE/soft-target/EDL/DAC/readout families now closed, the next candidate must introduce a genuinely new image-derived surface signal or a formally different supervision source, then pass direct keeper-relative recall and FP gates before any GPU smoke.
- [x] Implemented a train-only joint classification/localization gradient-readiness audit using exact crop/full-frame pairs, frozen keeper features, held-out audit sources, independently normalized objectness/localization components, and PCGrad projection. All local geometry checks passed, including class1 retained norm median `0.920122` and positive one-step class1 margin.
- [x] Rejected microbatch AdamW as an implementation mismatch for projected-gradient evidence: coordinate-wise optimizer normalization destroyed the intended equal-direction comparison. The official smoke instead used one manual encoder macro-step with control/candidate L2 step ratios both `1e-4` and absolute norm delta `2.315e-9`.
- [x] Full-val joint spatial-PCGrad smoke failed: keeper/control/candidate macro-class1 `0.882925/0.678261`, `0.874714/0.654867`, `0.873916/0.652941`; candidate versus control made `7/8` corrections/harms, removed/created `2/3` class1 FP, and failed six fixed gates. No probe/test/current-best update was permitted.
- [x] Reviewed full forensics, architecture trace, all 16 changed cases, candidate/keeper XAI, and contact sheets. Candidate Grad-CAM foreground fell `0.910216 -> 0.890438`, border mass rose `0.210001 -> 0.242096`, and every candidate changed case became a near tie. Wide context supplied border/hand/stem evidence rather than a transferable class1 surface cue.
- [x] Fixed the prediction-export contract to include canonical `target_index/prediction_index/y_true/y_pred`; added regression coverage after the old `label/prediction` schema caused an invalid near-random forensics result.
- [x] Added a fail-closed CLI guard so the rejected `adamw_microbatch` path requires a second explicit `--allow-rejected-adamw-ablation` flag; focused tests pass `11/11`, compileall and full pytest pass `676/676`.
- [ ] Do not repeat direct sequential detection curriculum, raw/total detection gradients, microbatch AdamW PCGrad, or sweep macro-step size, auxiliary ratio, detector warm-up, seed, query count, PCGrad/GradNorm variants. Reopen joint localization only after a new representation changes direct keeper FN/FP support and a matched classification-only control preserves the keeper.
- [x] Retained the audited joint smoke as 492 payloads (`40643232` bytes, SHA `37d39be8...eedc6`, no checkpoint/model/test), deleted two rejected checkpoints plus four exact temp preflights under a two-phase manifest, reclaimed `178580777` bytes, and passed retention over 566 directories with `blockers=[]`.
- [ ] Continue scientific worktree cleanup in dependency-coherent batches: lock protected user paths and current-best hash, classify every dirty path, stage explicit files only, inspect cached diff, test, commit/push, then rerun status. Never bulk-normalize, revert unclear changes, or delete generated evidence without a manifest and retention audit.
- [ ] Next candidate must change interior surface representation itself. Require a no-test, source-safe, matched-control readiness result that preserves keeper macro/class1/recall and demonstrates both keeper-FN rescue and conservative FP behavior before allocating another image-model smoke.
- [x] Re-read primary BBN, RIDE, ResLT, and SADE sources, then isolated the BBN natural-versus-reversed classifier-space claim before allocating another multi-expert image branch. The fixed protocol uses `C=0.3`, five source-grouped folds, exact reversed class mass proportional to `1/N_i`, and raw-logit `alpha=0.5` fusion with no sweep.
- [x] Exported and manifested a reusable frozen keeper cache covering all `9215/2606` train/validation rows, `8064/2577` source groups, and 256D embeddings (`12496767` bytes, payload SHA `86df8615...90e6`); no test/model/checkpoint/raw-data edit.
- [x] Self-review corrected in-sample keeper labeling, prohibited cross-scale keeper/logistic probability-delta claims, and added a hard train/validation source-overlap gate. The final source-guarded output has zero train/validation and OOF fit/hold overlap with bit-identical prediction CSVs across reruns.
- [x] BBN bilateral improved conventional OOF macro/class1 only `0.934635/0.802208 -> 0.936151/0.809365`, then reached validation `0.880665/0.666667`, below direct keeper `0.884073/0.684058`. Same-scale FN-versus-FP AUROC inverted `0.636277 -> 0.396057`.
- [x] Reviewed full folds, class metrics, weighting mass, prediction transitions, direction audit, README, manifest, and plot. Versus keeper, bilateral removed/created 32/7 class1 FP but rescued/broke only 1/18 FN/TP; six fixed gates failed and `image_smoke_permission=false`.
- [ ] Do not sweep BBN alpha/C/reversed weights/folds or launch a same-representation BBN/RIDE/ResLT/SADE image branch. Reopen multi-expert work only after a new representation independently preserves keeper macro/class1/recall and changes direct keeper FN/FP support.
- [x] Retained source-guarded BBN evidence (8 payloads, `6199651` bytes, SHA `c6655874...aea45b`), deleted two superseded roots under exact hash manifests (`12401130` bytes; observed free gain `12443648` bytes), and passed retention over 570 directories with `blockers=[]`.
- [x] BBN batch closure passed py-compile, compileall, focused tests `6/6`, full pytest `682/682`, manifest/plot/retention checks, unchanged keeper and command SHA `3c718130...57dd`, and explicit-path worktree review. Keep the seven protected user paths outside every stage/cleanup operation.
- [x] Re-read primary multiresolution uniform LBP and color-constant LBP, then added a locked no-test interior-surface readiness audit. The fixed 28D descriptor uses grayscale uniform `(P=8,R=1)` plus `(P=16,R=2)` histograms on the runtime object's 15%-eroded `128x128` ROI, five source-grouped folds, and a zero-init no-bias keeper-log-probability residual.
- [x] Self-review rejected the first 4096-row free-bias/summed-CE preflight as class-prior recalibration, then used a no-bias natural-frequency mean-CE `C=0.3` residual. The corrected partial prefix was still class-biased, so only the full `9215/2606`, `8064/2577`-source run with zero train/validation and fold overlap is decision evidence.
- [x] LBP failed direct transfer: standardized effective rank `4.386422`, standalone validation macro/class1 `0.368847/0.000000`, train in-sample keeper/grouped-OOF residual `0.939922/0.815729 -> 0.923667/0.814516`, and validation `0.884073/0.684058 -> 0.864965/0.645963`; class1 recall fell `0.781457 -> 0.688742`.
- [x] Reviewed all folds, descriptor normalization/rank, full OOF/validation predictions, transitions, FN-vs-FP direction, and the 13-row RGB/LBP contact sheet. Validation made `29/68` corrections/harms, removed/created 14/5 class1 FP, and rescued/broke only 1/15 FN/TP; direction AUROC was `0.484949/0.460128`.
- [ ] Do not sweep LBP channels, point/radius scales, ROI erosion/size, residual C/iterations, fold seeds/class balancing, or nearby LTP/CLBP variants. Fifteen checks failed and `image_smoke_permission=false`; reopen only after a new representation/supervision source changes direct keeper FN/FP support.
- [x] Retained full LBP evidence at `runs\diagnostic_lbp_surface_texture_full_20260712` (20 payloads, `6496709` bytes, SHA `195c8a0e...9f52919`, no model/checkpoint/test), compacted both preflight summaries, deleted 20 superseded files (`6665614` bytes; observed `6713344` free-byte gain), and passed retention over 572 directories with `blockers=[]`.
- [x] LBP closure passed py-compile, compileall, focused tests `6/6`, full pytest `688/688`, artifact/cleanup/retention hash checks, `git diff --check`, and explicit protected-path review; keeper and command SHA `3c718130...57dd` remain unchanged.
- [ ] Keep current-best full-train/export/video commands unchanged: LBP did not pass the independent validation gate. Continue worktree cleanup only through explicit dependency-coherent stages; the seven protected user paths remain out of scope.
- [x] Re-read the BioCLIP CVPR paper, official BioCLIP release, and OpenCLIP, then added a locked no-test matched-feature audit plus nine focused tests. Exact revisions/SHA-256 fail closed; OpenAI/BioCLIP use equal 86.193M-parameter ViT-B/16 vision towers, 512D normalized features, and bit-identical pixels, while the official QuickGELU/GELU difference is disclosed.
- [x] Corrected class-folder numeric-order and case-sensitive source-group alignment hazards before accepting the full run. Final coverage is `9215/2606` rows and `8064/2577` groups with zero train/validation or fold fit/hold overlap; no prompt, test, raw-data write, model, checkpoint, or trainable manifest.
- [x] BioCLIP improved matched OpenAI grouped OOF macro/class1 `0.832440/0.447581 -> 0.839207/0.474383`, but reversed on validation `0.851003/0.552448 -> 0.819665/0.444444`, below keeper `0.884073/0.684058`; class1 recall fell to `0.423841`.
- [x] Reviewed all folds, full predictions, transitions, domain direction, effective rank, and the 12-row mean-occlusion sheet. Versus keeper BioCLIP made `93/169` corrections/harms and rescued/broke only `8/62` class1 FN/TP; center evidence mass fell `0.442268 -> 0.162520`, shifting toward edges/context.
- [x] Audited keeper OOF availability: `yolof_oof_folds_train5_20260704` is data materialization only and current keeper train probabilities are in-sample. Therefore the stable BioCLIP-minus-OpenAI FN/FP AUROC `0.828627/0.854911` is not a valid keeper-relative residual/router target.
- [ ] Do not sweep BioCLIP versions/sizes, prompts, readout hyperparameters/folds, residual coefficients, routers/thresholds, or distillation adapters. Nine gates failed and transfer smoke is closed; reopen only with true current-keeper OOF plus an independently recall-safe validation action.
- [x] Retained BioCLIP evidence at `runs\diagnostic_bioclip_domain_feature_full_20260712` (9 payloads, `5908596` bytes, SHA `3ebb6b8a...c4b1c1`, no model/checkpoint/test), deleted only the reproducible 34.215 MB embedding cache under an exact-hash manifest, and passed retention over 574 directories with `blockers=[]`.
- [x] BioCLIP closure passed py-compile, compileall, focused tests `9/9`, full pytest `697/697`, JSON/payload-hash checks, and `git diff --check`; keeper and command SHA `3c718130...57dd` remain unchanged.
- [ ] Keep current-best train/export/video commands unchanged. Continue dependency-coherent worktree cleanup with explicit staging and keep all protected user paths outside this stage.
- [x] Re-read Deep TEN, DEP, and the official `PyTorch-Encoding` implementation at commit `ac748410...a20`, then added a locked no-test final-stem texture-encoding audit with a matched mean/std control and 11 focused tests.
- [x] Corrected the preflight's sparse-mask fallback to select the nine nearest valid bbox-center tokens for both branches. The corrected class-biased prefix verified protocol and reduced runtime `178.46s -> 81.82s`; only the full `9215/2606` source-guarded run was allowed to decide.
- [x] Deep-TEN lost to control in all five folds and collapsed independent validation keeper/control/candidate macro-class1 `0.884675/0.686047 -> 0.794480/0.363636 -> 0.779022/0.316327`; candidate class1 recall fell to `0.205298`.
- [x] Reviewed folds, curves, all predictions, transitions, direction, codeword telemetry, and 12-case signed XAI. Candidate made `82/199` corrections/harms, removed/created 62/1 class1 FP, rescued/broke 0/87 FN/TP, shifted center/border evidence `0.356843/0.030891 -> 0.278909/0.040745`, and had maximum codeword cosine `0.995986`.
- [ ] Do not implement or sweep Deep-TEN codewords/grid/projection/erosion/readout/optimizer/residual scale/folds/epochs, nor fit a router or threshold from its control-relative AUROC. Ten fixed gates failed and core image smoke is closed.
- [x] Retained full Deep-TEN evidence (11 payloads, `5671831` bytes, SHA `24ee3563...6924`, no model/checkpoint/test), deleted two superseded preflights under an exact-hash manifest (20 files, `5866265` bytes; observed `5902336` free-byte gain), and passed retention over 576 directories with `blockers=[]`.
- [x] Deep-TEN closure passed py-compile, compileall, focused tests `11/11`, full pytest `708/708`, artifact/cleanup/retention hashes, staged `git diff --check`, and protected-path review; keeper and command SHA remain unchanged.
- [ ] Keep current-best train/export/video commands unchanged. The next representation candidate must use genuinely local, class1-positive supervision and prove keeper-FN rescue plus recall preservation before any core-model smoke.
- [x] Re-read VICRegL and its official code at commit `803ae4c8...620`, then added a locked frozen-keeper crop-location readiness audit plus nine tests. It uses two overlapping 240px views, exact transformed metadata, post-pruning `patch_indices`, symmetric location/feature matching, and official gamma 20 without touching test or trainer.
- [x] Full source-safe coverage is `9215/2606` rows and `8064/2577` groups. Train/val location cosine `0.966519/0.965089` and retrieval top1 `0.893692/0.895635` show local matching is already near saturation, while FN-minus-TP mismatch is wrongly signed `-0.003113/-0.001721` and FN-vs-FP AUROC only `0.470264/0.411313`.
- [x] Crop-average regressed keeper validation macro/class1 `0.884675/0.686047 -> 0.881340/0.670487`; transitions were 35/38 corrections/harms, class1 FP remove/create 12/18, and FN-rescue/TP-break 5/6.
- [x] Reviewed full row metrics and the 12-case mismatch XAI. Mean mismatch was `0.010515`, with border/center mass `0.215986/0.149665`; residual differences track crop edges, padding, hands, stems, and object boundaries rather than missing class1 interior evidence.
- [ ] Do not implement or sweep VICRegL crop geometry/gamma/thresholds/projector/local weight/optimizer/epochs, nor use crop mismatch for routing, sample weights, or margins. Eleven gates failed and trainer smoke is closed.
- [x] Retained full VICRegL evidence (9 payloads, `4791831` bytes, SHA `f249f51c...f6281`, no model/checkpoint/test), deleted one preflight under exact hashes (8 files, `1776284` bytes; observed `1794048` free-byte gain), and passed retention over 578 directories with `blockers=[]`.
- [x] VICRegL closure passed py-compile, compileall, focused tests `9/9`, full pytest `717/717`, artifact/cleanup/retention hashes, staged `git diff --check`, and protected-path review; keeper and current-best command SHA remain unchanged.
- [ ] Keep current-best train/export/video commands unchanged. Continue scientific worktree cleanup only in explicit batches; the next method must expose new class1-positive semantics rather than more invariance on already stable final tokens.
- [x] Re-read FastViT, NTS-Net, and PMG before allocating another architecture run. FastViT-SA12 is three RepMixer stages plus one attention stage with kernel-3 RepMixer in local timm; reject a blind stock-backbone launch because it does not add a new class1-positive supervision source.
- [x] Added locked no-test original-resolution surface-tile readiness audit and seven focused tests. The protocol compares the exact keeper global crop against five fixed 70% bbox-crop tiles, preserves all bbox/mask/token-prior metadata, and uses one predeclared equal probability fusion without model/threshold fitting.
- [x] Full `9215/2606` audit reproduced global keeper validation `0.884675/0.686046`, while tile mean reached `0.862581/0.629080` and fixed fusion fell to `0.880824/0.666667`; class1 recall fell `0.781457 -> 0.748344`.
- [x] Reviewed complete rows, geometry, transitions, direction histograms, metrics, and the 12-row contact sheet. Fusion made 21/24 corrections/harms, class1 FP remove/create 10/10, and FN-rescue/TP-break 0/5. Only 9.27% of val class1 crops have raw min-side >=384 px, so resize loss is not the main class1 bottleneck.
- [ ] Do not sweep surface-tile scale/positions/count, fusion weight, image size, crop threshold, or fit a tile router/residual from AUROC alone. FN-vs-FP tile delta AUROC `0.844347/0.840000` is a confidence-contraction signal; a fixed matched readout remained only `0.869291/0.640212`, below keeper.
- [x] Retained full tile evidence (8 payloads, `8462921` bytes, SHA `010d368e...60de2`, no model/checkpoint/test) and removed the superseded 12/12 preflight. Cleanup manifest explicitly records the PowerShell 5.1 hash-capture provenance gap instead of inventing hashes.
- [x] Surface-tile closure passed py-compile, compileall, focused tests `7/7`, full pytest `724/724`, stable payload hashes, and retention over 580 directories with `blockers=[]`; keeper and command hashes remain unchanged.
- [ ] Next bounded method: audit Finer-CAM-style target-versus-confuser spatial attribution on keeper class1 FN/FP cases. Require deployable top-class/class1 reference selection, interior localization, positive FN-vs-FP crop direction, and direct recall-safe action before any trainer smoke.
- [ ] Keep current-best full-train/export/video commands unchanged; the surface-tile candidate failed the independent validation gate.
- [x] Re-read the Finer-CAM paper and current official code, then added a locked FP32 no-test class1-vs-three-logit-nearest-reference audit, a batch-context-preserving changed-case reviewer, and 13 focused tests. The target/reference/mask/action are deployable and never selected by ground truth.
- [x] Full `9215/2606`, `8064/2577`-group support shows Finer relative confidence drop improves over Grad-CAM on train `0.005317 > 0.005041` and val `0.006777 > 0.006346`, but contrast-gain FN-vs-FP AUROC transfers only `0.574541/0.616501` and fails the train gate.
- [x] The one fixed top-2 class1 action changes matched FP32 validation macro/class1 `0.882925/0.678261 -> 0.884183/0.682216`, precision `0.603093 -> 0.609375`, recall unchanged `0.774834`; transitions are 8/6 corrections/harms, FP remove/create 6/4, FN-rescue/TP-break 2/2.
- [x] Recomputed all 14 transitions in their original batch-32 windows over 334 context rows; every prediction reproduced and max relative-drop difference is `6.56e-7`. Visual review plus metrics show val interior mass falls `0.358258 -> 0.323801` while border rises `0.181833 -> 0.191826`.
- [ ] Do not sweep Finer-CAM alpha/layer/references/mask/CAM backend/residual/thresholds or fit a post-hoc filter/router. `13/17` gates pass but class1 gain is below `+0.005`, F1 below `0.70`, train direction fails, and recall action is neutral.
- [x] Retained full Finer-CAM evidence (8 payloads, `8770683` bytes, SHA `dfa7a1a5...40db`) plus exact changed-case review v2 (4 payloads, `1032981` bytes, SHA `10f48a65...67b9`), no model/checkpoint/test; AMP-vs-FP32 baseline difference is explicitly reconciled.
- [x] Removed 41 superseded Finer-CAM files under two exact-hash manifests (`3312956` bytes; observed free gain `3383296` bytes), then passed retention across 583 run directories with `blockers=[]`.
- [x] Finer-CAM closure passed py-compile, compileall, focused tests `13/13`, full pytest `737/737`, `git diff --check`, stable artifact/cleanup/retention hashes, and protected-path review.
- [ ] Keep current-best full-train/export/video commands unchanged; Finer-CAM did not pass its independent validation and recall-action gates.
- [x] Re-read reflection-separation and mango lenticel/optical-maturity sources,
  then added a locked surface-blob morphology readiness audit plus an exact
  changed-case reviewer. The achromatic top-5% `V*(1-S)` mask is documented as
  an exclusion proxy, not physical specular separation or a predictive feature.
- [x] Full source-safe support is `9215/2606` rows and `8064/2577` groups. The
  fixed 45D multiscale LoG block has effective rank `15.337912`, but standalone
  class1 F1 remains zero and OOF/val candidate-minus-control class1 gains are
  `-0.005881/+0.003402`; FN-vs-FP AUROC is inverted
  `0.425281/0.430451`.
- [x] Direct keeper/control/candidate validation macro-class1 is
  `0.884073/0.684058 -> 0.846421/0.624585 -> 0.848371/0.627986`.
  Candidate versus keeper makes `52/128` corrections/harms and rescues/breaks
  class1 FN/TP only `1/27`; `image_smoke_permission=false` with 17 failed gates.
- [x] Reproduced 24 prioritized transitions exactly with their original nine
  CUDA batch-64 windows (`576` context rows, descriptor max difference `0.0`).
  Visual review shows dense cross-class responses to spots, lesions, bruises,
  glare, edges, illumination, and hands rather than class1-positive morphology.
- [ ] Do not sweep LoG scales/background sigma/MAD threshold, highlight
  quantile, ROI size/erosion, residual C, folds, weights, or add a morphology
  branch/router. The stable wrong-sign direction is class1 suppression, not a
  recall-safe target.
- [x] Retained full morphology evidence (9 payloads, `9568281` bytes, SHA
  `eaf1b563...f624de`) and exact review (5 payloads, `1851974` bytes, SHA
  `9724432d...a6cad`), deleted 43 superseded files under a two-phase exact-hash
  manifest (`4242949` bytes; observed gain `4300800` bytes), and passed
  retention over 586 directories with `blockers=[]`.
- [x] Surface-morphology closure passed py-compile, compileall, focused tests
  `10/10`, full pytest `747/747`, artifact/cleanup/retention hashes,
  `git diff --check`, and protected-path review.
- [ ] Keep current-best train/export/video commands unchanged. The next method
  must introduce a genuinely different class1-positive supervision source and
  prove direct keeper FN rescue plus TP-recall preservation before GPU smoke.
- [x] Re-read the primary MCR2 paper and official head/loss/train/evaluation
  code, then added a locked 30-epoch frozen-keeper embedding audit plus seven
  tests. Five source-grouped folds compare the official-form MCR2 projection
  with the same PCA30 nearest-subspace readout on raw normalized embeddings.
- [x] Full `9215/2606`, `8064/2577`-group support has zero source overlap, all
  batches cover every class, and every loss decreases. MCR2 improves matched
  OOF control macro/class1 `0.891109/0.651852 -> 0.939210/0.818713` and
  validation `0.852374/0.563574 -> 0.875310/0.640523`.
- [x] Independently recomputed predictions, probabilities, confusion, and
  transitions. Candidate remains below keeper `0.884073/0.684058`, class1
  recall falls `0.781457 -> 0.649007`, and FN-vs-FP direction reverses from OOF
  `0.717030` to validation `0.385807`.
- [ ] Do not sweep MCR2 LR/epsilon/gamma/projection width/PCA rank/batch/
  optimizer/epochs/folds/temperature, and do not add a post-hoc MCR2 head,
  residual, or router. Seven gates fail; reopen only after a new image
  representation changes direct keeper FN/FP support.
- [x] Retained one final MCR2 root (8 payloads, `4863531` bytes, SHA
  `f5a32677...f6ae49`, no model/checkpoint/test). No superseded MCR2 output
  existed to delete; retention passed over 588 directories with `blockers=[]`.
- [x] MCR2 closure passed py-compile, compileall, focused tests `7/7`, full
  pytest `754/754`, independent CSV reconstruction, payload/retention checks,
  and explicit protected-path review.
- [ ] Keep current-best full-train/export/video commands unchanged. Select the
  next genuinely distinct class1-positive image supervision route only after a
  no-repeat literature/code audit and a predeclared keeper-relative gate.
- [x] Verified the original local AIDT ResNet50+ViT-B/16 architecture, exact
  timm weights/preprocessing, and added FP32 frozen feature extraction plus a
  five-fold no-test fusion readiness audit with 14 focused tests.
- [x] Full strict `class_f -> yolo_f` support is `9215/2606` rows and
  `8064/2577` groups with zero missing keys or source leakage. Raw fusion loses
  to ViT OOF `0.879585/0.600000 -> 0.878306/0.587121`, despite validation
  improving `0.894439/0.664407 -> 0.901745/0.691030`.
- [x] Direct keeper comparison rejects the validation-only gain: fusion recall
  is `0.688742` versus `0.781457`, FN-rescue/TP-break is `13/27`, all five OOF
  folds lose class1, direction AUROC is `0.430115`, and eight teacher gates
  fail.
- [x] Tested exactly one block-L2 correction after measuring the 6.4x branch
  norm mismatch. It worsened OOF fusion to `0.874501/0.568982`, stayed below
  class1 `0.70` on validation, and failed nine gates; close normalization and
  branch-scale sweeps.
- [x] Reviewed 32 balanced FP-remove/create and FN-rescue/TP-break cases with
  `class_f` crop plus `yolo_f` source/bbox. Rescued and broken class1 rows share
  the same pale-green/yellow/spot cues; no stable context rule appears.
- [ ] Do not use raw/block-L2 AIDT caches for KD, sample weighting, routing,
  thresholding, or command promotion. They are complementary validation
  diagnostics but not a fold-safe class1-positive teacher.
- [x] Removed eight superseded AIDT prefixes/reviews under exact hashes: 95
  files, `26639806` bytes; retained the two full audits, final visual review,
  and full feature/remap caches needed by the next declared stage.
- [ ] Next fixed method: extract raw pretrained AIDT features from a wider
  `yolo_f` bbox-context crop and compare object-only, context-only, and paired
  object+context fusion with source-grouped OOF, direct keeper recall safety,
  no test, and no normalization/fusion-weight sweep.
- [x] Raw-AIDT stage closure passed compileall, focused tests `16/16`, full
  pytest `769/769`, exact prediction/payload/cleanup checks, and retention over
  594 directories with `blockers=[]`; keeper and command hashes are unchanged.
- [x] Extended exact AIDT extraction to direct YOLO object rows with fixed bbox
  context margin `0.50`; full context cache matched all `9215/2606` object
  sample indices, labels, source stems, object indices, classes, and encoder
  hashes with zero train/validation source overlap.
- [x] Added strict diagnostic pairing `[class_f object 2816D | yolo_f context
  2816D]` and explicit `object/context/paired` audit semantics. Prefix schema,
  geometry, provenance, and end-to-end output passed before the sole full run.
- [x] Full OOF object/context/paired macro-class1 is
  `0.878306/0.587121`, `0.875488/0.578512`, and
  `0.878969/0.588566`; paired wins class1 in only 2/5 folds and misses both
  required gains.
- [x] Validation object/context/paired is `0.901745/0.691030`,
  `0.890328/0.645614`, and `0.899883/0.684746`; paired recall `0.668874` is
  below object and keeper, with FN-rescue/TP-break `9/12` versus object and
  `12/29` versus keeper.
- [x] Reviewed 32 balanced context-induced class1 changes. Hands, floor/table,
  foliage, shadows, neighboring fruit, object scale, and position occur in both
  fixes and harms; wide context is real but supplies scene/acquisition cues
  rather than stable maturity evidence.
- [ ] Do not sweep context margin/layout, branch normalization/scale, fusion
  weights, head settings, or create context routing/KD/weights/thresholds.
  Eleven fold-safe teacher gates fail and the current-best command remains
  unchanged.
- [x] Preserved full audit provenance and final review, then deleted eight
  reproducible cache/preflight roots under exact hashes: 55 files,
  `495809819` bytes; observed free gain `495398912` bytes.
- [ ] Select the next method only if it introduces a new class1-positive
  supervision/representation mechanism rather than another crop/context,
  frozen-readout, stock-hybrid, texture, uncertainty, or post-hoc variant.
- [x] Object-context closure passed compileall, focused tests `23/23`, full
  pytest `777/777`, bit-exact object-control reproduction, exact
  payload/cleanup verification, and retention over 595 directories with no
  blockers; keeper/current-best hashes remain unchanged.
- [x] Read the DINOv3 paper/repository and lock one new frozen representation
  comparison: timm DINOv2-S/14 versus DINOv3-S/16, released 224/256 recipes,
  equal 16x16 patch grids, equal 384/2688D descriptors, identical PCA/readout,
  five source folds, and OOF-only CLS-versus-dense selection.
- [x] Add a version-neutral DINO descriptor helper, exact DINOv2/DINOv3 weight
  hashes, strict retained-audit provenance, full prediction exports, a
  fail-closed gate, and a validation-only changed-case reviewer with 11 new
  tests.
- [x] Reject AMP for this comparison. A 16-versus-8+8 probe changed DINOv2
  descriptor values by up to `0.129207` and DINOv3 by `0.040240`; FP32 was
  bit-exact for both. Re-extract both controls in FP32 rather than comparing
  mismatched numeric modes.
- [x] Full `9215/2606`, `8064/2577`-group FP32 audit has zero overlap. OOF
  selected dense descriptors for both models; DINOv2/DINOv3 macro-class1 is
  `0.880034/0.602432 -> 0.878180/0.598714` OOF and
  `0.884337/0.641115 -> 0.883516/0.643599` validation.
- [x] DINOv3 wins class1 in only `2/5` folds and remains below keeper
  `0.884073/0.684058`; recall is `0.615894` versus `0.781457`. Versus keeper it
  rescues/breaks class1 FN/TP `9/34` while removing/creating FP `48/17`.
- [x] Review the eight largest changes in each of four class1 directions with
  matched FP32 DINOv2/DINOv3 patch-energy maps. DINOv3 is smoother and less
  interior-detailed; rescued/broken positives and created class0 FP share the
  same pale-green/yellow/mottled appearances, so no safe action rule appears.
- [ ] Do not sweep DINOv3 size/input/descriptor/PCA/readout/folds/thresholds or
  build a validation router, KD target, sample weight, or ensemble from this
  audit. Ten gates fail; reopen only with genuinely fold-safe keeper OOF
  evidence that protects the current `9/34` rescue/break direction.
- [x] Preserve compact DINOv3 evidence only: full audit 11 payloads/`8168884`
  bytes/SHA `2ef946a9...fe2b`, review 5 payloads/`471146` bytes/SHA
  `a3630a1e...6a57`; no descriptors/checkpoint/test/model binary. No
  superseded DINOv3 run existed to delete.
- [x] DINOv3 closure passed compileall, focused tests `18/18`, full pytest
  `789/789`, independent reconstruction of all eight CSV variants, payload
  hashes, and retention over 598 directories with `blockers=[]`.
- [ ] Keep current-best commands unchanged. Triage the next method only after
  excluding already closed SSL/MAE/DINO, ordinal, stock-hybrid, texture,
  context, uncertainty, frozen-readout, and post-hoc-router families; require
  a new class1-positive mechanism before another GPU smoke.
- [x] Read the primary MambaVision/CVPR sources and lock exactly one
  parameter-matched scratch-only selective-state-space gate: width `48`, depths
  `1/2/4/2`, late Mamba/attention layouts `2+2` and `1+1`, input 256, 6.172M
  parameters, with no architecture or optimization sweep.
- [x] Add checkpoint-safe `mambavision_nano`, exact structure/parameter guards,
  fixed ImageNet normalization, CUDA-only preflight, compact-launcher wiring,
  and regressions. BF16 batch 64 reached `137.20 img/s` at only
  `1150.68/1548.00 MiB` allocated/reserved; no package version changed.
- [x] Complete the 20-batch smoke, independent full-val reload, boundary audit,
  and 12-case robustness XAI. Pipeline passed with no test; Grad-CAM was
  foreground-clean and object desaturation dominated background perturbation,
  so the predeclared five-epoch run was permitted.
- [x] Reject MambaVision-Nano at epoch 5. Independent macro/class1 is only
  `0.780478/0.477327`, P/R `0.373134/0.662252`, below gates
  `0.83032/0.55340` and keeper `0.882925/0.678261`. It creates `168` class1 FP
  for `100` TP and has `51` FN; do not continue to epoch 10/15/30 or open test.
- [x] Make `trace_architecture` signature-aware for generic classifiers and
  capture five-class patch/stage traces. MambaVision reduces spatial features
  `64x64 -> 32x32 -> 16x16 -> 8x8`; generic RMS maps are structural only and
  must not be described as causal attention.
- [x] Audit two balanced cases for every `1->0/2/4` and `0/2/4->1` direction.
  Grad-CAM foreground/background is `0.93987/0.06013`; blur/gray drops are near
  zero versus object desaturation `0.13313`. Wide background is again not the
  bottleneck; broad object color/surface evidence remains class-unsafe.
- [x] Add and test a fail-closed rejected-run compactor. Correct the self-review
  bug that initially omitted nested source `summary.json` files, refresh all
  hashes, then retain `314` payloads/`48,325,685` bytes with manifest
  `2660ab55...446a1`; delete nine source roots and reclaim `346,447,872` bytes.
- [x] Close the MambaVision batch with compileall, focused tests `21/21`, full
  pytest `799/799`, corrected evidence/cleanup hashes, and retention over `601`
  run directories with `blockers=[]`. Keeper and command hashes are unchanged;
  `pip check` only reports pre-existing MambaVision/OpenCV pin conflicts.
- [ ] Do not sweep MambaVision size/width/depth/window/mixer ratio/input/drop
  path/LR/loss/sampler/teacher/augmentation or transplant the same mixer as an
  ungrounded adapter. Reopen SSM only after new class1-positive supervision
  independently preserves keeper TP and suppresses `0/2/4->1` FP.
- [ ] Keep current-best train/export/video commands unchanged. Continue worktree
  cleanup through explicit-path staging, cached-diff review, tests, retention,
  commit/push, then select a non-repeated supervision mechanism rather than
  another architecture catalogue substitution.
- [x] Re-read FAT, AdvProp, and LBGAT; lock one no-test Friendly Foreground
  Adversarial protocol before implementation with RGB `2/255`, step `1/255`,
  two early-stopped steps, eroded bbox mask, and class1 protect/suppress pools.
- [x] Complete full `9215/2606` FP32 readiness with zero source overlap, exact
  keeper reproduction, bounded/masked attacks, five source folds, and all
  `21/21` smoke-permission checks passing.
- [x] Add default-off shared attack/trainer/config/v8-launcher plumbing,
  telemetry, exact nested module-mode restoration, fail-closed primitive and
  1-based epoch validation, plus `15/15` focused tests.
- [x] Run the sole predeclared 40-batch/one-epoch smoke and independent full-val
  reload without test. Reject `0.882354/0.676471`, class1 P/R
  `0.608466/0.761589`, TP/FP `115/74`; macro, class1, and FP gates fail.
- [x] Compare every changed decision to keeper: `12` changes, `6/6`
  corrections/harms, class1 FP remove/create `3/0`, FN-rescue/TP-break `0/2`.
  No probe, test, or command promotion is allowed.
- [x] Complete boundary plus balanced six-direction XAI. Object desaturation
  `0.13788` dominates background blur/gray `0.00641/0.00602`, while maps and
  crops remain broad-color/silhouette/border driven.
- [x] Compact six rejected preflight/smoke/eval/boundary/XAI roots into 346
  verified payloads (SHA `c7c7418e...56a6c`), reclaim `396,460,032` bytes,
  retain full readiness separately, and pass retention over 603 directories.
- [x] Close the stage with compileall, PowerShell parse, full pytest `814/814`,
  unchanged keeper SHA `1f49d577...482677`, and unchanged command SHA
  `3c718130...57dd`.
- [x] Re-run the current-best one-command wrapper directly under Windows
  PowerShell with `-PreflightOnly`; confirm epochs 30, effective batch 64,
  train-side test skipped, FriendlyAdv weight `0`, and no run directory.
- [ ] Do not sweep FriendlyAdv epsilon/step/count/random-start/erosion/cohort/
  weight/start/LR/run length on this keeper. Reopen adversarial supervision
  only with a genuinely different independent target that proves direct FN
  rescue and TP preservation before training.
- [ ] Keep current-best full-train/export/video commands unchanged; this smoke
  did not beat the locked independent validation gate.
- [x] Re-read FACT/FDA primary papers and official code, verify the existing
  TRKH FFT is token-embedding filtering rather than RGB amplitude mixing, and
  lock a source-distinct same-class full-spectrum `lambda=0.5` protocol before
  code or GPU allocation.
- [x] Add a no-fit/no-test Fourier readiness tool plus six tests. Two capped
  preflights passed pairing, metadata, exact keeper forward, numerical identity,
  contact-sheet, manifest, and fail-closed gate checks on Windows/CUDA.
- [x] Complete full `9215/2606`, `8064/2577`-group evidence with zero source
  overlap and exact keeper `0.882925/0.678261`; all `11821` same-class and
  cross-control pairs have the locked label and a different source stem.
- [x] Reject same-class Fourier views at `20/26` gates: validation macro/class1
  collapses to `0.688662/0.358362`, clean/class1-TP retention is
  `0.768685/0.760684`, FN-rescue/TP-break is `16/28`, FP-remove/create
  `27/197`, and corrections/harms `65/554`.
- [x] Inspect the balanced 12-row contact sheet and independently rebuild all
  metrics/transitions/AUROC/hashes. Full-spectrum peer amplitude creates global
  color casts, halos, duplicated contours, crop-edge ringing, and background
  leakage; delta-p1 AUROC `0.802176` is unsafe broad movement, not a router.
- [ ] Do not sweep same-class full-spectrum lambda/randomness/peer policy/
  consistency, and do not use its delta as routing, weights, margins, or
  thresholds. A localized/low-frequency revisit needs a new anti-ringing
  mechanism and predeclared label-retention proof, not a nearby parameter run.
- [x] Keep current-best commands unchanged. Delete the two superseded
  preflights only after an 18-file exact-hash manifest, preserved full evidence/
  keeper/command verification, and manifest round-trip; reclaim an observed
  `6,356,992` bytes.
- [x] Close the Fourier stage with compileall, focused tests `6/6`, full pytest
  `820/820`, independent CSV/payload reconstruction, and retention over `605`
  directories plus `28` compaction manifests with `blockers=[]`.
- [x] Re-read the NeurIPS class-dependent augmentation study and lock one
  matched no-test candidate: reduce only class-1 spatial/local augmentation to
  `0.5`, keep classes `0/2/3/4`, photometric color handling, validation, model,
  sampling, losses, and optimizer unchanged.
- [x] Add default-off class-conditional scale plumbing and regressions for
  exact scale count/range, class routing, validation isolation, V8 forwarding,
  and bit-equivalent legacy behavior at scale 1.
- [x] Complete matched one-epoch/60-batch control and candidate continuations
  from the keeper, independent full-val FP32 reload, architecture trace, and no
  test. Control/candidate macro-class1 is
  `0.882725/0.680233 -> 0.881795/0.676385`.
- [x] Reject candidate before probe: versus control it changes one row and
  breaks one class-1 TP; versus keeper it has `5/7/1`
  corrections/harms/neutral, FP remove/create `2/1`, and FN rescue/TP break
  `1/2`. Five locked gates fail.
- [x] Complete candidate boundary scan and matched 13-case control/candidate
  all-method FP32 XAI. Maps are nearly unchanged; object desaturation remains
  dominant, while sample 1113 shifts candidate Grad-CAM toward hands/image
  edge and loses a true class-1 decision.
- [x] Compact the rejected control/candidate smoke, reload, transition,
  boundary, and XAI roots under exact hashes; verify keeper/current command,
  and retention. Evidence has 638 payloads/SHA `6de15fce...ab0d`; four
  checkpoint files were excluded, ten roots were deleted, and observed free
  gain is `422,850,560` bytes.
- [x] Re-run compileall, focused tests `22/22`, full pytest `825/825`, all four
  PowerShell parse checks, and current-best `-PreflightOnly`. It resolves 30
  epochs/effective batch 64/empty class scale/skip-test/exact keeper and creates
  no run directory; retention passes over 607 directories and 29 object
  manifests with no blockers.
- [x] Inspect the explicit staged diff and stage only the three
  class-conditional closure docs; protected user paths remain untracked.
- [ ] Do not sweep class-1 scale, transform subset, photometric scaling, other
  classes, seed, LR, or run length. Reopen class-dependent augmentation only
  with an independently measurable harm signal and direct keeper-TP
  protection.
- [ ] Keep current-best full-train/export/video commands unchanged; this
  matched candidate did not pass the locked validation gate.
- [x] Read the PDiscoFormer paper and official MIT implementation at commit
  `1a872e2b...f33`; distinguish semantic background/presence/equivariance/TV/
  entropy part discovery from the rejected four-part TRKH pairwise head.
- [x] Write a locked no-test semantic-part readiness protocol before code:
  keeper hash, K=2+background, exact CW90 view, matched CE control, official-
  form loss weights, five source folds, 12 epochs, direct class1/direction/
  collapse gates, and no hyperparameter sweep.
- [x] Add the standalone cache/head/audit tool and nine regressions. Fix the
  real keeper-head fallback plus `Subset.sample_paths` provenance bug before
  accepting the 128/128 preflight contract.
- [x] Complete full `9215/2606`, `8064/2577`-group evidence with zero overlap,
  exact keeper `0.882925/0.678261`, `16x16` grid, `167x256` post-pruning tokens,
  zero rotation-grid mismatch, five head-OOF folds, and no test/checkpoint.
- [x] Reject candidate: OOF control/candidate is
  `0.882467/0.619691 -> 0.865934/0.574402`; validation is
  `0.846990/0.560606 -> 0.812924/0.477733`; class1 wins only `1/5` folds.
- [x] Audit direct safety: versus keeper validation, candidate has
  `66/172/27` corrections/harms/neutral, FP remove/create `52/12`, and FN
  rescue/TP break `2/60`; do not convert AUROC `0.62185/0.57105` into routing.
- [x] Inspect the balanced 14-row contact sheet. Candidate background dominates
  inside the fruit (`0.79041`), foreground masses collapse to
  `0.05334/0.12963`, and parts map broad color plus hands/lower edges rather
  than stable class1-positive evidence.
- [x] Independently rebuild all CSV metrics/folds/transitions/AUROC and replay
  cache/artifact hashes. Compact about `2.085GB` of reproducible caches, keep
  13-file/`7,284,956`-byte full evidence, and pass retention over 610 runs.
- [x] Close with compileall, focused `16/16`, full pytest `834/834`, four
  PowerShell parse checks, exact keeper/current-command hashes, and current-
  best `-PreflightOnly` without a generated run.
- [ ] Do not sweep PDisco semantic-part K/loss weights/transform/temperature/
  layer/pruning/optimizer/LR/epochs/dropout/background prior/residual/router on
  this keeper. Reopen only after a new representation independently protects
  class1 TP and supplies stable foreground-positive support.
- [ ] Keep current-best train/export/video commands unchanged. Select the next
  method only if it introduces a new direct class1-positive representation or
  supervision source, not another part-map regularizer, prettier XAI target,
  frozen readout, context crop, or post-hoc suppressor.
- [x] Independently audit the user-completed 23-epoch random-init run. Reject
  `best.pt` at macro/class1 `0.874172/0.654639` and epoch-23 EMA at
  `0.877334/0.657895` versus keeper `0.882925/0.678261`.
- [x] Separate raw `last.pt/model_state` from validation `ema_model_state` with
  a strict 185-tensor extractor. Correct future run-summary semantics so
  selector epoch/macro and maximum-observed macro/epoch are explicit.
- [x] Lock and execute five source-grouped folds before test. Preserve the
  stable keeper/candidate `0.60/0.40`, class1-margin `0.034` precision rule:
  OOF `0.886214/0.683544`, locked val `0.890298/0.696774`, all 14 gates pass.
- [x] Open final test once with the frozen summary hash. Record macro/class1
  `0.887869/0.675676`, class1 P/R `0.666667/0.684932`; no further test-fitted
  ensemble, threshold, calibration, or member-weight variant is permitted.
- [x] Compare full clean/occlusion/dim/bright/low-contrast robustness. Keep the
  scratch member's real illumination-recall complement as a supervision clue,
  not as permission to promote its 110 validation class1 false positives.
- [x] Replace pruned-QKV pseudo-attention with validated full-grid native last-
  block attention. Rerun matched candidate/keeper smoke and manually inspect
  five hashed correction/harm/FP/FN cases with zero fallback.
- [x] Keep the current single-checkpoint full-train/export/video recipe on the
  deployable keeper. Publish the precision ensemble separately only after its
  packaged PyTorch and ONNX CPU paths pass full parity; do not require a failed
  TensorRT backend to relabel the two passing backends. This boundary is now
  implemented and documented.
- [ ] Do not repeat seed/epoch/SWA/soup/member-weight/margin/threshold sweeps.
  Next GPU work must use train-only supervision with separate late member
  paths, explicit keeper-TP protection, and conservative `0/2/4->1` control.
- [x] Remove only superseded blend/native-XAI-v1/final-state binaries through a
  reread manifest with exact hashes. Preserve `best.pt` and formal evidence;
  reclaim `147,976,192` observed bytes and pass retention over 613 directories.
- [x] Close this stage with focused `28/28`, full pytest `850/850`, four
  PowerShell parse checks, one-command full-pipeline preflight, TensorRT export
  preflight, and PyTorch video preflight. Protected user paths remain unstaged.
- [x] Harden probability/audit provenance after self-review: reject nonfinite,
  negative, and zero-sum vectors; require exact locked-summary SHA for test;
  validate each row split from `image_path`; refuse nonempty outputs; validate
  paired XAI cohort fields and hash every input `case.json`.
- [x] Replay all six readiness artifacts bit-identically at summary SHA
  `6c6397d...a8c8`; replay paired XAI `30/30` plus native `5/5` with known
  sources, zero missing tiles, and zero native fallback.
- [x] Re-run compile, focused tests `33/33`, and full pytest `859/859` after
  hardening; pass full-pipeline/export/video preflights and retention over 614
  run directories. Keep the deployable keeper and commands unchanged.
- [ ] Implement the next distinct train-only late-member representation only
  after this closure is committed. It must preserve keeper class-1 TP while
  learning the scratch member's illumination complement, and must pass direct
  `0/2/4->1` precision gates before any full 30-epoch run.
- [x] Implement and audit the keeper-initialized fork-after-6 late member,
  direct-splice control, frozen-primary fused-CE control, sparse directional
  teacher transfer, and strict full-validation transition accounting.
- [x] Reject sparse directional transfer: only about `21/1280` smoke rows were
  selected; independent macro/class1 is `0.880357/0.668622`, with
  FN-rescue/TP-break `1/4` and `10/13` corrections/harms.
- [x] Implement dense focal distillation of the frozen precision rule with
  model-owned fusion parameters and exact unit/runtime checks. Reject
  temperature-1 KL at independent `0.879181/0.664723`, class-1 precision
  `0.593750`, and `9/14` corrections/harms.
- [x] Complete native late-block-7 all-method XAI on every important class-1
  transition. Object desaturation remains causal; wide background does not.
- [x] Implement the single predeclared Positive-Congruent FD-LM distance
  (`0.5*L2` on normalized rule/student logits), run one finite-gradient
  preflight, then the matched 40-batch/full-val/no-test gate.
- [x] If and only if FD-LM passes precision `0.65`, recall `0.71`, class-1 F1
  `0.69`, macro `0.887`, FP `60`, TP-break `10`, and correction/harm gates,
  run the 120-batch/two-epoch probe. Otherwise close late-member dual-rule
  compression without a weight/temperature/seed/run-length sweep. FD-LM was
  rejected at `0.879775/0.666667`, class-1 P/R `0.596859/0.754967`, with
  `9/13` corrections/harms and FN-rescue/TP-break `1/4`; no probe was run.
- [x] Complete FD-LM transition/XAI closure on all 11 important class-1
  changes. Native late-block-7 attention had zero fallback; all cases were
  near-ties and precision harms still included edge/background peaks.
- [x] Lock a no-test feature-basis compatibility protocol before another
  architecture smoke. Compare keeper and scratch internal representations by
  direct matching on train-held-out source groups, account for pruned-token
  index intersections, and reject any adapter that only improves task loss
  through out-of-distribution stitched activations.
- [x] Complete full `9215`-row five-source-fold feature-basis audit. All forks
  pass OOF compatibility; fork 5 wins by probability MAE `0.009962` with token/
  prefix/patch/CNN R2 `0.8601/0.8760/0.8584/0.9947` and agreement `0.9572`.
- [x] Reject fork-5 linear stitching on its single validation opening. It raises
  macro/class1 F1 `0.882925/0.678261 -> 0.890221/0.700337`, raises class1
  precision `0.603093 -> 0.712329`, and cuts FP `77 -> 42`, but recall
  `0.688742` and TP-break `15` fail the locked `0.70/10` gates. No adapter
  checkpoint, probe, or test is permitted.
- [x] Compact 14 rejected late-member/stitch-preflight roots through verified
  evidence SHA `4da79083...984b`; preserve 1302 payloads, exclude 17 binaries,
  and reclaim `1,382,727,680` observed bytes. Keeper, scratch, full stitching
  audit, and current command hashes remain unchanged.
- [x] Close the stage with compileall, focused tests `34/34`, full pytest
  `889/889`, four PowerShell parse checks, full-pipeline/export/video
  preflights, and retention over `618` run directories with `blockers=[]`.
- [ ] Do not sweep feature-stitch fork/ridge/map/nonlinearity/token roles,
  fusion weight, margin, seed, or split. Keeper class1 TP-break and FP-removal
  confidence overlap too strongly for a nearby post-hoc suppressor.
- [ ] Keep current-best full-train/export/video commands unchanged until one
  single-checkpoint candidate passes the locked validation gate and probe.
  The next experiment must introduce direct train-only class1-positive evidence
  with explicit illumination robustness and TP preservation, not another
  output-rule compression or linear basis adapter.

## Cap nhat 2026-07-14 - Precision Deployment and Next Representation Gate

- [x] Build and strictly reload the self-contained precision package; verify
  source/protocol/package hashes and exact `2606`-row PyTorch validation replay.
- [x] Export the fixed-batch, three-input, six-output ONNX graph and pass full
  ONNX Runtime CPU parity at max probability error `2.980232e-7` with zero
  decision mismatch.
- [x] Reject ORT CUDA, TensorRT FP16, and TensorRT FP32 noTF32/O0 under the
  unchanged backend gates. Do not promote exact current-val argmax when the
  probability error is `0.008846 > 0.005`.
- [x] Repair actual TensorRT schema validation, add strict targeted ONNX probes,
  and make precision-ensemble TensorRT export fail closed by default.
- [x] Make package XAI explicitly member-specific and inference-only. Verify
  keeper/candidate provenance, native attention, all-method maps, robustness,
  package/member hashes, and no optimizer step.
- [x] Add and preflight a one-line-safe PowerShell pipeline for hash-locked
  package build, full PyTorch validation, ONNX CPU validation, and 12-case XAI
  per member. Keep the already frozen one-time test closed during replay.
- [x] Compact only superseded failed package/backend probe roots after copying
  complete summaries/row mismatches and writing a reread SHA manifest. Preserve
  the package, certified ONNX, full PyTorch/ONNX summaries, failed-backend full
  summaries, and every frozen validation/test/XAI artifact.
- [x] Run compileall, focused tests `162/162`, full pytest `919/919`, all
  full-pipeline/export/video/precision wrapper preflights, and retention over
  `637` run directories with `blockers=[]`. Keep protected user files outside
  selective staging; complete cached-diff review, commit, and push next.
- [x] Implement a no-test Counterfactual Illumination Disagreement Transfer
  readiness auditor. Use only `yolo_f/train`, exact source-group folds, fixed
  mild clean/dim/bright/low-contrast transforms, and the locked keeper/scratch
  hashes; write row-level correction/harm cohorts and transform provenance.
- [x] Before any trainer smoke, require the readiness audit to show a denser
  class1 rescue signal than the closed sparse clean disagreement (`~21/1280`),
  zero train/val/test source leakage, stable signed directions across folds,
  and an explicit keeper class1-TP protection cohort.
- [x] Close CIDT before training because the full audit failed its locked
  transform-label-retention gate. The signal was dense (`379` rescue events,
  `285` unique rows, all five folds favorable), but keeper class1-TP retention
  under dim/low-contrast was only `0.5133/0.5852 < 0.65`. Summary SHA is
  `d4891edf...d7ad`; do not sweep transform factors or reuse these views as
  distillation targets.
- [x] If readiness passes, add one default-off late-member directional loss:
  rescue true class1 only when the counterfactual scratch evidence is correct,
  preserve keeper class1 TP, and suppress candidate-created `0/2/4->1` FP.
  Run one finite-gradient preflight and one matched 40-batch/full-val/no-test
  smoke; do not sweep transforms, thresholds, weights, seed, LR, or run length.
  This conditional action is canceled for the locked CIDT protocol because its
  readiness gate failed.
- [x] Research CVPR 2026 CAR and transcribe equations 7-9 from the accepted PDF;
  the advertised official repository returned 404. Add tested paper CAR and a
  frequency-symmetric BiCAR primitive that weights both true-class columns and
  predicted-class rows of the EMA confusion matrix.
- [x] Lock the BiCAR train-only protocol before formal execution: exact CIDT
  summary hash, clean train rows only, strict sampler/seed, paper
  `alpha/beta/r0/gamma=0.5/0.5/0.2/0.1`, occurrence/fold gradient audits, and
  no validation/test predictions.
- [x] Run the formal CAR/BiCAR gradient audit. Only if every declared direction,
  focus-ratio, gradient-scale, and five-fold gate passes, implement trainer
  wiring and run one matched control/BiCAR 40-batch/full-val/no-test smoke.
  V1 correctly failed the strict sampler exposure gate; fix global remainder
  rotation without relaxing the gate. V2 then passed every gate with summary
  SHA `7998a85c...02c8`, FP/FN gains over CAR `3.7235x/1.2767x`, and weighted
  gradient ratio `0.2330x` CE.
- [x] Wire BiCAR into the trainer as default-off stateful EMA loss with locked
  paper values. Pass finite-gradient/config/checkpoint tests, then run exactly
  one matched control/BiCAR `40`-batch, one-epoch, full-validation, no-test
  smoke from scratch SHA `f8bd6309...1a549`; do not sweep any setting.
- [x] Fix the pre-train `args.use_sam` integration failure without changing the
  scientific protocol; add SAM on/off regression and preserve native launcher
  errors when a transcript omits Python stderr.
- [x] Reject BiCAR after independent full-validation reload. It only improves
  class1 precision `0.535865 -> 0.538136`, keeps recall `0.841060`, and changes
  two rows with corrections/harms `1/1`; it misses both the relative precision
  gate and absolute `0.880/0.680/0.60` probe gates.
- [x] Complete paired all-method XAI/robustness on both changed rows plus
  standard forensics/confusion/boundary audits. Candidate Grad-CAM foreground
  mass falls `0.9700 -> 0.4786` and border mass rises `0.1310 -> 0.6069`.
- [ ] Do not sweep CAR/BiCAR weight, margin, EMA, smoothing, sampler, seed, LR,
  checkpoint, or run length. Do not run its 120-batch probe or test.
- [x] Compact the five rejected BiCAR smoke/startup roots into `45` retained
  files and `47` verified payloads. Preserve payload manifest SHA
  `4cba5a42...156b19`, the live changed-case/XAI audit, keeper, scratch, and
  current command hashes; remove `574,625,725` bytes of reproducible payloads.
- [x] Close the combined stage with compileall, `162/162` focused tests,
  `919/919` full pytest, five-wrapper parse, four operational preflights, and
  retention over `637` run directories with `blockers=[]`.
- [ ] Keep current-best full-train/export/video commands unchanged. Select the
  next distinct method only after a train-only readiness audit proves a denser
  decision-level class1 precision effect with explicit TP preservation; a
  favorable infinitesimal gradient alone is insufficient.

## Cap nhat 2026-07-14 - MORE Parameter-Space Rebalancing

- [x] Research NeurIPS 2025 MORE and cross-check it against closed local
  families. Confirm that the keeper has exactly five mergeable `Conv2d`
  modules (`431,200` parameters), so this is a bounded representation change
  rather than another output head or loss-only retry.
- [x] Lock `docs/TRKH_5CLASS_MORE_READINESS_PROTOCOL_20260714.md` before code:
  all five convolutions, rank ratio `0.1`, seed `42`, paper `A'=2`/`A=10`,
  matched `A=0` control, exact CIDT five source folds, no validation/test, and
  no hyperparameter sweep.
- [x] Implement mergeable MORE convolution primitives and tests for zero-tail
  keeper equivalence, general-only path, finite tail gradients, exact merge,
  state-dict compatibility, and deterministic initialization.
- [x] Lock SGD LR `3e-4` from the paper after a metric-free 64-row train-only
  scale check (`2` steps: mean/max logit delta `0.00110/0.00386`). Keep
  deterministic no-TF32 merge gate `<=1e-5`; isolated keeper replay reaches
  approximately `1e-7`.
- [x] Run the locked fold-0 functional preflight at
  `runs/audit_more_functional_preflight_fold0_20b_20260714`. All structural,
  algebra, freeze, gradient, and merge gates pass, but MORE changes `0/1843`
  hard decisions versus control and has class1/nonfocus tail-logit L2 ratio
  only `0.9631x`; Stage A therefore fails exactly as predeclared.
- [x] Close full `9215`-row five-fold train-only OOF, trainer integration,
  validation, probe, test, and parameter sweeps for MORE. The candidate raises
  class1 probability on `1431/1843` rows with mean only `+0.000244`, strongest
  on true classes 2/3 rather than class1, so this is neither material nor a
  precision-selective tail signal.
- [x] Keep the current-best commands unchanged. MORE cannot reach promotion
  because Stage A failed, so no OOF, validation smoke, probe, or XAI run is
  authorized for this method.
- [ ] Do not revisit MORE by changing rank/layer/LR/amplitude/seed/budget or by
  unfreezing the base on this keeper. Select a genuinely different train-only
  representation signal with a predeclared decision-density and class1-TP/
  nonfocus-FP gate before allocating another image-model smoke.
- [x] Close MORE engineering with compileall, focused `12/12`, full pytest
  `931/931`, and retention over `639` run directories with `blockers=[]`;
  keeper/scratch/current-command hashes are unchanged.

## Cap nhat 2026-07-14 - Linear Differential Visual-Contrast Attention

- [x] Review the NeurIPS 2025 LinearDiff paper and official implementation at
  commit `3fb5ee1...e3d1`; verify source SHA `df6b8782...c8ab` and distinguish
  within-image inference VCA from the closed train-only natural-distractor PWCA.
- [x] Lock `docs/TRKH_5CLASS_VCA_READINESS_PROTOCOL_20260714.md` before code:
  all eight blocks, full `16x16` patch grid, `64` contrast tokens, no pruning
  in either matched variant, no new loss, seed 42, and no hyperparameter sweep.
- [x] Implement prefix-aware VCA, derived full-grid attention provenance,
  config/CLI/V8 wiring, conflict checks, checkpoint round trip, and focused
  shape/gradient/XAI tests without changing default behavior.
- [x] Pass the real-batch Stage-A gradient/resource preflight: all gates passed,
  `285184` added parameters, batch-32 AMP peak `2.979 GiB`, all component
  gradients live, and all eight `263x263` XAI maps normalized.
- [x] Run exactly one source-hash-locked matched `120b x 2e`, full-val, no-test
  control/candidate pair with `scripts/run_trkh_vca_matched_smoke.ps1`.
- [x] Audit the matched pair with independent reload, calibration,
  confusion/transitions, robustness, architecture trace, native attention,
  rollout, Grad-CAM, and changed-case XAI; advance only if every
  macro/class1 precision/F1/recall, transition, nonfocus, runtime, VRAM, and
  XAI gate passes. VCA failed eight material gates: macro/class1 F1 fell,
  class1 precision/recall fell, focus FP rose `141 -> 158`, TP breaks were
  `14`, harms exceeded corrections, and new `3->2` harms were `10`.
- [x] Harden grad-rollout provenance: candidate VCA had no usable gradient in
  all `16` cases, so remove `grad_rollout_*` deltas from valid attribution and
  retain native attention, ordinary rollout, Grad-CAM, and perturbation only.
- [x] Close VCA without a five-epoch probe, full train, test, or hyperparameter
  sweep. Compact rejected smoke evidence, repair mandatory binary exclusions,
  remove superseded XAI after replacement checks, and pass retention over
  `643` run directories; compileall, full pytest `956/956`, five wrapper parses,
  and current-best/VCA preflights also pass.
- [x] Keep current-best full-train/export/video commands unchanged because no
  single-checkpoint VCA candidate passed the independent validation gate.
- [ ] Research and lock the next distinct precision-selective representation
  method before implementation. It must create direct class1-positive surface
  evidence, preserve keeper TP, suppress `0/2/4->1`, and pass a train-only
  readiness gate before one matched no-test smoke.
- [x] Research and lock the MogaNet-XT stage-1-to-3 tokenizer at official
  commit `c83e328b...c0a1` and source SHA `1ac13dfb...79797`. Distinguish it
  from the rejected two-stage MBConv transplant and keep all eight TRKH
  Transformer blocks.
- [x] Implement default-off `moganet_xt_tokenizer` exactly as locked, then run
  the train-only shape/gradient/component/resource audit. Do not construct a
  validation/test loader during Stage A. All functional gates passed in v2;
  summary SHA is `98c27d5e...f406`.
- [x] Cancel the conditional matched smoke because candidate/control runtime
  was `1.9983x`, above the locked `1.75x` Stage-A limit. Do not run validation,
  probe, test, or a Moga depth/width/kernel/layout sweep.
- [x] Fix the baseline-wide Lab-chroma zero-gradient NaN exposed by Stage A,
  prove it affected control and candidate equally, and add a zero-chroma input-
  gradient regression. This XAI repair does not override the Moga rejection.
- [x] Reject `channels_last` after a metric-free five-repeat resource probe
  worsened candidate median `0.300960 -> 0.335850 s`; do not integrate it.
- [x] Close Moga engineering with compileall, focused `13/13`, full pytest
  `965/965`, five wrapper parses, three operational preflights, and retention
  over `645` run directories with `blockers=[]`; preserve all three protected
  hashes and leave current-best commands unchanged.
- [x] Research official InceptionNeXt `atto` at commit
  `3f9769c6...cb535`, source SHA `aed1b0a9...7b88`, and distinguish its
  parallel identity/`3x3`/`1x9`/`9x1` mixer from Moga and FasterNet.
- [x] Lock the InceptionNeXt-Atto stage-1-to-3 TRKH tokenizer protocol before
  implementation. Require exact official `2/2/6`, `40/80/160`, kernel `9`,
  ratio `0.25`, ten blocks, full five-class sensitivity, and a matched
  batch-32 runtime ratio no greater than `1.75x` before any validation smoke.
- [x] Implement default-off `inceptionnext_atto_tokenizer`, checkpoint/config/
  CLI wiring, exact branch/component tests, and the train-only Stage-A audit.
- [x] Run one matched `120b x 2e`, full-val, no-test precision smoke only if
  all InceptionNeXt Stage-A gates pass; then run complete transition,
  robustness, architecture, and paired-XAI audits before any continuation.
- [x] Close InceptionNeXt after Stage A passed but the matched candidate fell
  from control macro/class1 F1 `0.768491/0.455635` to
  `0.732042/0.378531`, class1 P/R fell to `0.330049/0.443709`, and
  correction/harm plus FN-rescue/TP-break counts were `77/144` and `5/33`.
- [x] Verify all five robustness conditions, architecture trace, full
  forensics/confusions, 16-case paired native-attention/rollout/Grad-CAM XAI,
  and perturbations. Candidate won `0/5` macro conditions, had worst class1
  recall delta `-0.36424`, and used zero attribution fallback.
- [x] Do not sweep InceptionNeXt depth/width/kernel/ratio/layer-scale,
  optimizer, LR, seed, loss, augmentation, checkpoint, or run length. Do not
  run its five-epoch continuation, full train, or test.
- [x] Compact only the two rejected InceptionNeXt smoke roots after preserving
  all nonbinary evidence and exact hashes; keep Stage A and the complete pair
  audit live, then rerun retention and protected-hash checks. Compaction kept
  `324` verified payloads, excluded four binaries, reclaimed `414,355,456`
  observed bytes, and retention passed over `649` directories with no blocker.
- [x] Close InceptionNeXt engineering with compileall, focused `151/151`, full
  pytest `979/979`, seven clean PowerShell parses, and five operational
  preflights. Preserve keeper/scratch/current-command hashes and do not promote
  the rejected tokenizer.
- [x] Research and lock official StarNet-S2 stages 1-3 before implementation.
  The next candidate must use local multiplicative feature interactions while
  retaining all eight TRKH Transformer blocks, and must gate class1 TP breaks,
  `0/2/4->1` FP, illumination behavior, runtime, and no-test provenance before
  any matched smoke.
- [x] Implement default-off `starnet_s2_tokenizer` exactly as locked, including
  same-weight star-vs-sum mechanism evidence, every-branch gradients, strict
  checkpoint/config round-trip, and a train-only Stage-A resource audit.
  Stage A passed with summary SHA `7fdb0192...ddcaa`, `1.0753x` runtime,
  `2.1281 GiB` peak, and all nine star interactions live.
- [x] Run one `120b x 2e`, full-validation, no-test matched smoke only if every
  StarNet Stage-A gate passes. The exact control/candidate macro/class1 F1 was
  `0.779446/0.487685 -> 0.769836/0.443850`; class1 P/R fell
  `0.388235/0.655629 -> 0.372197/0.549669`. No test split was opened.
- [x] Complete StarNet transitions, robustness, architecture trace, paired XAI,
  and perturbation review. It reduced locked `0/2/4->1` FP by `16`, but broke
  `23` class1 TP, rescued only `7` FN, created `43` new `3->2` harms, and won
  only `1/5` robustness macro conditions. Post-smoke summary SHA is
  `a8eab240...2a1db`; five-epoch/probe/full/test permission is denied.
- [x] Reject StarNet-S2 without a depth/width/kernel/expansion/LR/seed/loss/
  augmentation/run-length sweep. Its local multiplicative tokenizer made
  attention more fragmented and border-heavy on important harms while object
  desaturation remained much more causal than background perturbations.
- [x] Preserve and compact the two rejected StarNet smoke roots after all audit
  stages completed. Evidence root
  `runs\evidence_starnet_s2_matched_smoke_rejected_20260715` keeps `322`
  copied files and `324` verified payloads at manifest SHA
  `ce98ce76...bef86`; four checkpoint binaries totaling `375,651,277` bytes
  were excluded. Cleanup manifest SHA is `92cd39e9...c590`, observed free-space
  gain is `415,756,288` bytes, and both original smoke roots are absent.
- [x] Rerun StarNet retention and protected-hash checks. Audit
  `runs\artifact_retention_audit_20260715_starnet_closure` passed over `653`
  directories with `2/2` compacted originals absent, `blockers=[]`, and summary
  SHA `d87dabd4...c1c1`. Keeper, scratch complement, and current-command hashes
  remain `1f49d577...482677`, `f8bd6309...1a549`, and
  `36b9aa1a...40faf`.
- [x] Close StarNet engineering with compileall, focused tests `43/43`, full
  pytest `997/997`, ten clean PowerShell parses, and five operational
  preflights: current-best with optional final-test request, precision package,
  TensorRT export, PyTorch video, and StarNet matched smoke. All wrappers use
  direct native invocation with `$LASTEXITCODE`; no compacted checkpoint was
  recreated and no test split was opened by the closure checks.
- [x] Review the explicit StarNet/generic-tokenizer/audit cache diff and stage
  only the 28 dependency-coherent files. `BaoCao/` and both untracked
  deep-research reports remain unstaged; commit and push this closure batch.
- [x] Lock a train-only protocol for a compact learnable-Gabor texture residual
  before implementation. It must be an equation-traceable ICCV-2023 LHO/FCM
  adaptation, preserve the keeper semantic/CNN/Transformer path, use constrained
  low/high-frequency experts, remain deployable, and pass precision plus class1-
  TP preservation gates before one no-test matched smoke.
- [x] Implement the default-off learnable-Gabor residual, checkpoint/config/CLI
  wiring, trace surface, focused regressions, and train-only Stage-A audit. Do
  not construct validation/test loaders or run a smoke unless every functional,
  gradient, illumination, export, runtime, and VRAM gate passes. After fixing
  optional-module and parent-init RNG drift, the exact rerun passed all 50
  checks with summary SHA `e53d8c1a...81e8f`, runtime `1.05603x`, peak
  `2.58663 GiB`, and ONNX error `8.34e-7`.
- [x] If and only if Stage A passes, run the one locked keeper-initialized
  `120b x 2e` control/candidate full-validation smoke with no test; then inspect
  all generic and Gabor-specific XAI before granting any continuation. The
  RNG-neutral candidate failed six gates: macro/class1 F1 was
  `0.885917/0.689855 -> 0.884195/0.689266`, class1 precision fell
  `0.613402 -> 0.600985`, and focus FP removed/created was `8/15`.
- [x] Complete robustness, confusion, boundary, architecture, Gabor-mechanism,
  and 16-case paired XAI audits. Reconcile the one AMP-to-FP32 near-tie without
  weakening prediction validation. The gate remained `2.0e-6`, FCM/filter
  features collapsed, robustness won `2/5`, and harmful cases shifted Grad-CAM
  toward borders. No five-epoch/probe/full/test continuation is permitted.
- [x] Compact nine failed/rejected Gabor roots only after hash verification.
  Evidence contains `677` verified payloads with manifest SHA
  `67a29a3f...0352`; 14 binaries totaling `702287122` bytes were excluded.
  Retention passed over `657` run directories with `blockers=[]`; keeper,
  scratch complement, and current-command hashes remain unchanged.
- [x] Do not sweep the zero-gated keeper adapter. Lock and implement a separate
  deterministic from-initialization active Gabor protocol with direct
  texture-plus-semantic addition and no learned gate/scale/router. Train-only
  Stage A passed at summary SHA `09174249...efea3` with material fusion,
  `1.10627x` runtime, `2.57851 GiB` peak, and no validation/test loader.
- [x] Run the sole `120b x 5e` deterministic scratch control/candidate smoke
  and full validation. Candidate macro/class1 F1 fell
  `0.788949/0.478049 -> 0.784714/0.459948`; class1 precision/recall fell
  `0.378378/0.649007 -> 0.377119/0.589404`, with one FN rescue versus ten TP
  breaks. All continuation gates and all `5/5` robustness comparisons failed.
- [x] Repair the changed-case evidence after self-review found a 428-row
  keeper/control union. Comparator, cohort builder, and wrapper now enforce the
  exact 90 control-candidate decisions. Rebuilt 16-case FP32 paired XAI has no
  attention or grad-rollout fallback; summary SHA is `df8cc253...e8908b`.
- [x] Close the mechanism as material but collapsed: texture/semantic norm
  `2.11654`, sample cosine `0.998529`, filter-feature cosine `0.99999791`, and
  FCM entropy `0.99999237`. Object desaturation remains causal while background
  perturbations are near zero; do not interpret focus maps as a precision win.
- [x] Compact the two rejected active-Gabor smoke roots. Evidence retains `324`
  verified payloads at manifest SHA `285bf2d4...a195`; four checkpoints
  (`350099623` bytes) were excluded. Retention passed over `661` directories
  and `192` compacted originals with `blockers=[]`.
- [x] Complete active-route engineering verification: fail-closed cohort/XAI
  resume provenance, compile, focused tests `18/18`, full pytest `1028/1028`,
  six PowerShell parses, and five no-train preflights all passed.
- [x] Do not sweep Gabor filters/bands, LHO/FCM size, fusion scale/gate, LR,
  seed, loss, augmentation, or run length. Select the next genuinely distinct
  train-only representation route only after a local no-repeat and primary-
  literature cross-check; require diversity and class1 TP protection before a
  matched no-test smoke.
- [x] Complete the local no-repeat and primary-source screen for the next
  representation route. Reject HorNet as overlapping closed Moga/StarNet
  multiplicative interactions and Selective Kernel as weakly matched to the
  normalized-crop failure mode; select parameter-matched Octave Convolution.
- [x] Lock the sole `alpha=0.125` OctConv-stem experiment before code changes in
  `docs/TRKH_5CLASS_OCTAVE_CONV_STEM_READINESS_PROTOCOL_20260715.md`. It keeps
  the current three stem widths, stride-8 output, 16x16 patch grid, all eight
  Transformer blocks, and exact convolution parameter count.
- [x] Implement default-off `stem_architecture=octave_conv`, full config/CLI/
  checkpoint/trace wiring, equation-level path tests, RNG neutrality, and the
  train-only Stage-A audit. All gates passed at summary SHA
  `b27335e7...e159b`: exact parameter/RNG parity, live frequency paths,
  noncollapse, ONNX, `1.1917x` runtime, and `2.8077 GiB`; no validation/test
  loader was constructed.
- [x] Run the sole matched `120b x 2e`, full-validation, no-test control/
  candidate smoke only if every Stage-A functional, noncollapse, ONNX, runtime,
  VRAM, and provenance gate passes; then complete transition, robustness,
  OctConv-mechanism, architecture, and exact changed-case XAI audits. Candidate
  macro/class1 F1 fell `0.784059/0.477690 -> 0.775768/0.466844`; class1 P/R
  fell `0.395652/0.602649 -> 0.389381/0.582781`, with `40/59`
  corrections/harms and `2/5` FN-rescues/TP-breaks.
- [x] Complete all OctConv post-smoke audits. It won only `2/5` robustness
  conditions; paired native-attention/rollout/Grad-CAM XAI had zero fallback
  and cleaner foreground maps, but changed-case accuracy fell `0.500 -> 0.375`.
  Surface/color sensitivity still dominates background sensitivity.
- [x] Promote or update the best full-train command only if the independently
  reloaded single-checkpoint candidate materially raises macro/class1 F1 and
  class1 precision without violating recall/TP preservation. Otherwise close
  without alpha/path/width/depth/LR/seed/loss/augmentation/run-length sweeps.
  OctConv failed eight material gates, so no promotion or command edit occurred.
- [x] Preserve all nonbinary OctConv smoke evidence, exclude rejected
  checkpoints, verify hashes, delete only the two rejected smoke roots, then
  rerun artifact retention and protected keeper/scratch/command hash checks.
  Evidence keeps `324` verified payloads at manifest SHA
  `d2772bff...f87abd`; four checkpoints (`348,783,986` bytes) were excluded.
  Retention passed over `665` directories with no blocker, and all three
  protected hashes remain unchanged.
- [x] Close OctConv engineering with compileall, focused/full pytest, affected
  PowerShell parses, current-best/precision/export/video/matched-smoke
  preflights, and an explicit staging review. Compile passed, focused tests
  passed `24/24`, full pytest passed `1040/1040`, nine launchers parsed, and
  six no-train preflights passed.
- [x] Commit and push the explicit 21-path OctConv closure batch; confirm
  `BaoCao/` and both deep-research reports remain untracked and unstaged, then
  record the commit in the next coherent research update. Pushed commit
  `706c7f5`; the three user-owned paths remain the only untracked content.
- [x] Lock the next genuinely distinct channel-interaction protocol from the
  official XCiT Cross-Covariance Attention source. Preserve the current CNN
  stem, spatial MHSA, token pruning, and XAI; require deterministic train-only
  decision-level evidence, class1 TP protection, and focus-FP reduction before
  one no-test validation smoke. Protocol:
  `docs/TRKH_5CLASS_XCA_DUAL_AXIS_READINESS_PROTOCOL_20260715.md`; official
  commit `82f5291...9cdca`, source SHA `3e2d4be8...8b3e9e`.
- [x] Implement default-off shared-projection patch-only XCA in layers `2,5`,
  full config/CLI/checkpoint/trace wiring, source-equation tests, resume
  extension policy, and a train-only Stage-A auditor. The candidate adds
  exactly `1,040` parameters, preserves all eight spatial-MHSA/XAI layers and
  pruning, and passed RNG, checkpoint, FP32/BF16 gradient, equation, ablation,
  runtime, and VRAM checks without constructing validation/test loaders.
- [x] Run the locked deterministic fold-0 `20b x 32` paired adaptation and all
  functional/export/resource gates. Candidate-minus-control class1 F1,
  precision, and recall deltas were all exactly zero; it changed only two
  decisions (`1/1` correction/harm), removed zero focus FP, and failed six
  gates. Stage B, validation, probe, full train, test, and nearby sweeps are
  denied. Summary SHA is `5429a64f...f7b0`.
- [x] Complete XCA closure verification: fixed the self-reviewed VCA/gated-
  relative conflict regression, passed compileall, focused `12/12`, full
  pytest `1051/1051`, launcher parse/preflight, and retention over `667`
  directories with `194` compacted originals absent and `blockers=[]`. Pushed
  the explicit 13-path closure as commit `9354cbc`; all user-owned untracked
  paths remained untouched.
- [x] Screen local no-repeat evidence plus official primary sources for the
  next genuinely distinct precision-oriented TRKH representation mechanism.
  Reject GRN for this round because its primary evidence is strongest with
  FCMAE and newly adding it at fine-tuning is weak; select patch-style SRM
  because learned hidden mean/std gating directly targets the observed
  illumination/style false positives without repeating MixStyle or XCA.
- [x] Lock the exact Patch-Style SRM FFN protocol before implementation:
  equation-traceable patch-only SRM after GELU in layers `2,5`, unchanged
  prefix/spatial-MHSA/pruning/XAI paths, `8,192` new trainable parameters, and
  a source-disjoint head-only versus head+SRM train-only gate. Protocol SHA is
  `e71918dc...6779`; validation/test remain closed.
- [x] Implement default-off `patch_style_recalibration`, config/CLI/checkpoint/
  trace wiring, equation/RNG/gradient/export regressions, and a fail-closed
  Stage-A auditor. Exact state/RNG/equation/gradient/gate/ablation/static-ONNX
  checks passed; the candidate added exactly `8,192` parameters and no raw
  data, validation, or test path was touched.
- [x] Run the locked fold-0 `30b x 32` train-only adaptation plus clean and
  three-condition illumination gates. Authorize one `120b x 2e` full-val,
  no-test smoke only if class1 precision/F1 rises, restricted FP falls, TP is
  preserved, and every functional/export/resource gate passes. Nine gates
  failed: adapted class1 F1/P/R fell `-0.005739/-0.001206/-0.009174`, no
  restricted FP was removed, TP breaks exceeded rescues, low-contrast
  precision fell `-0.031674`, and runtime was `1.616455x`. Stage B is denied.
- [x] Close Patch-Style SRM without layer/CFC/BN/gate/LR/seed/fold/budget/loss/
  augmentation/run-length sweeps. Preserve the completed Stage-A evidence,
  compact only two superseded infrastructure roots, rerun retention, and keep
  the keeper/scratch/current-command hashes unchanged. Retention passed over
  `670` directories with `blockers=[]`; no best-command revision was added.
- [x] Complete the fresh no-repeat and primary-source screen. Reject DeepViT
  Re-Attention because the keeper has diverse, evolving heads/layers rather
  than attention collapse; retain ODConv exclusion; select official ViG
  max-relative dynamic patch graphs as a genuinely untried mechanism.
- [x] Precommit the exact ViG protocol at SHA `9a0ad9f9...82dd76`, then
  implement default-off graph layers `2,5`, deterministic `256->64` projection,
  `k=9` normalized kNN, max-relative message, zero-init residual, complete
  config/CLI/resume/trace wiring, tests, and a no-validation/no-test Stage-A
  auditor.
- [x] Run the locked source-disjoint `30b x 32` train-only paired adaptation.
  The graph was structurally live but reduced macro/class1 F1 by
  `-0.034419/-0.147450`; class1 precision rose `+0.010106` only while recall
  fell `-0.229358`, with `0/25` FN-rescues/TP-breaks and `8/29`
  corrections/harms. All three lighting class1 F1 deltas were negative.
  Stage B is denied; summary SHA is `8bcacda2...bde77`.
- [x] Close ViG without k/layer/width/type/self-neighbor/residual/LR/seed/fold/
  budget/loss/augmentation/run-length sweeps and without validation/test/XAI
  Stage B. Preserve nonbinary evidence, compact rejected ONNX binaries, rerun
  full engineering verification/retention, and leave the current-best command
  unchanged. Compaction retained seven verified payloads at manifest SHA
  `7452c05a...e7671`, excluded `31,168,862` ONNX bytes, and retention passed
  over `672` directories with zero originals remaining and `blockers=[]`.
  Compileall, focused `37/37`, full pytest `1078/1078`, five launcher parses,
  and current-best/TensorRT/video preflights passed.
- [x] Complete the post-ViG no-repeat and primary-source review. Reject CAL as
  overlapping closed bilinear attention crop/drop; select an exact deep
  class-prompt adaptation grounded in official Prompt-CAM and MCTformer code.
  Lock direct class1 TP protection, focus-FP suppression, spatial separation,
  illumination, export, runtime, and VRAM gates before implementation.
- [x] Implement default-off deep class prompts at all eight Transformer blocks,
  preserving legacy prefix/pruning/native-attention schemas while exporting
  separate reconstructed class-to-patch maps. Add exactly `11,009` parameters,
  complete config/CLI/resume/trace wiring, focused tests, and a train-only
  fail-closed Stage-A launcher/auditor.
- [x] Run the locked source-disjoint `60b x 32` head-only versus head+prompt
  adaptation without validation/test. Candidate macro/class1 F1 fell
  `-0.029655/-0.081406`; precision rose `+0.100952` only while recall fell
  `-0.229358`, with `27/62` corrections/harms and `0/25` class1 FN-rescues/
  TP-breaks. Every lighting class1 F1 delta was below `-0.11`; Stage B is
  denied.
- [x] Complete prompt XAI and resource/export review. Prompt maps were diverse
  and ONNX/runtime/VRAM passed, but target-versus-confuser separation fell
  `-0.034383`. Fix the independently found CUDA constructor-RNG restoration
  bug and verify it without rerunning or reinterpreting the rejected method.
- [x] Close deep class prompts without prompt-count/layer/init/fusion/head/LR/
  seed/fold/budget/loss/augmentation/run-length sweeps. Preserve seven verified
  nonbinary payloads at manifest SHA `d7b6b64b...e137d2`, exclude `30,699,520`
  reproducible binary bytes, verify source deletion, and pass retention over
  `674` directories with `blockers=[]`. Compileall, focused `10/10`, full
  pytest `1088/1088`, five launcher parses, and current-best/TensorRT/video
  preflights passed. Keep best commands unchanged.
- [x] Perform a fresh no-repeat and primary-source screen for the next distinct
  foreground surface/boundary mechanism. Reject hard V-MoE and another
  granularity-specific classifier because they require long schedules or
  overlap closed routers/heads. Select a small patch-token Soft MoE grounded
  in the official V-MoE implementation, with moderate-scale vision-MoE risk
  explicitly treated as a fail-closed hypothesis rather than assumed gain.
- [x] Lock the exact default-off Soft-MoE patch-adapter protocol before code:
  four experts with one slot each, `256->64->256`, layers `2,5`, normalized
  linear routing, zero-init `0.10` residual, exact `266,754` added parameters,
  matched fixed-router control, raw-keeper comparison, TP/precision/XAI/
  illumination/export/resource gates, and no validation/test access.
- [x] Implement the Soft-MoE module, model/config/CLI/resume/trace wiring,
  launcher, equation/RNG/checkpoint/gradient/export tests, and train-only
  Stage-A auditor without changing any raw data or best-command artifact.
- [x] Run the sole locked source-disjoint `60b x 32` Stage A. Inspect every
  decision, illumination, routing-collapse, bbox-residual XAI, expert-ablation,
  ONNX, runtime, and VRAM gate before deciding whether one no-test validation
  smoke is authorized. Candidate changed zero clean decisions and removed zero
  of 37 restricted focus FP; eight behavioral gates failed, so Stage B is
  denied.
- [x] Repair the audit-only FP32 metadata slicing mismatch and same-precision
  BF16 equation reference, add regressions, and preserve the immutable formal
  summary plus hash-locked correction without changing the failed outcome.
- [x] Close Soft MoE without expert/slot/layer/width/router/residual/LR/seed/
  fold/budget/loss/augmentation/run-length sweeps. Retain eight verified
  payloads at manifest SHA `75de0592...b0cd`, exclude `33,805,837` binary
  bytes, pass retention over `676` directories, compileall, focused `33/33`,
  full pytest `1103/1103`, five parses, and five operational preflights. Keep
  the best command/history unchanged.
- [x] Perform a fresh no-repeat and primary-source screen for the next distinct
  foreground surface/boundary representation mechanism. Differential
  Attention is a poor match because the keeper already has foreground-clean,
  diverse attention and negligible background sensitivity. Select an exact
  class-1-protected RSC training intervention from official ECCV-2020 source
  commit `bf6d280...59666`, while treating pretrained/long-schedule evidence
  as an explicit transfer risk.
- [x] Lock the class-1-protected pooled-channel RSC protocol before code or
  adaptation. A train-only FP32 A0 cohort used 36 restricted FP, all 109
  class-1 fold-0 rows, and 36 matched correct negatives. Eligible rows masked
  exactly `86/256` channels; class-1 rows had zero masked channels and zero
  prediction changes. Protocol SHA is `9227f5e5...59366`.
- [x] Implement the source-equation RSC helper and fail-closed Stage-A auditor,
  including exact class-1 protection, signed-gradient top-k, positive-drop
  batch selection, control/candidate RNG parity, full-model movement, clean
  and illumination decisions, ONNX, runtime, and VRAM checks. Native-grid
  positional interpolation is now skipped bit-exactly to permit deterministic
  CUDA backward; focused equation and gate tests cover the implementation.
- [x] Run the sole locked source-disjoint `60b x 32` Stage A. Authorize one
  no-test `120b x 2e` validation smoke only if class-1 precision and F1 rise,
  restricted FP fall, TP/recall are protected, and every engineering gate
  passes. Five gates failed: precision rose only `+0.002005`, restricted FP
  stayed `15 -> 15`, bright class-1 F1 fell `-0.012579` with one net new FP,
  and peak allocation was `3.907542 GiB`. Stage B is denied; no command
  revision occurred.
- [x] Complete RSC closure engineering: retain the four hash-verified
  nonbinary payloads, run compile/focused/full tests, parse and preflight all
  affected launchers, rerun artifact retention and protected-hash checks, then
  commit and push only the explicit RSC paths. Compile passed, focused/full
  pytest passed `12/12` and `1115/1115`, five launchers parsed, four
  operational preflights passed, and retention passed over `678` directories
  with `blockers=[]`. Protected hashes and three command revisions are exact.
- [x] Perform a fresh no-repeat and primary-source screen after RSC. Re-read
  official Swin and convolutional-stem sources, separate native stride-8 local
  attention from rejected image resizing/tiles/SPT/tokenizers, and precommit a
  train-only source-group-held-out information gate before architecture code.
- [x] Implement and run the exact high-resolution bridge A0 on all `9215`
  train rows. The `9215x288` aligned Haar detail had effective rank `72.27`
  but lost to the matched control in all five folds: macro/class1 F1
  `-0.032002/-0.061779`, class1 precision `-0.159006`, restricted FP
  `130 -> 249`, and `204/464` corrections/harms. Architecture implementation,
  validation, test, probe, and full train are denied.
- [x] Self-review the A0 comparator and artifacts. Independent CSV
  recomputation matched the summary; the collapsed control and all 15 LBFGS
  fits reaching the iteration cap add rejection evidence. Tighten the future
  optimizer-convergence gate, preserve six nonbinary payloads, pass retention
  over `680` directories with `blockers=[]`, compile/focused/full pytest
  `7/7` and `1122/1122`, parse/preflight the launcher, and leave the
  current-best command/history unchanged.
- [x] Perform a fresh no-repeat and primary-source screen after the
  high-resolution bridge. Select one matched natural-prior continuation to
  test whether scratch class1 overprediction came from `3.41x` strict-balanced
  exposure, while locking recall/TP protection before validation.
- [x] Run and fully audit the exact `2 x 60`-batch strict-versus-natural
  continuation. Natural sampling removed five restricted FP but broke six
  class1 TP, reduced class1 F1 `-0.012714` versus control, and failed 13 gates.
  XAI/robustness identify broad class suppression, so probe/full train/test and
  nearby sampler/LR/budget sweeps are denied.
- [x] Compact all natural-prior evidence after post-smoke robustness/XAI.
  Retain `420` verified payloads at manifest SHA `8c223f60...613ea9`, exclude
  `576,989,850` reproducible bytes, delete three verified sources, and pass
  retention over `682` directories with `blockers=[]`. Keep best commands and
  their three-revision history unchanged.
- [x] Complete the next primary-source screen. Select IP-DPP only for a
  no-training information gate; record that official schedules are
  `1000+100+100` epochs and paper Eq. 16 uses `/N` while official code uses
  `/N^2`, which may make the implemented kernel effectively random-balanced.
- [x] Lock, implement, and run a train-only IP-DPP A0 on the existing CIDT
  `9215`-row cache. Compare paper-`N`, code-`N^2`, and deterministic random
  balanced subsets with exact unique/source-fold checks, class1 TP retention,
  hard head-negative enrichment, spectral validity, and fold stability. Do
  not integrate a trainer, access validation/test, or sweep k/seed unless A0
  proves informative selection beyond random. A0 failed six mechanism gates:
  official default is identity, official extraction is uniform random,
  paper/code scales disagree, paper DPPy cannot initialize, code odds are only
  `1.000610/1.000721`, and no condition exceeds random hardness q95.
- [x] Close direct IP-DPP without k/seed/chain/kernel/probability-source/
  resampling/continuation sweeps. Preserve seven nonbinary payloads at manifest
  SHA `a77f9788...09608e`; independent replay matched exact subset hashes and
  null q95. Retention passed over `684` directories with `blockers=[]`. Deny
  trainer/validation/test/full train and keep best commands unchanged.
- [x] Perform a fresh no-repeat and primary-source screen after IP-DPP. Avoid
  another global rebalancing or subset-downsampling method: TRKH imbalance is
  only `4.66x`, while the measured problem is a narrow `0/2/4 <-> 1` boundary.
  Prefer a train-only mechanism that keeps every class1 row and applies
  class-selective pressure only to verified restricted head-class negatives.
- [x] Lock, implement, test, and run the sole class-1-reference A-GEM A0. The
  `186` hard-negative and `432` class1 gradients conflicted at cosine
  `-0.556244`; exact projection retained `83.10%` of the direction and
  improved recall versus unprojected control `0.880734 -> 0.926606`.
- [x] Close one-reference A-GEM at train-only A0. Versus the raw keeper it
  removed eight restricted FP and raised precision `+0.034694`, but broke six
  true class1 predictions, reduced recall `-0.055046`, reduced macro/class1 F1
  `-0.001452/-0.000467`, and failed low-contrast safety. Independent replay
  matched all `4 x 1843` rows; no binary model or validation/test artifact was
  produced. Keep the best command/history unchanged.
- [x] Complete A-GEM closure engineering: compileall, focused `7/7`, full
  pytest `1142/1142`, launcher parse/preflight, manifest/hash replay, four
  protected hashes, and retention over `687` directories all passed. Final
  closure SHA is `ceca3e91...326bbde`; command tracking remains `3` revisions
  and `2` updates after the initial revision.
- [x] Perform a fresh primary-source and no-repeat screen after A-GEM. Because
  the one averaged class1 CE constraint hid finite-step harm to vulnerable
  true positives, consider only a genuinely multi-constraint decision-margin
  method with raw keeper and one-reference A-GEM comparators. Precommit the
  strata, QP/equation, normalized step, and TP/support/illumination gates
  before implementation; do not sweep the closed A-GEM recipe.
- [x] Lock, implement, and independently audit the sole boundary-stratified
  GEM A0. Exact GEM satisfied every structural/QP gate but improved clean
  class1 F1 only `+0.003383`, removed one restricted FP, and lost class1 F1
  `-0.010314` to aggregate-margin A-GEM. Dim and low contrast created net FP;
  Stage B and all downstream access are denied.
- [x] Complete GEM closure engineering: run compile/focused/full tests, parse
  and preflight the launcher, rerun retention and all protected-hash checks,
  then commit and push only explicit GEM/doc paths. Keep command tracking at
  three revisions because no locked validation promotion occurred. Compile,
  focused `8/8`, full pytest `1150/1150`, parse/preflight, diff check, and
  retention over `689` directories all passed. Final closure SHA is
  `a94976b6...7b3af985`; all four protected hashes are exact.
- [x] Perform a fresh primary-source/no-repeat screen for a robust
  multi-condition training objective. The clean aggregate-margin signal is
  worth preserving, but any candidate must predefine clean/dim/bright/
  low-contrast groups, optimize worst-group risk without raw-data changes,
  protect class1 TP/support, and remain distinct from closed GEM/A-GEM and
  prior augmentation or sampling sweeps. Select V-REx from the ICML-2021 paper
  and DomainBed commit `b93c22a...f4139e`; direct GroupDRO and generic
  consistency/canonicalization are already closed.
- [x] Lock the boundary-balanced V-REx A0 before code. Use identical rows and
  labels in clean/dim/bright/low-contrast environments, equal hard-negative/
  class1 CE risk, official pre-anneal beta `1`, matched ERM, one normalized
  `1e-4` step, and raw/aggregate-margin comparators. Deny all downstream work
  unless every clean, TP/recall, FP, worst-condition, and risk-variance gate
  passes. Protocol SHA is `4a798895...0a48246`.
- [x] Implement the locked V-REx auditor and focused equation/gate tests.
  Require exact DomainBed/paper/local hashes, source-disjoint cohorts, matched
  environment rows, independent flat-gradient equations, risk telemetry,
  no validation/test/binary artifact, and a preflight that creates no run.
  Compile, focused `6/6`, PowerShell parse, diff check, and preflight passed;
  prior argmax/CIDT mismatch is zero.
- [x] Run the sole locked V-REx A0 and independently replay every metric,
  transition, risk, and artifact hash. Authorize Stage B only if all structural,
  clean precision/recall/FP, matched-ERM, worst-condition, and risk-variance
  gates pass; otherwise close without beta/environment/step sweeps. V-REx and
  ERM made identical clean decisions, reduced class1 F1/precision
  `-0.026129/-0.039642`, and created eight restricted FP; all `7372` replay
  rows matched exactly, so Stage B is denied.
- [x] Complete V-REx closure engineering: finalize closure/manifest hashes,
  compile, run focused and full pytest, parse and preflight the launcher, run
  read-only retention, verify all protected hashes and command revision count,
  then commit/push only the explicit V-REx batch. Compile/focused/full tests
  passed (`6/6`, `1156/1156`), parse/preflight and independent replay passed,
  and retention passed over `691` directories with `blockers=[]`. Final
  closure SHA is `b481e9a5...877920`; command tracking remains three revisions.
- [x] Perform a fresh primary-source/no-repeat screen after V-REx. Select one
  final condition-gradient A0 using NeurIPS-2021 CAGrad, because it optimizes
  worst local task improvement while retaining the average objective for
  `c<1`; keep the failed spatial PCGrad, GEM/A-GEM, GroupDRO, and V-REx routes
  closed.
- [x] Lock boundary-balanced CAGrad before code: `c=0.4` from the paper's
  NYU-v2 vision result, four identical-row condition risks, equal hard/class1
  CE, one normalized `1e-4` step, raw/margin/ERM/V-REx comparators, exact
  simplex/KKT/source-replay and precision/TP/FP/illumination gates. Protocol
  SHA is `79047d2b...48c9419`; no neighboring sweep is allowed.
- [x] Implement the fail-closed CAGrad auditor, official-source/paper-scale
  equation checks, deterministic multistart solver replay, focused tests, and
  PowerShell preflight. Do not touch the shared trainer or any raw dataset.
  Compile/focused `5/5`, launcher parse, diff check, and preflight passed with
  exact source/cohort/prior hashes and no output directory.
- [x] Run the sole locked train-only CAGrad A0 and independently replay every
  solver/CSV/risk/hash artifact. The valid conflict-averse direction still
  reduced clean macro/class1 F1 `-0.008905/-0.026129`, lowered precision
  `-0.039642`, and increased restricted FP `36 -> 44`; all shifted conditions
  also created FP. Stage B is denied and the condition-gradient family is
  closed without sweeps.
- [x] Complete CAGrad closure engineering: compilation/focused `5/5`, full
  pytest `1161/1161`, launcher parse/preflight, independent replay, and
  retention over `693` directories all passed; all protected hashes are exact.
  Current-best command tracking remains three revisions/two updates and is
  unchanged by CAGrad. Final closure SHA is `adf12c7e...db94ce6`.
- [x] Perform a fresh primary-source/no-repeat screen for a representation or
  objective mechanism outside the closed gradient-combination family. Select
  one favorable class-axis diagnostic inspired by CCAR and one fixed
  multimodal-density diagnostic inspired by Goto et al. WACV 2024; retain raw
  and aggregate-margin comparators and lock a train-only information gate
  before shared-trainer work.
- [x] Implement, test, precommit, and run the class-axis/multimodal A0 on the
  exact source-disjoint `7372/1843` train split. Class-axis energy collapsed to
  macro/class1 F1 `0.807973/0.494279` and increased restricted FP `36 -> 206`.
  The 10-mode GMM reached `0.929297/0.801688` with direction AUROC `0.652778`,
  but broke 13 class1 TP and reduced recall `0.981651 -> 0.871560`. Deny both
  routes without validation/test/trainer integration or parameter sweeps.
- [x] Independently replay all `1843` rows, decisions, confusion matrices,
  transitions, restricted-FP counts, and artifact hashes. Preserve four
  nonbinary payloads under
  `runs/audit_class_axis_multimodal_readiness_20260715`; leave current-best
  command tracking at three revisions/two updates. Compileall, focused `4/4`,
  full pytest `1165/1165`, launcher parse/preflight, protected hashes, and
  retention over `695` directories with `blockers=[]` all passed. Final
  closure SHA is `cbafd55b...f414c1`.
- [x] Complete a primary-source/no-repeat decision on GSFL-style
  shared/discriminative feature decomposition before code. Reconcile its
  pretrained VGG, cross-validated group count, `150+200` epochs, test-selected
  official loop, missing repository license, and `labels[0]` shared-center
  indexing with TRKH's scratch/`<=30e`/train-only rules. Only a prospectively
  fixed source-grouped frozen-adapter gate with raw/margin comparators and zero
  class1 TP loss may proceed; otherwise reject the family without a smoke. A
  paper-only adapter was judged distinct enough for one locked A0; protocol SHA
  is `1d5dba1...eaafb` and precommit/fix commits are `99b3f58`/`ca6fdb8`.
- [x] Run and independently replay the sole GSFL train-only adapter A0. Every
  structural gate passed, but precision `+0.104959` came with recall
  `-0.128440`: 20 restricted FP were removed while 14 class1 TP were broken,
  macro F1 fell `-0.003673`, and direction AUROC was `0.263889`. Deny trainer
  integration and all nearby decomposition/center/loss/width/epoch sweeps.
- [x] Complete GSFL closure engineering. Compile/focused `5/5`, full pytest
  `1170/1170`, launcher parse/preflight, independent replay, all protected
  hashes, and retention over `697` directories passed with `blockers=[]`.
  Summary/manifest SHAs are `e1396669...d801d`/`7c5d68a9...099f`; current-best
  command tracking remains three revisions/two updates. Final closure SHA is
  `49e213c2...c11c4`.
- [x] Perform a fresh primary-source/no-repeat screen for positive local class1
  evidence. Reject SIC/foundation-part prototypes/FAREL before code because of
  pretrained-feature, already-closed prototype/frequency, or `100-1600` epoch
  conflicts; select only DDHTS cross-layer plus cross-IUWT sign coding for one
  prospectively locked train-only A0.
- [x] Run the sole fixed DDHTS gate on exact source-disjoint `7372/1843`
  `yolo_f/train` rows. Clean changed zero decisions, removed `0/36` restricted
  FP, and reached direction AUROC `0.548546`; a zero-TP-loss holdout oracle also
  removed zero FP. Dim lighting broke one TP and removed no FP, so Stage B,
  validation, test, probe/full train, and command promotion are denied.
- [x] Complete DDHTS closure: inspect both frequency-view contact sheets,
  independently replay all `4 x 1843` decisions/confusions, preserve ten
  payloads, compile, pass focused/full pytest `6/6` and `1176/1176`, parse and
  preflight the launcher, verify protected hashes, and pass retention over
  `699` directories with `blockers=[]`. Summary/closure-manifest SHAs are
  `e65e2b65...d6512b`/`89d14eb3...efb76`; commands remain at three revisions.
  Final closure-document SHA is `145b6776...2519f`.
- [x] Perform a fresh primary-source/no-repeat screen outside fixed texture,
  part/prototype, frozen-readout, post-hoc threshold, and gradient-combination
  families. Reject AdvRF before code because its pretrained, reconstruction-
  heavy, `200`-epoch/four-A100 recipe violates the current constraints. Select
  one NeurIPS-2023 CEConv first-block residual A0 and lock exact official
  equations, a matched identity control, zero-gate equivalence, source-
  disjoint train cohorts, precision/TP/illumination/resource/export/XAI gates,
  and no sweep. Protocol SHA is `9c5af4e9...da57082`.
- [x] Implement, precommit, and run the sole CEConv train-only A0. Official
  equation/equivariance, pairing, gradient, frozen-state, deterministic, and
  ONNX checks passed, but clean macro/class1 F1 fell
  `-0.002254/-0.009991`, class1 precision fell `-0.015375`, and restricted FP
  increased `36 -> 39`. Dim/bright/low-contrast precision also fell about
  `0.040-0.042`; Stage B, validation, test, trainer integration, probe/full
  train, and command promotion are denied.
- [x] Complete CEConv closure engineering. Inspect all required XAI changes,
  replay all `4 x 1843` predictions, recompute nine payload hashes/sizes,
  compile, pass focused/full pytest `6/6` and `1182/1182`, parse/preflight the
  launcher, verify protected hashes, and pass read-only retention over `701`
  directories with `blockers=[]`. Summary/manifest SHAs are
  `8177f976...3bca7a`/`96fc5719...1f5972`; commands remain three revisions/two
  updates. Final closure SHA is `047e3ba6...83cfe0`.
- [x] Perform a primary-source/no-repeat screen outside color invariance and
  equivariance. Select NeurIPS-2021 AugSelf because color-parameter difference
  prediction preserves augmentation information; lock a paper-only final-head
  adapter with matched detached-gradient control, exact source-disjoint
  train-only cohorts, precision/TP/illumination/resource/export/XAI gates, and
  no validation/test access. Protocol SHA is `248f1ade...203ae`.
- [x] Implement, precommit, and run the sole AugSelf color-adapter A0. All
  structural/replay/deployment gates passed, but candidate color MSE was
  `1.06315x` control and clean macro/class1 F1 collapsed to
  `0.308590/0.149315`; class1 precision was `0.080681` and restricted FP rose
  `36 -> 740`. Deny Stage B and all nearby sweeps.
- [x] Complete AugSelf closure: inspect representative pages plus all-map
  statistics over 708 required XAI rows, replay all `7372` behavior rows and
  color views, pass compile/focused/full tests `6/6` and `1188/1188`, parse/
  preflight, protected hashes, and retention over `703` directories with
  `blockers=[]`. Commands remain three revisions/two updates.
- [x] Perform the next primary-source/no-repeat screen outside color adapters,
  invariance/equivariance, narrow-cohort CE updates, and every closed family in
  the journal. Stock hybrids and Conformer-style coupling were already closed;
  select TIP-2020 Mutual-Channel Loss as a distinct class-aligned channel-group
  mechanism, with official commit/source/license and paper hashes verified.
- [x] Lock the exact train-only final-patch MCL residual A0 before code. Reuse
  source-disjoint `7372/1843`, natural multiclass fit order, three channels per
  class, an identical CE-only control, official `alpha=1.5/beta=20`, ten frozen-
  feature epochs, and strict precision/TP/illumination/mechanism/deployment/XAI
  gates. Validation, test, trainer integration, sweeps, and command promotion
  remain forbidden unless every gate passes.
- [x] Correct the pre-metric patch-count assumption with a real keeper batch:
  final pruning retains `ceil(256 * 0.65) = 167` tokens because keep rates are
  absolute to the original grid, not sequential. Amend only this structural
  contract before formal training; no behavior metric or gate changed.
- [x] Implement the isolated MCL auditor, focused equation/replay tests, and a
  VS Code-safe PowerShell launcher without modifying shared model/trainer code.
  Correct the pre-metric patch count to 167, pass equation tests `8/8`, and
  pass formal hash/order/mask/source/license preflight without creating output.
- [x] Run and close the sole locked MCL A0. The mechanism worked (holdout group
  accuracy `0.912100`, max cosine `0.529963`, AUROC `0.946437`) but clean
  macro/class1 F1 fell `-0.049074/-0.158670`; 40 class1 TP broke while only 18
  net restricted FP were removed. Candidate also lost to natural CE control.
  Deny Stage B, validation/test, neighboring sweeps, probe/full train, and
  command promotion. Replay `7372/7372`, XAI `92/92`, ONNX, full pytest
  `1196/1196`, protected hashes, and retention over 705 directories passed.
- [x] Perform a fresh primary-source/no-repeat screen for a genuinely distinct
  representation that prospectively constrains natural-multiclass class1 TP
  and support under clean/dim/bright/low-contrast conditions. Select only
  dynamic routing capsules plus augmented-Lagrangian rank/support constraints;
  retain raw, natural-CE, and MCL comparators and forbid post-hoc calibration.
- [x] Lock, implement, precommit, and run the sole CapsALM train-only A0 on the
  exact source-disjoint `7372/1843` split. Support violation improved strongly
  versus CE control, but routing stayed near uniform and clean macro/class1 F1
  fell `-0.012661/-0.007905` versus raw; restricted FP increased `37 -> 38`
  and corrections/harms were `4/29`. Deny Stage B and every nearby sweep.
- [x] Complete CapsALM closure: inspect representative XAI pages including the
  zero-prior tiny-edge event, replay all `7372` predictions and 47 payload
  hashes, fix inference-tensor ONNX tracing, pass focused/full tests `7/7` and
  `1203/1203`, parse/preflight, protected hashes, and retention over 707
  directories with `blockers=[]`. Commands remain three revisions/two updates;
  final closure SHA is `54fddeb9...fa4fc`.
- [x] Perform a fresh primary-source/no-repeat screen outside capsules, channel
  diversity, color/frequency/style, prototype/subspace, output calibration,
  feature stitching, and closed gradient/objective families. Before training,
  require a train-only selectivity information gate on class1 TP versus hard
  negatives across clean/dim/bright/low-contrast, and reject nonselective
  spatial mechanisms without an image-model smoke. Select CVPR-2025
  Supervised Minority from the accepted CVF paper and official MIT repository;
  lock the exact condition-view A0 at protocol SHA `d4fb528c...61b4ce9`.
- [x] Implement/precommit A0 and diagnose its formal provenance stop. The full
  clean FP16 cache replayed deterministically but could not reproduce FP32
  CIDT: maximum probability error `0.01413372`, with sample `2589` changing
  class because FP16 tied p0/p1. Prove both model loaders/state dictionaries
  bit-exact and FP32-to-CIDT error `<=5.96e-8` on the 64 tightest-margin rows.
  Close A0 before selectivity/training without validation/test or command
  revision; closure SHA is `49420090...fac2b453`.
- [x] Implement, test, precommit, and run the sole A1 numeric correction at
  protocol SHA `e17f5cb5...744d9aa2`. Preserve A0 as reproducible history;
  select `a1_fp32_cidt`, extract all condition caches in FP32/batch64, require
  all-row CIDT error `<=1e-6` and zero argmax mismatch, then continue only if
  every inherited pre-training selectivity gate passes. Numeric replay passed
  at `8.94e-8/0` mismatch, but pre-training selectivity rejected before epoch
  1: clean TP-hard AUROC was `0.589174`, shifted class1 CAC was zero, and
  shifted SAA was only `0.131850-0.315247`.
- [x] Close A1 without training/XAI/deployment/validation/test. Independently
  replay the gate and all three manifest payloads, fix the post-run hard-coded
  A0 report title with a regression test, pass focused/full tests `14/14` and
  `1217/1217`, verify protected hashes, and pass retention over 709 directories
  with `blockers=[]`. Closure SHA is `aab02b4b...6f28005d`; commands remain
  three revisions/two updates and zero keeper replacements.
- [x] Perform the next primary-source/no-repeat screen outside frozen pooled
  condition alignment and every closed architecture/objective family. Select
  TransNeXt Aggregated Attention from the accepted CVPR-2024 paper and official
  Apache-2.0 repository, but authorize only one bounded block-1 replacement
  because the stock recipe uses 300 epochs.
- [x] Implement and preflight the sole locked FAA A1 on the exact source-
  disjoint `7372/1843` train-only fold. Official output/gradient replay, common
  state, gradients, BF16, ONNX, resources, and 41/42 checks passed, but only
  `0.599609` of patch queries used both routes above mass 0.05 versus the locked
  `0.95`. Deny the formal pair, XAI, validation/test, Stage B, and command
  promotion; do not sweep this family.
- [x] Preserve the FAA closure and compact only reproducible preflight payloads.
  Retain the exact summary/manifest, exclude the 30.789 MB reproducible ONNX,
  verify cleanup manifest SHA `e7278182...a20ff9c`, and pass read-only retention
  over 712 directories with `blockers=[]` and 71.226 GB free.
- [x] Perform a new primary-source/no-repeat screen outside static local/global
  attention competition. Select CVPR-2022 DAT from official Apache-2.0 tag
  `CVPR2022` at `566a593...b6f4ff`; reject stock 29M/300e DAT and distinguish
  continuous spatial K/V sampling from closed FAA and ViG.
- [x] Lock one exact block-2 DAT A1 at protocol SHA
  `6abbd94c...3f73cbc`: 2 groups, 5x5 offset kernel, stride 1, range 2,
  continuous RPE, global prefix path, exact pruning proxy, source-disjoint
  `7372/1843` scratch pair, precision/TP/condition/bbox-selectivity gates, and
  no validation/test/sweep/command promotion.
- [x] Implement the locked module, config/trainer/PowerShell integration,
  official-equation and bilinear-proxy preflight, deformable trace/audit, and
  focused tests. Commit/push through `aa26c09`; corrected formal preflight
  passed `65/65` and authorized exactly one five-epoch pair.
- [x] Complete the DAT A1 pair and corrected full audit. Clean class1 F1/
  precision improved `+0.015929/+0.046218`, but macro F1 fell `-0.003347`,
  errors were `10/23` corrections/harms, dim/low-contrast FP increased, clean
  selectivity AUROC stayed `0.603583`, outside-bbox distance worsened, and both
  stem/block-2 XAI foreground mass fell. Deny Stage B and all DAT sweeps.
- [x] Diagnose and fix the audit-only trace-path error. Separate standard-
  inference behavior logits from attention-trace spatial evidence, add a
  regression test, regenerate all 14,744 predictions and 16 XAI pages, and
  independently replay metrics to `1.11e-16` maximum error.
- [x] Complete DAT closure engineering: compile/parse, focused `19/19`, full
  pytest `1250/1250`, protected hashes, and read-only retention over 719
  directories all passed with `blockers=[]`. Compaction removed 520,134,563
  bytes; current-best commands remain three revisions/two updates.
- [x] Review only the explicit DAT closure paths and push commit
  `d6e40e4d6bd79a659b7ee64f64c335c500726b7a` without protected untracked
  reports. Final closure SHA is `e97b03ef...f5225a5`; HEAD equals upstream.
- [x] Perform the next accepted-primary-paper/official-code no-repeat screen.
  Reject HorNet/ODConv and stock Next-ViT/EfficientViT-style catalog hybrids as
  overlaps; defer unlicensed/pretrained FENet; select CVPR-2021 Diverse Branch
  Block from official Apache-2.0 commit `8d2b16b...ba929` as a distinct
  training-time structural re-parameterization route for the current hybrid.
- [x] Lock one exact three-block DBB stem A0 before implementation at protocol
  SHA `578b208f...5d22ff`: official four paths, unchanged GELU/max-pool and
  Transformer, exact deploy fusion, source-disjoint `7372/1843` scratch pair,
  precision/TP/condition/mechanism/XAI gates, and no validation/test/sweep or
  current-best command update.
- [x] Implement the Apache-attributed DBB stem, config/launcher integration,
  exact official-equation/deploy replay, common-state construction, ONNX and
  resource preflight, focused tests, and a VS Code-safe A0 wrapper. Commit
  `ed45a46` was pushed before formal execution; full pytest passed `1261/1261`.
- [x] Run the formal preflight gate before the authorized five-epoch pair. It
  passed 53/58 checks but denied the pair: official real-train replay
  `1.91e-6 > 1e-6`, BF16 deploy stem error `0.09375 > 0.002`, train runtime
  `1.841x > 1.50x`, VRAM ratio `1.557x > 1.25x`, and converted inference
  `1.083x > 1.05x`. Therefore no pair, behavior audit, or XAI was run.
- [x] Apply the locked DBB stop rule and compact rejected preflight evidence.
  Preserve summary/manifest SHAs `a60983ff...b55c8e`/`cd1e0fa6...16c37`,
  remove only the reproducible 30,419,262-byte ONNX through verified cleanup
  manifest `33724530...de722`; retention passed over 721 directories with
  `blockers=[]` at SHA `77cc4282...c381d`. Leave current-best commands
  unchanged.
- [x] Perform the next accepted-primary-paper and official-licensed-code
  no-repeat screen outside DBB/RepVGG/structural-reparameterization sweeps,
  DAT/FAA, persistent tokenizer branches, and all previously closed families.
  Reject SCConv/DilateFormer/LSKNet as license or local-overlap failures and
  select official MIT CVPR-2023 BiFormer BRA as the distinct query-specific
  context-filtering mechanism.
- [x] Lock the exact block-2 BRA A1 before code at protocol SHA
  `e7ce7cf3...c3ede3`: `16x16`, `S=4`, matched topk-16/topk-4 roles, identical
  QKV/projection/LCE state, global prefix handling, official replay, source-
  disjoint pre-training selectivity, resource/export/precision/TP/condition/
  XAI gates, and no validation/test/sweep/current-command update.
- [x] Implement the default-off MIT-attributed BRA module, model/config/train/
  trace wiring, exact official-equation and dense-MHSA parity tests, full
  engineering/selectivity auditor, route-overlay review gate, and VS Code-safe
  three-phase wrapper. Separate pruning attention from expensive audit trace,
  lock seven prefixes and incompatible routes, pass launcher/config preflight,
  focused `38/38`, and full pytest `1276/1276`. Commit/push this reviewed stage
  before the sole formal preflight.
- [x] Run the formal BRA preflight exactly once and apply the stop rule. The
  formal artifact passed 64/75 checks but failed inference runtime
  (`1.390526x`), two object-query coverage rows, clean positive-gain fraction
  (`0.486692`), dim route stability (`0.616128`), and visual review. Deny the
  pair, validation/test, full train, sweeps, and command promotion.
- [x] Correct the three BRA reporting defects without rerunning formal work.
  Hash-lock the original summary/CSV, replay role-RNG and deployment-compatible
  trace parity, aggregate finite geometry while preserving invalid rows, and
  confirm all five material rejection categories remain. Focused/full tests
  pass `18/18` and `1279/1279`; fix/replay commit `6eb272e` is pushed.
- [x] Complete BRA closure and compact only the reproducible 32,678,307-byte
  ONNX. Preserve formal/replay/CSV/visual/overlay evidence at payload-manifest
  SHA `5e19d833...f38773`; cleanup SHA is `cbbc2d8d...3bcb57`. Retention passes
  over 723 directories, all 205 compacted originals are absent, and
  `blockers=[]`. Closure SHA is `ab74dc74...766e4`; commands remain three
  revisions/two updates.
- [x] Perform a new accepted-primary-paper/official-licensed-code no-repeat
  screen outside BRA and every closed family. Require a prospective train-only
  gate that directly covers tiny edge objects, post-pruning spatial support,
  class1 precision/TP safety, and surface-boundary separability before any
  implementation or epoch.
- [x] Complete that screen against accepted papers and official licensed code.
  Select ICLR-2022 EViT inattentive-token fusion as the exact unresolved
  hard-drop operation; defer TNT/NesT/PaCaViT/Evo-ViT and forbid revisiting
  no-prune or higher keep rates.
- [x] Lock one parameter-free TRKH adaptation before code at
  `docs/TRKH_5CLASS_INATTENTIVE_TOKEN_FUSION_READINESS_PROTOCOL_20260716.md`:
  unchanged layers/keep rates/selector and spatial indices, one explicit
  non-spatial context token, exact source-equation replay, source-disjoint
  `7372/1843` gates, precision/TP/tiny-edge/context/XAI/resource checks, and no
  validation/test/sweep/current-command update. Current protocol SHA is
  `d3f8de6f...3ee4b`; it includes the prospective downstream-selector
  correction and exact metadata-only tiny/edge cohort definition
  before code so first-prune identity is exact and second-prune Jaccard is
  measured rather than logically forced to remain identical.
- [x] Implement and precommit the default-off A0 path, focused tests, formal
  engineering/selectivity auditor, overlay review, and a VS Code-safe wrapper.
  Before auditor execution, prospectively define the tiny/edge cohort from
  clean metadata only: linear bbox-area Q25 or normalized edge gap `<=1/16`,
  export membership/hash, and fail on invalid bbox metadata.
  Compileall/pyflakes, launcher configuration preflight, focused `151/151`,
  and full pytest `1296/1296` pass. The first full suite exposed and then
  verified the fix for a default-off deep-prompt prefix regression.
- [x] Run the formal EViT preflight once and correct its audit-order
  contamination with a hash-locked pristine replay. All engineering,
  source-disjoint behavior, deployment, and visual gates then passed and
  authorized exactly one five-epoch scratch pair.
- [x] Run and independently audit the sole control/fusion pair. Clean class1 F1
  rose `+0.017105`, but precision fell `-0.027903`, restricted FP worsened
  `13 -> 16`, bright class1 F1 fell `-0.041418`, and tiny/edge class1 F1/
  precision fell `-0.013285/-0.078947`. Review all 15 XAI pages and reject A0.
- [x] Hash-lock the failed renderer, replay four complete quantitative artifacts
  byte-for-byte after the scoped fix, and finalize visual review as fail at
  summary/prediction/visual SHAs `d7b3d1fe...c76348`/
  `82b41797...a1ff4`/`500c5374...251c4`. Deny validation/test/full train/
  current-command promotion and close nearby EViT fusion sweeps.
- [x] Compact the rejected pair only after recording all four checkpoint hashes.
  Preserve 332 non-binary files at payload SHA `7eed039e...d0ec6`, exclude
  exactly four checkpoints (`348,760,480` bytes), reclaim `332.312 MiB`, and
  pass retention over 728 directories/46 manifests with `blockers=[]` at SHA
  `260a34b2...afaf24`. Keeper and all 15 final XAI pages remain exact.
- [x] Screen accepted primary work and official licensed code outside EViT and
  every closed family. Select CVPR-2025 Token Cropr at official MIT
  commit/tree `fa259e9...61524de6f9`/`4a83993...86851d`, and lock one
  task-supervised routing A0 before implementation at protocol SHA
  `42e912a6...6a847`.
- [x] Implement the default-off two-layer Cropr scorer, matched native/learned
  routing roles, auxiliary loss/telemetry, official-equation replay, config/
  resume/launcher wiring, formal preflight, and VS Code-safe three-phase
  wrappers. Preflight passed `66/66` and authorized exactly one train-only
  five-epoch pair.
- [x] Run the sole source-disjoint `7372/1843` pair with identical natural
  occurrence records and no official validation/test. Preserve pair-manifest
  SHA `09a89cbe...0cc1e` and all four checkpoint hashes before any cleanup.
- [x] Audit all 1,843 holdout rows under clean/dim/bright/low-contrast. Reject
  learned routing: clean class1 F1 rose `+0.021631`, but precision fell
  `-0.024828`, restricted FP worsened `8 -> 10`, harms exceeded corrections
  `13 > 8`, and low-contrast class1 F1 fell `-0.010023`.
- [x] Preserve both failed XAI attempts and replay predictions, selector rows,
  perturbations, and events byte-exactly. Fix the auditor to hook the standard
  deployment forward while retaining the exact locked batch-32 composition;
  final parity is exact on all 57 requests per role.
- [x] Review all 29 hash-locked XAI pages and finalize visual fail. Candidate
  Cropr maps remain broad/high-entropy and mean layer-5 foreground mass falls
  `0.719379 -> 0.640683`. Final summary/visual SHAs are
  `ee2d4456...38c5b`/`1f5c2e57...69dc3`; deny validation/test/full train/
  sweeps/current-command promotion.
- [x] Compact the rejected pair after hash recording. Preserve 332 non-binary
  files at payload SHA `75a54745...a6df77`, exclude four checkpoints
  (`399,569,848` bytes), reclaim `446,504,960` measured bytes, and leave all
  three audit directories plus 29 final pages intact.
- [x] Complete Cropr closure verification. Compileall, both wrapper parses,
  focused `32/32`, full pytest `1332/1332`, protected hashes, and read-only
  retention over 734 directories/47 compaction manifests passed. All 209
  compacted originals are absent and `blockers=[]` at retention SHA
  `95128604...382e9`; final closure SHA is `dd54924d...2b42d`. Commit/push
  only explicit closure paths; current-best commands remain three revisions/
  two updates.
- [x] Cross-check ICLR-2021 NBDT against the accepted OpenReview paper and
  official MIT repository, screen the rejected semantic-attribute route, and
  prospectively lock one fixed domain tree `((0,1),(2,3))|4` before runtime
  implementation or data output.
- [x] Replay the official descendant-logit/path equations and independent
  gradients, HVP, finite differences, FP32/BF16, trace-only parity, ONNX, and
  resource behavior. Keep all prediction metrics on the standard deployment
  forward; never use the attention-return path as behavior evidence.
- [x] Apply the full-fit compatibility stop rule before training. On all 7,372
  fit rows, the flat keeper reached macro/class1 F1 `0.937980/0.809204`, while
  soft NBDT predicted class 4 for every row and reduced class1 TP `422 -> 0`.
  Deny the five-epoch pair and all holdout/validation/test/full-train access.
- [x] Record the unequal-depth diagnosis and remove the uncommitted NBDT
  runtime/trainer/launcher integration plus temporary ONNX artifact. Close
  topology/depth/loss/weight/schedule/inference/threshold variants and retain
  only the prospective protocol and dated closure document.
- [ ] Select the next distinct precision-first route only after a new
  accepted-primary-paper/official-licensed-code no-repeat screen. Require a
  pre-training gate that cannot be satisfied by background localization alone
  and that protects class1 TP as well as restricted false positives.
- [x] Complete the next accepted-primary-paper/official-code screen. Select
  ICML-2022 Sparse Over-Parameterization from the authors' MIT repository at
  commit/tree `4d991ce...18a77f`/`dadbf1b...0c6cd`; reject unlicensed SGN and
  checkpoint-only Label Wave for this precision question. Protocol SHA is
  `d2b94048...b687c8`; secondary/user reports remain hypothesis sources only.
- [ ] Implement and run only the locked full-fit SOP gradient-compatibility
  gate. Require exact official/independent equation replay and prove that SOP
  preserves the corrective class1 logit gradient on all 185 restricted FP and
  the support gradient on all 422 class1 TP. Any failure closes SOP before
  trainer integration, holdout, validation, test, or an image epoch.
