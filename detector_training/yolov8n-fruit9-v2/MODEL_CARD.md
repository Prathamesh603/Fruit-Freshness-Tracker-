# YOLOv8n nine-fruit detector v2

Detects nine fruit identities and boxes. Freshness is a separate model.
Trained on NVIDIA GTX 1650 CUDA from original COCO YOLOv8n weights, not from the v1 fine-tuned model. Input416, batch8, up to60 epochs, patience12, seed42. See reports/training_config.json for exact settings.

Prepared splits: {'train': 3761, 'test': 584, 'val': 591}. Original v1 splits frozen. Bohni filename families and rotation/reflection pHash components kept together. One representative view per new held-out component. Raw annotations have not all been manually verified. Physical-fruit and crop independence are not established.

Completed epochs: 43. Combined test: 584 images, 1486 boxes.
Metrics: {"metrics/precision(B)": 0.8344303494244768, "metrics/recall(B)": 0.715577324252868, "metrics/mAP50(B)": 0.7901417277557587, "metrics/mAP50-95(B)": 0.6203506410722665, "fitness": 0.6373297497406157}

| Fruit | Test images | Test boxes | AP50 |
|---|---:|---:|---:|
| apple | 238 | 333 | 99.08% |
| banana | 282 | 504 | 99.46% |
| grape | 23 | 59 | 82.27% |
| lemon | 8 | 29 | 26.99% |
| mango | 9 | 26 | 80.99% |
| orange | 253 | 387 | 98.98% |
| pomegranate | 13 | 34 | 88.89% |
| strawberry | 17 | 80 | 70.68% |
| tomato | 15 | 34 | 63.79% |

Original frozen test comparison: {"note": "Same original frozen test set, input416. New Bohni groups touching original holdouts were excluded by family/transform hash checks. Crop/physical-fruit independence is not guaranteed.", "v1": {"metrics/precision(B)": 0.6580508912547004, "metrics/recall(B)": 0.5861563605529482, "metrics/mAP50(B)": 0.5868262464637967, "metrics/mAP50-95(B)": 0.47723581237776336, "fitness": 0.48819485578636673}, "v2": {"metrics/precision(B)": 0.8435219847538786, "metrics/recall(B)": 0.4887060418648414, "metrics/mAP50(B)": 0.5235683855118997, "metrics/mAP50-95(B)": 0.433881633570301, "fitness": 0.44285030876446085}}

Final weights and export: artifacts/. Hashes, labels, size and preprocessing: artifacts/model.json. Parameters: 3012603.
Logs: logs/train.log. Losses and epoch metrics: runs/train/results.csv. Best and latest checkpoints: runs/train/weights/best.pt and last.pt.

This is same-source experimental evaluation. Desktop timings are not Android latency. No Android app was changed. Classic and v1 preserved. Lemon remains sparse; evaluate real camera photos before deployment. Source attribution: datasets/bohni-v13/metadata and datasets/fruit13/sources. Existing source image provenance limitations still apply.

Best validation epoch: 31. Training stopped automatically after 12 epochs without improved validation fitness. Combined-test AP50 is not directly comparable to v1 because the test composition differs. On the unchanged original test, AP50 fell from 58.68% to 52.36%, so v2 is not a universal improvement.
