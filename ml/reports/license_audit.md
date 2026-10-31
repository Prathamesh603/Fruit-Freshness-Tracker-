# License audit

This is an engineering record, not legal advice.

## Summary

| Component | License evidence | Redistribution status |
|---|---|---|
| Mendeley fruit dataset | CC BY 4.0 on the dataset landing page | Allowed with attribution and change indication; raw data is not bundled |
| Keras Applications MobileNetV3Small weights | Keras and TensorFlow are Apache-2.0; Keras Applications documentation states pretrained weights are Apache-licensed | Included as transformed parameters in `freshness.tflite` |
| TensorFlow, LiteRT, NumPy, Pillow, scikit-learn | Apache-2.0 / BSD-family open-source licenses | Used as tooling; no runtime binaries are bundled |
| Existing nine-class candidate | No LICENSE file at commit `200d8dffb48679567777a6ab5adc1d87eeea1c88`; no declared license | Not redistributed and not used in the bundle |
| Kaggle alternative dataset | License not established; credentials unavailable | Not used |
| Project repository | No root `LICENSE` file found | External model-bundle release terms remain unresolved |

## Dataset

Source page: https://data.mendeley.com/datasets/bdd69gyhv8/1

License: Creative Commons Attribution 4.0 International.

Obligations relevant here:

- credit the dataset authors;
- link to the license;
- indicate that the parity images were resized;
- do not imply endorsement.

Attribution is included in `ml/data/DATA_SOURCES.md` and in the bundle's `THIRD_PARTY_NOTICES.md` and `MODEL_CARD.md`. Six test images were resized to 224 x 224 and included in `parity/inputs/` solely for tensor-parity verification.

The source page should be re-checked before any public redistribution because dataset pages can change.

## Backbone

Keras is Apache-2.0:

https://github.com/keras-team/keras/blob/master/LICENSE

The Keras Applications page states that pretrained model weights are released under the Apache License:

https://keras.io/api/applications/

The ImageNet dataset itself is not redistributed. The bundle contains only the transformed model parameters after fine-tuning.

## Existing candidate

Repository: https://github.com/JosephMoustaid/smart-fruit-models-notebooks.git

Commit examined: `200d8dffb48679567777a6ab5adc1d87eeea1c88`

Evidence:

- No `LICENSE`, `LICENSE.md`, or `COPYING` file in the clone.
- No license declaration in the repository root listing.
- The notebook references an external “Fruit Ripeness: Unripe, Ripe, and Rotten” dataset but does not include that dataset or its license.

Conclusion: the candidate model and its weights must not be redistributed in the app or final bundle. They remain local reference material only.

## Kaggle alternative

The Kaggle dataset examined as a fallback, `sriramr/fruits-fresh-and-rotten-for-classification`, had no clear license evidence and required unavailable credentials. It was not downloaded or used.

## Project-level license gap

The repository has no root license file. That means:

- the code and trained model cannot be represented as having a confirmed project-wide license;
- the final bundle is an internal candidate artifact;
- before external publication, the project owner must choose and add a project license and confirm that the model release terms are compatible with the CC BY 4.0 dataset attribution.

The bundle deliberately states this gap rather than inventing a license.

## Bundle files and notices

- `model_exports/mobilenetv3small-sixclass-v1/THIRD_PARTY_NOTICES.md`
- `model_exports/mobilenetv3small-sixclass-v1/MODEL_CARD.md`
- `model_exports/mobilenetv3small-sixclass-v1/SHA256SUMS.txt`
