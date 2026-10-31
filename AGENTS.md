# App edition preservation

The user wants the existing pretrained-model app preserved for demonstrations and comparisons with later apps.

* `apps/01-fruit-lens-classic/` is the preserved **Fruit Lens Classic — Pretrained CPU** edition. Keep its model weights, labels, preprocessing, inference settings, archived APKs, and recorded baseline evidence unchanged unless the user explicitly requests changes to that edition.
* New custom/fine-tuned-model integration must use a separate folder such as `apps/02-fruit-lens-custom/`, launcher name **Fruit Lens Custom**, and Android application ID `com.fruitlens.custom` (or another unique, documented ID). Do not replace Classic with custom weights.
* Classic's ID is `com.fruitlens.classic`; the original prototype ID is `com.fruitlens.app`. Keep application IDs distinct so apps coexist on a phone.
* Source namespace/JNI symbols may remain `com.fruitlens.app` when reusing native code. Changing the application ID does not require changing JNI names. If source namespace changes, update/test JNI deliberately.
* Record model hashes, execution backend (CPU/GPU/NPU), preprocessing, labels, build identity, and actual verification for each new edition. Do not carry over accuracy/performance claims without matched evidence.
* Existing `ml/`, `model_exports/`, and `Training/` work belongs to the training workflow. Do not overwrite it as part of app packaging.
