# Independent phone-photo evaluation

This folder is reserved for new phone photos. It currently contains no evaluation
photos. Do not copy public training images or photographs of web search results
here and call them real-world test data.

Collect the supported fruits on different phones, backgrounds, days, and lighting
conditions. Include fresh-looking and visibly spoiled examples, multiple fruits,
occlusion, blur, unsupported fruits, and non-fruit objects. Use fruit/session IDs.
Avoid people and identifying details in the photos.

Record filename, fruit type, visible-condition label, reviewer, physical_fruit_id,
capture_session_id, phone model, lighting, scene, and annotation notes. For detector
evaluation also draw all target fruit boxes. Ambiguous condition is `uncertain`.

Keep this evaluation collection out of training and threshold selection. Once
used to repeatedly tune the model, it becomes development data; gather a new final
test set. No source dataset metric substitutes for this evaluation.
