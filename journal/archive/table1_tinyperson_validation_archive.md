# Archive: TinyPerson Validation Benchmark Results (1,764 Tiles)

* **Dataset**: TinyPerson Validation Set (`data/valid`, 131 full-scene images, 1,764 cropped $800 \times 800$ tiles)
* **Backbone**: ResNet-50-FPN
* **Protocol**: Fair-20 / Seed 42

| Method | Backbone | Loss / Assign | $\text{mAP}_{50}$ | $\text{AP}_{50:95}$ | $\text{AP}_{75}$ | $\text{AP}_{\text{micro}}$ | $\text{AP}_{\text{tiny}}$ | $\text{mAP}_{\text{scale}}$ | $\text{AP}_{\text{small}}$ | $\text{AR}_{100}$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Faster R-CNN Baseline | ResNet-50-FPN | IoU / Smooth-L1 | 0.3248 | 0.1136 | 0.0495 | 0.3020 | 0.5320 | 0.5278 | 0.5752 | 0.2531 |
| NWD (NeurIPS'21) | ResNet-50-FPN | NWD / NWD | 0.3260 | 0.1079 | 0.0360 | 0.3533 | 0.6046 | 0.5326 | 0.5348 | 0.2470 |
| IGWD (IEEE TMM'22) | ResNet-50-FPN | IGWD / Loss | 0.3250 | 0.1112 | 0.0478 | 0.2742 | 0.5287 | 0.5231 | 0.5724 | 0.2581 |
| RFLA (ECCV'22) | ResNet-50-FPN | RFLA / Smooth-L1 | 0.3315 | 0.1140 | 0.0510 | 0.3210 | 0.6350 | 0.6380 | 0.5820 | 0.2610 |
| **H-WIoU ($\sigma_0=8\text{px}$, Ours)** | ResNet-50-FPN | **H-WIoU / H-WIoU** | **0.4575** | **0.1560** | **0.0652** | **0.4711** | **0.7723** | **0.7572** | **0.8163** | **0.2850** |
| **H-WIoU ($\sigma_0=6\text{px}$, Ours)** | ResNet-50-FPN | **H-WIoU / H-WIoU** | **0.4618** | **0.1560** | **0.0628** | **0.4682** | **0.7695** | **0.7552** | **0.8140** | **0.2863** |
| **H-WIoU ($\sigma_0=10\text{px}$, Ours)** | ResNet-50-FPN | **H-WIoU / H-WIoU** | **0.4615** | **0.1568** | **0.0658** | **0.4697** | **0.7715** | **0.7542** | **0.8152** | **0.2854** |
