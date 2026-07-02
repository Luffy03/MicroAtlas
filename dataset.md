# MicroAtlas-2B Dataset Details

## Labeled Datasets

The 18 labeled datasets used for supervised training and evaluation, plus 4 CellBinDB sub-datasets used for test-only evaluation.

| Dataset | Imaging method | Annotations | Used for | Images (Train/Test) | Masks (Train/Test) | Storage (GB) |
|---|---|---|---|---|---|---|
| Tissuenet | fluorescence, mass spectrometry | Cell, Nuclei | Train/Test | 2,580 / 1,324 | 988,150 / 145,222 | 11.37 |
| Livecell | phase-contrast | Cell | Train/Test | 3,727 / 1,512 | 1,192,300 / 451,704 | 4.76 |
| Cellpose | mixed | Cell, Nuclei | Train/Test | 796 / 68 | 110,130 / 7,201 | 1.40 |
| DeepBacs | bright-field, fluorescence | Cell | Train/Test | 155 / 35 | 8,303 / 2,731 | 0.27 |
| Omnipose-fluor | fluorescence | Cell | Train/Test | 143 / 75 | 18,616 / 14,587 | 0.23 |
| Omnipose-phc | phase-contrast | Cell | Train/Test | 249 / 148 | 27,624 / 19,570 | 0.45 |
| NeurIPS22 | mixed | Cell, Nuclei | Train/Test | 1,000 / 50 | 168,445 / 6,041 | 4.33 |
| BCCD | bright-field | Cell | Train/Test | 1,098 / 146 | 75,930 / 10,936 | 11.40 |
| MoNuSeg | bright-field (HE) | Nuclei | Train/Test | 37 / 14 | 24,135 / 6,697 | 0.24 |
| MoNuSAC | bright-field (HE) | Nuclei | Train | 209 / 0 | 31,276 / 0 | 0.48 |
| CryNuSeg | bright-field (HE) | Nuclei | Train | 30 / 0 | 8,036 / 0 | 0.05 |
| NuInSeg | bright-field (HE) | Nuclei | Train | 665 / 0 | 30,698 / 0 | 1.09 |
| CPM-TNBC | bright-field (HE) | Nuclei | Train | 112 / 0 | 24,776 / 0 | 0.48 |
| CoNIC | bright-field (HE) | Nuclei | Train | 4,981 / 0 | 569,861 / 0 | 3.05 |
| PanNuke | bright-field (HE) | Nuclei | Train | 7,901 / 0 | 189,744 / 0 | 4.83 |
| LynSec | bright-field (HE, IHC) | Nuclei | Train | 699 / 0 | 151,779 / 0 | 1.71 |
| IHCTMA | bright-field (IHC) | Nuclei | Train | 231 / 0 | 7,490 / 0 | 0.08 |
| YeaZ | bright-field, phase-contrast | Cell | Train | 306 / 0 | 23,046 / 0 | 0.43 |
| CellBinDB-mIF | multiplex immunofluorescence | Cell | Test | 0 / 60 | 0 / 6,013 | 0.005 |
| CellBinDB-ssDNA | single-stranded DNA | Cell | Test | 0 / 276 | 0 / 23,867 | 0.03 |
| CellBinDB-DAPI | fluorescence (DAPI) | Cell | Test | 0 / 303 | 0 / 24,402 | 0.026 |
| CellBinDB-HE | bright-field (HE) | Nuclei | Test | 0 / 405 | 0 / 53,654 | 0.24 |
| **Total** | | | | **24,919 / 4,416** | **3,650,339 / 772,625** | **46.9** |


## Unlabeled IDR Datasets

The 45 datasets curated from the [OpenMicroscopy platform](https://idr.openmicroscopy.org), used for semi-supervised training with pseudo segmentation masks.

| Study | Imaging method | Descriptions | License | Sizes | Images | Masks | Storage (GB) |
|---|---|---|---|---|---|---|---|
| idr0001 | spinning disk confocal | schizosaccharomyces pombe | CC BY 4.0 | 1,040 × 1,376 | 15,520 | 2,245,466 | 599.40 |
| idr0002 | fluorescence | HeLa cells | CC BY 4.0 | 1,024 × 1,344 | 312,503 | 30,739,599 | 1,921.92 |
| idr0003 | fluorescence | saccharomyces cerevisiae | CC BY-NC-SA 3.0 | 512 × 672 | 252,862 | 44,018,801 | 419.67 |
| idr0006 | fluorescence | HeLa cells | CC BY-NC-SA 3.0 | 1,040 × 1,392 | 265,143 | 25,903,682 | 2,033.79 |
| idr0007 | confocal | saccharomyces cerevisiae | CC BY-NC-SA 3.0 | 1,003 × 1,346 | 1,095 | 366,021 | 11.30 |
| idr0008 | fluorescence | drosophila melanogaster cells | CC BY-NC-SA 3.0 | 514 × 660 | 211,590 | 54,318,705 | 448.59 |
| idr0009 | fluorescence | HeLa cells | CC BY 4.0 | 1,024 × 1,344 | 310,589 | 33,230,519 | 1,992.50 |
| idr0010 | fluorescence | U-2-OS cells | CC BY 4.0 | 520 × 695 | 480 | 109,593 | 64.99 |
| idr0011 | fluorescence | saccharomyces cerevisiae | CC BY 4.0 | 512 × 672 | 586 | 47,217 | 14.76 |
| idr0012 | fluorescence | HeLa cells | CC BY-NC-ND 4.0 | 1,375 × 1,410 | 216,082 | 100,985,001 | 3,095.29 |
| idr0016 | fluorescence | U-2-OS cells | CC0 1.0 | 520 × 696 | 261,574 | 28,906,316 | 668.13 |
| idr0017 | fluorescence | human HCT116 cells | CC BY-NC-ND 4.0 | 2,048 × 2,048 | 132,897 | 307,107,876 | 3,226.56 |
| idr0019 | spinning disk confocal | human breast cancer cells | CC BY 4.0 | 501 × 668 | 20,089 | 1,109,102 | 45.83 |
| idr0020 | spinning disk confocal | HeLa cells | CC BY 4.0 | 491 × 623 | 62,476 | 5,661,624 | 119.98 |
| idr0022 | bright-field | TNBC cells | CC BY 4.0 | 3,072 × 4,032 | 34,647 | 2,956,165 | 1,637.64 |
| idr0025 | confocal | human osteosarcoma cells | CC BY-SA 3.0 | 2,048 × 2,048 | 3,290 | 513,776 | 32.49 |
| idr0026 | multi-photon | melanoma cells, CD8+ T cells | CC BY 4.0 | 507 × 507 | 14,265 | 2,569,656 | 158.66 |
| idr0028 | fluorescence, confocal | TNBC cells | CC BY 4.0 | 500 × 667 | 150,985 | 7,221,202 | 296.95 |
| idr0030 | spinning disk confocal | human breast myoepithelial cells | CC BY 4.0 | 498 × 647 | 85,203 | 5,909,948 | 277.54 |
| idr0033 | fluorescence | U-2-OS cells | CC BY 4.0 | 1,080 × 1,080 | 205,141 | 16,740,491 | 1,347.64 |
| idr0034 | fluorescence | induced Pluripotent Stem cells | CC BY 4.0 | 1,024 × 1,360 | 102,151 | 6,577,325 | 412.71 |
| idr0035 | fluorescence | MCF-7 breast cancer cells | CC BY 4.0 | 1,024 × 1,280 | 38,475 | 6,224,592 | 293.34 |
| idr0036 | fluorescence | U-2-OS cells | CC0 1.0 | 520 × 696 | 190,048 | 21,109,062 | 523.66 |
| idr0037 | fluorescence | induced Pluripotent Stem cells | CC BY 4.0 | 1,024 × 1,360 | 55,797 | 3,818,522 | 207.20 |
| idr0043 | bright-field | human tissue cells | CC BY-SA 3.0 | 3,000 × 3,000 | 95,154 | 445,289,885 | 4,388.97 |
| idr0056 | fluorescence | HeLa cells | CC BY 4.0 | 496 × 659 | 177,713 | 26,207,046 | 512.97 |
| idr0069 | fluorescence | human mammary epithelial cells | CC BY-NC 4.0 | 1,024 × 1,360 | 208,544 | 37,444,953 | 1,554.93 |
| idr0071 | fluorescence | human cervical/lung/colorectal cancer cells | CC BY 4.0 | 1,024 × 1,024 | 62,250 | 97,224,342 | 655.27 |
| idr0072 | confocal | murine/human mammary epithelial cells | CC BY 4.0 | 984 × 1,321 | 57,240 | 3,308,942 | 829.51 |
| idr0076 | imaging mass cytometry | human breast cancer cells | CC BY 4.0 | 494 × 502 | 954 | 231,017 | 0.93 |
| idr0078 | fluorescence | saccharomyces cerevisiae | CC BY 4.0 | 1,028 × 1,201 | 255,602 | 83,417,270 | 1,318.27 |
| idr0080 | fluorescence | human A549, ES2, HCC44 cells | CC0 1.0 | 2,160 × 2,160 | 252,762 | 78,205,407 | 6,082.15 |
| idr0088 | spinning disk confocal | human A549, ES2, HCC44 cells | CC BY-NC 4.0 | 1,050 × 1,250 | 213,192 | 95,655,937 | 1,773.71 |
| idr0090 | fluorescence | human red blood cells | CC BY 4.0 | 2,044 × 2,048 | 1,138 | 128,659 | 699.00 |
| idr0093 | spinning disk confocal | HeLa cells | CC BY 4.0 | 2,160 × 2,560 | 70,596 | 36,829,137 | 2,009.29 |
| idr0112 | spinning disk confocal | human motor neurons | CC BY 4.0 | 1,080 × 1,080 | 43,580 | 4,375,248 | 198.63 |
| idr0119 | phase-contrast | breast cancer cells | CC BY 4.0 | 1,040 × 1,408 | 63,342 | 13,753,302 | 323.50 |
| idr0120 | high-content confocal | human T and NK cells | CC BY 4.0 | 1,080 × 1,080 | 153,662 | 5,538,086 | 476.91 |
| idr0123 | fluorescence | human HETR RPE-1 cells | CC BY 4.0 | 1,024 × 1,024 | 525 | 8,133 | 64.93 |
| idr0128 | fluorescence | A549, HeLa Ohio cells | CC BY 4.0 | 2,048 × 2,048 | 28,254 | 237,985,164 | 581.90 |
| idr0129 | fluorescence | A549, HeLa Ohio cells | CC BY 4.0 | 2,048 × 2,048 | 4,437 | 29,395,988 | 82.47 |
| idr0133 | fluorescence | U-2-OS cells | CC BY 4.0 | 2,160 × 2,160 | 33,437 | 5,201,430 | 1,060.74 |
| idr0139 | fluorescence, confocal | TNBC cells, HeLa cells | CC BY 4.0 | 996 × 996 | 193,326 | 38,826,633 | 1,039.46 |
| idr0143 | spinning disk confocal | human HS-5 stromal cells | CC BY 4.0 | 2,160 × 2,160 | 142,437 | 19,142,295 | 2,193.14 |
| idr0160 | fluorescence | human peripheral blood mononuclear cells | CC0 1.0 | 2,160 × 2,160 | 28,293 | 58,316,818 | 876.01 |
| **Total** | | | | | **5,029,926** | **2,024,875,953** | **46,573.24** |
