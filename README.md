# Cancer Type Classification Using RNA-Seq Gene Expression Data

**Author:** Jessica Rubí Lara Rosales  
**Contact:** [jessica.lara@cimat.mx](mailto:jessica.lara@cimat.mx)  
**Institution:** Center for Research in Mathematics (CIMAT)

---

## Project Overview

Large-scale genomic profiling enables the identification of molecular signatures that distinguish different tumor types. Using RNA-Seq gene expression data from [The Cancer Genome Atlas (TCGA)](https://portal.gdc.cancer.gov/) [Pan-Cancer Atlas](https://www.cell.com/pb-assets/consortium/pancanceratlas/pancani3/index.html) (20,502 genes across 11,568 tumor samples spanning 33 cancer types), this project investigates how effectively statistical and machine learning models can classify tumor types and uncover key biomarker genes.

## Objectives & Methodology

The main goal of this work is to evaluate supervised classification methods on high-dimensional gene expression profiles and identify the genes that contribute most to distinguishing between cancer types. Specifically, the project covers:

* **Dimensionality Reduction:** Applying dimensionality reduction techniques (such as Principal Component Analysis) to manage the high-dimensional gene space and extract informative representations across tumor cohorts.
* **Supervised Classification:** Training, evaluating, and comparing classical supervised learning classifiers to accurately predict the 33 tumor types from RNA-Seq profiles.
* **Feature Importance Analysis:** Quantifying variable importance to pinpoint the specific genes with the greatest weight in separating distinct tumor types.

## Repository Contents

* **`report/` (PDF):** Technical project report covering the background, methodology, classification performance, and analysis of key discriminant genes.
* **`code/`:** Scripts and notebooks implementing data preprocessing, dimensionality reduction, supervised classification models, and feature importance evaluation.
* **`presentation/`:** Slides summarizing the workflow and main findings of the project.

## References

1. Hoadley, K. A., Yau, C., Hinoue, T., Wolf, D. M., Lazar, A. J., Drill, E., ... & Laird, P. W. (2018). *Cell-of-Origin Patterns Dominate the Molecular Classification of 10,000 Tumors from 33 Types of Cancer*. **Cell**, 173(2), 291–304.
