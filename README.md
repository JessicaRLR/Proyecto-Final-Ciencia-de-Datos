# Cancer Type Classification Using RNA-Seq Gene Expression Data

**Author:** Jessica Rubí Lara Rosales  
**Contact:** [jessica.lara@cimat.mx](mailto:jessica.lara@cimat.mx)  
**Institution:** Center for Research in Mathematics (CIMAT)  
**Course:** Data Science  
**Instructor:** Dr. Marco Antonio Aquino López

## Project Overview
This repository contains the final project for a Data Science course at CIMAT, inspired by the findings of Hoadley et al. (2018).
The work focuses on supervised classification using a single molecular layer: patient RNA-Seq gene expression profiles (11,568 tumor samples and 20,502 genes across 33 cancer types from [The Cancer Genome Atlas (TCGA)](https://portal.gdc.cancer.gov/)). Specifically, it evaluates how effectively classical supervised machine learning models can recover the 33 known tumor types from gene expression data.

## Objectives & Methodology

- **Dimensionality Reduction:** Applying techniques such as Principal Component Analysis (PCA) to handle the high-dimensional gene expression space (20,502 features) and extract informative representations across tumor cohorts.
- **Supervised Classification:** Training, evaluating, and comparing standard supervised learning classifiers to predict the 33 cancer types from each patient's RNA-Seq profile.
- **Feature Importance Analysis:** Quantifying variable importance to identify which specific genes contribute most to distinguishing between distinct tumor types.

## Repository Contents

- **`Proyecto_Final_CdD.pdf`:** Technical project report detailing the background, methodology, classification performance metrics, and analysis of key discriminant genes.
- **`ProyectoCdD_Final.py`:** Scripts implementing data preprocessing, dimensionality reduction, supervised classification models, and feature importance evaluation.
- **`presentacion.pptx`:** Slides summarizing the workflow and main results of the project.

## References

1. Hoadley, K. A., Yau, C., Hinoue, T., Wolf, D. M., Lazar, A. J., Drill, E., ... & Laird, P. W. (2018). *Cell-of-Origin Patterns Dominate the Molecular Classification of 10,000 Tumors from 33 Types of Cancer*. **Cell**, 173(2), 291–304.
