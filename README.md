# AutoFetal-7: 3D Fetal Brain Volumetry & Z-Score Engine

**Translational research nnU-Net v2 pipeline for automated 7-class fetal brain segmentation and gestational-age-conditioned Z-score reporting.**

Validated for gestational ages **22.6–33.0 weeks**. This tool provides automated volumetric analysis and normative comparisons to aid in the quantitative assessment of fetal neurodevelopment.

![AutoFetal-7 Segmentation Sample](Harvard_sample.jpg)

---

## ⚠️ DEPLOYMENT GUARDRAIL
> **DOMAIN SHIFT WARNING:** AutoFetal-7 was trained exclusively on GE 1.5T/3T acquisition data (FeTA 2024). Application to non-GE platforms (e.g., Siemens 3T) will result in systematic underestimation of **Brainstem** (mean Z = -2.91) and **Cerebellum** (mean Z = -2.24) volumes. Do not use these specific Z-scores for clinical decision-making in heterogeneous environments without local scanner calibration.

---

## 1. Installation
Ensure you have `conda` or `pip` installed. Clone the repository and install the strict environment dependencies:

```bash
git clone [https://github.com/DrVox100/AutoFetal7-nnUNet.git](https://github.com/DrVox100/AutoFetal7-nnUNet.git)
cd AutoFetal7-nnUNet
pip install -r requirements.txt
```
2. Download Model Weights (Zenodo)
The trained nnU-Net v2 weights (244 MB) are hosted securely on Zenodo to ensure academic traceability.

Download DOI: https://doi.org/10.5281/zenodo.19398110

Setup: Unzip the folder. The engine expects the standard nnU-Net directory structure: Dataset501_FetalBrain/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/.

3. Running Inference
The pipeline is fully vectorized and executes via the inference.py wrapper. This script handles NIfTI loading, nnU-Net prediction, and automated Z-score calculation against Harvard CRL normative baselines.
```bash
python inference.py \
  --input /path/to/raw_nifti_folder \
  --output /path/to/save/results \
  --weights /path/to/unzipped/AutoFetal7_Weights \
  --ga 28.5
```
4. Segmented Anatomical Classes
The model outputs a multi-label NIfTI mask with the following label indices:

 1.eCSF (Extra-axial Cerebrospinal Fluid)

 2.Cortical Gray Matter

 3.White Matter (including Developmental Zones)

 4.Ventricles and Cavum

 5.Cerebellum

 6.Deep Gray Matter (incl. Ganglionic Eminence)

 7.Brainstem

5. Three-Tier Validation & Performance
The pipeline was validated through a three-tier clinical framework:

A. Internal Segmentation Accuracy (16 held-out Zurich FeTA cases):

Overall Mean DSC: 0.829 ± 0.087

White Matter: 0.907 ± 0.056

Lateral Ventricles: 0.870 ± 0.038

Deep Grey Matter: 0.869 ± 0.050

Cerebellum: 0.868 ± 0.104

Brainstem: 0.835 ± 0.099

Extra-axial CSF: 0.742 ± 0.266

Cortical Grey Matter: 0.711 ± 0.110

B. External Normative Calibration (163 Harvard CRL cases):
Validated for zero-shot domain shift, identifying the GE-to-Siemens scanner bias for the posterior fossa.

C. Pathological Stress Testing (Fidon SBA Atlas):
Validated to correctly identify multi-compartment disease signatures (e.g., massive Z-score elevations in Ventricles) in Spina Bifida Aperta.

6. Citations
If you use this tool in your research, please cite:

nnU-Net: Isensee, F., et al. (2021). Nature Methods.

FeTA Dataset: Payette, K., et al. (2024).

Normative Atlas: Gholipour, A., et al. (2017). Scientific Reports.

Pathological Atlas: Fidon, L., et al. (2021).

7. License
Apache 2.0. Developed by Dr. Abbu J, MD.
