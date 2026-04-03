# AutoFetal-7: 3D Fetal Brain Volumetry & Z-Score Engine

**Clinical-grade nnU-Net v2 pipeline for automated 7-class fetal brain segmentation and gestational-age-conditioned Z-score reporting.**

Validated for gestational ages 22.6–33.0 weeks. This tool provides automated volumetric analysis and normative comparisons to aid in the quantitative assessment of fetal neurodevelopment.

⚠️ **CLINICAL DEPLOYMENT GUARDRAIL**
> AutoFetal-7 was trained exclusively on GE 1.5T/3T acquisition data (FeTA 2024). Application to non-GE platforms (e.g., Siemens 3T) will result in systematic underestimation of Brainstem and Cerebellum volumes due to cross-scanner domain shift. Do not use Brainstem/Cerebellum Z-scores independently in heterogeneous environments.

## 1. Installation
Clone the repository and install the strict environment dependencies:
```bash
git clone [https://github.com/DrVox100/AutoFetal7-nnUNet.git](https://github.com/YOUR_GITHUB_USERNAME/AutoFetal7-nnUNet.git)
cd AutoFetal7-nnUNet
pip install -r requirements.txt

