# AutoFetal-7: 3D Fetal Brain Volumetry & Z-Score Engine

**Clinical-grade nnU-Net v2 pipeline for automated 7-class fetal brain segmentation and gestational-age-conditioned Z-score reporting.**

Validated for gestational ages 22.6–33.0 weeks. This tool provides automated volumetric analysis and normative comparisons to aid in the quantitative assessment of fetal neurodevelopment.

⚠️ **CLINICAL DEPLOYMENT GUARDRAIL**
> AutoFetal-7 was trained exclusively on GE 1.5T/3T acquisition data (FeTA 2024). Application to non-GE platforms (e.g., Siemens 3T) will result in systematic underestimation of Brainstem and Cerebellum volumes due to cross-scanner domain shift. Do not use Brainstem/Cerebellum Z-scores independently in heterogeneous environments.
2. Download Model Weights (Zenodo)
The heavy checkpoint_final.pth weights are hosted securely on Zenodo.

Download Weights Here: https://doi.org/10.5281/zenodo.19398110

Unzip the downloaded folder and note its absolute path on your machine.

3. Running Inference
The pipeline is fully vectorized and runs via the inference.py wrapper.

Bash
python inference.py \
  --input /path/to/your/raw_nifti_folder \
  --output /path/to/save/results \
  --weights /path/to/unzipped/AutoFetal7_Weights \
  --ga 28.5
Outputs
For each scan, the engine generates:

Seven-class .nii.gz segmentation masks.

AutoFetal7_Clinical_Report.csv detailing absolute volumes (mm³) and normative Z-scores derived from Harvard CRL baselines.

License
Apache 2.0. Open for clinical research and validation.
## 1. Installation
Clone the repository and install the strict environment dependencies:
```bash
git clone [https://github.com/DrVox100/AutoFetal7-nnUNet.git](https://github.com/DrVox100/AutoFetal7-nnUNet.git)
cd AutoFetal7-nnUNet
pip install -r requirements.txt

