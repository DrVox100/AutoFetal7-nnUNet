AutoFetal-7: 3D Fetal Brain Volumetry & Z-Score Engine
Clinical-grade nnU-Net v2 pipeline for automated 7-class fetal brain segmentation and gestational-age-conditioned Z-score reporting.

Validated for gestational ages 22.6–33.0 weeks. This tool provides automated volumetric analysis and normative comparisons to aid in the quantitative assessment of fetal neurodevelopment.

⚠️ CLINICAL DEPLOYMENT GUARDRAIL

AutoFetal-7 was trained exclusively on GE 1.5T/3T acquisition data (FeTA 2024). Application to non-GE platforms (e.g., Siemens 3T) will result in systematic underestimation of Brainstem and Cerebellum volumes due to cross-scanner domain shift. Do not use Brainstem/Cerebellum Z-scores independently in heterogeneous environments.

1. Installation
Clone the repository and install the strict environment dependencies:

Bash
git clone https://github.com/DrVox100/AutoFetal7-nnUNet.git
cd AutoFetal7-nnUNet
pip install -r requirements.txt

Download Model Weights (Zenodo)
The heavy checkpoint_final.pth weights and nnU-Net plans are hosted securely on Zenodo.

Download Weights Here: https://doi.org/10.5281/zenodo.19398110

Setup: Unzip the downloaded folder and ensure the directory structure remains intact (e.g., Dataset501_FetalBrain/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/).

3. Running Inference
The pipeline is fully vectorized and executes via the inference.py wrapper.

python inference.py \
  --input /path/to/your/raw_nifti_folder \
  --output /path/to/save/results \
  --weights /path/to/unzipped/AutoFetal7_Weights \
  --ga 28.5

 4. Outputs

    For each scan, the engine generates:
    Segmentation Masks: Seven-class .nii.gz NIfTI files.
    Clinical Report: AutoFetal7_Clinical_Report.csv detailing absolute volumes (mm^3) and normative Z-scores derived from Harvard CRL baselines.

 5. Segmented Classes
    
   1 eCSF (Extra-axial Cerebrospinal Fluid)

   2 Cortical Gray Matter

   3 White Matter (including Developmental Zones)

   4 Ventricles and Cavum

   5 Cerebellum

   6 Deep Gray Matter

License
Apache 2.0. Open for clinical research and validation.

    Brainstem
