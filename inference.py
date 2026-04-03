import os
import sys
import argparse
import logging
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

# Configure Elite Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [AutoFetal-7] - %(levelname)s - %(message)s'
)

class AutoFetalInference:
    """
    Translational-grade pipeline for 7-class fetal brain volumetry and Z-score reporting.
    """
    def __init__(self, input_dir: Path, output_dir: Path, weights_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.weights_dir = Path(weights_dir)
        
        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Class mapping based on FeTA 2024
        self.class_map = {
            1: "eCSF",
            2: "Cortical_Gray_Matter",
            3: "White_Matter",
            4: "Ventricles_and_Cavum",
            5: "Cerebellum",
            6: "Deep_Gray_Matter",
            7: "Brainstem"
        }

    def clinical_safety_check(self):
        """
        Mandatory domain-shift warning as documented in the EPNC 2026 manuscript.
        """
        logging.warning("="*75)
        logging.warning("CLINICAL DEPLOYMENT GUARDRAIL TRIGGERED")
        logging.warning("AutoFetal-7 was trained exclusively on GE 1.5T/3T acquisition data (FeTA 2024).")
        logging.warning("Application to non-GE platforms (e.g., Siemens 3T) will result in systematic ")
        logging.warning("underestimation of Brainstem and Cerebellum volumes due to domain shift.")
        logging.warning("Do not use Brainstem/Cerebellum Z-scores independently in heterogeneous environments.")
        logging.warning("="*75)

    def run_nnunet_prediction(self):
        """Executes the nnU-Net v2 prediction via subprocess."""
        logging.info("Initiating nnU-Net v2 Inference...")
        
        # THIS IS THE FIX: Tell nnU-Net exactly where your Zenodo weights are
        custom_env = os.environ.copy()
        custom_env["nnUNet_results"] = str(self.weights_dir)
        
        cmd = [
            "nnUNetv2_predict",
            "-i", str(self.input_dir),
            "-o", str(self.output_dir),
            "-d", "501",  # FeTA Dataset ID
            "-c", "3d_fullres",
            "-f", "0"     # Fold 0
        ]
        
        try:
            subprocess.run(cmd, env=custom_env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.info("Segmentation masks generated successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"nnU-Net prediction failed: {e.stderr.decode('utf-8')}")
            sys.exit(1)

    def extract_volumes_and_zscores(self, ga_weeks: float):
        """
        Parses the generated NIfTI masks, calculates volumes in mm^3, 
        and computes normative Z-scores based on Harvard CRL fits.
        """
        logging.info(f"Computing volumetrics and Z-scores for GA: {ga_weeks} weeks...")
        results = []
        
        for mask_file in self.output_dir.glob("*.nii.gz"):
            patient_id = mask_file.name.replace(".nii.gz", "")
            
            # Load NIfTI safely
            img = nib.load(mask_file)
            data = img.get_fdata()
            
            # Get voxel volume in mm^3
            header = img.header
            zooms = header.get_zooms()
            voxel_vol_mm3 = np.prod(zooms[:3])
            
            patient_data = {"Patient_ID": patient_id, "GA_Weeks": ga_weeks}
            
            for class_idx, class_name in self.class_map.items():
                # 1. Calculate Raw Volume
                voxel_count = np.sum(data == class_idx)
                vol_mm3 = voxel_count * voxel_vol_mm3
                patient_data[f"{class_name}_Vol_mm3"] = round(vol_mm3, 2)
                
                # 2. Compute Z-Score using Harvard CRL Equations
                expected_vol = self._get_expected_volume(class_name, ga_weeks) 
                std_dev = self._get_class_std(class_name)
                
                if std_dev > 0:
                    z_score = (vol_mm3 - expected_vol) / std_dev
                    patient_data[f"{class_name}_Z_Score"] = round(z_score, 2)
                else:
                    patient_data[f"{class_name}_Z_Score"] = None
                    
            results.append(patient_data)
            
        # Save clinical report
        if results:
            df = pd.DataFrame(results)
            report_path = self.output_dir / "AutoFetal7_Clinical_Report.csv"
            df.to_csv(report_path, index=False)
            logging.info(f"Clinical report generated: {report_path}")
        else:
            logging.error("No NIfTI masks found to process. Report generation failed.")

    def _get_expected_volume(self, class_name: str, ga: float) -> float:
        """
        Calculates expected normative volume (mm^3) based on Harvard CRL quadratic fits.
        """
        if class_name == "eCSF":
            return (-340.27 * (ga**2)) + (24558.44 * ga) - 344143.06
        elif class_name == "Cortical_Gray_Matter":
            return (308.44 * (ga**2)) - (12623.12 * ga) + 140382.44
        elif class_name == "White_Matter":
            return (10.75 * (ga**2)) + (8679.87 * ga) - 165151.04
        elif class_name == "Ventricles_and_Cavum":
            return (-21.85 * (ga**2)) + (1470.58 * ga) - 18449.31
        elif class_name == "Cerebellum":
            return (49.81 * (ga**2)) - (1861.91 * ga) + 18470.62
        elif class_name == "Deep_Gray_Matter":
            return (20.11 * (ga**2)) - (219.01 * ga) - 876.08
        elif class_name == "Brainstem":
            return (1.32 * (ga**2)) + (292.04 * ga) - 4883.74
        else:
            return 0.0

    def _get_class_std(self, class_name: str) -> float:
        """
        Returns the standard deviation (sigma) of the residuals from the Harvard CRL fits.
        """
        std_map = {
            "eCSF": 3821.40,
            "Cortical_Gray_Matter": 1116.21,
            "White_Matter": 4224.27,
            "Ventricles_and_Cavum": 193.51,
            "Cerebellum": 123.78,
            "Deep_Gray_Matter": 364.52,
            "Brainstem": 69.26
        }
        return std_map.get(class_name, 1.0) 

    def execute(self, ga_weeks: float):
        self.clinical_safety_check()
        self.run_nnunet_prediction()
        self.extract_volumes_and_zscores(ga_weeks=ga_weeks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoFetal-7: 3D Fetal Brain Volumetry & Z-Score Engine")
    parser.add_argument("--input", type=str, required=True, help="Directory containing raw NIfTI scans (.nii.gz)")
    parser.add_argument("--output", type=str, required=True, help="Directory to save segmentation masks and CSV reports")
    parser.add_argument("--weights", type=str, required=True, help="Directory containing the nnU-Net .pth weights")
    parser.add_argument("--ga", type=float, required=True, help="Gestational Age in weeks (e.g., 28.5)")
    
    args = parser.parse_args()
    
    pipeline = AutoFetalInference(
        input_dir=args.input,
        output_dir=args.output,
        weights_dir=args.weights
    )
    
    pipeline.execute(ga_weeks=args.ga)
