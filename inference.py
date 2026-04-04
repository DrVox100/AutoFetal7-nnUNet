"""
AutoFetal-7: 3D Fetal Brain Volumetry & Z-Score Engine
Clinical Inference Monolith v1.0
"""

import os
import sys
import argparse
import logging
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

# Elite Server Guardrail: Prevents GUI crashes on headless cloud instances (RunPod/Colab)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure Elite Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [AutoFetal-7] - %(levelname)s - %(message)s'
)

class AutoFetalInference:
    """
    Translational-grade pipeline for 7-class fetal brain volumetry and visual Z-score reporting.
    """
    def __init__(self, input_dir: str, output_dir: str, weights_dir: str):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.weights_dir = Path(weights_dir).resolve()
        
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

    def pre_flight_io_check(self):
        """Validates that external users formatted their input files correctly."""
        logging.info("Running pre-flight I/O validation...")
        input_files = list(self.input_dir.glob("*.nii*"))
        
        if not input_files:
            logging.error(f"CRITICAL: No NIfTI files found in {self.input_dir}")
            sys.exit(1)
            
        for f in input_files:
            if not f.name.endswith("_0000.nii.gz"):
                logging.error(f"CRITICAL FORMAT ERROR: File '{f.name}' is invalid.")
                logging.error("nnU-Net strictly requires all input files to end with '_0000.nii.gz'.")
                logging.error(f"Please rename '{f.name}' to '{f.name.split('.')[0]}_0000.nii.gz' and rerun.")
                sys.exit(1)
        logging.info("Pre-flight check passed. All files formatted correctly.")

    def run_nnunet_prediction(self):
        """Executes the nnU-Net v2 prediction via subprocess with isolated environment."""
        logging.info("Initiating nnU-Net v2 Inference Engine...")
        
        # Inject the weights path into the environment dynamically
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
            logging.info("3D Segmentation masks generated successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"nnU-Net prediction failed: {e.stderr.decode('utf-8')}")
            sys.exit(1)

    def generate_clinical_visual_report(self, patient_data: dict, patient_id: str, ga_weeks: float):
        """
        Generates a clinical-grade dashboard plotting the patient's volumetrics 
        against the Harvard CRL normative curves.
        """
        logging.info(f"Generating visual normative curve report for {patient_id}...")
        
        # Generate normative X-axis (Gestational Age range 20 to 38 weeks)
        ga_range = np.linspace(20, 38, 100)
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f"AutoFetal-7 Volumetric Analysis | Patient: {patient_id} | GA: {ga_weeks} Weeks", fontsize=18, fontweight='bold')
        axes = axes.flatten()
        
        for i, (class_idx, class_name) in enumerate(self.class_map.items()):
            ax = axes[i]
            
            # Calculate the normative curve for this specific class
            normative_volumes = [self._get_expected_volume(class_name, ga) for ga in ga_range]
            
            # Plot the expected normative curve
            ax.plot(ga_range, normative_volumes, color='blue', linestyle='--', label='Expected Volume (CRL)')
            
            # Extract and plot the patient's actual volume
            patient_vol = patient_data.get(f"{class_name}_Vol_mm3", 0)
            patient_z = patient_data.get(f"{class_name}_Z_Score", 0)
            
            # Determine dot color based on severity (Z-score > 2 or < -2 is abnormal)
            marker_color = 'red' if patient_z is not None and abs(patient_z) > 2.0 else 'green'
            
            ax.scatter(ga_weeks, patient_vol, color=marker_color, s=150, zorder=5, label='Patient Actual')
            
            # Formatting
            ax.set_title(f"{class_name.replace('_', ' ')}\nZ-Score: {patient_z}", fontsize=12)
            ax.set_xlabel("Gestational Age (Weeks)")
            ax.set_ylabel("Volume (mm³)")
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()
            
        # Clear empty subplots
        for j in range(7, 9):
            fig.delaxes(axes[j])
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save to output directory
        report_path = self.output_dir / f"{patient_id}_Visual_Report.png"
        plt.savefig(report_path, dpi=300)
        plt.close(fig)

    def _get_expected_volume(self, class_name: str, ga: float) -> float:
        """
        Calculates expected normative volume (mm³) based on Harvard CRL quadratic fits.
        Equation: V(t) = at^2 + bt + c
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

    def _get_dynamic_std(self, class_name: str, expected_volume_mm3: float) -> float:
        """
        Calculates standard deviation dynamically using the Coefficient of Variation (CV).
        Ensures variance scales appropriately with gestational age.
        """
        cv_map = {
            "eCSF": 0.15,
            "Cortical_Gray_Matter": 0.12,
            "White_Matter": 0.10,
            "Ventricles_and_Cavum": 0.18, 
            "Cerebellum": 0.08,           
            "Deep_Gray_Matter": 0.09,
            "Brainstem": 0.07             
        }
        cv = cv_map.get(class_name, 0.10)
        return expected_volume_mm3 * cv

    def extract_volumes_and_zscores(self, ga_weeks: float):
        """
        Parses NIfTI masks, calculates volumes in mm³, computes dynamic Z-scores,
        and triggers the visual report generation for each patient.
        """
        logging.info(f"Computing volumetrics and Z-scores for GA: {ga_weeks} weeks...")
        results = []
        
        for mask_file in self.output_dir.glob("*.nii.gz"):
            patient_id = mask_file.name.replace(".nii.gz", "")
            
            try:
                # Load NIfTI safely
                img = nib.load(mask_file)
                data = img.get_fdata()
                
                # Get voxel volume in mm³ to map physical geometry
                header = img.header
                zooms = header.get_zooms()
                voxel_vol_mm3 = float(np.prod(zooms[:3]))
                
                patient_data = {"Patient_ID": patient_id, "GA_Weeks": ga_weeks}
                
                for class_idx, class_name in self.class_map.items():
                    # 1. Calculate Raw Volume
                    voxel_count = np.sum(data == class_idx)
                    vol_mm3 = float(voxel_count * voxel_vol_mm3)
                    patient_data[f"{class_name}_Vol_mm3"] = round(vol_mm3, 2)
                    
                    # 2. Compute Expected Volume using Harvard CRL Equations
                    expected_vol = self._get_expected_volume(class_name, ga_weeks) 
                    
                    # 3. Compute Dynamic Standard Deviation
                    dynamic_std = self._get_dynamic_std(class_name, expected_vol)
                    
                    # 4. Calculate Z-Score
                    if dynamic_std > 0:
                        z_score = (vol_mm3 - expected_vol) / dynamic_std
                        patient_data[f"{class_name}_Z_Score"] = round(z_score, 2)
                    else:
                        patient_data[f"{class_name}_Z_Score"] = None
                        
                results.append(patient_data)
                
                # Generate the visual dashboard for this specific patient
                self.generate_clinical_visual_report(patient_data, patient_id, ga_weeks)
                
            except Exception as e:
                logging.error(f"Failed to process mask {mask_file.name}: {str(e)}")
            
        # Save aggregated clinical CSV report
        if results:
            df = pd.DataFrame(results)
            report_path = self.output_dir / "AutoFetal7_Clinical_Aggregate_Report.csv"
            df.to_csv(report_path, index=False)
            logging.info(f"Clinical aggregate CSV generated: {report_path}")
        else:
            logging.error("No valid NIfTI masks processed. Report generation failed.")

    def execute(self, ga_weeks: float):
        """Master execution block."""
        self.pre_flight_io_check()
        self.clinical_safety_check()
        self.run_nnunet_prediction()
        self.extract_volumes_and_zscores(ga_weeks=ga_weeks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoFetal-7: 3D Fetal Brain Volumetry & Z-Score Visualizer")
    parser.add_argument("--input", type=str, required=True, help="Directory containing raw NIfTI scans (must end in _0000.nii.gz)")
    parser.add_argument("--output", type=str, required=True, help="Directory to save segmentation masks, CSVs, and PNG reports")
    parser.add_argument("--weights", type=str, required=True, help="Root directory containing Dataset501_FetalBrain")
    parser.add_argument("--ga", type=float, required=True, help="Gestational Age in weeks (e.g., 28.5)")
    
    args = parser.parse_args()
    
    pipeline = AutoFetalInference(
        input_dir=args.input,
        output_dir=args.output,
        weights_dir=args.weights
    )
    
    pipeline.execute(ga_weeks=args.ga)
