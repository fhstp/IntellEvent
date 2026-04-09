import os
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import reshape_data, eval_te_data

def deterministic_ops(val):
    os.environ['PYTHONHASHSEED'] = str(val) 
    np.random.seed(val)
    tf.random.set_seed(val)
    tf.keras.utils.set_random_seed(val)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def run_analysis_for_phase(is_ic_phase, datasets, frequency_map):
    phase_label = "IC" if is_ic_phase else "FO"
    final_rows = []

    configs = [
        {"prefix": "IntellEvent", "path": f"../models/SingleCenter_IntellEvent_{phase_label}.keras"},
        {"prefix": "Multicenter", "path": f"../models/MultiCenter_IntellEvent_{phase_label}.keras"}
    ]

    for ds_name in datasets:
        data_path = f"../datasets/{ds_name}.pkl"
        if not os.path.exists(data_path):
            continue

        bl_data = pd.read_pickle(data_path)
        unique_labels = sorted(bl_data['Label'].unique())
        freq = frequency_map[ds_name]
        
        # 20ms threshold converted to frames: (20 / 1000) * freq
        frame_threshold = 0.020 * freq

        # Initialize rows for this dataset
        ds_results = {label: {"Center": ds_name, "Pathology": label} for label in unique_labels}

        for cfg in configs:
            if not os.path.exists(cfg['path']):
                continue
            
            model = tf.keras.models.load_model(cfg['path'])
            test_velocities, test_grf = reshape_data(bl_data['Velocity'], bl_data['GRF_Events'])
            preds = model.predict(test_velocities, batch_size=100, verbose=0)
            
            mae_all, error_list, _, _ = eval_te_data(
                bl_data['Label'], bl_data['DBid'], bl_data['Trial'], 
                preds, test_grf, is_ic_phase
            )

            for i, label_val in enumerate(unique_labels):
                errors = np.abs(error_list[i])
                total_events = len(errors)
                
                # Calculation: How many errors are within 20ms (frame_threshold)
                hits = np.sum(errors <= frame_threshold)
                dr_percent = (hits / total_events * 100) if total_events > 0 else 0
                
                # Convert frame-based MAE/STD to milliseconds for the table
                ms_mae = (mae_all[i] / freq) * 1000
                ms_std = (np.std(errors) / freq) * 1000

                # Store as individual numeric columns for future calculations
                ds_results[label_val]["No_Events"] = total_events
                ds_results[label_val][f"{cfg['prefix']}_MAE_ms"] = round(ms_mae, 2)
                ds_results[label_val][f"{cfg['prefix']}_STD_ms"] = round(ms_std, 2)
                ds_results[label_val][f"{cfg['prefix']}_DR_percent"] = round(dr_percent, 2)

        final_rows.extend(ds_results.values())

    return pd.DataFrame(final_rows)

def main():
    deterministic_ops(21)
    
    datasets = ["BL1", "BL2", "BL3", "BL4", "BL5", "BL6"]
    # Frequency mapping as provided
    frequency_map = {
        "BL1": 100, "BL2": 100, "BL3": 150, 
        "BL4": 100, "BL5": 120, "BL6": 120
    }

    print("Processing Initial Contact (IC)...")
    df_ic = run_analysis_for_phase(True, datasets, frequency_map)
    
    print("Processing Foot Off (FO)...")
    df_fo = run_analysis_for_phase(False, datasets, frequency_map)

    # Save to Excel with separate sheets for clean numeric access
    with pd.ExcelWriter("benchmark_laboratory_results.xlsx") as writer:
        df_ic.to_excel(writer, sheet_name="IC_Results", index=False)
        df_fo.to_excel(writer, sheet_name="FO_Results", index=False)

    print("\nSuccess! Results saved to 'benchmark_laboratory_results.xlsx'.")

if __name__ == "__main__":
    main()