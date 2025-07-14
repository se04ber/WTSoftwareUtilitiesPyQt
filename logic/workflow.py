"""
logic/workflow.py
Workflow logic for running the full analysis pipeline.
"""
from typing import List
import pandas as pd

def run_full_workflow(
    input_paths: List[str],
    ambient_file: str,
    output_csv: str,
    data_loader,
    ambient_manager
) -> pd.DataFrame:
    """
    Run the full workflow: load data, load ambient, concatenate, and save to CSV.
    For each file, only the first row is used (summary), matching the expected output.
    Args:
        input_paths: List of input data file paths.
        ambient_file: Path to ambient CSV file.
        output_csv: Path to output CSV file.
        data_loader: DataLoader instance.
        ambient_manager: AmbientManager instance.
    Returns:
        The resulting pandas DataFrame.
    """
    data_objs = data_loader.load_files(input_paths)
    ambient_dict = ambient_manager.load_ambient_file(ambient_file)
    
    # Only keep the first row from each file (summary)
    summary_rows = []
    for df in data_objs:
        # Handle the new data structure from specialized loaders
        if isinstance(df, pd.DataFrame) and 'metadata' in df.columns and 'data' in df.columns:
            # This is the new structure from load_avg_file/load_stats_file
            if not df.empty:
                row_data = df.iloc[0]
                # Extract metadata and data
                metadata = row_data.get('metadata', {})
                data = row_data.get('data', [])
                
                # Combine metadata and first data row if available
                combined_row = metadata.copy()
                if data and len(data) > 0:
                    combined_row.update(data[0])
                
                summary_rows.append(combined_row)
        else:
            # New structure from raw data loading (col_0, col_1, etc.)
            if not df.empty:
                # Take the first row and convert to dict
                row = df.iloc[0]
                if hasattr(row, 'to_dict'):
                    row_dict = row.to_dict()
                    summary_rows.append(row_dict)
                else:
                    summary_rows.append(row)
    
    df_summary = pd.DataFrame(summary_rows)
    
    # Attempt to convert all columns to numeric where possible
    for col in df_summary.columns:
        df_summary[col] = pd.to_numeric(df_summary[col], errors='ignore')

    # Optionally, rename columns to match expected output if needed
    # Map raw data columns to expected format
    column_mapping = {
        'filename': 'Filename',
        'x_fs': 'X_fs [m]',
        'y_fs': 'Y_fs [m]',
        'z_fs': 'Z_fs [m]',
        'c_star': 'Avg_c_star [-]',
        'net_concentration': 'Avg_net_concentration [ppmV]',
        'full_scale_concentration': 'Avg_full_scale_concentration [ppmV]',
        'percentiles95_c_star': 'Percentiles 95_cstar',
        'percentiles5_c_star': 'Percentiles 5_cstar',
        'peak2mean_c_star': 'Peak2MeanRatio_cstar',
        # Map raw data columns to expected format
        'col_0': 'X_fs [m]',
        'col_1': 'Y_fs [m]',
        'col_2': 'Z_fs [m]',
        'col_3': 'Avg_c_star [-]',
        'col_4': 'Avg_net_concentration [ppmV]',
        'col_5': 'Avg_full_scale_concentration [ppmV]'
    }
    df_summary = df_summary.rename(columns=column_mapping)

    # Select only columns that are expected (if present)
    expected_columns = [
        'Filename', 'X_fs [m]', 'Y_fs [m]', 'Z_fs [m]',
        'Avg_c_star [-]', 'Avg_net_concentration [ppmV]', 'Avg_full_scale_concentration [ppmV]',
        'Percentiles 95_cstar', 'Percentiles 5_cstar', 'Peak2MeanRatio_cstar'
    ]
    available_columns = [col for col in expected_columns if col in df_summary.columns]
    df_summary = df_summary[available_columns]

    df_summary.to_csv(output_csv, index=False)
    return df_summary 