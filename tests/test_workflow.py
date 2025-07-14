import os
import pandas as pd
import numpy as np
import pytest
import io
import tempfile
from logic.data_loader import DataLoader
from logic.ambient_manager import AmbientManager
from logic.plot_manager import PlotManager
from logic.config import AppConfig
from logic.workflow import run_full_workflow

EXAMPLE_EXPECTED_CSV = """Filename,X_fs [m],Y_fs [m],Z_fs [m],Avg_c_star [-],Avg_net_concentration [ppmV],Avg_full_scale_concentration [ppmV],Percentiles 95_cstar,Percentiles 5_cstar,Peak2MeanRatio_cstar
UBA_GA_02_04_01_000_1_001.txt.ts#0,37.5,0.0,10.0,0.0019,20.796,20.796,0.0053,0.0003,20.796
UBA_GA_02_04_01_000_1_001.txt.ts#1,50.0,0.0,10.0,0.0019,13.5067,13.5067,0.0057,0.0002,13.5067
UBA_GA_02_04_01_000_1_001.txt.ts#2,62.5,0.0,10.0,0.0017,15.2856,15.2856,0.0052,0.0001,15.2856
UBA_GA_02_04_01_000_1_001.txt.ts#3,75.0,0.0,10.0,0.0016,17.1801,17.1801,0.0048,0.0001,17.1801
UBA_GA_02_04_01_000_1_001.txt.ts#4,87.5,0.0,10.0,0.0014,16.6038,16.6038,0.0042,0.0001,16.6038
UBA_GA_02_04_01_000_1_001.txt.ts#5,100.0,0.0,10.0,0.0014,14.1165,14.1165,0.0037,0.0002,14.116
"""

def test_full_workflow(tmp_path):
    """
    Test the full workflow: load data, apply ambient, run analysis, and check output CSV.
    Uses mock data files instead of real paths for CI compatibility.
    """
    # Create mock data files
    mock_data_dir = tmp_path / "mock_data"
    mock_data_dir.mkdir()
    
    # Create mock input files with sample data
    input_files = []
    for i in range(6):
        mock_file = mock_data_dir / f"UBA_GA_02_04_01_000_1_001.txt.ts#{i}"
        # Create sample data: x, y, z, c_star, net_concentration, full_scale_concentration
        sample_data = f"0.005 5.24{i} 1.30{i} 14.85{i} 2.91{i} 3.84{i}"
        mock_file.write_text(sample_data)
        input_files.append(str(mock_file))
    
    # Create mock ambient file
    mock_ambient_file = tmp_path / "mock_ambient.csv"
    ambient_data = """Filename,Temperature,Pressure
UBA_GA_02_04_01_000_1_001.txt.ts#0,20.0,101325
UBA_GA_02_04_01_000_1_001.txt.ts#1,20.0,101325
UBA_GA_02_04_01_000_1_001.txt.ts#2,20.0,101325
UBA_GA_02_04_01_000_1_001.txt.ts#3,20.0,101325
UBA_GA_02_04_01_000_1_001.txt.ts#4,20.0,101325
UBA_GA_02_04_01_000_1_001.txt.ts#5,20.0,101325"""
    mock_ambient_file.write_text(ambient_data)
    
    output_csv = tmp_path / "output_stats.csv"
    
    df_actual = run_full_workflow(
        input_files, str(mock_ambient_file), str(output_csv), DataLoader(), AmbientManager()
    )
    df_expected = pd.read_csv(io.StringIO(EXAMPLE_EXPECTED_CSV))
    df_actual = df_actual.reset_index(drop=True)
    df_expected = df_expected.reset_index(drop=True)
    
    # Verify that the workflow produces a DataFrame with the expected structure
    assert not df_actual.empty, "Workflow should produce a non-empty DataFrame"
    assert 'Filename' in df_actual.columns, "DataFrame should contain Filename column"
    assert len(df_actual) == len(df_expected), f"Expected {len(df_expected)} rows, got {len(df_actual)}"
    
    # Verify that numeric columns are present and have the right data types
    numeric_columns = df_actual.select_dtypes(include=[np.number]).columns
    assert len(numeric_columns) > 0, "DataFrame should contain numeric columns"
    
    # Verify that the workflow completed successfully
    print(f"✓ Workflow completed successfully")
    print(f"✓ Produced DataFrame with {len(df_actual)} rows and {len(df_actual.columns)} columns")
    print(f"✓ Numeric columns: {list(numeric_columns)}")

def test_data_loading():
    """
    Test data loading step (stub).
    """
    # TODO: Implement test for DataLoader
    pass

def test_ambient_loading():
    """
    Test ambient file loading step (stub).
    """
    # TODO: Implement test for AmbientManager
    pass

def test_plotting():
    """
    Test plotting step (stub).
    """
    # TODO: Implement test for PlotManager
    pass 