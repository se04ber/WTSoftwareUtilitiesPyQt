from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton, QLabel, QListWidget, QCheckBox, QHBoxLayout
)
from PyQt5.QtCore import Qt

class SidebarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)

        # Step 1: Data Selection
        self.data_group = QGroupBox("Step 1: Select Data Directory and Files")
        self.data_layout = QVBoxLayout()
        self.data_group.setLayout(self.data_layout)
        self.layout.addWidget(self.data_group)

        self.dir_label = QLabel("No directory selected.")
        self.data_layout.addWidget(self.dir_label)
        self.dir_button = QPushButton("Choose Data Directory")
        self.data_layout.addWidget(self.dir_button)

        self.ambient_label = QLabel("No ambient conditions file selected.")
        self.data_layout.addWidget(self.ambient_label)
        self.ambient_button = QPushButton("Choose Ambient Conditions File (optional)")
        self.data_layout.addWidget(self.ambient_button)
        self.ambient_file = None

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.MultiSelection)
        self.data_layout.addWidget(self.file_list)

        # Step 2: Preprocessing Options (Placeholder)
        self.preprocess_group = QGroupBox("Step 2: Preprocessing Options (coming soon)")
        self.preprocess_layout = QFormLayout()
        self.preprocess_group.setLayout(self.preprocess_layout)
        self.layout.addWidget(self.preprocess_group)
        self.scale_input = QLineEdit("ms")
        self.preprocess_layout.addRow("Scale (ms/fs/nd):", self.scale_input)

        # Step 3: Analysis Selection
        self.analysis_group = QGroupBox("Step 3: Select Analyses/Plots")
        self.analysis_layout = QVBoxLayout()
        self.analysis_group.setLayout(self.analysis_layout)
        self.layout.addWidget(self.analysis_group)
        self.analysis_checks = {}
        self.analysis_options = [
            "Histogram", "Pdf", "Cdf", "Means",
            "ScatterPlot", "PowerDensity", "BoxPlot"
        ]
        for option in self.analysis_options:
            cb = QCheckBox(option)
            self.analysis_layout.addWidget(cb)
            self.analysis_checks[option] = cb

        self.run_button = QPushButton("Run Analysis")
        self.layout.addWidget(self.run_button)

        # Output options
        self.output_group = QGroupBox("Step 5: Output Options")
        self.output_layout = QFormLayout()
        self.output_group.setLayout(self.output_layout)
        self.layout.addWidget(self.output_group)
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("Choose Output Directory")
        output_dir_hbox = QHBoxLayout()
        output_dir_hbox.addWidget(self.output_dir_edit)
        output_dir_hbox.addWidget(self.output_dir_button)
        self.output_layout.addRow("Output Directory:", output_dir_hbox)
        self.output_name_edit = QLineEdit("output")
        self.output_layout.addRow("Base Output Filename:", self.output_name_edit)
        self.save_plots_cb = QCheckBox("Save Plots")
        self.save_data_cb = QCheckBox("Save Postprocessed Data (avg/stats)")
        self.combine_csv_cb = QCheckBox("Combine all saved avg/stats txt files to CSV")
        self.output_layout.addRow(self.save_plots_cb)
        self.output_layout.addRow(self.save_data_cb)
        self.output_layout.addRow(self.combine_csv_cb)

        # Ambient Defaults
        self.ambient_defaults_group = QGroupBox("Default Ambient Conditions (optional)")
        self.ambient_defaults_group.setCheckable(True)
        self.ambient_defaults_group.setChecked(False)
        self.ambient_defaults_layout = QFormLayout()
        self.ambient_defaults_group.setLayout(self.ambient_defaults_layout)
        self.layout.addWidget(self.ambient_defaults_group)
        self.ambient_fields = {}
        ambient_defaults = {
            'x_source': 0,
            'y_source': 0,
            'z_source': 0,
            'x_measure': 1020,
            'y_measure': 0,
            'z_measure': 5,
            'pressure': 101426.04472,
            'temperature': 23,
            'calibration_curve': 1.0,
            'mass_flow_controller': 0.3,
            'calibration_factor': 0,
            'gas_name': 'C12',
            'gas_factor': 0.5,
            'mol_weight': 29.0,
            'scale': 400,
            'scaling_factor': 0.5614882,
            'ref_length': 1/400,
            'ref_height': 100/400,
            'full_scale_wtref': 10,
            'full_scale_flow_rate': 0.002,
            'full_scale_temp': 20,
            'full_scale_pressure': 101325
        }
        for key, default in ambient_defaults.items():
            field = QLineEdit(str(default))
            self.ambient_defaults_layout.addRow(f"{key.replace('_', ' ').capitalize()}: ", field)
            self.ambient_fields[key] = field 