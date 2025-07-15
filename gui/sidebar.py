from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton, QLabel, QListWidget, QCheckBox, QHBoxLayout, QScrollArea
)
from PyQt5.QtCore import Qt

class SidebarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create scroll area for better UX
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Main content widget
        content_widget = QWidget()
        self.layout = QVBoxLayout(content_widget)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setSpacing(12)
        self.layout.setContentsMargins(16, 16, 16, 16)
        
        scroll_area.setWidget(content_widget)
        
        # Set up main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)

        # Step 1: Data Selection
        self.data_group = QGroupBox("📁 Step 1: Select Data Directory and Files")
        self.data_group.setToolTip("Choose your data directory and select concentration files to analyze")
        self.data_layout = QVBoxLayout()
        self.data_group.setLayout(self.data_layout)
        self.layout.addWidget(self.data_group)

        # Directory selection
        dir_label = QLabel("Data Directory:")
        dir_label.setStyleSheet("font-weight: 600; color: #0078d4;")
        self.data_layout.addWidget(dir_label)
        
        self.dir_label = QLabel("No directory selected")
        self.dir_label.setStyleSheet("color: #666666; font-style: italic; padding: 4px;")
        self.data_layout.addWidget(self.dir_label)
        
        self.dir_button = QPushButton("📂 Choose Data Directory")
        self.dir_button.setToolTip("Select the folder containing your concentration data files")
        self.data_layout.addWidget(self.dir_button)

        # Ambient conditions
        ambient_label = QLabel("Ambient Conditions File (Optional):")
        ambient_label.setStyleSheet("font-weight: 600; color: #0078d4; margin-top: 8px;")
        self.data_layout.addWidget(ambient_label)
        
        self.ambient_label = QLabel("No ambient conditions file selected")
        self.ambient_label.setStyleSheet("color: #666666; font-style: italic; padding: 4px;")
        self.data_layout.addWidget(self.ambient_label)
        
        self.ambient_button = QPushButton("📄 Choose Ambient Conditions File")
        self.ambient_button.setToolTip("Select CSV file with ambient conditions (optional)")
        self.data_layout.addWidget(self.ambient_button)
        self.ambient_file = None

        # File selection
        file_label = QLabel("Select Files to Analyze:")
        file_label.setStyleSheet("font-weight: 600; color: #0078d4; margin-top: 8px;")
        self.data_layout.addWidget(file_label)
        
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.MultiSelection)
        self.file_list.setToolTip("Select multiple files by holding Ctrl/Cmd while clicking")
        self.file_list.setMinimumHeight(120)
        self.data_layout.addWidget(self.file_list)

        # Step 2: Analysis Selection
        self.analysis_group = QGroupBox("📊 Step 2: Select Analyses/Plots")
        self.analysis_group.setToolTip("Choose which types of plots and analyses to generate")
        self.analysis_layout = QVBoxLayout()
        self.analysis_group.setLayout(self.analysis_layout)
        self.layout.addWidget(self.analysis_group)
        
        # Add select all/none buttons
        select_buttons_layout = QHBoxLayout()
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.setToolTip("Select all plot types")
        self.select_none_button = QPushButton("Select None")
        self.select_none_button.setToolTip("Deselect all plot types")
        select_buttons_layout.addWidget(self.select_all_button)
        select_buttons_layout.addWidget(self.select_none_button)
        self.analysis_layout.addLayout(select_buttons_layout)
        
        self.analysis_checks = {}
        self.analysis_options = [
            ("Histogram", "Distribution of concentration values"),
            ("Pdf", "Probability density function"),
            ("Cdf", "Cumulative distribution function"),
            ("Means", "Mean values with error bars"),
            ("ScatterPlot", "Scatter plot of concentration vs distance"),
            ("PowerDensity", "Power spectral density analysis"),
            ("BoxPlot", "Box plots showing statistics")
        ]
        
        for option, description in self.analysis_options:
            cb = QCheckBox(option)
            cb.setToolTip(description)
            self.analysis_layout.addWidget(cb)
            self.analysis_checks[option] = cb

        # Plot Configuration
        self.plot_config_group = QGroupBox("⚙️ Step 3: Plot Configuration")
        self.plot_config_group.setToolTip("Customize plot appearance and parameters")
        self.plot_config_layout = QVBoxLayout()
        self.plot_config_group.setLayout(self.plot_config_layout)
        self.layout.addWidget(self.plot_config_group)
        
        self.plot_config_button = QPushButton("🔧 Configure Plot Parameters")
        self.plot_config_button.setToolTip("Open plot configuration dialog to customize colors, labels, etc.")
        self.plot_config_layout.addWidget(self.plot_config_button)

        # Run Analysis Button
        self.run_button = QPushButton("🚀 Run Analysis")
        self.run_button.setToolTip("Start the analysis with selected files and plot types")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #107c10;
                color: white;
                font-weight: 600;
                font-size: 11pt;
                padding: 12px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #0e6e0e;
            }
            QPushButton:pressed {
                background-color: #0c5c0c;
            }
        """)
        self.layout.addWidget(self.run_button)

        # Output options
        self.output_group = QGroupBox("💾 Step 4: Output Options")
        self.output_group.setToolTip("Configure where and how to save results")
        self.output_layout = QFormLayout()
        self.output_group.setLayout(self.output_layout)
        self.layout.addWidget(self.output_group)
        
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Leave empty to use current directory")
        self.output_dir_button = QPushButton("📁 Choose Output Directory")
        self.output_dir_button.setToolTip("Select folder to save results")
        output_dir_hbox = QHBoxLayout()
        output_dir_hbox.addWidget(self.output_dir_edit)
        output_dir_hbox.addWidget(self.output_dir_button)
        self.output_layout.addRow("Output Directory:", output_dir_hbox)
        
        self.output_name_edit = QLineEdit("output")
        self.output_name_edit.setToolTip("Base name for output files")
        self.output_layout.addRow("Base Output Filename:", self.output_name_edit)
        
        self.save_plots_cb = QCheckBox("Save Plots as PNG")
        self.save_plots_cb.setToolTip("Save generated plots as PNG image files")
        self.save_data_cb = QCheckBox("Save Postprocessed Data")
        self.save_data_cb.setToolTip("Save processed data as avg/stats text files")
        self.combine_csv_cb = QCheckBox("Combine to CSV")
        self.combine_csv_cb.setToolTip("Combine all saved data files into CSV format")
        
        self.output_layout.addRow(self.save_plots_cb)
        self.output_layout.addRow(self.save_data_cb)
        self.output_layout.addRow(self.combine_csv_cb)

        # Ambient Defaults
        self.ambient_defaults_group = QGroupBox("🔧 Default Ambient Conditions (Optional)")
        self.ambient_defaults_group.setToolTip("Set default ambient conditions if not provided in CSV file")
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
            field.setToolTip(f"Default value for {key.replace('_', ' ')}")
            self.ambient_defaults_layout.addRow(f"{key.replace('_', ' ').capitalize()}: ", field)
            self.ambient_fields[key] = field 