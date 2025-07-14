import sys
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QListWidget, QListWidgetItem, QCheckBox, QTabWidget, QGroupBox, QFormLayout, QLineEdit, QMessageBox,
    QSplitter, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import importlib.util
import os
import csv
from windtunnel.concentration.utils import combine_to_csv

# Try to import PointConcentration and create_pdf
try:
    from windtunnel.concentration.PointConcentration import PointConcentration
    from windtunnel.concentration.CompareDatasets import (
        create_pdf, create_histogram, create_cdf, create_means, create_quantilplot,
        create_scatterplot, create_residualplot, create_autocorrelation, powerDensityPlot, create_boxplot
    )
except ImportError:
    PointConcentration = None
    create_pdf = create_histogram = create_cdf = create_means = create_quantilplot = None
    create_scatterplot = create_residualplot = create_autocorrelation = powerDensityPlot = create_boxplot = None

class ConcentrationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Concentration Analysis GUI")
        self.setGeometry(100, 100, 900, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.addWidget(self.splitter)

        # Sidebar widget for controls
        self.sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)
        self.sidebar_layout.setAlignment(Qt.AlignTop)

        # Step 1: Data Selection
        self.data_group = QGroupBox("Step 1: Select Data Directory and Files")
        self.data_layout = QVBoxLayout()
        self.data_group.setLayout(self.data_layout)
        self.sidebar_layout.addWidget(self.data_group)

        self.dir_label = QLabel("No directory selected.")
        self.data_layout.addWidget(self.dir_label)
        self.dir_button = QPushButton("Choose Data Directory")
        self.dir_button.clicked.connect(self.choose_directory)
        self.data_layout.addWidget(self.dir_button)

        # Ambient conditions file picker
        self.ambient_label = QLabel("No ambient conditions file selected.")
        self.data_layout.addWidget(self.ambient_label)
        self.ambient_button = QPushButton("Choose Ambient Conditions File (optional)")
        self.ambient_button.clicked.connect(self.choose_ambient_file)
        self.data_layout.addWidget(self.ambient_button)
        self.ambient_file = None

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.MultiSelection)
        self.data_layout.addWidget(self.file_list)

        # Step 2: Preprocessing Options (Placeholder)
        self.preprocess_group = QGroupBox("Step 2: Preprocessing Options (coming soon)")
        self.preprocess_layout = QFormLayout()
        self.preprocess_group.setLayout(self.preprocess_layout)
        self.sidebar_layout.addWidget(self.preprocess_group)
        # Example: Model scale/full scale/ND
        self.scale_input = QLineEdit("ms")
        self.preprocess_layout.addRow("Scale (ms/fs/nd):", self.scale_input)

        # Step 3: Analysis Selection
        self.analysis_group = QGroupBox("Step 3: Select Analyses/Plots")
        self.analysis_layout = QVBoxLayout()
        self.analysis_group.setLayout(self.analysis_layout)
        self.sidebar_layout.addWidget(self.analysis_group)

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
        self.run_button.clicked.connect(self.run_analysis)
        self.run_button.setMinimumHeight(40)
        font = self.run_button.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        self.run_button.setFont(font)
        self.sidebar_layout.addWidget(self.run_button)

        # Step 4: Results/Visualization (Placeholder)
        self.results_tabs = QTabWidget()
        self.results_tabs.setTabsClosable(True)
        self.results_tabs.tabCloseRequested.connect(self.close_tab)
        self.results_tabs.setMinimumWidth(400)
        self.sidebar_layout.addWidget(self.results_tabs)

        # Output options
        self.output_group = QGroupBox("Step 5: Output Options")
        self.output_layout = QFormLayout()
        self.output_group.setLayout(self.output_layout)
        self.sidebar_layout.addWidget(self.output_group)

        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("Choose Output Directory")
        self.output_dir_button.clicked.connect(self.choose_output_directory)
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

        self.output_dir = None

        # Data
        self.data_dir = None
        self.selected_files = []

        # --- Default Ambient Conditions Dropdown ---
        self.ambient_defaults_group = QGroupBox("Default Ambient Conditions (optional)")
        self.ambient_defaults_group.setCheckable(True)
        self.ambient_defaults_group.setChecked(False)
        self.ambient_defaults_layout = QFormLayout()
        self.ambient_defaults_group.setLayout(self.ambient_defaults_layout)
        self.sidebar_layout.addWidget(self.ambient_defaults_group)

        # Add fields for all relevant parameters
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

        # Make sidebar scrollable
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.sidebar_widget)
        self.splitter.addWidget(self.scroll_area)
        self.splitter.addWidget(self.results_tabs)
        self.splitter.setSizes([350, 900])

    def choose_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if dir_path:
            self.data_dir = dir_path
            self.dir_label.setText(f"Selected: {dir_path}")
            self.populate_file_list()

    def populate_file_list(self):
        self.file_list.clear()
        if self.data_dir:
            files = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]
            for f in files:
                item = QListWidgetItem(f)
                self.file_list.addItem(item)

    def choose_ambient_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Ambient Conditions CSV File", filter="CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.ambient_file = file_path
            self.ambient_label.setText(f"Selected: {file_path}")
        else:
            self.ambient_file = None
            self.ambient_label.setText("No ambient conditions file selected.")

    def choose_output_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_edit.setText(dir_path)

    def run_analysis(self):
        self.selected_files = [item.text() for item in self.file_list.selectedItems()]
        print(f"[DEBUG] Selected files: {self.selected_files}")
        selected_analyses = [k for k, v in self.analysis_checks.items() if v.isChecked()]
        if not self.data_dir or not self.selected_files:
            QMessageBox.warning(self, "Input Error", "Please select a data directory and at least one file.")
            return
        if not selected_analyses:
            QMessageBox.warning(self, "Input Error", "Please select at least one analysis/plot.")
            return
        # Output path and name
        output_dir = self.output_dir_edit.text().strip() or os.getcwd()
        base_name = self.output_name_edit.text().strip() or "output"
        save_plots = self.save_plots_cb.isChecked()
        save_data = self.save_data_cb.isChecked()
        combine_csv = self.combine_csv_cb.isChecked()
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                QMessageBox.critical(self, "Output Error", f"Failed to create output directory: {e}")
                return
        # --- Ambient conditions optimization ---
        ambient_dict = None
        if self.ambient_file:
            ambient_dict = {}
            try:
                with open(self.ambient_file, newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        # Use the filename or a unique key as needed
                        key = row.get('Filename') or row.get('name') or row.get('Name')
                        if key:
                            ambient_dict[key] = row
            except Exception as e:
                print(f"Failed to read ambient CSV: {e}")
                ambient_dict = None
        # Load PointConcentration objects and preprocess
        conc_objs = []
        fallback_files = []  # Track files where fallback to defaults is used
        avg_txt_files = []
        stats_txt_files = []
        for fname in self.selected_files:
            print(f"[DEBUG] Processing file: {fname}")
            fpath = os.path.join(self.data_dir, fname)
            conc = PointConcentration.from_file(fpath)
            ambient_found = False
            if self.ambient_file:
                try:
                    ambient = PointConcentration.get_ambient_conditions(path=self.data_dir, name=fname, input_file=self.ambient_file)
                    if ambient is not None:
                        (
                            x_source, y_source, z_source, x_measure, y_measure, z_measure, pressure, temperature,
                            calibration_curve, mass_flow_controller, calibration_factor, scaling_factor, scale, ref_length,
                            ref_height, gas_name, mol_weight, gas_factor, full_scale_wtref, full_scale_flow_rate,
                            full_scale_temp, full_scale_pressure
                        ) = PointConcentration.read_ambient_conditions(ambient, fname)
                        ambient_found = True
                except Exception as e:
                    print(f"Failed to read ambient conditions from CSV for {fname}: {e}")
            if not ambient_found:
                fallback_files.append(fname)
                # Use defaults from dropdown if enabled, else hardcoded
                if self.ambient_defaults_group.isChecked():
                    defval = lambda k, d: type(d)(self.ambient_fields[k].text()) if k in self.ambient_fields else d
                    x_source = defval('x_source', 0)
                    y_source = defval('y_source', 0)
                    z_source = defval('z_source', 0)
                    x_measure = defval('x_measure', 1020)
                    y_measure = defval('y_measure', 0)
                    z_measure = defval('z_measure', 5)
                    pressure = defval('pressure', 101426.04472)
                    temperature = defval('temperature', 23)
                    calibration_curve = defval('calibration_curve', 1.0)
                    mass_flow_controller = defval('mass_flow_controller', 0.3)
                    calibration_factor = defval('calibration_factor', 0)
                    gas_name = self.ambient_fields['gas_name'].text() if 'gas_name' in self.ambient_fields else 'C12'
                    gas_factor = defval('gas_factor', 0.5)
                    mol_weight = defval('mol_weight', 29.0)
                    scale = defval('scale', 400)
                    scaling_factor = defval('scaling_factor', 0.5614882)
                    ref_length = defval('ref_length', 1/400)
                    ref_height = defval('ref_height', 100/400)
                    full_scale_wtref = defval('full_scale_wtref', 10)
                    full_scale_flow_rate = defval('full_scale_flow_rate', 0.002)
                    full_scale_temp = defval('full_scale_temp', 20)
                    full_scale_pressure = defval('full_scale_pressure', 101325)
                else:
                    x_source = 0
                    y_source = 0
                    z_source = 0
                    mass_flow_controller = 0.3
                    calibration_curve = 1.0
                    calibration_factor = 0
                    gas_name = 'C12'
                    gas_factor = 0.5
                    mol_weight = 29.0
                    x_measure = 1020
                    y_measure = 0
                    z_measure = 5
                    pressure = 101426.04472
                    temperature = 23
                    scale = 400
                    scaling_factor = 0.5614882
                    ref_length = 1/400
                    ref_height = 100/400
                    full_scale_wtref = 10
                    full_scale_flow_rate = 0.002
                    full_scale_temp = 20
                    full_scale_pressure = 101325
            conc.ambient_conditions(
                x_source=x_source, y_source=y_source, z_source=z_source,
                x_measure=x_measure, y_measure=y_measure, z_measure=z_measure,
                pressure=pressure, temperature=temperature,
                calibration_curve=calibration_curve,
                mass_flow_controller=mass_flow_controller,
                calibration_factor=calibration_factor
            )
            conc.scaling_information(
                scaling_factor=scaling_factor, scale=scale,
                ref_length=ref_length, ref_height=ref_height
            )
            conc.tracer_information(
                gas_name=gas_name, mol_weight=mol_weight, gas_factor=gas_factor
            )
            conc.full_scale_information(
                full_scale_wtref=full_scale_wtref,
                full_scale_flow_rate=full_scale_flow_rate,
                full_scale_temp=full_scale_temp,
                full_scale_pressure=full_scale_pressure
            )
            conc.convert_temperature()
            conc.calc_wtref_mean()
            conc.calc_model_mass_flow_rate(usingMaxFlowRate="True", applyCalibration="False")
            conc.calc_net_concentration()
            conc.calc_c_star()
            conc.calc_full_scale_concentration()
            conc_objs.append(conc)
            # Save postprocessed data if requested
            if save_data:
                try:
                    avg_txt = f"{base_name}_{fname}"
                    stats_txt = f"{base_name}_{fname}"
                    out_dir = os.path.join(output_dir, '')
                    print(f"[DEBUG] Will save avg_txt as: {os.path.join(out_dir, avg_txt)}")
                    print(f"[DEBUG] Will save stats_txt as: {os.path.join(out_dir, stats_txt)}")
                    conc.save2file_avg(avg_txt, out_dir=out_dir)
                    conc.save2file_fullStats(stats_txt, out_dir=out_dir)
                    avg_txt_files.append(avg_txt)
                    stats_txt_files.append(stats_txt)
                except Exception as e:
                    QMessageBox.warning(self, "Save Data Error", f"Failed to save postprocessed data for {fname}: {e}")
        # Show warning if any files fell back to defaults
        if fallback_files:
            QMessageBox.information(self, "Ambient Conditions Fallback", f"The following files did not have ambient conditions in the CSV and default values were used:\n" + "\n".join(fallback_files))
        # Plotting logic for each selected analysis
        if len(conc_objs) == 0:
            QMessageBox.warning(self, "No Data", "No valid data objects to plot.")
            return
            
        for analysis in selected_analyses:
            try:
                fig = None
                if analysis == "Pdf" and create_pdf is not None:
                    fig = create_pdf(conc_objs, dimensionless="False", labels=None, xLabel=None, yLabel=None, xAchse=None, yAchse=None)
                    # Try to apply tight_layout for PDF plot
                    try:
                        if hasattr(fig, 'tight_layout'):
                            fig.tight_layout()
                    except Exception as e:
                        print(f"tight_layout() failed for PDF plot: {e}")
                elif analysis == "Histogram" and create_histogram is not None:
                    fig = create_histogram(conc_objs, dimensionless="False", labels=None, xLabel=None, yLabel=None, xAchse=None, yAchse=None)
                elif analysis == "Cdf" and create_cdf is not None:
                    fig = create_cdf(conc_objs, dimensionless="False", labels=None, xLabel=None, yLabel=None, xAchse=None, yAchse=None)
                elif analysis == "Means" and create_means is not None:
                    fig = create_means(conc_objs, error_values=None, dimensionless="False", labels=None, xLabel=None, yLabel=None, xAchse=None, yAchse=None)
                elif analysis == "ScatterPlot" and create_scatterplot is not None:
                    fig = create_scatterplot(conc_objs, dimensionless="False", labels=None, xLabel=None, yLabel=None, xAchse=None, yAchse=None)
                elif analysis == "PowerDensity" and powerDensityPlot is not None:
                    fig = powerDensityPlot(conc_objs, dimensionless="False", plot=True, labels=None, xLabel=None, yLabel=None, xAchse=None, yAchse=None)
                elif analysis == "BoxPlot" and create_boxplot is not None:
                    fig = create_boxplot(conc_objs, dimensionless="False", labels=None, xLabel=None, yLabel=None, xAchse=None, yAchse=None)
                if fig is None:
                    fig = plt.gcf()
                canvas = FigureCanvas(fig)
                result_widget = QWidget()
                layout = QVBoxLayout(result_widget)
                layout.addWidget(canvas)
                self.results_tabs.addTab(result_widget, f"{analysis} {self.results_tabs.count()+1}")
                if save_plots:
                    # Robust error handling and tight_layout for all plots
                    try:
                        plot_filename = f"{base_name}_{analysis}.png"
                        if hasattr(fig, 'tight_layout'):
                            fig.tight_layout()
                        fig.savefig(os.path.join(output_dir, plot_filename))
                    except Exception as e:
                        # Print debug info
                        try:
                            data_shapes = [getattr(obj, 'net_concentration', None) for obj in conc_objs]
                            fig_size = fig.get_size_inches() if hasattr(fig, 'get_size_inches') else 'unknown'
                            print(f"Plot error for {analysis}: data_shapes={data_shapes}, fig_size={fig_size}")
                        except Exception as debug_e:
                            print(f"Error while printing debug info: {debug_e}")
                        QMessageBox.warning(self, "Save Plot Error", f"Failed to save plot {analysis}: {e}")
                        continue
                plt.close(fig)
            except Exception as e:
                QMessageBox.critical(self, "Plot Error", f"Failed to plot {analysis}: {e}")
                continue
        # Combine to CSV if requested
        if combine_csv and save_data and avg_txt_files:
            print(f"[DEBUG] avg_txt_files: {avg_txt_files}")
            print(f"[DEBUG] stats_txt_files: {stats_txt_files}")
            print(f"[DEBUG] output_dir: {output_dir}")
            # Add back the manual prefixing to match saved files
            avg_txt_files = ["_avg_" + f for f in avg_txt_files]
            stats_txt_files = ["_stats_" + f for f in stats_txt_files]
            try:
                avg_csv = os.path.join(output_dir, f"{base_name}_combined_avg.csv")
                stats_csv = os.path.join(output_dir, f"{base_name}_combined_stats.csv")
                print(f"[DEBUG] Attempting to save avg CSV to: {avg_csv}")
                print(f"[DEBUG] Attempting to save stats CSV to: {stats_csv}")
                #combine_to_csv(avg_txt_files, output_dir + "/", file_type='avg', output_filename=avg_csv)
                combine_to_csv(stats_txt_files, output_dir + "/", file_type='stats', output_filename=stats_csv)
                # Check if files exist
                #avg_exists = os.path.exists(avg_csv)
                stats_exists = os.path.exists(stats_csv)
                #print(f"[DEBUG] avg_csv exists: {avg_exists}")
                print(f"[DEBUG] stats_csv exists: {stats_exists}")
                #if avg_exists and stats_exists:
                if stats_exists:
                    QMessageBox.information(self, "CSV Combined", f"Combined CSV files saved as:\n{avg_csv}\n{stats_csv}")
                else:
                    QMessageBox.warning(self, "CSV Not Found", f"CSV file(s) not found after saving.\nExpected:\n{stats_csv} (exists: {stats_exists})")
            except Exception as e:
                QMessageBox.warning(self, "Combine CSV Error", f"Failed to combine txt files to CSV: {e}")

    def close_tab(self, index):
        self.results_tabs.removeTab(index)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Apply a modern dark theme
    dark_stylesheet = """
    QWidget { background-color: #232629; color: #f0f0f0; }
    QGroupBox { border: 1px solid #444; margin-top: 10px; }
    QGroupBox:title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
    QPushButton { background-color: #31363b; color: #f0f0f0; border: 1px solid #444; border-radius: 4px; padding: 6px; }
    QPushButton:hover { background-color: #3a3f44; }
    QLineEdit, QComboBox, QListWidget, QTabWidget, QScrollArea, QCheckBox, QLabel {
        background-color: #232629; color: #f0f0f0; border: 1px solid #444; border-radius: 3px;
    }
    QTabBar::tab { background: #31363b; color: #f0f0f0; border: 1px solid #444; border-radius: 3px; padding: 6px; }
    QTabBar::tab:selected { background: #232629; }
    QTabBar::close-button { image: url(close.png); }
    QScrollBar:vertical { background: #232629; width: 12px; margin: 22px 0 22px 0; border: 1px solid #444; }
    QScrollBar::handle:vertical { background: #31363b; min-height: 20px; border-radius: 4px; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { background: none; }
    """
    app.setStyleSheet(dark_stylesheet)
    window = ConcentrationGUI()
    window.show()
    sys.exit(app.exec_()) 