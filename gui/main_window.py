# gui/main_window.py
"""
Main application window for the Concentration Analysis GUI.
Handles UI setup and delegates business logic to logic modules.
"""
import os
import csv
import sys
import pandas as pd
from typing import List, Optional
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QListWidget, QListWidgetItem, QCheckBox, QTabWidget, QGroupBox, QFormLayout, QLineEdit, QMessageBox,
    QSplitter, QScrollArea, QDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from windtunnel.concentration.utils import combine_to_csv
from gui.sidebar import SidebarWidget
from gui.results_tabs import ResultsTabWidget
from gui.plot_config_widget import PlotConfigurationDialog
from logic.data_loader import DataLoader
from logic.ambient_manager import AmbientManager
from logic.plot_manager import PlotManager
from logic.plot_config import PlotConfigManager
from logic.config import AppConfig

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

class MainWindow(QMainWindow):
    """
    Main application window. Sets up the UI and coordinates workflow.
    Delegates data, ambient, and plotting logic to logic modules.
    """
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Concentration Analysis GUI")
        self.setGeometry(100, 100, 900, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.addWidget(self.splitter)

        # Logic modules
        self.data_loader = DataLoader()
        self.ambient_manager = AmbientManager()
        self.plot_config_manager = PlotConfigManager()
        self.plot_manager = PlotManager(self.plot_config_manager)
        self.config = AppConfig()
        
        # Load saved plot configuration
        self._load_plot_configuration()
        
        # Initialize plot configuration dialog
        self.plot_config_dialog = None

        # Sidebar and Results Tabs widgets
        self.sidebar = SidebarWidget()
        self.results_tabs = ResultsTabWidget()

        # Make sidebar scrollable
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.sidebar)
        self.splitter.addWidget(self.scroll_area)
        self.splitter.addWidget(self.results_tabs)
        self.splitter.setSizes([350, 900])

        # Connect sidebar signals to main window logic
        self.sidebar.dir_button.clicked.connect(self.choose_directory)
        self.sidebar.ambient_button.clicked.connect(self.choose_ambient_file)
        self.sidebar.output_dir_button.clicked.connect(self.choose_output_directory)
        self.sidebar.run_button.clicked.connect(self.run_analysis)
        self.sidebar.plot_config_button.clicked.connect(self.open_plot_configuration)
        
        # Connect select all/none buttons
        self.sidebar.select_all_button.clicked.connect(self.select_all_analyses)
        self.sidebar.select_none_button.clicked.connect(self.select_none_analyses)

        self.data_dir: Optional[str] = None
        self.selected_files: List[str] = []
        self.last_plot_data: Optional[List[Any]] = None
        self.last_selected_analyses: Optional[List[str]] = None

        # Load and apply dark theme from QSS file
        self.apply_theme()

    def apply_theme(self) -> None:
        """
        Apply a modern, professional theme with better visual feedback.
        """
        # Modern professional theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
                color: #333333;
            }
            
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 9pt;
            }
            
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: 500;
                min-height: 20px;
            }
            
            QPushButton:hover {
                background-color: #106ebe;
            }
            
            QPushButton:pressed {
                background-color: #005a9e;
            }
            
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            
            QListWidget {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 4px;
                selection-background-color: #0078d4;
                selection-color: white;
            }
            
            QListWidget::item {
                padding: 6px;
                border-bottom: 1px solid #f0f0f0;
            }
            
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
            
            QListWidget::item:hover {
                background-color: #e6f3ff;
            }
            
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                background-color: white;
                border-radius: 4px;
            }
            
            QTabBar::tab {
                background-color: #f0f0f0;
                color: #333333;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 80px;
            }
            
            QTabBar::tab:selected {
                background-color: white;
                color: #0078d4;
                font-weight: 500;
            }
            
            QTabBar::tab:hover {
                background-color: #e6f3ff;
            }
            
            QTabBar::close-button {
                image: url(close.png);
                subcontrol-position: right;
                margin: 2px;
            }
            
            QTabBar::close-button:hover {
                background-color: #ff4444;
                border-radius: 2px;
            }
            
            QGroupBox {
                font-weight: 500;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
                color: #0078d4;
            }
            
            QCheckBox {
                spacing: 8px;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #d0d0d0;
                border-radius: 2px;
                background-color: white;
            }
            
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border-color: #0078d4;
                image: url(checkmark.png);
            }
            
            QLineEdit, QTextEdit {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 6px;
                background-color: white;
            }
            
            QLineEdit:focus, QTextEdit:focus {
                border-color: #0078d4;
            }
            
            QLabel {
                color: #333333;
            }
            
            QLabel[class="title"] {
                font-size: 14pt;
                font-weight: 600;
                color: #0078d4;
                padding: 8px 0;
            }
            
            QLabel[class="subtitle"] {
                font-size: 11pt;
                font-weight: 500;
                color: #666666;
                padding: 4px 0;
            }
            
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 12px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #c0c0c0;
                border-radius: 6px;
                min-height: 20px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #a0a0a0;
            }
            
            QMenuBar {
                background-color: #0078d4;
                color: white;
                border: none;
            }
            
            QMenuBar::item {
                background-color: transparent;
                padding: 8px 12px;
            }
            
            QMenuBar::item:selected {
                background-color: #106ebe;
            }
            
            QMenu {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 4px;
            }
            
            QMenu::item {
                padding: 6px 20px;
                border-radius: 2px;
            }
            
            QMenu::item:selected {
                background-color: #e6f3ff;
                color: #0078d4;
            }
            
            QToolBar {
                background-color: #f8f8f8;
                border: none;
                spacing: 4px;
                padding: 4px;
            }
            
            QStatusBar {
                background-color: #f0f0f0;
                color: #666666;
                border-top: 1px solid #d0d0d0;
            }
        """)

    def choose_directory(self) -> None:
        """
        Open a dialog to select the data directory and populate the file list.
        """
        dir_path = self.sidebar.dir_label.text()
        dir_path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if dir_path:
            self.data_dir = dir_path
            self.sidebar.dir_label.setText(f"Selected: {dir_path}")
            self.config.set('last_data_dir', dir_path)
            self.populate_file_list()

    def populate_file_list(self) -> None:
        """
        Populate the file list widget with files from the selected data directory.
        """
        self.sidebar.file_list.clear()
        if self.data_dir:
            files = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]
            for f in files:
                item = QListWidgetItem(f)
                self.sidebar.file_list.addItem(item)

    def choose_ambient_file(self) -> None:
        """
        Open a dialog to select the ambient conditions CSV file.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Ambient Conditions CSV File", filter="CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.sidebar.ambient_file = file_path
            self.sidebar.ambient_label.setText(f"Selected: {file_path}")
        else:
            self.sidebar.ambient_file = None
            self.sidebar.ambient_label.setText("No ambient conditions file selected.")

    def choose_output_directory(self) -> None:
        """
        Open a dialog to select the output directory.
        """
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.sidebar.output_dir = dir_path
            self.sidebar.output_dir_edit.setText(dir_path)
            self.config.set('last_output_dir', dir_path)

    def select_all_analyses(self) -> None:
        """Select all analysis/plot types."""
        for checkbox in self.sidebar.analysis_checks.values():
            checkbox.setChecked(True)

    def select_none_analyses(self) -> None:
        """Deselect all analysis/plot types."""
        for checkbox in self.sidebar.analysis_checks.values():
            checkbox.setChecked(False)

    def run_analysis(self) -> None:
        """
        Run the analysis workflow: load data, apply ambient, plot, and save results.
        """
        self.selected_files = [item.text() for item in self.sidebar.file_list.selectedItems()]
        print(f"[DEBUG] Selected files: {self.selected_files}")
        selected_analyses = [k for k, v in self.sidebar.analysis_checks.items() if v.isChecked()]
        if not self.data_dir or not self.selected_files:
            QMessageBox.warning(self, "Input Error", "Please select a data directory and at least one file.")
            return
        if not selected_analyses:
            QMessageBox.warning(self, "Input Error", "Please select at least one analysis/plot.")
            return
        output_dir = self.sidebar.output_dir_edit.text().strip() or os.getcwd()
        base_name = self.sidebar.output_name_edit.text().strip() or "output"
        save_plots = self.sidebar.save_plots_cb.isChecked()
        save_data = self.sidebar.save_data_cb.isChecked()
        combine_csv = self.sidebar.combine_csv_cb.isChecked()
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                QMessageBox.critical(self, "Output Error", f"Failed to create output directory: {e}")
                return
        ambient_file = getattr(self.sidebar, 'ambient_file', None)
        print(f"[DEBUG] Ambient file: {ambient_file}")
        conc_objs = []
        fallback_files = []
        avg_txt_files = []
        stats_txt_files = []
        for fname in self.selected_files:
            print(f"[DEBUG] Processing file: {fname}")
            fpath = os.path.join(self.data_dir, fname)
            conc = PointConcentration.from_file(fpath)
            ambient_found = False
            
            # Try to load ambient conditions using the original PointConcentration method
            if ambient_file:
                try:
                    print(f"[DEBUG] Trying to load ambient conditions for {fname} from {ambient_file}")
                    ambient_conditions = PointConcentration.get_ambient_conditions(
                        path=self.data_dir, 
                        name=fname, 
                        input_file=ambient_file
                    )
                    if ambient_conditions is not None:
                        (
                            x_source, y_source, z_source, x_measure, y_measure, z_measure, pressure, temperature,
                            calibration_curve, mass_flow_controller, calibration_factor, scaling_factor, scale, ref_length,
                            ref_height, gas_name, mol_weight, gas_factor, full_scale_wtref, full_scale_flow_rate,
                            full_scale_temp, full_scale_pressure
                        ) = PointConcentration.read_ambient_conditions(ambient_conditions, fname)
                        print(f"[DEBUG] Successfully loaded ambient conditions for {fname}")
                        ambient_found = True
                    else:
                        print(f"[DEBUG] No ambient conditions found for {fname} in {ambient_file}")
                except Exception as e:
                    print(f"Failed to read ambient conditions from CSV for {fname}: {e}")
            if not ambient_found:
                fallback_files.append(fname)
                if self.sidebar.ambient_defaults_group.isChecked():
                    defval = lambda k, d: type(d)(self.sidebar.ambient_fields[k].text()) if k in self.sidebar.ambient_fields else d
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
                    gas_name = self.sidebar.ambient_fields['gas_name'].text() if 'gas_name' in self.sidebar.ambient_fields else 'C12'
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
            
            # Add filename attribute for plotting
            conc.filename = fname
            
            conc_objs.append(conc)
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
        if fallback_files:
            QMessageBox.information(self, "Ambient Conditions Fallback", f"The following files did not have ambient conditions in the CSV and default values were used:\n" + "\n".join(fallback_files))
        if len(conc_objs) == 0:
            QMessageBox.warning(self, "No Data", "No valid data objects to plot.")
            return
        
        # Store data for potential refresh
        self.last_plot_data = conc_objs
        self.last_selected_analyses = selected_analyses
        
        # Debug: Check concentration objects
        print(f"[DEBUG] Number of concentration objects: {len(conc_objs)}")
        for i, conc_obj in enumerate(conc_objs):
            print(f"[DEBUG] Object {i}: filename={getattr(conc_obj, 'filename', 'unknown')}")
            print(f"[DEBUG] Object {i}: net_concentration length={len(getattr(conc_obj, 'net_concentration', []))}")
            print(f"[DEBUG] Object {i}: net_concentration type={type(getattr(conc_obj, 'net_concentration', []))}")
            print(f"[DEBUG] Object {i}: net_concentration sample={getattr(conc_obj, 'net_concentration', [])[:5] if len(getattr(conc_obj, 'net_concentration', [])) > 0 else 'No data'}")
            print(f"[DEBUG] Object {i}: c_star length={len(getattr(conc_obj, 'c_star', []))}")
            print(f"[DEBUG] Object {i}: c_star type={type(getattr(conc_obj, 'c_star', []))}")
            print(f"[DEBUG] Object {i}: x={getattr(conc_obj, 'x', 'None')}, y={getattr(conc_obj, 'y', 'None')}, z={getattr(conc_obj, 'z', 'None')}")
            
            # Check if the object has the expected attributes
            if hasattr(conc_obj, 'fast_FID'):
                print(f"[DEBUG] Object {i}: fast_FID length={len(conc_obj.fast_FID)}")
            if hasattr(conc_obj, 'slow_FID'):
                print(f"[DEBUG] Object {i}: slow_FID length={len(conc_obj.slow_FID)}")
            if hasattr(conc_obj, 'time'):
                print(f"[DEBUG] Object {i}: time length={len(conc_obj.time)}")
        
        for analysis in selected_analyses:
            try:
                fig = None
                
                # Try to use the new configurable plot manager first
                try:
                    # Store the original concentration objects for the plot manager
                    # This preserves the dataset structure and is more efficient
                    plot_data = conc_objs
                    
                    # Use the new plot manager with configurations
                    print(f"[DEBUG] Creating plot for {analysis} with {len(plot_data)} datasets")
                    print(f"[DEBUG] Data structure: {len(plot_data)} concentration objects")
                    fig = self.plot_manager.create_plot(plot_data, analysis)
                    
                except Exception as new_plot_error:
                    print(f"New plot manager failed for {analysis}: {new_plot_error}")
                    # Fallback to original plotting functions
                    if analysis == "Pdf" and create_pdf is not None:
                        fig = create_pdf(conc_objs, dimensionless="False", labels=None, xLabel=None, yLabel=None, xAchse=None, yAchse=None)
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
                self.results_tabs.add_plot_tab(result_widget, f"{analysis} {self.results_tabs.count()+1}")
                
                if save_plots:
                    try:
                        plot_filename = f"{base_name}_{analysis}.png"
                        if hasattr(fig, 'tight_layout'):
                            fig.tight_layout()
                        fig.savefig(os.path.join(output_dir, plot_filename))
                    except Exception as e:
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
        if combine_csv and save_data and avg_txt_files:
            print(f"[DEBUG] avg_txt_files: {avg_txt_files}")
            print(f"[DEBUG] stats_txt_files: {stats_txt_files}")
            print(f"[DEBUG] output_dir: {output_dir}")
            avg_txt_files = ["_avg_" + f for f in avg_txt_files]
            stats_txt_files = ["_stats_" + f for f in stats_txt_files]
            try:
                avg_csv = os.path.join(output_dir, f"{base_name}_combined_avg.csv")
                stats_csv = os.path.join(output_dir, f"{base_name}_combined_stats.csv")
                print(f"[DEBUG] Attempting to save avg CSV to: {avg_csv}")
                print(f"[DEBUG] Attempting to save stats CSV to: {stats_csv}")
                combine_to_csv(stats_txt_files, output_dir + "/", file_type='stats', output_filename=stats_csv)
                stats_exists = os.path.exists(stats_csv)
                print(f"[DEBUG] stats_csv exists: {stats_exists}")
                if stats_exists:
                    QMessageBox.information(self, "CSV Combined", f"Combined CSV files saved as:\n{avg_csv}\n{stats_csv}")
                else:
                    QMessageBox.warning(self, "CSV Not Found", f"CSV file(s) not found after saving.\nExpected:\n{stats_csv} (exists: {stats_exists})")
            except Exception as e:
                QMessageBox.warning(self, "Combine CSV Error", f"Failed to combine txt files to CSV: {e}")

    def open_plot_configuration(self) -> None:
        """
        Open the plot configuration dialog.
        """
        if self.plot_config_dialog is None or not self.plot_config_dialog.isVisible():
            self.plot_config_dialog = PlotConfigurationDialog(self.plot_config_manager, self)
            # Connect the configuration change signal to refresh plots
            self.plot_config_dialog.configuration_changed.connect(self.refresh_plots)
            print("[DEBUG] Connected configuration_changed signal to refresh_plots")
            self.plot_config_dialog.show()
        else:
            self.plot_config_dialog.raise_()
            self.plot_config_dialog.activateWindow()
    
    def refresh_plots(self) -> None:
        """
        Refresh all plots with current configuration.
        """
        print("[DEBUG] refresh_plots() called!")
        if self.last_plot_data is None or self.last_selected_analyses is None:
            print("[DEBUG] No plot data available for refresh")
            return
        
        print("[DEBUG] Refreshing plots with new configuration...")
        print(f"[DEBUG] Last plot data: {len(self.last_plot_data) if self.last_plot_data else 'None'} objects")
        print(f"[DEBUG] Last selected analyses: {self.last_selected_analyses}")
        
        # Clear existing plots
        self.results_tabs.clear_all_tabs()
        
        # Recreate plots with new configuration
        for analysis in self.last_selected_analyses:
            try:
                fig = None
                
                # Try to use the new configurable plot manager first
                try:
                    # Store the original concentration objects for the plot manager
                    # This preserves the dataset structure and is more efficient
                    plot_data = self.last_plot_data
                    
                    # Use the new plot manager with updated configurations
                    fig = self.plot_manager.create_plot(plot_data, analysis)
                    
                except Exception as new_plot_error:
                    print(f"New plot manager failed for {analysis}: {new_plot_error}")
                    # Fallback to original plotting functions (these won't use the new config)
                    if analysis == "Pdf" and create_pdf is not None:
                        fig = create_pdf(self.last_plot_data, dimensionless="False", labels=None, xLabel=None, yLabel=None, xAchse=None, yAchse=None)
                    elif analysis == "Histogram" and create_histogram is not None:
                        fig = create_histogram(self.last_plot_data, dimensionless="False", labels=None, xLabel=None, yLabel=None, xAchse=None, yAchse=None)
                    elif analysis == "Cdf" and create_cdf is not None:
                        fig = create_cdf(self.last_plot_data, dimensionless="False", labels=None, xLabel=None, yLabel=None, xAchse=None, yAchse=None)
                    elif analysis == "Means" and create_means is not None:
                        fig = create_means(self.last_plot_data, error_values=None, dimensionless="False", labels=None, xLabel=None, yLabel=None, xAchse=None, yAchse=None)
                    elif analysis == "ScatterPlot" and create_scatterplot is not None:
                        fig = create_scatterplot(self.last_plot_data, dimensionless="False", labels=None, xLabel=None, yLabel=None, xAchse=None, yAchse=None)
                    elif analysis == "PowerDensity" and powerDensityPlot is not None:
                        fig = powerDensityPlot(self.last_plot_data, dimensionless="False", plot=True, labels=None, xLabel=None, yLabel=None, xAchse=None, yAchse=None)
                    elif analysis == "BoxPlot" and create_boxplot is not None:
                        fig = create_boxplot(self.last_plot_data, dimensionless="False", labels=None, xLabel=None, yLabel=None, xAchse=None, yAchse=None)
                
                if fig is None:
                    fig = plt.gcf()
                
                canvas = FigureCanvas(fig)
                result_widget = QWidget()
                layout = QVBoxLayout(result_widget)
                layout.addWidget(canvas)
                self.results_tabs.add_plot_tab(result_widget, f"{analysis} {self.results_tabs.count()+1}")
                
                plt.close(fig)
                
            except Exception as e:
                print(f"Failed to refresh plot {analysis}: {e}")
                continue
    
    def _load_plot_configuration(self) -> None:
        """
        Load saved plot configuration from file.
        """
        config_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'plot_config.json')
        if os.path.exists(config_file):
            try:
                self.plot_config_manager.load_configurations(config_file)
                print(f"[DEBUG] Loaded plot configuration from {config_file}")
            except Exception as e:
                print(f"[DEBUG] Failed to load plot configuration: {e}")
    
    def _save_plot_configuration(self) -> None:
        """
        Save current plot configuration to file.
        """
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
        os.makedirs(config_dir, exist_ok=True)
        config_file = os.path.join(config_dir, 'plot_config.json')
        
        try:
            self.plot_config_manager.save_configurations(config_file)
            print(f"[DEBUG] Saved plot configuration to {config_file}")
        except Exception as e:
            print(f"[DEBUG] Failed to save plot configuration: {e}")
    
    def closeEvent(self, event) -> None:
        """
        Handle application close event - save configuration before closing.
        """
        self._save_plot_configuration()
        super().closeEvent(event) 