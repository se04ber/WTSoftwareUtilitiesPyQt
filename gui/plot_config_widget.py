"""
gui/plot_config_widget.py
GUI widgets for plot configuration.
"""
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QLabel, QPushButton, QScrollArea, QTabWidget, QFileDialog, QMessageBox, QDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from typing import Dict, Any, Optional
from logic.plot_config import PlotConfigManager, PlotType, PlotConfiguration

class PlotParameterWidget(QWidget):
    """Widget for configuring a single plot parameter."""
    
    def __init__(self, parameter, parent=None):
        super().__init__(parent)
        self.parameter = parameter
        
        # Parameter input widget
        self.input_widget = self._create_input_widget()
        
        # Set default value
        self.set_value(parameter.default_value)
    
    def _create_input_widget(self):
        """Create the appropriate input widget based on parameter type."""
        if self.parameter.parameter_type == "text":
            widget = QLineEdit()
            # Add placeholder text for specific parameters
            if self.parameter.name == "xAchse":
                widget.setPlaceholderText("e.g., (0, 100) or leave empty for auto")
            elif self.parameter.name == "yAchse":
                widget.setPlaceholderText("e.g., (0, 50) or leave empty for auto")
            elif self.parameter.name == "labels":
                widget.setPlaceholderText("e.g., Dataset1, Dataset2, Dataset3")
            else:
                widget.setPlaceholderText("Enter text...")
        elif self.parameter.parameter_type == "number":
            if isinstance(self.parameter.default_value, int):
                widget = QSpinBox()
                if self.parameter.min_value is not None:
                    widget.setMinimum(int(self.parameter.min_value))
                if self.parameter.max_value is not None:
                    widget.setMaximum(int(self.parameter.max_value))
            else:
                widget = QDoubleSpinBox()
                if self.parameter.min_value is not None:
                    widget.setMinimum(self.parameter.min_value)
                if self.parameter.max_value is not None:
                    widget.setMaximum(self.parameter.max_value)
                widget.setDecimals(3)
        elif self.parameter.parameter_type == "boolean":
            widget = QCheckBox()
        elif self.parameter.parameter_type == "choice":
            widget = QComboBox()
            if self.parameter.choices:
                widget.addItems(self.parameter.choices)
        else:
            widget = QLineEdit()
            widget.setPlaceholderText("Enter value...")
        
        return widget
    
    def get_value(self) -> Any:
        """Get the current value of the parameter."""
        if isinstance(self.input_widget, QLineEdit):
            return self.input_widget.text()
        elif isinstance(self.input_widget, (QSpinBox, QDoubleSpinBox)):
            return self.input_widget.value()
        elif isinstance(self.input_widget, QCheckBox):
            return self.input_widget.isChecked()
        elif isinstance(self.input_widget, QComboBox):
            return self.input_widget.currentText()
        else:
            return None
    
    def set_value(self, value: Any):
        """Set the value of the parameter."""
        if isinstance(self.input_widget, QLineEdit):
            self.input_widget.setText(str(value))
        elif isinstance(self.input_widget, (QSpinBox, QDoubleSpinBox)):
            self.input_widget.setValue(value)
        elif isinstance(self.input_widget, QCheckBox):
            self.input_widget.setChecked(bool(value))
        elif isinstance(self.input_widget, QComboBox):
            index = self.input_widget.findText(str(value))
            if index >= 0:
                self.input_widget.setCurrentIndex(index)

class PlotConfigurationWidget(QWidget):
    """Widget for configuring a specific plot type."""
    
    configuration_changed = pyqtSignal(PlotType, dict)
    
    def __init__(self, plot_type: PlotType, config: PlotConfiguration, parent=None):
        super().__init__(parent)
        self.plot_type = plot_type
        self.config = config
        self.parameter_widgets: Dict[str, PlotParameterWidget] = {}
        
        self.layout = QVBoxLayout(self)
        self._create_widgets()
    
    def _create_widgets(self):
        """Create the configuration widgets."""
        # Plot title and labels
        title_group = QGroupBox("Plot Labels")
        title_layout = QFormLayout()
        title_group.setLayout(title_layout)
        
        self.title_edit = QLineEdit(self.config.title)
        self.x_label_edit = QLineEdit(self.config.x_label)
        self.y_label_edit = QLineEdit(self.config.y_label)
        
        title_layout.addRow("Title:", self.title_edit)
        title_layout.addRow("X-Axis Label:", self.x_label_edit)
        title_layout.addRow("Y-Axis Label:", self.y_label_edit)
        
        self.layout.addWidget(title_group)
        
        # Parameters
        if self.config.parameters:
            param_group = QGroupBox("Plot Parameters")
            param_layout = QFormLayout()
            param_group.setLayout(param_layout)
            
            for parameter in self.config.parameters:
                param_widget = PlotParameterWidget(parameter)
                self.parameter_widgets[parameter.name] = param_widget
                # Only add the input widget to the form layout, not the entire widget
                param_layout.addRow(parameter.display_name, param_widget.input_widget)
            
            self.layout.addWidget(param_group)
        
        # Connect signals
        self.title_edit.textChanged.connect(self._on_configuration_changed)
        self.x_label_edit.textChanged.connect(self._on_configuration_changed)
        self.y_label_edit.textChanged.connect(self._on_configuration_changed)
        
        for widget in self.parameter_widgets.values():
            if isinstance(widget.input_widget, QLineEdit):
                widget.input_widget.textChanged.connect(self._on_configuration_changed)
            elif isinstance(widget.input_widget, (QSpinBox, QDoubleSpinBox)):
                widget.input_widget.valueChanged.connect(self._on_configuration_changed)
            elif isinstance(widget.input_widget, QCheckBox):
                widget.input_widget.toggled.connect(self._on_configuration_changed)
            elif isinstance(widget.input_widget, QComboBox):
                widget.input_widget.currentTextChanged.connect(self._on_configuration_changed)
    
    def _on_configuration_changed(self):
        """Emit signal when configuration changes."""
        config_dict = self.get_configuration_dict()
        self.configuration_changed.emit(self.plot_type, config_dict)
    
    def get_configuration_dict(self) -> Dict[str, Any]:
        """Get the current configuration as a dictionary."""
        config_dict = {
            "title": self.title_edit.text(),
            "x_label": self.x_label_edit.text(),
            "y_label": self.y_label_edit.text(),
            "parameters": {}
        }
        
        for param_name, widget in self.parameter_widgets.items():
            config_dict["parameters"][param_name] = widget.get_value()
        
        return config_dict
    
    def set_configuration_dict(self, config_dict: Dict[str, Any]):
        """Set the configuration from a dictionary."""
        if "title" in config_dict:
            self.title_edit.setText(config_dict["title"])
        if "x_label" in config_dict:
            self.x_label_edit.setText(config_dict["x_label"])
        if "y_label" in config_dict:
            self.y_label_edit.setText(config_dict["y_label"])
        
        if "parameters" in config_dict:
            for param_name, value in config_dict["parameters"].items():
                if param_name in self.parameter_widgets:
                    self.parameter_widgets[param_name].set_value(value)

class PlotConfigurationDialog(QDialog):
    """Main dialog for configuring all plot types."""
    
    configuration_changed = pyqtSignal()  # Signal emitted when any configuration changes
    
    def __init__(self, config_manager: PlotConfigManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.config_widgets: Dict[PlotType, PlotConfigurationWidget] = {}
        
        # Add debouncing timer to prevent excessive refreshes
        self.refresh_timer = QTimer()
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.timeout.connect(self._emit_refresh_signal)
        
        self.setWindowTitle("Plot Configuration")
        self.setModal(True)
        self.resize(600, 500)
        
        self.layout = QVBoxLayout(self)
        self._create_widgets()
    
    def _create_widgets(self):
        """Create the main configuration interface."""
        # Title
        title_label = QLabel("Plot Configuration")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        self.layout.addWidget(title_label)
        
        # Tab widget for different plot types
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        # Create tabs for each plot type
        for plot_type, config in self.config_manager.get_all_configurations().items():
            if config.enabled:
                config_widget = PlotConfigurationWidget(plot_type, config)
                self.config_widgets[plot_type] = config_widget
                config_widget.configuration_changed.connect(self._on_configuration_changed)
                
                self.tab_widget.addTab(config_widget, plot_type.value)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Configuration")
        self.save_button.clicked.connect(self.save_configuration)
        button_layout.addWidget(self.save_button)
        
        self.load_button = QPushButton("Load Configuration")
        self.load_button.clicked.connect(self.load_configuration)
        button_layout.addWidget(self.load_button)
        
        self.refresh_button = QPushButton("Refresh Plots")
        self.refresh_button.clicked.connect(self.manual_refresh_plots)
        button_layout.addWidget(self.refresh_button)
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.reset_button)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)
        
        self.layout.addLayout(button_layout)
    
    def _on_configuration_changed(self, plot_type: PlotType, config_dict: Dict[str, Any]):
        """Handle configuration changes."""
        print(f"[DEBUG] Configuration changed for {plot_type.value}")
        
        # Update the configuration manager
        config = self.config_manager.get_configuration(plot_type)
        if config:
            config.title = config_dict["title"]
            config.x_label = config_dict["x_label"]
            config.y_label = config_dict["y_label"]
            
            # Update parameter values
            for param_name, value in config_dict.get("parameters", {}).items():
                for param in config.parameters:
                    if param.name == param_name:
                        param.default_value = value
                        break
        
        # Auto-save configuration to file
        self._auto_save_configuration()
        
        # Use debounced refresh instead of immediate signal emission
        print("[DEBUG] Starting debounced refresh timer")
        self.refresh_timer.start(500)  # 500ms delay
    
    def _emit_refresh_signal(self):
        """Emit the refresh signal after debouncing delay."""
        print("[DEBUG] Emitting configuration_changed signal after debounce")
        self.configuration_changed.emit()
    
    def _auto_save_configuration(self):
        """Automatically save configuration to the default location."""
        try:
            config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
            os.makedirs(config_dir, exist_ok=True)
            config_file = os.path.join(config_dir, 'plot_config.json')
            self.config_manager.save_configurations(config_file)
        except Exception as e:
            print(f"[DEBUG] Auto-save failed: {e}")
    
    def get_all_configurations(self) -> Dict[PlotType, Dict[str, Any]]:
        """Get all current configurations."""
        configs = {}
        for plot_type, widget in self.config_widgets.items():
            configs[plot_type] = widget.get_configuration_dict()
        return configs
    
    def save_configuration(self):
        """Save the current configuration."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot Configuration", 
            filter="JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            try:
                self.config_manager.save_configurations(file_path)
                QMessageBox.information(self, "Success", f"Configuration saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save configuration: {e}")
    
    def load_configuration(self):
        """Load a configuration from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Plot Configuration", 
            filter="JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            try:
                self.config_manager.load_configurations(file_path)
                # Refresh the widgets with loaded configuration
                self._refresh_widgets()
                QMessageBox.information(self, "Success", f"Configuration loaded from {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration: {e}")
    
    def _refresh_widgets(self):
        """Refresh all configuration widgets with current configuration."""
        self.tab_widget.clear()
        self.config_widgets.clear()
        
        for plot_type, config in self.config_manager.get_all_configurations().items():
            if config.enabled:
                config_widget = PlotConfigurationWidget(plot_type, config)
                self.config_widgets[plot_type] = config_widget
                config_widget.configuration_changed.connect(self._on_configuration_changed)
                self.tab_widget.addTab(config_widget, plot_type.value)
    
    def reset_to_defaults(self):
        """Reset all configurations to defaults."""
        self.config_manager._initialize_default_configurations()
        # Recreate widgets with default configurations
        self.tab_widget.clear()
        self.config_widgets.clear()
        
        for plot_type, config in self.config_manager.get_all_configurations().items():
            if config.enabled:
                config_widget = PlotConfigurationWidget(plot_type, config)
                self.config_widgets[plot_type] = config_widget
                config_widget.configuration_changed.connect(self._on_configuration_changed)
                self.tab_widget.addTab(config_widget, plot_type.value)
    
    def manual_refresh_plots(self):
        """Manual refresh of plots."""
        print("[DEBUG] Manual refresh plots button clicked")
        # Stop any pending timer and emit immediately
        self.refresh_timer.stop()
        self.configuration_changed.emit() 