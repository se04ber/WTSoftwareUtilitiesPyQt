from PyQt5.QtWidgets import QTabWidget, QWidget, QVBoxLayout, QToolButton, QLabel
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QFont

class ResultsTabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        
        # Connect signals
        self.tabCloseRequested.connect(self.removeTab)
        
        # Set up tab bar properties for better UX
        self.tabBar().setExpanding(False)
        self.tabBar().setMovable(True)
        
        # Add tooltip for better user guidance
        self.setToolTip("Plot results - Click tabs to view different plots, click 'X' to close tabs")
        
        # Style the tab bar for better visual feedback
        self.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #0078d4;
                border-radius: 6px;
                background-color: white;
            }
            
            QTabBar::tab {
                background-color: #f0f0f0;
                color: #333333;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                min-width: 100px;
                font-weight: 500;
            }
            
            QTabBar::tab:selected {
                background-color: #0078d4;
                color: white;
                font-weight: 600;
            }
            
            QTabBar::tab:hover:!selected {
                background-color: #e6f3ff;
                color: #0078d4;
            }
            
            QTabBar::close-button {
                image: none;
                background-color: #ff4444;
                border-radius: 8px;
                margin: 2px;
                min-width: 16px;
                max-width: 16px;
                min-height: 16px;
                max-height: 16px;
            }
            
            QTabBar::close-button:hover {
                background-color: #ff6666;
            }
            
            QTabBar::close-button:pressed {
                background-color: #cc3333;
            }
        """)

    def add_plot_tab(self, widget: QWidget, title: str):
        """Add a new plot tab with improved styling and tooltips."""
        # Create a container widget for better layout
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(widget)
        
        # Add the tab
        tab_index = self.addTab(container, title)
        
        # Set tooltip for the tab
        self.setTabToolTip(tab_index, f"View {title} plot - Click 'X' to close")
        
        # Switch to the new tab
        self.setCurrentIndex(tab_index)
        
        return tab_index
    
    def clear_all_tabs(self):
        """Clear all tabs from the widget with confirmation."""
        if self.count() > 0:
            # Remove all tabs
            while self.count() > 0:
                self.removeTab(0)
    
    def close_current_tab(self):
        """Close the currently active tab."""
        current_index = self.currentIndex()
        if current_index >= 0:
            self.removeTab(current_index)
    
    def get_tab_count(self):
        """Get the number of open tabs."""
        return self.count()
    
    def has_tabs(self):
        """Check if there are any open tabs."""
        return self.count() > 0

    # Optionally, add more methods for advanced tab management 