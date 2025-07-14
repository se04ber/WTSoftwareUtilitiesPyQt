from PyQt5.QtWidgets import QTabWidget, QWidget, QVBoxLayout

class ResultsTabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMinimumWidth(400)
        self.tabCloseRequested.connect(self.removeTab)

    def add_plot_tab(self, widget: QWidget, title: str):
        self.addTab(widget, title)

    # Optionally, add more methods for advanced tab management 