import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
import io
import pandas as pd
import os
from typing import List

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 