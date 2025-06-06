# main.py
import sys
import logging
from PyQt6.QtWidgets import QApplication
from app_window import AppWindow

# (日志配置与上一版相同)

DARK_THEME_QSS = """
/* (大部分QSS与上一版相同) */

#StartButton {
    background-color: #0078d7; /* 蓝色 */
    color: white;
    font-weight: bold;
}
#StartButton:hover { background-color: #1082d7; }
#StartButton:pressed { background-color: #005a9e; }

/* 新增停止按钮样式 */
#StopButton {
    background-color: #c42b1c; /* 红色 */
    color: white;
    font-weight: bold;
}
#StopButton:hover { background-color: #d43b2c; }
#StopButton:pressed { background-color: #a41b0c; }

/* (其余QSS不变) */
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_THEME_QSS)
    window = AppWindow()
    window.show()
    sys.exit(app.exec())