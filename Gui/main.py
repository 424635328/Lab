# main.py
import sys
import logging
from PyQt6.QtWidgets import QApplication
from app_window import AppWindow

# --- 日志基础配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sniffer_gui.log", encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)

# --- [全新] Emo朋克 / 忧郁霓虹 (Melancholy Neon) QSS 样式表 ---
MELANCHOLY_NEON_QSS = """
/* --- 全局样式 --- */
QWidget {
    background-color: #161a21; /* 午夜蓝黑背景 */
    color: #bdc3c7; /* 柔和的银灰色文本 */
    font-family: "Roboto", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 10pt;
    border: none;
}

QMainWindow {
    background-color: #0e1116;
}

/* --- 标签与文本 --- */
QLabel {
    color: #7f8c8d; /* 石板灰 */
    padding-left: 3px;
    font-weight: bold;
}

/* --- 输入框、树、日志 --- */
QLineEdit, QTextEdit, QTreeWidget {
    background-color: #1f242d;
    color: #ecf0f1;
    border: 1px solid #2c3e50; /* 深邃的蓝灰色边框 */
    border-radius: 4px;
    padding: 8px;
    selection-background-color: #8e44ad; /* 选中文字的背景：忧郁紫 */
}

QLineEdit:focus, QTextEdit:focus, QTreeWidget:focus {
    border: 1px solid #3498db; /* 焦点时为电光蓝 */
}

QTreeWidget::item {
    padding: 4px 0;
}

QTreeWidget::item:hover {
    background-color: rgba(52, 152, 219, 0.1); /* 悬停时为半透明的电光蓝 */
}

QTreeWidget::item:selected {
    background-color: #8e44ad; /* 选中行为忧郁紫 */
    color: #ffffff;
}

/* --- 按钮 --- */
QPushButton {
    background-color: #2c3e50; /* 深蓝灰 */
    color: #ecf0f1;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
    outline: none;
}

QPushButton:hover {
    background-color: #34495e;
}

QPushButton:pressed {
    background-color: #2c3e50;
    padding-top: 9px;
    padding-bottom: 7px;
}

/* --- 特殊按钮：嗅探/下载 --- */
#StartButton, QPushButton[text=" 嗅探资源"] {
    background-color: #8e44ad; /* 忧郁紫 */
    color: white;
}
#StartButton:hover, QPushButton[text=" 嗅探资源"]:hover {
    background-color: #9b59b6;
}
#StartButton:pressed, QPushButton[text=" 嗅探资源"]:pressed {
    background-color: #8e44ad;
}

/* --- 特殊按钮：停止 --- */
#StopButton {
    background-color: #c0392b; /* 偏暗的石榴红 */
    color: white;
}
#StopButton:hover {
    background-color: #e74c3c;
}
#StopButton:pressed {
    background-color: #c0392b;
}

#StartButton:disabled, #StopButton:disabled, QPushButton:disabled {
    background-color: #2a2a2a;
    color: #666;
}

/* --- 复选框 --- */
QCheckBox {
    color: #95a5a6;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #2c3e50;
    border-radius: 9px; /* 圆形 */
    background-color: #1f242d;
}
QCheckBox::indicator:hover {
    border-color: #3498db;
}
QCheckBox::indicator:checked {
    background-color: #8e44ad;
    border-color: #8e44ad;
}

/* --- 进度条 --- */
QProgressBar {
    border: none;
    border-radius: 5px;
    text-align: center;
    color: rgba(255, 255, 255, 0.8);
    background-color: #1f242d;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: #8e44ad; /* 忧郁紫进度 */
    border-radius: 5px;
}

/* --- 表头 --- */
QHeaderView::section {
    background-color: #1f242d;
    color: #7f8c8d;
    padding: 8px;
    border: none;
    border-bottom: 2px solid #2c3e50;
}

/* --- 分割条 --- */
QSplitter::handle {
    background-color: #161a21;
    width: 1px;
}
QSplitter::handle:hover {
    background-color: #3498db;
}

/* --- 滚动条 --- */
QScrollBar:vertical, QScrollBar:horizontal {
    border: none;
    background: #1f242d;
    width: 8px;
    height: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #2c3e50;
    min-height: 25px;
    min-width: 25px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: #34495e;
}

/* --- 右键菜单 --- */
QMenu {
    background-color: #1f242d;
    border: 1px solid #2c3e50;
    padding: 5px;
}
QMenu::item {
    padding: 8px 25px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #8e44ad;
}
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 应用全新的QSS样式
    app.setStyleSheet(MELANCHOLY_NEON_QSS)
    
    # 创建并显示主窗口
    window = AppWindow()
    window.show()
    
    # 启动事件循环
    sys.exit(app.exec())