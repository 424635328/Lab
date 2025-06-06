# main.py
import sys
import logging
from PyQt6.QtWidgets import QApplication
from app_window import AppWindow

# (日志配置与上一版相同)

# --- [全新] 精致Emo朋克 (Polished Emo-Punk) QSS 样式表 ---
POLISHED_EMO_PUNK_QSS = """
/* --- 全局与字体 --- */
QWidget {
    background-color: transparent;
    color: #abb2bf; /* 柔和的灰白色 */
    font-family: "Inter", "Segoe UI", "Roboto", sans-serif; /* 现代、清晰的字体 */
    font-size: 10pt;
}

QMainWindow {
    background-color: #1e2127; /* 更深的蓝黑背景 */
}

/* --- 容器与分组 (使用QFrame) --- */
QFrame#UrlFrame, QFrame#DownloadFrame {
    background-color: rgba(44, 48, 58, 0.5); /* 半透明背景，模拟毛玻璃 */
    border-radius: 8px;
}

/* --- 标签 --- */
QLabel {
    color: #6d7789;
    font-weight: bold;
    padding-left: 3px;
}

/* --- 输入框、树、日志 --- */
QLineEdit, QTextEdit, QTreeWidget {
    background-color: #282c34;
    color: #dbe0e8;
    border: 1px solid #3a3f4b;
    border-radius: 6px;
    padding: 8px;
}

QLineEdit:focus, QTreeWidget:focus {
    border: 1px solid #61afef; /* 清澈的蓝色焦点 */
    background-color: #2c313a;
}

QTextEdit {
    font-family: "Consolas", "Courier New", monospace;
    font-size: 9.5pt;
}

/* --- 树形控件美化 --- */
QTreeWidget::item {
    padding: 6px;
    border-radius: 4px;
}
QTreeWidget::item:hover {
    background-color: rgba(97, 175, 239, 0.1); /* 悬停时为蓝色微光 */
}
QTreeWidget::item:selected {
    background-color: #61afef; /* 选中行为清澈蓝 */
    color: #1e2127;
}
QTreeWidget::branch:has-children:!has-siblings:closed,
QTreeWidget::branch:closed:has-children:has-siblings {
    border-image: none;
    image: url(none); /* 使用自定义或无图标 */
}
QTreeWidget::branch:open:has-children:!has-siblings,
QTreeWidget::branch:open:has-children:has-siblings {
    border-image: none;
    image: url(none); /* 使用自定义或无图标 */
}


/* --- 按钮 --- */
QPushButton {
    background-color: #3d4350;
    color: #c8cdd6;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #4a5160;
}

QPushButton:pressed {
    background-color: #353a45;
}

/* --- 特殊按钮：嗅探、下载、停止 --- */
#SniffButton, #StartButton {
    background-color: #61afef; /* 清澈蓝 */
    color: #1e2127;
}
#SniffButton:hover, #StartButton:hover {
    background-color: #71b9f9;
}

#StopButton {
    background-color: #e06c75; /* 柔和的红色 */
    color: #1e2127;
}
#StopButton:hover {
    background-color: #f07c85;
}

QPushButton:disabled {
    background-color: #2c313a;
    color: #5c6370;
}

/* --- 复选框 --- */
QCheckBox {
    color: #abb2bf;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #3a3f4b;
    border-radius: 5px;
}
QCheckBox::indicator:hover {
    border-color: #61afef;
}
QCheckBox::indicator:checked {
    background-color: #98c379; /* 选中时为清新的绿色 */
    border-color: #98c379;
}

/* --- 进度条 --- */
QProgressBar {
    height: 10px;
    border: none;
    border-radius: 5px;
    background-color: #2c313a;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #98c379; /* 清新绿进度 */
    border-radius: 5px;
}

/* --- 表头与分割条 --- */
QHeaderView::section {
    background-color: transparent;
    color: #6d7789;
    padding: 8px;
    border: none;
    border-bottom: 2px solid #2c313a;
    font-size: 9pt;
    text-transform: uppercase;
}

QSplitter::handle {
    background-color: #2c313a;
    width: 3px;
}
QSplitter::handle:hover {
    background-color: #61afef;
}
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(POLISHED_EMO_PUNK_QSS)
    window = AppWindow()
    window.show()
    sys.exit(app.exec())