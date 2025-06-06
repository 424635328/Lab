# app_window.py
import logging
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QMessageBox, QTextEdit, QSplitter, QTreeWidget, QTreeWidgetItem, QHeaderView,
    QStatusBar, QMenu, QFileDialog, QTreeWidgetItemIterator, QApplication, QLabel, QStyle
)
from PyQt6.QtCore import QThread, QSettings, QDir, Qt, QPoint
from PyQt6.QtGui import QFont, QIcon, QAction, QBrush, QColor

from worker import Worker  # 确保 worker.py 在同一目录下

logger = logging.getLogger(__name__)

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("终极资源嗅探下载器")
        self.setGeometry(100, 100, 1200, 800)

        # --- 初始化状态和设置 ---
        self.worker = None
        self.thread = None
        self.settings = QSettings("MyCompany", "UltimateSnifferGUI")
        self.current_task_data = {}

        self.setup_ui()
        self.connect_signals()
        self.load_settings()
        self.set_controls_for_idle()

    def setup_ui(self):
        """使用代码构建UI界面"""
        # --- 创建控件 ---
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("在此处粘贴要嗅探的URL")
        self.url_input.setFont(QFont("Segoe UI", 10))

        self.action_button_layout = QHBoxLayout()
        self.sniff_button = QPushButton(" 嗅探资源")
        self.download_button = QPushButton(" 下载选中项")
        self.stop_button = QPushButton(" 停止操作")
        
        self.download_button.setObjectName("StartButton")
        self.stop_button.setObjectName("StopButton")
        
        self.action_button_layout.addWidget(self.sniff_button)
        self.action_button_layout.addWidget(self.download_button)
        self.action_button_layout.addWidget(self.stop_button)

        self.task_tree = QTreeWidget()
        self.task_tree.setHeaderLabels(["任务URL", "标题"])
        self.task_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.task_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.task_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.resource_tree = QTreeWidget()
        self.resource_tree.setHeaderLabels(["格式", "编码", "分辨率", "大小", "备注"])
        self.resource_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.resource_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.resource_tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.resource_tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)

        self.path_input = QLineEdit()
        self.path_input.setReadOnly(True)
        self.browse_button = QPushButton(" 浏览...")

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Courier New", 9))
        
        # --- [修正] 设置图标 ---
        style = self.style()
        self.sniff_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))
        self.download_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_ArrowDown))
        self.stop_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.browse_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))

        # --- 布局 ---
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("URL:"))
        url_layout.addWidget(self.url_input, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addLayout(url_layout)
        left_layout.addWidget(self.task_tree)

        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("下载到:"))
        path_layout.addWidget(self.path_input, 1)
        path_layout.addWidget(self.browse_button)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self.resource_tree)
        right_layout.addLayout(path_layout)
        right_layout.addLayout(self.action_button_layout)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([450, 750])

        bottom_splitter = QSplitter(Qt.Orientation.Vertical)
        bottom_splitter.addWidget(main_splitter)
        bottom_splitter.addWidget(self.log_output)
        bottom_splitter.setSizes([600, 200])

        self.setCentralWidget(bottom_splitter)
        self.setStatusBar(QStatusBar(self))

    def connect_signals(self):
        self.sniff_button.clicked.connect(self.start_sniffing)
        self.url_input.returnPressed.connect(self.start_sniffing)
        self.task_tree.currentItemChanged.connect(self.display_resources)
        self.task_tree.customContextMenuRequested.connect(self.show_task_context_menu)
        self.browse_button.clicked.connect(self.browse_path)
        self.download_button.clicked.connect(self.start_downloading)
        self.stop_button.clicked.connect(self.stop_task)

    # --- [补全] 以下是所有被遗漏的辅助和事件处理函数 ---
    
    def load_settings(self):
        default_path = QDir.home().filePath("Downloads")
        path = self.settings.value("downloadPath", default_path)
        self.path_input.setText(path)

    def save_settings(self):
        self.settings.setValue("downloadPath", self.path_input.text())
        
    def browse_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择下载文件夹", self.path_input.text())
        if path:
            self.path_input.setText(path)

    def start_sniffing(self):
        url = self.url_input.text().strip()
        if not url: return

        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "提示", "已有任务在进行中，请稍候。")
            return

        self.set_controls_for_busy("正在嗅探...")
        self.log_output.append(f"<b>开始嗅探: {url}</b>")
        
        self.thread = QThread()
        worker = Worker("sniff", url=url)
        self.worker = worker
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.sniff_finished.connect(self.on_sniff_finished)
        self.worker.log.connect(self.log_output.append)
        
        self.thread.start()

    def on_sniff_finished(self, data, url):
        self.set_controls_for_idle()

        if data.get("error"):
            error_msg = data['error']
            self.log_output.append(f"<font color='red'>嗅探失败: {error_msg}</font>")
            QMessageBox.critical(self, "嗅探失败", error_msg)
        else:
            self.log_output.append(f"<font color='green'>嗅探成功: {data.get('title', '无标题')}</font>")
            if url not in self.current_task_data:
                self.current_task_data[url] = data
                task_item = QTreeWidgetItem(self.task_tree, [url, data.get('title', 'N/A')])
                self.task_tree.setCurrentItem(task_item)
        
        self.url_input.clear()
        self.thread.quit()
        self.thread.wait()

    def display_resources(self, current_item, previous_item):
        self.resource_tree.clear()
        if not current_item: return

        url = current_item.text(0)
        data = self.current_task_data.get(url)
        if not data: return
        
        style = self.style()
        video_icon = style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        audio_icon = style.standardIcon(QStyle.StandardPixmap.SP_MediaVolume)
        
        video_root = QTreeWidgetItem(self.resource_tree, ["视频流"])
        audio_root = QTreeWidgetItem(self.resource_tree, ["音频流"])
        
        video_root.setIcon(0, video_icon)
        audio_root.setIcon(0, audio_icon)

        for f in data.get("formats", []):
            is_video_only = f.get('vcodec') != 'none' and f.get('acodec') == 'none'
            is_audio_only = f.get('vcodec') == 'none' and f.get('acodec') != 'none'
            
            filesize = f"{(f.get('filesize') or f.get('filesize_approx', 0)) / 1024 / 1024:.2f} MB" if (f.get('filesize') or f.get('filesize_approx')) else "N/A"
            item_text = [
                f.get('format_note', f.get('format_id', 'N/A')),
                f"{f.get('vcodec', 'none')} / {f.get('acodec', 'none')}",
                f.get('resolution', '纯音频'),
                filesize,
                f.get('ext', 'N/A')
            ]
            
            parent = video_root if is_video_only else (audio_root if is_audio_only else video_root)
            
            item = QTreeWidgetItem(parent, item_text)
            item.setData(0, Qt.ItemDataRole.UserRole, f.get('format_id'))
            item.setCheckState(0, Qt.CheckState.Unchecked)
        
        self.resource_tree.expandAll()

    def start_downloading(self):
        selected_items = []
        iterator = QTreeWidgetItemIterator(self.resource_tree)
        while iterator.value():
            item = iterator.value()
            if item.checkState(0) == Qt.CheckState.Checked:
                format_id = item.data(0, Qt.ItemDataRole.UserRole)
                if format_id:
                    selected_items.append(format_id)
            iterator += 1

        if not selected_items:
            QMessageBox.warning(self, "提示", "请先在资源列表中勾选要下载的格式。")
            return
        
        format_string = "+".join(selected_items)
        current_task_item = self.task_tree.currentItem()
        if not current_task_item: return
        url = current_task_item.text(0)
        
        self.set_controls_for_busy("正在下载...")
        self.log_output.append(f"<b>开始下载: {url}，格式: {format_string}</b>")

        self.thread = QThread()
        worker = Worker("download", url=url, formats=format_string, download_path=self.path_input.text())
        self.worker = worker
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.download_finished.connect(self.on_download_finished)
        self.worker.download_progress.connect(self.update_progress)
        self.worker.log.connect(self.log_output.append)

        self.thread.start()

    def on_download_finished(self, success, message):
        self.set_controls_for_idle()
        if success:
            self.log_output.append(f"<font color='green'>{message}</font>")
            QMessageBox.information(self, "成功", message)
        else:
            self.log_output.append(f"<font color='red'>下载失败: {message}</font>")
            QMessageBox.critical(self, "失败", f"下载任务失败:\n{message}")
        
        self.thread.quit()
        self.thread.wait()

    def stop_task(self):
        if self.thread and self.thread.isRunning() and self.worker:
            self.statusBar().showMessage("正在发送停止信号...", 3000)
            self.log_output.append("<b>[用户操作] 发送停止信号...</b>")
            self.worker.stop()
        else:
            self.log_output.append("当前没有正在运行的任务可停止。")
            self.set_controls_for_idle()

    def update_progress(self, value):
        self.statusBar().showMessage(f"下载进度: {value}%")

    def show_task_context_menu(self, position: QPoint):
        item = self.task_tree.itemAt(position)
        if not item: return

        menu = QMenu()
        style = self.style()
        remove_action = menu.addAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton), "移除此任务")
        copy_url_action = menu.addAction(style.standardIcon(QStyle.StandardPixmap.SP_FileLinkIcon), "复制URL")
        
        action = menu.exec(self.task_tree.mapToGlobal(position))
        
        if action == remove_action:
            self.remove_task(item)
        elif action == copy_url_action:
            QApplication.clipboard().setText(item.text(0))
            self.statusBar().showMessage("URL已复制到剪贴板", 2000)

    def remove_task(self, item):
        url = item.text(0)
        self.current_task_data.pop(url, None)
        self.task_tree.takeTopLevelItem(self.task_tree.indexOfTopLevelItem(item))
        self.resource_tree.clear()

    def set_controls_for_idle(self):
        """设置UI为闲置状态"""
        self.sniff_button.setVisible(True)
        self.download_button.setVisible(True)
        self.stop_button.setVisible(False)
        self.url_input.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.task_tree.setEnabled(True)
        self.statusBar().showMessage("准备就绪")

    def set_controls_for_busy(self, message):
        """设置UI为忙碌状态"""
        self.sniff_button.setVisible(False)
        self.download_button.setVisible(False)
        self.stop_button.setVisible(True)
        self.url_input.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.task_tree.setEnabled(False)
        self.statusBar().showMessage(message)

    def closeEvent(self, event):
        self.save_settings()
        if self.thread and self.thread.isRunning():
            reply = QMessageBox.question(self, "确认退出", "任务仍在进行中，确定要退出吗？",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                if self.worker: self.worker.stop()
                self.thread.quit()
                self.thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            super().closeEvent(event)