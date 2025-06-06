# app_window.py
import logging
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QMessageBox, QTextEdit, QSplitter, QTreeWidget, QTreeWidgetItem, QHeaderView,
    QStatusBar, QMenu, QFileDialog, QTreeWidgetItemIterator, QCheckBox, QApplication, QStyle,QLabel
)
from PyQt6.QtCore import QThread, QSettings, QDir, Qt, QPoint
from PyQt6.QtGui import QFont, QIcon, QAction, QBrush, QColor

from worker import Worker

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("终极资源嗅探下载器 v2.1")
        self.setGeometry(100, 100, 1200, 800)

        self.download_queue = []
        self.current_download_info = {}
        self.worker = None
        self.thread = None
        self.settings = QSettings("MyCompany", "UltimateSnifferGUI")
        self.current_task_data = {}

        self.setup_ui()
        self.connect_signals()
        self.load_settings()
        self.set_controls_for_idle()

    def setup_ui(self):
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("在此处粘贴要嗅探的URL")
        self.url_input.setFont(QFont("Segoe UI", 10))
        self.sniff_button = QPushButton(" 嗅探资源")
        
        self.task_tree = QTreeWidget()
        self.task_tree.setHeaderLabels(["任务URL", "标题"])
        self.task_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.task_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.task_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.resource_tree = QTreeWidget()
        self.resource_tree.setHeaderLabels(["资源/格式", "类型/编码", "分辨率", "大小", "链接/备注"])
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

        self.merge_audio_checkbox = QCheckBox("自动合并最佳音轨 (仅对媒体流有效)")
        self.merge_audio_checkbox.setChecked(True)
        self.merge_audio_checkbox.setToolTip("勾选后，仅需选择视频流，程序将为每个视频流自动匹配最佳音轨进行合并。\n对直接下载链接无效。")
        
        self.download_button = QPushButton(" 下载选中项")
        self.download_button.setObjectName("StartButton")
        self.stop_button = QPushButton(" 停止操作")
        self.stop_button.setObjectName("StopButton")

        style = self.style()
        self.sniff_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))
        self.download_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_ArrowDown))
        self.stop_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.browse_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))

        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("URL:"))
        url_layout.addWidget(self.url_input, 1)
        url_layout.addWidget(self.sniff_button)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addLayout(url_layout)
        left_layout.addWidget(self.task_tree)

        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("下载到:"))
        path_layout.addWidget(self.path_input, 1)
        path_layout.addWidget(self.browse_button)
        
        download_options_layout = QHBoxLayout()
        download_options_layout.addWidget(self.merge_audio_checkbox)
        download_options_layout.addStretch()
        download_options_layout.addWidget(self.download_button)
        download_options_layout.addWidget(self.stop_button)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self.resource_tree)
        right_layout.addLayout(path_layout)
        right_layout.addLayout(download_options_layout)

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
        self.download_button.clicked.connect(self.prepare_downloads)
        self.stop_button.clicked.connect(self.stop_task)

    def start_sniffing(self):
        url = self.url_input.text().strip()
        if not url: return
        if self.thread and self.thread.isRunning(): return

        self.set_controls_for_busy("正在嗅探...")
        self.log_output.append(f"<b>开始嗅探: {url}</b>")
        
        self.thread = QThread()
        self.worker = Worker("sniff", url=url)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.sniff_finished.connect(self.on_sniff_finished)
        self.worker.log.connect(self.log_output.append)
        self.thread.start()

    def on_sniff_finished(self, data, url):
        self.set_controls_for_idle()

        existing_items = self.task_tree.findItems(url, Qt.MatchFlag.MatchExactly, 0)
        task_item = existing_items[0] if existing_items else QTreeWidgetItem(self.task_tree, [url, "正在获取标题..."])
        
        self.current_task_data[url] = data
        if data.get("error"):
            error_msg = data['error']
            self.log_output.append(f"<font color='red'>嗅探失败: {error_msg}</font>")
            task_item.setText(1, "[嗅探失败]")
            task_item.setForeground(1, QBrush(QColor("#ffc107")))
        else:
            title = data.get('title', '无标题')
            self.log_output.append(f"<font color='green'>嗅探成功: {title}</font>")
            task_item.setText(1, title)
            task_item.setForeground(1, QBrush(self.palette().text().color()))
        
        if not existing_items:
            self.task_tree.addTopLevelItem(task_item)
        self.task_tree.setCurrentItem(task_item)
        
        self.url_input.clear()
        self.thread.quit()
        self.thread.wait()
        self.thread, self.worker = None, None

    def display_resources(self, current_item, previous_item):
        self.resource_tree.clear()
        if not current_item: return

        url = current_item.text(0)
        data = self.current_task_data.get(url)
        if not data or data.get("error"): return
        
        engine = data.get("engine")
        if engine == "yt-dlp": self.display_yt_dlp_resources(data)
        elif engine == "html": self.display_html_resources(data)

    def display_yt_dlp_resources(self, data):
        style = self.style()
        video_icon = style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        audio_icon = style.standardIcon(QStyle.StandardPixmap.SP_MediaVolume)
        video_root = QTreeWidgetItem(self.resource_tree, ["视频流"]); video_root.setIcon(0, video_icon)
        audio_root = QTreeWidgetItem(self.resource_tree, ["音频流"]); audio_root.setIcon(0, audio_icon)

        for f in data.get("formats", []):
            is_video = f.get('vcodec') != 'none'
            is_audio = f.get('acodec') != 'none'
            filesize = f"{(f.get('filesize') or f.get('filesize_approx', 0)) / 1024 / 1024:.2f} MB" if (f.get('filesize') or f.get('filesize_approx')) else "N/A"
            item_text = [f.get('format_note', f.get('format_id', 'N/A')), f"{f.get('vcodec', 'none')} / {f.get('acodec', 'none')}", f.get('resolution', '纯音频'), filesize, f.get('ext', 'N/A')]
            
            parent = video_root if is_video else audio_root
            item = QTreeWidgetItem(parent, item_text)
            item.setData(0, Qt.ItemDataRole.UserRole, {"type": "yt-dlp", "format_id": f.get('format_id')})
            item.setCheckState(0, Qt.CheckState.Unchecked)
        self.resource_tree.expandAll()

    def display_html_resources(self, data):
        category_roots = {}
        for link in data.get("links", []):
            category_name = link.get("category", "其他")
            if category_name not in category_roots:
                category_roots[category_name] = QTreeWidgetItem(self.resource_tree, [category_name])

            parent = category_roots[category_name]
            filesize_mb = f"{link.get('size') / 1024 / 1024:.2f} MB" if isinstance(link.get('size'), int) else "未知"
            item_text = [link.get("filename", "N/A"), link.get("mime", link.get("ext")), "", filesize_mb, link.get("url")]
            item = QTreeWidgetItem(parent, item_text)
            item.setCheckState(0, Qt.CheckState.Unchecked)
            item.setData(0, Qt.ItemDataRole.UserRole, {"type": "direct", "url": link.get("url")})
        self.resource_tree.expandAll()

    def prepare_downloads(self):
        if self.thread and self.thread.isRunning(): return
        current_task_item = self.task_tree.currentItem()
        if not current_task_item: return
        self.current_download_info = {"base_url": current_task_item.text(0)}
        
        self.download_queue = []
        yt_dlp_formats, auto_merge = [], self.merge_audio_checkbox.isChecked()
        iterator = QTreeWidgetItemIterator(self.resource_tree)
        while iterator.value():
            item = iterator.value()
            if item.checkState(0) == Qt.CheckState.Checked:
                download_info = item.data(0, Qt.ItemDataRole.UserRole)
                if not download_info: continue
                
                if download_info["type"] == "direct": self.download_queue.append(download_info)
                elif download_info["type"] == "yt-dlp":
                    if auto_merge and item.parent() and item.parent().text(0) == "视频流":
                        self.download_queue.append({"type": "yt-dlp", "format_id": f"{download_info['format_id']}+bestaudio"})
                    elif not auto_merge:
                        yt_dlp_formats.append(download_info['format_id'])
            iterator += 1
        
        if not auto_merge and yt_dlp_formats:
            self.download_queue.append({"type": "yt-dlp", "format_id": "+".join(yt_dlp_formats)})
        if not self.download_queue: return
            
        self.log_output.append(f"<b>准备下载 {len(self.download_queue)} 个项目...</b>")
        self.process_next_in_queue()

    def process_next_in_queue(self):
        if not self.download_queue:
            self.log_output.append("<font color='green'><b>所有下载任务已处理完毕！</b></font>")
            self.set_controls_for_idle()
            QMessageBox.information(self, "完成", "所有下载任务已处理完毕！")
            return

        task = self.download_queue.pop(0)
        status_msg = f"正在下载 (队列剩余 {len(self.download_queue)} 个)..."
        self.set_controls_for_busy(status_msg)
        
        task_type = task["type"]
        if task_type == "direct":
            worker_kwargs = {"resource_type": "direct", "direct_url": task['url'], "download_path": self.path_input.text()}
            self.log_output.append(f"<b>开始直接下载: {task['url']}</b>")
        else: # yt-dlp
            worker_kwargs = {"resource_type": "yt-dlp", "url": self.current_download_info['base_url'], "formats": task['format_id'], "download_path": self.path_input.text()}
            self.log_output.append(f"<b>开始yt-dlp下载: {self.current_download_info['base_url']}，格式: {task['format_id']}</b>")

        self.thread = QThread()
        self.worker = Worker("download", **worker_kwargs)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.download_finished.connect(self.on_single_download_finished)
        self.worker.download_progress.connect(self.update_progress)
        self.worker.log.connect(self.log_output.append)
        self.thread.start()

    def on_single_download_finished(self, success, message):
        if success: self.log_output.append(f"<font color='green'>项目下载成功。</font>")
        else: self.log_output.append(f"<font color='red'>项目下载失败: {message}</font>")
        self.thread.quit()
        self.thread.wait()
        self.thread, self.worker = None, None
        self.process_next_in_queue()

    def stop_task(self):
        if self.worker:
            self.statusBar().showMessage("正在发送停止信号...", 3000)
            self.log_output.append("<b>[用户操作] 发送停止信号...</b>")
            self.worker.stop()
        else: self.set_controls_for_idle()

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
        if action == remove_action: self.remove_task(item)
        elif action == copy_url_action: QApplication.clipboard().setText(item.text(0))

    def remove_task(self, item):
        url = item.text(0)
        self.current_task_data.pop(url, None)
        self.task_tree.takeTopLevelItem(self.task_tree.indexOfTopLevelItem(item))
        self.resource_tree.clear()

    def set_controls_for_idle(self):
        self.sniff_button.setVisible(True)
        self.download_button.setVisible(True)
        self.merge_audio_checkbox.setVisible(True)
        self.stop_button.setVisible(False)
        for w in [self.url_input, self.browse_button, self.task_tree, self.resource_tree, self.merge_audio_checkbox]: w.setEnabled(True)
        self.statusBar().showMessage("准备就绪")

    def set_controls_for_busy(self, message):
        self.sniff_button.setVisible(False)
        self.download_button.setVisible(False)
        self.merge_audio_checkbox.setVisible(False)
        self.stop_button.setVisible(True)
        for w in [self.url_input, self.browse_button, self.task_tree, self.resource_tree]: w.setEnabled(False)
        self.statusBar().showMessage(message)

    def browse_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择下载文件夹", self.path_input.text())
        if path: self.path_input.setText(path)

    def load_settings(self):
        default_path = QDir.home().filePath("Downloads")
        path = self.settings.value("downloadPath", default_path)
        self.path_input.setText(path)
        self.merge_audio_checkbox.setChecked(self.settings.value("autoMergeAudio", True, type=bool))

    def save_settings(self):
        self.settings.setValue("downloadPath", self.path_input.text())
        self.settings.setValue("autoMergeAudio", self.merge_audio_checkbox.isChecked())

    def closeEvent(self, event):
        self.save_settings()
        if self.thread and self.thread.isRunning():
            reply = QMessageBox.question(self, "确认退出", "任务仍在进行中，确定要退出吗？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                if self.worker: self.worker.stop()
                self.thread.quit()
                self.thread.wait()
                event.accept()
            else: event.ignore()
        else: super().closeEvent(event)