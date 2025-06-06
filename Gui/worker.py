# worker.py
import logging
import subprocess
import json
import os
from PyQt6.QtCore import QObject, pyqtSignal

from backend_scraper import build_sniff_command, build_download_command

logger = logging.getLogger(__name__)

class Worker(QObject):
    sniff_finished = pyqtSignal(dict, str)
    download_finished = pyqtSignal(bool, str) # 返回成功状态和可能的错误信息
    log = pyqtSignal(str)
    # 新增下载进度信号
    download_progress = pyqtSignal(int)

    def __init__(self, task_type, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs
        self.process = None  # 用于持有子进程对象
        self.is_running = True

    def run(self):
        if not self.is_running: return

        if self.task_type == "sniff":
            self._run_sniff()
        elif self.task_type == "download":
            self._run_download()

    def _run_sniff(self):
        url = self.kwargs.get("url")
        command = build_sniff_command(url)
        self.log.emit(f"后台：执行嗅探命令: {' '.join(command)}")
        
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore', creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            stdout, stderr = self.process.communicate()

            if not self.is_running: # 检查是否在中途被停止
                self.sniff_finished.emit({"error": "操作被用户取消。"}, url)
                return

            if self.process.returncode == 0:
                first_line = stdout.strip().split('\n')[0]
                self.sniff_finished.emit(json.loads(first_line), url)
            else:
                self.sniff_finished.emit({"error": stderr.strip()}, url)
        except Exception as e:
            self.sniff_finished.emit({"error": str(e)}, url)

    def _run_download(self):
        url = self.kwargs.get("url")
        formats = self.kwargs.get("formats")
        download_path = self.kwargs.get("download_path")
        command = build_download_command(url, formats, download_path)
        self.log.emit(f"后台：执行下载命令: {' '.join(command)}")

        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore', creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            
            # 实时解析yt-dlp的进度输出
            for line in iter(self.process.stdout.readline, ''):
                if not self.is_running:
                    self.process.terminate()
                    self.process.wait()
                    self.log.emit("下载被用户取消。")
                    self.download_finished.emit(False, "操作被用户取消。")
                    return
                
                # yt-dlp的进度条格式通常是 [download] XX.X% of ...
                if "[download]" in line and "%" in line:
                    try:
                        percentage_str = line.split('%')[0].split()[-1]
                        percentage = int(float(percentage_str))
                        self.download_progress.emit(percentage)
                    except (ValueError, IndexError):
                        pass # 忽略无法解析的行
            
            self.process.wait()
            if self.process.returncode == 0:
                self.download_progress.emit(100)
                self.download_finished.emit(True, "下载成功完成。")
            else:
                self.download_finished.emit(False, f"下载失败，yt-dlp返回码: {self.process.returncode}")

        except Exception as e:
            self.download_finished.emit(False, str(e))

    def stop(self):
        self.log.emit("后台：收到停止信号...")
        self.is_running = False
        if self.process:
            try:
                self.log.emit(f"正在终止进程 ID: {self.process.pid}")
                self.process.terminate() # 优雅地尝试终止
            except ProcessLookupError:
                self.log.emit("进程已不存在。")