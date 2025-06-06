# worker.py
import logging
import re
import subprocess
import os
from PyQt6.QtCore import QObject, pyqtSignal

from backend_scraper import (
    sniff_with_yt_dlp, 
    sniff_with_deep_html_parser,
    sniff_github_release_api,
    build_download_command,
    download_direct_link
)

logger = logging.getLogger(__name__)

class Worker(QObject):
    sniff_finished = pyqtSignal(dict, str)
    download_finished = pyqtSignal(bool, str)
    download_progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, task_type, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs
        self.process = None
        self._is_running = True

    def run(self):
        if not self._is_running: return

        if self.task_type == "sniff":
            self._run_sniff()
        elif self.task_type == "download":
            self._run_download()

    def _run_sniff(self):
        url = self.kwargs.get("url")
        self.log.emit(f"后台：开始多策略嗅探 {url}...")
        
        result = None

        # 策略 1: 检查是否是 GitHub Release 页面
        if "github.com" in url and "/releases/tag/" in url:
            self.log.emit("检测到 GitHub Release 页面，使用专用API嗅探器...")
            result = sniff_github_release_api(url)
        else:
            # 策略 2: 默认的双引擎嗅探 (yt-dlp -> HTML)
            self.log.emit("阶段 1: 尝试 yt-dlp 引擎...")
            result = sniff_with_yt_dlp(url)
            
            is_unsupported_url = "unsupported url" in result.get("error", "").lower()
            if self._is_running and result.get("error") and is_unsupported_url:
                self.log.emit("yt-dlp 不支持，切换到HTML深度嗅探引擎作为备用方案...")
                result = sniff_with_deep_html_parser(url)

        if self._is_running:
            self.sniff_finished.emit(result, url)

    def _run_download(self):
        resource_type = self.kwargs.get("resource_type")
        
        if resource_type == "yt-dlp":
            self._run_yt_dlp_download()
        else: # "direct"
            self._run_direct_download()
            
    def _run_yt_dlp_download(self):
        url = self.kwargs.get("url")
        formats = self.kwargs.get("formats")
        download_path = self.kwargs.get("download_path")
        
        command = build_download_command(url, formats, download_path)
        
        try:
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                text=True, encoding='utf-8', errors='ignore',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
        except Exception as e:
            logger.error(f"启动 yt-dlp 下载进程失败: {e}")
            self.download_finished.emit(False, f"启动下载失败: {e}")
            return

        try:
            progress_pattern = re.compile(r"(\d+(\.\d+)?%)")
            for line in iter(self.process.stdout.readline, ''):
                if not self._is_running:
                    self.log.emit("检测到停止信号，终止下载...")
                    self.process.terminate()
                    self.process.wait(timeout=5)
                    self.download_finished.emit(False, "操作被用户取消。")
                    return
                
                self.log.emit(f"[yt-dlp] {line.strip()}")
                match = progress_pattern.search(line)
                if match:
                    try:
                        percentage = int(float(match.group(1).replace('%', '')))
                        self.download_progress.emit(percentage)
                    except (ValueError, IndexError):
                        pass
            
            return_code = self.process.wait()
            if self._is_running:
                if return_code == 0:
                    self.download_progress.emit(100)
                    self.download_finished.emit(True, "下载成功完成。")
                else:
                    self.download_finished.emit(False, f"下载失败，进程返回码: {return_code}")
        except Exception as e:
            logger.error(f"监控下载进程时出错: {e}")
            if self._is_running:
                self.download_finished.emit(False, f"监控下载时发生错误: {e}")

    def _run_direct_download(self):
        direct_url = self.kwargs.get("direct_url")
        download_path = self.kwargs.get("download_path")
        # 直接下载逻辑是阻塞的，未来可以改写成非阻塞
        success, msg = download_direct_link(direct_url, download_path, progress_callback=self.download_progress.emit)
        if self._is_running:
            self.download_finished.emit(success, msg)
            
    def stop(self):
        self._is_running = False
        self.log.emit("后台：收到停止信号...")
        if self.process and self.process.poll() is None:
            try:
                self.log.emit(f"正在终止进程 ID: {self.process.pid}")
                self.process.terminate()
            except Exception as e:
                self.log.emit(f"终止进程时出错: {e}")