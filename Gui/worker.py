# worker.py

import logging
import re
import subprocess
import os
from urllib.parse import urlparse
from PyQt6.QtCore import QObject, pyqtSignal

# [修改] 导入所有需要的后端函数和新的策略分析器
import backend_scraper
from strategy_profiler import select_best_strategy, AVAILABLE_STRATEGIES, update_experience_data

logger = logging.getLogger(__name__)

# 定义创建进程时使用的标志
if os.name == 'nt':
    CREATION_FLAGS = subprocess.CREATE_NO_WINDOW
else:
    CREATION_FLAGS = 0

class Worker(QObject):
    """
    后台工作线程。现在是一个策略执行器，负责调用策略分析器
    并按顺序执行返回的策略队列，并在成功后更新经验数据。
    """
    # --- 信号定义 (保持不变) ---
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
        """线程启动时的主执行函数。"""
        if not self._is_running: return

        if self.task_type == "sniff":
            self._run_intelligent_sniff()
        elif self.task_type == "download":
            self._run_download()

    def _run_intelligent_sniff(self):
        """[重构] 执行由策略分析器生成的策略队列。"""
        original_url = self.kwargs.get("url")
        self.log.emit(f"后台：启动贝叶斯策略嗅探 -> {original_url}")

        # 1. 获取按优先级排序的策略列表
        strategy_queue = select_best_strategy(original_url)

        if not strategy_queue:
            self.log.emit("<font color='red'>错误：没有找到适用于此URL的嗅探策略。</font>")
            self.sniff_finished.emit({"error": "没有适用的嗅探策略。"}, original_url)
            return

        # 2. 按顺序执行策略，直到成功或队列为空
        final_result = None
        for strategy_name in strategy_queue:
            if not self._is_running:
                final_result = {"error": "操作被用户取消。"}
                break

            self.log.emit(f"<b>策略执行: 尝试使用 '{strategy_name}' 引擎...</b>")
            
            backend_function_name = AVAILABLE_STRATEGIES.get(strategy_name)
            if not backend_function_name:
                self.log.emit(f"<font color='red'>配置错误：策略 '{strategy_name}' 没有对应的后端函数。</font>")
                continue
            
            # 处理“虚拟”的直接链接引擎
            if strategy_name == "direct_link":
                path = urlparse(original_url).path
                filename = os.path.basename(path)
                _, ext = os.path.splitext(path)
                final_result = {
                    "links": [{"url": original_url, "filename": filename, "category": "直接链接", "ext": ext}],
                    "title": filename,
                    "engine": "direct_link"
                }
                self.log.emit(f"<font color='green'>策略 '{strategy_name}' 成功。</font>")
                break
            
            # 动态获取并调用后端函数
            try:
                backend_function = getattr(backend_scraper, backend_function_name)
                result = backend_function(original_url)
            except AttributeError:
                msg = f"代码错误：后端模块中未找到函数 '{backend_function_name}'。"
                self.log.emit(f"<font color='red'>{msg}</font>")
                final_result = {"error": msg}
                break
            except Exception as e:
                result = {"error": f"执行后端函数时发生意外错误: {e}"}

            if result and not result.get("error"):
                self.log.emit(f"<font color='green'>策略 '{strategy_name}' 成功找到资源！</font>")
                final_result = result
                # [新增] 更新经验数据
                try:
                    domain = urlparse(original_url).netloc
                    update_experience_data(domain, strategy_name)
                except Exception as e:
                    logger.warning(f"更新经验数据失败: {e}")
                break
            else:
                error_msg = result.get("error", "未知错误")
                self.log.emit(f"策略 '{strategy_name}' 失败: {error_msg}")

        # 3. 发送最终结果
        if self._is_running:
            if not final_result:
                final_result = {"error": "所有推荐的嗅探策略均已尝试，未能找到任何资源。"}
            self.sniff_finished.emit(final_result, original_url)

    def _run_download(self):
        """根据资源类型调度不同的下载方法。"""
        resource_type = self.kwargs.get("resource_type")
        if resource_type == "yt-dlp":
            self._run_yt_dlp_download()
        else: # "direct"
            self._run_direct_download()
            
    def _run_yt_dlp_download(self):
        """执行并监控一个 yt-dlp 下载子进程。"""
        url, formats, download_path = self.kwargs.get("url"), self.kwargs.get("formats"), self.kwargs.get("download_path")
        
        command = backend_scraper.build_download_command(url, formats, download_path)
        
        try:
            self.process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                encoding='utf-8', 
                errors='ignore',
                creationflags=CREATION_FLAGS
            )
        except Exception as e:
            logger.error(f"启动 yt-dlp 下载进程时出错: {e}")
            self.download_finished.emit(False, f"启动下载失败: {e}")
            return

        try:
            progress_pattern = re.compile(r"download-stream:(\s*\d+\.?\d*%)")
            for line in self.process.stdout:
                if not self._is_running:
                    self.log.emit("检测到停止信号，终止下载...")
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                        self.log.emit("下载进程已成功终止。")
                    except subprocess.TimeoutExpired:
                        self.log.emit("终止进程超时，强制结束。")
                        self.process.kill()
                    self.download_finished.emit(False, "操作被用户取消。")
                    return
                
                line_strip = line.strip()
                if line_strip:
                    self.log.emit(f"[yt-dlp] {line_strip}")
                
                if match := progress_pattern.search(line):
                    try:
                        percentage_str = match.group(1).strip().replace('%', '')
                        percentage = int(float(percentage_str))
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
            logger.error(f"监控下载进程时发生未知错误: {e}")
            if self._is_running:
                self.download_finished.emit(False, f"监控下载时发生错误: {e}")

    def _run_direct_download(self):
        """执行直接链接的下载。"""
        direct_url = self.kwargs.get("direct_url")
        download_path = self.kwargs.get("download_path")
        success, msg = backend_scraper.download_direct_link(direct_url, download_path, progress_callback=self.download_progress.emit)
        if self._is_running:
            self.download_finished.emit(success, msg)
            
    def stop(self):
        """停止当前正在运行的任务。"""
        self.log.emit("后台：正在处理停止请求...")
        self._is_running = False
        if self.process and self.process.poll() is None:
            try:
                self.log.emit(f"尝试终止子进程 ID: {self.process.pid}...")
                self.process.terminate()
            except ProcessLookupError:
                self.log.emit(f"进程 {self.process.pid} 已不存在。")
            except Exception as e:
                logger.error(f"终止进程时发生未知错误: {e}")
                self.log.emit(f"终止进程时出错: {e}")
        else:
            self.log.emit("没有正在运行的子进程可供停止。")