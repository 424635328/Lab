# backend_scraper.py

import logging
import os
import sys  # 导入 sys 以便判断打包环境
import json
import subprocess

logger = logging.getLogger(__name__)

def get_executable_path(filename):
    """
    [新增] 智能获取可执行文件的路径。
    这对于 PyInstaller 打包至关重要。
    """
    # 如果是在 PyInstaller 打包的应用中运行
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    
    # 在开发环境中运行，假定可执行文件与脚本在同一目录或在PATH中
    # 为了稳定，我们优先检查脚本同级目录
    local_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(local_path):
        return local_path
        
    return filename # 回退到让系统在PATH中查找

def build_sniff_command(url, proxy_dict=None):
    """构建 yt-dlp 嗅探命令列表。"""
    yt_dlp_exe = get_executable_path("yt-dlp.exe") # 在打包时也需要
    command = [yt_dlp_exe, "--dump-json", "--no-warnings", "--encoding", "utf-8", url]
    if proxy_dict and (proxy_url := proxy_dict.get('https://') or proxy_dict.get('http://')):
        command.extend(["--proxy", proxy_url])
    logger.debug(f"构建嗅探命令: {' '.join(command)}")
    return command

def build_download_command(url, format_codes, download_dir, proxy_dict=None):
    """
    [修改] 构建 yt-dlp 下载命令，并明确指定 ffmpeg 的位置。
    """
    yt_dlp_exe = get_executable_path("yt-dlp.exe")
    # 获取 ffmpeg.exe 所在的目录路径
    ffmpeg_dir_path = os.path.dirname(get_executable_path("ffmpeg.exe"))

    command = [
        yt_dlp_exe,
        "-f", format_codes,
        "--output", os.path.join(download_dir, "%(title)s [%(id)s].%(ext)s"),
        "--merge-output-format", "mp4",
        "--no-warnings",
        "--progress",
        # --- [核心修复] ---
        # 明确告诉 yt-dlp 在哪里可以找到 ffmpeg
        "--ffmpeg-location", ffmpeg_dir_path
    ]
    if proxy_dict and (proxy_url := proxy_dict.get('https://') or proxy_dict.get('http://')):
        command.extend(["--proxy", proxy_url])
    command.append(url)
    logger.debug(f"构建下载命令: {' '.join(command)}")
    return command

# (这个文件中的其他函数，如代理、UA等，可以保持不变或删除)