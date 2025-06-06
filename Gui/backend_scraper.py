# backend_scraper.py
import logging
import os
import json
import subprocess

logger = logging.getLogger(__name__)

def build_sniff_command(url, proxy_dict=None):
    """[REFACTORED] 构建 yt-dlp 嗅探命令列表。"""
    command = ["yt-dlp", "--dump-json", "--no-warnings", "--encoding", "utf-8", url]
    if proxy_dict and (proxy_url := proxy_dict.get('https://') or proxy_dict.get('http://')):
        command.extend(["--proxy", proxy_url])
    logger.debug(f"构建嗅探命令: {' '.join(command)}")
    return command

def build_download_command(url, format_codes, download_dir, proxy_dict=None):
    """[REFACTORED] 构建 yt-dlp 下载命令列表。"""
    command = [
        "yt-dlp",
        "-f", format_codes,
        "--output", os.path.join(download_dir, "%(title)s [%(id)s][%(format_id)s].%(ext)s"),
        "--merge-output-format", "mp4",
        "--no-warnings",
        "--progress",
    ]
    if proxy_dict and (proxy_url := proxy_dict.get('https://') or proxy_dict.get('http://')):
        command.extend(["--proxy", proxy_url])
    command.append(url)
    logger.debug(f"构建下载命令: {' '.join(command)}")
    return command