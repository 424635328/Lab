# backend_scraper.py

import logging
import os
import sys
import json
import subprocess
import re
import time
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# --- 资源类型定义 ---
RESOURCE_CATEGORIES = {
    "视频": ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm', '.ts', '.m3u8'),
    "音频": ('.mp3', '.m4a', '.wav', '.aac', '.flac', '.ogg'),
    "图片": ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'),
    "压缩包": ('.zip', '.rar', '.7z', '.tar', '.gz', '.iso'),
    "可执行/安装包": ('.exe', '.msi', '.dmg', '.pkg', '.deb', '.rpm'),
    "文档": ('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt', '.md'),
}

# --- 核心: 智能路径定位函数 ---
def get_executable_path(filename):
    """
    获取可执行文件的正确路径，优先使用程序自带的版本。
    这对于开发和 PyInstaller 打包都至关重要。
    """
    # 1. 如果是在 PyInstaller 打包的应用中运行
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    
    # 2. 在开发环境中，优先查找与主启动脚本(main.py)同级的目录
    try:
        base_path = os.path.dirname(sys.argv[0])
        local_path = os.path.join(base_path, filename)
        if os.path.exists(local_path):
            logger.debug(f"在程序目录中找到 {filename}: {local_path}")
            return local_path
    except Exception:
        # 如果 sys.argv[0] 不可用（例如在某些特殊环境中），则回退
        pass
        
    # 3. 如果程序目录没有，再回退到让系统在PATH中查找
    logger.debug(f"在程序目录中未找到 {filename}，将依赖系统 PATH。")
    return filename

# --- 嗅探引擎 ---

def sniff_with_yt_dlp(url, proxy_dict=None):
    """使用 yt-dlp 嗅探媒体流，返回统一格式的字典。"""
    logger.info(f"引擎[yt-dlp]: 开始嗅探 {url}")
    yt_dlp_exe = get_executable_path("yt-dlp.exe")
    
    # 启动前再次检查路径是否有效
    if not os.path.exists(yt_dlp_exe):
        msg = f"yt-dlp.exe 未找到。检查路径: {yt_dlp_exe}"
        logger.error(msg)
        return {"error": msg, "engine": "yt-dlp"}

    command = [yt_dlp_exe, "--dump-json", "--no-warnings", url]
    if proxy_dict and (proxy_url := proxy_dict.get('https://')):
        command.extend(["--proxy", proxy_url])
    
    try:
        # 增加超时以防止永久卡死
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, 
            encoding='utf-8', errors='ignore', timeout=90
        )
        first_line = result.stdout.strip().split('\n')[0]
        data = json.loads(first_line)
        data['engine'] = 'yt-dlp'
        return data
    except subprocess.TimeoutExpired:
        msg = "yt-dlp 嗅探超时（超过90秒）。"
        logger.error(msg)
        return {"error": msg, "engine": "yt-dlp"}
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.strip()
        logger.error(f"yt-dlp 嗅探失败: {error_output}")
        return {"error": error_output, "engine": "yt-dlp"}
    except Exception as e:
        logger.error(f"嗅探时发生未知错误: {e}")
        return {"error": str(e), "engine": "yt-dlp"}

def sniff_with_deep_html_parser(url, proxy_dict=None):
    """使用 requests 和 BeautifulSoup 进行深度嗅探。"""
    logger.info(f"引擎[HTML]: 开始嗅探 {url}")
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
    if proxy_dict:
        session.proxies = proxy_dict

    try:
        main_response = session.get(url, timeout=20)
        main_response.raise_for_status()
        
        title = "Untitled"
        soup = BeautifulSoup(main_response.text, 'html.parser')
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # 从HTML中提取所有潜在链接
        potential_urls = set()
        for tag in soup.find_all(['a', 'img', 'video', 'audio', 'source'], href=True) + soup.find_all(['img', 'video', 'audio', 'source'], src=True):
            link = tag.get('href') or tag.get('src')
            if link and not link.startswith(('javascript:', '#', 'data:')):
                potential_urls.add(urljoin(url, link))

        logger.info(f"初步发现 {len(potential_urls)} 个潜在链接。")
        
        verified_links = []
        for link_url in potential_urls:
            path = urlparse(link_url).path
            filename, ext = os.path.splitext(path)
            if ext:
                for category, extensions in RESOURCE_CATEGORIES.items():
                    if ext.lower() in extensions:
                        verified_links.append({
                            "url": link_url,
                            "filename": os.path.basename(path) or "unknown_file",
                            "category": category,
                            "ext": ext.lower(),
                        })
                        break
        
        logger.info(f"验证后发现 {len(verified_links)} 个有效资源。")
        return {"links": verified_links, "title": title, "engine": "html"}

    except Exception as e:
        logger.error(f"深度HTML嗅探失败: {e}")
        return {"error": str(e), "engine": "html"}

# --- 下载引擎命令构建 ---

def build_download_command(url, format_codes, download_dir, proxy_dict=None):
    """构建 yt-dlp 下载命令，并明确指定 ffmpeg 的位置。"""
    yt_dlp_exe = get_executable_path("yt-dlp.exe")
    ffmpeg_exe_path = get_executable_path("ffmpeg.exe")

    command = [
        yt_dlp_exe,
        "-f", format_codes,
        "--output", os.path.join(download_dir, "%(title)s [%(id)s][%(format_id)s].%(ext)s"),
        "--merge-output-format", "mp4",
        "--no-warnings",
        "--progress",
        "--progress-template", "download-stream:%(progress._percent_str)s",
    ]
    
    if os.path.exists(ffmpeg_exe_path):
        command.extend(["--ffmpeg-location", os.path.dirname(ffmpeg_exe_path)])
    else:
        logger.warning("未能定位 ffmpeg.exe，合并可能会失败。")

    if proxy_dict and (proxy_url := proxy_dict.get('https://')):
        command.extend(["--proxy", proxy_url])
        
    command.append(url)
    logger.debug(f"构建下载命令: {' '.join(command)}")
    return command

# --- 直接下载函数 ---
def download_direct_link(url, download_dir, proxy_dict=None, progress_callback=None):
    """使用 requests 下载直接链接，并报告进度。"""
    logger.info(f"直接下载链接: {url}")
    try:
        filename = os.path.basename(urlparse(url).path)
        if not filename:
            filename = f"download_{int(time.time())}"
        filepath = os.path.join(download_dir, filename)

        with requests.get(url, stream=True, proxies=proxy_dict, timeout=(5, 300)) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded_size = 0
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0 and progress_callback:
                            percentage = int((downloaded_size / total_size) * 100)
                            progress_callback(percentage)
        
        if progress_callback:
            progress_callback(100)
        return True, "下载成功完成。"
    except Exception as e:
        logger.error(f"直接下载失败: {e}")
        return False, str(e)