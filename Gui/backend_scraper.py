# backend_scraper.py

import logging
import os
import sys
import time
import json
import re
import subprocess
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import certifi

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

def get_executable_path(filename):
    """
    智能获取可执行文件的路径。
    对于开发和 PyInstaller 打包都至关重要。
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    
    try:
        base_path = os.path.dirname(sys.argv[0])
        local_path = os.path.join(base_path, filename)
        if os.path.exists(local_path):
            logger.debug(f"在程序目录中找到 {filename}: {local_path}")
            return local_path
    except Exception:
        pass
        
    logger.debug(f"在程序目录中未找到 {filename}，将依赖系统 PATH。")
    return filename

# --- 嗅探引擎 ---

def sniff_with_yt_dlp(url, proxy_dict=None):
    """使用 yt-dlp 嗅探媒体流。"""
    logger.info(f"引擎[yt-dlp]: 开始嗅探 {url}")
    yt_dlp_exe = get_executable_path("yt-dlp.exe")
    if not os.path.exists(yt_dlp_exe):
        msg = f"yt-dlp.exe 未找到。请确保它在程序目录或系统PATH中。检查路径: {yt_dlp_exe}"
        return {"error": msg, "engine": "yt-dlp"}

    command = [yt_dlp_exe, "--dump-json", "--no-warnings", url]
    if proxy_dict and (proxy_url := proxy_dict.get('https://')):
        command.extend(["--proxy", proxy_url])
    
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, 
            encoding='utf-8', errors='ignore', timeout=90
        )
        first_line = result.stdout.strip().split('\n')[0]
        data = json.loads(first_line)
        data['engine'] = 'yt-dlp'
        return data
    except subprocess.TimeoutExpired:
        return {"error": "yt-dlp 嗅探超时（超过90秒）。", "engine": "yt-dlp"}
    except subprocess.CalledProcessError as e:
        return {"error": e.stderr.strip(), "engine": "yt-dlp"}
    except Exception as e:
        return {"error": str(e), "engine": "yt-dlp"}

def sniff_github_release_api(url):
    """通过直接请求 GitHub API 来嗅探 Release 页面的资源，并使用个人访问令牌。"""
    logger.info(f"引擎[GitHub API]: 开始嗅探 {url}")
    match = re.search(r'github\.com/([^/]+)/([^/]+)/releases/tag/([^/?#]+)', url)
    if not match:
        return {"error": "无法从URL中解析出 owner/repo/tag。", "engine": "github_api"}

    owner, repo, tag = match.groups()
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    logger.info(f"构造API请求URL: {api_url}")

    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'Ultimate-Sniffer-App/1.0'
    }
    github_token = os.environ.get('GITHUB_TOKEN')
    if github_token:
        headers['Authorization'] = f"token {github_token}"
        logger.info("检测到 GITHUB_TOKEN，已添加到请求头中。")
    else:
        logger.warning("未检测到 GITHUB_TOKEN 环境变量。将作为匿名用户请求，可能很快会达到速率限制。")

    try:
        response = requests.get(
            api_url, 
            headers=headers,
            timeout=30,
            verify=certifi.where() 
        )
        response.raise_for_status()
        data = response.json()

        verified_links = []
        for asset in data.get("assets", []):
            asset_url, asset_name, asset_size, asset_content_type = asset.get("browser_download_url"), asset.get("name"), asset.get("size"), asset.get("content_type")
            if asset_url and asset_name:
                _, ext = os.path.splitext(asset_name)
                category = "其他"
                for cat, exts in RESOURCE_CATEGORIES.items():
                    if ext.lower() in exts: category = cat; break
                verified_links.append({"url": asset_url, "filename": asset_name, "category": category, "size": asset_size, "mime": asset_content_type, "ext": ext.lower()})
        
        return {
            "links": verified_links, 
            "title": data.get("name", f"{owner}/{repo} - {tag}"), 
            "engine": "github_api"
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            msg = f"GitHub API 请求被拒绝 (403)。很可能是速率限制已超额。请设置 GITHUB_TOKEN 环境变量以提高限制。"
        else:
            msg = f"GitHub API 请求失败 (状态码: {e.response.status_code})。"
        logger.error(msg)
        return {"error": msg, "engine": "github_api"}
    except requests.exceptions.RequestException as e:
        msg = f"网络请求失败: {e}"
        return {"error": msg, "engine": "github_api"}
    except Exception as e:
        return {"error": f"处理GitHub API时发生未知错误: {e}", "engine": "github_api"}

def sniff_with_deep_html_parser(url, proxy_dict=None):
    """使用 requests 和 BeautifulSoup 进行深度嗅探。"""
    logger.info(f"引擎[HTML]: 开始嗅探 {url}")
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    if proxy_dict: session.proxies = proxy_dict

    try:
        main_response = session.get(url, timeout=20, verify=certifi.where())
        main_response.raise_for_status()
        soup = BeautifulSoup(main_response.text, 'html.parser')
        title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"
        
        potential_urls = set(urljoin(url, tag.get('href') or tag.get('src')) for tag in soup.find_all(['a', 'img', 'video', 'audio', 'source'], href=True) + soup.find_all(['img', 'video', 'audio', 'source'], src=True) if tag.get('href') or tag.get('src'))
        
        verified_links = []
        for link_url in potential_urls:
            path = urlparse(link_url).path
            _, ext = os.path.splitext(path)
            if ext:
                for category, extensions in RESOURCE_CATEGORIES.items():
                    if ext.lower() in extensions:
                        verified_links.append({"url": link_url, "filename": os.path.basename(path) or "unknown", "category": category, "ext": ext.lower()})
                        break
        return {"links": verified_links, "title": title, "engine": "html"}
    except Exception as e:
        return {"error": str(e), "engine": "html"}

# --- 下载引擎 ---

def build_download_command(url, format_codes, download_dir, proxy_dict=None):
    """构建 yt-dlp 下载命令。"""
    yt_dlp_exe = get_executable_path("yt-dlp.exe")
    ffmpeg_exe_path = get_executable_path("ffmpeg.exe")
    command = [
        yt_dlp_exe, "-f", format_codes,
        "--output", os.path.join(download_dir, "%(title)s [%(id)s][%(format_id)s].%(ext)s"),
        "--merge-output-format", "mp4", "--no-warnings",
        "--progress", "--progress-template", "download-stream:%(progress._percent_str)s",
    ]
    if os.path.exists(ffmpeg_exe_path):
        command.extend(["--ffmpeg-location", os.path.dirname(ffmpeg_exe_path)])
    else:
        logger.warning("未能定位 ffmpeg.exe，合并可能会失败。")
    if proxy_dict: command.extend(["--proxy", proxy_dict.get('https://')])
    command.append(url)
    return command

def download_direct_link(url, download_dir, proxy_dict=None, progress_callback=None):
    """使用 requests 下载直接链接，并报告进度。"""
    logger.info(f"直接下载链接: {url}")
    try:
        filename = os.path.basename(urlparse(url).path) or f"download_{int(time.time())}"
        filepath = os.path.join(download_dir, filename)

        with requests.get(url, stream=True, proxies=proxy_dict, timeout=(5, 300), verify=certifi.where()) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded_size = 0
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0 and progress_callback:
                            progress_callback(int((downloaded_size / total_size) * 100))
        if progress_callback: progress_callback(100)
        return True, "下载成功完成。"
    except Exception as e:
        return False, str(e)