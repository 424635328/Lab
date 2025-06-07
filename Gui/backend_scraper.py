# backend_scraper.py

import logging
import os
import sys
import json
import re
import subprocess
import time
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import certifi
import undetected_chromedriver as uc

logger = logging.getLogger(__name__)

# --- [新增] 仅在Windows上使用的启动标志，用于隐藏控制台窗口 ---
if os.name == 'nt':
    STARTUPINFO = subprocess.STARTUPINFO()
    STARTUPINFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
else:
    STARTUPINFO = None

# --- 资源类型定义 ---
RESOURCE_CATEGORIES = {
    "视频": ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm', '.ts', '.m3u8'),
    "音频": ('.mp3', '.m4a', '.wav', '.aac', '.flac', '.ogg'),
    "图片": ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'),
    "压缩包": ('.zip', '.rar', '.7z', '.tar', '.gz', '.iso'),
    "可执行/安装包": ('.exe', '.msi', '.dmg', '.pkg', '.deb', '.rpm'),
    "文档": ('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt', '.md'),
}

# --- 核心辅助函数 ---
def get_executable_path(filename):
    """智能获取可执行文件的路径。"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    try:
        base_path = os.path.dirname(sys.argv[0])
        local_path = os.path.join(base_path, filename)
        if os.path.exists(local_path):
            return local_path
    except Exception:
        pass
    return filename

def create_requests_session(proxy_dict=None):
    """创建一个配置好的 requests.Session 对象。"""
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
    if proxy_dict:
        session.proxies = proxy_dict
    session.verify = certifi.where()
    return session

def extract_links_from_html(base_url, html_content):
    """从HTML内容中提取所有潜在的资源链接。"""
    soup = BeautifulSoup(html_content, 'html.parser')
    found_urls = set()
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag.get('href')
        is_github_asset = a_tag.has_attr('data-skip-pjax')
        _, ext = os.path.splitext(urlparse(href).path)
        is_known_filetype = any(ext.lower() in exts for exts in RESOURCE_CATEGORIES.values())
        if href and not href.startswith(('javascript:', '#', 'data:')) and (is_github_asset or is_known_filetype):
            found_urls.add(urljoin(base_url, href))

    for tag in soup.find_all(['img', 'video', 'audio', 'source'], src=True):
        src = tag.get('src')
        if src and isinstance(src, str) and not src.startswith(('javascript:', '#', 'data:')):
            found_urls.add(urljoin(base_url, src))
            
    verified_links = []
    for link_url in found_urls:
        path = urlparse(link_url).path
        _, ext = os.path.splitext(path)
        if ext:
            for category, extensions in RESOURCE_CATEGORIES.items():
                if ext.lower() in extensions:
                    verified_links.append({"url": link_url, "filename": os.path.basename(path) or "unknown", "category": category, "ext": ext.lower()})
                    break
    return verified_links

# --- 嗅探引擎模块 ---
def sniff_engine_yt_dlp(url, proxy_dict=None):
    logger.info(f"引擎[yt-dlp]: 开始嗅探 {url}")
    yt_dlp_exe = get_executable_path("yt-dlp.exe")
    if not os.path.exists(yt_dlp_exe):
        return {"error": f"yt-dlp.exe 未找到", "engine": "yt-dlp"}
    command = [yt_dlp_exe, "--dump-json", "--no-warnings", url]
    if proxy_dict and (proxy_url := proxy_dict.get('https://')):
        command.extend(["--proxy", proxy_url])
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, 
            encoding='utf-8', errors='ignore', timeout=90,
            startupinfo=STARTUPINFO
        )
        data = json.loads(result.stdout.strip().split('\n')[0])
        data['engine'] = 'yt-dlp'
        return data
    except subprocess.TimeoutExpired:
        return {"error": "yt-dlp 嗅探超时（超过90秒）。", "engine": "yt-dlp"}
    except subprocess.CalledProcessError as e:
        return {"error": e.stderr.strip(), "engine": "yt-dlp"}
    except Exception as e:
        return {"error": str(e), "engine": "yt-dlp"}

def sniff_engine_github_api(url, proxy_dict=None):
    logger.info(f"引擎[GitHub API]: 开始嗅探 {url}")
    match = re.search(r'github\.com/([^/]+)/([^/]+)/releases/tag/([^/?#]+)', url)
    if not match:
        return {"error": "无法解析URL", "engine": "github_api"}
    owner, repo, tag = match.groups()
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if token := os.environ.get('GITHUB_TOKEN'):
        headers['Authorization'] = f"token {token}"
    session = create_requests_session(proxy_dict)
    session.headers.update(headers)
    try:
        response = session.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        links = []
        for asset in data.get("assets", []):
            asset_url, name, size, mime = asset.get("browser_download_url"), asset.get("name"), asset.get("size"), asset.get("content_type")
            if asset_url and name:
                _, ext = os.path.splitext(name)
                category = "其他"
                for cat, exts in RESOURCE_CATEGORIES.items():
                    if ext.lower() in exts: category = cat; break
                links.append({"url": asset_url, "filename": name, "category": category, "size": size, "mime": mime, "ext": ext.lower()})
        return {"links": links, "title": data.get("name", f"{owner}/{repo} - {tag}"), "engine": "github_api"}
    except requests.RequestException as e:
        return {"error": f"网络请求失败: {e}", "engine": "github_api"}

def sniff_engine_html_parser(url, proxy_dict=None):
    logger.info(f"引擎[HTML]: 开始嗅探 {url}")
    session = create_requests_session(proxy_dict)
    try:
        response = session.get(url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string.strip() if soup.title else "Untitled"
        links = extract_links_from_html(url, response.text)
        if not links:
            return {"error": "HTML引擎未发现可识别的链接。", "engine": "html"}
        return {"links": links, "title": title, "engine": "html"}
    except requests.RequestException as e:
        return {"error": f"网络请求失败: {e}", "engine": "html"}

def sniff_engine_browser(url, proxy_dict=None):
    logger.info(f"引擎[浏览器]: 启动无头浏览器嗅探 {url}")
    options = uc.ChromeOptions()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')
    options.add_argument('--log-level=3')
    if proxy_dict and (proxy_url := proxy_dict.get('https://')):
        options.add_argument(f'--proxy-server={proxy_url}')
    driver = None
    try:
        driver = uc.Chrome(options=options)
        driver.get(url)
        time.sleep(7)
        page_source, page_title = driver.page_source, driver.title
        logger.info("浏览器渲染完成，调用HTML解析器提取链接...")
        links = extract_links_from_html(url, page_source)
        if not links:
            return {"error": "浏览器成功渲染页面，但未发现可识别的链接。", "engine": "browser"}
        return {"links": links, "title": page_title, "engine": "browser"}
    except Exception as e:
        return {"error": f"浏览器引擎启动或执行失败: {e}", "engine": "browser"}
    finally:
        if driver: driver.quit()

# --- 下载相关函数 ---
def build_download_command(url, format_codes, download_dir, proxy_dict=None):
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
    if proxy_dict:
        command.extend(["--proxy", proxy_dict.get('https://')])
    command.append(url)
    return command

def download_direct_link(url, download_dir, proxy_dict=None, progress_callback=None):
    logger.info(f"直接下载链接: {url}")
    try:
        filename = os.path.basename(urlparse(url).path) or f"download_{int(time.time())}"
        filepath = os.path.join(download_dir, filename)
        session = create_requests_session(proxy_dict)
        with session.get(url, stream=True, timeout=(5, 300)) as r:
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
        if progress_callback:
            progress_callback(100)
        return True, "下载成功完成。"
    except Exception as e:
        return False, str(e)