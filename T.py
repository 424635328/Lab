# -*- coding: utf-8 -*-

# --- 标准库导入 ---
import logging
import os
import pickle
import http.cookiejar
import random
import time
import re
import subprocess
from urllib.parse import urlparse, unquote

# --- 第三方库导入 ---
import requests
import httpx
import cloudscraper
from curl_cffi import requests as curl_requests
from curl_cffi.requests.cookies import Cookies as CurlCffiCookies
from fake_useragent import UserAgent
from robotexclusionrulesparser import RobotExclusionRulesParser
from tqdm import tqdm

# ==============================================================================
# --- 全局配置区域 ---
# ==============================================================================

LOG_LEVEL = logging.INFO
LOG_FILE = "ultimate_scraper_zh.log"

REQUEST_TIMEOUT = 30
MIN_REQUEST_DELAY = 1
MAX_REQUEST_DELAY = 3
MAX_RETRIES_PER_ENGINE = 2
RETRY_BASE_DELAY = 2
RETRY_MAX_DELAY = 60

PROXY_FILE = "proxies_zh.txt"
SESSION_COOKIE_FILE_PREFIX = "session_cookies_zh_"
CURL_IMPERSONATE_OPTIONS = ["chrome110", "chrome116", "chrome120", "safari15_5", "firefox115", "random"]
DOWNLOAD_DIR = "downloads"
STRATEGY_PIPELINE = ["requests", "httpx", "cloudscraper", "curl_cffi"]

# ==============================================================================
# --- 任务定义列表 (已升级为字典格式) ---
# ==============================================================================
TARGET_URLS_TO_SCRAPE = [
    # {
    #     "url": "https://httpbin.org/anything",
    #     "method": "POST",
    #     "json_payload": {"message": "Hello from the ultimate scraper!", "task_id": 1},
    #     "custom_headers": {"X-Task-ID": "Task-001", "Authorization": "Bearer fake_token"},
    #     "timeout": 45,
    #     "start_engine": "requests",
    #     "ignore_robots": True,
    #     "description": "高级 POST 请求测试"
    # },
    # {
    #     "url": "https://nowsecure.nl/",
    #     "start_engine": "curl_cffi",
    #     "description": "TLS 指纹测试"
    # },
    # {
    #     "url": "http://vjs.zencdn.net/v/oceans.mp4",
    #     "download_method": "direct",
    #     "description": "直接下载MP4文件"
    # },
    {
        "url": "https://www.295yhw.com/play/8212-1-1.html",
        "download_method": "yt-dlp",
        "description": "使用 yt-dlp 下载B站视频"
    }
]
# ==============================================================================

# --- 日志与全局变量配置 ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s', handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8', mode='w'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

LOADED_PROXIES, BAD_PROXIES, ROBOTS_PARSERS = [], set(), {}
try:
    ua_generator = UserAgent(fallback='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
except Exception as e:
    ua_generator = None
    PREDEFINED_USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]
    logger.warning(f"fake_useragent 初始化失败 ({e})，将使用预定义列表。")

# ==============================================================================
# --- 辅助函数 (已补全和优化) ---
# ==============================================================================

def load_proxies_from_file(filename):
    """从文件加载代理列表。"""
    global LOADED_PROXIES
    if filename and os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                LOADED_PROXIES = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            logger.info(f"成功从 {filename} 加载了 {len(LOADED_PROXIES)} 个代理。")
        except Exception as e:
            logger.error(f"从 {filename} 加载代理时发生错误: {e}")
    else:
        logger.info(f"代理文件 '{filename}' 不存在，不使用文件代理。")

def get_random_proxy_dict():
    """获取一个随机的、未被标记为坏的代理。"""
    if not LOADED_PROXIES: return None
    available_proxies = [p for p in LOADED_PROXIES if p not in BAD_PROXIES]
    if not available_proxies:
        logger.warning("所有可用代理均已被标记为不可用。")
        return None
    proxy_url = random.choice(available_proxies)
    return {'http://': proxy_url, 'https://': proxy_url}

def mark_proxy_bad(proxy_dict):
    """将一个代理标记为不可用。"""
    if proxy_dict and (proxy_url := proxy_dict.get('http://') or proxy_dict.get('https://')):
        BAD_PROXIES.add(proxy_url)
        logger.warning(f"已将代理 {proxy_url} 标记为不可用。")

def get_random_user_agent_string():
    """获取一个随机的User-Agent字符串。"""
    if ua_generator:
        try: return ua_generator.random
        except Exception: pass
    return random.choice(PREDEFINED_USER_AGENTS)

def exponential_backoff_with_jitter(attempt):
    """计算带抖动的指数退避延迟时间。"""
    return min(RETRY_MAX_DELAY, (RETRY_BASE_DELAY * (2 ** attempt)) + random.uniform(0, 1))

def get_robots_parser(url, session_like_object, engine_name):
    """获取并缓存指定域名的robots.txt解析器。"""
    parsed_url = urlparse(url)
    domain_base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    if domain_base_url in ROBOTS_PARSERS: return ROBOTS_PARSERS[domain_base_url]
    
    parser = RobotExclusionRulesParser()
    try:
        robots_url = f"{domain_base_url}/robots.txt"
        headers = {'User-Agent': get_random_user_agent_string()}
        if engine_name == "curl_cffi": response = curl_requests.get(robots_url, headers=headers, timeout=10)
        else: response = session_like_object.get(robots_url, headers=headers, timeout=10)
        response.raise_for_status()
        parser.parse(response.text)
        logger.info(f"成功解析 {domain_base_url} 的 robots.txt。")
    except Exception as e:
        logger.warning(f"无法获取或解析 {domain_base_url} 的 robots.txt: {e}。假定允许所有路径。")
        parser.parse("User-agent: *\nAllow: /")
    ROBOTS_PARSERS[domain_base_url] = parser
    return parser

def is_url_allowed_by_robots(url, user_agent, session_like_object, engine_name):
    """检查给定的URL是否被robots.txt允许抓取。"""
    parser = get_robots_parser(url, session_like_object, engine_name)
    return parser.is_allowed(user_agent, url)

def save_cookies(cookie_jar_source, filepath):
    """将各种来源的cookie jar保存到文件。"""
    try:
        cookies_to_save = None
        if isinstance(cookie_jar_source, (requests.Session, cloudscraper.CloudScraper)): cookies_to_save = cookie_jar_source.cookies
        elif isinstance(cookie_jar_source, httpx.Client): cookies_to_save = cookie_jar_source.cookies
        elif isinstance(cookie_jar_source, (requests.cookies.RequestsCookieJar, httpx.Cookies, http.cookiejar.CookieJar, CurlCffiCookies)): cookies_to_save = cookie_jar_source
        else:
            logger.warning(f"保存Cookie时遇到未知源类型: {type(cookie_jar_source)}，跳过。")
            return
        if not cookies_to_save: return
        with open(filepath, 'wb') as f: pickle.dump(cookies_to_save, f)
        logger.debug(f"Cookies 已保存到: {filepath}")
    except Exception as e:
        logger.error(f"保存 cookies 到 {filepath} 时发生错误: {e}")

def load_cookies(target_ref_list, filepath):
    """从文件加载cookies并应用到目标对象。"""
    if not (os.path.exists(filepath) and os.path.getsize(filepath) > 0): return False
    try:
        with open(filepath, 'rb') as f: loaded_cookies = pickle.load(f)
        target_object = target_ref_list[0]
        if hasattr(target_object, 'cookies') and not callable(target_object.cookies): target_object.cookies.update(loaded_cookies)
        elif hasattr(target_object, 'update') and callable(target_object.update): target_object.update(loaded_cookies)
        else: return False
        logger.info(f"Cookies 已从 {filepath} 加载。")
        return True
    except Exception as e:
        logger.error(f"从 {filepath} 加载 cookies 时发生错误: {e}")
    return False

def get_consistent_headers(user_agent, last_url=None, custom_headers=None):
    """生成与User-Agent一致并合并自定义头的请求头。"""
    platform, is_mobile = "Windows", "?0"
    ua_lower = user_agent.lower()
    if "macintosh" in ua_lower or "mac os x" in ua_lower: platform = "macOS"
    elif "linux" in ua_lower and "android" not in ua_lower: platform = "Linux"
    elif "android" in ua_lower: platform, is_mobile = "Android", "?1"
    elif "iphone" in ua_lower or "ipad" in ua_lower: platform, is_mobile = "iOS", "?1"

    chrome_version_match = re.search(r'chrome/(\d+)', ua_lower)
    chrome_version = chrome_version_match.group(1) if chrome_version_match else "120"
    brand_list = [{"brand": "Not/A)Brand", "version": "99"}, {"brand": "Google Chrome", "version": chrome_version}, {"brand": "Chromium", "version": chrome_version}]
    random.shuffle(brand_list)
    
    headers = {"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9", "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7", "Accept-Encoding": "gzip, deflate, br", "Connection": "keep-alive", "Upgrade-Insecure-Requests": "1", "Sec-CH-UA": ", ".join([f'"{b["brand"]}";v="{b["version"]}"' for b in brand_list]), "Sec-CH-UA-Mobile": is_mobile, "Sec-CH-UA-Platform": f'"{platform}"', "Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Site": "cross-site" if not last_url else "same-origin", "Sec-Fetch-User": "?1"}
    if last_url: headers["Referer"] = last_url
    if custom_headers: headers.update(custom_headers)
    return headers

class ScrapeIdentity:
    """封装一个完整的爬虫身份，包括UA、代理、头信息等。"""
    def __init__(self, last_url=None, custom_headers=None):
        self.user_agent = get_random_user_agent_string()
        self.proxy_dict = get_random_proxy_dict()
        self.headers = get_consistent_headers(self.user_agent, last_url, custom_headers)
        self.curl_impersonate = random.choice(CURL_IMPERSONATE_OPTIONS)
        logger.info(f"创建新身份 -> UA: ...{self.user_agent[-30:]}, 代理: {self.proxy_dict.get('https') if self.proxy_dict else '无'}")

def is_captcha_block(response_text):
    """检查响应文本是否包含常见的CAPTCHA关键词。"""
    captcha_keywords = ["g-recaptcha", "h-captcha", "are you a robot", "人机验证", "验证码", "slide to verify"]
    text_lower = response_text.lower()
    return any(keyword in text_lower for keyword in captcha_keywords)

def clear_domain_cookies(session_like_object, url):
    """为特定域名清空会话中的cookies。"""
    domain = urlparse(url).netloc
    if hasattr(session_like_object, 'cookies') and isinstance(session_like_object.cookies, http.cookiejar.CookieJar):
        cookies_to_remove = [cookie.name for cookie in session_like_object.cookies if domain in cookie.domain]
        for name in cookies_to_remove:
            try: session_like_object.cookies.clear(domain=domain, path="/", name=name)
            except: pass
        if cookies_to_remove: logger.debug(f"已为域名 {domain} 清空了 {len(cookies_to_remove)} 个Cookies。")

def download_video_from_response(response, url):
    """从一个成功的响应中流式下载视频文件。"""
    try:
        content_disp = response.headers.get('content-disposition', '')
        filename_match = re.search(r'filename\*?=(?:UTF-8\'\')?([^;]+)', content_disp, flags=re.IGNORECASE)
        filename = unquote(filename_match.group(1).strip('"')) if filename_match else (os.path.basename(urlparse(url).path) or f"video_{int(time.time())}.mp4")
        
        if not os.path.exists(DOWNLOAD_DIR): os.makedirs(DOWNLOAD_DIR)
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        total_size = int(response.headers.get('content-length', 0))

        logger.info(f"开始直接下载: {filename} (大小: {total_size / 1024 / 1024:.2f} MB)")
        with open(filepath, 'wb') as f, tqdm(desc=filename, total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as bar:
            iterator = response.iter_bytes(8192) if hasattr(response, 'iter_bytes') else response.iter_content(8192)
            for chunk in iterator:
                f.write(chunk)
                bar.update(len(chunk))
        logger.info(f"直接下载完成: {filepath}")
    except Exception as e:
        logger.error(f"直接下载时发生错误: {e}", exc_info=True)

def download_with_yt_dlp(url, proxy_dict=None):
    """
    使用 yt-dlp 命令行工具下载视频，并直接将输出流式传输到控制台。
    """
    logger.info(f"将使用 yt-dlp 下载: {url}")
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        
    command = [
        "yt-dlp",
        # 输出模板，保存到 downloads 文件夹
        "--output", os.path.join(DOWNLOAD_DIR, "%(title)s [%(id)s].%(ext)s"),
        # 如果音视频分离，合并为mp4
        "--merge-output-format", "mp4",
        # 隐藏警告信息，让输出更干净
        "--no-warnings",
        # 显式要求 yt-dlp 显示进度条
        "--progress",
    ]

    # 添加代理参数
    if proxy_dict and (proxy_url := proxy_dict.get('https://') or proxy_dict.get('http://')):
        command.extend(["--proxy", proxy_url])
        logger.info(f"yt-dlp 将使用代理: {proxy_url}")
    
    # 将URL添加到命令末尾
    command.append(url)

    try:
        # 使用 subprocess.run，不捕获输出，让它直接显示在终端
        # check=True 确保如果 yt-dlp 失败（返回非零码），会抛出异常
        subprocess.run(command, check=True)
        
        logger.info("yt-dlp 进程成功执行。")
        return True
        
    except FileNotFoundError:
        logger.error("错误: 'yt-dlp' 命令未找到。请确保 yt-dlp 已安装并且在系统的 PATH 环境变量中。")
        return False
    except subprocess.CalledProcessError as e:
        # 如果 check=True 失败，会进入这里
        # 因为我们没有捕获输出，所以无法直接打印 e.stdout 或 e.stderr
        # 但 yt-dlp 的错误信息应该已经直接显示在控制台了
        logger.error(f"yt-dlp 执行失败，返回码: {e.returncode}。请检查上面的控制台输出获取详细错误信息。")
        return False
    except Exception as e:
        logger.error(f"调用 yt-dlp 时发生未知错误: {e}")
        return False

def process_response(response, task):
    """处理成功的响应，根据任务配置进行操作。"""
    url, download_method = task['url'], task.get("download_method")
    if download_method == "direct":
        download_video_from_response(response, url)
    else:
        try:
            content_preview = response.text[:300].replace('\n', ' ').replace('\r', '')
            logger.info(f"任务成功，内容预览: {content_preview}...")
        except Exception as e:
            logger.warning(f"无法获取文本内容预览 (可能是二进制文件): {e}")

# ==============================================================================
# --- 核心请求策略引擎 ---
# ==============================================================================
def fetch_url_with_strategy(task, session_pool, last_successful_url=None):
    """使用多层策略和字典式任务配置来获取URL。"""
    url, method = task['url'], task.get("method", "GET").upper()
    start_engine = task.get("start_engine", STRATEGY_PIPELINE[0])
    stream = task.get("download_method") == "direct"
    
    identity = ScrapeIdentity(last_successful_url, task.get("custom_headers"))

    try: active_pipeline = STRATEGY_PIPELINE[STRATEGY_PIPELINE.index(start_engine):]
    except ValueError: active_pipeline = STRATEGY_PIPELINE

    if not task.get("ignore_robots", False):
        if not is_url_allowed_by_robots(url, identity.user_agent, session_pool['requests'], 'requests'):
            logger.warning(f"根据 robots.txt 规则，任务被跳过: {url}")
            return None

    for engine in active_pipeline:
        logger.info(f"--- 策略升级: 尝试引擎 '{engine}' ---")
        session_like_object = session_pool.get(engine)
        if engine != 'curl_cffi': clear_domain_cookies(session_like_object, url)

        for attempt in range(MAX_RETRIES_PER_ENGINE):
            logger.info(f"引擎: {engine} | 身份尝试 {attempt + 1}/{MAX_RETRIES_PER_ENGINE} | {method} | URL: {url}")
            try:
                request_args = {k: v for k, v in {"headers": identity.headers, "timeout": task.get("timeout", REQUEST_TIMEOUT), "stream": stream, "params": task.get("params"), "data": task.get("data"), "json": task.get("json_payload")}.items() if v is not None}
                
                if engine in ["requests", "cloudscraper"]: response = session_like_object.request(method, url, proxies=identity.proxy_dict, **request_args)
                elif engine == "httpx":
                    session_like_object.proxies = identity.proxy_dict
                    response = session_like_object.request(method, url, **request_args)
                elif engine == "curl_cffi":
                    curl_args = request_args.copy()
                    curl_args.update({"impersonate": identity.curl_impersonate, "proxies": identity.proxy_dict})
                    response = curl_requests.request(method, url, **curl_args)
                
                if not stream and is_captcha_block(response.text):
                     logger.critical(f"检测到CAPTCHA页面，放弃此URL: {url}")
                     return None
                response.raise_for_status()
                logger.info(f"请求成功! 引擎: '{engine}', 状态码: {response.status_code}")
                cookie_src = response.cookies if engine == "curl_cffi" else session_like_object
                save_cookies(cookie_src, f"{SESSION_COOKIE_FILE_PREFIX}{engine}.pkl")
                return response
                
            except (requests.exceptions.HTTPError, httpx.HTTPStatusError) as e:
                status_code = getattr(e.response, 'status_code', -1)
                is_cloudflare = "cloudflare" in getattr(e.response, 'headers', {}).get("Server", "").lower()
                logger.warning(f"引擎 {engine} 遭遇HTTP错误: {status_code}")
                if is_cloudflare and engine not in ["cloudscraper", "curl_cffi"]: break
                if status_code in [403, 401, 407]:
                    logger.warning("遭遇硬封锁，执行身份重置！")
                    mark_proxy_bad(identity.proxy_dict)
                    identity = ScrapeIdentity(last_successful_url, task.get("custom_headers"))
            except Exception as e:
                logger.error(f"引擎 {engine} 发生网络或未知错误: {e}")
            if attempt < MAX_RETRIES_PER_ENGINE - 1: time.sleep(exponential_backoff_with_jitter(attempt))

    logger.error(f"所有策略管道均已用尽，未能成功获取URL: {url}")
    return None

# ==============================================================================
# --- 主程序逻辑 ---
# ==============================================================================
if __name__ == "__main__":
    load_proxies_from_file(PROXY_FILE)
    req_session, httpx_client, cs_scraper = requests.Session(), httpx.Client(http2=True, follow_redirects=True), cloudscraper.create_scraper()
    session_pool = {"requests": req_session, "httpx": httpx_client, "cloudscraper": cs_scraper, "curl_cffi": curl_requests}
    for name, client in session_pool.items():
        if name != 'curl_cffi': load_cookies([client], f"{SESSION_COOKIE_FILE_PREFIX}{name}.pkl")

    last_successful_url = None
    
    for i, task in enumerate(TARGET_URLS_TO_SCRAPE):
        url, download_method = task['url'], task.get("download_method")
        description = task.get("description", "无描述")
        
        logger.info(f"\n=======>>>>> 开始任务 {i+1}/{len(TARGET_URLS_TO_SCRAPE)} ({description}): {url} <<<<<=======")
        
        if download_method == "yt-dlp":
            download_with_yt_dlp(url, get_random_proxy_dict())
            continue

        response = fetch_url_with_strategy(task, session_pool, last_successful_url)
        if response:
            last_successful_url = url
            process_response(response, task)
    
    httpx_client.close()
    logger.info("\n--- 脚本执行完毕 ---")