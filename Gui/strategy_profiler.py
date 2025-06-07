# strategy_profiler.py (终极智能增强版)

import logging
from urllib.parse import urlparse, parse_qs
import json
import os

logger = logging.getLogger(__name__)

# --- [新增] 策略元数据：定义成本和后端函数 ---
# 成本：1=极低, 2=低, 3=中, 5=高, 10=极高(浏览器)
STRATEGY_METADATA = {
    "direct_link":    {"backend": "sniff_engine_direct_link", "cost": 1},
    "github_api":     {"backend": "sniff_engine_github_api", "cost": 2},
    "yt_dlp":         {"backend": "sniff_engine_yt_dlp", "cost": 5},
    "html_parser":    {"backend": "sniff_engine_html_parser", "cost": 3},
    "browser":        {"backend": "sniff_engine_browser", "cost": 10},
}
# 方便 worker.py 使用
AVAILABLE_STRATEGIES = {name: data["backend"] for name, data in STRATEGY_METADATA.items()}

# --- [新增] 动态经验数据 ---
# 在实际应用中，这应该被持久化到一个文件中
# key: 域名, value: (最常用且成功的策略, 次数)
experience_data = {
    "youtube.com": ("yt_dlp", 10),
    "bilibili.com": ("yt_dlp", 15)
}
EXPERIENCE_BOOST = 15 # 经验加分

# --- 特征提取与分析器 ---

def analyze_url_features(url_info):
    """从 urlparse 对象中提取更多特征。"""
    features = {
        'netloc': url_info.netloc,
        'path': url_info.path,
        'path_depth': url_info.path.count('/'),
        'query_params': parse_qs(url_info.query),
        'file_ext': os.path.splitext(url_info.path)[1].lower()
    }
    return features

def analyze_for_direct_link(features):
    known_exts = ('.mp4', '.mkv', '.zip', '.rar', '.exe', '.msi', '.pdf', '.mp3')
    if features['file_ext'] in known_exts:
        return 101 # 最高优先级
    return 0

def analyze_for_github_api(features):
    if "github.com" in features['netloc'] and "/releases/tag/" in features['path']:
        return 100
    return 0

def analyze_for_yt_dlp(features):
    score = 30 # 基础分
    known_media_domains = ["youtube.com", "youtu.be", "bilibili.com", "vimeo.com", "douyin.com", "ixigua.com"]
    if any(domain in features['netloc'] for domain in known_media_domains):
        score += 65
    if "youtube.com" in features['netloc'] and features['path'].startswith("/watch") and 'v' in features['query_params']:
        score += 10 # 额外加分
    if "bilibili.com" in features['netloc'] and features['path'].startswith("/video/"):
        score += 5
    return score

def analyze_for_html_parser(features):
    # 如果路径很深或者有很多查询参数，普通HTML解析可能有用
    score = 20 + features['path_depth'] * 2 + len(features['query_params'])
    return min(score, 40) # 封顶

def analyze_for_browser(features):
    score = 10 # 基础分很低
    js_heavy_sites = ["295yhw.com", "weibo.com"] # 已知的JS重灾区
    if any(domain in features['netloc'] for domain in js_heavy_sites):
        score += 80
    # 如果URL看起来像一个复杂的Web应用（例如，没有文件扩展名但路径很深）
    if not features['file_ext'] and features['path_depth'] > 2:
        score += 15
    return score


# 将所有分析器函数映射到策略名
STRATEGY_ANALYZERS = {
    "github_api": analyze_for_github_api,
    "yt_dlp": analyze_for_yt_dlp,
    "html_parser": analyze_for_html_parser,
    "browser": analyze_for_browser,
    "direct_link": analyze_for_direct_link
}

# --- 主决策函数 ---
def select_best_strategy(url):
    """
    [重构] 对给定的URL进行加权评分，选择最佳策略队列。
    """
    logger.info(f"为URL '{url}' 进行智能策略评估...")
    try:
        url_info = urlparse(url)
        features = analyze_url_features(url_info)
    except Exception as e:
        logger.error(f"URL解析失败: {e}")
        return []

    scores = {}
    for name, analyzer_func in STRATEGY_ANALYZERS.items():
        # 1. 计算基础信心分数
        raw_score = analyzer_func(features)
        
        # 2. 应用成本惩罚
        cost = STRATEGY_METADATA[name]["cost"]
        adjusted_score = raw_score / (cost ** 0.5) # 对成本开方，惩罚不至于太剧烈
        
        # 3. 应用经验加分
        domain = features['netloc'].replace("www.", "")
        if domain in experience_data and experience_data[domain][0] == name:
            adjusted_score += EXPERIENCE_BOOST
            
        scores[name] = adjusted_score
    
    # 按最终分数降序排序
    sorted_strategies = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    logger.info(f"策略评估结果 (策略, 最终得分): {[(s, round(v, 2)) for s, v in sorted_strategies]}")
    
    # 只返回分数大于某个阈值的策略
    return [strategy for strategy, score in sorted_strategies if score > 15]

def update_experience_data(domain, successful_strategy):
    """
    [新增] 模拟更新经验数据。
    在真实应用中，这里应该有读写文件的逻辑。
    """
    domain = domain.replace("www.", "")
    if domain in experience_data:
        if experience_data[domain][0] == successful_strategy:
            experience_data[domain] = (successful_strategy, experience_data[domain][1] + 1)
        else:
            # 如果新策略更成功，可以考虑替换
            pass
    else:
        experience_data[domain] = (successful_strategy, 1)
    logger.info(f"经验数据已更新: {experience_data}")