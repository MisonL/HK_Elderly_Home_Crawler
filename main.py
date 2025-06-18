import os
import logging
import time
import pandas as pd
import hashlib
import re
import argparse
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import google.api_core.exceptions # 导入用于处理API异常的模块
from selenium import webdriver
from dotenv import load_dotenv
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.remote.webelement import WebElement
from typing import List, Dict
from urllib.parse import urljoin # 导入 urljoin 用于处理URL拼接
from concurrent.futures import ThreadPoolExecutor # 导入线程池用于异步API调用
import json # 将json模块导入移到顶部，使其全局可用

# 优化：全局启用Pandas的Copy-on-Write，提高内存效率和预测性
pd.options.mode.copy_on_write = True

# --- 1. 配置日志 ---
# 创建日志目录
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志器
logging.basicConfig(
    level=logging.INFO,  # 默认日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'crawl.log'), encoding='utf-8'),  # 输出到文件
        logging.StreamHandler()  # 输出到控制台
    ]
)

logger = logging.getLogger(__name__)

# --- 2. 加载环境变量和配置Gemini API密钥 ---
load_dotenv() # 加载.env文件中的环境变量

# 修改为加载多个API Key
GEMINI_API_KEYS_STR = os.getenv('GEMINI_API_KEYS')
GEMINI_API_KEYS = [key.strip() for key in re.split(r'[\s,]+', GEMINI_API_KEYS_STR) if key.strip()] if GEMINI_API_KEYS_STR else []

if not GEMINI_API_KEYS:
    logger.critical("GEMINI_API_KEYS 环境变量未设置。行政区识别功能将无法使用。请在.env文件或系统环境变量中设置，多个密钥请用逗号分隔。")
    # 可以在这里选择退出程序或继续（不使用Gemini功能）
    # sys.exit(1) # 如果API密钥是强制的，可以取消注释此行

# 定义Gemini API处理类
class GeminiAPIHandler:
    def __init__(self, api_keys: List[str], model_name: str, prompt_template: str):
        self.all_api_keys = api_keys # 存储所有提供的API Key
        self.api_keys: List[str] = [] # 存储经过有效性测试后的可用API Key
        self.current_key_index = 0
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.model = None
        # 存储每个API Key上次使用的时间戳，用于轮转判断
        self.key_last_used_time: Dict[str, float] = {}
        # 存储每个API Key的冷却结束时间（429错误后）
        self.key_cooldown_until: Dict[str, float] = {}
        
        self._initialize_api_keys() # 初始化时进行API Key有效性测试
        self._configure_gemini_api() # 配置Gemini API

    def _initialize_api_keys(self):
        """测试并初始化可用的API Key列表。"""
        logger.info("开始测试Gemini API Key的有效性...")
        valid_keys = []
        current_time = time.time()
        for key in self.all_api_keys:
            if self._test_api_key_validity(key):
                valid_keys.append(key)
                # 初始化时记录当前时间，避免首次使用时显示大数值
                self.key_last_used_time[key] = current_time
                self.key_cooldown_until[key] = 0.0
        
        self.api_keys = valid_keys
        if not self.api_keys:
            logger.critical("所有提供的Gemini API Key均无效或未设置。行政区识别功能将无法使用。")
        else:
            logger.info(f"成功识别 {len(self.api_keys)} 个有效Gemini API Key。")

    def _test_api_key_validity(self, api_key: str) -> bool:
        """
        测试单个Gemini API Key的有效性。
        通过尝试列出模型来验证。
        """
        if not re.match(r'^[a-zA-Z0-9_-]{20,}$', api_key):
            logger.warning(f"API Key格式无效: {api_key[:5]}...。")
            return False
        
        try:
            genai.configure(api_key=api_key, transport='rest') # type: ignore[attr-defined, call-arg]
            # 尝试列出模型，这是一个轻量级的API调用，用于验证密钥
            list(genai.list_models()) # type: ignore
            logger.info(f"API Key {api_key[:5]}... 有效。")
            return True
        except (google.api_core.exceptions.GoogleAPIError,
                google.api_core.exceptions.InternalServerError,
                google.api_core.exceptions.ServiceUnavailable) as e:
            logger.warning(f"API Key {api_key[:5]}... 无效或连接失败: {e}")
            logger.info("请检查：")
            logger.info("1. API密钥是否正确（在 https://aistudio.google.com/app/apikey 查看）")
            logger.info("2. 网络连接是否正常（特别是代理设置）")
            logger.info("3. 是否启用了Gemini API（在 https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/ 启用）")
            return False
        except Exception as e:
            logger.warning(f"测试API Key {api_key[:5]}... 时发生未知错误: {e}")
            return False

    def _configure_gemini_api(self):
        """配置Gemini API，使用当前API Key。"""
        if not self.api_keys:
            logger.warning("没有可用的Gemini API密钥，无法配置模型。")
            self.model = None # 确保模型为None
            return

        current_api_key = self.api_keys[self.current_key_index]
        try:
            genai.configure(
                api_key=current_api_key,
                transport='rest'
            ) # type: ignore[attr-defined, call-arg]
            self.model = genai.GenerativeModel(self.model_name) # type: ignore
            logger.info(f"Gemini API 已配置，使用密钥索引: {self.current_key_index} (密钥: {current_api_key[:5]}...)，模型: {self.model_name}")
            # 成功配置后，更新当前key的上次使用时间
            self.key_last_used_time[current_api_key] = time.time()
        except Exception as e:
            logger.error(f"Gemini API 配置失败，密钥索引 {self.current_key_index}: {e}")
            logger.info("请检查：")
            logger.info("1. API密钥是否正确（在 https://aistudio.google.com/app/apikey 查看）")
            logger.info("2. 网络连接是否正常（特别是代理设置）")
            logger.info("3. 是否启用了Gemini API（在 https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/ 启用）")
            self.model = None
            # 如果当前密钥配置失败，尝试轮转到下一个密钥
            if self.api_keys: # 只有当还有其他密钥时才尝试轮转
                logger.info("当前密钥配置失败，尝试轮转到下一个可用密钥。")
                self.rotate_key(force_rotate=True) # 强制轮转到下一个密钥

    def rotate_key(self, force_rotate: bool = False):
        """
        轮转到下一个API Key。
        Args:
            force_rotate (bool): 如果为True，则强制轮转，跳过冷却时间检查。
        """
        if not self.api_keys:
            logger.warning("没有可用的API密钥进行轮转。")
            return

        current_time = time.time()
        
        # 调试日志：显示所有密钥状态
        logger.debug(f"开始密钥轮转，当前密钥索引: {self.current_key_index}, 强制轮转: {force_rotate}")
        for index, key in enumerate(self.api_keys):
            cooldown_until = self.key_cooldown_until.get(key, 0.0)
            cooldown_remaining = max(0.0, cooldown_until - current_time)
            time_since_last_use = current_time - self.key_last_used_time.get(key, 0.0)
            logger.debug(f"密钥 {index}: 冷却剩余 {cooldown_remaining:.1f}s, 上次使用 {time_since_last_use:.1f}s前")
        
        # 循环直到找到一个可用的密钥
        for _ in range(len(self.api_keys)):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            next_key = self.api_keys[self.current_key_index]
            
            cooldown_until = self.key_cooldown_until.get(next_key, 0.0)
            cooldown_remaining = max(0.0, cooldown_until - current_time)
            time_since_last_use = current_time - self.key_last_used_time.get(next_key, 0.0)
            
            # 强制轮转（因429错误）直接使用下一个密钥
            if force_rotate:
                logger.info(f"强制轮转: 跳过等待, 新密钥索引: {self.current_key_index} (密钥: {next_key[:5]}...)")
                self.key_last_used_time[next_key] = current_time
                self._configure_gemini_api()
                return
            
            # 检查密钥是否可用
            if cooldown_remaining <= 0:
                # 智能等待策略：距离上次使用不足60秒则等待
                if time_since_last_use < 60:
                    wait_time = 60 - time_since_last_use
                    logger.info(f"密钥 {self.current_key_index} 需要等待: {wait_time:.1f}s (上次使用: {time_since_last_use:.1f}s前)")
                    time.sleep(wait_time)
                
                logger.info(f"密钥 {self.current_key_index} 可用 (冷却已过, 密钥: {next_key[:5]}...)")
                self.key_last_used_time[next_key] = current_time
                self._configure_gemini_api()
                return
            else:
                logger.info(f"密钥 {self.current_key_index} 冷却中: 剩余 {cooldown_remaining:.1f}s (密钥: {next_key[:5]}...)")
        
        # 所有密钥都不可用时等待最长冷却时间
        max_cooldown = max(self.key_cooldown_until.values(), default=0) - current_time
        if max_cooldown > 0:
            logger.warning(f"所有密钥不可用，等待最长冷却时间: {max_cooldown:.1f}s")
            time.sleep(max_cooldown)
            self.rotate_key(force_rotate)  # 递归尝试
        else:
            logger.error("所有密钥均不可用且无有效冷却时间")

    def get_model(self):
        """获取当前配置的Gemini模型实例。"""
        return self.model

    def get_prompt(self, address: str) -> str:
        """生成带地址的提示词。"""
        return f"{self.prompt_template}：{address}"

# --- 3. 定义常量和文件路径 ---
DEFAULT_EXCEL_FILE = "護老院-爬虫.xlsx"
MAX_RETRIES = 5  # 最大重试次数
RETRY_DELAY = 30  # 重试延迟（秒）（增加到30秒）
PAGE_LOAD_TIMEOUT = 180  # 页面加载超时时间（秒）（增加到180秒）
ELEMENT_WAIT_TIMEOUT = 60 # 元素等待超时时间（秒）（增加到60秒）


# --- 6. 函数：加载已爬取数据并生成去重集合 ---
def load_existing_data_for_deduplication(excel_file: str = DEFAULT_EXCEL_FILE) -> set[str]:
    """
    加载已有的Excel数据，并生成用于去重的唯一标识符集合。
    唯一标识符为 '院舍名稱' + '地址' 的哈希值。
    """
    existing_hashes: set[str] = set()
    if os.path.exists(excel_file):
        try:
            # 添加更精确的类型忽略注释
            df_existing = pd.read_excel(excel_file, engine='openpyxl') # type: ignore[call-overload]
            if not df_existing.empty:
                # 确保关键列存在
                required_cols = ['院舍名稱', '地址']
                if all(col in df_existing.columns for col in required_cols):
                    # 为迭代添加类型忽略注释
                    for _, row in df_existing.iterrows(): # type: ignore[attr-defined]
                        # 明确 row 的类型
                        row_data: pd.Series = row # type: ignore[assignment]
                        # 增加空值检查
                        if pd.notna(row_data['院舍名稱']) and pd.notna(row_data['地址']):
                            # 对字符串进行Unicode规范化，确保不同表示形式的相同字符生成相同的哈希
                            normalized_name = str(row_data['院舍名稱']).strip().lower()
                            normalized_address = str(row_data['地址']).strip().lower()
                            unique_id = f"{normalized_name}||{normalized_address}".encode('utf-8')
                            hash_md5 = hashlib.md5(unique_id).hexdigest()
                            existing_hashes.add(hash_md5)
                        else:
                            logger.warning(f"发现去重数据中存在空值，跳过此行：{row_data.to_dict()}")
                    logger.info(f"成功加载 {len(df_existing)} 条现有数据用于去重。")
                else:
                    logger.warning(f"现有Excel文件 {excel_file} 缺少必要的列，无法进行去重。")
            else:
                logger.info(f"现有Excel文件 {excel_file} 为空，无需加载旧数据。")
        except Exception as e:
            logger.error(f"加载现有Excel文件 {excel_file} 失败: {e}，将不进行旧数据去重。") # 修正日志提示
    else:
        logger.info(f"未找到Excel文件 {excel_file}，将从空数据开始处理。")
    return existing_hashes

# --- 7. 函数：初始化 WebDriver ---
def initialize_webdriver() -> webdriver.Chrome | None:
    """
    初始化 Chrome WebDriver 实例，使用无头模式并优化性能。
    """
    options = webdriver.ChromeOptions()
    # 添加常见浏览器 User-Agent
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    options.add_argument(f"user-agent={user_agent}")
    
    options.add_argument("--headless")  # 无头模式
    options.add_argument("--disable-gpu")  # 禁用GPU加速
    options.add_argument("--no-sandbox")  # 禁用沙箱模式
    options.add_argument("--disable-dev-shm-usage")  # 解决Docker等环境下的/dev/shm问题
    options.add_argument("--log-level=3") # 设置日志级别，避免控制台输出过多信息
    # 为 add_experimental_option 添加类型忽略注释
    options.add_experimental_option("excludeSwitches", ["enable-automation"]) # type: ignore[call-arg] # 避免检测
    options.add_experimental_option('useAutomationExtension', False) # type: ignore[call-arg] # 避免检测
    options.add_argument("--disable-blink-features=AutomationControlled") # 避免被检测为自动化工具
    
    # 设置页面加载策略为"eager" - 不等待所有资源加载完成
    options.page_load_strategy = "eager"

    # 性能优化：禁用图片和CSS加载
    # 优化 prefs 字典的类型提示
    prefs: dict[str, int] = {
        "profile.managed_default_content_settings.images": 2,  # 禁用图片
        "profile.managed_default_content_settings.stylesheet": 2, # 禁用CSS
        "profile.default_content_setting_values.notifications": 2 # 禁用通知
    }
    # 添加更明确的类型忽略注释
    options.add_experimental_option("prefs", prefs) # type: ignore[call-arg] # 忽略 Pylance 的类型推断警告

    # 请根据您的Chrome WebDriver路径进行配置
    # 如果WebDriver在系统PATH中，则无需指定executable_path
    # from selenium.webdriver.chrome.service import Service # 导入 Service
    # service = Service(executable_path='/path/to/chromedriver') # 示例：Service(executable_path='/usr/local/bin/chromedriver')
    # driver = webdriver.Chrome(service=service, options=options)
    
    # 假设chromedriver在系统PATH中，或者与脚本在同一目录下
    try:
        driver = webdriver.Chrome(options=options)
        logger.info("WebDriver 初始化成功（无头模式）。")
        return driver
    except WebDriverException as e:
        logger.critical(f"WebDriver 初始化失败，请检查 ChromeDriver 是否已安装并配置正确，或路径是否正确: {e}")
        logger.critical("请确保您的 ChromeDriver 版本与 Chrome 浏览器版本兼容。")
        return None

# --- 8. 函数：使用Gemini API识别行政区 ---
def get_district(address: str, gemini_api_handler: GeminiAPIHandler) -> str | None:
    """
    使用Gemini API从香港地址中提取行政区和街道。
    Args:
        address (str): 需要识别的地址字符串。
        gemini_api_handler (GeminiAPIHandler): Gemini API处理实例，包含API密钥轮转和模型配置。
    Returns:
        str | None: 识别到的行政区信息，如果识别失败则返回None。
    """
    model = gemini_api_handler.get_model()
    if not model:
        logger.warning("Gemini模型未成功初始化，无法进行行政区识别。")
        return None

    prompt = gemini_api_handler.get_prompt(address)
    
    # API调用重试机制
    max_retries = 5 # 增加重试次数
    retry_delay = 60  # 秒，增加等待时间以应对速率限制
    for attempt in range(max_retries):
        # 定义默认重试延迟，以防API未提供
        default_retry_delay = 60 # 秒

        # 在每次API调用前增加延迟，以符合API速率限制
        # 第一次尝试前不等待，后续重试的等待由错误处理逻辑决定
        if attempt > 0:
            # 每次重试前，先尝试轮转密钥（非强制，会检查冷却）
            gemini_api_handler.rotate_key()
            # 确保模型已更新
            model = gemini_api_handler.get_model()
            if not model:
                logger.warning("Gemini模型在重试前未成功初始化，无法进行行政区识别。")
                return None

        try:
            # 增加API调用超时时间
            response = model.generate_content(
                prompt,
                generation_config=GenerationConfig(temperature=0), # type: ignore[attr-defined, call-arg]
                request_options={"timeout": 180}  # 增加超时时间到180秒
            )
            
            if response and response.text:
                raw_text = response.text.strip().replace("```json", "").replace("```", "").strip()
                try:
                    # 尝试解析为JSON
                    district_data = json.loads(raw_text)
                    district_info = district_data.get('行政区', '')
                    street_info = district_data.get('街道', '')
                    
                    if district_info and street_info:
                        result = f"行政区: {district_info}, 街道: {street_info}"
                    elif district_info:
                        result = f"行政区: {district_info}"
                    elif street_info:
                        result = f"街道: {street_info}"
                    else:
                        result = raw_text # 如果没有识别到特定字段，返回原始文本
                    
                    logger.info(f"地址 '{address}' 识别到行政区信息: {result}")
                    return result
                except json.JSONDecodeError:
                    logger.warning(f"Gemini API 返回的响应不是有效的JSON格式，地址: '{address}'。原始响应: {raw_text[:100]}...")
                    # 如果不是JSON，尝试使用正则表达式解析
                    # 修正正则表达式，使用贪婪匹配确保完整行政区名称，并处理可能没有街道的情况
                    match = re.match(r'行政区:\s*([^,]+),?\s*(?:街道:\s*(.+))?', raw_text)
                    if match:
                        district = match.group(1).strip()
                        street = match.group(2).strip() if match.group(2) else ''
                        result = f"行政区: {district}, 街道: {street}"
                        logger.info(f"地址 '{address}' 识别到行政区: {result} (通过正则解析)")
                        return result
                    else:
                        # 如果正则也无法解析，直接返回原始文本
                        logger.info(f"地址 '{address}' 识别到行政区: {raw_text} (非JSON/正则匹配)")
                        return raw_text
                except Exception as e:
                    logger.warning(f"处理Gemini API响应时发生错误: {e}，地址: '{address}'。原始响应: {raw_text[:100]}...")
                    return None
            else:
                logger.warning(f"Gemini API 未返回有效行政区信息，地址: '{address}'。请检查模型响应或提示词。")
                if attempt < max_retries - 1:
                    logger.info(f"无效响应，尝试切换API密钥并重试...")
                else:
                    logger.error(f"行政区识别多次重试后失败 (地址: '{address}'): 未返回有效信息。")
                    return None
                
        except google.api_core.exceptions.ResourceExhausted as e:
            logger.warning(f"行政区识别失败 (尝试 {attempt+1}/{max_retries}, 地址: '{address}'): 429 速率限制。")
            
            retry_after = default_retry_delay
            if hasattr(e, 'api_exception') and hasattr(e.api_exception, 'details'):
                for detail in e.api_exception.details:
                    if hasattr(detail, 'retry_delay') and hasattr(detail.retry_delay, 'seconds'):
                        retry_after = detail.retry_delay.seconds
                        logger.info(f"从API错误中提取到建议的重试延迟: {retry_after} 秒。")
                        break
            
            if attempt < max_retries - 1:
                logger.info(f"等待 {retry_after} 秒后重试...")
                time.sleep(retry_after)
                current_key = gemini_api_handler.api_keys[gemini_api_handler.current_key_index]
                gemini_api_handler.key_cooldown_until[current_key] = time.time() + retry_after
                gemini_api_handler.rotate_key(force_rotate=True)
            else:
                logger.error(f"行政区识别多次重试后失败 (地址: '{address}'): {e}")
                return None
        except Exception as e:
            logger.warning(f"行政区识别失败 (尝试 {attempt+1}/{max_retries}, 地址: '{address}'): {e}")
            logger.debug(f"完整错误信息: {str(e)}", exc_info=True)  # 添加详细错误日志
            if attempt < max_retries - 1:
                logger.info(f"非速率限制错误，尝试切换API密钥并重试...")
            else:
                logger.error(f"行政区识别多次重试后失败 (地址: '{address}'): {e}")
                return None

# --- 9. 主爬虫函数 ---
def crawl_elderly_homes(base_url: str, output_file: str, overwrite: bool, gemini_api_handler: GeminiAPIHandler, has_next_page: bool = True):
    """
    主爬虫函数，负责爬取安老院信息并保存。
    Args:
        base_url (str): 爬虫的起始URL。
        output_file (str): 数据保存的Excel文件名。
        overwrite (bool): 是否覆盖现有文件。
        gemini_api_handler (GeminiAPIHandler): Gemini API处理实例，包含API密钥轮转和模型配置。
    """
    driver: webdriver.Chrome | None = None
    all_elderly_homes_data: List[Dict[str, str]] = []
    existing_data_hashes: set[str] = load_existing_data_for_deduplication(output_file)
    current_page: int = 1

    # 定义保存数据的辅助函数
    def save_data_to_excel(data: List[Dict[str, str]], file_path: str, overwrite_mode: bool):
        """
        保存安老院数据到Excel文件
        Args:
            data: 安老院数据列表
            file_path: 输出文件路径
            overwrite_mode: 是否覆盖模式
        """
        if not data:
            logger.info("没有数据需要保存。")
            return

        # 定义期望的列顺序
        columns_order = ['行政区', '街道', '院舍名稱', '地址', '電話']
        df_to_save = pd.DataFrame(data, columns=columns_order)
        
        try:
            if overwrite_mode or not os.path.exists(file_path):
                # 覆盖模式或文件不存在时直接写入
                df_to_save.to_excel(file_path, index=False, engine='openpyxl') # type: ignore[call-overload]
                logger.info(f"保存 {len(df_to_save)} 条数据到 {file_path}。")
            else:
                # 追加模式：读取现有数据，合并后去重，再覆盖写入
                df_existing = pd.read_excel(file_path, engine='openpyxl') # type: ignore[call-overload]
                
                # 合并新旧数据
                df_combined = pd.concat([df_existing, df_to_save])
                
                # 去重（基于院舍名稱和地址）
                df_combined = df_combined.drop_duplicates(
                    subset=['院舍名稱', '地址'],
                    keep='last'  # 保留最新数据
                )
                
                # 覆盖写入整个文件
                df_combined.to_excel(file_path, index=False, engine='openpyxl') # type: ignore[call-overload]
                
                # 计算新增数据量
                new_count = len(df_combined) - len(df_existing)
                if new_count > 0:
                    logger.info(f"成功追加 {new_count} 条新数据到 {file_path}（文件现共有 {len(df_combined)} 条数据）。")
                else:
                    logger.info("没有新增数据（所有数据已存在）。")
        except Exception as e:
            logger.critical(f"保存数据到Excel文件 {file_path} 失败: {e}", exc_info=True)

    try:
        driver = initialize_webdriver()
        if not driver:
            return

        initial_url = base_url
        logger.info(f"开始爬取，初始URL: {initial_url}")
        
        retries = 0
        while retries < MAX_RETRIES:
            try:
                driver.get(initial_url)
                
                def page_loaded(driver: webdriver.Chrome) -> bool:
                    return "安老院" in driver.page_source or "elderly" in driver.page_source
                
                WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
                    lambda d: page_loaded(d) or EC.presence_of_element_located((By.CSS_SELECTOR, "div.view-search-result"))
                )
                
                if not page_loaded(driver):
                    raise TimeoutException("页面加载完成但未找到预期内容")
                    
                logger.info("页面加载完成。")
                break
            except TimeoutException as e:
                retries += 1
                if retries < MAX_RETRIES:
                    logger.warning(f"页面加载超时或内容未找到 ({retries}/{MAX_RETRIES}): {str(e)[:100]}")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error("页面加载多次重试后仍失败，请检查网络连接或网站状态。")
                    return
            except Exception as e:
                logger.error(f"初始页面加载时发生错误: {e}")
                if retries < MAX_RETRIES:
                    logger.warning(f"正在重试 ({retries}/{MAX_RETRIES})...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error("页面加载多次重试后仍失败，请检查网络连接或网站状态。")
                    return
        
        while True:
            logger.info(f"正在爬取第 {current_page} 页数据...")
            retries = 0
            page_data_extracted = False

            while retries < MAX_RETRIES:
                try:
                    WebDriverWait(driver, ELEMENT_WAIT_TIMEOUT).until(
                        EC.visibility_of_element_located((By.CSS_SELECTOR, "div.views-row"))
                    )
                    elderly_home_elements: List[WebElement] = driver.find_elements(By.CSS_SELECTOR, "div.col-md-6.col-xs-12.views-row")
                    
                    if not elderly_home_elements:
                        logger.warning(f"第 {current_page} 页未找到安老院列表项，可能已无数据或页面结构有变。尝试刷新页面并重试。")
                        driver.refresh()
                        WebDriverWait(driver, ELEMENT_WAIT_TIMEOUT).until(
                            EC.visibility_of_element_located((By.CSS_SELECTOR, "div.col-md-6.col-xs-12.views-row"))
                        )
                        elderly_home_elements = driver.find_elements(By.CSS_SELECTOR, "div.col-md-6.col-xs-12.views-row")
                        if not elderly_home_elements:
                            logger.error(f"刷新后仍未找到安老院列表项，跳过第 {current_page} 页。")
                            break

                    for element in elderly_home_elements:
                        name: str = ""
                        address: str = ""
                        phone: str = ""

                        try:
                            name_element: WebElement = element.find_element(By.CSS_SELECTOR, "div.title")
                            name = name_element.text.strip()
                        except NoSuchElementException:
                            logger.warning("未找到院舍名稱元素。")

                        try:
                            address_element: WebElement = element.find_element(By.CSS_SELECTOR, "div.basic_info div.address")
                            address = address_element.text.strip()
                        except NoSuchElementException:
                            logger.warning("未找到地址元素。")
                            address = ""

                        try:
                            phone_element: WebElement = element.find_element(By.CSS_SELECTOR, "div.basic_info div.tele")
                            phone = phone_element.text.strip().replace("電話:", "").strip()
                            if phone and not re.fullmatch(r'(\+\d{1,4}[\s-]?)?[\d\s()-]{5,}', phone):
                                logger.warning(f"电话号码格式异常: {phone}")
                        except NoSuchElementException:
                            logger.warning("未找到电话元素。")
                            phone = ""

                        if not name or not address or not phone:
                            logger.warning(f"发现不完整数据：名称='{name}', 地址='{address}', 电话='{phone}'。跳过此条。")
                            continue

                        unique_id = f"{name}||{address}".encode('utf-8')
                        hash_md5 = hashlib.md5(unique_id).hexdigest()
                        if hash_md5 in existing_data_hashes:
                            logger.info(f"数据已存在，跳过：{name} - {address}")
                            continue

                        elderly_home: Dict[str, str] = {
                            '院舍名稱': name,
                            '地址': address,
                            '電話': phone,
                            '行政区': '',  # 初始化行政区
                            '街道': ''     # 初始化街道
                        }
                        
                        # 检查是否配置了Gemini API Key，如果未配置则跳过行政区识别
                        if gemini_api_handler.api_keys:
                            # 使用线程池异步调用行政区识别函数，增加超时时间
                            with ThreadPoolExecutor(max_workers=1) as executor:
                                future = executor.submit(get_district, address, gemini_api_handler)
                                try:
                                    # 增加超时时间到180秒
                                    district_info_raw = future.result(timeout=180)
                                except Exception as e:
                                    logger.error(f"行政区识别异步调用失败或超时 (地址: '{address}'): {e}")
                                    district_info_raw = None

                            if district_info_raw:
                                # 尝试解析为JSON
                                try:
                                    district_data = json.loads(district_info_raw)
                                    elderly_home['行政区'] = district_data.get('行政区', '')
                                    elderly_home['街道'] = district_data.get('街道', '')
                                except json.JSONDecodeError:
                                    logger.warning(f"Gemini API 返回的响应不是有效的JSON格式，地址: '{address}'。原始响应: {district_info_raw[:100]}...")
                                    # 如果不是JSON，尝试使用正则表达式解析
                                    match = re.match(r'行政区:\s*([^,]+),?\s*(?:街道:\s*(.+))?', district_info_raw)
                                    if match:
                                        elderly_home['行政区'] = match.group(1).strip()
                                        if match.group(2):
                                            elderly_home['街道'] = match.group(2).strip()
                                    else:
                                        # 如果正则也无法解析，直接将原始文本作为行政区
                                        elderly_home['行政区'] = district_info_raw.strip()
                                        elderly_home['街道'] = ''
                                except Exception as e:
                                    logger.warning(f"处理Gemini API响应时发生错误: {e}，地址: '{address}'。原始响应: {district_info_raw[:100]}...")
                                    elderly_home['行政区'] = address # 如果识别失败，行政区默认为原地址
                                    elderly_home['街道'] = '' # 街道为空
                            else:
                                elderly_home['行政区'] = address # 如果识别失败，行政区默认为原地址
                                elderly_home['街道'] = '' # 街道为空
                        else:
                            logger.info("未配置Gemini API Key，跳过行政区识别功能。")
                            elderly_home['行政区'] = '' # 未启用功能时，行政区为空
                            elderly_home['街道'] = '' # 未启用功能时，街道为空
                        
                        all_elderly_homes_data.append(elderly_home)
                        existing_data_hashes.add(hash_md5)  # 添加到去重集合
                        logger.info(f"成功爬取安老院: {name} - {address}")

                    page_data_extracted = True
                    break
                except (TimeoutException, WebDriverException) as e:
                    retries += 1
                    logger.warning(f"第 {current_page} 页处理时WebDriver错误 ({retries}/{MAX_RETRIES}): {str(e)[:100]}")
                    
                    # 重启WebDriver
                    if driver:
                        try:
                            driver.quit()
                        except:
                            pass
                        driver = None
                    
                    driver = initialize_webdriver()
                    if not driver:
                        logger.error("WebDriver重启失败，终止爬取")
                        return
                    
                    logger.info(f"{RETRY_DELAY}秒后重试第 {current_page} 页...")
                    time.sleep(RETRY_DELAY)
                    
                    try:
                        driver.get(initial_url)
                        WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
                            lambda d: "安老院" in d.page_source or "elderly" in d.page_source
                        )
                        logger.info("页面重新加载成功，继续处理")
                    except Exception as reload_error:
                        logger.error(f"页面重新加载失败: {reload_error}")
                except Exception as e:
                    retries += 1
                    logger.warning(f"第 {current_page} 页处理时发生错误 ({retries}/{MAX_RETRIES}): {e}")
                    if retries < MAX_RETRIES:
                        logger.info(f"{RETRY_DELAY}秒后重试第 {current_page} 页...")
                        time.sleep(RETRY_DELAY)
                    else:
                        logger.error(f"第 {current_page} 页多次重试后仍失败，跳过此页。")
                        break # 如果多次重试失败，则跳出当前页的重试循环

            if not page_data_extracted:
                logger.warning(f"第 {current_page} 页未能成功提取数据，跳过翻页。")
                break # 如果当前页数据未能成功提取，则退出主循环

            if has_next_page:
                # 尝试查找下一页链接 (rel="next")
                next_page_link = None
                next_page_retries = 0 # 翻页重试计数器
                max_next_page_retries = 2 # 查找下一页链接的最大重试次数
                while next_page_retries < max_next_page_retries:
                    try:
                        next_page_link = WebDriverWait(driver, ELEMENT_WAIT_TIMEOUT).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "li.pager__item--next a[rel='next']"))
                        )
                        break # 找到链接，跳出重试循环
                    except (TimeoutException, NoSuchElementException):
                        logger.info(f"未找到下一页链接 (尝试 {next_page_retries+1}/{max_next_page_retries})。")
                        next_page_retries += 1
                        if next_page_retries < max_next_page_retries:
                            logger.info(f"等待 {RETRY_DELAY} 秒后重试查找下一页链接...")
                            time.sleep(RETRY_DELAY)
                        else:
                            logger.warning("多次尝试后仍未找到下一页链接。")
                            break # 达到最大重试次数，退出循环
                
                # 如果两次尝试后仍未找到下一页链接，尝试重新访问初始URL并再次查找
                if not next_page_link and next_page_retries == max_next_page_retries:
                    logger.info("尝试重新访问初始URL并再次查找下一页链接...")
                    try:
                        driver.get(initial_url)
                        WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
                            lambda d: "安老院" in d.page_source or "elderly" in d.page_source
                        )
                        # 确保获取到的是WebElement对象
                        next_page_link = WebDriverWait(driver, ELEMENT_WAIT_TIMEOUT).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "li.pager__item--next a[rel='next']"))
                        )
                        if next_page_link:
                            logger.info("重新访问后找到下一页链接。")
                        else:
                            logger.warning("重新访问后仍未找到下一页链接，视为爬取完成。")
                            break # 重新访问后仍未找到，视为爬取完成
                    except Exception as e:
                        logger.error(f"重新访问初始URL并查找下一页链接时出错: {e}，视为爬取完成。")
                        break # 重新访问失败，视为爬取完成

                if next_page_link:
                    try:
                        # 获取下一页的URL
                        next_page_url = next_page_link.get_attribute("href")
                        if not next_page_url or not next_page_url.startswith("http"): # 增加URL有效性检查
                            logger.warning("下一页链接的href属性为空或不是有效URL，爬取完成。")
                            break

                        # 检查是否是绝对URL，如果不是则拼接
                        # from urllib.parse import urljoin # 已在文件顶部导入
                        if not next_page_url.startswith("http"):
                            next_page_url = urljoin(driver.current_url, next_page_url)

                        logger.info(f"进入第 {current_page + 1} 页，URL: {next_page_url}")
                        driver.get(next_page_url)
                        current_page += 1
                        
                        # 等待页面加载，可以等待某个特定元素出现
                        WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "div.view-search-result"))
                        )
                        # 额外等待一小段时间，确保页面完全稳定
                        time.sleep(2)
                    except Exception as e:
                        logger.error(f"跳转到下一页或等待页面加载时出错: {e}")
                        break
                else:
                    logger.info("未找到下一页链接，爬取完成。")
                    break # 如果没有找到下一页链接，则退出循环
            else:
                logger.info("当前为无翻页模式，跳过翻页操作")
                break

    except Exception as e:
        logger.critical(f"爬虫主流程发生未捕获异常: {e}", exc_info=True)
        # 增强错误处理：保存当前已爬取数据到备份文件
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_file = f"backup_crawl_data_{timestamp}.json"
        try:
            with open(backup_file, "w", encoding='utf-8') as f:
                json.dump(all_elderly_homes_data, f, ensure_ascii=False, indent=4)
            logger.info(f"发生异常，已将当前数据备份到 {backup_file}")
        except Exception as backup_e:
            logger.error(f"数据备份失败: {backup_e}")
    finally:
        if driver:
            driver.quit()
            logger.info("WebDriver 已关闭。")

        # 保存剩余数据
        save_data_to_excel(all_elderly_homes_data, output_file, overwrite)

# --- 10. 主函数 ---
def main():
    parser = argparse.ArgumentParser(description="香港安老院信息爬虫")
    # 从环境变量获取URL，如果未设置则使用默认值
    # --base-url: 爬虫的目标网址 (必需参数)
    parser.add_argument("--url", type=str, required=True, help="安老院列表的URL")
    
    # 从环境变量获取输出文件名，否则使用默认值
    default_output_file = os.getenv("OUTPUT_FILE", DEFAULT_EXCEL_FILE)
    parser.add_argument("--output", type=str, default=default_output_file, help="输出Excel文件名")
    
    # 从环境变量获取覆盖模式，否则使用默认值
    default_overwrite = os.getenv("OVERWRITE", "False").lower() == "true"
    parser.add_argument("--overwrite", action="store_true", default=default_overwrite, help="覆盖现有Excel文件")
    
    # 从环境变量获取Gemini模型名称，否则使用默认值
    default_gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    parser.add_argument("--gemini-model", type=str, default=default_gemini_model, help="指定用于行政区识别的Gemini模型")
    
    # 从环境变量获取提示词，否则使用默认值
    default_prompt = os.getenv("PROMPT", "请从以下香港地址中提取行政区和街道信息。请严格按照以下JSON格式返回，不要包含其他任何文字或解释：\n```json\n{\"行政区\": \"<行政区名称>\", \"街道\": \"<街道名称>\"}\n```\n如果无法识别街道，请将街道字段留空。") # 优化提示词
    parser.add_argument("--prompt", type=str, default=default_prompt, help="自定义行政区识别的提示词")
    
    args = parser.parse_args()

    # 初始化Gemini API处理器
    gemini_api_handler = GeminiAPIHandler(
        api_keys=GEMINI_API_KEYS,
        model_name=args.gemini_model, # 使用命令行参数或环境变量获取的模型名称
        prompt_template=args.prompt # 使用命令行参数或环境变量获取的提示词
    ) # 确保括号正确闭合

    # 根据是否覆盖模式决定是否添加时间戳
    final_output_file = args.output
    if not args.overwrite:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # 在文件名中插入时间戳，例如：護老院-爬虫-20231027-153045.xlsx
        name, ext = os.path.splitext(args.output)
        final_output_file = f"{name}-{timestamp}{ext}"
        logger.info(f"已启用追加模式，输出文件将添加时间戳: {final_output_file}")
    
    crawl_elderly_homes(args.url, final_output_file, args.overwrite, gemini_api_handler)

if __name__ == "__main__":
    main()
