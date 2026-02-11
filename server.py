from flask import Flask, render_template, request, jsonify
from catboost import CatBoostRegressor
import pandas as pd
import os
import re
import requests
import json
from typing import List, Dict
try:
    from duckduckgo_search import DDGS
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    print("[警告] duckduckgo-search 未安装，联网搜索功能不可用")

app = Flask(__name__)

# --- 健康检查（用于排查“网络错误”到底是前端还是后端问题）---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

# --- 1. 核心辅助函数：把邮编翻译成大区 (必须和训练脚本一致) ---
def get_zone_from_postcode(val):
    # 使用正则表达式提取 4 位数字
    val_str = str(val)
    digits = re.findall(r'\b\d{4}\b', val_str)

    if not digits:
        return 'Unknown_Zone'

    postcode = int(digits[0])

    # 澳洲大区划分规则 (必须与 train_advanced.py 完全一致)
    if 2000 <= postcode <= 2234: return 'NSW_Metro'
    if 2500 <= postcode <= 2530: return 'NSW_Regional'
    if 2600 <= postcode <= 2620: return 'ACT_Metro'
    if 3000 <= postcode <= 3207: return 'VIC_Metro'
    if 4000 <= postcode <= 4179: return 'QLD_Metro'
    if 6000 <= postcode <= 6199: return 'WA_Metro'

    return 'Other_Regional'


# --- 2. 启动时加载模型和训练数据（用于置信度计算）---
MODEL_PATH = "telecom_model.cbm"
TRAINING_DATA_PATH = "training_mvp_v5.csv"

if not os.path.exists(MODEL_PATH):
    print(f"[错误] 严重错误：找不到模型文件 {MODEL_PATH}！")
    print("请先运行 python trainadvance.py 生成新模型！")
    exit()

print("正在加载增强版 AI 模型...")
model = CatBoostRegressor()
model.load_model(MODEL_PATH)
print("[成功] 模型加载完毕")

# 加载训练数据用于置信度计算
print("正在加载训练数据用于置信度分析...")
try:
    df_training = pd.read_csv(TRAINING_DATA_PATH)
    # 统一列名
    col_map = {
        'operator_manual': 'operator',
        'product_type_rule': 'product_type',
        'region_rule': 'region',
        'MRC (ex)': 'price',
        'mrc': 'price'
    }
    df_training = df_training.rename(columns=col_map)
    # 添加zone特征（与训练脚本一致）
    df_training['zone'] = df_training['region'].apply(get_zone_from_postcode)
    print(f"[成功] 训练数据加载完毕，共 {len(df_training)} 条记录")
except Exception as e:
    print(f"[警告] 无法加载训练数据: {e}，置信度计算将使用简化算法")
    df_training = None

print("服务启动中...")

# --- AI 模型配置（支持多个免费模型）---
# 可选模型：lmstudio, gemini, qwen, deepseek, ollama
AI_MODEL_TYPE = os.getenv('AI_MODEL_TYPE', 'lmstudio').lower()  # 默认使用 LM Studio（本地免费）

# LM Studio 配置（本地运行，完全免费）
LMSTUDIO_BASE_URL = os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234')
LMSTUDIO_MODEL = os.getenv('LMSTUDIO_MODEL', '')  # 使用 LM Studio 中加载的模型

# Ollama 配置（本地运行，完全免费）
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')  # 推荐模型：qwen2.5:7b, llama3.2, mistral

# Google Gemini 配置（免费额度大）
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyB-zg87-1nrJxDqV3UWUyxglEuvg_eXxXY')  # 默认使用提供的 API Key
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')  # 使用最新的 gemini-2.0-flash 模型
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# 通义千问配置（国内访问快，有免费额度）
QWEN_API_KEY = os.getenv('QWEN_API_KEY', '')
QWEN_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

# DeepSeek 配置（备用）
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# --- 加载公司知识库文件（company_knowledge.txt）---
COMPANY_KNOWLEDGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "company_knowledge.txt")

# 读取失败时使用的默认简介（确保程序不会崩溃）
_DEFAULT_COMPANY_CONTEXT = (
    "澳洲联通 (Australia Unicom) 是澳大利亚企业级网络服务提供商，"
    "提供企业光纤专线 (DIA)、SD-WAN、数据中心互联等服务。"
    "客服热线：1300-UNICOM，24小时中英双语服务，99.99% SLA 保障。"
)


def load_company_knowledge(filepath: str) -> str:
    """
    从 company_knowledge.txt 读取公司内部知识库。

    - 使用 UTF-8 编码打开，兼容中英文混合内容
    - 读取失败时打印错误日志并返回默认简介，不会导致程序崩溃

    Returns:
        str: 知识库文本内容
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            print(f"[警告] 知识库文件 {filepath} 为空，使用默认简介")
            return _DEFAULT_COMPANY_CONTEXT
        print(f"[成功] 公司知识库加载完成: {filepath} ({len(content)} 字符)")
        return content
    except FileNotFoundError:
        print(f"[警告] 找不到知识库文件: {filepath}，使用默认简介")
        return _DEFAULT_COMPANY_CONTEXT
    except UnicodeDecodeError as e:
        print(f"[错误] 知识库文件编码异常 (需要 UTF-8): {e}，使用默认简介")
        return _DEFAULT_COMPANY_CONTEXT
    except OSError as e:
        print(f"[错误] 读取知识库文件失败: {e}，使用默认简介")
        return _DEFAULT_COMPANY_CONTEXT


# 启动时加载公司知识库
company_context = load_company_knowledge(COMPANY_KNOWLEDGE_PATH)


# --- 构建训练数据知识库（用于 RAG 数据分析）---
def build_training_data_summary(df: pd.DataFrame) -> str:
    """基于训练数据构建统计摘要，作为 RAG 数据源补充"""
    if df is None or len(df) == 0:
        return ""
    
    parts: list[str] = []
    
    # 数据概览
    parts.append(f"## 训练数据统计摘要")
    parts.append(f"- 总数据量: {len(df)} 条记录")
    
    # 运营商分布
    operator_stats = df['operator'].value_counts()
    parts.append(f"\n## 运营商分布")
    for operator, count in operator_stats.items():
        parts.append(f"- {operator}: {count} 条 ({count / len(df) * 100:.1f}%)")
    
    # 产品类型分布
    product_stats = df['product_type'].value_counts()
    parts.append(f"\n## 产品类型分布")
    for product, count in product_stats.items():
        parts.append(f"- {product}: {count} 条 ({count / len(df) * 100:.1f}%)")
    
    # 地区分布
    zone_stats = df['zone'].value_counts()
    parts.append(f"\n## 地区分布")
    for zone, count in zone_stats.head(10).items():
        parts.append(f"- {zone}: {count} 条 ({count / len(df) * 100:.1f}%)")
    
    # 价格统计
    if 'price' in df.columns:
        ps = df['price'].describe()
        parts.append(f"\n## 价格统计（澳元/月）")
        parts.append(f"- 最低: ${ps['min']:.2f}  最高: ${ps['max']:.2f}")
        parts.append(f"- 平均: ${ps['mean']:.2f}  中位数: ${ps['50%']:.2f}")
    
    # 带宽统计
    if 'bandwidth_mbps' in df.columns:
        bs = df['bandwidth_mbps'].describe()
        parts.append(f"\n## 带宽统计（Mbps）")
        parts.append(f"- 范围: {bs['min']:.0f} ~ {bs['max']:.0f} Mbps")
        parts.append(f"- 平均: {bs['mean']:.0f} Mbps  中位数: {bs['50%']:.0f} Mbps")
    
    # 典型案例
    parts.append(f"\n## 典型价格案例")
    for operator in df['operator'].value_counts().head(5).index:
        op_data = df[df['operator'] == operator]
        for pt in op_data['product_type'].value_counts().head(2).index:
            pt_data = op_data[op_data['product_type'] == pt]
            if len(pt_data) > 0:
                parts.append(
                    f"- {operator} {pt}: 均价 ${pt_data['price'].mean():.2f}/月, "
                    f"均带宽 {pt_data['bandwidth_mbps'].mean():.0f} Mbps "
                    f"({len(pt_data)} 条)"
                )
    
    return "\n".join(parts)


# 构建训练数据摘要
training_data_summary = ""
if df_training is not None:
    training_data_summary = build_training_data_summary(df_training)
    print(f"[成功] 训练数据摘要构建完成，共 {len(training_data_summary)} 字符")


# --- 3. 访问主页 ---
@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/pricing')
def pricing():
    return render_template('index.html')

# --- 文章页面路由 ---
@app.route('/article/innovation-leadership')
def article_innovation():
    return render_template('article_innovation.html')

@app.route('/article/sme-communication')
def article_sme():
    return render_template('article_sme.html')

@app.route('/article/network-coverage')
def article_coverage():
    return render_template('article_coverage.html')


# --- 4. 预测接口 ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"收到请求: {data}")

        # 1. 提取用户输入的邮编
        user_region = str(data['region'])

        # 2. 翻译成 Zone (这是新模型需要的特征)
        derived_zone = get_zone_from_postcode(user_region)
        print(f"[地区转换] 用户输入 '{user_region}' -> AI识别为 '{derived_zone}'")

        # 3. 整理预测数据 (列名顺序要和训练时一致)
        X_pred = pd.DataFrame([{
            'operator': data['operator'],
            'product_type': data['product_type'],
            'bandwidth_mbps': float(data['bandwidth']),
            'term_months': int(data['term']),
            'zone': derived_zone  # 注意：这里传的是 zone，不是 region
        }])


        # 4. 预测
        raw_price = model.predict(X_pred)[0]

        # === [新增] 业务保底逻辑 (Business Rules) ===
        # 即使 AI 算出了负数，我们也不能展示给客户看
        MIN_PRICE = 50.0  # 设定：全网最低起步价 (你可以改成 100 或其他数字)

        if raw_price < MIN_PRICE:
            print(f"[警告] 触发保底: AI 算出 {raw_price}，已修正为 {MIN_PRICE}")
            price = MIN_PRICE
        else:
            price = raw_price

        # === [改进] 基于实际数据匹配度的置信度计算 ===
        confidence = 0
        
        if df_training is not None:
            # 方法1: 精确匹配 - 查找完全相同的参数组合
            exact_match = df_training[
                (df_training['operator'] == data['operator']) &
                (df_training['product_type'] == data['product_type']) &
                (df_training['zone'] == derived_zone) &
                (df_training['term_months'] == int(data['term']))
            ]
            
            if len(exact_match) > 0:
                # 有完全匹配的记录
                bandwidth = float(data['bandwidth'])
                # 检查带宽是否接近训练数据中的值
                bandwidth_diff = abs(exact_match['bandwidth_mbps'] - bandwidth)
                min_diff = bandwidth_diff.min()
                
                if min_diff <= 10:  # 带宽差异小于10Mbps
                    confidence = 85  # 高置信度
                elif min_diff <= 50:  # 带宽差异小于50Mbps
                    confidence = 70  # 中等置信度
                else:
                    confidence = 55  # 较低置信度
            else:
                # 方法2: 部分匹配 - 检查各个维度的数据覆盖度
                operator_count = len(df_training[df_training['operator'] == data['operator']])
                product_count = len(df_training[df_training['product_type'] == data['product_type']])
                zone_count = len(df_training[df_training['zone'] == derived_zone])
                term_count = len(df_training[df_training['term_months'] == int(data['term'])])
                
                total_records = len(df_training)
                
                # 计算各维度的覆盖率
                operator_coverage = operator_count / total_records * 100
                product_coverage = product_count / total_records * 100
                zone_coverage = zone_count / total_records * 100
                term_coverage = term_count / total_records * 100
                
                # 检查带宽范围
                bandwidth = float(data['bandwidth'])
                bandwidth_in_range = (df_training['bandwidth_mbps'].min() <= bandwidth <= df_training['bandwidth_mbps'].max())
                
                # 综合计算置信度（加权平均）
                # 运营商和产品类型权重较高
                base_confidence = (operator_coverage * 0.3 + product_coverage * 0.3 + 
                                  zone_coverage * 0.2 + term_coverage * 0.1)
                
                # 如果带宽在范围内，增加置信度
                if bandwidth_in_range:
                    base_confidence += 5
                
                # 由于数据量小（286条），整体降低置信度基准
                confidence = base_confidence * 0.85  # 降低15%以反映数据稀疏性
                
                # 如果某个维度数据很少，进一步降低
                if operator_count < 5 or product_count < 5 or zone_count < 3:
                    confidence *= 0.8  # 再降低20%
                
                # 确保置信度在合理范围内
                confidence = max(25, min(75, confidence))  # 限制在25-75之间（无精确匹配时）
        else:
            # 如果无法加载训练数据，使用简化算法
            confidence = 50  # 默认中等置信度
        
        # 最终限制在 0-100 范围内
        confidence = max(0, min(100, round(confidence, 0)))

        # 5. 返回结果
        return jsonify({
            'status': 'success',
            'price': round(price, 2),
            'confidence': round(confidence, 0),  # 返回置信度百分比
            'debug_info': f"已按 [{derived_zone}] 区域标准计算"  # 返回给前端看一眼，心里有底
        })

    except Exception as e:
        print(f"预测出错: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


# --- AI 模型调用函数 ---
def call_lmstudio(system_prompt: str, user_message: str) -> str:
    """调用 LM Studio 本地模型（完全免费）"""
    try:
        # 先检查服务器是否运行
        try:
            check_response = requests.get(f"{LMSTUDIO_BASE_URL}/v1/models", timeout=3)
            if check_response.status_code != 200:
                raise Exception("LM Studio 服务器未正常响应")
        except requests.exceptions.ConnectionError:
            raise Exception("无法连接到 LM Studio 服务器。\n\n请确保：\n1. LM Studio 已打开\n2. 模型已加载（Qwen3-VL-8B-Instruct）\n3. 本地服务器已启动（点击右下角 'Local Server' 或 'Server' 按钮）\n4. 服务器运行在端口 1234")
        except requests.exceptions.Timeout:
            raise Exception("LM Studio 服务器响应超时。\n\n可能原因：\n1. 服务器正在启动中，请等待 30 秒后重试\n2. 模型正在加载，请等待加载完成\n3. 服务器负载过高")
        
        # LM Studio 使用 OpenAI 兼容的 API 格式
        payload = {
            'model': LMSTUDIO_MODEL if LMSTUDIO_MODEL else 'local-model',  # 如果未指定，使用默认
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}
            ],
            'temperature': 0.7,
            'max_tokens': 1500,  # 降低 token 数量以加快生成速度
            'stream': False
        }
        
        url = f"{LMSTUDIO_BASE_URL}/v1/chat/completions"
        # 增加超时时间到 180 秒（8B 模型在 CPU 模式下可能需要 2-3 分钟）
        response = requests.post(url, json=payload, timeout=180)
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                raise Exception("LM Studio 返回格式异常")
        else:
            raise Exception(f"LM Studio API 错误: {response.status_code} - {response.text}")
    except requests.exceptions.Timeout:
        raise Exception("LM Studio 响应超时（180秒）。\n\n可能原因：\n1. 模型生成时间较长（8B 模型在 CPU 模式下需要 2-3 分钟）\n2. 服务器负载过高\n3. 未启用 GPU 加速\n\n建议：\n1. 在 LM Studio 设置中启用 GPU 加速（你有 RTX 3070 Ti）\n2. 检查 LM Studio 界面，确认模型已完全加载\n3. 尝试重启 LM Studio 服务器\n4. 如果持续超时，考虑使用更小的模型（7B Q4 量化版）")
    except requests.exceptions.ConnectionError:
        raise Exception("无法连接到 LM Studio 服务器。\n\n请确保：\n1. LM Studio 已打开\n2. 模型已加载（Qwen3-VL-8B-Instruct）\n3. 本地服务器已启动（点击右下角 'Local Server' 或 'Server' 按钮）\n4. 服务器运行在端口 1234")
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            raise Exception(f"LM Studio 响应超时。\n\n请检查：\n1. LM Studio 服务器是否正常运行\n2. 模型是否已完全加载\n3. 可以尝试重启 LM Studio 服务器")
        else:
            raise Exception(f"LM Studio 调用失败: {error_msg}")

def call_ollama(system_prompt: str, user_message: str) -> str:
    """调用 Ollama 本地模型（完全免费）"""
    try:
        payload = {
            'model': OLLAMA_MODEL,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}
            ],
            'stream': False
        }
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()['message']['content']
        else:
            raise Exception(f"Ollama API 错误: {response.status_code}")
    except requests.exceptions.ConnectionError:
        raise Exception("无法连接到 Ollama。请确保已安装并启动 Ollama 服务。\n安装: https://ollama.com/download\n启动后运行: ollama pull qwen2.5:7b")
    except Exception as e:
        raise Exception(f"Ollama 调用失败: {str(e)}")

def call_gemini(system_prompt: str, user_message: str) -> str:
    """调用 Google Gemini API（使用 gemini-2.0-flash 模型）"""
    if not GEMINI_API_KEY:
        raise Exception("Gemini API Key 未配置。请设置环境变量 GEMINI_API_KEY")
    
    try:
        # 构建完整的提示词
        full_prompt = f"{system_prompt}\n\n用户问题: {user_message}"
        
        # 使用 X-goog-api-key header 格式（根据你提供的 curl 示例）
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': GEMINI_API_KEY
        }
        
        payload = {
            'contents': [{
                'parts': [{
                    'text': full_prompt
                }]
            }]
        }
        
        # 使用 gemini-2.0-flash 模型
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            # 提取回复文本
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                raise Exception("Gemini API 返回格式异常")
        else:
            error_text = response.text
            raise Exception(f"Gemini API 错误: {response.status_code} - {error_text}")
    except Exception as e:
        raise Exception(f"Gemini 调用失败: {str(e)}")

def call_qwen(system_prompt: str, user_message: str) -> str:
    """调用通义千问 API（国内访问快，有免费额度）"""
    if not QWEN_API_KEY:
        raise Exception("通义千问 API Key 未配置。请设置环境变量 QWEN_API_KEY")
    
    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {QWEN_API_KEY}'
        }
        payload = {
            'model': 'qwen-turbo',
            'input': {
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_message}
                ]
            },
            'parameters': {
                'temperature': 0.7,
                'max_tokens': 2000
            }
        }
        response = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['output']['text']
        else:
            raise Exception(f"通义千问 API 错误: {response.status_code}")
    except Exception as e:
        raise Exception(f"通义千问调用失败: {str(e)}")

def call_deepseek(system_prompt: str, user_message: str) -> str:
    """调用 DeepSeek API（备用）"""
    if not DEEPSEEK_API_KEY:
        raise Exception("DeepSeek API Key 未配置。请设置环境变量 DEEPSEEK_API_KEY")
    
    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
        }
        payload = {
            'model': 'deepseek-chat',
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}
            ],
            'temperature': 0.7,
            'max_tokens': 2000
        }
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"DeepSeek API 错误: {response.status_code}")
    except Exception as e:
        raise Exception(f"DeepSeek 调用失败: {str(e)}")

# --- 网络搜索功能 ---
def search_web(query: str, max_results: int = 3) -> List[Dict]:
    """使用 DuckDuckGo 搜索网络资料（完全免费，无需 API Key）"""
    if not WEB_SEARCH_AVAILABLE:
        return []
    try:
        search_results = []
        with DDGS() as ddgs:
            # 搜索相关结果
            results = list(ddgs.text(query, max_results=max_results))
            for result in results:
                search_results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('body', ''),
                    'url': result.get('href', '')
                })
        return search_results
    except Exception as e:
        print(f"[搜索] 搜索失败: {e}")
        return []

def should_search(user_message: str) -> bool:
    """判断是否需要联网搜索"""
    # 关键词列表：如果问题包含这些词，可能需要搜索
    search_keywords = [
        '最新', '最近', '现在', '当前', '今年', '明年', '未来', '趋势',
        '新闻', '更新', '变化', '涨价', '降价', '政策', '规定',
        '市场', '行业', '竞争', '对比', '比较', '资料', '信息'
    ]
    user_lower = user_message.lower()
    return any(keyword in user_lower for keyword in search_keywords)

# --- 5. AI 分析接口（支持多个免费模型 + 联网搜索）---
@app.route('/deepseek/chat', methods=['POST'])
def ai_chat():
    """AI 对话接口，基于项目数据进行专有化分析，支持多个免费模型和联网搜索"""
    try:
        data = request.json
        user_message = data.get('message', '')
        requested_model = data.get('model', '').lower()  # 从前端获取请求的模型
        enable_search = data.get('enable_search', True)  # 默认启用搜索
        
        # 如果前端指定了模型，使用前端指定的；否则使用默认配置
        model_to_use = requested_model if requested_model in ['lmstudio', 'ollama', 'gemini', 'qwen', 'deepseek'] else AI_MODEL_TYPE
        
        if not user_message:
            return jsonify({'status': 'error', 'message': '消息不能为空'})
        
        # 联网搜索（如果需要）
        search_info = ""
        if enable_search and should_search(user_message):
            print(f"[搜索] 正在搜索: {user_message[:50]}...")
            search_results = search_web(user_message, max_results=3)
            if search_results:
                search_info = "\n\n=== 网络搜索结果 ===\n"
                for i, result in enumerate(search_results, 1):
                    search_info += f"\n{i}. {result['title']}\n"
                    search_info += f"   {result['snippet'][:200]}...\n"
                    search_info += f"   来源: {result['url']}\n"
                search_info += "\n请结合以上网络搜索结果和项目数据库来回答用户问题。"
                print(f"[搜索] 找到 {len(search_results)} 条结果")
            else:
                print(f"[搜索] 未找到相关结果")
        
        # === 构建系统提示词：公司知识库 + 训练数据 + 搜索结果 ===
        # 将 company_knowledge.txt 和训练数据摘要合并为完整的内部资料
        context_sections = [company_context]  # 公司知识库（来自 company_knowledge.txt）
        if training_data_summary:
            context_sections.append(training_data_summary)  # 训练数据统计摘要
        if search_info:
            context_sections.append(search_info)  # 网络搜索结果（如有）

        context_data = "\n\n".join(context_sections)

        system_prompt = (
            "你是澳洲联通 (Australia Unicom) 的 AI 客服。"
            "请根据以下内部资料回答用户问题。"
            "如果用户问的内容不在资料中，请礼貌地引导他们拨打客服热线 1300-UNICOM 或访问官网查询。\n\n"
            "[内部资料开始]\n"
            f"{context_data}\n"
            "[内部资料结束]\n\n"
            "回答要求：\n"
            "1. 严格基于内部资料回答，不要编造不存在的信息\n"
            "2. 涉及价格时，引用官方参考价格并说明实际价格可能浮动\n"
            "3. 涉及数据分析时，引用训练数据统计摘要中的具体数字\n"
            "4. 用中文回答，专业、简洁、易懂\n"
            "5. 如果信息不足，建议用户使用 AI 估价系统或联系 1300-UNICOM\n"
            "6. 如果引用了网络搜索结果，请注明信息来源"
        )
        
        print(f"[AI-{model_to_use.upper()}] 收到问题: {user_message[:50]}...")
        
        # 根据选择的模型类型调用相应的 API
        try:
            if model_to_use == 'lmstudio':
                ai_response = call_lmstudio(system_prompt, user_message)
            elif model_to_use == 'ollama':
                ai_response = call_ollama(system_prompt, user_message)
            elif model_to_use == 'gemini':
                ai_response = call_gemini(system_prompt, user_message)
            elif model_to_use == 'qwen':
                ai_response = call_qwen(system_prompt, user_message)
            elif model_to_use == 'deepseek':
                ai_response = call_deepseek(system_prompt, user_message)
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'不支持的模型类型: {model_to_use}。支持的类型: lmstudio, ollama, gemini, qwen, deepseek'
                })
            
            print(f"[AI-{model_to_use.upper()}] 回答生成成功")
            
            return jsonify({
                'status': 'success',
                'response': ai_response
            })
            
        except Exception as api_error:
            error_msg = str(api_error)
            print(f"[AI-{model_to_use.upper()}] 错误: {error_msg}")
            
            # 如果是 Gemini 配额错误，提供友好的提示
            if model_to_use == 'gemini' and ('429' in error_msg or 'quota' in error_msg.lower() or 'RESOURCE_EXHAUSTED' in error_msg):
                friendly_msg = """Gemini API 配额已用完。建议切换到其他免费模型：

1. **Ollama（推荐）** - 完全免费，本地运行，无配额限制
   - 需要先安装 Ollama：https://ollama.com/download
   - 然后运行：ollama pull qwen2.5:7b

2. **通义千问** - 有免费额度，国内访问快
   - 需要配置 QWEN_API_KEY

请在对话框顶部的模型选择器中切换到其他模型。"""
                return jsonify({
                    'status': 'error',
                    'message': friendly_msg
                })
            
            return jsonify({
                'status': 'error',
                'message': error_msg
            })
            
    except requests.exceptions.Timeout:
        return jsonify({'status': 'error', 'message': '请求超时，请稍后重试'})
    except Exception as e:
        print(f"[AI] 出错: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    # host='0.0.0.0' 允许局域网访问，避免出现 Connection Refused
    import socket
    # 获取本机IP地址
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"\n{'='*50}")
        print(f"服务器已启动")
        print(f"{'='*50}")
        print(f"本地访问: http://127.0.0.1:5000")
        print(f"局域网访问: http://{local_ip}:5000")
        print(f"{'='*50}")
        print(f"告诉同事在浏览器中输入: http://{local_ip}:5000")
        print(f"{'='*50}\n")
    except:
        print("\n服务器已启动，监听所有网络接口 (0.0.0.0:5000)")
        print("请运行 ipconfig 查看您的IP地址\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')