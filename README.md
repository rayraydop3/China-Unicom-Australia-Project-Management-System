# 澳洲电信专线价格预测系统

基于 CatBoost 机器学习模型的企业网络服务价格预测系统。

## 项目结构

```
PriceSystem_Web/
├── server.py              # Flask Web 服务器
├── trainadvance.py        # 模型训练脚本
├── telecom_model.cbm      # 训练好的模型文件
├── training_mvp_v5.csv    # 训练数据
├── templates/
│   └── index.html         # 前端界面
└── requirements.txt       # Python 依赖包
```

## 安装步骤

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型（如果模型文件不存在）

```bash
python trainadvance.py
```

这将生成 `telecom_model.cbm` 模型文件。

### 3. 启动 Web 服务器

```bash
python server.py
```

服务器将在 `http://localhost:5000` 启动。

## 使用方法

1. 打开浏览器访问 `http://localhost:5000`
2. 填写表单信息：
   - 运营商（Operator）
   - 带宽（Bandwidth，单位：Mbps）
   - 地区邮编（Region Code）
   - 产品类型（Type）
   - 服务周期（Term，月）
3. 点击"开始估价"按钮
4. 查看预测价格结果

## 功能特点

- 🤖 AI 驱动的价格预测
- 📍 智能地区归类（邮编自动转换为大区）
- 📊 数据增强技术提升模型准确性
- 🛡️ 价格保底机制（最低 50 澳元）
- 🎨 现代化黑色主题界面
- 💬 **AI 数据分析助手**：基于项目数据库的专有化 AI 分析
  - ✅ **支持 Ollama（完全免费，本地运行）** - 推荐！
  - ✅ 支持 Google Gemini（免费额度大）
  - ✅ 支持通义千问（国内访问快）
  - ✅ 支持 DeepSeek（备用）
  - 详见 `免费模型配置指南.md`

## 技术栈

- **后端**: Flask + CatBoost + DeepSeek API
- **前端**: HTML/CSS/JavaScript
- **机器学习**: CatBoost Regressor
- **AI 分析**: DeepSeek Chat API (RAG 检索增强生成)

## 注意事项

- 确保 `telecom_model.cbm` 文件存在，否则服务器无法启动
- 如果修改了训练脚本，需要重新训练模型
- 服务器默认运行在 `0.0.0.0:5000`，允许局域网访问

## 🚀 局域网部署

如果您需要将项目部署到公司局域网，让同事持续访问：

1. **查看部署指南**：`局域网部署指南.md` - 详细的部署步骤
2. **查看操作手册**：`操作手册.md` - 日常操作和维护说明
3. **使用启动脚本**：`启动服务器.bat` - 一键启动服务器
4. **配置自动启动**：`创建开机自启动.bat` - 设置开机自动启动（需管理员权限）
5. **配置防火墙**：`配置防火墙.bat` - 自动配置防火墙规则（需管理员权限）

### 快速部署步骤

1. 将项目复制到服务器电脑（建议路径：`D:\PriceSystem_Web`）
2. 安装 Python 和依赖：`pip install -r requirements.txt`
3. 配置防火墙：右键 `配置防火墙.bat` → 以管理员身份运行
4. 设置开机自启动：右键 `创建开机自启动.bat` → 以管理员身份运行
5. 测试启动：双击 `启动服务器.bat`
6. 记录 IP 地址，告诉同事访问：`http://[IP地址]:5000`

详细说明请查看 `局域网部署指南.md`。
