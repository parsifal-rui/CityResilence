# CityResilence - 城市韧性事件图谱抽取

基于 DeepSeek LLM 从新闻文本中抽取城市灾害事件图谱（事件类型 + 时空信息 + 关系）。

## 📊 项目简介

从新闻报道中自动抽取结构化的灾害事件信息：

- **事件类型**：Driver（驱动因素）/ Modulator（调节因素）/ Hazard（灾害）/ Impact（影响）
- **时空信息**：时间（精确到天）/ 地点（城市级别 + 行政层级）
- **事件关系**：引发 / 加剧 / 削弱 / 增强 / 缓解 / 抑制

## 🗂️ 数据集

- **测试样本**：`data/articles_test_20.csv`（20 条广东灾害新闻，2024年8-9月）
- **实体库**：`data/entities_by_type.json`（567 个预定义实体，分 4 类）
- **知识库**：`data/graph_database_export.xlsx`（454 条三元组）

## 🚀 快速开始

### 方案 A：本地运行（Without-RAG）

# 1. 克隆仓库

git clone git@github.com:parsifal-rui/CityResilence.git
cd CityResilence

# 2. 安装依赖

pip install -r requirements.txt

# 3. 设置 API Key

export DEEPSEEK_API_KEY="your_api_key_here"

# 4. 运行抽取（20 条测试样本）

python src/deepseek_event_schema.py
