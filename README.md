# 小桥 - 幼小衔接规划助手

基于 Streamlit 的 Web 应用，帮助 5-6 岁儿童完成幼小衔接的能力评估、计划生成与常见问题咨询。支持 OpenAI/Anthropic；可选启用知识库检索（RAG）。

**主要功能**
- 能力评估：语言、数学、社交、自理、运动等维度评分与建议
- 个性化计划：周目标、日活动、资源与评价标准
- 问答咨询：内置 FAQ；接入 LLM 后更灵活
- 知识库检索：基于 `knowledge_base.md`，可使用 Chroma 持久化向量库

**项目结构**
- `app.py` Streamlit Web 界面
- `assessment.py` 评估核心逻辑
- `kindergarten_agent_full.py` 主 Agent（含 RAG 检索）
- `kindergarten_agent.py` 简化版 Agent
- `knowledge_base.md` 知识库
- `requirements.txt` 依赖列表

**快速开始**
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

# Windows
copy .env.example .env
# macOS/Linux
cp .env.example .env

streamlit run app.py
```

如果未配置 LLM 的 API Key，应用会显示示例计划并使用本地问答。

**环境变量**
- `OPENAI_API_KEY` OpenAI API Key
- `OPENAI_BASE_URL` 自定义 OpenAI 兼容网关（可选）
- `OPENAI_MODEL` OpenAI 模型名（默认 `gpt-4o`）
- `OPENAI_EMBEDDING_MODEL` 向量模型名（默认 `text-embedding-3-small`）
- `OPENAI_USE_EMBEDDINGS` 是否启用向量检索（`1`/`0`）
- `KNOWLEDGE_BASE_PATH` 知识库路径（默认 `knowledge_base.md`）
- `CHROMA_DIR` Chroma 持久化目录（默认 `.chroma/kindergarten_transition`）
- `ANTHROPIC_AUTH_TOKEN` 或 `ANTHROPIC_API_KEY` Anthropic Key（可选）
- `ANTHROPIC_MODEL` Anthropic 模型名（启用 Anthropic 时必填）
- `ANTHROPIC_BASE_URL` Anthropic 网关地址（可选）

**说明**
- 首次运行并启用向量检索时会在 `.chroma/` 下创建本地向量库。
- 可按需替换 `knowledge_base.md` 以适配不同地区或口径。
