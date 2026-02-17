"""
幼小衔接规划 Agent - 完整版
包含 RAG 知识库检索功能
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:  # fallback for older langchain
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import StrOutputParser

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # fallback for older langchain
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from assessment import calculate_assessment

load_dotenv()

# ==================== 配置 ====================

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")
    EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
    OPENAI_USE_EMBEDDINGS = os.getenv("OPENAI_USE_EMBEDDINGS", "1").lower() not in ("0", "false", "no")
    KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "knowledge_base.md")
    CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma/kindergarten_transition")
    ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "")

# ==================== 数据模型 ====================

class LanguageAbility(BaseModel):
    listening: int = 3  # 倾听能力 1-5
    expression: int = 3  # 表达能力 1-5
    reading: int = 3  # 阅读能力 1-5
    writing_interest: int = 3  # 书写兴趣 1-5

class MathAbility(BaseModel):
    counting: int = 3  # 计数能力 1-5
    operation: int = 3  # 运算能力 1-5
    shapes: int = 3  # 图形认知 1-5
    space: int = 3  # 空间感知 1-5

class ChildProfile(BaseModel):
    name: str = ""
    age: float = 5.5
    language: LanguageAbility = LanguageAbility()
    math: MathAbility = MathAbility()
    social_level: int = 3
    self_care_level: int = 3
    motor_level: int = 3
    interests: List[str] = []
    concerns: List[str] = []

    def get_language_avg(self) -> float:
        return (self.language.listening + self.language.expression + 
                self.language.reading + self.language.writing_interest) / 4
    
    def get_math_avg(self) -> float:
        return (self.math.counting + self.math.operation + 
                self.math.shapes + self.math.space) / 4

class AssessmentResult(BaseModel):
    overall_level: str
    strengths: List[str]
    areas_to_improve: List[str]
    recommendations: List[str]

# ==================== RAG 知识库 ====================

class KnowledgeBase:
    def __init__(self):
        self.use_embeddings = bool(
            Config.OPENAI_API_KEY and Config.EMBEDDING_MODEL and Config.OPENAI_USE_EMBEDDINGS
        )
        self.embeddings = None
        self.vectorstore = None
        self.raw_chunks: List[str] = []
        self.knowledge_path = Path(Config.KNOWLEDGE_BASE_PATH)
        self.persist_dir = Path(Config.CHROMA_DIR)
        self._init_knowledge_base()
    
    def _init_knowledge_base(self):
        if not self.knowledge_path.exists():
            raise FileNotFoundError(f"知识库文件不存在: {self.knowledge_path}")

        if not self.use_embeddings:
            self.raw_chunks = self._load_raw_chunks()
            return

        try:
            self.embeddings = OpenAIEmbeddings(
                model=Config.EMBEDDING_MODEL,
                api_key=Config.OPENAI_API_KEY,
                base_url=Config.OPENAI_BASE_URL or None,
            )
            self.persist_dir.mkdir(parents=True, exist_ok=True)

            self.vectorstore = Chroma(
                collection_name="kindergarten_transition",
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_dir),
            )

            if self._is_empty():
                docs = self._load_documents()
                self.vectorstore.add_documents(docs)
                self.vectorstore.persist()
        except Exception:
            self.use_embeddings = False
            self.vectorstore = None
            self.raw_chunks = self._load_raw_chunks()

    def _load_documents(self):
        loader = TextLoader(str(self.knowledge_path), encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        return text_splitter.split_documents(documents)

    def _load_raw_chunks(self) -> List[str]:
        loader = TextLoader(str(self.knowledge_path), encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
        )
        chunks = text_splitter.split_documents(documents)
        return [doc.page_content for doc in chunks]

    def _is_empty(self) -> bool:
        try:
            return self.vectorstore._collection.count() == 0
        except Exception:
            return True
    
    def retrieve(self, query: str, k: int = 3) -> str:
        if self.use_embeddings and self.vectorstore is not None:
            docs = self.vectorstore.similarity_search(query, k=k)
            return "\n".join([doc.page_content for doc in docs])

        if not self.raw_chunks:
            return ""

        terms = set(re.findall(r"\w+", query.lower()))
        scored = []
        for chunk in self.raw_chunks:
            text = chunk.lower()
            score = sum(1 for t in terms if t in text)
            if score > 0:
                scored.append((score, chunk))

        if not scored:
            return "\n".join(self.raw_chunks[:k])

        scored.sort(key=lambda x: x[0], reverse=True)
        return "\n".join([chunk for _, chunk in scored[:k]])

# ==================== Agent 核心 ====================

class KindergartenAgent:
    def __init__(self):
        self.llm = self._build_llm()
        self.knowledge_base = KnowledgeBase()
        self.profile: Optional[ChildProfile] = None

    def _build_llm(self):
        if not Config.OPENAI_API_KEY:
            if Config.ANTHROPIC_AUTH_TOKEN or Config.ANTHROPIC_API_KEY:
                try:
                    from langchain_anthropic import ChatAnthropic
                except ImportError as exc:
                    raise ImportError("缺少依赖：langchain-anthropic") from exc

                model = Config.ANTHROPIC_MODEL
                if not model:
                    raise ValueError("ANTHROPIC_MODEL 未设置")

                api_key = Config.ANTHROPIC_API_KEY or Config.ANTHROPIC_AUTH_TOKEN
                default_headers = None
                if Config.ANTHROPIC_AUTH_TOKEN and not Config.ANTHROPIC_API_KEY:
                    default_headers = {"Authorization": f"Bearer {Config.ANTHROPIC_AUTH_TOKEN}"}

                return ChatAnthropic(
                    model=model,
                    temperature=0.7,
                    max_tokens=1024,
                    api_key=api_key,
                    base_url=Config.ANTHROPIC_BASE_URL or None,
                    default_headers=default_headers,
                )

            raise ValueError("未检测到可用的 LLM Key（OPENAI_API_KEY/ANTHROPIC_AUTH_TOKEN）")

        return ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=0.7,
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL or None,
        )
    
    def _build_system_prompt(self) -> str:
        return """你是"小桥"——幼小衔接规划专家，专为5-6岁儿童家庭和教育工作者服务。

## 你的专长
- 儿童发展评估与分析
- 个性化幼小衔接计划制定
- 入学准备指导
- 教育资源推荐

## 回答原则
1. 温暖、专业、实用
2. 给出具体可操作的建议
3. 根据孩子特点个性化建议
4. 必要时询问更多信息

## 知识库参考
{knowledge_base}"""
    
    def build_profile(self, profile_data: dict) -> ChildProfile:
        return ChildProfile(
            name=profile_data.get("name", ""),
            age=profile_data.get("age", 5.5),
            language=LanguageAbility(**profile_data.get("language", {})),
            math=MathAbility(**profile_data.get("math", {})),
            social_level=profile_data.get("social", 3),
            self_care_level=profile_data.get("self_care", 3),
            motor_level=profile_data.get("motor", 3),
            interests=profile_data.get("interests", []),
            concerns=profile_data.get("concerns", []),
        )

    def assess_child(self, profile: ChildProfile) -> AssessmentResult:
        """评估儿童发展水平 - 基于《3-6岁儿童学习与发展指南》"""
        profile_dict = {
            "language": profile.language.model_dump(),
            "math": profile.math.model_dump(),
            "social": profile.social_level,
            "self_care": profile.self_care_level,
            "motor": profile.motor_level,
        }
        result = calculate_assessment(profile_dict)
        return AssessmentResult(**result)
    
    def generate_plan(self, profile: ChildProfile, duration: str = "3个月") -> dict:
        """生成个性化计划"""
        assessment = self.assess_child(profile)
        
        prompt = f"""请为以下孩子生成一个{duration}的幼小衔接计划：

孩子信息：
- 年龄：{profile.age}岁
- 语言能力：倾听{profile.language.listening}/5，表达{profile.language.expression}/5，阅读{profile.language.reading}/5，书写兴趣{profile.language.writing_interest}/5
- 数学能力：计数{profile.math.counting}/5，运算{profile.math.operation}/5，图形{profile.math.shapes}/5，空间{profile.math.space}/5
- 社交能力：{profile.social_level}/5
- 自理能力：{profile.self_care_level}/5
- 运动能力：{profile.motor_level}/5
- 兴趣爱好：{', '.join(profile.interests)}
- 家长担忧：{', '.join(profile.concerns)}

评估结果：
- 整体水平：{assessment.overall_level}
- 优势：{', '.join(assessment.strengths) if assessment.strengths else '暂无明显优势'}
- 需加强：{', '.join(assessment.areas_to_improve) if assessment.areas_to_improve else '暂无明显不足'}

请生成：
1. 每周重点目标（4周）
2. 每日推荐活动
3. 推荐资源
4. 家长注意事项

请严格只返回JSON，不要包含解释、markdown或代码块。JSON结构示例：
{{
  "duration": "{duration}",
  "weekly_goals": ["..."],
  "daily_activities": [{{"time": "...", "activity": "...", "goal": "..."}}],
  "resources": ["..."],
  "parent_tips": ["..."],
  "evaluation_criteria": ["..."]
}}
"""
        response = self.llm.invoke(prompt)
        raw_content = response.content
        if isinstance(raw_content, dict):
            return raw_content
        if isinstance(raw_content, list):
            # 兼容 content blocks：优先取 type=text 的段落
            text_chunks = []
            for item in raw_content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        text_chunks.append(item["text"])
                elif isinstance(item, str):
                    text_chunks.append(item)
            if text_chunks:
                raw_content = "\n".join(text_chunks)
            else:
                return {"raw": str(raw_content)}
        if not isinstance(raw_content, (str, bytes, bytearray)):
            return {"raw": str(raw_content)}

        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            # 尝试提取JSON片段
            match = re.search(r"(\{.*\}|\[.*\])", raw_content, re.S)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            return {"raw": raw_content}
    
    def chat(self, message: str) -> str:
        """对话问答"""
        # 检索知识库
        relevant_knowledge = self.knowledge_base.retrieve(message)
        
        # 构建提示
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._build_system_prompt().format(knowledge_base=relevant_knowledge)),
            ("human", "{input}")
        ])
        
        # 生成回答
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"input": message})

# ==================== 主程序 ====================

def main():
    agent = KindergartenAgent()
    
    # 示例：创建孩子档案
    profile = ChildProfile(
        name="小明",
        age=5.5,
        language=LanguageAbility(
            listening=4,
            expression=4,
            reading=3,
            writing_interest=3
        ),
        math=MathAbility(
            counting=4,
            operation=3,
            shapes=3,
            space=3
        ),
        social_level=3,
        self_care_level=2,
        motor_level=4,
        interests=["画画", "拼图"],
        concerns=["自理能力", "专注力"]
    )
    
    # 1. 评估
    print("=" * 50)
    print("发展评估结果：")
    result = agent.assess_child(profile)
    print(f"整体水平: {result.overall_level}")
    print(f"优势: {result.strengths}")
    print(f"需加强: {result.areas_to_improve}")
    print(f"建议: {result.recommendations}")
    
    # 2. 生成计划
    print("\n" + "=" * 50)
    print("幼小衔接计划：")
    plan = agent.generate_plan(profile)
    print(plan)
    
    # 3. 问答
    print("\n" + "=" * 50)
    print("问答测试：")
    questions = [
        "孩子不想去小学怎么办？",
        "需要提前学拼音吗？"
    ]
    for q in questions:
        print(f"\n问: {q}")
        print(f"答: {agent.chat(q)}")

if __name__ == "__main__":
    main()
