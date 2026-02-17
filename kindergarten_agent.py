"""
幼小衔接规划 Agent - 基于 LangChain + OpenAI
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel
from typing import List, Optional
import json

load_dotenv()

try:
    from langchain_core.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
except ImportError:  # fallback for older langchain
    from langchain.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )

# ==================== 数据模型 ====================

class ChildAbility(BaseModel):
    language: int  # 1-5
    math: int
    social: int
    self_care: int
    motor: int

class ChildProfile(BaseModel):
    name: str
    age: float  # 5-6
    abilities: ChildAbility
    interests: List[str]
    strengths: List[str]
    areas_for_improvement: List[str]
    family_context: str

class TransitionPlan(BaseModel):
    duration: str
    weekly_goals: List[str]
    daily_activities: List[dict]
    resources: List[str]
    evaluation_criteria: List[str]

# ==================== Agent 工具 ====================

def assess_child_level(abilities: ChildAbility) -> dict:
    """评估儿童发展水平"""
    total = sum([abilities.language, abilities.math, abilities.social, 
                 abilities.self_care, abilities.motor])
    if total >= 20:
        return {"level": "优秀", "description": "孩子各项能力发展良好，可以顺利过渡到小学"}
    elif total >= 12:
        return {"level": "良好", "description": "孩子具备一定基础，需要针对性加强某些方面"}
    else:
        return {"level": "需关注", "description": "建议增加相关能力的培养时间"}

def generate_plan(profile: ChildProfile, duration: str = "3个月") -> TransitionPlan:
    """生成幼小衔接计划"""
    return TransitionPlan(
        duration=duration,
        weekly_goals=[
            "培养时间观念，养成固定作息",
            "加强语言表达和倾听能力",
            "提升自理能力（整理书包、如厕等）",
            "增强社交合作能力",
            "数学思维启蒙"
        ],
        daily_activities=[
            {"time": "早晨", "activity": "亲子阅读15分钟", "goal": "语言发展"},
            {"time": "下午", "activity": "益智游戏/积木", "goal": "数学思维"},
            {"time": "傍晚", "activity": "户外运动30分钟", "goal": "体能发展"},
            {"time": "睡前", "activity": "整理自己的书包/玩具", "goal": "自理能力"}
        ],
        resources=[
            "《我的第一本数学思维书》",
            "幼小衔接APP: 洪恩识字、斑马思维",
            "推荐动画片: 《蓝色小考拉》、《小鼠波波》"
        ],
        evaluation_criteria=[
            "能独立完成基本自理行为",
            "能完整表达自己的想法",
            "能与其他小朋友合作游戏",
            "对学习有积极兴趣"
        ]
    )

def answer_question(question: str, profile: Optional[ChildProfile] = None) -> str:
    """回答幼小衔接相关问题"""
    # 这里可以用 RAG 检索更精确的回答
    qa_database = {
        "入学前准备": "1. 心理准备：带孩子参观小学，减少焦虑\n2. 能力准备：自理、表达、倾听\n3. 物品准备：书包、文具、姓名贴\n4. 作息调整：早睡早起，适应小学时间",
        "要不要提前学拼音": "不建议系统学习拼音，但可以：\n1. 通过绘本亲子阅读培养语感\n2. 玩拼音相关游戏增加熟悉度\n3. 避免超前学习导致厌学",
        "选择公立还是私立": "建议根据：\n1. 家庭经济情况\n2. 孩子的性格特点\n3. 学校的教学理念\n4. 离家距离",
        "幼小衔接关键期": "大班下学期（5-6岁）是关键期，重点：\n1. 学习习惯培养\n2. 时间观念建立\n3. 社交能力提升\n4. 自理能力强化"
    }
    
    for key, answer in qa_database.items():
        if key in question:
            return answer
    return "这个问题建议咨询专业教育人士或查看当地教育部门官方指南。"

# ==================== 工具注册 ====================

tools = [
    Tool(
        name="assess_level",
        func=lambda x: json.dumps(assess_child_level(ChildAbility(**json.loads(x)))),
        description="评估儿童发展水平，输入JSON格式的能力数据"
    ),
    Tool(
        name="generate_plan",
        func=lambda x: json.dumps(generate_plan(ChildProfile(**json.loads(x))).model_dump(), ensure_ascii=False),
        description="生成幼小衔接计划，输入儿童档案JSON"
    ),
    Tool(
        name="answer_qa",
        func=answer_question,
        description="回答幼小衔接相关问题"
    )
]

# ==================== Prompt 模板 ====================

system_prompt = """你是"小桥"——幼小衔接规划专家，专为5-6岁儿童家庭和教育工作者服务。

## 你的职责
1. 帮助家长评估孩子的发展水平
2. 生成个性化的幼小衔接计划
3. 解答关于幼小衔接的问题
4. 推荐适合的学习资源和活动

## 能力范围
- 儿童发展评估（语言、数学、社交、自理、运动）
- 幼小衔接计划制定
- 入学准备咨询
- 教育资源推荐

## 回答风格
- 温暖、专业、实用
- 给出具体可操作的建议
- 适当举例说明

## 注意事项
- 如果信息不足，先询问必要信息
- 遇到不确定的问题，建议咨询专业人士"""

human_prompt = """{input}

{agent_scratchpad}"""

# ==================== Agent 构建 ====================

openai_api_key = os.getenv("OPENAI_API_KEY", "")
openai_base_url = os.getenv("OPENAI_BASE_URL", "")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY 未设置")

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    temperature=0.7,
    api_key=openai_api_key,
    base_url=openai_base_url or None,
)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(human_prompt)
    ]
)

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=ConversationBufferMemory())

# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例对话
    questions = [
        "我的孩子即将上小学，需要准备什么？",
        "孩子语言能力较好，但自理能力一般，有什么建议？"
    ]
    
    for q in questions:
        print(f"\n用户: {q}")
        result = agent_executor.invoke({"input": q})
        print(f"小桥: {result['output']}")
