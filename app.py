"""
幼小衔接规划Agent - Web界面
使用 Streamlit 构建
"""

import os

from dotenv import load_dotenv
import streamlit as st

from assessment import calculate_assessment

load_dotenv()


# ==================== 辅助函数 ====================
def llm_enabled() -> bool:
    return bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_AUTH_TOKEN")
        or os.getenv("ANTHROPIC_API_KEY")
    )


@st.cache_resource
def get_agent(cache_buster: float):
    from kindergarten_agent_full import KindergartenAgent

    return KindergartenAgent()


def set_menu(target: str) -> None:
    st.session_state["menu"] = target


def scored_radio(prompt, options, index=2, key=None):
    scores = list(range(1, len(options) + 1))
    return st.radio(
        prompt,
        scores,
        index=index,
        horizontal=True,
        format_func=lambda x: options[x - 1],
        key=key,
    )


def render_plan(plan: dict) -> None:
    if not isinstance(plan, dict):
        st.markdown(str(plan))
        return

    if "raw" in plan:
        st.markdown(plan["raw"])
        return

    duration = plan.get("duration")
    if duration:
        st.markdown(f"**周期：** {duration}")

    weekly_goals = plan.get("weekly_goals", [])
    if weekly_goals:
        st.markdown("### 每周重点目标")
        for item in weekly_goals:
            st.markdown(f"- {item}")

    daily_activities = plan.get("daily_activities", [])
    if daily_activities:
        st.markdown("### 每日推荐活动")
        st.table(daily_activities)

    resources = plan.get("resources", [])
    if resources:
        st.markdown("### 推荐资源")
        for item in resources:
            st.markdown(f"- {item}")

    parent_tips = plan.get("parent_tips", [])
    if parent_tips:
        st.markdown("### 家长注意事项")
        for item in parent_tips:
            st.markdown(f"- {item}")

    evaluation = plan.get("evaluation_criteria", [])
    if evaluation:
        st.markdown("### 评估标准")
        for item in evaluation:
            st.markdown(f"- {item}")


FALLBACK_QA = {
    "要不要提前学小学内容": "不建议系统学习小学内容，但可以通过游戏方式接触：\n\n1. **亲子阅读** - 培养语感和文字认知\n2. **数学游戏** - 通过积木、扑克牌等理解数概念\n3. **生活实践** - 认识时间、钱币等\n\n避免超前学习导致孩子入学后失去新鲜感，产生厌学情绪。",
    "孩子不想去小学": "可以尝试以下方法：\n\n1. **参观小学** - 熟悉校园环境\n2. **读绘本** - 《我上小学了》《小魔怪要上学》\n3. **认识新朋友** - 了解邻居的哥哥姐姐\n4. **正向引导** - 避免用'小学很辛苦'恐吓",
    "孩子注意力不集中": "建议：\n\n1. **时间管理** - 从15分钟开始训练\n2. **环境营造** - 保持安静，关掉电视\n3. **游戏培养** - 拼图、积木、棋类\n4. **一次一件事** - 避免边玩边学",
    "如何培养时间观念": "方法：\n\n1. **可视化计时器** - 沙漏、番茄钟\n2. **固定作息表** - 严格执行\n3. **提前提醒** - 还有5分钟要出发\n4. **参与管理** - 再玩5分钟回家",
    "需要提前学拼音": "不建议系统学习拼音，但可以：\n\n1. **亲子阅读** - 培养语感\n2. **拼音游戏** - 增加熟悉度\n3. **避免超前** - 以免入学后厌学",
}


def local_answer(question: str) -> str:
    for key, answer in FALLBACK_QA.items():
        if key in question:
            return answer
    return "这个问题建议咨询专业教育人士或查看当地教育部门官方指南。"

# 页面配置
st.set_page_config(
    page_title="小桥 - 幼小衔接规划助手",
    page_icon="🎒",
    layout="wide"
)

# 自定义CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .feature-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 15px;
        border-radius: 5px;
    }
    .header-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 10px;
    }
    .header-subtitle {
        font-size: 1.2em;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'assessment_result' not in st.session_state:
    st.session_state.assessment_result = None
if 'plan' not in st.session_state:
    st.session_state.plan = None

# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("🎒 小桥助手")
    st.markdown("---")
    
    menu_options = ["🏠 首页", "📋 能力评估", "📅 生成计划", "💬 问答咨询"]
    current_index = menu_options.index(st.session_state.get('menu', "🏠 首页"))
    
    menu = st.radio(
        "功能菜单",
        menu_options,
        index=current_index,
        key="menu",
    )
    
    st.markdown("---")
    st.markdown("### 📖 使用说明")
    st.markdown("""
    1. 先进行**能力评估**
    2. 根据评估结果**生成计划**
    3. 有问题可以**问答咨询**
    """)

    st.markdown("---")
    st.markdown("### 🔌 LLM 状态")
    if llm_enabled():
        st.success("已启用个性化计划与问答")
    else:
        st.warning("未检测到 OPENAI_API_KEY，将显示示例计划与本地问答")

# ==================== 首页 ====================
if menu == "🏠 首页":
    st.markdown('<p class="header-title">🎒 幼小衔接规划助手</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">帮助孩子顺利过渡到小学生活</p>', unsafe_allow_html=True)
    
    # 核心功能介绍
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📋 能力评估</h3>
            <p>根据《3-6岁儿童学习与发展指南》，评估孩子语言、数学、社交等各方面发展水平</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📅 个性化计划</h3>
            <p>根据评估结果，生成针对性的幼小衔接计划，每日活动推荐</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>💬 专家问答</h3>
            <p>解答关于入学准备、能力培养等方面的疑问</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 快速评估入口
    st.markdown("### 🚀 快速开始")
    st.button(
        "开始能力评估 →",
        use_container_width=True,
        on_click=set_menu,
        args=("📋 能力评估",),
    )

# ==================== 能力评估 ====================
elif menu == "📋 能力评估":
    st.title("📋 孩子能力评估")
    st.markdown("请根据孩子的日常表现选择最符合的选项")
    
    with st.form("assessment_form", clear_on_submit=False):
        # 基本信息
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("孩子姓名", placeholder="请输入姓名")
        with col2:
            age = st.number_input("年龄", min_value=5.0, max_value=6.5, value=5.5, step=0.5)
        
        st.markdown("---")
        st.markdown("### 👂 倾听理解")
        st.markdown("孩子能否听懂并按要求做事？")
        lang_listening = scored_radio(
            "选择最符合的描述：",
            [
                "只能听懂简单的词语和指令，需要反复提醒",
                "能听懂简单指令，但复杂指令需要重复或简化",
                "能听懂日常对话和简单指令，基本能按要求做事",
                "能听懂较复杂的指令，按要求做事较主动",
                "能很好理解对话内容，准确执行各种指令",
            ],
            index=2,
            key="lang_listening",
        )
        
        st.markdown("### 🗣️ 表达交流")
        st.markdown("孩子能否清楚表达自己的想法？")
        lang_expression = scored_radio(
            "选择最符合的描述：",
            [
                "较少主动表达，说话较短或不清楚",
                "能说简单句子，但不太连贯",
                "能基本清楚表达自己的想法，但有时需要引导",
                "能较流畅地表达，讲述事情较完整",
                "能流畅、完整地讲述事情，词汇丰富",
            ],
            index=2,
            key="lang_expression",
        )
        
        st.markdown("### 📖 阅读习惯")
        st.markdown("孩子对阅读的兴趣和表现如何？")
        lang_reading = scored_radio(
            "选择最符合的描述：",
            [
                "不太愿意听故事或看书",
                "愿意听故事，但注意力较短",
                "喜欢听故事，能安静听一会儿",
                "有阅读兴趣，能自己翻看图书",
                "非常喜欢阅读，能专注阅读15分钟以上",
            ],
            index=2,
            key="lang_reading",
        )
        
        st.markdown("### ✍️ 书写兴趣")
        st.markdown("孩子对写字、画画的态度？")
        lang_writing = scored_radio(
            "选择最符合的描述：",
            [
                "不太愿意拿笔或涂画",
                "愿意涂画但握笔姿势不正确",
                "愿意模仿写简单笔画，姿势基本正确",
                "能正确握笔，写自己的名字",
                "对书写很有兴趣，姿势正确，字迹清楚",
            ],
            index=2,
            key="lang_writing",
        )
        
        st.markdown("---")
        st.markdown("### 🔢 数数能力")
        st.markdown("孩子数数和点数的能力？")
        math_counting = scored_radio(
            "选择最符合的描述：",
            [
                "能数到10，但经常跳数或漏数",
                "能数到10，基本手口一致",
                "能数到20，手口基本一致",
                "能数到20以上，理解数的含义",
                "能数到100，理解数的组成和顺序",
            ],
            index=2,
            key="math_counting",
        )
        
        st.markdown("### ➕ 计算能力")
        st.markdown("孩子进行简单加减的能力？")
        math_operation = scored_radio(
            "选择最符合的描述：",
            [
                "不太理解数量的增加和减少",
                "能通过数实物进行简单加减",
                "能做5以内加减法",
                "能做10以内加减法",
                "能做20以内加减法，理解运算含义",
            ],
            index=2,
            key="math_operation",
        )
        
        st.markdown("### 🔺 图形认知")
        st.markdown("孩子认识图形的能力？")
        math_shapes = scored_radio(
            "选择最符合的描述：",
            [
                "能认识圆形",
                "能认识圆形、三角形",
                "能认识正方形、长方形、三角形、圆形",
                "能说出图形特点并进行简单分类",
                "能认识立体图形（正方体、球体等）",
            ],
            index=2,
            key="math_shapes",
        )
        
        st.markdown("### 🧭 空间方位")
        st.markdown("孩子对方位和空间的理解？")
        math_space = scored_radio(
            "选择最符合的描述：",
            [
                "不太理解上下、前后",
                "能理解上下、前后",
                "基本能区分上下、前后、左右",
                "能准确区分并表达方位",
                "能理解更复杂的空间关系",
            ],
            index=2,
            key="math_space",
        )
        
        st.markdown("---")
        st.markdown("### 👫 社交能力")
        st.markdown("孩子与同伴交往的表现？")
        social = scored_radio(
            "选择最符合的描述：",
            [
                "较害羞，不太愿意与同伴玩耍",
                "愿意与同伴玩，但不知道怎么加入",
                "能与同伴一起玩，但有时会有冲突",
                "能主动与同伴交往，合作游戏",
                "社交能力强，有很多好朋友",
            ],
            index=2,
            key="social",
        )
        
        st.markdown("### 🧹 自理能力")
        st.markdown("孩子独立做事的能力？")
        self_care = scored_radio(
            "选择最符合的描述：",
            [
                "依赖大人较多，需要帮助",
                "能做简单事情，如收拾玩具",
                "基本能自己穿脱衣服",
                "能自己整理书包，如厕",
                "自理能力强，基本不需要大人帮忙",
            ],
            index=2,
            key="self_care",
        )
        
        st.markdown("### 🏃 运动能力")
        st.markdown("孩子的运动和动手能力？")
        motor = scored_radio(
            "选择最符合的描述：",
            [
                "大运动和精细动作发展较慢",
                "能进行基本运动，精细动作稍弱",
                "运动能力发展正常",
                "运动能力强，精细动作好",
                "运动能力突出，动手能力强",
            ],
            index=2,
            key="motor",
        )
        
        st.markdown("---")
        st.markdown("### 其他信息")
        col1, col2 = st.columns(2)
        with col1:
            interests = st.multiselect(
                "兴趣爱好",
                ["画画", "拼图", "积木", "阅读", "运动", "音乐", "科学小实验"]
            )
        with col2:
            concerns = st.multiselect(
                "家长担忧的问题",
                ["语言表达", "数学基础", "自理能力", "社交能力", "专注力", "入学焦虑"]
            )
        
        submitted = st.form_submit_button("提交评估", use_container_width=True, type="primary")
        
        if submitted:
            if not name:
                st.error("请输入孩子姓名")
            else:
                # 保存评估数据
                st.session_state.profile = {
                    "name": name,
                    "age": age,
                    "language": {
                        "listening": lang_listening,
                        "expression": lang_expression,
                        "reading": lang_reading,
                        "writing_interest": lang_writing
                    },
                    "math": {
                        "counting": math_counting,
                        "operation": math_operation,
                        "shapes": math_shapes,
                        "space": math_space
                    },
                    "social": social,
                    "self_care": self_care,
                    "motor": motor,
                    "interests": interests,
                    "concerns": concerns
                }
                st.session_state.assessment_result = calculate_assessment(st.session_state.profile)
                st.session_state.plan = None
                st.success("评估完成！")
                
                # 显示评估结果
                result = st.session_state.assessment_result
                profile = st.session_state.profile
                
                st.markdown("---")
                st.markdown(f"## 📊 {profile['name']}的评估报告")
                
                # 整体评价
                level_colors = {"优秀": "🟢", "良好": "🟡", "需加强关注": "🔴"}
                st.info(f"{level_colors.get(result['overall_level'], '')} 整体水平: {result['overall_level']}")
                
                # 优势
                if result['strengths']:
                    st.markdown("### ✨ 优势")
                    for s in result['strengths']:
                        st.markdown(f"- {s}")
                
                # 需加强
                if result['areas_to_improve']:
                    st.markdown("### 📌 需加强")
                    for a in result['areas_to_improve']:
                        st.markdown(f"- {a}")
                
                # 建议
                if result['recommendations']:
                    st.markdown("### 💡 建议")
                    for r in result['recommendations']:
                        st.markdown(f"- {r}")
        
        
    # 生成计划按钮（放在表单外，避免表单回调限制）
    st.markdown("---")
    st.button(
        "根据评估结果生成计划 →",
        use_container_width=True,
        on_click=set_menu,
        args=("📅 生成计划",),
    )

# ==================== 生成计划 ====================
elif menu == "📅 生成计划":
    st.title("📅 幼小衔接计划")
    
    if not st.session_state.profile:
        st.warning("请先完成能力评估")
        st.button(
            "去评估 →",
            on_click=set_menu,
            args=("📋 能力评估",),
        )
    else:
        st.markdown(f"### 👶 {st.session_state.profile['name']}的个性化计划")

        if llm_enabled():
            if st.button("生成个性化计划", use_container_width=True, type="primary"):
                with st.spinner("生成计划中..."):
                    try:
                        agent = get_agent(os.path.getmtime("kindergarten_agent_full.py"))
                        child_profile = agent.build_profile(st.session_state.profile)
                        st.session_state.plan = agent.generate_plan(child_profile)
                    except Exception as exc:
                        st.error(f"计划生成失败：{exc}")

            if st.session_state.plan:
                render_plan(st.session_state.plan)
            else:
                st.info("点击上方按钮生成个性化计划。")
        else:
            st.warning("未检测到 OPENAI_API_KEY，显示示例计划。")
            st.markdown("""
            ### 第一周：习惯养成
            | 时间 | 活动 | 目标 |
            |------|------|------|
            | 早晨 | 亲子阅读15分钟 | 语言发展 |
            | 下午 | 益智游戏 | 数学思维 |
            | 傍晚 | 户外运动30分钟 | 体能发展 |
            | 睡前 | 整理书包 | 自理能力 |
            
            ### 第二周：能力提升
            | 时间 | 活动 | 目标 |
            |------|------|------|
            | 早晨 | 讲述昨天的事情 | 语言表达 |
            | 下午 | 简单加减法游戏 | 数学运算 |
            | 傍晚 | 与同伴游戏 | 社交能力 |
            | 睡前 | 整理衣物 | 自理能力 |
            
            ### 第三周：综合训练
            ### 第四周：巩固强化
            """)
        
        st.markdown("""
        <div class="info-box">
            <h4>📌 家长注意事项</h4>
            <ul>
                <li>每天坚持，形成习惯</li>
                <li>多鼓励、少批评</li>
                <li>保持耐心，循序渐进</li>
                <li>定期回顾调整</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.button(
            "有更多问题？去问答咨询 →",
            use_container_width=True,
            on_click=set_menu,
            args=("💬 问答咨询",),
        )

# ==================== 问答咨询 ====================
elif menu == "💬 问答咨询":
    st.title("💬 问答咨询")
    st.markdown("有什么关于幼小衔接的问题，欢迎提问")
    
    # 常见问题快速入口
    st.markdown("### 常见问题")
    common_questions = [
        "要不要提前学小学内容？",
        "孩子不想去小学怎么办？",
        "孩子注意力不集中怎么办？",
        "如何培养时间观念？",
        "需要提前学拼音吗？"
    ]
    
    cols = st.columns(2)
    for i, q in enumerate(common_questions):
        with cols[i % 2]:
            if st.button(q, key=f"q_{i}"):
                st.session_state['current_question'] = q
    
    # 问答输入
    st.markdown("---")
    st.markdown("### 提问")
    
    if 'current_question' in st.session_state:
        default_value = st.session_state['current_question']
    else:
        default_value = ""
    
    question = st.text_area("请输入你的问题", value=default_value, height=100)
    
    if st.button("获取回答", use_container_width=True):
        if question:
            with st.spinner("思考中..."):
                if llm_enabled():
                    try:
                        agent = get_agent(os.path.getmtime("kindergarten_agent_full.py"))
                        answer = agent.chat(question)
                    except Exception as exc:
                        st.error(f"调用问答失败：{exc}")
                        answer = local_answer(question)
                else:
                    answer = local_answer(question)

                st.markdown("### 💡 回答")
                st.markdown(answer)

if __name__ == "__main__":
    pass
