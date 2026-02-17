"""
幼小衔接评估核心逻辑
"""

from typing import Dict, List


LANG_KEYS = ["listening", "expression", "reading", "writing_interest"]
MATH_KEYS = ["counting", "operation", "shapes", "space"]


def _to_int(value: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 3


def _clamp(value: int, low: int = 1, high: int = 5) -> int:
    return max(low, min(high, value))


def _score(value: int) -> int:
    return _clamp(_to_int(value))


def calculate_assessment(profile: Dict) -> Dict:
    """计算评估结果"""
    language = profile.get("language", {})
    math = profile.get("math", {})

    lang_vals = [_score(language.get(k, 3)) for k in LANG_KEYS]
    math_vals = [_score(math.get(k, 3)) for k in MATH_KEYS]

    lang_avg = sum(lang_vals) / len(lang_vals)
    math_avg = sum(math_vals) / len(math_vals)

    social = _score(profile.get("social", 3))
    self_care = _score(profile.get("self_care", 3))
    motor = _score(profile.get("motor", 3))

    total = lang_avg + math_avg + social + self_care + motor

    strengths: List[str] = []
    areas_to_improve: List[str] = []
    recommendations: List[str] = []

    def add_feedback(score: int, strength: str, area: str, tip: str) -> None:
        if score >= 4:
            strengths.append(strength)
        elif score <= 2:
            areas_to_improve.append(area)
            recommendations.append(tip)

    # 语言分析
    add_feedback(
        _score(language.get("listening")),
        "倾听能力较好，能听懂指令",
        "倾听理解能力",
        "多与孩子交流复杂指令，锻炼理解能力",
    )
    add_feedback(
        _score(language.get("expression")),
        "语言表达清晰流畅",
        "语言表达能力",
        "每天15分钟亲子对话，鼓励孩子复述故事",
    )
    add_feedback(
        _score(language.get("reading")),
        "阅读兴趣浓厚",
        "阅读习惯",
        "建立固定阅读时间，选择孩子感兴趣的绘本",
    )
    add_feedback(
        _score(language.get("writing_interest")),
        "对书写有兴趣，能进行简单书写",
        "书写兴趣与握笔习惯",
        "用描红、描写名字等方式增强书写兴趣",
    )

    # 数学分析
    add_feedback(
        _score(math.get("counting")),
        "计数能力较强",
        "计数能力",
        "通过实物点数练习，20以内手口一致点数",
    )
    add_feedback(
        _score(math.get("operation")),
        "运算能力发展良好",
        "简单运算",
        "用实物游戏理解加减法含义",
    )
    add_feedback(
        _score(math.get("shapes")),
        "图形认知能力好",
        "图形认知",
        "通过积木、拼图认识基本几何图形",
    )
    add_feedback(
        _score(math.get("space")),
        "空间方位感较强",
        "空间感知",
        "多进行上下前后左右的方位游戏",
    )

    # 社交、自理、运动
    add_feedback(
        social,
        "社交能力强，愿意与同伴合作",
        "社交能力",
        "创造合作游戏机会，鼓励轮流与分享",
    )
    add_feedback(
        self_care,
        "自理能力强",
        "自理能力",
        "开始训练独立整理书包、穿脱衣物",
    )
    add_feedback(
        motor,
        "运动和动手能力好",
        "运动能力",
        "增加户外运动和精细动作练习",
    )

    if total >= 18:
        overall = "优秀"
        recommendations.append("孩子发展良好，可以顺利过渡到小学")
    elif total >= 12:
        overall = "良好"
    else:
        overall = "需加强关注"
        recommendations.append("建议增加幼小衔接训练的投入")

    return {
        "overall_level": overall,
        "strengths": strengths,
        "areas_to_improve": areas_to_improve,
        "recommendations": recommendations,
    }
