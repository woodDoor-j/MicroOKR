import json
import os
import re
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# ====== 页面配置 ======
st.set_page_config(
    page_title="Micro-OKR：微团队目标对齐器",
    page_icon="🎯",
    layout="wide",
)

# ====== 智谱 GLM-4 配置 ======
GLM_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLM_MODEL = "glm-4"
STATUS_OPTIONS = ["未开始", "进行中", "已完成"]

# ====== 本地持久化（保证刷新/重开不丢状态） ======
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")
_DATA_FILE = os.path.join(_DATA_DIR, "tasks.json")


def _load_state() -> dict | None:
    try:
        if os.path.exists(_DATA_FILE):
            with open(_DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _save_state() -> None:
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        payload = {
            "tasks": st.session_state.get("tasks", []),
            "goal": st.session_state.get("goal", ""),
            "members": st.session_state.get("members", ""),
            "weekly_report": st.session_state.get("weekly_report", ""),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _normalize_status(status: str) -> str:
    """把模型返回的状态值统一到应用内的状态枚举。"""
    s = (status or "").strip()
    mapping = {
        "未开始": "未开始",
        "待开始": "未开始",
        "进行中": "进行中",
        "已完成": "已完成",
        "完成": "已完成",
        "已结束": "已完成",
    }
    return mapping.get(s, "未开始")


def _extract_json_array(content: str) -> list:
    """从模型返回中提取纯 JSON 数组，兼容 ```json ... ``` 包裹。"""
    try:
        return json.loads(content)
    except Exception:
        pass
    cleaned = re.sub(r"```(?:json)?", "", content, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"\[\s*\{.*\}\s*\]", content, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("未能从模型返回中解析出 JSON 数组")


def decompose_okr(goal: str, api_key: str, members: str = "") -> list[dict]:
    """调用智谱 GLM-4，把团队核心目标拆解为 KR 与成员任务清单。

    members 为用户输入的团队成员名单（顿号/逗号分隔）；为空则让 AI 自行生成成员。
    """
    if not goal or not goal.strip():
        raise ValueError("请先输入团队核心目标")
    if not api_key or not api_key.strip():
        raise ValueError("请先在左侧栏填写智谱 API Key")

    # 解析成员名单：支持顿号、中英文逗号、分号、换行分隔
    member_list = [
        m.strip()
        for m in re.split(r"[、,，;；\n]+", members or "")
        if m.strip()
    ]

    if member_list:
        names = "、".join(member_list)
        member_rule = (
            f"团队成员名单：{names}\n"
            f"请只从上述名单中选择成员来拆解任务，不要虚构任何其他成员；"
            f"member 字段必须严格使用名单中的姓名。\n"
        )
    else:
        member_rule = "请自行生成 3-5 位中文成员姓名来拆解任务。\n"

    prompt = (
        "你是一个团队目标拆解专家。请根据下面给定的团队核心目标，"
        "拆解出团队成员的关键结果与具体任务。\n"
        f"{member_rule}"
        "严格只返回一个 JSON 数组，不要包含任何解释性文字、不要使用 Markdown 代码块标记。\n"
        "JSON 数组中每个元素的结构如下：\n"
        '[{"member": "成员姓名", "kr": "关键结果", "tasks": ["任务1", "任务2"], "status": "未开始"}]\n'
        "要求：kr 为可量化的关键结果；tasks 为 1-3 条具体任务；"
        "status 取值仅限 '未开始'、'进行中'、'已完成'。\n"
        f"团队核心目标：{goal}\n"
        "请返回 JSON 数组："
    )

    headers = {"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"}
    payload = {
        "model": GLM_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个严格按 JSON 格式输出的助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }

    resp = requests.post(GLM_API_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"GLM 接口返回错误（HTTP {resp.status_code}）：{resp.text[:200]}")
    content = resp.json()["choices"][0]["message"]["content"]
    items = _extract_json_array(content)

    rows = []
    for it in items:
        member = it.get("member") or "未指定"
        kr = it.get("kr") or ""
        status = _normalize_status(it.get("status"))
        tasks = it.get("tasks") or []
        if isinstance(tasks, str):
            tasks = [tasks]
        if not tasks:
            tasks = [""]
        for t in tasks:
            rows.append({"成员": member, "关键结果": kr, "具体任务": t, "状态": status})
    if not rows:
        raise ValueError("模型返回的任务清单为空")
    return rows


def generate_weekly_report(tasks: list[dict], goal: str, api_key: str) -> str:
    """调用智谱 GLM-4，基于当前任务与状态生成专业团队周报（Markdown）。"""
    if not api_key or not api_key.strip():
        raise ValueError("请先在左侧栏填写智谱 API Key")

    total = len(tasks)
    done = sum(1 for t in tasks if t.get("状态") == "已完成")
    doing = sum(1 for t in tasks if t.get("状态") == "进行中")
    todo = sum(1 for t in tasks if t.get("状态") == "未开始")
    progress = (done / total * 100) if total else 0

    lines = [
        f"{i}. 成员={t.get('成员','')} | 关键结果={t.get('关键结果','')} | "
        f"任务={t.get('具体任务','')} | 状态={t.get('状态','')}"
        for i, t in enumerate(tasks, 1)
    ]
    task_summary = "\n".join(lines) if lines else "（暂无任务）"

    prompt = (
        "你是一个团队周报撰写专家。请基于以下团队本阶段目标与任务清单，"
        "生成一份专业的团队周报（Markdown 格式）。\n"
        "周报必须包含三个章节：## 本周进展、## 风险与问题、## 下周计划。\n"
        "本周进展需结合任务完成率与各项状态进行总结；"
        "风险与问题需识别未完成或进行中任务的潜在风险点；"
        "下周计划需给出可执行的动作项。\n"
        f"团队核心目标：{goal or '（尚未设定）'}\n"
        f"任务总数：{total}，已完成：{done}（完成率 {progress:.1f}%），"
        f"进行中：{doing}，未开始：{todo}\n"
        f"任务清单：\n{task_summary}\n"
        "请直接输出 Markdown 周报正文，不要额外解释。"
    )

    headers = {"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"}
    payload = {
        "model": GLM_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个擅长撰写专业团队周报的助手，输出 Markdown。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.6,
    }

    resp = requests.post(GLM_API_URL, headers=headers, json=payload, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"GLM 接口返回错误（HTTP {resp.status_code}）：{resp.text[:200]}")
    return resp.json()["choices"][0]["message"]["content"]


# ====== 界面美化样式 ======
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1180px;}
    .hero {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        padding: 26px 30px; border-radius: 14px; margin-bottom: 20px;
        box-shadow: 0 6px 18px rgba(24,40,72,0.18);
    }
    .hero h1 {color: #ffffff; margin: 0; font-size: 1.7rem; font-weight: 700;}
    .hero p {color: #d8e2fb; margin: 6px 0 0; font-size: 0.95rem;}
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fb 100%);
        border: 1px solid #e8ecf3; border-radius: 12px; padding: 14px 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricValue"] {color: #2c3e50; font-weight: 700;}
    .stProgress > div > div {background: linear-gradient(90deg, #4b6cb7, #43b85d);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🎯 Micro-OKR：微团队目标对齐器</h1>'
    "<p>微团队目标对齐 · AI智能拆解 · 进度可视化</p></div>",
    unsafe_allow_html=True,
)

# ====== session_state 初始化（优先从本地缓存恢复，确保刷新不丢） ======
_saved = _load_state()
if "tasks" not in st.session_state:
    if _saved and _saved.get("tasks"):
        st.session_state.tasks = _saved["tasks"]
    else:
        st.session_state.tasks = [
            {"成员": "张明", "关键结果": "用户注册量提升30%", "具体任务": "优化注册流程", "状态": "进行中"},
            {"成员": "李华", "关键结果": "DAU突破5万", "具体任务": "设计增长活动", "状态": "已完成"},
            {"成员": "王芳", "关键结果": "留存率提升至60%", "具体任务": "优化新手引导", "状态": "未开始"},
            {"成员": "赵强", "关键结果": "NPS分数达到50", "具体任务": "用户调研访谈", "状态": "进行中"},
        ]
if "goal" not in st.session_state:
    st.session_state.goal = _saved.get("goal", "") if _saved else ""
if "members" not in st.session_state:
    st.session_state.members = _saved.get("members", "") if _saved else ""
if "weekly_report" not in st.session_state:
    st.session_state.weekly_report = _saved.get("weekly_report", "") if _saved else ""

# ====== 侧边栏：智谱 API Key 配置 ======
with st.sidebar:
    st.header("⚙️ 配置")
    st.text_input(
        "智谱 API Key",
        value=os.environ.get("ZHIPU_API_KEY", ""),
        type="password",
        key="glm_api_key",
        placeholder="粘贴 open.bigmodel.cn 的 API Key",
        help="Key 仅保存在本会话内存中，不会写入源码。",
    )
    st.caption("获取地址：https://open.bigmodel.cn → 控制台 → API Keys")
    st.divider()
    if st.button("🗑️ 重置数据为示例"):
        st.session_state.tasks = [
            {"成员": "张明", "关键结果": "用户注册量提升30%", "具体任务": "优化注册流程", "状态": "进行中"},
            {"成员": "李华", "关键结果": "DAU突破5万", "具体任务": "设计增长活动", "状态": "已完成"},
            {"成员": "王芳", "关键结果": "留存率提升至60%", "具体任务": "优化新手引导", "状态": "未开始"},
            {"成员": "赵强", "关键结果": "NPS分数达到50", "具体任务": "用户调研访谈", "状态": "进行中"},
        ]
        st.session_state.goal = ""
        st.session_state.members = ""
        st.session_state.weekly_report = ""
        _save_state()
        st.rerun()

# ====== 顶部：核心目标输入 + AI智能拆解 ======
with st.container(border=True):
    st.subheader("📌 本阶段核心目标")
    goal_input = st.text_area(
        "请输入团队本阶段的核心目标",
        value=st.session_state.goal,
        height=110,
        placeholder="例如：在 Q3 末实现产品 DAU 翻倍，并完成核心功能升级",
        label_visibility="visible",
    )
    members_input = st.text_area(
        "团队成员名单",
        value=st.session_state.members,
        height=80,
        placeholder="请输入团队成员名字，用顿号或逗号分隔，例如：张三, 李四, 王五",
        label_visibility="visible",
        help="留空则由 AI 自动生成 3-5 位成员",
    )
    if st.button("🤖 AI智能拆解", type="primary", use_container_width=True):
        api_key = st.session_state.get("glm_api_key", "")
        if not goal_input.strip():
            st.warning("请先输入团队核心目标，再进行 AI 拆解。")
        elif not api_key.strip():
            st.warning("请先在左侧栏填写智谱 API Key。")
        else:
            try:
                with st.spinner("AI 正在分析目标并拆解关键结果..."):
                    rows = decompose_okr(goal_input, api_key, members_input)
                st.session_state.tasks = rows
                st.session_state.goal = goal_input
                st.session_state.members = members_input
                st.success(f"AI 已基于核心目标拆解出 {len(rows)} 条任务。")
                _save_state()
                st.rerun()
            except Exception as e:
                st.error(f"AI 拆解失败：{e}")

# ====== 中部：左列任务清单（状态下拉） + 右列统计指标 ======
col_left, col_right = st.columns([3, 1.05])

with col_left:
    with st.container(border=True):
        st.subheader("📋 团队KR与成员任务清单")
        df = pd.DataFrame(st.session_state.tasks)
        edited = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "成员": st.column_config.TextColumn("成员", disabled=True),
                "关键结果": st.column_config.TextColumn("关键结果", disabled=True),
                "具体任务": st.column_config.TextColumn("具体任务", disabled=True),
                "状态": st.column_config.SelectboxColumn(
                    "状态", options=STATUS_OPTIONS, required=True
                ),
            },
            key="tasks_editor",
        )
        new_tasks = edited.to_dict("records")
        if new_tasks != st.session_state.tasks:
            st.session_state.tasks = new_tasks

with col_right:
    with st.container(border=True):
        st.subheader("📊 统计指标")
        total = len(st.session_state.tasks)
        done = sum(1 for t in st.session_state.tasks if t["状态"] == "已完成")
        doing = sum(1 for t in st.session_state.tasks if t["状态"] == "进行中")
        todo = sum(1 for t in st.session_state.tasks if t["状态"] == "未开始")
        progress = (done / total * 100) if total else 0

        m1, m2 = st.columns(2)
        m1.metric("任务总数", total)
        m2.metric("完成数", done)
        st.metric(
            "完成率",
            f"{progress:.1f}%",
            delta=f"{done}/{total}" if total else "0/0",
            delta_color="off" if progress < 1 else "normal",
        )
        st.progress(done / total if total else 0)
        st.caption(f"🟡 进行中 {doing}　⚪ 未开始 {todo}")

# ====== 底部：生成智能周报（调用 GLM-4） ======
with st.container(border=True):
    st.subheader("📝 智能周报")
    if st.button("📰 生成智能周报", type="primary", use_container_width=True):
        api_key = st.session_state.get("glm_api_key", "")
        if not api_key.strip():
            st.warning("请先在左侧栏填写智谱 API Key。")
        else:
            try:
                with st.spinner("AI 正在汇总本周进展、风险与下周计划..."):
                    report = generate_weekly_report(
                        st.session_state.tasks, st.session_state.goal, api_key
                    )
                st.session_state.weekly_report = report
                _save_state()
                st.success("周报已生成 ✅")
            except Exception as e:
                st.error(f"周报生成失败：{e}")
    if st.session_state.weekly_report:
        st.markdown(st.session_state.weekly_report)

# ====== 每次运行结束，落盘一次，保证刷新/重开可恢复 ======
_save_state()
