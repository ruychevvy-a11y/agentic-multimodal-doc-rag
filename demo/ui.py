import json

import requests
import streamlit as st

# 后端地址
API = "http://127.0.0.1:8000"

# 整页页图的显示宽度（像素）：缩略图，鼠标移上去点全屏才看细节
PAGE_IMAGE_WIDTH = 240

# 兜底作答的原因
FALLBACK_REASONS = {
    "max_model_calls": "模型调用次数用完，用看过的页强制作答",
    "timeout": "模型调用超时，用看过的页强制作答",
}

# 路由采信的路线
ROUTE_NAMES = {"agent": "agent", "baseline": "一次作答"}

st.set_page_config(page_title="混合多模态文档问答", layout="wide")


# 文档列表 {doc_id: 文档信息}
@st.cache_data
def load_documents():
    documents = requests.get(f"{API}/documents").json()
    return {document["doc_id"]: document for document in documents}


# 某份文档的题库 {query_id: 题}
@st.cache_data
def load_questions(doc_id):
    questions = requests.get(f"{API}/questions", params={"doc_id": doc_id}).json()
    return {question["query_id"]: question for question in questions}


# 文档、题库下拉框、问题框都切到这道题；换了题目就是新话题，对话清空
def use_question(question):
    st.session_state["doc_id"] = question["doc_id"]
    st.session_state["bank_choice"] = question["query_id"]
    st.session_state["question_text"] = question["question"]
    st.session_state["picked"] = question
    st.session_state["turns"] = []


# 随机抽题
def pick_random():
    use_question(requests.get(f"{API}/questions/random").json())


# 从题库下拉框选题
def pick_from_bank():
    query_id = st.session_state["bank_choice"]
    if query_id:
        use_question(load_questions(st.session_state["doc_id"])[query_id])


# 换文档时清空题库选择、问题和对话，免得拿别的文档的题来问
def reset_question():
    st.session_state["bank_choice"] = ""
    st.session_state["question_text"] = ""
    st.session_state["picked"] = None
    st.session_state["turns"] = []


# 问过的轮次全清掉
def clear_turns():
    st.session_state["turns"] = []


# 这一轮先记进对话、问题框清空；接口等渲染到这一轮时才调
def ask_question():
    question = st.session_state["question_text"].strip()
    # 按钮不设 disabled，这样打完字点一次就能问（输入框的内容点按钮时才提交）
    if not question:
        return
    picked = st.session_state.get("picked")
    st.session_state["turns"].append(
        {
            "question": question,
            "events": None,
            # 原样问题库题才附上标注，改过一个字就不算了
            "gold": picked if picked and picked["question"] == question else None,
        }
    )
    st.session_state["question_text"] = ""


# 页号列表转文字 [19, 20] --> '19、20'
def page_list(numbers):
    return "、".join(map(str, numbers))


# 整页页图横向排开，放不下自动换行
def show_pages(numbers, image_paths):
    with st.container(horizontal=True):
        for number, image_path in zip(numbers, image_paths):
            with st.container(width=PAGE_IMAGE_WIDTH):
                st.image(image_path, caption=f"第 {number} 页")


# 答成了的轮次 --> [{question, answer}]，发给后端把追问补成完整问题；答案取路由采信的那条
def history_before(turns):
    history = []
    for turn in turns:
        routes = [event for event in turn["events"] or [] if event["type"] == "route"]
        if routes:
            history.append(
                {"question": turn["question"], "answer": routes[-1]["answer"]}
            )
    return history


# 调 /ask 流式接口 --> 逐条交回事件，同时存进 collected
def ask_events(question, doc_id, history, collected):
    try:
        response = requests.post(
            f"{API}/ask",
            json={"question": question, "doc_id": doc_id, "history": history},
            stream=True,
        )
        response.raise_for_status()
        events = (
            json.loads(line[6:])
            for line in response.iter_lines()
            if line.startswith(b"data: ")
        )
    except requests.exceptions.ConnectionError:
        events = [{"type": "error", "message": f"连不上后端 {API}，先启动 uvicorn"}]
    except requests.exceptions.HTTPError as error:
        events = [{"type": "error", "message": str(error)}]
    for event in events:
        collected.append(event)
        yield event


# 路由依据 --> '分类器，agent 概率 0.73' / '一次作答 10 秒内没答完'
def route_basis(event):
    if event["agent_probability"] is None:
        return event["reason"]
    return f"{event['reason']}，agent 概率 {event['agent_probability']:.2f}"


# agent 每一步写进状态框，进行中显示「思考中」，结束后收起 --> 最后一条事件
def show_steps(events):
    last_event = None
    with st.status("检索中…", expanded=True) as status:
        for event in events:
            last_event = event
            if event["type"] == "retrieval":
                st.text(f"检索起点页：第 {page_list(event['pages'])} 页")
                # 运行中保持展开，每一步都看得到
                status.update(label="思考中…", expanded=True)
            elif event["type"] == "model_call":
                st.text(
                    f"第 {event['call_index'] + 1} 次调用模型，用时 {event['seconds']} s"
                )
            elif event["type"] == "tool" and event["error"]:
                # 参数不对、页号越界等，显示工具回给模型的提示
                st.warning(f"{event['name']}：{event['text']}")
            elif event["type"] == "tool" and event["name"] == "search_pages":
                st.text(f"搜索：{json.loads(event['arguments'])['query']}")
                st.code(event["text"], language=None, wrap_lines=True)
            elif event["type"] == "tool" and event["name"] == "view_pages":
                if event["shown_pages"]:
                    st.text(f"查看第 {page_list(event['shown_pages'])} 页")
                else:
                    st.text(event["text"])
            elif event["type"] == "tool":
                st.text(f"提交答案（{event['text']}）")
            elif event["type"] == "final":
                st.text("agent 答完，等一次作答后择一")
                status.update(label="路由中…", expanded=True)
            elif event["type"] == "route":
                st.text(f"路由：采信{ROUTE_NAMES[event['route']]}（{route_basis(event)}）")
            elif event["type"] == "error":
                st.error(event["message"])
        if last_event is None or last_event["type"] not in ("route", "error"):
            status.update(label="回答中途断了，再问一次", state="error", expanded=True)
        elif last_event["type"] == "error":
            status.update(label="出错了", state="error", expanded=True)
        else:
            status.update(label="回答完成", state="complete", expanded=False)
    return last_event


st.markdown(
    "<h1 style='text-align:center'>混合多模态文档问答</h1>", unsafe_allow_html=True
)

# 对话里问过的轮次 [{question, events, gold}]
st.session_state.setdefault("turns", [])

documents = load_documents()
doc_column, bank_column, random_column = st.columns(
    [3, 3, 1], vertical_alignment="bottom"
)
with doc_column:
    doc_id = st.selectbox(
        "文档",
        options=list(documents),
        key="doc_id",
        on_change=reset_question,
        persist_state="page",
        format_func=lambda doc_id: f"{doc_id}（{documents[doc_id]['pages']} 页，{documents[doc_id]['questions']} 题）",
    )
questions = load_questions(doc_id)
with bank_column:
    st.selectbox(
        "题库",
        options=[""] + list(questions),
        key="bank_choice",
        on_change=pick_from_bank,
        persist_state="page",
        format_func=lambda query_id: (
            f"{query_id}：{questions[query_id]['question'][:60]}"
            if query_id
            else "（选择问题或下方输入问题）"
        ),
    )
with random_column:
    st.button("随机抽题", on_click=pick_random, use_container_width=True)

# 一轮一问一答
for index, turn in enumerate(st.session_state["turns"]):
    with st.chat_message("user"):
        st.text(turn["question"])
    with st.chat_message("assistant"):
        if turn["events"] is None:
            turn["events"] = []
            events = ask_events(
                turn["question"],
                doc_id,
                history_before(st.session_state["turns"][:index]),
                turn["events"],
            )
        else:
            events = turn["events"]
        last_event = show_steps(events)
        if last_event and last_event["type"] == "route":
            with st.container(border=True, gap="xsmall"):
                st.markdown("**系统回答**")
                # 金额里的 $ 转义，防止两个 $ 之间被当成公式
                st.markdown(last_event["answer"].replace("$", r"\$"))
                st.text(f"引用页：{page_list(last_event['cited_pages']) or '无'}")
                st.caption(
                    f"采信{ROUTE_NAMES[last_event['route']]}（{route_basis(last_event)}）"
                )
                if last_event["fallback"]:
                    st.caption(FALLBACK_REASONS[last_event["fallback"]])
                with st.expander("两条路线各自的回答"):
                    st.markdown("**agent**")
                    st.text(last_event["agent_answer"])
                    st.markdown("**一次作答**")
                    st.text(last_event["baseline_answer"] or "没等到")
            show_pages(last_event["cited_pages"], last_event["cited_images"])
        # 原样问的题库题附上标注对照
        if turn["gold"]:
            # st.text 不解析 Markdown，答案里的 $ 不会变成公式
            with st.container(border=True, gap="xsmall"):
                st.markdown("**题库标注**")
                st.text(f"答案：{turn['gold']['answers'][0]}")
                st.text(
                    f"金标页：{', '.join(map(str, turn['gold']['gold_pages'])) or '无'}"
                )
                st.text(
                    f"证据类型：{', '.join(turn['gold']['evidence_sources']) or '无'}"
                )

st.text_area("问题", key="question_text", height=100, persist_state="page")

with st.container(horizontal=True):
    st.button("提问", type="primary", on_click=ask_question)
    st.button("新对话", disabled=not st.session_state["turns"], on_click=clear_turns)
