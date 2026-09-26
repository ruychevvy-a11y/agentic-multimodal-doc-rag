import json

import requests
import streamlit as st

# 后端地址
API = "http://127.0.0.1:8000"

# 整页页图的显示宽度（像素）：缩略图，鼠标移上去点全屏才看细节
PAGE_IMAGE_WIDTH = 240

# 界面文字 {语言: {文字名: 文字}}；后端只发代码（路由依据、兜底原因），在这里翻译
WORDS = {
    "zh": {
        "language": "中文",
        "title": "混合多模态文档问答",
        "fallback": {
            "max_model_calls": "模型调用次数用完，用看过的页强制作答",
            "timeout": "模型调用超时，用看过的页强制作答",
        },
        # agent 前加空格：中文里夹英文词隔开，「采信 agent」
        "routes": {"agent": " agent", "baseline": "一次作答"},
        "route_titles": {"agent": "agent", "baseline": "一次作答"},
        "reasons": {
            "classifier": "分类器",
            "rule": "手工规则：agent 检索过才采信",
            "baseline_timeout": "一次作答 {seconds} 秒内没答完",
            "baseline_error": "一次作答出错",
        },
        "with_probability": "{reason}，agent 概率 {probability:.2f}",
        "page": "第 {number} 页",
        "page_separator": "、",
        "no_backend": "连不上后端 {api}，先启动 uvicorn",
        "searching": "检索中…",
        "thinking": "思考中…",
        "routing": "路由中…",
        "done": "回答完成",
        "failed": "出错了",
        "broken": "回答中途断了，再问一次",
        "start_pages": "检索起点页：第 {pages} 页",
        "model_call": "第 {index} 次调用模型，用时 {seconds} s",
        "search": "搜索：{query}",
        "view": "查看第 {pages} 页",
        "submit": "提交答案（{text}）",
        "agent_done": "agent 答完，等一次作答后择一",
        "route_step": "路由：采信{route}（{basis}）",
        "adopted": "采信{route}（{basis}）",
        "document": "文档",
        "document_option": "{doc_id}（{pages} 页，{questions} 题）",
        "bank": "题库",
        "bank_placeholder": "（选择问题或下方输入问题）",
        "random": "随机抽题",
        "system_answer": "**系统回答**",
        "cited_pages": "引用页：{pages}",
        "none": "无",
        "both_answers": "两条路线各自的回答",
        "not_waited": "没等到",
        "gold": "**题库标注**",
        "gold_answer": "答案：{answer}",
        "gold_pages": "金标页：{pages}",
        "gold_sources": "证据类型：{sources}",
        "question": "问题",
        "ask": "提问",
        "new_chat": "新对话",
    },
    "en": {
        "language": "English",
        "title": "Hybrid Multimodal Document QA",
        "fallback": {
            "max_model_calls": "Model call limit reached; answered from the pages already viewed",
            "timeout": "Model call timed out; answered from the pages already viewed",
        },
        "routes": {"agent": "the agent", "baseline": "one-shot answering"},
        "route_titles": {"agent": "Agent", "baseline": "One-shot answering"},
        "reasons": {
            "classifier": "classifier",
            "rule": "rule: trust the agent only if it searched",
            "baseline_timeout": "one-shot answering did not finish within {seconds} s",
            "baseline_error": "one-shot answering failed",
        },
        "with_probability": "{reason}, agent probability {probability:.2f}",
        "page": "Page {number}",
        "page_separator": ", ",
        "no_backend": "Cannot reach the backend at {api}; start uvicorn first",
        "searching": "Retrieving…",
        "thinking": "Thinking…",
        "routing": "Routing…",
        "done": "Done",
        "failed": "Something went wrong",
        "broken": "The answer was cut off; please ask again",
        "start_pages": "Starting pages: {pages}",
        "model_call": "Model call {index}, {seconds} s",
        "search": "Search: {query}",
        "view": "Viewed pages: {pages}",
        "submit": "Submitted answer ({text})",
        "agent_done": "Agent finished; waiting for one-shot answering before choosing",
        "route_step": "Router: adopted {route} ({basis})",
        "adopted": "Adopted {route} ({basis})",
        "document": "Document",
        "document_option": "{doc_id} (pages: {pages}, questions: {questions})",
        "bank": "Question bank",
        "bank_placeholder": "(pick a question, or type one below)",
        "random": "Random question",
        "system_answer": "**Answer**",
        "cited_pages": "Cited pages: {pages}",
        "none": "none",
        "both_answers": "Answers from both routes",
        "not_waited": "not received in time",
        "gold": "**Dataset annotation**",
        "gold_answer": "Answer: {answer}",
        "gold_pages": "Gold pages: {pages}",
        "gold_sources": "Evidence types: {sources}",
        "question": "Question",
        "ask": "Ask",
        "new_chat": "New chat",
    },
}

# 界面语言：默认中文，地址带 ?lang=en 时直接用英文
st.session_state.setdefault(
    "language", st.query_params.get("lang") if st.query_params.get("lang") in WORDS else "zh"
)
words = WORDS[st.session_state["language"]]

st.set_page_config(page_title=words["title"], layout="wide")


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


# 换语言后下拉框按新语言重画：前端记着的是旧语言的选项文字，把选中的值原样写回 session_state
def refresh_selections():
    for key in ("doc_id", "bank_choice"):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


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


# 页号列表转文字 [19, 20] --> '19、20' / '19, 20'
def page_list(numbers):
    return words["page_separator"].join(map(str, numbers))


# 整页页图横向排开，放不下自动换行
def show_pages(numbers, image_paths):
    with st.container(horizontal=True):
        for number, image_path in zip(numbers, image_paths):
            with st.container(width=PAGE_IMAGE_WIDTH):
                st.image(image_path, caption=words["page"].format(number=number))


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
        events = [{"type": "error", "message": words["no_backend"].format(api=API)}]
    except requests.exceptions.HTTPError as error:
        events = [{"type": "error", "message": str(error)}]
    for event in events:
        collected.append(event)
        yield event


# 路由依据 --> '分类器，agent 概率 0.73' / 'one-shot answering did not finish within 10 s'
def route_basis(event):
    reason = words["reasons"][event["reason"]].format(seconds=event["wait_seconds"])
    if event["agent_probability"] is None:
        return reason
    return words["with_probability"].format(
        reason=reason, probability=event["agent_probability"]
    )


# agent 每一步写进状态框，进行中显示「思考中」，结束后收起 --> 最后一条事件
def show_steps(events):
    last_event = None
    with st.status(words["searching"], expanded=True) as status:
        for event in events:
            last_event = event
            if event["type"] == "retrieval":
                st.text(words["start_pages"].format(pages=page_list(event["pages"])))
                # 运行中保持展开，每一步都看得到
                status.update(label=words["thinking"], expanded=True)
            elif event["type"] == "model_call":
                st.text(
                    words["model_call"].format(
                        index=event["call_index"] + 1, seconds=event["seconds"]
                    )
                )
            elif event["type"] == "tool" and event["error"]:
                # 参数不对、页号越界等，显示工具回给模型的提示
                st.warning(f"{event['name']}: {event['text']}")
            elif event["type"] == "tool" and event["name"] == "search_pages":
                st.text(words["search"].format(query=json.loads(event["arguments"])["query"]))
                st.code(event["text"], language=None, wrap_lines=True)
            elif event["type"] == "tool" and event["name"] == "view_pages":
                if event["shown_pages"]:
                    st.text(words["view"].format(pages=page_list(event["shown_pages"])))
                else:
                    st.text(event["text"])
            elif event["type"] == "tool":
                st.text(words["submit"].format(text=event["text"]))
            elif event["type"] == "final":
                st.text(words["agent_done"])
                status.update(label=words["routing"], expanded=True)
            elif event["type"] == "route":
                st.text(
                    words["route_step"].format(
                        route=words["routes"][event["route"]], basis=route_basis(event)
                    )
                )
            elif event["type"] == "error":
                st.error(event["message"])
        if last_event is None or last_event["type"] not in ("route", "error"):
            status.update(label=words["broken"], state="error", expanded=True)
        elif last_event["type"] == "error":
            status.update(label=words["failed"], state="error", expanded=True)
        else:
            status.update(label=words["done"], state="complete", expanded=False)
    return last_event


# 右上角切换界面语言
with st.container(horizontal=True, horizontal_alignment="right"):
    st.segmented_control(
        "language",
        options=list(WORDS),
        format_func=lambda language: WORDS[language]["language"],
        key="language",
        on_change=refresh_selections,
        required=True,
        label_visibility="collapsed",
    )

st.markdown(
    f"<h1 style='text-align:center'>{words['title']}</h1>", unsafe_allow_html=True
)

# 对话里问过的轮次 [{question, events, gold}]
st.session_state.setdefault("turns", [])

documents = load_documents()
doc_column, bank_column, random_column = st.columns(
    [3, 3, 1], vertical_alignment="bottom"
)
with doc_column:
    doc_id = st.selectbox(
        words["document"],
        options=list(documents),
        key="doc_id",
        on_change=reset_question,
        persist_state="page",
        format_func=lambda doc_id: words["document_option"].format(
            doc_id=doc_id,
            pages=documents[doc_id]["pages"],
            questions=documents[doc_id]["questions"],
        ),
    )
questions = load_questions(doc_id)
with bank_column:
    st.selectbox(
        words["bank"],
        options=[""] + list(questions),
        key="bank_choice",
        on_change=pick_from_bank,
        persist_state="page",
        format_func=lambda query_id: (
            f"{query_id}: {questions[query_id]['question'][:60]}"
            if query_id
            else words["bank_placeholder"]
        ),
    )
with random_column:
    st.button(words["random"], on_click=pick_random, use_container_width=True)

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
                st.markdown(words["system_answer"])
                # 金额里的 $ 转义，防止两个 $ 之间被当成公式
                st.markdown(last_event["answer"].replace("$", r"\$"))
                st.text(
                    words["cited_pages"].format(
                        pages=page_list(last_event["cited_pages"]) or words["none"]
                    )
                )
                st.caption(
                    words["adopted"].format(
                        route=words["routes"][last_event["route"]],
                        basis=route_basis(last_event),
                    )
                )
                if last_event["fallback"]:
                    st.caption(words["fallback"][last_event["fallback"]])
                with st.expander(words["both_answers"]):
                    st.markdown(f"**{words['route_titles']['agent']}**")
                    st.text(last_event["agent_answer"])
                    st.markdown(f"**{words['route_titles']['baseline']}**")
                    st.text(last_event["baseline_answer"] or words["not_waited"])
            show_pages(last_event["cited_pages"], last_event["cited_images"])
        # 原样问的题库题附上标注对照
        if turn["gold"]:
            # st.text 不解析 Markdown，答案里的 $ 不会变成公式
            with st.container(border=True, gap="xsmall"):
                st.markdown(words["gold"])
                st.text(words["gold_answer"].format(answer=turn["gold"]["answers"][0]))
                st.text(
                    words["gold_pages"].format(
                        pages=", ".join(map(str, turn["gold"]["gold_pages"])) or words["none"]
                    )
                )
                st.text(
                    words["gold_sources"].format(
                        sources=", ".join(turn["gold"]["evidence_sources"]) or words["none"]
                    )
                )

st.text_area(words["question"], key="question_text", height=100, persist_state="page")

with st.container(horizontal=True):
    st.button(words["ask"], type="primary", on_click=ask_question)
    st.button(words["new_chat"], disabled=not st.session_state["turns"], on_click=clear_turns)
