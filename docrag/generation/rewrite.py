import requests
from dashscope import MultiModalConversation

from docrag import config
from docrag.embeddings import call_with_retry
from docrag.generation.answer import message_text

# 追问改写模型：只改写问题，不看页图，用便宜的文本模型
REWRITE_MODEL = "qwen3.7-flash"

# 改写调用超时（秒）
REWRITE_TIMEOUT = 15

# 改写指令：结合历史把追问补成能独立理解的问题
REWRITE_PROMPT = (
    "Below is a conversation about a document, followed by a new question.\n"
    "Rewrite the new question so that it can be understood on its own, without the conversation: "
    "replace pronouns and omitted references with what they refer to, and keep everything else as it is, "
    "including the language it is written in.\n"
    "If the new question already stands on its own, repeat it unchanged. Do not answer it. Reply with the question only.\n\n"
    "Conversation:\n{history}\n\nNew question: {question}"
)


# 历史问答 --> 提示词里的对话文字
def history_text(history):
    return "\n".join(f"Q: {turn['question']}\nA: {turn['answer']}" for turn in history)


# 追问 + 历史 --> 能独立理解的问题；调用失败或回空就用原问题
def rewrite_question(question, history):
    prompt = REWRITE_PROMPT.format(history=history_text(history), question=question)
    try:
        resp = call_with_retry(
            lambda: MultiModalConversation.call(
                model=REWRITE_MODEL,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                enable_thinking=False,
                temperature=0.0,
                request_timeout=REWRITE_TIMEOUT,
                api_key=config.API_KEY,
            )
        )
    except (RuntimeError, requests.exceptions.ReadTimeout):
        return question
    return message_text(resp.output.choices[0].message).strip() or question
