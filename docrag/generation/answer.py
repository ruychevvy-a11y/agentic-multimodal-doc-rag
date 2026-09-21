import os

from dashscope import MultiModalConversation

from docrag import config
from docrag.embeddings import call_with_retry

# 作答指令：只根据给的页回答；信息不够答 Not answerable；最后一行按 [Page N] 标签列出引用页
ANSWER_PROMPT = (
    "The images are pages retrieved from a document, each preceded by its label [Page N].\n"
    'Answer the question using only these pages. If they do not contain the information needed, answer "Not answerable".\n'
    'Give the answer first, then a brief explanation, and end with a line "Cited pages: N, M" using the [Page N] labels.\n\n'
    "Question: {question}"
)


# 页图内容
# pages 是页面记录，至少要有 page_number 和 image_path
def page_contents(pages):
    content = []
    for page in pages:
        content.append({"text": f"[Page {page['page_number']}]"})
        # 本地页图要写成 file:// 绝对路径
        image_path = os.path.abspath(page["image_path"]).replace("\\", "/")
        content.append({"image": f"file://{image_path}"})
    return content


# 从模型返回的消息里取出回答正文
def message_text(message):
    return "".join(part["text"] for part in message.content if "text" in part)


# 每页先放 [Page 页号] 标签再放页图，最后放指令和问题 --> MultiModalConversation 的 messages
def answer_messages(question, pages):
    content = page_contents(pages)
    content.append({"text": ANSWER_PROMPT.format(question=question)})
    return [{"role": "user", "content": content}]


# 作答：看页图回答问题 --> {response, input_tokens, output_tokens}；内容审核等不可重试的错误直接抛出，由调用方处理
def generate_answer(question, pages):
    resp = call_with_retry(
        lambda: MultiModalConversation.call(
            model=config.GENERATION_MODEL,
            messages=answer_messages(question, pages),
            api_key=config.API_KEY,
        )
    )
    return {
        "response": message_text(resp.output.choices[0].message),
        "input_tokens": resp.usage["input_tokens"],
        "output_tokens": resp.usage["output_tokens"],
    }
