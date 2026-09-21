import json
import re

from docrag import config

# 摘要窗口的词数
EXCERPT_WORDS = 30

# fmt: off
# 选摘要窗口时不算命中的常见词
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "by", "with", "as", "and", "or", "is", "are", "was", "were", "be", "it", "its", "this", "that", "there", "what", "which", "who", "how", "many", "much", "does", "do", "did", "according",
}
# fmt: on

# 三个工具的 function calling 定义
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_pages",
            "description": (
                f"Search the current document. Returns the {config.AGENT_SEARCH_RESULTS} most relevant pages, "
                "each with its page number and a short text snippet. Pages you have already seen are marked (viewed)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, in English keywords or a short sentence.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_pages",
            "description": (
                f"Look at page images by page number (any page in the document, not only search results). "
                f"At most {config.AGENT_VIEW_PAGES_PER_CALL} pages per call and {config.AGENT_MAX_VIEWED_PAGES} pages in total."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "1-based page numbers.",
                    }
                },
                "required": ["page_numbers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": 'Give the final answer and the viewed pages it is based on. Use "Not answerable" if the document does not contain the information.',
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "cited_pages": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Page numbers of viewed pages that support the answer.",
                    },
                },
                "required": ["answer", "cited_pages"],
            },
        },
    },
]


# 把工具的执行结果拼成统一结构，交回给循环
def tool_result(text, error=False, pages=(), answer=None):
    return {"text": text, "error": error, "pages": list(pages), "answer": answer}


# 判断模型给的页号参数是不是一串整数
def is_page_number_list(numbers):
    return isinstance(numbers, list) and all(
        isinstance(number, int) and not isinstance(number, bool) for number in numbers
    )


# 页号列表转文字
def page_numbers_text(numbers):
    return ", ".join(map(str, numbers))


# 在正文里滑窗找命中查询词最多的一段，截出来给模型看
def excerpt(text, query):
    words = text.split()
    if not words:
        return "(no text: image-only page)"
    # 查询关键词
    terms = set(re.findall(r"\w+", query.lower())) - STOPWORDS
    # 标出命中查询词的词 ['Figure', '3:', 'revenue'] --> [False, False, True]
    hits = [bool(terms & set(re.findall(r"\w+", word.lower()))) for word in words]
    # 滑动窗口找命中最多的一段
    window_hits = sum(hits[:EXCERPT_WORDS])
    best_start, best_hits = 0, window_hits
    for start in range(1, len(words) - EXCERPT_WORDS + 1):
        window_hits += hits[start + EXCERPT_WORDS - 1] - hits[start - 1]
        if window_hits > best_hits:
            best_start, best_hits = start, window_hits
    # 命中词居中
    hit_positions = [
        position
        for position, hit in enumerate(
            hits[best_start : best_start + EXCERPT_WORDS], best_start
        )
        if hit
    ]
    if hit_positions:
        slack = EXCERPT_WORDS - (hit_positions[-1] - hit_positions[0] + 1)
        best_start = min(
            max(hit_positions[0] - slack // 2, 0), max(len(words) - EXCERPT_WORDS, 0)
        )
    end = best_start + EXCERPT_WORDS
    prefix = "... " if best_start > 0 else ""
    suffix = " ..." if end < len(words) else ""
    return prefix + " ".join(words[best_start:end]) + suffix


# 单题工具箱
class DocumentTools:
    # 初始化工具箱
    def __init__(self, retrieval, doc_id, doc_pages, initial_pages):
        self.retrieval = retrieval
        self.search_filter = f'doc_id == "{doc_id}"'
        # 页号查页面
        self.pages_by_number = {page["page_number"]: page for page in doc_pages}
        # page_id 查页面
        self.pages_by_id = {page["page_id"]: page for page in doc_pages}
        # 看过的页
        self.viewed = {page["page_number"]: page for page in initial_pages}
        # 搜索缓存
        self.search_cache = {}

    # 调用工具
    def call_tool(self, name, arguments):
        # 工具名对应的方法
        handlers = {
            "search_pages": self.search_pages,
            "view_pages": self.view_pages,
            "answer": self.answer,
        }
        if name not in handlers:
            return tool_result(
                f"Unknown tool {name}. Available tools: {','.join(handlers)}.",
                error=True,
            )

        # 解析json，失败带原文
        try:
            parameters = json.loads(arguments)
        except json.JSONDecodeError:
            return tool_result(f"Arguments are not valid JSON: {arguments}", error=True)
        # 根据tool name拿出schema当中tool的需求
        required = next(
            tool["function"]["parameters"]["required"]
            for tool in TOOL_SCHEMAS
            if tool["function"]["name"] == name
        )
        # 检查参数是否填错，填错则返回所需求参数
        if not isinstance(parameters, dict) or set(parameters) != set(required):
            return tool_result(
                f"{name} takes these arguments: {', '.join(required)}.",
                error=True,
            )
        return handlers[name](**parameters)

    # 搜索页面
    def search_pages(self, query):
        if not isinstance(query, str) or not query.strip():
            return tool_result("query must be a non-empty string", error=True)
        # 规范化,未重复的写入缓存
        normalized_query = " ".join(query.lower().split())
        repeated = normalized_query in self.search_cache
        if not repeated:
            retrieved = self.retrieval.retrieve(query, self.search_filter)
            self.search_cache[normalized_query] = retrieved
        page_ids = self.search_cache[normalized_query][: config.AGENT_SEARCH_RESULTS]
        texts = self.retrieval.page_texts(page_ids)
        # 每页拼一行摘要
        lines = []
        for page_id in page_ids:
            page_number = self.pages_by_id[page_id]["page_number"]
            viewed = " (viewed)" if page_number in self.viewed else ""
            summary = excerpt(texts[page_id], query)
            lines.append(f"- Page {page_number}{viewed}: {summary}")
        header = (
            f'You already searched "{query}". Same results:'
            if repeated
            else f'Top pages for "{query}":'
        )
        return tool_result("\n".join([header, *lines]))

    # 看页图
    def view_pages(self, page_numbers):
        # []和[1.5]会被拦下
        if not page_numbers or not is_page_number_list(page_numbers):
            return tool_result(
                "page_numbers must be a non-empty list of integers.", error=True
            )
        # 保序去重 [5, 3, 5, 99] --> [5, 3, 99]
        requested = list(dict.fromkeys(page_numbers))
        # 分出越界和有效的页
        invalid = [number for number in requested if number not in self.pages_by_number]
        valid = [number for number in requested if number in self.pages_by_number]
        # 分出看过和没看过的页
        already_viewed = [number for number in valid if number in self.viewed]
        unseen = [number for number in valid if number not in self.viewed]
        # 这题还能看几页
        remaining = config.AGENT_MAX_VIEWED_PAGES - len(self.viewed)
        limit = min(config.AGENT_VIEW_PAGES_PER_CALL, remaining)
        shown, skipped = unseen[:limit], unseen[limit:]
        for number in shown:
            self.viewed[number] = self.pages_by_number[number]

        # 每种情况给模型一句说明
        messages = []
        if shown:
            messages.append(f"Pages {page_numbers_text(shown)} are shown below.")
        if already_viewed:
            messages.append(
                f"Pages {page_numbers_text(already_viewed)} are already shown above."
            )
        if invalid:
            messages.append(
                f"Pages {page_numbers_text(invalid)} do not exist: "
                f"the document has pages 1-{len(self.pages_by_number)}."
            )
        if skipped:
            messages.append(
                f"Pages {page_numbers_text(skipped)} are not shown: "
                f"at most {config.AGENT_VIEW_PAGES_PER_CALL} pages per call and "
                f"{config.AGENT_MAX_VIEWED_PAGES} in total, "
                f"{config.AGENT_MAX_VIEWED_PAGES - len(self.viewed)} left."
            )
        return tool_result(
            " ".join(messages),
            error=bool(invalid),
            pages=[self.pages_by_number[number] for number in shown],
        )

    # 作答
    def answer(self, answer, cited_pages):
        if not isinstance(answer, str) or not is_page_number_list(cited_pages):
            return tool_result(
                "answer must be a string and cited_pages a list of integers.",
                error=True,
            )
        # 去重
        cited_pages = list(dict.fromkeys(cited_pages))
        final_answer = {
            "text": answer,
            "cited_pages": [number for number in cited_pages if number in self.viewed],
        }
        return tool_result("Answer recorded.", answer=final_answer)
