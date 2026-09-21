import os

from dotenv import load_dotenv

load_dotenv()

# 只放两类常量：换数据集 / 做实验会改的开关，和多个模块必须一致的约定；只有一个模块用的常量写在那个模块顶部

# 通义百炼 API Key，全项目唯一读 .env 的地方
API_KEY = os.getenv("DASHSCOPE_API_KEY")


# =============================================#
#                   dataset                    #
# =============================================#

# 数据集登记表：每个数据集各自的选择都写在这里
#   source          = 页面形态：hf_images 每页一张图 / pdf 原生 PDF
#   hf_repo         = HuggingFace 仓库名
#   queries         = 取前多少道题，None 表示全部
#   text_model      = 文本 dense 路的嵌入模型
#   text_field      = 文本 dense 向量在 Milvus 里的字段名
#   convex_weights  = {凸组合路线名: 各路权重}，顺序同工厂函数里的检索器，交叉验证选定
#   rerank_instruct = 重排的英文任务说明
#   rerank_depth    = 第一阶段取前多少页交给重排
#   routes          = 这个数据集上评测的路线
DATASETS = {
    "mpdocvqa": {
        "source": "hf_images",
        "hf_repo": "lmms-lab/MP-DocVQA",
        "queries": 2000,
        "text_model": "text-embedding-v4",
        "text_field": "txt_qwv4",
        "convex_weights": {"convex_bm25_text": (0.6, 0.4)},
        "rerank_instruct": "Given a question about a scanned document, judge whether the page contains the answer",
        "rerank_depth": 20,
        "routes": (
            "image_dense",
            "text_dense",
            "text_bm25",
            "rrf_bm25_text",
            "convex_bm25_text",
            "rerank_convex_bm25_text",
        ),
    },
    "mmlongbench": {
        "source": "pdf",
        "hf_repo": "yubo2333/MMLongBench-Doc",
        "queries": None,
        "text_model": "qwen3.7-text-embedding",
        "text_field": "txt_qw37",
        "convex_weights": {"convex_bm25_text_image": (0.25, 0.55, 0.20)},
        "rerank_instruct": "Given a question about a document, judge whether the page contains the answer",
        "rerank_depth": 30,
        "routes": (
            "image_dense",
            "text_dense",
            "text_bm25",
            "convex_bm25_text_image",
            "rerank_convex_bm25_text_image",
        ),
    },
}

# 当前数据集：换数据集只改这一行
DATASET = "mmlongbench"  # mpdocvqa / mmlongbench

DATASET_SOURCE = DATASETS[DATASET]["source"]
HF_REPO = DATASETS[DATASET]["hf_repo"]
TARGET_QUERIES = DATASETS[DATASET]["queries"]
EVALUATION_ROUTES = DATASETS[DATASET]["routes"]

# 页图长边上限：OCR、bbox 坐标系、图像嵌入都用这个尺寸的页图
MAX_SIDE = 2000


# =============================================#
#                  data paths                  #
# =============================================#

DATA_DIR = f"data/{DATASET}"

# 页面清单 {page_id, doc_id, image_path, (page_number)}
PAGES_PATH = f"{DATA_DIR}/pages.jsonl"

# 检索语料：pages 加解析结果 {source, text, blocks, coord_size}
CORPUS_PATH = f"{DATA_DIR}/corpus.jsonl"

# 视觉描述 {page_id, description}：页上的图、表、照片写成文字，建索引时拼进正文
DESCRIPTIONS_PATH = f"{DATA_DIR}/descriptions.jsonl"

# 问题 {query_id, question, doc_id, (evidence_sources)}
QUERIES_PATH = f"{DATA_DIR}/queries.jsonl"

# 金标页 {query_id, page_id}，多金标的题有多行
QRELS_PATH = f"{DATA_DIR}/qrels.jsonl"

# 答案评测集：全部题（含不可回答题）的问题、金标答案和分组信息
ANSWERS_PATH = f"{DATA_DIR}/answers.jsonl"

IMAGE_DIR = f"{DATA_DIR}/images"

# 原始 PDF 和标注（source = "pdf" 的数据集用）
PDF_DIR = f"{DATA_DIR}/hf/documents"
ANNOTATIONS_PATH = f"{DATA_DIR}/annotations.json"

# 评测结果：evaluations/<数据集>/<closed|open>/results_<路线>.json + run_<路线>.jsonl
EVALUATION_DIR = f"evaluations/{DATASET}"


# =============================================#
#                 vector store                 #
# =============================================#

# 本地 Docker Milvus，没开鉴权
MILVUS_URI = "http://localhost:19530"

# 一个数据集一张表
PAGE_COLLECTION = f"{DATASET}_pages"

# 同一行里的三个向量字段
IMAGE_VECTOR_FIELD = "img_tevp"  # 图像 dense，tongyi-embedding-vision-plus
TEXT_VECTOR_FIELD = DATASETS[DATASET]["text_field"]  # 文本 dense
BM25_VECTOR_FIELD = "txt_bm25"  # BM25 稀疏向量，Milvus 从 text 字段自动算

# HNSW 搜索候选队列长度：越大越接近精确检索，须 ≥ 搜索的 limit
HNSW_SEARCH_EF = 256

# BM25 超参：k1 控制词频饱和，b 控制长文档惩罚
BM25_PARAMS = {"bm25_k1": 1.2, "bm25_b": 0.75}


# =============================================#
#               embedding models               #
# =============================================#

# 视觉描述模型：看页图把图表内容写成文字，关思考
DESCRIBE_MODEL = "qwen3.7-flash"

MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-plus"
TEXT_EMBEDDING_MODEL = DATASETS[DATASET]["text_model"]

# 向量维度，建表和嵌入必须一致
IMAGE_EMBEDDING_DIM = 1152
TEXT_EMBEDDING_DIM = 1024


# =============================================#
#                  retrieval                   #
# =============================================#

# 每题最终返回多少页
RETRIEVE_K = 10

# RRF 平滑常数：融合分 = Σ 1/(RRF_K + 名次)
RRF_K = 60

# 融合时每一路取前多少个候选
FUSION_DEPTH = 50

CONVEX_WEIGHTS = DATASETS[DATASET]["convex_weights"]


# =============================================#
#                    rerank                    #
# =============================================#

# 重排模型：读问题和一页正文，打 0~1 的相关分
RERANK_MODEL = "qwen3.7-text-rerank"
RERANK_INSTRUCT = DATASETS[DATASET]["rerank_instruct"]
RERANK_DEPTH = DATASETS[DATASET]["rerank_depth"]


# =============================================#
#                  generation                  #
# =============================================#

# 作答模型：看检索到的页图写答案
GENERATION_MODEL = "qwen3.8-flash"

# 每题给作答模型看前多少页
ANSWER_PAGES = 4


# =============================================#
#                    agent                     #
# =============================================#

# agent 调模型时的思考预算（token）：限制思考长度，压住延迟
AGENT_THINKING_BUDGET = 1024

# 单次模型调用超时（秒）：超时不重试，走兜底作答
AGENT_REQUEST_TIMEOUT = 45

# 一题最多调几次模型（不含兜底作答那一次）
AGENT_MAX_MODEL_CALLS = 5

# search_pages 每次返回前几页的摘要
AGENT_SEARCH_RESULTS = 5

# view_pages 每次最多附几张页图
AGENT_VIEW_PAGES_PER_CALL = 4

# 一题总共最多看几页页图（含起点给的 ANSWER_PAGES 页）
AGENT_MAX_VIEWED_PAGES = 12
