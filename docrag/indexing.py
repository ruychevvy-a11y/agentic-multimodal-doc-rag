import os

from pymilvus import DataType, Function, FunctionType, MilvusClient

from docrag import config
from docrag.embeddings import embed_image_pages, embed_text_pages
from docrag.jsonl import load_jsonl

# 两路 dense 的索引：HNSW 图索引 + 余弦距离
DENSE_INDEX_TYPE = "HNSW"
DENSE_METRIC_TYPE = "COSINE"

# HNSW 建图参数：M = 每个节点的出边数，efConstruction = 建图时的候选队列长度
HNSW_PARAMS = {"M": 16, "efConstruction": 200}

# page_id / doc_id 最大长度（MMLongBench 的 doc_id 是 PDF 文件名，可能很长）
ID_MAX_LENGTH = 256

# text 字段最大 UTF-8 字节数
TEXT_MAX_LENGTH = 65535

# 单次写入行数
INSERT_BATCH_SIZE = 200


# 连接 Milvus
def milvus_client():
    return MilvusClient(uri=config.MILVUS_URI)


# 一行（= 一页）的字段定义
def page_schema(client):
    schema = client.create_schema()
    schema.add_field("page_id", DataType.VARCHAR, max_length=ID_MAX_LENGTH, is_primary=True)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=ID_MAX_LENGTH)
    schema.add_field("image_path", DataType.VARCHAR, max_length=512)
    schema.add_field("source", DataType.VARCHAR, max_length=16)

    # 页面正文；enable_analyzer 是 BM25 Function 的前提
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=TEXT_MAX_LENGTH,
        enable_analyzer=True,
        analyzer_params={"type": "english"},
    )

    schema.add_field(config.IMAGE_VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=config.IMAGE_EMBEDDING_DIM)
    schema.add_field(config.TEXT_VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=config.TEXT_EMBEDDING_DIM)

    # BM25 稀疏向量：维度是词表大小，不用声明
    schema.add_field(config.BM25_VECTOR_FIELD, DataType.SPARSE_FLOAT_VECTOR)

    # text --> BM25 稀疏向量由 Milvus 自动算，写入时不提供这个字段
    schema.add_function(
        Function(
            name="bm25_from_text",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=[config.BM25_VECTOR_FIELD],
        )
    )
    return schema


# 三个向量字段的索引
def page_index_params(client):
    index_params = client.prepare_index_params()

    for field in (config.IMAGE_VECTOR_FIELD, config.TEXT_VECTOR_FIELD):
        index_params.add_index(
            field_name=field,
            index_type=DENSE_INDEX_TYPE,
            metric_type=DENSE_METRIC_TYPE,
            params=HNSW_PARAMS,
        )

    # 稀疏倒排索引，DAAT_MAXSCORE 剪枝
    index_params.add_index(
        field_name=config.BM25_VECTOR_FIELD,
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE", **config.BM25_PARAMS},
    )
    return index_params


# 建表（同时建索引、load）--> 是否建成；表已存在时问要不要删了重建
def create_page_collection():
    client = milvus_client()
    name = config.PAGE_COLLECTION

    if client.has_collection(name):
        count = client.get_collection_stats(name)["row_count"]
        answer = input(f"[{name}] 已有 {count} 行，是否删除重建？(三路向量会一起删除)。[y/N] ")
        if answer.strip().lower() != "y":
            print("已取消。")
            # 返回 False 让调用方跳过写入，避免重复主键
            return False
        client.drop_collection(name)
        print(f"已删除 [{name}]")

    client.create_collection(
        collection_name=name,
        schema=page_schema(client),
        index_params=page_index_params(client),
    )
    print(f"[{name}] created, indexes: {client.list_indexes(name)}")
    return True


# 建索引用的正文：解析出的正文 + 视觉描述；没有描述就是原正文
# 例：'Figure 3 ...' + 'Bar chart, 2015 highest' --> 'Figure 3 ...\nVisual content: Bar chart, 2015 highest'
def index_text(text, description):
    if not description:
        return text
    return f"{text}\nVisual content: {description}".strip()


# 一页语料 + 两路向量 --> 一行 Milvus 数据
def page_row(row, image_vec, text_vec):
    return {
        "page_id": row["page_id"],
        "doc_id": row["doc_id"],
        "image_path": row["image_path"],
        "source": row["source"],
        "text": row["text"],
        config.IMAGE_VECTOR_FIELD: image_vec,
        config.TEXT_VECTOR_FIELD: text_vec,
    }


# 建表、嵌入全部页、写进 Milvus --> 表的行数
def index_pages():
    if not create_page_collection():
        return 0

    client = milvus_client()
    rows = load_jsonl(config.CORPUS_PATH)
    # 有视觉描述的数据集把描述拼进正文，BM25 / 文本 dense / 重排读的都是拼好的正文
    descriptions = (
        {row["page_id"]: row["description"] for row in load_jsonl(config.DESCRIPTIONS_PATH)}
        if os.path.exists(config.DESCRIPTIONS_PATH)
        else {}
    )
    for row in rows:
        row["text"] = index_text(row["text"], descriptions.get(row["page_id"], ""))

    image_vecs = embed_image_pages([row["image_path"] for row in rows])
    text_vecs = embed_text_pages([row["text"] for row in rows])

    # 三者按下标对应，数量不对就是错位
    assert len(image_vecs) == len(text_vecs) == len(rows), "向量数量与语料对不上"

    for i in range(0, len(rows), INSERT_BATCH_SIZE):
        chunk = rows[i : i + INSERT_BATCH_SIZE]
        client.insert(
            collection_name=config.PAGE_COLLECTION,
            data=[page_row(row, image_vecs[i + j], text_vecs[i + j]) for j, row in enumerate(chunk)],
        )

    # flush 后数据才可搜索
    client.flush(config.PAGE_COLLECTION)
    count = client.get_collection_stats(config.PAGE_COLLECTION)["row_count"]
    print(f"[{config.PAGE_COLLECTION}] row_count = {count}")
    return count
