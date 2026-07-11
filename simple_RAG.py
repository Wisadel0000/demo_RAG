import chromadb
import ollama

#创建chroma客户端
chroma_client = chromadb.Client()

#创建模型对象
MODEL = "qwen3-embedding:4b"

def embed(text:str|list[str]):
    result =ollama.embed(model=MODEL, input=text)
    return result["embeddings"]

#收集集合创建
collecton = chroma_client.create_collection(name="my_collection")

documents=[
        "这是个关于菠萝的文档",
        "这是个关于橙子的文档"
    ]

vect = embed(documents)

#添加简单的文本（使用默认的all-MiniLM-L6-v2模型）
collecton.add(
    ids=["id1","id2"],
    embeddings=vect,
    documents=documents
)

query_text = ["这是个关于夏威夷的文档"]

#集合查询
result = collecton.query(
    # query_texts=query_text, #这句会被编码进行搜索
    query_embeddings=embed(query_text),
    n_results=2 #返回多少条消息
)

print(result["documents"][0])