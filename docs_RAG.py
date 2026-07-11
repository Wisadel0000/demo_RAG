import chromadb
import ollama
import pathlib as pl

_data_dir = pl.Path("./docs-langchain")
MODEL = "qwen3-embedding:4b"

_data_client : chromadb.ClientAPI
_data_base : chromadb.Collection


def embed(text:str|list[str]):
    result =ollama.embed(model=MODEL, input=text)
    return result["embeddings"]

def collection_init():
    global  _data_base , _data_client
    if _data_client is None:
        _data_client = chromadb.Client()
    if _data_base is None:
        _data_base = _data_client.create_collection("my_database")
    return

def docs_import():
    if _data_base is None:
        raise ValueError("请先调用collection_init()以初始化数据库！")

    def divide():
        raise NotImplementedError

    index = []
    docs = []
    doc_paths = (_data_dir/"components").rglob('*.md')
    for file in doc_paths:
        try:
            with open(file,'r',encoding='utf-8') as f:
                text =file.name + f.read()
                index.append(file.name)
                docs.append(text)
        except Exception as e:
            raise e
    vect = embed(docs)
    _data_base.add(
        ids= index,
        embeddings= vect,
        documents= docs,
    )
    return

def query(text:str,results:int):
    if _data_base is None:
        raise ValueError("请先调用collection_init()以初始化数据库！")
    vect = embed(text)
    result = _data_base.query(
        query_embeddings=vect,
        n_results=results,
    )
    return result["documents"][0]

def insert(text:str,id:str):
    if _data_base is None:
        raise ValueError("请先调用collection_init()以初始化数据库！")
    vect = embed(text)
    _data_base.add(
        ids= [id,],
        embeddings= [vect,],
        documents= [text,],
    )
    return