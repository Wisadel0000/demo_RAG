"""调用 Ollama 上的 Qwen3-embedding 模型生成文本向量。"""

import ollama


MODEL = "qwen3-embedding:4b"


def embed(text: str, model: str = MODEL) -> list[float]:
    """将单条文本转为 embedding 向量。"""
    resp = ollama.embed(model=model, input=text)
    return resp["embeddings"][0]


def embed_batch(texts: list[str], model: str = MODEL) -> list[list[float]]:
    """批量将多条文本转为 embedding 向量。"""
    resp = ollama.embed(model=model, input=texts)
    return resp["embeddings"]


if __name__ == "__main__":
    samples = [
        "深度学习是一种机器学习方法",
        "Transformer 架构彻底改变了 NLP 领域",
    ]

    vec = embed(samples[0])
    print(f"向量维度: {len(vec)}")
    print(f"前 5 个元素: {vec[:5]}")

    vecs = embed_batch(samples)
    print(f"批量向量数: {len(vecs)}")