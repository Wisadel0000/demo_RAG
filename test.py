from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pyexpat.errors import messages

import docs_RAG

model = ChatOpenAI(
    model = "deepseek-v4-flash",
    api_key= "***",
    base_url= "https://api.deepseek.com",
)

@tool
def seek_in_docs(input_text:str):
    """这个函数用于从说明文档中查找与input_text相关的文档，会返回5段相关可能性最大的原文"""
    result = docs_RAG.query(input_text,5)
    print(f"[查询事件]\nquery:{input_text}\nresult:{result}\n")
    return result

docs_RAG.collection_init()
docs_RAG.docs_import()


agent = create_agent(
    model= model,
    tools=[seek_in_docs],
    checkpointer=InMemorySaver(),
    system_prompt="你是一个文档助手，负责帮用户查阅文档和解答问题。对所有与文档内容可能相关的问题，请先使用工具查阅文档再进行回复。"
)

while True:
    user = input("input_message:")
    result = agent.invoke(
        {'messages' :[HumanMessage(user)]},
        config={"configurable": {"thread_id": 1}},
    )
    print(result["messages"][-1].content)

