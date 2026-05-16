
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import os
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, format_document,MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from file_history_store import get_history

class RagService:
    def __init__(self):

        load_dotenv()

        self.vector_store=VectorStoreService(
            embedding=DashScopeEmbeddings(
                model=config.embeddings_model_name,
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
            )
        )
        self.prompt_template=ChatPromptTemplate.from_messages(
            [
                ("system",config.system_prompt),
                MessagesPlaceholder("history"),
                ("user",config.user_prompt)
            ]
        )
        self.chat_model=ChatTongyi(
            model=config.chat_model_name,
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        self.chain=self.__get_chain()

    def __get_chain(self):
        retriever=self.vector_store.get_retriever()

        def format_document(docs:list[Document]):
            if docs is None:
                return "无相关数据"
            format_str=""
            for doc in docs:
                format_str+= f"文档片段: {doc.page_content}\n文档元数据: {doc.metadata}\n\n"
            return format_str
        def format_for_retriever(value):
            return value["input"]

        def format_for_prompt_template(value):
            new_value={}
            new_value["input"]=value["input"]["input"]
            new_value["history"]=value["input"]["history"]
            new_value["context"]=value["context"]
            return new_value

        chain =(
            {
                "input":RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document
            } | RunnableLambda(format_for_prompt_template) | self.prompt_template | self.chat_model | StrOutputParser()
        )

        conversation_chain=RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history"
        )

        return conversation_chain
