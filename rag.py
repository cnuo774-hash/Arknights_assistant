
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
from operator import itemgetter

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
        retriever = self.vector_store.get_retriever()

        # ==========================================
        # 1. 定义“问题改写”层的 Prompt
        # ==========================================
        condense_question_prompt = ChatPromptTemplate.from_messages([
            ("system", config.condense_question_system_template),
            MessagesPlaceholder("history"),
            ("user", "{input}")
        ])
        condense_question_chain = condense_question_prompt | self.chat_model | StrOutputParser()
        def format_document(docs: list[Document]):
            if not docs:
                return "无相关数据"
            format_str = ""
            for doc in docs:
                format_str += f"文档片段: {doc.page_content}\n文档元数据: {doc.metadata}\n\n"
            return format_str

        chain = (
                RunnablePassthrough.assign(
                    standalone_question=condense_question_chain
                )
                | RunnablePassthrough.assign(
            context=itemgetter("standalone_question") | RunnableLambda(retriever.invoke) | format_document
        )
                | self.prompt_template
                | self.chat_model
                | StrOutputParser()
        )
        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        return conversation_chain
