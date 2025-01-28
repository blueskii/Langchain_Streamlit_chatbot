import os
import streamlit as st
import tempfile

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# pysqlite3라는 라이브러리를 동적으로 임포트한 후, 이를 표준 라이브러리의 sqlite3 모듈로 대체
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_chroma import Chroma

#오픈AI API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

#Chroma tenant 오류 방지 위한 코드
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

#cache_resource로 한번 실행한 결과 캐싱해두기
@st.cache_resource
def load_pdf(_file):
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
        tmp_file.write(_file.getvalue())
        tmp_file_path = tmp_file.name
        #PDF 파일 업로드
        loader = PyPDFLoader(file_path=tmp_file_path)
        pages = loader.load_and_split()
    return pages

#텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)
    vectorstore = Chroma.from_documents(split_docs, OpenAIEmbeddings(model='text-embedding-3-small'))
    return vectorstore

#검색된 문서를 하나의 텍스트로 합치는 헬퍼 함수
def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

#PDF 문서 기반 RAG 체인 구축
@st.cache_resource
def chaining(_pages):
    vectorstore = create_vector_store(_pages)
    retriever = vectorstore.as_retriever()

    #이 부분의 시스템 프롬프트는 기호에 따라 변경하면 됩니다.
    qa_system_prompt = """
        당신은 질문-답변 작업을 위한 도우미입니다.
        다음에 제공된 검색된 문맥을 사용하여 질문에 답변하세요.
        만약 답을 모른다면, 모른다고만 말씀하세요.
        답변은 완벽하게 작성해주세요.
        답변에 이모지를 사용해주세요.
        답변은 한국어로 작성하며 존댓말을 사용해 주시기 바랍니다.
        {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o")
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Streamlit UI
st.header("ChatPDF 💬 📚")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    pages = load_pdf(uploaded_file)

    rag_chain = chaining(pages)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "무엇이든 물어보세요!"}]

    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

    if prompt_message := st.chat_input("질문을 입력해주세요 :)"):
        st.chat_message("human").write(prompt_message)
        st.session_state.messages.append({"role": "user", "content": prompt_message})
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(prompt_message)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
                
