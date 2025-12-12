import streamlit as st
import os
import glob
import cohere
from openai import OpenAI
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

# --- [1. 기본 설정 및 제목] ---
st.set_page_config(page_title="경인교육대학교 대학원 규정 챗봇", page_icon="🎓")

# 제목 스타일 조정 (가운데 정렬 + 크기 조절)
st.markdown(
    """
    <h1 style='text-align: center; font-size: 36px; margin-bottom: 30px;'>
        🎓 경인교육대학교 대학원 규정 안내 AI
    </h1>
    """, 
    unsafe_allow_html=True
)

# --- [사이드바: 라이센스 표기 (요청하신 부분)] ---
with st.sidebar:
    st.header("정보")
    st.info("이 챗봇은 경인교육대학교 대학원 규정 PDF 문서를 기반으로 답변합니다.")
    
    # 하단 공백 추가 (디자인상 아래로 밀어내기)
    st.markdown("<br>" * 10, unsafe_allow_html=True) 
    st.markdown("---")
    
    # 라이센스 및 저작자 표기
    st.markdown(
        """
        <div style='text-align: center; color: grey; font-size: 12px;'>
            Developed by <br>
            <b>Prof. LCH</b> <br>
            (<a href='mailto:leesleek@ginue.ac.kr' style='text-decoration: none; color: grey;'>leesleek@ginue.ac.kr</a>)
        </div>
        """, 
        unsafe_allow_html=True
    )

# --- [2. API 클라이언트 초기화] ---
@st.cache_resource
def init_clients():
    try:
        # Streamlit Secrets에서 API 키 로드
        co = cohere.Client(st.secrets["COHERE_API_KEY"])
        openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        chroma_client = chromadb.Client()
        return co, openai_client, chroma_client
    except Exception as e:
        st.error("API 키 설정 오류: .streamlit/secrets.toml 파일을 확인해주세요.")
        return None, None, None

co, openai_client, chroma_client = init_clients()

# --- [3. 지식베이스 구축 함수] ---
@st.cache_resource
def load_and_index_pdfs():
    collection_name = "pdf_knowledge_base"
    
    # 기존 컬렉션 초기화 (데이터 갱신용)
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
        
    collection = chroma_client.create_collection(name=collection_name)
    
    pdf_files = glob.glob("gradu_data/*.pdf")
    if not pdf_files:
        st.warning("⚠️ 'gradu_data' 폴더에 PDF 파일이 없습니다.")
        return None

    status_text = st.empty()
    status_text.info("📚 문서를 분석하고 있습니다... 잠시만 기다려 주세요.")

    # 청크 설정 (요청 사항 반영)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
    )

    all_chunks = []
    all_metadatas = []

    for file_path in pdf_files:
        try:
            reader = PdfReader(file_path)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text: full_text += text + "\n"
            
            # 텍스트 정제
            full_text = re.sub(r'\n{3,}', '\n\n', full_text)
            
            chunks = text_splitter.split_text(full_text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({"source": os.path.basename(file_path)})
        except Exception as e:
            st.warning(f"{file_path} 처리 중 오류 발생: {e}")

    # 임베딩 및 저장
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = all_chunks[i:i+batch_size]
        batch_metas = all_metadatas[i:i+batch_size]
        batch_ids = [str(hash(t)) for t in batch_texts]
        
        response = openai_client.embeddings.create(
            input=batch_texts,
            model="text-embedding-3-small"
        )
        embeddings = [data.embedding for data in response.data]
        
        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_metas,
            ids=batch_ids
        )
    
    status_text.success(f"✅ 총 {len(pdf_files)}개의 규정 문서 학습 완료!")
    return collection

# 앱 실행 시 지식베이스 로드
if openai_client:
    collection = load_and_index_pdfs()

# --- [4. 채팅 인터페이스] ---

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 대학원 규정에 대해 무엇이든 물어보세요."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if not collection:
            st.error("지식베이스가 로드되지 않았습니다.")
            st.stop()

        # 1. 임베딩 & 검색 (Vector Search)
        query_embed = openai_client.embeddings.create(
            input=[prompt],
            model="text-embedding-3-small"
        ).data[0].embedding
        
        results = collection.query(query_embeddings=[query_embed], n_results=30)
        retrieved_docs = results['documents'][0]
        retrieved_metas = results['metadatas'][0]

        if not retrieved_docs:
            full_response = "제공된 문서에서 관련 정보를 찾을 수 없습니다."
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.stop()

        # 2. Rerank (Cohere)
        rerank_results = co.rerank(
            query=prompt,
            documents=retrieved_docs,
            model="rerank-multilingual-v3.0",
            top_n=5
        )
        
        final_docs = []
        sources = set()
        for hit in rerank_results.results:
            final_docs.append(retrieved_docs[hit.index])
            sources.add(retrieved_metas[hit.index]['source'])

        context = "\n\n".join(final_docs)
        source_text = ", ".join(sources)

        # 3. LLM 생성 (OpenAI) - 프롬프트 적용
        system_prompt = f"""
        당신은 경인교육대학교 대학원 규정 안내 AI입니다.
        아래의 [컨텍스트]와 [참고 문서 목록]을 바탕으로 사용자의 질문에 답변하세요.

        답변 작성 지침:
        1. 컨텍스트에서 찾은 정보만을 사용하여 답변하세요.
        2. 정확한 정보를 제공하되, 확실하지 않은 내용은 추측하지 마세요.
        3. 답변은 명확하고 이해하기 쉽게 작성하세요.
        4. 컨텍스트에 관련 정보가 없으면 "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 답변하세요.

        답변 형식:
        - 질문에 직접 답하는 명확한 답변을 제공하세요.
        - 필요시 단계별로 설명하거나 예시를 포함하세요.
        - 출처가 명확한 경우 해당 문서나 섹션을 언급하세요.

        [컨텍스트]
        {context}
        
        [참고 문서 목록]
        {source_text}
        """

        full_response = ""
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        
        if sources:
            st.caption(f"📚 참고 문서: {source_text}")
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})