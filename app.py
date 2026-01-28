import streamlit as st
import os
from langchain_ollama import ChatOllama
from src.config_manager import ConfigManager
from src.loader import DocumentProcessor
from src.vector_store import VectorEngine
from src.log_utils import setup_logger, timer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 1. 系统初始化：加载配置与日志
st.set_page_config(page_title="Academic RAG Platform", layout="wide")
cfg = ConfigManager()
logger = setup_logger(cfg.configs['paths']['log_dir'])

# 2. 引擎实例化：注入 YAML 配置参数
# 从配置中动态读取，拒绝硬编码
rag_cfg = cfg.get_rag_config()
model_cfg = cfg.get_model_config()
path_cfg = cfg.configs['paths']

doc_processor = DocumentProcessor(
    chunk_size=rag_cfg['chunk_size'],
    chunk_overlap=rag_cfg['chunk_overlap']
)
vector_engine = VectorEngine(
    db_path=path_cfg['db_dir'],
    embedding_model=model_cfg['embedding']
)

# 3. Session State 状态管理
if "messages" not in st.session_state: st.session_state.messages = []
if "last_docs" not in st.session_state: st.session_state.last_docs = []

# --- 侧边栏：文件中心 ---
with st.sidebar:
    st.title("文件中心")
    uploaded_files = st.file_uploader("上传多篇学术论文 (PDF)", type="pdf", accept_multiple_files=True)

    if st.button("重建知识库", use_container_width=True):
        if uploaded_files:
            with st.spinner("正在清理旧索引并重新构建..."):
                # 处理上传文件并保存临时路径
                temp_paths = []
                for f in uploaded_files:
                    path = f"temp_{f.name}"
                    with open(path, "wb") as tmp:
                        tmp.write(f.getbuffer())
                    temp_paths.append(path)

                # 调用解耦后的模块执行重任
                chunks = doc_processor.process_files(temp_paths)
                vector_engine.rebuild_db(chunks)

                # 清理临时文件
                for p in temp_paths: os.remove(p)

            st.success(f"成功索引 {len(uploaded_files)} 篇文档！")
            logger.info(f"用户重建了知识库，文档数: {len(uploaded_files)}")
        else:
            st.warning("请先选择 PDF 文件")

    if st.button("清空对话记录", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 主界面布局：左侧对话，右侧证据 ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("对话")
    # 渲染历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 聊天输入逻辑
    if prompt := st.chat_input("基于上传的论文对比分析..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                # --- A. 动态参数计算 ---
                # 1. 获取文件数量，动态扩大初筛范围 (为 Rerank 提供候选池)
                num_files = len(uploaded_files) if uploaded_files else 1
                base_k = rag_cfg.get('base_k_per_file', 5)

                # 初筛数量 (Top-K)：通常取最终展示数量的 3-4 倍
                initial_k = max(base_k * num_files, 15)
                # 精排保留数量 (Top-N)：最终喂给大模型的高质量片段数
                final_n = rag_cfg.get('rerank_top_n', 5)

                # --- B. 初筛 (Retrieval) ---
                logger.info(f"开始初筛，动态 k={initial_k}")
                retriever = vector_engine.get_retriever(k=initial_k)
                initial_docs = retriever.get_relevant_documents(prompt)

                # --- C. 精排 (Rerank) ---
                # 使用你新写的 Reranker 类进行二次打分
                from src.reranker import Reranker

                # 建议在 app 启动时初始化 reranker 实例，此处为演示逻辑
                reranker = Reranker(model_name=model_cfg.get('reranker', 'BAAI/bge-reranker-base'))

                with st.spinner("🔍 正在精准过滤干扰片段..."):
                    final_docs = reranker.rerank(query=prompt, documents=initial_docs, top_n=final_n)

                st.session_state.last_docs = final_docs  # 实时同步证据栏
                logger.info(f"精排完成，保留最优片段数: {len(final_docs)}")

                # --- D. 组装 Context ---
                context = "\n\n".join([
                    f"来源《{d.metadata['source']}》P{d.metadata['page'] + 1}:\n{d.page_content}"
                    for d in final_docs
                ])

                # --- E. 推理：调用本地 Ollama ---
                llm = ChatOllama(model=model_cfg['llm'])
                full_prompt = f"你是一个严谨的学术助手。请根据资料回答：\n\n资料：\n{context}\n\n问题：{prompt}"

                # 流式展示正式答案
                response = st.write_stream(llm.stream(full_prompt))
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"发生错误: {str(e)}")
                logger.error(f"问答链路故障: {e}")

with col2:
    st.subheader("证据追溯")
    # 只有提问后才会显示相关片段
    if st.session_state.last_docs:
        for i, doc in enumerate(st.session_state.last_docs):
            with st.expander(f"证据 {i + 1}: {doc.metadata['source']} (P{doc.metadata['page'] + 1})"):
                st.write(doc.page_content)
    else:
        st.info("相关论文片段将在此显示。")