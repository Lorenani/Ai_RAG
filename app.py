import streamlit as st
import json
import re
import os
from pathlib import Path
from pyprojroot import here
from src.questions_processing import QuestionsProcessor
from src.pipeline import RunConfig, PipelineConfig

# 从Streamlit Secrets或环境变量读取API密钥
def get_dashscope_api_key():
    """从Streamlit Secrets或环境变量获取DashScope API密钥"""
    api_key = None
    
    # 优先从Streamlit Secrets读取（Streamlit Cloud使用这种方式）
    try:
        if hasattr(st, 'secrets'):
            # 尝试多种可能的访问方式
            if hasattr(st.secrets, 'get'):
                api_key = st.secrets.get('DASHSCOPE_API_KEY')
            elif isinstance(st.secrets, dict) and 'DASHSCOPE_API_KEY' in st.secrets:
                api_key = st.secrets['DASHSCOPE_API_KEY']
            elif hasattr(st.secrets, 'DASHSCOPE_API_KEY'):
                api_key = getattr(st.secrets, 'DASHSCOPE_API_KEY', None)
            
            # 如果获取到密钥，去除首尾空格
            if api_key:
                api_key = str(api_key).strip()
    except Exception as e:
        # 如果读取secrets失败，继续尝试环境变量
        pass
    
    # 从环境变量读取
    if not api_key:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            api_key = str(api_key).strip()
    
    return api_key

# 设置API密钥到环境变量（确保所有模块都能访问）
# 每次应用启动时都重新读取并设置
def ensure_api_key_set():
    """确保API密钥已设置到环境变量"""
    api_key = get_dashscope_api_key()
    if api_key:
        # 清理并设置
        api_key = str(api_key).strip()
        os.environ["DASHSCOPE_API_KEY"] = api_key
        return True
    return False

# 在模块加载时设置一次
ensure_api_key_set()

# 页面配置
st.set_page_config(
    page_title="企业知识库 RAG 问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .reasoning-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .page-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'data_path' not in st.session_state:
    st.session_state.data_path = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 侧边栏配置
# 数据路径选择（在sidebar外定义，以便主内容区也能访问）
data_path_option = st.sidebar.selectbox(
    "📁 选择数据集",
    ["erc2_set", "erc3_set"],
    index=0,  # 默认选择 erc2_set
    help="选择要使用的数据集"
)

data_root = here() / "data" / data_path_option

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h1>⚙️ 系统配置</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示数据集信息
    st.info(f"📂 数据集路径: `{data_root}`")
    
    # 显示API密钥状态
    api_key_status = get_dashscope_api_key()
    if api_key_status:
        # 显示密钥前缀用于验证（不显示完整密钥）
        key_prefix = api_key_status[:8] + "..." if len(api_key_status) > 8 else api_key_status
        key_length = len(api_key_status)
        st.success(f"🔑 API密钥: 已配置 ({key_prefix}, 长度: {key_length})")
        
        # 添加测试按钮
        if st.button("🧪 测试API密钥", help="点击测试API密钥是否有效"):
            with st.spinner("正在测试API密钥..."):
                try:
                    import dashscope
                    test_key = str(api_key_status).strip()
                    dashscope.api_key = test_key
                    # 尝试一个简单的embedding调用
                    rsp = dashscope.TextEmbedding.call(
                        model="text-embedding-v1",
                        input=["test"]
                    )
                    
                    # 检查响应
                    if isinstance(rsp, dict):
                        status_code = rsp.get('status_code')
                        code = rsp.get('code', '')
                    elif hasattr(rsp, 'status_code'):
                        status_code = rsp.status_code
                        code = getattr(rsp, 'code', '')
                    else:
                        status_code = None
                    
                    if status_code == 401 or code == 'InvalidApiKey':
                        st.error(f"❌ API密钥无效！\n错误代码: {code}\n请检查：\n1. 密钥是否正确\n2. 密钥是否过期\n3. 账户是否有权限")
                    elif status_code == 200 or (hasattr(rsp, 'output') and rsp.output):
                        st.success("✅ API密钥有效！可以正常使用")
                    else:
                        st.warning(f"⚠️ 测试结果不明确，状态码: {status_code}")
                except Exception as e:
                    st.error(f"❌ 测试失败: {str(e)}")
    else:
        st.warning("⚠️ API密钥: 未配置（请在Streamlit Cloud的Secrets中配置DASHSCOPE_API_KEY）")
    
    # 高级配置
    with st.expander("⚙️ 高级配置", expanded=False):
        use_reranking = st.checkbox("启用 LLM Reranking", value=True, help="使用 LLM 对检索结果进行重排序")
        top_n = st.slider("检索数量 (Top N)", min_value=5, max_value=30, value=10, help="从向量数据库检索的文档数量")
        rerank_sample_size = st.slider("Reranking 样本数", min_value=10, max_value=50, value=30, help="用于重排序的初始检索数量")
        
    # 初始化按钮
    if st.button("🚀 初始化系统", type="primary", use_container_width=True):
        with st.spinner("正在初始化 RAG 系统..."):
            try:
                # 创建配置
                run_config = RunConfig(
                    use_serialized_tables=False,
                    parent_document_retrieval=True,
                    llm_reranking=use_reranking,
                    llm_reranking_sample_size=rerank_sample_size,
                    top_n_retrieval=top_n,
                    parallel_requests=1,
                    api_provider="dashscope",
                    answering_model="qwen-turbo-latest",
                    full_context=False
                )
                
                # 初始化路径配置
                paths = PipelineConfig(
                    root_path=data_root,
                    subset_name="subset.csv",
                    questions_file_name="questions.json",
                    pdf_reports_dir_name="pdf_reports",
                    serialized=False,
                    config_suffix=""
                )
                
                # 确保API密钥已设置（在初始化processor之前）
                if not ensure_api_key_set():
                    st.error("❌ API密钥未配置，无法初始化系统")
                    st.info("💡 请在Streamlit Cloud的Secrets中配置DASHSCOPE_API_KEY")
                    st.stop()
                
                # 初始化问题处理器
                processor = QuestionsProcessor(
                    vector_db_dir=paths.vector_db_dir,
                    documents_dir=paths.documents_dir,
                    questions_file_path=None,  # 不使用文件，直接处理单个问题
                    new_challenge_pipeline=True,
                    subset_path=paths.subset_path,
                    parent_document_retrieval=run_config.parent_document_retrieval,
                    llm_reranking=run_config.llm_reranking,
                    llm_reranking_sample_size=run_config.llm_reranking_sample_size,
                    top_n_retrieval=run_config.top_n_retrieval,
                    parallel_requests=run_config.parallel_requests,
                    api_provider=run_config.api_provider,
                    answering_model=run_config.answering_model,
                    full_context=run_config.full_context
                )
                
                st.session_state.processor = processor
                st.session_state.data_path = data_root
                # 清除旧的companies_df缓存，确保使用新数据集
                if hasattr(st.session_state.processor, 'companies_df'):
                    delattr(st.session_state.processor, 'companies_df')
                st.success("✅ 系统初始化成功！")
                st.session_state.chat_history = []  # 清空历史记录
                
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ 初始化失败: {error_msg}")
                # 如果是API密钥相关错误，给出更明确的提示
                if "API" in error_msg or "api_key" in error_msg.lower() or "key" in error_msg.lower():
                    st.info("💡 提示：如果是在Streamlit Cloud上运行，请确保在应用设置的Secrets中配置了DASHSCOPE_API_KEY")
                st.exception(e)
    
    # 显示系统状态
    if st.session_state.processor:
        st.success("✅ 系统已就绪")
    else:
        st.warning("⚠️ 系统未初始化")

# 主界面
st.markdown("""
<div class="main-header">
    <h1>📚 企业知识库 RAG 问答系统</h1>
    <p style="margin: 0; opacity: 0.9;">基于检索增强生成的企业年报智能问答系统</p>
</div>
""", unsafe_allow_html=True)

# 检查系统是否已初始化，以及数据集是否匹配
if st.session_state.processor is None:
    st.warning("⚠️ 请先在侧边栏初始化系统")
    st.info("💡 提示：点击左侧的「初始化系统」按钮来加载向量数据库和配置")
    st.stop()
elif st.session_state.data_path != data_root:
    # 数据集已切换，需要重新初始化
    st.warning("⚠️ 数据集已切换，请重新初始化系统")
    st.info(f"💡 当前数据集：{data_path_option}，但系统使用的是：{st.session_state.data_path}")
    st.session_state.processor = None  # 清除旧的processor
    st.stop()

# 问题输入区域
st.markdown("### 💬 提问")
question = st.text_area(
    "请输入您的问题",
    height=120,
    placeholder='例如："Mercia Asset Management PLC"年报中是否提到了并购？\n或者："中芯国际"2024年主营业务的主要内容是什么？',
    help="💡 提示：问题中应包含公司名称（用引号括起来），系统会自动识别并检索相关信息"
)

# 问题类型选择（默认 string）
question_kind = st.selectbox(
    "问题类型",
    ["string", "boolean", "number", "names"],
    index=0,
    help="string: 开放性问题（默认）\nboolean: 是否类问题\nnumber: 数字类问题\nnames: 名称列表类问题"
)

# 提交按钮
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submit_button = st.button("🚀 提交问题", type="primary", use_container_width=True)

# 显示聊天历史
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 💬 历史对话")
    for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # 只显示最近5条
        with st.expander(f"Q{i+1}: {chat['question'][:50]}...", expanded=False):
            st.markdown(f"**问题：** {chat['question']}")
            st.markdown(f"**答案：** {chat['answer']}")
            if chat.get('pages'):
                st.markdown(f"**相关页码：** {', '.join([f'第 {p} 页' for p in chat['pages']])}")

# 显示结果区域
if submit_button and question:
    if not question.strip():
        st.warning("⚠️ 请输入问题")
    else:
        with st.spinner("🤔 正在思考中，请稍候..."):
            try:
                # 确保API密钥已设置（每次处理问题前都检查）
                if not ensure_api_key_set():
                    st.error("❌ API密钥未配置")
                    st.stop()
                
                # 处理问题
                answer_dict = st.session_state.processor.process_question(
                    question=question,
                    schema=question_kind
                )
                
                # 检查是否有错误
                if "error" in answer_dict:
                    st.error(f"❌ 处理出错: {answer_dict['error']}")
                    st.stop()
                
                # 解析答案（如果是 JSON 字符串）
                final_answer = answer_dict.get("final_answer", "")
                if isinstance(final_answer, str) and final_answer.startswith("```"):
                    # 尝试解析 JSON 字符串
                    try:
                        json_str = final_answer.strip("```json\n").strip("```").strip()
                        parsed = json.loads(json_str)
                        final_answer = parsed.get("final_answer", final_answer)
                        step_by_step = parsed.get("step_by_step_analysis", "")
                        reasoning = parsed.get("reasoning_summary", "")
                        relevant_pages = parsed.get("relevant_pages", [])
                    except:
                        step_by_step = answer_dict.get("step_by_step_analysis", "")
                        reasoning = answer_dict.get("reasoning_summary", "")
                        relevant_pages = answer_dict.get("relevant_pages", [])
                else:
                    step_by_step = answer_dict.get("step_by_step_analysis", "")
                    reasoning = answer_dict.get("reasoning_summary", "")
                    relevant_pages = answer_dict.get("relevant_pages", [])
                
                # 保存到聊天历史
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": final_answer,
                    "pages": relevant_pages
                })
                
                # 显示答案
                st.markdown("---")
                st.markdown("### 📝 答案")
                
                # 最终答案卡片 - 使用 Markdown 渲染
                st.markdown(f"""
                <div class="answer-card">
                    <h3 style="color: #667eea; margin-top: 0;">🎯 最终答案</h3>
                    <div style="font-size: 16px; line-height: 1.8; color: #333;">
                        {final_answer}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 推理摘要
                if reasoning:
                    st.markdown(f"""
                    <div class="reasoning-box">
                        <strong>📊 推理摘要：</strong><br>
                        {reasoning}
                    </div>
                    """, unsafe_allow_html=True)
                
                # 相关页码 - 使用 badge 样式
                if relevant_pages:
                    pages_html = " ".join([f'<span class="page-badge">第 {p} 页</span>' for p in relevant_pages])
                    st.markdown(f"""
                    <div style="margin: 1rem 0;">
                        <strong>📄 相关页码：</strong><br>
                        {pages_html}
                    </div>
                    """, unsafe_allow_html=True)
                
                # 分步分析 - 使用 Markdown 渲染
                if step_by_step:
                    with st.expander("🔍 详细分析过程", expanded=False):
                        # 将分步分析按行分割并格式化
                        steps = step_by_step.split('\n')
                        formatted_steps = []
                        for step in steps:
                            step = step.strip()
                            if step:
                                # 如果是数字开头的步骤，加粗
                                if re.match(r'^\d+\.', step):
                                    formatted_steps.append(f"**{step}**")
                                else:
                                    formatted_steps.append(step)
                        st.markdown('\n\n'.join(formatted_steps))
                
                # 引用信息
                references = answer_dict.get("references", [])
                if references:
                    with st.expander("📚 文档引用", expanded=False):
                        for ref in references:
                            st.json(ref)
                
            except ValueError as e:
                error_msg = str(e)
                # 显示中文错误信息（已经是中文的错误信息会直接显示）
                if "未在" in error_msg or "No company name found" in error_msg:
                    st.error(f"❌ {error_msg}")
                else:
                    st.error(f"❌ 错误: {error_msg}")
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ 处理问题时出错: {error_msg}")
                # 如果是API密钥相关错误，给出更明确的提示
                if "API" in error_msg or "api_key" in error_msg.lower() or "key" in error_msg.lower() or "None" in error_msg:
                    st.info("💡 提示：如果是在Streamlit Cloud上运行，请检查：\n"
                           "1. 在应用设置的Secrets中配置了DASHSCOPE_API_KEY\n"
                           "2. API密钥格式正确（一行，用引号包裹）\n"
                           "3. 保存后等待1-2分钟让配置生效")
                st.exception(e)

# 底部信息
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>💡 <strong>使用提示：</strong></p>
    <ul style="text-align: left; display: inline-block;">
        <li>问题中应包含公司名称（用引号括起来）</li>
        <li>系统会自动从向量数据库中检索相关信息并生成答案</li>
        <li>支持开放性问题、是否类问题、数字类问题和名称列表类问题</li>
    </ul>
</div>
""", unsafe_allow_html=True)
