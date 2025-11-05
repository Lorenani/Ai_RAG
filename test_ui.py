"""
测试 RAG 系统问答功能的脚本
用于验证系统是否正常工作
"""
import json
from pathlib import Path
from pyprojroot import here
from src.questions_processing import QuestionsProcessor
from src.pipeline import RunConfig, PipelineConfig

def test_single_question():
    """测试单个问题的处理"""
    print("=" * 60)
    print("开始测试 RAG 问答系统")
    print("=" * 60)
    
    # 配置路径
    data_root = here() / "data" / "test_set"
    paths = PipelineConfig(
        root_path=data_root,
        subset_name="subset.csv",
        questions_file_name="questions.json",
        pdf_reports_dir_name="pdf_reports",
        serialized=False,
        config_suffix=""
    )
    
    # 创建配置
    run_config = RunConfig(
        use_serialized_tables=False,
        parent_document_retrieval=True,
        llm_reranking=True,
        llm_reranking_sample_size=30,
        top_n_retrieval=10,
        parallel_requests=1,
        api_provider="dashscope",
        answering_model="qwen-turbo-latest",
        full_context=False
    )
    
    # 初始化处理器
    print("\n1. 正在初始化问题处理器...")
    processor = QuestionsProcessor(
        vector_db_dir=paths.vector_db_dir,
        documents_dir=paths.documents_dir,
        questions_file_path=None,
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
    print("✅ 处理器初始化成功")
    
    # 测试问题
    test_questions = [
        {
            "question": '"Mercia Asset Management PLC"年报中是否提到了并购？',
            "schema": "boolean"
        },
        {
            "question": '"Mercia Asset Management PLC"2022年主营业务的主要内容是什么？',
            "schema": "string"
        }
    ]
    
    print("\n2. 开始测试问题处理...")
    for i, test in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"测试问题 {i}:")
        print(f"问题: {test['question']}")
        print(f"类型: {test['schema']}")
        print("-" * 60)
        
        try:
            answer_dict = processor.process_question(
                question=test['question'],
                schema=test['schema']
            )
            
            if "error" in answer_dict:
                print(f"❌ 错误: {answer_dict['error']}")
            else:
                print("✅ 处理成功")
                print(f"最终答案: {answer_dict.get('final_answer', 'N/A')}")
                print(f"相关页码: {answer_dict.get('relevant_pages', [])}")
                print(f"推理摘要: {answer_dict.get('reasoning_summary', 'N/A')}")
                
        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_single_question()

