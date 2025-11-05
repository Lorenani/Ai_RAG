import json
import logging
from typing import List, Tuple, Dict, Union
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from src.reranking import LLMReranker

_log = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        # 初始化BM25检索器，指定BM25索引和文档目录
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir
        
    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        # 按公司名检索相关文本块，返回BM25分数最高的top_n个块
        document_path = None
        for path in self.documents_dir.glob("*.json"):
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                if doc["metainfo"]["company_name"] == company_name:
                    document_path = path
                    document = doc
                    break
                    
        if document_path is None:
            raise ValueError(f"No report found with '{company_name}' company name.")
            
        # 加载对应的BM25索引
        bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1_name']}.pkl"
        with open(bm25_path, 'rb') as f:
            bm25_index = pickle.load(f)
            
        # 获取文档内容和BM25索引
        document = document
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        # 计算BM25分数
        tokenized_query = query.split()
        scores = bm25_index.get_scores(tokenized_query)
        
        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]
        
        retrieval_results = []
        seen_pages = set()
        
        for index in top_indices:
            score = round(float(scores[index]), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": score,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": score,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
        
        return retrieval_results



class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path, embedding_provider: str = "dashscope"):
        # 初始化向量检索器，加载所有向量库和文档
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.all_dbs = self._load_dbs()
        # 默认使用 dashscope 作为 embedding provider
        self.embedding_provider = embedding_provider.lower()
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        # 根据 embedding_provider 初始化对应的 LLM 客户端
        # 注意：不在这里设置dashscope.api_key，因为此时环境变量可能还未设置
        # 而是在_get_embedding中每次使用时动态读取
        load_dotenv()
        if self.embedding_provider == "openai":
            llm = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=None,
                max_retries=2
            )
            return llm
        elif self.embedding_provider == "dashscope":
            # dashscope不需要client对象，API密钥在_get_embedding中动态设置
            return None
        else:
            raise ValueError(f"不支持的 embedding provider: {self.embedding_provider}")

    def _get_embedding(self, text: str):
        # 根据 embedding_provider 获取文本的向量表示
        if self.embedding_provider == "openai":
            embedding = self.llm.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return embedding.data[0].embedding
        elif self.embedding_provider == "dashscope":
            import dashscope
            # 确保API密钥已设置 - 每次使用时都重新读取，确保获取最新值
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                # 尝试从dashscope模块获取（如果之前设置过）
                if hasattr(dashscope, 'api_key') and dashscope.api_key:
                    api_key = dashscope.api_key
                else:
                    raise RuntimeError("DASHSCOPE_API_KEY环境变量未设置，请在Streamlit Secrets中配置")
            # 去除首尾空格，防止格式问题
            api_key = str(api_key).strip()
            if not api_key:
                raise RuntimeError("DASHSCOPE_API_KEY为空，请检查Streamlit Secrets配置")
            
            # 调试信息：检查密钥格式
            key_length = len(api_key)
            key_prefix = api_key[:10] if len(api_key) >= 10 else api_key
            key_suffix = api_key[-10:] if len(api_key) >= 10 else ""
            # 检查是否有特殊字符（如换行符、制表符等）
            has_newline = '\n' in api_key or '\r' in api_key
            has_tab = '\t' in api_key
            
            # 如果密钥长度不对或包含特殊字符，给出警告
            if key_length != 64:
                _log.warning(f"API密钥长度异常: {key_length} (期望64), 前缀: {key_prefix}, 后缀: {key_suffix}")
            if has_newline or has_tab:
                # 清理特殊字符
                api_key = api_key.replace('\n', '').replace('\r', '').replace('\t', '')
                api_key = api_key.strip()
                _log.warning(f"检测到API密钥中包含换行符或制表符，已清理。新长度: {len(api_key)}")
            
            # 每次调用都重新设置，确保使用最新的密钥
            dashscope.api_key = api_key
            rsp = dashscope.TextEmbedding.call(
                model="text-embedding-v1",
                input=[text]
            )
            # 检查响应是否为None
            if rsp is None:
                raise RuntimeError("DashScope API返回None，可能是API密钥无效或网络问题，请检查API密钥配置")
            
            # 检查是否为字典或对象，并获取状态码
            status_code = None
            if isinstance(rsp, dict):
                status_code = rsp.get('status_code')
                code = rsp.get('code', '')
                message = rsp.get('message', '')
            elif hasattr(rsp, 'status_code'):
                status_code = rsp.status_code
                code = getattr(rsp, 'code', '')
                message = getattr(rsp, 'message', '')
            
            # 如果状态码是401，说明API密钥无效
            if status_code == 401 or code == 'InvalidApiKey':
                # 显示密钥调试信息（不显示完整密钥）
                debug_info = f"密钥长度: {key_length}, 前缀: {key_prefix}, 后缀: {key_suffix}"
                if has_newline or has_tab:
                    debug_info += f", 检测到特殊字符已清理"
                
                raise RuntimeError(
                    f"❌ DashScope API密钥无效！\n"
                    f"错误代码: {code}\n"
                    f"错误信息: {message}\n"
                    f"调试信息: {debug_info}\n\n"
                    f"请检查：\n"
                    f"1. 在Streamlit Cloud的Secrets中配置了正确的DASHSCOPE_API_KEY\n"
                    f"2. API密钥格式: DASHSCOPE_API_KEY = \"完整密钥\"（一行，用引号包裹，等号前后有空格）\n"
                    f"3. 确保密钥没有多余空格或隐藏字符\n"
                    f"4. API密钥没有过期或被禁用\n"
                    f"5. 保存后等待1-2分钟让配置生效\n\n"
                    f"💡 提示：如果本地能运行但Streamlit Cloud不行，可能是Secrets中的密钥格式有问题。"
                    f"请删除Secrets中的内容，重新输入：DASHSCOPE_API_KEY = \"你的完整密钥\""
                )
            
            # 兼容 dashscope 返回格式，可能返回对象或字典
            # 先安全获取output
            output = None
            if hasattr(rsp, 'get'):
                output = rsp.get('output')
            elif hasattr(rsp, 'output'):
                output = rsp.output
            elif isinstance(rsp, dict) and 'output' in rsp:
                output = rsp['output']
            
            # 检查embeddings
            if output and isinstance(output, dict) and 'embeddings' in output:
                # 多条输入（本处只有一条）
                emb = output['embeddings'][0]
                if emb['embedding'] is None or len(emb['embedding']) == 0:
                    raise RuntimeError(f"DashScope返回的embedding为空，text_index={emb.get('text_index', None)}")
                return emb['embedding']
            elif output and isinstance(output, dict) and 'embedding' in output:
                # 兼容单条输入格式
                if output['embedding'] is None or len(output['embedding']) == 0:
                    raise RuntimeError("DashScope返回的embedding为空")
                return output['embedding']
            else:
                raise RuntimeError(f"DashScope embedding API返回格式异常: {rsp}")
        else:
            raise ValueError(f"不支持的 embedding provider: {self.embedding_provider}")

    @staticmethod
    def set_up_llm():
        # 静态方法，初始化OpenAI LLM
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
        )
        return llm

    def _load_dbs(self):
        # 加载所有向量库和对应文档，建立映射
        all_dbs = []
        # 获取所有JSON文档路径
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        vector_db_files = {db_path.stem: db_path for db_path in self.vector_db_dir.glob('*.faiss')}
        
        for document_path in all_documents_paths:
            stem = document_path.stem
            if stem not in vector_db_files:
                _log.warning(f"No matching vector DB found for document {document_path.name}")
                continue
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue
            
            # 校验文档结构
            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
                continue
            
            try:
                vector_db = faiss.read_index(str(vector_db_files[stem]))
            except Exception as e:
                _log.error(f"Error reading vector DB for {document_path.name}: {e}")
                continue
                
            report = {
                "name": stem,
                "vector_db": vector_db,
                "document": document
            }
            all_dbs.append(report)
        return all_dbs

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        # 计算两个字符串的余弦相似度（通过嵌入）
        llm = VectorRetriever.set_up_llm()
        embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-3-large")
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False) -> List[Tuple[str, float]]:
        # 按公司名检索相关文本块，返回向量距离最近的top_n个块
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                _log.error(f"Report '{report.get('name')}' is missing 'metainfo'!")
                raise ValueError(f"Report '{report.get('name')}' is missing 'metainfo'!")
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        actual_top_n = min(top_n, len(chunks))
        
        # 获取 query 的 embedding，支持 openai/dashscope
        embedding = self._get_embedding(query)
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
    
        retrieval_results = []
        seen_pages = set()
        
        for distance, index in zip(distances[0], indices[0]):
            distance = round(float(distance), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": distance,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": distance,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
            
        return retrieval_results

    def retrieve_all(self, company_name: str) -> List[Dict]:
        # 检索公司所有文本块，返回全部内容
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        pages = document["content"]["pages"]
        
        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {
                "distance": 0.5,
                "page": page["page"],
                "text": page["text"]
            }
            all_pages.append(result)
            
        return all_pages


class HybridRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.reranker = LLMReranker()
        
    def retrieve_by_company_name(
        self, 
        company_name: str, 
        query: str, 
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 2,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False
    ) -> List[Dict]:
        """
        Retrieve and rerank documents using hybrid approach.
        
        Args:
            company_name: Name of the company to search documents for
            query: Search query
            llm_reranking_sample_size: Number of initial results to retrieve from vector DB
            documents_batch_size: Number of documents to analyze in one LLM prompt
            top_n: Number of final results to return after reranking
            llm_weight: Weight given to LLM scores (0-1)
            return_parent_pages: Whether to return full pages instead of chunks
            
        Returns:
            List of reranked document dictionaries with scores
        """
        # Get initial results from vector retriever
        vector_results = self.vector_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages
        )
        
        # Rerank results using LLM
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )
        
        return reranked_results[:top_n]
