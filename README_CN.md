# 📚 企业知识库 RAG 智能问答系统

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![RAG](https://img.shields.io/badge/RAG-检索增强生成-FF6B6B?style=flat-square)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)

**基于检索增强生成（RAG）技术构建的企业年报智能问答系统**

[在线演示](#-在线演示) • [快速开始](#-快速开始) • [项目文档](#-项目文档) • [技术架构](#-技术架构)

</div>

---

## 🌟 项目简介

这是一个**企业级RAG（检索增强生成）智能问答系统**，专门用于处理企业年报PDF文档并回答相关问题。系统实现了完整的RAG流程，从PDF解析到向量检索，再到LLM生成答案，支持多公司、多数据库架构。

### 核心特性

- 📄 **智能PDF解析**：使用Docling进行结构化解析，保留表格、文本、元数据
- 🔍 **混合检索策略**：向量检索 + BM25 + LLM重排序，精准定位信息
- 🏗️ **多数据库架构**：每家公司独立向量数据库，智能路由
- 💬 **结构化输出**：CoT推理 + JSON Schema，自动引用页码验证
- 🌐 **Web交互界面**：基于Streamlit的现代化UI，实时问答

---

## 🚀 在线演示

> 💡 **简历展示链接**：[点击访问在线演示](https://你的应用名.streamlit.app)

### 演示功能

1. **初始化系统**：选择数据集，配置检索参数
2. **智能问答**：输入问题，系统自动检索并生成答案
3. **查看详情**：推理过程、引用页码、相关文档

---

## 🏗️ 技术架构

```
用户问题
   ↓
问题处理与路由（公司名提取、问题类型识别）
   ↓
混合检索（向量检索 + BM25 + LLM重排序）
   ↓
向量数据库（FAISS，多公司独立数据库）
   ↓
LLM生成答案（结构化输出 + CoT推理）
   ↓
返回答案（JSON + 引用页码）
```

### 技术栈

- **后端**：Python 3.8+, FAISS, LangChain, Docling
- **前端**：Streamlit
- **AI/ML**：DashScope/OpenAI, 向量检索, LLM重排序
- **数据处理**：Pandas, PyPDF2, Tiktoken

---

## ✨ 核心功能

### 1. PDF智能解析
- 使用Docling工具进行PDF结构化解析
- 保留表格、文本、页码等元数据
- 支持并行处理，提高效率

### 2. 混合检索策略
- **向量检索**：基于语义相似度的FAISS检索
- **BM25检索**：传统关键词检索
- **LLM重排序**：使用大模型对检索结果精确排序
- **父文档检索**：先检索摘要，再获取完整上下文

### 3. 多数据库架构
- 每家企业独立向量数据库
- 智能路由到对应数据库
- 支持多公司对比查询

### 4. 结构化输出
- Chain-of-Thought（CoT）分步推理
- JSON Schema结构化输出
- 自动引用页码验证
- 支持多种问题类型（布尔、数字、文本、列表）

---

## 🚀 快速开始

### 方式一：在线访问（推荐）

直接访问 [在线演示链接](https://你的应用名.streamlit.app)，无需安装即可体验。

### 方式二：本地运行

```bash
# 1. 克隆项目
git clone https://github.com/你的用户名/你的仓库.git
cd RAG-Challenge-2-main

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
# 创建 .env 文件，添加：
# DASHSCOPE_API_KEY=your_api_key_here

# 4. 运行Web界面
streamlit run app.py
```

### 使用步骤

1. **初始化系统**
   - 在侧边栏选择数据集（test_set 或 erc2_set）
   - 配置检索参数（Top N、Reranking等）
   - 点击"初始化系统"

2. **提问**
   - 输入问题（需包含公司名，用引号括起来）
   - 例如：`"Mercia Asset Management PLC"年报中是否提到了并购？`
   - 选择问题类型（string/boolean/number/names）
   - 点击"提交问题"

3. **查看结果**
   - 🎯 最终答案
   - 📊 推理摘要
   - 🔍 详细分析过程
   - 📄 相关页码
   - 📚 文档引用

---

## 📊 项目数据

- **支持文档类型**：PDF年报（多公司）
- **向量数据库**：FAISS索引
- **检索精度**：Top-10 + LLM重排序
- **响应时间**：平均3-5秒/问题
- **支持问题类型**：布尔、数字、文本、列表

---

## 🎯 项目亮点

1. **多策略检索融合**：向量检索 + BM25 + LLM重排序
2. **智能分块策略**：保留表格完整性，基于章节语义切分
3. **结构化输出设计**：CoT推理 + 自动页码引用验证
4. **可扩展架构**：多数据库路由 + 并行处理支持

---

## 📝 项目文档

- [系统架构梳理](RAG系统架构梳理.md) - 详细的技术架构说明
- [项目部署指南](项目部署指南.md) - 如何部署到线上
- [Streamlit界面使用说明](Streamlit界面使用说明.md) - 界面使用指南
- [项目展示文档](PROJECT_SHOWCASE.md) - 简历展示专用文档

---

## 🔧 环境配置

### 必需的环境变量

创建 `.env` 文件：

```bash
DASHSCOPE_API_KEY=your_api_key_here
# 可选
OPENAI_API_KEY=your_openai_key_here
```

### 依赖安装

```bash
pip install -r requirements.txt
```

### 模型下载

```bash
python main.py download-models
```

---

## 📈 项目价值

### 技术价值
- ✅ 完整的RAG系统实现
- ✅ 多策略检索融合
- ✅ LLM应用开发实践
- ✅ 向量数据库使用
- ✅ Web应用开发

### 业务价值
- 📄 企业年报智能问答
- 🔍 高效信息检索
- 💡 知识库构建
- 📊 数据分析支持

---

## 🎓 适合场景

- 📚 企业知识库问答
- 📄 文档智能检索
- 💼 财务报告分析
- 🔍 信息提取与问答

---

## 📞 联系方式

如有问题或建议，欢迎：
- 📧 发送Issue
- 💬 提交Pull Request
- 📝 查看详细文档

---

## 📄 License

MIT License

---

<div align="center">

**💡 这是一个简历展示项目，展示了RAG系统的完整实现和最佳实践**

Made with ❤️ by [你的名字]

</div>

