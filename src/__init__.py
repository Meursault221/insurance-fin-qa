"""上市保险公司财报问答知识库 —— 核心包。

第一阶段（当前）包含：
    config          路径与公司元数据
    schema          数据结构定义
    textnorm        文本清洗与表格线性化
    sections        章节标题识别
    tables          表格抽取后端（PyMuPDF / pdfplumber）
    pdf_extractor   PDF -> 结构化文档
    chunker         结构化文档 -> 检索单元
    pipeline        单份 / 批量的编排
    cli             命令行入口
"""

__version__ = "0.1.0"
__project_name__ = "基于向量检索与BM25的上市保险公司财报问答知识库"
