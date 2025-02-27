from pydantic import Field

from chatchat.settings import Settings
from chatchat.server.agent.tools_factory.tools_registry import (
    BaseToolOutput,
    regist_tool,
)
from chatchat.server.knowledge_base.kb_api import list_kbs
from chatchat.server.knowledge_base.kb_doc_api import search_docs

template = (
    "Use local knowledgebase from one or more of these:\n{KB_info}\n to get information，Only local data on "
    "this knowledge use this tool. The 'database' should be one of the above [{key}]."
)
KB_info_str = "\n".join([f"{key}: {value}" for key, value in Settings.kb_settings.KB_INFO.items()])
template_knowledge = template.format(KB_info=KB_info_str, key="samples")


# todo: 将配置中 search_knowledgebase 的相关配置文件干掉.
def search_knowledgebase(query: str, database: str, top_k: int, score_threshold: float):
    docs = search_docs(
        query=query,
        knowledge_base_name=database,
        top_k=top_k,
        score_threshold=score_threshold,
        file_name="",
        metadata={},
    )
    return {"knowledge_base": database, "docs": docs}


@regist_tool(description=template_knowledge, title="本地知识库")
def search_local_knowledgebase(
    database: str = Field(
        description="Database for Knowledge Search",
        choices=[kb.kb_name for kb in list_kbs().data],
    ),
    query: str = Field(description="Query for Knowledge Search"),
    top_k: int = Field(description="Top K for Knowledge Search"),
    score_threshold: float = Field(description="Score threshold for Knowledge Search")
):
    """temp docstr to avoid langchain error"""
    result = search_knowledgebase(
        query=query,
        database=database,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    # return BaseToolOutput(result, format=format_context)
    return BaseToolOutput(result)
