"""
Knowledge Anchor Algorithm Implementation
"""
from federatedscope.register import register_worker

# 注册worker
def call_my_worker(method):
    if method == 'knowledgeanchor':
        from federatedscope.llm.knowledge_anchor.server import KnowledgeAnchorServer
        from federatedscope.llm.knowledge_anchor.client import KnowledgeAnchorClient
        return {
            'server': KnowledgeAnchorServer,
            'client': KnowledgeAnchorClient
        }
    else:
        return None


register_worker('knowledgeanchor', call_my_worker)

