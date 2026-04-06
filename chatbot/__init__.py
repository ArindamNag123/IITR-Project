"""
chatbot — multi-agent LangGraph module for the Smart Retail App.

Public surface:
    from chatbot import retail_graph
    result = retail_graph.invoke({"messages": [HumanMessage(content="...")]})
"""

from chatbot.workflow_graph import retail_graph

__all__ = ["retail_graph"]
