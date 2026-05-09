from langchain_community.utilities import ArxivAPIWrapper
arxiv = ArxivAPIWrapper()
print(f"Default top_k_results: {arxiv.top_k_results}")
