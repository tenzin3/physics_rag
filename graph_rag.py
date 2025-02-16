import os 
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class GraphRag:
    def __init__(self):
        # Reference
        # https://neo4j.com/developer-blog/get-started-graphrag-python-package/
        URI = "neo4j+s://demo.neo4jlabs.com"
        AUTH = ("recommendations", "recommendations")
        # Connect to Neo4j database
        driver = GraphDatabase.driver(URI, auth=AUTH)
        self.retriever = self.get_retriever(driver)
        self.llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})
        self.rag = GraphRAG(retriever=self.retriever, llm=self.llm)

    def get_retriever(self, driver):
        embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
        retriever = VectorRetriever(
            driver,
            index_name="moviePlotsEmbedding",
            embedder=embedder,
            return_properties=["title", "plot"],
        )
        return retriever

    def retrieve(self, query:str, top_k:int=5):
        return self.retriever.search(query_text=query, top_k=3)
    
    def get_answer(self, query:str, top_k:int=5):
        response = self.rag.search(query_text=query, retriever_config={"top_k": top_k})
        return response

if __name__ == "__main__":
    graph_rag = GraphRag()
    answer = graph_rag.get_answer("What movies are sad romances?", top_k=3)
    print(answer)