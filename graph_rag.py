import os 
import dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG as Neo4jGraphRAG
from neo4j_graphrag.indexes import create_vector_index

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

load_status = dotenv.load_dotenv("neo4j_cred.txt")
if load_status is False:
    raise RuntimeError('Environment variables not loaded.')

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

class GraphRag:
    def __init__(self):
        # Connect to Neo4j database
        driver = GraphDatabase.driver(URI, auth=AUTH)
        self.retriever = self.get_retriever(driver)
        self.llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})
        self.rag = Neo4jGraphRAG(retriever=self.retriever, llm=self.llm)

    def get_retriever(self, driver):
        embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
        retriever = VectorRetriever(
            driver,
            index_name="vector_index",
            embedder=embedder
        )
        return retriever

    def retrieve(self, query:str, top_k:int=5):
        return self.retriever.search(query_text=query, top_k=3)
    
    def get_answer(self, query:str, top_k:int=5):
        response = self.rag.search(query_text=query, retriever_config={"top_k": top_k})
        return response

def create_vector_index():
    """
    This creates a vector retriever, i.e able to retrieve top k chunks
    """

    INDEX_NAME = "vector_index"
    DIMENSION=1536

    # Connect to Neo4j database
    driver = GraphDatabase.driver(URI, auth=AUTH)

    # Creating the index
    create_vector_index(
        driver,
        INDEX_NAME,
        label="Document",
        embedding_property="vectorProperty",
        dimensions=DIMENSION,
        similarity_fn="euclidean",
    )

if __name__ == "__main__":
    
    graphrag = GraphRag()
    query = "What is done in lhcb experiment?"
    response = graphrag.get_answer(query)
    print(response)