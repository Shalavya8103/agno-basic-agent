from fastapi import FastAPI
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.storage.agent.postgres import PostgresAgentStorage

app = FastAPI()

db_url = "postgresql+psycopg://agno:agno@db/agno"

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(table_name="recipes", db_url=db_url),
)
knowledge_base.load(recreate=True)

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You are a Thai cuisine expert!",
    knowledge=knowledge_base,
    storage=PostgresAgentStorage(table_name="agent_sessions", db_url=db_url),
    markdown=True,
)

@app.get("/ask")
async def ask(query: str):
    response = agent.run(query)
    return {"response": response.content}
