"""
FastAPI Entry Point
"""

from fastapi import FastAPI
from api.routes import router

app = FastAPI(

    title="Hybrid GraphRAG Question Paper Generator",

    version="1.0.0",

    description="Hybrid RAG + Knowledge Graph + Bloom's Taxonomy"

)

app.include_router(router)

@app.get("/")
def home():

    return {

        "message":

        "Hybrid GraphRAG API Running"

    }