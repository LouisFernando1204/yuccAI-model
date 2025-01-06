from fastapi import FastAPI, HTTPException
from app.services.populate_database import populate_database_service
from app.services.query_data import query_data_service
from app.model.request_models import QueryRequest, PopulateRequest
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="yuccAI Model API",
    description="API for querying the custom-trained **yuccAI** model, which utilizes embedded data for context-based responses. This API allows users to perform text-based searches, receive responses from the model based on relevant sources, and supports questions in Bahasa Indonesia."
)

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the yuccAI Model API!"}

@app.post("/populate-database")
def populate_database(populate: PopulateRequest):
    """
    Endpoint to populate the database with embeddings from PDF documents.
    """
    try:
        message = populate_database_service(populate.reset)
        return {"status": "success", "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_data(query: QueryRequest):
    """
    Endpoint to query the database and get a response from the model.
    """
    try:
        response = query_data_service(query.query_text)
        return {"status": "success", "response": response["answer"], "source": response["source"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))