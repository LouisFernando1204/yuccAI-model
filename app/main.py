from fastapi import FastAPI, HTTPException
from app.services.populate_database import populate_database_service
from app.services.query_data import query_data_service
from app.model.request_models import QueryRequest, PopulateResponse

app = FastAPI(
    title="yuccAI Model API",
    description="API for querying the custom-trained **yuccAI** model, which utilizes embedded data for context-based responses. This API allows users to perform text-based searches, receive responses from the model based on relevant sources, and supports questions in Bahasa Indonesia."
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the yuccAI Model API!"}

@app.post("/populate-database", response_model=PopulateResponse)
def populate_database(reset: bool = False):
    """
    Endpoint to populate the database with embeddings from PDF documents.
    """
    try:
        message = populate_database_service(reset)
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
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))