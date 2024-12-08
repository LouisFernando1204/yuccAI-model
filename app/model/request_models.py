from pydantic import BaseModel

class QueryRequest(BaseModel):
    query_text: str

class PopulateResponse(BaseModel):
    status: str
    message: str