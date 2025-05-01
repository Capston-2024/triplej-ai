from pydantic import BaseModel

class prediction_request(BaseModel):
    nationality: str
    education: str
    topik: str
    work: str