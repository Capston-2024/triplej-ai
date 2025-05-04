from pydantic import BaseModel

class feedback_request(BaseModel):
    missing_keywords: list[str]
    similarity: float
    job: str
    letter: str