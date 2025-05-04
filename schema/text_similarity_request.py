from pydantic import BaseModel

class text_similarity_request(BaseModel):
    job: str
    letter: str