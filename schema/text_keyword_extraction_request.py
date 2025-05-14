from pydantic import BaseModel

class text_keyword_extraction_request(BaseModel):
    text: str
    size: int