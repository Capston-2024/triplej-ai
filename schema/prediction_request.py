from pydantic import BaseModel

class prediction_request(BaseModel):
    aplicant_id : int