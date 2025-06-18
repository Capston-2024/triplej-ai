from pydantic import BaseModel

class prediction_request(BaseModel):
    applicant_id : int