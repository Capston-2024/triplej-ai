from fastapi import FastAPI
from pickin_api import router as pickin_router

app = FastAPI()
app.include_router(pickin_router)
