from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from model import Model, get_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelResponse(BaseModel):
    query: str
    attention_map: list


@app.options("/infer/{db_id}/{question}")
def options(db_id: str, question: str):
    return None


@app.get("/infer/{db_id}/{question}")
def infer(db_id: str, question: str, model: Model = Depends(get_model)):
    # try:
    query_output, attention_map = model.infer(question, db_id)
    return ModelResponse(query=query_output, attention_map=attention_map)
    # except:
    #     raise HTTPException(
    #         status_code=404, detail="The model couldn't process the question.")
