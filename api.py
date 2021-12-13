import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model import Model, get_model
from pydantic import BaseModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelResponse(BaseModel):
    sql_query: str
    attention_dict: str


@app.options("/infer/{db_id}/{question}")
def options(db_id: str, question: str):
    return None


@app.get("/infer/{db_id}/{question}")
def infer(db_id: str, question: str, model: Model = Depends(get_model)):
    try:
        query_output, attention_map = model.infer(question, db_id)
        sql_query = query_output[0]["inferred_code"]
        attention_dict = attention_map.to_json(orient='split')
        return ModelResponse(sql_query=sql_query, attention_dict=attention_dict)
    except:
        raise HTTPException(
            status_code=404, detail="The model couldn't process the question.")
