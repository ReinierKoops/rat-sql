from pydantic import BaseModel

# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.middleware.cors import CORSMiddleware

from flask import Flask
from flask_cors import CORS
from model import ModelSingleton


model = ModelSingleton()
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


class ModelResponse(BaseModel):
    query: str
    attention_map: list


# @app.options('/infer/{db_id}/{question}', methods=['OPTIONS'])
# def options(db_id: str, question: str):
#     return None


@app.route('/infer/{db_id}/{question}', methods=['GET'])
def infer(db_id: str, question: str):
    # try:
    query_output, attention_map = model.infer(question, db_id)
    return ModelResponse(query=query_output, attention_map=attention_map)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# class ModelResponse(BaseModel):
#     query: str
#     attention_map: list


# @app.options("/infer/{db_id}/{question}")
# def options(db_id: str, question: str):
#     return None


# @app.get("/infer/{db_id}/{question}")
# def infer(db_id: str, question: str, model: Model = Depends(get_model)):
#     # try:
#     query_output, attention_map = model.infer(question, db_id)
#     return ModelResponse(query=query_output, attention_map=attention_map)
#     # except:
#     #     raise HTTPException(
#     #         status_code=404, detail="The model couldn't process the question.")
