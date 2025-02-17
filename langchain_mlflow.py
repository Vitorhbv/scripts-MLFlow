import os
from operator import itemgetter

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

import mlflow

mlflow.set_experiment("langchain_mlflow")
# Enable mlflow langchain autologging
mlflow.langchain.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    registered_model_name="lc_model",
)

prompt_with_history_str = """
Here is a history between you and a human: {chat_history}

Now, please answer this question: {question}
"""
prompt_with_history = PromptTemplate(
    input_variables=["chat_history", "question"], template=prompt_with_history_str
)

def extract_question(input):
    return input[-1]["content"]

def extract_history(input):
    return input[:-1]

# Use Ollama instead of OpenAI
llm = Ollama(model="llama3.2:1b", temperature=0.6)

# Build a chain with LCEL
chain_with_history = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | prompt_with_history
    | llm
    | StrOutputParser()
)

inputs = {"messages": [{"role": "user", "content": "Who owns MLflow?"}]}

print(chain_with_history.invoke(inputs))

# We automatically log the model and trace related artifacts
# A model with name `lc_model` is registered, we can load it back as a PyFunc model
# model_name = "lc_model"
# model_version = 1
# loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
# print(loaded_model.predict(inputs))
mlflow.log_params({"model_type": "ollama-llm", "model_name": "llama3.2:1b"})
