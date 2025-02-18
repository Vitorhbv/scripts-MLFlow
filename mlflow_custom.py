import os
import openai
import pandas as pd
import mlflow
from langchain_community.llms import Ollama


#pip install textstat for text statistics
#mlflow.set_registry_uri("s3://test-bucket/")
base_url = "http://localhost:4000/v1"
api_key = os.getenv("OPENAI_API_KEY")
name_model = "lamma32"
client = openai.Client(api_key=api_key, base_url=base_url)
mlflow.set_experiment("mlflow_metric_genai")

eval_data = pd.DataFrame(
    {
        "inputs": [
            "Soltei os cachorros!",
            "Amigo da onça"
        ]
    }
)

def openai_qa(inputs):
    answers = []
    system_prompt="Traduza a sentença a seguir para o português brasileiro."
    for index, row in inputs.iterrows():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["inputs"]},
        ]
        response = client.chat.completions.create(
            model=name_model,
            messages=messages,
            temperature=0,
        )
        answers.append(response.choices[0].message.content)
    return answers

# Define the custom metric
cultural_sensitivity = mlflow.metrics.genai.make_genai_metric(
    name="cultural_sensitivity",
    definition="Compara o quão bom a tradução preserva o significado cultural em outro idioma",
    grading_prompt="Pontuação de 1 a 5, onde 1 é cuturalmente fraco e 5 é cuturalmente ciente.",
    examples=[
        mlflow.metrics.genai.EvaluationExample(
            input="Chutar o balde!",
            output="kick the bucket!",
            score=2,
            justification="É a tradução literal e não preserva o significado cultural."
        ),
        mlflow.metrics.genai.EvaluationExample(
            input="Chutar o balde!",
            output="Give up",
            score=5,
            justification="É a tradução cuturalmente correta."
        )
    ],

)

with mlflow.start_run() as run:
    results = mlflow.evaluate(
        openai_qa,
        eval_data,
        model_type="question-answering",  # model type indicates which metrics are relevant for this task
        evaluators="default",
        extra_metrics=[cultural_sensitivity],  # include the answer_similarity metric
    )

print(results.tables["eval_results_table"])