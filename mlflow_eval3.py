import os
import openai
import pandas as pd
import mlflow
from mlflow.metrics.genai import EvaluationExample, answer_similarity

# pip install textstat for text statistics
# mlflow.set_registry_uri("s3://test-bucket/")
base_url = "http://localhost:4000/v1"
api_key = os.getenv("OPENAI_API_KEY")
mlflow.set_experiment("mlflow_eval1")

eval_df = pd.DataFrame(
    {
        "inputs": [
            "How does useEffect() work?",
            "What does the static keyword in a function mean?",
            "What does the 'finally' block in Python do?",
            "What is the difference between multiprocessing and multithreading?"
        ],
        "ground_truth": [
            "The useEffect() hook tells React that your component needs to do something after render. React will remember the function you passed (we’ll refer to it as our “effect”), and call it later after performing the DOM updates.",
            "Static members belong to the class, rather than a specific instance. This means that only one instance of a static member exists, even if you create multiple objects of the class, or if you don't create any. It will be shared by all objects.",
            "'Finally' defines a block of code to run when the try... except...else block is final. The finally block will be executed no matter if the try block raises an error or not.",
            "Multithreading refers to the ability of a processor to execute multiple threads concurrently, where each thread runs a process. Whereas multiprocessing refers to the ability of a system to run multiple processors in parallel, where each processor can run one or more threads."
        ]
    }
)

model = "lamma32"
client = openai.Client(api_key=api_key, base_url=base_url)


@mlflow.trace
def openai_qa(inputs):
    answers = []
    system_prompt = "Answer the following question in two sentences"
    for _, row in inputs.iterrows():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["inputs"]},
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        answers.append(response.choices[0].message.content)
    return answers

predictions = openai_qa(eval_df)
eval_df["predictions"] = predictions

examples = [
    EvaluationExample(
        input=row["inputs"],
        output=row["predictions"],
        grading_context={"targets": row["ground_truth"]},
        score=3,
        justification="Autoevaluated by MLflow"
    )
    for _, row in eval_df.iterrows()
]

answer_similarity_metric = answer_similarity(examples=examples)

with mlflow.start_run() as run:
    results = mlflow.evaluate(
        eval_df,
        targets="ground_truth",  # specify which column corresponds to the expected output
        predictions="predictions",  # specify the column containing model outputs
        model_type="question-answering",  # model type indicates which metrics are relevant for this task
        extra_metrics=[answer_similarity_metric],  # include the answer_similarity metric
    )
    
    print("Evaluation Results:", results)