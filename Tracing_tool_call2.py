import mlflow
import litellm
import openai
import os

base_url = "http://localhost:4000/v1"
api_key = os.getenv("OPENAI_API_KEY")
# Enable MLflow auto-tracing for LiteLLM
mlflow.litellm.autolog()

mlflow.set_experiment("litellm")

# Define the tool function.
def get_weather(location: str) -> str:
    if location == "Tokyo":
        return "sunny"
    elif location == "Paris":
        return "rainy"
    return "unknown"

# Define function spec
get_weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "properties": {
                "location": {
                    "description": "The city and state, e.g., San Francisco, CA",
                    "type": "string",
                },
            },
            "required": ["location"],
            "type": "object",
        },
    },
}
model = "lamma32"
client = openai.Client(api_key=api_key, base_url=base_url)
# Call LiteLLM as usual
# Chamada ao LiteLLM apontando para seu endpoint local
with mlflow.start_run():
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What's the weather like in Paris today?"}],
        tools=[get_weather_tool]  # Define o endpoint da API LiteLLM
    )

    # Registrar logs no MLflow
    mlflow.log_param("model", "ollama/llama3.2:1b")
    mlflow.log_param("user_message", "What's the weather like in Paris today?")

print(response)