import mlflow
import litellm
import os
# Enable MLflow auto-tracing for LiteLLM
mlflow.litellm.autolog()
litellm._turn_on_debug()

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

# Call LiteLLM as usual
# Chamada ao LiteLLM apontando para seu endpoint local
with mlflow.start_run():
    response = litellm.completion(
        api_base="http://localhost:11434",
        messages=[{"role": "user", "content": "What's the weather like in Paris today?"}],
        tools=[get_weather_tool],
        model="ollama/llama3.2:1b",
        api_key= os.getenv("OPENAI_API_KEY"),
        #custom_llm_provider="ollama"# Define o endpoint da API LiteLLM
    )

    # Registrar logs no MLflow
    mlflow.log_param("model", "lamma32")
    mlflow.log_param("user_message", "What's the weather like in Paris today?")
    mlflow.log_metric("tokens", response.usage.total_tokens)
    print(response.choices[0].message.content)

#mlflow.get_tracking_client()._tracking_client.store.flush()

mlflow.end_run()