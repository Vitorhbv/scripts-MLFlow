import mlflow
import litellm

mlflow.litellm.autolog()
mlflow.set_experiment("litellm_test")
# Call Anthropic API via LiteLLM
litellm._turn_on_debug()

response = litellm.completion(
    model="ollama/llama3.2:1b",
    messages=[{"role": "user", "content": "Hey! how's it going?"}],
    api_base="http://localhost:11434"
)
