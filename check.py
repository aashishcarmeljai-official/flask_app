import openai

# If using API key
openai.api_key = "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"

# List available models
models = openai.models.list()

for model in models.data:
    print(model.id)