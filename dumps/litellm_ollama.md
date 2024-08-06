Ollama
LiteLLM supports all models from Ollama

Open In Colab
INFO
We recommend using ollama_chat for better responses.

Pre-requisites
Ensure you have your ollama server running

Example usage
from litellm import completion

response = completion(
    model="ollama/llama2", 
    messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
    api_base="http://localhost:11434"
)
print(response)


Example usage - Streaming
from litellm import completion

response = completion(
    model="ollama/llama2", 
    messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
    api_base="http://localhost:11434",
    stream=True
)
print(response)
for chunk in response:
    print(chunk['choices'][0]['delta'])


Example usage - Streaming + Acompletion
Ensure you have async_generator installed for using ollama acompletion with streaming

pip install async_generator

async def async_ollama():
    response = await litellm.acompletion(
        model="ollama/llama2", 
        messages=[{ "content": "what's the weather" ,"role": "user"}], 
        api_base="http://localhost:11434", 
        stream=True
    )
    async for chunk in response:
        print(chunk)

# call async_ollama
import asyncio
asyncio.run(async_ollama())


Example Usage - JSON Mode
To use ollama JSON Mode pass format="json" to litellm.completion()

from litellm import completion
response = completion(
  model="ollama/llama2",
  messages=[
      {
          "role": "user",
          "content": "respond in json, what's the weather"
      }
  ],
  max_tokens=10,
  format = "json"
)

Using ollama api/chat
In order to send ollama requests to POST /api/chat on your ollama server, set the model prefix to ollama_chat

from litellm import completion

response = completion(
    model="ollama_chat/llama2", 
    messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
)
print(response)

Ollama Models
Ollama supported models: https://github.com/ollama/ollama

Model Name	Function Call
Mistral	completion(model='ollama/mistral', messages, api_base="http://localhost:11434", stream=True)
Mistral-7B-Instruct-v0.1	completion(model='ollama/mistral-7B-Instruct-v0.1', messages, api_base="http://localhost:11434", stream=False)
Mistral-7B-Instruct-v0.2	completion(model='ollama/mistral-7B-Instruct-v0.2', messages, api_base="http://localhost:11434", stream=False)
Mixtral-8x7B-Instruct-v0.1	completion(model='ollama/mistral-8x7B-Instruct-v0.1', messages, api_base="http://localhost:11434", stream=False)
Mixtral-8x22B-Instruct-v0.1	completion(model='ollama/mixtral-8x22B-Instruct-v0.1', messages, api_base="http://localhost:11434", stream=False)
Llama2 7B	completion(model='ollama/llama2', messages, api_base="http://localhost:11434", stream=True)
Llama2 13B	completion(model='ollama/llama2:13b', messages, api_base="http://localhost:11434", stream=True)
Llama2 70B	completion(model='ollama/llama2:70b', messages, api_base="http://localhost:11434", stream=True)
Llama2 Uncensored	completion(model='ollama/llama2-uncensored', messages, api_base="http://localhost:11434", stream=True)
Code Llama	completion(model='ollama/codellama', messages, api_base="http://localhost:11434", stream=True)
Llama2 Uncensored	completion(model='ollama/llama2-uncensored', messages, api_base="http://localhost:11434", stream=True)
Meta LLaMa3 8B	completion(model='ollama/llama3', messages, api_base="http://localhost:11434", stream=False)
Meta LLaMa3 70B	completion(model='ollama/llama3:70b', messages, api_base="http://localhost:11434", stream=False)
Orca Mini	completion(model='ollama/orca-mini', messages, api_base="http://localhost:11434", stream=True)
Vicuna	completion(model='ollama/vicuna', messages, api_base="http://localhost:11434", stream=True)
Nous-Hermes	completion(model='ollama/nous-hermes', messages, api_base="http://localhost:11434", stream=True)
Nous-Hermes 13B	completion(model='ollama/nous-hermes:13b', messages, api_base="http://localhost:11434", stream=True)
Wizard Vicuna Uncensored	completion(model='ollama/wizard-vicuna', messages, api_base="http://localhost:11434", stream=True)
Ollama Vision Models
Model Name	Function Call
llava	completion('ollama/llava', messages)
Using Ollama Vision Models
Call ollama/llava in the same input/output format as OpenAI gpt-4-vision

LiteLLM Supports the following image types passed in url

Base64 encoded svgs
Example Request

import litellm

response = litellm.completion(
  model = "ollama/llava",
  messages=[
      {
          "role": "user",
          "content": [
                          {
                              "type": "text",
                              "text": "Whats in this image?"
                          },
                          {
                              "type": "image_url",
                              "image_url": {
                              "url": "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"
                              }
                          }
                      ]
      }
  ],
)
print(response)


LiteLLM/Ollama Docker Image
For Ollama LiteLLM Provides a Docker Image for an OpenAI API compatible server for local LLMs - llama2, mistral, codellama

Chat on WhatsApp Chat on Discord

An OpenAI API compatible server for local LLMs - llama2, mistral, codellama
Quick Start:
Docker Hub: For ARM Processors: https://hub.docker.com/repository/docker/litellm/ollama/general For Intel/AMD Processors: to be added

docker pull litellm/ollama

docker run --name ollama litellm/ollama

Test the server container
On the docker container run the test.py file using python3 test.py

Making a request to this server
import openai

api_base = f"http://0.0.0.0:4000" # base url for server

openai.api_base = api_base
openai.api_key = "temp-key"
print(openai.api_base)


print(f'LiteLLM: response from proxy with streaming')
response = openai.chat.completions.create(
    model="ollama/llama2", 
    messages = [
        {
            "role": "user",
            "content": "this is a test request, acknowledge that you got it"
        }
    ],
    stream=True
)

for chunk in response:
    print(f'LiteLLM: streaming response from proxy {chunk}')

Responses from this server
{
  "object": "chat.completion",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": " Hello! I acknowledge receipt of your test request. Please let me know if there's anything else I can assist you with.",
        "role": "assistant",
        "logprobs": null
      }
    }
  ],
  "id": "chatcmpl-403d5a85-2631-4233-92cb-01e6dffc3c39",
  "created": 1696992706.619709,
  "model": "ollama/llama2",
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 25,
    "total_tokens": 43
  }
}


----
"completion" is not accessedPylance
(function) def completion(
    model: str,
    messages: List = [],
    timeout: float | str | Timeout | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    n: int | None = None,
    stream: bool | None = None,
    stream_options: dict | None = None,
    stop: Any | None = None,
    max_tokens: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logit_bias: dict | None = None,
    user: str | None = None,
    response_format: dict | None = None,
    seed: int | None = None,
    tools: List | None = None,
    tool_choice: str | dict | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    deployment_id: Any | None = None,
    extra_headers: dict | None = None,
    functions: List | None = None,
    function_call: str | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    model_list: list | None = None,
    **kwargs: Any
) -> (ModelResponse | CustomStreamWrapper)
Perform a completion() using any of litellm supported llms (example gpt-4, gpt-3.5-turbo, claude-2, command-nightly) Parameters:
    model (str): The name of the language model to use for text completion. see all supported LLMs: https://docs.litellm.ai/docs/providers/
    messages (List): A list of message objects representing the conversation context (default is an empty list).

    OPTIONAL PARAMS
    functions (List, optional): A list of functions to apply to the conversation messages (default is an empty list).
    function_call (str, optional): The name of the function to call within the conversation (default is an empty string).
    temperature (float, optional): The temperature parameter for controlling the randomness of the output (default is 1.0).
    top_p (float, optional): The top-p parameter for nucleus sampling (default is 1.0).
    n (int, optional): The number of completions to generate (default is 1).
    stream (bool, optional): If True, return a streaming response (default is False).
    stream_options (dict, optional): A dictionary containing options for the streaming response. Only set this when you set stream: true. stop(string/list, optional): - Up to 4 sequences where the LLM API will stop generating further tokens.
    max_tokens (integer, optional): The maximum number of tokens in the generated completion (default is infinity).
    presence_penalty (float, optional): It is used to penalize new tokens based on their existence in the text so far.
    frequency_penalty: It is used to penalize new tokens based on their frequency in the text so far.
    logit_bias (dict, optional): Used to modify the probability of specific tokens appearing in the completion.
    user (str, optional): A unique identifier representing your end-user. This can help the LLM provider to monitor and detect abuse.
    logprobs (bool, optional): Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message
    top_logprobs (int, optional): An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.
    metadata (dict, optional): Pass in additional metadata to tag your completion calls - eg. prompt version, details, etc.
    api_base (str, optional): Base URL for the API (default is None).
    api_version (str, optional): API version (default is None).
    api_key (str, optional): API key (default is None).
    model_list (list, optional): List of api base, version, keys
    extra_headers (dict, optional): Additional headers to include in the request.

    LITELLM Specific Params
    mock_response (str, optional): If provided, return a mock completion response for testing or debugging purposes (default is None).
    custom_llm_provider (str, optional): Used for Non-OpenAI LLMs, Example usage for bedrock, set model="amazon.titan-tg1-large" and custom_llm_provider="bedrock"
    max_retries (int, optional): The number of retries to attempt (default is 0).
Returns:
    ModelResponse: A response object containing the generated completion and associated metadata.

Note:

This function is used to perform completions() using the specified language model.
It supports various optional parameters for customizing the completion behavior.
If 'mock_response' is provided, a mock completion response is returned for testing or debugging


---
Output
Format
Here's the exact json output and type you can expect from all litellm completion calls for all models

{
  'choices': [
    {
      'finish_reason': str,     # String: 'stop'
      'index': int,             # Integer: 0
      'message': {              # Dictionary [str, str]
        'role': str,            # String: 'assistant'
        'content': str          # String: "default message"
      }
    }
  ],
  'created': str,               # String: None
  'model': str,                 # String: None
  'usage': {                    # Dictionary [str, int]
    'prompt_tokens': int,       # Integer
    'completion_tokens': int,   # Integer
    'total_tokens': int         # Integer
  }
}


You can access the response as a dictionary or as a class object, just as OpenAI allows you

print(response.choices[0].message.content)
print(response['choices'][0]['message']['content'])

Here's what an example response looks like

{
  'choices': [
     {
        'finish_reason': 'stop',
        'index': 0,
        'message': {
           'role': 'assistant',
            'content': " I'm doing well, thank you for asking. I am Claude, an AI assistant created by Anthropic."
        }
      }
    ],
 'created': 1691429984.3852863,
 'model': 'claude-instant-1',
 'usage': {'prompt_tokens': 18, 'completion_tokens': 23, 'total_tokens': 41}
}


Additional Attributes
You can also access information like latency.

from litellm import completion
import os
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

messages=[{"role": "user", "content": "Hey!"}]

response = completion(model="claude-2", messages=messages)

print(response.response_ms) # 616.25# 616.25

--
Groq
https://groq.com/

We support ALL Groq models, just set groq/ as a prefix when sending completion requests

API Key
# env variable
os.environ['GROQ_API_KEY']

Sample Usage
from litellm import completion
import os

os.environ['GROQ_API_KEY'] = ""
response = completion(
    model="groq/llama2-70b-4096", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)

Sample Usage - Streaming
from litellm import completion
import os

os.environ['GROQ_API_KEY'] = ""
response = completion(
    model="groq/llama2-70b-4096", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)

Supported Models - ALL Groq Models Supported!
We support ALL Groq models, just set groq/ as a prefix when sending completion requests

Model Name	Function Call
llama3-8b-8192	completion(model="groq/llama3-8b-8192", messages)
llama3-70b-8192	completion(model="groq/llama3-70b-8192", messages)
llama2-70b-4096	completion(model="groq/llama2-70b-4096", messages)
mixtral-8x7b-32768	completion(model="groq/mixtral-8x7b-32768", messages)
gemma-7b-it	completion(model="groq/gemma-7b-it", messages)
Groq - Tool / Function Calling Example
# Example dummy function hard coded to return the current weather
import json
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})




# Step 1: send the conversation and available functions to the model
messages = [
    {
        "role": "system",
        "content": "You are a function calling LLM that uses the data extracted from get_current_weather to answer questions about the weather in San Francisco.",
    },
    {
        "role": "user",
        "content": "What's the weather like in San Francisco?",
    },
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]
response = litellm.completion(
    model="groq/llama2-70b-4096",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto is default, but we'll be explicit
)
print("Response\n", response)
response_message = response.choices[0].message
tool_calls = response_message.tool_calls


# Step 2: check if the model wanted to call a function
if tool_calls:
    # Step 3: call the function
    # Note: the JSON response may not always be valid; be sure to handle errors
    available_functions = {
        "get_current_weather": get_current_weather,
    }
    messages.append(
        response_message
    )  # extend conversation with assistant's reply
    print("Response message\n", response_message)
    # Step 4: send the info for each function call and function response to the model
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
    print(f"messages: {messages}")
    second_response = litellm.completion(
        model="groq/llama2-70b-4096", messages=messages
    )  # get a new response from the model where it can see the function response
    print("second response\n", second_response)


