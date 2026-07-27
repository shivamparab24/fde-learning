import json
from typing import Mapping, Sequence

from week_1.learning.tokens.token_count import count_tokens


WINDOW = 12800
OUTPUT_RESERVE  = 4096

INPUT_BUDGET = WINDOW - OUTPUT_RESERVE 

messages = [
    {
        "role": "system",
        "content": "You are a weather assistant."
    },
    {
        "role": "user",
        "content": "What's the weather in Pune?"
    },
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Pune"}'
                }
            }
        ]
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"temperature": 28, "condition": "Sunny"}'
    },
    {
        "role": "assistant",
        "content": "The current weather in Pune is sunny with a temperature of 28°C."
    }
]


# def count_message_tokens(messages):
#     total = 0

#     for message in messages:
#         # role
#         total += count_tokens(message.get("role", ""))

#         # content
#         total += count_tokens(message.get("content", ""))

#         # tool output id
#         total += count_tokens(message.get("tool_call_id", ""))

#         # assistant tool calls
#         if "tool_calls" in message:
#             total += count_tokens(json.dumps(message["tool_calls"]))

#     return total

# tokens = count_message_tokens(messages) # 79


def count_object_tokens(obj):
    """
    Recursively count tokens for every string contained
    in dictionaries, lists, tuples, etc.
    """

    if obj is None:
        return 0
    
    if isinstance(obj, (str, int, float, bool)):
        return count_tokens(obj)

    if isinstance(obj, Mapping):
        return sum(count_object_tokens(v) for v in obj.values())

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return sum(count_object_tokens(item) for item in obj)

    return 0
    

tokens = count_object_tokens(messages) # 59

if tokens > INPUT_BUDGET:
    print("Token limit crossed: ", tokens)
else:
    print("Tokens within limit: ", tokens)