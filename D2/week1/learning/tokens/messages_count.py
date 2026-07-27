from week_1.learning.tokens.token_count import count_tokens


messages = [
    {
        "role": "system",
        "content": "You are helpful."
    },
    {
        "role": "user",
        "content": "Explain AI."
    }
]





def count_message_tokens(messages):

    total = 0

    for message in messages:
        total += count_tokens(message["content"])

    return total

print(count_message_tokens(messages))