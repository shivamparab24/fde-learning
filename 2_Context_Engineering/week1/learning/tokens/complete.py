import tiktoken
from week_1.learning.tokens.data import history, retrieved, system, user

MODEL = "gpt-4o"
WINDOW = 128000
OUTPUT_RESERVE = 4096

encoding = tiktoken.encoding_for_model(MODEL)


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


system = system
history = history
retrieved = retrieved
user = user

used = (
    count_tokens(system)
    + count_tokens(history)
    + count_tokens(retrieved)
    + count_tokens(user)
)

# print(count_tokens(system))
# print(count_tokens(history))
# print(count_tokens(retrieved))
# print(count_tokens(user))

available = WINDOW - OUTPUT_RESERVE

print(f"Used: {used}")
print(f"Available: {available}")

if used <= available:
    print("✅ Safe to send")
else:
    print("❌ Trim context before sending")