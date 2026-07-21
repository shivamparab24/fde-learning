import tiktoken

text = """
Our support organization handles a high volume of inbound customer requests — order status checks, 
refund requests, subscription cancellations, product how-to questions, and basic account/spend 
lookups — through a single queue of human agents, regardless of how simple or repetitive the request is. 
Analysis of ticket volume shows that a large share of this volume (order status, "how do I…" questions, 
and simple account lookups in particular) requires no judgment call: the answer already exists in a policy doc, 
an order record, or a CRM field. Agents still have to open multiple systems, find the record, and type a response for each one.
"""


# encoding1 = tiktoken.encoding_for_model("gpt-4o")
# encoding2 = tiktoken.get_encoding("r50k_base")
# encoding3 = tiktoken.get_encoding("p50k_base")
# encoding4 = tiktoken.get_encoding("cl100k_base")


# tokens = encoding1.encode(text)
# print("\ngpt-4o: ", tokens)
# print("Tokens used: ", len(tokens))


# tokens = encoding2.encode(text)
# print("\nr50k_base: ", tokens)
# print("Tokens used: ", len(tokens))


# tokens = encoding3.encode(text)
# print("\np50k_base: ", tokens)
# print("Tokens used: ", len(tokens))


# tokens = encoding4.encode(text)
# print("\ncl100k_base: ", tokens)
# print("Tokens used: ", len(tokens))






encoding = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

# print(count_tokens(text))