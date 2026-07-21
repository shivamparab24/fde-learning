system = """
You are an expert AI tutor.

Rules:
- Explain concepts from first principles.
- Use simple language first, then introduce technical details.
- Give Python examples when appropriate.
- If the user asks about mathematics, derive the equations step by step.
- Never assume prior knowledge unless the user explicitly states it.
"""

history = """
User:
I'm trying to understand Neural Networks from scratch.

Assistant:
Great! Let's begin with the Perceptron. A perceptron computes:

y = f(w·x + b)

where
- w = weights
- x = inputs
- b = bias

User:
I understand the perceptron. Why do we need activation functions?

Assistant:
Without activation functions, stacking multiple linear layers is equivalent to a single linear transformation. Activation functions introduce non-linearity, allowing neural networks to model complex relationships.

User:
Can you explain ReLU?

Assistant:
ReLU(x) = max(0, x)

It outputs zero for negative inputs and the input itself for positive values.
"""


retrieved = """
Chapter: Attention Mechanism

Attention allows a neural network to focus on the most relevant parts of the input sequence.

Instead of compressing an entire sentence into one fixed vector, attention computes a weighted average of all input vectors.

Given:

Query (Q)
Key (K)
Value (V)

The attention score is computed as:

score = Q × Kᵀ

These scores are normalized using Softmax:

Attention(Q,K,V) = Softmax(QKᵀ / √dₖ) V

Advantages:
- Captures long-range dependencies
- Enables parallel computation
- Forms the basis of the Transformer architecture
"""


user = """
Can you explain why we divide by √dₖ in the attention formula?
"""