# def drop_truncate(text: str, max_chars: int, side: str = "right") -> str:
#     """Drops text exceeding the character budget."""
#     if len(text) <= max_chars:
#         return text
        
#     if side == "right":
#         return text[:max_chars]
#     elif side == "left":
#         return text[-max_chars:]
#     elif side == "center":
#         half = max_chars // 2
#         return text[:half] + "...[TRUNCATED]..." + text[-half:]
#     else:
#         raise ValueError("Side must be 'left', 'right', or 'center'")


def implement_drop_strategy(data_list: list, max_budget: int) -> list:
    """ Keeps only the newest items that fit within the numeric budget.
    
    Drops older items entirely.
    """
    # If we are already under budget, change nothing
    if len(data_list) <= max_budget:
        return data_list
        
    # Drop the oldest items, keeping only the 'max_budget' most recent items
    truncated_data = data_list[-max_budget:]
    return truncated_data

# --- Execution Example ---
chat_history = [
    "Message 1: User logged in",
    "Message 2: User opened settings",
    "Message 3: User changed theme to dark",
    "Message 4: User clicked upgrade button"
]
BUDGET = 2

result = implement_drop_strategy(chat_history, BUDGET)
print("--- DROP STRATEGY RESULT ---")
print(result)
# Output: ['Message 3: User changed theme to dark', 'Message 4: User clicked upgrade button']