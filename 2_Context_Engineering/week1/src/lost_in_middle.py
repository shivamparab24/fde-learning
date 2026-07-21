from .context_assembly import assemble_context

def generate_lost_in_the_middle_experiment(position: float) -> list[dict]:
    """
    Generates a payload where a needle fact is placed at a relative position (0.0 to 1.0)
    inside a sea of filler text to demonstrate retrieval degradation.
    """
    needle = "The secret activation code for the safe is: BLUE_MAGPIE_99."
    filler = (
        "The standard operating procedure dictates that all inventory forms must be signed in triplicate by a regional manager. "
        "Ensure that the air conditioning units are set to precisely 72 degrees Fahrenheit during weekends to conserve energy. "
        "The logistics team reports that shipping delays may occur if customs paperwork is missing an official government seal. "
        "Quarterly fiscal statements must be submitted no later than the final working day of the month to prevent compliance penalties. "
        "All employee badges must remain clearly visible above the waist while navigating the secure sectors of the main campus. "
        "IT support recommends clearing your browser cache and resetting your local network hardware if connection lag persists. "
        "The annual corporate retreat has been tentatively rescheduled for the first week of October pending budget finalization."
    )
    
    # Create slots of text
    total_slots = 200
    # target_slot = int(position * total_slots)
    target_slot = min(int(position * total_slots), total_slots - 1)
    
    docs = []
    for i in range(total_slots):
        if i == target_slot:
            docs.append(needle)
        else:
            docs.append(f"[ID: {i}] {filler}")
            
    system = "You are a helpful data assistant. Answer questions using only the provided context."
    user = "What is the secret activation code for the safe?"
    
    return assemble_context(
        system_prompt=system,
        user_prompt=user,
        history=[],
        retrieved_docs=docs,
        max_budget=30000
    )

# # Example runs to compare output payload structures:
# payload_beginning = generate_lost_in_the_middle_experiment(position=0.0)
# payload_middle = generate_lost_in_the_middle_experiment(position=0.5)
# payload_end = generate_lost_in_the_middle_experiment(position=1.0)


# def inspect_payload(name: str, payload: list[dict]):
#     print(f"=== {name.upper()} ===")
#     print(f"Total messages in payload: {len(payload)}")
    
#     # Typically, context_assembly returns a list of message dicts: 
#     # [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}]
#     for msg in payload:
#         role = msg.get("role")
#         content = msg.get("content", "")
        
#         print(f"\n[Role: {role}]")
        
#         # If the context is inside the user or system prompt, let's find our needle
#         if "BLUE_MAGPIE_99" in content:
#             print("-> NEEDLE FOUND IN THIS MESSAGE <-")
            
#             # Let's see where it is by looking at surrounding lines
#             lines = content.split("\n")
#             for idx, line in enumerate(lines):
#                 if "BLUE_MAGPIE_99" in line:
#                     print(f"Line {idx}: {line[:100]}...") 
#         else:
#             # Print a snippet of the message if it doesn't have the needle
#             print(f"NEEDLE NOT FOUND! Content snippet: {content[:150]}...")

#     print("\n" + "="*30 + "\n")

# # Run the inspection
# inspect_payload("Beginning Experiment (position=0.0)", payload_beginning)
# inspect_payload("Middle Experiment (position=0.5)", payload_middle)
# inspect_payload("End Experiment (position=1.0)", payload_end)