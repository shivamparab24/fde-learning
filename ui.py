import gradio as gr
import requests
from typing import List, Dict, Any

API_URL = "http://127.0.0.1:8000"

# --- API Helper Functions ---

def fetch_user_sessions(profile_id: str) -> List[str]:
    """Retrieves existing sessions for a given profile_id."""
    if not profile_id.strip():
        return []
    try:
        res = requests.get(f"{API_URL}/sessions/{profile_id.strip()}")
        if res.status_code == 200:
            return res.json().get("sessions", [])
    except Exception as e:
        print(f"Error fetching sessions: {e}")
    return []

def fetch_session_history(profile_id: str, session_id: str) -> List[Dict[str, str]]:
    """Retrieves chat history in Gradio's message format."""
    try:
        res = requests.get(f"{API_URL}/chat/history/{profile_id}/{session_id}")
        if res.status_code == 200:
            return [
                {"role": item["role"], "content": item["content"]}
                for item in res.json().get("history", [])
                if item.get("role") in ["user", "assistant"]
            ]
    except Exception as e:
        print(f"Error loading history: {e}")
    return []

def delete_user_session(profile_id: str, session_id: str) -> str:
    """Deletes a session from the backend database."""
    try:
        res = requests.delete(f"{API_URL}/session/{profile_id}/{session_id}")
        if res.status_code == 200:
            return f"✅ Deleted session '{session_id}'"
        return f"❌ Failed: {res.json().get('detail', 'Unknown error')}"
    except Exception as e:
        return f"⚠️ Error: {e}"

# --- Interface Event Handlers ---

def handle_login(profile_id: str):
    """Logs in user, fetches available sessions, and updates dropdown."""
    profile_id = profile_id.strip()
    if not profile_id:
        return (
            gr.update(visible=True),   # Login Section
            gr.update(visible=False),  # Session Section
            gr.update(visible=False),  # Chat Section
            [],                        # Dropdown choices
            "⚠️ Please enter a Profile ID."
        )
    
    sessions = fetch_user_sessions(profile_id)
    session_choices = sessions if sessions else ["-- No Existing Sessions --"]
    
    return (
        gr.update(visible=False),                             # Hide Login
        gr.update(visible=True),                              # Show Session Panel
        gr.update(visible=False),                             # Hide Chat Frame
        gr.Dropdown(choices=session_choices, value=None),    # Updated Sessions
        f"👤 Logged in as: **{profile_id}**"
    )

def handle_session_selection(session_mode: str, selected_existing: str, new_session_name: str):
    """Processes session choices (New vs Existing) and configures prompt requirements."""
    if session_mode == "Existing Session":
        if not selected_existing or selected_existing == "-- No Existing Sessions --":
            return gr.update(visible=True), gr.update(visible=False), "⚠️ Select a valid existing session."
        
        return (
            gr.update(visible=False), # Hide System Prompt input
            gr.update(visible=True),  # Enable Start Chat Button
            f"Ready to load existing session: `{selected_existing}`"
        )
    else:
        if not new_session_name.strip():
            return gr.update(visible=True), gr.update(visible=False), "⚠️ Please type a new Session ID."
        
        return (
            gr.update(visible=True),  # Show System Prompt for New Session
            gr.update(visible=True),  # Enable Start Chat Button
            f"Ready to create new session: `{new_session_name.strip()}`"
        )

def launch_chat_session(profile_id: str, session_mode: str, selected_existing: str, new_session_name: str, system_prompt: str):
    """Finalizes session loading/creation and opens the Chat window."""
    active_session = selected_existing if session_mode == "Existing Session" else new_session_name.strip()
    
    # Load history if existing session
    history = []
    if session_mode == "Existing Session":
        history = fetch_session_history(profile_id, active_session)
    
    status_msg = f"🟢 Connected | **Profile:** `{profile_id}` | **Session:** `{active_session}`"
    
    return (
        gr.update(visible=False),  # Hide Session Setup
        gr.update(visible=True),   # Show Main Chat Interface
        active_session,            # Store active session ID
        history,                   # Load history into Chatbot
        status_msg                 # Header Status
    )

def chat_step(message: str, history: List[Dict[str, str]], profile_id: str, session_id: str, system_prompt: str, mock_context: str):
    """Invokes backend multi-turn API with token budgeting."""
    if not message.strip():
        return "", history, "{}"

    # Build RAG docs structure if mock context provided
    retrieved_docs = []
    if mock_context.strip():
        retrieved_docs = [{"text": line.strip(), "relevance": 0.9} for line in mock_context.strip().split("\n") if line.strip()]

    payload = {
        "person_id": profile_id,
        "session_id": session_id,
        "system_prompt": system_prompt if system_prompt else "You are a helpful AI assistant.",
        "user_query": message,
        "retrieved_docs": retrieved_docs
    }

    try:
        res = requests.post(f"{API_URL}/chat/invoke", json=payload)
        if res.status_code == 200:
            data = res.json()
            bot_response = data.get("response", "No response received.")
            telemetry = data.get("telemetry", {})
            
            # Append interaction to chat history
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": bot_response})
            
            return "", history, telemetry
        else:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"❌ Error {res.status_code}: {res.text}"})
            return "", history, "{}"
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"⚠️ Connection Error: {e}"})
        return "", history, "{}"

def logout_user():
    """Resets UI to login screen."""
    return (
        gr.update(visible=True),   # Login visible
        gr.update(visible=False),  # Session setup hidden
        gr.update(visible=False),  # Chat hidden
        "",                        # Clear profile input
        "Logged out successfully."
    )

# --- Gradio UI Layout ---

with gr.Blocks(title="Context-Engineered Agent Workspace", theme=gr.themes.Soft()) as demo:
    
    # Session States
    active_profile_id = gr.State("")
    active_session_id = gr.State("")

    gr.Markdown("# 🧠 Context-Engineered Agent Workspace")

    # ==================== STEP 1: SIGN IN ====================
    with gr.Column(id="login_step", visible=True) as login_step:
        gr.Markdown("### 🔑 Step 1: Sign In or Identify Profile")
        profile_input = gr.Textbox(
            label="Profile ID", 
            placeholder="e.g. user_shivam or engineer_01", 
            info="Enter a new or existing profile ID to access saved sessions."
        )
        login_btn = gr.Button("Continue ➔", variant="primary")
        login_status = gr.Markdown("")

    # ==================== STEP 2: SESSION MANAGER ====================
    with gr.Column(id="session_step", visible=False) as session_step:
        gr.Markdown("### 📂 Step 2: Choose or Create Session")
        profile_display = gr.Markdown("")
        
        session_mode = gr.Radio(
            choices=["Existing Session", "New Session"], 
            value="Existing Session", 
            label="Session Option"
        )

        # Existing Session Picker
        with gr.Column(visible=True) as existing_group:
            existing_dropdown = gr.Dropdown(label="Select Existing Session", choices=[])
            delete_sess_btn = gr.Button("🗑️ Delete Selected Session", variant="stop", size="sm")
            delete_status = gr.Markdown("")

        # New Session Creator
        with gr.Column(visible=False) as new_group:
            new_session_input = gr.Textbox(
                label="New Session ID", 
                placeholder="e.g. debug_session_01",
                info="Name a unique session identifier."
            )

        # System Prompt Input (Required for New Session)
        system_prompt_box = gr.Column(visible=False)
        with system_prompt_box:
            sys_prompt_input = gr.Textbox(
                label="System Prompt", 
                value="You are a helpful, concise AI technical assistant.", 
                lines=3,
                info="Define system behaviour and rules before starting."
            )

        session_prep_status = gr.Markdown("")
        start_chat_btn = gr.Button("Start Chat 💬", variant="primary", visible=False)
        change_profile_btn = gr.Button("⬅️ Change Profile ID", variant="secondary", size="sm")

    # ==================== STEP 3: CHAT WINDOW ====================
    with gr.Column(id="chat_step", visible=False) as chat_step_view:
        chat_header = gr.Markdown("### 🟢 Chat Window")
        
        with gr.Row():
            # Main Chat Frame
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, type="messages")
                msg_input = gr.Textbox(placeholder="Type your query and press Enter...", label="Message Input")
                
                with gr.Row():
                    send_btn = gr.Button("Send 🚀", variant="primary")
                    clear_btn = gr.Button("Clear UI Screen", size="sm")

            # Context & Telemetry Panel (Right Drawer)
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Live Context & Telemetry")
                telemetry_box = gr.JSON(label="Last Turn Budget Metrics")
                
                gr.Markdown("### 📝 Inject Context (RAG)")
                mock_context_input = gr.Textbox(
                    label="Mock Document Context", 
                    placeholder="Add external knowledge lines here...", 
                    lines=4
                )
                
                exit_chat_btn = gr.Button("🔴 Exit / Switch Session", variant="secondary")

    # ==================== EVENT BINDINGS ====================

    # 1. Login Event
    login_btn.click(
        fn=handle_login,
        inputs=[profile_input],
        outputs=[login_step, session_step, chat_step_view, existing_dropdown, login_status]
    ).then(
        fn=lambda p: p.strip(),
        inputs=[profile_input],
        outputs=[active_profile_id]
    )

    # 2. Toggle Mode (New vs Existing)
    def toggle_session_mode(mode):
        if mode == "Existing Session":
            return gr.update(visible=True), gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=True)

    session_mode.change(
        fn=toggle_session_mode,
        inputs=[session_mode],
        outputs=[existing_group, new_group]
    )

    # 3. Dynamic Session Selection Handler
    for element in [session_mode, existing_dropdown, new_session_input]:
        element.change(
            fn=handle_session_selection,
            inputs=[session_mode, existing_dropdown, new_session_input],
            outputs=[system_prompt_box, start_chat_btn, session_prep_status]
        )

    # 4. Delete Session
    delete_sess_btn.click(
        fn=delete_user_session,
        inputs=[active_profile_id, existing_dropdown],
        outputs=[delete_status]
    ).then(
        fn=fetch_user_sessions,
        inputs=[active_profile_id],
        outputs=[existing_dropdown]
    )

    # 5. Launch Chat Session
    start_chat_btn.click(
        fn=launch_chat_session,
        inputs=[active_profile_id, session_mode, existing_dropdown, new_session_input, sys_prompt_input],
        outputs=[session_step, chat_step_view, active_session_id, chatbot, chat_header]
    )

    # 6. Messaging / Chat Step
    msg_input.submit(
        fn=chat_step,
        inputs=[msg_input, chatbot, active_profile_id, active_session_id, sys_prompt_input, mock_context_input],
        outputs=[msg_input, chatbot, telemetry_box]
    )
    send_btn.click(
        fn=chat_step,
        inputs=[msg_input, chatbot, active_profile_id, active_session_id, sys_prompt_input, mock_context_input],
        outputs=[msg_input, chatbot, telemetry_box]
    )

    # 7. Navigation Actions
    clear_btn.click(fn=lambda: [], outputs=[chatbot])
    
    change_profile_btn.click(
        fn=logout_user,
        outputs=[login_step, session_step, chat_step_view, profile_input, login_status]
    )
    
    exit_chat_btn.click(
        fn=lambda pid: handle_login(pid),
        inputs=[active_profile_id],
        outputs=[login_step, session_step, chat_step_view, existing_dropdown, login_status]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)