import streamlit as st
from supabase import create_client, Client

# Replace with your Supabase project URL and anon key
def get_supabase_client():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

# Registration function
def register_user(email: str, password: str):
    supabase: Client = get_supabase_client()
    response = supabase.auth.sign_up({"email": email, "password": password})
    return response

# Login function
def login_user(email: str, password: str):
    supabase: Client = get_supabase_client()
    response = supabase.auth.sign_in_with_password({"email": email, "password": password})
    if response.session:
        st.session_state["supabase_session"] = response.session
    return response

# Check if user is logged in
def get_current_user():
    # Check session state first
    session = st.session_state.get("supabase_session", None)
    if session and session.user:
        return session.user
    # Fallback to supabase client session
    supabase: Client = get_supabase_client()
    session = supabase.auth.get_session()
    return session.user if session and session.user else None

# Logout function
def logout_user():
    supabase: Client = get_supabase_client()
    supabase.auth.sign_out()
    st.session_state.pop("supabase_session", None)
