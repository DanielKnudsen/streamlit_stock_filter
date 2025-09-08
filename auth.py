import os
import streamlit as st
from supabase import create_client, Client
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timezone

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
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if response.session:
            st.session_state["supabase_session"] = response.session
        return response
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return None

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

# Password reset function
def reset_password(email: str):
    supabase: Client = get_supabase_client()
    try:
        supabase.auth.reset_password_email(email)
        st.write("Password reset email sent.")
        return True
    except Exception as e:
        st.error(f"Password reset failed: {str(e)}")
        return None

def check_membership_status_by_email(email):
    WP_URL = 'https://indicatum.se/wp-json/pmpro/v1/get_membership_level_for_user'
    WP_USERNAME = st.secrets["WP_USERNAME"]
    WP_APP_PASSWORD = st.secrets["WP_APP_PASSWORD"]
    try:
        # Gör API-anrop med email-parameter
        response = requests.get(
            f"{WP_URL}?email={email}",
            auth=HTTPBasicAuth(WP_USERNAME, WP_APP_PASSWORD),
            timeout=10
        )
        # Kontrollera status
        if response.status_code == 200:
            data = response.json()
            #print(f"API-svar: {data}")
            if data:
                if 'startdate' in data:
                    startdate = data.get('startdate')
                    # extract date from datetime
                    iso_start_date = datetime.fromtimestamp(int(startdate), timezone.utc).date()

                # Hantera enddate
                iso_end_date = None
                enddate = data.get('enddate')
                #print(f"Raw enddate from API: {enddate}")
                if enddate:
                    try:
                        # First try to parse as Unix timestamp (integer)
                        unix_timestamp = int(enddate)
                        iso_end_date = datetime.fromtimestamp(unix_timestamp, timezone.utc).date()
                        #print(f"Parsed enddate (Unix timestamp): {iso_end_date}")
                    except (ValueError, TypeError):
                        # If that fails, try to parse as MySQL datetime string
                        try:
                            iso_end_date = datetime.strptime(enddate, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).date()
                            #print(f"Parsed enddate (MySQL format): {iso_end_date}")
                        except ValueError as e:
                            #print(f"Fel vid parsning av enddate: {e}")
                            return False, None, None

                # Kontrollera om medlemskapet är aktivt
                current_time = datetime.now(timezone.utc).date()
                #print(f"Current datetime: {current_time}")
                #print(f"Membership enddate: {iso_end_date}")
                is_valid = enddate is None or iso_end_date >= current_time
                membership_id = data.get('id') if is_valid else None
                membership_name = data.get('name') if is_valid else None

                return is_valid, membership_id, membership_name, iso_start_date, iso_end_date
            else:
                return False, None, None, None, None
        else:
            #print(f"API-fel: {response.status_code} - {response.text}")
            return False, None, None, None, None
    except Exception as e:
        #print(f"Fel vid API-anrop: {e}")
        return False, None, None, None, None
