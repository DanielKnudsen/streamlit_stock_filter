"""
Authentication UI module for the Streamlit Stock Filter app.
Handles all authentication-related UI components and logic.
"""

import streamlit as st
import time
from auth import (
    register_user, login_user, get_current_user, logout_user, 
    check_membership_status_by_email, reset_password, 
    save_portfolio, get_user_portfolios, delete_portfolio
)


def show_auth_dialog():
    """Authentication dialog for login, registration, and password reset"""
    auth_mode = st.radio("Välj inloggningsläge:", ["Logga in", "Registrera", "Återställ lösenord"], horizontal=True)
    email = st.text_input("E-post")
    
    if auth_mode == "Logga in":
        password = st.text_input("Lösenord", type="password")
        if st.button("Logga in", width="stretch"):
            result = login_user(email, password)
            user_after_login = get_current_user()
            if user_after_login:
                progress_text = "Inloggning lyckades! Skickar dig till startsidan. Vänligen vänta."
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.015)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                my_bar.empty()
                
                st.rerun()
            else:
                st.error("Fel e-post eller lösenord.")
                time.sleep(4)
                st.rerun()
                
    elif auth_mode == "Registrera":
        password = st.text_input("Lösenord", type="password")
        if st.button("Registrera", width="stretch"):
            result = register_user(email, password)
            if result:
                st.success("Registrering lyckades! Kontrollera din e-post för bekräftelse.")
                time.sleep(3)
                st.rerun()
            else:
                st.error("Registrering misslyckades. Prova igen.")
                
    else:  # Reset password
        if st.button("Skicka återställningslänk", width="stretch"):
            if email:
                result = reset_password(email)
                if result:
                    st.success("En återställningslänk har skickats till din e-post.")
                else:
                    st.error("Det gick inte att skicka återställningslänken. Kontrollera din e-postadress.")
            else:
                st.error("Vänligen ange din e-postadress.")


def show_account_dialog():
    """Account information and portfolio management dialog"""
    user = get_current_user()
    if not user:
        return
        
    st.write(f"**Inloggad som:** {user.email}")
    
    # Check membership status
    is_valid, membership_id, membership_name, iso_start_date, iso_end_date = check_membership_status_by_email(user.email)
    
    if is_valid:
        st.success(f"✅ **Giltigt abonnemang:** {membership_name}")
        st.write(f"**Startdatum:** {iso_start_date}")
        st.write(f"**Slutdatum:** {iso_end_date}")
    else:
        st.error("❌ **Inget giltigt abonnemang**")
        st.write("Läs mer på [indicatum.se](https://indicatum.se/)")
    
    st.divider()
    
    # Portfolio Management Section
    st.subheader("📁 Mina Portföljer")
    
    portfolios = get_user_portfolios(user.id)
    
    if portfolios:
        for portfolio in portfolios:
            with st.expander(f"📊 {portfolio['name']} ({len(portfolio['tickers'])} aktier)"):
                st.write(f"**Skapad:** {portfolio['created_at'][:10]}")
                if portfolio.get('description'):
                    st.write(f"**Beskrivning:** {portfolio['description']}")
                
                # Show ticker list
                tickers_text = ", ".join(portfolio['tickers'])
                st.text_area("Aktier:", value=tickers_text, height=100, disabled=True)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("🔍 Ladda portfölj", key=f"load_{portfolio['id']}", help="Visa endast aktier från denna portfölj i resultattabellen"):
                        st.session_state.loaded_portfolio = {
                            'tickers': portfolio['tickers'],
                            'name': portfolio['name']
                        }
                        st.success(f"Portfölj '{portfolio['name']}' laddad som filter!")
                        st.rerun()
                
                with col2:
                    if st.button("🗑️ Ta bort", key=f"delete_{portfolio['id']}"):
                        if delete_portfolio(portfolio['id']):
                            st.success("Portfölj borttagen!")
                            st.rerun()
                        else:
                            st.error("Det gick inte att ta bort portföljen.")
                
                with col3:
                    # Create CSV data for download
                    csv_data = "\n".join(portfolio['tickers'])
                    st.download_button(
                        "📥 Ladda ner",
                        data=csv_data,
                        file_name=f"{portfolio['name'].replace(' ', '_')}.csv",
                        mime="text/csv",
                        key=f"download_{portfolio['id']}"
                    )
    else:
        st.info("📂 Inga sparade portföljer ännu. Använd shortlist-funktionen för att skapa din första!")
    
    st.divider()
    
    if st.button("Logga ut", width="stretch", type="primary"):
        logout_user()
        time.sleep(1)
        st.rerun()


def handle_authentication(ENABLE_AUTHENTICATION: bool):
    """
    Main authentication handler for the Streamlit app.
    
    Args:
        ENABLE_AUTHENTICATION: Boolean to enable/disable authentication
        
    Returns:
        tuple: (user, should_stop_execution)
        - user: Current user object or None
        - should_stop_execution: Boolean indicating if app should stop (for auth flow)
    """
    user = get_current_user()
    
    # If authentication is disabled, return None user and continue
    if not ENABLE_AUTHENTICATION:
        return None, False
    
    # If no user and authentication is enabled, show auth dialog
    if not user:
        @st.dialog("Logga in eller registrera dig")
        def show_auth_dialog_wrapper():
            show_auth_dialog()
        
        show_auth_dialog_wrapper()
        return None, True  # Stop execution to show auth dialog
    
    # Check if user has valid subscription
    is_valid, membership_id, membership_name, iso_start_date, iso_end_date = check_membership_status_by_email(user.email)
    if not is_valid:
        st.error(f"Hej {user.email}, tyvärr har du inget giltigt abonnemang. Läs mer på https://indicatum.se/")
        if st.button("Kontoinformation", type="secondary"):
            @st.dialog("Kontoinformation")
            def show_account_dialog_wrapper():
                show_account_dialog()
            show_account_dialog_wrapper()
        return user, True  # Stop execution due to invalid subscription
    
    return user, False  # Valid user, continue execution


def render_account_buttons(user, ENABLE_AUTHENTICATION: bool, get_concurrent_users_func):
    """
    Render account-related buttons in the main interface.
    
    Args:
        user: Current user object
        ENABLE_AUTHENTICATION: Boolean to enable/disable authentication features
        get_concurrent_users_func: Function to get concurrent users count
    """
    if not user or not ENABLE_AUTHENTICATION:
        return
    
    col1, col2, col3 = st.columns([5, 1, 1])
    with col2:
        if st.button("👤 Konto", help="Visa kontoinformation"):
            @st.dialog("Kontoinformation")
            def show_account_dialog_wrapper():
                show_account_dialog()
            show_account_dialog_wrapper()
    with col3:
        # Simple user monitoring (development mode)
        if st.button("📊", help="Användningsstatistik"):
            concurrent = get_concurrent_users_func()
            st.info(f"Aktiva användare: {concurrent}\nSession: {st.session_state.user_id[:8]}")


def handle_portfolio_save_dialog(user, df_display, current_time):
    """
    Handle the portfolio save dialog functionality.
    
    Args:
        user: Current user object
        df_display: DataFrame with current shortlist data
        current_time: Current timestamp string
        
    Returns:
        Boolean indicating if dialog was processed
    """
    if not st.session_state.get('show_save_portfolio', False):
        return False
    
    @st.dialog("Spara portfölj")
    def portfolio_save_dialog():
        with st.form("save_portfolio_form"):
            st.write("**Spara bevakningslista som portfölj**")
            portfolio_name = st.text_input("Portföljnamn:", value=f"Portfölj {current_time}")
            portfolio_description = st.text_area("Beskrivning (valfritt):", height=100)
            
            col_save, col_cancel = st.columns([1, 1])
            with col_save:
                if st.form_submit_button("Spara portfölj", width="stretch"):
                    if portfolio_name.strip():
                        # Get current filter settings
                        filter_settings = {
                            "timestamp": current_time,
                            "num_stocks": len(df_display),
                            # Add other relevant filter settings here
                        }
                        
                        # Save portfolio
                        tickers_list = df_display.index.tolist()
                        result = save_portfolio(user.id, portfolio_name, tickers_list, filter_settings, portfolio_description)
                        
                        if result:
                            st.success(f"Portfölj '{portfolio_name}' sparad!")
                            st.session_state.show_save_portfolio = False
                            st.rerun()
                        else:
                            st.error("Det gick inte att spara portföljen. Försök igen.")
                    else:
                        st.error("Vänligen ange ett portföljnamn.")
            
            with col_cancel:
                if st.form_submit_button("Avbryt", width="stretch"):
                    st.session_state.show_save_portfolio = False
                    st.rerun()
    
    portfolio_save_dialog()
    return True