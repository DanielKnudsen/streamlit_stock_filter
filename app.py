import streamlit as st
import pandas as pd
import plotly.graph_objects as go # Import Plotly
import plotly.express as px # Import Plotly Express for bubble plot
import numpy as np # For handling numerical operations
import pwlf
from pathlib import Path
from rank import load_config
import datetime
import time
import uuid
import json
from auth import register_user, login_user, get_current_user, logout_user, check_membership_status_by_email, reset_password, save_portfolio, get_user_portfolios, delete_portfolio

# =====================================================================
# STREAMLIT STOCK SCREENING APP - SWEDISH MARKETS
# =====================================================================

# Basic user tracking
def init_user_tracking():
    """Initialize basic user tracking"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
        st.session_state.session_start = datetime.datetime.now()
    
    # Log user activity
    try:
        log_entry = {
            'user_id': st.session_state.user_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'action': 'page_view'
        }
        
        with open(f'user_logs_{ENVIRONMENT}.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception:
        pass  # Fail silently if logging fails

def get_concurrent_users():
    """Get approximate concurrent users (active in last 5 minutes)"""
    try:
        from datetime import timedelta
        active_threshold = datetime.datetime.now() - timedelta(minutes=5)
        active_users = set()
        
        with open(f'user_logs_{ENVIRONMENT}.jsonl', 'r') as f:
            for line in f.readlines()[-200:]:  # Check last 200 entries
                try:
                    log = json.loads(line.strip())
                    log_time = datetime.datetime.fromisoformat(log['timestamp'])
                    if log_time > active_threshold:
                        active_users.add(log['user_id'])
                except Exception:
                    continue
                    
        return len(active_users)
    except Exception:
        return 0

# Initialize user tracking
init_user_tracking()

# Allow user to toggle between "wide" and "centered" layout
layout_mode = 'wide'#st.toggle("Bredd layout (wide)?", value=True)
st.set_page_config(
    layout="wide" if layout_mode == 'wide' else "centered",
    page_title="Indicatum Insights",
    page_icon="📈"
)
# Introduce the app and its purpose with enhanced visual appeal
with st.container(border=True):
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; color: white; margin-bottom: 20px;">
        <h1 style="color: white; margin-bottom: 15px;">✨ Välkommen till Indicatum Insights! ✨</h1>
        <h3 style="color: #f0f0f0; font-weight: normal; margin-bottom: 20px;">
            Förkorta din research. Förbättra dina edge.
        </h3>
        <p style="font-size: 18px; color: #e0e0e0;">
            Smart filtrering + djup analys = bättre investeringsbeslut
        </p>
    </div>
    """, unsafe_allow_html=True)
# =============================
# IMPORTS AND SETUP
# =============================

# Load environment variables
ENVIRONMENT = st.secrets["ENVIRONMENT"]

# Debug logging control - set to False for production, True for local debugging
ENABLE_DEBUG_LOGGING = ENVIRONMENT == 'local'

# Authentication control - disable for local testing, enable for production
ENABLE_AUTHENTICATION = ENVIRONMENT != 'local'

# Load configuration from YAML file
config = load_config("rank-config.yaml")

# --- Get directories for CSV files ---
CSV_PATH = Path('data') / ('local' if ENVIRONMENT == 'local' else 'remote')

show_Ratio_to_Rank =True


# --- Authentication UI ---

user = get_current_user()
#st.write(f"Current user: {user.email if user else 'None'}")
if not user and ENABLE_AUTHENTICATION:
    @st.dialog("Logga in eller registrera dig")
    def show_auth_dialog():
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

    show_auth_dialog()
    st.stop()
else:
    @st.dialog("Kontoinformation")
    def show_account_dialog():
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
        
        # Note: You'll need to implement get_user_portfolios function in auth.py
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
    
    # Check if user has valid subscription for app access (only when authentication is enabled)
    if ENABLE_AUTHENTICATION and user:
        is_valid, membership_id, membership_name, iso_start_date, iso_end_date = check_membership_status_by_email(user.email)
        if not is_valid:
            st.error(f"Hej {user.email}, tyvärr har du inget giltigt abonnemang. Läs mer på https://indicatum.se/")
            if st.button("Kontoinformation", type="secondary"):
                show_account_dialog()
            st.stop()
            

# Development mode indicator
if not ENABLE_AUTHENTICATION:
    st.info("🔧 **UTVECKLINGSLÄGE** - Autentisering är inaktiverad för lokal testning")

# Add account info and stats buttons after the welcome section (only when authentication is enabled)
if user and ENABLE_AUTHENTICATION:
    col1, col2, col3 = st.columns([5, 1, 1])
    with col2:
        if st.button("👤 Konto", help="Visa kontoinformation"):
            show_account_dialog()
    with col3:
        # Simple user monitoring (development mode)
        if st.button("📊", help="Användningsstatistik"):
            concurrent = get_concurrent_users()
            st.info(f"Aktiva användare: {concurrent}\nSession: {st.session_state.user_id[:8]}")

with st.container(border=True):
    

    
    # Three-step workflow cards using Streamlit columns
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.markdown("""
        <div style="padding: 20px; background: white; border-radius: 12px; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid #667eea; text-align: center;">
            <div style="font-size: 48px; margin-bottom: 10px;">🎯</div>
            <h4 style="color: #667eea; margin: 10px 0;">1. Filtrera</h4>
            <p style="color: #666; font-size: 14px; line-height: 1.5;">
                Välj bland 500+ svenska aktier med smarta filter för sektor, storlek och prestanda
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 20px; background: white; border-radius: 12px; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid #28a745; text-align: center;">
            <div style="font-size: 48px; margin-bottom: 10px;">📊</div>
            <h4 style="color: #28a745; margin: 10px 0;">2. Analysera</h4>
            <p style="color: #666; font-size: 14px; line-height: 1.5;">
                Djupdyk i nyckeltal, trender och teknisk analys för varje aktie som fångar ditt intresse
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="padding: 20px; background: white; border-radius: 12px; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid #ffc107; text-align: center;">
            <div style="font-size: 48px; margin-bottom: 10px;">💎</div>
            <h4 style="color: #e67e00; margin: 10px 0;">3. Investera</h4>
            <p style="color: #666; font-size: 14px; line-height: 1.5;">
                Bygg din bevakningslista och fatta välgrundade beslut baserat på data och trender
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Pro-tips section
    st.markdown("""
    <div style="margin-top: 25px; padding: 15px; background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); 
                border-radius: 10px; border: 1px solid #e1bee7; text-align: center;">
        <p style="color: #4a148c; font-size: 16px; margin: 5px 0; font-weight: 500;">
            💡 <strong>Pro-tips:</strong> Använd livbojen 🛟 i varje sektion för experttips och genvägar!
        </p>
        <p style="color: #6a1b9a; font-size: 14px; margin: 5px 0;">
            🎯 Kombinera flera filter → 📈 Analysera bubbeldiagram → ⭐ Shortlista favoriter → 🔍 Djupdykning per aktie
        </p>
    </div>
    """, unsafe_allow_html=True)

with st.expander("🛟 **Hur kan du använda detta verktyg?** (Klicka för att visa)", expanded=False):
    st.markdown(
        """
        **🚀 Från nybörjare till aktieproffs – här är din roadmap:**

        **🎯 För snabba resultat:**  
        • Aggregerad rank-reglage → Upptäck topp-prestanda direkt  
        • TTM-data → Fånga hetaste trenderna nu  
        • Trend 4 år → Hitta långsiktiga vinnare  

        **🔍 För detektiv-analys:**  
        • Kategori-filter → Lönsamhet, tillväxt, värdering  
        • Teknisk analys → SMA-breakouts och momentum  
        • Sector rotation → Vad är hett just nu?  

        **💰 Smart investeringsstrategier:**  
        • **Value hunting:** Stark tillväxt + låg kurs = underskattat?  
        • **Growth hacking:** TTM-acceleration + trend = raket på väg upp?  
        • **Turnaround plays:** Dålig historik + stark TTM = comeback?  
        • **Momentum riding:** Teknisk breakout + fundamental styrka = perfekt timing?  

        **🎨 Pro-workflow:**  
        1. **Filtrera** brett → **Shortlista** favoriter → **Djupdykning** per aktie  
        2. **Jämför** sektorer → **Identifiera** avvikare → **Validera** med teknisk analys  
        3. **Exportera** shortlist → **Bevaka** utveckling → **Uppdatera** regelbundet  

        **💡 Secret sauce:** TTM + Trend = magisk kombination för early detection!  
        """
    )
# Logga miljö och path för felsökning, samt datum för när filen stock_evaluations_result.csv senast uppdaterades   
st.write(f"Running in environment: {ENVIRONMENT}, using CSV path: {CSV_PATH}, data last updated: {pd.to_datetime(datetime.datetime.fromtimestamp(Path(CSV_PATH / config['results_file']).stat().st_mtime)).strftime('%Y-%m-%d %H:%M:%S')}")
# =============================
# HELPER FUNCTIONS
# =============================
# Funktion för att applicera bakgrundsfärger baserat på värden
def color_progress(val):
    # Get color ranges from config loaded at the top of the file
    color_ranges = config.get('color_ranges', [])
    for cr in color_ranges:
        # Each cr should be a dict with 'range' and 'color' keys
        if cr['range'][0] <= val <= cr['range'][1]:
            return f'background-color: {cr["color"]}'
    return ''

# Instead of storing SMAs, calculate them in the app
def add_moving_averages(df, short_window=config['SMA_short'], medium_window=config['SMA_medium'], long_window=config['SMA_long']):
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_medium'] = df['Close'].rolling(window=medium_window).mean() 
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
    return df

def create_slider(df, column_name, display_name_func, tooltip_func, step=1.0, format_str="%d%%"):
    """
    Skapar en Streamlit-slider för en given kolumn i en DataFrame.

    Parametrar:
        df: DataFrame som innehåller kolumnen
        column_name: Namn på kolumnen i DataFrame
        display_name_func: Funktion som returnerar visningsnamn för kolumnen
        tooltip_func: Funktion som returnerar tooltip-text för kolumnen
        step: Stegstorlek för slidern (default: 1.0)
        format_str: Formatsträng för sliderns värden (default: "%d%%")

    Returnerar:
        Tuple med valda min- och maxvärden från slidern
    """
    min_value = float(df[column_name].min())
    max_value = float(df[column_name].max())
    # Ensure the slider has a valid range
    if min_value == max_value:
        max_value += 0.001  # Ensure a valid range if min and max are equal
    
    return st.slider(
        label=display_name_func(column_name),
        min_value=min_value,
        max_value=max_value,
        value=(min_value, max_value),
        step=step,
        format=format_str,
        help=tooltip_func(column_name)
    )
def get_display_name(var_name):
    # Try to get a pretty name, fallback to a cleaned-up version
    return display_names.get(var_name, var_name.replace("_", " ").title())

def get_tooltip_text(var_name):
    # Try to get a tooltip text, fallback to an empty string
    return tooltip_texts.get(var_name, "")

def get_ratio_help_text(var_name):
    # Try to get a help text, fallback to an empty string
    return ratio_help_texts.get(var_name, "")

# Format the annotation value to be more readable (e.g., 60B instead of 60096000000.0)
def human_format(num):
    if num is None or pd.isna(num):
        return "N/A"
    num = float(num)
    # Use Swedish units: '', 't', 'Mkr', 'Mdkr', 'Bnkr'
    for unit in ['', ' t', ' Mkr', ' Mdkr', ' Bnkr']:
        if abs(num) < 1000.0:
            return f"{num:.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f} Bnkr"

# =============================
# LOAD DATA
# =============================
try:

    # Load main stock evaluation CSV (index_col=0 sets Ticker as index)
    df_new_ranks = pd.read_csv(CSV_PATH / config["results_file"], index_col=0)
    df_dividends = pd.read_csv(CSV_PATH / "dividends.csv", index_col='Ticker')
    unique_values_lista = df_new_ranks['Lista'].dropna().unique().tolist()
    unique_values_sector = df_new_ranks['Sektor'].dropna().unique().tolist()
    allCols_latest_ratioValue = [col for col in df_new_ranks.columns if col.endswith('_latest_ratioValue')]
    allCols_trend_ratioValue = [col for col in df_new_ranks.columns if col.endswith('_trend_ratioValue')]
    allCols_latest_ratioRank = [col for col in df_new_ranks.columns if col.endswith('_latest_ratioRank')]
    allCols_trend_ratioRank = [col for col in df_new_ranks.columns if col.endswith('_trend_ratioRank')]
    allCols_AvgGrowth_Rank = [col for col in df_new_ranks.columns if col.endswith('_AvgGrowth_Rank')]
    allCols_AvgGrowth = [col for col in df_new_ranks.columns if col.endswith('_AvgGrowth')] + ['cagr_close']
    # --- Create ratio-to-rank mapping dict from config['ratio_definitions'] ---
    ratio_definitions = config.get('ratio_definitions', {})
    ratio_to_rank_map = {}
    ratio_to_rank_latest_map = {}
    ratio_to_rank_trend_map = {}
    for ratio in ratio_definitions.keys():
        ratio_to_rank_map[f"{ratio}_latest_ratioValue"] = f"{ratio}_latest_ratioRank"
        ratio_to_rank_map[f"{ratio}_trend_ratioValue"] = f"{ratio}_trend_ratioRank"
        ratio_to_rank_latest_map[f"{ratio}_latest_ratioValue"] = f"{ratio}_latest_ratioRank"
        ratio_to_rank_trend_map[f"{ratio}_trend_ratioValue"] = f"{ratio}_trend_ratioRank"
    # =============================
    # COLUMN SELECTION FOR FILTERING AND DISPLAY
    # =============================
    # Filter columns that contain the string "catRank" for the main table
    rank_score_columns = [col for col in df_new_ranks.columns if "catRank" in col]
    latest_columns = [col for col in rank_score_columns if "latest" in col.lower()]
    trend_columns = [col for col in rank_score_columns if "trend" in col.lower()]
    ttm_columns = [col for col in rank_score_columns if "ttm" in col.lower()]
    rank_score_columns = rank_score_columns + ['Latest_clusterRank', 'Trend_clusterRank', 'TTM_clusterRank', 'Lista','personal_weights','QuarterDiff','TTM_diff_vs_pct_ch_20_d_diff','Latest_diff_vs_pct_ch_20_d_diff']  # Include total scores
    # Initialize a DataFrame that will be filtered by sliders
    df_filtered_by_sliders = df_new_ranks.copy()
    
    # DEBUG: Check QuarterDiff in original DataFrame
    if ENABLE_DEBUG_LOGGING:
        print(f"DEBUG: df_new_ranks QuarterDiff column dtype: {df_new_ranks['QuarterDiff'].dtype}")
        print(f"DEBUG: df_new_ranks QuarterDiff unique values: {df_new_ranks['QuarterDiff'].unique()}")
        print(f"DEBUG: df_filtered_by_sliders QuarterDiff unique values: {df_filtered_by_sliders['QuarterDiff'].unique()}")

    # =============================
    # PORTFOLIO FILTER
    # =============================
    # Check if a portfolio is loaded as filter
    if 'loaded_portfolio' in st.session_state:
        loaded_portfolio = st.session_state.loaded_portfolio
        portfolio_tickers = [ticker.upper() for ticker in loaded_portfolio['tickers']]
        
        # Create a container to show portfolio filter status
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"🔍 **Portföljfilter aktiv:** {loaded_portfolio['name']} ({len(portfolio_tickers)} aktier)")
            with col2:
                if st.button("❌ Ta bort filter", key="remove_portfolio_filter"):
                    del st.session_state.loaded_portfolio
                    st.rerun()
        
        # Apply portfolio filter to the dataframe
        df_filtered_by_sliders = df_filtered_by_sliders[
            df_filtered_by_sliders.index.str.upper().isin(portfolio_tickers)
        ]
    st.write(f"Totalt antal aktier efter portföljfilter: {len(df_filtered_by_sliders)}")
    # =============================
    # LOAD RANKING CATEGORIES FROM CONFIG
    # =============================

    if config:
        category_ratios = config.get("category_ratios", {})
        categories = list(category_ratios.keys())
        
        display_names = config.get("display_names", {})
        tooltip_texts = config.get("tooltip_texts", {})
        ratio_help_texts = config.get("ratio_help_texts", {})

        # new
        all_ratios = []
        for category, ratios in config['kategorier'].items():
            all_ratios.extend(ratios)
        #st.write("Ratios:", all_ratios)
        kategorier = config.get("kategorier", {}).keys()
        #st.write("Kategorier:", kategorier)
        kategorier_ratios = config.get("kategorier", {})
        #st.write("Kategorier med Ratios:", kategorier_ratios)
        cluster = config.get("cluster", {})
        #st.write("Kluster:", cluster)
    else:
        category_ratios = {}
        categories = []
    
    unique_category_display_names = list(set(get_display_name(cat.split("_")[0]) for cat in categories))
    # =============================
    # ENHETLIGT FILTERAVSNITT
    # =============================
    with st.container(border=True, key="filter_section"):
        st.subheader("🎯 Aktiefilter – Hitta dina favoriter")


        tab1, tab2, tab3, tab4 = st.tabs(["🛟 Info", "Förenklad filtrering", "Utökade filtermöjligheter", "Avancerad filtrering"])
        with tab1:
            with st.expander("🛟 **Hjälp med Filtrering?** (Klicka för att visa)", expanded=False):
                st.markdown("""
                **Tre sätt att hitta dina ideala aktier:**

                **1. 🚀 Förenklad filtrering:**  
                • Viktning av trend vs senaste året vs TTM  
                • Perfekt för snabb överblick  
                • Smart algoritm rankar åt dig  

                **2. 🎯 Utökade filtermöjligheter:**  
                • Finjustera med totalrank + tillväxt + teknisk analys  
                • Skriv in specifika tickers  
                • Resultatet uppdateras live  

                **3. 🔬 Avancerad filtrering:**  
                • Djupdykning i kategorier & nyckeltal  
                • För experter som vill ha full kontroll  
                • Skräddarsydda kombinationer  

                **🎨 Extra-tips:**  
                • **Lista/Sektor:** Klicka färgade "pills" för snabbval  
                • **Ticker-sök:** Skriv flera tickers separerade med komma  
                • **Kombination:** Använd flera filter samtidigt för laser-precision  
                """)

        with tab2:
            st.markdown("""
            ### 🎯 Din egen smarta ranking – väg ihop som du vill!

            • **Trend:** Hur bra var bolaget senaste 4 åren?  
            • **Senaste året:** Vad händer just nu?  
            • **TTM:** Senaste kvartalen (heta signaler!)  

            **Justera reglagen → Se resultatet live → Hitta dina favoriter!**
            """)
            # Tre sliders för preliminära värden
            col_trend, col_latest, col_ttm = st.columns(3, gap='medium', border=True)
            with col_trend:
                label = "Viktning för Trend (%)"
                trend = st.slider(label, label_visibility="visible", help=get_tooltip_text(label), min_value=0.0, max_value=100.0, value=33.3, step=1.0, key="trend_slider")
            with col_latest:
                label = "Viktning för Senaste (%)"
                latest = st.slider(label, label_visibility="visible", help=get_tooltip_text(label), min_value=0.0, max_value=100.0, value=33.3, step=1.0, key="latest_slider")
            with col_ttm:
                label = "Viktning för TTM (%)"
                ttm = st.slider(label, label_visibility="visible", help=get_tooltip_text(label), min_value=0.0, max_value=100.0, value=33.3, step=1.0, key="ttm_slider")

            # Beräkna summan av preliminära värden
            total = trend + latest + ttm

            # Normalisera värdena om summan inte är noll
            if total > 0:
                norm_trend = (trend / total) * 100
                norm_latest = (latest / total) * 100
                norm_ttm = (ttm / total) * 100
            else:
                norm_trend = 0.0
                norm_latest = 0.0
                norm_ttm = 0.0

            if total == 0:
                st.warning("⚠️ Alla viktningar är 0! Sätt minst en viktning > 0 för att få resultat.")

            df_filtered_by_sliders['personal_weights'] = (
            df_filtered_by_sliders['Trend_clusterRank'] * norm_trend +
            df_filtered_by_sliders['Latest_clusterRank'] * norm_latest +
            df_filtered_by_sliders['TTM_clusterRank'] * norm_ttm
            ) / 100
            df_filtered_by_sliders.sort_values(by='personal_weights', ascending=False, inplace=True)

        with tab3:
            st.markdown("""
            ### 🎚️ Finjustera med precision – mer kontroll!

            **Totalrank-reglage:**  
            • Trend, Senaste, TTM – sätt min/max gränser  

            **Extra filter:**  
            • CAGR-tillväxt för långsiktiga trender  
            • SMA-tekniska indikatorer för timing  
            • Ticker-sök för specifika bolag  

            **Kombinera filter → Smalna av resultatet → Hitta pärlorna!**
            """)
            col_total_trend, col_total_latest, col_total_ttm = st.columns(3, gap='medium', border=True)
            with col_total_trend:
                trend_range = create_slider(df_new_ranks, 'Trend_clusterRank', get_display_name, get_tooltip_text, 1.0, "%d")
            with col_total_latest:
                latest_range = create_slider(df_new_ranks, 'Latest_clusterRank', get_display_name, get_tooltip_text, 1.0, "%d")
            with col_total_ttm:
                ttm_range = create_slider(df_new_ranks, 'TTM_clusterRank', get_display_name, get_tooltip_text, 1.0, "%d")

            df_filtered_by_sliders = df_filtered_by_sliders[
            (df_filtered_by_sliders['Trend_clusterRank'] >= trend_range[0]) &
            (df_filtered_by_sliders['Trend_clusterRank'] <= trend_range[1]) &
            (df_filtered_by_sliders['Latest_clusterRank'] >= latest_range[0]) &
            (df_filtered_by_sliders['Latest_clusterRank'] <= latest_range[1]) &
            (df_filtered_by_sliders['TTM_clusterRank'] >= ttm_range[0]) &
            (df_filtered_by_sliders['TTM_clusterRank'] <= ttm_range[1])
            ]

            st.markdown("##### Filtrera efter genomsnittlig tillväxt")
            cagr_left, cagr_middle, cagr_right = st.columns(3, gap='medium', border=True)
            with cagr_left:
                cagr_range_left = create_slider(df_new_ranks, allCols_AvgGrowth_Rank[0], get_display_name, get_tooltip_text, 0.1, "%.1f")
                df_filtered_by_sliders = df_filtered_by_sliders[
                    (df_filtered_by_sliders[allCols_AvgGrowth_Rank[0]] >= cagr_range_left[0]) &
                    (df_filtered_by_sliders[allCols_AvgGrowth_Rank[0]] <= cagr_range_left[1])
                ]
            with cagr_middle:
                cagr_range_middle = create_slider(df_new_ranks, allCols_AvgGrowth_Rank[1], get_display_name, get_tooltip_text, 0.1, "%.1f")
                df_filtered_by_sliders = df_filtered_by_sliders[
                    (df_filtered_by_sliders[allCols_AvgGrowth_Rank[1]] >= cagr_range_middle[0]) &
                    (df_filtered_by_sliders[allCols_AvgGrowth_Rank[1]] <= cagr_range_middle[1])
                ]
            with cagr_right:
                cagr_range_right = create_slider(df_new_ranks, allCols_AvgGrowth_Rank[2], get_display_name, get_tooltip_text, 0.1, "%.1f")
                df_filtered_by_sliders = df_filtered_by_sliders[
                    (df_filtered_by_sliders[allCols_AvgGrowth_Rank[2]] >= cagr_range_right[0]) &
                    (df_filtered_by_sliders[allCols_AvgGrowth_Rank[2]] <= cagr_range_right[1])
                ]
            st.markdown("##### Filtrera efter SMA-differenser")
            col_diff_long_medium, col_diff_short_medium, col_diff_price_short = st.columns(3, gap='medium', border=True)
            with col_diff_long_medium:
                diff_long_medium_range = create_slider(df_new_ranks, 'pct_SMA_medium_vs_SMA_long', get_display_name, get_tooltip_text, 1.0, "%d%%")
            with col_diff_short_medium:
                diff_short_medium_range = create_slider(df_new_ranks, 'pct_SMA_short_vs_SMA_medium', get_display_name, get_tooltip_text, 1.0, "%d%%")
            with col_diff_price_short:
                diff_price_short_range = create_slider(df_new_ranks, 'pct_Close_vs_SMA_short', get_display_name, get_tooltip_text, 1.0, "%d%%")
            df_filtered_by_sliders = df_filtered_by_sliders[
            (df_filtered_by_sliders['pct_SMA_medium_vs_SMA_long'] >= diff_long_medium_range[0]) &
            (df_filtered_by_sliders['pct_SMA_medium_vs_SMA_long'] <= diff_long_medium_range[1]) &
            (df_filtered_by_sliders['pct_SMA_short_vs_SMA_medium'] >= diff_short_medium_range[0]) &
            (df_filtered_by_sliders['pct_SMA_short_vs_SMA_medium'] <= diff_short_medium_range[1]) &
            (df_filtered_by_sliders['pct_Close_vs_SMA_short'] >= diff_price_short_range[0]) &
            (df_filtered_by_sliders['pct_Close_vs_SMA_short'] <= diff_price_short_range[1])
            ]
            st.write(f"**Aktuella urval:** {df_filtered_by_sliders.shape[0]} aktier")

            ticker_input = st.text_input(
                "Filtrera på tickers (kommaseparerade, t.ex. VOLV-A,ERIC-B,ATCO-A):",
                value="",
                help="Skriv in en eller flera tickers separerade med komma för att endast visa dessa aktier."
                )
            if ticker_input.strip():
                tickers_to_keep = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
                df_filtered_by_sliders = df_filtered_by_sliders[df_filtered_by_sliders.index.str.upper().isin(tickers_to_keep)]

        with tab4:
            st.markdown("""
            ### 🔬 Expertnivå – full kontroll över varje nyckeltal!

            **För dig som vill micro-managea:**  
            • Filtrera på kategori-nivå (Trend, Senaste, TTM)  
            • Detaljstyrning av varje enskilt nyckeltal  
            • Skapa helt skräddarsydda urval  

            **Varning:** Här kan du gå ner i kaninhålet och komma fram 3 timmar senare! 🐰
            """)
            col_filter_left, col_filter_mid, col_filter_right = st.columns(3,gap='medium',border=True)
            with col_filter_left:
                st.markdown("###### Filtrera för kategori Trend-rankningar")
                if trend_columns:
                    for col in trend_columns:
                        with st.container(border=True,key=f"container_trend_{col}"):
                            min_val = df_filtered_by_sliders[col].min()
                            max_val = df_filtered_by_sliders[col].max()
                            slider_min = float(min_val)
                            slider_max = float(max_val)
                            if slider_min == slider_max:
                                slider_max += 0.001
                            current_min, current_max = st.slider(
                                f"{col.replace('_trend_catRank', ' trend Rank')}",
                                min_value=slider_min,
                                max_value=slider_max,
                                value=(slider_min, slider_max),
                                key=f"slider_trend_{col}",
                                step=1.0,
                                format="%d"
                            )
                            df_filtered_by_sliders = df_filtered_by_sliders[
                                (df_filtered_by_sliders[col] >= current_min) &
                                (df_filtered_by_sliders[col] <= current_max)
                            ]
                            category_name = col.replace("catRank", "ratioRank")
                            # Dynamiskt skapa flikar för varje trendkategori med nyckeltalsnamn
                            ratio_name = [r for r in category_ratios[category_name]]
                            ratio_name_display = [r.replace("_trend_ratioRank", "") for r in ratio_name] 
                            tab_labels = ['Info'] + ratio_name_display
                            tabs = st.tabs(tab_labels)
                            tabs[0].write(f"Detaljerad filtrering för *nyckeltal* i {category_name.replace('_trend_ratioRank', '')}:")
                            # Lägg till reglage för varje nyckeltalsflik (från index 1 och uppåt) trend_slope
                            for i, r in enumerate(ratio_name):
                                with tabs[i+1]:
                                    if r in df_filtered_by_sliders.columns:
                                        min_val = float(df_filtered_by_sliders[r].min())
                                        max_val = float(df_filtered_by_sliders[r].max())
                                        if min_val == max_val:
                                            max_val += 0.001
                                        slider_min, slider_max = st.slider(
                                            f"Filtrera {r.replace('_trend_ratioRank', ' trend Rank')} ",
                                            min_value=min_val,
                                            max_value=max_val,
                                            value=(min_val, max_val),
                                            key=f"slider_tab_trend_{category_name}_{r}",
                                            step=1.0,
                                            format="%d"
                                        )
                                        df_filtered_by_sliders = df_filtered_by_sliders[
                                            (df_filtered_by_sliders[r] >= slider_min) &
                                            (df_filtered_by_sliders[r] <= slider_max)
                                        ]
                                    else:
                                        st.info(f"Kolumn {r} saknas i data.")
                                    # Add filter for trendSlope, but do NOT exclude NaN values (keep them in the filtered DataFrame)
                                    r_data = f"{r.replace('_trend_ratioRank', '_trend_ratioValue')}"
                                    if r_data in df_filtered_by_sliders.columns:
                                        min_val = float(df_filtered_by_sliders[r_data].min(skipna=True))
                                        max_val = float(df_filtered_by_sliders[r_data].max(skipna=True))
                                        if min_val == max_val:
                                            max_val += 0.001
                                        slider_min, slider_max = st.slider(
                                            f"Filtrera {r_data.replace('_trend_ratioValue', ' trend Slope')}",
                                            min_value=min_val,
                                            max_value=max_val,
                                            value=(min_val, max_val),
                                            key=f"slider_tab_latest_{r_data}",
                                            step=0.1,
                                            format="%.1f"
                                        )
                                        # Only filter rows where the value is NOT NaN; keep NaN rows unfiltered
                                        mask = (df_filtered_by_sliders[r_data].isna()) | (
                                            (df_filtered_by_sliders[r_data] >= slider_min) & (df_filtered_by_sliders[r_data] <= slider_max)
                                        )
                                        df_filtered_by_sliders = df_filtered_by_sliders[mask]
                                    else:
                                        st.info(f"Kolumn {r_data} saknas i data.")

                else:
                    st.info("Inga 'trend'-kolumner hittades bland 'rank_Score'-kolumner för filtrering.")
                with col_filter_mid:
                    st.markdown("###### Filtrera för kategori Senaste-rankningar")
                    if latest_columns:
                        for col in latest_columns:
                            with st.container(border=True,key=f"container_trend_{col}"):
                                min_val = df_filtered_by_sliders[col].min()
                                max_val = df_filtered_by_sliders[col].max()
                                slider_min = float(min_val)
                                slider_max = float(max_val)
                                if slider_min == slider_max:
                                    slider_max += 0.001
                                current_min, current_max = st.slider(
                                    f"{col.replace('_latest_catRank', ' senaste Rank')}",
                                    min_value=slider_min,
                                    max_value=slider_max,
                                    value=(slider_min, slider_max),
                                    key=f"slider_latest_{col}",
                                    step=1.0,
                                    format="%d"
                                )
                                df_filtered_by_sliders = df_filtered_by_sliders[
                                    (df_filtered_by_sliders[col] >= current_min) &
                                    (df_filtered_by_sliders[col] <= current_max)
                                ]
                                category_name = col.replace("catRank", "ratioRank")
                                # Dynamiskt skapa flikar för varje senaste kategori med nyckeltalsnamn
                                ratio_name = [r for r in category_ratios[category_name]]
                                ratio_name_display = [r.replace("_latest_ratioRank", "") for r in ratio_name] 
                                tab_labels = ['Info'] + ratio_name_display
                                tabs = st.tabs(tab_labels)
                                tabs[0].write(f"Detaljerad filtrering för *nyckeltal* i {category_name.replace('_latest_ratioRank', '')}:")
                                # Lägg till reglage för varje nyckeltalsflik (från index 1 och uppåt)
                                for i, r in enumerate(ratio_name):
                                    with tabs[i+1]:
                                        if r in df_filtered_by_sliders.columns:
                                            min_val = float(df_filtered_by_sliders[r].min())
                                            max_val = float(df_filtered_by_sliders[r].max())
                                            if min_val == max_val:
                                                max_val += 0.001
                                            slider_min, slider_max = st.slider(
                                                f"Filtrera {r.replace('_latest_ratioRank', ' senaste Rank')} ",
                                                min_value=min_val,
                                                max_value=max_val,
                                                value=(min_val, max_val),
                                                key=f"slider_tab_latest_{category_name}_{r}",
                                                step=1.0,
                                                format="%d"
                                            )
                                            df_filtered_by_sliders = df_filtered_by_sliders[
                                                (df_filtered_by_sliders[r] >= slider_min) &
                                                (df_filtered_by_sliders[r] <= slider_max)
                                            ]
                                        else:
                                            st.info(f"Kolumn {r} saknas i data.")
                                        r_data = f"{r.replace('_latest_ratioRank', '_latest_ratioValue')}"
                                        if r_data in df_filtered_by_sliders.columns:
                                            min_val = float(df_filtered_by_sliders[r_data].min())
                                            max_val = float(df_filtered_by_sliders[r_data].max())
                                            if min_val == max_val:
                                                max_val += 0.001
                                            slider_min, slider_max = st.slider(
                                                f"Filtrera {r_data.replace('_latest_ratioValue', ' senaste Värde')}",
                                                min_value=min_val,
                                                max_value=max_val,
                                                value=(min_val, max_val),
                                                key=f"slider_tab_latest_{r_data}",
                                                step=0.1,
                                                format="%.1f"
                                            )
                                            # Only filter rows where the value is NOT NaN; keep NaN rows unfiltered
                                            mask = (df_filtered_by_sliders[r_data].isna()) | (
                                                (df_filtered_by_sliders[r_data] >= slider_min) & (df_filtered_by_sliders[r_data] <= slider_max)
                                            )
                                            df_filtered_by_sliders = df_filtered_by_sliders[mask]
                                        else:
                                            st.info(f"Kolumn {r_data} saknas i data.")
                    else:
                        st.info("Inga 'senaste'-kolumner hittades bland 'rank_Score'-kolumner för filtrering.")
                with col_filter_right:
                    st.markdown("###### Filtrera för kategori ttm-rankningar")
                    if ttm_columns:
                        for col in ttm_columns:
                            with st.container(border=True,key=f"container_trend_{col}"):
                                min_val = df_filtered_by_sliders[col].min()
                                max_val = df_filtered_by_sliders[col].max()
                                slider_min = float(min_val)
                                slider_max = float(max_val)
                                if slider_min == slider_max:
                                    slider_max += 0.001
                                current_min, current_max = st.slider(
                                    f"{col.replace('_ttm_catRank', ' ttm Rank')}",
                                    min_value=slider_min,
                                    max_value=slider_max,
                                    value=(slider_min, slider_max),
                                    key=f"slider_ttm_{col}",
                                    step=1.0,
                                    format="%d"
                                )
                                df_filtered_by_sliders = df_filtered_by_sliders[
                                    (df_filtered_by_sliders[col] >= current_min) &
                                    (df_filtered_by_sliders[col] <= current_max)
                                ]
                                category_name = col.replace("catRank", "ratioRank")
                                # Dynamiskt skapa flikar för varje ttm kategori med nyckeltalsnamn
                                ratio_name = [r for r in category_ratios[category_name]]
                                ratio_name_display = [r.replace("_ttm_ratioRank", "") for r in ratio_name] 
                                tab_labels = ['Info'] + ratio_name_display
                                tabs = st.tabs(tab_labels)
                                tabs[0].write(f"Detaljerad filtrering för *nyckeltal* i {category_name.replace('_ttm_ratioRank', '')}:")
                                # Lägg till reglage för varje nyckeltalsflik (från index 1 och uppåt)
                                for i, r in enumerate(ratio_name):
                                    with tabs[i+1]:
                                        if r in df_filtered_by_sliders.columns:
                                            min_val = float(df_filtered_by_sliders[r].min())
                                            max_val = float(df_filtered_by_sliders[r].max())
                                            if min_val == max_val:
                                                max_val += 0.001
                                            slider_min, slider_max = st.slider(
                                                f"Filtrera {r.replace('_ttm_ratioRank', ' ttm Rank')} ",
                                                min_value=min_val,
                                                max_value=max_val,
                                                value=(min_val, max_val),
                                                key=f"slider_tab_ttm_{category_name}_{r}",
                                                step=1.0,
                                                format="%d"
                                            )
                                            df_filtered_by_sliders = df_filtered_by_sliders[
                                                (df_filtered_by_sliders[r] >= slider_min) &
                                                (df_filtered_by_sliders[r] <= slider_max)
                                            ]
                                        else:
                                            st.info(f"Kolumn {r} saknas i data.")
                                        r_data = f"{r.replace('_ttm_ratioRank', '_ttm_ratioValue')}"
                                        if r_data in df_filtered_by_sliders.columns:
                                            min_val = float(df_filtered_by_sliders[r_data].min())
                                            max_val = float(df_filtered_by_sliders[r_data].max())
                                            if min_val == max_val:
                                                max_val += 0.001
                                            slider_min, slider_max = st.slider(
                                                f"Filtrera {r_data.replace('_ttm_ratioValue', ' ttm Värde')}",
                                                min_value=min_val,
                                                max_value=max_val,
                                                value=(min_val, max_val),
                                                key=f"slider_tab_ttm_{r_data}",
                                                step=0.1,
                                                format="%.1f"
                                            )
                                            # Only filter rows where the value is NOT NaN; keep NaN rows unfiltered
                                            mask = (df_filtered_by_sliders[r_data].isna()) | (
                                                (df_filtered_by_sliders[r_data] >= slider_min) & (df_filtered_by_sliders[r_data] <= slider_max)
                                            )
                                            df_filtered_by_sliders = df_filtered_by_sliders[mask]
                                        else:
                                            st.info(f"Kolumn {r_data} saknas i data.")
                    else:
                        st.info("Inga 'ttm'-kolumner hittades bland 'rank_Score'-kolumner för filtrering.")
        with st.expander("Välj eller uteslut från sektor eller lista (klicka på färgade 'pills')"):
            col_lista, col_sektor = st.columns(2, gap='medium', border=True)
            with col_lista:
                if 'Lista' in df_filtered_by_sliders.columns:

                    # Use pills for selection, all enabled by default
                    lista_selected = st.pills(
                        "Välj/uteslut Lista:",
                        options=unique_values_lista,
                        default=unique_values_lista,
                        selection_mode='multi',
                        key="segmented_lista"
                    )
                    # Filter df_filtered_by_sliders by selected Lista values
                    if lista_selected:
                        df_filtered_by_sliders = df_filtered_by_sliders[df_filtered_by_sliders['Lista'].isin(lista_selected)]
                    else:
                        df_filtered_by_sliders = df_filtered_by_sliders.iloc[0:0]  # Show nothing if none selected
            with col_sektor:
                # --- Sektor toggles for bubble plot ---
                if 'Sektor' in df_filtered_by_sliders.columns:

                    # Use st.pills for multi-select, all enabled by default
                    sektor_selected = st.pills(
                        "Välj/uteslut Sektor:",
                        options=unique_values_sector,
                        default=unique_values_sector,
                        selection_mode='multi',
                        key="pills_sektor"
                    )
                    # Filter df_filtered_by_sliders by selected Sektor values
                    if sektor_selected:
                        df_filtered_by_sliders = df_filtered_by_sliders[df_filtered_by_sliders['Sektor'].isin(sektor_selected)]
                    else:
                        df_filtered_by_sliders = df_filtered_by_sliders.iloc[0:0]  # Show nothing if none selected

    # =============================
    # FILTERED RESULTS AND BUBBLE PLOT
    # =============================
    st.markdown("<br>", unsafe_allow_html=True) # Lägger till tre radbrytningar
    st.write(f"✅ Data loaded: {df_filtered_by_sliders.shape[0]} aktier, {df_filtered_by_sliders.shape[1]} kolumner")

    with st.container(border=True, key="filtered_results"):
        # Get the number of stocks after filtering by sliders
        #num_filtered_stocks = len(df_display)
        st.subheader(f"🎉 Träffar: {df_filtered_by_sliders.shape[0]} aktier som matchar dina filter!")

        with st.expander('🛟 **Hjälp med filtreringsresultat** (Klicka för att visa)', expanded=False):
                                        st.markdown(
                                                """
                                                **Dina filtrerade aktier – nu kör vi!**  

                                                **📊 Bubbelplotten:**  
                                                • Varje bubbla = en aktie som matchar dina filter  
                                                • Storlek = marknadsvärde, färg = börs-lista  
                                                • Anpassa axlarna för att hitta dolda mönster  
                                                • Toggle tickers på/av för renare vy  

                                                **📋 Resultattabellen:**  
                                                • 'Välj' → Djupdykning i en aktie (grafer + analys)  
                                                • 'Shortlist' → Lägg till i din bevakningslista  
                                                • Sortering: Klicka kolumnnamn för stigande/fallande  
                                                • Antal rader: Justera med segmentreglaget  

                                                **⭐ Bevakningslistan:**  
                                                • Samlar dina utvalda aktier  
                                                • Ladda ner som CSV för vidare analys  
                                                • Spara som portfölj för framtida filtrering  
                                                • Perfect för att hålla koll på favoriter  

                                                **💾 Portföljhantering:**  
                                                • Spara din shortlist som namngiven portfölj  
                                                • Ladda tidigare sparade portföljer som filter  
                                                • Kombinera portföljfilter med andra filter  

                                                **🔬 Detaljanalys:**  
                                                När du väljer en aktie får du: kurscharts, tillväxtgrafer, ranking breakdown och teknisk analys.  

                                                **💡 Pro-tips:** Kombinera filter → Analysera bubblor → Shortlista kandidater → Djupdykning per aktie!  
                                                """
                    )

        # bubble plot
        with st.container(border=True, key="bubble_plot_container"):
            show_tickers = st.toggle('Visa tickers i bubbelplotten', value=True)
            if 'market_cap' in df_filtered_by_sliders.columns:
                df_filtered_by_sliders['market_cap_MSEK'] = (df_filtered_by_sliders['market_cap'] / 1_000_000).round().astype('Int64').map(lambda x: f"{x:,}".replace(",", " ") + " MSEK" if pd.notna(x) else "N/A")
            
            if len(df_filtered_by_sliders) > 0:
                # Assign fixed colors to Lista values using all possible values from the full dataset
                lista_color_map = {
                    'Large Cap': '#1f77b4',
                    'Mid Cap': '#ff7f0e',
                    'Small Cap': '#2ca02c',
                    'First North': '#d62728',
                    'Other': '#9467bd'
                }
                if 'Lista' in df_new_ranks.columns:
                    # Get all unique Lista values from the full dataset for stable color mapping
                    all_lista = df_new_ranks['Lista'].dropna().unique().tolist()
                    plotly_colors = px.colors.qualitative.Plotly
                    for i, lista in enumerate(all_lista):
                        if lista not in lista_color_map:
                            lista_color_map[lista] = plotly_colors[i % len(plotly_colors)]
                    # Use the full color map, but only show legend for filtered values
                    color_discrete_map = {k: v for k, v in lista_color_map.items()}
                else:
                    color_discrete_map = None

                # --- Robust handling of NaN values for bubble plot ---
                # Drop rows with NaN in required columns for the plot
                required_cols = ['Trend_clusterRank', 'Latest_clusterRank', 'TTM_clusterRank']
                # Let user decide which two dimensions to plot using st.segmented_control
                axis_options = [
                    ('Trend_clusterRank', 'Latest_clusterRank'),
                    ('Trend_clusterRank', 'TTM_clusterRank'),
                    ('Latest_clusterRank', 'TTM_clusterRank')
                ]
                axis_labels = [
                    'Trend vs Senaste',
                    'Trend vs TTM',
                    'Senaste vs TTM'
                ]
                selected_axis = st.segmented_control(
                    'Välj axlar för bubbelplotten:',
                    options=axis_labels,
                    selection_mode='single',
                    default=axis_labels[0],
                    key='bubble_axis_selector'
                )
                # Map selection to axis columns
                axis_map = dict(zip(axis_labels, axis_options))
                x_col, y_col = axis_map[selected_axis]
                plot_required_cols = [x_col, y_col]
                if 'Lista' in df_filtered_by_sliders.columns:
                    plot_required_cols.append('Lista')
                plot_df = df_filtered_by_sliders.dropna(subset=plot_required_cols, how='any').copy()
                # Handle market_cap for size
                if 'market_cap' in plot_df.columns:
                    size_raw = plot_df['market_cap'].fillna(20)
                    size = size_raw
                else:
                    size = [20] * len(plot_df)

                if len(plot_df) > 0:
                    bubble_fig = px.scatter(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        color='Lista' if 'Lista' in plot_df.columns else None,
                        color_discrete_map=color_discrete_map,
                        hover_name=plot_df.index if show_tickers else None,
                        text=plot_df.index if show_tickers else None,
                        size=size_raw, # if 'market_cap' in plot_df.columns else [20]*len(plot_df),
                        hover_data={},
                        labels={
                            x_col: get_display_name(x_col),
                            y_col: get_display_name(y_col),
                            'Lista': get_display_name('Lista'),
                            #'hover_summary': 'Summary',
                            'size': 'Market Cap'
                        },
                        title='Bubbeldiagram',
                        width=900,
                        height=600
                    )
                    bubble_fig.update_layout(
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                            title_text=None  # Hide the legend title
                        ),
                        showlegend=True
                    )
                    bubble_fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
                    if show_tickers:
                        bubble_fig.update_traces(textposition='top center')
                    st.plotly_chart(bubble_fig, config={"displayModeBar": False}, use_container_width=True)
                else:
                    st.info('No stocks in the selected score range (after removing rows with saknade värden).')
            else:
                st.info('No stocks in the selected score range.')


        # =============================
        # MAIN TABLE DISPLAY
        # =============================
        with st.container(border=True, key="main_table_container"):
            st.markdown("##### Resultattabell med filtrerade aktier")
                        # get input from user on how many stocks they like to see
            num_stocks_options = ["5", "10", "20", "50", "Alla"]

            selected_num_stocks = st.segmented_control(
                "Här kan du välja att begränsa antal aktier i resultat-tabellen",
                options=num_stocks_options,
                selection_mode='single',
                default="Alla",
                key="num_stocks_input"
            )
            # filter the DataFrame based on the selected number of stocks
            if selected_num_stocks != "Alla":
                df_filtered_by_sliders = df_filtered_by_sliders.head(int(selected_num_stocks))
            # Create a DataFrame for display in the main table.
            # This DataFrame is now based on the slider-filtered data
            # and contains only 'rank_Score' columns, keeping the Ticker as index.
            df_display = df_filtered_by_sliders[rank_score_columns].copy() # Keep index
            #df_display.sort_index(inplace=True)

            # Rename the rank_Score columns for display
            # Create a dictionary for renaming using get_display_name
            rename_mapping = {col: get_display_name(col) for col in rank_score_columns}
            df_display.rename(columns=rename_mapping, inplace=True)

            # Add a "Välj" column for plotting the graph
            # Initialize all checkboxes to False
            df_display['Välj'] = False

            # Add a "Shortlist" column to save stocks
            # Initialize all checkboxes to False
            df_display['Shortlist'] = False

            cols = df_display.columns.tolist()
            cols.insert(0, cols.pop(cols.index('Lista')))
            cols.insert(0, cols.pop(cols.index('Agg. Rank ttm diff')))
            cols.insert(0, cols.pop(cols.index('Agg. Rank sen. året'))) 
            cols.insert(0, cols.pop(cols.index('Agg. Rank trend 4 år'))) 
            cols.insert(0, cols.pop(cols.index('Shortlist'))) 
            cols.insert(0, cols.pop(cols.index('Välj')))
              # Move 'Lista' to the front
            df_display = df_display[cols]  # Reorder columns
            # Update rank_score_columns to reflect the new names for shortlist display
            display_rank_score_columns = df_display.columns.tolist()

            df_display = df_display[display_rank_score_columns]
            edited_df = st.data_editor(
                df_display,
                width="stretch",
                hide_index=False, # Set to False to always show the index (Ticker)
                column_config={
                    # No need to configure "Ticker" as it's now the index
                    "Välj": st.column_config.CheckboxColumn(
                        "Välj", # Header for the checkbox column to plot
                        help="Select a stock to display its price development",
                        default=False,
                        width="small",
                        pinned=True
                    ),
                    "Shortlist": st.column_config.CheckboxColumn(
                        "Shortlist", # Header for the checkbox column for shortlist
                        help="Lägg till aktien i din bevakningslista",
                        default=False,
                        width="small",
                        pinned=True 
                    ),
                    "Lista": st.column_config.TextColumn(
                        "Lista", # Header for the Lista column",
                        default="",
                        width="small",
                        pinned=False
                    ),
                },
                key="stock_selection_editor" # Unique key to manage state
            )

            # Logic to handle checkbox selection for plotting
            selected_rows_plot = edited_df[edited_df['Välj']]
            #st.info("Markera rutan under 'Välj' för att visa aktiedata. Markera rutan under 'Shortlist' för att lägga till aktien i din bevakningslista.")

            # Ensure only one stock can be selected at a time for plotting.
            if len(selected_rows_plot) > 1:
                st.warning("Endast en aktie kan väljas åt gången för prisutveckling. Visar graf för den första valda aktien.")
                selected_stock_ticker = selected_rows_plot.index[0] # Access Ticker from index
            elif len(selected_rows_plot) == 1:
                selected_stock_ticker = selected_rows_plot.index[0] # Access Ticker from index
            else:
                selected_stock_ticker = None # No stock selected for plotting

            # Create a dict for the selected stock's data for easy access
            selected_stock_dict = None
            if selected_stock_ticker is not None:
                if ENABLE_DEBUG_LOGGING:
                    print(f"DEBUG: About to create selected_stock_dict for ticker: '{selected_stock_ticker}' (type: {type(selected_stock_ticker)})")
                try:
                    selected_stock_dict = df_new_ranks.loc[selected_stock_ticker].to_dict()
                    # DEBUG: Check QuarterDiff immediately after dict creation
                    if ENABLE_DEBUG_LOGGING:
                        print(f"DEBUG: selected_stock_dict created for {selected_stock_ticker}")
                        print(f"DEBUG: QuarterDiff in dict: {selected_stock_dict.get('QuarterDiff', 'NOT_FOUND')} (type: {type(selected_stock_dict.get('QuarterDiff', None))})")
                except KeyError as e:
                    if ENABLE_DEBUG_LOGGING:
                        print(f"ERROR: selected_stock_ticker '{selected_stock_ticker}' not found in df_new_ranks index: {e}")
                        print(f"Available tickers: {list(df_new_ranks.index[:5])}...")
                    selected_stock_dict = None
                except Exception as e:
                    if ENABLE_DEBUG_LOGGING:
                        print(f"ERROR: Failed to create selected_stock_dict for ticker '{selected_stock_ticker}': {e}")
                    selected_stock_dict = None
            # Logic to handle Shortlist
            shortlisted_stocks = edited_df[edited_df['Shortlist']]
            with st.container(border=True, key="shortlist_container"):
                st.markdown("##### Bevakningslista (Shortlist)")

                if not shortlisted_stocks.empty:
                    df_display = shortlisted_stocks.copy()
                    # Merge with new ranks for additional info (ensure index is Ticker)
                    df_display = df_display.merge(
                        df_new_ranks[['Lista', 'Sektor']],
                        left_index=True, right_index=True, how='left'
                    )
                    # Display only Ticker (index) and the renamed rank_Score columns for shortlist
                    download_columns = ['Lista', 'Sektor'] + [
                        col for col in display_rank_score_columns if col not in ['Shortlist', 'Välj']
                    ]
                    # Only keep columns that exist in df_display
                    download_columns = [col for col in download_columns if col in df_display.columns]

                    st.dataframe(
                        df_display[download_columns], # Ticker is already the index
                        hide_index=False,
                        width="stretch"
                    )

                    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    file_name = f"shortlist_{current_time}.csv"
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.download_button(
                            "📥 Ladda ner som CSV",
                            data=df_display[download_columns].to_csv(),
                            file_name=file_name,
                            mime="text/csv",
                            width="stretch"
                        )
                    
                    with col2:
                        if st.button("💾 Spara som portfölj", width="stretch"):
                            st.session_state.show_save_portfolio = True
                    
                    # Portfolio save dialog
                    if st.session_state.get('show_save_portfolio', False):
                        with st.form("save_portfolio_form"):
                            st.write("**Spara bevakningslista som portfölj**")
                            portfolio_name = st.text_input("Portföljnamn:", value=f"Portfölj {current_time}")
                            portfolio_description = st.text_area("Beskrivning (valfritt):", height=100)
                            
                            col_save, col_cancel = st.columns([1, 1])
                            with col_save:
                                if st.form_submit_button("Spara portfölj", width="stretch"):
                                    if portfolio_name.strip():
                                        # Get current filter settings (you'll need to implement this)
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
                else:
                    pass

    st.markdown("<br>", unsafe_allow_html=True) # Lägger till tre radbrytningar

    with st.container(border=True, key="stock_details_container"):
        st.subheader("🔍 **Djupdykning i din valda aktie**")
        with st.expander("🛟 **Hjälp om aktieinformation** (Klicka för att visa)", expanded=False):
            st.markdown(
                """
                **Djupdykning i din valda aktie – här är guiden:**

                **📋 Grundinfo:**  
                • Ticker, sektor, lista, marknadsvärde  
                • Klicka företagsbeskrivningen för hela storyn  

                **📈 Tillväxtanalys (CAGR):**  
                • Stapeldiagram för 4-årsperioden  
                • Expandera för detaljvy med TTM-data  
                • Grönt/rött = bra/dålig TTM-utveckling  

                **💹 Kursutveckling:**  
                • Prisgraf med volym och glidande medelvärden  
                • Justerbar trendlinje (PWLF) med standardavvikelser  
                • SMA-differenser i procent  

                **🏆 Ranking breakdown:**  
                • Sammanvägd: Totalbild per kategori  
                • Detaljerad: Varje nyckeltal med trendutveckling  
                • Färgkodade staplar: Röd = svag, Grön = stark  

                **🎯 Ratio 2 Rank:**  
                • Scatterplot: Ditt bolag vs konkurrenterna  
                • Röd korslinje = din valda aktie  
                • Bakgrundsfärger = ranking-zoner  

                **💡 Smart-tips:** Datadump längst ner för full transparens!  
                """
            )
        if selected_stock_dict is not None and selected_stock_ticker is not None:
            st.subheader(f"**{selected_stock_dict['Name'] if 'Name' in selected_stock_dict else 'N/A'}**")

        with st.container(border=True, key="stock_details"):
            if selected_stock_ticker:
                #st.subheader(f"Kort info om: {selected_stock_dict['Name'] if 'Name' in selected_stock_dict else 'N/A'}")
                selected_stock_lista = selected_stock_dict['Lista'] if 'Lista' in selected_stock_dict else 'N/A'
                selected_stock_sektor = selected_stock_dict['Sektor'] if 'Sektor' in selected_stock_dict else 'N/A'
                
                # DEBUG: Check QuarterDiff value before conversion
                if ENABLE_DEBUG_LOGGING:
                    quarter_diff_value = selected_stock_dict.get('QuarterDiff', 'N/A')
                    print(f"DEBUG: QuarterDiff value for {selected_stock_ticker}: {quarter_diff_value} (type: {type(quarter_diff_value)})")
                
                # More robust conversion with error handling
                try:
                    quarter_diff_value = selected_stock_dict.get('QuarterDiff', 'N/A')
                    if quarter_diff_value == 'N/A' or quarter_diff_value is None or pd.isna(quarter_diff_value):
                        selected_stock_ttm_offset = 0
                    else:
                        selected_stock_ttm_offset = int(quarter_diff_value)
                    if ENABLE_DEBUG_LOGGING:
                        print(f"DEBUG: Successfully converted QuarterDiff to: {selected_stock_ttm_offset}")
                except (ValueError, TypeError) as e:
                    if ENABLE_DEBUG_LOGGING:
                        print(f"ERROR: Failed to convert QuarterDiff '{quarter_diff_value}' to int: {e}")
                        print(f"ERROR: selected_stock_dict keys: {list(selected_stock_dict.keys())}")
                        print(f"ERROR: selected_stock_dict QuarterDiff: {selected_stock_dict.get('QuarterDiff', 'NOT_FOUND')}")
                    selected_stock_ttm_offset = 0
                left_col, right_col = st.columns([2,3], gap='medium', border=False)
                with left_col:
                    
                    st.write(f"**Ticker:**   \n{selected_stock_ticker}")
                    st.write(f"**Lista:**   \n{selected_stock_lista}")
                    st.write(f"**Sektor:**   \n{selected_stock_sektor}")
                    st.write(f"**Marknadsvärde:**   \n{human_format(selected_stock_dict['market_cap'] if 'market_cap' in selected_stock_dict else 'N/A')}")
                    st.write(f"**Senaste årsrapport:**   \n{selected_stock_dict['LatestReportDate_Y'] if 'LatestReportDate_Y' in selected_stock_dict else 'N/A'}")
                    st.write(f"**Senaste kvartalsrapport:**   \n{selected_stock_dict['LatestReportDate_Q'] if 'LatestReportDate_Q' in selected_stock_dict else 'N/A'}")
                    st.write(f"**Antal kvartalsrapporter efter årsrapport:**   \n{selected_stock_ttm_offset}")
                with right_col:
                    #st.subheader("Företagsbeskrivning")
                    LongBusinessSummary = selected_stock_dict['LongBusinessSummary'] if 'LongBusinessSummary' in selected_stock_dict else 'N/A'
                    with st.popover(f"{LongBusinessSummary[0:500]}...",width="stretch"):
                        st.write(LongBusinessSummary if LongBusinessSummary else "Ingen lång företagsbeskrivning tillgänglig för denna aktie.")
                                # --- Plot annual dividends for selected_stock_ticker ---
                if selected_stock_ticker is not None and 'df_dividends' in locals():
                    dividends_df = df_dividends[df_dividends.index == selected_stock_ticker]
                    if not dividends_df.empty:
                        # Ensure Year and Value columns exist
                        if 'Year' in dividends_df.columns and 'Value' in dividends_df.columns:
                            # Convert Year to datetime for proper sorting
                            dividends_df = dividends_df.copy()
                            #dividends_df['Year'] = pd.to_datetime(dividends_df['Year'], errors='coerce')
                            dividends_df.sort_values('Year', inplace=True)
                            fig_div = go.Figure(go.Bar(
                                x=dividends_df['Year'],
                                y=dividends_df['Value'],
                                marker_color='gold',
                                text=[f"{v:.2f}" for v in dividends_df['Value']],
                                textposition='auto',
                            ))
                            fig_div.update_layout(
                                title=f"Utdelningar för {selected_stock_ticker}",
                                xaxis_title="År",
                                yaxis_title="SEK",
                                height=150,
                                margin=dict(l=10, r=10, t=40, b=10),
                                xaxis=dict(type='category')
                            )
                            st.plotly_chart(fig_div, config={"displayModeBar": False}, use_container_width=True, key=f"dividends_bar_{selected_stock_ticker}")
                        else:
                            st.info(f"Dividend-data saknar nödvändiga kolumner ('Year', 'Value') för {selected_stock_ticker}.")
                    else:
                        st.info(f"Ingen utdelningsdata för {selected_stock_ticker}.")
        if selected_stock_ticker is not None:
            with st.popover(f"Datadump av {selected_stock_ticker}", width="stretch"):
                st.write(f"Datadump av {selected_stock_ticker}")
                # Convert dataframe to consistent types for display
                display_df = df_new_ranks.loc[selected_stock_ticker].to_frame()
                # Convert numeric columns to float, keep others as strings
                for col in display_df.columns:
                    if display_df[col].dtype == 'object':
                        # Try to convert to numeric, keep as string if it fails
                        try:
                            display_df[col] = pd.to_numeric(display_df[col], errors='ignore')
                        except (ValueError, TypeError):
                            pass
                st.dataframe(display_df)
        with st.container(border=True, key="cagr_container"):
            st.subheader("📈 Tillväxthistorik senaste 4 åren")
            # Only show the following sections if a stock is selected
            if selected_stock_dict is not None and selected_stock_ticker is not None:
                # Bar plot for all cagr columns for selected_stock_ticker using selected_stock_dict
                cagr_cols = [col for col in allCols_AvgGrowth]
                if cagr_cols:
                    cagr_values = [float(selected_stock_dict.get(col, float('nan'))) for col in cagr_cols]
                    bar_colors = ['royalblue' for v in cagr_values]
                    bar_text = [
                        "N/A" if pd.isna(v) else f"{v*100:.2f}%"
                        for v in cagr_values
                    ]
                    y_values = [v*100 if not pd.isna(v) else None for v in cagr_values]
                    x_labels = [get_display_name(col) for col in cagr_cols]
                    fig_cagr = go.Figure(go.Bar(
                        x=x_labels,
                        y=y_values,
                        marker_color=bar_colors,
                        text=bar_text,
                        textposition='auto',
                    ))
                    # Add 'Data missing' annotation for each NaN
                    for i, v in enumerate(cagr_values):
                        if pd.isna(v):
                            fig_cagr.add_annotation(
                                x=x_labels[i],
                                y=0,
                                text="Data missing",
                                showarrow=False,
                                font=dict(color="#b30000", size=13),
                                bgcolor="#ffe5e5",
                                bordercolor="#ffcccc",
                                borderwidth=1,
                                yshift=20
                            )
                    fig_cagr.update_layout(
                        title=f"Genomsnittlig förändring över 4 år för {selected_stock_dict['Name']} ({selected_stock_ticker})",
                        xaxis_title="Mått",
                        yaxis_title="Procent",
                        height=350,
                        margin=dict(l=10, r=10, t=40, b=10),
                        yaxis=dict(ticksuffix="%", tickformat=".0f")
                    )
                    st.plotly_chart(fig_cagr, config={"displayModeBar": False}, use_container_width=True, key=f"cagr_bar_{selected_stock_ticker}")
                with st.expander("**📊 Detaljerade tillväxtgrafer + TTM-signaler** (Klicka för att dölja)", expanded=True):
                    def plot_cagr_bar(df, selected_stock_ticker, base_ratio, key_prefix, ttm_q_offset, ttm_value, ttm_diff_value,higher_is_better):
                        year_cols = [col for col in df.columns if col.startswith(base_ratio + '_year_')]
                        year_cols = [col for col in year_cols if not pd.isna(df.loc[selected_stock_ticker, col])]
                        year_cols_sorted = sorted(year_cols, key=lambda x: int(x.split('_')[-1]), reverse=False)
                        year_cols_last4 = year_cols_sorted[-4:]
                        higher_is_better = True if higher_is_better is None else higher_is_better
                        if year_cols_last4:
                            values = df.loc[selected_stock_ticker, year_cols_last4].values.astype(float)
                            years = [int(col.split('_')[-1]) for col in year_cols_last4]
                            fig = go.Figure()
                            colors = ['lightblue'] * (len(years) - 1) + ['royalblue']
                            bar_text = [f"{human_format(v)}" for v in values]
                            if ttm_value is not None:
                                colors.append('gold')
                                values = np.append(values, ttm_value)
                                ttm_label = f'ttm (+{ttm_q_offset}Q)'
                                years.append(ttm_label)
                            fig.add_trace(go.Bar(x=years, y=values, marker_color=colors, name=base_ratio,text=bar_text, showlegend=False))
                            fig.add_trace(go.Scatter(
                                x=[years[0], years[-1 if ttm_value is None else -2]],
                                y=[values[0], values[-1 if ttm_value is None else -2]],
                                mode='lines',
                                name='Trend',
                                line=dict(color='#888888', dash='dot', width=6),
                                showlegend=False
                            ))
                                    # add trendline between last full year and ttm
                            if ttm_value is not None and not pd.isna(ttm_value):
                                fig.add_trace(go.Scatter(
                                    x=[years[-2], ttm_label],
                                    y=[values[-2], ttm_value],
                                    mode='lines',
                                    name='Trend',
                                    line=dict(color="#0D0D0D", dash='dot', width=6),
                                    showlegend=False
                                ))
                            if ttm_value is not None and not pd.isna(ttm_value) and ttm_diff_value is not None and not pd.isna(ttm_diff_value):
                                # Use higher_is_better to determine color
                                if higher_is_better:
                                    color = "green" if ttm_diff_value >= 0 else "red"
                                else:
                                    color = "red" if ttm_diff_value >= 0 else "green"
                                y_shift = 20 if ttm_value >= 0 else -20

                                fig.add_annotation(
                                    x=ttm_label,
                                    y=ttm_value,
                                    text=f"{human_format(ttm_value)}",
                                    showarrow=False,
                                    font=dict(color=color, size=14, family="Arial"),
                                    yshift=y_shift
                                ) # annotation for pct change for ttm
                            fig.update_layout(title=f"{base_ratio}", 
                                            height=250, 
                                            yaxis_title="SEK",
                                            margin=dict(l=10, r=10, t=30, b=10), 
                                            showlegend=False,
                                            xaxis=dict(type='category'))
                            st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True, key=f"{key_prefix}_{base_ratio}_cagr_bar")
                    left_col, middle_col, right_col = st.columns(3, gap='medium', border=False)
                    base_ratio_left = allCols_AvgGrowth[0].replace("_AvgGrowth", "")  # Use the first column as base for left side
                    base_ratio_middle = allCols_AvgGrowth[1].replace("_AvgGrowth", "")  # Use the second column as base for middle
                    base_ratio_right = allCols_AvgGrowth[2].replace("_AvgGrowth", "")  # Use the third column as base for right

                    with left_col:
                        ttm_col = f"{base_ratio_left}_ttm"
                        ttm_value = df_new_ranks.loc[selected_stock_ticker, ttm_col] if ttm_col in df_new_ranks.columns else None
                        ttm_diff = f"{base_ratio_left}_ttm_diff"
                        ttm_diff_value = df_new_ranks.loc[selected_stock_ticker, ttm_diff] if ttm_diff in df_new_ranks.columns else None
                        plot_cagr_bar(df_new_ranks, selected_stock_ticker, base_ratio_left, "left", selected_stock_ttm_offset, ttm_value, ttm_diff_value, higher_is_better=True)
                    with middle_col:
                        ttm_col = f"{base_ratio_middle}_ttm"
                        ttm_value = df_new_ranks.loc[selected_stock_ticker, ttm_col] if ttm_col in df_new_ranks.columns else None
                        #st.write(f"ttm_value: {ttm_value}")
                        ttm_diff = f"{base_ratio_middle}_ttm_diff"
                        ttm_diff_value = df_new_ranks.loc[selected_stock_ticker, ttm_diff] if ttm_diff in df_new_ranks.columns else None
                        #st.write(f"ttm_diff_value: {ttm_diff_value}")
                        #st.write(f"selected_stock_ttm_offset: {selected_stock_ttm_offset}")
                        plot_cagr_bar(df_new_ranks, selected_stock_ticker, base_ratio_middle, "middle", selected_stock_ttm_offset, ttm_value, ttm_diff_value, higher_is_better=True)
                    with right_col:
                        ttm_col = f"{base_ratio_right}_ttm"
                        ttm_value = df_new_ranks.loc[selected_stock_ticker, ttm_col] if ttm_col in df_new_ranks.columns else None
                        ttm_diff = f"{base_ratio_right}_ttm_diff"
                        ttm_diff_value = df_new_ranks.loc[selected_stock_ticker, ttm_diff] if ttm_diff in df_new_ranks.columns else None
                        plot_cagr_bar(df_new_ranks, selected_stock_ticker, base_ratio_right, "right", selected_stock_ttm_offset, ttm_value, ttm_diff_value, higher_is_better=True)

        with st.container(border=True, key="stock_price_trend_container"):
            st.subheader("💹 Kursutveckling & Smart Trendanalys")

            if selected_stock_ticker:
                # Add slider for PWLF
                label = "Antal linjesegment för trendlinje"
                linjesegments =[1, 2, 3, 4, 5]
                num_segments = st.segmented_control(label, linjesegments, selection_mode='single', default=1, key="pwlf_slider")
                price_file_path = CSV_PATH / config["price_data_file"]
                if price_file_path.exists():
                    df_price_all = pd.read_csv(price_file_path)
                    df_price = df_price_all[df_price_all['Ticker'] == selected_stock_ticker].copy()
                    df_price['Date'] = pd.to_datetime(df_price['Date']) # Convert 'Date' to datetime object
                    df_price = add_moving_averages(df_price)
                    # PWLF calculation
                    x_hat = None
                    y_hat = None
                    std_devs = None
                    if len(df_price) > num_segments:
                        x = np.arange(len(df_price['Date']))
                        y = df_price['Close'].values

                        my_pwlf = pwlf.PiecewiseLinFit(x, y)
                        # fit the data for a given number of line segments
                        res = my_pwlf.fit(num_segments)
                        # predict for the determined breaks
                        x_hat = np.linspace(x.min(), x.max(), 100)
                        y_hat = my_pwlf.predict(x_hat)

                        # Calculate standard deviation of residuals (difference between actual and fitted)
                        y_fitted = my_pwlf.predict(x)
                        residuals = y - y_fitted
                        std_devs = [np.std(residuals) * i for i in [1, 2, 3]]

                    # Create Plotly figure
                    fig = go.Figure()

                    # Add Close price
                    if 'Close' in df_price.columns:
                        fig.add_trace(go.Scatter(x=df_price['Date'], y=df_price['Close'],
                            mode='lines', name='Stängningskurs',
                            line=dict(color='blue', width=2)))

                    # Add PWLF trendline to the plot
                    if x_hat is not None and y_hat is not None:
                        # Create a new date range for the predicted values
                        date_range = pd.to_datetime(np.linspace(df_price['Date'].min().value, df_price['Date'].max().value, len(x_hat)))
                        fig.add_trace(go.Scatter(x=date_range, y=y_hat,
                                    mode='lines', name='Trendlinje',
                                    line=dict(color='orange', width=3, dash='dash')))
                        # Add dotted lines for +- 1, 2, 3 standard deviations from trendline
                        if std_devs is not None:
                            for i, std in enumerate(std_devs, 1):
                                fig.add_trace(go.Scatter(
                                    x=date_range, y=y_hat + std,
                                    mode='lines',
                                    name=f'+{i}σ',
                                    line=dict(color='gray', width=1, dash='dot'),
                                    showlegend=True
                                ))
                                fig.add_trace(go.Scatter(
                                    x=date_range, y=y_hat - std,
                                    mode='lines',
                                    name=f'-{i}σ',
                                    line=dict(color='gray', width=1, dash='dot'),
                                    showlegend=True
                                ))

                    # Add SMA_short
                    if 'SMA_short' in df_price.columns:
                        fig.add_trace(go.Scatter(x=df_price['Date'], y=df_price['SMA_short'],
                            mode='lines', name=get_display_name('SMA_short'),
                            line=dict(color='red', width=1, dash='dot')))

                    # Add SMA_medium
                    if 'SMA_medium' in df_price.columns:
                        fig.add_trace(go.Scatter(x=df_price['Date'], y=df_price['SMA_medium'],
                            mode='lines', name=get_display_name('SMA_medium'),
                            line=dict(color='green', width=1, dash='dash')))

                    # Add SMA_long
                    if 'SMA_long' in df_price.columns:
                        fig.add_trace(go.Scatter(x=df_price['Date'], y=df_price['SMA_long'],
                            mode='lines', name=get_display_name('SMA_long'),
                            line=dict(color='purple', width=1, dash='longdash')))

                    # Add Volume as a secondary y-axis
                    if 'Volume' in df_price.columns:
                        fig.add_trace(go.Bar(x=df_price['Date'], y=df_price['Volume'],
                            name='Volym', marker_color='gray', opacity=0.3, yaxis='y2'))

                    # Update layout for the chart
                    fig.update_layout(
                        title=f"Pris & Volym för {selected_stock_dict['Name']} ({selected_stock_ticker})",
                        xaxis_title="Datum",
                        yaxis_title="Pris",
                        hovermode=False,  # Disable all hover interactions
                        legend_title="Legend",
                        height=500,
                        yaxis2=dict(title="Volym", overlaying="y", side="right", showgrid=False),
                        legend=dict(
                            x=0.01,
                            y=0.99,
                            xanchor="left",
                            yanchor="top",
                            bgcolor="rgba(255,255,255,0.7)",
                            bordercolor="rgba(0,0,0,0.1)",
                            borderwidth=1
                        )
                    )

                    # Remove hover for all traces
                    for trace in fig.data:
                        trace.update(hoverinfo="skip", hovertemplate=None)

                    st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)
                else:
                    st.warning(f"Prisdatafil saknas: {price_file_path}. Kontrollera att filen finns i mappen '{CSV_PATH}/'.")

            
            # =============================
            # PERCENTAGE BAR PLOTS
            # =============================
            # Bar plot for all pct_ columns for selected_stock_ticker
            if selected_stock_ticker:
                with st.expander("**SMA differenser (%)** (Klicka för att visa)", expanded=False):
                    pct_cols = [col for col in selected_stock_dict.keys() if col.startswith('pct_')]
                    if pct_cols:
                        pct_values = [float(selected_stock_dict.get(col, float('nan'))) for col in pct_cols]
                        fig_pct = go.Figure(go.Bar(
                            x=[get_display_name(col) for col in pct_cols],
                            y=pct_values,
                            marker_color='royalblue',
                            text=[f"{v:.2f}%" for v in pct_values],
                            textposition='auto',
                        ))
                        fig_pct.update_layout(
                            title=f"Kursutveckling for {selected_stock_ticker}",
                            xaxis_title="Metric",
                            yaxis_title="Percentage",
                            height=350,
                            margin=dict(l=10, r=10, t=40, b=10),
                            yaxis=dict(ticksuffix="%", tickformat=".0f")
                        )
                        st.plotly_chart(fig_pct, config={"displayModeBar": False}, use_container_width=True, key=f"pct_bar_{selected_stock_ticker}")

            
        with st.container(border=True, key="ratios_container"):
            # =============================
            # RANKING FOR SELECTED STOCK
            # =============================
            st.subheader("🏆 Ranking breakdown – Var står bolaget?")
            if not df_filtered_by_sliders.empty and categories and selected_stock_ticker is not None:
                #st.markdown("**Trend senaste 4 åren & Senaste året**")
                clusterRank_trend_items = {col: val for col, val in selected_stock_dict.items() if "_clusterRank" in col and "trend" in col.lower()}
                df_clusterRank_trend = pd.DataFrame.from_dict(clusterRank_trend_items, orient='index', columns=['Trend Rank'])
                df_clusterRank_trend['Kategori']= 'AGG. RANK'
                catRank_trend_items = {col: val for col, val in selected_stock_dict.items() if "_catRank" in col and "trend" in col.lower()}
                df_catRank_trend = pd.DataFrame.from_dict(catRank_trend_items, orient='index', columns=['Trend Rank']).reset_index()
                df_catRank_trend['Kategori'] = df_catRank_trend['index'].str.replace('_trend_catRank','').str.replace('_',' ')
                df_trend_combined = pd.concat([df_catRank_trend, df_clusterRank_trend.reset_index()], ignore_index=True, sort=False)

                clusterRank_latest_items = {col: val for col, val in selected_stock_dict.items() if "_clusterRank" in col and "latest" in col.lower()}
                df_clusterRank_latest = pd.DataFrame.from_dict(clusterRank_latest_items, orient='index', columns=['Latest Rank'])
                df_clusterRank_latest['Kategori']= 'AGG. RANK'
                catRank_latest_items = {col: val for col, val in selected_stock_dict.items() if "_catRank" in col and "latest" in col.lower()}
                df_catRank_latest = pd.DataFrame.from_dict(catRank_latest_items, orient='index', columns=['Latest Rank']).reset_index()
                df_catRank_latest['Kategori'] = df_catRank_latest['index'].str.replace('_latest_catRank','').str.replace('_',' ')
                df_latest_combined = pd.concat([df_catRank_latest, df_clusterRank_latest.reset_index()], ignore_index=True, sort=False)

                clusterRank_ttm_items = {col: val for col, val in selected_stock_dict.items() if "_clusterRank" in col and "ttm" in col.lower()}
                df_clusterRank_ttm = pd.DataFrame.from_dict(clusterRank_ttm_items, orient='index', columns=['TTM Rank'])
                df_clusterRank_ttm['Kategori']= 'AGG. RANK'
                catRank_ttm_items = {col: val for col, val in selected_stock_dict.items() if "_catRank" in col and "ttm" in col.lower()}
                df_catRank_ttm = pd.DataFrame.from_dict(catRank_ttm_items, orient='index', columns=['TTM Rank']).reset_index()
                df_catRank_ttm['Kategori'] = df_catRank_ttm['index'].str.replace('_ttm_catRank','').str.replace('_',' ')
                df_ttm_combined = pd.concat([df_catRank_ttm, df_clusterRank_ttm.reset_index()], ignore_index=True, sort=False)

                # Merge the trend and latest DataFrames on 'Kategori'
                df_catRank_merged = pd.merge(df_trend_combined, df_latest_combined, on='Kategori', suffixes=('_trend', '_latest'))
                df_catRank_merged = pd.merge(df_catRank_merged, df_ttm_combined, on='Kategori', suffixes=('', '_ttm'))
                # -------------------------------------------------------------
                # PROGRESS BARS: LATEST AND TREND RANKINGS
                # -------------------------------------------------------------

                st.dataframe(
                    df_catRank_merged[['Kategori', 'Trend Rank', 'Latest Rank', 'TTM Rank']]
                    .style.map(color_progress, subset=['Trend Rank', 'Latest Rank', 'TTM Rank']),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Trend Rank": st.column_config.ProgressColumn(
                                "Trend Rank",
                                help="Rankingvärde (0-100)",
                                min_value=0,
                                max_value=100,
                                format="%.1f"
                            ),
                        "Latest Rank": st.column_config.ProgressColumn(
                                "Latest Rank",
                                help="Rankingvärde (0-100)",
                                min_value=0,
                                max_value=100,
                                format="%.1f"
                            ),
                        "TTM Rank": st.column_config.ProgressColumn(
                                "TTM Rank",
                                help="Rankingvärde (0-100)",
                                min_value=0,
                                max_value=100,
                                format="%.1f"
                            ),
                    }
                )


                # -------------------------------------------------------------
                # TREND RATIO BREAKDOWN BAR CHARTS
                # -------------------------------------------------------------
                st.markdown('---')
                st.subheader('🔬 Detaljerad ranking per kategori')
                st.markdown("**Trend senaste 4 åren & Senaste året**")
                # Create DataFrames for trend and latest ratio ranks
                ratioRank_latest_items = {col: val for col, val in selected_stock_dict.items() if "_ratioRank" in col and "latest" in col.lower()}
                df_ratioRank_latest = pd.DataFrame.from_dict(ratioRank_latest_items, orient='index', columns=['Rank']).reset_index()
                df_ratioRank_latest['Ratio_name'] = df_ratioRank_latest['index'].str.replace('_latest_ratioRank','')

                ratioRank_trend_items = {col: val for col, val in selected_stock_dict.items() if "_ratioRank" in col and "trend" in col.lower()}
                df_ratioRank_trend = pd.DataFrame.from_dict(ratioRank_trend_items, orient='index', columns=['Rank']).reset_index()
                df_ratioRank_trend['Ratio_name'] = df_ratioRank_trend['index'].str.replace('_trend_ratioRank','')

                ratioRank_ttm_items = {col: val for col, val in selected_stock_dict.items() if "_ratioRank" in col and "ttm" in col.lower()}
                df_ratioRank_ttm = pd.DataFrame.from_dict(ratioRank_ttm_items, orient='index', columns=['Rank']).reset_index()
                df_ratioRank_ttm['Ratio_name'] = df_ratioRank_ttm['index'].str.replace('_ttm_ratioRank','')

                df_ratioRank_merged = pd.merge(df_ratioRank_trend, df_ratioRank_latest, on='Ratio_name', suffixes=('_trend', '_latest'))
                df_ratioRank_merged = pd.merge(df_ratioRank_merged, df_ratioRank_ttm, on='Ratio_name', suffixes=('', '_ttm'))
                df_ratioRank_merged.rename(columns={'Rank_trend': 'Trend Rank', 'Rank_latest': 'Latest Rank', 'Rank': 'TTM Rank'}, inplace=True)
                
                for cat, cat_dict in category_ratios.items():

                    if cat.endswith('trend_ratioRank'):
                        display_cat = cat.replace('_trend_ratioRank', '').replace('_', ' ')
                        # Use a visually distinct box for each category, with extra margin for spacing
                        with st.container(border=True):
                            st.subheader(f"{get_display_name(display_cat)}")
                            st.markdown("**Rank för Trend senaste 4 åren & Senaste året**")
                            st.dataframe(
                                df_catRank_merged[df_catRank_merged['Kategori'] == display_cat][[ 'Trend Rank', 'Latest Rank', 'TTM Rank']].style.map(color_progress, subset=['Trend Rank', 'Latest Rank', 'TTM Rank']),
                                hide_index=True,
                                width="stretch",
                                column_config={
                                    "Latest Rank": st.column_config.ProgressColumn(
                                            "Latest Rank",
                                            help="Rankingvärde (0-100)",
                                            min_value=0,
                                            max_value=100,
                                            format="%.1f",
                                            width="small",
                                        ),
                                    "Trend Rank": st.column_config.ProgressColumn(
                                            "Trend Rank",
                                            help="Rankingvärde (0-100)",
                                            min_value=0,
                                            max_value=100,
                                            format="%.1f",
                                            width="small"
                                        ),
                                    "TTM Rank": st.column_config.ProgressColumn(
                                            "TTM Rank",
                                            help="Rankingvärde (0-100)",
                                            min_value=0,
                                            max_value=100,
                                            format="%.1f",
                                            width="small"
                                        )
                                }
                            )
                            
                            ratios = [ratio for ratio in cat_dict]
                            st.markdown(f"Ingående Nyckeltal för **{get_display_name(display_cat)}** med Rank för *Trend senaste 4 åren*, *Senaste året* samt *TTM* (om tillgänglig)")
                            cols = st.columns(len(ratios), border=True,gap="small") if ratios else []
                            for idx, ratio in enumerate(ratios):
                                base_ratio = ratio.replace('_trend_ratioRank', '')
                                # Load higher_is_better from config if available
                                higher_is_better = True
                                if 'config' in globals() or 'config' in locals():
                                    ratio_defs = config.get('ratio_definitions', {})
                                    if base_ratio in ratio_defs and 'higher_is_better' in ratio_defs[base_ratio]:
                                        higher_is_better = ratio_defs[base_ratio]['higher_is_better']
                                # year cols
                                year_cols = [col for col in df_new_ranks.columns if col.startswith(base_ratio + '_year_')]
                                year_cols = [col for col in year_cols if not pd.isna(df_new_ranks.loc[selected_stock_ticker, col]) and col.split('_')[-1].isdigit()]
                                year_cols_sorted = sorted(year_cols, key=lambda x: int(x.split('_')[-1]), reverse=False)
                                year_cols_last4 = year_cols_sorted[-4:]

                                # quarter cols - look for columns with quarter pattern
                                quarter_cols = [col for col in df_new_ranks.columns if '_quarter_2' in col and base_ratio in col]
                                # Filter for valid quarter columns and non-NaN values
                                def is_valid_quarter(col):
                                    last_part = col.split('_')[-1]
                                    return ('Q' in last_part and any(c.isdigit() for c in last_part) and 
                                           not pd.isna(df_new_ranks.loc[selected_stock_ticker, col]))
                                
                                quarter_cols = [col for col in quarter_cols if is_valid_quarter(col)]
                                
                                # Sort quarters chronologically (e.g., 2024Q4, 2025Q1, 2025Q2)
                                def quarter_sort_key(col):
                                    last_part = col.split('_')[-1]  # e.g., "2025Q1"
                                    if 'Q' in last_part:
                                        year_part = last_part.split('Q')[0]
                                        quarter_part = last_part.split('Q')[1]
                                        try:
                                            return int(year_part), int(quarter_part)
                                        except ValueError:
                                            return (0, 0)  # fallback
                                    return (0, 0)
                                
                                quarter_cols_sorted = sorted(quarter_cols, key=quarter_sort_key, reverse=False)
                                quarter_cols_last2 = quarter_cols_sorted[-2:]

                                latest_rank_col = f"{base_ratio}_latest_ratioRank"
                                trend_rank_col = f"{base_ratio}_trend_ratioRank"
                                #ttm_col = f"{base_ratio}_ttm_ratioValue"
                                #ttm_value = df_new_ranks.loc[selected_stock_ticker, ttm_col] if ttm_col in df_new_ranks.columns else None
                                ttm_diff = f"{base_ratio}_ttm_diff"
                                ttm_diff_value = df_new_ranks.loc[selected_stock_ticker, ttm_diff] if ttm_diff in df_new_ranks.columns else None
                                #st.write(f"Looking for ttm_col:{ttm_col}", ttm_col in df_new_ranks.columns)
                                #st.write("ttm_value:", ttm_value)
                                #st.write("ttm_diff_value:", ttm_diff_value)
                                with cols[idx]:
                                    if year_cols_last4 and quarter_cols_last2:
                                        try:
                                            # get values for the 4 years and then the two most recent quarters
                                            raw_values = df_new_ranks.loc[selected_stock_ticker, year_cols_last4 + quarter_cols_last2].values
                                            # Convert to numeric, handling any non-numeric values
                                            values = pd.to_numeric(raw_values, errors='coerce').astype(float)

                                        except Exception as e:
                                            values = np.array([])  # Empty array to skip plotting
                                        # Safely parse year columns (convert to int if possible, else skip)
                                        years_numeric = []
                                        years_labels = []
                                        for col in year_cols_last4:
                                            try:
                                                year_val = int(col.split('_')[-1])
                                                years_numeric.append(year_val)
                                                years_labels.append(str(year_val))
                                            except (ValueError, IndexError):
                                                years_numeric.append(col.split('_')[-1])  # fallback
                                                years_labels.append(col.split('_')[-1])
                                        
                                        # For quarters, convert to numeric values and keep labels
                                        def quarter_to_numeric(quarter_str):
                                            if 'Q' in quarter_str:
                                                try:
                                                    year = int(quarter_str.split('Q')[0])
                                                    quarter = int(quarter_str.split('Q')[1])
                                                    return year + (quarter - 1) / 4.0  # Q1=0, Q2=0.25, Q3=0.5, Q4=0.75
                                                except (ValueError, IndexError):
                                                    return quarter_str  # fallback to string
                                            return quarter_str
                                        
                                        for col in quarter_cols_last2:
                                            quarter_str = col.split('_')[-1]
                                            years_numeric.append(quarter_to_numeric(quarter_str))
                                            years_labels.append(quarter_str)
                                        
                                        years = years_numeric  # for calculations
                                        years_display = years_labels  # for display
                                        
                                        # Check if we have valid numeric values
                                        if len(values) == 0 or pd.isna(values).all():
                                            st.warning(f"Ingen giltig data för {base_ratio}. Hoppar över graf.")
                                            continue
                                        
                                        # Filter out NaN values and corresponding years
                                        valid_indices = ~pd.isna(values)
                                        values = values[valid_indices]
                                        years = [y for y, valid in zip(years, valid_indices) if valid]
                                        years_display = [y for y, valid in zip(years_display, valid_indices) if valid]
                                        
                                        if len(values) == 0:
                                            st.warning(f"Ingen giltig data för {base_ratio}. Hoppar över graf.")
                                            continue
                                        
                                        bar_colors = (
                                            ['lightblue'] * (len(year_cols_last4) - 1) +
                                            ['royalblue'] +
                                            ['gold'] * len(quarter_cols_last2)
                                        )[:len(values)]  # Adjust colors to match actual data length
                                        # Prepare bar data: years + quarters (variable length)
                                        bar_x = years_display
                                        bar_y = list(values)
                                        bar_text = [f"{v:.2f}" for v in values]
                                        ttm_label = None  # Ensure ttm_label is always defined
                                        # st.write("bar_x:", bar_x)
                                        # st.write("bar_y:", bar_y)
                                        # st.write("bar_colors:", bar_colors)
                                        fig = go.Figure()
                                        # Add bars for 4 years + ttm (if present)
                                        fig.add_trace(go.Bar(x=bar_x, y=bar_y, marker_color=bar_colors, name=base_ratio, showlegend=False, text=bar_text, textposition='auto'))
                                        # Add trend line (only for the 4 years)
                                        if len(years) > 1:
                                            # Only fit the trend line to the first 4 items (years, not quarters)
                                            trend_years = years[:4]
                                            trend_values = values[:4]
                                            trend_x_display = years_display[:4]
                                            if len(trend_years) > 1:
                                                coeffs = np.polyfit(trend_years, trend_values, 1)
                                                trend_vals = np.polyval(coeffs, trend_years)
                                                fig.add_trace(go.Scatter(
                                                    x=trend_x_display,
                                                    y=trend_vals,
                                                    mode='lines',
                                                    name='Trend',
                                                    line=dict(color='#888888', dash='dot', width=6),
                                                    showlegend=False
                                                ))
                                        else:
                                            trend_vals = values

                                        # add trendline between the last two items (the quarters)
                                        if len(bar_x) >= 2:
                                            fig.add_trace(go.Scatter(
                                                x=[bar_x[-2], bar_x[-1]],
                                                y=[bar_y[-2], bar_y[-1]],
                                                mode='lines',
                                                name='Quarter Trend',
                                                line=dict(color="#0D0D0D", dash='dot', width=6),
                                                showlegend=False
                                            ))
                                        # Add annotation above ttm bar if available
                                        if ttm_label and ttm_value is not None and not pd.isna(ttm_value) and ttm_diff_value is not None and not pd.isna(ttm_diff_value):
                                            pct_text = f"{ttm_diff_value:+.2f}"
                                            # Use higher_is_better to determine color
                                            if higher_is_better:
                                                color = "green" if ttm_diff_value >= 0 else "red"
                                            else:
                                                color = "red" if ttm_diff_value >= 0 else "green"
                                            y_shift = 20 if ttm_value >= 0 else -20
                                            fig.add_annotation(x=ttm_label,y=ttm_value,text=pct_text,showarrow=False,font=dict(color=color, size=14, family="Arial"),yshift=y_shift) # annotation for pct change for ttm
                                        fig.update_layout(title=f"{base_ratio}",
                                                        height=250,
                                                        margin=dict(l=10, r=10, t=30, b=10),
                                                        showlegend=False,
                                                        xaxis=dict(type='category'))
                                        st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True, key=f"{cat}_{base_ratio}_bar")
                                        #latest_rank = df_new_ranks.loc[selected_stock_ticker, latest_rank_col] if latest_rank_col in df_new_ranks.columns else 'N/A'
                                        #trend_rank = df_new_ranks.loc[selected_stock_ticker, trend_rank_col] if trend_rank_col in df_new_ranks.columns else 'N/A'
                                    else:
                                        st.warning(f"Ingen data för de senaste 4 åren för {base_ratio}. Trend Rank och Latest Rank sätts till 50 (neutral).")
                                    # Bullet plots for the two ranks in two columns: trend (left), latest (right)
                                    #st.write(f"**{ratio}**")
                                    # Dataframe for the ranks
                                    st.dataframe(
                                        df_ratioRank_merged[df_ratioRank_merged['index_trend'] == ratio][['Trend Rank', 'Latest Rank','TTM Rank']].style.map(color_progress, subset=['Trend Rank', 'Latest Rank','TTM Rank']),
                                        hide_index=True,
                                        width="stretch",
                                        column_config={
                                            "Latest Rank": st.column_config.ProgressColumn(
                                                    "Latest Rank",
                                                    #help=ratio_help_texts.get(ratio),
                                                    min_value=0,
                                                    max_value=100,
                                                    format="%.1f",
                                                    width="small",
                                                ),
                                            "Trend Rank": st.column_config.ProgressColumn(
                                                    "Trend Rank",
                                                    #help=ratio_help_texts.get(ratio),
                                                    min_value=0,
                                                    max_value=100,
                                                    format="%.1f",
                                                    width="small"
                                                ),
                                            "TTM Rank": st.column_config.ProgressColumn(
                                                    "TTM Rank",
                                                    #help=ratio_help_texts.get(ratio),
                                                    min_value=0,
                                                    max_value=100,
                                                    format="%.1f",
                                                    width="small"
                                                )
                                        }
                                    )
                                    
                    # Clear the empty space before each category
                    st.markdown("<br>", unsafe_allow_html=True) # Lägger till tre radbrytningar
        if show_Ratio_to_Rank:
            with st.container(border=True, key="ratio_rank_container"):
                st.subheader("**🎯 Ratio 2 Rank – Hitta avvikarna!**")
                if selected_stock_ticker is not None:
                    st.markdown(f"**{selected_stock_ticker}, {selected_stock_lista}, {selected_stock_sektor}**")
                    with st.expander("🛟 **Hjälp om Ratio 2 Rank** (Klicka för att visa)", expanded=False):
                        st.markdown(
                            """
                            **Scatterplot-magi: Hitta avvikarna och guldkornen!**

                            **🎯 Vad du ser:**  
                            • X-axel = Nyckeltalet (faktiska värdet)  
                            • Y-axel = Ranking (0-100, högre = bättre)  
                            • Din aktie = röd punkt med korslinje  
                            • Alla andra = blå punkter  

                            **🎨 Bakgrundsfärger (5 zoner):**  
                            • Mörkröd = mycket svag ranking (0-20)  
                            • Röd = svag ranking (21-40)  
                            • Gul = okej ranking (41-60)  
                            • Ljusgrön = bra ranking (61-80)  
                            • Mörkgrön = utmärkt ranking (81-100)  

                            **🔧 Kontroller:**  
                            • **Område:** Trend (4 år) vs Senaste året  
                            • **Sektor/Lista:** Jämför äpplen med äpplen  
                            • **Nyckeltal:** Välj vad du vill analysera  

                            **💰 Vad du kan upptäcka:**  
                            • Din aktie i grön zon = stark prestanda inom detta nyckeltal  
                            • Din aktie i röd zon = svag prestanda, kanske förbättringspotential  
                            • Outliers (avvikare) = aktier som sticker ut från mängden  
                            • Kluster = grupper av aktier med liknande prestanda  
                            • Jämförelse inom sektor/lista = hur din aktie presterar mot liknande bolag  

                            **🔍 Viktigt att komma ihåg:**  
                            Plotten visar din aktie jämfört med andra aktier i samma sektor eller börs-lista - så du jämför verkligen äpplen med äpplen!  

                            **Pro-tips:** Använd olika nyckeltal för att få en helhetsbild av bolagets styrkor och svagheter!  
                            """
                        )
                    col_left, col_mid, col_right = st.columns(3, gap='medium', border=False)
                    with col_left:
                        selected_ratio_area = st.radio(
                            "Välj område att visa:",
                            options=['Trend senaste 4 åren', 'Senaste ttm', 'Diff senaste ttm mot föregående ttm'],
                            index=0,
                            key="selected_ratio_area"
                        )
                        ratioValue_map = {
                            'Trend senaste 4 åren': 'year_trend',
                            'Senaste ttm': 'quarter_latest',
                            'Diff senaste ttm mot föregående ttm': 'quarter_trend'
                        }
                        ratio_to_value_map_temp = ratioValue_map.get(selected_ratio_area, 'quarter_latest')
                        ratioRank_map = {
                            'Trend senaste 4 åren': 'trend',
                            'Senaste ttm': 'latest',
                            'Diff senaste ttm mot föregående ttm': 'ttm'
                        }
                        ratio_to_rank_map_temp = ratioRank_map.get(selected_ratio_area, 'quarter_latest')
                    with col_mid:
                        sektors_all = [selected_stock_sektor, 'Alla']
                        display_stock_sektor_selector = st.radio(
                            "Välj Sektor att visa:",
                            options=sektors_all,
                            index=sektors_all.index(selected_stock_sektor) if selected_stock_sektor in sektors_all else 0,
                            key="display_stock_sektor"
                        )
                        display_stock_sektor = display_stock_sektor_selector if display_stock_sektor_selector != 'Alla' else unique_values_sector
                        display_stock_sektor = [display_stock_sektor] if isinstance(display_stock_sektor, str) else display_stock_sektor

                    with col_right:
                        lists_all = [selected_stock_lista, 'Alla']
                        display_stock_lista_selector = st.radio(
                            "Välj Lista att visa:",
                            options=lists_all,
                            index=lists_all.index(selected_stock_lista) if selected_stock_lista in lists_all else 0,
                            key="display_stock_lista"
                        )
                        display_stock_lista = display_stock_lista_selector if display_stock_lista_selector != 'Alla' else unique_values_lista
                        display_stock_lista = [display_stock_lista] if isinstance(display_stock_lista, str) else display_stock_lista
                    # Plotly scatter plot for selected ratio and rank
                    filtered_scatter_df = df_new_ranks[df_new_ranks['Sektor'].isin(display_stock_sektor) & df_new_ranks['Lista'].isin(display_stock_lista)]
                    display_ratio_selector = st.selectbox(
                        "Välj ett nyckeltal att visa detaljerad information om:",
                        options=all_ratios
                    )
                    display_ratio=f"{display_ratio_selector}_{ratio_to_value_map_temp}_ratioValue"
                    display_rank = f"{display_ratio_selector}_{ratio_to_rank_map_temp}_ratioRank"
                    col_left, col_right = st.columns(2, gap='medium', border=False)

                    if (
                        display_ratio and display_rank and
                        display_ratio in df_new_ranks.columns and display_rank in df_new_ranks.columns and
                        not filtered_scatter_df.empty
                    ):
                        # Create color array: red for selected_stock_ticker, royalblue for others
                        marker_colors = [
                            'red' if idx == selected_stock_ticker else 'royalblue'
                            for idx in filtered_scatter_df.index
                        ]
                        scatter_fig = go.Figure()

                        # Add horizontal background color bars for Rank value ranges
                        color_ranges = config.get('color_ranges', [])
                        x_min = filtered_scatter_df[display_ratio].min()
                        x_max = filtered_scatter_df[display_ratio].max()
                        st.write(f"X-axel intervall: {x_min:.2f} till {x_max:.2f}")
                        for cr in color_ranges:
                            y0 = cr['range'][0]
                            y1 = cr['range'][1]
                            scatter_fig.add_shape(
                                type="rect",
                                x0=x_min, x1=x_max,
                                y0=y0, y1=y1,
                                fillcolor=cr['color'],
                                opacity=0.25,
                                line=dict(width=0),
                                layer="below"
                            )

                        # Add scatter points
                        scatter_fig.add_trace(go.Scatter(
                            x=filtered_scatter_df[display_ratio],
                            y=filtered_scatter_df[display_rank],
                            mode='markers',
                            marker=dict(size=8, color=marker_colors),
                            text=filtered_scatter_df.index,
                            hoverinfo='text+x+y',
                            #name=f"{display_ratio} vs {display_rank}"
                        ))

                        # Add crosshair for selected_stock_ticker if present in filtered_scatter_df and has valid display_ratio value
                        if (
                            selected_stock_ticker in filtered_scatter_df.index and
                            pd.notna(filtered_scatter_df.loc[selected_stock_ticker, display_ratio]) and
                            pd.notna(filtered_scatter_df.loc[selected_stock_ticker, display_rank])
                        ):
                            x_val = filtered_scatter_df.loc[selected_stock_ticker, display_ratio]
                            y_val = filtered_scatter_df.loc[selected_stock_ticker, display_rank]
                            scatter_fig.add_shape(
                                type="line",
                                x0=x_val, x1=x_val,
                                y0=filtered_scatter_df[display_rank].min(), y1=filtered_scatter_df[display_rank].max(),
                                line=dict(color="red", width=2, dash="dot"),
                            )
                            scatter_fig.add_shape(
                                type="line",
                                x0=filtered_scatter_df[display_ratio].min(), x1=filtered_scatter_df[display_ratio].max(),
                                y0=y_val, y1=y_val,
                                line=dict(color="red", width=2, dash="dot"),
                            )
                        else:
                            st.warning(f"Valt bolag {selected_stock_ticker} saknar giltiga värden för {display_ratio} eller {display_rank}. Ingen korslinje visas.")

                        scatter_fig.update_layout(
                            #title=f"Scatterplot: {display_ratio} vs {display_rank}",
                            xaxis_title=f"{display_ratio_selector} {ratio_to_rank_map_temp} Värde",
                            yaxis_title=f"{display_ratio_selector} {ratio_to_rank_map_temp} Rank",
                            height=400,
                            margin=dict(l=10, r=10, t=40, b=10)
                        )
                        st.plotly_chart(scatter_fig, config={"displayModeBar": False}, use_container_width=True, key=f"scatter_{display_ratio}_{display_rank}")
                        with st.expander(f"🛟 **Hjälp om  {{display_ratio_selector}}_{ratio_to_rank_map_temp}** (Klicka för att visa)"):
                            st.write(get_ratio_help_text(f"{display_ratio_selector}_{ratio_to_rank_map_temp}"))

                    elif display_ratio and display_rank and display_ratio in df_new_ranks.columns and display_rank in df_new_ranks.columns:
                        st.info("Ingen data att visa för scatterplotten med nuvarande filter.")


                            
                # --- END: Show ratio bar charts for each _trend_rank category ---

except FileNotFoundError:
    st.error(f"Error: Main file '{CSV_PATH / config['results_file']}' not found in directory '{CSV_PATH}'. Check the path.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")




st.markdown("---")
st.subheader("ℹ️ Om Indicatum Insights")
st.info("🧪 **Beta-läge:** Data från Yahoo Finance | Endast för analys & utbildning | Inte finansiell rådgivning | Investera smart & ansvarsfullt!")
# --- END: Main app logic ---
