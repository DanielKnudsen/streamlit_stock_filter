import streamlit as st
import pandas as pd
import plotly.graph_objects as go # Import Plotly
import plotly.express as px # Import Plotly Express for bubble plot
import numpy as np # For handling numerical operations
from pathlib import Path
from rank import load_config
import datetime
import uuid
import json
from auth_ui import handle_authentication, render_account_buttons, handle_portfolio_save_dialog
from config_mappings import ConfigMappings
from app_helper import get_ratio_ranks_by_period,get_ratio_values_by_period,get_category_ranks_by_period,visualize_dataframe_with_progress
from app_plots import create_trend_momentum_plot, generate_price_chart,plot_ratio_values,generate_scatter_plot

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

# Initialize ConfigMappings for dynamic column handling
try:
    mappings = ConfigMappings(config)
    if ENABLE_DEBUG_LOGGING:
        st.write("✅ ConfigMappings initialized successfully")
except Exception as e:
    st.error(f"❌ Failed to initialize ConfigMappings: {e}")
    st.stop()

# --- Get directories for CSV files ---
CSV_PATH = Path('data') / ('local' if ENVIRONMENT == 'local' else 'remote')

show_Ratio_to_Rank =True


# --- Authentication Handling ---
user, should_stop = handle_authentication(ENABLE_AUTHENTICATION)

if should_stop:
    st.stop()
            

# Development mode indicator
if not ENABLE_AUTHENTICATION:
    st.info("🔧 **UTVECKLINGSLÄGE** - Autentisering är inaktiverad för lokal testning")

# Add account info and stats buttons after the welcome section (only when authentication is enabled)
if user and ENABLE_AUTHENTICATION:
    render_account_buttons(user, ENABLE_AUTHENTICATION, get_concurrent_users)

with st.container(border=False):
    
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

def create_slider_and_filter_df(df, column_name, tooltip_func, step=1.0, format_str="%d%%", key_suffix=""):
    """
    Skapar en Streamlit-slider för en given kolumn i en DataFrame.

    Parametrar:
        df: DataFrame som innehåller kolumnen
        column_name: Namn på kolumnen i DataFrame
        tooltip_func: Funktion som returnerar tooltip-text för kolumnen
        step: Stegstorlek för slidern (default: 1.0)
        format_str: Formatsträng för sliderns värden (default: "%d%%")
        key_suffix: Valfritt suffix för slider-nyckeln (default: "")

    Returnerar:
        filtered_df: Filtrerad DataFrame baserat på slidervärden
    """
    slider_key = f"slider_{column_name}{key_suffix}"
    min_max_key = f"minmax_{column_name}{key_suffix}"  # ← NEW: Store original min/max
    
    # Store original min/max on FIRST initialization only
    if min_max_key not in st.session_state:
        st.session_state[min_max_key] = (float(df[column_name].min()), float(df[column_name].max()))
    
    min_value, max_value = st.session_state[min_max_key]
    
    # Ensure the slider has a valid range
    if min_value == max_value:
        max_value += 0.001
    
    # Initialize slider value on first use
    if slider_key not in st.session_state:
        st.session_state[slider_key] = (min_value, max_value)
    
    # Get current slider value and clamp it
    current_value = st.session_state[slider_key]
    clamped_value = (
        max(min_value, min(current_value[0], max_value)),
        max(min_value, min(current_value[1], max_value))
    )

    slider_values = st.slider(
        label=get_display_name(column_name),
        min_value=min_value,
        max_value=max_value,
        value=clamped_value,
        step=step,
        format=format_str,
        help=tooltip_func(column_name),
        key=slider_key
    )

    # Filter using the ORIGINAL dataframe passed in
    filtered_df = df[
        ((df[column_name] >= slider_values[0]) & (df[column_name] <= slider_values[1])) | 
        (df[column_name].isna())
    ]
    
    return filtered_df

def create_pills_and_filter_df(df, column_name, tooltip_func):
    """
    Skapar Streamlit-piller för en given kolumn i en DataFrame.

    Parametrar:
        df: DataFrame som innehåller kolumnen
        column_name: Namn på kolumnen i DataFrame
        tooltip_func: Funktion som returnerar tooltip-text för kolumnen

    Returnerar:
        filtered_df: Filtrerad DataFrame baserat på pillerval
    """
    unique_values = df[column_name].dropna().unique().tolist()
    
    # Count occurrences of each value and create labels with counts
    value_counts = df[column_name].value_counts()
    labeled_values = [f"{val} ({value_counts[val]})" for val in unique_values]
    
    selected_labeled_values = st.pills(
        column_name,
        options=labeled_values,
        selection_mode='multi',
        default=labeled_values,
        key=f"pills_{column_name}",
        help=tooltip_func(column_name)
    )

    # Extract original values (without counts) for filtering
    selected_values = [val.rsplit(' (', 1)[0] for val in selected_labeled_values]
    
    # Filter the DataFrame based on the selected pill values
    return df[df[column_name].isin(selected_values)]


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

def prepare_sector_comparison(df):
    df_sector_avg = df[['Sektor','ttm_momentum_clusterRank_Sektor_avg', 'pct_ch_3_m_Sektor_avg']].drop_duplicates()
    df_sector_avg['is_sektor'] = True
    df_sector_avg.rename(columns={
        'ttm_momentum_clusterRank_Sektor_avg': 'ttm_momentum_clusterRank',
        'pct_ch_3_m_Sektor_avg': 'pct_ch_3_m'
    }, inplace=True)
    df_sector_avg.set_index('Sektor', inplace=True, drop=False)
    df_sector_avg['Lista'] = df_sector_avg['Sektor'] + ' Genomsnitt'
    df = df.drop(columns=['ttm_momentum_clusterRank_Sektor_avg', 'pct_ch_3_m_Sektor_avg'])
    df['is_sektor'] = False
    return pd.concat([df, df_sector_avg], axis=0)

# =============================
# LOAD DATA
# =============================


try:

    # Load main stock evaluation CSV (index_col=0 sets Ticker as index)
    df_new_ranks = pd.read_csv(CSV_PATH / config["results_file"], index_col=0)
    df_dividends = pd.read_csv(CSV_PATH / "dividends.csv", index_col='Ticker')
    df_agr_yearly = pd.read_csv(CSV_PATH / "agr_results_yearly_melted.csv", index_col='Ticker')
    df_agr_quarterly = pd.read_csv(CSV_PATH / "agr_results_quarterly_melted.csv", index_col='Ticker')
    df_agr_all_results = pd.read_csv(CSV_PATH / "agr_all_results_melted.csv", index_col='Ticker')
    unique_values_lista = df_new_ranks['Lista'].dropna().unique().tolist()
    unique_values_sector = df_new_ranks['Sektor'].dropna().unique().tolist()
    

    allCols_AvgGrowth_Rank = [col for col in df_new_ranks.columns if col.endswith('_AvgGrowth_Rank')]
    #st.write("allCols_AvgGrowth_Rank:", allCols_AvgGrowth_Rank)
    allCols_AvgGrowth = [col for col in df_new_ranks.columns if col.endswith('_AvgGrowth')] + ['cagr_close']
    # --- Create ratio-to-rank mapping using ConfigMappings ---
    ratio_definitions = config.get('ratio_definitions', {})
    # ConfigMappings handles the ratio-to-rank mapping dynamically
    # No need for manual mapping creation
    # =============================
    # COLUMN SELECTION FOR FILTERING AND DISPLAY
    # =============================
    # Filter columns that contain the string "catRank" for the main table

    # get all column groups from config / result_columns
    all_column_groups = config.get("result_columns", {})
    #st.write("all_column_groups before sektor_avg addition:", all_column_groups)
    # add sektor_avg as key (take from config, but add two versions to the items, one ending with _Sektor_avg, and one ending with _Sektor_diff)
    all_column_groups["sektor_avg"] = config.get("sektor_avg", [])
    all_column_groups["sektor_avg"] = [col + "_Sektor_avg" for col in all_column_groups["sektor_avg"]] + [col + "_Sektor_diff" for col in all_column_groups["sektor_avg"]]
    #st.write("all_column_groups:", all_column_groups)
    # Initialize a DataFrame that will be filtered by sliders
    df_filtered_by_sliders = df_new_ranks.copy()


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
    # st.write(f"Totalt antal aktier efter portföljfilter: {len(df_filtered_by_sliders)}")
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
    with st.container(border=False, key="filter_section"):
        st.subheader(f"🎯 Aktiefilter – Hitta dina favoriter bland {len(df_filtered_by_sliders)} aktier")

        with st.expander("🛟 **Hjälp med Filtrering?**", expanded=False):
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
        with st.expander(f"🎯 **Välj eller uteslut från sektor eller lista** {len(df_filtered_by_sliders)}"):
            col_lista, col_sektor = st.columns(2, gap='medium', border=True)
            with col_lista:
                df_filtered_by_sliders = create_pills_and_filter_df(df_filtered_by_sliders, 'Lista', get_tooltip_text)

            with col_sektor:
                df_filtered_by_sliders = create_pills_and_filter_df(df_filtered_by_sliders, 'Sektor', get_tooltip_text)

        with st.expander(f"🎯 **Total Rank** {len(df_filtered_by_sliders)}", expanded=False):
            st.markdown("##### Filtrera efter Total Rank")
            total_rank_columns = all_column_groups['Total Rank'] # ['long_trend_clusterRank','ttm_momentum_clusterRank','ttm_current_clusterRank']

            # loop through total_rank_columns and create sliders
            columns = st.columns(len(total_rank_columns), gap='medium', border=True)
            for total_rank_col, col in zip(total_rank_columns, columns):
                with col:
                    df_filtered_by_sliders = create_slider_and_filter_df(df_filtered_by_sliders, total_rank_col, get_tooltip_text, 1.0, "%d")

        with st.expander(f"🎯 **Periodtyp Rank** {len(df_filtered_by_sliders)}", expanded=False):
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
            time_periods = mappings.cluster_columns # ['long_trend_clusterRank','ttm_momentum_clusterRank','ttm_current_clusterRank']
            
            # loop through time periods and create sliders
            columns = st.columns(len(time_periods), gap='medium', border=True)
            for period, col in zip(time_periods, columns):
                with col:
                    df_filtered_by_sliders = create_slider_and_filter_df(df_filtered_by_sliders, period, get_tooltip_text, 1.0, "%d")
            
            cluster_rank_sums_diffs = ['clusterRank_sums','clusterRank_diffs']
            columns = st.columns(len(cluster_rank_sums_diffs), gap='medium', border=True)
            for period, col in zip(cluster_rank_sums_diffs, columns):
                with col:
                    df_filtered_by_sliders = create_slider_and_filter_df(df_filtered_by_sliders, period, get_tooltip_text, 1.0, "%d")
            
            # scatter plot of ttm_momentum_clusterRank vs long_trend_clusterRank
            if not df_filtered_by_sliders.empty and all(col in df_filtered_by_sliders.columns for col in time_periods):
                scatter_df = df_filtered_by_sliders[time_periods + ['Lista']].copy()
                scatter_df = scatter_df.dropna(subset=['long_trend_clusterRank', 'ttm_momentum_clusterRank', 'ttm_current_clusterRank','Lista'])
                
                if not scatter_df.empty:
                    create_trend_momentum_plot(get_display_name, scatter_df)
                else:
                    st.info("Ingen data tillgänglig för scatterplotten efter borttagning av saknade värden.")
            else:
                st.warning("De nödvändiga kolumnerna är inte tillgängliga för scatterplotten.")

        with st.expander(f"🎯 **Teknisk analys: SMA-differenser** {len(df_filtered_by_sliders)}", expanded=False):
            st.markdown("##### Filtrera efter SMA-differenser")
            sma_periods = all_column_groups['Glidande medelvärde'] # ['long_trend_clusterRank','ttm_momentum_clusterRank','ttm_current_clusterRank']
                        
            # loop through sma_periods and create sliders
            columns = st.columns(len(sma_periods), gap='medium', border=True)
            for sma_period, col in zip(sma_periods, columns):
                with col:
                    df_filtered_by_sliders = create_slider_and_filter_df(df_filtered_by_sliders, sma_period, get_tooltip_text, 1.0, "%d")

        with st.expander(f"🎯 **Omsättningstillväxt Rank** {len(df_filtered_by_sliders)}", expanded=False):
            st.markdown("##### Filtrera efter Omsättningstillväxt")
            revenue_columns = all_column_groups['Omsättningstillväxt']  # Assume this method exists in ConfigMappings
            
            # loop through revenue_columns and create sliders
            columns = st.columns(len(revenue_columns), gap='medium', border=True)
            for revenue_col, col in zip(revenue_columns, columns):
                with col:
                    df_filtered_by_sliders = create_slider_and_filter_df(df_filtered_by_sliders, revenue_col, get_tooltip_text, 1.0, "%d")
            revenue_growth_columns = all_column_groups['Omsättningstillväxt_values']
            columns = st.columns(len(revenue_growth_columns), gap='medium', border=True)
            for revenue_growth_col, col in zip(revenue_growth_columns, columns):
                with col:
                    df_filtered_by_sliders = create_slider_and_filter_df(df_filtered_by_sliders, revenue_growth_col, get_tooltip_text, 1.0, "%2.2f")

        with st.expander(f"🎯 **Vinsttillväxt per aktie Rank** {len(df_filtered_by_sliders)}", expanded=False):
            st.markdown("##### Filtrera efter Vinsttillväxt per aktie")
            eps_columns = all_column_groups['EPS']  # Assume this method exists in ConfigMappings

            # loop through eps_columns and create sliders
            columns = st.columns(len(eps_columns), gap='medium', border=True)
            for eps_col, col in zip(eps_columns, columns):
                with col:
                    df_filtered_by_sliders = create_slider_and_filter_df(df_filtered_by_sliders, eps_col, get_tooltip_text, 1.0, "%d")
            eps_growth_columns = all_column_groups['EPS_values']
            columns = st.columns(len(eps_growth_columns), gap='medium', border=True)
            for eps_growth_col, col in zip(eps_growth_columns, columns):
                with col:
                    df_filtered_by_sliders = create_slider_and_filter_df(df_filtered_by_sliders, eps_growth_col, get_tooltip_text, 1.0, "%2.2f")

        with st.expander(f"🎯 **Sektoranalys** {len(df_filtered_by_sliders)}", expanded=False):
            st.markdown("##### Filtrera efter Sektoranalys")
            sektor_columns =['ttm_momentum_clusterRank','pct_ch_3_m','sektor_avg_diffs']
            # loop through sektor_columns and create sliders
            columns = st.columns(len(sektor_columns), gap='medium', border=True)
            for sektor_col, col in zip(sektor_columns, columns):
                with col:
                    df_filtered_by_sliders = create_slider_and_filter_df(df_filtered_by_sliders, sektor_col, get_tooltip_text, 1.0, "%d", key_suffix="_sektor")
            
            df_scatter_to_use=prepare_sector_comparison(df_filtered_by_sliders[['Lista','Sektor','ttm_momentum_clusterRank','ttm_momentum_clusterRank_Sektor_avg','pct_ch_3_m','pct_ch_3_m_Sektor_avg']])
            fig = generate_scatter_plot(df_scatter_to_use)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander(f"🎯 **Expertnivå: Detaljerad nyckeltalsfiltrering** {len(df_filtered_by_sliders)}", expanded=False):
            st.markdown("""
            ### 🔬 Expertnivå – full kontroll över varje nyckeltal!

            **För dig som vill micro-managea:**  
            • Filtrera på kategori-nivå (Trend, Senaste, TTM)  
            • Detaljstyrning av varje enskilt nyckeltal  
            • Skapa helt skräddarsydda urval  

            **Varning:** Här kan du gå ner i kaninhålet och komma fram 3 timmar senare! 🐰
            """)

            # Extract periods dynamically from ConfigMappings
            all_periods = mappings.period_types
            columns = st.columns(len(all_periods),gap='medium',border=True)
            for period, col in zip(all_periods, columns):
                with col:
                    # Get category rank columns for this period using ConfigMappings
                    category_rank_cols = mappings.get_category_rank_columns_for_period(period)
                    for category_rank_col in category_rank_cols:
                        with st.container(border=True,key=f"V2_container_trend_{category_rank_col}"):
                            df_filtered_by_sliders = create_slider_and_filter_df(df_filtered_by_sliders, category_rank_col, get_tooltip_text, 1.0, "%d")
                            # Get underlying ratio_rank_columns for this category rank column
                            ratio_rank_columns = mappings.get_underlying_ratios_for_category_rank(category_rank_col)['ratio_rank_columns']
                            # Get underlying ratio_value_columns for this category rank column
                            ratio_value_columns = mappings.get_underlying_ratios_for_category_rank(category_rank_col)['ratio_value_columns']

                            # create tabs for each ratio and create one slider for ratio rank and one slider for ratio value
                            tab_labels = ['Info'] + ratio_rank_columns
                            tabs = st.tabs(tab_labels)
                            for i, ratio_rank in enumerate(ratio_rank_columns):
                                ratio_value = ratio_value_columns[i]
                                with tabs[i+1]:
                                    df_filtered_by_sliders = create_slider_and_filter_df(df_filtered_by_sliders, ratio_rank, get_tooltip_text, 1.0, "%d")
                                    df_filtered_by_sliders = create_slider_and_filter_df(df_filtered_by_sliders, ratio_value, get_tooltip_text, 0.01, "%2.2f")


        with st.expander("🎯 **Eller ange tickers direkt**", expanded=False):
            ticker_input = st.text_input(
                "Filtrera på tickers (kommaseparerade, t.ex. VOLV-A, ERIC-B, ATCO-A):",
                value="",
                help="Skriv in en eller flera tickers separerade med komma för att endast visa dessa aktier."
                )
            if ticker_input.strip():
                tickers_to_keep = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
                df_filtered_by_sliders = df_filtered_by_sliders[df_filtered_by_sliders.index.str.upper().isin(tickers_to_keep)]
          
    # =============================
    # FILTERED RESULTS AND BUBBLE PLOT
    # =============================
    st.markdown("<br>", unsafe_allow_html=True) # Lägger till tre radbrytningar

    with st.container(border=False, key="filtered_results"):

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

        # =============================
        # MAIN TABLE DISPLAY
        # =============================
        with st.container(border=False, key="main_table_container"):
            st.markdown("##### Resultattabell med filtrerade aktier")

            selected_column_keys = st.segmented_control(
                "Välj kolumner i resultat-tabellen",
                options=all_column_groups.keys(),
                selection_mode='multi',
                default="Bas Data",
                key="selected_column_keys_input"
            )

            # Aggregate columns from all selected groups
            if isinstance(selected_column_keys, list):
                rank_score_columns = sum([all_column_groups[k] for k in selected_column_keys], [])
            else:
                rank_score_columns = all_column_groups[selected_column_keys]

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
            # Initialize checkbox states in session state if not already present
            if "checkbox_states" not in st.session_state:
                st.session_state.checkbox_states = {}
            
            # Get or initialize checkbox values for each ticker
            df_display['Välj'] = df_display.index.map(
                lambda ticker: st.session_state.checkbox_states.get(f"{ticker}_valj", False)
            )

            # Add a "Shortlist" column to save stocks
            # Get or initialize shortlist values for each ticker
            df_display['Shortlist'] = df_display.index.map(
                lambda ticker: st.session_state.checkbox_states.get(f"{ticker}_shortlist", False)
            )

            cols = df_display.columns.tolist()
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
                    )
                },
                key="stock_selection_editor" # Unique key to manage state
            )
            
            # Track if any checkbox state changed
            state_changed = False
            # Update session state with the new checkbox values from the editor
            for ticker in edited_df.index:
                new_valj = edited_df.loc[ticker, 'Välj']
                new_shortlist = edited_df.loc[ticker, 'Shortlist']
                old_valj = st.session_state.checkbox_states.get(f"{ticker}_valj", False)
                old_shortlist = st.session_state.checkbox_states.get(f"{ticker}_shortlist", False)
                
                if new_valj != old_valj or new_shortlist != old_shortlist:
                    state_changed = True
                
                st.session_state.checkbox_states[f"{ticker}_valj"] = new_valj
                st.session_state.checkbox_states[f"{ticker}_shortlist"] = new_shortlist
            
            # Force immediate rerun if state changed to ensure new state loads on next render
            if state_changed:
                st.rerun()

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
                
                try:
                    selected_stock_dict = df_new_ranks.loc[selected_stock_ticker].to_dict()
                    # DEBUG: Check QuarterDiff immediately after dict creation
                except KeyError as e:
                    selected_stock_dict = None
                except Exception as e:
                    selected_stock_dict = None
            # Logic to handle Shortlist
            shortlisted_stocks = edited_df[edited_df['Shortlist']]
            with st.container(border=True, key="shortlist_container"):
                st.markdown("##### Bevakningslista (Shortlist)")

                if not shortlisted_stocks.empty:
                    df_display = shortlisted_stocks.copy()
                    
                    # Remove ['Shortlist', 'Välj'] columns for download
                    df_display = df_display.drop(columns=['Shortlist', 'Välj'], errors='ignore')
                    download_columns = df_display.columns.tolist()
                    # Only keep columns that exist in df_display
                    download_columns = [col for col in download_columns if col in df_display.columns]

                    st.dataframe(
                        df_display,#[download_columns], # Ticker is already the index
                        hide_index=False,
                        width="stretch"
                    )

                    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    file_name = f"shortlist_{current_time}.csv"
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
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
                    
                    with col3:
                        # Create a comma-separated list of tickers for easy copying
                        tickers_list = ", ".join(shortlisted_stocks.index.tolist())

                        st.write(f"**📋 Kopiera tickerlista:** {tickers_list}")
                    
                    # Portfolio save dialog
                    handle_portfolio_save_dialog(user, shortlisted_stocks, current_time)
                else:
                    pass

    st.markdown("<br>", unsafe_allow_html=True) # Lägger till tre radbrytningar

    with st.container(border=False, key="stock_details_container"):
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

        with st.container(border=False, key="stock_details"):
            if selected_stock_ticker:
                #st.subheader(f"Kort info om: {selected_stock_dict['Name'] if 'Name' in selected_stock_dict else 'N/A'}")
                selected_stock_lista = selected_stock_dict['Lista'] if 'Lista' in selected_stock_dict else 'N/A'
                selected_stock_sektor = selected_stock_dict['Sektor'] if 'Sektor' in selected_stock_dict else 'N/A'
                
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
                    #st.data_editor(dividends_df, width="stretch", height=200, key=f"dividends_data_{selected_stock_ticker}")
                    # Only keep the 10 latest years
                    if not dividends_df.empty and 'Year' in dividends_df.columns:
                        dividends_df = dividends_df.sort_values('Year', ascending=False).head(10).sort_values('Year')
                    if not dividends_df.empty:
                        # Ensure Year and Value columns exist
                        if 'Year' in dividends_df.columns and 'Value' in dividends_df.columns:
                            # Convert Year to datetime for proper sorting
                            dividends_df = dividends_df.copy()

                            dividends_df.sort_values('Year', inplace=True)
                            fig_div = go.Figure(go.Bar(
                                x=dividends_df['Year'],
                                y=dividends_df['Value'],
                                marker_color='gold',
                                text=[f"{v:.2f}" for v in dividends_df['Value']],
                                textposition='auto',
                            ))
                            fig_div.update_layout(
                                title=f"Utdelningar för {selected_stock_ticker}, senaste 10 åren",
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
                    bar_colors = ['gold' if 'quarterly' in col else 'royalblue' for col in cagr_cols]
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
                        title=f"Genomsnittlig förändring över 4 år (blå) eller senaste kvartal (gul) för {selected_stock_dict['Name']} ({selected_stock_ticker})",
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
                            colors = ['royalblue'] * len(years)
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
                    
                    # define the three columns for the three ratios
                    left_col,  right_col = st.columns(2, gap='medium', border=False)

                    def plot_agr_data(cols):
                        df_tot_rev_yearly = df_agr_yearly[
                            (df_agr_yearly.index == selected_stock_ticker) &
                            (df_agr_yearly['Growth_Metric'].str.startswith(f'{cols}_year_2'))
                        ]
                        
                        # Select rows for the ticker and Growth_metric starting with 'Total_Revenue_quarterly_2'
                        df_tot_rev_quarterly = df_agr_quarterly[
                            (df_agr_quarterly.index == selected_stock_ticker) &
                            (df_agr_quarterly['Growth_Metric'].str.startswith(f'{cols}_quarterly_2'))
                        ]
                        df_tot_rev = pd.concat([df_tot_rev_yearly, df_tot_rev_quarterly])
                        df_tot_rev['Period'] = df_tot_rev['Growth_Metric'].apply(lambda x: x.replace(f'{cols}_year_', 'Year ').replace(f'{cols}_quarterly_', 'TTM '))
                        # st.data_editor(df_tot_rev, key="debug_tot_rev_data_editor", hide_index=True, width=900)

                        # plot df_tot_rev as bar chart with Period on x-axis and Value on y-axis
                        fig_tot_rev = px.bar(
                            df_tot_rev,
                            x='Period',
                            y='Value',
                            text='Value',
                            labels={'Value': 'SEK', 'Period': 'Period'},
                            title=f"{cols} för {selected_stock_ticker}",
                            height=350
                        )
                        marker_colors = ['royalblue'] * 4 + ['gold'] * 2
                        fig_tot_rev.update_traces(texttemplate='%{text:.3s}', textposition='inside', marker_color=marker_colors)
                        fig_tot_rev.update_layout(margin=dict(l=10, r=10, t=100, b=40), yaxis_title="SEK", xaxis=dict(type='category'))
                        st.plotly_chart(fig_tot_rev, config={"displayModeBar": False}, use_container_width=True, key=f"{cols}_bar_{selected_stock_ticker}")

                    with left_col:
                        cols = "Total_Revenue"
                        plot_agr_data(cols)

                    with right_col:
                        cols= "Basic_EPS"
                        plot_agr_data(cols)
                    
        with st.container(border=True, key="stock_price_trend_container"):
            st.subheader("💹 Kursutveckling & Smart Trendanalys")

            generate_price_chart(config, CSV_PATH, add_moving_averages, get_display_name, selected_stock_ticker, selected_stock_dict)

            
            # =============================
            # PERCENTAGE BAR PLOTS
            # =============================
            # Bar plot for all pct_ columns for selected_stock_ticker
            if selected_stock_ticker:
                with st.expander("**SMA differenser (%)** (Klicka för att visa)", expanded=False):
                    pct_cols = all_column_groups['Glidande medelvärde']
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
                st.markdown("**Trend senaste 4 åren & Senaste året**")
                # get catRank create a dataframe with category ranks as rows and cluster periods as columns for long_trend, ttm_current and ttm_momentum
                df_ranking = get_category_ranks_by_period(selected_stock_ticker, df_new_ranks, mappings)

                # -------------------------------------------------------------
                # PROGRESS BARS: LATEST AND TREND RANKINGS
                # -------------------------------------------------------------

                visualize_dataframe_with_progress(color_progress, df_ranking)
                # -------------------------------------------------------------
                # TREND RATIO BREAKDOWN BAR CHARTS
                # -------------------------------------------------------------

                st.markdown('---')
                st.subheader('🔬 Detaljerad ranking per kategori')
                st.markdown("**Trend senaste 4 åren & Senaste året**")

                for category in mappings.category_bases:
                    with st.container(border=False, key=f"category_container_{category}"):
                        st.subheader(category)
                        visualize_dataframe_with_progress(color_progress, df_ranking.loc[[category]], hide_index=True)
                        # get ratios in this category
                        ratios_in_category = mappings.category_to_ratios[category]

                        cols = st.columns(len(ratios_in_category), border=True,gap="small")
                        for idx, ratio in enumerate(ratios_in_category):
                            with cols[idx]:
                                st.markdown(f"**{ratio}**")
                                df_ratio_rank = get_ratio_ranks_by_period(selected_stock_ticker, ratio, df_new_ranks, mappings)
                                #st.dataframe(df_ratio_rank, hide_index=True)
                                test_df = get_ratio_values_by_period(selected_stock_ticker, ratio, df_new_ranks)
                                #st.dataframe(test_df, hide_index=False)
                                if not test_df.empty:
                                    fig = plot_ratio_values(test_df, mappings=mappings)
                                    # show fig
                                    fig
                                else:
                                    st.info(f"Inga data för {ratio} tillgängliga, rank sätts till 50 som standard.")
                                visualize_dataframe_with_progress(color_progress, df_ratio_rank, hide_index=True)

                             
                                    
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
                        # Get column display names
                        col_names = {period: mappings.get_cluster_col_name(period) for period in mappings.period_types}
                        selected_ratio_area = st.radio(
                            "Välj period att visa:",
                            options=list(col_names.values()),
                            index=0,
                            key="selected_ratio_area"
                        )
                        # Map back to period type
                        period_type_map = {v: k for k, v in col_names.items()}
                        selected_period_type = period_type_map[selected_ratio_area]
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
                        options=mappings.ratio_bases
                    )
                    # Construct column names using period_type and ratio
                    display_ratio = f"{display_ratio_selector}_{selected_period_type}_ratioValue"
                    display_rank = f"{display_ratio_selector}_{selected_period_type}_ratioRank"
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
                            xaxis_title=f"{display_ratio_selector} {selected_period_type} Värde",
                            yaxis_title=f"{display_ratio_selector} {selected_period_type} Rank",
                            height=400,
                            margin=dict(l=10, r=10, t=40, b=10)
                        )
                        st.plotly_chart(scatter_fig, config={"displayModeBar": False}, use_container_width=True, key=f"scatter_{display_ratio}_{display_rank}")
                        with st.expander(f"🛟 **Hjälp om {display_ratio_selector}_{selected_period_type}** (Klicka för att visa)"):
                            st.write(get_ratio_help_text(f"{display_ratio_selector}_{selected_period_type}"))

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
