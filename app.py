import streamlit as st
import pandas as pd
import os
import yaml
import plotly.graph_objects as go # Import Plotly
import plotly.express as px # Import Plotly Express for bubble plot
import numpy as np # For handling numerical operations
from collections import OrderedDict
import pwlf
import streamlit as st
# =====================================================================
# STREAMLIT STOCK SCREENING APP - SWEDISH MARKETS
# =====================================================================

# =============================
# IMPORTS AND SETUP
# =============================

# --- Define directories for CSV files ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DATA_DIR = "csv-data" # Directory for the main CSV (e.g., "stock_evaluation_results.csv")
PRICE_DATA_DIR = "data" # Directory for individual stock price CSV files (e.g., "data/Company A.csv")
CSV_FILE_NAME = "stock_evaluation_results.csv"
full_csv_path = os.path.join(CSV_DATA_DIR, CSV_FILE_NAME)

# =============================
# STREAMLIT APPLICATION
# =============================
st.set_page_config(layout="centered", page_title="Stock Evaluation", page_icon="📈")

st.title("📈 Stock Screening Tool for Swedish Markets")
st.markdown(
    """
    Detta verktyg ger en omfattande översikt över svenska aktier och rankar dem inom flera finansiella kategorier och trender.
    Investerare kan använda dessa rankningar och filter för att identifiera lovande aktier, jämföra prestationer och bygga en personlig bevakningslista för vidare analys.
    """
)

# =============================
# HELPER FUNCTIONS
# =============================
# Funktion för att applicera bakgrundsfärger baserat på värden
def color_progress(val):
    color_ranges = [
        {'range': [0, 20], 'color': '#ffcccc'},    # Light Red
        {'range': [20, 40], 'color': '#ffe5cc'},   # Light Orange
        {'range': [40, 60], 'color': '#ffffcc'},   # Light Yellow
        {'range': [60, 80], 'color': '#e6ffe6'},   # Very Light Green
        {'range': [80, 100], 'color': '#ccffcc'}   # Light Green
    ]
    
    for cr in color_ranges:
        if cr['range'][0] <= val <= cr['range'][1]:
            return f'background-color: {cr["color"]}'
    return ''


# =============================
# LOAD DATA
# =============================
try:
    # Load main stock evaluation CSV (index_col=0 sets Ticker as index)
    df_new_ranks = pd.read_csv(full_csv_path, index_col=0)
    df_long_business_summary = pd.read_csv(os.path.join(CSV_DATA_DIR, "longBusinessSummary.csv"), index_col=0)

    # =============================
    # ADD HOVER SUMMARY COLUMN
    # =============================
    def get_truncated_summary(ticker):
        try:
            summary = str(df_long_business_summary.loc[ticker].values[0])
            return summary[:130] + ("..." if len(summary) > 130 else "")
        except Exception:
            return "No summary available."
    df_new_ranks['hover_summary'] = df_new_ranks.index.map(get_truncated_summary)
    
    # =============================
    # COLUMN SELECTION FOR FILTERING AND DISPLAY
    # =============================
    # Filter columns that contain the string "catRank" for the main table
    rank_score_columns = [col for col in df_new_ranks.columns if "catRank" in col]
    latest_columns = [col for col in rank_score_columns if "latest" in col.lower()]
    trend_columns = [col for col in rank_score_columns if "trend" in col.lower()]
    rank_score_columns = rank_score_columns + ['Latest_clusterRank', 'Trend_clusterRank']  # Include total scores
    # Initialize a DataFrame that will be filtered by sliders
    df_filtered_by_sliders = df_new_ranks.copy()

    # =============================
    # LOAD RANKING CATEGORIES FROM CONFIG
    # =============================
    rank_config_path = os.path.join(BASE_DIR, "rank-config.yaml")
    if os.path.exists(rank_config_path):
        with open(rank_config_path, "r") as f:
            config = yaml.safe_load(f)
        category_ratios = config.get("category_ratios", {})
        categories = list(category_ratios.keys())
        display_names = config.get("display_names", {})
        tooltip_texts = config.get("tooltip_texts", {})
    else:
        category_ratios = {}
        categories = []
    
    def get_display_name(var_name):
        # Try to get a pretty name, fallback to a cleaned-up version
        return display_names.get(var_name, var_name.replace("_", " ").title())
    
    def get_tooltip_text(var_name):
        # Try to get a tooltip text, fallback to an empty string
        return tooltip_texts.get(var_name, "")


    # =============================
    # ENHETLIGT FILTERAVSNITT
    # =============================
    with st.container(border=True):
        st.subheader("Aktiefilter")
        st.markdown(
            """
            Använd reglagen nedan för att filtrera aktier utifrån Totalrank, SMA-differenser och detaljerade rankningar inom olika finansiella kategorier.
            - **Totalrank:** Filtrera aktier baserat på deras aggregerade 'Trend'- och 'Senaste'-rank.
            - **SMA-differenser:** Begränsa urvalet med hjälp av skillnader mellan glidande medelvärden.
            - **Utökade filter:** Expandera för avancerad filtrering på kategori- och nyckeltalsnivå.
            Justera inställningarna för att hitta aktier som matchar dina investeringskriterier.
            """
        )

        # --- NY: Manuell ticker-filtrering ---
        ticker_input = st.text_input(
            "Filtrera på tickers (kommaseparerade, t.ex. VOLV-A,ERIC-B,ATCO-A):",
            value="",
            help="Skriv in en eller flera tickers separerade med komma för att endast visa dessa aktier."
        )
        if ticker_input.strip():
            tickers_to_keep = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
            df_filtered_by_sliders = df_filtered_by_sliders[df_filtered_by_sliders.index.str.upper().isin(tickers_to_keep)]

        # --- Filtrera efter SMA-differenser ---
        st.markdown("##### Filtrera efter SMA-differenser")
        col_diff_long_medium, col_diff_short_medium, col_diff_price_short = st.columns(3,gap='medium',border=True)
        with col_diff_long_medium:
            # Filter by Long-Medium SMA difference
            min_diff_long_medium = float(df_new_ranks['pct_SMA_medium_vs_SMA_long'].min())
            max_diff_long_medium = float(df_new_ranks['pct_SMA_medium_vs_SMA_long'].max())
            diff_long_medium_range = st.slider(
                get_display_name('pct_SMA_medium_vs_SMA_long'),
                min_value=min_diff_long_medium,
                max_value=max_diff_long_medium,
                value=(min_diff_long_medium, max_diff_long_medium),
                step=1.0,
                format="%d",
                help=get_tooltip_text('pct_SMA_medium_vs_SMA_long')
            )
        with col_diff_short_medium:
            # Filter by Short-Medium SMA difference
            min_diff_short_medium = float(df_new_ranks['pct_SMA_short_vs_SMA_medium'].min())
            max_diff_short_medium = float(df_new_ranks['pct_SMA_short_vs_SMA_medium'].max())
            diff_short_medium_range = st.slider(
                get_display_name('pct_SMA_short_vs_SMA_medium'),
                min_value=min_diff_short_medium,
                max_value=max_diff_short_medium,
                value=(min_diff_short_medium, max_diff_short_medium),
                step=1.0,
                format="%d",
                help=get_tooltip_text('pct_SMA_short_vs_SMA_medium')
            )
        with col_diff_price_short:
            # Filter by Price-Short SMA difference
            min_diff_price_short = float(df_new_ranks['pct_Close_vs_SMA_short'].min())
            max_diff_price_short = float(df_new_ranks['pct_Close_vs_SMA_short'].max())
            diff_price_short_range = st.slider(
                get_display_name('pct_Close_vs_SMA_short'),
                min_value=min_diff_price_short,
                max_value=max_diff_price_short,
                value=(min_diff_price_short, max_diff_price_short),
                step=1.0,
                format="%d",
                help=get_tooltip_text('pct_Close_vs_SMA_short')
            )

        df_filtered_by_sliders = df_filtered_by_sliders[(df_filtered_by_sliders['pct_SMA_short_vs_SMA_medium'] >= diff_long_medium_range[0]) & (df_filtered_by_sliders['pct_SMA_short_vs_SMA_medium'] <= diff_long_medium_range[1]) &
                                                          (df_filtered_by_sliders['pct_SMA_short_vs_SMA_medium'] >= diff_short_medium_range[0]) & (df_filtered_by_sliders['pct_SMA_short_vs_SMA_medium'] <= diff_short_medium_range[1]) &
                                                          (df_filtered_by_sliders['pct_Close_vs_SMA_short'] >= diff_price_short_range[0]) & (df_filtered_by_sliders['pct_Close_vs_SMA_short'] <= diff_price_short_range[1])]
        # --- Reglage för totalrank (överst, nu i två kolumner) ---
        st.markdown('##### Filtrera efter Aggregerad rankn')
        col_total_trend, col_total_latest = st.columns(2,gap='medium',border=True)
        with col_total_trend:
            min_trend = float(df_new_ranks['Trend_clusterRank'].min())
            max_trend = float(df_new_ranks['Trend_clusterRank'].max())
            trend_range = st.slider(
                get_display_name('Trend_clusterRank'),
                min_value=min_trend,
                max_value=max_trend,
                value=(min_trend, max_trend),
                step=1.0,
                format="%d",
                help=get_tooltip_text('Trend_clusterRank')
            )
        with col_total_latest:
            min_latest = float(df_new_ranks['Latest_clusterRank'].min())
            max_latest = float(df_new_ranks['Latest_clusterRank'].max())
            latest_range = st.slider(
                get_display_name('Latest_clusterRank'),
                min_value=min_latest,
                max_value=max_latest,
                value=(min_latest, max_latest),
                step=1.0,
                format="%d",
                help=get_tooltip_text('Latest_clusterRank')
            )
        
        df_filtered_by_sliders = df_filtered_by_sliders[(df_filtered_by_sliders['Trend_clusterRank'] >= trend_range[0]) & (df_filtered_by_sliders['Trend_clusterRank'] <= trend_range[1]) &
                                    (df_filtered_by_sliders['Latest_clusterRank'] >= latest_range[0]) & (df_filtered_by_sliders['Latest_clusterRank'] <= latest_range[1])]
        st.write("Antal kvarvarande aktier efter filtrering:", df_filtered_by_sliders.shape[0])
        # --- Senaste/Trend-reglage (under totalreglage) ---
        with st.expander('**Utökade filtermöjligheter**', expanded=False):
            col_filter_left, col_filter_right = st.columns(2,gap='medium',border=True)
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
                                    r_data = f"{r.replace('_trend_ratioRank', '_ratio_trendSlope')}"
                                    if r_data in df_filtered_by_sliders.columns:
                                        min_val = float(df_filtered_by_sliders[r_data].min(skipna=True))
                                        max_val = float(df_filtered_by_sliders[r_data].max(skipna=True))
                                        if min_val == max_val:
                                            max_val += 0.001
                                        slider_min, slider_max = st.slider(
                                            f"Filtrera {r_data.replace('_ratio_trendSlope', ' trend Slope')}",
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
            with col_filter_right:
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
                                    r_data = f"{r.replace('_latest_ratioRank', '_ratio_latest')}"
                                    if r_data in df_filtered_by_sliders.columns:
                                        min_val = float(df_filtered_by_sliders[r_data].min())
                                        max_val = float(df_filtered_by_sliders[r_data].max())
                                        if min_val == max_val:
                                            max_val += 0.001
                                        slider_min, slider_max = st.slider(
                                            f"Filtrera {r_data.replace('_ratio_latest', ' senaste Värde')}",
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
        # --- Reglage för kategoripoäng: En expander per kategori (ingen nästling) ---
        for cat in categories:
            with st.expander(f"Filtrera efter kategori: {cat}", expanded=False):
                pass
        st.write("Antal kvarvarande aktier efter filtrering:", df_filtered_by_sliders.shape[0])

    # =============================
    # BUBBLE PLOT SECTION
    # =============================
    with st.container(border=True):
        st.subheader("Bubbelplot med filtrering")
        # --- Bubble Plot: Total_Trend_Score vs Total_Latest_Score (filtered) ---

        # --- Lista toggles for bubble plot ---
        lista_values = []
        if 'Lista' in df_filtered_by_sliders.columns:
            with st.container(border=True, key="lista_toggles"):
                lista_values = df_filtered_by_sliders['Lista'].dropna().unique().tolist()
                lista_values = lista_values[:5]  # Limit to 5 unique values
                # Use pills for selection, all enabled by default
                lista_selected = st.pills(
                    "Välj/uteslut Lista:",
                    options=lista_values,
                    default=lista_values,
                    selection_mode='multi',
                    key="segmented_lista"
                )
                # Filter df_filtered_by_sliders by selected Lista values
                if lista_selected:
                    df_filtered_by_sliders = df_filtered_by_sliders[df_filtered_by_sliders['Lista'].isin(lista_selected)]
                else:
                    df_filtered_by_sliders = df_filtered_by_sliders.iloc[0:0]  # Show nothing if none selected
        # --- Sektor toggles for bubble plot ---
        sektor_values = []
        if 'Sektor' in df_filtered_by_sliders.columns:
            with st.container(border=True, key="sektor_toggles"):
                sektor_values = df_filtered_by_sliders['Sektor'].dropna().unique().tolist()
                # Use st.pills for multi-select, all enabled by default
                sektor_selected = st.pills(
                    "Välj/uteslut Sektor:",
                    options=sektor_values,
                    default=sektor_values,
                    selection_mode='multi',
                    key="pills_sektor"
                )
                # Filter df_filtered_by_sliders by selected Sektor values
                if sektor_selected:
                    df_filtered_by_sliders = df_filtered_by_sliders[df_filtered_by_sliders['Sektor'].isin(sektor_selected)]
                else:
                    df_filtered_by_sliders = df_filtered_by_sliders.iloc[0:0]  # Show nothing if none selected
        # Format marketCap for hover (MSEK, rounded, with space as thousands separator)

        with st.container(border=True, key="bubble_plot_container"):
            show_tickers = st.toggle('Visa tickers i bubbelplotten', value=True)
        if 'marketCap' in df_filtered_by_sliders.columns:
            df_filtered_by_sliders['marketCap_MSEK'] = (df_filtered_by_sliders['marketCap'] / 1_000_000).round().astype('Int64').map(lambda x: f"{x:,}".replace(",", " ") + " MSEK" if pd.notna(x) else "N/A")
        
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
            required_cols = ['Trend_clusterRank', 'Latest_clusterRank']
            if 'Lista' in df_filtered_by_sliders.columns:
                required_cols.append('Lista')
            plot_df = df_filtered_by_sliders.dropna(subset=required_cols, how='any').copy()
            # Handle marketCap for size
            if 'marketCap' in plot_df.columns:
                size_raw = plot_df['marketCap'].fillna(20)
                size = size_raw
            else:
                size = [20] * len(plot_df)

            if len(plot_df) > 0:
                bubble_fig = px.scatter(
                    plot_df,
                    x='Trend_clusterRank',
                    y='Latest_clusterRank',
                    color='Lista' if 'Lista' in plot_df.columns else None,
                    color_discrete_map=color_discrete_map,
                    hover_name=plot_df.index if show_tickers else None,
                    text=plot_df.index if show_tickers else None,
                    size=size_raw, # if 'marketCap' in plot_df.columns else [20]*len(plot_df),
                    hover_data={},
                    labels={
                        'Trend_clusterRank': get_display_name('Trend_clusterRank'),
                        'Latest_clusterRank': get_display_name('Latest_clusterRank'),
                        'Lista': get_display_name('Lista'),
                        #'hover_summary': 'Summary',
                        'size': 'Market Cap'
                    },
                    title='Total Trend Score vs Total Latest Score',
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
                st.plotly_chart(bubble_fig, use_container_width=True)
            else:
                st.info('No stocks in the selected score range (after removing rows with saknade värden).')
        else:
            st.info('No stocks in the selected score range.')

    # =============================
    # MAIN TABLE DISPLAY
    # =============================
    # Create a DataFrame for display in the main table.
    # This DataFrame is now based on the slider-filtered data
    # and contains only 'rank_Score' columns, keeping the Ticker as index.
    df_display = df_filtered_by_sliders[rank_score_columns].copy() # Keep index

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
    cols.insert(0, cols.pop(cols.index('Agg. Rank trend 4 år'))) 
    cols.insert(0, cols.pop(cols.index('Agg. Rank sen. året'))) 
    cols.insert(0, cols.pop(cols.index('Shortlist'))) 
    cols.insert(0, cols.pop(cols.index('Välj')))  # Move 'Agg. Rank trend 4 år' to the front
    df_display = df_display[cols]  # Reorder columns
    # Update rank_score_columns to reflect the new names for shortlist display
    display_rank_score_columns = df_display.columns.tolist()

    # Get the number of stocks after filtering by sliders
    num_filtered_stocks = len(df_display)
    st.subheader(f"Filtered Stock Information ({num_filtered_stocks} aktier)")

    # Use st.data_editor to display the table with interactive checkboxes
    edited_df = st.data_editor(
        df_display,
        use_container_width=True,
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
                help="Add the stock to your personal shortlist",
                default=False,
                width="small",
                pinned=True
            )
        },
        key="stock_selection_editor" # Unique key to manage state
    )

    # Logic to handle checkbox selection for plotting
    selected_rows_plot = edited_df[edited_df['Välj']]
    st.info("Markera rutan under 'Välj' för att visa aktiedata. Markera rutan under 'Shortlist' för att lägga till aktien i din bevakningslista.")

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
        selected_stock_dict = df_new_ranks.loc[selected_stock_ticker].to_dict()
    # Logic to handle Shortlist
    shortlisted_stocks = edited_df[edited_df['Shortlist']]

    st.markdown("---")
    st.subheader("Your Shortlist")

    if not shortlisted_stocks.empty:
        # Display only Ticker (index) and the renamed rank_Score columns for shortlist
        download_columns = [col for col in display_rank_score_columns if col != 'Shortlist' and col != 'Välj']  # Remove 'Shortlist' and 'Välj' columns from display
        st.dataframe(
            shortlisted_stocks[download_columns], # Ticker is already the index
            hide_index=False, # Show the index (Ticker) for the shortlist as well
            use_container_width=True
        )
        
        st.download_button("Ladda ner bevakningslista", data=shortlisted_stocks[download_columns].to_csv(), file_name="shortlist.csv", mime="text/csv")
    else:
        st.info("Din bevakningslista är tom. Markera rutan under 'Shortlist' för att lägga till aktier.")

    st.markdown("---")        
    if selected_stock_ticker:
        st.subheader(f"Kort info om: {selected_stock_dict['Name'] if 'Name' in selected_stock_dict else 'N/A'}")
        left_col, right_col = st.columns([1,2], gap='medium', border=False)
        with left_col:
            
            st.write(f"**Ticker:**   \n{selected_stock_ticker}")
            st.write(f"**Lista:**   \n{selected_stock_dict['Lista'] if 'Lista' in selected_stock_dict else 'N/A'}")
            st.write(f"**Sektor:**   \n{selected_stock_dict['Sektor'] if 'Sektor' in selected_stock_dict else 'N/A'}")
            st.write(f"**Marknadsvärde:**   \n{selected_stock_dict['marketCap_MSEK'] if 'marketCap_MSEK' in selected_stock_dict else 'N/A'}")
        with right_col:
            #st.subheader("Företagsbeskrivning")
            longBusinessSummary = df_long_business_summary.loc[selected_stock_ticker]
            with st.popover(f"{longBusinessSummary.values[0][0:500]}...",use_container_width=True):
                st.write(longBusinessSummary.values[0] if not longBusinessSummary.empty else "Ingen lång företagsbeskrivning tillgänglig för denna aktie.")

    st.subheader("Kursutveckling och Trendlinje")

    if selected_stock_ticker:
        # Add slider for PWLF
        label = "Antal linjesegment för trendlinje"
        linjesegments =[1, 2, 3, 4, 5]
        num_segments = st.segmented_control(label, linjesegments, selection_mode='single', default=1, key="pwlf_slider")
        price_file_path = os.path.join(CSV_DATA_DIR, "price_data.csv")
        if os.path.exists(price_file_path):
            df_price_all = pd.read_csv(price_file_path)
            df_price = df_price_all[df_price_all['Ticker'] == selected_stock_ticker].copy()
            df_price['Date'] = pd.to_datetime(df_price['Date']) # Convert 'Date' to datetime object

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
                    mode='lines', name='SMA Kort',
                    line=dict(color='red', width=1, dash='dot')))

            # Add SMA_medium
            if 'SMA_medium' in df_price.columns:
                fig.add_trace(go.Scatter(x=df_price['Date'], y=df_price['SMA_medium'],
                    mode='lines', name='SMA Medel',
                    line=dict(color='green', width=1, dash='dash')))

            # Add SMA_long
            if 'SMA_long' in df_price.columns:
                fig.add_trace(go.Scatter(x=df_price['Date'], y=df_price['SMA_long'],
                    mode='lines', name='SMA Lång',
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
                hovermode="x unified",
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

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning(f"Prisdatafil saknas: {price_file_path}. Kontrollera att filen finns i mappen '{CSV_DATA_DIR}/'.")

    else:
        st.info("Markera en ruta under 'Välj' i tabellen ovan för att visa prisutvecklingen.")

    # Bar plot for all pct_ columns for selected_stock_ticker
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
        st.plotly_chart(fig_pct, use_container_width=True, key=f"pct_bar_{selected_stock_ticker}")

    # Only show the following sections if a stock is selected
    if selected_stock_dict is not None and selected_stock_ticker is not None:
        # Bar plot for all cagr columns for selected_stock_ticker using selected_stock_dict
        cagr_cols = [col for col in selected_stock_dict.keys() if col.startswith('cagr')]
        if cagr_cols:
            cagr_values = [float(selected_stock_dict.get(col, float('nan'))) for col in cagr_cols]
            fig_cagr = go.Figure(go.Bar(
                x=[get_display_name(col) for col in cagr_cols],
                y=[v * 100 for v in cagr_values],  # Convert to percent
                marker_color='royalblue',
                text=[f"{v*100:.2f}%" if not pd.isna(v) else "" for v in cagr_values],
                textposition='auto',
            ))
            fig_cagr.update_layout(
                title=f"CAGR över 4 år för {selected_stock_dict['Name']} ({selected_stock_ticker})",
                xaxis_title="Mått",
                yaxis_title="Procent",
                height=350,
                margin=dict(l=10, r=10, t=40, b=10),
                yaxis=dict(ticksuffix="%", tickformat=".0f")
            )
            st.plotly_chart(fig_cagr, use_container_width=True, key=f"cagr_bar_{selected_stock_ticker}")


        

        # =============================
        # RANKING FOR SELECTED STOCK
        # =============================
        st.subheader(f"Sammanvägd rank för: {selected_stock_dict['Name']} ({selected_stock_ticker})")
        if not df_filtered_by_sliders.empty and categories:
            clusterRank_trend_items = {col: val for col, val in selected_stock_dict.items() if "_clusterRank" in col and "trend" in col.lower()}
            df_clusterRank_trend = pd.DataFrame.from_dict(clusterRank_trend_items, orient='index', columns=['Trend Rank'])
            df_clusterRank_trend['Kategori']= 'AGGREGERAD RANK'
            catRank_trend_items = {col: val for col, val in selected_stock_dict.items() if "_catRank" in col and "trend" in col.lower()}
            df_catRank_trend = pd.DataFrame.from_dict(catRank_trend_items, orient='index', columns=['Trend Rank']).reset_index()
            df_catRank_trend['Kategori'] = df_catRank_trend['index'].str.split('_', expand=True)[0]
            df_trend_combined = pd.concat([df_catRank_trend, df_clusterRank_trend.reset_index()], ignore_index=True, sort=False)

            clusterRank_latest_items = {col: val for col, val in selected_stock_dict.items() if "_clusterRank" in col and "latest" in col.lower()}
            df_clusterRank_latest = pd.DataFrame.from_dict(clusterRank_latest_items, orient='index', columns=['Latest Rank'])
            df_clusterRank_latest['Kategori']= 'AGGREGERAD RANK'
            catRank_latest_items = {col: val for col, val in selected_stock_dict.items() if "_catRank" in col and "latest" in col.lower()}
            df_catRank_latest = pd.DataFrame.from_dict(catRank_latest_items, orient='index', columns=['Latest Rank']).reset_index()
            df_catRank_latest['Kategori'] = df_catRank_latest['index'].str.split('_', expand=True)[0]
            df_latest_combined = pd.concat([df_catRank_latest, df_clusterRank_latest.reset_index()], ignore_index=True, sort=False)
            # Merge the trend and latest DataFrames on 'Kategori'
            df_catRank_merged = pd.merge(df_trend_combined, df_latest_combined, on='Kategori', suffixes=('_trend', '_latest'))
            # -------------------------------------------------------------
            # PROGRESS BARS: LATEST AND TREND RANKINGS
            # -------------------------------------------------------------
            col_left, col_right = st.columns(2, gap='medium', border=False)

            with col_left:
                st.dataframe(
                    df_catRank_merged[['Kategori', 'Trend Rank']] # Select columns first
                    .style.map(color_progress, subset=['Trend Rank']), # Apply progress bar coloring
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Trend Rank": st.column_config.ProgressColumn(
                                "Trend Rank",
                                help="Rankingvärde (0-100)",
                                min_value=0,
                                max_value=100,
                                format="%.1f"
                            ),
                    }
                )

            with col_right:
                st.dataframe(
                    df_catRank_merged[['Kategori', 'Latest Rank']] # Select columns first
                    .style.map(color_progress, subset=['Latest Rank']), # Apply progress bar coloring
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Latest Rank": st.column_config.ProgressColumn(
                                "Latest Rank",
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
            st.subheader('Category Trend Ratio Breakdown (Last 4 Years)')
            # Create DataFrames for trend and latest ratio ranks
            ratioRank_latest_items = {col: val for col, val in selected_stock_dict.items() if "_ratioRank" in col and "latest" in col.lower()}
            df_ratioRank_latest = pd.DataFrame.from_dict(ratioRank_latest_items, orient='index', columns=['Rank']).reset_index()
            df_ratioRank_latest['Ratio_name'] = df_ratioRank_latest['index'].str.split('_', expand=True)[0]

            ratioRank_trend_items = {col: val for col, val in selected_stock_dict.items() if "_ratioRank" in col and "trend" in col.lower()}
            df_ratioRank_trend = pd.DataFrame.from_dict(ratioRank_trend_items, orient='index', columns=['Rank']).reset_index()
            df_ratioRank_trend['Ratio_name'] = df_ratioRank_trend['index'].str.split('_', expand=True)[0]
            df_ratioRank_merged = pd.merge(df_ratioRank_trend, df_ratioRank_latest, on='Ratio_name', suffixes=('_trend', '_latest'))
            df_ratioRank_merged.rename(columns={'Rank_trend': 'Trend Rank', 'Rank_latest': 'Latest Rank'}, inplace=True)

            # Load help texts from config if available
            ratio_help_texts = config.get('ratio_help_texts', {}) if 'config' in locals() or 'config' in globals() else {}
            #st.write("category_ratios:", category_ratios.items())  # Debugging line to show category_ratios
            for cat, cat_dict in category_ratios.items():

                if cat.endswith('trend_ratioRank'):
                    display_cat = cat.replace('_trend_ratioRank', '')
                    # Use a visually distinct box for each category, with extra margin for spacing
                    with st.container(border=True):
                        st.dataframe(
                            df_catRank_merged[df_catRank_merged['Kategori'] == display_cat][['Kategori', 'Trend Rank', 'Latest Rank']].style.map(color_progress, subset=['Trend Rank', 'Latest Rank']),
                            hide_index=True,
                            use_container_width=True,
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
                                    )
                            }
                        )
                        
                        ratios = [ratio for ratio in cat_dict]
                        cols = st.columns(len(ratios), border=True,gap="small") if ratios else []
                        for idx, ratio in enumerate(ratios):
                            
                            base_ratio = ratio.replace('_trend_ratioRank', '')
                            year_cols = [col for col in df_new_ranks.columns if col.startswith(base_ratio + '_year_')]
                            # Filter out columns where the value for the selected stock is NaN
                            year_cols = [col for col in year_cols if not pd.isna(df_new_ranks.loc[selected_stock_ticker, col])]
                            year_cols_sorted = sorted(year_cols, key=lambda x: int(x.split('_')[-1]), reverse=False)
                            year_cols_last4 = year_cols_sorted[-4:]
                            latest_rank_col = f"{base_ratio}_latest_ratioRank"
                            trend_rank_col = f"{base_ratio}_trend_ratioRank"
                            with cols[idx]:
                                if year_cols_last4:
                                    values = df_new_ranks.loc[selected_stock_ticker, year_cols_last4].values.astype(float)
                                    years = [int(col.split('_')[-1]) for col in year_cols_last4]
                                    # Linear regression for trend line
                                    if len(years) > 1:
                                        coeffs = np.polyfit(years, values, 1)
                                        trend_vals = np.polyval(coeffs, years)
                                    else:
                                        trend_vals = values
                                    fig = go.Figure()
                                    colors = ['lightblue'] * (len(years) - 1) + ['royalblue']
                                    fig.add_trace(go.Bar(x=years, y=values, marker_color=colors, name=base_ratio, showlegend=False))
                                    fig.add_trace(go.Scatter(
                                        x=years, 
                                        y=trend_vals, 
                                        mode='lines', 
                                        name='Trend',
                                        line=dict(color='#888888', dash='dot', width=6),  # Medium-dark gray
                                        showlegend=False
                                    ))
                                    fig.update_layout(title=f"{base_ratio}", 
                                                      height=250, 
                                                      margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                                    st.plotly_chart(fig, use_container_width=True, key=f"{cat}_{base_ratio}_bar")
                                    latest_rank = df_new_ranks.loc[selected_stock_ticker, latest_rank_col] if latest_rank_col in df_new_ranks.columns else 'N/A'
                                    trend_rank = df_new_ranks.loc[selected_stock_ticker, trend_rank_col] if trend_rank_col in df_new_ranks.columns else 'N/A'
                                else:
                                    st.warning(f"Ingen data för de senaste 4 åren för {base_ratio}. Trend Rank och Latest Rank sätts till 50 (neutral).")
                                # Bullet plots for the two ranks in two columns: trend (left), latest (right)
                                #st.write(f"**{ratio}**")
                                st.dataframe(
                                    df_ratioRank_merged[df_ratioRank_merged['index_trend'] == ratio][['Trend Rank', 'Latest Rank']].style.map(color_progress, subset=['Trend Rank', 'Latest Rank']),
                                    hide_index=True,
                                    use_container_width=True,
                                    column_config={
                                        "Latest Rank": st.column_config.ProgressColumn(
                                                "Latest Rank",
                                                help=ratio_help_texts.get(ratio),
                                                min_value=0,
                                                max_value=100,
                                                format="%.1f",
                                                width="small",
                                            ),
                                        "Trend Rank": st.column_config.ProgressColumn(
                                                "Trend Rank",
                                                help=ratio_help_texts.get(ratio),
                                                min_value=0,
                                                max_value=100,
                                                format="%.1f",
                                                width="small"
                                            )
                                    }
                                )
                                

                st.markdown("<br>", unsafe_allow_html=True) # Lägger till tre radbrytningar
                # Clear the empty space before each category
            # --- END: Show ratio bar charts for each _trend_rank category ---

except FileNotFoundError:
    st.error(f"Error: Main file '{CSV_FILE_NAME}' not found in directory '{CSV_DATA_DIR}'. Check the path.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.subheader("About this application")
st.info("To run this app locally: Save the code as a .py file (e.g., `app.py`) and run `streamlit run app.py` in your terminal.")
st.caption("Make sure your CSV files are in the specified folders (`csv-data` for the main file and `data` for the price files).")

