import streamlit as st
import numpy as np
import pandas as pd
import pwlf
import plotly.graph_objects as go # Import Plotly

def generate_price_chart(config, CSV_PATH, add_moving_averages, get_display_name, selected_stock_ticker, selected_stock_dict):
    if selected_stock_ticker:
                # Add slider for PWLF
        label = "Antal linjesegment för trendlinje"
        linjesegments =[1, 2, 3, 4, 5]
        num_segments = st.segmented_control(label, linjesegments, selection_mode='single', default=1, key="pwlf_slider")
        label = "Historik (år bakåt i tiden)"
        linjesegments =[4,1, 0.5, 0.25] # years
        historik_segments = st.segmented_control(label, linjesegments, selection_mode='single', default=4, key="historik_segments")
        price_file_path = CSV_PATH / config["price_data_file"]
        if price_file_path.exists():
            df_price_all = pd.read_csv(price_file_path)
            df_price_all['Date'] = pd.to_datetime(df_price_all['Date']) # Convert 'Date' to datetime object before filtering
            df_price = df_price_all[df_price_all['Ticker'] == selected_stock_ticker].copy()
            df_price = add_moving_averages(df_price)
                    # Cap data to historik_segments years back
            if not df_price.empty and historik_segments:
                max_date = df_price['Date'].max()
                        # Use days for fractional years to avoid DateOffset ambiguity
                try:
                    years_float = float(historik_segments)
                    if years_float.is_integer():
                        min_date = max_date - pd.DateOffset(years=int(years_float))
                    else:
                                # Approximate 1 year as 365.25 days
                        days = int(years_float * 365.25)
                        min_date = max_date - pd.DateOffset(days=days)
                except Exception as e:
                    min_date = max_date - pd.DateOffset(years=1)  # fallback to 1 year
                df_price = df_price[df_price['Date'] >= min_date]
                    #df_price = add_moving_averages(df_price)
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
