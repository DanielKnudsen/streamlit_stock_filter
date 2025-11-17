#!/usr/bin/env python3
"""
Email notification system for saved filter results.

This script processes all users with saved filters and sends email notifications
based on their selected frequency (daily, weekly, monthly).
"""

import os
import pandas as pd
from pathlib import Path
from supabase import create_client
from rank import load_config
from datetime import datetime


def get_supabase_client():
    """Get Supabase client using environment variables"""
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_ANON_KEY')
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY environment variables required")
    return create_client(url, key)


def load_stock_data():
    """Load current stock evaluation data"""
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'remote')
    CSV_PATH = Path('data') / ('local' if ENVIRONMENT == 'local' else 'remote')
    config = load_config("rank-config.yaml")

    df = pd.read_csv(CSV_PATH / config["results_file"], index_col=0)
    return df, config


def apply_saved_filter(df, filter_data):
    """Apply saved filter configuration to dataframe"""
    try:
        filtered_df = df.copy()

        # Apply slider filters
        for key, value in filter_data.items():
            if key.startswith('slider_') and isinstance(value, (list, tuple)) and len(value) == 2:
                column_name = key.replace('slider_', '')
                # Remove suffixes like _sektor
                if column_name.endswith('_sektor'):
                    column_name = column_name[:-7]
                min_val, max_val = value
                if column_name in filtered_df.columns:
                    filtered_df = filtered_df[
                        (filtered_df[column_name] >= min_val) &
                        (filtered_df[column_name] <= max_val) |
                        (filtered_df[column_name].isna())
                    ]

        # Apply pill filters
        for key, value in filter_data.items():
            if key.startswith('pills_') and isinstance(value, list):
                column_name = key.replace('pills_', '')
                if column_name in filtered_df.columns and value:
                    # Extract original values (remove counts)
                    clean_values = [val.rsplit(' (', 1)[0] for val in value]
                    filtered_df = filtered_df[filtered_df[column_name].isin(clean_values)]

        # Apply ticker filter
        if 'ticker_input' in filter_data and filter_data['ticker_input'].strip():
            tickers = [t.strip().upper() for t in filter_data['ticker_input'].split(',') if t.strip()]
            if tickers:
                filtered_df = filtered_df[filtered_df.index.str.upper().isin(tickers)]

        return filtered_df

    except Exception as e:
        print(f"Error applying filter: {e}")
        return pd.DataFrame()


def generate_email_content(user_email, filter_name, filtered_results, df_original, config):
    """Generate HTML email content"""

    # Get top 10 results
    top_results = filtered_results.head(10)

    # Calculate summary stats
    total_stocks = len(df_original)
    matching_stocks = len(filtered_results)
    match_percentage = (matching_stocks / total_stocks) * 100 if total_stocks > 0 else 0

    # Create HTML content
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                     color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
            .summary {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #667eea; color: white; }}
            .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee;
                     font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📈 Indicatum Insights - Filterresultat</h1>
            <h2>Filter: {filter_name}</h2>
        </div>

        <div class="summary">
            <h3>Sammanfattning</h3>
            <p><strong>Matchande aktier:</strong> {matching_stocks} av {total_stocks} ({match_percentage:.1f}%)</p>
            <p><strong>Genererad:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>

        <h3>Topp 10 resultat</h3>
        <table>
            <tr>
                <th>Ticker</th>
                <th>Namn</th>
                <th>Sektor</th>
                <th>Lista</th>
                <th>Total Rank</th>
            </tr>
    """

    for ticker, row in top_results.iterrows():
        html += f"""
            <tr>
                <td>{ticker}</td>
                <td>{row.get('longBusinessSummary', 'N/A')[:50]}...</td>
                <td>{row.get('Sektor', 'N/A')}</td>
                <td>{row.get('Lista', 'N/A')}</td>
                <td>{row.get('total_rank', 'N/A')}</td>
            </tr>
        """

    html += """
        </table>

        <div class="footer">
            <p><strong>Om Indicatum Insights:</strong> Smart filtrering + djup analys = bättre investeringsbeslut</p>
            <p>Du får detta mejl eftersom du har sparat ett filter med e-postaviseringar aktiverat.</p>
            <p>Vill du ändra dina inställningar? Logga in på <a href="https://indicatum-insights.streamlit.app">appen</a> och uppdatera dina filter.</p>
            <p><em>Endast för analys & utbildning. Inte finansiell rådgivning.</em></p>
        </div>
    </body>
    </html>
    """

    return html


def send_email(recipient, subject, html_content):
    """Send email using SMTP (placeholder - implement with your email service)"""
    # This is a placeholder - implement with SendGrid, Mailgun, or SMTP
    print(f"Would send email to {recipient} with subject: {subject}")
    print(f"Email content length: {len(html_content)} characters")

    # TODO: Implement actual email sending
    # Example with smtplib:
    # msg = MIMEMultipart('alternative')
    # msg['Subject'] = subject
    # msg['From'] = 'noreply@indicatum.se'
    # msg['To'] = recipient
    #
    # part = MIMEText(html_content, 'html')
    # msg.attach(part)
    #
    # server = smtplib.SMTP('smtp.gmail.com', 587)
    # server.starttls()
    # server.login('your-email@gmail.com', 'your-password')
    # server.sendmail('your-email@gmail.com', recipient, msg.as_string())
    # server.quit()

    return True


def send_filter_emails(frequency='daily'):
    """Main function to send emails for specified frequency"""

    print(f"Starting email notifications for frequency: {frequency}")

    try:
        # Initialize clients and data
        supabase = get_supabase_client()
        df_current, config = load_stock_data()

        # Get users with filters for this frequency
        # Note: This assumes you have a user_profiles table with email_notifications_enabled
        # For now, we'll get all filter_states with the matching frequency
        response = supabase.table('filter_states').select("""
            id, user_id, name, filter_data, description, frequency,
            auth.users(email)
        """).eq('frequency', frequency).execute()

        filters_to_process = response.data
        print(f"Found {len(filters_to_process)} filters to process")

        emails_sent = 0
        for filter_item in filters_to_process:
            try:
                user_email = filter_item.get('email')
                if not user_email:
                    print(f"No email found for filter {filter_item['id']}")
                    continue

                # Apply filter
                filtered_results = apply_saved_filter(df_current, filter_item['filter_data'])

                if len(filtered_results) == 0:
                    print(f"No results for filter {filter_item['name']} - skipping email")
                    continue

                # Generate email content
                html_content = generate_email_content(
                    user_email,
                    filter_item['name'],
                    filtered_results,
                    df_current,
                    config
                )

                # Send email
                subject = f"📈 Indicatum Insights - {filter_item['name']} ({len(filtered_results)} resultat)"
                success = send_email(user_email, subject, html_content)

                if success:
                    emails_sent += 1
                    print(f"Email sent to {user_email} for filter '{filter_item['name']}'")

                    # Log the email send
                    supabase.table('email_notifications').insert({
                        'user_id': filter_item['user_id'],
                        'filter_id': filter_item['id'],
                        'status': 'sent',
                        'recipient_email': user_email,
                        'frequency': frequency,
                        'results_count': len(filtered_results)
                    }).execute()

            except Exception as e:
                print(f"Error processing filter {filter_item.get('id')}: {e}")
                # Log failed send
                try:
                    supabase.table('email_notifications').insert({
                        'user_id': filter_item.get('user_id'),
                        'filter_id': filter_item.get('id'),
                        'status': 'failed',
                        'recipient_email': filter_item.get('email')
                    }).execute()
                except Exception as e:
                    print(f"Error logging failed send: {e}")
                    # Still try to log the failure
                    try:
                        supabase.table('email_notifications').insert({
                            'user_id': filter_item.get('user_id'),
                            'filter_id': filter_item.get('id'),
                            'status': 'failed',
                            'recipient_email': filter_item.get('email'),
                            'frequency': frequency
                        }).execute()
                    except Exception as log_error:
                        print(f"Failed to log error: {log_error}")

        print(f"Email sending complete. Sent {emails_sent} emails.")

    except Exception as e:
        print(f"Critical error in send_filter_emails: {e}")
        raise


if __name__ == "__main__":
    # Get frequency from environment variable (set by GitHub Actions)
    frequency = os.getenv('FREQUENCY', 'daily')
    send_filter_emails(frequency)