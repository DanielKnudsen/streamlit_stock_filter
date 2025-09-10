# User Monitoring Implementation for Streamlit App

## Simple Session Tracking

Add this to your `app.py` to track concurrent users:

```python
import streamlit as st
import time
import json
import os
from datetime import datetime, timedelta
import uuid

# User tracking functions
def init_user_tracking():
    """Initialize user tracking in session state"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
        st.session_state.session_start = datetime.now()
    
    # Log user activity
    log_user_activity()

def log_user_activity():
    """Log user activity to a simple JSON file"""
    user_log = {
        'user_id': st.session_state.user_id,
        'timestamp': datetime.now().isoformat(),
        'page': st.get_option('page_title') or 'main',
        'action': 'page_view'
    }
    
    # Simple file-based logging (you can upgrade to database later)
    log_file = 'user_activity.jsonl'
    with open(log_file, 'a') as f:
        f.write(json.dumps(user_log) + '\n')

def get_concurrent_users():
    """Get number of concurrent users (active in last 5 minutes)"""
    try:
        if not os.path.exists('user_activity.jsonl'):
            return 0
            
        active_threshold = datetime.now() - timedelta(minutes=5)
        active_users = set()
        
        with open('user_activity.jsonl', 'r') as f:
            for line in f:
                try:
                    log = json.loads(line.strip())
                    log_time = datetime.fromisoformat(log['timestamp'])
                    if log_time > active_threshold:
                        active_users.add(log['user_id'])
                except:
                    continue
                    
        return len(active_users)
    except:
        return 0

def display_user_stats():
    """Display user statistics in sidebar (admin only)"""
    if st.sidebar.checkbox("Show User Stats (Debug)", key="debug_stats"):
        concurrent = get_concurrent_users()
        st.sidebar.metric("Concurrent Users", concurrent)
        st.sidebar.metric("Session ID", st.session_state.user_id[:8])
        st.sidebar.metric("Session Duration", 
                         str(datetime.now() - st.session_state.session_start).split('.')[0])

# Add to your main app.py
def main():
    # Initialize user tracking at the start
    init_user_tracking()
    
    # Your existing app code here...
    
    # Optional: Display stats in development
    if st.sidebar.checkbox("Debug Mode"):
        display_user_stats()
```

## Advanced Monitoring with Database

For production use, implement proper database tracking:

```python
import sqlite3
from datetime import datetime, timedelta

def init_user_db():
    """Initialize SQLite database for user tracking"""
    conn = sqlite3.connect('user_activity.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            user_id TEXT,
            session_start TIMESTAMP,
            last_activity TIMESTAMP,
            page_views INTEGER DEFAULT 1,
            user_agent TEXT,
            ip_address TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS page_views (
            user_id TEXT,
            timestamp TIMESTAMP,
            page TEXT,
            action TEXT,
            processing_time REAL
        )
    ''')
    
    conn.commit()
    conn.close()

def log_user_session():
    """Log user session to database"""
    import streamlit.web.server.websocket_headers as ws_headers
    
    # Get user info
    user_agent = st.get_option('browser.gatherUsageStats', '')
    
    conn = sqlite3.connect('user_activity.db')
    cursor = conn.cursor()
    
    # Update or insert user session
    cursor.execute('''
        INSERT OR REPLACE INTO user_sessions 
        (user_id, session_start, last_activity, page_views, user_agent)
        VALUES (?, ?, ?, 
                COALESCE((SELECT page_views FROM user_sessions WHERE user_id = ?), 0) + 1,
                ?)
    ''', (
        st.session_state.user_id,
        st.session_state.session_start,
        datetime.now(),
        st.session_state.user_id,
        user_agent
    ))
    
    conn.commit()
    conn.close()

def get_detailed_user_stats():
    """Get detailed user statistics"""
    conn = sqlite3.connect('user_activity.db')
    cursor = conn.cursor()
    
    # Concurrent users (active in last 5 minutes)
    cursor.execute('''
        SELECT COUNT(DISTINCT user_id) 
        FROM user_sessions 
        WHERE last_activity > datetime('now', '-5 minutes')
    ''')
    concurrent = cursor.fetchone()[0]
    
    # Daily active users
    cursor.execute('''
        SELECT COUNT(DISTINCT user_id) 
        FROM user_sessions 
        WHERE DATE(last_activity) = DATE('now')
    ''')
    daily_active = cursor.fetchone()[0]
    
    # Total page views today
    cursor.execute('''
        SELECT COUNT(*) 
        FROM page_views 
        WHERE DATE(timestamp) = DATE('now')
    ''')
    daily_views = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'concurrent_users': concurrent,
        'daily_active_users': daily_active,
        'daily_page_views': daily_views
    }
```

## Google Analytics Integration

For comprehensive tracking, add Google Analytics:

```python
# Add to your app.py
def inject_ga():
    """Inject Google Analytics tracking"""
    ga_code = """
    <!-- Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'GA_MEASUREMENT_ID');
    </script>
    """
    
    st.components.v1.html(ga_code, height=0)

# Add to main function
def main():
    inject_ga()  # Track with Google Analytics
    init_user_tracking()  # Your custom tracking
    # ... rest of app
```

## Real-time Dashboard

Create a simple monitoring dashboard:

```python
# monitoring_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

def create_monitoring_dashboard():
    st.title("Indicatum User Monitoring")
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    stats = get_detailed_user_stats()
    
    with col1:
        st.metric("Concurrent Users", stats['concurrent_users'])
    with col2:
        st.metric("Daily Active", stats['daily_active_users'])
    with col3:
        st.metric("Daily Views", stats['daily_page_views'])
    with col4:
        st.metric("Performance", "Good" if stats['concurrent_users'] < 50 else "Warning")
    
    # Usage over time chart
    df_usage = get_hourly_usage_data()
    fig = px.line(df_usage, x='hour', y='users', title='Hourly Active Users')
    st.plotly_chart(fig)
    
    # Performance warnings
    if stats['concurrent_users'] > 50:
        st.warning("⚠️ High concurrent usage detected. Consider scaling soon.")
    if stats['concurrent_users'] > 80:
        st.error("🚨 Critical usage levels. App performance may be degraded.")

if __name__ == "__main__":
    create_monitoring_dashboard()
```

## Simple Implementation to Start

Add this basic tracking to your current `app.py`:

```python
# At the top of app.py
import uuid
import json
from datetime import datetime

# Initialize session tracking
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.start_time = datetime.now()

# Log activity (simple file-based)
def log_activity(action="page_view"):
    log_entry = {
        'user_id': st.session_state.user_id,
        'timestamp': datetime.now().isoformat(),
        'action': action
    }
    
    with open('user_logs.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

# Call this on each page load
log_activity()

# Optional: Show current stats in sidebar (development only)
if st.sidebar.button("Show User Count"):
    # Simple concurrent user count
    concurrent = len(set([json.loads(line)['user_id'] 
                         for line in open('user_logs.jsonl', 'r').readlines()[-100:]]))
    st.sidebar.write(f"Recent active users: {concurrent}")
```

This gives you immediate visibility into usage patterns and helps you plan for scaling before you hit Streamlit Cloud's limits!
