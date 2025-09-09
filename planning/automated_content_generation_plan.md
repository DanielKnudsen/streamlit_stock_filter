# Automated Content Generation Plan for Stock Screening App

**Date:** September 9, 2025  
**Context:** Strategic planning for transforming stock screening app into automated market intelligence system

## Business Vision

Transform the existing Streamlit stock screening application into an automated market intelligence system that generates WordPress blog posts for Swedish stock market investors. The goal is to create a subscription-based service that delivers data-driven insights through automated content generation.

## Target Content Types

1. **Overlooked Opportunities** (Monday)
   - Stocks with strong fundamentals but low market attention
   - Focus on undervalued companies with improving metrics

2. **Sector Pulse** (Wednesday) 
   - Sector rotation analysis and capital flow detection
   - Identify which industries are gaining/losing favor

3. **Quarterly Breakthroughs** (Friday)
   - Companies showing substantial quarterly improvements
   - TTM (trailing twelve months) vs previous year comparisons

## Technical Implementation Strategy

### Phase 1: Data Infrastructure Enhancement

**Objective:** Add user interaction tracking and content generation foundation

**Tasks:**
- Add user interaction tracking to existing Supabase database
- Create content generation tables (articles, publication schedule, subscriber preferences)
- Implement user preference capture (sector interests, risk tolerance, portfolio size)
- Add engagement metrics tracking (which stocks users shortlist, filter preferences)

**Database Schema Extensions:**
```sql
-- User interaction tracking
CREATE TABLE user_interactions (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    ticker VARCHAR(10),
    interaction_type VARCHAR(50), -- 'shortlist', 'filter_view', 'detail_view'
    timestamp TIMESTAMP DEFAULT NOW(),
    session_id VARCHAR(100)
);

-- Content generation
CREATE TABLE generated_articles (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200),
    content TEXT,
    article_type VARCHAR(50), -- 'overlooked', 'sector_pulse', 'quarterly'
    publication_date DATE,
    wordpress_post_id INTEGER,
    tickers_mentioned TEXT[], -- Array of ticker symbols
    created_at TIMESTAMP DEFAULT NOW()
);

-- User preferences
CREATE TABLE user_preferences (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id),
    preferred_sectors TEXT[],
    risk_tolerance VARCHAR(20), -- 'conservative', 'moderate', 'aggressive'
    portfolio_size_range VARCHAR(30), -- 'under_100k', '100k_500k', 'over_500k'
    content_frequency VARCHAR(20), -- 'daily', 'weekly', 'monthly'
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Phase 2: Automated Content Algorithms

**2.1 Overlooked Opportunities Detection**

```python
def find_overlooked_opportunities(df_ranks, lookback_days=30):
    """
    Identify stocks with strong fundamentals but low market attention
    
    Criteria:
    - High fundamental ranks (>70) in multiple categories
    - Low recent user interaction volume
    - Price below long-term moving averages (potential value)
    - Improving TTM metrics vs previous year
    """
    # Strong fundamentals filter
    fundamental_strong = (
        (df_ranks['Profitability_latest_catRank'] > 70) &
        (df_ranks['Growth_latest_catRank'] > 70) &
        (df_ranks['Financial_Health_latest_catRank'] > 60)
    )
    
    # Low attention filter (would need user interaction data)
    # low_attention = df_ranks['user_interaction_score'] < 0.3
    
    # Value indicators
    value_indicators = (
        (df_ranks['pct_SMA_long'] < -5) &  # Below long-term average
        (df_ranks['PE_latest_ratioRank'] > 60)  # Reasonable valuation
    )
    
    # Improving metrics
    improving_metrics = (
        (df_ranks['Revenue_ttm_diff'] > 0) &
        (df_ranks['NetIncome_ttm_diff'] > 0)
    )
    
    overlooked = df_ranks[
        fundamental_strong & value_indicators & improving_metrics
    ].head(10)
    
    return overlooked

# Content generation template
def generate_overlooked_content(overlooked_stocks):
    """Generate WordPress content for overlooked opportunities"""
    
    content_template = """
    # Veckans Underskattade Möjligheter
    
    Vår automatiserade analys har identifierat {count} aktier som uppvisar starka fundamenta 
    men verkar ha undgått marknadens uppmärksamhet. Dessa bolag kombinerar:
    
    - Hög lönsamhet och tillväxtpotential
    - Attraktiva värderingar relativt historiska nivåer  
    - Förbättrade senaste månader resultat
    
    ## Denna Veckas Fynd
    
    {stock_analysis}
    
    ## Analysmetodik
    
    Vår algoritm utvärderar över {total_stocks} svenska aktier dagligen och rangordnar dem 
    baserat på {criteria_count} finansiella nyckeltal. Aktier kvalificerar sig för 
    "underskattade möjligheter" när de uppfyller strikta kriterier för fundamental styrka 
    samtidigt som de handlas under sitt historiska genomsnitt.
    
    *Automatiskt genererat {date} | Baserat på senaste finansiella rapporter*
    """
    
    return content_template.format(
        count=len(overlooked_stocks),
        stock_analysis=format_stock_analysis(overlooked_stocks),
        total_stocks=len(df_ranks),
        criteria_count=50,  # Number of financial ratios analyzed
        date=datetime.now().strftime("%Y-%m-%d")
    )
```

**2.2 Sector Rotation Detection**

```python
def detect_sector_rotation(df_ranks, lookback_periods=[30, 90, 180]):
    """
    Detect capital flow between sectors using momentum and performance metrics
    
    Algorithm:
    1. Calculate sector-weighted performance across multiple timeframes
    2. Identify momentum shifts (acceleration/deceleration)
    3. Compare relative strength between sectors
    4. Flag significant capital rotation events
    """
    
    sector_performance = {}
    
    for period in lookback_periods:
        # Calculate sector averages for key momentum indicators
        sector_stats = df_ranks.groupby('Sektor').agg({
            f'pct_SMA_short': 'mean',  # Short-term momentum
            f'pct_SMA_medium': 'mean',  # Medium-term momentum  
            f'pct_SMA_long': 'mean',   # Long-term momentum
            'marketCap': 'sum',        # Total sector market cap
            'Volume_latest_ratioRank': 'mean'  # Trading activity
        })
        
        # Calculate relative strength vs market
        market_avg = df_ranks[f'pct_SMA_short'].mean()
        sector_stats[f'relative_strength_{period}d'] = (
            sector_stats[f'pct_SMA_short'] - market_avg
        )
        
        sector_performance[period] = sector_stats
    
    # Identify sectors with significant momentum changes
    rotation_signals = {}
    for sector in sector_stats.index:
        # Compare short vs long-term relative performance
        short_momentum = sector_performance[30].loc[sector, 'relative_strength_30d']
        long_momentum = sector_performance[180].loc[sector, 'relative_strength_180d']
        
        momentum_acceleration = short_momentum - long_momentum
        
        # Flag significant rotations (threshold can be tuned)
        if abs(momentum_acceleration) > 5.0:  # 5% threshold
            rotation_signals[sector] = {
                'direction': 'inflow' if momentum_acceleration > 0 else 'outflow',
                'magnitude': abs(momentum_acceleration),
                'short_term_performance': short_momentum,
                'long_term_performance': long_momentum
            }
    
    return rotation_signals

def generate_sector_pulse_content(rotation_signals):
    """Generate sector rotation analysis content"""
    
    content_template = """
    # Sektorspuls: Kapitalflöden och Rotation
    
    Vår sektoranalys visar tydliga tecken på kapitalrotation mellan svenska branschsegment. 
    Baserat på momentum- och volymanalys har vi identifierat {signal_count} sektorer med 
    signifikanta förändringar i investerarnas intresse.
    
    ## Kapital Strömmar In Till:
    {inflow_sectors}
    
    ## Kapital Lämnar:
    {outflow_sectors}
    
    ## Teknisk Analys
    
    {technical_commentary}
    
    ## Investeringsimplikationer
    
    {investment_implications}
    
    *Sektoranalys uppdaterad {date} | Baserad på {data_points} datapunkter*
    """
    
    return content_template
```

**2.3 Quarterly Breakthrough Detection**

```python
def find_quarterly_breakthroughs(df_ranks, improvement_threshold=20):
    """
    Identify companies with substantial quarterly improvements
    
    Focus on TTM vs previous year comparisons across key metrics
    """
    
    # Define key breakthrough metrics
    breakthrough_metrics = [
        'Revenue_ttm_diff',
        'NetIncome_ttm_diff', 
        'OperatingIncome_ttm_diff',
        'FreeCashFlow_ttm_diff'
    ]
    
    breakthroughs = []
    
    for metric in breakthrough_metrics:
        if metric in df_ranks.columns:
            # Find stocks with significant positive changes
            improved_stocks = df_ranks[
                (df_ranks[metric] > improvement_threshold) &
                (df_ranks[metric].notna())
            ].copy()
            
            # Sort by improvement magnitude
            improved_stocks = improved_stocks.sort_values(
                metric, ascending=False
            )
            
            # Add context about the improvement
            improved_stocks['improvement_metric'] = metric
            improved_stocks['improvement_value'] = improved_stocks[metric]
            
            breakthroughs.append(improved_stocks.head(5))
    
    # Combine and deduplicate
    all_breakthroughs = pd.concat(breakthroughs, ignore_index=True)
    
    # Group by ticker to avoid duplicates, keep best improvement
    best_breakthroughs = all_breakthroughs.loc[
        all_breakthroughs.groupby('Ticker')['improvement_value'].idxmax()
    ]
    
    return best_breakthroughs.head(10)

def generate_quarterly_breakthrough_content(breakthroughs):
    """Generate quarterly breakthrough analysis"""
    
    content_template = """
    # Kvartalets Genombrott: Bolag i Acceleration
    
    Analys av senaste kvartalsrapporter avslöjar {count} svenska bolag som visar 
    exceptionella förbättringar jämfört med föregående år. Dessa företag uppvisar 
    tydliga tecken på operationell vändning eller acceleration.
    
    ## Störst Förbättringar
    
    {breakthrough_analysis}
    
    ## Gemensamma Framgångsfaktorer
    
    {success_factors}
    
    ## Risker att Beakta
    
    {risk_factors}
    
    *Kvartalsgenomgång {date} | {reports_analyzed} rapporter analyserade*
    """
    
    return content_template
```

### Phase 3: WordPress Integration & Automation

**3.1 WordPress API Integration**

```python
import requests
from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import NewPost, GetPost

class WordPressPublisher:
    def __init__(self, site_url, username, app_password):
        self.site_url = site_url
        self.username = username
        self.app_password = app_password
        
    def publish_article(self, title, content, category="market-analysis", tags=None):
        """Publish article to WordPress via REST API"""
        
        endpoint = f"{self.site_url}/wp-json/wp/v2/posts"
        
        headers = {
            'Authorization': f'Basic {self._get_auth_string()}',
            'Content-Type': 'application/json'
        }
        
        post_data = {
            'title': title,
            'content': content,
            'status': 'draft',  # Start as draft for review
            'categories': [self._get_category_id(category)],
            'tags': tags or []
        }
        
        response = requests.post(endpoint, json=post_data, headers=headers)
        
        if response.status_code == 201:
            return response.json()['id']
        else:
            raise Exception(f"WordPress publish failed: {response.text}")
    
    def schedule_publication(self, post_id, publish_datetime):
        """Schedule post for future publication"""
        
        endpoint = f"{self.site_url}/wp-json/wp/v2/posts/{post_id}"
        
        headers = {
            'Authorization': f'Basic {self._get_auth_string()}',
            'Content-Type': 'application/json'
        }
        
        update_data = {
            'status': 'future',
            'date': publish_datetime.isoformat()
        }
        
        response = requests.post(endpoint, json=update_data, headers=headers)
        return response.status_code == 200
```

**3.2 Content Generation Scheduler**

```python
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta

def setup_content_scheduler():
    """Setup automated content generation schedule"""
    
    scheduler = BackgroundScheduler()
    
    # Monday: Overlooked Opportunities (6 AM)
    scheduler.add_job(
        func=generate_and_publish_overlooked,
        trigger="cron",
        day_of_week='mon',
        hour=6,
        minute=0,
        id='overlooked_opportunities'
    )
    
    # Wednesday: Sector Pulse (6 AM)  
    scheduler.add_job(
        func=generate_and_publish_sector_pulse,
        trigger="cron", 
        day_of_week='wed',
        hour=6,
        minute=0,
        id='sector_pulse'
    )
    
    # Friday: Quarterly Breakthroughs (6 AM)
    scheduler.add_job(
        func=generate_and_publish_breakthroughs,
        trigger="cron",
        day_of_week='fri', 
        hour=6,
        minute=0,
        id='quarterly_breakthroughs'
    )
    
    scheduler.start()
    return scheduler

def generate_and_publish_overlooked():
    """Full workflow for overlooked opportunities content"""
    
    # Load latest data
    df_ranks = load_latest_ranking_data()
    
    # Find opportunities
    overlooked = find_overlooked_opportunities(df_ranks)
    
    if len(overlooked) > 0:
        # Generate content
        content = generate_overlooked_content(overlooked)
        
        # Publish to WordPress
        wp_publisher = WordPressPublisher(
            site_url=config['wordpress']['site_url'],
            username=config['wordpress']['username'], 
            app_password=config['wordpress']['app_password']
        )
        
        post_id = wp_publisher.publish_article(
            title=f"Underskattade Möjligheter - Vecka {datetime.now().strftime('%W')}",
            content=content,
            category="overlooked-opportunities",
            tags=[stock['Ticker'] for stock in overlooked.to_dict('records')]
        )
        
        # Log to database
        log_generated_content(
            article_type='overlooked',
            wordpress_post_id=post_id,
            tickers_mentioned=overlooked.index.tolist()
        )
        
        return post_id
    
    return None
```

## Implementation Timeline

### Week 1-2: Foundation
- [ ] Set up planning folder and documentation
- [ ] Design database schema extensions
- [ ] Create user interaction tracking system
- [ ] Set up WordPress development environment

### Week 3-4: Core Algorithms  
- [ ] Implement overlooked opportunities detection
- [ ] Develop sector rotation analysis
- [ ] Create quarterly breakthrough identification
- [ ] Build content generation templates

### Week 5-6: Integration & Testing
- [ ] WordPress API integration
- [ ] Content scheduling system
- [ ] End-to-end testing with sample data
- [ ] Performance optimization

### Week 7-8: Launch Preparation
- [ ] User preference collection system
- [ ] Subscription management integration
- [ ] Content quality review process
- [ ] Launch beta testing with select users

## Success Metrics

### Content Quality
- Unique insights per article (target: 5+ actionable points)
- Accuracy of predictions (track 3-month performance)
- User engagement (time on page, shares, comments)

### Business Impact  
- Subscriber conversion rate (target: 2-5% of free users)
- Monthly recurring revenue growth
- User retention and content consumption patterns

### Technical Performance
- Content generation time (target: <5 minutes per article)
- System uptime and reliability (target: 99.5%)
- Data freshness (update within 24 hours of new reports)

## Risk Mitigation

### Content Quality Control
- Manual review process for first 50 articles
- A/B testing of content formats and styles
- User feedback integration system

### Technical Risks
- Fallback to manual content creation if automation fails
- Multiple data source validation
- WordPress backup and recovery procedures

### Business Risks
- Gradual rollout to manage subscriber expectations
- Clear disclaimers about automated analysis limitations
- Professional investment advice disclaimers

## Next Steps

1. **Immediate (This Week)**
   - Review and refine this plan based on feedback
   - Set up development environment for WordPress integration
   - Begin database schema design

2. **Short Term (Next Month)**
   - Implement basic user interaction tracking
   - Develop prototype of overlooked opportunities algorithm
   - Create first content generation template

3. **Medium Term (Next Quarter)**
   - Full implementation of all three content types
   - WordPress integration and automated publishing
   - Beta testing with select users

This plan transforms your existing stock screening application into a comprehensive market intelligence platform that generates valuable, automated content for Swedish investors while creating a sustainable subscription business model.
