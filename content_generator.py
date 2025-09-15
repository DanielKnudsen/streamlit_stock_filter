"""
🤖 Automated Blog Post Generator
================================

Creates data-driven blog posts from quarterly earnings analysis.
Integrates with WordPress API for hands-off publishing.

Features:
- Parses quarterly_changes_analysis.txt for content
- Quality control thresholds
- WordPress REST API integration
- Swedish market optimization
- SEO-friendly content generation
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import requests
from dataclasses import dataclass

# Environment detection pattern (consistent with project)
ENVIRONMENT = os.getenv('ENVIRONMENT', 'local')
DATA_PATH = Path('data') / ('local' if ENVIRONMENT == 'local' else 'remote')

@dataclass
class ContentTemplate:
    """Blog post content structure"""
    title: str
    introduction: str
    analysis_section: str
    stock_highlights: List[str]
    conclusion: str
    tags: List[str]
    category: str = "Quarterly Analysis"

class ContentGenerator:
    """Automated content creation from earnings analysis"""
    
    def __init__(self):
        self.wp_url = os.getenv('WORDPRESS_URL', 'https://indicatum.se')
        self.wp_user = os.getenv('WORDPRESS_USER')
        self.wp_password = os.getenv('WORDPRESS_APP_PASSWORD')
        self.data_path = DATA_PATH
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def load_quarterly_analysis(self) -> Dict:
        """Load latest quarterly analysis results"""
        analysis_file = Path("quarterly_changes_analysis.txt")
        
        if not analysis_file.exists():
            logging.warning("No quarterly analysis file found")
            return {}
            
        # Parse analysis file
        with open(analysis_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract structured data from analysis
        return self._parse_analysis_content(content)
    
    def _parse_analysis_content(self, content: str) -> Dict:
        """Parse quarterly analysis text into structured data"""
        analysis_data = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'significant_changes': [],
            'top_performers': [],
            'underperformers': [],
            'sector_trends': [],
            'summary_stats': {},
            'total_stocks': 0
        }
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Extract total stocks count
            if "Total stocks with quarterly updates:" in line:
                try:
                    analysis_data['total_stocks'] = int(line.split(':')[1].strip())
                except Exception:
                    pass
            
            # Section detection
            elif 'TOP QUARTERLY PERFORMERS' in line:
                current_section = 'top_performers'
            elif 'QUARTERLY UNDERPERFORMERS' in line:
                current_section = 'underperformers'
            elif 'SECTOR PERFORMANCE' in line:
                current_section = 'sector_trends'
            elif line.startswith('#') and current_section == 'top_performers':
                # Parse top performer entry
                if '.' in line and '-' in line:
                    analysis_data['top_performers'].append(line)
            elif line.startswith('•') and current_section == 'underperformers':
                # Parse underperformer entry
                analysis_data['underperformers'].append(line.replace('•', '').strip())
            elif line.startswith('•') and current_section == 'sector_trends':
                # Parse sector trend entry
                analysis_data['sector_trends'].append(line.replace('•', '').strip())
        
        return analysis_data
    
    def evaluate_content_worthiness(self, analysis_data: Dict) -> bool:
        """Determine if analysis warrants blog post creation"""
        # Quality thresholds
        min_total_stocks = 5
        min_top_performers = 3
        min_sectors = 2
        
        # Check if analysis is from today
        is_fresh = analysis_data.get('date') == datetime.now().strftime('%Y-%m-%d')
        
        criteria = [
            analysis_data.get('total_stocks', 0) >= min_total_stocks,
            len(analysis_data.get('top_performers', [])) >= min_top_performers,
            len(analysis_data.get('sector_trends', [])) >= min_sectors,
            is_fresh
        ]
        
        # Force content creation if environment variable is set
        force_content = os.getenv('FORCE_CONTENT', 'false').lower() == 'true'
        
        if force_content:
            logging.info("🔄 Content generation forced via FORCE_CONTENT=true")
            return True
        
        meets_criteria = sum(criteria) >= 3  # At least 3 of 4 criteria
        logging.info(f"📊 Content worthiness check: {sum(criteria)}/4 criteria met")
        
        return meets_criteria
    
    def generate_content_template(self, analysis_data: Dict) -> ContentTemplate:
        """Create blog post content from analysis data"""
        date_str = datetime.now().strftime('%B %Y')
        
        # Dynamic title based on data
        total_stocks = analysis_data.get('total_stocks', 0)
        title = f"Kvartalsrapporter påverkar {total_stocks} svenska aktier - {date_str}"
        
        # Introduction
        intro = f"""
        Nya kvartalsrapporter har lett till betydande förändringar i våra aktierankningar. 
        Under {date_str} har vi identifierat {total_stocks} svenska aktier med uppdaterade kvartalsdata 
        som påverkat deras placering i vårt rankingsystem. Här är en analys av vilka aktier som 
        presterat bäst och sämst efter sina senaste rapporter.
        """
        
        # Analysis section
        analysis_section = self._build_analysis_section(analysis_data)
        
        # Stock highlights (top 3 performers)
        highlights = self._extract_stock_highlights(analysis_data)
        
        # Conclusion
        conclusion = f"""
        Kvartalsrapporterna visar tydligt hur snabbt aktierankningar kan förändras baserat på 
        företagens prestationer. De {len(analysis_data.get('top_performers', []))} toppresterande 
        aktierna visar positiva trender inom ROE och intäktstillväxt, medan underpresterarna 
        behöver ses över mer noggrant. Använd vårt aktiefilter för att få fullständig analys 
        och hitta de bästa investeringsmöjligheterna bland svenska aktier.
        """
        
        # SEO tags (Swedish focus)
        tags = [
            'kvartalsrapporter', 
            'svenska aktier', 
            'aktieanalys', 
            'stockholm börsen',
            'aktierankningar',
            'ROE analys',
            'kvartalsresultat'
        ]
        
        return ContentTemplate(
            title=title,
            introduction=intro.strip(),
            analysis_section=analysis_section,
            stock_highlights=highlights,
            conclusion=conclusion.strip(),
            tags=tags
        )
    
    def _build_analysis_section(self, analysis_data: Dict) -> str:
        """Build main analysis content section"""
        sections = []
        
        # Top performers section
        if analysis_data.get('top_performers'):
            sections.append("## 🎯 Kvartalsvinnarerna\n")
            sections.append("Dessa aktier har visat bäst utveckling efter sina kvartalsrapporter:\n")
            
            for i, performer in enumerate(analysis_data['top_performers'][:5], 1):
                # Clean up the performer text
                clean_performer = performer.replace('#', '').strip()
                if clean_performer:
                    sections.append(f"**{i}.** {clean_performer}")
            sections.append("")
        
        # Underperformers section
        if analysis_data.get('underperformers'):
            sections.append("## ⚠️ Aktier att bevaka\n")
            sections.append("Följande aktier har visat svagare utveckling i sina senaste kvartalsrapporter:\n")
            
            for underperformer in analysis_data['underperformers'][:3]:
                if underperformer:
                    sections.append(f"- {underperformer}")
            sections.append("")
        
        # Sector trends
        if analysis_data.get('sector_trends'):
            sections.append("## 🏭 Sektortrender\n")
            sections.append("Branschvis utveckling baserat på kvartalsrapporter:\n")
            
            for trend in analysis_data['sector_trends'][:5]:
                if trend:
                    sections.append(f"- {trend}")
            sections.append("")
        
        return "\n".join(sections)
    
    def _extract_stock_highlights(self, analysis_data: Dict) -> List[str]:
        """Extract individual stock highlights for summary bullets"""
        highlights = []
        
        for performer in analysis_data.get('top_performers', [])[:3]:
            if performer and '.' in performer and '-' in performer:
                try:
                    # Extract ticker and company name
                    parts = performer.split('.', 1)[1].strip()  # Remove number
                    if ' - ' in parts:
                        ticker_name = parts.split(' - ')[0].strip()
                        highlights.append(f"**{ticker_name}** visar stark kvartalsprestation")
                except Exception:
                    continue
        
        # Add sector highlight if available
        if analysis_data.get('sector_trends'):
            best_sector = analysis_data['sector_trends'][0] if analysis_data['sector_trends'] else None
            if best_sector and ':' in best_sector:
                sector_name = best_sector.split(':')[0].strip()
                highlights.append(f"**{sector_name}** är starkaste sektorn denna kvartal")
        
        return highlights
    
    def publish_to_wordpress(self, content: ContentTemplate) -> bool:
        """Publish blog post to WordPress via REST API"""
        if not all([self.wp_user, self.wp_password]):
            logging.error("❌ WordPress credentials not configured")
            logging.info("Set WORDPRESS_USER and WORDPRESS_APP_PASSWORD environment variables")
            return False
        
        # WordPress REST API endpoint
        api_url = f"{self.wp_url}/wp-json/wp/v2/posts"
        
        # Prepare post data
        post_data = {
            'title': content.title,
            'content': self._format_wordpress_content(content),
            'status': 'publish',  # Change to 'draft' for manual review
            'categories': [75],  # Kvartalsrapporter category
            'tags': content.tags,
            'excerpt': content.introduction[:150] + "...",
            'meta': {
                'description': f"Analys av svenska aktier efter kvartalsrapporter - {datetime.now().strftime('%B %Y')}"
            }
        }
        
        # Authentication
        auth = (self.wp_user, self.wp_password)
        
        try:
            response = requests.post(
                api_url, 
                json=post_data, 
                auth=auth,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 201:
                post_data = response.json()
                post_id = post_data['id']
                post_url = post_data.get('link', 'N/A')
                logging.info("✅ Blog post published successfully")
                logging.info(f"📝 Post ID: {post_id}")
                logging.info(f"🔗 URL: {post_url}")
                return True
            else:
                logging.error(f"❌ WordPress API error: {response.status_code}")
                logging.error(f"Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Failed to connect to WordPress: {e}")
            return False
        except Exception as e:
            logging.error(f"❌ Failed to publish to WordPress: {e}")
            return False
    
    def _format_wordpress_content(self, content: ContentTemplate) -> str:
        """Format content for WordPress HTML"""
        html_content = f"""
        <p>{content.introduction}</p>
        
        {self._convert_markdown_to_html(content.analysis_section)}
        
        <h3>📋 Sammanfattning</h3>
        <ul>
        """
        
        for highlight in content.stock_highlights:
            html_content += f"<li>{highlight}</li>\n"
        
        html_content += f"""
        </ul>
        
        <p>{content.conclusion}</p>
        
        <div style="background-color: #f8f9fa; padding: 20px; margin: 20px 0; border-left: 4px solid #007cba;">
        <h4>🔍 Djupare analys</h4>
        <p>Vill du få tillgång till fullständiga aktierankningar och detaljerad analys? 
        <a href="{self.wp_url}/aktiefilter" style="color: #007cba; font-weight: bold;">Testa vårt aktiefilter här →</a></p>
        </div>
        
        <p><small><em>Analys baserad på data från {datetime.now().strftime('%Y-%m-%d')}. 
        Rankningar uppdateras dagligen baserat på senaste kvartalsrapporter och finansiella nyckeltal.</em></small></p>
        """
        
        return html_content
    
    def _convert_markdown_to_html(self, markdown_text: str) -> str:
        """Simple markdown to HTML conversion for WordPress"""
        html = markdown_text
        
        # Convert headers
        html = html.replace('## ', '<h3>').replace('\n\n', '</h3>\n\n')
        
        # Convert bold
        html = html.replace('**', '<strong>').replace('**', '</strong>')
        
        # Convert bullet points
        lines = html.split('\n')
        in_list = False
        html_lines = []
        
        for line in lines:
            if line.strip().startswith('- '):
                if not in_list:
                    html_lines.append('<ul>')
                    in_list = True
                html_lines.append(f"<li>{line.strip()[2:]}</li>")
            else:
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(line)
        
        if in_list:
            html_lines.append('</ul>')
        
        return '\n'.join(html_lines)
    
    def generate_and_publish(self) -> bool:
        """Main workflow: analyze, generate, and publish content"""
        try:
            logging.info("🚀 Starting automated content generation...")
            
            # Load analysis data
            analysis_data = self.load_quarterly_analysis()
            
            if not analysis_data:
                logging.info("❌ No analysis data available - skipping content generation")
                return False
            
            # Check if content is worth creating
            if not self.evaluate_content_worthiness(analysis_data):
                logging.info("📊 Analysis doesn't meet content thresholds - skipping")
                return False
            
            logging.info("✅ Content generation criteria met - proceeding...")
            
            # Generate content
            content_template = self.generate_content_template(analysis_data)
            logging.info(f"📝 Generated content: '{content_template.title}'")
            
            # Publish to WordPress
            success = self.publish_to_wordpress(content_template)
            
            if success:
                # Log successful publication
                self._log_publication(content_template)
                logging.info("🎉 Content generation workflow completed successfully")
            else:
                logging.error("❌ Failed to publish content to WordPress")
            
            return success
            
        except Exception as e:
            logging.error(f"💥 Content generation workflow failed: {e}")
            return False
    
    def _log_publication(self, content: ContentTemplate):
        """Log successful content publication"""
        log_file = self.data_path / "content_generation_log.txt"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp}: Published '{content.title}'\n")
            f.write(f"  - Tags: {', '.join(content.tags)}\n")
            f.write(f"  - Highlights: {len(content.stock_highlights)} items\n")
            f.write("  - Status: SUCCESS\n\n")
        
        logging.info(f"📋 Publication logged to: {log_file}")


def main():
    """CLI entry point for content generation"""
    print("🤖 Automated Blog Post Generator")
    print("=" * 40)
    
    generator = ContentGenerator()
    success = generator.generate_and_publish()
    
    if success:
        print("✅ Content generated and published successfully")
        return 0
    else:
        print("❌ Content generation skipped or failed")
        return 1


if __name__ == "__main__":
    exit(main())