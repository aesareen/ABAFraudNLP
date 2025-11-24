"""
Visualizations module for ABA Fraud NLP project.

This module provides various visualization functions to analyze and display
fraud article data, including temporal trends, keyword analysis, and content statistics.
"""

import json
import glob
import os
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any
import re

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Project root for data access
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/scraped_json_results")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "visualizations_output")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_articles() -> List[Dict[str, Any]]:
    """
    Load all articles from the scraped JSON results directory.
    
    Returns:
        List of article dictionaries
    """
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    articles = []
    
    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    articles.append(data[0])
                else:
                    print(f"Warning: Skipping {file} - unexpected data format")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Error loading {file}: {e}")
    
    return articles


def parse_date(date_str: str) -> datetime:
    """
    Parse date string in format 'Month Day, Year'.
    
    Args:
        date_str: Date string to parse
        
    Returns:
        datetime object or None if parsing fails
    """
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%B %d, %Y")
    except (ValueError, AttributeError, TypeError):
        return None


def plot_articles_over_time(articles: List[Dict[str, Any]], save: bool = True):
    """
    Create a timeline plot showing article publication dates.
    
    Args:
        articles: List of article dictionaries
        save: Whether to save the plot to file
    """
    # Extract and parse dates
    dates = []
    for article in articles:
        date_str = article.get('date_published')
        if date_str:
            date = parse_date(date_str)
            if date:
                dates.append(date)
    
    if not dates:
        print("No valid dates found in articles")
        return
    
    # Sort dates and count articles per month
    dates_sorted = sorted(dates)
    df = pd.DataFrame({'date': dates_sorted})
    df['year_month'] = df['date'].dt.to_period('M')
    monthly_counts = df.groupby('year_month').size()
    
    # Create plot
    plt.figure(figsize=(14, 6))
    monthly_counts.plot(kind='bar', color='steelblue', alpha=0.7)
    plt.title('Articles Published Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save:
        output_path = os.path.join(OUTPUT_DIR, 'articles_timeline.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved timeline plot to {output_path}")
    
    plt.show()


def create_wordcloud(articles: List[Dict[str, Any]], save: bool = True):
    """
    Generate a word cloud from all article content.
    
    Args:
        articles: List of article dictionaries
        save: Whether to save the plot to file
    """
    # Combine all article content
    all_text = " ".join([article.get('raw_content', '') for article in articles])
    
    # Create word cloud
    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color='white',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(all_text)
    
    # Display
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Fraud Articles', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save:
        output_path = os.path.join(OUTPUT_DIR, 'wordcloud.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved word cloud to {output_path}")
    
    plt.show()


def plot_article_lengths(articles: List[Dict[str, Any]], save: bool = True):
    """
    Create a histogram of article content lengths.
    
    Args:
        articles: List of article dictionaries
        save: Whether to save the plot to file
    """
    lengths = [len(article.get('raw_content', '')) for article in articles]
    
    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=15, color='coral', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Article Lengths', fontsize=16, fontweight='bold')
    plt.xlabel('Content Length (characters)', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.axvline(sum(lengths) / len(lengths), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {sum(lengths) / len(lengths):.0f}')
    plt.legend()
    plt.tight_layout()
    
    if save:
        output_path = os.path.join(OUTPUT_DIR, 'article_lengths.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved article lengths plot to {output_path}")
    
    plt.show()


def extract_keywords_from_articles(articles: List[Dict[str, Any]]) -> Counter:
    """
    Extract keywords/topics from article content using simple pattern matching.
    Counts the number of articles containing each keyword (not total occurrences).
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Counter object with keyword frequencies (articles containing keyword)
    """
    # Common fraud-related keywords to look for
    keywords = [
        'fraud', 'scam', 'phishing', 'cybersecurity', 'ransomware',
        'check fraud', 'debit card', 'credit card', 'identity theft',
        'elder fraud', 'business email compromise', 'deepfake', 'atm',
        'investment', 'romance scam', 'wire transfer', 'cryptocurrency',
        'authentication', 'verification', 'prevention', 'detection'
    ]
    
    keyword_counts = Counter()
    
    for article in articles:
        content = article.get('raw_content', '').lower()
        for keyword in keywords:
            # Check if keyword appears in the article (at least once)
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', content):
                keyword_counts[keyword] += 1
    
    return keyword_counts


def plot_top_keywords(articles: List[Dict[str, Any]], top_n: int = 15, save: bool = True):
    """
    Create a bar chart of top keywords/topics mentioned in articles.
    
    Args:
        articles: List of article dictionaries
        top_n: Number of top keywords to display
        save: Whether to save the plot to file
    """
    keyword_counts = extract_keywords_from_articles(articles)
    
    if not keyword_counts:
        print("No keywords found")
        return
    
    # Get top N keywords
    top_keywords = keyword_counts.most_common(top_n)
    keywords, counts = zip(*top_keywords)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(keywords)), counts, color='teal', alpha=0.7)
    plt.yticks(range(len(keywords)), keywords)
    plt.xlabel('Number of Articles Mentioning', fontsize=12)
    plt.title(f'Top {top_n} Fraud-Related Keywords in Articles', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(count + 0.1, i, str(count), va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save:
        output_path = os.path.join(OUTPUT_DIR, 'top_keywords.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved top keywords plot to {output_path}")
    
    plt.show()


def generate_summary_statistics(articles: List[Dict[str, Any]]):
    """
    Generate and print summary statistics about the article collection.
    
    Args:
        articles: List of article dictionaries
    """
    total_articles = len(articles)
    
    # Date statistics
    dates = [parse_date(article.get('date_published', '')) for article in articles]
    valid_dates = [d for d in dates if d is not None]
    
    # Content statistics
    lengths = [len(article.get('raw_content', '')) for article in articles]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    min_length = min(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0
    
    # Print summary
    print("\n" + "="*60)
    print("ABA FRAUD ARTICLES - SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Articles: {total_articles}")
    print(f"Articles with Dates: {len(valid_dates)}")
    if valid_dates:
        print(f"Date Range: {min(valid_dates).strftime('%B %d, %Y')} to {max(valid_dates).strftime('%B %d, %Y')}")
    print(f"\nContent Statistics:")
    print(f"  Average Length: {avg_length:.0f} characters")
    print(f"  Shortest Article: {min_length} characters")
    print(f"  Longest Article: {max_length} characters")
    print(f"  Total Content: {sum(lengths):,} characters")
    
    # Top keywords
    keyword_counts = extract_keywords_from_articles(articles)
    if keyword_counts:
        print(f"\nTop 5 Keywords:")
        for keyword, count in keyword_counts.most_common(5):
            print(f"  {keyword}: {count} articles")
    
    print("="*60 + "\n")


def create_all_visualizations(articles: List[Dict[str, Any]] = None, save: bool = True):
    """
    Generate all visualizations for the fraud articles.
    
    Args:
        articles: List of article dictionaries (if None, will load from disk)
        save: Whether to save plots to files
    """
    if articles is None:
        articles = load_articles()
    
    print(f"Loaded {len(articles)} articles")
    print(f"Generating visualizations in: {OUTPUT_DIR}\n")
    
    # Generate summary statistics
    generate_summary_statistics(articles)
    
    # Generate all plots
    print("Creating visualizations...")
    plot_articles_over_time(articles, save=save)
    create_wordcloud(articles, save=save)
    plot_article_lengths(articles, save=save)
    plot_top_keywords(articles, save=save)
    
    print("\n✓ All visualizations generated successfully!")


if __name__ == "__main__":
    # Load articles and create all visualizations
    articles = load_articles()
    create_all_visualizations(articles, save=True)
