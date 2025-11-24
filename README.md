Coming Soon!

## Visualizations

The `visualizations.py` module provides comprehensive data visualization capabilities for analyzing ABA fraud articles. It generates various plots and charts to help understand fraud trends, article patterns, and keyword distributions.

### Features

- **Timeline Analysis**: View article publication trends over time
- **Word Clouds**: Visualize the most prominent terms across all articles
- **Content Statistics**: Analyze article length distributions
- **Keyword Analysis**: Identify top fraud-related topics and themes
- **Summary Statistics**: Get detailed statistics about the article collection

### Usage

To generate all visualizations:

```bash
python visualizations.py
```

This will create visualizations in the `visualizations_output/` directory:
- `articles_timeline.png` - Timeline of article publications
- `wordcloud.png` - Word cloud of article content
- `article_lengths.png` - Distribution of article lengths
- `top_keywords.png` - Most frequently mentioned fraud topics

### Programmatic Usage

You can also import and use individual visualization functions:

```python
from visualizations import load_articles, create_wordcloud, plot_top_keywords

# Load articles
articles = load_articles()

# Generate specific visualizations
create_wordcloud(articles, save=True)
plot_top_keywords(articles, top_n=10, save=True)
```

### Requirements

The visualizations module requires:
- matplotlib
- seaborn
- wordcloud
- pandas

These are automatically included in the project dependencies.
