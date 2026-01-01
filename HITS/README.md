# HITS Algorithm

Implementation of the HITS (Hyperlink-Induced Topic Search) algorithm for analyzing web page authority and hub scores.

## Files

- **[learn_hits.py](learn_hits.py)** - Educational implementation with a simple example network
- **[github_hits.py](github_hits.py)** - Real-world application analyzing GitHub repositories

## What is HITS?

HITS identifies two types of pages in a network:
- **Authorities**: Pages with high-quality content (receive many links)
- **Hubs**: Pages that link to many authorities (curated lists)

## Usage

### Learning Example
```bash
python learn_hits.py
```

### GitHub Analysis
```bash
# Set up environment
cp .env.example .env
# Add your GITHUB_TOKEN to .env

# Install dependencies
pip install -r requirements.txt

# Run analysis
python github_hits.py
```

## Key Insight

HITS separates content creators (authorities) from content curators (hubs) - something PageRank alone cannot do.
