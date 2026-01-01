# GitHub Hidden Gems Finder

A small project to learn and apply the PageRank algorithm by discovering underrated GitHub repositories.

## What it does

- Fetches GitHub repositories by topic using the GitHub API
- Extracts cross-repository links from README files
- Builds a directed graph of repository references
- Calculates PageRank scores using NetworkX
- Identifies "hidden gems" - repos with high PageRank but relatively low stars
- Generates a network visualization

## Requirements

```bash
pip install requests python-dotenv networkx matplotlib
```

## Setup

Create a `.env` file with your GitHub token:

```
GITHUB_TOKEN=your_token_here
```

## Usage

```bash
python github-gems.py
```

Results are saved to `hidden_gems_{TOPIC}.txt` and `network_{TOPIC}.png`.

## Learning Goals

Understanding how PageRank works by applying it to real-world data from GitHub's repository ecosystem.

## Reference

Based on the PageRank algorithm described in:
- Dode, A., & Hasani, S. (2017). PageRank Algorithm. *IOSR Journal of Computer Engineering*, 19(01), 01-07. DOI: [10.9790/0661-1901030107](https://doi.org/10.9790/0661-1901030107)
- Available on [ResearchGate](https://www.researchgate.net/publication/314235791_PageRank_Algorithm)
