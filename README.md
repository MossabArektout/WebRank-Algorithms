# WebRank Algorithms

Implementations of web ranking algorithms applied to GitHub repository analysis.

## Algorithms

### [PageRank](PageRank/)
Finds "hidden gems" - underrated repositories with high influence in the citation network.

**Key idea**: A repository is important if it's referenced by other important repositories.

### [HITS](HITS/)
Separates repositories into authorities (high-quality tools) and hubs (curated lists).

**Key idea**: Authorities are linked to by good hubs. Hubs link to good authorities.

### [SALSA](SALSA/)
Stochastic approach to identify authorities and hubs using random walks on bipartite graphs.

**Key idea**: Uses random walks instead of mutual reinforcement to find authorities (high-quality content) and hubs (gateway content). More resistant to manipulation than HITS.

## Quick Start

Each folder contains:
- Educational implementation with simple examples
- Real-world GitHub repository analyzer
- README with usage instructions

## Requirements

- Python 3.8+
- GitHub API token (for real-world analyzers)

## Purpose

This project was created for educational purposes to learn and understand how link analysis algorithms work by implementing them from scratch and applying them to real GitHub data.
