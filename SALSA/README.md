# SALSA Algorithm - Movie Recommendation Network

SALSA (Stochastic Approach for Link-Structure Analysis) is a link analysis algorithm that identifies both **authorities** (high-quality content) and **hubs** (good gateways to content) in a network.

## What is SALSA?

SALSA improves on HITS by using random walks instead of mutual reinforcement. It creates two separate graphs:

- **Authority Graph**: Connects nodes that share common recommenders (movies recommended by the same sources)
- **Hub Graph**: Connects nodes that recommend similar content (movies that recommend the same targets)

Random walks on these graphs produce stable authority and hub scores.

## Key Concepts

- **Authority Score**: Measures content quality (highly recommended movies)
- **Hub Score**: Measures gateway quality (movies that lead to other good movies)

## Example Network

The implementation uses 8 classic movies:
- The Godfather, Pulp Fiction, The Shawshank Redemption, Fight Club
- Inception, The Matrix, Goodfellas, Forrest Gump

Each movie recommends similar movies, creating a recommendation network.

## How It Works

1. **Build Authority Graph**: Connect movies with shared recommenders
2. **Build Hub Graph**: Connect movies that recommend the same targets
3. **Random Walks**: Perform power iteration to find stationary distributions
4. **Score Assignment**: Converged probabilities become authority/hub scores

## Running the Code

```bash
python movies_salsa.py
```

### Output

The program generates:
- Console output showing network structure, graph construction, and score calculations
- Three visualizations saved as PNG files:
  - `salsa_movie_network.png` - Network with score visualization
  - `salsa_score_comparison.png` - Bar charts comparing scores
  - `salsa_authority_vs_hub.png` - Scatter plot showing movie positioning

## Visualizations

### 1. Network Visualization
Shows the original network alongside authority and hub score distributions (node size = score magnitude).

### 2. Score Comparison
Side-by-side bar charts ranking movies by authority and hub scores.

### 3. Authority vs Hub Scatter Plot
Positions movies in a 2D space showing which are:
- High authority + high hub (best overall)
- High authority + low hub (quality content)
- Low authority + high hub (gateway movies)
- Low authority + low hub (niche content)

## Dependencies

```bash
pip install numpy matplotlib
```

## Key Differences from HITS

- Uses random walks instead of mutual reinforcement
- More resistant to manipulation (link spam)
- Probabilistic interpretation (stationary distributions)
- Separates authority and hub computations into different graphs

## Example Results

In the movie network:
- **Best Authority**: The Shawshank Redemption (most recommended by others)
- **Best Hub**: Pulp Fiction (leads to many good movies)
