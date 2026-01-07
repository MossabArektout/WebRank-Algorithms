import numpy as np


# Movies network
movies = [
    'The Godfather',      # 0
    'Pulp Fiction',       # 1
    'The Shawshank',      # 2
    'Fight Club',         # 3
    'Inception',          # 4
    'The Matrix',         # 5
    'Goodfellas',         # 6
    'Forrest Gump',       # 7
]

# links[i][j] = 1 means "if you like movie i, watch movie j"
links = [
    # God  Pul  Shaw Figh Inc  Mat  Good Forr
    [ 0,   0,   1,   0,   0,   0,   1,   1  ],  # Godfather → Shawshank, Goodfellas, Forrest
    [ 0,   0,   1,   1,   1,   1,   0,   0  ],  # Pulp → Shawshank, Fight, Inception, Matrix
    [ 1,   1,   0,   0,   0,   0,   0,   1  ],  # Shawshank → Godfather, Pulp, Forrest
    [ 0,   1,   0,   0,   1,   1,   0,   0  ],  # Fight → Pulp, Inception, Matrix
    [ 0,   1,   1,   1,   0,   1,   0,   0  ],  # Inception → Pulp, Shawshank, Fight, Matrix
    [ 0,   1,   1,   1,   1,   0,   0,   0  ],  # Matrix → Pulp, Shawshank, Fight, Inception
    [ 1,   0,   0,   0,   0,   0,   0,   1  ],  # Goodfellas → Godfather, Forrest
    [ 1,   0,   1,   0,   0,   0,   1,   0  ],  # Forrest → Godfather, Shawshank, Goodfellas
]

# Pulp Fiction is a "hub" - leads you to many good movies
# Shawshank is an "authority" - many movies recommend it!

def print_network_info(movies, links):
    print("🎬 MOVIE RECOMMENDATION NETWORK")
    print("=" * 70)
    
    # Count recommendations
    for i, movie in enumerate(movies):
        out_links = sum(links[i]) 
        in_links = sum(links[j][i] for j in range(len(movies)))  # How many recommend this?

        print(f"{movie:<20} → Recommends {out_links} movies | Recommended by {in_links} movies")
    
    print("=" * 70)
    print()

print_network_info(movies, links)

        