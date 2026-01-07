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

# =============================================================
# SALSA's key idea:

# Authority Graph: Connect movies that share common recommenders
# Hub Graph: Connect movies that recommend the same movies
# ==============================================================

def build_authority_graph(links):
    n = len(links)
    authority_graph = np.zeros((n,n))
    
    print("🏆 BUILDING AUTHORITY GRAPH")
    print("=" * 70)
    print("Connecting movies that share common recommenders...\n")

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            common_recommenders = 0
            recommender_names =[]

            for k in range(n):
                if links[k][i] == 1 and links[k][j] == 1:
                    common_recommenders += 1
                    recommender_names.append(movies[k])
            if common_recommenders > 0:
                authority_graph[i][j] = 1
                
                # Print for first few to show what's happening
                if i < 3 and j < 3 and i < j:
                    print(f"  {movies[i]} ↔ {movies[j]}")
                    print(f"    Shared recommenders: {', '.join(recommender_names)}")
                    print()
    print("=" * 70)
    print(f"✅ Authority graph complete!")
    print(f"   Total connections: {int(np.sum(authority_graph) / 2)}")
    print()
    
    return authority_graph

def build_hub_graph(links):
    n = len(links)
    hub_graph = np.zeros((n, n))
    
    print("📚 BUILDING HUB GRAPH")
    print("=" * 70)
    print("Connecting movies that recommend similar movies...\n")
    
    # For each pair of movies
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # Skip self-connections
            
            # Count how many movies are recommended by BOTH i and j
            common_recommendations = 0
            shared_movies = []
            
            for k in range(n):
                # Do both i and j recommend movie k?
                if links[i][k] == 1 and links[j][k] == 1:
                    common_recommendations += 1
                    shared_movies.append(movies[k])
            
            # If they recommend similar movies, connect them
            if common_recommendations > 0:
                hub_graph[i][j] = 1
                
                # Print for first few to show what's happening
                if i < 3 and j < 3 and i < j:
                    print(f"  {movies[i]} ↔ {movies[j]}")
                    print(f"    Both recommend: {', '.join(shared_movies)}")
                    print()
    
    print("=" * 70)
    print(f"✅ Hub graph complete!")
    print(f"   Total connections: {int(np.sum(hub_graph) / 2)}")
    print()
    
    return hub_graph

def visualize_graphs(authority_graph, hub_graph, movies):
    n = len(movies)
    
    print("\n" + "=" * 70)
    print("📊 AUTHORITY GRAPH (Movies with common recommenders)")
    print("=" * 70)
    
    for i in range(n):
        connections = []
        for j in range(n):
            if authority_graph[i][j] == 1:
                connections.append(movies[j])
        
        if connections:
            print(f"{movies[i]:<20} ↔ {', '.join(connections)}")
    
    print("\n" + "=" * 70)
    print("📊 HUB GRAPH (Movies that recommend similar movies)")
    print("=" * 70)
    
    for i in range(n):
        connections = []
        for j in range(n):
            if hub_graph[i][j] == 1:
                connections.append(movies[j])
        
        if connections:
            print(f"{movies[i]:<20} ↔ {', '.join(connections)}")
    
    print()

def main():
    """Main function to run SALSA Phase 2"""
    
    # Show original network
    print_network_info(movies, links)
    
    # Build authority graph
    authority_graph = build_authority_graph(links)
    
    # Build hub graph  
    hub_graph = build_hub_graph(links)
    
    # Visualize both graphs
    visualize_graphs(authority_graph, hub_graph, movies)
    
    # Show graph statistics
    print("=" * 70)
    print("📈 GRAPH STATISTICS")
    print("=" * 70)
    
    for i, movie in enumerate(movies):
        auth_connections = int(np.sum(authority_graph[i]))
        hub_connections = int(np.sum(hub_graph[i]))
        print(f"{movie:<20} Auth connections: {auth_connections:2d} | Hub connections: {hub_connections:2d}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()