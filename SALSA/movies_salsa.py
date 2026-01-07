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

def create_transition_matrix(graph):
    """
    Convert graph to transition matrix for random walk
    
    Transition matrix: Each row is a probability distribution
    - If movie i connects to movies j, k, l → equal probability to each
    - If movie i has no connections → equal probability to ALL movies
    
    Args:
        graph: adjacency matrix (authority or hub graph)
    
    Returns:
        transition_matrix: stochastic matrix for random walk
    """
    n = len(graph)
    transition = np.zeros((n, n))
    
    for i in range(n):
        # Get all connections from movie i
        row_sum = np.sum(graph[i])
        
        if row_sum > 0:
            # Normalize: equal probability to each connected movie
            transition[i] = graph[i] / row_sum
        else:
            # Isolated movie: equal probability to ALL movies
            transition[i] = np.ones(n) / n
    
    return transition

def power_iteration(transition_matrix, max_iterations=100, tolerance=1e-6, verbose=True):
    """
    Perform power iteration to find stationary distribution
    
    This simulates a random walk until probabilities stabilize
    
    Args:
        transition_matrix: stochastic matrix
        max_iterations: maximum iterations
        tolerance: convergence threshold
        verbose: print progress
    
    Returns:
        scores: stationary distribution (converged probabilities)
    """
    n = len(transition_matrix)
    
    # Start with uniform distribution
    scores = np.ones(n) / n
    
    if verbose:
        print("  Starting random walk...")
        print(f"  Initial scores: all = {scores[0]:.4f}")
        print()
    
    for iteration in range(max_iterations):
        # One step of random walk
        # New probability = sum of (probability of being at j) × (prob of j→i)
        new_scores = transition_matrix.T @ scores
        
        # Normalize (should already sum to 1, but numerical stability)
        new_scores = new_scores / np.sum(new_scores)
        
        # Check convergence
        change = np.max(np.abs(new_scores - scores))
        
        if verbose and (iteration < 3 or (iteration + 1) % 20 == 0):
            top_idx = np.argmax(new_scores)
            print(f"  Iteration {iteration + 1:3d}: max change = {change:.6f}, top movie = {movies[top_idx]}")
        
        scores = new_scores
        
        if change < tolerance:
            if verbose:
                print(f"\n  ✅ Converged after {iteration + 1} iterations!")
            break
    
    return scores

def calculate_salsa_scores(authority_graph, hub_graph, max_iterations=100):
    """
    Calculate SALSA authority and hub scores
    
    Args:
        authority_graph: graph of movies with common recommenders
        hub_graph: graph of movies with common recommendations
        max_iterations: max iterations for convergence
    
    Returns:
        authority_scores: authority score for each movie
        hub_scores: hub score for each movie
    """
    print("\n" + "=" * 70)
    print("🔮 CALCULATING SALSA SCORES")
    print("=" * 70)
    
    # Step 1: Create transition matrices
    print("\n📐 Creating transition matrices...")
    auth_transition = create_transition_matrix(authority_graph)
    hub_transition = create_transition_matrix(hub_graph)
    print("  ✅ Transition matrices ready")
    
    # Step 2: Calculate authority scores
    print("\n🏆 Calculating AUTHORITY scores (random walk on authority graph):")
    print("-" * 70)
    authority_scores = power_iteration(auth_transition, max_iterations)
    
    # Step 3: Calculate hub scores
    print("\n📚 Calculating HUB scores (random walk on hub graph):")
    print("-" * 70)
    hub_scores = power_iteration(hub_transition, max_iterations)
    
    print("\n" + "=" * 70)
    print("✅ SALSA CALCULATION COMPLETE!")
    print("=" * 70)
    
    return authority_scores, hub_scores

def display_salsa_results(authority_scores, hub_scores, movies):
    """
    Display SALSA results in a nice format
    """
    print("\n" + "=" * 80)
    print("🎬 SALSA RESULTS - MOVIE RANKINGS")
    print("=" * 80)
    
    # Sort by authority
    auth_ranking = sorted(range(len(movies)), 
                         key=lambda i: authority_scores[i], 
                         reverse=True)
    
    # Sort by hub
    hub_ranking = sorted(range(len(movies)), 
                        key=lambda i: hub_scores[i], 
                        reverse=True)
    
    # Display side by side
    print(f"\n{'AUTHORITIES (Best Movies)':<40} {'HUBS (Gateway Movies)'}")
    print(f"{'='*40} {'='*40}")
    
    for rank in range(len(movies)):
        auth_idx = auth_ranking[rank]
        hub_idx = hub_ranking[rank]
        
        auth_name = movies[auth_idx]
        hub_name = movies[hub_idx]
        
        auth_score = authority_scores[auth_idx]
        hub_score = hub_scores[hub_idx]
        
        auth_bar = "█" * int(auth_score * 100)
        hub_bar = "█" * int(hub_score * 100)
        
        print(f"{rank+1}. {auth_name:<25} {auth_score:>6.4f} {auth_bar[:10]:<10} | "
              f"{rank+1}. {hub_name:<25} {hub_score:>6.4f} {hub_bar[:10]}")
    
    print("\n" + "=" * 80)
    print("💡 INTERPRETATION")
    print("=" * 80)
    print(f"🏆 Best Authority: {movies[auth_ranking[0]]}")
    print(f"   → Most recommended by other movies (highest quality)")
    print()
    print(f"📚 Best Hub: {movies[hub_ranking[0]]}")
    print(f"   → Best gateway to other good movies (leads to quality)")
    print("=" * 80)

def main():
    """Main function - now with SALSA scoring!"""
    
    # Phase 1: Show network
    print_network_info(movies, links)
    
    # Phase 2: Build graphs
    authority_graph = build_authority_graph(links)
    hub_graph = build_hub_graph(links)
    visualize_graphs(authority_graph, hub_graph, movies)
    
    # Phase 3: Calculate SALSA scores
    authority_scores, hub_scores = calculate_salsa_scores(
        authority_graph, 
        hub_graph, 
        max_iterations=100
    )
    
    # Display results
    display_salsa_results(authority_scores, hub_scores, movies)
    
    # Store results for potential Phase 4
    return authority_scores, hub_scores

if __name__ == "__main__":
    auth_scores, hub_scores = main()