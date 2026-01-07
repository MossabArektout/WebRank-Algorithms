import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_network_with_scores(links, authority_scores, hub_scores, movies):
    """
    Create visual representation of the network with SALSA scores
    
    Creates 3 subplots:
    1. Original recommendation network
    2. Authority scores (node size = authority score)
    3. Hub scores (node size = hub score)
    """
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('SALSA Algorithm - Movie Recommendation Network', 
                 fontsize=16, fontweight='bold')
    
    n = len(movies)
    
    # Create a circular layout for nodes
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {i: (np.cos(angle), np.sin(angle)) for i, angle in enumerate(angles)}
    
    # ========================================================================
    # SUBPLOT 1: Original Network
    # ========================================================================
    ax1 = axes[0]
    ax1.set_title('Original Recommendation Network', fontsize=12, fontweight='bold')
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.axis('off')
    
    # Draw edges (recommendations)
    for i in range(n):
        for j in range(n):
            if links[i][j] == 1:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                ax1.arrow(x1, y1, (x2-x1)*0.85, (y2-y1)*0.85,
                         head_width=0.05, head_length=0.05,
                         fc='gray', ec='gray', alpha=0.3, linewidth=0.5)
    
    # Draw nodes (equal size)
    for i in range(n):
        x, y = pos[i]
        circle = plt.Circle((x, y), 0.15, color='lightblue', ec='black', linewidth=2)
        ax1.add_patch(circle)
        
        # Add movie name
        ax1.text(x, y-0.35, movies[i], ha='center', va='top', 
                fontsize=8, fontweight='bold')
    
    # ========================================================================
    # SUBPLOT 2: Authority Scores
    # ========================================================================
    ax2 = axes[1]
    ax2.set_title('Authority Scores (Best Movies)', fontsize=12, fontweight='bold')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.axis('off')
    
    # Normalize scores for visualization
    max_auth = max(authority_scores)
    
    # Draw edges
    for i in range(n):
        for j in range(n):
            if links[i][j] == 1:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                ax2.arrow(x1, y1, (x2-x1)*0.85, (y2-y1)*0.85,
                         head_width=0.05, head_length=0.05,
                         fc='gray', ec='gray', alpha=0.2, linewidth=0.5)
    
    # Draw nodes (size = authority score)
    for i in range(n):
        x, y = pos[i]
        size = 0.1 + (authority_scores[i] / max_auth) * 0.3  # Scale size
        
        # Color intensity based on score
        intensity = authority_scores[i] / max_auth
        color = plt.cm.Reds(0.3 + intensity * 0.7)
        
        circle = plt.Circle((x, y), size, color=color, ec='darkred', linewidth=2)
        ax2.add_patch(circle)
        
        # Add movie name and score
        ax2.text(x, y-size-0.15, f"{movies[i]}\n{authority_scores[i]:.3f}", 
                ha='center', va='top', fontsize=7, fontweight='bold')
    
    # ========================================================================
    # SUBPLOT 3: Hub Scores
    # ========================================================================
    ax3 = axes[2]
    ax3.set_title('Hub Scores (Gateway Movies)', fontsize=12, fontweight='bold')
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.axis('off')
    
    # Normalize scores
    max_hub = max(hub_scores)
    
    # Draw edges
    for i in range(n):
        for j in range(n):
            if links[i][j] == 1:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                ax3.arrow(x1, y1, (x2-x1)*0.85, (y2-y1)*0.85,
                         head_width=0.05, head_length=0.05,
                         fc='gray', ec='gray', alpha=0.2, linewidth=0.5)
    
    # Draw nodes (size = hub score)
    for i in range(n):
        x, y = pos[i]
        size = 0.1 + (hub_scores[i] / max_hub) * 0.3
        
        # Color intensity based on score
        intensity = hub_scores[i] / max_hub
        color = plt.cm.Blues(0.3 + intensity * 0.7)
        
        circle = plt.Circle((x, y), size, color=color, ec='darkblue', linewidth=2)
        ax3.add_patch(circle)
        
        # Add movie name and score
        ax3.text(x, y-size-0.15, f"{movies[i]}\n{hub_scores[i]:.3f}", 
                ha='center', va='top', fontsize=7, fontweight='bold')
    
    # Add legend
    fig.text(0.5, 0.02, 
             'Node size = score magnitude | Darker color = higher score | Arrows = recommendations',
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save
    filename = 'salsa_movie_network.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n💾 Visualization saved: {filename}")
    
    plt.show()

def plot_score_comparison(authority_scores, hub_scores, movies):
    """
    Create bar chart comparing authority vs hub scores
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('SALSA Scores Comparison', fontsize=16, fontweight='bold')
    
    # Sort by scores
    auth_sorted_idx = np.argsort(authority_scores)[::-1]
    hub_sorted_idx = np.argsort(hub_scores)[::-1]
    
    # ========================================================================
    # Authority Scores
    # ========================================================================
    ax1.barh(range(len(movies)), 
             [authority_scores[i] for i in auth_sorted_idx],
             color='crimson', alpha=0.7, edgecolor='darkred', linewidth=1.5)
    ax1.set_yticks(range(len(movies)))
    ax1.set_yticklabels([movies[i] for i in auth_sorted_idx])
    ax1.set_xlabel('Authority Score', fontsize=11, fontweight='bold')
    ax1.set_title('🏆 Authority Scores\n(Best Quality Movies)', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Add score labels
    for i, idx in enumerate(auth_sorted_idx):
        score = authority_scores[idx]
        ax1.text(score + 0.005, i, f'{score:.3f}', 
                va='center', fontsize=9, fontweight='bold')
    
    # ========================================================================
    # Hub Scores
    # ========================================================================
    ax2.barh(range(len(movies)), 
             [hub_scores[i] for i in hub_sorted_idx],
             color='royalblue', alpha=0.7, edgecolor='darkblue', linewidth=1.5)
    ax2.set_yticks(range(len(movies)))
    ax2.set_yticklabels([movies[i] for i in hub_sorted_idx])
    ax2.set_xlabel('Hub Score', fontsize=11, fontweight='bold')
    ax2.set_title('📚 Hub Scores\n(Best Gateway Movies)', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    # Add score labels
    for i, idx in enumerate(hub_sorted_idx):
        score = hub_scores[idx]
        ax2.text(score + 0.005, i, f'{score:.3f}', 
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    filename = 'salsa_score_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"💾 Comparison chart saved: {filename}")
    
    plt.show()

def plot_authority_vs_hub(authority_scores, hub_scores, movies):
    """
    Scatter plot: Authority vs Hub scores
    Shows which movies are authorities, hubs, or both
    """
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(hub_scores, authority_scores, 
               s=300, c='purple', alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add movie labels
    for i, movie in enumerate(movies):
        plt.annotate(movie, (hub_scores[i], authority_scores[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # Add quadrant lines
    mean_auth = np.mean(authority_scores)
    mean_hub = np.mean(hub_scores)
    
    plt.axhline(y=mean_auth, color='red', linestyle='--', alpha=0.5, linewidth=1)
    plt.axvline(x=mean_hub, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    
    # Label quadrants
    max_auth = max(authority_scores)
    max_hub = max(hub_scores)
    
    plt.text(max_hub * 0.8, max_auth * 0.95, 
            'High Authority\nHigh Hub\n(Best Overall)', 
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.text(max_hub * 0.2, max_auth * 0.95, 
            'High Authority\nLow Hub\n(Quality Content)', 
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.text(max_hub * 0.8, max_auth * 0.2, 
            'Low Authority\nHigh Hub\n(Gateway Movies)', 
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.text(max_hub * 0.2, max_auth * 0.2, 
            'Low Authority\nLow Hub\n(Niche Movies)', 
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    plt.xlabel('Hub Score (Gateway Quality)', fontsize=12, fontweight='bold')
    plt.ylabel('Authority Score (Content Quality)', fontsize=12, fontweight='bold')
    plt.title('SALSA: Authority vs Hub Scores\nMovie Positioning', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Save
    filename = 'salsa_authority_vs_hub.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"💾 Scatter plot saved: {filename}")
    
    plt.show()

def main():
    """Main function with visualizations"""
    
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
    
    # Phase 4: CREATE VISUALIZATIONS!
    print("\n" + "=" * 70)
    print("🎨 CREATING VISUALIZATIONS...")
    print("=" * 70)
    
    # 1. Network with scores
    visualize_network_with_scores(links, authority_scores, hub_scores, movies)
    
    # 2. Bar chart comparison
    plot_score_comparison(authority_scores, hub_scores, movies)
    
    # 3. Authority vs Hub scatter
    plot_authority_vs_hub(authority_scores, hub_scores, movies)
    
    print("\n" + "=" * 70)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  📊 salsa_movie_network.png - Network with score visualization")
    print("  📊 salsa_score_comparison.png - Bar charts of scores")
    print("  📊 salsa_authority_vs_hub.png - Scatter plot comparison")
    print("=" * 70)
    
    return authority_scores, hub_scores

if __name__ == "__main__":
    auth_scores, hub_scores = main()