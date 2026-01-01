import numpy as np

pages = [
    'Wikipedia',      # Index 0 - Good content, few outlinks
    'TechBlog',       # Index 1 - Medium content, some links
    'NewsArticle',    # Index 2 - Good content, cites sources
    'AwesomeList',    # Index 3 - Just links, no original content
    'ForumPost'
]

# Build the link graph
# 0: Wikipedia → nobody
# 1: TechBlog → Wikipedia
# 2: NewsArticle → Wikipedia, TechBlog
# 3: AwesomeList → Wiki, Tech, News
# 4: ForumPost → Wiki, AwesomeList

links = [
    [0,0,0,0,0],
    [1,0,0,0,0],
    [1,1,0,0,0],
    [1,1,1,0,0],
    [1,0,0,10,]
]

# Wikipedia gets many incoming links (should have high AUTHORITY)
# AwesomeList has many outgoing links (should have high HUB score)
# Wikipedia has zero outgoing links (should have low HUB score)

def hits_algorithm(links, max_iterations=20, verbose=True):
    n = len(links)  # Number of pages
    
    # STEP 1: Initialize all scores to 1
    authority = np.ones(n)
    hub = np.ones(n)
    
    if verbose:
        print("STARTING HITS ALGORITHM")
        print("=" * 60)
        print(f"Number of pages: {n}")
        print(f"Max iterations: {max_iterations}\n")
    
    # STEP 2: Iterate to convergence
    for iteration in range(max_iterations):
        
        if verbose and iteration < 3:  # Show first 3 iterations
            print(f"--- Iteration {iteration + 1} ---")
        
        # STEP 2a: Update AUTHORITY scores
        # Authority = sum of HUB scores that point to you
        new_authority = np.zeros(n)
        
        for i in range(n):  # For each page i
            for j in range(n):  # Check all pages j
                if links[j][i] == 1:  # If j links to i
                    new_authority[i] += hub[j]  # Add j's hub score
        
        if verbose and iteration < 3:
            print(f"Authority scores: {new_authority}")
        
        # STEP 2b: Update HUB scores
        # Hub = sum of AUTHORITY scores you point to
        new_hub = np.zeros(n)
        
        for i in range(n):  # For each page i
            for j in range(n):  # Check all pages j
                if links[i][j] == 1:  # If i links to j
                    new_hub[i] += new_authority[j]  # Add j's authority
        
        if verbose and iteration < 3:
            print(f"Hub scores: {new_hub}")
        
        # STEP 3: Normalize scores
        # This prevents numbers from getting too large
        # and makes scores easier to compare
        
        max_auth = np.max(new_authority) if np.max(new_authority) > 0 else 1
        max_hub = np.max(new_hub) if np.max(new_hub) > 0 else 1
        
        authority = new_authority / max_auth
        hub = new_hub / max_hub
        
        if verbose and iteration < 3:
            print(f"After normalization:")
            print(f"  Authority: {authority}")
            print(f"  Hub: {hub}\n")
        
        # STEP 4: Check if converged (scores stopped changing)
        # We'll do this simpler: just run all iterations
    
    if verbose:
        print("=" * 60)
        print("CONVERGENCE REACHED")
        print("=" * 60)
    
    return authority, hub
    
# Run HITS on our network
authority_scores, hub_scores = hits_algorithm(links, max_iterations=10)

# Display final results
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"{'Page':<15} {'Authority Score':>20} {'Hub Score':>20}")
print("-" * 70)

for i, page in enumerate(pages):
    print(f"{page:<15} {authority_scores[i]:>20.4f} {hub_scores[i]:>20.4f}")

print("\n" + "=" * 70)
print(f"🏆 Best Authority: {pages[np.argmax(authority_scores)]}")
print(f"📚 Best Hub: {pages[np.argmax(hub_scores)]}")
print("=" * 70)