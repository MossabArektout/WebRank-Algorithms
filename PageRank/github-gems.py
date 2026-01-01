import requests
import time
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import os
import re
import networkx as nx

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
TOPIC = "machine-learning"

def search_repos(topic, max_repos = 30):
    url = "https://api.github.com/search/repositories"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    params = { 
        "q": f"topic {topic} stars:>50",
        "sort": "stars",
        "order": "desc",
        "per_page": max_repos
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.json())
        return []
    
    repos = response.json()["items"]
    
    repo_list = []
    for repo in repos:
        repo_list.append({
            "owner": repo["owner"]["login"],
            "name": repo["name"],
            "full_name": repo["full_name"],
            "stars": repo["stargazers_count"],
            "url": repo["html_url"]
        })
    
    print(f"Found {len(repo_list)} repositories")
    return repo_list

def get_readme(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.raw"  # Get raw markdown
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.text
    return ""

def extract_github_links(readme_text):
    pattern = r'github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)'
    matches = re.findall(pattern, readme_text)

    links = [f"{owner}/{repo}" for owner, repo in matches]

    links = list(set(links))

    return links

def build_link_graph(repos):
    """
    For each repo, find what other repos it links to
    
    Returns: Dictionary like {'user/repo': ['linked/repo1', 'linked/repo2']}
    """
    graph = {}
    repo_names = {repo['full_name'] for repo in repos}  # Set of our repos
    
    for i, repo in enumerate(repos):
        print(f"Processing {i+1}/{len(repos)}: {repo['full_name']}")
        
        # Get README
        readme = get_readme(repo['owner'], repo['name'])
        
        if not readme:
            graph[repo['full_name']] = []
            continue
        
        # Extract links
        all_links = extract_github_links(readme)
        
        # Only keep links to repos in our dataset
        valid_links = [link for link in all_links if link in repo_names]
        
        graph[repo['full_name']] = valid_links
        
        print(f"  Found {len(valid_links)} links to other repos in our set")
        
        # Be nice to GitHub API - wait a bit between requests
        time.sleep(0.5)
    
    return graph


def calculate_pagerank(repos, link_graph):
    G = nx.DiGraph()

    for repo in repos : 
        G.add_node(repo['full_name'], stars=repo['stars'], url=repo['url'])

    for source, targets in link_graph.items():
        for target in targets:
            G.add_edge(source, target)

    print(f"\nGraph stats:")
    print(f"  Nodes (repos): {G.number_of_nodes()}")
    print(f"  Edges (links): {G.number_of_edges()}")
    
    # Calculate PageRank!
    pagerank_scores = nx.pagerank(G, alpha=0.85)

    return G, pagerank_scores

def find_hidden_gems(repos, pagerank_scores, top_n=10):
    """
    Find repos with high PageRank but low stars (hidden gems!)
    """
    gems = []
    
    for repo in repos:
        name = repo['full_name']
        stars = repo['stars']
        pr_score = pagerank_scores.get(name, 0)
        
        # Hidden gem score = PageRank / Stars
        # High PageRank but low stars = high gem score
        if stars > 0:  # Avoid division by zero
            gem_score = pr_score / stars * 1000000  # Multiply for readability
        else:
            gem_score = 0
        
        gems.append({
            'name': name,
            'stars': stars,
            'pagerank': pr_score,
            'gem_score': gem_score,
            'url': repo['url']
        })
    
    # Sort by gem score
    gems.sort(key=lambda x: x['gem_score'], reverse=True)
    
    return gems[:top_n]

def print_results(gems):
    """
    Pretty print the hidden gems
    """
    print("\n" + "="*70)
    print("🌟 HIDDEN GEMS - High PageRank, Low Stars 🌟")
    print("="*70)
    
    for i, gem in enumerate(gems, 1):
        print(f"\n{i}. {gem['name']}")
        print(f"   ⭐ Stars: {gem['stars']}")
        print(f"   📊 PageRank: {gem['pagerank']:.6f}")
        print(f"   💎 Gem Score: {gem['gem_score']:.2f}")
        print(f"   🔗 {gem['url']}")

def visualize_network(G, pagerank_scores, top_gems):
    plt.figure(figsize=(12, 8))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Node sizes based on PageRank
    node_sizes = [pagerank_scores.get(node, 0) * 10000 for node in G.nodes()]
    
    # Color hidden gems differently
    gem_names = {gem['name'] for gem in top_gems}
    node_colors = ['red' if node in gem_names else 'lightblue' 
                   for node in G.nodes()]
    
    # Draw
    nx.draw(G, pos, 
            node_size=node_sizes,
            node_color=node_colors,
            with_labels=False,
            arrows=True,
            edge_color='gray',
            alpha=0.6)
    
    plt.title(f"Repository Network - {TOPIC}\n(Red = Hidden Gems, Size = PageRank)")
    plt.savefig(f'network_{TOPIC}.png', dpi=300, bbox_inches='tight')
    print(f"📊 Network visualization saved to: network_{TOPIC}.png")
    plt.show()

def main():
    print(f"Searching for hidden gems in: {TOPIC}")
    print("="*70)
    
    # Step 1: Get repos
    print("\n[1/4] Fetching repositories...")
    repos = search_repos(TOPIC, max_repos=30)
    
    if not repos:
        print("No repos found. Check your token and topic.")
        return
    
    # Step 2: Build link graph
    print("\n[2/4] Building link graph...")
    link_graph = build_link_graph(repos)
    
    # Step 3: Calculate PageRank
    print("\n[3/4] Calculating PageRank...")
    G, pagerank_scores = calculate_pagerank(repos, link_graph)
    
    # Step 4: Find gems
    print("\n[4/4] Finding hidden gems...")
    gems = find_hidden_gems(repos, pagerank_scores, top_n=10)
    
    # Show results
    print_results(gems)
    visualize_network(G, pagerank_scores, gems)
    
    # Optional: Save to file
    with open(f'hidden_gems_{TOPIC}.txt', 'w') as f:
        f.write(f"Hidden Gems in {TOPIC}\n")
        f.write("="*70 + "\n\n")
        for i, gem in enumerate(gems, 1):
            f.write(f"{i}. {gem['name']}\n")
            f.write(f"   Stars: {gem['stars']}\n")
            f.write(f"   PageRank: {gem['pagerank']:.6f}\n")
            f.write(f"   Gem Score: {gem['gem_score']:.2f}\n")
            f.write(f"   URL: {gem['url']}\n\n")
    
    print(f"\n✅ Results saved to: hidden_gems_{TOPIC}.txt")


if __name__ == "__main__":
    main()
    
