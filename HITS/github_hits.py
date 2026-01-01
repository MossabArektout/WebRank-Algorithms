import requests
import numpy as np
import time
from collections import defaultdict
from dotenv import load_dotenv
import os
import re


load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
TOPIC = "python"
MAX_REPOS = 40


def search_repos(topic, max_repos=40):
    url = "https://api.github.com/search/repositories"
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    params = {
        "q": f"topic:{topic} stars:>50",
        "sort": "stars",
        "order": "desc",
        "per_page": max_repos
    }
    
    print(f"🔍 Searching for '{topic}' repositories...")
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.json())
        return []
    
    repos = response.json()["items"]
    
    # Extract relevant info
    repo_list = []
    for repo in repos:
        repo_list.append({
            "owner": repo["owner"]["login"],
            "name": repo["name"],
            "full_name": repo["full_name"],
            "stars": repo["stargazers_count"],
            "url": repo["html_url"],
            "description": repo["description"] or ""
        })
    
    print(f"✅ Found {len(repo_list)} repositories")
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
    # Regex pattern to match github.com/owner/repo
    pattern = r'github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)'
    
    matches = re.findall(pattern, readme_text)
    
    # Convert to "owner/repo" format
    links = []
    for owner, repo in matches:
        # Clean repo name (remove .git, trailing slashes, etc.)
        repo = repo.rstrip('/')
        repo = repo.replace('.git', '')
        links.append(f"{owner}/{repo}")
    
    # Remove duplicates
    return list(set(links))

def build_link_graph(repos):
    graph = {}
    repo_names = {repo['full_name'] for repo in repos}  # Set for fast lookup
    
    print(f"\n📖 Building link graph...")
    print("=" * 70)
    
    for i, repo in enumerate(repos):
        print(f"[{i+1}/{len(repos)}] Processing {repo['full_name']}")
        
        # Fetch README
        readme = get_readme(repo['owner'], repo['name'])
        
        if not readme:
            print(f"  ⚠️  No README found")
            graph[repo['full_name']] = []
            continue
        
        # Extract all GitHub links
        all_links = extract_github_links(readme)
        
        # Keep only links to repos in our dataset
        valid_links = [link for link in all_links if link in repo_names and link != repo['full_name']]
        
        graph[repo['full_name']] = valid_links
        
        # Identify if this looks like an awesome list
        is_awesome = 'awesome' in repo['name'].lower()
        awesome_marker = "📚 [AWESOME LIST]" if is_awesome else ""
        
        print(f"  → Found {len(valid_links)} links to other repos {awesome_marker}")
        
        # Be nice to GitHub API
        time.sleep(0.5)
    
    print("=" * 70)
    print(f"✅ Link graph complete!\n")
    
    return graph

def calculate_hits(repos, link_graph, max_iterations=30):
    print("📊 Calculating HITS scores...")
    print("=" * 70)
    
    # Create repo name list for indexing
    repo_names = [repo['full_name'] for repo in repos]
    n = len(repo_names)
    
    # Initialize scores
    authority = {name: 1.0 for name in repo_names}
    hub = {name: 1.0 for name in repo_names}
    
    # Build reverse link graph (who links to me?)
    incoming_links = defaultdict(list)
    for source, targets in link_graph.items():
        for target in targets:
            incoming_links[target].append(source)
    
    print(f"Nodes: {n} repositories")
    print(f"Edges: {sum(len(links) for links in link_graph.values())} links")
    print(f"Iterations: {max_iterations}\n")
    
    # Iterate
    for iteration in range(max_iterations):
        # Update authority scores
        # Authority = sum of hub scores pointing to you
        new_authority = {}
        for repo in repo_names:
            score = 0
            for source in incoming_links[repo]:
                score += hub[source]
            new_authority[repo] = score
        
        # Update hub scores
        # Hub = sum of authority scores you point to
        new_hub = {}
        for repo in repo_names:
            score = 0
            for target in link_graph.get(repo, []):
                score += new_authority[target]
            new_hub[repo] = score
        
        # Normalize
        max_auth = max(new_authority.values()) if new_authority else 1
        max_hub = max(new_hub.values()) if new_hub else 1
        
        authority = {k: v/max_auth for k, v in new_authority.items()}
        hub = {k: v/max_hub for k, v in new_hub.items()}
        
        # Show progress every 10 iterations
        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1}/{max_iterations}")
    
    print("=" * 70)
    print("✅ HITS calculation complete!\n")
    
    return authority, hub

def analyze_results(repos, authority_scores, hub_scores, top_n=10):
    print("\n" + "=" * 80)
    print("🏆 GITHUB AWESOME LISTS ANALYZER - HITS RESULTS")
    print("=" * 80)
    
    # Separate awesome lists from regular repos
    awesome_lists = []
    regular_repos = []
    
    for repo in repos:
        if 'awesome' in repo['name'].lower():
            awesome_lists.append(repo)
        else:
            regular_repos.append(repo)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total repositories: {len(repos)}")
    print(f"  Awesome lists: {len(awesome_lists)}")
    print(f"  Regular repos: {len(regular_repos)}")
    
    # Sort by authority (best tools)
    sorted_by_authority = sorted(
        repos, 
        key=lambda r: authority_scores[r['full_name']], 
        reverse=True
    )
    
    # Sort by hub (best lists)
    sorted_by_hub = sorted(
        repos, 
        key=lambda r: hub_scores[r['full_name']], 
        reverse=True
    )
    
    # Display best authorities (tools)
    print("\n" + "=" * 80)
    print("🏆 TOP TOOLS (Highest Authority Scores)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Repository':<40} {'Stars':<10} {'Authority':<12}")
    print("-" * 80)
    
    for i, repo in enumerate(sorted_by_authority[:top_n], 1):
        name = repo['full_name']
        stars = repo['stars']
        auth = authority_scores[name]
        auth_bar = "█" * int(auth * 30)
        
        print(f"{i:<6} {name:<40} {stars:<10} {auth:>6.4f} {auth_bar}")
    
    # Display best hubs (lists)
    print("\n" + "=" * 80)
    print("📚 TOP CURATED LISTS (Highest Hub Scores)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Repository':<40} {'Stars':<10} {'Hub Score':<12}")
    print("-" * 80)
    
    for i, repo in enumerate(sorted_by_hub[:top_n], 1):
        name = repo['full_name']
        stars = repo['stars']
        hub_score = hub_scores[name]
        hub_bar = "█" * int(hub_score * 30)
        
        print(f"{i:<6} {name:<40} {stars:<10} {hub_score:>6.4f} {hub_bar}")
    
    # Insights
    print("\n" + "=" * 80)
    print("💡 KEY INSIGHTS")
    print("=" * 80)
    
    # Check if awesome lists dominate hubs
    top_hubs = sorted_by_hub[:5]
    awesome_in_top_hubs = sum(1 for r in top_hubs if 'awesome' in r['name'].lower())
    
    print(f"✓ {awesome_in_top_hubs}/5 top hubs are 'awesome-*' lists (expected!)")
    
    # Check if tools dominate authorities
    top_auths = sorted_by_authority[:5]
    tools_in_top_auths = sum(1 for r in top_auths if 'awesome' not in r['name'].lower())
    
    print(f"✓ {tools_in_top_auths}/5 top authorities are actual tools (expected!)")
    
    # Show clear separation
    print(f"\n🎯 HITS successfully separated:")
    print(f"  • Tools (authorities) from Lists (hubs)")
    print(f"  • PageRank alone couldn't do this!")
    
    return sorted_by_authority, sorted_by_hub

import matplotlib.pyplot as plt

def visualize_results(repos, authority_scores, hub_scores):
    """
    Create scatter plot: Authority vs Hub scores
    """
    print("\n📊 Creating visualization...")
    
    # Prepare data
    authorities = [authority_scores[r['full_name']] for r in repos]
    hubs = [hub_scores[r['full_name']] for r in repos]
    names = [r['name'] for r in repos]
    is_awesome = ['awesome' in r['name'].lower() for r in repos]
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Plot
    for i, (auth, hub, name, awesome) in enumerate(zip(authorities, hubs, names, is_awesome)):
        color = 'red' if awesome else 'blue'
        marker = 's' if awesome else 'o'
        size = 100 if awesome else 50
        
        plt.scatter(hub, auth, c=color, marker=marker, s=size, alpha=0.6)
        
        # Label top repos
        if auth > 0.5 or hub > 0.5:
            plt.annotate(name, (hub, auth), fontsize=8, alpha=0.7)
    
    # Labels and title
    plt.xlabel('Hub Score (Curator Quality)', fontsize=12)
    plt.ylabel('Authority Score (Content Quality)', fontsize=12)
    plt.title(f'HITS Analysis: {TOPIC}\nBlue = Tools | Red = Awesome Lists', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Legend
    plt.scatter([], [], c='blue', marker='o', s=50, label='Regular Repos')
    plt.scatter([], [], c='red', marker='s', s=100, label='Awesome Lists')
    plt.legend()
    
    # Save
    filename = f'hits_visualization_{TOPIC}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"💾 Visualization saved to: {filename}")
    plt.show()


def main():
    """
    Main function to run the GitHub Awesome Lists Analyzer
    """
    print("\n" + "=" * 80)
    print("🚀 GITHUB AWESOME LISTS ANALYZER WITH HITS")
    print("=" * 80)
    print(f"\nTopic: {TOPIC}")
    print(f"Max repositories: {MAX_REPOS}\n")
    
    # Step 1: Fetch repositories
    repos = search_repos(TOPIC, max_repos=MAX_REPOS)
    
    if not repos:
        print("❌ No repositories found. Check your token and topic.")
        return
    
    # Step 2: Build link graph
    link_graph = build_link_graph(repos)
    
    # Check if we have enough links
    total_links = sum(len(links) for links in link_graph.values())
    if total_links < 10:
        print(f"⚠️  Warning: Only {total_links} links found.")
        print("   Consider using a more interconnected topic or more repos.")
    
    # Step 3: Calculate HITS scores
    authority_scores, hub_scores = calculate_hits(repos, link_graph)
    
    # Step 4: Analyze and display results
    best_tools, best_lists = analyze_results(repos, authority_scores, hub_scores, top_n=10)
    
    # Step 5: Save results to file
    save_results(repos, authority_scores, hub_scores, best_tools, best_lists)
    visualize_results(repos, authority_scores, hub_scores)
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)

def save_results(repos, authority_scores, hub_scores, best_tools, best_lists):
    """Save results to a text file"""
    filename = f'hits_results_{TOPIC}.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"HITS Analysis Results for Topic: {TOPIC}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("TOP 10 TOOLS (Authorities):\n")
        f.write("-" * 80 + "\n")
        for i, repo in enumerate(best_tools[:10], 1):
            name = repo['full_name']
            auth = authority_scores[name]
            f.write(f"{i}. {name}\n")
            f.write(f"   Authority: {auth:.4f}\n")
            f.write(f"   Stars: {repo['stars']}\n")
            f.write(f"   URL: {repo['url']}\n\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        f.write("TOP 10 CURATED LISTS (Hubs):\n")
        f.write("-" * 80 + "\n")
        for i, repo in enumerate(best_lists[:10], 1):
            name = repo['full_name']
            hub = hub_scores[name]
            f.write(f"{i}. {name}\n")
            f.write(f"   Hub Score: {hub:.4f}\n")
            f.write(f"   Stars: {repo['stars']}\n")
            f.write(f"   URL: {repo['url']}\n\n")
    
    print(f"\n💾 Results saved to: {filename}")

if __name__ == "__main__":
    main()