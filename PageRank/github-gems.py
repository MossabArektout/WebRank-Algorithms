import requests
import time
from dotenv import load_dotenv
import os
import re

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
    graph = {}
    repos_names = {repo['full_name'] for repo in repos}

    for i, repo in enumerate(repos):
        print(f"Processing {i+1}/{len(repos)}: {repo['full_name']}")

        readme = get_readme(repo['owner'], repo['name'])
        if not readme : 
            graph[repo['full_name']] = []
            continue

        all_links = extract_github_links(readme)
        
        # Only keep links to repos in our dataset
        valid_links = [link for link in all_links if link in repo_names]
        
        graph[repo['full_name']] = valid_links
        
        print(f"  Found {len(valid_links)} links to other repos in our set")

        time.sleep(0.5)

    return graph

# Test it
if __name__ == "__main__":
    repos = search_repos(TOPIC, max_repos=30)
    for repo in repos[:5]:  # Print first 5
        print(f"{repo['full_name']} - {repo['stars']} stars")
    
