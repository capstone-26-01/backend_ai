import requests
import base64

def get_file_tree(repo_path):
    api_url = f'https://api.github.com/repos/{repo_path}/git/trees/HEAD?recursive=1'
    response = requests.get(api_url)

    if response.status_code != 200:
        return None

    data = response.json()
    files = []
    for item in data.get('tree', []):
        if item['type'] == 'blob':
            files.append(item['path'])
    return files


def get_file_content(repo_path, file_path):
    api_url = f'https://api.github.com/repos/{repo_path}/contents/{file_path}'
    response = requests.get(api_url)

    if response.status_code != 200:
        return None

    content = response.json().get('content', '')
    return base64.b64decode(content).decode('utf-8')