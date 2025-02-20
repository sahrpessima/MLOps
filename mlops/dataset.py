import os
from pathlib import Path as P
from concurrent.futures import ThreadPoolExecutor
from duckduckgo_search import DDGS

def scrap_images(query, n_results=40):
    """Скрапинг изображений из DuckDuckGo."""
    with DDGS() as ddgs:
        ddgs_images_gen = ddgs.images(
            query,
            region="wt-wt",
            size="Medium",
            type_image="photo",
            max_results=n_results,
        )
        return [r['image'] for r in ddgs_images_gen]

def download_files(urls, label, target_dir):
    """Загрузка изображений по URL-адресам."""
    def download_1(u, label):
        os.system(f"wget -P '{P(target_dir) / label}' '{u.strip()}'")
        print(f"Скачано: {u.strip()}")
    
    os.makedirs(P(target_dir) / label, exist_ok=True)
    
    with ThreadPoolExecutor() as executor:
        for url in urls:
            executor.submit(download_1, url, label)
