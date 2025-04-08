"""
Downloads pretrained models for nowcast and synrad
"""
# import pandas as pd
# import urllib.request
# import os

# def main():
#     model_info = pd.read_csv('model_urls.csv')
#     for i,r in model_info.iterrows():
#         print(f'Downloading {r.model}...')
#         download_file(r.url,f'{r.application}/{r.model}')

# def download_file(url,filename):
#     os.system(f'wget -O {filename} {url}')

# if __name__=='__main__':
#     main()

import pandas as pd
import os
import requests

def main():
    model_info = pd.read_csv('model_urls.csv')
    for i, r in model_info.iterrows():
        print(f'Downloading {r.model}...')
        download_file(r.url, f'{r.application}/{r.model}')

def download_file(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"Failed to download {filename}")

if __name__ == '__main__':
    main()



