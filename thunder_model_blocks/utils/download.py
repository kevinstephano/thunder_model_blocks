import urllib.request

def download(url, module_name):
    # Download the Python file
    urllib.request.urlretrieve(url, f"{module_name}.py")
