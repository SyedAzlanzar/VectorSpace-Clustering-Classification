import urllib.request
import os

def download_test_image(imageNo='01'):
    if not os.path.exists("output"):
        os.makedirs("output")
    url = f"https://www.iqandreas.com/sample-images/100-100-color/{imageNo}.jpg"
    # Add User-Agent to avoid HTTP 403 errors from servers that block non-browser requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        with open("output/real_test.jpg", "wb") as f:
            f.write(response.read())

    return url
