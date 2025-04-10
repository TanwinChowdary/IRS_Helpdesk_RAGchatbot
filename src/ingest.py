import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
FAQ_FILE = os.path.join(DATA_DIR, "irs_faq.txt")

os.makedirs(PDF_DIR, exist_ok=True)

# -------------------------------
# Function to scrape IRS FAQs
# -------------------------------
def fetch_irs_faq():
    url = "https://www.irs.gov/help/ita"
    print(f"Scraping FAQ from: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    faq_divs = soup.find_all("div", class_="card-body")
    faqs = [div.get_text(strip=True) for div in faq_divs if div.get_text(strip=True)]

    with open(FAQ_FILE, "w", encoding="utf-8") as f:
        for faq in faqs:
            f.write(faq + "\n\n")

    print(f"✅ IRS FAQs saved to {FAQ_FILE}")


# -------------------------------
# Function to download IRS PDFs
# -------------------------------
PDF_LINKS = {
    "Publication 17": "https://www.irs.gov/pub/irs-pdf/p17.pdf",
    "Publication 505": "https://www.irs.gov/pub/irs-pdf/p505.pdf",
    "Form 1040 Instructions": "https://www.irs.gov/pub/irs-pdf/i1040gi.pdf"
}

def download_irs_pdfs():
    for name, url in PDF_LINKS.items():
        filename = os.path.join(PDF_DIR, url.split('/')[-1])
        print(f"Downloading {name}...")
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✅ Saved: {filename}")


# -------------------------------
# Run Everything
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting IRS data ingestion...\n")
    fetch_irs_faq()
    download_irs_pdfs()
    print("\n🎯 Done! Your data lives in the 'data/' directory.")
