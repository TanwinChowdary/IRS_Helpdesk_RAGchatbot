import requests
from bs4 import BeautifulSoup
import pdfplumber

def fetch_faq():
    response = requests.get("https://www.irs.gov/help/ita")
    soup = BeautifulSoup(response.text, 'html.parser')
    faq_sections = soup.find_all('div', class_='card-body')
    return [faq.get_text(strip=True) for faq in faq_sections]

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or '' for page in pdf.pages)
