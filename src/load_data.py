import os
import pdfplumber

def extract_text_from_pdfs(pdf_dir="data/pdfs"):
    all_text = ""
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            try:
                with pdfplumber.open(os.path.join(pdf_dir, filename)) as pdf:
                    all_text += "\n".join([page.extract_text() or '' for page in pdf.pages]) + "\n\n"
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return all_text

def load_faq_text(faq_file="data/irs_faq.txt"):
    return open(faq_file, "r", encoding="utf-8").read() if os.path.exists(faq_file) else ""

def load_all_irs_text():
    combined_file = "data/combined_irs_text.txt"
    if os.path.exists(combined_file):
        return open(combined_file, "r", encoding="utf-8").read()
    faq_text = load_faq_text()
    pdf_text = extract_text_from_pdfs()
    full_text = f"{faq_text}\n\n{pdf_text}"
    with open(combined_file, "w", encoding="utf-8") as f:
        f.write(full_text)
    return full_text
