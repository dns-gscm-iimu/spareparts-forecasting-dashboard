from pypdf import PdfReader

def extract_text():
    try:
        reader = PdfReader("project report 2.pdf")
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
            
        with open("extracted_report_context.txt", "w") as f:
            f.write(text)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_text()
