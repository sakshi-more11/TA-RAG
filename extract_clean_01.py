import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        text = page.get_text("text")
        full_text += text + "\n"

    return full_text

def clean_text(text):
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Remove page numbers
        if line.isdigit():
            continue

        # Remove very short meaningless lines
        if len(line) < 3:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

# Process all PDFs in folder
input_folder = "data/raw"
output_folder = "data/cleaned"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(input_folder, file)
        text = extract_text_from_pdf(pdf_path)
        cleaned = clean_text(text)

        output_path = os.path.join(output_folder, file.replace(".pdf", ".txt"))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"Processed: {file}")
