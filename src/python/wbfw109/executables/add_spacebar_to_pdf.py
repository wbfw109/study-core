import fitz  # PyMuPDF
import spacy

# Open PDF file
pdf_document: str = "cc.pdf"
pdf_doc: fitz.Document = fitz.open(pdf_document)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Iterate over all pages
for page_num in range(pdf_doc.page_count):
    print(page_num)
    page: fitz.Page = pdf_doc.load_page(page_num)

    # Extract text with position information (including bounding boxes)
    extracted_text_instances = page.get_text("dict")  # Get text with coordinates

    # Collect all spans to process them in bulk
    all_texts = []
    all_spans = []
    for block in extracted_text_instances["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    original_text = span["text"]
                    all_texts.append(original_text)
                    all_spans.append(span)

    # Process all texts at once using spaCy
    corrected_texts = []
    spacy_docs = nlp(" ".join(all_texts))  # Process all texts in a single batch
    token_index = 0

    for token in spacy_docs:
        corrected_texts.append(token.text)

    # Re-apply corrected texts to spans and update PDF
    for span, corrected_text in zip(all_spans, corrected_texts):
        rect = fitz.Rect(span["bbox"])  # Use the original bounding box

        # Step 1: Draw a white rectangle over the existing text
        page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))  # Erase original text

        # Step 2: Insert corrected text at the same position
        page.insert_text(rect.top_left, corrected_text, fontsize=span["size"])
    if page_num >= 5:
        break
# Save the PDF with corrected spacing
pdf_doc.save("cc_spacing_corrected.pdf")
pdf_doc.close()
