
import os
from llama_parse import LlamaParse
from dotenv import load_dotenv
load_dotenv()

def parse_s3_pdf_to_markdown_table(
    pdf_path: str,
    output_dir: str = "markdowns_20",
    old_pdfs_dir: str = "old_pdfs",
    pdfs_dir: str = "pdfs"
):
    try:
        # Extract the PDF filename from the path
        pdf_filename = pdf_path.split("/")[-1]
        
        # Check if the PDF already exists in old_pdfs directory
        old_pdfs_path = os.path.join(os.path.dirname(__file__), old_pdfs_dir)
        old_pdf_file_path = os.path.join(old_pdfs_path, pdf_filename)
        
        if os.path.exists(old_pdf_file_path):
            print(f"PDF {pdf_filename} already exists in old_pdfs directory. Skipping processing...")
            return None
        
        print(f"Processing PDF: {pdf_filename}")
        
        parser = LlamaParse(
                        api_key=os.getenv("LLAMA_PARSE_KEY"),
                        result_type="markdown",
                        auto_mode=True,
                        auto_mode_trigger_on_table_in_page=True,
                        skip_diagonal_text=True,
                        disable_ocr=True,
                        disable_image_extraction=True,
                        do_not_cache=True,
                        verbose=True, 
                    ).load_data(pdf_path)
        markdown_text = "\n".join(doc.text for doc in parser)

        # Create markdowns directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save markdown text to file
        markdown_filename = f"{pdf_filename.split('.')[0]}.md"
        markdown_file_path = os.path.join(output_dir, markdown_filename)
        
        with open(markdown_file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)

        print(f"Markdown saved to: {markdown_file_path}")
        print(markdown_text)
        return markdown_text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

def process_all_pdfs_from_directory(
    pdfs_dir: str = "pdfs",
    old_pdfs_dir: str = "old_pdfs", 
    output_dir: str = "markdowns_20"
):
    """
    Process all PDFs from the pdfs directory, skipping those that already exist in old_pdfs
    """
    pdfs_path = os.path.join(os.path.dirname(__file__), pdfs_dir)
    
    if not os.path.exists(pdfs_path):
        print(f"PDFs directory {pdfs_path} does not exist!")
        return
    
    pdf_files = [f for f in os.listdir(pdfs_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdfs_path}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    processed_count = 0
    skipped_count = 0
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdfs_path, pdf_file)
        result = parse_s3_pdf_to_markdown_table(
            pdf_path=pdf_path,
            output_dir=output_dir,
            old_pdfs_dir=old_pdfs_dir,
            pdfs_dir=pdfs_dir
        )
        
        if result is None:
            skipped_count += 1
        else:
            processed_count += 1
    
    print(f"\nProcessing complete:")
    print(f"  - Processed: {processed_count} PDFs")
    print(f"  - Skipped: {skipped_count} PDFs")

# Example usage
if __name__ == "__main__":
  
    parse_s3_pdf_to_markdown_table("try.pdf")