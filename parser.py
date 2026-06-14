import time
import gc
import torch
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

def parse_pdf_with_images(input_path, output_path, total_pages=64):
    print(f"Starting Advanced Multimodal Docling conversion for: {input_path}")
    print("Note: Processing in batches of 10 to prevent VRAM overflow.\n")

    start_time = time.time()

    # 1. Configure the pipeline for image extraction and description
    pipeline_options = PdfPipelineOptions()

    # Optimizations for large documents to prevent std::bad_alloc OOM
    pipeline_options.generate_picture_images = True
    pipeline_options.do_picture_description = True
    pipeline_options.images_scale = 1.2

    # 2. Initialize the converter with our custom options
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # 3. Clear the output file before starting so we can append to it cleanly
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("")

    # 4. Process the PDF in batches of 10
    batch_size = 10
    print("Analyzing layout, extracting text, and reading diagrams... Please wait.")

    for start_page in range(1, total_pages + 1, batch_size):
        end_page = min(start_page + batch_size - 1, total_pages)
        print(f"-> Processing pages {start_page} to {end_page}...")

        # Process just this chunk
        result = converter.convert(input_path, page_range=(start_page, end_page))

        # Export the chunk to Markdown
        markdown_output = result.document.export_to_markdown()

        # 5. Append this batch to the file
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(markdown_output + "\n\n---\n\n")

        # 6. CRITICAL VRAM PURGE: Delete the memory objects and empty the GPU cache
        del result
        gc.collect()
        torch.cuda.empty_cache()

    end_time = time.time()

    print("\n--- Advanced Parsing Complete ---")
    print(f"Successfully saved to: {output_path}")
    print(f"Time taken: {round(end_time - start_time, 2)} seconds")


if __name__ == "__main__":
    pdf_file = "data/sample_report_2.pdf"
    md_file = "data/sample_report_2_parsed.md"

    # Set total_pages to the exact length of your PDF
    parse_pdf_with_images(pdf_file, md_file, total_pages=64)