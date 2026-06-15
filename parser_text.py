import time
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


def parse_fast_text(input_path, output_path):
    print(f"Starting Fast Text-Only Docling conversion for: {input_path}")
    print("Note: OCR and Vision models are DISABLED. This will run on CPU instantly.\n")

    start_time = time.time()

    # 1. Configure the pipeline for Text-Only extraction
    pipeline_options = PdfPipelineOptions()

    # The Magic Switches: Turn off all heavy Vision and OCR engines
    pipeline_options.do_ocr = False
    pipeline_options.generate_picture_images = False
    pipeline_options.do_picture_description = False

    # 2. Initialize the converter with our fast options
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # 3. Process the PDF (No batching needed for text-only)
    print("Extracting native text... Please wait.")
    result = converter.convert(input_path)

    # 4. Export and Save
    markdown_output = result.document.export_to_markdown()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_output)

    end_time = time.time()

    print("\n--- Fast Parsing Complete ---")
    print(f"Successfully saved to: {output_path}")
    print(f"Time taken: {round(end_time - start_time, 2)} seconds")


if __name__ == "__main__":
    pdf_file = "data/sample_report_3.pdf"
    md_file = "data/sample_report_3_text_parsed.md"

    parse_fast_text(pdf_file, md_file)