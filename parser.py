import re
import time
import gc
import torch
from collections import Counter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from pypdf import PdfReader


# Images scale presets per document type.
# Higher scale = better quality for vision model, higher VRAM cost.
_SCALE_DEFAULTS: dict[str, float] = {
    "novel":     0.75,   # text-heavy, minimal diagrams — VRAM safety first
    "general":   1.0,    # balanced default
    "technical": 2.0,    # formulas + architecture diagrams need full detail
}


def remove_running_headers(
    markdown_text: str,
    total_pages: int,
    threshold_ratio: float = 0.3,
) -> str:
    """
    Detect and deduplicate running page headers from markdown output.

    Headings that appear on more than `threshold_ratio` of the total pages are
    treated as running headers — kept at their first occurrence, removed
    everywhere else.

    Args:
        markdown_text:    Full markdown content to process.
        total_pages:      Total page count of the source PDF.
        threshold_ratio:  Fraction of total_pages at which a heading is
                          considered a running header (default: 0.3 = 30%).

    Returns:
        Cleaned markdown string.
    """
    lines = markdown_text.split('\n')

    heading_counts = Counter(
        line.strip() for line in lines if line.strip().startswith('#')
    )

    threshold = max(3, int(total_pages * threshold_ratio))
    running_headers = {h for h, n in heading_counts.items() if n >= threshold}

    if running_headers:
        print(f"[INFO] Detected {len(running_headers)} running page header(s) — keeping first occurrence only:")
        for h in sorted(running_headers):
            print(f"  '{h}' appeared {heading_counts[h]}x")
    else:
        print("[INFO] No running page headers detected.")

    result_lines = []
    seen = set()
    for line in lines:
        stripped = line.strip()
        if stripped in running_headers:
            if stripped not in seen:
                seen.add(stripped)
                result_lines.append(line)
            # else: silently drop the duplicate
        else:
            result_lines.append(line)

    return '\n'.join(result_lines)


def clean_for_rag(
    markdown_text: str,
    total_pages: int,
    threshold_ratio: float = 0.3,
) -> str:
    """
    Strip running page headers and pipeline artifacts from markdown before
    chunking for a RAG pipeline.

    Unlike remove_running_headers, this removes ALL occurrences of running
    headers (including the first), since they carry no semantic value for
    embedding. Also strips batch boundary comments left by the parser.

    Args:
        markdown_text:    Full markdown content to process.
        total_pages:      Total page count of the source PDF.
        threshold_ratio:  Fraction of total_pages at which a heading is
                          considered a running header (default: 0.3 = 30%).

    Returns:
        RAG-ready markdown string.
    """
    # Strip batch boundary markers
    text = re.sub(r'<!-- batch_end pages=\d+-\d+ -->', '', markdown_text)

    lines = text.split('\n')
    heading_counts = Counter(
        line.strip() for line in lines if line.strip().startswith('#')
    )

    threshold = max(3, int(total_pages * threshold_ratio))
    running_headers = {h for h, n in heading_counts.items() if n >= threshold}

    if running_headers:
        print(f"[RAG] Removing {len(running_headers)} running header(s) entirely:")
        for h in sorted(running_headers):
            print(f"  '{h}' appeared {heading_counts[h]}x")

    # Drop ALL occurrences — no first-occurrence exception for RAG
    cleaned = '\n'.join(
        line for line in lines if line.strip() not in running_headers
    )

    # Collapse excess blank lines left behind
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()


def parse_heavy_multimodal(
    input_path: str,
    output_path: str,
    batch_size: int = 10,
    images_scale: float | None = None,
    describe_pictures: bool = True,
    enrich_formulas: bool = False,
    header_threshold_ratio: float = 0.3,
    doc_type: str = "general",
) -> str:
    """
    Convert a PDF to Markdown using Docling's multimodal pipeline, processing
    in page batches to prevent VRAM overflow on consumer GPUs (e.g. RTX 4060).

    Args:
        input_path:             Path to the source PDF file.
        output_path:            Path where the output Markdown file will be written.
        batch_size:             Number of pages to process per batch (default: 10).
        images_scale:           Render scale for extracted images. Overrides the
                                doc_type default when set explicitly. Values < 1.0
                                reduce VRAM; values > 1.0 improve quality for the
                                vision model at higher memory cost.
        describe_pictures:      Enable vision-model captioning of figures. The most
                                expensive flag in the pipeline — disable if speed
                                matters more than diagram descriptions.
        enrich_formulas:        Enable LaTeX OCR for detected formula regions.
                                Requires `pip install "docling[formula]"`.
                                Auto-enabled when doc_type="technical".
        header_threshold_ratio: Headings appearing on more than this fraction of
                                total pages are treated as running page headers and
                                deduplicated. Default 0.3 (30%). Lower if legitimate
                                repeated headings are being stripped; raise if headers
                                still leak through.
        doc_type:               Document type preset — controls default images_scale
                                and formula enrichment.
                                  "novel"     → scale 0.75, formulas off
                                  "general"   → scale 1.0,  formulas off  (default)
                                  "technical" → scale 2.0,  formulas auto-enabled

    Returns:
        The output_path on success.
    """
    # --- Resolve doc_type defaults ---
    if images_scale is None:
        images_scale = _SCALE_DEFAULTS.get(doc_type, 1.0)

    if doc_type == "technical" and not enrich_formulas:
        enrich_formulas = True
        print("[INFO] doc_type='technical': formula enrichment auto-enabled.")

    print(f"Starting Advanced Multimodal Docling conversion for: {input_path}")
    print(f"  doc_type={doc_type} | images_scale={images_scale} | "
          f"enrich_formulas={enrich_formulas} | describe_pictures={describe_pictures}")

    reader = PdfReader(input_path)
    total_pages = len(reader.pages)
    print(f"Detected {total_pages} pages in the document.")
    print(f"Processing in batches of {batch_size} to prevent VRAM overflow.\n")

    start_time = time.time()

    # Configure the pipeline
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.do_picture_description = describe_pictures
    pipeline_options.images_scale = images_scale
    pipeline_options.do_formula_enrichment = enrich_formulas

    # Initialize the converter once; reused across all batches
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Clear the output file before appending
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("")

    print("Analyzing layout, extracting text, and reading diagrams... Please wait.")

    failed_batches = []

    for start_page in range(1, total_pages + 1, batch_size):
        end_page = min(start_page + batch_size - 1, total_pages)
        print(f"-> Processing pages {start_page} to {end_page}...")

        result = None
        try:
            result = converter.convert(input_path, page_range=(start_page, end_page))
            markdown_output = result.document.export_to_markdown()

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(markdown_output)
                f.write(f"\n\n<!-- batch_end pages={start_page}-{end_page} -->\n\n")

        except Exception as e:
            print(f"  [WARNING] Failed on pages {start_page}-{end_page}: {e}")
            failed_batches.append((start_page, end_page))

        finally:
            del result
            gc.collect()
            torch.cuda.empty_cache()

    # --- Post-processing: remove running page headers ---
    print("\nPost-processing: scanning for running page headers...")
    with open(output_path, "r", encoding="utf-8") as f:
        raw_markdown = f.read()

    cleaned_markdown = clean_for_rag(raw_markdown, total_pages, header_threshold_ratio)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_markdown)

    end_time = time.time()

    print("\n--- Advanced Parsing Complete ---")
    print(f"Successfully saved to: {output_path}")
    print(f"Time taken: {round(end_time - start_time, 2)} seconds")

    if failed_batches:
        print(f"\n[WARNING] {len(failed_batches)} batch(es) failed and were skipped:")
        for s, e in failed_batches:
            print(f"  - Pages {s}-{e}")

    return output_path


if __name__ == "__main__":

    # Research paper — high-res images, LaTeX formula decoding
    parse_heavy_multimodal(
        "data/prob.pdf",
        "data/prob_parsed.md",
        doc_type="technical",
    )