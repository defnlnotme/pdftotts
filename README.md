# pdftotts: PDF/Image-to-TTS-Optimized Text Transcription

`pdftotts` is a Python script that leverages Large Language Models (LLMs) via the CAMEL-AI framework to transcribe text from sequences of book page images or directly from PDF files. The key goal is to produce clean, continuous prose optimized for Text-to-Speech (TTS) synthesis, omitting disruptive elements like page numbers, references, and structural markers.

## Features

*   **Processes PDFs or Image Directories:** Can directly convert PDFs to images and process them, or work with pre-extracted page images.
*   **TTS-Optimized Output:**
    *   Seamlessly concatenates text across page boundaries.
    *   Omits page numbers, running headers/footers, citations, figure/table labels, footnote markers, and URLs.
    *   Attempts to describe the content/purpose of images, tables, and equations instead of just noting their presence.
    *   Converts lists and formatting into flowing prose.
*   **Contextual Transcription:** Uses a 3-page sliding window (previous, current, next page images) to provide visual and textual context to the LLM, improving transcription accuracy and flow, especially across page breaks.
*   **Ancillary Page Skipping:** Intelligently skips common front/back matter pages (Cover, TOC, Index, Bibliography, etc.) based on LLM analysis, outputting `SKIPPED` for these pages.
*   **Automatic Page Range Detection (PDFs):** Can use an LLM to analyze the PDF structure and identify the start and end pages of the main content body (e.g., Chapters), skipping front/back matter automatically.
*   **Manual Page Range Control:** Allows specifying exact start and end page numbers to process.
*   **Append Mode:** Can resume processing by appending to existing output files, using the end of the previous text as context.
*   **Flexible Model Selection:** Supports various multimodal LLMs available through the CAMEL-AI `ModelFactory` (e.g., GPT-4o/vision, Gemini Pro/Flash, Claude 3.5 Sonnet).
*   **Dependency Management:** Uses external tools `pdftoppm` (poppler-utils) and `montage` (ImageMagick).

## Prerequisites

1.  **Python:** Python 3.8 or higher is recommended.
2.  **Pip:** Python package installer.
3.  **External Tools:**
    *   **poppler-utils:** Provides `pdftoppm` for PDF-to-image conversion.
        *   Ubuntu/Debian: `sudo apt update && sudo apt install poppler-utils`
        *   macOS (Homebrew): `brew install poppler`
        *   Windows: Requires manual installation, e.g., via [Chocolatey](https://chocolatey.org/packages/poppler) or downloading binaries. Ensure `pdftoppm.exe` is in your system's PATH.
    *   **ImageMagick:** Provides `montage` for creating composite context images.
        *   Ubuntu/Debian: `sudo apt update && sudo apt install imagemagick`
        *   macOS (Homebrew): `brew install imagemagick`
        *   Windows: Download from the [official ImageMagick website](https://imagemagick.org/script/download.php). Ensure `montage.exe` is in your system's PATH.
4.  **LLM API Keys:** You need API keys for the LLM provider(s) you intend to use (e.g., OpenAI, Google AI Studio, Anthropic).

## Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/defnlnotme/pdftotts.git # Replace with your repo URL
    cd pdftotts
    ```

2.  **Install Python Dependencies:**
    ```bash
    pip install python-dotenv camel-ai Pillow # Pillow might be needed by CAMEL
    # Or, if a requirements.txt is provided:
    # pip install -r requirements.txt
    ```
    *Note: `camel-ai` might have additional dependencies depending on the specific models/toolkits you use. Refer to the [CAMEL-AI documentation](https://www.camel-ai.org/).*

3.  **Configure API Keys:**
    Create a file named `.env` in the root directory of the cloned repository (e.g., alongside the `pdftotts` subdirectory). Add your API keys like this:
    ```dotenv
    # .env file
    OPENAI_API_KEY="sk-..."
    GEMINI_API_KEY="AIza..."
    ANTHROPIC_API_KEY="sk-ant-..."
    # Add other keys as needed by CAMEL for your chosen models
    ```
    The script uses `python-dotenv` to load these keys automatically.

## Usage

The script is run from the command line.

```bash
python pdftotts/pdftotts.py [MODE] [OPTIONS]
```

**Modes (Choose One):**

*   `-i, --input-dir PATH`: Specify a directory containing **pre-extracted page images**. Images **must** be named with a page number suffix, like `book-name-001.png`, `page-042.jpg`, etc. The script extracts the number after the last hyphen (`-`).
*   `-b, --book-dir PATH`: Specify a directory containing **PDF files**. The script will convert each PDF to images in a temporary directory and process them sequentially.

**Common Options:**

*   `-o, --output PATH`:
    *   With `--input-dir`: Path to the **single output text file**. If omitted, output goes to the console.
    *   With `--book-dir`: Path to the **output directory**. A separate `.txt` file will be created for each processed PDF inside this directory (e.g., `book1.txt`, `book2.txt`). If omitted, defaults to a `books_tts` subdirectory within the `--book-dir`.
*   `-a, --append`: Append to the output file(s) instead of overwriting. Reads the last ~4KB of the existing file to provide context for the first page being transcribed in the current run.
*   `--start-page INT`: Manually specify the first page number to transcribe (inclusive). Overrides LLM analysis for PDFs. Requires `--end-page`.
*   `--end-page INT`: Manually specify the last page number to transcribe (inclusive). Overrides LLM analysis for PDFs. Requires `--start-page`.
*   `--analysis-model MODEL_ID`: String identifier for the LLM used for PDF structure analysis (page range detection). Must match a value in CAMEL's `ModelType` enum. Default: `google/gemini-2.5-pro-exp-03-25:free` (Example, check defaults in script).
*   `--transcription-model MODEL_ID`: String identifier for the LLM used for image transcription. Must match a value in CAMEL's `ModelType` enum. Default: `google/gemini-2.0-flash-exp:free` (Example, check defaults in script).
*   `--log-level LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: `INFO`.

**Examples:**

1.  **Transcribe pre-extracted images to a single file:**
    ```bash
    python pdftotts/pdftotts.py --input-dir path/to/book_page_images -o my_book_transcription.txt
    ```

2.  **Transcribe images, append to existing file, use GPT-4o:**
    ```bash
    python pdftotts/pdftotts.py -i path/to/images -o output.txt -a --transcription-model gpt-4o
    ```

3.  **Process all PDFs in a directory, auto-detect page ranges, save to default `books_tts` dir:**
    ```bash
    python pdftotts/pdftotts.py --book-dir path/to/my_pdfs
    ```

4.  **Process PDFs, auto-detect ranges, save to specific dir, append, use Claude 3.5 Sonnet for analysis and GPT-4o Mini for transcription:**
    ```bash
    python pdftotts/pdftotts.py -b path/to/pdfs -o /data/transcriptions -a \
        --analysis-model claude-3-5-sonnet \
        --transcription-model gpt-4o-mini
    ```

5.  **Process PDFs, force pages 50-150 for all PDFs, save to specific dir:**
    ```bash
    python pdftotts/pdftotts.py -b path/to/pdfs --start-page 50 --end-page 150 -o /data/transcriptions_p50-150
    ```

6.  **Run with debug logging:**
    ```bash
    python pdftotts/pdftotts.py -b path/to/pdfs --log-level DEBUG
    ```

**Run `python pdftotts/pdftotts.py --help` to see all options and default model values.**

## Key Concepts & Workflow

1.  **PDF Conversion (if `--book-dir`):** `pdftoppm` converts the PDF into individual page images (e.g., PNG) in a temporary directory.
2.  **Image Sorting:** Images (either pre-existing or converted) are sorted based on the page number extracted from their filenames.
3.  **Page Range Filtering:** Pages outside the specified (manually or auto-detected) range are skipped.
4.  **Composite Image Creation:** For each target page within the range (that isn't the very first or last), `montage` creates a temporary composite image by placing the previous page, the current (target) page, and the next page side-by-side (`prev | current | next`). This provides visual context to the LLM.
5.  **Contextual Prompting:** A detailed prompt instructs the LLM to:
    *   Focus *only* on transcribing the **middle** image.
    *   Check if the middle page is ancillary content (TOC, Index, etc.) and output `SKIPPED` if so.
    *   Use the *textual transcription* from the end of the previously transcribed page (if available, passed via `previous_transcription_context_section`) to ensure a seamless start for the current page's transcription, avoiding repetition.
    *   Follow strict rules for omitting page numbers, references, etc., and describing visuals.
6.  **LLM Interaction:** The composite image and the prompt (including previous text context) are sent to the selected transcription LLM via the `camel-ai` `ImageAnalysisToolkit`.
7.  **Output Handling:**
    *   If `SKIPPED` is returned, it's logged, and no text is added to the output.
    *   Otherwise, the cleaned transcription is appended to the output file (or printed). The successful transcription becomes the context for the *next* page.
8.  **Cleanup:** Temporary composite images are deleted after use. Temporary directories created for PDF image conversion are removed after processing each PDF.

## Model Selection

*   You specify models using the `--analysis-model` and `--transcription-model` arguments.
*   The values provided **must** correspond to valid string identifiers defined in the `camel.types.ModelType` enum within the `camel-ai` library.
*   Run `python pdftotts/pdftotts.py --help` to see the default model values and a list of *some* available `ModelType` values (if the script can dynamically generate it).
*   Refer to the [CAMEL-AI documentation](https://www.camel-ai.org/) for a more comprehensive list of supported models and their identifiers.
*   Ensure you have the necessary API keys in your `.env` file for the models you select.
*   **Important:** Models suitable for PDF analysis (page range detection) might differ from those best for detailed page transcription. Both need to be multimodal (capable of vision).

## Troubleshooting & Notes

*   **`pdftoppm` or `montage` not found:** Ensure poppler-utils and ImageMagick are installed correctly and their executables are in your system's PATH.
*   **API Key Errors:** Double-check that your `.env` file is correctly named, in the right location (root of the project), and contains the correct keys for the selected models.
*   **Image Naming:** If using `--input-dir`, ensure your images follow the `prefix-NUMBER.extension` naming convention (e.g., `book_page-001.png`).
*   **Temporary Files:** If the script is interrupted forcefully (e.g., Ctrl+C during PDF conversion), temporary image directories (like `owl_pdf_*` or `owl_montage_*` in your system's temp location) might remain. You may need to clean these up manually.
*   **LLM Variability:** Transcription quality, adherence to instructions (like skipping pages or omitting references), and the accuracy of page range detection can vary depending on the LLM used, the quality of the input PDF/images, and the complexity of the page layout. You may need to experiment with different models.
*   **Cost:** Using commercial LLMs via API incurs costs. Be mindful of the number of pages you are processing, especially with high-resolution images or expensive models.
*   **Rate Limits:** You might encounter API rate limits if processing very large documents quickly. The script includes basic retries, but persistent issues may require adjusting usage or contacting the API provider.

## Contributing
