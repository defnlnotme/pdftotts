import os
import sys
from pathlib import Path

# --- Add Owl directory to Python path ---
script_dir = Path(__file__).resolve().parent
owl_dir = script_dir.parent
if str(owl_dir) not in sys.path:
    sys.path.insert(0, str(owl_dir))
    root_dir = owl_dir.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
# --- End Path Addition ---

from enum import Enum
import subprocess
import glob
import argparse
import logging
import re # Import regex module
from typing import List, Optional, Tuple, Type, Dict, Any, Union, Set
from time import sleep
import tempfile
import datetime # Import datetime for append message
import shutil # Import for directory removal
import json # Import JSON for parsing state
import concurrent.futures # <--- Added for threading

# --- ADDED: Import PyPDF2 ---
try:
    import PyPDF2
    from PyPDF2.errors import PdfReadError, DependencyError
except ImportError:
    print("Error: PyPDF2 library not found. Please install it: pip install pypdf2")
    sys.exit(1)

from dotenv import load_dotenv

# --- Import necessary CAMEL components ---
from camel.toolkits import ImageAnalysisToolkit
from camel.models import ModelFactory, BaseModelBackend
from camel.messages import BaseMessage
from camel.types import ModelPlatformType, ModelType, UnifiedModelType

# --- Model Definitions (Example Unified Models) ---
# Keep your custom models if needed, example remains
class mymodel(UnifiedModelType, Enum):
    OPENROUTER_QUASAR = "openrouter/quasar-alpha"
    OPENROUTER_SCOUT = "meta-llama/llama-4-scout:free"
    OPENROUTER_MAVERICK = "meta-llama/llama-4-maverick:free"
    OPENROUTER_GEMINI_PRO_2_5 = "google/gemini-2.5-pro-exp-03-25:free"
    OPENROUTER_GEMINI_FLASH_THINKING_2_0 = "google/gemini-2.0-flash-thinking-exp:free"
    OPENROUTER_GEMINI_FLASH_2_0 = "google/gemini-2.0-flash-exp:free"
    OPENROUTER_GEMINI_LEARN_LM = "google/learnlm-1.5-pro-experimental:free"
    OPENROUTER_QWEN_2_5_VL_3B = "qwen/qwen2.5-vl-3b-instruct:free"
    OPENROUTER_QWEN_2_5_VL_32B = "qwen/qwen2.5-vl-32b-instruct:free"

# --- Default Models & Failover Configuration ---
DEFAULT_TRANSCRIPTION_MODEL = "gemini-2.0-flash-lite-preview-02-05" # Using a Gemini model as default
DEFAULT_ANALYSIS_MODEL = "google/gemini-2.5-pro-exp-03-25:free" # Default analysis model
DEFAULT_BACKUP_MODEL = "google/gemini-2.0-flash-exp:free" # Default backup model
FAILURE_THRESHOLD = 3 # Number of consecutive failures to trigger switch

# --- Prompt Templates ---
# Transcription Prompt (Keep PROMPT_TEMPLATE as before)
PROMPT_TEMPLATE = """
# Prompt for TTS Book Page Transcription (Optimized for Continuity & Excluding References/Markers)

*Objective:** Transcribe the text from the **middle** book page image provided into clear, natural-sounding prose optimized for Text-to-Speech (TTS) playback. The key goals are a smooth narrative flow between pages, completely **excluding** any reference to external sources (books, papers) or internal structural pointers and markers (page numbers, figure/table/equation numbers, chapter/section numbers, list numbers, running headers/footers, etc.).

*Input:** Three images are provided side-by-side:
1.  Previous Page (Left Image) - **CONTEXT ONLY for visuals**
2.  **Target Page (Middle Image) - TRANSCRIBE THIS PAGE**
3.  Next Page (Right Image) - **CONTEXT ONLY for visuals**

*Context from Previous Page's Transcription:**
{previous_transcription_context_section}

*Critical Instructions:**

1.  **Focus EXCLUSIVELY on the Middle Page:** Do NOT transcribe any content from the left (previous) or right (next) page images. They are provided solely for contextual understanding when describing visuals *on the middle page*.
2.  **Ancillary Page Check (Middle Page Only):**
    * **CRITICAL Check:** If the **middle page** is clearly identifiable as **primarily** containing content such as a book cover, title page, copyright information, Preface, Foreword, Dedication, Table of Contents, List of Figures/Tables, Acknowledgments, Introduction (if clearly marked as introductory material *before* Chapter 1 or the main narrative), Notes section, Bibliography, References, Index, Appendix, Glossary, or other front/back matter **not part of the main narrative body/chapters**, **output the single word: `SKIPPED`**.
    * **Strict Adherence:** **Do NOT transcribe these ancillary page types.** If the page header, title, or main content block clearly indicates it's one of these types (e.g., starts with "Preface", "Table of Contents", "Index"), it **must** be skipped. Only transcribe pages that are part of the core chapters or narrative content.

*Transcription Guidelines (For Non-Skipped Middle Pages):**

The following guidelines ensure the transcription flows naturally as spoken prose, focusing on the core content while omitting disruptive elements like references and structural markers.*

1.  **Continuity from Previous Page:**
    * If `<previous_transcription>` context is provided, the transcription for the *middle page* **must seamlessly concatenate** with the very end of the provided context. This means:
        * **Start Precisely:** Begin the new transcription *exactly* where the previous context finished, even if it's mid-sentence or mid-paragraph.
        * **No Interruptions:** Add absolutely **no** extra pauses, breaks, paragraph starts, or introductory filler words between the context and the new text. The join must be direct and immediate.
        * **No Repetitions:** **Critically avoid repeating** any words, phrases, or sentences from the end of the previous context. **Specifically, if the text at the very top of the middle page begins with words identical to the end of the `previous_transcription` context, you MUST skip these duplicated words on the middle page** and start the transcription immediately following them.
        * **No Broken Flow:** Ensure sentences and paragraphs flow naturally across the page boundary without being artificially broken, cut off mid-word (unless hyphenated in source), or restarted incorrectly.
    The final combined text must read as a single, uninterrupted piece of prose, making the transition between the previous context and the new transcription **completely undetectable** in the final audio output.
    * If no `<previous_transcription>` context is provided (e.g., for the very first page being transcribed), start the transcription naturally based on the middle page content.
2.  **Verbatim Text & Order (Middle Page):** Transcribe the textual content **from the main body** of the middle page accurately, preserving the original order of information, *subject to the critical omissions below and the continuity guideline above*. Focus on the primary block(s) of text containing the narrative or core information, excluding isolated header/footer lines unless they introduce a new section directly within that main block.
3.  **Handling Visuals (Images, Tables, Charts, Diagrams, Equations on Middle Page):**
    * **Describe the Content/Purpose:** When a visual element appears in the text flow, describe **what it shows or illustrates** rather than just noting its presence. Focus on its main point, key takeaway, or the information it conveys relevant to the surrounding text.
    * **If a visual has NO caption:** Insert a concise, contextual description. Start with varied, natural phrasing (e.g., "Here is a diagram illustrating...", "The following table compares...", "A photograph shows...", "This chart displays trends in...", "An equation shows the relationship between..."). Use surrounding text (and context from previous/next pages if helpful) to inform the description.
    * **If a visual HAS a caption:** Integrate the caption smoothly into the narration *as part of the description*. Do not just state "Caption reads..."; weave it in naturally (e.g., "As shown in the illustration depicting regional deforestation rates...", "The table, captioned '[caption text]', highlights...", "An equation, described as '[caption text]', defines..."). Read the essential parts of the caption text itself as part of the description.
    * **Brevity for Visuals:** Keep descriptions concise. Summarize key trends, comparisons, the main subject of an image, or the purpose of an equation. Avoid listing exhaustive details or data points from tables/charts.
    * **Crucially: Omit Numbers/Labels in Description:** When describing the visual or integrating its caption, do **not** mention its original number or label (e.g., 'Figure 3.1', 'Table 2', 'Equation 5'). The description should stand alone, focusing solely on the visual's content and purpose.
4.  **Omissions (Middle Page Text):**
    * **a. Page Numbers:** Omit entirely (from headers, footers, or body text).
    * **b. Structural Numbers (Chapters, Sections, Headers, Lists):** Omit numbers that identify chapters, sections, sub-sections, or items in a numbered list, *when these numbers appear in titles, headers, or at the start of the list item itself*. (e.g., Read "Chapter: Title" not "Chapter 1: Title"; transcribe list items as continuous prose without the numbers).
    * **c. Figure/Table/Equation Labels:** Omit standalone labels like "Figure 3.1", "Table 2a", "Chart 5", "Eq. 5".
    * **d. Footnote/Endnote Markers:** Omit symbols or numbers indicating footnotes/endnotes (e.g., *, †, ¹, ²).
    * **e. URLs/Web Addresses:** Omit entirely.
    * **f. Citations, References, and Internal Pointers:** **MOST CRITICAL OMISSION:** **Absolutely and completely omit *any* text, marker, or phrase that refers to external sources, bibliographic entries, other authors' works, specific publications, OR internal pointers within the *same* book (page numbers, figure numbers, etc.).** Connect surrounding sentences smoothly. (Examples: `[15]`, `(Smith 2023)`, `"see Fig. 4"`, `"on page 123"` MUST BE OMITTED).
    * **g. Running Headers/Footers:** **CRITICAL OMISSION:** Omit text appearing consistently in the header (top margin) or footer (bottom margin) area, clearly separate from the main text block(s), especially if it repeats chapter/section titles. **Focus transcription solely on the primary narrative content block(s).** Do *not* repeat a chapter or section title if it appears as a running header/footer on a page *after* the section has already begun. However, if a title or heading clearly marks the *beginning* of a new chapter or major section *within the main text block* on *this specific page*, it **should** be included (unless it's just a number, per 4b).
5.  **Replacements & Formatting for TTS:**
    * **a. Symbols:** Replace with spoken equivalents ("%" -> "percent"). Describe math meaning, don't read complex symbols literally.
    * **b. Abbreviations:** Spell out common ones ("Figure" not "Fig."). Use common acronyms (NASA) if used as such.
    * **c. Punctuation:** Use effectively for natural pauses.
    * **d. Formatting Conversion:** Convert bullet points/numbered lists into flowing prose, **omitting the bullets/numbers**. Use connecting phrases if appropriate (e.g., "First,... Second,... Finally,...").
    * **e. Disruptive Characters:** Avoid #, *, excessive parentheses.

*Output Format:**

* If the page is skipped (per Instruction #2), output **only** the single word `SKIPPED`.
* If the page is transcribed, provide **only** the final, clean transcribed text.
* **Crucially: Do NOT wrap the output text in ``` ```, backticks, code blocks, quotation marks, or any other formatting markers.** The output should be plain text suitable for direct use.

---

*Now, please process the provided middle book page image based on these strictly revised guidelines, ensuring smooth continuity if previous transcription context is provided.**
"""

# --- ADDED: TOC Analysis Prompt ---
PROMPT_TOC_ANALYSIS = """
# Prompt for PDF Table of Contents (TOC) Analysis

**Objective:** Analyze the provided Table of Contents (TOC) text extracted from a PDF book and determine the page range corresponding to the **main narrative content** (i.e., the core chapters or primary sections). Exclude pages belonging to typical front matter and back matter.

**Input:** The following text represents the Table of Contents extracted from the PDF:
```text
{toc_text}
```

**Definitions:**

*   **Main Narrative Content:** The primary body of the book, usually organized into chapters (e.g., Chapter 1, Chapter 2...) or major titled sections that constitute the core story, argument, or information being presented.
*   **Front Matter (Exclude):** Pages typically appearing *before* the main narrative. Examples include: Book Cover, Title Page, Copyright Page, Dedication, Epigraph, Table of Contents (the TOC itself), List of Figures, List of Tables, Foreword, Preface, Acknowledgments, Introduction (if it serves as preliminary material *before* the first main chapter/section).
*   **Back Matter (Exclude):** Pages typically appearing *after* the main narrative. Examples include: Appendix/Appendices, Notes, Glossary, Bibliography, References, Index, Colophon, About the Author.

**Instructions:**

1.  **Analyze the TOC:** Carefully examine the structure and titles in the provided `{toc_text}`. Identify the entries that most likely mark the beginning and end of the main narrative content, according to the definitions above.
2.  **Identify Start Page:** Determine the page number associated with the *first* entry you identify as part of the main narrative content (e.g., the start of "Chapter 1", "Part I", or the first main section).
3.  **Identify End Page:** Determine the page number associated with the *last* entry you identify as part of the main narrative content (e.g., the end of the last chapter or main section, *before* any back matter like Appendices or Index begins). If the end page isn't explicitly listed for the last main content item, use the page number of the *next* item (likely the start of back matter) minus one, or make a reasonable estimate based on surrounding page numbers.
4.  **Handle Ambiguity:** If the TOC structure is unclear, or if it's difficult to definitively separate main content from front/back matter, make your best reasonable judgment. If you cannot confidently determine a range, state this in your reasoning.
5.  **Output Format:** Respond **only** with a valid JSON object containing the determined start and end page numbers. Use `null` if a value cannot be determined.

    ```json
    {{
      "reasoning": "Brief explanation of how the start and end pages were identified (e.g., 'Main content starts with Chapter 1 on page 5 and ends with Chapter 10 on page 250, before the Appendix on page 251.'). If unable to determine, explain why.",
      "start_page": <start_page_number_or_null>,
      "end_page": <end_page_number_or_null>
    }}
    ```

**Example Output 1 (Successful):**
```json
{{
  "reasoning": "Identified 'Chapter 1: The Beginning' starting on page 12 as the main content start. The last main section 'Chapter 20: Conclusions' appears to end before the 'Appendix A' which starts on page 345. Estimated end page as 344.",
  "start_page": 12,
  "end_page": 344
}}
```

**Example Output 2 (Cannot Determine):**
```json
{{
  "reasoning": "The provided TOC is very sparse and lacks clear chapter/section markers or page numbers. Cannot reliably distinguish main content from other sections.",
  "start_page": null,
  "end_page": null
}}
```

**Now, analyze the provided TOC text and return the JSON output.**
"""

# --- Logging Setup ---
logger = logging.getLogger(__name__)
# Basic config will be set later based on args

# Load environment variables from .env file
load_dotenv(".env", verbose=True)

# --- Constants ---
RESUME_STATE_FILENAME = "_pdftotts_resume_state.json"
END_MARKER = "The End."
TEMP_IMAGE_SUBDIR = ".pdftotts_images" # Used ONLY in --book-dir mode
SYSTEM_TEMP_DIR_PATTERN = "/tmp/image_*.png" # For Camel toolkit temp files

# --- Helper Functions ---
def extract_page_number(filename: Path) -> Optional[int]:
    """Extracts the page number from filenames like 'prefix-digits.ext'. (Used only for --input-dir mode)"""
    match = re.search(r"-(\d+)\.\w+$", filename.name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            logger.warning(f"Could not convert extracted number to int in filename: {filename.name}")
            return None
    else:
        # Try alternative: only digits before extension
        match_alt = re.search(r"(\d+)\.\w+$", filename.name)
        if match_alt:
             try:
                 # Less stringent check for image mode
                 logger.debug(f"Found digits {match_alt.group(1)} in {filename.name}. Assuming it's the page number.")
                 return int(match_alt.group(1))
             except ValueError:
                 logger.warning(f"Could not convert extracted number to int in filename: {filename.name}")
                 return None
        else:
            logger.warning(f"Could not extract page number (format '-digits.' or 'digits.') from filename: {filename.name}")
            return None

# --- MODIFIED: Helper Function to generate composite image (can be run in thread) ---
def create_composite_image_task(image_paths: List[Path], output_dir: Path, base_name: str) -> Optional[Path]:
    """Creates a temporary composite image from exactly 3 images using ImageMagick's montage tool.
    This is the function to be executed in a background thread."""
    if not image_paths or len(image_paths) != 3:
        logger.warning(f"Cannot create composite task: Need exactly 3 image paths, got {len(image_paths)} for base {base_name}.")
        return None

    # Ensure all input paths actually exist before attempting montage
    missing_files = [p for p in image_paths if not p.exists()]
    if missing_files:
        logger.error(f"Cannot create composite task: Missing input image(s): {[str(p.name) for p in missing_files]} for base {base_name}.")
        return None

    image_paths_str = ' '.join(f'"{path}"' for path in image_paths)
    # Composites will go directly into the specified output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{base_name}_composite.png"

    # If composite already exists, return it (might happen on retry/overlap)
    if output_path.exists():
        logger.debug(f"Composite image {output_path.name} already exists. Returning path.")
        return output_path

    command = f"montage -mode concatenate -tile x1 {image_paths_str} \"{output_path}\""

    try:
        logger.debug(f"Running montage command in background thread: {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, encoding='utf-8')
        if output_path.exists():
            logger.info(f"Background task created composite image: {output_path.name}")
            return output_path
        else:
             logger.error(f"Montage command ran (bg) but output file not found: {output_path.name}")
             logger.error(f"Montage (bg) stdout: {result.stdout}")
             logger.error(f"Montage (bg) stderr: {result.stderr}")
             return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating composite image (bg) with command '{command}': {e}")
        logger.error(f"Montage (bg) stdout: {e.stdout}")
        logger.error(f"Montage (bg) stderr: {e.stderr}")
        # Raise the error so the Future captures it
        raise e
    except FileNotFoundError:
        logger.error("Error: 'montage' command not found. Please ensure ImageMagick is installed and in your system's PATH.")
        raise RuntimeError("ImageMagick 'montage' command not found.")
    except Exception as e:
        logger.error(f"Unexpected error creating composite image {base_name} (bg): {e}", exc_info=True)
        # Raise the error so the Future captures it
        raise e

def transcribe_image(
    active_toolkit: ImageAnalysisToolkit,
    image_path: Path,
    original_middle_page_name: str,
    previous_transcription: Optional[str] = None,
    prompt_template: str = PROMPT_TEMPLATE
) -> Tuple[Optional[str], bool]:
    """
    Uses the provided ImageAnalysisToolkit instance to transcribe text from a composite image.
    Handles internal retries but signals overall success/failure for external failover logic.

    Returns:
        Tuple[Optional[str], bool]: (Transcription text or None, success_flag)
                                    success_flag is True if API call worked AND returned valid content.
                                                     False otherwise (API error, timeout, empty/invalid content).
    """
    logger.info(f"Transcribing composite image representing middle page: {original_middle_page_name} using {active_toolkit.model.model_type.value}")

    if previous_transcription:
        max_context_chars = 2000
        truncated_context = previous_transcription[-max_context_chars:]
        context_section = f"<previous_transcription>\n{truncated_context}\n</previous_transcription>"
        logger.debug(f"Providing last ~{len(truncated_context)} chars of previous transcription as context.")
    else:
        context_section = "<previous_transcription>No previous transcription context provided.</previous_transcription>"
        logger.debug("No previous transcription context provided.")

    final_prompt = prompt_template.format(previous_transcription_context_section=context_section)

    transcription = None
    try:
        retry_delay = 2
        max_retries = 3 # Internal retries per toolkit attempt
        retries = 0

        while retries < max_retries:
            call_success = False
            try:
                # Call the model
                transcription = active_toolkit.ask_question_about_image(
                    image_path=str(image_path),
                    question=final_prompt
                )
                call_success = True # If no exception, API call itself succeeded

            except Exception as e_inner:
                 logger.warning(f"Transcription attempt {retries + 1}/{max_retries} API call failed for {original_middle_page_name} (using {active_toolkit.model.model_type.value}): {e_inner}. Retrying in {retry_delay}s...")
                 # call_success remains False

            # Process the result if the API call worked
            if call_success:
                if transcription:
                    # Clean up potential markdown/code blocks
                    transcription = transcription.strip()
                    transcription = re.sub(r'^```(text|markdown)?\s*', '', transcription, flags=re.IGNORECASE | re.MULTILINE)
                    transcription = re.sub(r'\s*```$', '', transcription, flags=re.IGNORECASE | re.MULTILINE)
                    transcription = transcription.strip('"\'') # Remove leading/trailing quotes

                    if transcription and "Analysis failed" not in transcription and len(transcription) > 5:
                        logger.info(f"Transcription successful for {original_middle_page_name} (attempt {retries+1} with {active_toolkit.model.model_type.value}). Length: {len(transcription)}")
                        return transcription, True # Return result and signal external success
                    else:
                        logger.warning(f"Transcription attempt {retries + 1}/{max_retries} produced short/invalid output for {original_middle_page_name} (using {active_toolkit.model.model_type.value}). Content: '{transcription[:50]}...'. Treating as failure for this attempt.")
                        transcription = None # Reset for retry
                        # Do NOT proceed to retry wait/increment here; let the outer loop handle it
                else: # API call succeeded but returned empty transcription
                    logger.warning(f"Received empty transcription on attempt {retries + 1}/{max_retries} for {original_middle_page_name} (using {active_toolkit.model.model_type.value}). Treating as failure for this attempt.")
                    # Do NOT proceed to retry wait/increment here; let the outer loop handle it

            # Retry logic (only if needed based on call_success or invalid content)
            if not call_success or transcription is None:
                sleep(retry_delay)
                retry_delay *= 2
                retries += 1
                continue # Go to next retry
            else: # Should only happen if content was valid
                break # Exit the loop, transcription is valid

        # If loop finished without returning valid data
        if transcription is None:
            logger.error(f"Transcription FAILED after {max_retries} internal attempts for {original_middle_page_name} (using {active_toolkit.model.model_type.value}).")
            return None, False # Return failure signal

        # This path should ideally not be reached if logic is correct, but as safety:
        return transcription, True # Assume success if we broke loop with valid transcription

    except Exception as e:
        logger.error(f"Unhandled error during transcription retry loop for {image_path.name} ({original_middle_page_name}): {e}", exc_info=True)
        return None, False # Return failure signal
    finally:
        cleanup_system_temp_files() # Cleanup Camel's temp files


# --- NEW: Function to convert a single PDF page ---
def convert_pdf_page_to_image(
    pdf_path: Path,
    page_number: int,
    output_dir: Path,
    image_format: str = 'png',
    dpi: int = 300
) -> Optional[Path]:
    """
    Converts a single page of a PDF file to an image using pdftoppm.

    Args:
        pdf_path: Path to the input PDF file.
        page_number: The 1-based page number to convert.
        output_dir: The directory to save the output image.
        image_format: Output image format (e.g., 'png', 'jpeg').
        dpi: Resolution for the output image.

    Returns:
        The Path to the created image file, or None if conversion failed.
    """
    if not pdf_path.is_file():
        logger.error(f"PDF file not found for page conversion: {pdf_path}")
        return None
    if page_number < 1:
        logger.error(f"Invalid page number {page_number}. Must be 1 or greater.")
        return None

    if shutil.which("pdftoppm") is None:
        logger.critical("Critical Error: 'pdftoppm' command not found. Please ensure poppler-utils is installed.")
        raise RuntimeError("pdftoppm not found") # Raise error to stop processing

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine padding based on total pages
    total_pages = get_pdf_page_count(pdf_path) # Use helper function
    if total_pages is None:
        logger.warning(f"Could not get total page count for {pdf_path.name}, using default padding.")
        padding_width = 3
    else:
        padding_width = len(str(total_pages))

    # Construct expected filename based on pdftoppm behavior
    output_prefix_path_part = output_dir / pdf_path.stem
    expected_output_filename = f"{pdf_path.stem}-{page_number:0{padding_width}d}.{image_format}"
    expected_output_path = output_dir / expected_output_filename

    # If the file already exists, assume it's correct and return path
    if expected_output_path.exists():
        logger.debug(f"Image for page {page_number} already exists: {expected_output_path.name}. Skipping conversion.")
        return expected_output_path

    command = [
        "pdftoppm",
        f"-{image_format}",
        "-r", str(dpi),
        "-f", str(page_number), # First page to process
        "-l", str(page_number), # Last page to process
        str(pdf_path),
        str(output_prefix_path_part) # Base path/prefix for output
    ]

    try:
        logger.debug(f"Converting page {page_number} of '{pdf_path.name}' using command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        logger.debug(f"pdftoppm stdout for page {page_number}: {result.stdout}")
        if result.stderr:
             logger.debug(f"pdftoppm stderr for page {page_number}: {result.stderr}")

        # Verify the specific output file was created
        if expected_output_path.exists():
            logger.info(f"Successfully converted page {page_number} to: {expected_output_path.name}")
            return expected_output_path
        else:
             logger.error(f"pdftoppm command ran for page {page_number} but expected output file not found: {expected_output_path.name}")
             # Log directory contents for debugging
             try:
                  files_in_dir = list(output_dir.glob(f'{pdf_path.stem}-*.{image_format}')) # Be more specific
                  logger.error(f"PDF-related files currently in {output_dir}: {[f.name for f in files_in_dir]}")
                  # Attempt to find a file with the correct number but maybe different padding
                  alternative_match = list(output_dir.glob(f"{pdf_path.stem}-{page_number}.{image_format}")) # Check without strict padding
                  if alternative_match:
                      logger.warning(f"Found match with potentially different padding: {alternative_match[0].name}. Returning this.")
                      return alternative_match[0]
                  else: logger.error("No alternative file match found either.")
             except Exception as e_list: logger.error(f"Could not list directory contents for debugging: {e_list}")
             return None # Indicate failure

    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting page {page_number} of '{pdf_path.name}': {e}")
        logger.error(f"pdftoppm stdout: {e.stdout}")
        logger.error(f"pdftoppm stderr: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during PDF page {page_number} conversion for {pdf_path.name}: {e}", exc_info=True)
        return None

# --- State Management Functions ---
# load_resume_state and save_resume_state remain unchanged

def load_resume_state(state_file_path: Path) -> Dict[str, Any]:
    """Loads the resume state from a JSON file."""
    if not state_file_path.exists():
        return {}
    try:
        with open(state_file_path, 'r', encoding='utf-8') as f:
            raw_state_data = json.load(f)
        logger.info(f"Loaded resume state from {state_file_path}")

        # Check format and convert if necessary (keeping previous logic)
        converted_state_data = {}
        needs_resave = False
        for pdf_filename, data in raw_state_data.items():
            if isinstance(data, int): # Old format
                logger.warning(f"Detected old state format for '{pdf_filename}'. Converting.")
                converted_state_data[pdf_filename] = {
                    "last_processed": data,
                    "detected_start": None, "detected_end": None
                }
                needs_resave = True
            elif isinstance(data, dict): # Assume new format
                if "last_processed" not in data: data["last_processed"] = None
                converted_state_data[pdf_filename] = data
            else: logger.warning(f"Skipping unrecognized state entry for '{pdf_filename}'. Data type: {type(data)}")

        if needs_resave:
            logger.info("Resaving state file in new format after conversion.")
            save_resume_state(state_file_path, converted_state_data)

        return converted_state_data
    except json.JSONDecodeError: logger.warning(f"Error decoding JSON state: {state_file_path}. Starting fresh."); return {}
    except IOError as e: logger.error(f"Error reading state file {state_file_path}: {e}. Starting fresh."); return {}
    except Exception as e: logger.error(f"Unexpected error loading state {state_file_path}: {e}. Starting fresh.", exc_info=True); return {}

def save_resume_state(state_file_path: Path, state_data: Dict[str, Any]):
    """Saves the resume state (new format) to a JSON file."""
    try:
        state_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Filter out entries that are completely empty or only have nulls
        cleaned_state = {}
        for pdf, data in state_data.items():
            if isinstance(data, dict) and any(v is not None for v in data.values()):
                 cleaned_state[pdf] = data

        if not cleaned_state:
            logger.debug(f"State data is empty, removing state file: {state_file_path}")
            if state_file_path.exists(): state_file_path.unlink(missing_ok=True)
            return

        with open(state_file_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_state, f, indent=4)
        logger.debug(f"Saved resume state to {state_file_path}")
    except IOError as e: logger.error(f"Error writing state file {state_file_path}: {e}")
    except Exception as e: logger.error(f"Unexpected error saving state {state_file_path}: {e}", exc_info=True)


# --- MODIFIED: PDF Page Processing Function handles failover state ---
def process_pdf_pages(
    pdf_path: Path,
    num_pdf_pages: int,
    pdf_image_dir: Path,
    output_path: Optional[Path],
    append_mode: bool,
    start_page: Optional[int],
    end_page: Optional[int],
    failover_state: Dict[str, Any], # Pass mutable state dictionary
    prompt_template: str,
    initial_context: Optional[str],
    pdf_filename: str,
    state_file_path: Optional[Path],
    state_data: Dict[str, Any],
    image_format: str = 'png',
    dpi: int = 300,
    executor: concurrent.futures.ThreadPoolExecutor = None
) -> Tuple[bool, Optional[str]]:
    """
    Processes pages of a PDF with async composite generation and uses failover models.
    Updates the passed `failover_state` dictionary directly.
    """
    logger.info(f"Processing PDF: {pdf_filename} [Effective Pages {start_page or 1} to {end_page or num_pdf_pages}] with Async Composite")

    # Get toolkit instances from failover state for easier access
    t_primary_tk = failover_state['t_primary_tk']
    t_backup_tk = failover_state['t_backup_tk']
    # t_active_tk and t_failures will be accessed directly via failover_state dict

    files_processed_count = 0
    last_successful_transcription = initial_context
    processing_successful = True
    current_images_on_disk: Set[Path] = set()
    if not executor:
         logger.critical("ThreadPoolExecutor must be provided for process_pdf_pages.")
         return False, initial_context

    temp_composite_dir = pdf_image_dir / ".composites"

    def get_image_path(page_num: int) -> Path:
        # Reusing get_pdf_page_count to determine padding needed
        total_pages = get_pdf_page_count(pdf_path)
        padding_width = len(str(total_pages)) if total_pages else 3
        return pdf_image_dir / f"{pdf_path.stem}-{page_num:0{padding_width}d}.{image_format}"

    def ensure_page_image_exists(page_num: int) -> Optional[Path]:
        if page_num < 1 or page_num > num_pdf_pages: return None
        img_path = get_image_path(page_num)
        if img_path not in current_images_on_disk and not img_path.exists():
            logger.info(f"Generating missing image for page {page_num}...")
            generated_path = convert_pdf_page_to_image(pdf_path, page_num, pdf_image_dir, image_format, dpi)
            if generated_path: current_images_on_disk.add(generated_path); return generated_path
            else: logger.error(f"Failed to generate required image for page {page_num}."); return None
        elif img_path.exists():
             if img_path not in current_images_on_disk: current_images_on_disk.add(img_path)
             return img_path
        else: logger.error(f"Logic error: Image path {img_path.name} not in cache set and does not exist."); return None

    def delete_page_image(page_num: int):
        if page_num < 1: return
        img_path = get_image_path(page_num)
        if img_path in current_images_on_disk:
            try: img_path.unlink(); logger.debug(f"Deleted page image: {img_path.name}"); current_images_on_disk.remove(img_path)
            except OSError as e_del: logger.warning(f"Could not delete page image {img_path.name}: {e_del}")
        elif img_path.exists():
             try: img_path.unlink(); logger.debug(f"Deleted stray page image: {img_path.name}")
             except OSError as e_del_stray: logger.warning(f"Could not delete stray page image {img_path.name}: {e_del_stray}")

    try:
        pdf_image_dir.mkdir(parents=True, exist_ok=True)
        temp_composite_dir.mkdir(exist_ok=True)
        logger.info(f"Using temporary image directory: {pdf_image_dir}")
        logger.info(f"Using temporary composite directory: {temp_composite_dir}")
    except OSError as e_comp_dir:
        logger.critical(f"Could not create temporary dirs under {pdf_image_dir}: {e_comp_dir}. Cannot process PDF.")
        return False, initial_context

    if output_path and not append_mode:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f_init: f_init.truncate(0)
            logger.info(f"Output file {output_path} truncated (or created).")
        except IOError as e: logger.error(f"Error preparing output file {output_path}: {e}"); return False, initial_context

    loop_start = start_page if start_page is not None else 1
    loop_end = end_page if end_page is not None else num_pdf_pages
    logger.info(f"Looping through pages {loop_start} to {loop_end} (inclusive) for {pdf_filename}.")
    pdf_state = state_data.get(pdf_filename, {})

    current_composite_future: Optional[concurrent.futures.Future] = None
    next_composite_future: Optional[concurrent.futures.Future] = None

    # Pre-submission logic (remains the same)
    first_transcribe_page = max(loop_start, 2 if loop_start < 2 else loop_start) # Ensure we don't try to get page 0
    if first_transcribe_page <= loop_end:
         logger.info(f"Pre-submitting composite generation task for first transcribable page: {first_transcribe_page}")
         p_prev_first, p_curr_first, p_next_first = first_transcribe_page - 1, first_transcribe_page, first_transcribe_page + 1
         if p_prev_first >= 1 and p_next_first <= num_pdf_pages:
             paths_first = [ensure_page_image_exists(p) for p in [p_prev_first, p_curr_first, p_next_first]]
             if all(p is not None for p in paths_first):
                 base_name_first = f"page_{p_curr_first:04d}"
                 current_composite_future = executor.submit(create_composite_image_task, paths_first, temp_composite_dir, base_name_first)
                 logger.debug(f"Submitted initial composite task for page {p_curr_first}")
             else: logger.error(f"Could not generate necessary images for the first composite ({p_curr_first}). Cannot start pre-processing."); processing_successful = False
         else: logger.info(f"First transcribable page {first_transcribe_page} too close to edge {1}-{num_pdf_pages}. Will handle in loop.")
    else: logger.info(f"No pages within the specified range {loop_start}-{loop_end} to process."); processing_successful = True

    try:
        for page_num in range(loop_start, loop_end + 1):
            if not processing_successful:
                logger.warning(f"Terminating loop for {pdf_filename} due to prior failure.")
                break # Exit loop if a fatal error occurred

            logger.info(f"--- Current target page: {page_num}/{loop_end} ({pdf_filename}) ---")
            pdf_state = state_data.get(pdf_filename, {}) # Refresh state in case it was modified

            # Submit next composite task in background (same logic)
            next_composite_page_num = page_num + 1
            p_curr_next, p_next_next, p_after_next = next_composite_page_num - 1, next_composite_page_num, next_composite_page_num + 1
            if next_composite_page_num <= loop_end and p_after_next <= num_pdf_pages:
                paths_next = [ensure_page_image_exists(p) for p in [p_curr_next, p_next_next, p_after_next]]
                if all(p is not None for p in paths_next):
                    base_name_next = f"page_{next_composite_page_num:04d}"
                    next_composite_future = executor.submit(create_composite_image_task, paths_next, temp_composite_dir, base_name_next)
                    logger.info(f"Submitted composite task for next page {next_composite_page_num}")
                else: logger.warning(f"Could not generate images for next composite ({next_composite_page_num})."); next_composite_future = None
            else: logger.debug(f"Not submitting next composite task: page {next_composite_page_num} is near end or out of bounds."); next_composite_future = None

            p_prev = page_num - 1
            p_curr = page_num
            p_next = page_num + 1

            # --- Wait + Transcription + Failover Logic ---
            # Only transcribe if we can form the 3-page window
            if p_prev < 1 or p_next > num_pdf_pages:
                logger.info(f"Skipping transcription for edge page {page_num}: Cannot form 3-page window.")
                if state_file_path:
                     pdf_state["last_processed"] = page_num
                     state_data[pdf_filename] = pdf_state
                     save_resume_state(state_file_path, state_data)
                # Move to next composite immediately if available
                current_composite_future = next_composite_future
                next_composite_future = None
                delete_page_image(page_num - 2) # Cleanup previous page image
                continue # Skip to next page_num in loop

            # We can form a window, wait for the *current* composite future
            if current_composite_future:
                logger.debug(f"Waiting for composite result for page {page_num}...")
                temp_composite_img_path = None
                composite_exception = None
                try:
                    temp_composite_img_path = current_composite_future.result(timeout=600) # Wait up to 10 mins
                except concurrent.futures.TimeoutError: logger.error(f"Timeout waiting for composite generation for page {page_num}."); composite_exception = TimeoutError("Composite generation timed out")
                except Exception as e_future: logger.error(f"Composite generation failed for page {page_num}: {e_future}"); composite_exception = e_future

                if temp_composite_img_path and temp_composite_img_path.exists():
                    logger.info(f"Composite ready for page {page_num}: {temp_composite_img_path.name}")
                    files_processed_count += 1
                    middle_page_name = get_image_path(p_curr).name
                    transcription = None
                    call_success = False

                    try:
                        # --- Transcribe using the ACTIVE toolkit ---
                        # transcribe_image internally handles retries for the ACTIVE toolkit, but returns success/fail status
                        transcription, call_success = transcribe_image(
                            active_toolkit=failover_state['t_active_tk'], # Use active from state
                            image_path=temp_composite_img_path,
                            original_middle_page_name=middle_page_name,
                            previous_transcription=last_successful_transcription,
                            prompt_template=prompt_template
                        )

                        # --- Handle Transcription Result and Failover ---
                        if call_success:
                            # SUCCESS: Reset failure counter
                            failover_state['t_failures'] = 0
                            logger.debug(f"Transcription API call successful for page {page_num}, reset failure counter to 0.")

                            # Process the transcription (skipped or content)
                            transcription = transcription.strip() # transcribe_image should have cleaned, but be safe
                            if transcription.upper() == "SKIPPED":
                                logger.info(f"SKIPPED page {page_num} ({middle_page_name}) due to ancillary content.")
                                # Update state, no context change, NOT a failure
                                if state_file_path:
                                    pdf_state["last_processed"] = page_num
                                    state_data[pdf_filename] = pdf_state
                                    save_resume_state(state_file_path, state_data)
                            else:
                                # Write to output
                                if output_path:
                                    try:
                                        with open(output_path, 'a', encoding='utf-8') as f: f.write(transcription + "\n\n")
                                        logger.info(f"Appended transcription for page {page_num} to {output_path.name}")
                                    except IOError as e: logger.error(f"Error writing to {output_path}: {e}"); processing_successful = False; break # Break if write fails
                                else: print(f"--- Page {page_num} ({middle_page_name}) ---\n{transcription}\n--- End ---\n")

                                last_successful_transcription = transcription # Update context FOR NEXT PAGE
                                # Update resume state ONLY after successful processing & write
                                if state_file_path:
                                    pdf_state["last_processed"] = page_num
                                    state_data[pdf_filename] = pdf_state
                                    save_resume_state(state_file_path, state_data)
                        else: # FAILURE (API error, timeout, or invalid content from transcribe_image)
                            logger.warning(f"--- Transcription FAILED for page {page_num} ({middle_page_name}) using {failover_state['t_active_tk'].model.model_type.value} ---")
                            failover_state['t_failures'] += 1
                            logger.info(f"Transcription failure count incremented to: {failover_state['t_failures']}")
                            # Do NOT update last_processed or context on failure
                            processing_successful = False # Signal failure for this book

                            # Check if failover threshold reached
                            if failover_state['t_failures'] >= FAILURE_THRESHOLD:
                                current_active_model = failover_state['t_active_tk'].model.model_type.value
                                if failover_state['t_active_tk'] == t_primary_tk and t_backup_tk:
                                    failover_state['t_active_tk'] = t_backup_tk
                                    logger.warning(f"Reached {FAILURE_THRESHOLD} consecutive transcription failures on PRIMARY ({current_active_model}). Switching to BACKUP model: {t_backup_tk.model.model_type.value}")
                                elif failover_state['t_active_tk'] == t_backup_tk and t_primary_tk:
                                     failover_state['t_active_tk'] = t_primary_tk
                                     logger.warning(f"Reached {FAILURE_THRESHOLD} consecutive transcription failures on BACKUP ({current_active_model}). Switching back to PRIMARY model: {t_primary_tk.model.model_type.value}")
                                else:
                                     logger.error(f"Reached {FAILURE_THRESHOLD} failures, but cannot switch from {current_active_model} - primary or backup missing or same?")

                                if t_primary_tk != t_backup_tk: # Only reset if models actually switched
                                     failover_state['t_failures'] = 0 # Reset counter after switching
                                     logger.info("Transcription failure counter reset after model switch.")
                                else:
                                    logger.warning("Primary and backup models are the same, fail counter not reset after threshold.")
                            # Break the loop for this PDF on failure
                            break

                    except Exception as e_transcribe:
                         logger.error(f"Unhandled error during transcription block for page {page_num} ({middle_page_name}): {e_transcribe}", exc_info=True)
                         processing_successful = False
                         # Count exception as failure for failover
                         failover_state['t_failures'] += 1
                         logger.info(f"Transcription failure count incremented to: {failover_state['t_failures']} due to exception.")
                         # Repeat failover check logic (consider refactoring this)
                         if failover_state['t_failures'] >= FAILURE_THRESHOLD:
                              current_active_model = failover_state['t_active_tk'].model.model_type.value
                              if failover_state['t_active_tk'] == t_primary_tk and t_backup_tk: failover_state['t_active_tk'] = t_backup_tk; logger.warning(f"Reached {FAILURE_THRESHOLD} failures (exception). Switching to BACKUP: {t_backup_tk.model.model_type.value}")
                              elif failover_state['t_active_tk'] == t_backup_tk and t_primary_tk: failover_state['t_active_tk'] = t_primary_tk; logger.warning(f"Reached {FAILURE_THRESHOLD} failures (exception). Switching back to PRIMARY: {t_primary_tk.model.model_type.value}")
                              else: logger.error(f"Reached {FAILURE_THRESHOLD} failures (exception), but cannot switch from {current_active_model}.")
                              if t_primary_tk != t_backup_tk: failover_state['t_failures'] = 0; logger.info("Transcription failure counter reset after model switch.")
                              else: logger.warning("Primary and backup models are the same, fail counter not reset after threshold.")
                         break # Break loop on unhandled error

                    finally:
                         if temp_composite_img_path and temp_composite_img_path.exists():
                             try: temp_composite_img_path.unlink(); logger.debug(f"Deleted used composite: {temp_composite_img_path.name}")
                             except OSError as e_del_comp: logger.warning(f"Could not delete used composite {temp_composite_img_path.name}: {e_del_comp}")

                elif composite_exception:
                     logger.error(f"Skipping transcription for page {page_num} because composite generation failed: {composite_exception}")
                     processing_successful = False; break # Fail this book
                else:
                     logger.error(f"Composite future completed for page {page_num}, but path is invalid or file missing. Skipping.")
                     processing_successful = False; break # Fail this book
            else:
                 logger.warning(f"No current composite future available to wait for page {page_num}. Check pre-submission logic. Skipping transcription attempt.")
                 # Don't necessarily fail the whole book, but log warning

            # --- Advance to next page's composite future ---
            current_composite_future = next_composite_future
            next_composite_future = None # Clear it so we know to submit next iteration
            delete_page_image(page_num - 2) # Cleanup image from two pages ago

        # End of page loop
        if not processing_successful:
            logger.error(f"Loop terminated prematurely for PDF {pdf_filename} due to failures.")

    finally:
        logger.debug(f"Performing final image cleanup for {pdf_filename}...")
        images_to_delete = list(current_images_on_disk)
        for img_path in images_to_delete:
             try: delete_page_image(int(re.search(r"-(\d+)\.[^.]+$", img_path.name).group(1)))
             except (AttributeError, ValueError, TypeError) as e_parse: logger.warning(f"Could not parse page num from {img_path.name} for final cleanup: {e_parse}")
             except Exception as e_final_del: logger.warning(f"Error in final cleanup helper for {img_path.name}: {e_final_del}")

        if temp_composite_dir.exists():
            try: shutil.rmtree(temp_composite_dir); logger.debug(f"Cleaned up composites directory: {temp_composite_dir}")
            except OSError as e_final_clean: logger.warning(f"Could not clean up composites directory {temp_composite_dir}: {e_final_clean}")

    logger.info(f"Finished loop for PDF {pdf_filename}. Processed {files_processed_count} pages within effective range.")
    return processing_successful, last_successful_transcription


# --- Model Parsing and Platform Inference ---
# parse_model_string_to_enum remains mostly unchanged
def parse_model_string_to_enum(model_string: str, model_purpose: str) -> Optional[Union[ModelType, str]]:
    """Parses a model string to a base ModelType enum or a recognized unified string."""
    logger.debug(f"Attempting to parse model string for {model_purpose}: '{model_string}'")
    if not isinstance(model_string, str): logger.error(f"Invalid model string type ({type(model_string)})."); return None

    # 1. Try base ModelType enum
    try:
        model_enum = ModelType(model_string); logger.info(f"Parsed '{model_string}' as base ModelType: {model_enum}"); return model_enum
    except ValueError: logger.debug(f"'{model_string}' not a base ModelType value.")

    # 2. Try custom 'mymodel' enum
    try:
        custom_enum_member = mymodel(model_string); logger.info(f"Parsed '{model_string}' as custom 'mymodel'. Returning value: '{custom_enum_member.value}'"); return custom_enum_member.value
    except ValueError: logger.debug(f"'{model_string}' not a custom 'mymodel' value.")

    # 3. Check for unified 'provider/model' format (if not already matched by custom enum)
    if "/" in model_string:
        logger.info(f"'{model_string}' looks like a generic unified model string."); return model_string

    # 4. Fallback
    logger.error(f"Invalid or unrecognized model string for {model_purpose}: '{model_string}'.")
    return None

# get_platform_from_model_type remains mostly unchanged
def get_platform_from_model_type(model_type_repr: Union[ModelType, str]) -> Optional[ModelPlatformType]:
    """Infers the ModelPlatformType from a ModelType enum or model string."""
    unified_platform_map = {
        "openrouter": ModelPlatformType.OPENROUTER, "meta-llama": ModelPlatformType.OPENROUTER,
        "google": ModelPlatformType.OPENROUTER, # Assume OpenRouter for google/ prefix
        "qwen": ModelPlatformType.QWEN, "anthropic": ModelPlatformType.ANTHROPIC,
        "deepseek": ModelPlatformType.DEEPSEEK, "groq": ModelPlatformType.GROQ,
    }
    original_input = model_type_repr

    if isinstance(model_type_repr, str):
        model_string_cleaned = model_type_repr.split(':')[0]
        try: enum_match = ModelType(model_string_cleaned); model_type_repr = enum_match # Try convert to base enum
        except ValueError:
            if "/" in model_string_cleaned:
                provider = model_string_cleaned.split('/')[0].lower()
                platform = unified_platform_map.get(provider)
                if platform: logger.debug(f"Inferred platform '{platform}' from unified: {original_input}"); return platform
                else: logger.warning(f"Unknown provider prefix '{provider}'. Defaulting to OPENAI."); return ModelPlatformType.OPENAI
            else: logger.error(f"Could not determine platform for string: {original_input}."); return None

    if isinstance(model_type_repr, ModelType):
        model_name = model_type_repr.name.upper()
        if model_name.startswith("GPT") or model_name.startswith("O3"): return ModelPlatformType.OPENAI
        if model_name.startswith("GEMINI"): return ModelPlatformType.GEMINI # Direct Google API
        if model_name.startswith("AZURE"): return ModelPlatformType.AZURE
        if model_name.startswith("QWEN"): return ModelPlatformType.QWEN
        if model_name.startswith("DEEPSEEK"): return ModelPlatformType.DEEPSEEK
        if model_name.startswith("GROQ"): return ModelPlatformType.GROQ
        if "LLAMA" in model_name or "MIXTRAL" in model_name: logger.warning(f"Assuming OPENAI compatibility for {model_name}. Check env."); return ModelPlatformType.OPENAI
        if model_name.startswith("CLAUDE"): return ModelPlatformType.ANTHROPIC
        logger.warning(f"Could not infer platform for enum {model_type_repr}. Defaulting to OPENAI."); return ModelPlatformType.OPENAI
    else: logger.error(f"Invalid type '{type(original_input)}' in get_platform_from_model_type."); return None

# check_if_file_completed remains unchanged
def check_if_file_completed(file_path: Path, marker: str) -> bool:
    """Checks if the file exists and ends with the specified marker."""
    if not file_path.exists() or file_path.stat().st_size == 0: return False
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            read_size = len(marker) + 100; f.seek(max(0, file_path.stat().st_size - read_size))
            return f.read().strip().endswith(marker)
    except Exception as e: logger.warning(f"Could not read end of {file_path}: {e}"); return False

# --- ADDED: TOC Extraction Function ---
# extract_toc remains unchanged
def extract_toc(pdf_path: Path) -> Optional[str]:
    """Extracts the table of contents (outline) from a PDF."""
    toc_entries = []
    try:
        reader = PyPDF2.PdfReader(str(pdf_path))
        if not reader.outline: logger.warning(f"No Outline/TOC found in '{pdf_path.name}'"); return None
        logger.info(f"Extracting TOC from '{pdf_path.name}'...")
        def _recursive_outline_extract(items, level=0):
            for item in items:
                if isinstance(item, list): _recursive_outline_extract(item, level + 1)
                else:
                    try:
                        page_num = item.page_number + 1 # PyPDF2 v3+ preference
                        title = item.title.strip() if item.title else "Untitled"
                        toc_entries.append(f"{'  ' * level}- {title} (Page {page_num})")
                    except AttributeError: logger.warning(f"Skipping TOC item without standard page_number/title: {item}")
                    except Exception as e_item: logger.warning(f"Error processing TOC item '{getattr(item, 'title', 'N/A')}': {e_item}")
        _recursive_outline_extract(reader.outline)
        if not toc_entries: logger.warning(f"PDF '{pdf_path.name}' has outline, but no entries extracted."); return None
        toc_text = "\n".join(toc_entries); logger.info(f"Extracted {len(toc_entries)} TOC entries."); return toc_text
    except (PdfReadError, DependencyError) as e: logger.error(f"Error reading PDF '{pdf_path.name}' for TOC: {e}"); return None
    except Exception as e: logger.error(f"Unexpected TOC extraction error '{pdf_path.name}': {e}", exc_info=True); return None

# --- MODIFIED: TOC Analysis Function handles failover state ---
def get_main_content_page_range(
    toc_text: str,
    failover_state: Dict[str, Any], # Pass mutable state
    prompt_template: str = PROMPT_TOC_ANALYSIS
) -> Tuple[Optional[int], Optional[int]]:
    """
    Uses the active AI model (from failover state) to analyze TOC text.
    Handles failover logic internally. Updates failover_state.
    """
    # Get model backends from failover state
    a_primary_backend = failover_state['a_primary_backend']
    a_backup_backend = failover_state['a_backup_backend']
    # a_active_backend and a_failures are accessed directly

    logger.info(f"Analyzing TOC with ACTIVE model: {failover_state['a_active_backend'].model_type.value}")
    final_prompt = prompt_template.format(toc_text=toc_text)

    start_page, end_page = None, None
    try:
        retry_delay = 2; max_retries = 2; retries = 0; raw_response = None; call_success = False

        while retries < max_retries:
            active_backend = failover_state['a_active_backend'] # Use current active
            try:
                messages = [{"role": "user", "content": final_prompt}]
                response_obj = active_backend.run(messages=messages)

                if response_obj and response_obj.choices and response_obj.choices[0].message:
                     raw_response = response_obj.choices[0].message.content
                     logger.debug(f"Raw analysis response (attempt {retries + 1} using {active_backend.model_type.value}):\n{raw_response[:500]}...")
                     call_success = True; break # Success
                else: logger.warning(f"TOC analysis ({active_backend.model_type.value}) returned invalid response structure (attempt {retries+1})."); raw_response = None; call_success = False
            except Exception as e_inner: logger.warning(f"TOC analysis API call failed (attempt {retries + 1}/{max_retries} using {active_backend.model_type.value}): {e_inner}. Retrying in {retry_delay}s..."); call_success = False

            if not call_success: sleep(retry_delay); retry_delay *= 2; retries += 1
            else: break # Should have broken already

        # --- Handle Analysis Result and Failover ---
        if call_success and raw_response:
            failover_state['a_failures'] = 0 # SUCCESS: Reset failure counter
            logger.debug("Analysis API call successful, reset failure counter.")
            try:
                json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_response, re.DOTALL | re.IGNORECASE) or re.search(r"(\{.*?\})", raw_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1); data = json.loads(json_str)
                    logger.info(f"Parsed TOC analysis result: {data.get('reasoning', 'N/A')}")
                    s_page, e_page = data.get("start_page"), data.get("end_page")
                    if isinstance(s_page, int) and s_page > 0: start_page = s_page
                    elif s_page is not None: logger.warning(f"Invalid start_page: {s_page}.")
                    if isinstance(e_page, int) and e_page > 0: end_page = e_page
                    elif e_page is not None: logger.warning(f"Invalid end_page: {e_page}.")
                    if start_page is not None and end_page is not None and start_page > end_page: logger.warning(f"Start > end ({start_page} > {end_page}). Discarding."); return None, None
                    logger.info(f"Determined main content range: Start={start_page}, End={end_page}"); return start_page, end_page
                else: logger.error(f"Could not find JSON in analysis response:\n{raw_response}"); call_success = False # Treat as failure
            except (json.JSONDecodeError, Exception) as e_parse: logger.error(f"Error processing analysis JSON '{e_parse}':\n{raw_response}", exc_info=True); call_success = False # Treat as failure

        # --- Handle Failure and Failover (if call_success is False) ---
        if not call_success:
            logger.error(f"Failed to get valid analysis after {max_retries} attempts using {failover_state['a_active_backend'].model_type.value}.")
            failover_state['a_failures'] += 1
            logger.info(f"Analysis failure count incremented to: {failover_state['a_failures']}")
            if failover_state['a_failures'] >= FAILURE_THRESHOLD:
                current_active_model = failover_state['a_active_backend'].model_type.value
                switched = False
                if failover_state['a_active_backend'] == a_primary_backend and a_backup_backend:
                    failover_state['a_active_backend'] = a_backup_backend; logger.warning(f"Reached {FAILURE_THRESHOLD} analysis failures on PRIMARY ({current_active_model}). Switching to BACKUP: {a_backup_backend.model_type.value}"); switched = True
                elif failover_state['a_active_backend'] == a_backup_backend and a_primary_backend:
                     failover_state['a_active_backend'] = a_primary_backend; logger.warning(f"Reached {FAILURE_THRESHOLD} analysis failures on BACKUP ({current_active_model}). Switching back to PRIMARY: {a_primary_backend.model_type.value}"); switched = True
                else: logger.error(f"Reached {FAILURE_THRESHOLD} failures, cannot switch analysis model from {current_active_model}.")
                if switched and a_primary_backend != a_backup_backend: failover_state['a_failures'] = 0; logger.info("Analysis failure counter reset after model switch.")
            return None, None # Analysis failed

    except Exception as e: # Catch unhandled errors in the try block
        logger.error(f"Unhandled error during TOC analysis: {e}", exc_info=True)
        failover_state['a_failures'] += 1 # Count exception as failure
        logger.info(f"Analysis failure count incremented to: {failover_state['a_failures']} due to exception.")
        if failover_state['a_failures'] >= FAILURE_THRESHOLD:
             current_active_model = failover_state['a_active_backend'].model_type.value; switched=False
             if failover_state['a_active_backend'] == a_primary_backend and a_backup_backend: failover_state['a_active_backend'] = a_backup_backend; logger.warning(f"Reached {FAILURE_THRESHOLD} failures (exception). Switching analysis to BACKUP: {a_backup_backend.model_type.value}"); switched=True
             elif failover_state['a_active_backend'] == a_backup_backend and a_primary_backend: failover_state['a_active_backend'] = a_primary_backend; logger.warning(f"Reached {FAILURE_THRESHOLD} failures (exception). Switching analysis back to PRIMARY: {a_primary_backend.model_type.value}"); switched=True
             else: logger.error(f"Reached {FAILURE_THRESHOLD} failures (exception), cannot switch analysis from {current_active_model}.")
             if switched and a_primary_backend != a_backup_backend: failover_state['a_failures'] = 0; logger.info("Analysis failure counter reset after model switch.")
        return None, None


# --- System Temp File Cleanup ---
# cleanup_system_temp_files unchanged
def cleanup_system_temp_files():
    """Finds and deletes temporary files created by the toolkit in /tmp."""
    logger.debug(f"Attempting periodic cleanup of '{SYSTEM_TEMP_DIR_PATTERN}'...")
    count = 0; deleted_something = False
    try:
        for temp_file_path_str in glob.iglob(SYSTEM_TEMP_DIR_PATTERN):
            try:
                p = Path(temp_file_path_str)
                if p.is_file(): p.unlink(); logger.debug(f"Deleted tmp file: {p.name}"); count += 1; deleted_something = True
                else: logger.debug(f"Skipping non-file glob entry: {p.name}")
            except Exception as e_unlink: logger.warning(f"Could not delete tmp file {temp_file_path_str}: {e_unlink}")
        if deleted_something: logger.debug(f"Periodic cleanup removed {count} tmp file(s).")
    except Exception as e_glob: logger.error(f"Error during tmp cleanup glob: {e_glob}")

# --- NEW: Get PDF Page Count ---
# get_pdf_page_count remains unchanged
def get_pdf_page_count(pdf_path: Path) -> Optional[int]:
    """Gets the total number of pages in a PDF using PyPDF2."""
    try:
        reader = PyPDF2.PdfReader(str(pdf_path)); num_pages = len(reader.pages)
        logger.debug(f"PDF '{pdf_path.name}' has {num_pages} pages."); return num_pages
    except (PdfReadError, DependencyError) as e: logger.error(f"Error reading PDF '{pdf_path.name}' for page count: {e}"); return None
    except Exception as e: logger.error(f"Unexpected page count error '{pdf_path.name}': {e}", exc_info=True); return None

# --- Main Logic Function ---
def run_main_logic(args):
    # Set up logging based on args
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level_int) # Set root logger level
    formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] %(message)s')
    root_logger = logging.getLogger()
    if root_logger.hasHandlers(): root_logger.handlers.clear()
    ch = logging.StreamHandler(); ch.setLevel(log_level_int); ch.setFormatter(formatter); root_logger.addHandler(ch)
    logger.setLevel(log_level_int)

    # --- Input validation ---
    if args.input_dir and args.book_dir: logger.critical("Cannot specify both --input-dir and --book-dir."); sys.exit(1)
    if not args.input_dir and not args.book_dir: logger.critical("Must specify either --input-dir or --book-dir."); sys.exit(1)
    source_path_str = args.input_dir if args.input_dir else args.book_dir
    source_dir = Path(source_path_str).resolve()
    if not source_dir.is_dir(): logger.critical(f"Input source directory not found: {source_dir}"); sys.exit(1)
    process_pdfs = bool(args.book_dir)
    mode_str = "PDFs in book directory" if process_pdfs else "images in directory"
    logger.info(f"Mode: Processing {mode_str}: {source_dir}")

    user_start_page_arg, user_end_page_arg = args.start_page, args.end_page
    if (user_start_page_arg is not None and user_end_page_arg is None) or (user_start_page_arg is None and user_end_page_arg is not None): logger.critical("Must provide BOTH --start-page and --end-page if specifying one."); sys.exit(1)
    if user_start_page_arg is not None and user_end_page_arg is not None and user_start_page_arg > user_end_page_arg: logger.critical(f"Start page ({user_start_page_arg}) > end page ({user_end_page_arg})."); sys.exit(1)
    if user_start_page_arg is not None and user_start_page_arg < 1: logger.critical(f"Start page ({user_start_page_arg}) must be >= 1."); sys.exit(1)

    # --- Output path setup ---
    single_output_file_path: Optional[Path] = None
    output_directory_path: Optional[Path] = None
    resume_state_file_path: Optional[Path] = None
    final_append_mode = args.append

    if not process_pdfs:
        if args.output: single_output_file_path = Path(args.output).resolve(); logger.info(f"Output --> SINGLE file: {single_output_file_path}")
        else: logger.info("Output --> CONSOLE.")
        logger.warning("Image dir mode assumes images are pre-generated. Async composites N/A. Persistent range N/A.")
    else:
        output_directory_path = Path(args.output).resolve() if args.output else (source_dir / "books_tts")
        try: output_directory_path.mkdir(parents=True, exist_ok=True); logger.info(f"Output directory for books: {output_directory_path}")
        except OSError as e: logger.critical(f"Failed to create output directory {output_directory_path}: {e}"); sys.exit(1)
        resume_state_file_path = output_directory_path / RESUME_STATE_FILENAME
        logger.info(f"Resume state file: {resume_state_file_path}")

    # --- Model selection and initialization (with backup) ---
    logger.info("Parsing model arguments...")
    primary_t_model_repr = parse_model_string_to_enum(args.transcription_model, "Primary Transcription")
    primary_a_model_repr = parse_model_string_to_enum(args.analysis_model, "Primary Analysis")
    backup_model_repr = parse_model_string_to_enum(DEFAULT_BACKUP_MODEL, "Backup")

    if not primary_t_model_repr: logger.critical("Invalid primary transcription model. Cannot proceed."); sys.exit(1)
    analysis_model_needed = process_pdfs and not (user_start_page_arg and user_end_page_arg)
    if analysis_model_needed and not primary_a_model_repr: logger.critical("Invalid primary analysis model (required for auto range). Cannot proceed."); sys.exit(1)
    if not backup_model_repr: logger.critical(f"Invalid backup model string ({DEFAULT_BACKUP_MODEL}). Cannot proceed."); sys.exit(1)

    logger.info("Determining model platforms...")
    primary_t_platform = get_platform_from_model_type(primary_t_model_repr)
    primary_a_platform = get_platform_from_model_type(primary_a_model_repr) if primary_a_model_repr else None
    backup_platform = get_platform_from_model_type(backup_model_repr)

    if not primary_t_platform: logger.critical("Could not determine platform for primary transcription model."); sys.exit(1)
    if analysis_model_needed and not primary_a_platform: logger.critical("Could not determine platform for primary analysis model."); sys.exit(1)
    if not backup_platform: logger.critical("Could not determine platform for backup model."); sys.exit(1)

    logger.info(f"Primary T: Model='{args.transcription_model}', Platform={primary_t_platform}")
    logger.info(f"Backup T/A: Model='{DEFAULT_BACKUP_MODEL}', Platform={backup_platform}")
    if primary_a_model_repr: logger.info(f"Primary A: Model='{args.analysis_model}', Platform={primary_a_platform}")

    logger.info("Initializing models and toolkits...")
    t_primary_backend, t_backup_backend = None, None
    a_primary_backend, a_backup_backend = None, None
    t_primary_tk, t_backup_tk = None, None
    try:
        t_primary_backend = ModelFactory.create(model_platform=primary_t_platform, model_type=primary_t_model_repr)
        t_primary_tk = ImageAnalysisToolkit(model=t_primary_backend)
        logger.info(f"Initialized Primary T Toolkit: {t_primary_backend.model_type.value}")
        if primary_t_model_repr != backup_model_repr:
            t_backup_backend = ModelFactory.create(model_platform=backup_platform, model_type=backup_model_repr)
            t_backup_tk = ImageAnalysisToolkit(model=t_backup_backend)
            logger.info(f"Initialized Backup T Toolkit: {t_backup_backend.model_type.value}")
        else: t_backup_tk = t_primary_tk; logger.info("Backup T model is same as primary, reusing toolkit.")

        if analysis_model_needed:
            a_primary_backend = ModelFactory.create(model_platform=primary_a_platform, model_type=primary_a_model_repr)
            logger.info(f"Initialized Primary A Backend: {a_primary_backend.model_type.value}")
            if primary_a_model_repr != backup_model_repr:
                a_backup_backend = ModelFactory.create(model_platform=backup_platform, model_type=backup_model_repr)
                logger.info(f"Initialized Backup A Backend: {a_backup_backend.model_type.value}")
            else: a_backup_backend = a_primary_backend; logger.info("Backup A model is same as primary, reusing backend.")
    except Exception as e: logger.critical(f"Failed to initialize models/toolkits: {e}", exc_info=True); sys.exit(1)

    if not t_primary_tk or not t_backup_tk: logger.critical("Transcription toolkit initialization failed."); sys.exit(1)
    if analysis_model_needed and (not a_primary_backend or not a_backup_backend): logger.critical("Analysis backend initialization failed."); sys.exit(1)

    # --- Setup Failover State Dictionary ---
    failover_state = {
        't_primary_tk': t_primary_tk,
        't_backup_tk': t_backup_tk,
        't_active_tk': t_primary_tk, # Start with primary
        't_failures': 0,
        'a_primary_backend': a_primary_backend,
        'a_backup_backend': a_backup_backend,
        'a_active_backend': a_primary_backend, # Start with primary
        'a_failures': 0,
    }
    logger.info(f"Initial active transcription model: {failover_state['t_active_tk'].model.model_type.value}")
    if failover_state['a_active_backend']: logger.info(f"Initial active analysis model: {failover_state['a_active_backend'].model_type.value}")
    else: logger.info("Analysis model not used in this run.")

    # --- Main Processing Logic ---
    if not process_pdfs:
        # --- Single Image Directory Processing ---
        logger.info(f"Processing pre-existing images from: {source_dir}")
        overall_initial_context: Optional[str] = None
        if single_output_file_path and final_append_mode and single_output_file_path.exists() and single_output_file_path.stat().st_size > 0:
            try:
                with open(single_output_file_path, 'r', encoding='utf-8') as f:
                    f.seek(max(0, single_output_file_path.stat().st_size - 4096)); overall_initial_context = f.read().strip() or None
                if overall_initial_context: logger.info(f"Loaded context (last ~{len(overall_initial_context)} chars) from single file.")
            except Exception as e: logger.error(f"Error reading context from {single_output_file_path}: {e}")
            try:
                 with open(single_output_file_path, 'a', encoding='utf-8') as f: f.write(f"\n\n{'='*20} Appending at {datetime.datetime.now()} {'='*20}\n\n")
            except IOError as e: logger.error(f"Error adding append separator to {single_output_file_path}: {e}")
        elif final_append_mode: logger.info(f"Append mode active, but {single_output_file_path} missing/empty. Starting fresh.")

        image_files_with_nums = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.ppm"]:
            for fp in source_dir.glob(ext):
                 pn = extract_page_number(fp);
                 if pn is not None: image_files_with_nums.append((pn, fp))
                 else: logger.debug(f"Skipping non-standard named file: {fp.name}")
        if not image_files_with_nums: logger.warning(f"No valid images found in {source_dir}. Exiting."); sys.exit(0)
        image_files_with_nums.sort(key=lambda x: x[0])
        image_files = [item[1] for item in image_files_with_nums]
        logger.info(f"Found {len(image_files)} valid images to process.")

        last_successful_transcription = overall_initial_context
        processing_successful_overall = True
        temp_composite_dir = source_dir / ".composites_img_mode"; temp_composite_dir.mkdir(exist_ok=True)
        if single_output_file_path and not final_append_mode:
             try: single_output_file_path.parent.mkdir(exist_ok=True, parents=True); open(single_output_file_path, 'w').close(); logger.info(f"Truncated output file: {single_output_file_path}")
             except IOError as e: logger.critical(f"Error preparing output file {single_output_file_path}: {e}"); sys.exit(1)

        for i, current_image_path in enumerate(image_files):
            मध्य_पृष्ठ_नाम = current_image_path.name # Using a unique variable name to avoid potential shadowing if we rename
            page_num = extract_page_number(current_image_path) or -1
            if page_num < 0: logger.warning(f"Could not extract page num from {मध्य_पृष्ठ_नाम} in loop, skipping."); continue
            if user_start_page_arg is not None and page_num < user_start_page_arg: logger.debug(f"Skipping {मध्य_पृष्ठ_नाम} (page {page_num}): before user start {user_start_page_arg}."); continue
            if user_end_page_arg is not None and page_num > user_end_page_arg: logger.info(f"Reached user end page {user_end_page_arg}. Stopping."); break
            can_get_prev, can_get_next = i > 0, i < len(image_files) - 1
            if not can_get_prev or not can_get_next: logger.info(f"Skipping edge image {मध्य_पृष्ठ_नाम}: Cannot form 3-page set."); continue

            logger.info(f"Processing image {मध्य_पृष्ठ_नाम} (Page: {page_num})")
            montage_input_paths = [image_files[i-1], image_files[i], image_files[i+1]]
            base_name = f"img_page_{page_num:04d}"
            temp_composite_img_path = None
            image_processing_failed = False

            try:
                temp_composite_img_path = create_composite_image_task(montage_input_paths, temp_composite_dir, base_name)
                if temp_composite_img_path and temp_composite_img_path.exists():
                    transcription, call_success = transcribe_image(
                        active_toolkit=failover_state['t_active_tk'], # Use active from state
                        image_path=temp_composite_img_path,
                        original_middle_page_name=मध्य_पृष्ठ_नाम,
                        previous_transcription=last_successful_transcription
                    )
                    if call_success: # API/content was okay
                        failover_state['t_failures'] = 0 # Reset on success
                        transcription = transcription.strip()
                        if transcription.upper() == "SKIPPED": logger.info(f"SKIPPED image {मध्य_पृष्ठ_नाम}.")
                        else:
                            if single_output_file_path:
                                try:
                                    with open(single_output_file_path, 'a', encoding='utf-8') as f: f.write(transcription + "\n\n")
                                    logger.info(f"Appended transcription for {मध्य_पृष्ठ_नाम} to {single_output_file_path.name}")
                                except IOError as e: logger.error(f"Error writing to {single_output_file_path}: {e}"); image_processing_failed = True
                            else: print(f"--- {मध्य_पृष्ठ_नाम} ---\n{transcription}\n--- End ---\n")
                            last_successful_transcription = transcription
                    else: # API call failed or returned bad content
                        logger.warning(f"--- Transcription FAILED for {मध्य_पृष्ठ_नाम} using {failover_state['t_active_tk'].model.model_type.value} ---")
                        failover_state['t_failures'] += 1
                        image_processing_failed = True
                        if failover_state['t_failures'] >= FAILURE_THRESHOLD:
                            current_active_model = failover_state['t_active_tk'].model.model_type.value
                            switched = False
                            if failover_state['t_active_tk'] == t_primary_tk and t_backup_tk: failover_state['t_active_tk'] = t_backup_tk; logger.warning(f"Failover: Switching T to BACKUP: {t_backup_tk.model.model_type.value}"); switched=True
                            elif failover_state['t_active_tk'] == t_backup_tk and t_primary_tk: failover_state['t_active_tk'] = t_primary_tk; logger.warning(f"Failover: Switching T back to PRIMARY: {t_primary_tk.model.model_type.value}"); switched=True
                            else: logger.error(f"Failover error: Cannot switch T from {current_active_model}.")
                            if switched and t_primary_tk != t_backup_tk: failover_state['t_failures'] = 0; logger.info("Reset T failure counter.")
                else: logger.error(f"Composite creation failed for {मध्य_पृष्ठ_नाम}. Skipping."); image_processing_failed = True
            except Exception as e_outer: logger.error(f"Error processing image {मध्य_पृष्ठ_नाम}: {e_outer}", exc_info=True); image_processing_failed = True
            finally:
                 if temp_composite_img_path and temp_composite_img_path.exists():
                     try: temp_composite_img_path.unlink(); logger.debug(f"Deleted composite: {temp_composite_img_path.name}")
                     except OSError as e_del_comp: logger.warning(f"Could not delete composite {temp_composite_img_path.name}: {e_del_comp}")
            if image_processing_failed: processing_successful_overall = False # Mark overall failure

        # --- Cleanup after image loop ---
        if temp_composite_dir.exists():
            try: shutil.rmtree(temp_composite_dir)
            except OSError as e_final_clean: logger.warning(f"Could not clean up {temp_composite_dir}: {e_final_clean}")
        if not processing_successful_overall: logger.error(f"Processing encountered errors for image directory: {source_dir}")
        logger.info("Image directory processing finished.")

    else:
        # --- PDF Directory Processing ---
        pdf_files = sorted(list(source_dir.glob('*.pdf')))
        if not pdf_files: logger.warning(f"No PDF files found in book directory: {source_dir}"); sys.exit(0)
        logger.info(f"Found {len(pdf_files)} PDFs to process.")
        current_state_data = load_resume_state(resume_state_file_path) if resume_state_file_path else {}
        images_base_dir = source_dir / TEMP_IMAGE_SUBDIR

        with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='CompositeGen') as executor:
            for pdf_path in pdf_files:
                pdf_filename = pdf_path.name
                logger.info(f"--- Starting PDF: {pdf_filename} ---")
                book_specific_output_path = output_directory_path / f"{pdf_path.stem}.txt"
                pdf_image_dir_path = images_base_dir / pdf_path.stem
                logger.info(f"Output file: {book_specific_output_path}")
                logger.info(f"Temp images dir: {pdf_image_dir_path}")

                if check_if_file_completed(book_specific_output_path, END_MARKER):
                    logger.info(f"'{pdf_filename}' already completed ('{END_MARKER}' found). Skipping.");
                    if resume_state_file_path and pdf_filename in current_state_data:
                         logger.info(f"Removing completed '{pdf_filename}' from resume state.")
                         del current_state_data[pdf_filename]; save_resume_state(resume_state_file_path, current_state_data)
                    if pdf_image_dir_path.exists():
                        try: shutil.rmtree(pdf_image_dir_path); logger.debug("Cleaned leftover image dir.")
                        except OSError as e_clean: logger.warning(f"Could not clean leftover dir {pdf_image_dir_path}: {e_clean}")
                    continue

                num_pdf_pages = get_pdf_page_count(pdf_path)
                if num_pdf_pages is None: logger.error(f"Cannot get page count for '{pdf_filename}'. Skipping."); continue

                # --- Determine Base Page Range Logic (includes analysis with failover) ---
                base_start_page, base_end_page, range_source = None, None, "undetermined"
                pdf_state = current_state_data.get(pdf_filename, {})
                if user_start_page_arg is not None and user_end_page_arg is not None:
                    base_start_page, base_end_page, range_source = user_start_page_arg, user_end_page_arg, "user"
                    logger.info(f"Using user range for '{pdf_filename}': {base_start_page}-{base_end_page}")
                    if "detected_start" in pdf_state or "detected_end" in pdf_state: # Clear old detected if user overrides
                        pdf_state.pop("detected_start", None); pdf_state.pop("detected_end", None)
                        if resume_state_file_path: current_state_data[pdf_filename] = pdf_state; save_resume_state(resume_state_file_path, current_state_data)
                elif "detected_start" in pdf_state or "detected_end" in pdf_state:
                     loaded_start, loaded_end = pdf_state.get("detected_start"), pdf_state.get("detected_end")
                     if (isinstance(loaded_start, int) or loaded_start is None) and (isinstance(loaded_end, int) or loaded_end is None): base_start_page, base_end_page, range_source = loaded_start, loaded_end, "state"; logger.info(f"Using range from state '{pdf_filename}': Start={base_start_page}, End={base_end_page}")
                     else: logger.warning(f"Invalid range in state for '{pdf_filename}': Start={loaded_start}, End={loaded_end}. Will attempt auto."); pdf_state.pop("detected_start", None); pdf_state.pop("detected_end", None); range_source = "state_invalid"

                if range_source in ["undetermined", "state_invalid"]:
                    logger.info(f"No valid range from user/state for '{pdf_filename}'. Attempting auto detection...")
                    range_source = "auto"
                    toc_text = extract_toc(pdf_path)
                    if toc_text and failover_state.get('a_active_backend'):
                        # --- Call analysis with failover state ---
                        detected_start, detected_end = get_main_content_page_range(toc_text, failover_state)
                        if detected_start is not None or detected_end is not None: base_start_page, base_end_page = detected_start, detected_end; logger.info(f"Auto-detected range '{pdf_filename}': Start={base_start_page}, End={base_end_page}")
                        else: logger.warning(f"Auto range detection failed for '{pdf_filename}'. Processing all."); range_source = "failed_auto"; base_start_page, base_end_page = None, None
                        pdf_state["detected_start"], pdf_state["detected_end"] = base_start_page, base_end_page # Store result or None
                    else: logger.warning(f"Cannot attempt auto range: {'No TOC' if not toc_text else 'No analysis model'}. Processing all."); range_source = "no_toc/model"; base_start_page, base_end_page = None, None; pdf_state["detected_start"], pdf_state["detected_end"] = None, None
                    if resume_state_file_path: current_state_data[pdf_filename] = pdf_state; save_resume_state(resume_state_file_path, current_state_data); logger.info(f"Saved analysis result to state for '{pdf_filename}'.")

                # --- Determine Effective Page Range & Resumption ---
                last_processed_page = pdf_state.get("last_processed")
                resuming = last_processed_page is not None
                resume_start_page = (last_processed_page + 1) if resuming else 1
                logger.debug(f"Range Source: {range_source}. Base: {base_start_page}-{base_end_page}. Resume state: last_processed={last_processed_page}")
                effective_start_page = max(resume_start_page if resuming else 1, base_start_page or 1)
                effective_end_page = min(base_end_page if base_end_page is not None else num_pdf_pages, num_pdf_pages)
                logger.info(f"Effective range for '{pdf_filename}': Start={effective_start_page}, End={effective_end_page}")

                if effective_start_page > effective_end_page: logger.warning(f"Effective start {effective_start_page} > end {effective_end_page}. Nothing to process for '{pdf_filename}'."); continue

                # --- Load Context Logic ---
                current_book_initial_context = None
                if (resuming or (final_append_mode and book_specific_output_path.exists() and book_specific_output_path.stat().st_size > 0)):
                    try:
                        logger.info(f"Reading context from {book_specific_output_path}")
                        with open(book_specific_output_path, 'r', encoding='utf-8') as bf: bf.seek(max(0, book_specific_output_path.stat().st_size - 4096)); current_book_initial_context = bf.read().strip() or None
                        if current_book_initial_context: logger.info(f"Loaded context ({len(current_book_initial_context)} chars) for {pdf_filename}.")
                        else: logger.warning(f"Context file empty/unreadable: {book_specific_output_path}")
                    except Exception as e: logger.error(f"Error reading context {book_specific_output_path}: {e}")
                elif final_append_mode: logger.info(f"Append mode active, but {book_specific_output_path} missing/empty. Starting context fresh.")
                if final_append_mode and book_specific_output_path.exists() and book_specific_output_path.stat().st_size > 0 and not resuming:
                     try:
                         with open(book_specific_output_path, 'a', encoding='utf-8') as bf: bf.write(f"\n\n{'='*20} Appending ({range_source}) at {datetime.datetime.now()} {'='*20}\n\n")
                     except IOError as e: logger.error(f"Error prepping append for {book_specific_output_path}: {e}. Skipping."); continue

                # --- Process the PDF using Async & Failover ---
                pdf_processing_success = False
                try:
                    pdf_image_dir_path.mkdir(parents=True, exist_ok=True)
                    process_append_mode = resuming or final_append_mode

                    # Pass the mutable failover state dict
                    pdf_processing_success, _ = process_pdf_pages(
                        pdf_path=pdf_path, num_pdf_pages=num_pdf_pages,
                        pdf_image_dir=pdf_image_dir_path, output_path=book_specific_output_path,
                        append_mode=process_append_mode, start_page=effective_start_page,
                        end_page=effective_end_page, failover_state=failover_state,
                        prompt_template=PROMPT_TEMPLATE, initial_context=current_book_initial_context,
                        pdf_filename=pdf_filename, state_file_path=resume_state_file_path,
                        state_data=current_state_data, image_format=args.image_format,
                        dpi=args.dpi, executor=executor
                    )

                    if pdf_processing_success:
                        logger.info(f"Successfully processed PDF: {pdf_filename}")
                        try:
                            with open(book_specific_output_path, 'a', encoding='utf-8') as f_end: f_end.write(f"\n\n{END_MARKER}\n")
                            logger.info(f"Added '{END_MARKER}' to completed file: {book_specific_output_path.name}")
                            if resume_state_file_path and pdf_filename in current_state_data: logger.info(f"Removing completed '{pdf_filename}' from state."); del current_state_data[pdf_filename]; save_resume_state(resume_state_file_path, current_state_data)
                        except IOError as e: logger.error(f"Error adding '{END_MARKER}' to {book_specific_output_path.name}: {e}"); pdf_processing_success = False # Mark failed if end marker fails
                    else: logger.error(f"Processing failed for PDF: {pdf_filename}.")

                except Exception as e: logger.critical(f"Critical error processing PDF {pdf_filename}: {e}", exc_info=True)
                finally:
                    if pdf_image_dir_path.exists():
                        try: shutil.rmtree(pdf_image_dir_path); logger.info(f"Cleaned up PDF image directory: {pdf_image_dir_path}")
                        except Exception as e_clean: logger.warning(f"Error cleaning image dir {pdf_image_dir_path}: {e_clean}")
                    elif pdf_image_dir_path: logger.info(f"Keeping image dir: {pdf_image_dir_path}")
                logger.info(f"--- Finished PDF: {pdf_filename} ---")

        # --- Cleanup after PDF loop ---
        if images_base_dir.exists():
            try:
                if not any(images_base_dir.iterdir()): images_base_dir.rmdir(); logger.info(f"Cleaned up empty base image directory: {images_base_dir}")
                else: logger.warning(f"Base image directory {images_base_dir} not empty, تركها.")
            except Exception as e_clean_base: logger.error(f"Error cleaning base image directory {images_base_dir}: {e_clean_base}")
        elif images_base_dir.exists(): logger.info(f"Keeping base image directory: {images_base_dir}")

        logger.info("All PDF processing finished.")


# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe text from book page images or PDFs using Owl/CAMEL-AI with async composite generation and model failover.",
        formatter_class=argparse.RawTextHelpFormatter # Keep for multi-line help
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--input-dir", help="Directory containing pre-generated image files (named 'prefix-digits.ext' or similar).")
    input_group.add_argument("-b", "--book-dir", help="Directory containing PDF book files to process (images generated on-the-fly).")
    parser.add_argument("-o", "--output", help="Output path (file for --input-dir, directory for --book-dir). Defaults to console/book_dir+'books_tts'.")
    parser.add_argument("-a", "--append", action="store_true", help=f"Append to output file(s). Reads context/checks '{END_MARKER}'/uses state file.")
    parser.add_argument("--start-page", type=int, help="Manual start page (1-based). Overrides auto/state.")
    parser.add_argument("--end-page", type=int, help="Manual end page (1-based). Overrides auto/state.")
    parser.add_argument("--transcription-model", type=str, default=DEFAULT_TRANSCRIPTION_MODEL, help=f"Primary model for transcription (default: {DEFAULT_TRANSCRIPTION_MODEL}).")
    parser.add_argument("--analysis-model", type=str, default=DEFAULT_ANALYSIS_MODEL, help=f"Primary model for TOC analysis (default: {DEFAULT_ANALYSIS_MODEL}).")
    parser.add_argument("--image-format", default='png', choices=['png', 'jpeg', 'tiff', 'ppm'], help="Format for generated PDF images (default: png).")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution (DPI) for generated PDF images (default: 300).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level (default: INFO).")

    # Generate epilog (remains the same, but add failover note)
    try:
        available_base = [f"'{m.value}'" for m in ModelType]; available_custom = [f"'{m.value}'" for m in mymodel]
        all_available = sorted(list(set(available_base + available_custom))); available_models_str = ", ".join(all_available)
        model_help_epilog = f"\n\nAvailable ModelType/Unified strings:\n{available_models_str}"
    except Exception: model_help_epilog = "\n\n(Could not list available models)"

    parser.epilog = f"""
Example Usage:
  # Images -> Single file (default models + failover)
  python %(prog)s --input-dir path/pages -o out.txt
  # PDFs -> Specific dir (override primaries, default backup, ASYNC)
  python %(prog)s --book-dir path/pdfs -o path/out --transcription-model 'oai/gpt-4o-mini' --analysis-model 'oai/gpt-4o'
  # PDFs -> Default dir, append/resume, ASYNC (default models)
  python %(prog)s --book-dir path/pdfs --append
  # PDFs -> Specific dir (manual range, ASYNC, default models)
  python %(prog)s --book-dir path/pdfs --start-page 50 --end-page 150 -o path/out --append

Notes:
- Requires 'pdftoppm' (poppler-utils) and 'montage' (ImageMagick).
- Requires 'PyPDF2' (`pip install pypdf2`).
- PDF Mode (--book-dir): Images generated on-the-fly. Async composite generation.
- Image Mode (--input-dir): Assumes images pre-generated. No async composites.
- Configure API keys in a .env file.
- Auto page range (--book-dir w/o --start/--end) uses TOC + analysis model.
- --append (--book-dir) uses '{END_MARKER}' & '{RESUME_STATE_FILENAME}'.
- **Failover:** Switches between primary/backup ('{DEFAULT_BACKUP_MODEL}') after {FAILURE_THRESHOLD} consecutive failures.
{model_help_epilog}
"""
    parsed_args = parser.parse_args()

    # Check prerequisites early for PDF mode
    if parsed_args.book_dir and (shutil.which("montage") is None or shutil.which("pdftoppm") is None):
        logging.critical("CRITICAL: 'montage' (ImageMagick) and/or 'pdftoppm' (poppler-utils) not found. Required for --book-dir mode.")
        sys.exit(1)

    try:
        run_main_logic(parsed_args)
    except Exception as e:
        logging.critical(f"Main script execution failed: {e}", exc_info=True)
        sys.exit(1) # Exit with error code on critical unexpected failure
    finally:
        logger.info("Performing final cleanup check of system temporary files in /tmp...")
        cleanup_system_temp_files()
        logger.info("Script finished.")
