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
# (Keep your existing model definitions or use ModelType directly)
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
                 # Check if parent dir name hints at being from pdftoppm (less relevant here)
                 if filename.parent.name == filename.stem:
                     logger.debug(f"Assuming page number {match_alt.group(1)} from filename {filename.name} based on parent dir structure.")
                     return int(match_alt.group(1))
                 else:
                      # In image mode, accept digits if that's all we have
                      logger.warning(f"Found digits {match_alt.group(1)} in {filename.name}. Assuming it's the page number.")
                      return int(match_alt.group(1))
             except ValueError:
                 logger.warning(f"Could not convert extracted number to int in filename: {filename.name}")
                 return None
        else:
            logger.warning(f"Could not extract page number (format '-digits.' or 'digits.') from filename: {filename.name}")
            return None

def create_temporary_composite_image(image_paths: List[Path], output_dir: Path, base_name: str) -> Optional[Path]:
    """Creates a temporary composite image from exactly 3 images using ImageMagick's montage tool."""
    if not image_paths or len(image_paths) != 3:
        logger.warning(f"Cannot create composite: Need exactly 3 image paths, got {len(image_paths)} for base {base_name}.")
        return None

    # Ensure all input paths actually exist before attempting montage
    missing_files = [p for p in image_paths if not p.exists()]
    if missing_files:
        logger.error(f"Cannot create composite: Missing input image(s): {[str(p.name) for p in missing_files]} for base {base_name}.")
        return None

    image_paths_str = ' '.join(f'"{path}"' for path in image_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{base_name}_composite.png"
    command = f"montage -mode concatenate -tile x1 {image_paths_str} \"{output_path}\""

    try:
        logger.debug(f"Running montage command: {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, encoding='utf-8')
        if output_path.exists():
            logger.info(f"Temporary composite image created: {output_path}")
            return output_path
        else:
             logger.error(f"Montage command ran but output file not found: {output_path}")
             logger.error(f"Montage stdout: {result.stdout}")
             logger.error(f"Montage stderr: {result.stderr}")
             return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating composite image with command '{command}': {e}")
        logger.error(f"Montage stdout: {e.stdout}")
        logger.error(f"Montage stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error("Error: 'montage' command not found. Please ensure ImageMagick is installed and in your system's PATH.")
        # Raise an exception or return None to stop further processing
        raise RuntimeError("ImageMagick 'montage' command not found.")
    except Exception as e:
        logger.error(f"Unexpected error creating composite image {base_name}: {e}", exc_info=True)
        return None

def transcribe_image(
    toolkit: ImageAnalysisToolkit,
    image_path: Path,
    original_middle_page_name: str,
    previous_transcription: Optional[str] = None,
    prompt_template: str = PROMPT_TEMPLATE
) -> Optional[str]:
    """Uses the ImageAnalysisToolkit to transcribe text from a composite image."""
    logger.info(f"Transcribing composite image representing middle page: {original_middle_page_name} using {toolkit.model.model_type.value}")

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
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                transcription = toolkit.ask_question_about_image(
                    image_path=str(image_path),
                    question=final_prompt
                )
            except Exception as e_inner:
                 logger.warning(f"Transcription attempt {retries + 1}/{max_retries} API call failed for {original_middle_page_name}: {e_inner}. Retrying in {retry_delay}s...")
                 sleep(retry_delay)
                 retry_delay *= 2
                 retries += 1
                 continue

            if transcription:
                # Clean up potential markdown/code blocks
                transcription = transcription.strip()
                transcription = re.sub(r'^```(text|markdown)?\s*', '', transcription, flags=re.IGNORECASE | re.MULTILINE)
                transcription = re.sub(r'\s*```$', '', transcription, flags=re.IGNORECASE | re.MULTILINE)
                transcription = transcription.strip('"\'') # Remove leading/trailing quotes

                if transcription and "Analysis failed" not in transcription and len(transcription) > 5:
                    logger.info(f"Transcription successful for {original_middle_page_name} (attempt {retries+1}). Length: {len(transcription)}")
                    return transcription
                else:
                    logger.warning(f"Transcription attempt {retries + 1}/{max_retries} failed or produced short/invalid output for middle page: {original_middle_page_name}. Content: '{transcription[:50]}...'. Retrying in {retry_delay}s...")
                    transcription = None # Reset for retry
                    sleep(retry_delay)
                    retry_delay *= 2
                    retries += 1
                    continue
            else:
                logger.warning(f"Received empty transcription on attempt {retries + 1}/{max_retries} for middle page: {original_middle_page_name}. Retrying in {retry_delay}s...")
                sleep(retry_delay)
                retry_delay *= 2
                retries += 1
                continue

        logger.error(f"Transcription FAILED after {max_retries} attempts for middle page: {original_middle_page_name} (composite: {image_path.name}). Returning None.")
        return None

    except Exception as e:
        logger.error(f"Unhandled error during transcription retry loop for image {image_path.name} (middle page: {original_middle_page_name}): {e}", exc_info=True)
        return None
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
        logger.error("Critical Error: 'pdftoppm' command not found. Please ensure poppler-utils is installed.")
        raise RuntimeError("pdftoppm not found") # Raise error to stop processing

    output_dir.mkdir(parents=True, exist_ok=True)

    # Define the output filename structure: <pdf_stem>-<page_number>.<format>
    # pdftoppm automatically appends the page number.
    output_prefix_path_part = output_dir / pdf_path.stem
    # Adjust expected filename based on pdftoppm's behavior (padding with zeros)
    # Let's determine padding based on rough number of digits needed
    try:
        reader = PyPDF2.PdfReader(str(pdf_path)) # Re-read for safety, maybe cache this later
        total_pages = len(reader.pages)
        padding_width = len(str(total_pages))
    except Exception:
        padding_width = 3 # Default padding if count fails
    expected_output_filename = f"{pdf_path.stem}-{page_number:0{padding_width}d}.{image_format}"
    # expected_output_filename = f"{pdf_path.stem}-{page_number:d}.{image_format}" # Use standard formatting # OLD
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
                  files_in_dir = list(output_dir.glob('*.*'))
                  logger.error(f"Files currently in {output_dir}: {[f.name for f in files_in_dir]}")
                  # Attempt to find a file with the correct number but different padding
                  alternative_match = list(output_dir.glob(f"{pdf_path.stem}-{page_number}.{image_format}")) # Check without padding
                  if not alternative_match:
                      alternative_match = list(output_dir.glob(f"{pdf_path.stem}-*?{page_number}.{image_format}")) # More generic check

                  if alternative_match:
                      logger.warning(f"Found possible match with different naming/padding: {alternative_match[0].name}. Returning this.")
                      return alternative_match[0]
                  else:
                      logger.error("No alternative file match found either.")

             except Exception as e_list:
                  logger.error(f"Could not list directory contents for debugging: {e_list}")
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
# --- MODIFIED: load_resume_state to handle new structure ---
def load_resume_state(state_file_path: Path) -> Dict[str, Any]:
    """
    Loads the resume state from a JSON file.
    Handles both old ({pdf: page}) and new ({pdf: {last_processed: page, ...}}) formats.
    """
    if not state_file_path.exists():
        return {}
    try:
        with open(state_file_path, 'r', encoding='utf-8') as f:
            raw_state_data = json.load(f)
        logger.info(f"Loaded resume state from {state_file_path}")

        # Check format and convert if necessary
        converted_state_data = {}
        needs_resave = False
        for pdf_filename, data in raw_state_data.items():
            if isinstance(data, int): # Old format detected
                logger.warning(f"Detected old state format for '{pdf_filename}'. Converting.")
                converted_state_data[pdf_filename] = {
                    "last_processed": data,
                    "detected_start": None, # Add placeholders
                    "detected_end": None
                }
                needs_resave = True
            elif isinstance(data, dict): # Assume new format
                 # Ensure essential key exists, even if None initially
                if "last_processed" not in data:
                    data["last_processed"] = None
                converted_state_data[pdf_filename] = data
            else:
                logger.warning(f"Skipping unrecognized state entry for '{pdf_filename}'. Unexpected data type: {type(data)}")

        if needs_resave:
            logger.info("Resaving state file in new format after conversion.")
            save_resume_state(state_file_path, converted_state_data) # Save converted data

        return converted_state_data # Return the potentially converted data

    except json.JSONDecodeError:
        logger.warning(f"Error decoding JSON from state file: {state_file_path}. Starting fresh.")
        return {}
    except IOError as e:
        logger.error(f"Error reading state file {state_file_path}: {e}. Starting fresh.")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading state file {state_file_path}: {e}. Starting fresh.", exc_info=True)
        return {}

def save_resume_state(state_file_path: Path, state_data: Dict[str, Any]):
    """Saves the resume state (new format) to a JSON file."""
    try:
        state_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Filter out entries where last_processed is None *unless* detected range exists
        # Also filter out completely empty dicts if they sneak in
        cleaned_state = {}
        for pdf, data in state_data.items():
            if isinstance(data, dict) and (data.get("last_processed") is not None or
                                           data.get("detected_start") is not None or
                                           data.get("detected_end") is not None):
                 cleaned_state[pdf] = data

        if not cleaned_state:
            logger.debug(f"State data is empty, deleting state file: {state_file_path}")
            if state_file_path.exists():
                state_file_path.unlink()
            return # No need to save empty state

        with open(state_file_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_state, f, indent=4) # Use cleaned state
        logger.debug(f"Saved resume state to {state_file_path}")
    except IOError as e:
        logger.error(f"Error writing state file {state_file_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving state file {state_file_path}: {e}", exc_info=True)


# --- REFACTORED: PDF Page Processing Function ---
# --- MODIFIED: process_pdf_pages updates state correctly ---
def process_pdf_pages(
    pdf_path: Path,
    num_pdf_pages: int,
    pdf_image_dir: Path, # Directory for this PDF's temporary images
    output_path: Optional[Path],
    append_mode: bool,
    start_page: Optional[int], # Effective start page
    end_page: Optional[int],   # Effective end page
    image_toolkit: ImageAnalysisToolkit,
    prompt_template: str,
    initial_context: Optional[str],
    pdf_filename: str, # Original PDF filename for state tracking
    state_file_path: Optional[Path],
    state_data: Dict[str, Any], # Pass the whole state dictionary
    image_format: str = 'png',
    dpi: int = 300
) -> Tuple[bool, Optional[str]]:
    """
    Processes pages of a PDF by generating images on the fly, transcribing,
    and cleaning up images in a rolling window.

    Args:
        pdf_path: Path to the PDF file.
        num_pdf_pages: Total number of pages in the PDF.
        pdf_image_dir: Directory to store temporary page images for this PDF.
        output_path: Path to the output text file (or None for console).
        append_mode: Whether to append to the output file.
        start_page: The 1-based page number to start processing from (inclusive).
        end_page: The 1-based page number to stop processing at (inclusive).
        image_toolkit: Initialized ImageAnalysisToolkit.
        prompt_template: The prompt template for transcription.
        initial_context: Transcription context from previous runs/files.
        pdf_filename: The name of the PDF file (for state management).
        state_file_path: Path to the resume state JSON file.
        state_data: The *entire* loaded resume state dictionary.
        image_format: Format for temporary images (e.g., 'png').
        dpi: Resolution for temporary images.

    Returns:
        A tuple: (success_flag, last_successful_transcription).
    """
    logger.info(f"Processing PDF: {pdf_filename} [Effective Pages {start_page or 1} to {end_page or num_pdf_pages}]")

    files_processed_count = 0
    last_successful_transcription = initial_context
    processing_successful = True
    # Keep track of images currently existing on disk for this PDF
    current_images_on_disk: Set[Path] = set()

    # --- Helper to generate image path ---
    def get_image_path(page_num: int) -> Path:
        # Determine padding dynamically based on total pages
        try:
            reader = PyPDF2.PdfReader(str(pdf_path)) # Re-read for safety
            total_pages = len(reader.pages)
            padding_width = len(str(total_pages))
        except Exception:
            padding_width = 3 # Default padding
        # pdftoppm uses 0-padding based on total pages. e.g., page 1 of 150 becomes -001
        return pdf_image_dir / f"{pdf_path.stem}-{page_num:0{padding_width}d}.{image_format}"

    # --- Helper to safely delete an image ---
    def delete_page_image(page_num: int):
        if page_num < 1: return # Cannot delete non-positive page
        img_path = get_image_path(page_num)
        if img_path in current_images_on_disk:
            try:
                img_path.unlink()
                logger.debug(f"Deleted page image: {img_path.name}")
                current_images_on_disk.remove(img_path)
            except FileNotFoundError:
                logger.debug(f"Image {img_path.name} already deleted.")
                current_images_on_disk.discard(img_path) # Ensure removal from set
            except OSError as e_del:
                logger.warning(f"Could not delete page image {img_path.name}: {e_del}")
        elif img_path.exists(): # Safety check if not in set but exists
             try:
                 img_path.unlink()
                 logger.debug(f"Deleted stray page image: {img_path.name}")
             except OSError as e_del_stray:
                 logger.warning(f"Could not delete stray page image {img_path.name}: {e_del_stray}")


    # Ensure temporary directories exist
    temp_composite_dir = pdf_image_dir / ".composites"
    try:
        pdf_image_dir.mkdir(parents=True, exist_ok=True)
        temp_composite_dir.mkdir(exist_ok=True)
        logger.info(f"Using temporary image directory: {pdf_image_dir}")
        logger.info(f"Using temporary composite directory: {temp_composite_dir}")
    except OSError as e_comp_dir:
        logger.error(f"Could not create temporary dirs under {pdf_image_dir}: {e_comp_dir}. Cannot process PDF.")
        return False, initial_context

    # Prepare output file if not appending
    if output_path and not append_mode:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f_init:
                f_init.truncate(0)
            logger.info(f"Output file {output_path} truncated (or created).")
        except IOError as e:
            logger.error(f"Error preparing output file {output_path}: {e}")
            return False, initial_context

    # Determine the loop range
    loop_start = start_page if start_page is not None else 1
    loop_end = end_page if end_page is not None else num_pdf_pages

    logger.info(f"Looping through pages {loop_start} to {loop_end} (inclusive) for {pdf_filename}.")

    # Get the specific state dict for this PDF
    pdf_state = state_data.get(pdf_filename, {}) # Default to empty dict if not present

    try:
        for page_num in range(loop_start, loop_end + 1):
            logger.info(f"--- Current target page: {page_num}/{loop_end} ({pdf_filename}) ---")

            # --- Check Edge Cases for 3-Page Window ---
            # Need page_num-1, page_num, page_num+1
            p_prev = page_num - 1
            p_curr = page_num
            p_next = page_num + 1

            if p_prev < 1 or p_next > num_pdf_pages:
                logger.info(f"Skipping transcription for edge page {page_num}: Cannot form 3-page window (Needs {p_prev} and {p_next}).")
                # Update state even for skipped edge page IF it's within range and state tracking is active
                if state_file_path:
                     logger.debug(f"Updating state to reflect skipped edge page {page_num}")
                     pdf_state["last_processed"] = page_num # Update specific dict
                     state_data[pdf_filename] = pdf_state # Put updated dict back
                     save_resume_state(state_file_path, state_data) # Save entire state
                # Still need to generate current/next if they are needed for the *next* iteration's window
                # (Convert only if not already present)
                curr_path = get_image_path(p_curr)
                if not curr_path.exists():
                    # Generate image if it's within overall bounds
                    if p_curr <= num_pdf_pages:
                         img_curr = convert_pdf_page_to_image(pdf_path, p_curr, pdf_image_dir, image_format, dpi)
                         if img_curr: current_images_on_disk.add(img_curr)
                elif curr_path not in current_images_on_disk: # If exists but not tracked
                    current_images_on_disk.add(curr_path)

                if p_next <= num_pdf_pages:
                    next_path = get_image_path(p_next)
                    if not next_path.exists():
                         img_next = convert_pdf_page_to_image(pdf_path, p_next, pdf_image_dir, image_format, dpi)
                         if img_next: current_images_on_disk.add(img_next)
                    elif next_path not in current_images_on_disk:
                        current_images_on_disk.add(next_path)
                # Cleanup page - 2 as it's definitely not needed now
                delete_page_image(page_num - 2)
                continue # Move to the next page number

            # --- Generate Required Images ---
            required_paths: Dict[str, Path] = {
                "prev": get_image_path(p_prev),
                "curr": get_image_path(p_curr),
                "next": get_image_path(p_next),
            }
            montage_input_paths: List[Path] = []
            generation_failed = False

            for role, page_idx in [("prev", p_prev), ("curr", p_curr), ("next", p_next)]:
                img_path = required_paths[role]
                if img_path not in current_images_on_disk and not img_path.exists():
                    logger.info(f"Generating missing image for page {page_idx} ({role})...")
                    generated_path = convert_pdf_page_to_image(
                        pdf_path, page_idx, pdf_image_dir, image_format, dpi
                    )
                    if generated_path:
                        current_images_on_disk.add(generated_path)
                        montage_input_paths.append(generated_path)
                    else:
                        logger.error(f"Failed to generate required image for page {page_idx} ({role}). Skipping transcription for page {p_curr}.")
                        generation_failed = True
                        break # Stop trying to generate for this window
                elif img_path.exists(): # Already exists (or just generated by another role check)
                     if img_path not in current_images_on_disk: # Track if untracked
                        current_images_on_disk.add(img_path)
                     montage_input_paths.append(img_path)
                else: # Should not happen if logic is correct
                     logger.error(f"Logic error: Image path {img_path.name} not in set and does not exist.")
                     generation_failed = True
                     break

            if generation_failed or len(montage_input_paths) != 3:
                logger.warning(f"Skipping transcription for page {p_curr} due to missing required images.")
                processing_successful = False # Mark failure for this PDF
                # Clean up page - 2 if possible
                delete_page_image(page_num - 2)
                continue # Try next page number

            # --- Create Composite and Transcribe ---
            middle_page_name = required_paths["curr"].name
            files_processed_count += 1
            base_name = f"page_{page_num:04d}" # Composite name base
            temp_composite_img_path = None

            try:
                temp_composite_img_path = create_temporary_composite_image(
                    montage_input_paths, temp_composite_dir, base_name
                )

                if temp_composite_img_path and temp_composite_img_path.exists():
                    transcription = None
                    try:
                        transcription = transcribe_image(
                            toolkit=image_toolkit,
                            image_path=temp_composite_img_path,
                            original_middle_page_name=middle_page_name,
                            previous_transcription=last_successful_transcription,
                            prompt_template=prompt_template
                        )

                        if transcription:
                            transcription = transcription.strip() # Already cleaned in transcribe_image
                            if transcription.upper() == "SKIPPED":
                                logger.info(f"SKIPPED page {page_num} ({middle_page_name}) due to ancillary content.")
                                # --- Update State for SKIPPED page ---
                                if state_file_path:
                                    pdf_state["last_processed"] = page_num # Update specific dict
                                    state_data[pdf_filename] = pdf_state # Put updated dict back
                                    save_resume_state(state_file_path, state_data) # Save entire state
                                    logger.debug(f"Updated resume state after skipping page {page_num}")
                                # DO NOT update last_successful_transcription for skipped pages
                            else:
                                if output_path:
                                    try:
                                        with open(output_path, 'a', encoding='utf-8') as f:
                                            f.write(transcription + "\n\n") # Add spacing
                                        logger.info(f"Appended transcription for page {page_num} to {output_path.name}")

                                        # --- Update State AFTER successful write ---
                                        if state_file_path:
                                            pdf_state["last_processed"] = page_num
                                            state_data[pdf_filename] = pdf_state
                                            save_resume_state(state_file_path, state_data)
                                            logger.debug(f"Updated resume state for {pdf_filename} to page {page_num}")

                                    except IOError as e:
                                        logger.error(f"Error writing transcription for page {page_num} to {output_path}: {e}")
                                        processing_successful = False
                                        # Don't update state if write fails
                                else: # Output to console
                                    print(f"--- Transcription for Page {page_num} ({middle_page_name}) ---")
                                    print(transcription)
                                    print("--- End Transcription ---\n")
                                    # Also update state if writing to console was implicitly successful
                                    if state_file_path:
                                            pdf_state["last_processed"] = page_num
                                            state_data[pdf_filename] = pdf_state
                                            save_resume_state(state_file_path, state_data)
                                            logger.debug(f"Updated resume state for {pdf_filename} to page {page_num} (console output)")


                                # Update context ONLY if transcription was successful AND NOT skipped
                                last_successful_transcription = transcription
                        else:
                            logger.warning(f"--- Transcription FAILED (empty/failed result) for page {page_num} ({middle_page_name}) ---")
                            processing_successful = False
                            # Do not update state if transcription fails

                    except Exception as e_transcribe:
                         logger.error(f"Error during transcription/writing for page {page_num} ({middle_page_name}): {e_transcribe}", exc_info=True)
                         processing_successful = False
                else:
                    logger.error(f"Failed to create composite image for page {page_num} ({middle_page_name}). Skipping.")
                    processing_successful = False

            except Exception as e_outer:
                logger.error(f"Outer loop error for page {page_num} ({middle_page_name}): {e_outer}", exc_info=True)
                processing_successful = False
            finally:
                if temp_composite_img_path and temp_composite_img_path.exists():
                    try:
                        temp_composite_img_path.unlink()
                        logger.debug(f"Deleted temporary composite: {temp_composite_img_path.name}")
                    except OSError as e_del_comp:
                        logger.warning(f"Could not delete temporary composite {temp_composite_img_path.name}: {e_del_comp}")

            # --- Rolling Cleanup: Delete oldest image (page_num - 2) ---
            delete_page_image(page_num - 2)

            # Check if overall processing failed to break early
            if not processing_successful:
                 logger.error(f"Stopping processing for {pdf_filename} due to error.")
                 break # Exit the loop for this PDF

    finally:
        # --- Final Cleanup for this PDF ---
        logger.debug(f"Performing final image cleanup for {pdf_filename}...")
        # Create a copy of the set to iterate over, as we modify it inside loop
        images_to_delete = list(current_images_on_disk)
        for img_path in images_to_delete:
            # Extract page num from name to call delete helper correctly
            match = re.search(r"-(\d+)\.[^.]+$", img_path.name)
            if match:
                 try:
                     pn = int(match.group(1))
                     delete_page_image(pn)
                 except ValueError:
                      logger.warning(f"Could not parse page number from {img_path.name} for final cleanup.")
                 except Exception as e_final_del_helper:
                      logger.warning(f"Error in final cleanup helper for {img_path.name}: {e_final_del_helper}")
            else:
                 # If name doesn't match, just try to delete the file directly
                 try:
                      if img_path.exists():
                           img_path.unlink()
                           logger.debug(f"Deleted non-standard named image in final cleanup: {img_path.name}")
                 except OSError as e_final_del_direct:
                      logger.warning(f"Could not delete non-standard image in final cleanup {img_path.name}: {e_final_del_direct}")


        # Clean up composite directory
        if temp_composite_dir.exists():
            try:
                shutil.rmtree(temp_composite_dir)
                logger.debug(f"Cleaned up composites directory: {temp_composite_dir}")
            except OSError as e_final_clean:
                 logger.warning(f"Could not clean up composites directory {temp_composite_dir}: {e_final_clean}")

    logger.info(f"Finished loop for PDF {pdf_filename}. Processed {files_processed_count} pages within effective range.")
    return processing_successful, last_successful_transcription

# --- Model Parsing and Platform Inference (Unchanged) ---
def parse_model_string_to_enum(model_string: str, model_purpose: str) -> Optional[Union[ModelType, str]]:
    """
    Tries to parse a string into a ModelType enum member.
    If it's a base enum, returns the enum member.
    If it's a known custom/unified string (like 'provider/model'), returns the string itself.
    Returns None if the string is invalid or unrecognized.
    """
    try:
        # Try direct conversion to base ModelType enum first
        model_enum = ModelType(model_string)
        logger.debug(f"Successfully parsed {model_purpose} model string '{model_string}' to base enum {model_enum}")
        return model_enum # Return the ENUM MEMBER
    except ValueError:
        # Not a base enum value. Check if it's potentially a unified string or known custom type.
        # We assume ModelFactory can handle strings like 'provider/model-name[:tag]'
        if "/" in model_string:
            logger.warning(f"Model string '{model_string}' for {model_purpose} is not a base ModelType enum. "
                           f"Assuming it's a unified model string and returning the string directly for ModelFactory.")
            # Check if it exists in our custom mymodel enum for logging/validation purposes
            try:
                _ = mymodel(model_string) # Check if it's in our custom enum
                logger.debug(f"Model string '{model_string}' matches a known custom 'mymodel' enum value.")
            except ValueError:
                logger.warning(f"Model string '{model_string}' does not match 'mymodel' enum either, but proceeding with the string.")
            return model_string # Return the STRING for ModelFactory to handle
        else:
            # Not a base enum and not in unified format provider/model
            logger.error(f"Invalid or unrecognized model string provided for {model_purpose}: '{model_string}'. "
                         f"It does not match any known ModelType enum value and is not in 'provider/model' format.")
            available_base = [m.value for m in ModelType]
            available_custom = [m.value for m in mymodel]
            logger.info(f"Available base ModelType enum values: {available_base}")
            logger.info(f"Available custom 'mymodel' enum values: {available_custom}")
            return None # Indicate failure

def get_platform_from_model_type(model_type_repr: Union[ModelType, str]) -> Optional[ModelPlatformType]:
    """
    Tries to infer the ModelPlatformType from a ModelType enum or model string.
    Returns None if the platform cannot be determined.
    """
    unified_platform_map = {
        "openrouter": ModelPlatformType.OPENAI, # Default assumption for OpenRouter
        "meta-llama": ModelPlatformType.OPENAI, # Default for Llama via OpenAI-compatible API
        "google": ModelPlatformType.GEMINI,
        "qwen": ModelPlatformType.QWEN,
        "anthropic": ModelPlatformType.OPENAI, # Default assumption for Claude via compatible API
        "deepseek": ModelPlatformType.DEEPSEEK,
        "groq": ModelPlatformType.GROQ,
        # Add other known provider prefixes if necessary
    }

    original_input = model_type_repr # Keep for logging

    if isinstance(model_type_repr, str):
        # Handle potential suffixes like ':free', ':latest', etc.
        model_string_cleaned = model_type_repr.split(':')[0]

        try:
            # Attempt to convert the *cleaned* string to a base ModelType enum first.
            enum_match = ModelType(model_string_cleaned)
            logger.debug(f"Successfully converted cleaned string '{model_string_cleaned}' to base enum {enum_match}")
            model_type_repr = enum_match # Replace original string with the enum for further processing

        except ValueError:
            # Not a base enum value. Check if it's a unified string format.
            if "/" in model_string_cleaned:
                provider = model_string_cleaned.split('/')[0].lower()
                platform = unified_platform_map.get(provider)
                if platform:
                    logger.debug(f"Inferred platform '{platform}' from unified model string: {original_input}")
                    return platform
                else:
                    # Provider prefix not in our map, make a best guess or default
                    logger.warning(f"Unknown provider prefix '{provider}' in model string '{original_input}'. Defaulting platform to OPENAI.")
                    return ModelPlatformType.OPENAI # Default fallback
            else:
                # String is not unified format and not a base enum.
                logger.error(f"Could not determine platform for model string: {original_input}. It's not a recognized base enum or known unified format.")
                return None # Indicate failure clearly

    # --- If we reach here, model_type_repr should be a ModelType enum instance ---
    if isinstance(model_type_repr, ModelType):
        model_name = model_type_repr.name.upper()

        # Check by name prefix (common pattern)
        if model_name.startswith("GPT") or model_name.startswith("O3"): return ModelPlatformType.OPENAI
        if model_name.startswith("GEMINI"): return ModelPlatformType.GEMINI
        if model_name.startswith("AZURE"): return ModelPlatformType.AZURE
        if model_name.startswith("QWEN"): return ModelPlatformType.QWEN
        if model_name.startswith("DEEPSEEK"): return ModelPlatformType.DEEPSEEK
        if model_name.startswith("GROQ"): return ModelPlatformType.GROQ
        if "LLAMA" in model_name or "MIXTRAL" in model_name:
             logger.warning(f"Assuming OPENAI platform compatibility for base Llama/Mixtral enum: {model_name}. Check .env for provider/base URL.")
             return ModelPlatformType.OPENAI # Default assumption, may need adjustment based on env
        if model_name.startswith("CLAUDE"):
             logger.warning("Assuming OPENAI platform compatibility for Claude model enum. Verify API provider/base URL.")
             return ModelPlatformType.OPENAI

        # Fallback for other base enums not explicitly handled above
        logger.warning(f"Could not infer platform for base model enum: {model_type_repr}. Defaulting to OPENAI.")
        return ModelPlatformType.OPENAI # Default fallback
    else:
         # This path should ideally not be reached if the logic is sound
         logger.error(f"Invalid type '{type(original_input)}' encountered unexpectedly in get_platform_from_model_type. Value: {original_input}")
         return None # Indicate failure

def check_if_file_completed(file_path: Path, marker: str) -> bool:
    """Checks if the file exists and ends with the specified marker."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read a reasonably sized chunk from the end
            read_size = len(marker) + 100 # Add buffer
            f.seek(max(0, file_path.stat().st_size - read_size))
            last_chunk = f.read()
            # Check if the marker is present at the very end after stripping whitespace
            return last_chunk.strip().endswith(marker)
    except IOError as e:
        logger.warning(f"Could not read end of file {file_path} to check completion: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking file completion for {file_path}: {e}", exc_info=True)
        return False

# --- ADDED: TOC Extraction Function (Unchanged) ---
def extract_toc(pdf_path: Path) -> Optional[str]:
    """Extracts the table of contents (outline) from a PDF."""
    toc_entries = []
    try:
        reader = PyPDF2.PdfReader(str(pdf_path))
        if reader.outline:
            logger.info(f"Extracting Table of Contents (Outline) from '{pdf_path.name}'...")

            def _recursive_outline_extract(outline_items, level=0):
                for item in outline_items:
                    if isinstance(item, list): # Recursive call for nested items
                        _recursive_outline_extract(item, level + 1)
                    else: # It's a Destination object
                        try:
                            page_num_0_indexed = item.page_number # PyPDF2 v3+
                            page_num_1_indexed = page_num_0_indexed + 1
                            title = item.title.strip() if item.title else "Untitled Section"
                            indent = "  " * level
                            toc_entries.append(f"{indent}- {title} (Page {page_num_1_indexed})")
                        except AttributeError: # Fallback for older PyPDF2 or different structure
                             try:
                                 page_obj = item.page
                                 page_num_0_indexed = reader.get_page_number(page_obj)
                                 page_num_1_indexed = page_num_0_indexed + 1
                                 title = item.title.strip() if item.title else "Untitled Section"
                                 indent = "  " * level
                                 toc_entries.append(f"{indent}- {title} (Page {page_num_1_indexed})")
                             except Exception as e_item_fallback:
                                logger.warning(f"Could not process TOC item '{getattr(item, 'title', 'N/A')}' (fallback attempt failed): {e_item_fallback}")
                        except Exception as e_item:
                            logger.warning(f"Could not process TOC item '{getattr(item, 'title', 'N/A')}': {e_item}")

            _recursive_outline_extract(reader.outline)

            if toc_entries:
                toc_text = "\n".join(toc_entries)
                logger.info(f"Successfully extracted {len(toc_entries)} TOC entries.")
                logger.debug(f"Extracted TOC:\n{toc_text[:500]}...") # Log beginning
                return toc_text
            else:
                logger.warning(f"PDF '{pdf_path.name}' has an outline structure, but no entries could be extracted.")
                return None
        else:
            logger.warning(f"No Table of Contents (Outline) found in PDF: '{pdf_path.name}'")
            return None
    except PdfReadError as e:
        logger.error(f"Error reading PDF '{pdf_path.name}' for TOC extraction: {e}. Is it password protected or corrupted?")
        return None
    except DependencyError as e:
         logger.error(f"PyPDF2 dependency error reading '{pdf_path.name}': {e}. Check crypto library (e.g., 'pip install pypdf2[crypto]') if encrypted.")
         return None
    except Exception as e:
        logger.error(f"Unexpected error extracting TOC from '{pdf_path.name}': {e}", exc_info=True)
        return None

# --- ADDED: TOC Analysis Function (Unchanged) ---
def get_main_content_page_range(
    toc_text: str,
    analysis_model: BaseModelBackend,
    prompt_template: str = PROMPT_TOC_ANALYSIS
) -> Tuple[Optional[int], Optional[int]]:
    """Uses an AI model to analyze TOC text and determine the main content page range."""
    logger.info(f"Analyzing TOC with model: {analysis_model.model_type.value}")
    final_prompt = prompt_template.format(toc_text=toc_text)

    start_page, end_page = None, None
    try:
        retry_delay = 2
        max_retries = 2 # Fewer retries for analysis
        retries = 0
        raw_response = None

        while retries < max_retries:
            try:
                # Use the model's run method directly for simpler interaction
                messages = [{"role": "user", "content": final_prompt}]
                response_obj = analysis_model.run(messages=messages)

                # Extract content correctly based on CAMEL's response structure
                if response_obj and response_obj.choices and response_obj.choices[0].message:
                     raw_response = response_obj.choices[0].message.content
                     logger.debug(f"Raw analysis response received (attempt {retries + 1}):\n{raw_response}")
                     break # Success
                else:
                     logger.warning(f"TOC analysis model returned invalid response structure (attempt {retries+1}).")
                     raw_response = None # Ensure it's None

            except Exception as e_inner:
                 logger.warning(f"TOC analysis API call failed (attempt {retries + 1}/{max_retries}): {e_inner}. Retrying in {retry_delay}s...")

            # Retry logic
            sleep(retry_delay)
            retry_delay *= 2
            retries += 1

        if not raw_response:
             logger.error(f"Failed to get valid response from TOC analysis model after {max_retries} attempts.")
             return None, None

        # Attempt to extract JSON from the response
        try:
            # Be robust: find JSON block even if there's surrounding text
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_response, re.DOTALL | re.IGNORECASE)
            if not json_match:
                json_match = re.search(r"(\{.*?\})", raw_response, re.DOTALL) # Try finding any JSON block

            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                logger.info(f"Parsed TOC analysis result: {data.get('reasoning', 'No reasoning provided.')}")

                s_page = data.get("start_page")
                e_page = data.get("end_page")

                # Validate and convert to int
                if isinstance(s_page, int) and s_page > 0:
                    start_page = s_page
                elif s_page is not None:
                    logger.warning(f"Invalid start_page value received: {s_page}. Ignoring.")

                if isinstance(e_page, int) and e_page > 0:
                    end_page = e_page
                elif e_page is not None:
                    logger.warning(f"Invalid end_page value received: {e_page}. Ignoring.")

                # Basic sanity check
                if start_page is not None and end_page is not None and start_page > end_page:
                    logger.warning(f"Analysis resulted in start_page ({start_page}) > end_page ({end_page}). Discarding range.")
                    return None, None

                logger.info(f"Determined main content range: Start={start_page}, End={end_page}")
                return start_page, end_page
            else:
                logger.error(f"Could not find JSON block in TOC analysis response:\n{raw_response}")
                return None, None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from TOC analysis response: {e}\nResponse was:\n{raw_response}")
            return None, None
        except Exception as e_parse:
             logger.error(f"Error processing TOC analysis JSON data: {e_parse}\nResponse was:\n{raw_response}", exc_info=True)
             return None, None

    except Exception as e:
        logger.error(f"Unhandled error during TOC analysis: {e}", exc_info=True)
        return None, None

# --- System Temp File Cleanup (Unchanged) ---
def cleanup_system_temp_files():
    """Finds and deletes temporary files created by the toolkit in /tmp."""
    logger.debug(f"Attempting periodic cleanup of '{SYSTEM_TEMP_DIR_PATTERN}'...")
    count = 0
    try:
        for temp_file_path_str in glob.iglob(SYSTEM_TEMP_DIR_PATTERN):
            try:
                p = Path(temp_file_path_str)
                if p.is_file():
                    p.unlink()
                    logger.debug(f"Deleted system temporary file: {p.name}")
                    count += 1
                else:
                    logger.debug(f"Skipping non-file entry found by glob: {p.name}")
            except FileNotFoundError:
                 logger.debug(f"System temporary file already gone: {temp_file_path_str}")
            except OSError as e_unlink:
                logger.warning(f"Could not delete system temporary file {temp_file_path_str}: {e_unlink}")
            except Exception as e_unlink_other:
                 logger.warning(f"Unexpected error deleting system temporary file {temp_file_path_str}: {e_unlink_other}")
        if count > 0:
            logger.debug(f"Periodic cleanup removed {count} system temporary file(s).")
    except Exception as e_glob:
        logger.error(f"Error during periodic cleanup glob operation: {e_glob}")

# --- NEW: Get PDF Page Count ---
def get_pdf_page_count(pdf_path: Path) -> Optional[int]:
    """Gets the total number of pages in a PDF using PyPDF2."""
    try:
        reader = PyPDF2.PdfReader(str(pdf_path))
        num_pages = len(reader.pages)
        logger.debug(f"PDF '{pdf_path.name}' has {num_pages} pages.")
        return num_pages
    except PdfReadError as e:
        logger.error(f"Error reading PDF '{pdf_path.name}' to get page count: {e}. Encrypted or corrupted?")
        return None
    except DependencyError as e:
        logger.error(f"PyPDF2 dependency error reading '{pdf_path.name}' for page count: {e}.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting page count for '{pdf_path.name}': {e}", exc_info=True)
        return None

# --- Main Logic Function ---
# --- MODIFIED: run_main_logic incorporates persistent page range ---
def run_main_logic(args):
    # Set up logging based on args
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level_int) # Set root logger level
    formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] %(message)s')
    root_logger = logging.getLogger()
    # Clear existing handlers to avoid duplicate logs if run multiple times in same session
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    # Add console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level_int)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)
    # Set this script's logger level as well
    logger.setLevel(log_level_int)

    # --- Input validation ---
    if args.input_dir and args.book_dir:
        logger.error("Cannot specify both --input-dir and --book-dir.")
        sys.exit(1)
    if not args.input_dir and not args.book_dir:
         logger.error("Must specify either --input-dir or --book-dir.")
         sys.exit(1)

    source_path_str = args.input_dir if args.input_dir else args.book_dir
    source_dir = Path(source_path_str).resolve()
    if not source_dir.is_dir():
        logger.error(f"Input source directory not found: {source_dir}")
        sys.exit(1)

    process_pdfs = bool(args.book_dir)
    mode_str = "PDFs in book directory" if process_pdfs else "images in directory"
    logger.info(f"Mode: Processing {mode_str}: {source_dir}")

    user_start_page_arg = args.start_page
    user_end_page_arg = args.end_page

    if (user_start_page_arg is not None and user_end_page_arg is None) or \
       (user_start_page_arg is None and user_end_page_arg is not None):
        logger.error("Both --start-page and --end-page must be provided together if specified by user.")
        sys.exit(1)
    if user_start_page_arg is not None and user_end_page_arg is not None and user_start_page_arg > user_end_page_arg:
        logger.error(f"User-provided start page ({user_start_page_arg}) cannot be > end page ({user_end_page_arg}).")
        sys.exit(1)
    if user_start_page_arg is not None and user_start_page_arg < 1:
         logger.error(f"User-provided start page ({user_start_page_arg}) must be 1 or greater.")
         sys.exit(1)

    # --- Output path setup ---
    single_output_file_path: Optional[Path] = None
    output_directory_path: Optional[Path] = None
    resume_state_file_path: Optional[Path] = None
    final_append_mode = args.append

    if not process_pdfs: # Image directory mode (No change needed here)
        if args.output:
            single_output_file_path = Path(args.output).resolve()
            logger.info(f"Output will be to SINGLE file: {single_output_file_path}")
        else:
            logger.info("Output will be to console.")
        # --- Add warning for image mode ---
        logger.warning("Processing in image directory mode. Assumes images are pre-generated.")
        logger.warning("The 'rolling window' optimization only applies when processing PDFs directly (--book-dir mode).")
        logger.warning("Persistent page range detection/storage is not applicable in image directory mode.") # Added warning

    else: # PDF Mode
        if args.output:
            output_directory_path = Path(args.output).resolve()
        else:
            output_directory_path = source_dir / "books_tts"
        try:
            output_directory_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory for books: {output_directory_path}")
            # --- MODIFIED: State file path defined here ---
            resume_state_file_path = output_directory_path / RESUME_STATE_FILENAME
            logger.info(f"Resume state will be managed in: {resume_state_file_path}")
        except OSError as e:
            logger.error(f"Failed to create output directory {output_directory_path}: {e}")
            sys.exit(1)

    # --- Model selection and initialization (Unchanged logic, uses helpers) ---
    parsed_transcription_model_repr = parse_model_string_to_enum(args.transcription_model, "Transcription")
    parsed_analysis_model_repr = parse_model_string_to_enum(args.analysis_model, "Analysis") # ADDED

    # --- Validate Parsed Models ---
    if parsed_transcription_model_repr is None:
        logger.critical("Invalid transcription model type provided. Cannot proceed.")
        sys.exit(1)
    # Analysis model is only critical if processing PDFs
    analysis_model_needed_for_auto = process_pdfs and not (user_start_page_arg and user_end_page_arg)
    if analysis_model_needed_for_auto and parsed_analysis_model_repr is None:
         logger.critical("Invalid analysis model type provided, required for automatic page range detection in PDF mode (when --start/--end are not used). Cannot proceed.")
         sys.exit(1)

    # --- Determine Platforms ---
    transcription_platform = get_platform_from_model_type(parsed_transcription_model_repr)
    analysis_platform = get_platform_from_model_type(parsed_analysis_model_repr) if parsed_analysis_model_repr else None

    # --- Validate Platforms ---
    if transcription_platform is None:
        logger.critical(f"Could not determine platform for transcription model: {args.transcription_model}. Cannot proceed.")
        sys.exit(1)
    if analysis_model_needed_for_auto and analysis_platform is None:
        logger.critical(f"Could not determine platform for required analysis model: {args.analysis_model}. Cannot proceed.")
        sys.exit(1)

    logger.info(f"Transcription: Model='{args.transcription_model}', Representation={type(parsed_transcription_model_repr).__name__}, Platform={transcription_platform}")
    if parsed_analysis_model_repr: # Log analysis info only if it was parsed
         logger.info(f"Analysis: Model='{args.analysis_model}', Representation={type(parsed_analysis_model_repr).__name__}, Platform={analysis_platform or 'N/A (not needed or platform undetermined)'}")

    # --- Initialize models and toolkits ---
    transcription_toolkit: Optional[ImageAnalysisToolkit] = None
    analysis_model_backend: Optional[BaseModelBackend] = None

    try:
        logger.info(f"Initializing Transcription model...")
        transcription_model = ModelFactory.create(
            model_platform=transcription_platform,
            model_type=parsed_transcription_model_repr
        )
        transcription_toolkit = ImageAnalysisToolkit(model=transcription_model)
        logger.info(f"Initialized Transcription Toolkit with model: {transcription_model.model_type}")

        # Initialize analysis model ONLY IF it's needed and platform could be determined
        if analysis_model_needed_for_auto and analysis_platform and parsed_analysis_model_repr:
            logger.info(f"Initializing Analysis model...")
            analysis_model_backend = ModelFactory.create(
                model_platform=analysis_platform,
                model_type=parsed_analysis_model_repr
            )
            logger.info(f"Initialized Analysis Model Backend: {analysis_model_backend.model_type}")
        elif analysis_model_needed_for_auto:
            # This case should be caught earlier by platform validation, but double-check
            logger.critical("Analysis model needed but could not be initialized (platform/model issue).")
            sys.exit(1)
        else:
            logger.info("Analysis model not needed or not specified.")


    except ValueError as ve:
        logger.error(f"Failed to initialize models or toolkits: {ve}", exc_info=True)
        logger.error(f"Check model string ('{args.transcription_model}'/'{args.analysis_model}'), inferred platform, and API keys in .env.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize models or toolkits: {e}", exc_info=True)
        sys.exit(1)

    # --- Final Validation after Initialization Attempt ---
    if transcription_toolkit is None:
         logger.critical("Transcription toolkit initialization failed unexpectedly.")
         sys.exit(1)
    # Check if analysis model is needed but wasn't initialized
    if analysis_model_needed_for_auto and analysis_model_backend is None:
         logger.critical("Analysis model required for auto page range but failed to initialize.")
         sys.exit(1)


    # --- Main Processing Logic ---
    if not process_pdfs:
        # --- Single Image Directory Processing (Handles pre-generated images) ---
        # (This section remains largely unchanged as state/range persistence doesn't apply)
        logger.info(f"Processing pre-existing images directly from: {source_dir}")
        # --- Load context if appending ---
        overall_initial_context: Optional[str] = None
        if single_output_file_path and final_append_mode:
            if single_output_file_path.exists() and single_output_file_path.stat().st_size > 0:
                try:
                    with open(single_output_file_path, 'r', encoding='utf-8') as f:
                        read_size = 4096
                        f.seek(max(0, single_output_file_path.stat().st_size - read_size))
                        overall_initial_context = f.read().strip() or None
                        if overall_initial_context: logger.info(f"Loaded context (last ~{len(overall_initial_context)} chars) from single file.")
                except Exception as e: logger.error(f"Error reading context from {single_output_file_path}: {e}")
            else: logger.info(f"Append mode active, but output file {single_output_file_path} missing/empty. Starting fresh.")
            # Add separator
            try:
                 if single_output_file_path.exists():
                     with open(single_output_file_path, 'a', encoding='utf-8') as f:
                         separator = f"\n\n{'='*20} Appending image dir transcriptions at {datetime.datetime.now()} {'='*20}\n\n"
                         f.write(separator)
            except IOError as e: logger.error(f"Error adding append separator to {single_output_file_path}: {e}")

        # Find and sort existing images using their extracted numbers
        image_files_with_nums = []
        supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.ppm"]
        for ext in supported_extensions:
            for file_path in source_dir.glob(ext):
                page_num = extract_page_number(file_path)
                if page_num is not None:
                    image_files_with_nums.append((page_num, file_path))
                else:
                    logger.debug(f"Skipping file due to non-standard name format or failed number extraction: {file_path.name}")

        if not image_files_with_nums:
            logger.warning(f"No image files with extractable page numbers found in {source_dir}. Exiting.")
            sys.exit(0)

        image_files_with_nums.sort(key=lambda x: x[0])
        image_files = [item[1] for item in image_files_with_nums] # Sorted list of Paths
        logger.info(f"Found {len(image_files)} valid images to process in {source_dir.name}.")

        files_processed_count = 0
        last_successful_transcription = overall_initial_context
        processing_successful = True
        temp_composite_dir = source_dir / ".composites_img_mode" # Separate composite dir

        try:
            temp_composite_dir.mkdir(exist_ok=True)
        except OSError as e_comp_dir:
            logger.error(f"Could not create composite dir {temp_composite_dir}: {e_comp_dir}. Cannot proceed.")
            sys.exit(1)

        # Prepare output file
        if single_output_file_path and not final_append_mode:
             try:
                 single_output_file_path.parent.mkdir(parents=True, exist_ok=True)
                 with open(single_output_file_path, 'w', encoding='utf-8') as f_init: f_init.truncate(0)
                 logger.info(f"Output file {single_output_file_path} truncated (or created).")
             except IOError as e: logger.error(f"Error preparing output file {single_output_file_path}: {e}"); sys.exit(1)

        # Loop through EXISTING images
        for i, current_image_path in enumerate(image_files):
             middle_page_name = current_image_path.name
             # Extract page number *again* for range check (already extracted for sorting)
             page_num = extract_page_number(current_image_path)
             if page_num is None:
                 logger.warning(f"Could not extract page number from image {middle_page_name} during processing loop, skipping.")
                 continue

             # Apply user range if provided
             if user_start_page_arg is not None and page_num < user_start_page_arg:
                  logger.debug(f"Skipping image {middle_page_name} (page {page_num}): before user start {user_start_page_arg}.")
                  continue
             if user_end_page_arg is not None and page_num > user_end_page_arg:
                  logger.info(f"Reached user end page {user_end_page_arg}. Stopping image processing.")
                  break

             # Check for 3-image window (using indices)
             can_get_prev = i > 0
             can_get_next = i < len(image_files) - 1
             if not can_get_prev or not can_get_next:
                  logger.info(f"Skipping edge image {middle_page_name}: Cannot form 3-page set.")
                  continue

             logger.info(f"Processing image {middle_page_name} (Page: {page_num}, Index: {i})")
             files_processed_count += 1

             # Create composite from EXISTING images
             montage_input_paths = [image_files[i-1], image_files[i], image_files[i+1]]
             base_name = f"img_page_{page_num:04d}" # Use page num for composite name base
             temp_composite_img_path = None

             try:
                  temp_composite_img_path = create_temporary_composite_image(
                      montage_input_paths, temp_composite_dir, base_name
                  )

                  if temp_composite_img_path and temp_composite_img_path.exists():
                      transcription = None
                      try:
                          # Transcribe
                          transcription = transcribe_image(
                              toolkit=transcription_toolkit,
                              image_path=temp_composite_img_path,
                              original_middle_page_name=middle_page_name,
                              previous_transcription=last_successful_transcription,
                              prompt_template=PROMPT_TEMPLATE
                          )

                          if transcription:
                              transcription = transcription.strip()
                              if transcription.upper() == "SKIPPED":
                                  logger.info(f"SKIPPED image {middle_page_name} due to ancillary content.")
                              else:
                                  # Write output
                                  if single_output_file_path:
                                      try:
                                          with open(single_output_file_path, 'a', encoding='utf-8') as f:
                                              f.write(transcription + "\n\n")
                                          logger.info(f"Appended transcription for image {middle_page_name} to {single_output_file_path.name}")
                                      except IOError as e:
                                          logger.error(f"Error writing transcription for {middle_page_name} to {single_output_file_path}: {e}")
                                          processing_successful = False
                                  else: # Output to console
                                      print(f"--- Transcription for Image {middle_page_name} ---")
                                      print(transcription)
                                      print("--- End Transcription ---\n")

                                  # Update context ONLY if successful AND NOT skipped
                                  last_successful_transcription = transcription
                          else:
                              logger.warning(f"--- Transcription FAILED for image {middle_page_name} ---")
                              processing_successful = False

                      except Exception as e_transcribe:
                           logger.error(f"Error during transcription/writing for image {middle_page_name}: {e_transcribe}", exc_info=True)
                           processing_successful = False
                  else:
                      logger.error(f"Failed to create composite image for {middle_page_name}. Skipping.")
                      processing_successful = False

             except Exception as e_outer:
                  logger.error(f"Outer loop error for image {middle_page_name}: {e_outer}", exc_info=True)
                  processing_successful = False
             finally:
                  if temp_composite_img_path and temp_composite_img_path.exists():
                      try:
                          temp_composite_img_path.unlink()
                          logger.debug(f"Deleted temporary composite: {temp_composite_img_path.name}")
                      except OSError as e_del_comp:
                          logger.warning(f"Could not delete temporary composite {temp_composite_img_path.name}: {e_del_comp}")

             # No rolling delete needed in this mode

        # Cleanup composite dir for image mode
        if temp_composite_dir.exists():
             try: shutil.rmtree(temp_composite_dir)
             except OSError as e_final_clean: logger.warning(f"Could not clean up composites directory {temp_composite_dir}: {e_final_clean}")

        if not processing_successful:
             logger.error(f"Processing encountered errors for image directory: {source_dir}")
        logger.info("Image directory processing finished.")

    else:
        # --- PDF Directory Processing with Rolling Window ---
        pdf_files = sorted(list(source_dir.glob('*.pdf')))
        if not pdf_files:
            logger.warning(f"No PDF files found in book directory: {source_dir}")
            sys.exit(0)

        logger.info(f"Found {len(pdf_files)} PDFs to process.")
        # --- MODIFIED: Load state file ---
        current_state_data = load_resume_state(resume_state_file_path) if resume_state_file_path else {}
        images_base_dir = source_dir / TEMP_IMAGE_SUBDIR # Parent dir for all PDF temp images

        for pdf_path in pdf_files:
            pdf_filename = pdf_path.name
            logger.info(f"--- Starting processing for PDF: {pdf_filename} ---")
            book_specific_output_path = output_directory_path / f"{pdf_path.stem}.txt"
            pdf_image_dir_path = images_base_dir / pdf_path.stem # Specific subdir for this PDF's images

            logger.info(f"Output for '{pdf_filename}' will be: {book_specific_output_path}")
            logger.info(f"Temporary images for '{pdf_filename}' will be in: {pdf_image_dir_path}")

            if check_if_file_completed(book_specific_output_path, END_MARKER):
                logger.info(f"'{pdf_filename}' already marked as completed ('{END_MARKER}' found). Skipping.")
                # --- MODIFIED: Remove completed book from state if present ---
                if resume_state_file_path and pdf_filename in current_state_data:
                    logger.info(f"Removing completed book '{pdf_filename}' from resume state.")
                    del current_state_data[pdf_filename]
                    save_resume_state(resume_state_file_path, current_state_data)
                # Clean up leftover image dir if it exists and not keeping images
                if pdf_image_dir_path.exists() and not args.keep_images:
                    try:
                        shutil.rmtree(pdf_image_dir_path)
                        logger.debug(f"Cleaned up leftover image directory for completed PDF: {pdf_image_dir_path}")
                    except OSError as e_clean_comp:
                         logger.warning(f"Could not clean up leftover image dir {pdf_image_dir_path}: {e_clean_comp}")
                continue

            # --- Get Total Page Count ---
            num_pdf_pages = get_pdf_page_count(pdf_path)
            if num_pdf_pages is None:
                logger.error(f"Could not determine page count for '{pdf_filename}'. Skipping this PDF.")
                continue

            # --- Determine Base Page Range (User Args OR State OR Auto-Detect) ---
            base_start_page = None
            base_end_page = None
            range_source = "undetermined"
            analysis_performed_this_run = False # Flag to avoid saving state unnecessarily

            # --- Get the specific state for this PDF ---
            pdf_state = current_state_data.get(pdf_filename, {})

            # Priority 1: User Arguments
            if user_start_page_arg is not None and user_end_page_arg is not None:
                base_start_page = user_start_page_arg
                base_end_page = user_end_page_arg
                range_source = "user"
                logger.info(f"Using user-provided page range for '{pdf_filename}': {base_start_page}-{base_end_page}")
                # User override: Clear any detected range in state if it exists
                if "detected_start" in pdf_state or "detected_end" in pdf_state:
                    logger.debug(f"Clearing previously detected range from state for '{pdf_filename}' due to user override.")
                    pdf_state.pop("detected_start", None)
                    pdf_state.pop("detected_end", None)
                    # Save state immediately if changes were made
                    if resume_state_file_path:
                        current_state_data[pdf_filename] = pdf_state
                        save_resume_state(resume_state_file_path, current_state_data)


            # Priority 2: Load from State
            elif "detected_start" in pdf_state or "detected_end" in pdf_state:
                 # Check if *both* are present and potentially valid (allow None)
                 loaded_start = pdf_state.get("detected_start")
                 loaded_end = pdf_state.get("detected_end")
                 # Basic validation: check if they are ints or None
                 if (isinstance(loaded_start, int) or loaded_start is None) and \
                    (isinstance(loaded_end, int) or loaded_end is None):
                      base_start_page = loaded_start
                      base_end_page = loaded_end
                      range_source = "state"
                      logger.info(f"Using page range from state file for '{pdf_filename}': Start={base_start_page}, End={base_end_page}")
                 else:
                      logger.warning(f"Invalid page range data found in state for '{pdf_filename}': Start={loaded_start}, End={loaded_end}. Will attempt auto-detection.")
                      # Clear invalid state entries
                      pdf_state.pop("detected_start", None)
                      pdf_state.pop("detected_end", None)
                      range_source = "state_invalid" # Fall through to analysis

            # Priority 3: Auto-Detect (if user args not given AND not found/invalid in state)
            if range_source in ["undetermined", "state_invalid"]:
                logger.info(f"No valid page range from user or state for '{pdf_filename}'. Attempting automatic detection...")
                range_source = "auto"
                toc_text = extract_toc(pdf_path)
                if toc_text and analysis_model_backend:
                    detected_start, detected_end = get_main_content_page_range(
                        toc_text, analysis_model_backend
                    )
                    analysis_performed_this_run = True # Analysis was attempted
                    if detected_start is not None or detected_end is not None:
                         logger.info(f"Auto-detected range for '{pdf_filename}': Start={detected_start}, End={detected_end}")
                         base_start_page = detected_start # Can be None
                         base_end_page = detected_end   # Can be None
                         # --- MODIFIED: Store detected range in state ---
                         pdf_state["detected_start"] = base_start_page
                         pdf_state["detected_end"] = base_end_page
                    else:
                         logger.warning(f"Automatic page range detection failed for '{pdf_filename}'. Processing all pages (subject to resume).")
                         range_source = "failed_auto"
                         # --- MODIFIED: Store failure (null) in state ---
                         pdf_state["detected_start"] = None
                         pdf_state["detected_end"] = None
                else:
                     logger.warning(f"Could not extract TOC or analysis model not available for '{pdf_filename}'. Processing all pages (subject to resume).")
                     range_source = "no_toc"
                     analysis_performed_this_run = True # Attempted but failed due to missing prereq
                     # --- MODIFIED: Store failure (null) in state ---
                     pdf_state["detected_start"] = None
                     pdf_state["detected_end"] = None

                # --- MODIFIED: Save state immediately after analysis attempt ---
                if analysis_performed_this_run and resume_state_file_path:
                    current_state_data[pdf_filename] = pdf_state # Ensure pdf_state is in main dict
                    save_resume_state(resume_state_file_path, current_state_data)
                    logger.info(f"Saved analysis result (or failure) to state file for '{pdf_filename}'.")


            # --- Determine Effective Page Range & Resumption ---
            last_processed_page = pdf_state.get("last_processed") # Get from specific PDF state
            resuming = last_processed_page is not None
            resume_start_page = (last_processed_page + 1) if resuming else 1

            logger.debug(f"Range Source: {range_source}. Base range: {base_start_page}-{base_end_page}. Resume state: last_processed={last_processed_page}")

            # Calculate final effective start page
            effective_start_page = base_start_page # Start with base (can be None)
            if resuming:
                 # Ensure resume starts *at or after* the base start page
                 effective_start_page = max(resume_start_page, base_start_page or 1)
            elif base_start_page is None:
                 effective_start_page = 1 # If no base, start at 1

            # Effective end is the base end, capped by total pages
            effective_end_page = base_end_page if base_end_page is not None else num_pdf_pages
            effective_end_page = min(effective_end_page, num_pdf_pages)

            logger.info(f"Effective processing range for '{pdf_filename}': Start={effective_start_page}, End={effective_end_page}")

            if effective_start_page > effective_end_page:
                 logger.warning(f"Effective start page {effective_start_page} is after effective end page {effective_end_page} for '{pdf_filename}'. Nothing to process.")
                 # Check if this is because we resumed *after* the end page
                 if resuming and last_processed_page >= effective_end_page:
                     logger.info(f"PDF '{pdf_filename}' already processed up to or beyond effective end page based on resume state.")
                     # Consider marking as complete here? For now, just skip.
                 continue

            # --- Load Context for Append/Resume ---
            current_book_initial_context = None
            load_context_needed = resuming or final_append_mode
            if load_context_needed and book_specific_output_path.exists() and book_specific_output_path.stat().st_size > 0:
                try:
                    logger.info(f"Reading context from existing file: {book_specific_output_path} (Reason: {'resume' if resuming else 'append'})")
                    with open(book_specific_output_path, 'r', encoding='utf-8') as bf:
                        read_size = 4096
                        bf.seek(max(0, book_specific_output_path.stat().st_size - read_size))
                        current_book_initial_context = bf.read().strip() or None
                        if current_book_initial_context: logger.info(f"Loaded context for book {pdf_filename}. Length: {len(current_book_initial_context)}")
                        else: logger.warning(f"Could not read context from existing file: {book_specific_output_path}")
                except Exception as e: logger.error(f"Error reading context for {book_specific_output_path}: {e}")
            elif load_context_needed:
                 logger.info(f"Append/Resume mode active, but file {book_specific_output_path} missing/empty. Starting context fresh.")

            # Add append separator if explicitly appending and file exists
            if final_append_mode and book_specific_output_path.exists() and book_specific_output_path.stat().st_size > 0 and not resuming: # Only add separator if NOT resuming
                 try:
                     with open(book_specific_output_path, 'a', encoding='utf-8') as bf:
                         separator = f"\n\n{'='*20} Appending ({range_source}) at {datetime.datetime.now()} {'='*20}\n\n"
                         bf.write(separator)
                 except IOError as e:
                    logger.error(f"Error preparing file {book_specific_output_path} for appending: {e}. Skipping book.")
                    continue

            # --- Process the PDF using Rolling Window ---
            pdf_processing_success = False
            try:
                # Ensure image directory exists (might have been cleaned if completed before)
                pdf_image_dir_path.mkdir(parents=True, exist_ok=True) # <<< ENSURE DIR EXISTS >>>

                # Determine the actual mode for process_pdf_pages's append logic
                process_append_mode = resuming or final_append_mode

                # <<< REMOVED BULK CONVERSION CALL >>>

                # Call the refactored PDF page processing function
                pdf_processing_success, _ = process_pdf_pages(
                    pdf_path=pdf_path,
                    num_pdf_pages=num_pdf_pages,
                    pdf_image_dir=pdf_image_dir_path, # Pass the dedicated dir
                    output_path=book_specific_output_path,
                    append_mode=process_append_mode, # Use calculated append mode
                    start_page=effective_start_page,
                    end_page=effective_end_page,
                    image_toolkit=transcription_toolkit,
                    prompt_template=PROMPT_TEMPLATE,
                    initial_context=current_book_initial_context,
                    pdf_filename=pdf_filename,
                    state_file_path=resume_state_file_path, # Pass state file path
                    state_data=current_state_data, # Pass the entire state dict
                    image_format=args.image_format, # Pass from args
                    dpi=args.dpi # Pass from args
                )

                if pdf_processing_success:
                    logger.info(f"Successfully processed PDF: {pdf_filename}")
                    try:
                        with open(book_specific_output_path, 'a', encoding='utf-8') as f_end:
                            f_end.write(f"\n\n{END_MARKER}\n")
                        logger.info(f"Added '{END_MARKER}' to completed file: {book_specific_output_path.name}")
                        # --- MODIFIED: Clear state for completed book ---
                        if resume_state_file_path and pdf_filename in current_state_data:
                             logger.info(f"Removing completed book '{pdf_filename}' from resume state.")
                             del current_state_data[pdf_filename]
                             save_resume_state(resume_state_file_path, current_state_data)

                    except IOError as e:
                         logger.error(f"Error adding '{END_MARKER}' to {book_specific_output_path.name}: {e}")
                         pdf_processing_success = False # Mark as failed if end marker fails
                else:
                    logger.error(f"Processing failed for PDF: {pdf_filename}.")

            except Exception as e:
                 logger.critical(f"Critical error processing PDF {pdf_filename}: {e}", exc_info=True)
            finally:
                # --- Cleanup image directory for this PDF ---
                if pdf_image_dir_path.exists() and not args.keep_images:
                    try:
                        shutil.rmtree(pdf_image_dir_path)
                        logger.info(f"Cleaned up PDF image directory: {pdf_image_dir_path}")
                    except Exception as e_clean:
                         logger.error(f"Error cleaning up PDF image dir {pdf_image_dir_path}: {e_clean}")
                elif pdf_image_dir_path and args.keep_images:
                    logger.info(f"Keeping PDF image directory due to --keep-images flag: {pdf_image_dir_path}")

            logger.info(f"--- Finished processing for PDF: {pdf_filename} ---")

        # --- Final cleanup of the main temp image subdir ---
        if images_base_dir.exists() and not args.keep_images:
            try:
                # Check if it's empty first (might be if all subdirs were cleaned)
                if not any(images_base_dir.iterdir()):
                     images_base_dir.rmdir()
                     logger.info(f"Cleaned up empty base image directory: {images_base_dir}")
                else: # Should not happen if sub-cleanups work, but just in case
                     logger.warning(f"Base image directory {images_base_dir} not empty after processing, leaving it.")
            except Exception as e_clean_base:
                 logger.error(f"Error during final cleanup of base image directory {images_base_dir}: {e_clean_base}")
        elif images_base_dir.exists() and args.keep_images:
            logger.info(f"Keeping base image directory due to --keep-images flag: {images_base_dir}")

        logger.info("All PDF processing finished.")


# --- Script Entry Point ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    DEFAULT_TRANSCRIPTION_MODEL = ModelType.GEMINI_2_0_FLASH_LITE_PREVIEW.value
    DEFAULT_ANALYSIS_MODEL = ModelType.GEMINI_2_5_PRO_EXP.value
    DEFAULT_IMAGE_FORMAT = 'png'
    DEFAULT_DPI = 300

    parser = argparse.ArgumentParser(
        description="Transcribe text from book page images or PDFs using Owl/CAMEL-AI.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--input-dir", help="Directory containing pre-generated image files (named 'prefix-digits.ext' or similar).")
    input_group.add_argument("-b", "--book-dir", help="Directory containing PDF book files to process (images generated on-the-fly).")

    parser.add_argument(
        "-o", "--output",
        help="Output path. Behavior depends on input type:\n"
             "  --input-dir: Path to the SINGLE output file (optional, defaults to console).\n"
             "  --book-dir: Path to the output DIRECTORY for '<pdf_name>.txt' files.\n"
             "              (optional, defaults to '<book_dir>/books_tts')."
    )
    parser.add_argument(
        "-a", "--append",
        action="store_true",
        help=f"Append to the output file(s). Reads context from end of existing file(s).\n"
              f"In --book-dir mode, also checks for '{END_MARKER}' to skip completed PDFs\n"
              f"and uses '{RESUME_STATE_FILENAME}' in the output dir to resume incomplete ones,\n"
              f"including previously detected page ranges." # Updated help text
    )
    parser.add_argument("--start-page", type=int, help="Manually specify the first page number (1-based, inclusive) to process. Overrides auto-detection and state.")
    parser.add_argument("--end-page", type=int, help="Manually specify the last page number (1-based, inclusive) to process. Overrides auto-detection and state.")

    parser.add_argument(
        "--transcription-model", type=str, default=DEFAULT_TRANSCRIPTION_MODEL,
        help=f"Model string for image transcription (default: '{DEFAULT_TRANSCRIPTION_MODEL}'). Matches ModelType enum or supported string."
    )
    parser.add_argument(
        "--analysis-model", type=str, default=DEFAULT_ANALYSIS_MODEL,
        help=f"Model string for TOC analysis (default: '{DEFAULT_ANALYSIS_MODEL}'). Used only in --book-dir mode if --start/--end-page are not provided AND range not found in state."
    )
    parser.add_argument(
        "--image-format", default=DEFAULT_IMAGE_FORMAT, choices=['png', 'jpeg', 'tiff', 'ppm'], # Added choices
        help=f"Format for temporary images generated from PDFs (default: {DEFAULT_IMAGE_FORMAT})."
    )
    parser.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI,
        help=f"Resolution (DPI) for temporary images generated from PDFs (default: {DEFAULT_DPI}). Higher DPI means better quality but larger files."
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)."
    )
    parser.add_argument(
        "--keep-images", action="store_true",
        help=f"Keep the temporary per-PDF image directories created in '{TEMP_IMAGE_SUBDIR}' within the book directory (only applies to --book-dir mode)."
    )

    # Generate epilog with available models (Unchanged)
    try:
        available_base_models = [f"'{m.value}'" for m in ModelType]
        available_custom_models = [f"'{m.value}'" for m in mymodel]
        all_available_models = sorted(list(set(available_base_models + available_custom_models)))
        available_models_str = ", ".join(all_available_models)
        model_help_epilog = f"\n\nAvailable ModelType/Unified strings for --transcription-model and --analysis-model:\n{available_models_str}"
    except Exception as e:
        logger.warning(f"Could not dynamically list all available models for help text: {e}")
        model_help_epilog = "\n\n(Could not list all available ModelType/Unified values)"

    parser.epilog = f"""
Example Usage:
  # Images (pre-generated) -> Single file (overwrite)
  python %(prog)s --input-dir path/to/pages -o output.txt
  # Images (pre-generated) -> Console
  python %(prog)s --input-dir path/to/pages
  # Images (pre-generated) -> Single file (append, specific model)
  python %(prog)s --input-dir path/to/pages -o output.txt --append --transcription-model {DEFAULT_TRANSCRIPTION_MODEL}

  # PDFs -> Specific output dir (AUTO page range [or state], overwrite)
  python %(prog)s --book-dir path/to/pdfs -o path/to/output_dir
  # PDFs -> Default output dir 'books_tts' (AUTO page range [or state], overwrite)
  python %(prog)s --book-dir path/to/pdfs
  # PDFs -> Specific dir (AUTO page range [or state], append/resume, specific models, jpeg format)
  python %(prog)s --book-dir path/to/pdfs -o path/to/output_dir --append \\
    --transcription-model {DEFAULT_TRANSCRIPTION_MODEL} --analysis-model {DEFAULT_ANALYSIS_MODEL} \\
    --image-format jpeg --dpi 200
  # PDFs -> Specific dir (MANUAL page range, append/resume)
  python %(prog)s --book-dir path/to/pdfs --start-page 50 --end-page 150 -o path/to/output_dir --append
  # PDFs -> Default dir (AUTO page range [or state], debug, append/resume, keep images)
  python %(prog)s --book-dir path/to/pdfs --log-level DEBUG --append --keep-images

Notes:
- Requires 'pdftoppm' (from poppler-utils) and 'montage' (from ImageMagick).
- Requires 'PyPDF2' (`pip install pypdf2`).
- **PDF Mode (--book-dir):** Images are generated on-the-fly for pages needed in the 3-page transcription window. Older images are deleted automatically unless --keep-images is used.
- **Image Mode (--input-dir):** Assumes images are already generated and correctly named (e.g., 'prefix-001.png', 'prefix-002.png'). Does not use the rolling window optimization. Requires images sorted numerically by page number in filename. Does not use state file or page range detection.
- Configure API keys in a .env file.
- Use model strings matching CAMEL's ModelType enum values or supported unified strings (like 'provider/model-name').
- Auto page range detection (in --book-dir mode without --start/--end-page):
    - Uses PyPDF2 to extract the Table of Contents (outline).
    - Uses the --analysis-model to determine the main content page range, excluding front/back matter.
    - Falls back to processing all pages if TOC extraction or analysis fails.
    - **Detected range is saved to the state file ({RESUME_STATE_FILENAME}) and reused on subsequent runs unless overridden by --start/--end-page.**
- --append mode (for --book-dir):
    - Checks for '{END_MARKER}' in existing .txt files to skip completed PDFs.
    - Uses '{RESUME_STATE_FILENAME}' in output dir to resume incomplete PDFs from last successful page **and** uses the stored detected page range (if present).
    - State is updated after *each* successful page transcription/skip.
    - Completed PDFs are removed from the state file.
- Interrupted processes in PDF mode may leave temporary image directories ({TEMP_IMAGE_SUBDIR}). Use --keep-images to prevent cleanup.
{model_help_epilog}
"""
    parsed_args = parser.parse_args()

    # Check ImageMagick early if processing PDFs
    if parsed_args.book_dir and shutil.which("montage") is None:
        logger.critical("Critical Error: 'montage' command not found (required for PDF processing). Please install ImageMagick.")
        sys.exit(1)
    # Check poppler-utils early if processing PDFs
    if parsed_args.book_dir and shutil.which("pdftoppm") is None:
        logger.critical("Critical Error: 'pdftoppm' command not found (required for PDF processing). Please install poppler-utils.")
        sys.exit(1)

    try:
        run_main_logic(parsed_args)
    finally:
        logger.info("Performing final cleanup check of system temporary files in /tmp...")
        cleanup_system_temp_files()
        logger.info("Script finished.")
