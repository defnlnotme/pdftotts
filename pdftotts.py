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
from typing import List, Optional, Tuple, Type, Dict, Any, Union
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
TEMP_IMAGE_SUBDIR = ".pdftotts_images"
SYSTEM_TEMP_DIR_PATTERN = "/tmp/image_*.png"

# --- Helper Functions ---
def extract_page_number(filename: Path) -> Optional[int]:
    """Extracts the page number from filenames like 'prefix-digits.ext'."""
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
                 # Check if parent dir name hints at being from pdftoppm
                 if filename.parent.name == filename.stem:
                     logger.debug(f"Assuming page number {match_alt.group(1)} from filename {filename.name} based on parent dir structure.")
                     return int(match_alt.group(1))
                 else:
                      logger.warning(f"Found digits {match_alt.group(1)} in {filename.name} but parent dir doesn't match stem, cannot confirm page number.")
                      return None
             except ValueError:
                 logger.warning(f"Could not convert extracted number to int in filename: {filename.name}")
                 return None
        else:
            logger.warning(f"Could not extract page number (format '-digits.' or 'digits.') from filename: {filename.name}")
            return None

def find_and_sort_image_files(directory: Path) -> List[Path]:
    """Finds image files and sorts them based on the extracted page number."""
    supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.ppm"]
    image_files_with_nums = []
    for ext in supported_extensions:
        for file_path in directory.glob(ext):
            page_num = extract_page_number(file_path)
            if page_num is not None:
                image_files_with_nums.append((page_num, file_path))
            else:
                logger.debug(f"Skipping file due to non-standard name format or failed number extraction: {file_path.name}")

    if not image_files_with_nums:
        logger.warning(f"No image files with extractable page numbers found in {directory}")
        return []

    image_files_with_nums.sort(key=lambda x: x[0])
    return [item[1] for item in image_files_with_nums]

def create_temporary_composite_image(image_paths: List[Path], output_dir: Path, base_name: str) -> Optional[Path]:
    """Creates a temporary composite image from 1-3 images using ImageMagick's montage tool."""
    if not image_paths or len(image_paths) != 3:
        logger.warning(f"Cannot create composite: Need exactly 3 image paths, got {len(image_paths)} for base {base_name}.")
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
        return None
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
                transcription = transcription.strip()
                transcription = re.sub(r'^```(text)?\s*', '', transcription, flags=re.IGNORECASE | re.MULTILINE)
                transcription = re.sub(r'\s*```$', '', transcription, flags=re.IGNORECASE | re.MULTILINE)
                transcription = transcription.strip('"\'')

                if transcription and "Analysis failed" not in transcription and len(transcription) > 5:
                    return transcription
                else:
                    logger.warning(f"Transcription attempt {retries + 1}/{max_retries} failed or produced short/invalid output for middle page: {original_middle_page_name}. Content: '{transcription[:50]}...'. Retrying in {retry_delay}s...")
                    transcription = None
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
        cleanup_system_temp_files()

def convert_pdf_to_images(pdf_path: Path, output_dir: Path, image_format: str = 'png', dpi: int = 300) -> bool:
    """Converts a PDF file to images using pdftoppm."""
    if not pdf_path.is_file():
        logger.error(f"PDF file not found: {pdf_path}")
        return False

    if shutil.which("pdftoppm") is None:
        logger.error("Error: 'pdftoppm' command not found. Please ensure poppler-utils is installed and in your system's PATH.")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    # Change output prefix to just the dir, pdftoppm adds the stem
    # output_prefix = output_dir / pdf_path.stem
    output_prefix_path_part = output_dir / pdf_path.stem # Path prefix for pdftoppm

    command = [
        "pdftoppm",
        f"-{image_format}",
        "-r", str(dpi),
        str(pdf_path),
        str(output_prefix_path_part) # Use path prefix
    ]

    try:
        logger.info(f"Converting PDF '{pdf_path.name}' to {image_format.upper()} images in '{output_dir.name}'...")
        logger.debug(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        logger.info(f"Successfully converted PDF: {pdf_path.name}")
        logger.debug(f"pdftoppm stdout: {result.stdout}")
        if result.stderr:
             logger.debug(f"pdftoppm stderr: {result.stderr}")

        # Verify output files were created
        expected_files = list(output_dir.glob(f"{pdf_path.stem}-*.{image_format}"))
        if not expected_files:
             logger.warning(f"pdftoppm ran but no output images found for {pdf_path.name} matching pattern '{pdf_path.stem}-*.{image_format}' in {output_dir}")
             # Attempt to list ANY files created to help debug
             all_files = list(output_dir.glob(f'*.*'))
             if all_files:
                  logger.warning(f"Files actually found in {output_dir}: {[f.name for f in all_files]}")
             else:
                  logger.warning(f"No files at all found in {output_dir} after pdftoppm.")
             return False
        else:
             logger.info(f"Found {len(expected_files)} image files after conversion.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting PDF '{pdf_path.name}' with command '{' '.join(command)}': {e}")
        logger.error(f"pdftoppm stdout: {e.stdout}")
        logger.error(f"pdftoppm stderr: {e.stderr}")
        return False
    except FileNotFoundError:
         logger.error("Error: 'pdftoppm' command not found during execution (should have been checked earlier).")
         return False
    except Exception as e:
        logger.error(f"Unexpected error during PDF conversion for {pdf_path.name}: {e}", exc_info=True)
        return False

# --- State Management Functions ---
def load_resume_state(state_file_path: Path) -> Dict[str, Any]:
    """Loads the resume state from a JSON file."""
    if not state_file_path.exists():
        return {}
    try:
        with open(state_file_path, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
            logger.info(f"Loaded resume state from {state_file_path}")
            return state_data
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
    """Saves the resume state to a JSON file."""
    try:
        state_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=4)
        logger.debug(f"Saved resume state to {state_file_path}")
    except IOError as e:
        logger.error(f"Error writing state file {state_file_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving state file {state_file_path}: {e}", exc_info=True)

# --- Image Processing Function ---
def process_images_in_directory(
    image_directory: Path,
    output_path: Optional[Path],
    append_mode: bool,
    start_page: Optional[int],
    end_page: Optional[int],
    image_toolkit: ImageAnalysisToolkit,
    prompt_template: str,
    initial_context: Optional[str],
    pdf_filename: Optional[str] = None,
    state_file_path: Optional[Path] = None,
    state_data: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Processes images, performs transcription within range, updates state if applicable.
    """
    logger.info(f"Processing directory: {image_directory.name} [Pages {start_page or 'start'} to {end_page or 'end'}]")

    image_files = find_and_sort_image_files(image_directory)
    if not image_files:
        logger.warning(f"No valid image files found in {image_directory}. Skipping.")
        return True, initial_context

    logger.info(f"Found {len(image_files)} valid images to potentially process in {image_directory.name}.")

    files_processed_count = 0
    last_successful_transcription = initial_context
    processing_successful = True

    temp_composite_dir = image_directory / ".composites"
    try:
        temp_composite_dir.mkdir(exist_ok=True)
        logger.info(f"Using temporary directory for composites: {temp_composite_dir}")
    except OSError as e_comp_dir:
        logger.error(f"Could not create composite dir {temp_composite_dir}: {e_comp_dir}. Cannot proceed.")
        return False, initial_context

    if output_path and not append_mode:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f_init:
                f_init.truncate(0)
            logger.info(f"Output file {output_path} truncated (or created).")
        except IOError as e:
            logger.error(f"Error preparing output file {output_path}: {e}")
            return False, initial_context

    try:
        for i, current_image_path in enumerate(image_files):
            middle_page_name = current_image_path.name
            page_num = extract_page_number(current_image_path)
            if page_num is None:
                logger.warning(f"Could not extract page number from {middle_page_name}, skipping.")
                continue

            if start_page is not None and page_num < start_page:
                logger.debug(f"Skipping page {page_num} ({middle_page_name}): before effective start {start_page}.")
                continue
            if end_page is not None and page_num > end_page:
                logger.info(f"Reached effective end page {end_page}. Stopping processing for this directory.")
                break

            # Need context from previous and next pages
            can_get_prev = i > 0
            can_get_next = i < len(image_files) - 1
            if not can_get_prev or not can_get_next:
                 logger.info(f"Skipping edge page {page_num} ({middle_page_name}): Cannot form 3-page set for transcription context.")
                 # Update state even for skipped edge page IF it's within range and state tracking is active
                 if pdf_filename and state_file_path and state_data is not None:
                      if page_num >= (start_page or 1): # Only update state if we are skipping a page we *would* have processed
                          logger.debug(f"Updating state to reflect skipped edge page {page_num}")
                          state_data[pdf_filename] = page_num
                          save_resume_state(state_file_path, state_data)
                 continue # Skip to next page

            logger.info(f"Processing page {page_num} ({middle_page_name}) (Index {i})")
            files_processed_count += 1

            montage_input_paths = [image_files[i-1], image_files[i], image_files[i+1]]
            base_name = f"page_{page_num:04d}"
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
                            transcription = transcription.strip()
                            if transcription.upper() == "SKIPPED":
                                logger.info(f"SKIPPED page {page_num} ({middle_page_name}) due to ancillary content.")
                                # --- Update State for SKIPPED page ---
                                if pdf_filename and state_file_path and state_data is not None:
                                    state_data[pdf_filename] = page_num # Update to the skipped page number
                                    save_resume_state(state_file_path, state_data)
                                    logger.debug(f"Updated resume state after skipping page {page_num}")
                            else:
                                if output_path:
                                    try:
                                        with open(output_path, 'a', encoding='utf-8') as f:
                                            f.write(transcription + "\n\n")

                                        # --- Update State AFTER successful write ---
                                        if pdf_filename and state_file_path and state_data is not None:
                                            state_data[pdf_filename] = page_num
                                            save_resume_state(state_file_path, state_data)
                                            logger.debug(f"Updated resume state for {pdf_filename} to page {page_num}")

                                    except IOError as e:
                                        logger.error(f"Error writing transcription for page {page_num} to {output_path}: {e}")
                                        processing_successful = False
                                else:
                                    print(f"--- Transcription for Page {page_num} ({middle_page_name}) ---")
                                    print(transcription)
                                    print("--- End Transcription ---\n")

                                last_successful_transcription = transcription # Update context only if NOT skipped
                        else:
                            logger.warning(f"--- Transcription FAILED (empty/failed result) for page {page_num} ({middle_page_name}) ---")
                            processing_successful = False

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
                    except OSError as e_del:
                        logger.warning(f"Could not delete temporary composite {temp_composite_img_path.name}: {e_del}")

    finally:
        if temp_composite_dir.exists():
            try:
                shutil.rmtree(temp_composite_dir)
                logger.debug(f"Cleaned up composites directory: {temp_composite_dir}")
            except OSError as e_final_clean:
                 logger.warning(f"Could not clean up composites directory {temp_composite_dir}: {e_final_clean}")

    logger.info(f"Finished loop for directory {image_directory.name}. Processed {files_processed_count} pages within effective range.")
    return processing_successful, last_successful_transcription

# --- Model Parsing and Platform Inference ---
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

# --- Main Logic Function (Relevant Initialization Part) ---
def get_platform_from_model_type(model_type: ModelType) -> Optional[ModelPlatformType]:
    """
    Infers the ModelPlatformType STRICTLY from a ModelType enum member.
    Assumes the input is already a valid ModelType enum.
    """
    # --- Ensure input is ModelType enum ---
    if not isinstance(model_type, ModelType):
        logger.error(f"get_platform_from_model_type received an invalid type: {type(model_type)}. Expected ModelType enum.")
        return None # Or raise TypeError

    # --- Logic based on ModelType enum member name ---
    model_name = model_type.name.upper()

    if model_name.startswith("GPT") or model_name.startswith("O3"): return ModelPlatformType.OPENAI
    if model_name.startswith("GEMINI"): return ModelPlatformType.GEMINI
    if model_name.startswith("AZURE"): return ModelPlatformType.AZURE
    if model_name.startswith("QWEN"): return ModelPlatformType.QWEN
    if model_name.startswith("DEEPSEEK"): return ModelPlatformType.DEEPSEEK
    if model_name.startswith("GROQ"): return ModelPlatformType.GROQ
    if "LLAMA" in model_name or "MIXTRAL" in model_name: # Base enums for these often imply specific platforms
         logger.warning(f"Assuming GROQ platform for base Llama/Mixtral enum: {model_name}. Check .env for GROQ keys or adjust logic if using another provider.")
         return ModelPlatformType.GROQ
    if model_name.startswith("CLAUDE"):
         # Claude models typically require specific handling
         logger.warning("Assuming OPENAI platform compatibility for Claude model enum. Verify API provider/base URL in .env.")
         return ModelPlatformType.OPENAI # Default assumption, may need env var check

    # Fallback for other base enums not explicitly handled above
    logger.warning(f"Could not infer platform for base model enum: {model_type}. Defaulting to OPENAI.")
    return ModelPlatformType.OPENAI # Default fallback



    """Tries to infer the ModelPlatformType from a ModelType enum or model string."""
    unified_platform_map = {
        "openrouter": ModelPlatformType.OPENAI, # Adjust if needed, OpenRouter often mimics OpenAI
        "meta-llama": ModelPlatformType.OPENAI, # Adjust based on actual provider (Groq, Together, etc.)
        "google": ModelPlatformType.GEMINI,
        "qwen": ModelPlatformType.QWEN,
        # Add other known provider prefixes
    }

    original_input = model_type # Keep for logging

    if isinstance(model_type, str):
        # Handle potential suffixes like ':free', ':latest', etc.
        model_string_cleaned = model_type.split(':')[0]

        try:
            # Attempt to convert the *cleaned* string to a base ModelType enum first.
            # If successful, fall through to the enum logic below.
            enum_match = ModelType(model_string_cleaned)
            logger.debug(f"Successfully converted cleaned string '{model_string_cleaned}' to base enum {enum_match}")
            model_type = enum_match # Replace original string with the enum for further processing

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

    # --- If we reach here, model_type should be a ModelType enum instance ---
    if isinstance(model_type, ModelType):
        model_name = model_type.name.upper()

        # Check by name prefix (common pattern)
        if model_name.startswith("GPT") or model_name.startswith("O3"): return ModelPlatformType.OPENAI
        if model_name.startswith("GEMINI"): return ModelPlatformType.GEMINI
        if model_name.startswith("AZURE"): return ModelPlatformType.AZURE
        if model_name.startswith("QWEN"): return ModelPlatformType.QWEN
        if model_name.startswith("DEEPSEEK"): return ModelPlatformType.DEEPSEEK
        if model_name.startswith("GROQ"): return ModelPlatformType.GROQ
        if "LLAMA" in model_name or "MIXTRAL" in model_name: # Base enums for these often imply specific platforms
             logger.warning(f"Assuming GROQ platform for base Llama/Mixtral enum: {model_name}. Check .env for GROQ keys or adjust logic if using another provider.")
             return ModelPlatformType.GROQ
        if model_name.startswith("CLAUDE"):
             # Claude models typically require specific handling, often via OpenAI platform on OpenRouter or dedicated Anthropic SDK
             logger.warning("Assuming OPENAI platform compatibility for Claude model enum. Verify API provider/base URL.")
             return ModelPlatformType.OPENAI

        # Fallback for other base enums not explicitly handled above
        logger.warning(f"Could not infer platform for base model enum: {model_type}. Defaulting to OPENAI.")
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
            read_size = len(marker) + 50
            f.seek(max(0, file_path.stat().st_size - read_size))
            last_chunk = f.read()
            return last_chunk.strip().endswith(marker)
    except IOError as e:
        logger.warning(f"Could not read end of file {file_path} to check completion: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking file completion for {file_path}: {e}", exc_info=True)
        return False

# --- ADDED: TOC Extraction Function ---
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
                            # PyPDF2 >= 3.X.X uses .page_number (0-indexed)
                            # PyPDF2 < 3.X.X might use .page (PageObject) - need to get index
                            page_num_0_indexed = item.page_number if hasattr(item, 'page_number') else reader.get_page_number(item.page)

                            page_num_1_indexed = page_num_0_indexed + 1
                            title = item.title.strip() if item.title else "Untitled Section"
                            indent = "  " * level
                            toc_entries.append(f"{indent}- {title} (Page {page_num_1_indexed})")
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

# --- ADDED: TOC Analysis Function ---
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

# --- System Temp File Cleanup ---
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

# --- Main Logic Function ---
def run_main_logic(args):
    # Set up logging based on args
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level_int)
    formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] %(message)s')
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setLevel(log_level_int)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)
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

    # --- Output path setup ---
    single_output_file_path: Optional[Path] = None
    output_directory_path: Optional[Path] = None
    resume_state_file_path: Optional[Path] = None
    final_append_mode = args.append

    if not process_pdfs: # Image directory mode
        if args.output:
            single_output_file_path = Path(args.output).resolve()
            logger.info(f"Output will be to SINGLE file: {single_output_file_path}")
        else:
            logger.info("Output will be to console.")
    else: # PDF Mode
        if args.output:
            output_directory_path = Path(args.output).resolve()
        else:
            output_directory_path = source_dir / "books_tts"
        try:
            output_directory_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory for books: {output_directory_path}")
            resume_state_file_path = output_directory_path / RESUME_STATE_FILENAME
            logger.info(f"Resume state will be managed in: {resume_state_file_path}")
        except OSError as e:
            logger.error(f"Failed to create output directory {output_directory_path}: {e}")
            sys.exit(1)

    # --- Model selection and initialization ---
    # Parse the model strings first to get either enum or string representation
    parsed_transcription_model_repr = parse_model_string_to_enum(args.transcription_model, "Transcription")
    parsed_analysis_model_repr = parse_model_string_to_enum(args.analysis_model, "Analysis") # ADDED

    # --- Validate Parsed Models ---
    if parsed_transcription_model_repr is None:
        logger.critical("Invalid transcription model type provided. Cannot proceed.")
        sys.exit(1)
    if parsed_analysis_model_repr is None and process_pdfs and not (user_start_page_arg and user_end_page_arg):
         logger.critical("Invalid analysis model type provided, required for automatic page range detection in PDF mode. Cannot proceed.")
         sys.exit(1)

    # --- Determine Platforms ---
    transcription_platform: Optional[ModelPlatformType] = None
    analysis_platform: Optional[ModelPlatformType] = None

    # Determine platform for Transcription Model
    if isinstance(parsed_transcription_model_repr, ModelType):
        transcription_platform = get_platform_from_model_type(parsed_transcription_model_repr)
    elif isinstance(parsed_transcription_model_repr, str):
        # Infer platform from unified string prefix
        if "/" in parsed_transcription_model_repr:
            provider = parsed_transcription_model_repr.split('/')[0].lower()
            if provider == "google": transcription_platform = ModelPlatformType.GEMINI
            elif provider == "qwen": transcription_platform = ModelPlatformType.QWEN
            # Add mappings for other providers like 'meta-llama', 'openrouter' etc.
            elif provider in ["openrouter", "meta-llama", "anthropic"]:
                # Defaulting to OPENAI compatibility for these, adjust if needed
                logger.warning(f"Assuming OPENAI platform compatibility for unified provider '{provider}'. Check .env.")
                transcription_platform = ModelPlatformType.OPENAI
            else:
                 logger.warning(f"Unknown provider prefix '{provider}' for transcription model. Defaulting platform to OPENAI.")
                 transcription_platform = ModelPlatformType.OPENAI
        else:
            logger.error(f"Cannot determine platform for transcription string '{parsed_transcription_model_repr}' - not a base enum or unified format.")
            transcription_platform = None # Explicitly None if cannot determine
    else: # Should not happen if parse_model_string_to_enum works correctly
         logger.error(f"Unexpected type for parsed transcription model: {type(parsed_transcription_model_repr)}")
         transcription_platform = None

    # Determine platform for Analysis Model (only if needed and parsed)
    if parsed_analysis_model_repr:
        if isinstance(parsed_analysis_model_repr, ModelType):
            analysis_platform = get_platform_from_model_type(parsed_analysis_model_repr)
        elif isinstance(parsed_analysis_model_repr, str):
            # Infer platform from unified string prefix
            if "/" in parsed_analysis_model_repr:
                provider = parsed_analysis_model_repr.split('/')[0].lower()
                if provider == "google": analysis_platform = ModelPlatformType.GEMINI
                elif provider == "qwen": analysis_platform = ModelPlatformType.QWEN
                elif provider in ["openrouter", "meta-llama", "anthropic"]:
                    logger.warning(f"Assuming OPENAI platform compatibility for unified provider '{provider}'. Check .env.")
                    analysis_platform = ModelPlatformType.OPENAI
                else:
                    logger.warning(f"Unknown provider prefix '{provider}' for analysis model. Defaulting platform to OPENAI.")
                    analysis_platform = ModelPlatformType.OPENAI
            else:
                logger.error(f"Cannot determine platform for analysis string '{parsed_analysis_model_repr}' - not a base enum or unified format.")
                analysis_platform = None
        else:
            logger.error(f"Unexpected type for parsed analysis model: {type(parsed_analysis_model_repr)}")
            analysis_platform = None

    # --- Validate Platforms ---
    if transcription_platform is None:
        logger.critical(f"Could not determine platform for transcription model: {args.transcription_model}. Cannot proceed.")
        sys.exit(1)
    # Check analysis platform validity only if it was required and parsed
    analysis_required = process_pdfs and not (user_start_page_arg and user_end_page_arg)
    if analysis_required and parsed_analysis_model_repr and analysis_platform is None:
        logger.critical(f"Could not determine platform for required analysis model: {args.analysis_model}. Cannot proceed.")
        sys.exit(1)

    logger.info(f"Transcription: Model='{args.transcription_model}', Representation={type(parsed_transcription_model_repr).__name__}, Platform={transcription_platform}")
    if parsed_analysis_model_repr: # Log analysis info only if it was parsed
         logger.info(f"Analysis: Model='{args.analysis_model}', Representation={type(parsed_analysis_model_repr).__name__}, Platform={analysis_platform or 'N/A (not needed or failed)'}")

    # --- Initialize models and toolkits (passes the correctly determined platform) ---
    transcription_toolkit: Optional[ImageAnalysisToolkit] = None
    analysis_model_backend: Optional[BaseModelBackend] = None

    try:
        logger.info(f"Initializing Transcription model...")
        transcription_model = ModelFactory.create(
            model_platform=transcription_platform, # Use determined platform
            model_type=parsed_transcription_model_repr # Use parsed repr (enum or string)
        )
        transcription_toolkit = ImageAnalysisToolkit(model=transcription_model)
        logger.info(f"Initialized Transcription Toolkit with model: {transcription_model.model_type}")

        # Initialize analysis model if needed and platform determined
        if analysis_required and parsed_analysis_model_repr and analysis_platform:
            logger.info(f"Initializing Analysis model...")
            analysis_model_backend = ModelFactory.create(
                model_platform=analysis_platform, # Use determined platform
                model_type=parsed_analysis_model_repr # Use parsed repr (enum or string)
            )
            logger.info(f"Initialized Analysis Model Backend: {analysis_model_backend.model_type}")

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
    if analysis_required and analysis_model_backend is None:
         # Check if it failed because platform couldn't be determined or factory failed
         if parsed_analysis_model_repr and analysis_platform is None:
              logger.critical("Analysis model initialization failed because platform could not be determined.")
         else: # Platform was determined, but ModelFactory failed
             logger.critical("Analysis model backend initialization failed (ModelFactory error). Required for auto page range detection.")
         sys.exit(1)

    logger.info(f"Transcription: Model='{args.transcription_model}', Platform={transcription_platform}")
    if analysis_platform:
         logger.info(f"Analysis: Model='{args.analysis_model}', Platform={analysis_platform}")

    # Initialize models and toolkits
    transcription_toolkit: Optional[ImageAnalysisToolkit] = None
    analysis_model_backend: Optional[BaseModelBackend] = None # ADDED

    try:
        logger.info(f"Initializing Transcription model...")
        transcription_model = ModelFactory.create(
            model_platform=transcription_platform,
            model_type=parsed_transcription_model_repr
        )
        transcription_toolkit = ImageAnalysisToolkit(model=transcription_model)
        logger.info(f"Initialized Transcription Toolkit with model: {transcription_model.model_type}")

        # Initialize analysis model if needed
        if analysis_platform and parsed_analysis_model_repr:
            logger.info(f"Initializing Analysis model...")
            analysis_model_backend = ModelFactory.create(
                model_platform=analysis_platform,
                model_type=parsed_analysis_model_repr
            )
            logger.info(f"Initialized Analysis Model Backend: {analysis_model_backend.model_type}")

    except Exception as e:
        logger.error(f"Failed to initialize models or toolkits: {e}", exc_info=True)
        sys.exit(1)

    if transcription_toolkit is None:
         logger.critical("Transcription toolkit initialization failed.")
         sys.exit(1)
    # Analysis model is optional if only processing images or user provides range
    if process_pdfs and not (user_start_page_arg and user_end_page_arg) and analysis_model_backend is None:
         logger.critical("Analysis model initialization failed, required for auto page range detection.")
         sys.exit(1)


    # --- Load context if appending (single file mode only) ---
    overall_initial_context: Optional[str] = None
    if single_output_file_path and final_append_mode:
        if single_output_file_path.exists() and single_output_file_path.stat().st_size > 0:
            try:
                logger.info(f"Append mode: Reading context from end of single file: {single_output_file_path}")
                with open(single_output_file_path, 'r', encoding='utf-8') as f:
                    read_size = 4096
                    f.seek(max(0, single_output_file_path.stat().st_size - read_size))
                    overall_initial_context = f.read().strip() or None
                    if overall_initial_context:
                         logger.info(f"Loaded context (last ~{len(overall_initial_context)} chars) from single file.")
                    else:
                         logger.warning(f"Could not read context from existing file: {single_output_file_path}")
            except IOError as e:
                 logger.error(f"Error reading context from {single_output_file_path}: {e}")
            except Exception as e:
                 logger.error(f"Unexpected error loading context from {single_output_file_path}: {e}", exc_info=True)
        else:
             logger.info(f"Append mode active, but output file {single_output_file_path} missing/empty. Starting fresh.")
        # Add separator
        try:
            if single_output_file_path.exists():
                 with open(single_output_file_path, 'a', encoding='utf-8') as f:
                     separator = f"\n\n{'='*20} Appending transcriptions at {datetime.datetime.now()} {'='*20}\n\n"
                     f.write(separator)
        except IOError as e:
             logger.error(f"Error adding append separator to {single_output_file_path}: {e}")

    # --- Main Processing Logic ---
    if not process_pdfs:
        # --- Single Image Directory Processing ---
        logger.info(f"Processing images directly from: {source_dir}")
        success, _ = process_images_in_directory(
            image_directory=source_dir,
            output_path=single_output_file_path,
            append_mode=final_append_mode,
            start_page=user_start_page_arg, # Use direct args
            end_page=user_end_page_arg,
            image_toolkit=transcription_toolkit,
            prompt_template=PROMPT_TEMPLATE,
            initial_context=overall_initial_context,
            pdf_filename=None,
            state_file_path=None,
            state_data=None
        )
        if not success:
             logger.error(f"Processing failed for image directory: {source_dir}")
        logger.info("Image directory processing finished.")
    else:
        # --- PDF Directory Processing with State/Resume ---
        pdf_files = sorted(list(source_dir.glob('*.pdf')))
        if not pdf_files:
            logger.warning(f"No PDF files found in book directory: {source_dir}")
            sys.exit(0)

        logger.info(f"Found {len(pdf_files)} PDFs to process.")
        current_state_data = load_resume_state(resume_state_file_path) if resume_state_file_path else {}

        for pdf_path in pdf_files:
            pdf_filename = pdf_path.name
            logger.info(f"--- Starting processing for PDF: {pdf_filename} ---")
            book_specific_output_path = output_directory_path / f"{pdf_path.stem}.txt"
            logger.info(f"Output for '{pdf_filename}' will be: {book_specific_output_path}")

            if check_if_file_completed(book_specific_output_path, END_MARKER):
                logger.info(f"'{pdf_filename}' already marked as completed ('{END_MARKER}' found). Skipping.")
                continue

            # --- Determine Base Page Range (User Args OR Auto-Detect) ---
            base_start_page = user_start_page_arg
            base_end_page = user_end_page_arg
            range_source = "user"

            if base_start_page is None and base_end_page is None:
                logger.info("No user page range provided. Attempting automatic detection...")
                range_source = "auto"
                toc_text = extract_toc(pdf_path)
                if toc_text and analysis_model_backend:
                    detected_start, detected_end = get_main_content_page_range(
                        toc_text, analysis_model_backend
                    )
                    if detected_start is not None or detected_end is not None:
                         logger.info(f"Auto-detected range for '{pdf_filename}': Start={detected_start}, End={detected_end}")
                         base_start_page = detected_start # Can be None
                         base_end_page = detected_end   # Can be None
                    else:
                         logger.warning(f"Automatic page range detection failed for '{pdf_filename}'. Processing all pages (subject to resume).")
                         range_source = "failed_auto"
                else:
                     logger.warning(f"Could not extract TOC or analysis model not available for '{pdf_filename}'. Processing all pages (subject to resume).")
                     range_source = "no_toc"
            else:
                 logger.info(f"Using user-provided page range: {base_start_page}-{base_end_page}")

            # --- Determine Effective Page Range & Resumption ---
            last_processed_page = current_state_data.get(pdf_filename)
            resuming = last_processed_page is not None
            resume_start_page = (last_processed_page + 1) if resuming else 1

            logger.debug(f"Range Source: {range_source}. Base range: {base_start_page}-{base_end_page}. Resume state: last_processed={last_processed_page}")

            # Calculate final effective start page
            effective_start_page = base_start_page # Start with base (can be None)
            if resuming:
                 # Effective start is the MAX of resume point and base start (treat None base_start as 1)
                 effective_start_page = max(resume_start_page, base_start_page or 1)
            elif base_start_page is None:
                 effective_start_page = 1 # Not resuming and no base_start means start from 1

            # Effective end is just the base end (can be None)
            effective_end_page = base_end_page

            logger.info(f"Effective processing range for '{pdf_filename}': Start={effective_start_page}, End={effective_end_page or 'end'}")

            if effective_start_page is not None and effective_end_page is not None and effective_start_page > effective_end_page:
                 logger.warning(f"Effective start page {effective_start_page} is after effective end page {effective_end_page} for '{pdf_filename}'. Nothing to process.")
                 continue

            # --- Load Context for Append/Resume ---
            current_book_initial_context = None
            # Determine if we NEED to load context (either resuming or explicitly appending)
            load_context_needed = resuming or final_append_mode
            if load_context_needed and book_specific_output_path.exists() and book_specific_output_path.stat().st_size > 0:
                try:
                    logger.info(f"Reading context from existing file: {book_specific_output_path} (Reason: {'resume' if resuming else 'append'})")
                    with open(book_specific_output_path, 'r', encoding='utf-8') as bf:
                        read_size = 4096
                        bf.seek(max(0, book_specific_output_path.stat().st_size - read_size))
                        current_book_initial_context = bf.read().strip() or None
                        if current_book_initial_context:
                            logger.info(f"Loaded context for book {pdf_filename}. Length: {len(current_book_initial_context)}")
                        else:
                             logger.warning(f"Could not read context from existing file: {book_specific_output_path}")
                except IOError as e:
                    logger.error(f"Error reading context for {book_specific_output_path}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error loading context from {book_specific_output_path}: {e}", exc_info=True)
            elif load_context_needed:
                 logger.info(f"Append/Resume mode active, but file {book_specific_output_path} missing/empty. Starting context fresh.")


            # Add append separator if explicitly appending and file exists
            if final_append_mode and book_specific_output_path.exists():
                 try:
                     with open(book_specific_output_path, 'a', encoding='utf-8') as bf:
                         separator = f"\n\n{'='*20} Appending ({range_source}) at {datetime.datetime.now()} {'='*20}\n\n"
                         bf.write(separator)
                 except IOError as e:
                    logger.error(f"Error preparing file {book_specific_output_path} for appending: {e}. Skipping book.")
                    continue

            # --- Process the PDF ---
            images_base_dir = source_dir / TEMP_IMAGE_SUBDIR
            pdf_image_dir_path = images_base_dir / pdf_path.stem
            pdf_processing_success = False

            try:
                images_base_dir.mkdir(exist_ok=True)
                if pdf_image_dir_path.exists():
                    # Always clean up old images before conversion
                    try:
                        shutil.rmtree(pdf_image_dir_path)
                        logger.debug(f"Cleaned up pre-existing image directory: {pdf_image_dir_path}")
                    except OSError as e_clean_old:
                        logger.warning(f"Could not clean up pre-existing image directory {pdf_image_dir_path}: {e_clean_old}")

                try:
                    pdf_image_dir_path.mkdir(parents=True, exist_ok=True) # Ensure it exists
                    logger.info(f"Image dir for '{pdf_filename}': {pdf_image_dir_path}")
                except OSError as e_mkdir:
                    logger.error(f"Failed to create image directory {pdf_image_dir_path}: {e_mkdir}")
                    continue

                if not convert_pdf_to_images(pdf_path, pdf_image_dir_path):
                    logger.error(f"Failed PDF conversion for '{pdf_filename}'. Skipping.")
                else:
                    # Determine the actual mode for process_images_in_directory's append logic
                    # It should append if resuming OR if explicitly told to append by user
                    process_append_mode = resuming or final_append_mode

                    # Call image processing with the calculated effective range
                    pdf_processing_success, _ = process_images_in_directory(
                        image_directory=pdf_image_dir_path,
                        output_path=book_specific_output_path,
                        append_mode=process_append_mode, # Use calculated append mode
                        start_page=effective_start_page,
                        end_page=effective_end_page,
                        image_toolkit=transcription_toolkit,
                        prompt_template=PROMPT_TEMPLATE,
                        initial_context=current_book_initial_context,
                        pdf_filename=pdf_filename,
                        state_file_path=resume_state_file_path,
                        state_data=current_state_data
                    )

                    if pdf_processing_success:
                        logger.info(f"Successfully processed images for PDF: {pdf_filename}")
                        try:
                            with open(book_specific_output_path, 'a', encoding='utf-8') as f_end:
                                f_end.write(f"\n\n{END_MARKER}\n")
                            logger.info(f"Added '{END_MARKER}' to completed file: {book_specific_output_path.name}")
                            # Clear state for completed book
                            if resume_state_file_path and pdf_filename in current_state_data:
                                 del current_state_data[pdf_filename]
                                 save_resume_state(resume_state_file_path, current_state_data)
                                 logger.info(f"Removed completed book '{pdf_filename}' from resume state.")
                        except IOError as e:
                             logger.error(f"Error adding '{END_MARKER}' to {book_specific_output_path.name}: {e}")
                             pdf_processing_success = False # Mark as failed if end marker fails
                    else:
                        logger.error(f"Processing failed for images from PDF: {pdf_filename}.")

            except Exception as e:
                 logger.critical(f"Critical error processing PDF {pdf_filename}: {e}", exc_info=True)
            finally:
                if pdf_image_dir_path and pdf_image_dir_path.exists() and not args.keep_images:
                    try:
                        shutil.rmtree(pdf_image_dir_path)
                        logger.info(f"Cleaned up PDF image directory: {pdf_image_dir_path}")
                    except Exception as e_clean:
                         logger.error(f"Error cleaning up PDF image dir {pdf_image_dir_path}: {e_clean}")
                elif pdf_image_dir_path and args.keep_images:
                    logger.info(f"Keeping PDF image directory due to --keep-images flag: {pdf_image_dir_path}")

            logger.info(f"--- Finished processing for PDF: {pdf_filename} ---")

        logger.info("All PDF processing finished.")

# --- Script Entry Point ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    DEFAULT_TRANSCRIPTION_MODEL = ModelType.GEMINI_2_0_FLASH_LITE_PREVIEW.value # Example default
    # Use a capable model for analysis, GPT-4o Mini is a good choice if available
    DEFAULT_ANALYSIS_MODEL = ModelType.GEMINI_2_5_PRO_EXP.value

    parser = argparse.ArgumentParser(
        description="Transcribe text from book page images or PDFs using Owl/CAMEL-AI.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--input-dir", help="Directory containing image files (named 'prefix-digits.ext').")
    input_group.add_argument("-b", "--book-dir", help="Directory containing PDF book files to process.")

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
              f"and uses '{RESUME_STATE_FILENAME}' in the output dir to resume incomplete ones."
    )
    parser.add_argument("--start-page", type=int, help="Manually specify the first page number (inclusive) to process. Overrides auto-detection.")
    parser.add_argument("--end-page", type=int, help="Manually specify the last page number (inclusive) to process. Overrides auto-detection.")

    parser.add_argument(
        "--transcription-model", type=str, default=DEFAULT_TRANSCRIPTION_MODEL,
        help=f"Model string for image transcription (default: '{DEFAULT_TRANSCRIPTION_MODEL}'). Matches ModelType enum or supported string."
    )
    # --- ADDED: Analysis Model Argument ---
    parser.add_argument(
        "--analysis-model", type=str, default=DEFAULT_ANALYSIS_MODEL,
        help=f"Model string for TOC analysis (default: '{DEFAULT_ANALYSIS_MODEL}'). Used only in --book-dir mode if --start/--end-page are not provided."
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)."
    )
    parser.add_argument(
        "--keep-images", action="store_true",
        help=f"Keep the per-PDF image directories created in '{TEMP_IMAGE_SUBDIR}' within the book directory (only applies to --book-dir mode)."
    )

    # Generate epilog with available models
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
  # Images -> Single file (overwrite)
  python %(prog)s --input-dir path/to/pages -o output.txt
  # Images -> Console
  python %(prog)s --input-dir path/to/pages
  # Images -> Single file (append)
  python %(prog)s --input-dir path/to/pages -o output.txt --append --transcription-model {DEFAULT_TRANSCRIPTION_MODEL}

  # PDFs -> Specific output dir (AUTO page range detection, overwrite)
  python %(prog)s --book-dir path/to/pdfs -o path/to/output_dir
  # PDFs -> Default output dir 'books_tts' (AUTO page range, overwrite)
  python %(prog)s --book-dir path/to/pdfs
  # PDFs -> Specific dir (AUTO page range, append/resume, specific models)
  python %(prog)s --book-dir path/to/pdfs -o path/to/output_dir --append \\
    --transcription-model {DEFAULT_TRANSCRIPTION_MODEL} --analysis-model {DEFAULT_ANALYSIS_MODEL}
  # PDFs -> Specific dir ()
  python %(prog)s --book-dir path/to/pdfs --start-page 50 --end-page 150 -o path/to/output_dir --append
  # PDFs -> Default dir (AUTO page range, debug, append/resume, keep images)
  python %(prog)s --book-dir path/to/pdfs --log-level DEBUG --append --keep-images

Notes:
- Requires 'pdftoppm' (poppler-utils) and 'montage' (ImageMagick).
- Requires 'PyPDF2' (`pip install pypdf2`).
- Images must be named like '-[digits].ext' (or 'prefix-[digits].ext'). Needs 3-page context for transcription.
- Configure API keys in a .env file.
- Use model strings matching CAMEL's ModelType enum values or supported unified strings (like 'provider/model-name').
- Auto page range detection (in --book-dir mode without --start/--end-page):
    - Uses PyPDF2 to extract the Table of Contents (outline).
    - Uses the --analysis-model to determine the main content page range, excluding front/back matter.
    - Falls back to processing all pages if TOC extraction or analysis fails.
- --append mode (for --book-dir):
    - Checks for '{END_MARKER}' in existing .txt files to skip completed PDFs.
    - Uses '{RESUME_STATE_FILENAME}' in output dir to resume incomplete PDFs from last successful page.
    - State is updated after *each* successful page transcription/skip.
- Interrupted processes may leave temporary directories ({TEMP_IMAGE_SUBDIR}). Use --keep-images to prevent cleanup.
{model_help_epilog}
"""
    parsed_args = parser.parse_args()

    try:
        run_main_logic(parsed_args)
    finally:
        logger.info("Performing final cleanup check of temporary files in /tmp...")
        cleanup_system_temp_files()
        logger.info("Script finished.")
