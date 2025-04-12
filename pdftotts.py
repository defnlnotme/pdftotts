from enum import Enum
import subprocess
import os
import sys
import glob
import argparse
import logging
from pathlib import Path
import re # Import regex module
from typing import List, Optional, Tuple, Type
from time import sleep
import tempfile
import datetime # Import datetime for append message
import shutil # Import for directory removal
import json # Import JSON for parsing analysis response

# --- Add Owl directory to Python path ---
script_dir = Path(__file__).parent
# Navigate up to the 'owl' directory from 'examples'
owl_dir = script_dir.parent
if str(owl_dir) not in sys.path:
    # Insert the 'owl' directory itself
    sys.path.insert(0, str(owl_dir))
    # Also insert the parent of 'owl' (the root where 'camel' might be)
    root_dir = owl_dir.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

# --- End Path Addition ---

from dotenv import load_dotenv

from camel.toolkits import ImageAnalysisToolkit
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, UnifiedModelType

# --- Model Definitions (Example Unified Models) ---
# Note: These are just examples. The user will provide model strings via CLI.
#       The core ModelType enum will be used for parsing.
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

# --- Prompt Templates (Unchanged) ---
PROMPT_TEMPLATE = """
# Prompt for TTS Book Page Transcription (Optimized for Continuity & Excluding References/Markers)

**Objective:** Transcribe the text from the **middle** book page image provided into clear, natural-sounding prose optimized for Text-to-Speech (TTS) playback. The key goals are a smooth narrative flow between pages, completely **excluding** any reference to external sources (books, papers) or internal structural pointers and markers (page numbers, figure/table/equation numbers, chapter/section numbers, list numbers, running headers/footers, etc.).

**Input:** Three images are provided side-by-side:
1.  Previous Page (Left Image) - **CONTEXT ONLY for visuals**
2.  **Target Page (Middle Image) - TRANSCRIBE THIS PAGE**
3.  Next Page (Right Image) - **CONTEXT ONLY for visuals**

**Context from Previous Page's Transcription:**
{previous_transcription_context_section}

**Critical Instructions:**

1.  **Focus EXCLUSIVELY on the Middle Page:** Do NOT transcribe any content from the left (previous) or right (next) page images. They are provided solely for contextual understanding when describing visuals *on the middle page*.
2.  **Ancillary Page Check (Middle Page Only):**
    * **CRITICAL Check:** If the **middle page** is clearly identifiable as **primarily** containing content such as a book cover, title page, copyright information, Preface, Foreword, Dedication, Table of Contents, List of Figures/Tables, Acknowledgments, Introduction (if clearly marked as introductory material *before* Chapter 1 or the main narrative), Notes section, Bibliography, References, Index, Appendix, Glossary, or other front/back matter **not part of the main narrative body/chapters**, **output the single word: `SKIPPED`**.
    * **Strict Adherence:** **Do NOT transcribe these ancillary page types.** If the page header, title, or main content block clearly indicates it's one of these types (e.g., starts with "Preface", "Table of Contents", "Index"), it **must** be skipped. Only transcribe pages that are part of the core chapters or narrative content.

**Transcription Guidelines (For Non-Skipped Middle Pages):**

*The following guidelines ensure the transcription flows naturally as spoken prose, focusing on the core content while omitting disruptive elements like references and structural markers.*

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

**Output Format:**

* If the page is skipped (per Instruction #2), output **only** the single word `SKIPPED`.
* If the page is transcribed, provide **only** the final, clean transcribed text.
* **Crucially: Do NOT wrap the output text in ``` ```, backticks, code blocks, quotation marks, or any other formatting markers.** The output should be plain text suitable for direct use.

---

**Now, please process the provided middle book page image based on these strictly revised guidelines, ensuring smooth continuity if previous transcription context is provided.**
"""

PDF_ANALYSIS_PROMPT = """
Analyze the provided PDF document structure to identify the page range containing the main content suitable for transcription (e.g., chapters of a book).

**Objective:** Determine the *first page number* where the primary narrative or core content begins (typically Chapter 1 or equivalent, *after* any front matter like Title Page, Copyright, Table of Contents, Preface, Introduction) and the *last page number* where this main content ends (typically *before* back matter like Appendices, Bibliography, Index, Glossary).

**Instructions:**
1.  Examine the overall structure of the PDF.
2.  Identify the start of the main content body. Note the page number **printed on that page**, if available. If not printed, use the sequential page number from the PDF viewer (1-based index).
3.  Identify the end of the main content body. Note the page number **printed on that page**, if available. If not printed, use the sequential page number.
4.  Exclude standard front matter (cover, title, copyright, dedication, ToC, lists of figures/tables, preface, foreword, acknowledgments, sometimes introduction if clearly preliminary).
5.  Exclude standard back matter (appendices, notes, bibliography/references, glossary, index, colophon).
6.  Focus on the core chapters or sections that constitute the main narrative or argument of the work.

**Output Format:**
*   Provide the result as a JSON object.
*   The JSON object should have two keys: `start_page` and `end_page`.
*   The values should be the identified integer page numbers.
*   Example: `{"start_page": 15, "end_page": 350}`
*   If you cannot reliably determine the start or end page (e.g., unclear structure, missing page numbers, analysis error), output a JSON object with an "error" key: `{"error": "Could not reliably determine page range."}`

**Provide ONLY the JSON object as the output.**
"""

# --- Logging Setup ---
logger = logging.getLogger(__name__)
# Basic config will be set later based on args

# Load environment variables from .env file
load_dotenv()

# --- Helper Functions (Unchanged) ---
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
        logger.warning(f"Could not extract page number (format '-digits.') from filename: {filename.name}")
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
                logger.warning(f"Skipping file due to invalid name format or failed number extraction: {file_path.name}")

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
    # Ensure output directory exists (important if using subdirs within the main temp dir)
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

# --- New Function for PDF Analysis ---
def analyze_pdf_structure(
    analysis_toolkit: ImageAnalysisToolkit,
    pdf_path: Path
) -> Optional[str]:
    """
    Uses a specified LLM toolkit to analyze the PDF structure and identify
    the start and end pages of the main content.
    """
    logger.info(f"Analyzing PDF structure for: {pdf_path.name} using model {analysis_toolkit.model.model_type.value}") # Log enum value
    try:
        # Using ask_question_about_image, assuming the toolkit/model can handle PDF path
        # for structural analysis. If this fails, need to adjust (e.g., text extraction).
        response = analysis_toolkit.ask_question_about_image(
            image_path=str(pdf_path), # Pass PDF path
            question=PDF_ANALYSIS_PROMPT
        )
        if response:
             # Clean up potential markdown code blocks around JSON
            response = re.sub(r'^```(json)?\s*', '', response.strip(), flags=re.IGNORECASE | re.MULTILINE)
            response = re.sub(r'\s*```$', '', response.strip(), flags=re.IGNORECASE | re.MULTILINE)
            logger.info(f"Received PDF analysis response for {pdf_path.name}")
            logger.debug(f"Raw analysis response: {response}")
            return response.strip()
        else:
            logger.warning(f"Received empty response during PDF structure analysis for {pdf_path.name}.")
            return None
    except Exception as e:
        logger.error(f"Error during PDF structure analysis for {pdf_path.name}: {e}", exc_info=True)
        return None

# --- New Function to Parse Analysis Response ---
def parse_pdf_analysis_response(response: str) -> Tuple[Optional[int], Optional[int]]:
    """Parses the JSON response from the PDF analysis LLM call."""
    if not response:
        return None, None
    try:
        data = json.loads(response)
        if "error" in data:
            logger.warning(f"PDF analysis failed: {data['error']}")
            return None, None
        start_page = data.get("start_page")
        end_page = data.get("end_page")
        if isinstance(start_page, int) and isinstance(end_page, int):
            if start_page > 0 and end_page >= start_page:
                 logger.info(f"Successfully parsed page range: Start={start_page}, End={end_page}")
                 return start_page, end_page
            else:
                logger.warning(f"Parsed page range invalid: Start={start_page}, End={end_page}. Ignoring.")
                return None, None
        else:
            logger.warning(f"Could not parse valid integer start/end pages from JSON: {response}")
            return None, None
    except json.JSONDecodeError:
        logger.warning(f"Failed to decode JSON from PDF analysis response: {response}")
        # Fallback: Try regex for the older format just in case
        start_match = re.search(r"Start:\s*(\d+)", response, re.IGNORECASE)
        end_match = re.search(r"End:\s*(\d+)", response, re.IGNORECASE)
        if start_match and end_match:
             try:
                 start_page = int(start_match.group(1))
                 end_page = int(end_match.group(1))
                 if start_page > 0 and end_page >= start_page:
                     logger.info(f"Successfully parsed page range via REGEX fallback: Start={start_page}, End={end_page}")
                     return start_page, end_page
                 else:
                    logger.warning(f"Regex parsed page range invalid: Start={start_page}, End={end_page}. Ignoring.")
                    return None, None
             except ValueError:
                 logger.warning(f"Regex fallback found digits but failed int conversion in: {response}")
                 return None, None
        logger.warning("Could not parse page range using JSON or regex fallback.")
        return None, None
    except Exception as e:
         logger.error(f"Unexpected error parsing PDF analysis response '{response}': {e}", exc_info=True)
         return None, None


def transcribe_image(
    toolkit: ImageAnalysisToolkit,
    image_path: Path,
    original_middle_page_name: str,
    previous_transcription: Optional[str] = None, # Added parameter
    prompt_template: str = PROMPT_TEMPLATE # Use the template
) -> Optional[str]:
    """Uses the ImageAnalysisToolkit to transcribe text from a composite image, using previous context."""
    logger.info(f"Transcribing composite image representing middle page: {original_middle_page_name} using {toolkit.model.model_type.value}") # Log enum value

    # --- Dynamically Construct Prompt ---
    if previous_transcription:
        context_snippet = previous_transcription[-500:] # Use last 500 chars
        context_section = f"""
<!-- Context from end of previous transcribed page -->
<previous_transcription_end>
...{context_snippet}
</previous_transcription_end>
*Instruction Reminder: Start the transcription for the current middle page smoothly following this context.*
"""
        logger.debug(f"Using context snippet: ...{context_snippet}")
    else:
        context_section = "*No previous transcription context provided (this might be the first page being transcribed).*"
        logger.debug("No previous transcription context provided.")

    final_prompt = prompt_template.format(previous_transcription_context_section=context_section)
    # --- End Dynamic Prompt Construction ---

    try:
        retry_delay = 2
        max_retries = 3
        retries = 0
        transcription = None
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
                # Handle potential LLM wrapping
                transcription = re.sub(r'^```(text)?\s*', '', transcription, flags=re.IGNORECASE | re.MULTILINE)
                transcription = re.sub(r'\s*```$', '', transcription, flags=re.IGNORECASE | re.MULTILINE)
                transcription = transcription.strip('"\'')

                if transcription and "Analysis failed" not in transcription and len(transcription) > 5:
                    # logger.info(f"Successfully transcribed middle page: {original_middle_page_name}") # Redundant log
                    return transcription
                else:
                    logger.warning(f"Transcription attempt {retries + 1}/{max_retries} failed or produced short/invalid output for middle page: {original_middle_page_name}. Content: '{transcription[:50]}...'. Retrying in {retry_delay}s...")
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
        logger.error(f"Unhandled error during transcription process for image {image_path.name} (middle page: {original_middle_page_name}): {e}", exc_info=True)
        return None


def convert_pdf_to_images(pdf_path: Path, output_dir: Path, image_format: str = 'png', dpi: int = 300) -> bool:
    """Converts a PDF file to images using pdftoppm."""
    if not pdf_path.is_file():
        logger.error(f"PDF file not found: {pdf_path}")
        return False

    if shutil.which("pdftoppm") is None:
        logger.error("Error: 'pdftoppm' command not found. Please ensure poppler-utils is installed and in your system's PATH.")
        return False

    output_prefix = output_dir / pdf_path.stem
    command = [
        "pdftoppm",
        f"-{image_format}",
        "-r", str(dpi),
        str(pdf_path),
        str(output_prefix)
    ]

    try:
        logger.info(f"Converting PDF '{pdf_path.name}' to {image_format.upper()} images in '{output_dir.name}'...")
        logger.debug(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        logger.info(f"Successfully converted PDF: {pdf_path.name}")
        logger.debug(f"pdftoppm stdout: {result.stdout}")
        # stderr might contain warnings, log it as debug
        if result.stderr:
             logger.debug(f"pdftoppm stderr: {result.stderr}")

        if not list(output_dir.glob(f"{output_prefix.name}-*.{image_format}")):
             logger.warning(f"pdftoppm ran but no output images found for {pdf_path.name} in {output_dir}")
             return False
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


def process_images_in_directory(
    image_directory: Path,
    output_path: Optional[Path],
    append_mode: bool,
    start_page: Optional[int], # This is the *effective* start page
    end_page: Optional[int],   # This is the *effective* end page
    image_toolkit: ImageAnalysisToolkit, # Transcription toolkit
    prompt_template: str,
    initial_context: Optional[str]
) -> Tuple[bool, Optional[str]]:
    """
    Processes images found in a directory, performing transcription within the effective page range.
    Ensures temporary composite images are deleted after use.
    """
    logger.info(f"Starting processing for image directory: {image_directory}")
    if start_page is not None and end_page is not None:
        logger.info(f"Effective processing range: Pages {start_page} to {end_page} (inclusive).")
    elif start_page is not None:
         logger.info(f"Effective processing range: Starting from page {start_page}.")
    elif end_page is not None:
        logger.info(f"Effective processing range: Up to page {end_page}.")
    else:
        logger.info("Effective processing range: All available pages.")


    image_files = find_and_sort_image_files(image_directory)
    if not image_files:
        logger.warning(f"No valid image files found in directory: {image_directory}. Skipping this directory.")
        return True, initial_context

    logger.info(f"Found {len(image_files)} valid image files to potentially process in {image_directory.name}.")

    files_processed_count = 0
    last_successful_transcription = initial_context
    processing_successful = True

    # Create a dedicated subdirectory within the system's temp location for composites
    # This outer directory will be managed by tempfile.TemporaryDirectory context manager
    with tempfile.TemporaryDirectory(prefix=f"owl_montage_dir_{image_directory.stem}_") as temp_dir:
        temp_composite_path_base = Path(temp_dir)
        logger.info(f"Using temporary directory for composites: {temp_composite_path_base}")

        if output_path and not append_mode:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f_init:
                    f_init.truncate(0)
                logger.info(f"Output file {output_path} truncated (or created).")
            except IOError as e:
                logger.error(f"Error preparing output file {output_path}: {e}")
                return False, initial_context

        for i, current_image_path in enumerate(image_files):
            middle_page_name = current_image_path.name
            page_num = extract_page_number(current_image_path)
            if page_num is None:
                logger.warning(f"Could not extract page number from {middle_page_name}, skipping.")
                continue

            # --- Apply Effective Page Range Filter ---
            if start_page is not None and page_num < start_page:
                logger.debug(f"Skipping page {page_num} ({middle_page_name}): before effective start page {start_page}.")
                continue
            if end_page is not None and page_num > end_page:
                logger.info(f"Reached effective end page {end_page}. Stopping processing for this directory.")
                break

            # --- Check if it can be a 'middle' page (index check remains same) ---
            can_get_prev = i > 0
            can_get_next = i < len(image_files) - 1

            if not can_get_prev or not can_get_next:
                 logger.info(f"Skipping transcription for edge page number {page_num} ({middle_page_name}): Cannot form a 3-page set needed for context.")
                 continue

            # --- Proceed with Transcription ---
            logger.info(f"Processing page {page_num} ({middle_page_name}) (Index {i})")
            files_processed_count += 1

            montage_input_paths = [image_files[i-1], image_files[i], image_files[i+1]]
            base_name = f"page_{page_num:04d}"

            temp_composite_img_path = None # Initialize before try block
            try:
                temp_composite_img_path = create_temporary_composite_image(
                    montage_input_paths,
                    temp_composite_path_base, # Use the main temp dir for output
                    base_name
                )

                if temp_composite_img_path and temp_composite_img_path.exists():
                    transcription = None
                    try:
                        transcription = transcribe_image(
                            toolkit=image_toolkit, # Use the transcription toolkit
                            image_path=temp_composite_img_path,
                            original_middle_page_name=middle_page_name,
                            previous_transcription=last_successful_transcription,
                            prompt_template=prompt_template # Pass the template
                        )

                        if transcription:
                            transcription = transcription.strip()
                            if transcription.upper() == "SKIPPED":
                                logger.info(f"SKIPPED page {page_num} ({middle_page_name}) due to ancillary content.")
                            else:
                                # logger.info(f"Storing transcription for page {page_num} ({middle_page_name})") # Redundant
                                if output_path:
                                    try:
                                        with open(output_path, 'a', encoding='utf-8') as f:
                                            f.write(transcription + "\n\n")
                                    except IOError as e:
                                        logger.error(f"Error writing transcription for page {page_num} to {output_path}: {e}")
                                        processing_successful = False # Mark as failed if write error
                                else:
                                    print(f"--- Transcription for Page {page_num} ({middle_page_name}) ---")
                                    print(transcription)
                                    print("--- End Transcription ---\n")

                                last_successful_transcription = transcription
                        else:
                            logger.warning(f"--- Transcription FAILED (empty/failed result) for page {page_num} ({middle_page_name}) ---")
                            # Explicitly set processing_successful to False if transcription fails
                            processing_successful = False

                    except Exception as e_transcribe:
                         logger.error(f"An unexpected error occurred during transcription/writing for page {page_num} ({middle_page_name}): {e_transcribe}", exc_info=True)
                         processing_successful = False
                else:
                    logger.error(f"Failed to create or find composite image for page {page_num} ({middle_page_name}). Skipping.")
                    processing_successful = False # Mark as failed if composite wasn't created

            except Exception as e_outer:
                logger.error(f"Outer loop error for page {page_num} ({middle_page_name}): {e_outer}", exc_info=True)
                processing_successful = False
            finally:
                # *** IMMEDIATE CLEANUP of the composite image ***
                if temp_composite_img_path and temp_composite_img_path.exists():
                    try:
                        temp_composite_img_path.unlink()
                        logger.debug(f"Deleted temporary composite: {temp_composite_img_path.name}")
                    except OSError as e_del:
                        logger.warning(f"Could not delete temporary composite {temp_composite_img_path.name}: {e_del}")
                # The main temp directory `temp_composite_path_base` will be cleaned up by the `with` statement exit

        logger.info(f"Finished processing for directory {image_directory.name}. Attempted to process {files_processed_count} pages within the effective range.")
        return processing_successful, last_successful_transcription


# --- Function to parse model string to ModelType enum ---
def parse_model_string_to_enum(model_string: str, model_purpose: str) -> Optional[ModelType]:
    """
    Attempts to parse a string identifier into a ModelType enum member.

    Args:
        model_string: The string identifier from the command line.
        model_purpose: A string describing the purpose (e.g., "Analysis", "Transcription") for logging.

    Returns:
        The corresponding ModelType enum member if found, otherwise None.
    """
    try:
        # Attempt direct lookup using the string value
        model_enum = ModelType(model_string)
        logger.debug(f"Successfully parsed {model_purpose} model string '{model_string}' to enum {model_enum}")
        return model_enum
    except ValueError:
        # Handle cases where the string is not a direct value of any ModelType member
        logger.error(f"Invalid model string provided for {model_purpose}: '{model_string}'. "
                     f"It does not match any known ModelType enum value.")
        # Optionally list available values for better user feedback
        available_values = [m.value for m in ModelType]
        logger.info(f"Available ModelType values: {available_values}")
        return None

# --- Main Execution ---
def main(args):
    # Set logging level
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    # Force=True is important if basicConfig might have been called elsewhere
    logging.basicConfig(level=log_level_int, format='%(asctime)s - %(name)s [%(levelname)s] %(message)s', force=True)
    # Ensure our specific logger also respects the level
    logger.setLevel(log_level_int)
    # Optionally silence other noisy libraries if needed
    # logging.getLogger("some_library").setLevel(logging.WARNING)

    # --- Argument Validation ---
    if args.input_dir and args.book_dir:
        logger.error("Cannot specify both --input-dir and --book-dir. Choose one.")
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
    if process_pdfs:
        logger.info(f"Mode: Processing PDFs from directory: {source_dir}")
    else:
        logger.info(f"Mode: Processing images from directory: {source_dir}")

    # --- Store User-Provided Page Range ---
    user_start_page_arg = args.start_page
    user_end_page_arg = args.end_page

    if (user_start_page_arg is not None and user_end_page_arg is None) or \
       (user_start_page_arg is None and user_end_page_arg is not None):
        logger.error("Both --start-page and --end-page must be provided together if one is specified.")
        sys.exit(1)
    if user_start_page_arg is not None and user_end_page_arg is not None and user_start_page_arg > user_end_page_arg:
        logger.error(f"Start page ({user_start_page_arg}) cannot be greater than end page ({user_end_page_arg}).")
        sys.exit(1)

    # --- Determine Output Path and Mode ---
    single_output_file_path: Optional[Path] = None
    output_directory_path: Optional[Path] = None
    final_append_mode = args.append

    if not process_pdfs:
        if args.output:
            single_output_file_path = Path(args.output).resolve()
            logger.info(f"Output will be written to SINGLE file: {single_output_file_path}")
        else:
            logger.info("Output will be printed to console.")
    else:
        if args.output:
            output_directory_path = Path(args.output).resolve()
        else:
            output_directory_path = source_dir / "books_tts"
            logger.info(f"--output directory not specified, using default: {output_directory_path}")

        try:
            output_directory_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory for book transcriptions: {output_directory_path}")
        except OSError as e:
            logger.error(f"Failed to create output directory {output_directory_path}: {e}")
            sys.exit(1)

    # --- Parse Model Strings to Enums ---
    analysis_model_enum = parse_model_string_to_enum(args.analysis_model, "Analysis")
    transcription_model_enum = parse_model_string_to_enum(args.transcription_model, "Transcription")

    if analysis_model_enum is None or transcription_model_enum is None:
        logger.critical("Invalid model type provided. Cannot proceed.")
        sys.exit(1)

    # --- Initialize Models and Toolkits ---
    try:
        # == Analysis Model (for PDF structure) ==
        logger.info(f"Attempting to initialize PDF Analysis model enum: {analysis_model_enum}")
        analysis_model = ModelFactory.create(
            model_type=analysis_model_enum, # Use parsed enum
        )
        analysis_toolkit = ImageAnalysisToolkit(model=analysis_model)
        logger.info(f"Initialized PDF Analysis Toolkit with model: {analysis_model.model_type.value} (Platform: {analysis_model.model_platform.value})")


        # == Transcription Model (for page images) ==
        logger.info(f"Attempting to initialize Transcription model enum: {transcription_model_enum}")
        transcription_model = ModelFactory.create(
            model_type=transcription_model_enum, # Use parsed enum
        )
        transcription_toolkit = ImageAnalysisToolkit(model=transcription_model)
        logger.info(f"Initialized Transcription Toolkit with model: {transcription_model.model_type.value} (Platform: {transcription_model.model_platform.value})")

    except Exception as e:
        logger.error(f"Failed to initialize models or toolkits: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Initial Context (ONLY for single file output with append) ---
    overall_initial_context: Optional[str] = None
    if single_output_file_path and final_append_mode:
        if single_output_file_path.exists() and single_output_file_path.stat().st_size > 0:
            try:
                logger.info(f"Append mode active for single file. Reading context from end of: {single_output_file_path}")
                with open(single_output_file_path, 'r', encoding='utf-8') as f:
                    read_size = 4096 # Increased chunk size
                    f.seek(max(0, single_output_file_path.stat().st_size - read_size))
                    context_from_file = f.read()
                    if context_from_file:
                        overall_initial_context = context_from_file.strip()
                        logger.info(f"Successfully loaded context from single file. Last ~{len(overall_initial_context)} chars will be used.")
                    else:
                        logger.warning(f"Could not read context from {single_output_file_path}, though it exists and is not empty.")
            except IOError as e:
                logger.error(f"Error reading context from {single_output_file_path}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading context from {single_output_file_path}: {e}", exc_info=True)
        else:
            logger.info(f"Append mode active for single file, but output file {single_output_file_path} does not exist or is empty. Starting fresh.")

        try:
            if single_output_file_path.exists():
                 with open(single_output_file_path, 'a', encoding='utf-8') as f:
                    separator = f"\n\n{'='*20} Appending transcriptions starting at {datetime.datetime.now()} {'='*20}\n\n"
                    f.write(separator)
                    logger.info(f"Added append separator to {single_output_file_path}")
        except IOError as e:
            logger.error(f"Error adding append separator to {single_output_file_path}: {e}")


    # --- Main Processing ---
    if not process_pdfs:
        # --- Process Single Image Directory ---
        logger.info(f"Processing images directly from: {source_dir}")
        success, _ = process_images_in_directory(
            image_directory=source_dir,
            output_path=single_output_file_path,
            append_mode=final_append_mode,
            start_page=user_start_page_arg,
            end_page=user_end_page_arg,
            image_toolkit=transcription_toolkit,
            prompt_template=PROMPT_TEMPLATE,
            initial_context=overall_initial_context
        )
        if not success:
             logger.error(f"Processing failed for image directory: {source_dir}")
        logger.info("Image directory processing finished.")

    else:
        # --- Process PDFs in Book Directory (Output to Directory) ---
        pdf_files = sorted(list(source_dir.glob('*.pdf')))
        if not pdf_files:
            logger.warning(f"No PDF files found in the book directory: {source_dir}")
            sys.exit(0)

        logger.info(f"Found {len(pdf_files)} PDF files to process.")

        for pdf_path in pdf_files:
            logger.info(f"--- Starting processing for PDF: {pdf_path.name} ---")
            book_specific_output_path = output_directory_path / f"{pdf_path.stem}.txt"
            logger.info(f"Output for '{pdf_path.name}' will be saved to: {book_specific_output_path}")

            # --- Determine Effective Page Range for this PDF ---
            effective_start_page = user_start_page_arg
            effective_end_page = user_end_page_arg

            if user_start_page_arg is None and user_end_page_arg is None:
                logger.info(f"Attempting to auto-detect content page range for {pdf_path.name}...")
                analysis_response = analyze_pdf_structure(analysis_toolkit, pdf_path)
                llm_start_page, llm_end_page = parse_pdf_analysis_response(analysis_response)
                if llm_start_page is not None and llm_end_page is not None:
                    effective_start_page = llm_start_page
                    effective_end_page = llm_end_page
                    logger.info(f"Using LLM-determined page range for {pdf_path.name}: {effective_start_page}-{effective_end_page}")
                else:
                    logger.warning(f"Could not auto-detect page range for {pdf_path.name}. Will process all pages.")
            else:
                 logger.info(f"Using user-provided page range for {pdf_path.name}: {effective_start_page}-{effective_end_page}")

            current_book_initial_context = None
            if final_append_mode and book_specific_output_path.exists() and book_specific_output_path.stat().st_size > 0:
                try:
                    logger.info(f"Append mode active for '{pdf_path.name}'. Reading context from: {book_specific_output_path}")
                    with open(book_specific_output_path, 'r', encoding='utf-8') as bf:
                        read_size = 4096
                        bf.seek(max(0, book_specific_output_path.stat().st_size - read_size))
                        current_book_initial_context = bf.read().strip()
                        if current_book_initial_context:
                             logger.info(f"Loaded context for book {pdf_path.name}. Length: {len(current_book_initial_context)}")
                        else:
                             logger.warning(f"Could not read context from existing file: {book_specific_output_path}")
                except IOError as e:
                    logger.error(f"Error reading context for {book_specific_output_path}: {e}")
                except Exception as e:
                     logger.error(f"Unexpected error loading context from {book_specific_output_path}: {e}", exc_info=True)

            if final_append_mode:
                try:
                    if book_specific_output_path.exists():
                         with open(book_specific_output_path, 'a', encoding='utf-8') as bf:
                             separator = f"\n\n{'='*20} Appending transcriptions starting at {datetime.datetime.now()} {'='*20}\n\n"
                             bf.write(separator)
                             logger.info(f"Added append separator to {book_specific_output_path}")
                except IOError as e:
                    logger.error(f"Error preparing book output file {book_specific_output_path} for appending: {e}. Skipping this book.")
                    continue

            temp_image_dir_path = None
            try:
                temp_dir_name = f"owl_pdf_{pdf_path.stem}_{os.getpid()}"
                temp_image_dir_path = Path(tempfile.gettempdir()) / temp_dir_name
                temp_image_dir_path.mkdir(exist_ok=True)

                logger.info(f"Created temporary image directory for '{pdf_path.name}': {temp_image_dir_path}")

                if not convert_pdf_to_images(pdf_path, temp_image_dir_path):
                    logger.error(f"Failed to convert PDF '{pdf_path.name}' to images. Skipping this book.")
                    if temp_image_dir_path and temp_image_dir_path.exists():
                         shutil.rmtree(temp_image_dir_path, ignore_errors=True)
                         logger.info(f"Cleaned up temporary directory after failed conversion: {temp_image_dir_path}")
                    continue

                success, _ = process_images_in_directory(
                    image_directory=temp_image_dir_path,
                    output_path=book_specific_output_path,
                    append_mode=final_append_mode,
                    start_page=effective_start_page,
                    end_page=effective_end_page,
                    image_toolkit=transcription_toolkit,
                    prompt_template=PROMPT_TEMPLATE,
                    initial_context=current_book_initial_context
                )

                if success:
                    logger.info(f"Successfully processed images for PDF: {pdf_path.name}")
                else:
                    logger.error(f"Processing failed for images converted from PDF: {pdf_path.name}.")

                if temp_image_dir_path and temp_image_dir_path.exists():
                    shutil.rmtree(temp_image_dir_path, ignore_errors=True)
                    logger.info(f"Cleaned up temporary image directory: {temp_image_dir_path}")

            except Exception as e:
                 logger.critical(f"A critical error occurred processing PDF {pdf_path.name}: {e}", exc_info=True)
                 if temp_image_dir_path and temp_image_dir_path.exists():
                    logger.warning(f"Attempting cleanup after critical error for: {temp_image_dir_path}")
                    shutil.rmtree(temp_image_dir_path, ignore_errors=True)


            logger.info(f"--- Finished processing for PDF: {pdf_path.name} ---")

        logger.info("All PDF processing finished.")


if __name__ == "__main__":
    # Define default model strings using values from ModelType enum for validation
    DEFAULT_ANALYSIS_MODEL = ModelType.GEMINI_2_5_PRO_EXP.value
    DEFAULT_TRANSCRIPTION_MODEL = ModelType.GEMINI_2_FLASH_LITE_PREVIEW.value


    parser = argparse.ArgumentParser(
        description="Transcribe text from book page images or PDFs using Owl/CAMEL-AI.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--input-dir", help="Directory containing the image files (named like 'prefix-digits.ext').")
    input_group.add_argument("-b", "--book-dir", help="Directory containing PDF book files to process.")

    parser.add_argument(
        "-o", "--output",
        help="Output path. If using --input-dir, this is the path to the SINGLE output file (optional, defaults to console).\n"
             "If using --book-dir, this is the path to the output DIRECTORY where individual '<pdf_name>.txt' files will be saved.\n"
             "(optional, defaults to a 'books_tts' subdirectory within the --book-dir)."
    )
    parser.add_argument(
        "-a", "--append",
        action="store_true",
        help="Append to the output file(s) instead of overwriting. Reads context from end of existing file(s). Respects the output mode."
    )
    parser.add_argument("--start-page", type=int, help="Manually specify the first page number to transcribe (inclusive). Overrides LLM analysis if provided for PDFs.")
    parser.add_argument("--end-page", type=int, help="Manually specify the last page number to transcribe (inclusive). Overrides LLM analysis if provided for PDFs.")

    # --- Updated Model Arguments ---
    parser.add_argument(
        "--analysis-model",
        type=str,
        default=DEFAULT_ANALYSIS_MODEL,
        help=f"Model string identifier for PDF structure analysis (default: '{DEFAULT_ANALYSIS_MODEL}').\n"
             f"Must match a value in CAMEL's ModelType enum."
    )
    parser.add_argument(
        "--transcription-model",
        type=str,
        default=DEFAULT_TRANSCRIPTION_MODEL,
        help=f"Model string identifier for image transcription (default: '{DEFAULT_TRANSCRIPTION_MODEL}').\n"
              f"Must match a value in CAMEL's ModelType enum."
    )
    # --- End Updated Model Arguments ---

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)."
    )

    # Dynamically add available model types to help string if possible
    try:
        available_models_str = ", ".join(f"'{m.value}'" for m in ModelType)
        model_help_epilog = f"\n\nAvailable ModelType values for --analysis-model and --transcription-model:\n{available_models_str}"
    except NameError: # ModelType might not be defined if CAMEL import fails
        model_help_epilog = "\n\n(Could not list available ModelType values)"

    parser.epilog = f"""
Example Usage:
  # Transcribe images, overwriting single output file (default models)
  python %(prog)s --input-dir path/to/book_pages -o output.txt

  # Transcribe images, print to console (default models)
  python %(prog)s --input-dir path/to/book_pages

  # Transcribe images, append to single output file, using specific transcription model
  python %(prog)s --input-dir path/to/pages -o output.txt --append --transcription-model gpt-4o-mini

  # Transcribe PDFs, auto-detect range, save to specific dir (default models)
  python %(prog)s --book-dir path/to/pdfs -o path/to/output_dir

  # Transcribe PDFs, auto-detect range, save to default dir 'books_tts' (default models)
  python %(prog)s --book-dir path/to/pdfs

  # Transcribe PDFs, auto-detect range, save to specific dir, append, use specific models
  python %(prog)s --book-dir path/to/pdfs -o path/to/output_dir --append \\
    --analysis-model claude-3-5-sonnet \\
    --transcription-model gpt-4o

  # Transcribe PDFs, force pages 50-150, save to specific dir, append (default models)
  python %(prog)s --book-dir path/to/pdfs --start-page 50 --end-page 150 -o path/to/output_dir --append

  # Run PDF processing with auto-detect, debug logging, output to default dir, append (default models)
  python %(prog)s --book-dir path/to/pdfs --log-level DEBUG --append

Notes:
- Requires 'pdftoppm' (poppler-utils) and 'montage' (ImageMagick).
- Images must be named like '-[digits].ext'.
- First/last image files are skipped (need 3-page context for transcription).
- Configure API keys in a .env file (e.g., GEMINI_API_KEY=..., OPENAI_API_KEY=..., ANTHROPIC_API_KEY=...).
- Use model string identifiers that match values in CAMEL's ModelType enum.
- PDF page range auto-detection runs only if --start-page/--end-page are NOT provided.
- When using --append:
    - With --input-dir: Reads context from the end of the single output file.
    - With --book-dir: Reads context from the end of the *specific* '<pdf_name>.txt' file for each book.
- Interrupted processes may leave temporary directories. Check and clean manually if needed.
{model_help_epilog}
"""
    parsed_args = parser.parse_args()

    main(parsed_args)
