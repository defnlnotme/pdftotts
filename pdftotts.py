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
from typing import List, Optional, Tuple, Type, Dict, Any
from time import sleep
import tempfile
import datetime # Import datetime for append message
import shutil # Import for directory removal
import json # Import JSON for parsing analysis response and state

from dotenv import load_dotenv

from camel.toolkits import ImageAnalysisToolkit
from camel.models import ModelFactory
# <<< ADDED: Import ModelPlatformType >>>
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
load_dotenv(".env", verbose=True)

# --- Constants ---
RESUME_STATE_FILENAME = "_pdftotts_resume_state.json"
END_MARKER = "The End."
TEMP_IMAGE_SUBDIR = ".pdftotts_images" # Subdirectory name for images within book_dir
SYSTEM_TEMP_DIR_PATTERN = "/tmp/image_*.png" # Pattern for library temp files


# --- Helper Functions (Unchanged + State Management) ---
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

def analyze_pdf_structure(
    analysis_toolkit: ImageAnalysisToolkit,
    pdf_path: Path
) -> Optional[str]:
    """Uses a specified LLM toolkit to analyze the PDF structure."""
    logger.info(f"Analyzing PDF structure for: {pdf_path.name} using model {analysis_toolkit.model.model_type.value}")
    response = None
    try:
        response = analysis_toolkit.ask_question_about_image(
            image_path=str(pdf_path),
            question=PDF_ANALYSIS_PROMPT
        )
        if response:
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
    finally:
        # <<< ADDED: Cleanup after analysis attempt >>>
        cleanup_system_temp_files()

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
    previous_transcription: Optional[str] = None,
    prompt_template: str = PROMPT_TEMPLATE
) -> Optional[str]:
    """Uses the ImageAnalysisToolkit to transcribe text from a composite image."""
    logger.info(f"Transcribing composite image representing middle page: {original_middle_page_name} using {toolkit.model.model_type.value}")


    # --- ADDED: Prepare context_section ---
    if previous_transcription:
        # Optional: Limit context length if very long
        max_context_chars = 2000 # Example limit
        truncated_context = previous_transcription[-max_context_chars:]
        context_section = f"<previous_transcription>\n{truncated_context}\n</previous_transcription>"
        logger.debug(f"Providing last ~{len(truncated_context)} chars of previous transcription as context.")
    else:
        context_section = "<previous_transcription>No previous transcription context provided.</previous_transcription>"
        logger.debug("No previous transcription context provided.")

    # ... (context preparation logic remains the same) ...
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
                # <<< MOVED: Cleanup is now in the finally block below >>>
            except Exception as e_inner:
                 logger.warning(f"Transcription attempt {retries + 1}/{max_retries} API call failed for {original_middle_page_name}: {e_inner}. Retrying in {retry_delay}s...")
                 sleep(retry_delay)
                 retry_delay *= 2
                 retries += 1
                 continue # Move to next retry attempt

            # --- Process successful transcription attempt ---
            if transcription:
                transcription = transcription.strip()
                transcription = re.sub(r'^```(text)?\s*', '', transcription, flags=re.IGNORECASE | re.MULTILINE)
                transcription = re.sub(r'\s*```$', '', transcription, flags=re.IGNORECASE | re.MULTILINE)
                transcription = transcription.strip('"\'')

                if transcription and "Analysis failed" not in transcription and len(transcription) > 5:
                    return transcription # Success! Return the result
                else:
                    # Transcription succeeded but content is invalid/short
                    logger.warning(f"Transcription attempt {retries + 1}/{max_retries} failed or produced short/invalid output for middle page: {original_middle_page_name}. Content: '{transcription[:50]}...'. Retrying in {retry_delay}s...")
                    # Reset transcription for the next retry
                    transcription = None
                    sleep(retry_delay)
                    retry_delay *= 2
                    retries += 1
                    continue # Move to next retry attempt
            else:
                # Transcription call returned None or empty
                logger.warning(f"Received empty transcription on attempt {retries + 1}/{max_retries} for middle page: {original_middle_page_name}. Retrying in {retry_delay}s...")
                sleep(retry_delay)
                retry_delay *= 2
                retries += 1
                continue # Move to next retry attempt

        # If loop finishes without returning, all retries failed
        logger.error(f"Transcription FAILED after {max_retries} attempts for middle page: {original_middle_page_name} (composite: {image_path.name}). Returning None.")
        return None # Explicitly return None after all retries fail

    except Exception as e:
        # Catch any unexpected error during the retry loop itself
        logger.error(f"Unhandled error during transcription retry loop for image {image_path.name} (middle page: {original_middle_page_name}): {e}", exc_info=True)
        return None # Return None on unexpected error
    finally:
        # <<< ADDED: Cleanup after transcription attempt (success, failure, or error) >>>
        cleanup_system_temp_files()

def convert_pdf_to_images(pdf_path: Path, output_dir: Path, image_format: str = 'png', dpi: int = 300) -> bool:
    """Converts a PDF file to images using pdftoppm."""
    if not pdf_path.is_file():
        logger.error(f"PDF file not found: {pdf_path}")
        return False

    if shutil.which("pdftoppm") is None:
        logger.error("Error: 'pdftoppm' command not found. Please ensure poppler-utils is installed and in your system's PATH.")
        return False

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

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
        state_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with open(state_file_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=4)
        logger.debug(f"Saved resume state to {state_file_path}")
    except IOError as e:
        logger.error(f"Error writing state file {state_file_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving state file {state_file_path}: {e}", exc_info=True)

# --- Modified Image Processing Function ---
def process_images_in_directory(
    image_directory: Path,
    output_path: Optional[Path],
    append_mode: bool,
    start_page: Optional[int], # Effective start page for this run
    end_page: Optional[int],   # Effective end page for this run
    image_toolkit: ImageAnalysisToolkit,
    prompt_template: str,
    initial_context: Optional[str],
    # --- New state parameters ---
    pdf_filename: Optional[str] = None, # Only provided in book mode
    state_file_path: Optional[Path] = None, # Only provided in book mode
    state_data: Optional[Dict[str, Any]] = None # Only provided in book mode
) -> Tuple[bool, Optional[str]]:
    """
    Processes images, performs transcription within range, updates state if applicable.
    """
    logger.info(f"Processing directory: {image_directory.name} [Pages {start_page or 'start'} to {end_page or 'end'}]")
    # (Rest of logging based on start/end page remains the same)

    image_files = find_and_sort_image_files(image_directory)
    if not image_files:
        logger.warning(f"No valid image files found in {image_directory}. Skipping.")
        return True, initial_context # Still success, just nothing to do

    logger.info(f"Found {len(image_files)} valid images to potentially process in {image_directory.name}.")

    files_processed_count = 0
    # Use initial_context for the very first transcription, then update last_successful
    last_successful_transcription = initial_context
    processing_successful = True # Assume success unless error occurs

    # Create a temporary directory for composites *within* the image directory itself
    temp_composite_dir = image_directory / ".composites"
    try:
        temp_composite_dir.mkdir(exist_ok=True)
        logger.info(f"Using temporary directory for composites: {temp_composite_dir}")
    except OSError as e_comp_dir:
        logger.error(f"Could not create composite dir {temp_composite_dir}: {e_comp_dir}. Cannot proceed.")
        return False, initial_context

    if output_path and not append_mode:
        # Ensure output file is truncated only if not appending
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f_init:
                f_init.truncate(0)
            logger.info(f"Output file {output_path} truncated (or created).")
        except IOError as e:
            logger.error(f"Error preparing output file {output_path}: {e}")
            # Don't delete temp_composite_dir here, outer finally will handle it
            return False, initial_context

    try: # Wrap the main loop for composite dir cleanup
        for i, current_image_path in enumerate(image_files):
            middle_page_name = current_image_path.name
            page_num = extract_page_number(current_image_path)
            if page_num is None:
                logger.warning(f"Could not extract page number from {middle_page_name}, skipping.")
                continue

            # --- Apply Effective Page Range Filter ---
            # Note: start_page here is already adjusted for resumption if needed
            if start_page is not None and page_num < start_page:
                logger.debug(f"Skipping page {page_num} ({middle_page_name}): before effective start {start_page}.")
                continue
            if end_page is not None and page_num > end_page:
                logger.info(f"Reached effective end page {end_page}. Stopping processing for this directory.")
                break # Stop processing this directory

            can_get_prev = i > 0
            can_get_next = i < len(image_files) - 1
            if not can_get_prev or not can_get_next:
                 logger.info(f"Skipping edge page {page_num} ({middle_page_name}): Cannot form 3-page set.")
                 continue

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
                        # Use last_successful_transcription for context between pages
                        transcription = transcribe_image(
                            toolkit=image_toolkit,
                            image_path=temp_composite_img_path,
                            original_middle_page_name=middle_page_name,
                            previous_transcription=last_successful_transcription, # Pass context
                            prompt_template=prompt_template
                        )

                        if transcription:
                            transcription = transcription.strip()
                            if transcription.upper() == "SKIPPED":
                                logger.info(f"SKIPPED page {page_num} ({middle_page_name}) due to ancillary content.")
                                # Still update state if skipping non-narrative page successfully
                                if pdf_filename and state_file_path and state_data is not None:
                                    state_data[pdf_filename] = page_num
                                    save_resume_state(state_file_path, state_data)
                                    logger.debug(f"Updated resume state after skipping page {page_num}")

                            else:
                                if output_path:
                                    try:
                                        with open(output_path, 'a', encoding='utf-8') as f:
                                            f.write(transcription + "\n\n") # Add spacing

                                        # --- Update State AFTER successful write ---
                                        if pdf_filename and state_file_path and state_data is not None:
                                            state_data[pdf_filename] = page_num
                                            save_resume_state(state_file_path, state_data)
                                            logger.debug(f"Updated resume state for {pdf_filename} to page {page_num}")

                                    except IOError as e:
                                        logger.error(f"Error writing transcription for page {page_num} to {output_path}: {e}")
                                        processing_successful = False # Mark as failed
                                else: # Print to console if no output path
                                    print(f"--- Transcription for Page {page_num} ({middle_page_name}) ---")
                                    print(transcription)
                                    print("--- End Transcription ---\n")
                                    # Cannot update state reliably without output file

                                # Update context for the *next* page
                                last_successful_transcription = transcription
                        else:
                            logger.warning(f"--- Transcription FAILED (empty/failed result) for page {page_num} ({middle_page_name}) ---")
                            processing_successful = False # Mark as failed

                    except Exception as e_transcribe:
                         logger.error(f"Error during transcription/writing for page {page_num} ({middle_page_name}): {e_transcribe}", exc_info=True)
                         processing_successful = False # Mark as failed
                else:
                    logger.error(f"Failed to create composite image for page {page_num} ({middle_page_name}). Skipping.")
                    processing_successful = False # Mark as failed

            except Exception as e_outer:
                logger.error(f"Outer loop error for page {page_num} ({middle_page_name}): {e_outer}", exc_info=True)
                processing_successful = False # Mark as failed
            finally:
                # Clean up individual composite after use
                if temp_composite_img_path and temp_composite_img_path.exists():
                    try:
                        temp_composite_img_path.unlink()
                        logger.debug(f"Deleted temporary composite: {temp_composite_img_path.name}")
                    except OSError as e_del:
                        logger.warning(f"Could not delete temporary composite {temp_composite_img_path.name}: {e_del}")

    finally: # Cleanup the main composite directory
        if temp_composite_dir.exists():
            try:
                shutil.rmtree(temp_composite_dir)
                logger.debug(f"Cleaned up composites directory: {temp_composite_dir}")
            except OSError as e_final_clean:
                 logger.warning(f"Could not clean up composites directory {temp_composite_dir}: {e_final_clean}")


    logger.info(f"Finished loop for directory {image_directory.name}. Processed {files_processed_count} pages within effective range.")
    return processing_successful, last_successful_transcription

# --- Function to parse model string to ModelType enum (Unchanged) ---
def parse_model_string_to_enum(model_string: str, model_purpose: str) -> Optional[ModelType]:
    try:
        model_enum = ModelType(model_string)
        logger.debug(f"Successfully parsed {model_purpose} model string '{model_string}' to enum {model_enum}")
        return model_enum
    except ValueError:
        logger.error(f"Invalid model string provided for {model_purpose}: '{model_string}'. "
                     f"It does not match any known ModelType enum value.")
        available_values = [m.value for m in ModelType]
        logger.info(f"Available ModelType values: {available_values}")
        return None

# --- Function to check if file ends with marker ---
def check_if_file_completed(file_path: Path, marker: str) -> bool:
    """Checks if the file exists and ends with the specified marker."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read a chunk from the end
            read_size = len(marker) + 50 # Read a bit more than the marker
            f.seek(max(0, file_path.stat().st_size - read_size))
            last_chunk = f.read()
            # Check if the marker is present at the very end, allowing for trailing whitespace
            return last_chunk.strip().endswith(marker)
    except IOError as e:
        logger.warning(f"Could not read end of file {file_path} to check completion: {e}")
        return False # Assume not completed if read fails
    except Exception as e:
        logger.error(f"Unexpected error checking file completion for {file_path}: {e}", exc_info=True)
        return False


# --- ADDED: Helper function to get platform from ModelType ---
def get_platform_from_model_type(model_type: ModelType) -> Optional[ModelPlatformType]:
    """Tries to infer the ModelPlatformType from a ModelType enum."""
    model_name = model_type.name.upper()
    model_value = model_type.value

    # Check by name convention
    if model_name.startswith("GPT") or model_name.startswith("O3"):
        return ModelPlatformType.OPENAI
    elif model_name.startswith("GEMINI"):
        return ModelPlatformType.GEMINI
    elif model_name.startswith("AZURE"):
        return ModelPlatformType.AZURE
    elif model_name.startswith("QWEN"):
        return ModelPlatformType.QWEN
    elif model_name.startswith("DEEPSEEK"):
        return ModelPlatformType.DEEPSEEK
    elif model_name.startswith("GROQ") or model_name.startswith("LLAMA") or model_name.startswith("MIXTRAL"): # Groq specific
        return ModelPlatformType.GROQ
    elif model_name.startswith("CLAUDE"):
        # Assuming Claude might be handled via OpenAI compatible or specific platform
        # Let's default to OPENAI for now, might need adjustment if CAMEL has ANTHROPIC platform
        logger.warning(f"Assuming OPENAI platform for Claude model: {model_name}. Adjust if needed.")
        return ModelPlatformType.OPENAI

    # Check by value convention (e.g., "openrouter/...")
    if isinstance(model_value, str) and "/" in model_value:
        platform_str = model_value.split('/')[0].upper()
        try:
            # Try to find a matching enum member by the derived upper-case name
            platform_enum = getattr(ModelPlatformType, platform_str, None)
            if platform_enum:
                logger.debug(f"Inferred platform '{platform_enum}' from model value: {model_value}")
                return platform_enum
            else:
                 logger.warning(f"Could not map derived platform string '{platform_str}' to ModelPlatformType enum from value '{model_value}'.")
        except Exception as e:
             logger.warning(f"Error trying to map platform string '{platform_str}' from value '{model_value}': {e}")


    # Default or fallback
    logger.warning(f"Could not infer platform for model type: {model_type}. Defaulting to OPENAI. Check if this is correct.")
    return ModelPlatformType.OPENAI


def run_main_logic(args):
    # ... (The logic inside run_main_logic remains the same) ...
    # Calls to analyze_pdf_structure and process_images_in_directory (which calls transcribe_image)
    # will now trigger the cleanup internally after each major toolkit operation.
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level_int, format='%(asctime)s - %(name)s [%(levelname)s] %(message)s', force=True)
    logger.setLevel(log_level_int)

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
        logger.error("Both --start-page and --end-page must be provided together if specified.")
        sys.exit(1)
    if user_start_page_arg is not None and user_end_page_arg is not None and user_start_page_arg > user_end_page_arg:
        logger.error(f"Start page ({user_start_page_arg}) cannot be > end page ({user_end_page_arg}).")
        sys.exit(1)

    single_output_file_path: Optional[Path] = None
    output_directory_path: Optional[Path] = None
    resume_state_file_path: Optional[Path] = None # Path to the state file
    final_append_mode = args.append

    if not process_pdfs:
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
            # Define state file path only in book mode
            resume_state_file_path = output_directory_path / RESUME_STATE_FILENAME
            logger.info(f"Resume state will be managed in: {resume_state_file_path}")
        except OSError as e:
            logger.error(f"Failed to create output directory {output_directory_path}: {e}")
            sys.exit(1)

    analysis_model_enum = parse_model_string_to_enum(args.analysis_model, "Analysis")
    transcription_model_enum = parse_model_string_to_enum(args.transcription_model, "Transcription")

    if analysis_model_enum is None or transcription_model_enum is None:
        logger.critical("Invalid model type provided. Cannot proceed.")
        sys.exit(1)

    # <<< Determine platforms >>>
    analysis_platform = get_platform_from_model_type(analysis_model_enum)
    transcription_platform = get_platform_from_model_type(transcription_model_enum)

    if analysis_platform is None:
        logger.critical(f"Could not determine platform for analysis model: {analysis_model_enum.value}. Cannot proceed.")
        sys.exit(1)
    if transcription_platform is None:
        logger.critical(f"Could not determine platform for transcription model: {transcription_model_enum.value}. Cannot proceed.")
        sys.exit(1)

    logger.info(f"Determined analysis platform: {analysis_platform}")
    logger.info(f"Determined transcription platform: {transcription_platform}")

    try:
        logger.info(f"Initializing PDF Analysis model: {analysis_model_enum.value} (Platform: {analysis_platform})")
        analysis_model = ModelFactory.create(
            model_platform=analysis_platform,
            model_type=analysis_model_enum
        )
        analysis_toolkit = ImageAnalysisToolkit(model=analysis_model)
        logger.info(f"Initialized PDF Analysis Toolkit: {analysis_model.model_type.value}")

        logger.info(f"Initializing Transcription model: {transcription_model_enum.value} (Platform: {transcription_platform})")
        transcription_model = ModelFactory.create(
            model_platform=transcription_platform,
            model_type=transcription_model_enum
        )
        transcription_toolkit = ImageAnalysisToolkit(model=transcription_model)
        logger.info(f"Initialized Transcription Toolkit: {transcription_model.model_type.value}")

    except Exception as e:
        logger.error(f"Failed to initialize models or toolkits: {e}", exc_info=True)
        sys.exit(1)

    overall_initial_context: Optional[str] = None
    if single_output_file_path and final_append_mode:
        # (Context loading for single file remains the same as before)
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
        try: # Add separator
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
            start_page=user_start_page_arg, # Use direct args here
            end_page=user_end_page_arg,
            image_toolkit=transcription_toolkit,
            prompt_template=PROMPT_TEMPLATE,
            initial_context=overall_initial_context,
            # No state needed for single image dir mode
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
        # Load the overall state ONCE before the loop
        current_state_data = load_resume_state(resume_state_file_path) if resume_state_file_path else {}

        for pdf_path in pdf_files:
            pdf_filename = pdf_path.name
            logger.info(f"--- Starting processing for PDF: {pdf_filename} ---")
            book_specific_output_path = output_directory_path / f"{pdf_path.stem}.txt"
            logger.info(f"Output for '{pdf_filename}' will be: {book_specific_output_path}")

            # --- Check if already completed ---
            if check_if_file_completed(book_specific_output_path, END_MARKER):
                logger.info(f"'{pdf_filename}' already marked as completed ('{END_MARKER}' found). Skipping.")
                continue

            # --- Determine Effective Page Range & Resumption ---
            effective_start_page = user_start_page_arg
            effective_end_page = user_end_page_arg
            resuming = False

            # Get last processed page from state
            last_processed_page = current_state_data.get(pdf_filename)

            if last_processed_page is not None:
                 resume_start_page = last_processed_page + 1
                 logger.info(f"Found resume state for '{pdf_filename}': Last processed page was {last_processed_page}. Attempting to resume from page {resume_start_page}.")
                 resuming = True
            else:
                 resume_start_page = 1 # Default if no state

            # Determine base start/end (LLM or user args)
            base_start_page = user_start_page_arg
            base_end_page = user_end_page_arg
            if base_start_page is None: # No user args, try LLM analysis
                logger.info(f"Attempting LLM analysis for page range of {pdf_filename}...")
                analysis_response = analyze_pdf_structure(analysis_toolkit, pdf_path) # Cleanup happens inside
                llm_start, llm_end = parse_pdf_analysis_response(analysis_response)
                if llm_start is not None and llm_end is not None:
                    base_start_page = llm_start
                    base_end_page = llm_end
                    logger.info(f"Using LLM-determined range: {base_start_page}-{base_end_page}")
                else:
                    logger.warning(f"LLM analysis failed for {pdf_filename}. Will process all pages or use resume state.")
                    # Leave base_start/end as None, rely on resume or full processing
            else:
                logger.info(f"Using user-provided range: {base_start_page}-{base_end_page}")

            # Calculate final effective start page
            if resuming:
                # If resuming, the start page is the *later* of the resume point or the base start
                effective_start_page = max(resume_start_page, base_start_page or 1)
            else:
                # If not resuming, use the base start page (or None if LLM failed and no user args)
                effective_start_page = base_start_page

            effective_end_page = base_end_page # End page is just the base end page

            if effective_start_page is not None and effective_end_page is not None and effective_start_page > effective_end_page:
                 logger.warning(f"Effective start page {effective_start_page} is after effective end page {effective_end_page} for '{pdf_filename}'. Nothing to process.")
                 continue # Skip this PDF if range is invalid after resumption logic


            # --- Load Context for Append/Resume ---
            current_book_initial_context = None
            if (final_append_mode or resuming) and book_specific_output_path.exists() and book_specific_output_path.stat().st_size > 0:
                try:
                    logger.info(f"Reading context from existing file: {book_specific_output_path}")
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


            # Add append separator if appending and file exists
            if final_append_mode and book_specific_output_path.exists():
                 try:
                     with open(book_specific_output_path, 'a', encoding='utf-8') as bf:
                         separator = f"\n\n{'='*20} Appending at {datetime.datetime.now()} {'='*20}\n\n"
                         bf.write(separator)
                 except IOError as e:
                    logger.error(f"Error preparing file {book_specific_output_path} for appending: {e}. Skipping book.")
                    continue

            # --- Process the PDF ---
            # Create a subdirectory within the book directory for temporary images
            images_base_dir = source_dir / TEMP_IMAGE_SUBDIR
            pdf_image_dir_path = images_base_dir / pdf_path.stem # Subdir named after the PDF stem
            pdf_processing_success = False # Track success for adding "The End."

            try:
                # Ensure base dir exists
                images_base_dir.mkdir(exist_ok=True)
                # Clean up any old images for this PDF first
                if pdf_image_dir_path.exists():
                    try:
                        shutil.rmtree(pdf_image_dir_path)
                        logger.debug(f"Cleaned up pre-existing image directory: {pdf_image_dir_path}")
                    except OSError as e_clean_old:
                        logger.warning(f"Could not clean up pre-existing image directory {pdf_image_dir_path}: {e_clean_old}")
                # Create the specific dir for this PDF's images
                try:
                    pdf_image_dir_path.mkdir(parents=False, exist_ok=False) # Should not exist after cleanup
                    logger.info(f"Image dir for '{pdf_filename}': {pdf_image_dir_path}")
                except FileExistsError:
                    logger.warning(f"Image directory {pdf_image_dir_path} still exists after cleanup attempt. Proceeding.")
                except OSError as e_mkdir:
                    logger.error(f"Failed to create image directory {pdf_image_dir_path}: {e_mkdir}")
                    continue # Skip this PDF if dir creation fails

                if not convert_pdf_to_images(pdf_path, pdf_image_dir_path):
                    logger.error(f"Failed PDF conversion for '{pdf_filename}'. Skipping.")
                    # No need to update state if conversion failed before processing
                else:
                    pdf_processing_success, _ = process_images_in_directory(
                        image_directory=pdf_image_dir_path,
                        output_path=book_specific_output_path,
                        append_mode=final_append_mode or resuming, # Append if either flag is true or resuming
                        start_page=effective_start_page,
                        end_page=effective_end_page,
                        image_toolkit=transcription_toolkit,
                        prompt_template=PROMPT_TEMPLATE,
                        initial_context=current_book_initial_context,
                        # Pass state info
                        pdf_filename=pdf_filename,
                        state_file_path=resume_state_file_path,
                        state_data=current_state_data # Pass the dictionary
                    )

                    if pdf_processing_success:
                        logger.info(f"Successfully processed images for PDF: {pdf_filename}")
                        # --- Add "The End." marker ---
                        try:
                            with open(book_specific_output_path, 'a', encoding='utf-8') as f_end:
                                f_end.write(f"\n\n{END_MARKER}\n")
                            logger.info(f"Added '{END_MARKER}' to completed file: {book_specific_output_path.name}")
                        except IOError as e:
                             logger.error(f"Error adding '{END_MARKER}' to {book_specific_output_path.name}: {e}")
                             pdf_processing_success = False # Mark as failed if marker write fails
                    else:
                        logger.error(f"Processing failed for images from PDF: {pdf_filename}.")
                        # State was hopefully updated during the failed run up to the last good page

            except Exception as e:
                 logger.critical(f"Critical error processing PDF {pdf_filename}: {e}", exc_info=True)
            finally: # Cleanup pdf image dir
                if pdf_image_dir_path and pdf_image_dir_path.exists() and not args.keep_images: # Only delete if keep_images is False
                    try:
                        shutil.rmtree(pdf_image_dir_path)
                        logger.info(f"Cleaned up PDF image directory: {pdf_image_dir_path}")
                    except Exception as e_clean:
                         logger.error(f"Error cleaning up PDF image dir {pdf_image_dir_path}: {e_clean}")
                elif pdf_image_dir_path and args.keep_images:
                    logger.info(f"Keeping PDF image directory due to --keep-images flag: {pdf_image_dir_path}")

            logger.info(f"--- Finished processing for PDF: {pdf_filename} ---")

        logger.info("All PDF processing finished.")


    # ... (The logic inside run_main_logic remains the same) ...
    # Calls to analyze_pdf_structure and process_images_in_directory (which calls transcribe_image)
    # will now trigger the cleanup internally after each major toolkit operation.
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level_int, format='%(asctime)s - %(name)s [%(levelname)s] %(message)s', force=True)
    logger.setLevel(log_level_int)

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
        logger.error("Both --start-page and --end-page must be provided together if specified.")
        sys.exit(1)
    if user_start_page_arg is not None and user_end_page_arg is not None and user_start_page_arg > user_end_page_arg:
        logger.error(f"Start page ({user_start_page_arg}) cannot be > end page ({user_end_page_arg}).")
        sys.exit(1)

    single_output_file_path: Optional[Path] = None
    output_directory_path: Optional[Path] = None
    resume_state_file_path: Optional[Path] = None # Path to the state file
    final_append_mode = args.append

    if not process_pdfs:
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
            # Define state file path only in book mode
            resume_state_file_path = output_directory_path / RESUME_STATE_FILENAME
            logger.info(f"Resume state will be managed in: {resume_state_file_path}")
        except OSError as e:
            logger.error(f"Failed to create output directory {output_directory_path}: {e}")
            sys.exit(1)

    analysis_model_enum = parse_model_string_to_enum(args.analysis_model, "Analysis")
    transcription_model_enum = parse_model_string_to_enum(args.transcription_model, "Transcription")

    if analysis_model_enum is None or transcription_model_enum is None:
        logger.critical("Invalid model type provided. Cannot proceed.")
        sys.exit(1)

    # <<< Determine platforms >>>
    analysis_platform = get_platform_from_model_type(analysis_model_enum)
    transcription_platform = get_platform_from_model_type(transcription_model_enum)

    if analysis_platform is None:
        logger.critical(f"Could not determine platform for analysis model: {analysis_model_enum.value}. Cannot proceed.")
        sys.exit(1)
    if transcription_platform is None:
        logger.critical(f"Could not determine platform for transcription model: {transcription_model_enum.value}. Cannot proceed.")
        sys.exit(1)

    logger.info(f"Determined analysis platform: {analysis_platform}")
    logger.info(f"Determined transcription platform: {transcription_platform}")

    try:
        logger.info(f"Initializing PDF Analysis model: {analysis_model_enum.value} (Platform: {analysis_platform})")
        analysis_model = ModelFactory.create(
            model_platform=analysis_platform,
            model_type=analysis_model_enum
        )
        analysis_toolkit = ImageAnalysisToolkit(model=analysis_model)
        logger.info(f"Initialized PDF Analysis Toolkit: {analysis_model.model_type.value}")

        logger.info(f"Initializing Transcription model: {transcription_model_enum.value} (Platform: {transcription_platform})")
        transcription_model = ModelFactory.create(
            model_platform=transcription_platform,
            model_type=transcription_model_enum
        )
        transcription_toolkit = ImageAnalysisToolkit(model=transcription_model)
        logger.info(f"Initialized Transcription Toolkit: {transcription_model.model_type.value}")

    except Exception as e:
        logger.error(f"Failed to initialize models or toolkits: {e}", exc_info=True)
        sys.exit(1)

    overall_initial_context: Optional[str] = None
    if single_output_file_path and final_append_mode:
        # (Context loading for single file remains the same as before)
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
        try: # Add separator
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
            start_page=user_start_page_arg, # Use direct args here
            end_page=user_end_page_arg,
            image_toolkit=transcription_toolkit,
            prompt_template=PROMPT_TEMPLATE,
            initial_context=overall_initial_context,
            # No state needed for single image dir mode
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
        # Load the overall state ONCE before the loop
        current_state_data = load_resume_state(resume_state_file_path) if resume_state_file_path else {}

        for pdf_path in pdf_files:
            pdf_filename = pdf_path.name
            logger.info(f"--- Starting processing for PDF: {pdf_filename} ---")
            book_specific_output_path = output_directory_path / f"{pdf_path.stem}.txt"
            logger.info(f"Output for '{pdf_filename}' will be: {book_specific_output_path}")

            # --- Check if already completed ---
            if check_if_file_completed(book_specific_output_path, END_MARKER):
                logger.info(f"'{pdf_filename}' already marked as completed ('{END_MARKER}' found). Skipping.")
                continue

            # --- Determine Effective Page Range & Resumption ---
            effective_start_page = user_start_page_arg
            effective_end_page = user_end_page_arg
            resuming = False

            # Get last processed page from state
            last_processed_page = current_state_data.get(pdf_filename)

            if last_processed_page is not None:
                 resume_start_page = last_processed_page + 1
                 logger.info(f"Found resume state for '{pdf_filename}': Last processed page was {last_processed_page}. Attempting to resume from page {resume_start_page}.")
                 resuming = True
            else:
                 resume_start_page = 1 # Default if no state

            # Determine base start/end (LLM or user args)
            base_start_page = user_start_page_arg
            base_end_page = user_end_page_arg
            if base_start_page is None: # No user args, try LLM analysis
                logger.info(f"Attempting LLM analysis for page range of {pdf_filename}...")
                analysis_response = analyze_pdf_structure(analysis_toolkit, pdf_path) # Cleanup happens inside
                llm_start, llm_end = parse_pdf_analysis_response(analysis_response)
                if llm_start is not None and llm_end is not None:
                    base_start_page = llm_start
                    base_end_page = llm_end
                    logger.info(f"Using LLM-determined range: {base_start_page}-{base_end_page}")
                else:
                    logger.warning(f"LLM analysis failed for {pdf_filename}. Will process all pages or use resume state.")
                    # Leave base_start/end as None, rely on resume or full processing
            else:
                logger.info(f"Using user-provided range: {base_start_page}-{base_end_page}")

            # Calculate final effective start page
            if resuming:
                # If resuming, the start page is the *later* of the resume point or the base start
                effective_start_page = max(resume_start_page, base_start_page or 1)
            else:
                # If not resuming, use the base start page (or None if LLM failed and no user args)
                effective_start_page = base_start_page

            effective_end_page = base_end_page # End page is just the base end page

            if effective_start_page is not None and effective_end_page is not None and effective_start_page > effective_end_page:
                 logger.warning(f"Effective start page {effective_start_page} is after effective end page {effective_end_page} for '{pdf_filename}'. Nothing to process.")
                 continue # Skip this PDF if range is invalid after resumption logic


            # --- Load Context for Append/Resume ---
            current_book_initial_context = None
            if (final_append_mode or resuming) and book_specific_output_path.exists() and book_specific_output_path.stat().st_size > 0:
                try:
                    logger.info(f"Reading context from existing file: {book_specific_output_path}")
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


            # Add append separator if appending and file exists
            if final_append_mode and book_specific_output_path.exists():
                 try:
                     with open(book_specific_output_path, 'a', encoding='utf-8') as bf:
                         separator = f"\n\n{'='*20} Appending at {datetime.datetime.now()} {'='*20}\n\n"
                         bf.write(separator)
                 except IOError as e:
                    logger.error(f"Error preparing file {book_specific_output_path} for appending: {e}. Skipping book.")
                    continue

            # --- Process the PDF ---
            # Create a subdirectory within the book directory for temporary images
            images_base_dir = source_dir / TEMP_IMAGE_SUBDIR
            pdf_image_dir_path = images_base_dir / pdf_path.stem # Subdir named after the PDF stem
            pdf_processing_success = False # Track success for adding "The End."

            try:
                # Ensure base dir exists
                images_base_dir.mkdir(exist_ok=True)
                # Clean up any old images for this PDF first
                if pdf_image_dir_path.exists():
                    try:
                        shutil.rmtree(pdf_image_dir_path)
                        logger.debug(f"Cleaned up pre-existing image directory: {pdf_image_dir_path}")
                    except OSError as e_clean_old:
                        logger.warning(f"Could not clean up pre-existing image directory {pdf_image_dir_path}: {e_clean_old}")
                # Create the specific dir for this PDF's images
                try:
                    pdf_image_dir_path.mkdir(parents=False, exist_ok=False) # Should not exist after cleanup
                    logger.info(f"Image dir for '{pdf_filename}': {pdf_image_dir_path}")
                except FileExistsError:
                    logger.warning(f"Image directory {pdf_image_dir_path} still exists after cleanup attempt. Proceeding.")
                except OSError as e_mkdir:
                    logger.error(f"Failed to create image directory {pdf_image_dir_path}: {e_mkdir}")
                    continue # Skip this PDF if dir creation fails

                if not convert_pdf_to_images(pdf_path, pdf_image_dir_path):
                    logger.error(f"Failed PDF conversion for '{pdf_filename}'. Skipping.")
                    # No need to update state if conversion failed before processing
                else:
                    pdf_processing_success, _ = process_images_in_directory(
                        image_directory=pdf_image_dir_path,
                        output_path=book_specific_output_path,
                        append_mode=final_append_mode or resuming, # Append if either flag is true or resuming
                        start_page=effective_start_page,
                        end_page=effective_end_page,
                        image_toolkit=transcription_toolkit,
                        prompt_template=PROMPT_TEMPLATE,
                        initial_context=current_book_initial_context,
                        # Pass state info
                        pdf_filename=pdf_filename,
                        state_file_path=resume_state_file_path,
                        state_data=current_state_data # Pass the dictionary
                    )

                    if pdf_processing_success:
                        logger.info(f"Successfully processed images for PDF: {pdf_filename}")
                        # --- Add "The End." marker ---
                        try:
                            with open(book_specific_output_path, 'a', encoding='utf-8') as f_end:
                                f_end.write(f"\n\n{END_MARKER}\n")
                            logger.info(f"Added '{END_MARKER}' to completed file: {book_specific_output_path.name}")
                        except IOError as e:
                             logger.error(f"Error adding '{END_MARKER}' to {book_specific_output_path.name}: {e}")
                             pdf_processing_success = False # Mark as failed if marker write fails
                    else:
                        logger.error(f"Processing failed for images from PDF: {pdf_filename}.")
                        # State was hopefully updated during the failed run up to the last good page

            except Exception as e:
                 logger.critical(f"Critical error processing PDF {pdf_filename}: {e}", exc_info=True)
            finally: # Cleanup pdf image dir
                if pdf_image_dir_path and pdf_image_dir_path.exists() and not args.keep_images: # Only delete if keep_images is False
                    try:
                        shutil.rmtree(pdf_image_dir_path)
                        logger.info(f"Cleaned up PDF image directory: {pdf_image_dir_path}")
                    except Exception as e_clean:
                         logger.error(f"Error cleaning up PDF image dir {pdf_image_dir_path}: {e_clean}")
                elif pdf_image_dir_path and args.keep_images:
                    logger.info(f"Keeping PDF image directory due to --keep-images flag: {pdf_image_dir_path}")

            logger.info(f"--- Finished processing for PDF: {pdf_filename} ---")

        logger.info("All PDF processing finished.")

def cleanup_system_temp_files():
    """Finds and deletes temporary files created by the toolkit in /tmp."""
    logger.debug(f"Attempting periodic cleanup of '{SYSTEM_TEMP_DIR_PATTERN}'...")
    count = 0
    try:
        # Use glob.iglob for potentially better memory efficiency if many files exist
        for temp_file_path_str in glob.iglob(SYSTEM_TEMP_DIR_PATTERN):
            try:
                p = Path(temp_file_path_str)
                if p.is_file(): # Double-check it's a file before unlinking
                    p.unlink()
                    logger.debug(f"Deleted system temporary file: {p.name}")
                    count += 1
                else:
                    logger.debug(f"Skipping non-file entry found by glob: {p.name}")
            except FileNotFoundError:
                 logger.debug(f"System temporary file already gone: {temp_file_path_str}")
            except OSError as e_unlink:
                # More specific logging for permission errors etc.
                logger.warning(f"Could not delete system temporary file {temp_file_path_str}: {e_unlink}")
            except Exception as e_unlink_other:
                 logger.warning(f"Unexpected error deleting system temporary file {temp_file_path_str}: {e_unlink_other}")
        # Log only if files were actually deleted to reduce noise
        if count > 0:
            logger.debug(f"Periodic cleanup removed {count} system temporary file(s).")
    except Exception as e_glob:
        # Catch errors during the glob operation itself
        logger.error(f"Error during periodic cleanup glob operation: {e_glob}")
# --- END: Dedicated cleanup function ---

# --- Script Entry Point with Cleanup ---
if __name__ == "__main__":
    # ... (Argument parsing remains the same) ...
    DEFAULT_ANALYSIS_MODEL = ModelType.GEMINI_2_5_PRO_EXP.value # Changed to standard enum value
    DEFAULT_TRANSCRIPTION_MODEL = ModelType.GEMINI_2_0_FLASH_LITE_PREVIEW.value # Changed to standard enum value

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
    parser.add_argument("--start-page", type=int, help="Manually specify the first page number (inclusive). Overrides LLM analysis.")
    parser.add_argument("--end-page", type=int, help="Manually specify the last page number (inclusive). Overrides LLM analysis.")

    parser.add_argument(
        "--analysis-model", type=str, default=DEFAULT_ANALYSIS_MODEL,
        help=f"Model string for PDF analysis (default: '{DEFAULT_ANALYSIS_MODEL}'). Matches ModelType enum."
    )
    parser.add_argument(
        "--transcription-model", type=str, default=DEFAULT_TRANSCRIPTION_MODEL,
        help=f"Model string for image transcription (default: '{DEFAULT_TRANSCRIPTION_MODEL}'). Matches ModelType enum."
    )

    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)."
    )
    parser.add_argument(
        "--keep-images", action="store_true",
        help=f"Keep the per-PDF image directories created in '{TEMP_IMAGE_SUBDIR}' within the book directory (only applies to --book-dir mode)."
    )

    try:
        available_models_str = ", ".join(f"'{m.value}'" for m in ModelType)
        model_help_epilog = f"\n\nAvailable ModelType values:\n{available_models_str}"
    except NameError:
        model_help_epilog = "\n\n(Could not list available ModelType values)"

    parser.epilog = f"""
Example Usage:
  # Images -> Single file (overwrite)
  python %(prog)s --input-dir path/to/pages -o output.txt
  # Images -> Console
  python %(prog)s --input-dir path/to/pages
  # Images -> Single file (append)
  python %(prog)s --input-dir path/to/pages -o output.txt --append --transcription-model {ModelType.GPT_4O_MINI.value}

  # PDFs -> Specific output dir (auto-detect range, overwrite)
  python %(prog)s --book-dir path/to/pdfs -o path/to/output_dir
  # PDFs -> Default output dir 'books_tts' (auto-detect range, overwrite)
  python %(prog)s --book-dir path/to/pdfs
  # PDFs -> Specific dir (auto-detect range, append/resume, specific models)
  python %(prog)s --book-dir path/to/pdfs -o path/to/output_dir --append \\
    --analysis-model {ModelType.CLAUDE_3_5_SONNET.value} \\
    --transcription-model {ModelType.GPT_4O.value}
  # PDFs -> Specific dir (force pages 50-150, append/resume)
  python %(prog)s --book-dir path/to/pdfs --start-page 50 --end-page 150 -o path/to/output_dir --append
  # PDFs -> Default dir (auto-detect, debug, append/resume, keep images)
  python %(prog)s --book-dir path/to/pdfs --log-level DEBUG --append --keep-images

Notes:
- Requires 'pdftoppm' (poppler-utils) and 'montage' (ImageMagick).
- Images must be named like '-[digits].ext'. Needs 3-page context.
- Configure API keys in a .env file.
- Use model strings matching CAMEL's ModelType enum values.
- PDF page range auto-detection runs only if --start-page/--end-page are NOT provided.
- --append mode (for --book-dir):
    - Checks for '{END_MARKER}' in existing .txt files to skip completed PDFs.
    - Uses '{RESUME_STATE_FILENAME}' in output dir to resume incomplete PDFs from last successful page.
    - State is updated after *each* successful page transcription.
- Interrupted processes may leave temporary directories ({TEMP_IMAGE_SUBDIR}). Use --keep-images to prevent cleanup.
{model_help_epilog}
"""
    parsed_args = parser.parse_args()

    try:
        run_main_logic(parsed_args)
    finally:
        # Keep the final cleanup as a safety net, although most should be cleaned periodically
        logger.info("Performing final cleanup check of temporary files in /tmp...")
        cleanup_system_temp_files() # Call the same cleanup function one last time
        logger.info("Script finished.")
