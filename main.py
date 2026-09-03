# main.py -- Transcription Orchestrator
#
# Pipeline:
#   1. Discover all PDFs in INPUT_PDF_DIR
#   2. Rasterise each PDF into page images
#   3. Issue anonymous answer_ids via IdentityLedger
#   4. Transcribe each page with Qwen2.5-VL (Pass 1)
#   5. Clean answer text using same model in text-only mode (Pass 2)
#   6. Write outputs/transcriptions.json  (safe to process further)
#   7. Write outputs/MASTER_LEDGER.json   (NEVER leave this machine)
#
# Run:
#   ./launch.sh 1,4,7
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES  -- set by launch.sh
#   USE_INFERENCE_SERVER  -- set to 1 to use keep_alive.py server instead of loading model

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import config
from anonymizer import IdentityLedger
from segmenter import extract_roll_number, pdf_to_images, save_segment

# Use inference server if available, otherwise load model directly
_USE_SERVER = os.environ.get("USE_INFERENCE_SERVER", "0") == "1"

if _USE_SERVER:
    from transcriber import InferenceServerTranscriber as _TranscriberClass
else:
    from transcriber import QwenTranscriber as _TranscriberClass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.makedirs(config.OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.OUTPUT_DIR, "dgx_engine.log")),
    ],
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_completed_ids(path: str) -> set:
    """Return set of answer_ids already in transcriptions.json."""
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        data = json.load(f)
    return {entry["answer_id"] for entry in data}


def save_transcriptions(entries: list, path: str):
    """Atomically write transcriptions to disk."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    logger.info("Checkpoint written -> %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(config.OUTPUT_DIR,    exist_ok=True)
    os.makedirs(config.SEGMENTS_DIR,  exist_ok=True)
    os.makedirs(config.INPUT_PDF_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GradeFlow -- Transcription Engine (page-by-page mode)")
    logger.info("  Mode       : %s", "inference server" if _USE_SERVER else "local model")
    logger.info("  Input PDFs : %s", config.INPUT_PDF_DIR)
    logger.info("  Output     : %s", config.TRANSCRIPTIONS_PATH)
    logger.info("  Ledger     : %s", config.LEDGER_PATH)
    logger.info("=" * 60)

    pdf_files = sorted(Path(config.INPUT_PDF_DIR).glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDFs found in %s. Exiting.", config.INPUT_PDF_DIR)
        sys.exit(1)
    logger.info("Found %d PDF(s).", len(pdf_files))

    ledger        = IdentityLedger(config.LEDGER_PATH)
    completed_ids = load_completed_ids(config.TRANSCRIPTIONS_PATH)
    logger.info("Checkpoint: %d page(s) already transcribed.", len(completed_ids))

    # Load model / connect to server
    if _USE_SERVER:
        transcriber = _TranscriberClass(
            primary_prompt      = config.TRANSCRIPTION_PROMPT,
            continuation_prompt = config.CONTINUATION_PROMPT,
            continuation_marker = config.SPILLOVER_CONTINUATION_MARKER,
            max_spillover_pages = config.SPILLOVER_MAX_EXTRA_PAGES,
        )
    else:
        transcriber = _TranscriberClass(
            model_path          = config.MODEL_PATH,
            primary_prompt      = config.TRANSCRIPTION_PROMPT,
            continuation_prompt = config.CONTINUATION_PROMPT,
            continuation_marker = config.SPILLOVER_CONTINUATION_MARKER,
            max_new_tokens      = config.MAX_NEW_TOKENS,
            max_spillover_pages = config.SPILLOVER_MAX_EXTRA_PAGES,
        )

    # Load existing entries from checkpoint
    entries: List[dict] = []
    if os.path.exists(config.TRANSCRIPTIONS_PATH):
        with open(config.TRANSCRIPTIONS_PATH) as f:
            entries = json.load(f)

    for pdf_path in pdf_files:
        pdf_name    = pdf_path.name
        roll_number = extract_roll_number(pdf_name)

        logger.info("")
        logger.info("---- %s (roll=%s) ----", pdf_name, roll_number)

        try:
            pages = pdf_to_images(str(pdf_path), dpi=config.PDF_DPI)
        except Exception as e:
            logger.error("Could not rasterise %s: %s", pdf_name, e)
            continue

        logger.info("  %d page(s) found.", len(pages))

        for page_index, page_img in enumerate(pages):
            page_number = page_index + 1

            answer_id = ledger.issue_id(
                roll_number = roll_number,
                source_pdf  = pdf_name,
                page_number = page_number,
                question_id = "page_{}".format(page_number),
            )

            if answer_id in completed_ids:
                logger.info("  Skipping page %d (already done)", page_number)
                continue

            # Save full page image to disk
            page_path = save_segment(
                page_img,
                config.SEGMENTS_DIR,
                roll_number,
                "full_page",
                page_number,
            )

            logger.info("  Transcribing page %d -> %s", page_number, answer_id)

            # Pass 1: VLM reads the image
            extracted_text = transcriber.transcribe_answer(
                overlap_image_path=page_path,
            )

            # Pass 2: text-only cleaning -- remove question/headers using known question text
            q_key   = config.PAGE_TO_QUESTION.get(page_number, "")
            q_text  = config.MARKING_SCHEMES.get(q_key, {}).get("question_text", "")
            extracted_text = transcriber.clean_answer(extracted_text, question_text=q_text)

            preview = extracted_text[:80] + ("..." if len(extracted_text) > 80 else "")
            logger.info("    -> %s", preview)

            entries.append({
                "answer_id":      answer_id,
                "roll_number":    roll_number,
                "source_pdf":     pdf_name,
                "page_number":    page_number,
                "extracted_text": extracted_text,
            })
            completed_ids.add(answer_id)

            # Checkpoint after every page
            save_transcriptions(entries, config.TRANSCRIPTIONS_PATH)

    save_transcriptions(entries, config.TRANSCRIPTIONS_PATH)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Transcription complete.")
    logger.info(ledger.summary())
    logger.info("Output : %s", config.TRANSCRIPTIONS_PATH)
    logger.info("Ledger : %s  <-- NEVER share this file", config.LEDGER_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
