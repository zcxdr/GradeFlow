# build_grading_payload.py -- Build grading_payload.json from transcriptions.json
#
# Reads transcriptions.json (one entry per page, with roll_number)
# Groups by question using PAGE_TO_QUESTION mapping
# Strips roll_number (anonymisation -- only answer_id remains)
# Attaches marking scheme from config
# Writes grading_payload.json (safe to send to grading API)
#
# Run: python build_grading_payload.py

import json
import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("build_payload")

OUTPUT = os.path.join(config.OUTPUT_DIR, "grading_payload.json")

# Strip model preamble patterns that sometimes leak into transcriptions
PREAMBLE_RE = re.compile(
    r"^(Sure[,.].*?[-]{3,}\s*|Here is.*?:\s*|The following.*?:\s*)",
    re.DOTALL | re.IGNORECASE,
)
SUFFIX_RE = re.compile(r"\s*[-]{3,}\s*$", re.DOTALL)


def clean(text: str) -> str:
    text = PREAMBLE_RE.sub("", text)
    text = SUFFIX_RE.sub("", text)
    return text.strip()


def main():
    if not os.path.exists(config.TRANSCRIPTIONS_PATH):
        logger.error("transcriptions.json not found at %s", config.TRANSCRIPTIONS_PATH)
        sys.exit(1)

    with open(config.TRANSCRIPTIONS_PATH) as f:
        entries = json.load(f)
    logger.info("Loaded %d transcription(s)", len(entries))

    questions = {}
    skipped   = []

    for entry in entries:
        page = entry["page_number"]
        qid  = config.PAGE_TO_QUESTION.get(page)
        text = clean(entry["extracted_text"])

        if qid is None:
            skipped.append(entry["answer_id"])
            continue
        if not text or text in ("[BLANK]", "[ERROR]"):
            skipped.append(entry["answer_id"])
            continue

        scheme = config.MARKING_SCHEMES.get(qid, {})
        if qid not in questions:
            questions[qid] = {
                "question_id":    qid,
                "question_text":  scheme.get("question_text", ""),
                "marking_scheme": scheme.get("marking_scheme", ""),
                "max_marks":      scheme.get("max_marks", 0),
                "submissions":    [],
            }

        # answer_id only -- roll_number deliberately excluded
        questions[qid]["submissions"].append({
            "answer_id":      entry["answer_id"],
            "extracted_text": text,
        })

    payload = [questions[qid] for qid in sorted(questions.keys())]

    with open(OUTPUT, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Written -> %s", OUTPUT)
    logger.info("Questions: %d", len(payload))
    for block in payload:
        logger.info(
            "  %s (%d marks): %d submission(s)",
            block["question_id"], block["max_marks"], len(block["submissions"]),
        )
    if skipped:
        logger.info("Skipped %d entry(s): %s", len(skipped), skipped)


if __name__ == "__main__":
    main()
