# config.py -- Central configuration for the GradeFlow pipeline.
# Edit this file before running the pipeline.

import os

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR            = os.path.expanduser("~/shubh/data/Project")
MODEL_PATH          = os.path.join(BASE_DIR, "models/Qwen2.5-VL-7B")
INPUT_PDF_DIR       = os.path.join(BASE_DIR, "input_pdfs")
OUTPUT_DIR          = os.path.join(BASE_DIR, "outputs")
SEGMENTS_DIR        = os.path.join(OUTPUT_DIR, "segments")
LEDGER_PATH         = os.path.join(OUTPUT_DIR, "MASTER_LEDGER.json")   # NEVER share
TRANSCRIPTIONS_PATH = os.path.join(OUTPUT_DIR, "transcriptions.json")

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 1024

# ---------------------------------------------------------------------------
# PDF -> IMAGE
# ---------------------------------------------------------------------------
PDF_DPI = 100   # 100 DPI balances quality vs memory on V100s

# ---------------------------------------------------------------------------
# SPILLOVER  (for multi-page answers)
# ---------------------------------------------------------------------------
SPILLOVER_CONTINUATION_MARKER = "[CONTINUES]"
SPILLOVER_MAX_EXTRA_PAGES     = 2

# ---------------------------------------------------------------------------
# PAGE -> QUESTION MAPPING
# ---------------------------------------------------------------------------
# Maps page_number (int) -> question_id (str) for this exam format.
# One page per question. Edit if your exam layout differs.
PAGE_TO_QUESTION = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}

# ---------------------------------------------------------------------------
# MARKING SCHEMES  (edit per exam before each run)
# ---------------------------------------------------------------------------
MARKING_SCHEMES = {
    "Q1": {
        "question_text": (
            "Four validator nodes (Aditya BH1, Himanshu BH2, Ram BH3, Juice Shop) maintain a shared "
            "ledger recording Juice Coin balances. Aditya broadcasts a transaction to deduct 100 coins. "
            "Ram may maliciously broadcast a conflicting transaction to double-spend. Describe the "
            "consensus algorithm used and how conflicting decisions are prevented."
        ),
        "max_marks": 5,
        "marking_scheme": (
            "The question asks the student to: (1) identify a suitable consensus algorithm for a "
            "network with a potentially malicious node, and (2) explain how double-spending or "
            "conflicting transactions are prevented. Award marks holistically as follows:\n\n"
            "5 marks: Student correctly identifies a Byzantine Fault Tolerant algorithm (PBFT or "
            "equivalent). Clearly explains that nodes broadcast and cross-check messages with each "
            "other (not just trust the leader). Explains that a majority/quorum of matching messages "
            "is required before a transaction is finalised. Explains that a malicious node cannot "
            "succeed because it cannot forge the majority -- other nodes will detect the conflict.\n\n"
            "3-4 marks: Correctly identifies BFT/PBFT. Explains the broadcast and majority mechanism "
            "reasonably well but misses some detail, OR explains the mechanism well without naming "
            "the algorithm explicitly.\n\n"
            "1-2 marks: Shows some understanding -- mentions majority voting, or mentions that nodes "
            "communicate with each other, but explanation is incomplete or partially incorrect.\n\n"
            "0 marks: No relevant content, or describes a completely wrong algorithm.\n\n"
            "IMPORTANT: Do NOT penalise for not naming specific phases (Pre-Prepare, Prepare, Commit). "
            "Award full marks if the student demonstrates correct understanding regardless of terminology."
        ),
    },
    "Q2": {
        "question_text": (
            "Paxos system: Aditya (BH1) is proposer, Himanshu (BH2), Ram (BH3), Kavya (GH) are "
            "acceptors. Messages may be delayed but not lost. Multiple proposers may attempt "
            "simultaneously. If the prepare phase is removed and direct accept requests are allowed, "
            "what could happen?"
        ),
        "max_marks": 5,
        "marking_scheme": (
            "The question asks what COULD HAPPEN if the prepare phase is removed. Award marks "
            "holistically for understanding the failure mode:\n\n"
            "5 marks: Student clearly explains that without the prepare phase, there is no mechanism "
            "to prevent multiple proposers from getting different acceptors to accept different values "
            "simultaneously. Demonstrates this with a concrete scenario. Correctly concludes that "
            "consensus may not be reached -- either nodes end up agreeing on different values "
            "(safety violation), or proposals keep overwriting each other and nothing gets finalised "
            "(liveness violation). Both safety and liveness covered.\n\n"
            "3-4 marks: Correctly identifies the core problem. Gives a reasonable explanation or "
            "example but covers only one of safety/liveness, or the example is slightly imprecise.\n\n"
            "1-2 marks: Understands something goes wrong but explanation is vague or partially "
            "incorrect.\n\n"
            "0 marks: No relevant content.\n\n"
            "IMPORTANT: Accept any reasonable failure scenario. Do not require exact Paxos "
            "terminology. A student who explains the problem correctly in plain language should "
            "receive full credit."
        ),
    },
    "Q3": {
        "question_text": (
            "Aditya (BH1) and Ram (BH3) run Bitcoin version A. Himanshu (BH2) runs version B which "
            "interprets transaction rules slightly differently. All nodes are honest. Aditya mines a "
            "block version A considers valid. Himanshu mines a different block at the same height "
            "that version B considers valid. Both blocks propagate, nodes temporarily follow different "
            "chains. Describe how block validation differences matter in forking and how it will be "
            "overcome."
        ),
        "max_marks": 10,
        "marking_scheme": (
            "The question has two parts: (1) how validation differences cause the fork, and "
            "(2) how it resolves. Award up to 5 marks for each part.\n\n"
            "Part 1 -- How validation differences cause the fork (5 marks):\n"
            "5 marks: Explains that because the two versions have different rules, each version "
            "rejects the other's block as invalid. Both miners produce a valid block at the same "
            "height simultaneously. This causes the network to split -- nodes running version A "
            "follow one chain, nodes running version B follow another.\n"
            "3-4 marks: Explains the split reasonably but misses a detail.\n"
            "1-2 marks: Mentions that different software causes different chains but explanation "
            "is superficial.\n\n"
            "Part 2 -- How the fork resolves (5 marks):\n"
            "5 marks: Explains that Bitcoin uses the longest chain (most accumulated work) as the "
            "canonical chain. Whichever fork grows longer first wins. The minority fork is abandoned "
            "and those miners switch to the winning chain. The fork is temporary.\n"
            "3-4 marks: Mentions longest chain rule and that one fork wins, but misses details.\n"
            "1-2 marks: Vaguely mentions computational power or majority decides.\n\n"
            "IMPORTANT: Do not require specific terms like 'orphaned block' or 'hash rate'. "
            "Award marks for correct conceptual understanding."
        ),
    },
    "Q4": {
        "question_text": (
            "Route: Aditya -> Himanshu -> Juice Shop. Juice Shop generates invoice, sends to Aditya. "
            "(a) Aditya initiates Lightning payment. Juice Shop goes offline before claiming. "
            "What happens to the conditional payment locked in each channel? [5] "
            "(b) Himanshu did a transaction with Juice Shop recorded in a commitment transaction. "
            "Juice Shop refuses cooperative channel closure. Himanshu force-closes. "
            "How will Himanshu close the channel and what transaction is broadcast? [5]"
        ),
        "max_marks": 10,
        "marking_scheme": (
            "Part (a) -- Juice Shop goes offline (5 marks):\n"
            "5 marks: Student explains that the payment uses time-locked contracts (HTLCs). "
            "Because Juice Shop is offline, the secret/key is never revealed. Without the secret, "
            "Himanshu cannot claim from the H-JuiceShop channel. After the timeout expires, the "
            "locked funds return to Himanshu. Similarly the A-H channel times out and funds return "
            "to Aditya. No one loses money.\n"
            "3-4 marks: Correctly explains that timeouts protect the funds and money is returned, "
            "but misses one step in the chain.\n"
            "1-2 marks: Understands that the payment fails and money is somehow returned, but "
            "explanation is vague or missing the mechanism.\n\n"
            "Part (b) -- Himanshu force-closes (5 marks):\n"
            "5 marks: Student explains that Himanshu broadcasts the latest/most recent commitment "
            "transaction to the main blockchain unilaterally. There is a waiting period (timelock) "
            "before he can claim funds. Since this is the latest transaction, it reflects the "
            "correct final state and Juice Shop cannot dispute it. After the waiting period, "
            "Himanshu receives his funds.\n"
            "3-4 marks: Correctly identifies that Himanshu broadcasts the commitment transaction "
            "and there is a timelock, but misses the detail about why Juice Shop cannot contest it.\n"
            "1-2 marks: Mentions broadcasting to blockchain but is vague.\n\n"
            "CRITICAL: Award 0 for part (b) only if the student describes Himanshu broadcasting "
            "an OLD revoked transaction to cheat. If the student describes legitimate force closure "
            "even imprecisely, award partial marks.\n"
            "Do NOT penalise for not using exact terms like HTLC, CSV, or commitment transaction."
        ),
    },
}

# ---------------------------------------------------------------------------
# TRANSCRIPTION PROMPT  (Pass 1 -- VLM reads the image)
# ---------------------------------------------------------------------------
TRANSCRIPTION_PROMPT = (
    "You are reading a student exam answer sheet. "
    "Your ONLY job is to output the student's handwritten answer text.\n\n"
    "STRICT RULES:\n"
    "1. OUTPUT ONLY handwritten text -- text written by hand in pen or pencil by the student.\n"
    "2. NEVER output printed or typed text. "
    "Printed text includes: the question text, exam instructions, course name, "
    "college name, date, max marks, table headers, and any pre-printed content.\n"
    "3. The student's handwritten NAME and ROLL NUMBER at the top are also "
    "header info -- do NOT include them.\n"
    "4. DO include: the student's handwritten answer paragraphs, handwritten headings "
    "they wrote like 'Case 1:' or 'i)', diagram labels they drew, "
    "and any handwritten annotations or working.\n"
    "5. If you cannot find any handwritten answer content, output exactly: [BLANK]\n\n"
    "Output the handwritten answer text only. "
    "Do not explain, do not add labels, do not repeat the question."
)

# ---------------------------------------------------------------------------
# CONTINUATION PROMPT  (used if a page ends with [CONTINUES])
# ---------------------------------------------------------------------------
CONTINUATION_PROMPT = (
    "This image is the next page of a student exam answer sheet.\n"
    "The student's answer was cut off on the previous page. "
    "Here is what was captured so far:\n"
    "--- PARTIAL ANSWER SO FAR ---\n"
    "{previous_text}\n"
    "--- END OF PARTIAL ANSWER ---\n\n"
    "Instructions:\n"
    "- Transcribe the handwritten text on this page that continues from the partial answer above.\n"
    "- Do NOT repeat text already captured in the partial answer.\n"
    "- If the answer still continues beyond this image, end with exactly: [CONTINUES]\n"
    "- If no continuation is found, respond with exactly: [BLANK]"
)
