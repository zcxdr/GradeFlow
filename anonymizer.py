# anonymizer.py -- Identity firewall between student identity and anonymous answer_ids.
#
# The MASTER_LEDGER is the ONLY file that maps answer_id <-> (roll_number, question_id).
# It must NEVER be sent to any external API or cloud service.
#
# Ledger schema (MASTER_LEDGER.json):
# {
#   "created_at": "...",
#   "exam_session": "...",
#   "entries": {
#     "ans_9f8d7a": {
#       "roll_number": "2021CS042",
#       "source_pdf":  "2021CS042_paper.pdf",
#       "page_number": 1,
#       "question_id": "Q1",
#       "issued_at":   "2025-01-01T10:00:00"
#     }
#   }
# }

import json
import logging
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class IdentityLedger:
    """
    Issues cryptographically random answer_ids and persists the mapping to disk.
    Atomic writes ensure no data loss on crash.
    """

    def __init__(self, ledger_path: str, exam_session: str = "default"):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        if self.ledger_path.exists():
            logger.info("Loading existing ledger from %s", self.ledger_path)
            with open(self.ledger_path) as f:
                self._data = json.load(f)
        else:
            logger.info("Creating new ledger at %s", self.ledger_path)
            self._data = {
                "created_at":   datetime.now().isoformat(),
                "exam_session": exam_session,
                "entries":      {},
            }
            self._save()

    def issue_id(
        self,
        roll_number: str,
        source_pdf: str,
        page_number: int,
        question_id: str,
    ) -> str:
        """
        Generate a unique answer_id and record it in the ledger.
        Returns the answer_id string (e.g. 'ans_9f8d7a').
        """
        answer_id = "ans_" + secrets.token_hex(3)
        while answer_id in self._data["entries"]:
            answer_id = "ans_" + secrets.token_hex(3)

        self._data["entries"][answer_id] = {
            "roll_number": roll_number,
            "source_pdf":  source_pdf,
            "page_number": page_number,
            "question_id": question_id,
            "issued_at":   datetime.now().isoformat(),
        }
        self._save()
        logger.debug("Issued %s -> roll=%s q=%s", answer_id, roll_number, question_id)
        return answer_id

    def lookup(self, answer_id: str) -> Optional[dict]:
        """Reverse lookup: answer_id -> identity dict. Used at reconciliation."""
        return self._data["entries"].get(answer_id)

    def get_all_entries(self) -> dict:
        return self._data["entries"]

    def summary(self) -> str:
        n = len(self._data["entries"])
        return "Ledger: {} entries | session='{}' | path={}".format(
            n, self._data["exam_session"], self.ledger_path
        )

    def _save(self):
        """Atomically write ledger to disk."""
        tmp = self.ledger_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        tmp.replace(self.ledger_path)
