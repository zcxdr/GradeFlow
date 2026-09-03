# transcriber.py -- Qwen2.5-VL inference engine
#
# V100 constraints:
#   - torch.float32      (bfloat16 not supported; float16 causes repetition loops)
#   - attn_implementation="eager"  (no FlashAttention on V100)
#   - device_map="auto"  (HuggingFace spreads model across all visible GPUs)
#
# Two classes:
#   QwenTranscriber          -- loads model locally, used in normal pipeline
#   InferenceServerTranscriber -- talks to keep_alive.py HTTP server

import json as _json
import logging
import time
import urllib.request
from typing import Callable, List, Optional

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


class QwenTranscriber:
    """
    Loads Qwen2.5-VL once and provides transcription + answer-cleaning.
    Instantiate once at startup, call transcribe_answer() per page.
    """

    def __init__(
        self,
        model_path:          str,
        primary_prompt:      str,
        continuation_prompt: str,
        continuation_marker: str = "[CONTINUES]",
        max_new_tokens:      int = 1024,
        max_spillover_pages: int = 2,
    ):
        self.primary_prompt      = primary_prompt
        self.continuation_prompt = continuation_prompt
        self.continuation_marker = continuation_marker
        self.max_new_tokens      = max_new_tokens
        self.max_spillover_pages = max_spillover_pages

        logger.info("Loading Qwen2.5-VL from %s ...", model_path)
        logger.info("  dtype=float32 | attn=eager | device_map=auto")
        t0 = time.time()

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto",
            attn_implementation="eager",
        )
        self.model.eval()

        # max_pixels controls vision token budget per image.
        # 1280*28*28 is the Qwen default -- full resolution.
        # Lower this if you hit OOM during inference (not model loading).
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        logger.info("Model ready in %.1fs", time.time() - t0)

    # -------------------------------------------------------------------------
    # Low-level inference
    # -------------------------------------------------------------------------

    def _infer(self, image_path: str, prompt_text: str) -> str:
        """Single-image inference. Returns decoded string."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text",  "text": prompt_text},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )

        trimmed = generated_ids[0][len(inputs.input_ids[0]):]
        return self.processor.decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    # -------------------------------------------------------------------------
    # Spillover resolution
    # -------------------------------------------------------------------------

    def _resolve_spillover(
        self,
        partial_text:             str,
        continuation_crop_getter: Callable[[], Optional[str]],
    ) -> str:
        """
        If the model emitted [CONTINUES], fetch continuation crops and
        append until the answer is complete or max_spillover_pages reached.
        """
        collected = partial_text.rstrip()
        if collected.endswith(self.continuation_marker):
            collected = collected[:-len(self.continuation_marker)].rstrip()

        for attempt in range(self.max_spillover_pages):
            next_path = continuation_crop_getter()
            if next_path is None:
                break

            logger.info("  Spillover detected -- fetching continuation (attempt %d)", attempt + 1)
            cont_prompt = self.continuation_prompt.format(previous_text=collected)

            try:
                cont_text = self._infer(next_path, cont_prompt)
            except Exception as e:
                logger.error("  Continuation inference failed: %s", e)
                break

            if cont_text in ("[BLANK]", "", "[ERROR]"):
                break

            if cont_text.endswith(self.continuation_marker):
                collected = collected + " " + cont_text[:-len(self.continuation_marker)].rstrip()
            else:
                collected = collected + " " + cont_text
                break

        return collected.strip()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def transcribe_answer(
        self,
        overlap_image_path:       str,
        continuation_crop_getter: Optional[Callable[[], Optional[str]]] = None,
        retries:                  int   = 3,
        retry_delay:              float = 2.0,
    ) -> str:
        """
        Transcribe a single page image.
        Resolves spillover automatically if continuation_crop_getter is provided.
        """
        result = None
        for attempt in range(1, retries + 1):
            try:
                result = self._infer(overlap_image_path, self.primary_prompt)
                break
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM on attempt %d. Clearing cache ...", attempt)
                torch.cuda.empty_cache()
                time.sleep(retry_delay)
            except Exception as e:
                logger.error("Inference error attempt %d: %s", attempt, e)
                time.sleep(retry_delay)

        if result is None:
            logger.error("All %d attempts failed for %s", retries, overlap_image_path)
            return "[ERROR]"

        if self.continuation_marker in result and continuation_crop_getter:
            result = self._resolve_spillover(result, continuation_crop_getter)

        return result

    def clean_answer(self, raw_text: str, question_text: str = "") -> str:
        """
        Text-only Pass 2: remove question text / headers from raw transcription.
        Reuses the already-loaded model -- no extra GPU memory needed.
        Providing question_text helps the model know exactly what to remove.
        """
        if raw_text in ("[BLANK]", "[ERROR]", ""):
            return raw_text

        question_context = (
            "The printed question on this page is:\n{}\n\n".format(question_text)
            if question_text
            else "The page contains a printed exam question followed by the student answer.\n\n"
        )

        system = (
            "You are extracting a student answer from a raw exam page transcription.\n\n"
            + question_context
            + "The raw transcription contains everything on the page mixed together: "
            "college header, student name, roll number, the printed question text above, "
            "and the student handwritten answer.\n\n"
            "Your job: return ONLY the student handwritten answer -- "
            "the part written by the student in response to the question above.\n"
            "Keep it exactly as the student wrote it, including spelling mistakes.\n"
            "Do NOT include the question text, headers, name, roll number, or instructions.\n"
            "If no answer is present, return exactly: [BLANK]\n"
            "Return only the answer text. No labels, no explanation."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": "RAW TRANSCRIPTION:\n{}".format(raw_text)},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        trimmed = out[0][len(inputs.input_ids[0]):]
        result  = self.processor.decode(trimmed, skip_special_tokens=True).strip()
        return result if result else raw_text


# ---------------------------------------------------------------------------
# Inference server client (for keep_alive.py workflow)
# ---------------------------------------------------------------------------

class InferenceServerTranscriber:
    """
    Drop-in replacement for QwenTranscriber that calls the keep_alive.py server.
    Use when the model is already loaded in keep_alive.py to avoid reloading.
    Set USE_INFERENCE_SERVER=1 environment variable to activate.
    """

    def __init__(
        self,
        host:                str   = "localhost",
        port:                int   = 8765,
        primary_prompt:      str   = "",
        continuation_prompt: str   = "",
        continuation_marker: str   = "[CONTINUES]",
        max_spillover_pages: int   = 2,
        **kwargs,
    ):
        self.url                  = "http://{}:{}/infer".format(host, port)
        self.primary_prompt       = primary_prompt
        self.continuation_prompt  = continuation_prompt
        self.continuation_marker  = continuation_marker
        self.max_spillover_pages  = max_spillover_pages

    def _call(self, image_path: str, prompt: str) -> str:
        body = _json.dumps({"image_path": image_path, "prompt": prompt}).encode()
        req  = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return _json.loads(r.read())["text"]

    def transcribe_answer(
        self,
        overlap_image_path:       str,
        continuation_crop_getter: Optional[Callable[[], Optional[str]]] = None,
        **kwargs,
    ) -> str:
        result = self._call(overlap_image_path, self.primary_prompt)

        if self.continuation_marker in result and continuation_crop_getter:
            collected = result.replace(self.continuation_marker, "").strip()
            for _ in range(self.max_spillover_pages):
                next_path = continuation_crop_getter()
                if not next_path:
                    break
                cont_prompt = self.continuation_prompt.format(previous_text=collected)
                cont = self._call(next_path, cont_prompt)
                if cont in ("[BLANK]", "", "[ERROR]"):
                    break
                if cont.endswith(self.continuation_marker):
                    collected += " " + cont.replace(self.continuation_marker, "").strip()
                else:
                    collected += " " + cont
                    break
            return collected.strip()

        return result

    def clean_answer(self, raw_text: str, question_text: str = "") -> str:
        """Cleaning not supported over inference server -- return raw text."""
        return raw_text
