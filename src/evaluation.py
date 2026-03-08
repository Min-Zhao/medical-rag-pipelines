"""
RAG Evaluation Module
=====================
Computes faithfulness, relevancy, context precision/recall,
and BERTScore for RAG pipeline outputs.

Supports both automated (LLM-as-judge) and reference-based metrics.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RAGSample:
    """One evaluated Q&A sample."""
    question: str
    answer: str
    contexts: list[str]
    reference_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Evaluation scores for a single sample."""
    question: str
    answer: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0
    bert_f1: float = 0.0
    bert_precision: float = 0.0
    bert_recall: float = 0.0
    rouge_l: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_recall": self.context_recall,
            "context_precision": self.context_precision,
            "bert_f1": self.bert_f1,
            "bert_precision": self.bert_precision,
            "bert_recall": self.bert_recall,
            "rouge_l": self.rouge_l,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# LLM-as-judge helpers
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """You are evaluating whether an AI-generated answer is faithful to the provided context.

Context:
{context}

Answer:
{answer}

Rate faithfulness on a scale from 0 to 1, where:
1.0 = Every claim in the answer is directly supported by the context
0.5 = Most claims are supported but some go beyond the context
0.0 = The answer contradicts or ignores the context

Respond ONLY with a JSON object: {{"score": <float 0-1>, "reason": "<brief reason>"}}"""

RELEVANCY_PROMPT = """You are evaluating whether an AI answer is relevant to the question.

Question: {question}
Answer: {answer}

Rate answer relevancy from 0 to 1:
1.0 = Directly and completely answers the question
0.5 = Partially answers but misses key aspects
0.0 = Does not address the question

Respond ONLY with: {{"score": <float 0-1>, "reason": "<brief reason>"}}"""

CONTEXT_RECALL_PROMPT = """Given the reference answer and retrieved context, evaluate how much of the reference information is covered.

Reference Answer: {reference}
Retrieved Context: {context}

Score from 0-1:
1.0 = All key information in the reference is present in the context
0.5 = About half the reference information is in the context
0.0 = The context contains none of the reference information

Respond ONLY with: {{"score": <float 0-1>, "reason": "<brief reason>"}}"""


def _parse_score_from_json(response_text: str) -> float:
    """Parse score from LLM JSON response, with fallback regex."""
    try:
        data = json.loads(response_text.strip())
        return float(data.get("score", 0.0))
    except (json.JSONDecodeError, ValueError):
        match = re.search(r'"score"\s*:\s*([0-9.]+)', response_text)
        if match:
            return float(match.group(1))
        match = re.search(r'\b(0\.\d+|1\.0|1)\b', response_text)
        if match:
            return float(match.group(1))
        return 0.0


class LLMJudge:
    """Uses an LLM to score faithfulness, relevancy, and context recall."""

    def __init__(self, llm_client, model: str = "gpt-4o-mini"):
        self.client = llm_client
        self.model = model

    def _score(self, prompt: str) -> float:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
            )
            return _parse_score_from_json(response.choices[0].message.content)
        except Exception as e:
            logger.warning("LLM judge call failed: %s", e)
            return 0.0

    def faithfulness(self, answer: str, contexts: list[str]) -> float:
        context_str = "\n---\n".join(contexts[:3])
        return self._score(FAITHFULNESS_PROMPT.format(context=context_str, answer=answer))

    def relevancy(self, question: str, answer: str) -> float:
        return self._score(RELEVANCY_PROMPT.format(question=question, answer=answer))

    def context_recall(self, reference: str, contexts: list[str]) -> float:
        if not reference:
            return 0.0
        context_str = "\n---\n".join(contexts[:3])
        return self._score(CONTEXT_RECALL_PROMPT.format(reference=reference, context=context_str))


# ---------------------------------------------------------------------------
# Reference-based metrics
# ---------------------------------------------------------------------------

class BERTScoreEvaluator:
    """Computes BERTScore F1 between generated and reference answers."""

    def __init__(self, model_type: str = "distilbert-base-uncased"):
        try:
            from bert_score import score as bert_score_fn
            self._score_fn = bert_score_fn
            self.model_type = model_type
        except ImportError:
            logger.warning("bert-score not installed. BERTScore will return 0.")
            self._score_fn = None

    def compute(self, predictions: list[str], references: list[str]) -> dict[str, list[float]]:
        if self._score_fn is None or not references or not any(references):
            return {"precision": [0.0] * len(predictions),
                    "recall": [0.0] * len(predictions),
                    "f1": [0.0] * len(predictions)}
        try:
            P, R, F = self._score_fn(
                predictions, references,
                model_type=self.model_type,
                verbose=False,
            )
            return {
                "precision": P.tolist(),
                "recall": R.tolist(),
                "f1": F.tolist(),
            }
        except Exception as e:
            logger.warning("BERTScore failed: %s", e)
            return {"precision": [0.0] * len(predictions),
                    "recall": [0.0] * len(predictions),
                    "f1": [0.0] * len(predictions)}


class ROUGEEvaluator:
    """Computes ROUGE-L between generated and reference answers."""

    def __init__(self):
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        except ImportError:
            logger.warning("rouge-score not installed. ROUGE will return 0.")
            self.scorer = None

    def compute(self, prediction: str, reference: str) -> float:
        if self.scorer is None or not reference:
            return 0.0
        try:
            scores = self.scorer.score(reference, prediction)
            return scores["rougeL"].fmeasure
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Context precision (simple keyword overlap fallback)
# ---------------------------------------------------------------------------

def compute_context_precision_simple(question: str, contexts: list[str]) -> float:
    """
    Heuristic context precision: fraction of contexts that share keywords
    with the question. Used as a cheap fallback when no LLM judge is available.
    """
    q_tokens = set(re.findall(r'\b\w+\b', question.lower())) - {"what", "is", "are", "how", "the", "a", "an", "of", "in", "for"}
    if not q_tokens:
        return 0.0
    relevant = sum(
        1 for ctx in contexts
        if q_tokens & set(re.findall(r'\b\w+\b', ctx.lower()))
    )
    return relevant / len(contexts) if contexts else 0.0


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Computes multiple evaluation metrics for RAG pipeline outputs.

    Example:
        evaluator = RAGEvaluator(llm_client=openai_client)
        result = evaluator.evaluate_sample(sample)
        df = evaluator.evaluate_batch(samples)
    """

    def __init__(
        self,
        llm_client=None,
        judge_model: str = "gpt-4o-mini",
        bertscore_model: str = "distilbert-base-uncased",
        use_llm_judge: bool = True,
    ):
        self.judge = LLMJudge(llm_client, model=judge_model) if llm_client and use_llm_judge else None
        self.bertscore = BERTScoreEvaluator(model_type=bertscore_model)
        self.rouge = ROUGEEvaluator()

    def evaluate_sample(self, sample: RAGSample) -> EvalResult:
        result = EvalResult(question=sample.question, answer=sample.answer)

        if self.judge:
            result.faithfulness = self.judge.faithfulness(sample.answer, sample.contexts)
            result.answer_relevancy = self.judge.relevancy(sample.question, sample.answer)
            if sample.reference_answer:
                result.context_recall = self.judge.context_recall(sample.reference_answer, sample.contexts)
        else:
            result.faithfulness = 0.0
            result.answer_relevancy = 0.0
            result.context_recall = 0.0

        result.context_precision = compute_context_precision_simple(sample.question, sample.contexts)

        if sample.reference_answer:
            bs = self.bertscore.compute([sample.answer], [sample.reference_answer])
            result.bert_precision = bs["precision"][0]
            result.bert_recall = bs["recall"][0]
            result.bert_f1 = bs["f1"][0]
            result.rouge_l = self.rouge.compute(sample.answer, sample.reference_answer)

        return result

    def evaluate_batch(self, samples: list[RAGSample]) -> list[EvalResult]:
        results = []
        for i, sample in enumerate(samples):
            logger.info("Evaluating sample %d/%d: %s", i + 1, len(samples), sample.question[:60])
            result = self.evaluate_sample(sample)
            results.append(result)
        return results

    def summarize(self, results: list[EvalResult]) -> dict[str, float]:
        """Return mean scores across all evaluated samples."""
        if not results:
            return {}
        metrics = ["faithfulness", "answer_relevancy", "context_recall",
                   "context_precision", "bert_f1", "rouge_l"]
        return {
            metric: round(sum(getattr(r, metric) for r in results) / len(results), 4)
            for metric in metrics
        }

    def save_results(self, results: list[EvalResult], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info("Evaluation results saved to %s", output_path)

    @staticmethod
    def load_eval_questions(eval_path: str | Path) -> list[dict]:
        with open(eval_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("questions", [])
