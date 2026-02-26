"""Source file for deduplication utilities for tasks."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from src.schemas.task_schemas import Task


load_dotenv()


def get_correct_choice_text(task: Task) -> str:
    """
    Get the text of the correct choice for a multiple-choice task.

    Args:
        task (Task): The task object containing generation metadata and choices.

    Returns
    -------
        str: The text of the correct choice, or an empty string if not found.
    """
    meta = task.generation_metadata or {}
    label = meta.get("correct_answer")
    if not label or not task.choices:
        return ""

    label = str(label).strip()
    for ch in task.choices:
        if str(ch.get("label", "")).strip() == label:
            return str(ch.get("solution", "")).strip()

    return ""


def build_dedup_text(task: Task) -> str:
    """
    Create deduptext for a task by combining the task statement and the correct answer.

    Args:
        task (Task): The task object.

    Returns
    -------
        str: The deduplication text.
    """
    stmt = (task.task_statement or "").strip()
    correct_text = get_correct_choice_text(task)

    if correct_text:
        return f"{stmt}\n\nCORRECT: {correct_text}"
    return stmt


def _sha256(text: str) -> str:
    """Return sha256 hex digest of input text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _l2_normalize(vec: List[float]) -> List[float]:
    """L2-normalize a vector."""
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine_dot(u_norm: List[float], v_norm: List[float]) -> float:
    """Calculate cosine similarity between two L2-normalized vectors."""
    return sum(a * b for a, b in zip(u_norm, v_norm))


def embed_texts_openai(
    client: OpenAI,
    model: str,
    texts: List[str],
    batch_size: int = 128,
) -> List[List[float]]:
    """
    Return embeddings as List[List[float]] aligned with input texts.

    Args:
        client (OpenAI): OpenAI client instance.
        model (str): Embedding model name.
        texts (List[str]): List of texts to embed.
        batch_size (int): Number of texts to embed per API call.

    Returns
    -------
        List[List[float]]: List of embeddings.
    """
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out


def embed_with_cache(
    client: OpenAI,
    model: str,
    texts: List[str],
    cache_path: Optional[Path],
    batch_size: int = 128,
) -> List[List[float]]:
    """
    Cache embeddings keyed by sha256(text) to avoid redundant API calls.

    Args:
        client (OpenAI): OpenAI client instance.
        model (str): Embedding model name.
        texts (List[str]): List of texts to embed.
        cache_path (Optional[Path]): Path to cache file. If None, no caching is used.
        batch_size (int): Number of texts to embed per API call.

    Returns
    -------
        List[List[float]]: List of embeddings aligned with input texts.
    """
    if cache_path is None:
        return embed_texts_openai(client, model, texts, batch_size=batch_size)

    cache: Dict[str, Any] = {"model": model, "items": {}}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {"model": model, "items": {}}

    items: Dict[str, Any] = cache.get("items", {})
    out: List[Optional[List[float]]] = [None] * len(texts)

    to_embed_idx: List[int] = []
    to_embed_sha: List[str] = []
    to_embed_txt: List[str] = []

    for i, t in enumerate(texts):
        h = _sha256(t)
        cached = items.get(h)
        if isinstance(cached, list) and cached:
            out[i] = cached
        else:
            to_embed_idx.append(i)
            to_embed_sha.append(h)
            to_embed_txt.append(t)

    if to_embed_txt:
        new_embs = embed_texts_openai(
            client, model, to_embed_txt, batch_size=batch_size
        )
        for idx, h, emb in zip(to_embed_idx, to_embed_sha, new_embs):
            items[h] = emb
            out[idx] = emb

        cache["model"] = model
        cache["items"] = items
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    return [e if e is not None else [] for e in out]


def deduplicate_tasks_for_chapter(
    tasks: List[Task],
    *,
    chapter_id: str,
    embedding_model: str,
    threshold: float,
    keep_policy: str = "first",
    cache_path: Optional[Path] = None,
    batch_size: int = 128,
) -> Tuple[List[Task], List[Task], Dict[str, Any]]:
    """
    Create chapter-level dedup based on cosine similarity between embeddings of:
    (task_statement + correct_answer_text).

    Args:
        tasks (List[Task]): List of tasks to deduplicate.
        chapter_id (str): Chapter identifier.
        embedding_model (str): OpenAI embedding model name.
        threshold (float): Cosine similarity threshold for deduplication.
        keep_policy (str): "first" to keep first, "prefer_longer" to keep longer text.
        cache_path (Optional[Path]): Path to embedding cache file.
        batch_size (int): Number of texts to embed per API call.

    Returns
    -------
        Tuple[List[Task], List[Task], Dict[str, Any]]:
            - kept tasks
            - discarded tasks
            - deduplication report
    """  # noqa: D205
    if not tasks:
        report = {
            "chapter_id": chapter_id,
            "num_before": 0,
            "num_after": 0,
            "num_discarded": 0,
            "discarded": [],
            "kept_source_task_ids": [],
        }
        return [], [], report

    # 1) build dedup texts
    texts = [build_dedup_text(t) for t in tasks]

    # 2) embeddings
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embs = embed_with_cache(
        client=client,
        model=embedding_model,
        texts=texts,
        cache_path=cache_path,
        batch_size=batch_size,
    )

    # 3) normalize
    embs_norm = [_l2_normalize(e) for e in embs]

    # 4) order by keep policy
    idxs = list(range(len(tasks)))
    if keep_policy == "prefer_longer":
        idxs.sort(key=lambda i: len(texts[i]), reverse=True)
    else:
        idxs.sort()  # "first"

    kept_idxs: List[int] = []
    kept_embs: List[List[float]] = []
    discarded_idxs: List[int] = []
    discarded_meta: List[Dict[str, Any]] = []

    # 5) greedy dedup
    for i in idxs:
        if not kept_idxs:
            kept_idxs.append(i)
            kept_embs.append(embs_norm[i])
            continue

        best_sim = -1.0
        best_kept_pos = -1
        for pos, ke in enumerate(kept_embs):
            sim = _cosine_dot(embs_norm[i], ke)
            if sim > best_sim:
                best_sim = sim
                best_kept_pos = pos

        matched_keep_idx = kept_idxs[best_kept_pos]
        if best_sim >= threshold:
            discarded_idxs.append(i)
            discarded_meta.append(
                {
                    "discarded_source_task_id": tasks[i].task_id,
                    "matched_kept_source_task_id": tasks[matched_keep_idx].task_id,
                    "similarity": best_sim,
                }
            )
        else:
            kept_idxs.append(i)
            kept_embs.append(embs_norm[i])

    kept_tasks = [tasks[i] for i in kept_idxs]
    discarded_tasks = [tasks[i] for i in discarded_idxs]

    report = {
        "chapter_id": chapter_id,
        "embedding_model": embedding_model,
        "threshold": threshold,
        "keep_policy": keep_policy,
        "num_before": len(tasks),
        "num_after": len(kept_tasks),
        "num_discarded": len(discarded_tasks),
        "kept_source_task_ids": [t.task_id for t in kept_tasks],
        "discarded": discarded_meta,
    }
    return kept_tasks, discarded_tasks, report


def assign_chapter_level_task_ids(
    kept_tasks: List[Task],
    *,
    chapter_id: str,
    prefix: Optional[str] = None,
) -> List[Task]:
    """
    Assign new task IDs to kept tasks after deduplication.

    Args:
        kept_tasks (List[Task]): List of kept tasks.
        chapter_id (str): Chapter identifier.
        prefix (Optional[str]): Optional prefix for task IDs. If None, use chapter_id.

    Returns
    -------
        List[Task]: List of kept tasks with updated task IDs and metadata.
    """
    if prefix is None:
        prefix = chapter_id

    for i, t in enumerate(kept_tasks):
        meta = t.generation_metadata or {}
        meta["source_task_id"] = t.task_id
        meta["dedup_kept"] = True
        meta["chapter_id"] = meta.get("chapter_id") or chapter_id
        t.generation_metadata = meta

        t.task_id = f"{prefix}__task_{i:03d}"

    return kept_tasks


def mark_discarded_metadata(
    discarded_tasks: List[Task],
    *,
    chapter_id: str,
    dedup_report: Dict[str, Any],
) -> List[Task]:
    """
    Mark discarded tasks with dedup metadata so you can save them too.

    Args:
        discarded_tasks (List[Task]): List of discarded tasks.
        chapter_id (str): Chapter identifier.
        dedup_report (Dict[str, Any]): Dedup report containing matched kept task IDs.

    Returns
    -------
        List[Task]: List of discarded tasks with updated metadata.
    """
    m: Dict[str, Tuple[str, float]] = {}
    for row in dedup_report.get("discarded", []):
        d = row.get("discarded_source_task_id")
        k = row.get("matched_kept_source_task_id")
        s = row.get("similarity")
        if isinstance(d, str) and isinstance(k, str) and isinstance(s, (int, float)):
            m[d] = (k, float(s))

    for t in discarded_tasks:
        meta = t.generation_metadata or {}
        meta["dedup_kept"] = False
        meta["chapter_id"] = meta.get("chapter_id") or chapter_id
        matched = m.get(t.task_id)
        if matched:
            meta["dedup_matched_kept_source_task_id"] = matched[0]
            meta["dedup_similarity"] = matched[1]
        t.generation_metadata = meta

    return discarded_tasks
