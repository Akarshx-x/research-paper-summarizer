"""
update_3_clustering.py
======================
Unsupervised Extractive Research Paper Summarizer
Update 3: Unsupervised Clustering & Multi-Length Summary Generation Engine

Pipeline position:
    EmbeddingResult (sentences + embeddings)  ──►  SummaryResult (selected sentences)
    ← Update 2 output                              → Update 4 input (final formatting)

Strategy:
    K-Means is fitted on the L2-normalized SciBERT embedding matrix. For each of
    the K clusters, the single sentence whose embedding vector lies closest to that
    cluster's centroid is selected as the cluster's representative. Those K sentences
    are then re-sorted into their original document order to produce a coherent,
    chronologically-ordered extractive summary.

Three summary length modes:
    "short"    → K = 4          (macro thesis, ~4 sentences)
    "one_page" → K = 11         (detailed overview, ~300-400 words)
    "full"     → K = max(5, round(0.15 × N))  (15% compression of full paper)

Install requirements:
    pip install scikit-learn numpy

Usage:
    python update_3_clustering.py          # runs interactive demo
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("update_3_clustering")


# ═══════════════════════════════════════════════════════════════════════════════
# WHY K-MEANS ON L2-NORMALIZED EMBEDDINGS WORKS AS EXTRACTIVE SUMMARIZATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# INTUITION
# ─────────
# After Update 2, every sentence is a unit vector in R^768. Sentences about the
# same topic (e.g., "attention mechanism", "training setup", "results") point in
# roughly the same direction in that space — they cluster together.
#
# K-Means partitions this space into K non-overlapping regions and computes the
# arithmetic mean of all member vectors as the centroid of each region. The centroid
# is NOT itself a sentence — it is a synthetic "average meaning" point. The sentence
# whose vector is closest to the centroid is the most representative real sentence
# for that semantic cluster.
#
# MATH: WHY EUCLIDEAN DISTANCE ≡ COSINE PROXIMITY ON THE UNIT SPHERE
# ────────────────────────────────────────────────────────────────────
# For any two L2-normalized vectors u and v (i.e., ||u|| = ||v|| = 1):
#
#     ||u - v||²  =  ||u||² + ||v||² - 2·(u · v)
#                 =   1     +   1    - 2·cos(u, v)
#                 =   2  -  2·cos(u, v)
#
# Minimizing Euclidean distance ||u - v||² is therefore IDENTICAL to maximizing
# cosine similarity cos(u, v). This is why normalizing in Update 2 was load-bearing:
# it lets sklearn's standard Euclidean KMeans produce cosine-optimal clusters
# without requiring a custom metric.
#
# NOTE: K-Means centroids are the mean of their member vectors. The mean of unit
# vectors is NOT a unit vector (it has norm < 1). The centroid lives INSIDE the
# unit sphere. When we compute distances from member vectors to this centroid,
# we are finding which member most closely matches the "average direction" of the
# cluster — the correct notion of "most representative sentence".
#
# CHRONOLOGICAL RE-SORTING: WHY IT IS MANDATORY
# ───────────────────────────────────────────────
# K-Means assigns clusters based on semantic similarity, not document position.
# A naive output might order sentences as:
#     [sentence 89 (conclusion), sentence 12 (intro), sentence 45 (methods)]
#
# This violates the logical narrative flow of a research paper:
# Abstract → Motivation → Methodology → Results → Conclusion.
#
# After selecting the K centroid sentences, we sort their original document indices
# in ascending order (sorted(selected_indices)). This maps the K sentences back
# onto the paper's timeline, producing a summary that reads in natural order.
#
# THREE-MODE K SELECTION RATIONALE
# ──────────────────────────────────
# "short"    K=4  — One sentence per major paper section (motivation, method,
#                   results, conclusion). Forces extreme compression; good for
#                   a one-paragraph abstract substitute.
#
# "one_page" K=11 — Empirically ~300-400 words for papers with 80-200 sentences.
#                   Covers all major subsections without exhausting a reader.
#
# "full"     K=max(5, round(0.15*N)) — 15% extraction ratio. For a 200-sentence
#                   paper → K=30 sentences. Mirrors human abstractors who typically
#                   compress by 85-90% for comprehensive summaries.
#                   The floor of 5 prevents degenerate summaries on very short inputs.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusteringConfig:
    """
    Controls all runtime behaviour of ScientificPaperSummarizer.

    Attributes:
        short_k:        Fixed K for "short" mode. 4 captures the four canonical
                        paper sections: background, method, experiment, conclusion.
        one_page_k:     Fixed K for "one_page" mode. 11 gives ~300-400 words for
                        typical NLP conference papers (80-200 sentences).
        full_k_ratio:   Fractional sentence extraction rate for "full" mode.
                        0.15 = extract 15% of sentences.
        full_k_floor:   Minimum K for "full" mode regardless of document length.
                        Prevents degenerate 1-2 sentence summaries on short inputs.
        random_state:   KMeans random seed for reproducible cluster assignments.
                        Fixed at 42 so the same paper always produces the same summary.
        n_init:         Number of KMeans restarts with different centroid seeds.
                        "auto" uses sklearn's default (10 for KMeans, 1 for KMeans++).
        max_iter:       Maximum KMeans iterations per restart before forced convergence.
    """

    short_k:       int              = 4
    one_page_k:    int              = 11
    full_k_ratio:  float            = 0.15
    full_k_floor:  int              = 5
    random_state:  int              = 42
    n_init:        Union[int, str]  = "auto"
    max_iter:      int              = 300


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT CONTAINER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SummaryResult:
    """
    Carries the complete clustering output and the final extractive summary.

    Keeping selected_indices alongside the sentence strings is important for
    Update 4's formatting stage: it may need the original position of each
    sentence (e.g., to cite section numbers or to detect introduction sentences
    by their low index values).

    Attributes:
        mode:               One of "short", "one_page", "full".
        k_used:             The actual K that was fitted. May be lower than
                            requested if N < requested K (capped at N-1).
        n_total_sentences:  N from the input EmbeddingResult.
        selected_indices:   Original sentence indices in ascending doc order.
                            len(selected_indices) == len(sentences) == k_used
                            (minus any empty clusters, which are extremely rare).
        sentences:          The actual selected sentence strings, in doc order.
        word_count:         Sum of word counts across all selected sentences.
                            Approximate — splits on whitespace.
        compression_ratio:  len(sentences) / n_total_sentences.
                            0.15 for "full" mode, lower for "short".
        silhouette:         Mean silhouette coefficient (-1 to 1) of the fitted
                            K-Means clustering; higher means better-separated,
                            more semantically coherent clusters. None when it
                            can't be computed (k=1 or k >= N-1 edge cases).
    """

    mode:               str
    k_used:             int
    n_total_sentences:  int
    selected_indices:   List[int]
    sentences:          List[str]
    word_count:         int
    compression_ratio:  float
    silhouette:         Optional[float] = None

    def __repr__(self) -> str:
        silhouette_str = f"{self.silhouette:.4f}" if self.silhouette is not None else "n/a"
        return (
            f"SummaryResult("
            f"mode='{self.mode}', "
            f"k={self.k_used}, "
            f"sentences={len(self.sentences)}/{self.n_total_sentences}, "
            f"words≈{self.word_count}, "
            f"compression={self.compression_ratio:.1%}, "
            f"silhouette={silhouette_str})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ScientificPaperSummarizer:
    """
    Converts an EmbeddingResult (Update 2 output) into an extractive summary
    via K-Means clustering on the L2-normalized SciBERT embedding space.

    Public API:
        summarizer = ScientificPaperSummarizer()
        result     = summarizer.summarize(embedding_result, mode="one_page")

    Diagnostic API:
        inertia_by_k = summarizer.compute_elbow_curve(embedding_result.embeddings)
        # Returns {k: inertia} dict — plot to find natural elbow point.
    """

    _VALID_MODES = frozenset({"short", "one_page", "full"})

    def __init__(self, config: Optional[ClusteringConfig] = None) -> None:
        self.config = config or ClusteringConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    # ── Public: Main Summarization Entry Point ────────────────────────────────

    def summarize(self, embedding_result, mode: str = "one_page") -> SummaryResult:
        """
        Produce an extractive summary for the given mode.

        Args:
            embedding_result: EmbeddingResult from update_2_embeddings.embed_sentences().
                              Must have .sentences (List[str]) and .embeddings (np.ndarray
                              of shape (N, 768), L2-normalized).
            mode:             One of "short", "one_page", "full".

        Returns:
            SummaryResult with .sentences in chronological document order.

        Raises:
            ValueError: If mode is not one of the three valid strings.
            ValueError: If embedding_result contains fewer than 2 sentences
                        (K-Means requires at least 2 data points).
        """
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(self._VALID_MODES)}, got '{mode}'."
            )

        sentences:  List[str]  = embedding_result.sentences
        embeddings: np.ndarray = embedding_result.embeddings
        N = len(sentences)

        if N < 2:
            raise ValueError(
                f"Need at least 2 sentences to cluster; got {N}. "
                "Check that Update 1's pipeline ran successfully."
            )

        k = self._resolve_k(mode, N)

        self.logger.info(
            f"{'─'*60}\n"
            f"  Summarization  |  mode='{mode}'  |  N={N}  |  K={k}\n"
            f"{'─'*60}"
        )

        # ── Core pipeline ─────────────────────────────────────────────────────
        labels, centroids, silhouette = self._fit_kmeans(embeddings, k)
        selected_indices  = self._select_centroid_sentences(embeddings, labels, centroids, k)
        selected_indices  = sorted(selected_indices)                     # chronological sort
        selected_sents    = [sentences[i] for i in selected_indices]

        # ── Statistics ────────────────────────────────────────────────────────
        word_count         = sum(len(s.split()) for s in selected_sents)
        compression_ratio  = len(selected_sents) / N

        result = SummaryResult(
            mode              = mode,
            k_used            = k,
            n_total_sentences = N,
            selected_indices  = selected_indices,
            sentences         = selected_sents,
            word_count        = word_count,
            compression_ratio = compression_ratio,
            silhouette        = silhouette,
        )

        self.logger.info(
            f"Done  →  {len(selected_sents)} sentences selected  |  "
            f"~{word_count} words  |  {compression_ratio:.1%} of original"
        )
        return result

    # ── Public: Diagnostic Elbow Curve ────────────────────────────────────────

    def compute_elbow_curve(
        self,
        embeddings: np.ndarray,
        k_range:    Optional[range] = None,
    ) -> Dict[int, float]:
        """
        Compute KMeans inertia for a range of K values.

        Inertia = within-cluster sum of squared Euclidean distances from each
        point to its assigned centroid. On L2-normalized vectors this equals
        2K - 2·(total cosine similarity to assigned centroid), so lower inertia
        means tighter, more coherent semantic clusters.

        The "elbow" is the K where the inertia curve transitions from steep to
        flat — the point of diminishing returns. Below the elbow, adding one more
        cluster meaningfully separates a semantic topic. Above it, new clusters
        just split already-coherent groups into redundant sub-clusters.

        Args:
            embeddings: L2-normalized embedding matrix, shape (N, 768).
            k_range:    Range of K values to evaluate. Defaults to range(2, min(21, N)).

        Returns:
            Dict mapping k → inertia, in ascending k order.
        """
        N = len(embeddings)
        if k_range is None:
            k_range = range(2, min(21, N))

        self.logger.info(
            f"Computing elbow curve for K in [{k_range.start}, {k_range.stop - 1}]..."
        )

        inertia_by_k: Dict[int, float] = {}
        for k in k_range:
            if k >= N:
                self.logger.debug(f"K={k} >= N={N}, stopping elbow scan.")
                break
            km = KMeans(
                n_clusters   = k,
                random_state = self.config.random_state,
                n_init       = self.config.n_init,
                max_iter     = self.config.max_iter,
            )
            km.fit(embeddings)
            inertia_by_k[k] = float(km.inertia_)
            self.logger.debug(f"  K={k:3d}  inertia={km.inertia_:.4f}")

        self.logger.info("Elbow curve complete.")
        return inertia_by_k

    # ── Private: K Resolution ─────────────────────────────────────────────────

    def _resolve_k(self, mode: str, n_sentences: int) -> int:
        """
        Compute the target K for the given mode and document size.

        All three modes cap K at (n_sentences - 1) to guarantee that every
        cluster has at least one member. KMeans requires strictly fewer clusters
        than data points; violating this raises a ValueError from sklearn.

        Mode logic:
            "short"    → config.short_k (fixed, default 4)
            "one_page" → config.one_page_k (fixed, default 11)
            "full"     → max(config.full_k_floor, round(config.full_k_ratio × N))
        """
        if mode == "short":
            raw_k = self.config.short_k
        elif mode == "one_page":
            raw_k = self.config.one_page_k
        else:  # "full"
            raw_k = max(
                self.config.full_k_floor,
                round(self.config.full_k_ratio * n_sentences),
            )
            self.logger.info(
                f"'full' mode: K = max({self.config.full_k_floor}, "
                f"round({self.config.full_k_ratio} × {n_sentences})) = {raw_k}"
            )

        k = min(raw_k, n_sentences - 1)
        if k < raw_k:
            self.logger.warning(
                f"Requested K={raw_k} but document only has {n_sentences} sentences. "
                f"Capping K at {k}."
            )
        return k

    # ── Private: KMeans Fitting ───────────────────────────────────────────────

    def _fit_kmeans(
        self,
        embeddings: np.ndarray,
        k:          int,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        """
        Fit KMeans and return (labels, centroids, silhouette).

        KMeans initialization uses k-means++ by default in sklearn, which selects
        initial centroids with probability proportional to their squared distance
        from already-chosen centroids. This spreads initial seeds across the
        embedding space and typically converges in fewer iterations than random
        initialization, producing more stable cluster assignments.

        n_init="auto" (sklearn >= 1.2): runs 10 restarts and picks the result
        with the lowest inertia. This is important because K-Means is sensitive
        to initialization; a single run can converge to a local minimum.

        random_state=42 is set on config so that for the same paper and the same
        mode, the summary is always identical — deterministic output is essential
        for a system marketed as having zero hallucinations.

        Returns:
            labels:     np.ndarray (N,) — cluster index (0..K-1) per sentence.
            centroids:  np.ndarray (K, 768) — arithmetic mean of member embeddings
                        per cluster. NOT L2-normalized (means of unit vectors are
                        inside the unit sphere, not on it).
            silhouette: mean silhouette coefficient over all points, or None if
                        k doesn't satisfy 2 <= k <= N-1 (silhouette is undefined
                        outside that range, e.g. tiny documents forced to k=1).
        """
        self.logger.info(
            f"Fitting KMeans: K={k}, n_init={self.config.n_init}, "
            f"random_state={self.config.random_state}"
        )

        kmeans = KMeans(
            n_clusters   = k,
            random_state = self.config.random_state,
            n_init       = self.config.n_init,
            max_iter     = self.config.max_iter,
        )
        kmeans.fit(embeddings)

        self.logger.info(
            f"KMeans converged in {kmeans.n_iter_} iterations  |  "
            f"inertia={kmeans.inertia_:.4f}"
        )

        silhouette = None
        n_labels = len(np.unique(kmeans.labels_))  # may be < k if a cluster ended up empty
        if 2 <= n_labels <= len(embeddings) - 1:
            silhouette = float(silhouette_score(embeddings, kmeans.labels_, metric="cosine"))
            self.logger.info(f"Silhouette score: {silhouette:.4f}")
        else:
            self.logger.warning(
                f"Skipping silhouette score: {n_labels} distinct cluster(s) "
                f"present (need 2..{len(embeddings) - 1})."
            )

        return kmeans.labels_, kmeans.cluster_centers_, silhouette

    # ── Private: Centroid-Closest Sentence Selection ──────────────────────────

    def _select_centroid_sentences(
        self,
        embeddings: np.ndarray,   # (N, 768) L2-normalized sentence vectors
        labels:     np.ndarray,   # (N,)     cluster assignment per sentence
        centroids:  np.ndarray,   # (K, 768) cluster center vectors
        k:          int,
    ) -> List[int]:
        """
        For each cluster, identify the sentence whose embedding is closest to
        that cluster's centroid in Euclidean space.

        ALGORITHM (per cluster c):
            1. member_indices = np.where(labels == c)[0]
               → original sentence indices assigned to cluster c
            2. member_embeddings = embeddings[member_indices]
               → the actual embedding vectors for those sentences
            3. distances = ||member_embeddings - centroids[c]||₂  (row-wise)
               → Euclidean distance from each member to the centroid
            4. selected = member_indices[argmin(distances)]
               → the sentence index with minimum distance to the centroid,
                 i.e., maximum cosine similarity (because inputs are L2-normalized)

        WHY NOT JUST USE np.argmin ON THE FULL MATRIX?
            Computing distances from all N sentences to centroid c and calling
            argmin would also work, but it wastes computation on the K-1 clusters
            that are not c. The per-cluster approach only evaluates the sentences
            that actually belong to each cluster — O(N) total instead of O(N*K).

        Empty cluster handling:
            Extremely rare with K-Means++ initialization on dense embeddings, but
            theoretically possible if two cluster centers collapse to the same
            point. We skip empty clusters with a warning rather than crashing.

        Returns:
            List of K (or fewer, if empty clusters) original sentence indices.
            NOT yet sorted — caller is responsible for chronological sorting.
        """
        selected_indices: List[int] = []

        for cluster_id in range(k):
            member_indices = np.where(labels == cluster_id)[0]  # original indices

            if len(member_indices) == 0:
                self.logger.warning(
                    f"Cluster {cluster_id} is empty (possible with pathological input). "
                    f"Skipping — final summary will have {k - 1} sentences."
                )
                continue

            member_embeddings = embeddings[member_indices]           # (n_members, 768)
            centroid          = centroids[cluster_id]                # (768,)

            # Row-wise L2 distance: shape (n_members,)
            distances         = np.linalg.norm(member_embeddings - centroid, axis=1)
            closest_local_idx = int(np.argmin(distances))
            original_idx      = int(member_indices[closest_local_idx])

            selected_indices.append(original_idx)
            self.logger.debug(
                f"  Cluster {cluster_id:3d}: "
                f"{len(member_indices):3d} members → "
                f"selected sentence [{original_idx:4d}] "
                f"(dist={distances[closest_local_idx]:.4f})"
            )

        return selected_indices


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTIONAL ENTRY POINT  (called by Update 4)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_summary(
    embedding_result,
    mode:   str                        = "one_page",
    config: Optional[ClusteringConfig] = None,
) -> SummaryResult:
    """
    Functional entry point: instantiate summarizer, run clustering, return result.

    Args:
        embedding_result: EmbeddingResult from update_2_embeddings.embed_sentences().
        mode:             "short" | "one_page" | "full".
        config:           Optional ClusteringConfig. Defaults are suitable for
                          most NLP conference papers.

    Returns:
        SummaryResult with .sentences in chronological document order.
    """
    cfg        = config or ClusteringConfig()
    summarizer = ScientificPaperSummarizer(cfg)
    return summarizer.summarize(embedding_result, mode)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE EXECUTION BLOCK
# Run:  python update_3_clustering.py
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    DIV = "=" * 72
    SUB = "-" * 72

    # ── Synthetic EmbeddingResult (mimics Update 2 output) ────────────────────
    # 20 sentences drawn from different semantic sections of
    # "Attention Is All You Need" — enough variety for K-Means to find real clusters.

    from dataclasses import dataclass as _dc, field as _field

    @_dc
    class _MockEmbeddingResult:
        sentences:   list
        embeddings:  np.ndarray
        model_name:  str = "allenai/scibert_scivocab_uncased"
        device_used: str = "cpu"

    SAMPLE_SENTENCES = [
        # ── Background / Motivation (indices 0-4) ─────────────────────────────
        "The dominant sequence transduction models are based on complex recurrent "
        "or convolutional neural networks that include an encoder and a decoder.",

        "Recurrent models typically factor computation along the positions of the "
        "input and output sequences, generating a sequence of hidden states.",

        "Attention mechanisms have become an integral part of compelling sequence "
        "modeling and transduction models in various tasks.",

        "The fundamental constraint of sequential computation, however, remains a "
        "barrier to parallelization within training examples.",

        "In this work we propose the Transformer, a model architecture eschewing "
        "recurrence and instead relying entirely on an attention mechanism.",

        # ── Architecture / Method (indices 5-9) ───────────────────────────────
        "The Transformer follows an encoder-decoder structure using stacked "
        "self-attention and point-wise, fully connected layers.",

        "The encoder maps an input sequence of symbol representations to a "
        "sequence of continuous representations.",

        "Multi-head attention allows the model to jointly attend to information "
        "from different representation subspaces at different positions.",

        "Residual connections are employed around each of the two sub-layers, "
        "followed by layer normalization.",

        "We also use learned embeddings to convert the input tokens and output "
        "tokens to vectors of dimension d_model.",

        # ── Training Setup (indices 10-13) ────────────────────────────────────
        "We trained on the standard WMT 2014 English-German dataset consisting "
        "of about 4.5 million sentence pairs.",

        "We trained our models on one machine with 8 NVIDIA P100 GPUs and each "
        "training step took about 0.4 seconds.",

        "We used the Adam optimizer with a custom learning rate schedule that "
        "increases linearly for the first warmup_steps steps.",

        "We employed three types of regularization during training: residual "
        "dropout, attention dropout, and label smoothing.",

        # ── Positional Encoding (indices 14-15) ───────────────────────────────
        "Since our model contains no recurrence and no convolution, in order for "
        "the model to make use of the order of the sequence we must inject some "
        "information about the relative or absolute position of the tokens.",

        "We also experimented with using learned positional embeddings instead, "
        "and found that the two versions produced nearly identical results.",

        # ── Results (indices 16-19) ───────────────────────────────────────────
        "On the WMT 2014 English-to-German translation task, the big transformer "
        "model outperforms the best previously reported models by more than 2.0 BLEU.",

        "On the WMT 2014 English-to-French translation task, our big model "
        "achieves a BLEU score of 41.0, outperforming all of the previously "
        "published single models at less than one quarter the training cost.",

        "The Transformer generalizes well to other tasks, as shown by its "
        "performance on English constituency parsing.",

        "We are excited about the future of attention-based models and plan to "
        "apply the Transformer to other tasks involving input and output modalities "
        "beyond text.",
    ]

    N = len(SAMPLE_SENTENCES)

    # Build synthetic embeddings using Update 2's EmbeddingResult structure.
    # In a real run these come from ScientificSentenceEmbedder.embed().
    # Here we use seeded random vectors (L2-normalized) as a functional stand-in.
    rng = np.random.default_rng(seed=42)
    raw_emb = rng.standard_normal((N, 768)).astype(np.float32)
    norms   = np.linalg.norm(raw_emb, axis=1, keepdims=True)
    raw_emb = raw_emb / norms

    mock_result = _MockEmbeddingResult(
        sentences  = SAMPLE_SENTENCES,
        embeddings = raw_emb,
    )

    summarizer = ScientificPaperSummarizer()

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  INTERACTIVE DEMO — Update 3: K-Means Extractive Summarization")
    print(f"{DIV}")
    print(
        f"\n  Input : {N} sentences from 'Attention Is All You Need'\n"
        f"  Model : allenai/scibert_scivocab_uncased (mocked with seeded random)\n"
    )

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{SUB}")
    print("  PHASE A — Elbow Curve Diagnostic")
    print(f"{SUB}\n")
    print(
        "  Inertia = within-cluster sum of squared distances.\n"
        "  On L2-normalized vectors: lower inertia = tighter semantic clusters.\n"
        "  The 'elbow' is where the curve transitions from steep to flat.\n"
    )

    inertia_map = summarizer.compute_elbow_curve(
        mock_result.embeddings,
        k_range=range(2, min(16, N)),
    )

    max_inertia = max(inertia_map.values())
    bar_width   = 35
    print(f"  {'K':>3}   {'Inertia':>10}   Chart")
    print(f"  {'-'*3}   {'-'*10}   {'-'*(bar_width)}")
    for k_val, inertia in inertia_map.items():
        bar_len = int((inertia / max_inertia) * bar_width)
        bar     = "█" * bar_len + "░" * (bar_width - bar_len)
        print(f"  {k_val:3d}   {inertia:10.4f}   {bar}")

    # ─────────────────────────────────────────────────────────────────────────
    for mode in ("short", "one_page", "full"):
        print(f"\n{DIV}")
        print(f"  MODE: '{mode.upper()}'")
        print(f"{DIV}\n")

        result = summarizer.summarize(mock_result, mode=mode)

        k_desc = {
            "short":    f"K={result.k_used} (fixed — one sentence per major section)",
            "one_page": f"K={result.k_used} (fixed — detailed overview target)",
            "full":     (
                f"K={result.k_used} "
                f"(= max(5, round(0.15 × {N})) "
                f"= max(5, {round(0.15 * N)}) — 15% compression)"
            ),
        }[mode]

        print(f"  K resolution:  {k_desc}")
        print(f"  Result:        {result}\n")
        print(f"  ── Extractive Summary ({result.word_count} words) ──────────────\n")

        for rank, (idx, sent) in enumerate(
            zip(result.selected_indices, result.sentences), start=1
        ):
            prefix = f"  [{rank:02d}] (sent #{idx:03d})"
            indent = " " * len(prefix)
            words  = sent.split()
            lines, line = [], []
            line_len = 0
            max_line = 63
            for w in words:
                if line_len + len(w) + 1 > max_line and line:
                    lines.append(" ".join(line))
                    line, line_len = [w], len(w)
                else:
                    line.append(w)
                    line_len += len(w) + 1
            if line:
                lines.append(" ".join(line))

            print(f"{prefix}  {lines[0]}")
            for l in lines[1:]:
                print(f"{indent}  {l}")
            print()

        print(f"  Sentences selected : {len(result.sentences)}/{result.n_total_sentences}")
        print(f"  Approximate words  : {result.word_count}")
        print(f"  Compression ratio  : {result.compression_ratio:.1%} of original")
        print(f"  Chronological?     : {result.selected_indices == sorted(result.selected_indices)}")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  PHASE B — Index Chronology Verification")
    print(f"{SUB}\n")
    print("  Verifying that selected_indices are always in ascending document order:\n")
    for mode in ("short", "one_page", "full"):
        r = summarizer.summarize(mock_result, mode=mode)
        is_sorted = r.selected_indices == sorted(r.selected_indices)
        print(f"  mode='{mode}' → indices={r.selected_indices}  chronological={is_sorted}")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  Demo complete.")
    print(f"  SummaryResult.sentences → ready for Update 4 (formatting & output).")
    print(f"{DIV}\n")

    # ── Integration snippet for Update 4 ──────────────────────────────────────
    #
    # from paper_preprocessor   import DocumentIngestionPipeline
    # from update_2_embeddings  import embed_sentences, EmbeddingConfig
    # from update_3_clustering  import generate_summary
    #
    # pipeline  = DocumentIngestionPipeline()
    # doc       = pipeline.run("attention_is_all_you_need.pdf")
    #
    # emb_result = embed_sentences(
    #     doc.sentences,
    #     config=EmbeddingConfig(batch_size=16, normalize=True),
    # )
    #
    # for mode in ("short", "one_page", "full"):
    #     summary = generate_summary(emb_result, mode=mode)
    #     print(f"\n=== {mode.upper()} SUMMARY ({summary.word_count} words) ===")
    #     for sent in summary.sentences:
    #         print(f"  • {sent}")
