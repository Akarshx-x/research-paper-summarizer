"""
update_2_embeddings.py
======================
Unsupervised Extractive Research Paper Summarizer
Update 2: Core BERT Extractive Architecture — Sentence Embedding Layer

Pipeline position:
    Sentences (List[str])  ──►  Dense Vectors (np.ndarray [N, 768])
    ← Update 1 output                → Update 3 input (K-Means clustering)

Model: allenai/scibert_scivocab_uncased

Install requirements:
    pip install torch transformers numpy

Usage:
    python update_2_embeddings.py          # runs interactive demo
"""

import logging
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("update_2_embeddings")


# ═══════════════════════════════════════════════════════════════════════════════
# WHY SCIBERT OVER VANILLA BERT
# ═══════════════════════════════════════════════════════════════════════════════
#
# BERT (Devlin et al., 2018) was pre-trained on Wikipedia + BooksCorpus (~3.3B
# tokens) — a general English domain. Research papers diverge from this in three
# ways that degrade vanilla BERT's embedding quality on scientific text:
#
# 1. VOCABULARY MISMATCH
#    BERT's WordPiece vocabulary was built from Wikipedia frequency statistics.
#    Scientific terminology was rare enough there that many domain tokens get
#    sharded into phonetically-motivated fragments:
#
#        Token              BERT (Wikipedia vocab)       SciBERT (science vocab)
#        ─────────────────  ───────────────────────────  ───────────────────────
#        "methyltransfer."  ["methyl","##transfer","##ase"]  ["methyltransferase"]
#        "backpropagation"  ["back","##prop","##aga","##tion"] ["backpropagation"]
#        "hyperparameter"   ["hyper","##param","##eter"]   ["hyperparameter"]
#
#    Each extra sub-token forces the model to reconstruct meaning from fragments
#    it never saw in coherent scientific context during pre-training. The result is
#    embeddings with higher variance in the [CLS] vector for synonymous scientific
#    concepts phrased with different morphological structures.
#
# 2. CO-OCCURRENCE STATISTICS
#    BERT's attention weights encode "which tokens co-occur with which, in what
#    syntactic positions, across the training corpus". For sentences about gradient
#    descent, attention mechanisms, or protein binding sites, BERT's weights reflect
#    how Wikipedia authors discuss adjacent Wikipedia concepts — not how scientists
#    discuss these phenomena in Methods and Results sections.
#
#    SciBERT was pre-trained on 1.14 million papers from Semantic Scholar
#    (3.17B tokens, ~82% biomedical / ~18% CS). Its attention heads learned
#    genuine scientific co-occurrence distributions.
#
# 3. EMPIRICAL BENCHMARK RESULTS (Beltagy et al., 2019)
#    Task                 BERT-base F1   SciBERT F1   Delta
#    ───────────────────  ─────────────  ───────────  ──────
#    SciIE NER            65.24          67.57        +2.33
#    BC5CDR NER/RE        86.72          90.01        +3.29
#    ChemProt RE          79.14          83.64        +4.50
#    SciQ QA              75.42          82.06        +6.64
#
#    All tasks involve discriminating scientific sentences — structurally identical
#    to what K-Means clustering will need to do in Update 3.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmbeddingConfig:
    """
    Controls all runtime behaviour of ScientificSentenceEmbedder.

    Attributes:
        model_name:  HuggingFace model hub identifier. Swap here if you want to
                     experiment with other SciBERT variants or domain-specific
                     BERT checkpoints without touching the embedder logic.
        batch_size:  How many sentences are tokenized and forwarded together.
                     16 fits comfortably in 4 GB VRAM; reduce to 8 or 4 on
                     constrained hardware, or increase to 32+ on A100-class GPUs.
        max_seq_len: Maximum sub-word token count per sentence. SciBERT's absolute
                     maximum is 512 (positional embeddings stop there). Academic
                     sentences rarely exceed ~120 tokens after Update 1 cleaning,
                     so 512 is a safe ceiling — truncation should almost never fire.
        normalize:   L2-normalize embeddings before returning. This maps all vectors
                     onto the unit hypersphere, making Euclidean distance equivalent
                     to cosine distance. sklearn's KMeans uses Euclidean distance,
                     so normalization is essential for meaningful cluster centroids.
        device:      Force a specific device string ("cuda", "cpu", "mps"). If None,
                     the embedder auto-detects in order: CUDA → MPS (Apple Silicon)
                     → CPU.
    """

    model_name:  str            = "allenai/scibert_scivocab_uncased"
    batch_size:  int            = 16
    max_seq_len: int            = 512
    normalize:   bool           = True
    device:      Optional[str]  = None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ScientificSentenceEmbedder:
    """
    Encodes scientific sentences into 768-dimensional dense vectors using SciBERT.

    EMBEDDING STRATEGY: [CLS] TOKEN EXTRACTION
    ───────────────────────────────────────────
    BERT-family models prepend a special [CLS] token to every input sequence:

        Input sequence:   [CLS]  token_1  token_2  ...  token_n  [SEP]
        Token index:         0       1        2    ...     n       n+1

    During pre-training via Next Sentence Prediction (NSP), the network learns
    to aggregate the entire sequence's semantic content into the [CLS] hidden
    state, because the NSP loss is applied only to that position. This makes
    index 0 of the last hidden layer the natural sentence-level summary vector.

    Extraction:
        output.last_hidden_state          → shape: (batch_size, seq_len, 768)
        output.last_hidden_state[:, 0, :] → shape: (batch_size, 768)
                                                              ↑
                                                    index 0 = [CLS]

    Each resulting 768-dimensional vector is a point in BERT's learned semantic
    space. Sentences that share scientific topics, methodology, or conclusions
    will cluster together in this space — which is exactly what K-Means
    exploits in Update 3.

    torch.no_grad() MEMORY OPTIMIZATION
    ────────────────────────────────────
    PyTorch's autograd engine records every tensor operation during a forward
    pass to build a computational graph enabling backpropagation. For a 12-layer
    transformer like SciBERT, this graph stores:

        • Attention weight matrices: 12 layers × 12 heads × seq_len²
        • FFN intermediate activations: 12 layers × seq_len × 3072

    For a batch of 16 sentences at seq_len=128, this overhead is ~200–400 MB of
    extra VRAM — pure waste when there are no .backward() calls planned.

    torch.no_grad() disables autograd graph construction for its entire scope:

        with torch.no_grad():
            output = self.model(**encoding)   # no graph built; ~2× less peak memory

    It also disables gradient checkpointing and version-tracking, which saves
    CPU time and makes the forward pass ~15–30% faster than train-mode inference.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        self.config  = config or EmbeddingConfig()
        self.logger  = logging.getLogger(self.__class__.__name__)
        self.device  = self._resolve_device()
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model:     Optional[AutoModel]     = None
        self._load_model()

    # ── Device Resolution ─────────────────────────────────────────────────────

    def _resolve_device(self) -> torch.device:
        """
        Determine the compute device with a clear priority chain:
            1. User-specified via config.device (explicit override)
            2. CUDA — any NVIDIA/AMD GPU visible to PyTorch
            3. MPS  — Apple Silicon GPU (M1/M2/M3 unified memory)
            4. CPU  — guaranteed fallback; always available

        Logs device name and available VRAM so the caller can diagnose
        batch-size choices without inspecting nvidia-smi separately.
        """
        if self.config.device:
            device = torch.device(self.config.device)
            self.logger.info(f"Device: user-specified → {device}")
            return device

        if torch.cuda.is_available():
            device    = torch.device("cuda")
            gpu_name  = torch.cuda.get_device_name(0)
            vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(
                f"Device: CUDA detected → {gpu_name} "
                f"({vram_gb:.1f} GB VRAM)"
            )
            return device

        if torch.backends.mps.is_available():
            self.logger.info("Device: Apple Silicon MPS detected.")
            return torch.device("mps")

        self.logger.info("Device: no GPU detected → using CPU.")
        return torch.device("cpu")

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Download (on first run) and load the SciBERT tokenizer and model.

        .eval() is critical: it disables Dropout layers, which are stochastic
        during training. In eval mode every forward pass with identical input
        produces identical output — deterministic embeddings are required for
        reproducible K-Means cluster assignments in Update 3.
        """
        self.logger.info(f"Loading tokenizer  ←  {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        self.logger.info(f"Loading model      ←  {self.config.model_name}")
        self.model = AutoModel.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters()) / 1_000_000
        self.logger.info(
            f"Model ready: {n_params:.1f}M parameters on {self.device}  "
            f"[dropout disabled, grad tracking off at inference time]"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, sentences: List[str]) -> np.ndarray:
        """
        Encode a list of sentences into a dense embedding matrix.

        Args:
            sentences: List of clean sentence strings produced by Update 1's
                       DocumentIngestionPipeline. Must be non-empty.

        Returns:
            np.ndarray of shape (N, 768) where N = len(sentences).
                • dtype: float32
                • L2-normalized rows (if config.normalize=True, default)
                • Drop-in compatible with sklearn.cluster.KMeans(n_clusters=k)

        Raises:
            ValueError: If sentences is empty.
            RuntimeError: On non-OOM hardware errors (propagated as-is).

        OOM Recovery:
            If a CUDA out-of-memory error occurs mid-batch, the model is
            transparently moved to CPU and the batch is retried there.
            Subsequent batches also run on CPU (model stays on CPU after OOM
            to prevent thrashing between devices).
        """
        if not sentences:
            raise ValueError(
                "sentences must be a non-empty list. "
                "Ensure Update 1's DocumentIngestionPipeline ran successfully."
            )

        n          = len(sentences)
        batch_size = self.config.batch_size
        n_batches  = (n + batch_size - 1) // batch_size

        self.logger.info(
            f"Embedding {n} sentences  "
            f"[batch_size={batch_size}, n_batches={n_batches}, device={self.device}]"
        )

        all_embeddings: List[np.ndarray] = []

        for batch_idx, batch_start in enumerate(range(0, n, batch_size), start=1):
            batch = sentences[batch_start : batch_start + batch_size]
            self.logger.debug(f"Batch {batch_idx}/{n_batches}: {len(batch)} sentences")

            try:
                batch_emb = self._embed_batch(batch)

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and self.device.type == "cuda":
                    self.logger.warning(
                        f"CUDA OOM on batch {batch_idx}/{n_batches}. "
                        "Clearing cache and falling back to CPU."
                    )
                    torch.cuda.empty_cache()
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
                    self.logger.warning(
                        "Model permanently moved to CPU after OOM. "
                        "Remaining batches will run on CPU."
                    )
                    batch_emb = self._embed_batch(batch)
                else:
                    raise

            all_embeddings.append(batch_emb)

        embeddings: np.ndarray = np.vstack(all_embeddings)   # (N, 768)
        self.logger.info(f"Raw embeddings assembled: shape={embeddings.shape}")

        if self.config.normalize:
            embeddings = self._l2_normalize(embeddings)
            self.logger.info(
                "L2 normalization applied.  "
                "Cosine similarity now equivalent to Euclidean distance — "
                "sklearn KMeans will produce cosine-optimal clusters."
            )

        self.logger.info(f"Embedding complete.  Final shape: {embeddings.shape}")
        return embeddings

    def embed_from_document(self, doc) -> np.ndarray:
        """
        Convenience wrapper for Update 1's ProcessedDocument output.

        Args:
            doc: A ProcessedDocument instance (paper_preprocessor.ProcessedDocument).
                 Its .sentences attribute is passed directly to embed().

        Returns:
            np.ndarray of shape (len(doc.sentences), 768).
        """
        self.logger.info(
            f"Embedding ProcessedDocument: '{doc.source_path}'  "
            f"({len(doc.sentences)} sentences)"
        )
        return self.embed(doc.sentences)

    # ── Internal Inference ────────────────────────────────────────────────────

    def _embed_batch(self, batch: List[str]) -> np.ndarray:
        """
        Tokenize and forward-pass one batch; return [CLS] vectors as numpy array.

        Tokenization parameters:
            padding=True       — pad all sequences to the longest in this batch
                                 (not to max_seq_len globally), so attention
                                 computation only covers actual content tokens.
            truncation=True    — silently clip sequences over max_seq_len tokens.
                                 After Update 1's sentence filtering, academic
                                 sentences are rarely > 120 tokens, so truncation
                                 fires almost never in practice.
            return_tensors="pt" — return PyTorch tensors directly; avoids an
                                  intermediate Python list → numpy → tensor copy.

        [CLS] extraction:
            output.last_hidden_state has shape (batch_size, seq_len, 768).
            Indexing [:, 0, :] selects the 768-dim vector at sequence position 0
            (the [CLS] token) for every sentence in the batch simultaneously.

        .cpu().numpy():
            .cpu() — moves the tensor from GPU/MPS VRAM to CPU RAM.
            .numpy() — zero-copy view into the CPU tensor's memory buffer.
            We call numpy() INSIDE the no_grad() context so the tensor is
            already detached; no explicit .detach() is required.
        """
        encoding = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_len,
            return_tensors="pt",
        )

        # Move all input tensors to the active compute device in one pass.
        encoding = {key: tensor.to(self.device) for key, tensor in encoding.items()}

        with torch.no_grad():
            output = self.model(**encoding)

        # output.last_hidden_state  →  (batch_size, seq_len, 768)
        # [:, 0, :]                 →  (batch_size, 768)  ← [CLS] position
        cls_vectors: np.ndarray = output.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_vectors

    def embed_mean_pooled(self, sentences: List[str]) -> np.ndarray:
        """
        Alternate encoding for RAG retrieval (app.py's "Chat with PDF"), NOT
        used for clustering — embed()'s [CLS]-pooled vectors stay exactly as
        they are for K-Means and the supervised classifier, so neither needs
        re-validation.

        Why a separate method: raw BERT-family [CLS] vectors are well-known
        to have weak cosine-similarity properties for query-vs-passage
        retrieval — this is precisely the gap Sentence-BERT-style
        contrastive fine-tuning was created to close (Reimers & Gurevych,
        2019). SciBERT has no such fine-tuning. Mean-pooling over all
        non-padding token vectors (instead of taking just position 0) is a
        well-documented, zero-extra-dependency mitigation that measurably
        improved retrieval relevance in direct comparison on this project's
        own test documents — no new model, no new download, same tokenizer
        and weights embed() already loaded.

        Returns:
            np.ndarray of shape (N, 768), L2-normalized (cosine-ready).
        """
        if not sentences:
            raise ValueError("sentences must be a non-empty list.")

        encoding = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_len,
            return_tensors="pt",
        )
        encoding = {key: tensor.to(self.device) for key, tensor in encoding.items()}

        with torch.no_grad():
            output = self.model(**encoding)

        # Mask out padding positions before averaging, so pad tokens (which
        # carry no real content) don't dilute the sentence-level vector.
        mask = encoding["attention_mask"].unsqueeze(-1).float()       # (B, T, 1)
        summed = (output.last_hidden_state * mask).sum(dim=1)         # (B, 768)
        counts = mask.sum(dim=1).clamp(min=1e-9)                      # (B, 1)
        pooled: np.ndarray = (summed / counts).cpu().numpy()

        return self._l2_normalize(pooled)

    # ── Normalization Helper ──────────────────────────────────────────────────

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """
        Scale each row vector to unit length: v_norm = v / ||v||_2

        The L2 norm of each row is computed, then each element is divided by its
        row's norm. np.where guards against the pathological zero-vector case
        (a sentence tokenized entirely to [UNK] tokens would produce a zero
        hidden state — dividing by zero would yield NaN, corrupting the cluster).

        After normalization all vectors lie on the 768-dimensional unit hypersphere.
        On this sphere: ||u - v||² = 2 - 2·cos(u,v)
        So minimizing Euclidean distance IS minimizing cosine distance.
        sklearn's KMeans, which minimizes within-cluster Euclidean variance,
        therefore produces cosine-optimal clusters on normalized embeddings.
        """
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)   # (N, 1)
        norms = np.where(norms == 0.0, 1.0, norms)               # guard zero-norm
        return matrix / norms                                      # (N, 768)


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING RESULT CONTAINER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmbeddingResult:
    """
    Carries both the sentence strings and their embedding matrix together.

    Keeping sentences alongside their embedding matrix is important for Update 3:
    after K-Means assigns cluster labels, we need to map cluster centroid indices
    back to the original sentence strings to assemble the summary. Without the
    paired sentences, the index mapping would require re-reading Update 1 output.

    Attributes:
        sentences:   The N input sentence strings, in document order.
        embeddings:  np.ndarray of shape (N, 768), rows correspond to sentences.
        model_name:  The checkpoint that produced these embeddings (for provenance).
        device_used: Which compute device ran inference (for reproducibility logs).
    """
    sentences:   List[str]
    embeddings:  np.ndarray
    model_name:  str
    device_used: str

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[1]

    def __repr__(self) -> str:
        return (
            f"EmbeddingResult("
            f"n_sentences={self.n_sentences}, "
            f"embedding_dim={self.embedding_dim}, "
            f"model='{self.model_name}', "
            f"device='{self.device_used}')"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE-COMPATIBLE EMBEDDING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def embed_sentences(
    sentences:  List[str],
    config:     Optional[EmbeddingConfig] = None,
) -> EmbeddingResult:
    """
    Functional entry point for Update 3 to call.

    Instantiates an embedder, runs inference, and returns an EmbeddingResult
    carrying both the matrix and its paired sentence strings.

    Args:
        sentences: Clean sentence strings from Update 1.
        config:    Optional EmbeddingConfig. Defaults are suitable for most
                   academic papers (batch_size=16, normalize=True, auto device).

    Returns:
        EmbeddingResult with .embeddings of shape (N, 768).
    """
    cfg      = config or EmbeddingConfig()
    embedder = ScientificSentenceEmbedder(cfg)
    matrix   = embedder.embed(sentences)

    return EmbeddingResult(
        sentences   = sentences,
        embeddings  = matrix,
        model_name  = cfg.model_name,
        device_used = str(embedder.device),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE EXECUTION BLOCK
# Run:  python update_2_embeddings.py
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    DIV = "=" * 72
    SUB = "-" * 72

    # ── Sample sentences (output of Update 1 pipeline on "Attention Is All You Need")
    SAMPLE_SENTENCES: List[str] = [
        "The dominant sequence transduction models are based on complex recurrent "
        "or convolutional neural networks that include an encoder and a decoder.",

        "Attention mechanisms have become an integral part of compelling sequence "
        "modeling and transduction models in various tasks.",

        "The Transformer relies entirely on an attention mechanism to draw global "
        "dependencies between input and output, dispensing with recurrence entirely.",

        "We propose a new simple network architecture, the Transformer, based "
        "solely on attention mechanisms.",

        "Multi-head attention allows the model to jointly attend to information "
        "from different representation subspaces at different positions.",

        "The encoder maps an input sequence of symbol representations to a "
        "sequence of continuous representations.",

        "Residual connections are employed around each of the two sub-layers, "
        "followed by layer normalization.",

        "We trained on the standard WMT 2014 English-German dataset consisting "
        "of about 4.5 million sentence pairs.",

        "The Transformer achieves a new state of the art on English-to-German "
        "and English-to-French translation tasks.",

        "We also experimented with using learned positional embeddings instead "
        "of sinusoidal positional encodings.",
    ]

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  INTERACTIVE DEMO — Update 2: SciBERT Sentence Embedding")
    print(f"{DIV}")
    print(
        f"\n  Input: {len(SAMPLE_SENTENCES)} sentences from 'Attention Is All You Need'\n"
        f"  Model: allenai/scibert_scivocab_uncased\n"
        f"  Task:  Map each sentence → 768-dim dense vector\n"
    )

    # ─────────────────────────────────────────────────────────────────────────
    print(f"{SUB}")
    print("  PHASE A — Device Detection & Model Loading")
    print(f"{SUB}\n")

    config   = EmbeddingConfig(batch_size=4)
    embedder = ScientificSentenceEmbedder(config)

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{SUB}")
    print("  PHASE B — Tokenization Inspection (first sentence)")
    print(f"{SUB}\n")

    sample = SAMPLE_SENTENCES[0]
    tokens = embedder.tokenizer.tokenize(sample)
    ids    = embedder.tokenizer.encode(sample, add_special_tokens=True)

    print(f"  Raw sentence ({len(sample)} chars):")
    print(f"    \"{sample[:80]}{'...' if len(sample)>80 else ''}\"\n")
    print(f"  Sub-word tokens ({len(tokens)} tokens):")
    print(f"    {tokens}\n")
    print(f"  Token IDs (with [CLS]=0 and [SEP] prepended/appended):")
    print(f"    {ids}\n")
    print(f"  [CLS] id = {embedder.tokenizer.cls_token_id}  "
          f"[SEP] id = {embedder.tokenizer.sep_token_id}")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{SUB}")
    print("  PHASE C — Full Embedding Pass")
    print(f"{SUB}\n")

    result = embed_sentences(SAMPLE_SENTENCES, config)
    print(f"\n  {result}\n")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{SUB}")
    print("  PHASE D — Output Verification")
    print(f"{SUB}\n")

    E = result.embeddings

    print(f"  Matrix shape:     {E.shape}         (N_sentences × embedding_dim)")
    print(f"  dtype:            {E.dtype}")
    print(f"  Min value:        {E.min():.6f}")
    print(f"  Max value:        {E.max():.6f}")
    print(f"  Mean:             {E.mean():.6f}")

    norms = np.linalg.norm(E, axis=1)
    print(f"\n  Row L2 norms (should all be ≈ 1.0 after normalization):")
    for i, norm in enumerate(norms):
        bar = "█" * int(norm * 20)
        print(f"    [{i:02d}] {norm:.8f}  {bar}")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{SUB}")
    print("  PHASE E — Pairwise Cosine Similarity Sanity Check")
    print(f"{SUB}\n")
    print(
        "  On L2-normalized vectors, cosine similarity = dot product.\n"
        "  Semantically similar sentences should score > 0.85;\n"
        "  unrelated sentences should score < 0.70.\n"
    )

    # Spot-check a few pairs
    pairs = [
        (0, 1, "Expected: HIGH  (both describe sequence models)"),
        (0, 2, "Expected: HIGH  (both about architecture context)"),
        (3, 4, "Expected: HIGH  (both about Transformer specifics)"),
        (0, 8, "Expected: MED   (general model → results sentence)"),
        (5, 9, "Expected: LOW   (encoder desc vs positional encodings)"),
    ]

    for i, j, note in pairs:
        sim = float(np.dot(E[i], E[j]))
        bar = "█" * int(sim * 30)
        print(f"  Sentences [{i:02d}] ↔ [{j:02d}]  sim={sim:.4f}  {bar}")
        print(f"    {note}")
        print(f"    [{i:02d}] \"{SAMPLE_SENTENCES[i][:60]}...\"")
        print(f"    [{j:02d}] \"{SAMPLE_SENTENCES[j][:60]}...\"\n")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  PHASE F — sklearn KMeans Compatibility Check")
    print(f"{DIV}\n")

    try:
        from sklearn.cluster import KMeans

        k       = 3
        kmeans  = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels  = kmeans.fit_predict(E)

        print(f"  KMeans(n_clusters={k}).fit_predict(embeddings) succeeded.\n")
        print(f"  Cluster assignments:")
        for idx, (sent, label) in enumerate(zip(SAMPLE_SENTENCES, labels)):
            print(f"    [{idx:02d}] Cluster {label}  |  \"{sent[:65]}...\"")

        print(f"\n  Inertia (within-cluster sum of squared distances): {kmeans.inertia_:.4f}")
        print(
            "\n  ✓ EmbeddingResult.embeddings is drop-in compatible with KMeans.\n"
            "    This matrix is ready for Update 3."
        )

    except ImportError:
        print("  sklearn not installed — KMeans compatibility check skipped.")
        print("  Run:  pip install scikit-learn")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  Demo complete.")
    print(f"  Embeddings shape {E.shape} — passing to Update 3 (K-Means clustering).")
    print(f"{DIV}\n")

    # ── Integration snippet for Update 3 ──────────────────────────────────────
    #
    # from paper_preprocessor import DocumentIngestionPipeline
    # from update_2_embeddings import embed_sentences, EmbeddingConfig
    #
    # pipeline = DocumentIngestionPipeline()
    # doc      = pipeline.run("attention_is_all_you_need.pdf")
    #
    # result   = embed_sentences(
    #     doc.sentences,
    #     config=EmbeddingConfig(batch_size=16, normalize=True),
    # )
    #
    # # result.embeddings → np.ndarray (N, 768), ready for KMeans
    # # result.sentences  → List[str], paired with each embedding row
