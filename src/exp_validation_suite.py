"""
Phase 3: Robustness Validation Experiments
==========================================

This script runs the Phase 3 robustness experiments with CORRECT scaling,
matching the methodology in run_full_experiment.py.

Experiments:
- Exp 3A: Real English Control (Does the wormhole persist with real concepts?)
- Exp 3B: Unrelated Baseline (Do "Cousins" differ from "Random Strangers"?)
- Exp 3C: Trained Mapper (Does supervised training separate Truth from Hallucination?)


Usage:
    python exp_validation_suite.py --model qwen --seed <seed>
    python exp_validation_suite.py --model deepseek --seed <seed>
"""
 ## this code has more sample size

import os
import sys
import re
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import threading
from queue import Queue
import geoopt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import random
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# OPTIMIZATION: Cache generated outputs to avoid duplicate generation
GENERATION_CACHE = {}

def get_cached_generation(model, prompt, max_tokens, temperature=0.0):
    """
    Generate text with caching to avoid duplicate generation.
    2x speedup when same prompt is used multiple times.
    """
    cache_key = hash(prompt + str(max_tokens))
    if cache_key in GENERATION_CACHE:
        return GENERATION_CACHE[cache_key]
    
    output = model.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
    output_str = output if isinstance(output, str) else model.tokenizer.decode(output[0])
    GENERATION_CACHE[cache_key] = output_str
    return output_str

def clear_generation_cache():
    """Clear the generation cache (call between different experiments)."""
    global GENERATION_CACHE
    GENERATION_CACHE = {}

# ==========================================
# CONFIGURATION (No Hardcoded Values!)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

# ==========================================
# MULTI-GPU CONFIGURATION (Auto-Detection)
# ==========================================
def get_available_gpus():
    """Automatically detect all available GPUs."""
    if not torch.cuda.is_available():
        return []
    n_gpus = torch.cuda.device_count()
    gpus = []
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            'id': i,
            'name': props.name,
            'memory_gb': props.total_memory / (1024**3)
        })
    return gpus

AVAILABLE_GPUS = get_available_gpus()
N_GPUS = len(AVAILABLE_GPUS)
USE_MULTI_GPU = N_GPUS > 1

if AVAILABLE_GPUS:
    print(f"\n🎮 GPU Configuration:")
    print(f"  Found {N_GPUS} GPU(s):")
    for gpu in AVAILABLE_GPUS:
        print(f"    GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
    if USE_MULTI_GPU:
        print(f"  ✅ Multi-GPU mode ENABLED - will parallelize across {N_GPUS} GPUs")
    else:
        print(f"  Single GPU mode")
else:
    print("\n⚠️  No GPUs found, running on CPU")

# Model-Specific Configuration
CONFIG = {
    'qwen': {
        'layer': 23,              # Optimal layer for Qwen (validated)
        'samples_default': 300,   # More samples since faster
        'reasoning_tokens': 0,    # No reasoning for standard models
        'probe_after_reasoning': False,
    },
    'deepseek': {
        'layer': 23,              # Optimal layer for DeepSeek (validated experimentally)
        'samples_default': 300,   # Minimum recommended (80 train, 20 test)
        'reasoning_tokens': 512,  # Tokens to generate for reasoning probe (matching run_full_experiment.py)
        'probe_after_reasoning': True,
    }
}

# Auth - Load HF_TOKEN from environment
if not os.environ.get("HF_TOKEN"):
    print("Warning: HF_TOKEN not found in environment. Please set it in .env file.")

# ==========================================
# THINKING TOKEN DETECTION (MI Peaks Alignment)
# ==========================================
# Based on: "Demystifying Reasoning Dynamics with Mutual Information"
# These tokens show sudden MI spikes during reasoning

THINKING_TOKENS = [
    # Hesitation/reflection tokens
    "hmm", "wait", "let me", "actually", "oh", "so",
    # Logical connectors
    "therefore", "thus", "hence", "because", "since",
    # Conditional reasoning
    "if", "then", "but", "however", "although",
    # Sequential markers
    "first", "second", "next", "finally", "step",
    # Conclusion tokens
    "means", "implies", "shows", "proves"
]

# OPTIMIZATION: Pre-compile regex for 5x faster matching
THINKING_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(t) for t in THINKING_TOKENS) + r')\b', 
    re.IGNORECASE
)

def find_thinking_positions(output_tokens, return_tokens=False):
    """
    Identify thinking token positions (analogous to MI peaks).
    """
    positions = []
    found_tokens = []
    
    for i, tok in enumerate(output_tokens):
        tok_str = str(tok).lower().strip()
        if THINKING_PATTERN.search(tok_str):
            positions.append(i)
            if return_tokens:
                found_tokens.append(tok_str)
    
    if return_tokens:
        return positions, found_tokens
    return positions

def compute_thinking_metrics(output_tokens):
    """
    Compute MI Peaks-aligned metrics from thinking tokens.
    """
    positions = find_thinking_positions(output_tokens)
    n_tokens = len(output_tokens)
    n_thinking = len(positions)
    
    density = n_thinking / max(n_tokens, 1)
    
    if n_thinking >= 2:
        intervals = [positions[i+1] - positions[i] for i in range(n_thinking - 1)]
        mean_interval = sum(intervals) / len(intervals)
        interval_variance = sum((x - mean_interval)**2 for x in intervals) / len(intervals)
    else:
        mean_interval = 0.0
        interval_variance = 0.0
    
    return {
        'thinking_positions': positions,
        'thinking_count': n_thinking,
        'thinking_density': density,
        'mean_interval': mean_interval,
        'interval_variance': interval_variance
    }

def extract_thinking_activations(model, prompt, layer_idx, max_tokens=None):
    """
    Extract activations at thinking token positions (MI Peaks methodology).
    Stops at min(max_tokens, </think> position).
    """
    # Use CONFIG if max_tokens not specified
    if max_tokens is None:
        max_tokens = CONFIG['deepseek']['reasoning_tokens']
    
    # Generate with caching
    output_str = get_cached_generation(model, prompt, max_tokens)
    
    # Tokenize
    all_str_tokens = model.to_str_tokens(output_str)
    prompt_toks = model.to_tokens(prompt)[0]
    prompt_len = len(prompt_toks)
    new_tokens = all_str_tokens[prompt_len:]
    
    # Find </think> position (stop point)
    think_end_pos = -1
    for i, tok in enumerate(new_tokens):
        if "</think>" in tok or "think>" in tok:
            think_end_pos = i
            break
    
    # Determine effective token range
    if think_end_pos > 0:
        effective_tokens = new_tokens[:think_end_pos]
        n_effective = think_end_pos
    else:
        effective_tokens = new_tokens
        n_effective = len(new_tokens)
    
    # Find thinking positions in effective range
    thinking_positions = find_thinking_positions(effective_tokens)
    thinking_metrics = compute_thinking_metrics(effective_tokens)
    
    # Get activations
    with torch.no_grad():
        _, cache = model.run_with_cache(output_str, 
                                        names_filter=lambda x: x.endswith("hook_resid_post"))
        full_acts = cache[f"blocks.{layer_idx}.hook_resid_post"][0]
        
        # Slice to generated tokens only
        gen_acts = full_acts[prompt_len:]
        
        # Apply effective range cutoff
        if think_end_pos > 0 and think_end_pos < gen_acts.shape[0]:
            gen_acts = gen_acts[:think_end_pos]
        
        # Extract thinking-only activations
        if len(thinking_positions) > 0:
            valid_positions = [p for p in thinking_positions if p < gen_acts.shape[0]]
            if valid_positions:
                thinking_acts = gen_acts[valid_positions].cpu()
            else:
                thinking_acts = gen_acts[-1:].cpu()
        else:
            thinking_acts = gen_acts[-1:].cpu()
        
        all_acts = gen_acts.cpu()
        del cache
        torch.cuda.empty_cache()
    
    return {
        'output_str': output_str,
        'all_acts': all_acts,
        'thinking_acts': thinking_acts,
        'thinking_positions': thinking_positions,
        'thinking_metrics': thinking_metrics,
        'think_end_pos': think_end_pos,
        'n_effective_tokens': n_effective
    }

# ==========================================
# REPRODUCIBILITY: Global Seed
# ==========================================
def set_global_seed(seed=42):
    """Ensure reproducibility across all random sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🎲 Global seed set to {seed} for reproducibility")

def generate_balanced_dataset(generator, n_samples, seed=42):
    """
    Generate dataset with BALANCED class distribution and depth stratification.
    Target: 1/3 True, 1/3 Hallucination, 1/3 Unrelated
    """
    rng = random.Random(seed)
    n_per_class = n_samples // 3
    
    collected = {
        'true': {'shallow': [], 'deep': []},
        'hallucination': [],
        'unrelated': []
    }
    
    max_attempts = n_samples * 20
    attempts = 0
    
    while attempts < max_attempts:
        batch = generator.generate_sample()
        if batch is None:
            attempts += 1
            continue
            
        for sample in batch:
            t = sample['type']
            if t == 'true':
                depth = sample.get('depth', 1)
                bucket = 'shallow' if depth <= 2 else 'deep'
                if len(collected['true'][bucket]) < n_per_class // 2:
                    collected['true'][bucket].append(sample)
            elif t == 'hallucination' and len(collected['hallucination']) < n_per_class:
                collected['hallucination'].append(sample)
            elif t == 'unrelated' and len(collected['unrelated']) < n_per_class:
                collected['unrelated'].append(sample)
        
        true_count = len(collected['true']['shallow']) + len(collected['true']['deep'])
        if (true_count >= n_per_class and 
            len(collected['hallucination']) >= n_per_class and
            len(collected['unrelated']) >= n_per_class):
            break
        attempts += 1
    
    n_shallow = n_per_class // 2
    n_deep = n_per_class - n_shallow
    
    dataset = []
    dataset.extend(collected['true']['shallow'][:n_shallow])
    dataset.extend(collected['true']['deep'][:n_deep])
    dataset.extend(collected['hallucination'][:n_per_class])
    dataset.extend(collected['unrelated'][:n_per_class])
    
    rng.shuffle(dataset)
    
    actual_true = len([d for d in dataset if d['type'] == 'true'])
    actual_hall = len([d for d in dataset if d['type'] == 'hallucination'])
    actual_unrel = len([d for d in dataset if d['type'] == 'unrelated'])
    
    print(f"📊 Balanced Dataset (seed={seed}): {len(dataset)} samples")
    print(f"   • True: {actual_true}, Hall: {actual_hall}, Unrel: {actual_unrel}")
    
    return dataset

# ==========================================
# Model Loading (Identical to run_full_experiment.py)
# ==========================================
def get_model(model_type="standard", target_device=None):
    """
    Load model onto a specific device.
    
    Args:
        model_type: "standard" (Qwen) or "reasoning" (DeepSeek)
        target_device: Specific device like "cuda:0", "cuda:1", etc.
                      If None, uses the global 'device' variable.
    """
    if target_device is None:
        target_device = device
    
    if model_type == "standard":
        print(f"🔧 Loading Standard Model: Qwen/Qwen2.5-7B-Instruct on {target_device}")
        return HookedTransformer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device=target_device)
    elif model_type == "reasoning":
        print(f"🔧 Loading Reasoning Model: DeepSeek-R1-Distill-Qwen-7B on {target_device}")
        # IMPORTANT: Load HF model to CPU first to avoid double GPU memory allocation
        print("...downloading via Transformers (AutoModel) to CPU...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
            torch_dtype=torch.float16,  # Always use FP16 for memory efficiency
            trust_remote_code=True,
            device_map="cpu",  # Load to CPU first to avoid OOM when wrapping
        )
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        
        print(f"...wrapping DeepSeek weights in HookedTransformer on {target_device}...")
        hooked_model = HookedTransformer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=target_device,
            fold_ln=False,
            fold_value_biases=False,   # Disable to avoid meta tensor issues
            center_writing_weights=False,
            center_unembed=False
        )
        
        # Free CPU memory from original HF model
        del hf_model
        torch.cuda.empty_cache()
        
        print("✅ DeepSeek-R1 model loaded successfully!")
        return hooked_model

# ==========================================
# 1. Data Generation (Identical to run_full_experiment.py)
# ==========================================
class LogicTreeGenerator:
    """
    Generate hierarchical logic trees.
    
    Supports multiple domains for cross-domain generalization testing:
    - 'fiction': Nonsense words (wumpus, fele, lorpus)
    - 'animals': Real animal hierarchy (animal, mammal, cat, dog)
    - 'geography': Geographic hierarchy (place, continent, country, city)
    """
    def __init__(self, mode='fiction', depth=5, seed=42):
        self.mode = mode
        self.depth = depth
        self.rng = random.Random(seed)
        
        # Domain-specific vocabulary
        if mode == 'fiction':
            self.nouns = ["wumpus", "fele", "lorpus", "grumpus", "tumpus", 
                          "zumpus", "yumpus", "brompus", "numpus", "umpus"]
            self.subject = "Max"
        elif mode == 'animals' or mode == 'real':  # 'real' for backward compat
            self.nouns = ["animal", "mammal", "vertebrate", "carnivore", "feline", 
                          "canine", "reptile", "cat", "dog", "snake", "lizard", 
                          "poodle", "tabby", "cobra"]
            self.subject = "Spot"
        elif mode == 'geography':
            # NEW: Geography domain for cross-domain generalization
            self.nouns = ["place", "continent", "country", "region", "state", 
                          "city", "town", "America", "Europe", "France", 
                          "California", "Paris", "London", "Berlin"]
            self.subject = "Location X"
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'fiction', 'animals', or 'geography'")
            
    def generate_sample(self):
        concepts = self.rng.sample(self.nouns, min(len(self.nouns), 10))
        G = nx.DiGraph()
        root = concepts[0]; G.add_node(root); current = [root]; used = {root}
        for d in range(self.depth):
            nxt = []
            for p in current:
                for _ in range(self.rng.randint(1, 2)):
                    rem = [c for c in concepts if c not in used]
                    if not rem: break
                    child = rem[0]; used.add(child); G.add_edge(child, p); nxt.append(child)
            current = nxt; 
            if not current: break
            
        if len(G.nodes) < 3: return None
        leaves = [n for n in G.nodes if G.in_degree(n) == 0]
        if not leaves: return None
        start = self.rng.choice(leaves)
        subject = self.subject  # Use domain-specific subject
        
        edges = list(G.edges()); self.rng.shuffle(edges)
        ctx = " ".join([f"Every {u} is a {v}." for u, v in edges]) + f" {subject} is a {start}."
        
        samples = []
        true_types = set(nx.descendants(G, start)) | {start}
        anc = list(true_types - {start})
        if anc:
            tgt = self.rng.choice(anc)
            hops = nx.shortest_path_length(G, start, tgt)
            samples.append({"context": ctx, "query": f"Is {subject} a {tgt}?", "label": "TRUE", "depth": hops, "type": "true"})
            
        false_t = list(set(G.nodes) - true_types)
        if false_t:
            tgt = self.rng.choice(false_t)
            samples.append({"context": ctx, "query": f"Is {subject} a {tgt}?", "label": "FALSE", "depth": -1, "type": "hallucination"})
            
        strangers = [n for n in self.nouns if n not in G.nodes]
        if strangers:
            tgt = self.rng.choice(strangers)
            samples.append({"context": ctx, "query": f"Is {subject} a {tgt}?", "label": "FALSE", "depth": -2, "type": "unrelated"})
        return samples

# ==========================================
# 2. Hyperbolic Mapper (Stable + Discriminative)
# ==========================================
class HyperbolicMapper(nn.Module):
    """
    Hyperbolic Mapper that balances stability with discriminative power.
    
    Key insight: Input normalization destroys magnitude information!
    With use_layer_norm=True (default): Uses LayerNorm for stability
    With use_layer_norm=False: Preserves magnitude for ablation study
    
    Architecture: ϕ(h) = exp_0^D(scale · W · [LayerNorm(h) or h])
    """
    def __init__(self, input_dim, output_dim=16, c=1.0, use_layer_norm=True):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=c)
        self.use_layer_norm = use_layer_norm
        
        # LayerNorm option (can be disabled for ablation)
        if use_layer_norm:
            self.input_norm = nn.LayerNorm(input_dim)
        else:
            # Simple scaling for stability when not using LayerNorm
            self.register_buffer('input_scale', torch.tensor(0.01))
            
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        # Spectral norm prevents weight explosion
        self.fc = nn.utils.spectral_norm(self.fc)
        # Learnable scale (initialized small)
        self.log_scale = nn.Parameter(torch.tensor(-2.0))
    
    def forward(self, x):
        # 1. Normalize (or scale if no LayerNorm)
        if self.use_layer_norm:
            x = self.input_norm(x)
        else:
            x = x * self.input_scale  # Simple scaling for stability
        # 2. Project
        v = self.fc(x)
        # 3. Learnable scale (bounded to prevent saturation)
        scale = torch.sigmoid(self.log_scale) * 0.5
        v = v * scale
        # 4. Map to Poincaré ball
        return self.manifold.expmap0(v)
    
    def dist(self, u, v):
        """Hyperbolic distance in Poincaré Ball"""
        return self.manifold.dist(u, v)
    
    def radius(self, u):
        """Euclidean radius"""
        return torch.norm(u, dim=-1)


# ==========================================
# 2b. Hyperbolic Classifier (TRUE vs HALLUCINATION)
# ==========================================
class HyperbolicClassifier(nn.Module):
    """
    Binary classifier using hyperbolic geometry.
    
    This fixes the circular training logic by learning to classify
    TRUE vs HALLUCINATION directly, instead of regressing on depth.
    
    Architecture: 
    1. Project to hyperbolic space
    2. Learn class centroids (TRUE centroid, HALLUCINATION centroid)
    3. Classify based on distance to centroids
    """
    def __init__(self, input_dim, hidden_dim=64, c=1.0, use_layer_norm=True):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=c)
        self.use_layer_norm = use_layer_norm
        
        if use_layer_norm:
            self.input_norm = nn.LayerNorm(input_dim)
        else:
            self.register_buffer('input_scale', torch.tensor(0.01))
        
        # Two-layer projection to hyperbolic space
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 16)
        
        # Learnable class centroids in tangent space
        self.true_centroid = nn.Parameter(torch.randn(16) * 0.1)
        self.hall_centroid = nn.Parameter(torch.randn(16) * 0.1)
        
    def forward(self, x):
        """Project to hyperbolic space."""
        if self.use_layer_norm:
            x = self.input_norm(x)
        else:
            x = x * self.input_scale
            
        x = torch.relu(self.fc1(x))
        x = self.fc2(x) * 0.1  # Keep small for stability
        return self.manifold.expmap0(x)
    
    def classify(self, x):
        """
        Classify samples as TRUE or HALLUCINATION.
        Returns logits (positive = TRUE, negative = HALLUCINATION)
        """
        emb = self.forward(x)
        
        # Map centroids to hyperbolic space
        true_c = self.manifold.expmap0(self.true_centroid.unsqueeze(0))
        hall_c = self.manifold.expmap0(self.hall_centroid.unsqueeze(0))
        
        # Distances to centroids
        dist_true = self.manifold.dist(emb, true_c.expand(emb.shape[0], -1))
        dist_hall = self.manifold.dist(emb, hall_c.expand(emb.shape[0], -1))
        
        # Logits: negative distance to TRUE + positive distance to HALL
        # (closer to TRUE → positive logit → class 1)
        logits = dist_hall - dist_true
        return logits
    
    def predict(self, x):
        """Return binary predictions (1=TRUE, 0=HALLUCINATION)"""
        return (self.classify(x) > 0).long()


def train_classifier(acts, metas, epochs=300, lr=0.01, use_layer_norm=True, verbose=True):
    """
    Train binary classifier: TRUE vs HALLUCINATION.
    
    This is the proper way to test if geometries are distinct -
    train a classifier and measure accuracy/AUROC.
    
    Returns:
        classifier: Trained HyperbolicClassifier
        metrics: Dict with accuracy, auroc, etc.
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    # Prepare data (TRUE=1, HALLUCINATION=0)
    true_idxs = [i for i, m in enumerate(metas) if m['label'] == 'TRUE']
    hall_idxs = [i for i, m in enumerate(metas) if m['type'] == 'hallucination']
    
    if len(true_idxs) < 5 or len(hall_idxs) < 5:
        print(f"  ⚠️ Not enough samples: TRUE={len(true_idxs)}, HALL={len(hall_idxs)}")
        return None, {'accuracy': 0.5, 'auroc': 0.5}
    
    # Create balanced dataset
    n_samples = min(len(true_idxs), len(hall_idxs))
    true_idxs = true_idxs[:n_samples]
    hall_idxs = hall_idxs[:n_samples]
    
    all_idxs = true_idxs + hall_idxs
    X = acts[all_idxs].to(device).float()
    y = torch.tensor([1] * len(true_idxs) + [0] * len(hall_idxs), device=device).float()
    
    # Train/test split (80/20)
    split = int(0.8 * len(X))
    perm = torch.randperm(len(X))
    train_idx, test_idx = perm[:split], perm[split:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    if verbose:
        print(f"  📊 Training classifier: {len(true_idxs)} TRUE, {len(hall_idxs)} HALL samples")
        print(f"     Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"     LayerNorm: {use_layer_norm}")
    
    # Create classifier
    classifier = HyperbolicClassifier(X.shape[1], use_layer_norm=use_layer_norm).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = classifier.classify(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer.step()
        
        if verbose and epoch % 100 == 0:
            with torch.no_grad():
                preds = (logits > 0).float()
                acc = (preds == y_train).float().mean().item()
                print(f"     Epoch {epoch}: Loss = {loss.item():.4f}, Train Acc = {acc:.2%}")
    
    # Evaluate on test set
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier.classify(X_test)
        test_preds = (test_logits > 0).float()
        
        accuracy = (test_preds == y_test).float().mean().item()
        
        # AUROC
        try:
            auroc = roc_auc_score(y_test.cpu().numpy(), test_logits.cpu().numpy())
        except:
            auroc = 0.5
    
    if verbose:
        print(f"  ✅ Test Results: Accuracy = {accuracy:.2%}, AUROC = {auroc:.3f}")
    
    metrics = {
        'accuracy': accuracy,
        'auroc': auroc,
        'n_true': len(true_idxs),
        'n_hall': len(hall_idxs),
        'use_layer_norm': use_layer_norm
    }
    
    return classifier, metrics


def compute_perplexity(model, prompts, max_samples=100):
    """
    Compute perplexity for each prompt.
    
    This is a control to check if hyperbolic distance is just measuring
    uncertainty/perplexity rather than logical structure.
    
    If correlation(hyperbolic_distance, perplexity) > 0.9:
        We're just measuring uncertainty, not truth geometry.
    """
    perplexities = []
    
    for prompt in tqdm(prompts[:max_samples], desc="Computing perplexity"):
        with torch.no_grad():
            tokens = model.to_tokens(prompt)
            logits = model(tokens)
            
            # Compute cross-entropy loss (=log perplexity)
            shift_logits = logits[0, :-1, :]
            shift_labels = tokens[0, 1:]
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits, shift_labels)
            perplexity = torch.exp(loss).item()
            
            perplexities.append(perplexity)
            torch.cuda.empty_cache()
    
    return perplexities


def perplexity_correlation_check(model, acts, metas, mapper, prompts, max_samples=None):
    """
    Check correlation between hyperbolic distance and perplexity.
    
    Returns:
        correlation: Pearson r between distance and perplexity
        interpretation: String explaining what the correlation means
    """
    from scipy.stats import pearsonr
    
    # Use full dataset if max_samples not specified
    if max_samples is None:
        max_samples = len(prompts)
    
    print("\n📊 Perplexity Control Check:")
    print(f"  Computing perplexity for {max_samples} samples...")
    
    # Get perplexities
    perplexities = compute_perplexity(model, prompts, max_samples)
    
    # Get hyperbolic distances
    with torch.no_grad():
        emb = mapper(acts[:max_samples].to(device).float())
        distances = mapper.radius(emb).cpu().numpy()
    
    # Compute correlation
    corr, pval = pearsonr(distances, perplexities)
    
    print(f"  Perplexity range: [{min(perplexities):.1f}, {max(perplexities):.1f}]")
    print(f"  Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
    print(f"  Correlation: r = {corr:.3f}, p = {pval:.4f}")
    
    if abs(corr) > 0.7:
        interpretation = "❌ HIGH CORRELATION: Distance is likely just measuring perplexity/uncertainty"
    elif abs(corr) > 0.3:
        interpretation = "⚠️ MODERATE: Distance partially reflects perplexity (~" + f"{corr**2*100:.0f}% variance explained)"
    else:
        interpretation = "✅ LOW: Distance captures something beyond perplexity"
    
    print(f"  {interpretation}")
    
    return corr, interpretation

def train_mapper(acts, metas, epochs=300, lr=0.01, shuffle_labels=False, use_layer_norm=True):
    """
    Train hyperbolic mapper to predict logical depth from activations.
    
    Uses Euclidean radius for training (stable gradients), then
    hyperbolic distance for evaluation.
    
    Args:
        use_layer_norm: If True (default), uses LayerNorm. Set False for ablation.
    """
    true_idxs = [i for i, m in enumerate(metas) if m['label']=='TRUE']
    
    if len(true_idxs) < 10:
        print(f"⚠️  WARNING: Only {len(true_idxs)} TRUE samples found!")
    
    X = acts[true_idxs].to(device).float()
    
    if shuffle_labels:
        fixed_depths = [m['depth'] for m in np.array(metas)[true_idxs]]
        random.shuffle(fixed_depths)
        depths = torch.tensor(fixed_depths, device=device).float()
    else:
        depths = torch.tensor([m['depth'] for m in np.array(metas)[true_idxs]], device=device).float()
    
    depth_min, depth_max = depths.min().item(), depths.max().item()
    print(f"  📊 Training on {len(true_idxs)} TRUE samples, depths range: [{depth_min:.0f}, {depth_max:.0f}]")
    
    # Adaptive target scaling: map depths to Euclidean radii [0.1, 0.4]
    # (Euclidean radius is stable, hyperbolic distance is for evaluation)
    alpha = 0.075  # Scaling factor
    targets = alpha * depths  # depth=1 → 0.075, depth=5 → 0.375
    
    target_min, target_max = targets.min().item(), targets.max().item()
    print(f"  🎯 Targets (α={alpha}): [{target_min:.3f}, {target_max:.3f}]")
        
    mapper = HyperbolicMapper(X.shape[1], 16, use_layer_norm=use_layer_norm).to(device)

    
    # Use standard Adam (more stable than RiemannianAdam for this setup)
    optimizer = optim.Adam(mapper.parameters(), lr=lr)
    
    initial_loss = None
    final_loss = None
    
    for e in range(epochs):
        optimizer.zero_grad()
        emb = mapper(X)
        # Train on EUCLIDEAN RADIUS (stable gradients)
        # Hyperbolic distance is used for evaluation only
        d_origin = mapper.radius(emb)
        loss = torch.mean((d_origin - targets)**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), max_norm=1.0)
        optimizer.step()
        
        if e == 0:
            initial_loss = loss.item()
        if e == epochs - 1:
            final_loss = loss.item()
            
        if e % 100 == 0:
            dist_mean = d_origin.mean().item()
            dist_std = d_origin.std().item()
            print(f"  Epoch {e}: Loss = {loss.item():.6f}, Dist: mean={dist_mean:.3f}, std={dist_std:.3f}")
            
            # VALIDATION 4: Check for saturation (all same radius)
            if dist_std < 0.001 and e > 50:
                print(f"  ⚠️  WARNING: Distance std={dist_std:.6f} is too low! Possible saturation.")
                
            # VALIDATION 5: Check for boundary saturation
            if dist_mean > 5.0:
                print(f"  ⚠️  WARNING: Mean distance={dist_mean:.2f} is very high! Likely boundary saturation.")
    
    # VALIDATION 6: Check training actually improved
    if initial_loss is not None and final_loss is not None:
        improvement = (initial_loss - final_loss) / initial_loss * 100
        if improvement < 10:
            print(f"  ⚠️  WARNING: Loss only improved by {improvement:.1f}%. Training may have failed.")
        else:
            print(f"  ✅ Training complete: Loss improved by {improvement:.1f}% ({initial_loss:.4f} → {final_loss:.4f})")
            
    return mapper

# ==========================================
# 3. Inference
# ==========================================
def get_activations(model, data, probe_after_reasoning=False):
    """
    Extract activations from model.
    
    Args:
        probe_after_reasoning: If True (for DeepSeek), generate output and probe
                               AFTER the </think> token to capture reasoned state.
    """
    acts = []
    metas = []
    
    # Model-Specific Layer Selection from CONFIG
    model_key = 'deepseek' if probe_after_reasoning else 'qwen'
    layer_idx = CONFIG[model_key]['layer']
    reasoning_tokens = CONFIG[model_key]['reasoning_tokens']
    
    print(f"Probing Layer {layer_idx} on {len(data)} samples (from CONFIG['{model_key}'])")
    print(f"  Probe after reasoning: {probe_after_reasoning}")
    if probe_after_reasoning:
        print(f"  Reasoning tokens: {reasoning_tokens}")
    
    # Logging for thinking token probe
    thinking_token_counts = []
    fallback_count = 0
    
    for item in tqdm(data, disable=not sys.stdout.isatty()):
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{item['context']}\n{item['query']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        with torch.no_grad():
            if probe_after_reasoning:
                # FOR REASONING MODELS: Use MI Peaks methodology
                # Probe at ALL thinking tokens and mean pool
                extraction = extract_thinking_activations(
                    model, prompt, layer_idx, reasoning_tokens
                )
                
                thinking_acts = extraction['thinking_acts']
                thinking_count = len(extraction['thinking_positions'])
                thinking_token_counts.append(thinking_count)
                
                # Mean pool thinking token activations for static analysis
                if thinking_count > 0:
                    act = thinking_acts.mean(dim=0)
                else:
                    act = extraction['all_acts'][-1]
                    fallback_count += 1
                
                # Store thinking info in metadata
                item = {**item, 
                        'thinking_count': thinking_count,
                        'thinking_density': extraction['thinking_metrics']['thinking_density'],
                        'think_end_found': extraction['think_end_pos'] > 0}
            else:
                # FOR STANDARD MODELS: Probe at end of prompt
                _, cache = model.run_with_cache(prompt, 
                                                names_filter=lambda x: x.endswith("hook_resid_post"))
                act = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1, :].cpu()
                del cache; torch.cuda.empty_cache()
            
            acts.append(act)
            metas.append(item)
    
    # Log thinking token probe stats
    if probe_after_reasoning:
        total = len(thinking_token_counts)
        avg_thinking = sum(thinking_token_counts) / total if total > 0 else 0
        print(f"\n🧠 Thinking Token Probe Stats (MI Peaks):")
        print(f"  Avg thinking tokens per sample: {avg_thinking:.1f}")
        print(f"  Samples with 0 thinking tokens (fallback): {fallback_count}/{total} ({100*fallback_count/total:.1f}%)")
        if thinking_token_counts:
            print(f"  Min/Max thinking tokens: {min(thinking_token_counts)}/{max(thinking_token_counts)}")
    
    return torch.stack(acts), metas

# ==========================================
# 3b. MULTI-GPU PARALLEL INFERENCE (SEQUENTIAL LOADING)
# ==========================================

def get_activations_parallel(data, model_type, probe_after_reasoning=False):
    """
    Extract activations using ALL available GPUs in parallel.
    
    Automatically:
    - Detects number of GPUs
    - Splits data evenly across GPUs
    - Loads models SEQUENTIALLY to avoid race conditions
    - Runs inference in parallel
    - Returns one model for reuse
    
    Falls back to single-GPU if only 1 GPU or issues arise.
    """
    if not USE_MULTI_GPU:
        print("⚠️  Single GPU detected, using standard inference")
        model = get_model(model_type=model_type)
        acts, metas = get_activations(model, data, probe_after_reasoning)
        return acts, metas, model
    
    print(f"\n🚀 MULTI-GPU PARALLEL INFERENCE")
    print(f"  Distributing {len(data)} samples across {N_GPUS} GPUs")
    print(f"  ~{len(data) // N_GPUS} samples per GPU")
    
    # Split data into chunks for each GPU
    chunk_size = len(data) // N_GPUS
    data_chunks = []
    for i in range(N_GPUS):
        start = i * chunk_size
        end = start + chunk_size if i < N_GPUS - 1 else len(data)
        data_chunks.append(data[start:end])
    
    # Clear all GPU memory before loading
    torch.cuda.empty_cache()
    for i in range(N_GPUS):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    
    # SEQUENTIAL model loading to avoid race conditions
    print("  Loading models SEQUENTIALLY to avoid race conditions...")
    models = {}
    for gpu_id in range(N_GPUS):
        print(f"  Loading model on GPU {gpu_id}...")
        try:
            models[gpu_id] = get_model(model_type=model_type, target_device=f"cuda:{gpu_id}")
            print(f"  ✓ GPU {gpu_id} model loaded")
        except Exception as e:
            print(f"  ⚠️ GPU {gpu_id} failed to load: {e}")
            models[gpu_id] = None
    
    # Check if we have at least one working GPU
    working_gpus = [gpu_id for gpu_id, m in models.items() if m is not None]
    if not working_gpus:
        print("⚠️  All GPU loads failed, falling back to single GPU")
        torch.cuda.empty_cache()
        model = get_model(model_type=model_type, target_device="cuda:0")
        acts, metas = get_activations(model, data, probe_after_reasoning)
        return acts, metas, model
    
    # Get config
    model_key = 'deepseek' if probe_after_reasoning else 'qwen'
    layer_idx = CONFIG[model_key]['layer']
    reasoning_tokens = CONFIG[model_key]['reasoning_tokens']
    
    # Worker function for inference on a single GPU
    def gpu_inference_worker(gpu_id, model, data_chunk, result_queue):
        try:
            acts = []
            metas = []
            think_found = 0
            think_not_found = 0
            
            for item in tqdm(data_chunk, desc=f"GPU {gpu_id}", position=gpu_id):
                prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{item['context']}\n{item['query']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                
                with torch.no_grad():
                    if probe_after_reasoning:
                        output = model.generate(prompt, max_new_tokens=reasoning_tokens, temperature=0.0)
                        output_str = output if isinstance(output, str) else model.tokenizer.decode(output[0])
                        _, cache = model.run_with_cache(output_str, 
                                                        names_filter=lambda x: x.endswith("hook_resid_post"))
                        
                        think_end_pos = -1
                        try:
                            if "</think>" in output_str:
                                output_tokens = model.to_str_tokens(output_str)
                                for i, tok in enumerate(output_tokens):
                                    if "</think>" in tok or "think>" in tok:
                                        think_end_pos = i
                                        break
                        except:
                            pass
                        
                        if think_end_pos > 0:
                            act = cache[f"blocks.{layer_idx}.hook_resid_post"][0, think_end_pos, :].cpu()
                            think_found += 1
                        else:
                            act = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1, :].cpu()
                            think_not_found += 1
                    else:
                        _, cache = model.run_with_cache(prompt, 
                                                        names_filter=lambda x: x.endswith("hook_resid_post"))
                        act = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1, :].cpu()
                    
                    acts.append(act)
                    metas.append(item)
                    del cache
                    torch.cuda.empty_cache()
            
            result_queue.put({
                'acts': torch.stack(acts) if acts else None,
                'metas': metas,
                'think_found': think_found,
                'think_not_found': think_not_found,
                'gpu_id': gpu_id
            })
        except Exception as e:
            print(f"⚠️  GPU {gpu_id} inference error: {e}")
            result_queue.put(None)
    
    # Run parallel inference (models already loaded)
    result_queue = Queue()
    threads = []
    
    for gpu_id in working_gpus:
        if gpu_id < len(data_chunks):
            t = threading.Thread(
                target=gpu_inference_worker, 
                args=(gpu_id, models[gpu_id], data_chunks[gpu_id], result_queue)
            )
            threads.append(t)
            t.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    # Keep GPU 0 model for subsequent operations, delete others
    model_to_return = models.get(0)
    for gpu_id, m in models.items():
        if gpu_id != 0 and m is not None:
            del m
    torch.cuda.empty_cache()
    
    # Collect results
    results = []
    while not result_queue.empty():
        r = result_queue.get()
        if r is not None and r['acts'] is not None:
            results.append(r)
    
    if not results:
        print("⚠️  Multi-GPU failed, falling back to single GPU")
        model = get_model(model_type=model_type)
        acts, metas = get_activations(model, data, probe_after_reasoning)
        return acts, metas, model
    
    # Sort by GPU ID to maintain order
    results.sort(key=lambda x: x['gpu_id'])
    
    # Combine results
    all_acts = torch.cat([r['acts'] for r in results], dim=0)
    all_metas = []
    total_think_found = 0
    total_think_not_found = 0
    
    for r in results:
        all_metas.extend(r['metas'])
        total_think_found += r['think_found']
        total_think_not_found += r['think_not_found']
    
    print(f"\n✅ Multi-GPU inference complete!")
    print(f"  Total samples processed: {len(all_metas)}")
    
    if probe_after_reasoning:
        total = total_think_found + total_think_not_found
        if total > 0:
            print(f"\n📊 Reasoning Probe Stats:")
            print(f"  </think> found: {total_think_found}/{total} ({100*total_think_found/total:.1f}%)")
            print(f"  </think> NOT found: {total_think_not_found}/{total}")
    
    return all_acts, all_metas, model_to_return


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 3 Robustness Experiments")
    parser.add_argument("--model", type=str, default="qwen", choices=["qwen", "deepseek"], 
                        help="Choose model: 'qwen' (Standard) or 'deepseek' (Reasoning)")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples (default: from CONFIG)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--multi-gpu", action="store_true", default=True, help="Enable multi-GPU parallel inference")
    parser.add_argument("--single-gpu", action="store_true", help="Force single-GPU mode")
    args = parser.parse_args()
    
    # Set global seed FIRST for reproducibility
    set_global_seed(args.seed)
    
    # Handle GPU mode
    use_parallel = args.multi_gpu and not args.single_gpu and USE_MULTI_GPU
    if args.single_gpu:
        print("⚠️  Single-GPU mode forced by --single-gpu flag")
    
    # Use CONFIG default if --samples not specified
    if args.samples is None:
        N_SAMPLES = CONFIG[args.model]['samples_default']
        print(f"📊 Using CONFIG default samples: {N_SAMPLES} (for {args.model})")
    else:
        N_SAMPLES = args.samples
        print(f"📊 Using specified samples: {N_SAMPLES}")
    
    prefix = args.model  # "qwen" or "deepseek"
    
    # Create output directory with prefix
    output_dir = f"results/phase3_corrected/{prefix}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate BALANCED datasets (reproducible with seed)
    print(f"\n--- Generating Balanced Datasets (seed={args.seed}) ---")
    
    print("\nGenerating fiction data...")
    gen_f = LogicTreeGenerator(mode='fiction', seed=args.seed)
    fiction_ds = generate_balanced_dataset(gen_f, N_SAMPLES, seed=args.seed)
    
    print("\nGenerating Animals (real English) data...")
    gen_r = LogicTreeGenerator(mode='animals', seed=args.seed + 1)  # Different seed offset
    real_ds = generate_balanced_dataset(gen_r, N_SAMPLES, seed=args.seed + 1)
    
    # Geography domain for cross-domain generalization test
    print("\nGenerating Geography data (for cross-domain test)...")
    gen_g = LogicTreeGenerator(mode='geography', seed=args.seed + 2)
    geo_ds = generate_balanced_dataset(gen_g, N_SAMPLES, seed=args.seed + 2)
    
    # ==========================================
    # DATA VALIDATION
    # ==========================================
    def validate_dataset(ds, name):
        """Validate dataset has proper distribution of types and depths."""
        types = [d['type'] for d in ds]
        labels = [d['label'] for d in ds]
        depths = [d['depth'] for d in ds if d['label'] == 'TRUE']
        
        n_true = types.count('true')
        n_hall = types.count('hallucination')
        n_unrel = types.count('unrelated')
        
        print(f"\n📊 {name} Dataset Validation:")
        print(f"   Total: {len(ds)} samples")
        print(f"   • TRUE: {n_true} ({100*n_true/len(ds):.1f}%)")
        print(f"   • Hallucination: {n_hall} ({100*n_hall/len(ds):.1f}%)")
        print(f"   • Unrelated: {n_unrel} ({100*n_unrel/len(ds):.1f}%)")
        
        if depths:
            print(f"   • Depth range: {min(depths)} to {max(depths)}")
            depth_1 = sum(1 for d in depths if d == 1)
            depth_5 = sum(1 for d in depths if d >= 4)
            print(f"   • 1-hop: {depth_1}, 5-hop: {depth_5}")
        
        # Warnings
        if n_true < 10:
            print(f"   ⚠️  WARNING: Too few TRUE samples ({n_true})!")
        if n_hall < 10:
            print(f"   ⚠️  WARNING: Too few Hallucination samples ({n_hall})!")
        if n_unrel < 5:
            print(f"   ⚠️  WARNING: Too few Unrelated samples ({n_unrel})!")
        if depths and max(depths) < 3:
            print(f"   ⚠️  WARNING: No deep (5-hop) samples found!")
            
    validate_dataset(fiction_ds, "Fiction")
    validate_dataset(real_ds, "Real English")
    
    # Model type
    model_type = "standard" if args.model == "qwen" else "reasoning"
    
    # CRITICAL: For DeepSeek, probe AFTER reasoning
    is_reasoning_model = args.model == "deepseek"
    print(f"\n🔬 Probe after reasoning: {is_reasoning_model}")
    
    # Get Activations - use parallel if multiple GPUs available
    if use_parallel:
        print(f"🚀 Using MULTI-GPU parallel inference ({N_GPUS} GPUs)")
        
        print("\nExtracting activations (Fiction)...")
        fiction_acts, fiction_meta, model = get_activations_parallel(
            fiction_ds, model_type=model_type, probe_after_reasoning=is_reasoning_model)
        
        print("\nExtracting activations (Real English)...")
        # For second dataset, pass data directly to get_activations using existing model
        real_acts, real_meta = get_activations(model, real_ds, probe_after_reasoning=is_reasoning_model)
    else:
        print("Using single-GPU inference")
        model = get_model(model_type=model_type)
        
        print("\nExtracting activations (Fiction)...")
        fiction_acts, fiction_meta = get_activations(model, fiction_ds, probe_after_reasoning=is_reasoning_model)
        
        print("\nExtracting activations (Real English)...")
        real_acts, real_meta = get_activations(model, real_ds, probe_after_reasoning=is_reasoning_model)
    
    # Train Mapper on Fiction Data
    print("\nTraining base mapper on fiction data...")
    base_mapper = train_mapper(fiction_acts, fiction_meta, epochs=300)
    
    # ==========================================
    # Run Phase 3 Experiments (with prefix)
    # ==========================================
    
    # Exp 3A: Real English
    print("\n=== Exp 3A: Real English Control ===")
    with torch.no_grad():
        emb = base_mapper(real_acts.to(device).float())
        dists_3a = base_mapper.dist(emb, torch.zeros_like(emb)).cpu().numpy()
        
    data_3a = []
    for i, m in enumerate(real_meta):
        if m['type'] == 'true':
            if m['depth'] == 1: lbl = "True (1 Hop)"
            elif m['depth'] >= 4: lbl = "True (5 Hop)"
            else: continue
        elif m['type'] == 'hallucination': lbl = "Hallucination"
        elif m['type'] == 'unrelated': lbl = "Unrelated"
        else: continue
        data_3a.append({'Hyperbolic Distance': dists_3a[i], 'Condition': lbl})
        
    df_3a = pd.DataFrame(data_3a)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.kdeplot(data=df_3a, x='Hyperbolic Distance', hue='Condition', fill=True, ax=axes[0],
                palette={'True (1 Hop)': 'blue', 'True (5 Hop)': 'orange', 'Hallucination': 'red', 'Unrelated': 'green'},
                warn_singular=False)
    axes[0].set_title(f"Exp 3A: Real English ({prefix}) - Density")
    order = ['True (1 Hop)', 'True (5 Hop)', 'Hallucination', 'Unrelated']
    sns.violinplot(data=df_3a, x='Condition', y='Hyperbolic Distance', hue='Condition', order=order, ax=axes[1],
                   palette={'True (1 Hop)': 'blue', 'True (5 Hop)': 'orange', 'Hallucination': 'red', 'Unrelated': 'green'},
                   legend=False)
    axes[1].set_title(f"Exp 3A: Real English ({prefix}) - Distribution")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_exp3a_real_english.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/{prefix}_exp3a_real_english.png")
    for cond in order:
        subset = df_3a[df_3a['Condition'] == cond]['Hyperbolic Distance']
        if len(subset) > 0: print(f"  {cond}: Mean = {subset.mean():.2f}, Std = {subset.std():.2f}, N = {len(subset)}")
    
    # Exp 3B: Unrelated Baseline
    print("\n=== Exp 3B: Unrelated Baseline ===")
    with torch.no_grad():
        emb = base_mapper(fiction_acts.to(device).float())
        dists_3b = base_mapper.dist(emb, torch.zeros_like(emb)).cpu().numpy()
        
    data_3b = []
    for i, m in enumerate(fiction_meta):
        if m['type'] == 'true':
            if m['depth'] == 1: lbl = "True (1 Hop)"
            elif m['depth'] >= 4: lbl = "True (5 Hop)"
            else: continue
        elif m['type'] == 'hallucination': lbl = "Hallucination"
        elif m['type'] == 'unrelated': lbl = "Unrelated"
        else: continue
        data_3b.append({'Hyperbolic Distance': dists_3b[i], 'Condition': lbl})
        
    df_3b = pd.DataFrame(data_3b)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.kdeplot(data=df_3b, x='Hyperbolic Distance', hue='Condition', fill=True, ax=axes[0],
                palette={'True (1 Hop)': 'blue', 'True (5 Hop)': 'orange', 'Hallucination': 'red', 'Unrelated': 'green'},
                warn_singular=False)
    axes[0].set_title(f"Exp 3B: Unrelated Baseline ({prefix}) - Density")
    sns.violinplot(data=df_3b, x='Condition', y='Hyperbolic Distance', hue='Condition', order=order, ax=axes[1],
                   palette={'True (1 Hop)': 'blue', 'True (5 Hop)': 'orange', 'Hallucination': 'red', 'Unrelated': 'green'},
                   legend=False)
    axes[1].set_title(f"Exp 3B: Unrelated Baseline ({prefix}) - Distribution")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_exp3b_unrelated_baseline.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/{prefix}_exp3b_unrelated_baseline.png")
    for cond in order:
        subset = df_3b[df_3b['Condition'] == cond]['Hyperbolic Distance']
        if len(subset) > 0: print(f"  {cond}: Mean = {subset.mean():.2f}, Std = {subset.std():.2f}, N = {len(subset)}")
    
    # Exp 3C: Trained Probe
    print("\n=== Exp 3C: Trained Probe ===")
    print("Training fresh probe on fiction data...")
    trained_mapper = train_mapper(fiction_acts, fiction_meta, epochs=300)
    with torch.no_grad():
        emb = trained_mapper(fiction_acts.to(device).float())
        dists_3c = trained_mapper.dist(emb, torch.zeros_like(emb)).cpu().numpy()
        
    data_3c = []
    for i, m in enumerate(fiction_meta):
        if m['type'] == 'true':
            if m['depth'] == 1: lbl = "True (1 Hop)"
            elif m['depth'] >= 4: lbl = "True (5 Hop)"
            else: continue
        elif m['type'] == 'hallucination': lbl = "Hallucination"
        elif m['type'] == 'unrelated': lbl = "Unrelated"
        else: continue
        data_3c.append({'Hyperbolic Distance': dists_3c[i], 'Condition': lbl})
        
    df_3c = pd.DataFrame(data_3c)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.kdeplot(data=df_3c, x='Hyperbolic Distance', hue='Condition', fill=True, ax=axes[0],
                palette={'True (1 Hop)': 'blue', 'True (5 Hop)': 'orange', 'Hallucination': 'red', 'Unrelated': 'green'},
                warn_singular=False)
    axes[0].set_title(f"Exp 3C: Trained Probe ({prefix}) - Density")
    sns.violinplot(data=df_3c, x='Condition', y='Hyperbolic Distance', hue='Condition', order=order, ax=axes[1],
                   palette={'True (1 Hop)': 'blue', 'True (5 Hop)': 'orange', 'Hallucination': 'red', 'Unrelated': 'green'},
                   legend=False)
    axes[1].set_title(f"Exp 3C: Trained Probe ({prefix}) - Distribution")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_exp3c_trained_probe.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/{prefix}_exp3c_trained_probe.png")
    for cond in order:
        subset = df_3c[df_3c['Condition'] == cond]['Hyperbolic Distance']
        if len(subset) > 0: print(f"  {cond}: Mean = {subset.mean():.2f}, Std = {subset.std():.2f}, N = {len(subset)}")
    
    # ==========================================
    # Exp 3D: TRUE Cross-Domain Generalization
    # Train on Animals → Test on Geography
    # ==========================================
    print("\n=== Exp 3D: Cross-Domain Generalization (Animals → Geography) ===")
    print("  This tests if hierarchy encoding generalizes across domains!")
    
    # Get Geography activations - use existing model (already loaded)
    print("\nExtracting activations (Geography)...")
    geo_acts, geo_meta = get_activations(model, geo_ds, probe_after_reasoning=is_reasoning_model)
    
    # Train mapper on Animals (real) data
    print("\nTraining mapper on Animals domain...")
    animals_mapper = train_mapper(real_acts, real_meta, epochs=300)
    
    # Evaluate on Geography (completely different domain!)
    with torch.no_grad():
        emb = animals_mapper(geo_acts.to(device).float())
        dists_3d = animals_mapper.dist(emb, torch.zeros_like(emb)).cpu().numpy()
        
    data_3d = []
    for i, m in enumerate(geo_meta):
        if m['type'] == 'true':
            if m['depth'] == 1: lbl = "True (1 Hop)"
            elif m['depth'] >= 4: lbl = "True (5 Hop)"
            else: continue
        elif m['type'] == 'hallucination': lbl = "Hallucination"
        elif m['type'] == 'unrelated': lbl = "Unrelated"
        else: continue
        data_3d.append({'Hyperbolic Distance': dists_3d[i], 'Condition': lbl})
        
    df_3d = pd.DataFrame(data_3d)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.kdeplot(data=df_3d, x='Hyperbolic Distance', hue='Condition', fill=True, ax=axes[0],
                palette={'True (1 Hop)': 'blue', 'True (5 Hop)': 'orange', 'Hallucination': 'red', 'Unrelated': 'green'},
                warn_singular=False)
    axes[0].set_title(f"Exp 3D: Cross-Domain (Train:Animals→Test:Geography) ({prefix})")
    sns.violinplot(data=df_3d, x='Condition', y='Hyperbolic Distance', hue='Condition', order=order, ax=axes[1],
                   palette={'True (1 Hop)': 'blue', 'True (5 Hop)': 'orange', 'Hallucination': 'red', 'Unrelated': 'green'},
                   legend=False)
    axes[1].set_title(f"Exp 3D: Cross-Domain Generalization ({prefix})")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_exp3d_cross_domain.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/{prefix}_exp3d_cross_domain.png")
    for cond in order:
        subset = df_3d[df_3d['Condition'] == cond]['Hyperbolic Distance']
        if len(subset) > 0: print(f"  {cond}: Mean = {subset.mean():.2f}, Std = {subset.std():.2f}, N = {len(subset)}")
    
    # ==========================================
    # NEW EXPERIMENTS: Addressing Methodology Critique
    # ==========================================
    
    print("\n" + "="*60)
    print("PHASE 4: METHODOLOGY VALIDATION EXPERIMENTS")
    print("="*60)
    print("These experiments address the circular training logic critique.")
    
    # ==========================================
    # Exp 4A: TRUE vs HALLUCINATION Classification
    # ==========================================
    print("\n=== Exp 4A: TRUE vs HALLUCINATION Classification ===")
    print("  Training classifier on BOTH classes (not just TRUE)...")
    print("  This is the proper test of geometric separability.\n")
    
    classifier_ln, metrics_ln = train_classifier(fiction_acts, fiction_meta, 
                                                   epochs=300, use_layer_norm=True)
    
    # Save classification results
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['Accuracy', 'AUROC'], [metrics_ln['accuracy'], metrics_ln['auroc']], 
                   color=['steelblue', 'darkorange'])
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random Baseline')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title(f"Exp 4A: TRUE vs HALLUCINATION Classification ({prefix})")
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2%}', ha='center', va='bottom', fontsize=12)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_exp4a_classification.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/{prefix}_exp4a_classification.png")
    
    # Interpret results
    if metrics_ln['accuracy'] > 0.8:
        print("  🎉 HIGH ACCURACY: Geometries ARE distinct! Hypothesis supported.")
    elif metrics_ln['accuracy'] > 0.6:
        print("  ⚠️ MODERATE ACCURACY: Some geometric separation exists.")
    else:
        print("  ❌ LOW ACCURACY: Geometries are NOT distinct. Hypothesis may fail.")
    
    # ==========================================
    # Exp 4B: LayerNorm Ablation Study  
    # ==========================================
    print("\n=== Exp 4B: LayerNorm Ablation Study ===")
    print("  Testing if magnitude information (removed by LayerNorm) is important...")
    
    # Train without LayerNorm
    classifier_no_ln, metrics_no_ln = train_classifier(fiction_acts, fiction_meta,
                                                         epochs=300, use_layer_norm=False)
    
    # Train mapper without LayerNorm for comparison
    print("\n  Training depth mapper without LayerNorm...")
    mapper_no_ln = train_mapper(fiction_acts, fiction_meta, epochs=300, use_layer_norm=False)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Classification comparison
    x = ['With LayerNorm', 'Without LayerNorm']
    acc = [metrics_ln['accuracy'], metrics_no_ln['accuracy']]
    auroc = [metrics_ln['auroc'], metrics_no_ln['auroc']]
    
    x_pos = np.arange(len(x))
    width = 0.35
    
    axes[0].bar(x_pos - width/2, acc, width, label='Accuracy', color='steelblue')
    axes[0].bar(x_pos + width/2, auroc, width, label='AUROC', color='darkorange')
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(x)
    axes[0].set_ylabel('Score')
    axes[0].set_title(f"Exp 4B: LayerNorm Ablation - Classification ({prefix})")
    axes[0].legend()
    axes[0].set_ylim(0, 1.0)
    
    # Depth prediction comparison (using mapper)
    with torch.no_grad():
        emb_ln = base_mapper(fiction_acts.to(device).float())
        dists_ln = base_mapper.radius(emb_ln).cpu().numpy()
        
        emb_no_ln = mapper_no_ln(fiction_acts.to(device).float())
        dists_no_ln = mapper_no_ln.radius(emb_no_ln).cpu().numpy()
    
    true_depths = [m['depth'] for m in fiction_meta if m['label'] == 'TRUE']
    true_idxs = [i for i, m in enumerate(fiction_meta) if m['label'] == 'TRUE']
    
    axes[1].scatter(true_depths, dists_ln[true_idxs], alpha=0.5, label='With LayerNorm')
    axes[1].scatter(true_depths, dists_no_ln[true_idxs], alpha=0.5, label='Without LayerNorm')
    axes[1].set_xlabel('True Depth')
    axes[1].set_ylabel('Hyperbolic Distance')
    axes[1].set_title(f"Exp 4B: LayerNorm Ablation - Depth Correlation ({prefix})")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_exp4b_layernorm_ablation.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/{prefix}_exp4b_layernorm_ablation.png")
    
    # Interpret
    acc_diff = metrics_no_ln['accuracy'] - metrics_ln['accuracy']
    if acc_diff > 0.05:
        print(f"  📈 WITHOUT LayerNorm is BETTER (+{acc_diff:.1%})")
        print("     → Magnitude information WAS being lost!")
    elif acc_diff < -0.05:
        print(f"  📉 WITH LayerNorm is BETTER ({-acc_diff:.1%})")
        print("     → LayerNorm helps stability without hurting performance")
    else:
        print(f"  ≈ Similar performance (diff={acc_diff:.1%})")
        print("     → LayerNorm doesn't significantly affect results")
    
    # ==========================================
    # Exp 4C: Perplexity Control
    # ==========================================
    print("\n=== Exp 4C: Perplexity Control ===")
    print("  Checking if hyperbolic distance is just measuring perplexity...")
    
    # Create prompts for perplexity
    fiction_prompts = [f"{d['context']}\n{d['query']}" for d in fiction_ds]
    
    # Run perplexity check
    corr, interpretation = perplexity_correlation_check(
        model, fiction_acts, fiction_meta, base_mapper, fiction_prompts, 
        max_samples=min(50, len(fiction_prompts))
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    bar = ax.bar(['Distance-Perplexity\nCorrelation'], [abs(corr)], 
                  color='coral' if abs(corr) > 0.7 else 'lightgreen')
    ax.axhline(y=0.9, color='red', linestyle='--', label='Danger Zone (r > 0.9)')
    ax.axhline(y=0.7, color='orange', linestyle='--', label='Warning Zone (r > 0.7)')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('|Pearson Correlation|')
    ax.set_title(f"Exp 4C: Perplexity Control ({prefix})")
    ax.legend()
    
    ax.text(bar[0].get_x() + bar[0].get_width()/2., bar[0].get_height() + 0.02,
            f'r = {corr:.3f}', ha='center', va='bottom', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_exp4c_perplexity_control.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/{prefix}_exp4c_perplexity_control.png")
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*60)
    print(f"ALL EXPERIMENTS COMPLETE ({prefix.upper()})")
    print("="*60)
    print(f"Results saved to: {output_dir}/")
    
    print("\n📊 PHASE 3 (Original Experiments):")
    print("  - Exp 3A: Same-domain test (Fiction→Animals)")
    print("  - Exp 3B: Unrelated baseline")
    print("  - Exp 3C: Fresh trained probe")
    print("  - Exp 3D: Cross-domain generalization")
    
    print("\n📊 PHASE 4 (Methodology Validation):")
    print(f"  - Exp 4A: Classification Accuracy = {metrics_ln['accuracy']:.1%}, AUROC = {metrics_ln['auroc']:.3f}")
    print(f"  - Exp 4B: LayerNorm Ablation: With LN = {metrics_ln['accuracy']:.1%}, Without LN = {metrics_no_ln['accuracy']:.1%}")
    print(f"  - Exp 4C: Perplexity Correlation: r = {corr:.3f}")
    
    print("\n📋 INTERPRETATION:")
    if metrics_ln['accuracy'] > 0.8 and abs(corr) < 0.7:
        print("  ✅ HYPOTHESIS SUPPORTED: Distinct geometries, not just perplexity")
    elif metrics_ln['accuracy'] > 0.6:
        print("  ⚠️ PARTIAL SUPPORT: Some separation, needs more investigation")
    else:
        print("  ❌ HYPOTHESIS CHALLENGED: Poor classification or high perplexity correlation")
    
    print("\nRun for other model:")
    if args.model == "qwen":
        print("  python run_phase3_corrected.py --model deepseek")
    else:
        print("  python run_phase3_corrected.py --model qwen")
