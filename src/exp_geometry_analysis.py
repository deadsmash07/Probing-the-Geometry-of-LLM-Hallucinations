"""
Geometrical Analysis
======================
This code runs experiemtns for complexity analysis, ID analysis and trajectory analysis

usage: python exp_geometry_analysis.py --model <model_name> --seed <seed>
"""

import os
import sys
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import geoopt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import random
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GENERATION_CACHE = {}

def get_cached_generation(model, prompt, max_tokens, temperature=0.0):
    """
    Generate text with caching to avoid duplicate generation.
    
    2x speedup when same prompt is used in get_activations and get_trajectory_metrics.
    
    Args:
        model: HookedTransformer model
        prompt: Input prompt string
        max_tokens: Max new tokens to generate
        temperature: Generation temperature
        
    Returns:
        output_str: Generated text string
    """
    # Create unique cache key
    cache_key = hash(prompt + str(max_tokens))
    
    if cache_key in GENERATION_CACHE:
        return GENERATION_CACHE[cache_key]
    
    # Generate and cache
    with torch.no_grad():
        output = model.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
        output_str = output if isinstance(output, str) else model.tokenizer.decode(output[0])
    
    GENERATION_CACHE[cache_key] = output_str
    return output_str

def clear_generation_cache():
    """Clear the generation cache (call between different experiments)."""
    global GENERATION_CACHE
    GENERATION_CACHE = {}
    torch.cuda.empty_cache()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")
os.makedirs("results/plots", exist_ok=True)

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
        'samples_default': 300,   # Increased for statistical significance
        'reasoning_tokens': 0,    # No reasoning for standard models
        'probe_after_reasoning': False,
    },
    'deepseek': {
        'layer': 23,              # Optimal layer for DeepSeek
        'samples_default': 300,   # ~100 TRUE samples needed for stable training
        'reasoning_tokens': 512,  # Tokens to generate for reasoning probe
        'probe_after_reasoning': True,
    }
}

# Auth - Load HF_TOKEN from environment
if not os.environ.get("HF_TOKEN"):
    print("Warning: HF_TOKEN not found in environment. Please set it in .env file.")

# ==========================================
# REPRODUCIBILITY: Global Seed
# ==========================================
def set_global_seed(seed=42):
    """
    Ensure reproducibility across all random sources.
    
    Academic standard: All experiments should be reproducible with same seed.
    """
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
    
    Academic Standard: Equal representation prevents bias in metrics.
    
    Target: 1/3 True, 1/3 Hallucination, 1/3 Unrelated (equal representation)
    Depth: Stratified sampling to ensure equal 1-hop and 5-hop representation
    
    Args:
        generator: LogicTreeGenerator instance
        n_samples: Total samples desired
        seed: Random seed for reproducibility
    
    Returns:
        Balanced dataset list
    """
    rng = random.Random(seed)
    
    # Target: Equal thirds (most academically rigorous)
    n_per_class = n_samples // 3
    
    # Collect samples by type AND depth
    collected = {
        'true': {'shallow': [], 'deep': []},  # shallow: depth 1-2, deep: depth 3+
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
                target_per_bucket = n_per_class // 2
                if len(collected['true'][bucket]) < target_per_bucket:
                    collected['true'][bucket].append(sample)
            elif t == 'hallucination':
                if len(collected['hallucination']) < n_per_class:
                    collected['hallucination'].append(sample)
            elif t == 'unrelated':
                if len(collected['unrelated']) < n_per_class:
                    collected['unrelated'].append(sample)
        
        # Check if all targets met
        true_count = len(collected['true']['shallow']) + len(collected['true']['deep'])
        if (true_count >= n_per_class and 
            len(collected['hallucination']) >= n_per_class and
            len(collected['unrelated']) >= n_per_class):
            break
        attempts += 1
    
    # Combine: balance true samples by depth
    n_shallow = n_per_class // 2
    n_deep = n_per_class - n_shallow
    
    dataset = []
    dataset.extend(collected['true']['shallow'][:n_shallow])
    dataset.extend(collected['true']['deep'][:n_deep])
    dataset.extend(collected['hallucination'][:n_per_class])
    dataset.extend(collected['unrelated'][:n_per_class])
    
    # Shuffle deterministically
    rng.shuffle(dataset)
    
    # Report balance
    actual_true = len([d for d in dataset if d['type'] == 'true'])
    actual_hall = len([d for d in dataset if d['type'] == 'hallucination'])
    actual_unrel = len([d for d in dataset if d['type'] == 'unrelated'])
    true_depths = [d['depth'] for d in dataset if d['type'] == 'true']
    
    print(f"📊 Balanced Dataset Generated (seed={seed}):")
    print(f"   Total: {len(dataset)} samples")
    print(f"   • True: {actual_true} ({100*actual_true/len(dataset):.1f}%)")
    print(f"   • Hallucination: {actual_hall} ({100*actual_hall/len(dataset):.1f}%)")
    print(f"   • Unrelated: {actual_unrel} ({100*actual_unrel/len(dataset):.1f}%)")
    if true_depths:
        shallow = sum(1 for d in true_depths if d <= 2)
        deep = sum(1 for d in true_depths if d >= 3)
        print(f"   • True depths: {shallow} shallow (1-2 hop), {deep} deep (3+ hop)")
    
    return dataset

# ==========================================
# 1. Data Generation
# ==========================================
class LogicTreeGenerator:
    def __init__(self, mode='fiction', depth=5, seed=42):
        self.mode = mode
        self.depth = depth
        self.rng = random.Random(seed)
        if mode == 'fiction':
            self.nouns = ["wumpus", "fele", "lorpus", "grumpus", "tumpus", "zumpus", "yumpus", "brompus", "numpus", "umpus"]
        else:
            self.nouns = ["animal", "mammal", "vertebrate", "carnivore", "feline", "canine", "reptile", "cat", "dog", "snake", "lizard", "poodle", "tabby", "cobra"]
            
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
        subject = "Max" if self.mode=='fiction' else "Spot"
        
        edges = list(G.edges()); self.rng.shuffle(edges)
        ctx = " ".join([f"Every {u} is a {v}." for u, v in edges]) + f" {subject} is a {start}."
        
        samples = []
        # True
        true_types = set(nx.descendants(G, start)) | {start}
        anc = list(true_types - {start})
        if anc:
            tgt = self.rng.choice(anc)
            hops = nx.shortest_path_length(G, start, tgt)
            samples.append({"context": ctx, "query": f"Is {subject} a {tgt}?", "label": "TRUE", "depth": hops, "type": "true"})
            
        # Hallucination
        false_t = list(set(G.nodes) - true_types)
        if false_t:
            tgt = self.rng.choice(false_t)
            samples.append({"context": ctx, "query": f"Is {subject} a {tgt}?", "label": "FALSE", "depth": -1, "type": "hallucination"})
            
        # Unrelated
        strangers = [n for n in self.nouns if n not in G.nodes]
        if strangers:
            tgt = self.rng.choice(strangers)
            samples.append({"context": ctx, "query": f"Is {subject} a {tgt}?", "label": "FALSE", "depth": -2, "type": "unrelated"})
        return samples

# ==========================================
# 2. Model & Inference
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
    
    # Toggle between Standard and Reasoning Models
    if model_type == "standard":
        print(f"Loading Standard Model: Qwen/Qwen2.5-7B-Instruct on {target_device}")
        return HookedTransformer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device=target_device)
    elif model_type == "reasoning":
        print(f"Loading Reasoning Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B on {target_device}")
        # WORKAROUND: TransformerLens doesn't officially support this name yet.
        # But it IS a Qwen2.5 model. We load it as a raw HF model, then 
        # tell TransformerLens to treat it as "Qwen/Qwen2.5-7B-Instruct".
        
        # IMPORTANT: Load HF model to CPU first to avoid double GPU memory allocation
        # HookedTransformer.from_pretrained will copy weights to target_device
        print("...downloading via Transformers (AutoModel) to CPU...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
            torch_dtype=torch.float16,  # Always use FP16 for memory efficiency
            trust_remote_code=True,
            device_map="cpu",  # Load to CPU first to avoid OOM when wrapping
        )
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        
        print(f"...wrapping in HookedTransformer on {target_device}...")
        hooked_model = HookedTransformer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", 
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=target_device,
            fold_ln=False,
            fold_value_biases=False,   # Disable to avoid tensor issues
            center_writing_weights=False,
            center_unembed=False
        )
        
        # Free CPU memory from original HF model
        del hf_model
        torch.cuda.empty_cache()
        
        return hooked_model

def get_activations(model, data, probe_after_reasoning=False):
    """
    Extract activations from model for each data sample.
    
    Args:
        model: HookedTransformer model
        data: List of data samples with 'context' and 'query'
        probe_after_reasoning: If True (for DeepSeek), generate output and probe
                               AFTER the </think> token to capture reasoned state.
                               If False (for Qwen), probe at end of prompt.
    
    CRITICAL: For reasoning models like DeepSeek-R1, probing at end of prompt
    captures the "pre-reasoning" state. We must probe AFTER the model has
    generated its <think>...</think> chain.
    """
    acts = []
    metas = []
    print(f"Inference on {len(data)} samples...")
    print(f"  In-Process Reasoning: {probe_after_reasoning}")
    
    # Model-Specific Layer Selection from CONFIG
    model_key = 'deepseek' if probe_after_reasoning else 'qwen'
    layer_idx = CONFIG[model_key]['layer']
    reasoning_tokens = CONFIG[model_key]['reasoning_tokens']
    print(f"  Probing Layer {layer_idx} (from CONFIG['{model_key}'])")
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
                    act = thinking_acts.mean(dim=0)  # Mean pool
                else:
                    act = extraction['all_acts'][-1]  # Fallback to last token
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
        print(f"  Min/Max thinking tokens: {min(thinking_token_counts)}/{max(thinking_token_counts)}")
    
    return torch.stack(acts), metas


# ==========================================
# 2a-2. THINKING TOKEN DETECTION (MI Peaks Alignment)
# ==========================================
# Based on: "Demystifying Reasoning Dynamics with Mutual Information"
# Key insight: MI peaks occur at "thinking tokens" like "Hmm", "Wait", "So"
# These are the critical moments where reasoning happens

THINKING_TOKENS = [
    # Reflective/pause tokens (highest MI in paper)
    "wait", "hmm", "hm", "okay", "ok", "well",
    # Transition tokens  
    "so", "therefore", "thus", "hence", "now",
    # Self-correction tokens
    "actually", "but", "however", "although",
    # Planning tokens
    "let me", "first", "next", "then", "finally",
    # Conclusion tokens
    "means", "implies", "shows", "proves"
]

# OPTIMIZATION: Pre-compile regex for faster matching
THINKING_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(t) for t in THINKING_TOKENS) + r')\b', 
    re.IGNORECASE
)

def find_thinking_positions(output_tokens, return_tokens=False):
    """
    Identify thinking token positions (analogous to MI peaks).
    
    Based on MI Peaks paper: These tokens show sudden, significant increases
    in mutual information with the correct answer.
    
    OPTIMIZATION: Uses pre-compiled regex instead of nested loops.
    
    Args:
        output_tokens: List of token strings
        return_tokens: If True, also return the actual tokens found
        
    Returns:
        positions: List of indices where thinking tokens occur
        (optionally) tokens: List of thinking tokens found
    """
    positions = []
    found_tokens = []
    
    for i, tok in enumerate(output_tokens):
        if THINKING_PATTERN.search(tok):
            positions.append(i)
            found_tokens.append(tok)
    
    if return_tokens:
        return positions, found_tokens
    return positions


def compute_thinking_metrics(output_tokens):
    """
    Compute thinking token metrics (MI Peaks alignment).
    
    Paper finding: LRMs have ~0.5-5% thinking token ratio.
    Hypothesis: TRUE samples should have higher thinking density.
    
    Returns:
        dict with:
        - density: n_thinking / n_total
        - count: absolute number of thinking tokens
        - positions: list of positions
        - mean_interval: average gap between thinking tokens
        - interval_variance: variance in gaps (low = regular, high = irregular)
    """
    positions = find_thinking_positions(output_tokens)
    n_total = len(output_tokens)
    n_thinking = len(positions)
    
    # Density (paper: ~0.5-5% for LRMs)
    density = n_thinking / n_total if n_total > 0 else 0
    
    # Interval analysis (paper: non-uniform distribution)
    if len(positions) >= 2:
        intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        mean_interval = sum(intervals) / len(intervals)
        interval_variance = sum((x - mean_interval)**2 for x in intervals) / len(intervals)
    else:
        mean_interval = n_total  # No peaks = max interval
        interval_variance = 0
    
    return {
        'thinking_density': density,
        'thinking_count': n_thinking,
        'thinking_positions': positions,
        'mean_interval': mean_interval,
        'interval_variance': interval_variance
    }


def extract_thinking_activations(model, prompt, layer_idx, max_tokens=None):
    """
    Extract activations at thinking token positions (MI Peaks methodology).
    
    Based on paper: "Demystifying Reasoning Dynamics with Mutual Information"
    Key insight: Probe ONLY at thinking tokens where MI peaks occur.
    
    Stops at min(max_tokens, </think> position).
    
    Args:
        model: HookedTransformer model
        prompt: Input prompt string
        layer_idx: Layer to extract from
        max_tokens: Max tokens to generate (from CONFIG)
        
    Returns:
        dict with:
        - output_str: Full generated text
        - all_acts: All token activations [N, Hidden]
        - thinking_acts: Activations at thinking positions only [M, Hidden]
        - thinking_positions: List of thinking token indices
        - thinking_metrics: Dict with density, count, intervals
        - think_end_pos: Position of </think> if found, else -1
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
    
    # Determine effective token range: stop at </think> if found
    if think_end_pos > 0:
        effective_tokens = new_tokens[:think_end_pos]
    else:
        effective_tokens = new_tokens
    
    # Find thinking token positions within effective range
    thinking_positions = find_thinking_positions(effective_tokens)
    thinking_metrics = compute_thinking_metrics(effective_tokens)
    
    # Run model to get activations
    with torch.no_grad():
        _, cache = model.run_with_cache(output_str, 
                                        names_filter=lambda x: x.endswith("hook_resid_post"))
        
        full_acts = cache[f"blocks.{layer_idx}.hook_resid_post"][0]  # [Seq, Hidden]
        
        # Slice to get only generated tokens (after prompt)
        if full_acts.shape[0] > prompt_len:
            gen_acts = full_acts[prompt_len:]  # All generated token activations
            
            # Apply </think> cutoff
            if think_end_pos > 0:
                gen_acts = gen_acts[:think_end_pos]
            
            # Extract thinking-only activations
            if len(thinking_positions) > 0:
                valid_positions = [p for p in thinking_positions if p < gen_acts.shape[0]]
                if len(valid_positions) > 0:
                    thinking_acts = gen_acts[valid_positions]
                else:
                    thinking_acts = gen_acts[-1:] # Fallback to last token
            else:
                thinking_acts = gen_acts[-1:]  # Fallback to last token
        else:
            gen_acts = full_acts[-1:]
            thinking_acts = full_acts[-1:]
        
        del cache
        torch.cuda.empty_cache()
    
    return {
        'output_str': output_str,
        'all_acts': gen_acts.cpu(),
        'thinking_acts': thinking_acts.cpu(),
        'thinking_positions': thinking_positions,
        'thinking_metrics': thinking_metrics,
        'think_end_pos': think_end_pos,
        'n_effective_tokens': len(effective_tokens)
    }


# ==========================================
# 2a-3. TRAJECTORY ANALYSIS (for Reasoning Models)
# ==========================================

def get_trajectory_metrics(model, data, mapper, layer_idx=None, max_tokens=None):
    """
    Analyze the TRAJECTORY of reasoning tokens through hyperbolic space.
    
    Instead of probing a single snapshot (e.g., </think>), this captures:
    - How far the representation travels (arc length)
    - Net displacement (does it actually go somewhere?)
    - Tortuosity (efficiency of thought)
    
    Hypothesis:
    - TRUE (deep): High arc length (deep thought requires movement)
    - HALLUCINATION: Low arc length ("short-circuit" - lazy jump)
    
    Args:
        model: HookedTransformer model
        data: List of samples with 'context', 'query', 'type', 'depth'
        mapper: Trained HyperbolicMapper
        layer_idx: Layer to probe (default: from CONFIG)
        max_tokens: Max reasoning tokens to generate
        
    Returns:
        pd.DataFrame with metrics per sample
    """
    import pandas as pd
    
    if layer_idx is None:
        layer_idx = CONFIG['deepseek']['layer']
    
    # Use CONFIG if max_tokens not specified
    if max_tokens is None:
        max_tokens = CONFIG['deepseek']['reasoning_tokens']
    
    print(f"\n🔬 TRAJECTORY ANALYSIS (Layer {layer_idx}, {max_tokens} tokens)")
    print(f"  Analyzing {len(data)} samples...")
    print(f"  🧠 MI Peaks alignment: detecting thinking tokens")
    
    metrics = []
    
    for item in tqdm(data, disable=not sys.stdout.isatty()):
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{item['context']}\n{item['query']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Use unified extraction function
        extraction = extract_thinking_activations(model, prompt, layer_idx, max_tokens)
        
        all_acts = extraction['all_acts'].to(device).float()
        thinking_acts = extraction['thinking_acts'].to(device).float()
        thinking_positions = extraction['thinking_positions']
        thinking_metrics = extraction['thinking_metrics']
        n_tokens = extraction['n_effective_tokens']
        n_thinking = len(thinking_positions)
        
        # Skip if no tokens generated
        if all_acts.shape[0] == 0:
            metrics.append({
                'type': item['type'],
                'depth': item.get('depth', -1),
                'label': item.get('label', 'UNKNOWN'),
                # All-token trajectory metrics
                'all_arc_length': 0.0,
                'all_displacement': 0.0,
                'all_tortuosity': 1.0,
                'all_mean_radius': 0.0,
                'n_tokens': 0,
                # Thinking-only trajectory metrics
                'think_arc_length': 0.0,
                'think_displacement': 0.0,
                'think_tortuosity': 1.0,
                'think_mean_radius': 0.0,
                'n_thinking': 0,
                # MI Peaks metrics
                'thinking_density': 0.0,
                'thinking_count': 0,
            })
            continue
        
        # ==========================================
        # TRAJECTORY 1: ALL TOKENS
        # ==========================================
        emb_all = mapper(all_acts)
        all_radii = mapper.radius(emb_all)
        all_mean_radius = all_radii.mean().item()
        
        if all_acts.shape[0] >= 2:
            all_steps = mapper.dist(emb_all[:-1], emb_all[1:])
            all_arc_length = all_steps.sum().item()
            all_displacement = mapper.dist(emb_all[0:1], emb_all[-1:]).item()
        else:
            all_arc_length = 0.0
            all_displacement = 0.0
        all_tortuosity = all_arc_length / (all_displacement + 1e-6)
        
        # ==========================================
        # TRAJECTORY 2: THINKING TOKENS ONLY (MI Peaks)
        # ==========================================
        if n_thinking >= 2:
            emb_think = mapper(thinking_acts)
            think_radii = mapper.radius(emb_think)
            think_mean_radius = think_radii.mean().item()
            
            think_steps = mapper.dist(emb_think[:-1], emb_think[1:])
            think_arc_length = think_steps.sum().item()
            think_displacement = mapper.dist(emb_think[0:1], emb_think[-1:]).item()
            think_tortuosity = think_arc_length / (think_displacement + 1e-6)
        elif n_thinking == 1:
            emb_think = mapper(thinking_acts)
            think_mean_radius = mapper.radius(emb_think).item()
            think_arc_length = 0.0
            think_displacement = 0.0
            think_tortuosity = 1.0
        else:
            think_mean_radius = all_mean_radius  # Fallback
            think_arc_length = 0.0
            think_displacement = 0.0
            think_tortuosity = 1.0
        
        metrics.append({
            'type': item['type'],
            'depth': item.get('depth', -1),
            'label': item.get('label', 'UNKNOWN'),
            # All-token trajectory metrics
            'all_arc_length': all_arc_length,
            'all_displacement': all_displacement,
            'all_tortuosity': min(all_tortuosity, 100.0),
            'all_mean_radius': all_mean_radius,
            'n_tokens': n_tokens,
            # Thinking-only trajectory metrics
            'think_arc_length': think_arc_length,
            'think_displacement': think_displacement,
            'think_tortuosity': min(think_tortuosity, 100.0),
            'think_mean_radius': think_mean_radius,
            'n_thinking': n_thinking,
            # MI Peaks metrics
            'thinking_density': thinking_metrics['thinking_density'],
            'thinking_count': thinking_metrics['thinking_count'],
        })
    
    df = pd.DataFrame(metrics)
    
    # Summary stats - ALL TOKENS
    print(f"\n📊 Trajectory Summary (ALL TOKENS):")
    for sample_type in df['type'].unique():
        subset = df[df['type'] == sample_type]
        print(f"  {sample_type}: Arc={subset['all_arc_length'].mean():.2f}±{subset['all_arc_length'].std():.2f}, "
              f"Disp={subset['all_displacement'].mean():.2f}, "
              f"Tort={subset['all_tortuosity'].mean():.2f}")
    
    # Summary stats - THINKING TOKENS ONLY
    print(f"\n🧠 Trajectory Summary (THINKING TOKENS ONLY - MI Peaks):")
    for sample_type in df['type'].unique():
        subset = df[df['type'] == sample_type]
        print(f"  {sample_type}: Arc={subset['think_arc_length'].mean():.2f}±{subset['think_arc_length'].std():.2f}, "
              f"ThinkCount={subset['n_thinking'].mean():.1f}, "
              f"Density={subset['thinking_density'].mean():.3f}")
    
    return df

# ==========================================
# 2b. MULTI-GPU PARALLEL INFERENCE
# ==========================================

def get_activations_parallel(data, model_type, probe_after_reasoning=False):
    """
    Extract activations using ALL available GPUs in parallel.
    
    Automatically:
    - Detects number of GPUs
    - Splits data evenly across GPUs
    - Loads models SEQUENTIALLY to avoid race conditions
    - Runs inference in parallel
    - Combines results
    
    Falls back to single-GPU if only 1 GPU or issues arise.
    """
    if not USE_MULTI_GPU:
        print("⚠️  Single GPU detected, using standard inference")
        model = get_model(model_type=model_type)
        acts, metas = get_activations(model, data, probe_after_reasoning)
        return acts, metas, model  # Return model for reuse
    
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
    # This is critical - concurrent loading causes "meta device" errors
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
    working_gpus = [gpu_id for gpu_id, model in models.items() if model is not None]
    if not working_gpus:
        print("⚠️  All GPU loads failed, falling back to single GPU")
        torch.cuda.empty_cache()
        model = get_model(model_type=model_type, target_device="cuda:0")
        acts, metas = get_activations(model, data, probe_after_reasoning)
        return acts, metas, model  # Return model for reuse
    
    # Get config
    model_key = 'deepseek' if probe_after_reasoning else 'qwen'
    layer_idx = CONFIG[model_key]['layer']
    reasoning_tokens = CONFIG[model_key]['reasoning_tokens']
    
    # Worker function for inference on a single GPU
    def gpu_inference_worker(gpu_id, model, data_chunk, result_queue):
        try:
            acts = []
            metas = []
            thinking_token_counts = []
            fallback_count = 0
            
            for item in tqdm(data_chunk, desc=f"GPU {gpu_id}", position=gpu_id):
                prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{item['context']}\n{item['query']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                
                with torch.no_grad():
                    if probe_after_reasoning:
                        # USE SAME METHODOLOGY AS SINGLE-GPU
                        extraction = extract_thinking_activations(
                            model, prompt, layer_idx, reasoning_tokens
                        )
                        
                        thinking_acts = extraction['thinking_acts']
                        thinking_count = len(extraction['thinking_positions'])
                        thinking_token_counts.append(thinking_count)
                        
                        # Mean pool thinking token activations
                        if thinking_count > 0:
                            act = thinking_acts.mean(dim=0)
                        else:
                            act = extraction['all_acts'][-1]
                            fallback_count += 1
                        
                        # Store thinking info in metadata
                        item = {**item, 
                                'thinking_count': thinking_count,
                                'thinking_density': extraction['thinking_metrics']['thinking_density']}
                    else:
                        _, cache = model.run_with_cache(prompt, 
                                                        names_filter=lambda x: x.endswith("hook_resid_post"))
                        act = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1, :].cpu()
                        del cache
                        torch.cuda.empty_cache()
                    
                    acts.append(act)
                    metas.append(item)
            
            result_queue.put({
                'acts': torch.stack(acts) if acts else None,
                'metas': metas,
                'thinking_counts': thinking_token_counts,
                'fallback_count': fallback_count,
                'gpu_id': gpu_id
            })
        except Exception as e:
            print(f"⚠️  GPU {gpu_id} inference error: {e}")
            result_queue.put(None)
    
    # Run parallel inference (models already loaded)
    import threading
    from queue import Queue
    
    result_queue = Queue()
    threads = []
    
    # Start threads for inference (models already loaded)
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
    for gpu_id, model in models.items():
        if gpu_id != 0 and model is not None:
            del model
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
# 3. Mappers
# ==========================================
class HyperbolicMapper(nn.Module):
    """
    Hyperbolic Mapper that balances stability with discriminative power.
    
    Key insight: Input normalization destroys magnitude information!
    Instead, we use LayerNorm which preserves relative differences.
    
    Architecture: ϕ(h) = exp_0^D(scale · W · LayerNorm(h))
    """
    def __init__(self, input_dim, output_dim=16, c=1.0):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=c)
        # LayerNorm preserves relative differences (unlike unit normalization)
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        # Spectral norm prevents weight explosion
        self.fc = nn.utils.spectral_norm(self.fc)
        # Learnable scale (initialized small)
        self.log_scale = nn.Parameter(torch.tensor(-2.0))
    
    def forward(self, x):
        # 1. LayerNorm (preserves relative differences, stabilizes magnitude)
        x = self.input_norm(x)
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

class EuclideanMapper(nn.Module):
    """
    Euclidean Mapper with same stabilization as HyperbolicMapper for fair comparison.
    """
    def __init__(self, input_dim, output_dim=16):
        super().__init__()
        # Same stabilization as HyperbolicMapper
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.fc = nn.utils.spectral_norm(self.fc)
        # Learnable scale
        self.log_scale = nn.Parameter(torch.tensor(0.0))  # Larger init for Euclidean
    
    def forward(self, x):
        x = self.input_norm(x)
        v = self.fc(x)
        scale = torch.sigmoid(self.log_scale) * 2.0  # max scale = 2.0 for Euclidean
        return v * scale
    
    def dist(self, u, v):
        return torch.norm(u - v, dim=-1)
    
    def radius(self, u):
        """Euclidean norm from origin"""
        return torch.norm(u, dim=-1)

# ==========================================
# 3b. Intrinsic Dimension Analysis (SVD-based)
# ==========================================
def analyze_intrinsic_dimension(acts, metas, threshold=0.9):
    """
    Analyze Intrinsic Dimension using SVD as a proxy.
    
    Theory: Premature Manifold Collapse
    - Truth: High ID (model is exploring complex dependency space)
    - Hallucination: Low ID (model collapsed early into decision subspace)
    
    Args:
        acts: Tensor of activations [N, hidden_dim]
        metas: List of metadata dicts with 'type' field
        threshold: Variance threshold for effective dimension (default 90%)
    
    Returns:
        Dict with ID metrics for True vs Hallucination samples
    """
    print(f"\n{'='*60}")
    print("INTRINSIC DIMENSION ANALYSIS (SVD Proxy)")
    print(f"{'='*60}")
    
    # Separate True vs Hallucination
    true_idxs = [i for i, m in enumerate(metas) if m.get('type') == 'true']
    hall_idxs = [i for i, m in enumerate(metas) if m.get('type') == 'hallucination']
    
    if len(true_idxs) < 5 or len(hall_idxs) < 5:
        print("⚠️  Not enough samples for ID analysis")
        return None
    
    true_acts = acts[true_idxs].float()
    hall_acts = acts[hall_idxs].float()
    
    def compute_effective_dim(X, threshold=0.9):
        """Compute effective dimension using SVD"""
        # Center the data
        X_centered = X - X.mean(dim=0)
        
        # SVD
        try:
            U, S, V = torch.svd(X_centered)
        except:
            return -1, None
        
        # Calculate effective dimension (how many singular values for threshold% variance)
        S_squared = S ** 2
        total_var = S_squared.sum()
        cumsum = torch.cumsum(S_squared, dim=0)
        effective_dim = (cumsum < threshold * total_var).sum().item() + 1
        
        # Also compute spectral decay rate
        if len(S) > 1:
            decay_rate = (S[0] / S[-1]).item() if S[-1] > 0 else float('inf')
        else:
            decay_rate = 1.0
        
        return effective_dim, S.cpu().numpy()
    
    # Compute for each group
    true_dim, true_S = compute_effective_dim(true_acts, threshold)
    hall_dim, hall_S = compute_effective_dim(hall_acts, threshold)
    
    print(f"\n📊 Effective Dimension (explains {int(threshold*100)}% variance):")
    print(f"  True samples (n={len(true_idxs)}):        ID = {true_dim}")
    print(f"  Hallucination samples (n={len(hall_idxs)}): ID = {hall_dim}")
    
    if true_dim > 0 and hall_dim > 0:
        ratio = true_dim / hall_dim
        print(f"\n  Ratio (True/Hall): {ratio:.2f}x")
        
        if ratio > 1.2:
            print(f"  ✅ CONFIRMED: Truth has higher ID ({ratio:.2f}x) - NOT premature collapse")
        elif ratio < 0.8:
            print(f"  ⚠️ INVERTED: Hallucination has higher ID!")
        else:
            print(f"  ➡️ SIMILAR: No significant ID difference")
    
    # Plot singular value spectrum
    plt.figure(figsize=(10, 5))
    
    if true_S is not None and hall_S is not None:
        plt.subplot(1, 2, 1)
        plt.plot(true_S[:50], 'b-', label='True', linewidth=2)
        plt.plot(hall_S[:50], 'r--', label='Hallucination', linewidth=2)
        plt.xlabel('Singular Value Index')
        plt.ylabel('Magnitude')
        plt.title('Singular Value Spectrum (First 50)')
        plt.legend()
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        # Normalized cumulative variance
        true_cumvar = np.cumsum(true_S**2) / np.sum(true_S**2)
        hall_cumvar = np.cumsum(hall_S**2) / np.sum(hall_S**2)
        plt.plot(true_cumvar[:50], 'b-', label='True', linewidth=2)
        plt.plot(hall_cumvar[:50], 'r--', label='Hallucination', linewidth=2)
        plt.axhline(y=threshold, color='gray', linestyle=':', label=f'{int(threshold*100)}% threshold')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance Explained')
        plt.title('Variance Explained')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/plots/intrinsic_dimension_analysis.png", dpi=150)
    plt.close()
    print(f"\n✅ Saved: results/plots/intrinsic_dimension_analysis.png")
    
    return {
        'true_dim': true_dim,
        'hall_dim': hall_dim,
        'true_n': len(true_idxs),
        'hall_n': len(hall_idxs),
        'threshold': threshold
    }

def train_mapper(mapper_cls, acts, metas, epochs=300, lr=0.01, is_hyperbolic=True, shuffle_labels=False):
    """
    Train mapper to predict logical depth from activations.
    
    Uses Euclidean radius for training (stable gradients), then
    hyperbolic distance for evaluation.
    """
    true_idxs = [i for i, m in enumerate(metas) if m['label']=='TRUE']
    
    if len(true_idxs) < 10:
        print(f"⚠️  WARNING: Only {len(true_idxs)} TRUE samples! Need at least 10.")
    
    X = acts[true_idxs].to(device).float()
    
    if shuffle_labels:
        fixed_depths = [m['depth'] for m in np.array(metas)[true_idxs]]
        random.shuffle(fixed_depths)
        depths = torch.tensor(fixed_depths, device=device).float()
    else:
        depths = torch.tensor([m['depth'] for m in np.array(metas)[true_idxs]], device=device).float()
    
    depth_min, depth_max = depths.min().item(), depths.max().item()
    print(f"  📊 Training {mapper_cls.__name__}: {len(true_idxs)} samples, depths [{depth_min:.0f}, {depth_max:.0f}]")
    
    # Adaptive target scaling: map depths to Euclidean radii
    if is_hyperbolic:
        alpha = 0.075  # Scaling factor
        targets = alpha * depths  # depth=1 → 0.075, depth=5 → 0.375
    else:
        targets = 0.2 + (depths - 1) * 0.2  # Euclidean uses larger range
    
    target_min, target_max = targets.min().item(), targets.max().item()
    print(f"  🎯 Targets (α={alpha if is_hyperbolic else 'N/A'}): [{target_min:.3f}, {target_max:.3f}]")
        
    mapper = mapper_cls(X.shape[1], 16).to(device)
    
    # Use standard Adam (more stable)
    optimizer = optim.Adam(mapper.parameters(), lr=lr)
    
    initial_loss = None
    final_loss = None
    
    for e in range(epochs):
        optimizer.zero_grad()
        emb = mapper(X)
        # Train on EUCLIDEAN RADIUS (stable gradients)
        if hasattr(mapper, 'radius'):
            d_origin = mapper.radius(emb)
        else:
            d_origin = torch.norm(emb, dim=-1)
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
            
            # VALIDATION 4: Check for saturation
            if dist_std < 0.001 and e > 50:
                print(f"  ⚠️  WARNING: Distance std={dist_std:.6f} too low! Possible saturation.")
            if dist_mean > 5.0:
                print(f"  ⚠️  WARNING: Mean dist={dist_mean:.2f} very high! Boundary saturation.")
    
    # VALIDATION 5: Check training improved
    if initial_loss and final_loss:
        improvement = (initial_loss - final_loss) / initial_loss * 100
        if improvement < 10:
            print(f"  ⚠️  WARNING: Loss only improved {improvement:.1f}%!")
        else:
            print(f"  ✅ Complete: {improvement:.1f}% improvement ({initial_loss:.4f} → {final_loss:.4f})")
            
    return mapper

# ==========================================
# 4. Analysis & Plotting
# ==========================================
def run_control_studies(acts, metas, prefix=""):
    print(f"\n--- Running Phase 7 Control Studies ({prefix}) ---")
    
    # Control 1: Random Baseline (Untrained)
    # Just project linearly and map to ball
    mapper_rand = HyperbolicMapper(acts.shape[1], 16).to(device)
    with torch.no_grad():
        emb_r = mapper_rand(acts.to(device).float())
        d_r = mapper_rand.dist(emb_r, torch.zeros_like(emb_r)).cpu().numpy()
        
    # Control 2: Shuffled Labels (Trained on Noise)
    mapper_shuff = train_mapper(HyperbolicMapper, acts, metas, is_hyperbolic=True, shuffle_labels=True)
    with torch.no_grad():
        emb_s = mapper_shuff(acts.to(device).float())
        d_s = mapper_shuff.dist(emb_s, torch.zeros_like(emb_s)).cpu().numpy()
        
    # Plot Comparisons
    data = []
    types = [m['type'] for m in metas]
    for i, t in enumerate(types):
        if t not in ['true', 'hallucination']: continue
        lbl = "True" if t == 'true' else "Hallucination"
        
        data.append({'Condition': 'Random (Untrained)', 'Distance': d_r[i], 'Type': lbl})
        data.append({'Condition': 'Shuffled (Noise)', 'Distance': d_s[i], 'Type': lbl})
        
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=pd.DataFrame(data), x='Condition', y='Distance', hue='Type', split=True, palette={'True': 'blue', 'Hallucination': 'red'})
    plt.title(f"Rigorous Controls ({prefix}): Signal vs Noise")
    plt.savefig(f"results/plots/{prefix}_control_studies.png")

def compare_geometries(acts, metas, prefix=""):
    """Compare Hyperbolic vs Euclidean geometry with proper train/test split."""
    
    # TRAIN/TEST SPLIT: 80/20 to measure GENERALIZATION, not memorization
    all_idxs = list(range(len(metas)))
    np.random.seed(42)
    np.random.shuffle(all_idxs)
    split = int(0.8 * len(all_idxs))
    train_idxs = all_idxs[:split]
    test_idxs = all_idxs[split:]
    
    train_acts = acts[train_idxs]
    train_metas = [metas[i] for i in train_idxs]
    test_acts = acts[test_idxs]
    test_metas = [metas[i] for i in test_idxs]
    
    print(f"\n📊 Train/Test Split: {len(train_idxs)} train, {len(test_idxs)} test")
    
    # Train mappers on TRAIN set only
    hyp_mapper = train_mapper(HyperbolicMapper, train_acts, train_metas, is_hyperbolic=True)
    euc_mapper = train_mapper(EuclideanMapper, train_acts, train_metas, is_hyperbolic=False)
    
    # Run Controls First (on full data for visualization)
    run_control_studies(acts, metas, prefix=prefix)
    
    # 1. Stress Test on TEST SET (GENERALIZATION ERROR)
    # BUG FIX: Use CORRECT scaling that matches training targets!
    ALPHA_HYP = 0.075  # Must match train_mapper!
    
    test_true_idxs = [i for i, m in enumerate(test_metas) if m['label']=='TRUE']
    if len(test_true_idxs) < 5:
        print(f"⚠️ Only {len(test_true_idxs)} TRUE samples in test set, using train set for stress test")
        test_true_idxs = [i for i, m in enumerate(train_metas) if m['label']=='TRUE']
        X_stress = train_acts[test_true_idxs].to(device).float()
        depths_stress = np.array([m['depth'] for m in np.array(train_metas)[test_true_idxs]])
    else:
        X_stress = test_acts[test_true_idxs].to(device).float()
        depths_stress = np.array([m['depth'] for m in np.array(test_metas)[test_true_idxs]])
    
    print(f"  Stress Test on {len(test_true_idxs)} held-out TRUE samples")
    
    with torch.no_grad():
        # Get raw distances from probes
        h_emb = hyp_mapper(X_stress)
        h_d = hyp_mapper.radius(h_emb).cpu().numpy()
        e_emb = euc_mapper(X_stress)
        e_d = euc_mapper.radius(e_emb).cpu().numpy()
        
        # CORRECT RESCALING
        h_d_rescaled = h_d / ALPHA_HYP
        e_d_rescaled = (e_d - 0.2) / 0.2 + 1
        
    data = []
    for i, d in enumerate(depths_stress):
        data.append({'Geometry': 'Hyperbolic', 'True Depth': d, 'Error': abs(h_d_rescaled[i] - d)})
        data.append({'Geometry': 'Euclidean', 'True Depth': d, 'Error': abs(e_d_rescaled[i] - d)})
        
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=pd.DataFrame(data), x='True Depth', y='Error', hue='Geometry', marker='o')
    plt.title(f"Stress Test ({prefix}): GENERALIZATION Error (held-out test set)")
    plt.savefig(f"results/plots/{prefix}_stress_test.png")
    plt.close()
    
    # 2. Geometry Wars (Hallucination vs Truth)
    hall_idxs = [i for i, m in enumerate(metas) if m['label']=='FALSE' and m['type']=='hallucination']
    if not hall_idxs:
        print("WARNING: No hallucination samples found! Skipping Geometry Wars plot.")
        return {'hyp_avg': 0.0, 'euc_avg': 0.0, 'hyp_raw': 0.0, 'euc_raw': 0.0}
    
    X_h = acts[hall_idxs].to(device).float()
    with torch.no_grad():
        h_h = hyp_mapper.radius(hyp_mapper(X_h)).cpu().numpy()
        h_h_rescaled = h_h / ALPHA_HYP  # CORRECT scaling!
        e_h = euc_mapper.radius(euc_mapper(X_h)).cpu().numpy()
        e_h_rescaled = (e_h - 0.2) / 0.2 + 1  # CORRECT scaling!
        
    avg_h = np.mean(h_h_rescaled)
    avg_e = np.mean(e_h_rescaled)
    raw_h = np.mean(h_h)  # Also save raw distances for analysis
    raw_e = np.mean(e_h)
    
    plt.figure(figsize=(6, 5))
    sns.barplot(x=['Hyperbolic', 'Euclidean'], y=[avg_h, avg_e])
    plt.axhline(y=1.0, color='r', linestyle='--', label='Truth (1-Hop)')
    plt.ylabel('Predicted Depth')
    plt.title(f"Geometry Wars ({prefix}): Hallucination Predicted Depth")
    plt.savefig(f"results/plots/{prefix}_geometry_wars.png")
    plt.close()
    
    return {'hyp_avg': avg_h, 'euc_avg': avg_e, 'hyp_raw': raw_h, 'euc_raw': raw_e}

# ==========================================
# 5. Phase 1 & 2 Analysis (Legacy Plots)
# ==========================================
# NOTE: calculate_delta removed - it returned a placeholder 0.0 and was never used.
# If Gromov delta is needed, implement proper O(N^4) computation or sampling-based approximation.

def run_phase_1_2(model, datasets, prefix="", probe_after_reasoning=False, f_acts=None, f_meta=None):
    """
    Run Phase 1 & 2 Analysis with REAL DATA.
    
    FIG 1: Complexity Scaling - Compare depth radii
    FIG 2: Wormhole Signature - KDE plot
    
    Note: Layer sweep has been moved to run_layer_validation.py
    """
    print(f"Running Phase 1 & 2 Analysis ({prefix})...")
    print(f"  Probe after reasoning: {probe_after_reasoning}")
    
    # Use pre-computed activations if provided, otherwise compute
    if f_acts is None or f_meta is None:
        print("  Computing activations (no pre-computed data provided)...")
        f_acts, f_meta = get_activations(model, datasets['fiction'], 
                                          probe_after_reasoning=probe_after_reasoning)
    else:
        print(f"  Using pre-computed activations ({len(f_meta)} samples)")
    
    # FIG 1: Complexity Scaling - REAL depth comparison
    print("Computing complexity scaling (REAL data)...")
    mapper = train_mapper(HyperbolicMapper, f_acts, f_meta, epochs=200)
    
    depth_radii = {}
    with torch.no_grad():
        for d in range(1, 6):
            idxs = [i for i, m in enumerate(f_meta) if m['label']=='TRUE' and m['depth']==d]
            if idxs:
                X = f_acts[idxs].to(device).float()
                emb = mapper(X)
                r = mapper.radius(emb).mean().item()
                depth_radii[d] = r
    
    if depth_radii:
        depths = list(depth_radii.keys())
        radii = list(depth_radii.values())
        plt.figure(figsize=(8, 5))
        bars = plt.bar([f'Depth {d}' for d in depths], radii, color='purple')
        plt.ylabel("Mean Hyperbolic Radius")
        plt.title(f"Complexity Scaling ({prefix}) - REAL DATA")
        for bar, r in zip(bars, radii):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f'{r:.3f}', ha='center', fontsize=10)
        plt.savefig(f"results/plots/{prefix}_complexity_scaling.png")
        plt.close()
    
    # FIG 2: Basic Wormhole (uses CONFIG layer)
    model_key = 'deepseek' if prefix == 'deepseek' else 'qwen'
    layer_used = CONFIG[model_key]['layer']
    
    with torch.no_grad():
        emb = mapper(f_acts.to(device).float())
        r = mapper.radius(emb).cpu().numpy()
        
    # Plot basic distribution
    data = []
    for i, m in enumerate(f_meta):
        if m['type'] == 'true': lbl = "True"
        elif m['type'] == 'hallucination': lbl = "Hallucination"
        else: continue
        data.append({'Radius': r[i], 'Type': lbl, 'Depth': m.get('depth', -1)})
    
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Radius', hue='Type', fill=True, 
                palette={'True': 'blue', 'Hallucination': 'red'}, alpha=0.5)
    
    # Add vertical lines for mean radius by depth (TRUE only)
    true_depths = df[df['Type'] == 'True'].groupby('Depth')['Radius'].mean()
    for depth, mean_r in true_depths.items():
        if depth > 0:
            plt.axvline(x=mean_r, color='gray', linestyle='--', alpha=0.5, 
                       label=f'Depth {depth}: {mean_r:.2f}')
    
    probe_state = "In-Process Reasoning" if prefix == 'deepseek' else "Standard"
    plt.title(f"Wormhole Signature (Layer {layer_used}) - {prefix.upper()} ({probe_state})")
    plt.xlabel("Hyperbolic Radius")
    plt.ylabel("Density")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', title='Sample Type')
    plt.tight_layout()
    plt.savefig(f"results/plots/{prefix}_wormhole.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  \u2713 Saved complexity scaling and wormhole plots")




# NOTE: Layer validation moved to separate script: run_layer_validation.py
# Run: python run_layer_validation.py --model deepseek --samples 50

# ==========================================
# Main Execution
# ==========================================
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Geometric Analysis on LLMs")
    parser.add_argument("--model", type=str, default="qwen", choices=["qwen", "deepseek"], help="Choose model: 'qwen' (Standard) or 'deepseek' (Reasoning)")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples (default: from CONFIG - 200 for qwen, 150 for deepseek)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--multi-gpu", action="store_true", default=True, help="Enable multi-GPU parallel inference (default: True, auto-detects GPUs)")
    parser.add_argument("--single-gpu", action="store_true", help="Force single-GPU mode (disables multi-GPU)")
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
    
    # 1. Generate BALANCED Data (reproducible with seed)
    print(f"\n--- Generating Balanced Datasets (seed={args.seed}) ---")
    datasets = {}
    for mode in ['fiction', 'real']:
        print(f"\nGenerating {mode} data...")
        gen = LogicTreeGenerator(mode=mode, seed=args.seed)
        datasets[mode] = generate_balanced_dataset(gen, N_SAMPLES, seed=args.seed)
        
    # 2. Select Model Type
    model_type = "standard" if args.model == "qwen" else "reasoning"
    
    # 3. Infer & Analyze
    # CRITICAL: For DeepSeek (reasoning), probe AFTER the <think>...</think> phase
    is_reasoning_model = args.model == "deepseek"
    print(f"\n🔬 In-Process Reasoning State: {is_reasoning_model}")
    
    # Use parallel inference if multiple GPUs available
    if use_parallel:
        print(f"🚀 Using MULTI-GPU parallel inference ({N_GPUS} GPUs)")
        f_acts, f_meta, model = get_activations_parallel(
            datasets['fiction'], 
            model_type=model_type,
            probe_after_reasoning=is_reasoning_model
        )
    else:
        print("Using single-GPU inference")
        model = get_model(model_type=model_type)
        f_acts, f_meta = get_activations(model, datasets['fiction'], probe_after_reasoning=is_reasoning_model)
    
    # Use Prefix based on model type
    prefix = args.model  # "qwen" or "deepseek"
    print(f"Saving plots with prefix: {prefix}_")
    
    # Pass pre-computed activations to avoid redundant inference
    run_phase_1_2(model, datasets, prefix=prefix, probe_after_reasoning=is_reasoning_model, 
                  f_acts=f_acts, f_meta=f_meta)
    
    # Capture metrics from comparison
    metrics = compare_geometries(f_acts, f_meta, prefix=prefix)
    
    # Run Intrinsic Dimension Analysis (SVD-based)
    id_results = analyze_intrinsic_dimension(f_acts, f_meta)
    
    # Save Metrics to Text File
    with open(f"results/plots/{prefix}_metrics.txt", "w") as f:
        f.write(f"--- Metrics for {prefix} ---\n")
        f.write(f"Model: {prefix}\n")
        f.write(f"Layer: {CONFIG[prefix]['layer']}\n")
        f.write(f"Reasoning Tokens: {CONFIG[prefix]['reasoning_tokens']}\n")
        f.write(f"In-Process Reasoning: {is_reasoning_model}\n")
        f.write(f"\n--- Wormhole Results ---\n")
        f.write(f"Hyperbolic Avg Dist (Hallucination): {metrics['hyp_avg']:.4f}\n")
        f.write(f"Euclidean Avg Dist (Hallucination): {metrics['euc_avg']:.4f}\n")
        f.write(f"Ratio (Separability): {metrics['hyp_avg'] / metrics['euc_avg']:.2f}x\n")
        
        # Add ID analysis results
        if id_results:
            f.write(f"\n--- Intrinsic Dimension Analysis ---\n")
            f.write(f"True ID (90% variance): {id_results['true_dim']}\n")
            f.write(f"Hallucination ID: {id_results['hall_dim']}\n")
            f.write(f"Ratio (True/Hall): {id_results['true_dim']/id_results['hall_dim']:.2f}x\n")
    
    # ==========================================
    # TRAJECTORY ANALYSIS (DeepSeek Only)
    # ==========================================
    if is_reasoning_model:
        print("\n" + "="*60)
        print("TRAJECTORY ANALYSIS: Reasoning Dynamics")
        print("="*60)
        print("Analyzing how representations move through hyperbolic space during reasoning...")
        
        # Train a mapper for trajectory analysis (use HyperbolicMapper class defined in this file)
        print("\nTraining mapper on fiction data for trajectory analysis...")
        traj_mapper = train_mapper(HyperbolicMapper, f_acts, f_meta, epochs=200)
        
        # Run trajectory analysis on a subset (expensive due to generation)
        traj_subset = datasets['fiction']  # Use all samples for statistical power
        traj_df = get_trajectory_metrics(model, traj_subset, traj_mapper)
        
        # ==========================================
        # Exp 5A: Arc Length (Work Done)
        # ==========================================
        print("\n=== Exp 5A: Arc Length by Sample Type ===")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # By type
        type_order = ['true', 'hallucination', 'unrelated']
        type_colors = {'true': 'green', 'hallucination': 'red', 'unrelated': 'gray'}
        
        arc_by_type = traj_df.groupby('type')['all_arc_length'].mean()
        types_present = [t for t in type_order if t in arc_by_type.index]
        
        bars = axes[0].bar(types_present, [arc_by_type[t] for t in types_present],
                           color=[type_colors[t] for t in types_present])
        axes[0].set_ylabel('Mean Arc Length')
        axes[0].set_title(f'Exp 5A: Arc Length by Type ({prefix})')
        axes[0].set_xlabel('Sample Type')
        
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
        
        # By depth (for TRUE samples)
        true_df = traj_df[traj_df['type'] == 'true']
        if len(true_df) > 0:
            depth_arc = true_df.groupby('depth')['all_arc_length'].mean()
            axes[1].bar(depth_arc.index.astype(str), depth_arc.values, color='steelblue')
            axes[1].set_ylabel('Mean Arc Length')
            axes[1].set_title(f'Exp 5A: Arc Length by Depth (TRUE only)')
            axes[1].set_xlabel('Logical Depth')
        
        plt.tight_layout()
        plt.savefig(f"results/plots/{prefix}_exp5a_arc_length.png", dpi=150)
        plt.close()
        print(f"Saved: results/plots/{prefix}_exp5a_arc_length.png")
        
        # ==========================================
        # Exp 5B: Tortuosity (Efficiency)
        # ==========================================
        print("\n=== Exp 5B: Tortuosity (Efficiency of Thought) ===")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # By type
        tort_by_type = traj_df.groupby('type')['all_tortuosity'].mean()
        
        bars = axes[0].bar(types_present, [tort_by_type.get(t, 0) for t in types_present],
                           color=[type_colors[t] for t in types_present])
        axes[0].set_ylabel('Mean Tortuosity')
        axes[0].set_title(f'Exp 5B: Tortuosity by Type ({prefix})')
        axes[0].set_xlabel('Sample Type')
        axes[0].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfectly Direct (τ=1)')
        axes[0].legend()
        
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom')
        
        # Distribution comparison
        for t in types_present:
            subset = traj_df[traj_df['type'] == t]['all_tortuosity']
            if len(subset) > 0:
                axes[1].hist(subset, alpha=0.5, label=t, bins=15, color=type_colors[t])
        
        axes[1].set_xlabel('Tortuosity')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Exp 5B: Tortuosity Distribution ({prefix})')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f"results/plots/{prefix}_exp5b_tortuosity.png", dpi=150)
        plt.close()
        print(f"Saved: results/plots/{prefix}_exp5b_tortuosity.png")
        
        # Interpretation
        hall_arc = traj_df[traj_df['type'] == 'hallucination']['all_arc_length'].mean() if 'hallucination' in traj_df['type'].values else 0
        true_arc = traj_df[traj_df['type'] == 'true']['all_arc_length'].mean() if 'true' in traj_df['type'].values else 0
        
        # Also compare thinking-only trajectory
        hall_think = traj_df[traj_df['type'] == 'hallucination']['think_arc_length'].mean() if 'hallucination' in traj_df['type'].values else 0
        true_think = traj_df[traj_df['type'] == 'true']['think_arc_length'].mean() if 'true' in traj_df['type'].values else 0
        
        print(f"\n📊 TRAJECTORY INTERPRETATION:")
        if true_arc > hall_arc * 1.2:
            print(f"  ✅ TRUE samples travel MORE ({true_arc:.2f}) than HALLUCINATION ({hall_arc:.2f})")
            print(f"     → Supports 'Lazy Hallucination' hypothesis!")
        elif hall_arc > true_arc * 1.2:
            print(f"  ⚠️ HALLUCINATION travels MORE - unexpected!")
        else:
            print(f"  ≈ Similar arc lengths - no clear difference")
        
        # ==========================================
        # Exp 5C: Thinking Token Density (MI Peaks)
        # ==========================================
        print("\n=== Exp 5C: Thinking Token Density (MI Peaks Alignment) ===")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Thinking density by type
        density_by_type = traj_df.groupby('type')['thinking_density'].mean()
        bars = axes[0].bar(types_present, [density_by_type.get(t, 0) for t in types_present],
                          color=[type_colors[t] for t in types_present])
        axes[0].set_ylabel('Thinking Token Density')
        axes[0].set_title(f'Exp 5C: Thinking Density by Type ({prefix})')
        axes[0].set_xlabel('Sample Type')
        axes[0].axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='Paper avg (~5%)')
        axes[0].legend()
        
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # Thinking count histogram
        for t in types_present:
            subset = traj_df[traj_df['type'] == t]['n_thinking']
            if len(subset) > 0:
                axes[1].hist(subset, alpha=0.5, label=t, bins=15, color=type_colors[t])
        
        axes[1].set_xlabel('Number of Thinking Tokens')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Exp 5C: Thinking Token Count Distribution ({prefix})')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f"results/plots/{prefix}_exp5c_thinking_density.png", dpi=150)
        plt.close()
        print(f"Saved: results/plots/{prefix}_exp5c_thinking_density.png")
        
        # ==========================================
        # Exp 5D: Thinking-Only vs All-Token Trajectory
        # ==========================================
        print("\n=== Exp 5D: Thinking vs All Trajectory Comparison ===")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Side-by-side bar comparison
        x = np.arange(len(types_present))
        width = 0.35
        
        all_arcs = [traj_df[traj_df['type'] == t]['all_arc_length'].mean() for t in types_present]
        think_arcs = [traj_df[traj_df['type'] == t]['think_arc_length'].mean() for t in types_present]
        
        bars1 = axes[0].bar(x - width/2, all_arcs, width, label='All Tokens', color='steelblue')
        bars2 = axes[0].bar(x + width/2, think_arcs, width, label='Thinking Only', color='coral')
        
        axes[0].set_ylabel('Arc Length')
        axes[0].set_title(f'Exp 5D: All vs Thinking Trajectory ({prefix})')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(types_present)
        axes[0].legend()
        
        # Ratio plot: think_arc / all_arc
        ratios = [think_arcs[i] / (all_arcs[i] + 1e-6) for i in range(len(types_present))]
        bars = axes[1].bar(types_present, ratios, color=[type_colors[t] for t in types_present])
        axes[1].set_ylabel('Think Arc / All Arc Ratio')
        axes[1].set_title(f'Exp 5D: Trajectory Concentration Ratio ({prefix})')
        axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"results/plots/{prefix}_exp5d_thinking_trajectory.png", dpi=150)
        plt.close()
        print(f"Saved: results/plots/{prefix}_exp5d_thinking_trajectory.png")
        
        # ==========================================
        # Exp 5E: Think Arc vs Thinking Count Correlation
        # ==========================================
        print("\n=== Exp 5E: Thinking Arc vs Count Correlation ===")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for t in types_present:
            subset = traj_df[traj_df['type'] == t]
            ax.scatter(subset['n_thinking'], subset['think_arc_length'], 
                      alpha=0.6, label=t, color=type_colors[t], s=50)
        
        ax.set_xlabel('Number of Thinking Tokens')
        ax.set_ylabel('Thinking-Only Arc Length')
        ax.set_title(f'Exp 5E: Thinking Token Count vs Arc Length ({prefix})')
        ax.legend()
        
        # Add correlation line for TRUE
        true_subset = traj_df[traj_df['type'] == 'true']
        if len(true_subset) > 5:
            z = np.polyfit(true_subset['n_thinking'], true_subset['think_arc_length'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(true_subset['n_thinking'].min(), true_subset['n_thinking'].max(), 100)
            ax.plot(x_line, p(x_line), 'g--', alpha=0.8, label=f'TRUE trend (slope={z[0]:.2f})')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"results/plots/{prefix}_exp5e_think_correlation.png", dpi=150)
        plt.close()
        print(f"Saved: results/plots/{prefix}_exp5e_think_correlation.png")
        
        # Thinking interpretation
        print(f"\n🧠 THINKING TRAJECTORY INTERPRETATION:")
        print(f"  TRUE thinking arc: {true_think:.2f}")
        print(f"  HALL thinking arc: {hall_think:.2f}")
        if true_think > hall_think * 1.2:
            print(f"  ✅ TRUE has MORE movement at thinking tokens - MI Peaks hypothesis supported!")
        elif hall_think > true_think * 1.2:
            print(f"  ⚠️ HALLUCINATION has more thinking movement - needs investigation")
        else:
            print(f"  ≈ Similar thinking trajectory lengths")
        
        # Save trajectory metrics
        traj_df.to_csv(f"results/plots/{prefix}_trajectory_metrics.csv", index=False)
        print(f"Saved trajectory metrics: results/plots/{prefix}_trajectory_metrics.csv")
        
    print(f"\nDone! Charts saved to results/plots/{prefix}_*.png")
    print(f"Metrics saved to results/plots/{prefix}_metrics.txt")
    print(f"\n💡 TIP: Run 'python run_layer_validation.py --model {prefix}' to validate optimal layer")

