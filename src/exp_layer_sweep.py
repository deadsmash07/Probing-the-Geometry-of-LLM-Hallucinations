"""
Layer Validation Script
=======================

Standalone script to find the optimal probing layer for a given model.
Tests True vs Hallucination separability across layers and saves a visualization.

Usage:
    python exp_layer_sweep.py --model qwen --samples 300
    python exp_layer_sweep.py --model deepseek --samples 300
"""

import os
import sys
import re
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import geoopt
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OPTIMIZATION: Cache generated outputs to avoid duplicate generation
GENERATION_CACHE = {}

def get_cached_generation(model, prompt, max_tokens, temperature=0.0):
    """Generate text with caching for 2x speedup."""
    cache_key = hash(prompt + str(max_tokens))
    if cache_key in GENERATION_CACHE:
        return GENERATION_CACHE[cache_key]
    
    output = model.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
    output_str = output if isinstance(output, str) else model.tokenizer.decode(output[0])
    GENERATION_CACHE[cache_key] = output_str
    return output_str

def clear_generation_cache():
    """Clear the generation cache."""
    global GENERATION_CACHE
    GENERATION_CACHE = {}

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")
os.makedirs("results/layer_validation", exist_ok=True)

# Model-Specific Configuration
CONFIG = {
    'qwen': {
        'layer': 23,              # Current optimal layer for Qwen
        'reasoning_tokens': 0,
        'probe_after_reasoning': False,
        'test_layers': list(range(0, 28, 3)) + [23],  # Every 3rd layer + L23
    },
    'deepseek': {
        'layer': 23,              # Current optimal layer for DeepSeek
        'reasoning_tokens': 512,  # Increased from 128 to match run_full_experiment.py
        'probe_after_reasoning': True,
        'test_layers': [16, 19, 21, 23, 25],
    }
}

# Auth - Load HF_TOKEN from environment
if not os.environ.get("HF_TOKEN"):
    print("Warning: HF_TOKEN not found in environment. Please set it in .env file.")

# ==========================================
# THINKING TOKEN DETECTION (MI Peaks Alignment)
# ==========================================
THINKING_TOKENS = [
    "hmm", "wait", "let me", "actually", "oh", "so",
    "therefore", "thus", "hence", "because", "since",
    "if", "then", "but", "however", "although",
    "first", "second", "next", "finally", "step",
    "means", "implies", "shows", "proves"
]

THINKING_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(t) for t in THINKING_TOKENS) + r')\b', 
    re.IGNORECASE
)

def find_thinking_positions(output_tokens):
    """Identify thinking token positions (analogous to MI peaks)."""
    positions = []
    for i, tok in enumerate(output_tokens):
        tok_str = str(tok).lower().strip()
        if THINKING_PATTERN.search(tok_str):
            positions.append(i)
    return positions

# ==========================================
# Data Generator (Same as run_full_experiment.py)
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
        """Generate samples with True, Hallucination, and Unrelated types."""
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
        # True: reachable ancestors
        true_types = set(nx.descendants(G, start)) | {start}
        anc = list(true_types - {start})
        if anc:
            tgt = self.rng.choice(anc)
            hops = nx.shortest_path_length(G, start, tgt)
            samples.append({"context": ctx, "query": f"Is {subject} a {tgt}?", "label": "TRUE", "depth": hops, "type": "true"})
        
        # Hallucination: in graph but not reachable
        false_t = list(set(G.nodes) - true_types)
        if false_t:
            tgt = self.rng.choice(false_t)
            samples.append({"context": ctx, "query": f"Is {subject} a {tgt}?", "label": "FALSE", "depth": -1, "type": "hallucination"})
        
        # Unrelated: not in graph at all
        strangers = [n for n in self.nouns if n not in G.nodes]
        if strangers:
            tgt = self.rng.choice(strangers)
            samples.append({"context": ctx, "query": f"Is {subject} a {tgt}?", "label": "FALSE", "depth": -2, "type": "unrelated"})
        return samples

def generate_balanced_dataset(generator, n_samples, seed=42):
    """Generate dataset with BALANCED class distribution."""
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
                target_per_bucket = n_per_class // 2
                if len(collected['true'][bucket]) < target_per_bucket:
                    collected['true'][bucket].append(sample)
            elif t == 'hallucination':
                if len(collected['hallucination']) < n_per_class:
                    collected['hallucination'].append(sample)
            elif t == 'unrelated':
                if len(collected['unrelated']) < n_per_class:
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
    
    # Report balance
    actual_true = len([d for d in dataset if d['type'] == 'true'])
    actual_hall = len([d for d in dataset if d['type'] == 'hallucination'])
    actual_unrel = len([d for d in dataset if d['type'] == 'unrelated'])
    
    print(f"📊 Balanced Dataset Generated (seed={seed}):")
    print(f"   Total: {len(dataset)} samples")
    print(f"   • True: {actual_true} ({100*actual_true/len(dataset) if dataset else 0:.1f}%)")
    print(f"   • Hallucination: {actual_hall} ({100*actual_hall/len(dataset) if dataset else 0:.1f}%)")
    print(f"   • Unrelated: {actual_unrel} ({100*actual_unrel/len(dataset) if dataset else 0:.1f}%)")
    
    return dataset

# ==========================================
# Model Loading
# ==========================================
def get_model(model_type="qwen"):
    if model_type == "qwen":
        print("=" * 60)
        print("🔧 Loading: Qwen/Qwen2.5-7B-Instruct")
        print("=" * 60)
        return HookedTransformer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device=device)
    elif model_type == "deepseek":
        print("=" * 60)
        print("🔧 Loading: DeepSeek-R1-Distill-Qwen-7B")
        print("=" * 60)
        # Load to CPU first to avoid double GPU memory allocation
        print("...downloading via Transformers (AutoModel) to CPU...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu",  # Load to CPU first
        )
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        
        print(f"...wrapping in HookedTransformer on {device}...")
        hooked_model = HookedTransformer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=device,
            fold_ln=False,
            fold_value_biases=False,
            center_writing_weights=False,
            center_unembed=False
        )
        
        # Free CPU memory
        del hf_model
        torch.cuda.empty_cache()
        
        return hooked_model

# ==========================================
# HyperbolicMapper (For quick probe training)
# ==========================================
class HyperbolicMapper(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=16, c=1.0):
        """
        output_dim=16 to match main experiments.
        REMOVED tanh constraint - let expmap handle the geometry naturally.
        """
        super().__init__()
        self.c = c
        self.ball = geoopt.PoincareBall(c=c)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Learnable scale to prevent initial saturation
        self.log_scale = nn.Parameter(torch.tensor(-1.0))
        
    def forward(self, x):
        v = self.proj(x)
        # Learnable bounded scale instead of fixed tanh constraint
        scale = torch.sigmoid(self.log_scale) * 0.8 + 0.1  # [0.1, 0.9]
        v = v * scale
        return self.ball.expmap0(v)
    
    def radius(self, x):
        return self.ball.dist0(x)

# ==========================================
# Layer Validation
# ==========================================
def run_layer_validation(model, data, model_name):
    """Find optimal layer by measuring True vs Hallucination separability.
    
    OPTIMIZED: Generates text ONCE and caches activations for ALL layers.
    This is 4x more efficient than generating per-layer.
    """
    print(f"\n{'='*60}")
    print(f"LAYER VALIDATION: {model_name.upper()}")
    print(f"{'='*60}")
    
    config = CONFIG[model_name]
    probe_after_reasoning = config['probe_after_reasoning']
    reasoning_tokens = config['reasoning_tokens']
    test_layers = sorted(set([l for l in config['test_layers'] if l < model.cfg.n_layers]))
    
    print(f"Probe after reasoning: {probe_after_reasoning}")
    print(f"Testing layers: {test_layers}")
    print(f"Using {len(data)} samples")
    
    # ==========================================
    # STEP 1: Generate ONCE, Cache ALL Layers
    # ==========================================
    print(f"\n📊 Generating and caching activations for ALL {len(test_layers)} layers...")
    print(f"   (This is 4x more efficient than regenerating per layer)")
    
    # Storage: {layer_idx: [act1, act2, ...]} 
    layer_acts = {l: [] for l in test_layers}
    
    for item in tqdm(data, desc="Caching activations", disable=not sys.stdout.isatty()):
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{item['context']}\n{item['query']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        with torch.no_grad():
            if probe_after_reasoning:
                # Generate ONCE with caching
                output_str = get_cached_generation(model, prompt, reasoning_tokens)
                
                # Tokenize to find thinking positions
                all_str_tokens = model.to_str_tokens(output_str)
                prompt_toks = model.to_tokens(prompt)[0]
                prompt_len = len(prompt_toks)
                new_tokens = all_str_tokens[prompt_len:]
                
                # Find </think> position
                think_end_pos = -1
                for i, tok in enumerate(new_tokens):
                    if "</think>" in tok or "think>" in tok:
                        think_end_pos = i
                        break
                
                # Determine effective tokens (stop at </think>)
                effective_tokens = new_tokens[:think_end_pos] if think_end_pos > 0 else new_tokens
                
                # Find thinking token positions (MI Peaks)
                thinking_positions = find_thinking_positions(effective_tokens)
                
                # Cache ALL layers at once
                layer_filter = lambda x: any(x == f"blocks.{l}.hook_resid_post" for l in test_layers)
                _, cache = model.run_with_cache(output_str, names_filter=layer_filter)
                
                # Extract all layers with thinking token mean pooling
                for l in test_layers:
                    full_acts = cache[f"blocks.{l}.hook_resid_post"][0]
                    gen_acts = full_acts[prompt_len:]
                    
                    # Apply effective range cutoff
                    if think_end_pos > 0 and think_end_pos < gen_acts.shape[0]:
                        gen_acts = gen_acts[:think_end_pos]
                    
                    # Mean pool thinking tokens (MI Peaks methodology)
                    if len(thinking_positions) > 0:
                        valid_pos = [p for p in thinking_positions if p < gen_acts.shape[0]]
                        if valid_pos:
                            act = gen_acts[valid_pos].mean(dim=0).cpu()
                        else:
                            act = gen_acts[-1].cpu()
                    else:
                        act = gen_acts[-1].cpu()
                    
                    layer_acts[l].append(act)
            else:
                # Non-reasoning model: cache all layers
                layer_filter = lambda x: any(x == f"blocks.{l}.hook_resid_post" for l in test_layers)
                _, cache = model.run_with_cache(prompt, names_filter=layer_filter)
                
                for l in test_layers:
                    act = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].cpu()
                    layer_acts[l].append(act)
            
            del cache
            torch.cuda.empty_cache()
    
    # Convert to tensors
    for l in test_layers:
        layer_acts[l] = torch.stack(layer_acts[l])
    
    print(f"✅ Cached {len(data)} samples × {len(test_layers)} layers")
    
    # ==========================================
    # STEP 2: Train Probes (Now Using Cached Data)
    # ==========================================
    layer_scores = {}
    true_idxs = [i for i, m in enumerate(data) if m['label'] == 'TRUE']
    hall_idxs = [i for i, m in enumerate(data) if m.get('type') == 'hallucination']
    
    if len(true_idxs) < 5 or len(hall_idxs) < 3:
        print(f"⚠️ Not enough samples: TRUE={len(true_idxs)}, HALL={len(hall_idxs)}")
        return None, {}
    
    print(f"\n📊 Training probes: {len(true_idxs)} TRUE, {len(hall_idxs)} HALL samples")
    
    for layer_idx in tqdm(test_layers, desc="Training probes"):
        acts = layer_acts[layer_idx]
        
        X = acts[true_idxs].to(device).float()
        depths = torch.tensor([data[i]['depth'] for i in true_idxs], device=device).float()
        targets = 0.075 * depths
        
        # Create mapper with improved architecture
        mapper = HyperbolicMapper(X.shape[1], hidden_dim=64, output_dim=16).to(device)
        
        # Improved optimizer with scheduler
        optimizer = optim.Adam(mapper.parameters(), lr=0.005)  # Lower LR for stability
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
        
        # FIXED: 500 epochs instead of 100
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(500):
            optimizer.zero_grad()
            emb = mapper(X)
            d_origin = mapper.radius(emb)
            loss = torch.mean((d_origin - targets)**2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step(loss)
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter > 100:  # Early stop if no improvement
                break
        
        with torch.no_grad():
            true_emb = mapper(X)
            hall_emb = mapper(acts[hall_idxs].to(device).float())
            true_radii = mapper.radius(true_emb)
            hall_radii = mapper.radius(hall_emb)
            
            true_r = true_radii.mean().item()
            hall_r = hall_radii.mean().item()
            true_std = true_radii.std().item()
            hall_std = hall_radii.std().item()
        
        # Simple separability (original metric)
        raw_separability = abs(true_r - hall_r)
        
        # Fisher's Linear Discriminant (more robust)
        # SNR = |mean1 - mean2| / (std1 + std2)
        # Higher = better separation
        fisher_ratio = abs(true_r - hall_r) / (true_std + hall_std + 1e-6)
        
        # Classification accuracy (threshold-based)
        # Find optimal threshold and measure accuracy
        all_radii = torch.cat([true_radii, hall_radii]).cpu().numpy()
        all_labels = [1] * len(true_radii) + [0] * len(hall_radii)  # 1=True, 0=Hall
        
        # Try thresholds and find best accuracy
        best_acc = 0.5
        best_thresh = 0.0
        for thresh in np.linspace(all_radii.min(), all_radii.max(), 50):
            # Two hypotheses: True > Hall or True < Hall
            pred_high = (all_radii > thresh).astype(int)  # True predicted if radius > thresh
            pred_low = (all_radii <= thresh).astype(int)  # True predicted if radius <= thresh
            
            acc_high = np.mean([p == l for p, l in zip(pred_high, all_labels)])
            acc_low = np.mean([p == l for p, l in zip(pred_low, all_labels)])
            
            if acc_high > best_acc:
                best_acc = acc_high
                best_thresh = thresh
            if acc_low > best_acc:
                best_acc = acc_low
                best_thresh = thresh
        
        layer_scores[layer_idx] = {
            'separability': raw_separability,
            'fisher_ratio': fisher_ratio,
            'classification_acc': best_acc,
            'true_r': true_r,
            'hall_r': hall_r,
            'true_std': true_std,
            'hall_std': hall_std,
            'best_threshold': best_thresh
        }
        
        # Direction indicator: 'T>H' means True has higher radius than Hall
        direction = 'T>H' if true_r > hall_r else 'T<H'
        print(f"  Layer {layer_idx}: True r={true_r:.3f}±{true_std:.3f}, Hall r={hall_r:.3f}±{hall_std:.3f}, "
              f"Fisher={fisher_ratio:.3f}, Acc={best_acc:.1%} [{direction}]")
    
    # Find best layer by CLASSIFICATION ACCURACY (most unbiased metric)
    best_layer_sep = max(layer_scores, key=lambda l: layer_scores[l]['separability'])
    best_layer_fisher = max(layer_scores, key=lambda l: layer_scores[l]['fisher_ratio'])
    best_layer_acc = max(layer_scores, key=lambda l: layer_scores[l]['classification_acc'])
    
    # Use classification accuracy as the PRIMARY metric (most robust)
    best_layer = best_layer_acc
    best_acc = layer_scores[best_layer]['classification_acc']
    best_fisher = layer_scores[best_layer]['fisher_ratio']
    
    current_layer = config['layer']
    current_acc = layer_scores.get(current_layer, {}).get('classification_acc', 0.5)
    current_fisher = layer_scores.get(current_layer, {}).get('fisher_ratio', 0)
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"✅ OPTIMAL LAYER (by Classification Accuracy): {best_layer_acc} ({layer_scores[best_layer_acc]['classification_acc']:.1%})")
    print(f"   OPTIMAL LAYER (by Fisher Ratio): {best_layer_fisher} ({layer_scores[best_layer_fisher]['fisher_ratio']:.3f})")
    print(f"   OPTIMAL LAYER (by Raw Separability): {best_layer_sep} ({layer_scores[best_layer_sep]['separability']:.4f})")
    print(f"\n   Current CONFIG layer: {current_layer}")
    print(f"     - Classification Acc: {current_acc:.1%}")
    print(f"     - Fisher Ratio: {current_fisher:.3f}")
    
    if best_layer == current_layer:
        print(f"   ✅ CONFIG is already optimal!")
    elif current_acc > 0.9 * best_acc:
        print(f"   ✅ CONFIG is near-optimal")
    else:
        print(f"   ⚠️ Consider updating CONFIG to layer {best_layer}")
    
    # Check for "Short-Circuit" pattern
    best_info = layer_scores[best_layer]
    if best_info['true_r'] > best_info['hall_r']:
        print(f"\n   📊 Pattern: TRUE > HALLUCINATION (Wormhole hypothesis)")
    else:
        print(f"\n   📊 Pattern: TRUE < HALLUCINATION (Short-Circuit hypothesis)")
        print(f"      This is interesting! Hallucinations may mimic 'deep' truths.")
    
    # Plot (4 panels)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    layers = sorted(layer_scores.keys())
    
    # Panel 1: Classification Accuracy (PRIMARY METRIC)
    accs = [layer_scores[l]['classification_acc'] for l in layers]
    bars = axes[0, 0].bar(layers, accs, color='steelblue')
    if best_layer_acc in layers:
        bars[layers.index(best_layer_acc)].set_color('green')
    if current_layer in layers:
        axes[0, 0].axvline(x=current_layer, color='red', linestyle='--', label=f'CONFIG L{current_layer}')
    axes[0, 0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Random')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Classification Accuracy')
    axes[0, 0].set_title(f'Best Layer by Accuracy: {best_layer_acc} ({layer_scores[best_layer_acc]["classification_acc"]:.1%})')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0.4, 1.0)
    
    # Panel 2: Fisher Discriminant Ratio
    fishers = [layer_scores[l]['fisher_ratio'] for l in layers]
    bars = axes[0, 1].bar(layers, fishers, color='darkorange')
    if best_layer_fisher in layers:
        bars[layers.index(best_layer_fisher)].set_color('green')
    if current_layer in layers:
        axes[0, 1].axvline(x=current_layer, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Fisher Ratio (SNR)')
    axes[0, 1].set_title(f'Fisher Discriminant: {best_layer_fisher} ({layer_scores[best_layer_fisher]["fisher_ratio"]:.2f})')
    
    # Panel 3: Mean Radius by Type
    true_rs = [layer_scores[l]['true_r'] for l in layers]
    hall_rs = [layer_scores[l]['hall_r'] for l in layers]
    axes[1, 0].plot(layers, true_rs, 'b-o', label='True', linewidth=2, markersize=8)
    axes[1, 0].plot(layers, hall_rs, 'r-o', label='Hallucination', linewidth=2, markersize=8)
    axes[1, 0].fill_between(layers, true_rs, hall_rs, alpha=0.2)
    if current_layer in layers:
        axes[1, 0].axvline(x=current_layer, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Mean Radius')
    axes[1, 0].set_title(f'Radius by Layer: {model_name}')
    axes[1, 0].legend()
    
    # Panel 4: Radius with Error Bars
    true_stds = [layer_scores[l]['true_std'] for l in layers]
    hall_stds = [layer_scores[l]['hall_std'] for l in layers]
    axes[1, 1].errorbar(layers, true_rs, yerr=true_stds, fmt='b-o', label='True', capsize=3)
    axes[1, 1].errorbar(layers, hall_rs, yerr=hall_stds, fmt='r-o', label='Hallucination', capsize=3)
    if current_layer in layers:
        axes[1, 1].axvline(x=current_layer, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Mean Radius ± Std')
    axes[1, 1].set_title('Radius with Variance')
    axes[1, 1].legend()
    
    plt.tight_layout()
    save_path = f"results/layer_validation/{model_name}_layer_validation.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n✅ Plot saved: {save_path}")
    
    # Save results
    results_path = f"results/layer_validation/{model_name}_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Layer Validation Results: {model_name}\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"PRIMARY METRIC: Classification Accuracy\n")
        f.write(f"Optimal Layer: {best_layer_acc} (Acc={layer_scores[best_layer_acc]['classification_acc']:.1%})\n\n")
        f.write(f"SECONDARY METRIC: Fisher Discriminant Ratio\n")
        f.write(f"Optimal Layer: {best_layer_fisher} (Fisher={layer_scores[best_layer_fisher]['fisher_ratio']:.3f})\n\n")
        f.write(f"Current CONFIG Layer: {current_layer}\n")
        f.write(f"Current Acc: {current_acc:.1%}, Fisher: {current_fisher:.3f}\n\n")
        f.write(f"All Layers:\n")
        for l in sorted(layer_scores.keys()):
            s = layer_scores[l]
            direction = 'T>H' if s['true_r'] > s['hall_r'] else 'T<H'
            f.write(f"  L{l}: Acc={s['classification_acc']:.1%}, Fisher={s['fisher_ratio']:.2f}, "
                    f"True={s['true_r']:.3f}±{s['true_std']:.3f}, Hall={s['hall_r']:.3f}±{s['hall_std']:.3f} [{direction}]\n")
    print(f"✅ Results saved: {results_path}")
    
    return best_layer, layer_scores

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal probing layer for a model")
    parser.add_argument("--model", type=str, default="qwen", choices=["qwen", "deepseek"],
                        help="Model to validate")
    parser.add_argument("--samples", type=int, default=150, 
                        help="Number of samples (default: 150, min for stats significance)")
    args = parser.parse_args()
    
    print(f"\n📊 Layer Validation for {args.model.upper()}")
    print(f"   Samples: {args.samples}")
    
    # Generate BALANCED data
    print("\nGenerating balanced data...")
    gen = LogicTreeGenerator(mode='fiction', seed=42)
    data = generate_balanced_dataset(gen, args.samples, seed=42)
    
    # Load model
    model = get_model(args.model)
    
    # Run validation
    best_layer, scores = run_layer_validation(model, data, args.model)
    
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION")
    print(f"{'='*60}")
    print(f"Use layer {best_layer} for {args.model} in CONFIG")
