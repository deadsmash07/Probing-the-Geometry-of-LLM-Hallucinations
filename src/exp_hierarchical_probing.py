"""

Key differences from our previous approach:
1. PAIRWISE distance loss: Σᵢⱼ (||Hhᵢ - Hhⱼ|| - Dᵢⱼ)²
2. Depth probe: Logistic regression (not MSE regression)
3. Datasets: Dyck strings, Binary trees (not just PrOntoQA)
4. Analysis: Layer sweep with Pearson correlation

Usage:
    python run_hprobes_replication.py --model qwen --experiment dyck
    python run_hprobes_replication.py --model deepseek --experiment binary_tree
    python run_hprobes_replication.py --model qwen --experiment all
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import geoopt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from collections import defaultdict

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")
os.makedirs("results/hprobes", exist_ok=True)

# Model-Specific Configuration
CONFIG = {
    'qwen': {
        'layer': 23,                  # Optimal layer for Qwen (validated)
        'samples_default': 300,       # Default sample count
        'dyck_samples': 300,          # Dyck string samples
        'dyck_max_depth': 5,          # Max bracket depth
        'binary_tree_samples': 300,   # Binary tree samples
        'binary_tree_depth': 5,       # Tree depth
    },
    'deepseek': {
        'layer': 23,                  # Optimal layer for DeepSeek
        'samples_default': 300,       # Default sample count
        'dyck_samples': 300,          # Dyck string samples
        'dyck_max_depth': 5,          # Max bracket depth
        'binary_tree_samples': 300,   # Binary tree samples
        'binary_tree_depth': 5,       # Tree depth
    }
}

# Auth - Load HF_TOKEN from environment
if not os.environ.get("HF_TOKEN"):
    print("Warning: HF_TOKEN not found in environment. Please set it in .env file.")

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

# ==========================================
# 1. DATASET GENERATORS
# ==========================================

class DyckGenerator:
    """
    Generate Dyck string completion tasks.
    
    Dyck strings are balanced bracket sequences like:
    "((()[]))" or "[{()}]"
    
    We track the depth at each position for probing.
    """
    def __init__(self, max_depth=4, num_samples=400, seed=42):
        self.max_depth = max_depth
        self.num_samples = num_samples
        self.rng = random.Random(seed)
        self.brackets = [('(', ')'), ('[', ']')]
    
    def generate_dyck_string(self, target_length=20):
        """Generate a random balanced Dyck string with depth tracking."""
        result = []
        depths = []
        stack = []
        current_depth = 0
        
        while len(result) < target_length:
            # Decide: open or close?
            can_open = current_depth < self.max_depth
            can_close = len(stack) > 0
            
            if can_open and (not can_close or self.rng.random() < 0.6):
                # Open a bracket
                bracket_type = self.rng.choice(self.brackets)
                result.append(bracket_type[0])
                stack.append(bracket_type[1])
                current_depth += 1
                depths.append(current_depth)
            elif can_close:
                # Close a bracket
                result.append(stack.pop())
                current_depth -= 1
                depths.append(current_depth)
            else:
                break
        
        # Close remaining brackets
        while stack:
            result.append(stack.pop())
            current_depth -= 1
            depths.append(current_depth)
        
        return ''.join(result), depths
    
    def generate_dataset(self):
        """Generate dataset of Dyck strings with depth labels."""
        samples = []
        for _ in range(self.num_samples):
            length = self.rng.randint(10, 30)
            dyck_str, depths = self.generate_dyck_string(length)
            
            # Create completion task: give prefix, predict next bracket
            if len(dyck_str) > 5:
                split_point = self.rng.randint(3, len(dyck_str) - 2)
                prefix = dyck_str[:split_point]
                next_char = dyck_str[split_point]
                depth_at_position = depths[split_point - 1] if split_point > 0 else 0
                
                samples.append({
                    'prefix': prefix,
                    'next': next_char,
                    'depth': depth_at_position,
                    'full_string': dyck_str
                })
        
        return samples


class BinaryTreeGenerator:
    """
    Generate binary tree traversal tasks FROM A SINGLE SHARED TREE.
    
    CRITICAL FIX: H-Probes requires pairwise distances between samples.
    This only makes sense when samples reference nodes in the SAME tree.
    
    Following the paper: "500 random start and end node pairs drawn from 
    trees of depths 1–4"
    """
    def __init__(self, tree_depth=5, num_samples=400, seed=42):
        self.tree_depth = tree_depth
        self.num_samples = num_samples
        self.rng = random.Random(seed)
        
        # Generate ONE shared tree for all samples
        self.nodes, self.children, self.tree_structure = self._generate_shared_tree()
        
    def _generate_shared_tree(self):
        """Generate a single full binary tree that all samples will reference."""
        # Nodes are numbered 1, 2, 3, ... (level order)
        num_nodes = (2 ** self.tree_depth) - 1
        nodes = list(range(1, num_nodes + 1))
        
        # Compute parent-child relationships
        children = {}
        for i in nodes:
            left = 2 * i
            right = 2 * i + 1
            if left <= num_nodes:
                children[i] = [left, right] if right <= num_nodes else [left]
            else:
                children[i] = []
        
        # Build edge description (shared across all prompts)
        edge_descriptions = []
        for parent, childs in children.items():
            for child in childs:
                edge_descriptions.append(f"Node {parent} connects to Node {child}")
        
        # Shuffle but keep consistent (seeded)
        self.rng.shuffle(edge_descriptions)
        tree_structure = ". ".join(edge_descriptions)
        
        return nodes, children, tree_structure
    
    def tree_distance(self, node1, node2):
        """Compute tree distance (path length) between two nodes."""
        # Find path from each node to root, then compute LCA distance
        def path_to_root(n):
            path = []
            while n >= 1:
                path.append(n)
                n = n // 2
            return path
        
        path1 = path_to_root(node1)
        path2 = path_to_root(node2)
        
        # Find LCA (lowest common ancestor)
        set1 = set(path1)
        lca = None
        for n in path2:
            if n in set1:
                lca = n
                break
        
        if lca is None:
            return float('inf')
        
        # Distance = depth(node1) - depth(lca) + depth(node2) - depth(lca)
        dist1 = path1.index(lca)
        dist2 = path2.index(lca)
        return dist1 + dist2
    
    def node_depth(self, node):
        """Compute depth of a node (root = depth 0)."""
        depth = 0
        while node > 1:
            node = node // 2
            depth += 1
        return depth
    
    def generate_dataset(self):
        """Generate dataset of node pairs with tree distances - ALL FROM SAME TREE."""
        samples = []
        
        for _ in range(self.num_samples):
            # Sample random pair FROM THE SHARED TREE
            node1, node2 = self.rng.sample(self.nodes, 2)
            dist = self.tree_distance(node1, node2)
            depth1 = self.node_depth(node1)
            depth2 = self.node_depth(node2)
            
            # Create traversal prompt WITH THE SHARED TREE STRUCTURE
            prompt = (
                f"Binary tree structure: {self.tree_structure}. "
                f"Find the path from Node {node1} to Node {node2}."
            )
            
            samples.append({
                'prompt': prompt,
                'node1': node1,
                'node2': node2,
                'distance': dist,  # ACTUAL tree path distance
                'depth1': depth1,
                'depth2': depth2,
                'tree_depth': self.tree_depth,
            })
        
        return samples
    
    def compute_pairwise_distances(self, samples):
        """
        Compute the TRUE pairwise distance matrix between all samples.
        
        CRITICAL: Uses actual tree path distance, NOT depth difference.
        
        For samples i and j:
        - If they reference different node pairs, distance = 0 (they represent the same query type)
        - The H-Probes loss compares activations based on NODE distances within the tree
        """
        N = len(samples)
        distance_matrix = torch.zeros(N, N)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    # Use actual tree path distance between the PRIMARY nodes
                    # (node1 of sample i vs node1 of sample j)
                    dist = self.tree_distance(samples[i]['node1'], samples[j]['node1'])
                    distance_matrix[i, j] = dist
        
        return distance_matrix


# ==========================================
# 2. PROBES (H-Probes Paper Methodology)
# ==========================================

class PairwiseDistanceProbe(nn.Module):
    """
    Euclidean Pairwise Distance Probe.
    
    H-probes paper methodology:
    L = Σᵢⱼ (||H·hᵢ - H·hⱼ|| - Dᵢⱼ)²
    
    Where H is a learnable projection matrix.
    """
    def __init__(self, input_dim, proj_dim=16):
        super().__init__()
        self.H = nn.Linear(input_dim, proj_dim, bias=False)
        nn.init.xavier_uniform_(self.H.weight)
    
    def forward(self, h):
        return self.H(h)
    
    def compute_pairwise_distances(self, z):
        """Compute all pairwise Euclidean distances."""
        # z: [N, proj_dim]
        N = z.shape[0]
        z_i = z.unsqueeze(1)  # [N, 1, proj_dim]
        z_j = z.unsqueeze(0)  # [1, N, proj_dim]
        return torch.norm(z_i - z_j, dim=-1)  # [N, N]
    
    def pairwise_loss(self, h_batch, distance_matrix):
        """Compute pairwise distance loss."""
        z = self.forward(h_batch)
        pred_dist = self.compute_pairwise_distances(z)
        
        # MSE loss, only upper triangle (avoid double counting)
        N = z.shape[0]
        mask = torch.triu(torch.ones(N, N, device=z.device), diagonal=1)
        loss = ((pred_dist - distance_matrix) ** 2 * mask).sum() / mask.sum()
        return loss, pred_dist


class HyperbolicPairwiseProbe(nn.Module):
    """
    Hyperbolic Pairwise Distance Probe.
    
    Same as Euclidean but uses Poincaré ball distance.
    """
    def __init__(self, input_dim, proj_dim=16, c=1.0):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=c)
        self.H = nn.Linear(input_dim, proj_dim, bias=False)
        # Small init for stability
        nn.init.normal_(self.H.weight, std=0.01)
        self.scale = 0.1  # Keep outputs small for expmap stability
    
    def forward(self, h):
        v = self.H(h) * self.scale
        return self.manifold.expmap0(v)
    
    def compute_pairwise_distances(self, z):
        """
        Compute all pairwise hyperbolic distances - OPTIMIZED.
        
        Instead of O(N²) Python loop, use vectorized geoopt operations.
        """
        N = z.shape[0]
        # Vectorized: compute all pairs at once
        # Expand z to [N, 1, D] and [1, N, D]
        z_i = z.unsqueeze(1).expand(N, N, -1)  # [N, N, D]
        z_j = z.unsqueeze(0).expand(N, N, -1)  # [N, N, D]
        
        # Reshape for batch distance computation
        z_i_flat = z_i.reshape(N * N, -1)  # [N*N, D]
        z_j_flat = z_j.reshape(N * N, -1)  # [N*N, D]
        
        # Compute all distances at once
        dists_flat = self.manifold.dist(z_i_flat, z_j_flat)  # [N*N]
        pred_dist = dists_flat.reshape(N, N)  # [N, N]
        
        return pred_dist
    
    def pairwise_loss(self, h_batch, distance_matrix):
        """Compute pairwise hyperbolic distance loss."""
        z = self.forward(h_batch)
        pred_dist = self.compute_pairwise_distances(z)
        
        N = z.shape[0]
        mask = torch.triu(torch.ones(N, N, device=z.device), diagonal=1)
        loss = ((pred_dist - distance_matrix) ** 2 * mask).sum() / mask.sum()
        return loss, pred_dist


class DepthProbe:
    """
    Depth Probe using Logistic Regression.
    
    Following the paper: "For tree depth: Perform logistic regression"
    """
    def __init__(self):
        self.classifier = None
    
    def fit(self, X, y):
        """Fit logistic regression."""
        self.classifier = LogisticRegression(max_iter=1000)
        self.classifier.fit(X, y)
    
    def predict(self, X):
        """Predict depth labels."""
        return self.classifier.predict(X)
    
    def score(self, X, y):
        """Compute accuracy."""
        return self.classifier.score(X, y)


# ==========================================
# 3. MODEL LOADING
# ==========================================

def get_model(model_type="standard"):
    """Load model - supports both Qwen (standard) and DeepSeek (reasoning)"""
    if model_type == "standard" or model_type == "qwen":
        print("=" * 60)
        print("🔧 Loading Standard Model: Qwen/Qwen2.5-7B-Instruct")
        print("=" * 60)
        return HookedTransformer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device=device)
    elif model_type == "reasoning" or model_type == "deepseek":
        print("=" * 60)
        print("🔧 Loading Reasoning Model: DeepSeek-R1-Distill-Qwen-7B")
        print("=" * 60)
        # Load HF model to CPU first to avoid OOM when wrapping
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
            fold_value_biases=False,   # Disable to avoid meta tensor issues
            center_writing_weights=False,
            center_unembed=False
        )
        
        # Free CPU memory
        del hf_model
        torch.cuda.empty_cache()
        
        return hooked_model


# ==========================================
# 4. ACTIVATION EXTRACTION
# ==========================================

def extract_activations_all_layers(model, prompts, max_samples=100):
    """
    Extract activations from ALL layers for layer sweep analysis.
    
    Returns: dict[layer_idx] -> tensor of shape [N, hidden_dim]
    """
    n_layers = model.cfg.n_layers
    all_acts = {i: [] for i in range(n_layers)}
    
    print(f"Extracting activations from all {n_layers} layers...")
    for prompt in tqdm(prompts[:max_samples]):
        with torch.no_grad():
            _, cache = model.run_with_cache(
                prompt, 
                names_filter=lambda x: x.endswith("hook_resid_post")
            )
            for layer_idx in range(n_layers):
                act = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1, :].cpu()
                all_acts[layer_idx].append(act)
            del cache
            torch.cuda.empty_cache()
    
    # Stack into tensors
    for layer_idx in range(n_layers):
        all_acts[layer_idx] = torch.stack(all_acts[layer_idx])
    
    return all_acts


# ==========================================
# 5. TRAINING FUNCTIONS
# ==========================================

def train_pairwise_probe(probe, activations, distance_matrix, epochs=200, lr=0.01):
    """Train a pairwise distance probe with LR scheduler and early stopping."""
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5, min_lr=1e-5)
    
    activations = activations.to(device).float()
    distance_matrix = distance_matrix.to(device).float()
    
    losses = []
    best_corr = -1.0
    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 50
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, pred_dist = probe.pairwise_loss(activations, distance_matrix)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step(loss)  # LR scheduler
        losses.append(loss.item())
        
        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            if epoch % 50 != 0:  # Print if we haven't already
                mask = torch.triu(torch.ones_like(distance_matrix), diagonal=1).bool()
                pred_flat = pred_dist[mask].detach().cpu().numpy()
                true_flat = distance_matrix[mask].cpu().numpy()
                corr, _ = pearsonr(pred_flat, true_flat)
                print(f"  Epoch {epoch}: Loss = {loss.item():.4f}, Pearson r = {corr:.4f} (early stop)")
            break
        
        if epoch % 50 == 0:
            # Compute correlation
            mask = torch.triu(torch.ones_like(distance_matrix), diagonal=1).bool()
            pred_flat = pred_dist[mask].detach().cpu().numpy()
            true_flat = distance_matrix[mask].cpu().numpy()
            corr, _ = pearsonr(pred_flat, true_flat)
            best_corr = max(best_corr, corr)
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}, Pearson r = {corr:.4f}")
    
    return losses, best_corr


def train_with_hyperparameter_sweep(probe_cls, input_dim, proj_dim, activations, distance_matrix):
    """
    Train probe with hyperparameter sweep to ensure FAIR comparison.
    
    This addresses the concern that Euclidean "failure" might be optimization artifact.
    We try multiple learning rates and epochs to find the best configuration.
    
    Returns the best probe and its correlation.
    """
    print("  Running hyperparameter sweep for fair comparison...")
    
    learning_rates = [0.001, 0.01, 0.05]
    epoch_counts = [200, 400]
    
    best_probe = None
    best_corr = -1.0
    best_config = {}
    
    for lr in learning_rates:
        for epochs in epoch_counts:
            # Create fresh probe
            probe = probe_cls(input_dim, proj_dim).to(device)
            
            # Train
            _, corr = train_pairwise_probe(probe, activations, distance_matrix, 
                                           epochs=epochs, lr=lr)
            
            print(f"    Config lr={lr}, epochs={epochs}: r = {corr:.4f}")
            
            if corr > best_corr:
                best_corr = corr
                best_probe = probe
                best_config = {'lr': lr, 'epochs': epochs}
    
    print(f"  ✅ Best config: lr={best_config['lr']}, epochs={best_config['epochs']}, r={best_corr:.4f}")
    return best_probe, best_corr, best_config


def compute_distance_correlation(probe, activations, distance_matrix):
    """Compute Pearson correlation between predicted and true distances."""
    probe.eval()
    with torch.no_grad():
        activations = activations.to(device).float()
        distance_matrix = distance_matrix.to(device).float()
        
        z = probe.forward(activations)
        pred_dist = probe.compute_pairwise_distances(z)
        
        mask = torch.triu(torch.ones_like(distance_matrix), diagonal=1).bool()
        pred_flat = pred_dist[mask].cpu().numpy()
        true_flat = distance_matrix[mask].cpu().numpy()
        
        corr, pval = pearsonr(pred_flat, true_flat)
    return corr, pval


# ==========================================
# 6. EXPERIMENTS
# ==========================================

def run_dyck_experiment(model, model_name):
    """Run Dyck string completion experiment."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Dyck String Completion")
    print("=" * 60)
    
    # Get CONFIG values
    config = CONFIG[model_name]
    dyck_samples = config['dyck_samples']
    dyck_max_depth = config['dyck_max_depth']
    
    # Generate data using CONFIG
    generator = DyckGenerator(max_depth=dyck_max_depth, num_samples=dyck_samples)
    data = generator.generate_dataset()
    
    print(f"  📊 Using CONFIG: dyck_samples={dyck_samples}, dyck_max_depth={dyck_max_depth}")
    
    # Create prompts
    prompts = [f"Complete the Dyck string: {d['prefix']}" for d in data]
    depths = [d['depth'] for d in data]
    
    # Extract activations from all layers (use full dataset)
    all_acts = extract_activations_all_layers(model, prompts, max_samples=len(prompts))
    # depths already full, no slicing needed
    
    # Depth probe (logistic regression) per layer
    print("\n📊 Depth Probe (Logistic Regression) per Layer:")
    depth_correlations = []
    for layer_idx in range(model.cfg.n_layers):
        X = all_acts[layer_idx].numpy()
        y = np.array(depths)
        
        # Split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        probe = DepthProbe()
        probe.fit(X_train, y_train)
        
        # Compute correlation (between predicted and true depths)
        y_pred = probe.predict(X_test)
        corr, _ = pearsonr(y_pred, y_test)
        depth_correlations.append(corr)
        
        if layer_idx % 5 == 0:
            acc = accuracy_score(y_test, y_pred)
            print(f"  Layer {layer_idx}: Accuracy = {acc:.2%}, Pearson r = {corr:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(model.cfg.n_layers), depth_correlations, 'b-o', label='Depth Probe')
    plt.xlabel('Layer')
    plt.ylabel('Pearson Correlation')
    plt.title(f'Dyck String Depth Probe - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/hprobes/{model_name}_dyck_depth_correlation.png', dpi=150)
    plt.close()
    
    print(f"\n✅ Saved: results/hprobes/{model_name}_dyck_depth_correlation.png")
    return depth_correlations


def run_binary_tree_experiment(model, model_name):
    """Run binary tree traversal experiment with PAIRWISE distance probes."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Binary Tree Traversal (Pairwise Distance)")
    print("=" * 60)
    
    # Get CONFIG values
    config = CONFIG[model_name]
    tree_samples = config['binary_tree_samples']
    tree_depth = config['binary_tree_depth']
    
    # Generate data FROM A SINGLE SHARED TREE using CONFIG
    generator = BinaryTreeGenerator(tree_depth=tree_depth, num_samples=tree_samples)
    data = generator.generate_dataset()
    
    print(f"  📊 Using CONFIG: binary_tree_samples={tree_samples}, binary_tree_depth={tree_depth}")
    print(f"  📊 Generated {len(data)} samples from a shared depth-{generator.tree_depth} tree")
    print(f"  📊 Tree has {len(generator.nodes)} nodes")
    
    # Create prompts
    prompts = [d['prompt'] for d in data]
    
    # Build distance matrix using ACTUAL TREE PATH DISTANCES
    # CRITICAL FIX: Uses path length, NOT depth difference
    print("  📊 Computing pairwise tree distances (actual path lengths)...")
    distance_matrix = generator.compute_pairwise_distances(data)
    
    # Extract activations
    all_acts = extract_activations_all_layers(model, prompts, max_samples=len(data))
    
    # Train probes on middle layer
    target_layer = model.cfg.n_layers // 2
    acts = all_acts[target_layer]
    
    print(f"\n📊 Training Pairwise Probes on Layer {target_layer}:")
    print("  Using hyperparameter sweep for FAIR comparison (addresses optimizer artifact concern)")
    
    # TRAIN/TEST SPLIT: 80/20 to measure generalization, not memorization
    N = len(data)
    split_idx = int(0.8 * N)
    train_acts = acts[:split_idx]
    test_acts = acts[split_idx:]
    train_dist = distance_matrix[:split_idx, :split_idx]
    test_dist = distance_matrix[split_idx:, split_idx:]
    
    print(f"  Train: {len(train_acts)} samples, Test: {len(test_acts)} samples")
    
    # Euclidean probe WITH HYPERPARAMETER SWEEP
    print("\n  🔷 Euclidean Pairwise Probe (with hyperparameter sweep):")
    euc_probe, euc_corr_train, euc_config = train_with_hyperparameter_sweep(
        PairwiseDistanceProbe, train_acts.shape[1], 16, train_acts, train_dist
    )
    # Evaluate on TEST set
    euc_corr_test, _ = compute_distance_correlation(euc_probe, test_acts, test_dist)
    
    # Hyperbolic probe WITH HYPERPARAMETER SWEEP
    print("\n  🔶 Hyperbolic Pairwise Probe (with hyperparameter sweep):")
    hyp_probe, hyp_corr_train, hyp_config = train_with_hyperparameter_sweep(
        HyperbolicPairwiseProbe, train_acts.shape[1], 16, train_acts, train_dist
    )
    # Evaluate on TEST set
    hyp_corr_test, _ = compute_distance_correlation(hyp_probe, test_acts, test_dist)
    
    print(f"\n📊 Results (GENERALIZATION on held-out test set):")
    print(f"  Euclidean:  Train r={euc_corr_train:.4f}, Test r={euc_corr_test:.4f}")
    print(f"  Hyperbolic: Train r={hyp_corr_train:.4f}, Test r={hyp_corr_test:.4f}")
    
    # Use TEST correlation for comparison (not train!)
    euc_corr = euc_corr_test
    hyp_corr = hyp_corr_test
    
    # Layer sweep
    print("\n📊 Layer Sweep (Euclidean Probe):")
    layer_correlations = []
    for layer_idx in tqdm(range(model.cfg.n_layers)):
        acts = all_acts[layer_idx]
        probe = PairwiseDistanceProbe(acts.shape[1], proj_dim=16).to(device)
        
        # Quick training
        optimizer = optim.Adam(probe.parameters(), lr=0.01)
        acts_dev = acts.to(device).float()
        dist_dev = distance_matrix.to(device).float()
        
        for _ in range(100):
            optimizer.zero_grad()
            loss, _ = probe.pairwise_loss(acts_dev, dist_dev)
            loss.backward()
            optimizer.step()
        
        corr, _ = compute_distance_correlation(probe, acts, distance_matrix)
        layer_correlations.append(corr)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(model.cfg.n_layers), layer_correlations, 'r-o', label='Distance Probe (Euclidean)')
    plt.xlabel('Layer')
    plt.ylabel('Pearson Correlation')
    plt.title(f'Binary Tree Distance Probe - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/hprobes/{model_name}_binary_tree_correlation.png', dpi=150)
    plt.close()
    
    print(f"\n✅ Saved: results/hprobes/{model_name}_binary_tree_correlation.png")
    
    # Euclidean vs Hyperbolic comparison
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Euclidean', 'Hyperbolic'], [euc_corr, hyp_corr], 
                   color=['steelblue', 'darkorange'])
    plt.ylabel('Pearson Correlation')
    plt.title(f'Geometry Comparison: {model_name}')
    plt.ylim(0, 1)
    for bar, val in zip(bars, [euc_corr, hyp_corr]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{val:.3f}', ha='center')
    plt.savefig(f'results/hprobes/{model_name}_geometry_comparison.png', dpi=150)
    plt.close()
    
    print(f"✅ Saved: results/hprobes/{model_name}_geometry_comparison.png")
    
    return layer_correlations, euc_corr, hyp_corr


def run_hallucination_extension(model, model_name):
    """
    EXTENSION: Apply H-probes methodology to hallucination detection.
    
    CRITICAL FIX: Train ONLY on TRUE-TRUE pairs, then TEST on TRUE-HALLUCINATION.
    This avoids circular training logic where we explicitly push hallucinations away.
    
    Scientific hypothesis:
    - If the probe learns valid reasoning geometry from TRUE samples,
    - Then hallucinations should naturally fall "off the manifold" 
    - (Either very large distance or anomalous small distance = "short circuit")
    """
    print("\n" + "=" * 60)
    print("EXTENSION: Hallucination Detection with H-Probes")
    print("=" * 60)
    print("⚠️  METHODOLOGY: Train ONLY on TRUE, Test on HALLUCINATION")
    print("   (This avoids circular training logic)")
    
    # Import our existing data generator
    from run_full_experiment import LogicTreeGenerator
    
    # Generate data
    gen = LogicTreeGenerator(mode='fiction', depth=5, seed=42)
    data = []
    for _ in range(300):
        samples = gen.generate_sample()
        if samples:
            data.extend(samples)
    
    # Separate by type
    true_samples = [d for d in data if d['label'] == 'TRUE'][:100]
    hall_samples = [d for d in data if d['type'] == 'hallucination'][:100]
    
    n_true = len(true_samples)
    n_hall = len(hall_samples)
    
    print(f"  TRAIN set: {n_true} TRUE samples (only)")
    print(f"  TEST set: {n_hall} HALLUCINATION samples (held out)")
    
    # Create prompts
    true_prompts = [f"{d['context']}\n{d['query']}" for d in true_samples]
    hall_prompts = [f"{d['context']}\n{d['query']}" for d in hall_samples]
    
    # Extract activations
    target_layer = CONFIG[model_name]['layer']
    print(f"\n📊 Extracting activations from Layer {target_layer}...")
    
    true_acts = []
    for prompt in tqdm(true_prompts, desc="TRUE"):
        with torch.no_grad():
            _, cache = model.run_with_cache(
                prompt, names_filter=lambda x: x.endswith("hook_resid_post"))
            act = cache[f"blocks.{target_layer}.hook_resid_post"][0, -1, :].cpu()
            true_acts.append(act)
            del cache
    true_acts = torch.stack(true_acts)
    
    hall_acts = []
    for prompt in tqdm(hall_prompts, desc="HALLUCINATION"):
        with torch.no_grad():
            _, cache = model.run_with_cache(
                prompt, names_filter=lambda x: x.endswith("hook_resid_post"))
            act = cache[f"blocks.{target_layer}.hook_resid_post"][0, -1, :].cpu()
            hall_acts.append(act)
            del cache
    hall_acts = torch.stack(hall_acts)
    
    # Build TRAINING distance matrix (TRUE-TRUE only)
    # Distance = depth difference (logical hop distance)
    print("\n📊 Building TRUE-TRUE distance matrix (for training)...")
    train_dist = torch.zeros(n_true, n_true)
    for i in range(n_true):
        for j in range(n_true):
            if i == j:
                train_dist[i, j] = 0
            else:
                d1 = true_samples[i]['depth']
                d2 = true_samples[j]['depth']
                train_dist[i, j] = abs(d1 - d2)
    
    # Train probe ONLY on TRUE samples
    print("\n📊 Training Pairwise Probe on TRUE samples ONLY:")
    print("   (NOT optimizing on hallucination - this is the unsupervised test)")
    probe = PairwiseDistanceProbe(true_acts.shape[1], proj_dim=16).to(device)
    train_pairwise_probe(probe, true_acts, train_dist, epochs=200)
    
    # NOW: Apply trained probe to BOTH true and hallucination
    probe.eval()
    all_acts = torch.cat([true_acts, hall_acts], dim=0)
    
    with torch.no_grad():
        z = probe.forward(all_acts.to(device).float())
        pred_dist = probe.compute_pairwise_distances(z).cpu()
    
    # Analyze distances
    true_idxs = list(range(n_true))
    hall_idxs = list(range(n_true, n_true + n_hall))
    
    # Group TRUE by depth
    true_1hop = [i for i, s in enumerate(true_samples) if s['depth'] == 1]
    true_5hop = [i for i, s in enumerate(true_samples) if s['depth'] >= 4]
    
    dist_1hop_1hop = pred_dist[np.ix_(true_1hop, true_1hop)].mean().item() if len(true_1hop) > 1 else 0
    dist_5hop_5hop = pred_dist[np.ix_(true_5hop, true_5hop)].mean().item() if len(true_5hop) > 1 else 0
    dist_1hop_5hop = pred_dist[np.ix_(true_1hop, true_5hop)].mean().item() if len(true_1hop) > 0 and len(true_5hop) > 0 else 0
    
    # Critical: TRUE-HALLUCINATION distance (NOT trained on this!)
    dist_true_hall = pred_dist[np.ix_(true_idxs, hall_idxs)].mean().item()
    dist_hall_hall = pred_dist[np.ix_(hall_idxs, hall_idxs)].mean().item()
    
    print(f"\n📊 RESULTS (Probe trained on TRUE only):")
    print(f"  TRAIN SET (TRUE-TRUE):")
    print(f"    1-hop ↔ 1-hop: {dist_1hop_1hop:.3f}")
    print(f"    5-hop ↔ 5-hop: {dist_5hop_5hop:.3f}")
    print(f"    1-hop ↔ 5-hop: {dist_1hop_5hop:.3f}")
    print(f"\n  TEST SET (held out - NOT used in training):")
    print(f"    TRUE ↔ HALLUCINATION: {dist_true_hall:.3f}")
    print(f"    HALLUCINATION ↔ HALLUCINATION: {dist_hall_hall:.3f}")
    
    # Interpretation
    print(f"\n📊 INTERPRETATION:")
    if dist_true_hall > dist_1hop_5hop * 1.5:
        print(f"  ✅ Hallucinations are FURTHER from truths than deep truths are from shallow truths!")
        print(f"     → Probe detects hallucinations WITHOUT being trained on them!")
    elif dist_true_hall < dist_1hop_1hop:
        print(f"  ⚠️ Hallucinations SHORT-CIRCUIT: Closer than 1-hop truths!")
        print(f"     → 'Lazy hallucination' hypothesis supported")
    else:
        print(f"  ≈ Hallucinations fall within the TRUE distribution")
        print(f"     → Probe cannot distinguish (hypothesis fails)")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['1-hop↔1-hop\n(TRAIN)', '5-hop↔5-hop\n(TRAIN)', '1-hop↔5-hop\n(TRAIN)', 
                  'TRUE↔HALL\n(TEST)', 'HALL↔HALL\n(TEST)']
    values = [dist_1hop_1hop, dist_5hop_5hop, dist_1hop_5hop, dist_true_hall, dist_hall_hall]
    colors = ['lightblue', 'steelblue', 'royalblue', 'salmon', 'darkred']
    
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel('Mean Pairwise Distance (Projected)')
    ax.set_title(f'H-Probes Hallucination Analysis - {model_name}\n(Trained on TRUE only, tested on HALLUCINATION)')
    ax.set_ylim(0, max(values) * 1.3)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.2f}', ha='center', fontsize=10)
    
    # Add line separating train/test
    ax.axvline(x=2.5, color='black', linestyle='--', alpha=0.7)
    ax.text(1.0, max(values) * 1.1, 'TRAIN', ha='center', fontweight='bold')
    ax.text(3.5, max(values) * 1.1, 'TEST', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'results/hprobes/{model_name}_hallucination_pairwise.png', dpi=150)
    plt.close()
    
    print(f"\n✅ Saved: results/hprobes/{model_name}_hallucination_pairwise.png")
    
    return {
        '1hop_1hop': dist_1hop_1hop,
        '5hop_5hop': dist_5hop_5hop,
        '1hop_5hop': dist_1hop_5hop,
        'true_hall': dist_true_hall,
        'hall_hall': dist_hall_hall
    }


# ==========================================
# 7. MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="H-Probes Paper Replication")
    parser.add_argument('--model', type=str, default='qwen', 
                        choices=['qwen', 'deepseek'],
                        help='Model to use')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['dyck', 'binary_tree', 'hallucination', 'all'],
                        help='Experiment to run')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    
    # Set global seed FIRST for reproducibility
    set_global_seed(args.seed)
    
    # Load model
    model = get_model(args.model)
    model_name = args.model
    
    results = {}
    
    if args.experiment in ['dyck', 'all']:
        results['dyck'] = run_dyck_experiment(model, model_name)
    
    if args.experiment in ['binary_tree', 'all']:
        results['binary_tree'] = run_binary_tree_experiment(model, model_name)
    
    if args.experiment in ['hallucination', 'all']:
        results['hallucination'] = run_hallucination_extension(model, model_name)
    
    print("\n" + "=" * 60)
    print(f"H-PROBES EXPERIMENTS COMPLETE ({model_name.upper()})")
    print("=" * 60)
    print(f"Results saved to: results/hprobes/")


if __name__ == "__main__":
    main()
