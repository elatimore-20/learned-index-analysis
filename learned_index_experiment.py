import os
import sys
import time
import random
import bisect
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DATASET_SIZES = [50_000, 100_000, 200_000]
NUM_QUERIES = 3000
NUM_TRIALS = 2

ZIPF_A = 1.5
RMI_NUM_EXPERTS = 10
BTREE_T = 32

BABY_NAMES_PATH = "baby_names.csv"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def now():
    return time.perf_counter()


def safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def deep_sizeof(obj, seen=None):
    """Rough recursive memory estimate."""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)
    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(deep_sizeof(k, seen) + deep_sizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset, deque)):
        size += sum(deep_sizeof(x, seen) for x in obj)
    elif hasattr(obj, "__dict__"):
        size += deep_sizeof(vars(obj), seen)

    return size


def sample_existing_queries(keys, num_queries=3000):
    """Sample keys that definitely exist."""
    idx = np.random.choice(len(keys), size=min(num_queries, len(keys)), replace=False)
    sampled_keys = keys[idx]
    true_positions = {int(k): int(i) for k, i in zip(sampled_keys, idx)}
    return sampled_keys.tolist(), true_positions


def bounded_binary_search(keys, q, lo, hi):
    lo = max(0, lo)
    hi = min(len(keys) - 1, hi)

    while lo <= hi:
        mid = (lo + hi) // 2

        if keys[mid] == q:
            return mid
        elif keys[mid] < q:
            lo = mid + 1
        else:
            hi = mid - 1

    return -1


# ============================================================
# DATASETS
# ============================================================

def generate_uniform_dataset(n):
    """
    Synthetic uniform dataset:
    sorted unique integers with near-linear CDF.
    """
    return np.arange(1, n + 1, dtype=np.int64)


def generate_zipf_dataset(n, a=1.5):
    """
    Synthetic Zipf dataset:
    skewed values, duplicates removed, sorted.
    """
    target = n
    size = max(n * 5, 1000)

    values = np.random.zipf(a=a, size=size)
    keys = np.unique(values.astype(np.int64))

    while len(keys) < target:
        extra = np.random.zipf(a=a, size=size)
        keys = np.unique(np.concatenate([keys, extra.astype(np.int64)]))

    keys = np.sort(keys[:target])
    return keys.astype(np.int64)


def load_baby_names_dataset(n, csv_path="baby_names.csv"):
    """
    Loads U.S. Baby Names dataset if available.
    Expected columns include Year and Name.

    If the file is missing, the code uses a pseudo-real-world fallback
    so the experiment still runs tonight.
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] Baby names file '{csv_path}' not found.")
        print("[WARN] Using pseudo real-world fallback dataset.")

        years = np.random.randint(1880, 2025, size=n * 2)
        base = np.random.randint(1, 200000, size=n * 2)
        keys = np.unique(years.astype(np.int64) * 1_000_000 + base.astype(np.int64))

        while len(keys) < n:
            years = np.random.randint(1880, 2025, size=n)
            base = np.random.randint(1, 200000, size=n)
            extra = years.astype(np.int64) * 1_000_000 + base.astype(np.int64)
            keys = np.unique(np.concatenate([keys, extra]))

        return np.sort(keys[:n]).astype(np.int64)

    df = pd.read_csv(csv_path)

    year_col = None
    name_col = None

    for c in df.columns:
        lc = c.lower()
        if "year" in lc:
            year_col = c
        if "name" in lc:
            name_col = c

    if year_col is None or name_col is None:
        raise ValueError("Baby names CSV must contain Year and Name columns.")

    name_codes, _ = pd.factorize(df[name_col].astype(str))
    years = df[year_col].astype(np.int64).to_numpy()

    keys = years * 1_000_000 + name_codes.astype(np.int64)
    keys = np.unique(keys)
    keys = np.sort(keys)

    if len(keys) < n:
        n = len(keys)

    return keys[:n].astype(np.int64)


def load_dataset(dataset_name, n):
    if dataset_name == "uniform":
        return generate_uniform_dataset(n)
    elif dataset_name == "zipf":
        return generate_zipf_dataset(n, ZIPF_A)
    elif dataset_name == "baby_names":
        return load_baby_names_dataset(n, BABY_NAMES_PATH)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ============================================================
# BINARY SEARCH BASELINE
# ============================================================

class BinarySearchIndex:
    def __init__(self):
        self.keys = None

    def build(self, keys):
        self.keys = keys

    def lookup(self, q):
        i = bisect.bisect_left(self.keys, q)

        if i < len(self.keys) and self.keys[i] == q:
            return i

        return -1

    def memory_bytes(self):
        return deep_sizeof(self.keys)


# ============================================================
# B-TREE IMPLEMENTATION
# ============================================================

class BTreeNode:
    def __init__(self, leaf=False):
        self.leaf = leaf
        self.keys = []
        self.values = []
        self.children = []


class BTree:
    def __init__(self, t=32):
        self.t = t
        self.root = BTreeNode(leaf=True)

    def search(self, k, node=None):
        if node is None:
            node = self.root

        i = 0
        while i < len(node.keys) and k > node.keys[i]:
            i += 1

        if i < len(node.keys) and k == node.keys[i]:
            return node.values[i]

        if node.leaf:
            return -1

        return self.search(k, node.children[i])

    def split_child(self, parent, i):
        t = self.t
        y = parent.children[i]
        z = BTreeNode(leaf=y.leaf)

        median_key = y.keys[t - 1]
        median_value = y.values[t - 1]

        z.keys = y.keys[t:]
        z.values = y.values[t:]

        y.keys = y.keys[:t - 1]
        y.values = y.values[:t - 1]

        if not y.leaf:
            z.children = y.children[t:]
            y.children = y.children[:t]

        parent.children.insert(i + 1, z)
        parent.keys.insert(i, median_key)
        parent.values.insert(i, median_value)

    def insert(self, k, v):
        root = self.root

        if len(root.keys) == 2 * self.t - 1:
            new_root = BTreeNode(leaf=False)
            new_root.children.append(root)
            self.root = new_root
            self.split_child(new_root, 0)
            self._insert_non_full(new_root, k, v)
        else:
            self._insert_non_full(root, k, v)

    def _insert_non_full(self, node, k, v):
        i = len(node.keys) - 1

        if node.leaf:
            node.keys.append(0)
            node.values.append(0)

            while i >= 0 and k < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.values[i + 1] = node.values[i]
                i -= 1

            node.keys[i + 1] = k
            node.values[i + 1] = v

        else:
            while i >= 0 and k < node.keys[i]:
                i -= 1

            i += 1

            if len(node.children[i].keys) == 2 * self.t - 1:
                self.split_child(node, i)

                if k > node.keys[i]:
                    i += 1

            self._insert_non_full(node.children[i], k, v)


class BTreeIndex:
    def __init__(self, t=32):
        self.t = t
        self.tree = None

    def build(self, keys):
        self.tree = BTree(t=self.t)

        for pos, key in enumerate(keys):
            self.tree.insert(int(key), int(pos))

    def lookup(self, q):
        return self.tree.search(int(q))

    def memory_bytes(self):
        return deep_sizeof(self.tree)


# ============================================================
# LINEAR LEARNED INDEX
# ============================================================

class LinearLearnedIndex:
    def __init__(self):
        self.a = None
        self.b = None
        self.max_error = None
        self.keys = None

    def build(self, keys):
        self.keys = keys

        x = keys.astype(np.float64)
        y = np.arange(len(keys), dtype=np.float64)

        self.a, self.b = np.polyfit(x, y, 1)

        preds = self.a * x + self.b
        errors = np.abs(preds - y)

        self.max_error = int(np.ceil(np.max(errors)))

    def predict_position(self, q):
        pred = self.a * float(q) + self.b
        pred = int(round(pred))
        pred = max(0, min(len(self.keys) - 1, pred))
        return pred

    def lookup(self, q):
        pred = self.predict_position(q)
        return bounded_binary_search(
            self.keys,
            q,
            pred - self.max_error,
            pred + self.max_error
        )

    def lookup_with_metrics(self, q, true_pos):
        pred = self.predict_position(q)

        t0 = now()
        pos = bounded_binary_search(
            self.keys,
            q,
            pred - self.max_error,
            pred + self.max_error
        )
        t1 = now()

        abs_error = abs(pred - true_pos)
        correction_time = t1 - t0

        return pos, abs_error, correction_time

    def memory_bytes(self):
        return deep_sizeof({
            "a": self.a,
            "b": self.b,
            "max_error": self.max_error
        })


# ============================================================
# SIMPLE 2-STAGE RMI
# ============================================================

class SimpleRMI:
    """
    Stage 1: linear model predicts expert id.
    Stage 2: each expert predicts final key position.

    This version includes routed error-bound correction so RMI avoids misses
    when Stage 1 routes a key to a nearby/wrong expert.
    """
    def __init__(self, n_experts=10):
        self.n_experts = n_experts
        self.top_a = None
        self.top_b = None
        self.experts = []
        self.keys = None

    def build(self, keys):
        self.keys = keys
        n = len(keys)

        x = keys.astype(np.float64)
        positions = np.arange(n, dtype=np.float64)

        seg_ids = np.minimum(
            (positions * self.n_experts) // n,
            self.n_experts - 1
        ).astype(np.float64)

        # Stage 1: key -> expert/segment id
        self.top_a, self.top_b = np.polyfit(x, seg_ids, 1)

        self.experts = []

        # Stage 2: train one linear model per segment
        for seg in range(self.n_experts):
            mask = seg_ids == seg
            x_seg = x[mask]
            y_seg = positions[mask]

            if len(x_seg) < 2:
                a = 0.0
                b = float(y_seg[0]) if len(y_seg) else 0.0
                max_error = 0
            else:
                a, b = np.polyfit(x_seg, y_seg, 1)
                preds = a * x_seg + b
                max_error = int(np.ceil(np.max(np.abs(preds - y_seg))))

            self.experts.append({
                "a": a,
                "b": b,
                "max_error": max_error
            })

        # ----------------------------------------------------
        # FIX: recompute error bounds after actual Stage 1 routing
        # ----------------------------------------------------
        routed_errors = [0 for _ in range(self.n_experts)]

        for key, true_pos in zip(keys, positions):
            expert_id = self.predict_expert(key)
            expert = self.experts[expert_id]

            pred = expert["a"] * float(key) + expert["b"]
            pred = int(round(pred))
            pred = max(0, min(len(self.keys) - 1, pred))

            err = abs(pred - int(true_pos))
            routed_errors[expert_id] = max(
                routed_errors[expert_id],
                int(np.ceil(err))
            )

        for expert_id in range(self.n_experts):
            self.experts[expert_id]["max_error"] = routed_errors[expert_id]

    def predict_expert(self, q):
        pred = self.top_a * float(q) + self.top_b
        pred = int(round(pred))
        pred = max(0, min(self.n_experts - 1, pred))
        return pred

    def predict_position(self, q):
        expert_id = self.predict_expert(q)
        expert = self.experts[expert_id]

        pred = expert["a"] * float(q) + expert["b"]
        pred = int(round(pred))
        pred = max(0, min(len(self.keys) - 1, pred))

        return pred, expert_id, expert["max_error"]

    def lookup(self, q):
        pred, expert_id, max_error = self.predict_position(q)

        return bounded_binary_search(
            self.keys,
            q,
            pred - max_error,
            pred + max_error
        )

    def lookup_with_metrics(self, q, true_pos):
        pred, expert_id, max_error = self.predict_position(q)

        t0 = now()
        pos = bounded_binary_search(
            self.keys,
            q,
            pred - max_error,
            pred + max_error
        )
        t1 = now()

        abs_error = abs(pred - true_pos)
        correction_time = t1 - t0

        return pos, abs_error, correction_time

    def memory_bytes(self):
        return deep_sizeof({
            "top_a": self.top_a,
            "top_b": self.top_b,
            "experts": self.experts
        })


# ============================================================
# BENCHMARKING
# ============================================================

def make_model(model_name):
    if model_name == "binary_search":
        return BinarySearchIndex()
    elif model_name == "btree":
        return BTreeIndex(t=BTREE_T)
    elif model_name == "linear":
        return LinearLearnedIndex()
    elif model_name == "rmi":
        return SimpleRMI(n_experts=RMI_NUM_EXPERTS)

    raise ValueError(f"Unknown model: {model_name}")


def benchmark_one_model(model_name, keys, queries, true_positions):
    model = make_model(model_name)

    t0 = now()
    model.build(keys)
    t1 = now()
    build_time = t1 - t0

    total_lookup_start = now()

    abs_errors = []
    correction_times = []
    misses = 0

    for q in queries:
        true_pos = true_positions[int(q)]

        if model_name in ("linear", "rmi"):
            found_pos, abs_error, correction_time = model.lookup_with_metrics(q, true_pos)
            abs_errors.append(abs_error)
            correction_times.append(correction_time)
        else:
            found_pos = model.lookup(q)

        if found_pos != true_pos:
            misses += 1

    total_lookup_end = now()
    total_lookup_time = total_lookup_end - total_lookup_start

    avg_lookup_us = (total_lookup_time / len(queries)) * 1e6
    memory_bytes = model.memory_bytes()

    return {
        "model": model_name,
        "build_time_sec": build_time,
        "avg_lookup_us": avg_lookup_us,
        "mean_abs_error": safe_mean(abs_errors),
        "max_abs_error": float(max(abs_errors)) if abs_errors else 0.0,
        "mean_correction_us": safe_mean(correction_times) * 1e6 if correction_times else 0.0,
        "memory_bytes": memory_bytes,
        "misses": misses
    }


def run_experiments():
    dataset_names = ["uniform", "zipf", "baby_names"]
    model_names = ["binary_search", "btree", "linear", "rmi"]

    results = []

    for dataset_name in dataset_names:
        for n in DATASET_SIZES:
            print(f"\n=== DATASET: {dataset_name} | SIZE: {n} ===")

            keys = load_dataset(dataset_name, n)
            queries, true_positions = sample_existing_queries(keys, NUM_QUERIES)

            for model_name in model_names:
                print(f"Running model: {model_name}")

                trial_metrics = []

                for trial in range(NUM_TRIALS):
                    metrics = benchmark_one_model(model_name, keys, queries, true_positions)
                    trial_metrics.append(metrics)

                    print(
                        f"  Trial {trial + 1}/{NUM_TRIALS} | "
                        f"lookup={metrics['avg_lookup_us']:.3f} us | "
                        f"build={metrics['build_time_sec']:.4f} sec | "
                        f"misses={metrics['misses']}"
                    )

                final = {
                    "dataset": dataset_name,
                    "size": n,
                    "model": model_name,
                    "build_time_sec": safe_mean([m["build_time_sec"] for m in trial_metrics]),
                    "avg_lookup_us": safe_mean([m["avg_lookup_us"] for m in trial_metrics]),
                    "mean_abs_error": safe_mean([m["mean_abs_error"] for m in trial_metrics]),
                    "max_abs_error": safe_mean([m["max_abs_error"] for m in trial_metrics]),
                    "mean_correction_us": safe_mean([m["mean_correction_us"] for m in trial_metrics]),
                    "memory_bytes": safe_mean([m["memory_bytes"] for m in trial_metrics]),
                    "misses": safe_mean([m["misses"] for m in trial_metrics]),
                }

                results.append(final)

                print(
                    f"  FINAL | dataset={dataset_name} | size={n} | model={model_name} | "
                    f"avg_lookup={final['avg_lookup_us']:.3f} us | "
                    f"mean_error={final['mean_abs_error']:.3f} | "
                    f"misses={final['misses']}"
                )

    return pd.DataFrame(results)


# ============================================================
# PLOTTING
# ============================================================

def ensure_results_dir():
    if not os.path.exists("results"):
        os.makedirs("results")


def plot_latency(df):
    for dataset_name in df["dataset"].unique():
        sub = df[df["dataset"] == dataset_name]

        plt.figure(figsize=(8, 5))

        for model in sub["model"].unique():
            part = sub[sub["model"] == model].sort_values("size")
            plt.plot(part["size"], part["avg_lookup_us"], marker="o", label=model)

        plt.title(f"Lookup Latency vs Dataset Size ({dataset_name})")
        plt.xlabel("Dataset Size")
        plt.ylabel("Average Lookup Time (microseconds)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/latency_{dataset_name}.png")
        plt.close()


def plot_error(df):
    for dataset_name in df["dataset"].unique():
        sub = df[
            (df["dataset"] == dataset_name) &
            (df["model"].isin(["linear", "rmi"]))
        ]

        plt.figure(figsize=(8, 5))

        for model in sub["model"].unique():
            part = sub[sub["model"] == model].sort_values("size")
            plt.plot(part["size"], part["mean_abs_error"], marker="o", label=model)

        plt.title(f"Mean Prediction Error vs Dataset Size ({dataset_name})")
        plt.xlabel("Dataset Size")
        plt.ylabel("Mean Absolute Error")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/error_{dataset_name}.png")
        plt.close()


def plot_memory(df):
    for dataset_name in df["dataset"].unique():
        sub = df[df["dataset"] == dataset_name]

        plt.figure(figsize=(8, 5))

        for model in sub["model"].unique():
            part = sub[sub["model"] == model].sort_values("size")
            plt.plot(part["size"], part["memory_bytes"], marker="o", label=model)

        plt.title(f"Memory Usage vs Dataset Size ({dataset_name})")
        plt.xlabel("Dataset Size")
        plt.ylabel("Memory (bytes)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/memory_{dataset_name}.png")
        plt.close()


def plot_correction_cost(df):
    for dataset_name in df["dataset"].unique():
        sub = df[
            (df["dataset"] == dataset_name) &
            (df["model"].isin(["linear", "rmi"]))
        ]

        plt.figure(figsize=(8, 5))

        for model in sub["model"].unique():
            part = sub[sub["model"] == model].sort_values("size")
            plt.plot(part["size"], part["mean_correction_us"], marker="o", label=model)

        plt.title(f"Last-Mile Search Cost vs Dataset Size ({dataset_name})")
        plt.xlabel("Dataset Size")
        plt.ylabel("Correction Time (microseconds)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/correction_{dataset_name}.png")
        plt.close()


def generate_plots(df):
    ensure_results_dir()
    plot_latency(df)
    plot_error(df)
    plot_memory(df)
    plot_correction_cost(df)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Starting learned index experiments...")

    ensure_results_dir()

    df = run_experiments()

    output_path = "results/experiment_results.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")
    print("\nFinal Results:")
    print(df.sort_values(["dataset", "size", "model"]).to_string(index=False))

    generate_plots(df)

    print("\nSaved plots in results/ folder:")
    print("- latency_uniform.png / latency_zipf.png / latency_baby_names.png")
    print("- error_uniform.png / error_zipf.png / error_baby_names.png")
    print("- memory_uniform.png / memory_zipf.png / memory_baby_names.png")
    print("- correction_uniform.png / correction_zipf.png / correction_baby_names.png")


if __name__ == "__main__":
    main()