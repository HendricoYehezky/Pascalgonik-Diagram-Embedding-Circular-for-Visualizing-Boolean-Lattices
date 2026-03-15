# ============================================================
# PASCALGONIC DIAGRAM
# Stable Google Colab Version (Fixed Hungarian Step Tables)
# ============================================================

import math
import os
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import ipywidgets as widgets
from IPython.display import display, clear_output


def ui_display(x):
    try:
        display(x)
    except Exception:
        print(x)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class PascalgonicConfig:
    p: int = 5
    target_binary: str = "00001"
    rotation_deg: float = 90.0
    alpha: float = 1.0
    beta: float = 0.20
    figsize: tuple = (10, 10)

    # "figure_only", "summary_and_figure", "full_and_figure"
    display_mode: str = "summary_and_figure"

    show_legend: bool = True
    show_cell_labels: bool = True

    save_final_figure: bool = False
    save_step_figures: bool = False

    output_dir_steps: str = "pascalgonic_steps"
    output_final_name: str = "pascalgonic_final.png"

    ring_r_min: float = 0.20
    ring_r_max: float = 1.00
    sector_samples: int = 80

    background_color: str = "#f2f2f2"
    cell_normal_color: str = "#e7e7e7"
    cell_highlight_color: str = "#ecd98b"
    structure_line_color: str = "#8a8a8a"
    outer_line_color: str = "#111111"
    text_color: str = "#111111"

    fast_render_mode: bool = False
    visual_method: str = "hungarian"

    compare_mode: bool = False
    compare_primes: tuple = (5, 7, 11, 13)
    compare_figsize: tuple = (16, 16)
    compare_output_name: str = "pascalgonic_compare.png"


# ============================================================
# BASIC UTILITIES
# ============================================================

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True


def popcount(x: int) -> int:
    return bin(x).count("1")


def mask_to_binary(mask: int, p: int) -> str:
    return format(mask, f"0{p}b")


def binary_to_mask(binary: str) -> int:
    return int(binary, 2)


def masks_in_layer(p: int, k: int):
    return [m for m in range(2 ** p) if popcount(m) == k]


def valid_target_binaries(p: int):
    return [format(m, f"0{p}b") for m in range(1, 2 ** p)]


def singleton_target_binary_last(p: int) -> str:
    return "0" * (p - 1) + "1"


def apply_fast_render_defaults(config: PascalgonicConfig) -> PascalgonicConfig:
    cfg = PascalgonicConfig(**vars(config))
    if cfg.fast_render_mode:
        cfg.sector_samples = min(cfg.sector_samples, 12)
        cfg.show_cell_labels = False
        if cfg.compare_mode:
            cfg.show_legend = False
        if cfg.display_mode == "full_and_figure":
            cfg.display_mode = "summary_and_figure"
    return cfg


def resolve_visual_method(config: PascalgonicConfig) -> str:
    return config.visual_method


def pretty_method_name(method: str) -> str:
    return {
        "hungarian": "Hungarian Assignment",
        "angular_sorting": "Angular Sorting Heuristic",
    }.get(method, method)


# ============================================================
# GEOMETRY
# ============================================================

def regular_polygon_vertices(p: int, radius: float = 1.0, rotation_deg: float = 90.0):
    angles = np.linspace(0, 2 * np.pi, p, endpoint=False) + np.deg2rad(rotation_deg)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack([x, y])


def interpolate_polygon_by_angle(p: int, radius: float, theta: float, rotation_deg: float = 90.0):
    verts = regular_polygon_vertices(p, radius=radius, rotation_deg=rotation_deg)
    direction = np.array([math.cos(theta), math.sin(theta)], dtype=float)

    best_t = None
    best_point = None

    for i in range(p):
        a = verts[i]
        b = verts[(i + 1) % p]
        edge = b - a

        M = np.array(
            [
                [direction[0], -edge[0]],
                [direction[1], -edge[1]],
            ],
            dtype=float,
        )

        det = np.linalg.det(M)
        if abs(det) < 1e-12:
            continue

        t, u = np.linalg.solve(M, a)
        if t >= -1e-10 and -1e-10 <= u <= 1 + 1e-10:
            pt = t * direction
            if best_t is None or t < best_t:
                best_t = t
                best_point = pt

    return best_point


def safe_point(p: int, radius: float, theta: float, rotation_deg: float = 90.0):
    pt = interpolate_polygon_by_angle(p, radius, theta, rotation_deg)
    if pt is not None:
        return pt

    for delta in [1e-8, -1e-8, 1e-7, -1e-7, 1e-6, -1e-6, 1e-5, -1e-5]:
        pt = interpolate_polygon_by_angle(p, radius, theta + delta, rotation_deg)
        if pt is not None:
            return pt

    raise ValueError(f"Failed to intersect polygon at theta={theta}")


def build_ring_sector_geometry(
    p: int,
    r_inner: float,
    r_outer: float,
    edges: list,
    rotation_deg: float = 90.0,
    samples: int = 40,
):
    m = len(edges) - 1
    outer_arcs = []
    inner_arcs = []

    for j in range(m):
        theta1 = edges[j]
        theta2 = edges[j + 1]

        outer_thetas = np.linspace(theta1, theta2, samples)
        inner_thetas = np.linspace(theta2, theta1, samples)

        outer_arc = np.array([safe_point(p, r_outer, th, rotation_deg) for th in outer_thetas])
        inner_arc = np.array([safe_point(p, r_inner, th, rotation_deg) for th in inner_thetas])

        outer_arc[0] = safe_point(p, r_outer, theta1, rotation_deg)
        outer_arc[-1] = safe_point(p, r_outer, theta2, rotation_deg)
        inner_arc[-1] = safe_point(p, r_inner, theta1, rotation_deg)
        inner_arc[0] = safe_point(p, r_inner, theta2, rotation_deg)

        outer_arcs.append(outer_arc)
        inner_arcs.append(inner_arc)

    return [np.vstack([outer_arcs[j], inner_arcs[j]]) for j in range(m)]


def build_ring_band_polygon(p: int, r_inner: float, r_outer: float, rotation_deg: float = 90.0):
    outer = regular_polygon_vertices(p, radius=r_outer, rotation_deg=rotation_deg)
    inner = regular_polygon_vertices(p, radius=r_inner, rotation_deg=rotation_deg)[::-1]
    return np.vstack([outer, inner])


# ============================================================
# ANGLE UTILITIES
# ============================================================

def normalize_angle_0_2pi(theta: float) -> float:
    return theta % (2 * math.pi)


def circular_angle_distance(a: float, b: float) -> float:
    d = abs(normalize_angle_0_2pi(a - b))
    return min(d, 2 * math.pi - d)


def circular_slot_distance(i: int, j: int, m: int) -> int:
    d = abs(i - j)
    return min(d, m - d)


def build_bit_angles(p: int, rotation_deg: float = 90.0):
    angles = np.linspace(0, 2 * np.pi, p, endpoint=False) + np.deg2rad(rotation_deg)
    return {1 << i: angles[i] for i in range(p)}


def mask_direction_angle(mask: int, bit_angles: dict) -> float:
    if mask == 0:
        return 0.0

    vx, vy = 0.0, 0.0
    for bit, ang in bit_angles.items():
        if mask & bit:
            vx += math.cos(ang)
            vy += math.sin(ang)

    return normalize_angle_0_2pi(math.atan2(vy, vx))


def initial_ring_order(masks, bit_angles):
    return sorted(masks, key=lambda m: mask_direction_angle(m, bit_angles))


def cyclic_order_by_reference(masks, angle_fn, ref_angle):
    if not masks:
        return []

    items = [(m, normalize_angle_0_2pi(angle_fn(m))) for m in masks]
    items.sort(key=lambda t: t[1])

    deltas = [normalize_angle_0_2pi(a - ref_angle) for _, a in items]
    start_idx = int(np.argmin(deltas))
    rotated = items[start_idx:] + items[:start_idx]
    return [m for m, _ in rotated]


def build_ring_partition_centers(m: int, rotation_deg: float = 90.0):
    sector_width = 2 * math.pi / m
    start_center = np.deg2rad(rotation_deg)
    return [normalize_angle_0_2pi(start_center + j * sector_width) for j in range(m)]


def ordered_cyclic_block_slots(start: int, length: int, m: int):
    return [(start + t) % m for t in range(length)]


def ordered_complement_slots(block_slots, m: int):
    block_set = set(block_slots)
    return [j for j in range(m) if j not in block_set]


# ============================================================
# RELATIONS
# ============================================================

def is_subset_mask(a: int, b: int) -> bool:
    return (a & b) == a


def is_superset_mask(a: int, b: int) -> bool:
    return (a & b) == b


def proper_subset_mask(a: int, b: int) -> bool:
    return a != b and is_subset_mask(a, b)


def proper_superset_mask(a: int, b: int) -> bool:
    return a != b and is_superset_mask(a, b)


def relation_to_target(mask: int, target_mask: int) -> str:
    if mask == target_mask:
        return "target"
    if proper_subset_mask(mask, target_mask):
        return "proper subset"
    if proper_superset_mask(mask, target_mask):
        return "proper superset"
    return "unrelated"


def is_related(mask: int, target_mask: int) -> bool:
    return relation_to_target(mask, target_mask) != "unrelated"


# ============================================================
# HUNGARIAN ALGORITHM
# ============================================================

def zero_mask(M, tol=1e-12):
    return np.abs(M) <= tol


def maximum_bipartite_matching_zero(Z):
    n = Z.shape[0]
    row_to_col = [-1] * n
    col_to_row = [-1] * n

    def dfs(r, seen_cols):
        for c in range(n):
            if Z[r, c] and not seen_cols[c]:
                seen_cols[c] = True
                if col_to_row[c] == -1 or dfs(col_to_row[c], seen_cols):
                    row_to_col[r] = c
                    col_to_row[c] = r
                    return True
        return False

    for r in range(n):
        seen_cols = [False] * n
        dfs(r, seen_cols)

    match_size = sum(x != -1 for x in row_to_col)
    return row_to_col, col_to_row, match_size


def minimum_vertex_cover_from_matching(Z, row_to_col, col_to_row):
    n = Z.shape[0]
    unmatched_rows = [r for r in range(n) if row_to_col[r] == -1]

    visited_rows = [False] * n
    visited_cols = [False] * n
    stack = unmatched_rows[:]

    for r in unmatched_rows:
        visited_rows[r] = True

    while stack:
        r = stack.pop()
        for c in range(n):
            if Z[r, c]:
                if row_to_col[r] != c and not visited_cols[c]:
                    visited_cols[c] = True
                    rr = col_to_row[c]
                    if rr != -1 and not visited_rows[rr]:
                        visited_rows[rr] = True
                        stack.append(rr)

    cover_rows = [r for r in range(n) if not visited_rows[r]]
    cover_cols = [c for c in range(n) if visited_cols[c]]
    return cover_rows, cover_cols


def apply_cover_update(M, cover_rows, cover_cols):
    n = M.shape[0]
    cover_rows_set = set(cover_rows)
    cover_cols_set = set(cover_cols)

    uncovered = [
        M[i, j]
        for i in range(n)
        for j in range(n)
        if i not in cover_rows_set and j not in cover_cols_set
    ]

    if not uncovered:
        return M.copy(), 0.0

    delta = min(uncovered)
    newM = M.copy()

    for i in range(n):
        for j in range(n):
            row_cov = i in cover_rows_set
            col_cov = j in cover_cols_set

            if not row_cov and not col_cov:
                newM[i, j] -= delta
            elif row_cov and col_cov:
                newM[i, j] += delta

    return newM, delta


def hungarian_with_steps(cost_matrix):
    C = np.array(cost_matrix, dtype=float)
    n = C.shape[0]
    steps = []

    steps.append({"name": "Initial cost matrix", "matrix": C.copy()})

    row_mins = C.min(axis=1)
    C = C - row_mins[:, None]
    steps.append({"name": "After row reduction", "matrix": C.copy(), "row_mins": row_mins.copy()})

    col_mins = C.min(axis=0)
    C = C - col_mins[None, :]
    steps.append({"name": "After column reduction", "matrix": C.copy(), "col_mins": col_mins.copy()})

    iteration = 0
    while True:
        Z = zero_mask(C)
        row_to_col, col_to_row, match_size = maximum_bipartite_matching_zero(Z)

        steps.append(
            {
                "name": f"Zero matching check {iteration}",
                "matrix": C.copy(),
                "zero_mask": Z.copy(),
                "row_to_col": row_to_col[:],
                "match_size": match_size,
            }
        )

        if match_size == n:
            assignment = {r: row_to_col[r] for r in range(n)}
            steps.append(
                {
                    "name": "Final independent zeros / optimal assignment",
                    "matrix": C.copy(),
                    "assignment": assignment.copy(),
                }
            )
            return assignment, steps

        cover_rows, cover_cols = minimum_vertex_cover_from_matching(Z, row_to_col, col_to_row)
        updated_C, delta = apply_cover_update(C, cover_rows, cover_cols)

        steps.append(
            {
                "name": f"Cover/update iteration {iteration}",
                "matrix_before": C.copy(),
                "cover_rows": cover_rows[:],
                "cover_cols": cover_cols[:],
                "delta": delta,
                "matrix_after": updated_C.copy(),
            }
        )

        C = updated_C
        iteration += 1


def solve_square_assignment(cost_matrix, collect_steps=False):
    C = np.array(cost_matrix, dtype=float)
    n = C.shape[0]

    if n == 0:
        return {}, ([] if collect_steps else None), 0.0

    assignment, steps = hungarian_with_steps(C)
    objective = sum(C[r, c] for r, c in assignment.items())
    return assignment, (steps if collect_steps else None), objective


# ============================================================
# COST MODEL
# ============================================================

def build_base_cost_matrix(ordered_masks, partition_centers, bit_angles, alpha=1.0, beta=0.20):
    m = len(ordered_masks)
    sector_width = 2 * math.pi / m
    cost = np.zeros((m, m), dtype=float)

    for i, mask in enumerate(ordered_masks):
        theta = mask_direction_angle(mask, bit_angles)
        for j, phi in enumerate(partition_centers):
            angular_cost = circular_angle_distance(theta, phi) / sector_width
            displacement_cost = circular_slot_distance(i, j, m)
            cost[i, j] = alpha * angular_cost + beta * displacement_cost

    return cost


def compactify_ring_by_block_hungarian(
    masks,
    bit_angles,
    target_mask,
    p,
    rotation_deg=90.0,
    alpha=1.0,
    beta=0.20,
    collect_steps=True,
):
    ordered_masks = initial_ring_order(masks, bit_angles)
    m = len(ordered_masks)
    partition_centers = build_ring_partition_centers(m, rotation_deg=rotation_deg)

    base_cost = build_base_cost_matrix(
        ordered_masks=ordered_masks,
        partition_centers=partition_centers,
        bit_angles=bit_angles,
        alpha=alpha,
        beta=beta,
    )

    related_rows = [i for i, mask in enumerate(ordered_masks) if is_related(mask, target_mask)]
    unrelated_rows = [i for i, mask in enumerate(ordered_masks) if not is_related(mask, target_mask)]
    h = len(related_rows)

    if h == 0 or h == m:
        assignment, steps = hungarian_with_steps(base_cost)
        assigned_masks = [None] * m
        for r, c in assignment.items():
            assigned_masks[c] = ordered_masks[r]

        return {
            "ordered_masks": ordered_masks,
            "partition_centers": partition_centers,
            "base_cost_matrix": base_cost,
            "used_cost_matrix": base_cost.copy(),
            "assignment_row_to_col": assignment,
            "assigned_masks_by_partition": assigned_masks,
            "block_start": 0 if h == m else None,
            "block_slots": list(range(m)) if h == m else [],
            "related_count": h,
            "candidate_history": [],
            "steps": steps if collect_steps else None,
            "step_row_labels": [mask_to_binary(mk, p) for mk in ordered_masks],
            "step_col_labels": [f"P{j+1}" for j in range(m)],
            "step_scope": "full_layer",
        }

    best = None
    candidate_history = []

    full_row_labels = [mask_to_binary(mk, p) for mk in ordered_masks]

    for start in range(m):
        block_slots = ordered_cyclic_block_slots(start, h, m)
        other_slots = ordered_complement_slots(block_slots, m)

        C_rel = base_cost[np.ix_(related_rows, block_slots)]
        rel_assignment_local, rel_steps, rel_obj = solve_square_assignment(
            C_rel,
            collect_steps=collect_steps
        )

        C_unrel = base_cost[np.ix_(unrelated_rows, other_slots)]
        unrel_assignment_local, _, unrel_obj = solve_square_assignment(
            C_unrel,
            collect_steps=False
        )

        assignment = {}
        for r_local, c_local in rel_assignment_local.items():
            assignment[related_rows[r_local]] = block_slots[c_local]
        for r_local, c_local in unrel_assignment_local.items():
            assignment[unrelated_rows[r_local]] = other_slots[c_local]

        objective = rel_obj + unrel_obj

        assigned_masks = [None] * m
        for r, c in assignment.items():
            assigned_masks[c] = ordered_masks[r]

        rel_row_labels = [full_row_labels[r] for r in related_rows]
        rel_col_labels = [f"P{c+1}" for c in block_slots]

        candidate = {
            "block_start": start,
            "block_slots": block_slots[:],
            "objective": objective,
            "used_cost_matrix": base_cost.copy(),
            "assignment_row_to_col": assignment.copy(),
            "assigned_masks_by_partition": assigned_masks[:],
            "steps_related_only": rel_steps if collect_steps else None,
            "step_row_labels": rel_row_labels,
            "step_col_labels": rel_col_labels,
            "step_scope": "related_block_only",
        }
        candidate_history.append(candidate)

        if best is None or objective < best["objective"]:
            best = candidate

    return {
        "ordered_masks": ordered_masks,
        "partition_centers": partition_centers,
        "base_cost_matrix": base_cost,
        "used_cost_matrix": best["used_cost_matrix"],
        "assignment_row_to_col": best["assignment_row_to_col"],
        "assigned_masks_by_partition": best["assigned_masks_by_partition"],
        "block_start": best["block_start"],
        "block_slots": best["block_slots"],
        "related_count": h,
        "candidate_history": candidate_history,
        "steps": best["steps_related_only"] if collect_steps else None,
        "step_row_labels": best["step_row_labels"],
        "step_col_labels": best["step_col_labels"],
        "step_scope": best["step_scope"],
    }


# ============================================================
# ANGULAR SORTING HEURISTIC
# ============================================================

def fast_compact_ring_assignment(masks, bit_angles, target_mask, p, rotation_deg=90.0):
    ordered_masks = initial_ring_order(masks, bit_angles)
    m = len(ordered_masks)
    partition_centers = build_ring_partition_centers(m, rotation_deg=rotation_deg)

    related = [x for x in ordered_masks if is_related(x, target_mask)]
    unrelated = [x for x in ordered_masks if not is_related(x, target_mask)]
    h = len(related)

    if h == 0 or h == m:
        assigned = ordered_masks[:]
        return {
            "ordered_masks": ordered_masks,
            "partition_centers": partition_centers,
            "base_cost_matrix": np.zeros((m, m)),
            "used_cost_matrix": np.zeros((m, m)),
            "assignment_row_to_col": {i: i for i in range(m)},
            "assigned_masks_by_partition": assigned,
            "block_start": 0 if h == m else None,
            "block_slots": list(range(m)) if h == m else [],
            "related_count": h,
            "candidate_history": [],
            "steps": None,
            "step_row_labels": [mask_to_binary(mk, p) for mk in ordered_masks],
            "step_col_labels": [f"P{j+1}" for j in range(m)],
            "step_scope": "heuristic",
        }

    target_theta = mask_direction_angle(target_mask, bit_angles)
    sector_width = 2 * math.pi / m

    best_start = None
    best_score = None

    for start in range(m):
        center_slot = (start + 0.5 * (h - 1)) % m
        block_center_theta = normalize_angle_0_2pi(
            partition_centers[0] + center_slot * sector_width
        )
        score = circular_angle_distance(block_center_theta, target_theta)

        if best_score is None or score < best_score:
            best_score = score
            best_start = start

    related_order = cyclic_order_by_reference(
        related,
        lambda mk: mask_direction_angle(mk, bit_angles),
        target_theta,
    )
    unrelated_order = cyclic_order_by_reference(
        unrelated,
        lambda mk: mask_direction_angle(mk, bit_angles),
        target_theta,
    )

    assigned = [None] * m
    block_slots = ordered_cyclic_block_slots(best_start, h, m)

    for t in range(h):
        assigned[(best_start + t) % m] = related_order[t]

    free_slots = [j for j in range(m) if assigned[j] is None]
    for slot, mask in zip(free_slots, unrelated_order):
        assigned[slot] = mask

    row_index = {mask: i for i, mask in enumerate(ordered_masks)}
    assignment = {row_index[mask]: j for j, mask in enumerate(assigned)}

    return {
        "ordered_masks": ordered_masks,
        "partition_centers": partition_centers,
        "base_cost_matrix": np.zeros((m, m)),
        "used_cost_matrix": np.zeros((m, m)),
        "assignment_row_to_col": assignment,
        "assigned_masks_by_partition": assigned,
        "block_start": best_start,
        "block_slots": block_slots,
        "related_count": h,
        "candidate_history": [],
        "steps": None,
        "step_row_labels": [mask_to_binary(mk, p) for mk in ordered_masks],
        "step_col_labels": [f"P{j+1}" for j in range(m)],
        "step_scope": "heuristic",
    }


# ============================================================
# SOLVERS
# ============================================================

def validate_config(config: PascalgonicConfig):
    if config.p <= 2:
        raise ValueError("Parameter p harus memenuhi p > 2.")
    if not is_prime(config.p):
        raise ValueError("Implementasi ini ditujukan untuk p prima > 2.")
    if len(config.target_binary) != config.p:
        raise ValueError("Panjang target_binary harus sama dengan p.")
    if set(config.target_binary) - {"0", "1"}:
        raise ValueError("target_binary harus berupa string biner.")
    if binary_to_mask(config.target_binary) == 0:
        raise ValueError("target_binary tidak boleh himpunan kosong.")


def solve_pascalgonic_compact_hungarian(config: PascalgonicConfig):
    validate_config(config)

    p = config.p
    target_mask = binary_to_mask(config.target_binary)
    bit_angles = build_bit_angles(p, rotation_deg=config.rotation_deg)
    collect_steps = (config.display_mode == "full_and_figure") and (not config.fast_render_mode)

    results = {
        "p": p,
        "target_binary": config.target_binary,
        "target_mask": target_mask,
        "bit_angles": bit_angles,
        "rotation_deg": config.rotation_deg,
        "alpha": config.alpha,
        "beta": config.beta,
        "center_mask": (1 << p) - 1,
        "config": config,
        "layers": {},
        "model_mode": "hungarian",
    }

    for k in range(p + 1):
        masks = masks_in_layer(p, k)

        if k == 0 or k == p:
            results["layers"][k] = {
                "masks": masks,
                "ordered_masks": masks[:],
                "partition_centers": [0.0],
                "base_cost_matrix": np.array([[0.0]]),
                "used_cost_matrix": np.array([[0.0]]),
                "assignment_row_to_col": {0: 0},
                "assigned_masks_by_partition": masks[:],
                "block_start": None,
                "block_slots": [0],
                "related_count": sum(is_related(m, target_mask) for m in masks),
                "candidate_history": [],
                "steps": None,
                "step_row_labels": [mask_to_binary(mk, p) for mk in masks],
                "step_col_labels": ["P1"],
                "step_scope": "trivial",
            }
        else:
            results["layers"][k] = compactify_ring_by_block_hungarian(
                masks=masks,
                bit_angles=bit_angles,
                target_mask=target_mask,
                p=p,
                rotation_deg=config.rotation_deg,
                alpha=config.alpha,
                beta=config.beta,
                collect_steps=collect_steps,
            )

    return results


def solve_pascalgonic_angular_sorting(config: PascalgonicConfig):
    validate_config(config)

    p = config.p
    target_mask = binary_to_mask(config.target_binary)
    bit_angles = build_bit_angles(p, rotation_deg=config.rotation_deg)

    results = {
        "p": p,
        "target_binary": config.target_binary,
        "target_mask": target_mask,
        "bit_angles": bit_angles,
        "rotation_deg": config.rotation_deg,
        "alpha": config.alpha,
        "beta": config.beta,
        "center_mask": (1 << p) - 1,
        "config": config,
        "layers": {},
        "model_mode": "angular_sorting",
    }

    for k in range(p + 1):
        masks = masks_in_layer(p, k)

        if k == 0 or k == p:
            results["layers"][k] = {
                "masks": masks,
                "ordered_masks": masks[:],
                "partition_centers": [0.0],
                "base_cost_matrix": np.array([[0.0]]),
                "used_cost_matrix": np.array([[0.0]]),
                "assignment_row_to_col": {0: 0},
                "assigned_masks_by_partition": masks[:],
                "block_start": None,
                "block_slots": [0],
                "related_count": sum(is_related(m, target_mask) for m in masks),
                "candidate_history": [],
                "steps": None,
                "step_row_labels": [mask_to_binary(mk, p) for mk in masks],
                "step_col_labels": ["P1"],
                "step_scope": "trivial",
            }
        else:
            results["layers"][k] = fast_compact_ring_assignment(
                masks=masks,
                bit_angles=bit_angles,
                target_mask=target_mask,
                p=p,
                rotation_deg=config.rotation_deg,
            )

    return results


# ============================================================
# TABLE UTILITIES
# ============================================================

def degree_str(theta):
    return f"{math.degrees(theta):.1f}°"


def matrix_to_dataframe(M, row_labels, col_labels, decimals=4):
    return pd.DataFrame(np.round(M, decimals), index=row_labels, columns=col_labels)


def make_mask_table(masks, p, bit_angles, target_mask=None):
    rows = []
    for idx, mask in enumerate(masks, start=1):
        rows.append(
            {
                "No": idx,
                "Subset": mask_to_binary(mask, p),
                "Cardinality": popcount(mask),
                "Direction angle": degree_str(mask_direction_angle(mask, bit_angles)),
                "Relation": relation_to_target(mask, target_mask) if target_mask is not None else "",
            }
        )
    return pd.DataFrame(rows)


def layer_content_table(results):
    p = results["p"]
    target_mask = results["target_mask"]

    rows = []
    for k in range(p + 1):
        masks = results["layers"][k]["ordered_masks"]
        highlighted = [mask_to_binary(m, p) for m in masks if is_related(m, target_mask)]
        rows.append(
            {
                "Layer": f"R{k}",
                "Cardinality": k,
                "Count": len(masks),
                "Members": ", ".join(mask_to_binary(m, p) for m in masks),
                "Highlighted members": ", ".join(highlighted) if highlighted else "-",
            }
        )

    return pd.DataFrame(rows)


def assignment_table_for_layer(results, k):
    p = results["p"]
    layer = results["layers"][k]

    rows = []
    for j, mask in enumerate(layer["assigned_masks_by_partition"]):
        rows.append(
            {
                "Partition": f"P{j+1}",
                "Partition center": degree_str(layer["partition_centers"][j]),
                "Assigned subset": mask_to_binary(mask, p),
            }
        )

    return pd.DataFrame(rows)


def assignment_cost_table_for_layer(results, k):
    p = results["p"]
    layer = results["layers"][k]
    ordered_masks = layer["ordered_masks"]
    assigned_masks = layer["assigned_masks_by_partition"]
    cost_matrix = layer["base_cost_matrix"]

    row_index = {mask: i for i, mask in enumerate(ordered_masks)}
    rows = []

    for j, mask in enumerate(assigned_masks):
        r = row_index[mask]
        rows.append(
            {
                "Partition": f"P{j+1}",
                "Assigned subset": mask_to_binary(mask, p),
                "Cost value": round(float(cost_matrix[r, j]), 6),
            }
        )

    return pd.DataFrame(rows)


def candidate_block_table_for_layer(results, k):
    p = results["p"]
    layer = results["layers"][k]
    rows = []

    for cand in layer["candidate_history"]:
        assigned = cand["assigned_masks_by_partition"]
        rows.append(
            {
                "Layer": f"R{k}",
                "Block start": cand["block_start"],
                "Block slots": str(cand["block_slots"]),
                "Objective": round(float(cand["objective"]), 6),
                "Assigned subsets by partition": ", ".join(mask_to_binary(m, p) for m in assigned),
            }
        )

    if not rows:
        return pd.DataFrame(
            [{
                "Layer": f"R{k}",
                "Block start": "-",
                "Block slots": "-",
                "Objective": "-",
                "Assigned subsets by partition": "-",
            }]
        )

    return pd.DataFrame(rows).sort_values(by="Objective", ascending=True).reset_index(drop=True)


def matching_table(step, row_labels, col_labels):
    rows = []
    for i, c in enumerate(step["row_to_col"]):
        rows.append(
            {
                "Row subset": row_labels[i],
                "Matched column": "-" if c == -1 else col_labels[c],
            }
        )
    return pd.DataFrame(rows)


def final_assignment_from_step(step, row_labels, col_labels):
    rows = []
    for r, c in step["assignment"].items():
        rows.append(
            {
                "Row subset": row_labels[r],
                "Assigned partition": col_labels[c],
            }
        )
    return pd.DataFrame(rows)


def print_summary_tables(results):
    print("\n" + "#" * 100)
    print("LAYER SUMMARY TABLE R0 TO Rp")
    print("#" * 100)
    ui_display(layer_content_table(results))

    for k in range(1, results["p"]):
        print(f"\nAssignment summary for layer R{k}")
        ui_display(assignment_table_for_layer(results, k))


def print_complete_layer_table(results, k):
    p = results["p"]
    bit_angles = results["bit_angles"]
    layer = results["layers"][k]

    print("\n" + "#" * 110)
    print(f"COMPLETE TABLES FOR LAYER R{k}")
    print("#" * 110)

    print("\n1. Ordered subsets in the layer")
    ui_display(make_mask_table(layer["ordered_masks"], p, bit_angles, results["target_mask"]))

    print("\n2. Partition centers")
    centers_df = pd.DataFrame(
        {
            "Partition": [f"P{j+1}" for j in range(len(layer["partition_centers"]))],
            "Center angle": [degree_str(x) for x in layer["partition_centers"]],
        }
    )
    ui_display(centers_df)

    row_labels = [mask_to_binary(m, p) for m in layer["ordered_masks"]]
    col_labels = [f"P{j+1}" for j in range(len(layer["ordered_masks"]))]

    print("\n3. Base cost matrix")
    ui_display(matrix_to_dataframe(layer["base_cost_matrix"], row_labels, col_labels, decimals=4))

    print("\n4. Used cost matrix")
    ui_display(matrix_to_dataframe(layer["used_cost_matrix"], row_labels, col_labels, decimals=4))

    print("\n5. Zero pattern of used cost matrix (1 = zero entry)")
    zero_df = pd.DataFrame(
        (np.abs(layer["used_cost_matrix"]) <= 1e-12).astype(int),
        index=row_labels,
        columns=col_labels,
    )
    ui_display(zero_df)

    print("\n6. Candidate block summary")
    ui_display(candidate_block_table_for_layer(results, k))

    print("\n7. Final partition-wise assignment")
    ui_display(assignment_table_for_layer(results, k))

    print("\n8. Final assignment costs")
    ui_display(assignment_cost_table_for_layer(results, k))

    steps = layer["steps"]
    if steps is None:
        print("\nThis layer does not store detailed Hungarian iteration steps.")
        return

    step_row_labels = layer.get("step_row_labels", row_labels)
    step_col_labels = layer.get("step_col_labels", col_labels)
    step_scope = layer.get("step_scope", "unknown")

    print("\n9. Hungarian detailed steps")
    print(f"Step scope: {step_scope}")

    for idx, s in enumerate(steps, start=1):
        print("\n" + "=" * 110)
        print(f"Step {idx}: {s['name']}")
        print("=" * 110)

        if "row_mins" in s:
            ui_display(
                pd.DataFrame(
                    {
                        "Row subset": step_row_labels,
                        "Row minimum": np.round(s["row_mins"], 6),
                    }
                )
            )

        if "col_mins" in s:
            ui_display(
                pd.DataFrame(
                    {
                        "Partition": step_col_labels,
                        "Column minimum": np.round(s["col_mins"], 6),
                    }
                )
            )

        if "matrix" in s:
            ui_display(matrix_to_dataframe(s["matrix"], step_row_labels, step_col_labels, decimals=6))

        if "zero_mask" in s:
            print("Zero matching pattern")
            ui_display(
                pd.DataFrame(
                    s["zero_mask"].astype(int),
                    index=step_row_labels,
                    columns=step_col_labels,
                )
            )

        if "row_to_col" in s:
            ui_display(matching_table(s, step_row_labels, step_col_labels))
            print("Matching size:", s["match_size"])

        if "cover_rows" in s:
            ui_display(
                pd.DataFrame(
                    [
                        {
                            "Covered rows": ", ".join(step_row_labels[i] for i in s["cover_rows"]) if s["cover_rows"] else "-",
                            "Covered cols": ", ".join(step_col_labels[j] for j in s["cover_cols"]) if s["cover_cols"] else "-",
                            "Delta": round(float(s["delta"]), 6),
                        }
                    ]
                )
            )
            print("Matrix before update")
            ui_display(matrix_to_dataframe(s["matrix_before"], step_row_labels, step_col_labels, decimals=6))
            print("Matrix after update")
            ui_display(matrix_to_dataframe(s["matrix_after"], step_row_labels, step_col_labels, decimals=6))

        if "assignment" in s:
            ui_display(final_assignment_from_step(s, step_row_labels, step_col_labels))


def print_all_main_tables(results):
    print("\n" + "#" * 100)
    print("FULL TABLES FOR ALL LAYERS")
    print("#" * 100)
    ui_display(layer_content_table(results))

    for k in range(1, results["p"]):
        print_complete_layer_table(results, k)


# ============================================================
# DRAWING UTILITIES
# ============================================================

def compute_polygon_radii(config: PascalgonicConfig):
    return np.linspace(config.ring_r_min, config.ring_r_max, config.p)


def build_state_assignments(results, optimized_until_layer=None, use_final=True):
    p = results["p"]
    state = {}

    for k in range(p + 1):
        layer = results["layers"][k]

        if k == 0 or k == p:
            state[k] = layer["assigned_masks_by_partition"][:]
            continue

        if optimized_until_layer is None:
            state[k] = layer["ordered_masks"][:]
        else:
            if use_final and k <= optimized_until_layer:
                state[k] = layer["assigned_masks_by_partition"][:]
            else:
                state[k] = layer["ordered_masks"][:]

    return state


def adaptive_fontsize(p: int, k: int, m: int):
    base = 11.0 - 0.35 * p
    density_penalty = 0.08 * m
    ring_penalty = 0.35 * max(0, k - 2)
    fs = base - density_penalty - ring_penalty
    return max(5.5, min(10.5, fs))


def label_radial_bias(p: int, k: int):
    if k == 1:
        return 0.50
    if k == 2:
        return 0.52
    if k == 3:
        return 0.56
    if k >= 4:
        return 0.62
    return 0.50


def compute_sector_label_position(p, r_inner, r_outer, theta1, theta2, k, rotation_deg=90.0):
    theta_mid = normalize_angle_0_2pi(0.5 * (theta1 + theta2))
    bias = label_radial_bias(p, k)
    r_label = r_inner + bias * (r_outer - r_inner)
    return safe_point(p, r_label, theta_mid, rotation_deg=rotation_deg)


def draw_structure_boundaries(ax, p, polygon_radii, results, rotation_deg=90, config=None):
    structure_color = config.structure_line_color
    outer_color = config.outer_line_color

    for i, r in enumerate(polygon_radii):
        verts = regular_polygon_vertices(p, radius=r, rotation_deg=rotation_deg)
        patch = Polygon(
            verts,
            closed=True,
            fill=False,
            edgecolor=structure_color if i < len(polygon_radii) - 1 else outer_color,
            linewidth=1.0 if i < len(polygon_radii) - 1 else 1.4,
            joinstyle="miter",
            antialiased=False,
        )
        ax.add_patch(patch)

    for k in range(1, p):
        m = len(results["layers"][k]["assigned_masks_by_partition"])
        r_inner = polygon_radii[k - 1]
        r_outer = polygon_radii[k]
        sector_width = 2 * math.pi / m
        centers = results["layers"][k]["partition_centers"]
        start_center = centers[0]
        edges = [start_center - sector_width / 2 + j * sector_width for j in range(m + 1)]

        for th in edges:
            p1 = safe_point(p, r_inner, th, rotation_deg=rotation_deg)
            p2 = safe_point(p, r_outer, th, rotation_deg=rotation_deg)
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=structure_color,
                linewidth=1.0,
                solid_capstyle="butt",
                antialiased=False,
            )


def draw_pascalgonic_on_ax(ax, results, state_assignments, title="Pascalgonic Diagram", show_title=True, show_legend=True):
    config = results["config"]
    p = results["p"]
    target_mask = results["target_mask"]
    center_mask = results["center_mask"]
    rotation_deg = results["rotation_deg"]
    polygon_radii = compute_polygon_radii(config)

    ax.set_facecolor(config.background_color)
    ax.set_aspect("equal")
    ax.axis("off")

    inner_verts = regular_polygon_vertices(p, radius=polygon_radii[0], rotation_deg=rotation_deg)
    center_fc = config.cell_highlight_color if is_related(center_mask, target_mask) else config.cell_normal_color

    ax.add_patch(
        Polygon(
            inner_verts,
            closed=True,
            fill=True,
            facecolor=center_fc,
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            joinstyle="miter",
        )
    )

    center_text_pos = np.mean(inner_verts, axis=0)
    if config.show_cell_labels:
        ax.text(
            center_text_pos[0],
            center_text_pos[1],
            mask_to_binary(center_mask, p),
            ha="center",
            va="center",
            fontsize=max(5.0, 10.0 - 0.3 * p),
            color=config.text_color,
        )

    for k in range(1, p):
        assigned_masks = state_assignments[k]
        m = len(assigned_masks)
        r_inner = polygon_radii[k - 1]
        r_outer = polygon_radii[k]
        sector_width = 2 * math.pi / m
        centers = results["layers"][k]["partition_centers"]
        start_center = centers[0]
        edges = [start_center - sector_width / 2 + j * sector_width for j in range(m + 1)]

        ring_band = build_ring_band_polygon(
            p=p,
            r_inner=r_inner,
            r_outer=r_outer,
            rotation_deg=rotation_deg,
        )
        ax.add_patch(
            Polygon(
                ring_band,
                closed=True,
                fill=True,
                facecolor=config.cell_normal_color,
                edgecolor="none",
                linewidth=0,
                antialiased=False,
                joinstyle="miter",
            )
        )

        ring_polys = build_ring_sector_geometry(
            p=p,
            r_inner=r_inner,
            r_outer=r_outer,
            edges=edges,
            rotation_deg=rotation_deg,
            samples=config.sector_samples,
        )

        for j, mask in enumerate(assigned_masks):
            if is_related(mask, target_mask):
                cell = ring_polys[j]
                ax.add_patch(
                    Polygon(
                        cell,
                        closed=True,
                        fill=True,
                        facecolor=config.cell_highlight_color,
                        edgecolor="none",
                        linewidth=0,
                        antialiased=False,
                        joinstyle="miter",
                    )
                )

            if config.show_cell_labels:
                theta1 = edges[j]
                theta2 = edges[j + 1]
                label_pos = compute_sector_label_position(
                    p=p,
                    r_inner=r_inner,
                    r_outer=r_outer,
                    theta1=theta1,
                    theta2=theta2,
                    k=k,
                    rotation_deg=rotation_deg,
                )
                fs = adaptive_fontsize(p=p, k=k, m=m)
                ax.text(
                    label_pos[0],
                    label_pos[1],
                    mask_to_binary(mask, p),
                    ha="center",
                    va="center",
                    fontsize=fs,
                    color=config.text_color,
                )

    draw_structure_boundaries(
        ax=ax,
        p=p,
        polygon_radii=polygon_radii,
        results=results,
        rotation_deg=rotation_deg,
        config=config,
    )

    if show_title:
        method_name = pretty_method_name(results.get("model_mode", "hungarian"))
        ax.set_title(f"{title}\nmethod = {method_name}", fontsize=12, pad=10)

    if show_legend:
        method_name = pretty_method_name(results.get("model_mode", "hungarian"))
        legend_text = (
            f"target {results['target_binary']}\n"
            f"gold = related, gray = unrelated\n"
            f"method = {method_name}"
        )
        ax.text(
            -1.22,
            -1.18,
            legend_text,
            ha="left",
            va="bottom",
            fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", alpha=0.95),
        )

    lim = 1.28
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


def draw_pascalgonic_state(results, state_assignments, title="Pascalgonic Diagram", save_path=None, show_title=True, show_legend=True):
    config = results["config"]
    fig, ax = plt.subplots(figsize=config.figsize)
    fig.patch.set_facecolor(config.background_color)

    draw_pascalgonic_on_ax(
        ax=ax,
        results=results,
        state_assignments=state_assignments,
        title=title,
        show_title=show_title,
        show_legend=show_legend,
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )

    plt.show()


def solve_by_selected_method(config: PascalgonicConfig):
    method = resolve_visual_method(config)
    if method == "angular_sorting":
        return solve_pascalgonic_angular_sorting(config)
    return solve_pascalgonic_compact_hungarian(config)


def make_unique_primes(primes):
    seen = set()
    cleaned = []
    for p in primes:
        if p not in seen:
            cleaned.append(p)
            seen.add(p)
    return cleaned


def subplot_shape(n):
    if n == 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n in [3, 4]:
        return 2, 2
    raise ValueError("n must be between 1 and 4")


def draw_compare_figure(base_config: PascalgonicConfig, save_path=None):
    compare_primes = list(base_config.compare_primes)
    compare_primes = make_unique_primes(compare_primes)

    if len(compare_primes) == 0:
        raise ValueError("Compare mode requires at least one prime.")
    if len(compare_primes) > 4:
        raise ValueError("Compare mode supports at most 4 prime values.")

    n = len(compare_primes)
    nrows, ncols = subplot_shape(n)

    fig, axes = plt.subplots(nrows, ncols, figsize=base_config.compare_figsize)
    fig.patch.set_facecolor(base_config.background_color)

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    compare_results = {}
    start_time = time.perf_counter()

    for ax, p in zip(axes, compare_primes):
        cfg = PascalgonicConfig(**vars(base_config))
        cfg.p = p
        cfg.target_binary = singleton_target_binary_last(p)
        cfg.display_mode = "figure_only"
        cfg.compare_mode = False
        cfg.show_legend = False
        cfg.save_step_figures = False
        cfg.save_final_figure = False

        results = solve_by_selected_method(cfg)
        final_state = build_state_assignments(results, optimized_until_layer=cfg.p - 1)

        draw_pascalgonic_on_ax(
            ax=ax,
            results=results,
            state_assignments=final_state,
            title=f"p = {p}, target = {cfg.target_binary}",
            show_title=True,
            show_legend=False,
        )

        compare_results[p] = results

    for ax in axes[len(compare_primes):]:
        ax.axis("off")

    end_time = time.perf_counter()
    runtime = end_time - start_time

    title_text = "Pascalgonic comparison: p = " + ", ".join(str(x) for x in compare_primes)
    plt.suptitle(title_text, fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.965])

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )

    plt.show()
    return compare_results, runtime


# ============================================================
# STEP-BY-STEP VISUALIZATION
# ============================================================

def draw_all_optimization_stages(results):
    if results.get("model_mode") == "angular_sorting":
        print("Step-by-step is not available for Angular Sorting Heuristic.")
        return

    config = results["config"]
    Path(config.output_dir_steps).mkdir(parents=True, exist_ok=True)

    p = results["p"]

    state = build_state_assignments(results, optimized_until_layer=None)
    draw_pascalgonic_state(
        results=results,
        state_assignments=state,
        title="Initial arrangement before optimization",
        save_path=os.path.join(config.output_dir_steps, "step_00_initial.png"),
        show_legend=config.show_legend,
    )

    for k in range(1, p):
        state = build_state_assignments(results, optimized_until_layer=k)
        draw_pascalgonic_state(
            results=results,
            state_assignments=state,
            title=f"After optimizing layer R{k}",
            save_path=os.path.join(config.output_dir_steps, f"step_{k:02d}_after_R{k}.png"),
            show_legend=config.show_legend,
        )

    state = build_state_assignments(results, optimized_until_layer=p - 1)
    draw_pascalgonic_state(
        results=results,
        state_assignments=state,
        title="Final Pascalgonic diagram after all layer optimizations",
        save_path=os.path.join(config.output_dir_steps, "step_final.png"),
        show_legend=config.show_legend,
    )


# ============================================================
# MAIN RUNNER
# ============================================================

def run_pascalgonic(config: PascalgonicConfig):
    config = apply_fast_render_defaults(config)
    start_time = time.perf_counter()

    if config.compare_mode:
        compare_results, compare_runtime = draw_compare_figure(
            base_config=config,
            save_path=config.compare_output_name if config.save_final_figure else None,
        )
        end_time = time.perf_counter()
        return {
            "compare_mode": True,
            "compare_results": compare_results,
            "runtime_seconds": end_time - start_time,
            "compare_runtime_seconds": compare_runtime,
            "config": config,
        }

    results = solve_by_selected_method(config)

    if results.get("model_mode") == "hungarian":
        if config.display_mode == "summary_and_figure":
            print_summary_tables(results)
        elif config.display_mode == "full_and_figure":
            print_all_main_tables(results)
    else:
        if config.display_mode != "figure_only":
            print("Angular Sorting Heuristic is active: heuristic compact-block is used, without detailed Hungarian steps.")
            if config.display_mode == "full_and_figure":
                print_all_main_tables(results)
            else:
                print_summary_tables(results)

    if config.save_step_figures and (not config.fast_render_mode):
        draw_all_optimization_stages(results)

    final_state = build_state_assignments(results, optimized_until_layer=config.p - 1)
    draw_pascalgonic_state(
        results=results,
        state_assignments=final_state,
        title="Final Pascalgonic Diagram",
        save_path=config.output_final_name if config.save_final_figure else None,
        show_legend=config.show_legend,
    )

    end_time = time.perf_counter()
    results["runtime_seconds"] = end_time - start_time
    return results


# ============================================================
# USER INTERFACE
# ============================================================

def launch_pascalgonic_ui():
    prime_options = [3, 5, 7, 11, 13, 17, 23]

    title = widgets.HTML(
        value=(
            "<h3>Pascalgonic Diagram Generator</h3>"
            "<p>"
            "Stable Google Colab version.<br>"
            "Methods: <b>Hungarian Assignment</b> and <b>Angular Sorting Heuristic</b>.<br>"
            "Compare mode supports <b>1 to 4</b> selected prime values."
            "</p>"
        )
    )

    p_widget = widgets.Dropdown(
        options=prime_options,
        value=5,
        description="p:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="380px"),
    )

    target_widget = widgets.Dropdown(
        options=valid_target_binaries(5),
        value=singleton_target_binary_last(5),
        description="target_binary:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="380px"),
    )

    rotation_widget = widgets.FloatText(
        value=90.0,
        description="rotation_deg:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="380px"),
    )

    alpha_widget = widgets.FloatText(
        value=1.0,
        description="alpha:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="380px"),
    )

    beta_widget = widgets.FloatText(
        value=0.20,
        description="beta:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="380px"),
    )

    visual_method_widget = widgets.Dropdown(
        options=[
            ("Hungarian Assignment", "hungarian"),
            ("Angular Sorting Heuristic", "angular_sorting"),
        ],
        value="hungarian",
        description="Visual Method:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="430px"),
    )

    display_mode_widget = widgets.Dropdown(
        options=[
            ("Figure only", "figure_only"),
            ("Summary + figure", "summary_and_figure"),
            ("Full tables + figure", "full_and_figure"),
        ],
        value="summary_and_figure",
        description="Display mode:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="380px"),
    )

    fast_render_widget = widgets.Checkbox(
        value=False,
        description="Fast render mode",
    )

    compare_mode_widget = widgets.Checkbox(
        value=False,
        description="Compare selected p in one picture",
    )

    compare_count_widget = widgets.Dropdown(
        options=[1, 2, 3, 4],
        value=4,
        description="Compare count:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="380px"),
    )

    compare_p1_widget = widgets.Dropdown(
        options=prime_options,
        value=5,
        description="Compare p1:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="380px"),
    )

    compare_p2_widget = widgets.Dropdown(
        options=prime_options,
        value=7,
        description="Compare p2:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="380px"),
    )

    compare_p3_widget = widgets.Dropdown(
        options=prime_options,
        value=11,
        description="Compare p3:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="380px"),
    )

    compare_p4_widget = widgets.Dropdown(
        options=prime_options,
        value=13,
        description="Compare p4:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="380px"),
    )

    show_labels_widget = widgets.Checkbox(value=True, description="Show cell labels")
    show_legend_widget = widgets.Checkbox(value=True, description="Show legend")

    figsize_w_widget = widgets.IntSlider(
        value=10, min=6, max=20, step=1,
        description="fig width:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="480px"),
    )

    figsize_h_widget = widgets.IntSlider(
        value=10, min=6, max=20, step=1,
        description="fig height:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="480px"),
    )

    compare_figsize_w_widget = widgets.IntSlider(
        value=16, min=8, max=28, step=1,
        description="compare width:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="480px"),
    )

    compare_figsize_h_widget = widgets.IntSlider(
        value=16, min=8, max=28, step=1,
        description="compare height:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="480px"),
    )

    save_final_widget = widgets.Checkbox(value=False, description="Save final figure")
    save_steps_widget = widgets.Checkbox(value=False, description="Save step figures")

    output_dir_widget = widgets.Text(
        value="pascalgonic_steps_ui",
        description="steps folder:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="480px"),
    )

    output_name_widget = widgets.Text(
        value="pascalgonic_final_ui.png",
        description="final image:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="480px"),
    )

    compare_output_name_widget = widgets.Text(
        value="pascalgonic_compare_selected.png",
        description="compare image:",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="480px"),
    )

    info_box = widgets.HTML()
    output = widgets.Output()

    generate_button = widgets.Button(description="Generate", button_style="success", icon="play")
    clear_button = widgets.Button(description="Clear Output", button_style="warning", icon="trash")

    def update_target_options(*args):
        p = p_widget.value
        old_value = target_widget.value
        new_options = valid_target_binaries(p)

        target_widget.options = new_options
        if old_value in new_options:
            target_widget.value = old_value
        else:
            target_widget.value = singleton_target_binary_last(p)

        info_box.value = (
            f"<b>Info:</b> For p = {p}, there are {len(new_options)} valid target binaries. "
            f"When compare mode is active, the target is automatically set to the last singleton for each selected p. "
            f"Hungarian Assignment is exact, while Angular Sorting Heuristic is faster."
        )

    def update_compare_widgets_visibility(*args):
        active = compare_mode_widget.value
        count = compare_count_widget.value

        compare_count_widget.layout.display = "" if active else "none"
        compare_p1_widget.layout.display = "" if active else "none"
        compare_p2_widget.layout.display = "" if (active and count >= 2) else "none"
        compare_p3_widget.layout.display = "" if (active and count >= 3) else "none"
        compare_p4_widget.layout.display = "" if (active and count >= 4) else "none"
        compare_figsize_w_widget.layout.display = "" if active else "none"
        compare_figsize_h_widget.layout.display = "" if active else "none"
        compare_output_name_widget.layout.display = "" if active else "none"

    p_widget.observe(update_target_options, names="value")
    compare_mode_widget.observe(update_compare_widgets_visibility, names="value")
    compare_count_widget.observe(update_compare_widgets_visibility, names="value")

    update_target_options()
    update_compare_widgets_visibility()

    def on_generate_clicked(_):
        with output:
            clear_output(wait=True)

            p = p_widget.value
            target_binary = target_widget.value
            compare_mode = compare_mode_widget.value

            compare_primes_raw = [
                compare_p1_widget.value,
                compare_p2_widget.value,
                compare_p3_widget.value,
                compare_p4_widget.value,
            ]
            compare_count = compare_count_widget.value
            selected_compare_primes = compare_primes_raw[:compare_count]

            if compare_mode:
                if len(set(selected_compare_primes)) != len(selected_compare_primes):
                    print("Error: selected compare p values must be distinct.")
                    return
            else:
                if not is_prime(p):
                    print(f"Error: p = {p} is not prime.")
                    return
                if len(target_binary) != p:
                    print("Error: target_binary length must equal p.")
                    return

            print("=" * 110)
            print("RUNNING PASCALGONIC DIAGRAM GENERATOR")
            print("=" * 110)
            print(f"p : {p}")
            print(f"target_binary : {target_binary}")
            print(f"rotation_deg : {rotation_widget.value}")
            print(f"alpha : {alpha_widget.value}")
            print(f"beta : {beta_widget.value}")
            print(f"visual_method : {visual_method_widget.value}")
            print(f"display_mode : {display_mode_widget.value}")
            print(f"fast_render_mode : {fast_render_widget.value}")
            print(f"compare_mode : {compare_mode_widget.value}")
            if compare_mode:
                print(f"compare_count : {compare_count}")
                print(f"compare_primes : {selected_compare_primes}")
            print(f"show_cell_labels : {show_labels_widget.value}")
            print(f"show_legend : {show_legend_widget.value}")
            print(f"save_final : {save_final_widget.value}")
            print(f"save_steps : {save_steps_widget.value}")
            print("=" * 110)

            config = PascalgonicConfig(
                p=p,
                target_binary=target_binary,
                rotation_deg=rotation_widget.value,
                alpha=alpha_widget.value,
                beta=beta_widget.value,
                figsize=(figsize_w_widget.value, figsize_h_widget.value),
                display_mode=display_mode_widget.value,
                show_legend=show_legend_widget.value,
                show_cell_labels=show_labels_widget.value,
                save_final_figure=save_final_widget.value,
                save_step_figures=save_steps_widget.value,
                output_dir_steps=output_dir_widget.value,
                output_final_name=output_name_widget.value,
                fast_render_mode=fast_render_widget.value,
                visual_method=visual_method_widget.value,
                compare_mode=compare_mode_widget.value,
                compare_primes=tuple(selected_compare_primes),
                compare_figsize=(compare_figsize_w_widget.value, compare_figsize_h_widget.value),
                compare_output_name=compare_output_name_widget.value,
            )

            try:
                results = run_pascalgonic(config)

                runtime_seconds = results.get("runtime_seconds", None)
                print("\nFinished.")

                if runtime_seconds is not None:
                    print(f"Total runtime: {runtime_seconds:.4f} seconds")

                if compare_mode:
                    print(f"Compare mode is active: showing selected p values in one picture: {list(config.compare_primes)}")
                    if save_final_widget.value:
                        print(f"Compare figure saved to: {config.compare_output_name}")
                else:
                    method_used = results.get("model_mode", resolve_visual_method(config))
                    print(f"Method used: {pretty_method_name(method_used)}")

                    if save_final_widget.value:
                        print(f"Final figure saved to: {config.output_final_name}")

                    if save_steps_widget.value and (not config.fast_render_mode):
                        print(f"Step figures saved to folder: {config.output_dir_steps}")

                    if config.fast_render_mode:
                        print("Fast render mode is active: labels are hidden and sector sampling is reduced.")

            except Exception as e:
                print("An error occurred while running the program:")
                print(str(e))

    def on_clear_clicked(_):
        with output:
            clear_output()

    generate_button.on_click(on_generate_clicked)
    clear_button.on_click(on_clear_clicked)

    left_box = widgets.VBox(
        [
            p_widget,
            target_widget,
            rotation_widget,
            alpha_widget,
            beta_widget,
            visual_method_widget,
            display_mode_widget,
            fast_render_widget,
            compare_mode_widget,
            compare_count_widget,
            compare_p1_widget,
            compare_p2_widget,
            compare_p3_widget,
            compare_p4_widget,
        ]
    )

    right_box = widgets.VBox(
        [
            show_labels_widget,
            show_legend_widget,
            figsize_w_widget,
            figsize_h_widget,
            compare_figsize_w_widget,
            compare_figsize_h_widget,
            save_final_widget,
            save_steps_widget,
            output_dir_widget,
            output_name_widget,
            compare_output_name_widget,
        ]
    )

    controls = widgets.HBox([left_box, right_box])
    buttons = widgets.HBox([generate_button, clear_button])

    ui = widgets.VBox([title, info_box, controls, buttons, output])
    display(ui)


launch_pascalgonic_ui()
