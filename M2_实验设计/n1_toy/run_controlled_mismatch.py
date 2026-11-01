"""W7 controlled mismatch: exact constrained directions and first-round algebra only."""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import common as C
from theory import feedthrough


OUT = Path(__file__).resolve().parent / "outputs"
JSON_PATH = OUT / "controlled_mismatch.json"
PNG_PATH = OUT / "controlled_mismatch.png"

P = C.P
MU = C.MU
ALPHA_R, ALPHA_H = C.SPECTRA["A"]
BETA = C.make_beta_star(P, C.S_SMOOTH)
A, B = C.make_spectra(ALPHA_R, ALPHA_H, P)
D0 = A + MU
J = C.coord_index(P)
W = 2.0 * MU * A * BETA * B / D0**2
C_INF = float(np.sum(2.0 * MU**2 * A * BETA**2 * (B - A) / D0**3))
F_R = feedthrough(A, BETA)
SIGMA_TILDE_R2 = C.SIG_R2 + F_R
K = float(2.0 * np.sum(
    SIGMA_TILDE_R2 * (A**2 / D0**2 + A**2 * (B - A) / D0**3)
))

ENERGIES = (1e-4, 1e-2, 1e-6)
SHAPES = ("low", "white", "high")
COMMON_SHAPE = {"low": "lowfreq", "white": "white", "high": "highfreq"}
UNCONTROLLED_NORMS = (0.5, 1.0, 2.0)
ALGEBRA_SEED = 731904
RESIDUAL_TOL = 1e-8
SIGN_TOL = 1e-12
NONTRIVIAL_A_RANGE_TOL = max(1e-12, 1e-8 * abs(C_INF))
AMPLITUDE_RELATIVE_TOL = 0.10
MIN_REPEAT_CONFIGS = 2
BRENTQ_XTOL = 1e-14
BRENTQ_RTOL = 1e-14
BRENTQ_MAXITER = 200
PAIR_PROXY_COUNT = 8
PAIR_QUANTILE_COUNT = 17
PAIR_RANDOM_COUNT = 3
ALGORITHM_VERSION = "analytic_domain_constructive_q_path_v1"


def resource_snapshot():
    commands = {}
    command_env = os.environ.copy()
    command_env.pop("OMP_NUM_THREADS", None)
    for name, command in (("nproc", ["nproc"]), ("nproc_all", ["nproc", "--all"]),
                          ("uptime", ["uptime"]), ("free_h", ["free", "-h"])):
        completed = subprocess.run(command, check=True, text=True, capture_output=True,
                                   env=command_env)
        commands[name] = {"command": command, "stdout": completed.stdout.strip(),
                          "environment": "OMP_NUM_THREADS removed so nproc reports CPU availability rather than the thread cap"}
    load1, load5, load15 = os.getloadavg()
    workers = 8 if load1 > 80.0 else 24
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "commands": commands,
        "load_average": [load1, load5, load15],
        "load_threshold_for_eight_workers": 80.0,
        "selected_worker_cap_if_parallel": workers,
        "first_round_workers": 1,
        "parallel_policy": "first round is serial; any E1 process pool uses the selected total cap (24 normally, 8 when load1>80), never full nproc",
        "thread_environment": {name: os.environ[name] for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "gpu_policy": "disabled with empty CUDA_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES; no GPU code is used",
        "process_policy": "no process is killed or reniced",
    }


def constraint_residuals(z, energy, c_abs=None):
    z = np.asarray(z)
    residuals = {
        "norm_squared_minus_one": float(z @ z - 1.0),
        "energy_minus_target": float(A @ (z * z) - energy),
        "alignment_abs_minus_target": None if c_abs is None else float(BETA @ z - c_abs),
        "nonnegativity_violation": float(max(0.0, -np.min(z))),
    }
    equality_values = [abs(residuals["norm_squared_minus_one"]),
                       abs(residuals["energy_minus_target"])]
    if c_abs is not None:
        equality_values.append(abs(residuals["alignment_abs_minus_target"]))
    residuals["max_abs_equality"] = float(max(equality_values))
    residuals["max_abs_including_nonnegativity"] = float(max(
        residuals["max_abs_equality"], residuals["nonnegativity_violation"]))
    return residuals


def position_metrics(z, energy, c_abs):
    z = np.asarray(z)
    mass = np.cumsum(z)
    total = float(mass[-1])
    coordinates = {}
    for fraction in (0.01, 0.10, 0.50):
        index = int(np.searchsorted(mass, fraction * total, side="left"))
        index = min(index, P - 1)
        coordinates[f"{fraction:.2f}"] = index + 1
    return {
        "energy_centroid_j": float((A * z * z) @ J / energy),
        "alignment_centroid_j": float((BETA * z) @ J / c_abs),
        "absolute_z_mass_cumulative_coordinate_1based": coordinates,
        "absolute_z_mass_definition": "first 1-based j whose cumulative sum_{k<=j} z_k reaches the stated fraction of sum_j z_j",
    }


def algebra_for_z(z):
    a_inf = float(W @ z)
    h = a_inf - C_INF
    return {
        "A_inf": a_inf,
        "C_inf": C_INF,
        "H": h,
        "K": K,
        "ncrit": float(K / h) if h > 0.0 else None,
        "ncrit_condition": "K/H only when H>0; otherwise null",
    }


def direction_record(z, energy, c_abs_target=None, projection_error=None):
    z = np.asarray(z, dtype=float)
    actual_energy = float(A @ (z * z))
    actual_c_abs = float(BETA @ z)
    delta = -z
    return {
        "z_abs_delta": z,
        "delta": delta,
        "delta_sign_convention": "delta_j=-z_j, z_j>=0",
        "norm": float(np.linalg.norm(delta)),
        "norm_squared": float(z @ z),
        "E": actual_energy,
        "c_delta_beta_star": float(delta @ BETA),
        "abs_c": actual_c_abs,
        "target_E": energy,
        "target_abs_c": c_abs_target,
        "support_size": int(np.count_nonzero(z > 0.0)),
        "position": position_metrics(z, actual_energy, actual_c_abs),
        "residuals": constraint_residuals(z, energy, c_abs_target),
        "projection_error_squared_l2": projection_error,
        **algebra_for_z(z),
    }


def build_pair_catalog(energy):
    i_parts, k_parts, q_parts = [], [], []
    for i in range(P - 1):
        ks = np.arange(i + 1, P)
        q = (energy - A[ks]) / (A[i] - A[ks])
        valid = (q >= 0.0) & (q <= 1.0)
        if np.any(valid):
            i_parts.append(np.full(int(valid.sum()), i, dtype=int))
            k_parts.append(ks[valid])
            q_parts.append(q[valid])
    if not i_parts:
        raise RuntimeError(f"no pair vertex for E={energy}")
    i = np.concatenate(i_parts)
    k = np.concatenate(k_parts)
    q = np.concatenate(q_parts)
    sqrt_q = np.sqrt(q)
    sqrt_one_minus_q = np.sqrt(1.0 - q)
    return {
        "i": i, "k": k, "q": q,
        "sqrt_q": sqrt_q, "sqrt_one_minus_q": sqrt_one_minus_q,
        "c": BETA[i] * sqrt_q + BETA[k] * sqrt_one_minus_q,
        "position": (A[i] * q * J[i] + A[k] * (1.0 - q) * J[k]) / energy,
        "A_inf": W[i] * sqrt_q + W[k] * sqrt_one_minus_q,
    }


def catalog_pair_z(catalog, index):
    z = np.zeros(P)
    z[catalog["i"][index]] = catalog["sqrt_q"][index]
    z[catalog["k"][index]] = catalog["sqrt_one_minus_q"][index]
    return z


def pair_certificates(catalog):
    specifications = (
        ("c_min", "c", np.argmin),
        ("c_max_pair", "c", np.argmax),
        ("position_min", "position", np.argmin),
        ("position_max", "position", np.argmax),
    )
    certificates = {}
    for name, key, selector in specifications:
        index = int(selector(catalog[key]))
        z = catalog_pair_z(catalog, index)
        certificates[name] = {
            "value": float(catalog[key][index]),
            "z": z,
            "support": np.flatnonzero(z > 0.0) + 1,
            "catalog_index": index,
        }
    return certificates


def analytic_c_max(energy):
    def weighted_energy(r):
        denominator = 1.0 + r * A
        weights = (BETA / denominator)**2
        return float(A @ weights / weights.sum())

    epsilon = 1e-12
    for _ in range(20):
        lower = -1.0 / float(A.max()) + epsilon
        if weighted_energy(lower) > energy:
            break
        epsilon *= 0.1
    else:
        raise RuntimeError("could not bracket the lower KKT multiplier ratio")
    upper = 1.0
    while weighted_energy(upper) > energy:
        upper *= 2.0
        if upper > 1e18:
            raise RuntimeError("could not bracket the upper KKT multiplier ratio")
    r = brentq(lambda value: weighted_energy(value) - energy, lower, upper,
               xtol=BRENTQ_XTOL, rtol=BRENTQ_RTOL, maxiter=BRENTQ_MAXITER)
    denominator = 1.0 + r * A
    raw = BETA / denominator
    raw_norm = float(np.linalg.norm(raw))
    z = raw / raw_norm
    multiplier_u = raw_norm / 2.0
    multiplier_v = r * multiplier_u
    stationarity = BETA - 2.0 * (multiplier_u + multiplier_v * A) * z
    residuals = constraint_residuals(z, energy)
    certificate = {
        "method": "sufficient KKT for maximizing concave c(q)=sum beta*sqrt(q) over the two-linear-constraint nonnegative q polytope",
        "global_optimality_claim": True,
        "multiplier_ratio_r": float(r),
        "multiplier_u": float(multiplier_u),
        "multiplier_v": float(multiplier_v),
        "r_bracket": [float(lower), float(upper)],
        "weighted_energy_at_r": weighted_energy(r),
        "minimum_dual_denominator": float(denominator.min()),
        "dual_infeasibility": float(max(0.0, -denominator.min())),
        "max_abs_stationarity_residual": float(np.max(np.abs(stationarity))),
        "primal_residuals": residuals,
        "z_formula": "z=beta/(1+r*a)/norm(beta/(1+r*a)); u=norm/2, v=r*u",
    }
    if (residuals["max_abs_including_nonnegativity"] >= RESIDUAL_TOL or
            certificate["max_abs_stationarity_residual"] >= RESIDUAL_TOL or
            certificate["dual_infeasibility"] != 0.0):
        raise RuntimeError(f"invalid analytic c_max certificate: {certificate}")
    return float(BETA @ z), z, certificate


def alignment_path(z_start, z_end, c_target, energy):
    q_start = z_start * z_start
    q_end = z_end * z_end

    def c_on_path(t):
        return float(BETA @ np.sqrt((1.0 - t) * q_start + t * q_end))

    c_start = c_on_path(0.0)
    c_end = c_on_path(1.0)
    tolerance = 1e-13 * max(1.0, abs(c_start), abs(c_end), abs(c_target))
    if c_target < min(c_start, c_end) - tolerance or c_target > max(c_start, c_end) + tolerance:
        raise ValueError("alignment target is not bracketed by path endpoints")
    if abs(c_start - c_target) <= tolerance:
        root = 0.0
    elif abs(c_end - c_target) <= tolerance:
        root = 1.0
    elif abs(c_end - c_start) <= tolerance:
        root = 0.5
    else:
        root = brentq(lambda t: c_on_path(t) - c_target, 0.0, 1.0,
                      xtol=BRENTQ_XTOL, rtol=BRENTQ_RTOL, maxiter=BRENTQ_MAXITER)
    z = np.sqrt((1.0 - root) * q_start + root * q_end)
    return z, {
        "method": "one-dimensional brentq on q(t)=(1-t)q_start+tq_end; norm and energy are affine in q",
        "root_t": float(root),
        "c_path_start": c_start,
        "c_path_end": c_end,
        "c_target": float(c_target),
        "residuals": constraint_residuals(z, energy, c_target),
    }


def midpoint_feasible_z(z_min, z_max, c_min, c_max, energy):
    c_target = float((c_min + c_max) / 2.0)
    z, info = alignment_path(z_min, z_max, c_target, energy)
    info["role"] = "guaranteed feasible path between global c_min and global c_max"
    return z, c_target, info


def degenerate_vertex(energy):
    if energy == float(A[-1]):
        index = P - 1
    elif energy == float(A[0]):
        index = 0
    else:
        return None
    z = np.zeros(P)
    z[index] = 1.0
    return z


def target_vectors():
    return {
        shape: np.abs(C.make_delta(COMMON_SHAPE[shape], 1.0, BETA, P, anti_align=True))
        for shape in SHAPES
    }


def select_pair_endpoints(catalog, targets, rng):
    count = len(catalog["i"])
    sources = {}

    def add(index, source):
        sources.setdefault(int(index), []).append(source)

    def add_extremes(values, name, both=True):
        order = np.argsort(values, kind="stable")
        take = min(PAIR_PROXY_COUNT, count)
        selected = [(int(order[index]), "smallest") for index in range(take)]
        if both:
            selected += [(int(order[count - 1 - index]), "largest") for index in range(take)]
        else:
            selected = [(int(order[count - 1 - index]), "largest") for index in range(take)]
        for index, extreme in selected:
            add(index, f"{name}_{extreme}_proxy")

    add_extremes(catalog["c"], "alignment")
    add_extremes(catalog["A_inf"], "A_inf")
    add_extremes(catalog["position"], "energy_centroid")
    for shape, target in targets.items():
        score = (target[catalog["i"]] * catalog["sqrt_q"] +
                 target[catalog["k"]] * catalog["sqrt_one_minus_q"])
        add_extremes(score, f"project_{shape}", both=False)
    for key in ("A_inf", "position"):
        order = np.argsort(catalog[key], kind="stable")
        quantile_count = min(PAIR_QUANTILE_COUNT, count)
        positions = np.unique(np.rint(np.linspace(0, count - 1, quantile_count)).astype(int))
        for position in positions:
            add(order[position], f"{key}_ordered_quantile_{int(position)}")
    random_indices = rng.choice(count, size=min(PAIR_RANDOM_COUNT, count), replace=False)
    for random_index, index in enumerate(random_indices, 1):
        add(index, f"fixed_seed_random_pair_{random_index}")
    return sorted(sources), sources


def generate_candidates(energy, c_target, z_cmin, z_cmax, z_mid, midpoint_info,
                        catalog, targets, rng):
    started = time.time()
    endpoint_indices, endpoint_sources = select_pair_endpoints(catalog, targets, rng)
    candidates = []
    rejected = []

    def append_candidate(z, construction):
        residuals = constraint_residuals(z, energy, c_target)
        if (not np.isfinite(z).all() or
                residuals["max_abs_including_nonnegativity"] >= RESIDUAL_TOL):
            rejected.append({"construction": construction, "residuals": residuals})
            return
        candidates.append({
            "z": z,
            "construction": construction,
            "residuals": residuals,
            "A_inf": float(W @ z),
            "energy_centroid": float((A * z * z) @ J / energy),
            "projection_errors": {
                shape: float(np.sum((z - target)**2)) for shape, target in targets.items()
            },
        })

    append_candidate(z_mid, {"source": "global_c_min_to_global_c_max_midpoint",
                             "path": midpoint_info})
    for index in endpoint_indices:
        z_pair = catalog_pair_z(catalog, index)
        c_pair = float(catalog["c"][index])
        if c_pair <= c_target:
            z_other = z_cmax
            other_name = "global_c_max"
        else:
            z_other = z_cmin
            other_name = "global_c_min"
        try:
            z, path_info = alignment_path(z_pair, z_other, c_target, energy)
        except ValueError as error:
            rejected.append({"pair_catalog_index": index, "reason": str(error)})
            continue
        construction = {
            "source": f"pair_vertex_to_{other_name}",
            "pair_catalog_index": index,
            "pair_support_1based": [int(catalog["i"][index] + 1),
                                    int(catalog["k"][index] + 1)],
            "pair_selection_sources": endpoint_sources[index],
            "path": path_info,
        }
        append_candidate(z, construction)

    metadata = {
        "method_class": "constructive",
        "candidate_family": "the guaranteed midpoint path plus one-dimensional q-paths from predeclared pair-catalog endpoints to global c_min or c_max",
        "global_projection_or_extremum_claim": False,
        "range_interpretation": "all selected projections and extrema are optima only over this residual-certified constructive candidate family",
        "pair_catalog_size": len(catalog["i"]),
        "selected_pair_endpoint_count": len(endpoint_indices),
        "accepted_candidate_count": len(candidates),
        "rejected_candidate_count": len(rejected),
        "rejected_candidates": rejected,
        "max_accepted_constraint_residual": float(max(
            candidate["residuals"]["max_abs_including_nonnegativity"] for candidate in candidates
        )) if candidates else None,
        "pair_proxy_count_per_extreme": PAIR_PROXY_COUNT,
        "pair_ordered_quantile_count": PAIR_QUANTILE_COUNT,
        "fixed_seed_random_pair_count": PAIR_RANDOM_COUNT,
        "selected_endpoints": [
            {"catalog_index": index,
             "support_1based": [int(catalog["i"][index] + 1), int(catalog["k"][index] + 1)],
             "selection_sources": endpoint_sources[index]}
            for index in endpoint_indices
        ],
        "elapsed_seconds": time.time() - started,
    }
    return candidates, metadata


def candidate_optimizer_metadata(candidate, candidates, objective, projection=False):
    return {
        "method": "constructive feasible-path candidate-family selection; no high-dimensional optimizer",
        "method_class": "constructive",
        "selection_objective": objective,
        "candidate_family_size": len(candidates),
        "candidate_construction": candidate["construction"],
        "candidate_family_minimum" if projection else "candidate_family_extremum": True,
        "global_projection" if projection else "global_extremum": False,
        "scope": "selected only over the predeclared constructive candidate family, not over all feasible directions",
    }


def select_directions(candidates, targets, energy, c_target):
    directions = {}
    for shape in SHAPES:
        candidate = min(candidates, key=lambda item: item["projection_errors"][shape])
        error = candidate["projection_errors"][shape]
        record = direction_record(candidate["z"], energy, c_target, error)
        record.update({
            "feasible": True,
            "target_shape_definition": f"{COMMON_SHAPE[shape]}, normalized to unit L2 before projection",
            "optimizer": candidate_optimizer_metadata(
                candidate, candidates, f"squared L2 projection to {shape}", projection=True),
        })
        directions[f"project_{shape}"] = record
    specifications = (
        ("A_inf_min", "A_inf", min),
        ("A_inf_max", "A_inf", max),
        ("energy_centroid_min", "energy_centroid", min),
        ("energy_centroid_max", "energy_centroid", max),
    )
    for key, objective, selector in specifications:
        candidate = selector(candidates, key=lambda item: item[objective])
        record = direction_record(candidate["z"], energy, c_target)
        record.update({
            "feasible": True,
            "optimizer": candidate_optimizer_metadata(candidate, candidates, objective),
        })
        directions[key] = record
    return directions


def controlled_config(energy, rng):
    vertex = degenerate_vertex(energy)
    config = {
        "E_target": energy,
        "spectrum": "A",
        "algorithm_version": ALGORITHM_VERSION,
        "degenerate_single_point_feasible_set": vertex is not None,
    }
    targets = target_vectors()
    if vertex is not None:
        c_value = float(BETA @ vertex)
        position_value = float(J[-1] if energy == A[-1] else J[0])
        config["feasible_domain"] = {
            "method": "exact unique simplex vertex because E equals the minimum/maximum eigenvalue",
            "global_optimality_claim": True,
            "c_interval": [c_value, c_value],
            "c_midpoint_abs": c_value,
            "position_interval_without_alignment_constraint": [position_value, position_value],
            "pair_vertex_certificates": {},
        }
        config["midpoint_construction"] = {"method": "exact unique feasible vertex"}
        config["midpoint_feasible_direction"] = direction_record(vertex, energy, c_value)
        directions = {}
        for shape in SHAPES:
            error = float(np.sum((vertex - targets[shape])**2))
            record = direction_record(vertex, energy, c_value, error)
            record.update({"feasible": True,
                           "optimizer": {"method": "exact unique feasible point",
                                         "method_class": "exact",
                                         "global_projection": True,
                                         "global_extremum": True}})
            directions[f"project_{shape}"] = record
        for key in ("A_inf_min", "A_inf_max", "energy_centroid_min", "energy_centroid_max"):
            record = direction_record(vertex, energy, c_value)
            record.update({"feasible": True,
                           "optimizer": {"method": "exact unique feasible point",
                                         "method_class": "exact",
                                         "global_extremum": True}})
            directions[key] = record
        config["directions"] = directions
        config["construction"] = {"method_class": "exact unique vertex",
                                  "accepted_candidate_count": 1,
                                  "global_projection_or_extremum_claim": True}
        config["method_scope"] = "unique feasible point; all projections and extrema are globally exact"
        config["feasible"] = True
        config["all_directions_feasible"] = True
        config["benchmark_seconds"] = {"total_construction": 0.0}
        return config

    pair_started = time.time()
    catalog = build_pair_catalog(energy)
    certificates = pair_certificates(catalog)
    pair_elapsed = time.time() - pair_started
    cmax_started = time.time()
    c_max, z_cmax, cmax_certificate = analytic_c_max(energy)
    cmax_elapsed = time.time() - cmax_started
    c_min = certificates["c_min"]["value"]
    z_cmin = certificates["c_min"]["z"]
    if c_min > c_max + RESIDUAL_TOL:
        config.update({"feasible": False, "reason": "computed c interval is reversed",
                       "directions": {}})
        return config
    z_mid, c_mid, midpoint_info = midpoint_feasible_z(z_cmin, z_cmax, c_min, c_max, energy)
    config["feasible_domain"] = {
        "method": "complete pair-vertex enumeration plus analytic scalar-KKT c_max; no SLSQP",
        "c_min_certificate": "c(q) is concave, so a global minimum over the compact convex q polytope occurs at an extreme point; every two-coordinate vertex is enumerated",
        "position_certificate": "energy centroid is linear in q, so its unconstrained-c global extrema occur at pair vertices; every vertex is enumerated",
        "pair_vertex_certificates": certificates,
        "c_max_kkt_certificate": cmax_certificate,
        "c_interval": [c_min, c_max],
        "c_midpoint_abs": c_mid,
        "position_interval_without_alignment_constraint": [
            certificates["position_min"]["value"], certificates["position_max"]["value"]],
        "c_min_direction": direction_record(z_cmin, energy),
        "c_max_direction": direction_record(z_cmax, energy),
        "position_min_direction": direction_record(certificates["position_min"]["z"], energy),
        "position_max_direction": direction_record(certificates["position_max"]["z"], energy),
    }
    config["midpoint_construction"] = midpoint_info
    config["midpoint_feasible_direction"] = direction_record(z_mid, energy, c_mid)
    if constraint_residuals(z_mid, energy, c_mid)["max_abs_including_nonnegativity"] >= RESIDUAL_TOL:
        config.update({"feasible": False, "reason": "midpoint path did not meet residual tolerance",
                       "directions": {}})
        return config
    candidates, construction = generate_candidates(
        energy, c_mid, z_cmin, z_cmax, z_mid, midpoint_info, catalog, targets, rng)
    config["construction"] = construction
    config["method_scope"] = "constructive candidate-family projections and extrema; not global over the full feasible set"
    if not candidates:
        config.update({"feasible": False,
                       "reason": "no residual-certified constructive candidate; not relabelled as full-domain infeasibility",
                       "directions": {}})
        return config
    selection_started = time.time()
    directions = select_directions(candidates, targets, energy, c_mid)
    selection_elapsed = time.time() - selection_started
    config["directions"] = directions
    config["feasible"] = True
    config["all_directions_feasible"] = all(
        record.get("feasible", False) and
        record["residuals"]["max_abs_including_nonnegativity"] < RESIDUAL_TOL
        for record in directions.values())
    config["benchmark_seconds"] = {
        "pair_catalog_and_certificates": pair_elapsed,
        "analytic_c_max": cmax_elapsed,
        "candidate_generation": construction["elapsed_seconds"],
        "selection": selection_elapsed,
    }
    return config


def finite_values(records, key):
    return np.array([record[key] for record in records if record.get(key) is not None], dtype=float)


def summarize_config(config):
    records = [record for record in config["directions"].values() if record.get("feasible", False)]
    if not records:
        return {"feasible": False, "category_input": "no residual-certified controlled direction"}
    h = finite_values(records, "H")
    a_inf = finite_values(records, "A_inf")
    abs_h = np.abs(h)
    ncrit = finite_values(records, "ncrit")
    signs = np.where(h > SIGN_TOL, 1, np.where(h < -SIGN_TOL, -1, 0))
    a_range = float(a_inf.max() - a_inf.min())
    h_range = float(h.max() - h.min())
    abs_h_relative = float((abs_h.max() - abs_h.min()) / max(abs_h.max(), SIGN_TOL))
    ncrit_relative = None
    if len(ncrit) >= 2:
        ncrit_relative = float((ncrit.max() - ncrit.min()) / ncrit.max())
    sign_flip = bool(np.any(signs < 0) and np.any(signs > 0))
    nontrivial = bool(a_range > NONTRIVIAL_A_RANGE_TOL)
    stable_amplitude = bool(
        not sign_flip and nontrivial and
        (abs_h_relative >= AMPLITUDE_RELATIVE_TOL or
         (ncrit_relative is not None and ncrit_relative >= AMPLITUDE_RELATIVE_TOL))
    )
    projection_summary = {}
    for shape in SHAPES:
        record = config["directions"].get(f"project_{shape}", {})
        if record.get("feasible", False):
            projection_summary[shape] = {
                key: record[key] for key in ("E", "c_delta_beta_star", "abs_c", "H", "A_inf", "K", "ncrit", "projection_error_squared_l2")
            }
            projection_summary[shape]["position"] = record["position"]
            projection_summary[shape]["max_constraint_residual"] = record["residuals"]["max_abs_including_nonnegativity"]
        else:
            projection_summary[shape] = {"feasible": False,
                                         "reason": record.get("reason", "not solved")}
    return {
        "feasible": True,
        "method_scope": config["method_scope"],
        "ranges_are_global": config["degenerate_single_point_feasible_set"],
        "n_directions": len(records),
        "all_directions_feasible": config["all_directions_feasible"],
        "max_constraint_residual": float(max(record["residuals"]["max_abs_including_nonnegativity"] for record in records)),
        "H_min": float(h.min()), "H_max": float(h.max()),
        "H_range": h_range,
        "A_inf_min": float(a_inf.min()), "A_inf_max": float(a_inf.max()),
        "A_inf_range": a_range,
        "A_inf_range_is_nontrivial": nontrivial,
        "H_signs": signs.tolist(),
        "sign_flip": sign_flip,
        "abs_H_min": float(abs_h.min()), "abs_H_max": float(abs_h.max()),
        "abs_H_relative_range": abs_h_relative,
        "ncrit_min_if_at_least_two_positive_H": float(ncrit.min()) if len(ncrit) >= 2 else None,
        "ncrit_max_if_at_least_two_positive_H": float(ncrit.max()) if len(ncrit) >= 2 else None,
        "ncrit_relative_range": ncrit_relative,
        "stable_repeatable_amplitude_candidate": stable_amplitude,
        "H_approximately_invariant": bool(h_range <= NONTRIVIAL_A_RANGE_TOL),
        "low_white_high": projection_summary,
    }


def uncontrolled_records():
    records = []
    for norm in UNCONTROLLED_NORMS:
        for shape in SHAPES:
            delta = C.make_delta(COMMON_SHAPE[shape], norm, BETA, P, anti_align=True)
            z = np.abs(delta)
            record = direction_record(z, float(A @ (z * z)))
            record.update({
                "shape": shape,
                "requested_norm": norm,
                "control_type": "original uncontrolled N1b direction; no E or c constraint imposed",
                "norm_constraint_residual": float(z @ z - norm**2),
            })
            records.append(record)
    return records


def classify(configs):
    eligible = [config for config in configs if config.get("summary", {}).get("feasible", False)]
    sign_flip_configs = [config for config in eligible
                         if config["summary"]["sign_flip"] and config["summary"]["A_inf_range_is_nontrivial"]]
    amplitude_configs = [config for config in eligible
                         if config["summary"]["stable_repeatable_amplitude_candidate"]]
    if not eligible:
        category = "不可行"
        reason = "no (E,c) configuration has a residual-certified controlled direction"
    elif len(sign_flip_configs) >= MIN_REPEAT_CONFIGS:
        category = "强位置效应"
        reason = f"sign flip with nontrivial A_inf range in {len(sign_flip_configs)} configurations (threshold {MIN_REPEAT_CONFIGS})"
    elif len(sign_flip_configs) == 1:
        category = "未达强位置效应双配置门槛（单配置符号翻转）"
        reason = "one sign-flip configuration is real evidence but does not satisfy W7's at-least-two-configurations strong-position category; it is not relabelled amplitude-only"
    elif len(amplitude_configs) >= MIN_REPEAT_CONFIGS:
        category = "仅幅度效应"
        reason = f"no sign flip; stable |H| or ncrit variation in {len(amplitude_configs)} configurations"
    elif all(config["summary"]["H_approximately_invariant"] for config in eligible):
        category = "无独立位置效应"
        reason = "H ranges are within the predeclared numerical nontrivial-range tolerance in every feasible configuration"
    else:
        category = "无独立位置效应"
        reason = "no sign flip and the observed variation does not meet the predeclared at-least-two-configurations repeatable-amplitude threshold; no stable independent effect is claimed"
    second_round_required = bool(sign_flip_configs or len(amplitude_configs) >= MIN_REPEAT_CONFIGS)
    triggered = sign_flip_configs if sign_flip_configs else amplitude_configs
    selection = None
    if second_round_required:
        selected = triggered[0]
        directions = selected["directions"]
        a_min_record = directions["A_inf_min"]
        a_max_record = directions["A_inf_max"]
        if not a_min_record.get("feasible", False) or not a_max_record.get("feasible", False):
            second_round_required = False
            reason += "; E1 not launched because an A_inf extremum lacks a certified direction"
        else:
            midpoint = 0.5 * (a_min_record["A_inf"] + a_max_record["A_inf"])
            candidates = [(key, record) for key, record in directions.items()
                          if record.get("feasible", False) and key not in ("A_inf_min", "A_inf_max")]
            intermediate_key, _ = min(candidates, key=lambda item: abs(item[1]["A_inf"] - midpoint))
            selection = {
                "selection_rule": "first triggered configuration in preregistered ENERGIES order; A_inf min/max and the other certified direction whose A_inf is closest to their midpoint; no outcome ranking",
                "E_target": selected["E_target"],
                "c_midpoint_abs": selected["feasible_domain"]["c_midpoint_abs"],
                "A_inf_midpoint": midpoint,
                "direction_keys_in_order_min_intermediate_max": ["A_inf_min", intermediate_key, "A_inf_max"],
                "directions": [directions["A_inf_min"], directions[intermediate_key], directions["A_inf_max"]],
            }
    if second_round_required:
        second_round_reason = "sign flip present" if sign_flip_configs else "stable strong amplitude effect in at least two configurations"
        status = "required; first-round runner does not run Monte-Carlo"
    else:
        second_round_reason = "no nontrivial sign flip and no stable strong amplitude effect meeting the two-configuration threshold"
        status = "not required"
    return {
        "category": category,
        "reason": reason,
        "method_scope": "nondegenerate-E ranges and selected extrema are constructive candidate-family results, not global-extremum claims; approximate invariance within the family is not a proof of global invariance",
        "sign_flip_config_count": len(sign_flip_configs),
        "sign_flip_E_values": [config["E_target"] for config in sign_flip_configs],
        "stable_amplitude_config_count": len(amplitude_configs),
        "stable_amplitude_E_values": [config["E_target"] for config in amplitude_configs],
        "criteria": {
            "constraint_residual_strict_upper_bound": RESIDUAL_TOL,
            "sign_absolute_tolerance": SIGN_TOL,
            "nontrivial_A_inf_absolute_range_threshold": NONTRIVIAL_A_RANGE_TOL,
            "amplitude_relative_range_threshold": AMPLITUDE_RELATIVE_TOL,
            "relative_range_definition": "(max-min)/max for |H| and for positive-H ncrit",
            "minimum_repeat_configurations": MIN_REPEAT_CONFIGS,
            "strong_position": "at least two configurations with both a non-zero-tolerance H sign flip and nontrivial A_inf range",
            "strong_amplitude_for_E1": "no sign flip within a configuration and nontrivial A_inf range plus |H| or ncrit relative range >= threshold, repeated in at least two configurations",
            "single_sign_flip_exception": "W7 has no matching one-of-four final label for exactly one sign-flip configuration; it is reported explicitly and never called amplitude-only",
        },
        "second_round": {
            "required": second_round_required,
            "status": status,
            "reason": second_round_reason,
            "representative_selection": selection,
            "required_protocol_if_triggered": "new independent E1 seeds, 400 fresh reps, six near-zero lambdas, per-rep quadratic first coefficient, bootstrap CI",
        },
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(result):
    OUT.mkdir(exist_ok=True)
    safe = json_safe(result)
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(safe, handle, ensure_ascii=False, indent=1, allow_nan=False)
        handle.write("\n")
    print(f"saved {JSON_PATH}", flush=True)
    return safe


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(21, 8.5))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.35, 1.0, 1.0))
    ax_table = fig.add_subplot(grid[0, 0])
    ax_controlled = fig.add_subplot(grid[0, 1])
    ax_original = fig.add_subplot(grid[0, 2])

    rows = []
    for config in result["configurations"]:
        for shape in SHAPES:
            item = config["summary"]["low_white_high"][shape]
            if not item.get("feasible", True):
                rows.append([f"{config['E_target']:.0e}", shape, "INFEASIBLE", "", "", "", ""])
                continue
            rows.append([
                f"{config['E_target']:.0e}", shape,
                f"{item['E']:.6e}", f"{item['c_delta_beta_star']:.6e}",
                f"{item['position']['energy_centroid_j']:.2f}",
                f"{item['H']:.4e}", f"{item['max_constraint_residual']:.2e}",
            ])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=rows,
        colLabels=["target E", "shape", "actual E", "signed c", "energy centroid", "H", "max residual"],
        loc="center", cellLoc="center", colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.45)
    ax_table.set_title("Controlled low/white/high: actual invariants and residuals")

    colors = {1e-4: "C0", 1e-2: "C1", 1e-6: "C2"}
    markers = {"low": "o", "white": "s", "high": "^"}
    for config in result["configurations"]:
        energy = config["E_target"]
        if not config.get("summary", {}).get("feasible", False):
            continue
        for key, record in config["directions"].items():
            if not record.get("feasible", False):
                continue
            x = record["position"]["energy_centroid_j"]
            y = record["H"]
            if key.startswith("project_"):
                shape = key.removeprefix("project_")
                ax_controlled.scatter(x, y, color=colors[energy], marker=markers[shape], s=58,
                                      edgecolor="black", linewidth=0.35,
                                      label=f"E={energy:.0e} {shape}")
            else:
                ax_controlled.scatter(x, y, color=colors[energy], marker="x", s=42, alpha=0.8)
    ax_controlled.axhline(0.0, color="black", lw=0.9)
    ax_controlled.set_xlabel("energy centroid (1-based coordinate)")
    ax_controlled.set_ylabel(r"$H=A_\infty-C_\infty$")
    ax_controlled.set_title("Constructive candidate family; x markers are selected extrema (not global)")
    ax_controlled.grid(alpha=0.25)
    ax_controlled.legend(fontsize=7, ncol=2)

    for record in result["uncontrolled_controls"]:
        shape = record["shape"]
        norm = record["requested_norm"]
        ax_original.scatter(record["position"]["energy_centroid_j"], record["H"],
                            marker=markers[shape], s=35 + 35 * norm,
                            color={"low": "C3", "white": "C4", "high": "C5"}[shape],
                            alpha=0.75, edgecolor="black", linewidth=0.3,
                            label=f"{shape}, norm={norm:g}")
    ax_original.axhline(0.0, color="black", lw=0.9)
    ax_original.set_xlabel("energy centroid (1-based coordinate)")
    ax_original.set_ylabel(r"$H$")
    ax_original.set_title("Original uncontrolled N1b controls (E and c vary)")
    ax_original.grid(alpha=0.25)
    ax_original.legend(fontsize=7, ncol=2)

    verdict = result["verdict"]
    verdict_label = {
        "强位置效应": "strong position effect",
        "仅幅度效应": "amplitude-only effect",
        "无独立位置效应": "no independent position effect",
        "不可行": "infeasible",
        "未达强位置效应双配置门槛（单配置符号翻转）": "single sign flip; strong-effect threshold not met",
    }.get(verdict["category"], "see JSON")
    fig.suptitle(
        f"W7 controlled mismatch | constructive, not global | per-coordinate feedthrough K | "
        f"verdict: {verdict_label} | E1 required: {verdict['second_round']['required']}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PNG_PATH, dpi=160)
    plt.close(fig)
    print(f"saved {PNG_PATH}", flush=True)


def build_result(configs, resources, started, run_state="complete"):
    verdict = classify(configs)
    return {
        "experiment": "W7 controlled mismatch: spectrum position versus target-weighted energy and alignment",
        "preregistration": "M2_实验设计/W7_受控失配实验方案.md v1.0",
        "round": "first: exact algebra only; no Monte-Carlo and no E2/E3/E4 or training",
        "run_state": run_state,
        "verdict_is_final": run_state == "complete",
        "actual_workers": resources["actual_workers"],
        "config": {
            "algorithm_version": ALGORITHM_VERSION,
            "p": P, "s": C.S_SMOOTH, "mu": MU,
            "spectrum_family": "A", "alpha_r": ALPHA_R, "alpha_h": ALPHA_H,
            "sigma_r2": C.SIG_R2, "sigma_h2": C.SIG_H2,
            "n_r_for_reference": C.N_R, "n_h_for_reference": C.N_H,
            "norm_squared": 1.0,
            "energies_in_preregistered_order": ENERGIES,
            "completed_energy_values": [config["E_target"] for config in configs],
            "c_rule": "midpoint of the exact pair-certified c_min and analytic scalar-KKT global c_max interval at each E; no hand-written c",
            "target_shapes": {"low": "j^-1", "white": "1", "high": "(j/p)^2"},
            "target_normalization": "unit L2 before projection",
            "candidate_sources": "pair proxy extrema for each target, A_inf, position and alignment; A/position ordered quantiles; uniform is the white target; three fixed-seed random pair endpoints; guaranteed midpoint path",
            "optimizer": "complete pair enumeration plus scalar KKT brentq for c_max and one-dimensional q-path brentq for fixed c; no SLSQP and no dense high-dimensional optimization",
            "brentq_xtol": BRENTQ_XTOL, "brentq_rtol": BRENTQ_RTOL,
            "brentq_maxiter": BRENTQ_MAXITER,
            "pair_proxy_count_per_extreme": PAIR_PROXY_COUNT,
            "pair_ordered_quantile_count": PAIR_QUANTILE_COUNT,
            "fixed_seed_random_pair_count": PAIR_RANDOM_COUNT,
            "algebra_seed": ALGEBRA_SEED,
            "algebra_seed_scheme": "default_rng(SeedSequence([731904,0,energy_index])) with energy_index 0,1,2",
            "uncontrolled_norms": UNCONTROLLED_NORMS,
        },
        "resources_before_recompute": resources,
        "constants": {
            "C_inf": C_INF,
            "K": K,
            "K_formula": "2*sum_j (sigma_r2+F_r,j)*(a_j^2/D0_j^2+a_j^2*(b_j-a_j)/D0_j^3)",
            "feedthrough_source": "theory.feedthrough(A,BETA), elementwise F_r,j=sum_{k!=j}a_k beta_k^2",
            "feedthrough_is_per_coordinate": True,
            "total_feedthrough_used_for_K": False,
            "W_definition": "W_j=2 mu a_j beta_j b_j/D0_j^2; A_inf=W@z",
            "C_inf_definition": "sum_j 2 mu^2 a_j beta_j^2 (b_j-a_j)/D0_j^3",
        },
        "configurations": configs,
        "uncontrolled_controls": uncontrolled_records(),
        "verdict": verdict,
        "implementation_notes": [
            "K deliberately does not reuse run_n1d.py's scalar total-feedthrough approximation; the explicit W7/user requirement is per-coordinate theory.feedthrough.",
            "At E=amin=1e-6 the nonnegative unit-norm feasible set is exactly the last-coordinate vertex.",
            "W7 gives qualitative rather than numerical amplitude/nontrivial-range thresholds; fixed operational thresholds are exposed under verdict.criteria and were not tuned to an observed sign flip.",
            "Original uncontrolled controls retain every N1b norm 0.5,1,2 rather than selecting a norm after seeing H.",
            "The previous dense SLSQP approach and its results are not used.",
        ],
        "implementation_deviations": [
            "W7 section 4's high-dimensional multi-start SLSQP is replaced by exact feasible-domain certificates and constructive one-dimensional q-paths. Nondegenerate-E low/white/high projections and A_inf/position extrema are selected only within the disclosed candidate family and are not global claims.",
            "The target/uniform/random initial-value protocol is replaced by target-derived pair proxies, the uniform white target, and three fixed-seed random pair endpoints. Candidate sources are fixed before H is evaluated; no configuration is selected to manufacture a sign flip.",
        ],
        "elapsed_seconds_at_json_write": time.time() - started,
    }


def main():
    started = time.time()
    resources = resource_snapshot()
    resources["actual_workers"] = 1
    print(json.dumps({"resource_check": resources["commands"]["uptime"]["stdout"],
                      "actual_workers": 1}, ensure_ascii=False), flush=True)
    configs = []
    for energy_index, energy in enumerate(ENERGIES):
        t0 = time.time()
        print(f"[algebra] constructing feasible domain and constructive directions for E={energy}", flush=True)
        rng = np.random.default_rng(np.random.SeedSequence([ALGEBRA_SEED, 0, energy_index]))
        config = controlled_config(energy, rng)
        config["summary"] = summarize_config(config)
        config["elapsed_seconds"] = time.time() - t0
        configs.append(config)
        print(json.dumps({"E": energy, "c_interval": config.get("feasible_domain", {}).get("c_interval"),
                          "summary": config["summary"]}, default=str), flush=True)
    verdict = classify(configs)
    result = build_result(configs, resources, started, "complete")
    safe_result = write_json(result)
    plot(safe_result)
    print(json.dumps({"final_category": verdict["category"],
                      "reason": verdict["reason"],
                      "second_round": verdict["second_round"]["reason"],
                      "representative_keys": None if verdict["second_round"]["representative_selection"] is None else verdict["second_round"]["representative_selection"]["direction_keys_in_order_min_intermediate_max"],
                      "elapsed_seconds": time.time() - started}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
