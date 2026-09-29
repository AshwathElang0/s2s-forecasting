"""Train and evaluate S2S prediction baselines and the proper learnable weighting factor models.

Models compared:
  1. Raw Ensemble
  2. Climatology
  3. Fixed-Strength Correction (Zhang et al. style)
  4. Heuristic Spread-Gated Correction (MVP hand-crafted percentile gate)
  5. Learned Parametric Gate (Calibrated logistic gate directly optimizing Brier Score)
  6. Learned Neural Dual-Head Network (PyTorch end-to-end differentiable confidence gate)

Features:
  - Temporal (rolling-origin) train/test split.
  - Physical teleconnection attribution (MJO amplitude/phase, ENSO ONI).
  - Spatial skill mapping across CONUS.
  - Publication-ready diagnostic visualizations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from s2s_data.config import CACHE_DIR, TARGET_LAT, TARGET_LON
from s2s_models.baselines import (
    ClimatologyBaseline,
    FixedCorrectionBaseline,
    LearnedParametricGateBaseline,
    RawEnsembleBaseline,
    SpreadGatedBaseline,
)
from s2s_models.features import DatasetLoader
from s2s_models.metrics import compute_all_metrics, reliability_curve_data
from s2s_models.neural_gated import NeuralGatedBaseline


def plot_results(
    results_by_lead: dict[int, dict[str, dict[str, float]]],
    reliability_data: dict[int, dict[str, dict[str, np.ndarray]]],
    output_dir: Path,
) -> None:
    """Generate and save publication-quality diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lead_weeks = sorted(results_by_lead.keys())
    models = [
        "raw_ensemble",
        "climatology",
        "fixed_correction",
        "spread_gated",
        "learned_param",
        "learned_neural",
    ]
    colors = {
        "raw_ensemble": "#4A90E2",
        "climatology": "#9E9E9E",
        "fixed_correction": "#F5A623",
        "spread_gated": "#2ECC71",
        "learned_param": "#1ABC9C",
        "learned_neural": "#9B59B6",
    }
    labels = {
        "raw_ensemble": "B1: Raw Ensemble",
        "climatology": "B2: Climatology",
        "fixed_correction": "B3: Fixed Correction",
        "spread_gated": "B4: Heuristic Spread-Gated",
        "learned_param": "B5: Learned Parametric Gate",
        "learned_neural": "B6: Learned Neural Dual-Head (Proposed)",
    }

    # 1. Brier Skill Score vs Lead Week
    plt.figure(figsize=(10, 6))
    for m in models:
        if m in results_by_lead[lead_weeks[0]]:
            scores = [results_by_lead[w][m]["brier_skill_score"] for w in lead_weeks]
            plt.plot(lead_weeks, scores, marker="o", linewidth=2.2, label=labels[m], color=colors[m])

    plt.axhline(0.0, color="#666666", linestyle="--", alpha=0.7, label="Climatology Skill (0.0)")
    plt.xlabel("Forecast Lead Week", fontsize=12, fontweight="bold")
    plt.ylabel("Brier Skill Score (BSS vs. Climatology)", fontsize=12, fontweight="bold")
    plt.title("S2S Extreme Precipitation Forecast Skill: Baselines vs. Learned Gating", fontsize=13, fontweight="bold")
    plt.xticks(lead_weeks, [f"Week {w}\n(Days {7*(w-1)+1}–{7*w})" for w in lead_weeks], fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "bss_by_lead_week.png", dpi=200)
    plt.close()

    # 2. Reliability Calibration Curves
    n_leads = len(lead_weeks)
    fig, axes = plt.subplots(1, n_leads, figsize=(5 * n_leads, 4.5), sharey=True)
    if n_leads == 1:
        axes = [axes]

    for idx, w in enumerate(lead_weeks):
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect Calibration")
        for m in ["fixed_correction", "spread_gated", "learned_neural"]:
            if m in reliability_data[w]:
                cal = reliability_data[w][m]
                valid = ~np.isnan(cal["mean_predicted"]) & ~np.isnan(cal["fraction_positives"])
                if np.any(valid):
                    ax.plot(
                        cal["mean_predicted"][valid],
                        cal["fraction_positives"][valid],
                        marker="s",
                        linewidth=1.8,
                        label=labels[m],
                        color=colors[m],
                    )
        ax.set_title(f"Lead Week {w}", fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Observed Frequency", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Reliability Calibration Curves by Lead Week", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "reliability_curves.png", dpi=200)
    plt.close()


def plot_physical_attribution(
    attribution_data: dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot relationship between confidence c and physical teleconnection indices (MJO, ONI)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Confidence vs MJO Amplitude
    ax = axes[0]
    mjo_amp = attribution_data["mjo_amplitude"]
    conf = attribution_data["confidence"]
    ax.scatter(mjo_amp, conf, alpha=0.25, color="#2980B9", s=12)
    valid = ~np.isnan(mjo_amp) & ~np.isnan(conf)
    if np.sum(valid) > 5:
        m, b = np.polyfit(mjo_amp[valid], conf[valid], 1)
        r = float(np.corrcoef(mjo_amp[valid], conf[valid])[0, 1])
        x_vals = np.linspace(np.min(mjo_amp[valid]), np.max(mjo_amp[valid]), 100)
        ax.plot(x_vals, m * x_vals + b, color="#C0392B", linewidth=2.2, label=f"Trend (r = {r:+.3f})")
        ax.legend()
    ax.set_xlabel("MJO Amplitude", fontweight="bold")
    ax.set_ylabel("Learned Model Confidence c", fontweight="bold")
    ax.set_title("Confidence vs. MJO Strength", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)

    # 2. Confidence by MJO Phase (1 to 8)
    ax = axes[1]
    phases = list(range(1, 9))
    phase_means = [attribution_data["phase_means"].get(p, np.nan) for p in phases]
    ax.bar(phases, phase_means, color="#27AE60", alpha=0.85, edgecolor="#1E8449")
    ax.set_xlabel("MJO Phase (Wheeler-Hendon RMM)", fontweight="bold")
    ax.set_ylabel("Mean Confidence c", fontweight="bold")
    ax.set_title("Confidence by MJO Convective Phase", fontweight="bold")
    ax.set_xticks(phases)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    # 3. Confidence vs Oceanic Nino Index (ONI)
    ax = axes[2]
    oni = attribution_data["oni"]
    ax.scatter(oni, conf, alpha=0.25, color="#8E44AD", s=12)
    valid_oni = ~np.isnan(oni) & ~np.isnan(conf)
    if np.sum(valid_oni) > 5:
        m_oni, b_oni = np.polyfit(oni[valid_oni], conf[valid_oni], 1)
        r_oni = float(np.corrcoef(oni[valid_oni], conf[valid_oni])[0, 1])
        x_vals = np.linspace(np.min(oni[valid_oni]), np.max(oni[valid_oni]), 100)
        ax.plot(x_vals, m_oni * x_vals + b_oni, color="#C0392B", linewidth=2.2, label=f"Trend (r = {r_oni:+.3f})")
        ax.legend()
    ax.set_xlabel("Oceanic Niño Index (ONI / ENSO)", fontweight="bold")
    ax.set_ylabel("Learned Model Confidence c", fontweight="bold")
    ax.set_title("Confidence vs. ENSO State", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Physical Teleconnection Attribution of Learned Confidence", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "physical_attribution.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_spatial_skill(
    spatial_skill_by_lead: dict[int, np.ndarray],
    loader: DatasetLoader,
    output_dir: Path,
) -> None:
    """Plot 2D CONUS map of Brier Skill Score for each lead week."""
    lead_weeks = sorted(spatial_skill_by_lead.keys())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, w in enumerate(lead_weeks):
        ax = axes[idx]
        grid_2d = np.full((len(TARGET_LAT), len(TARGET_LON)), np.nan)
        grid_2d[loader.land_mask] = spatial_skill_by_lead[w]

        im = ax.imshow(
            grid_2d,
            origin="lower",
            extent=[float(TARGET_LON.min()), float(TARGET_LON.max()), float(TARGET_LAT.min()), float(TARGET_LAT.max())],
            cmap="RdYlBu",
            vmin=-0.05,
            vmax=0.05,
            aspect="auto",
        )
        ax.set_title(f"Lead Week {w} (Days {7*(w-1)+1}–{7*w})", fontweight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, label="BSS vs Climatology")

    fig.suptitle("CONUS Spatial Brier Skill Score by Lead Week (Learned Neural Model)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "spatial_skill_maps.png", dpi=200, bbox_inches="tight")
    plt.close()


def run_comparison(
    lead_weeks: tuple[int, ...] = (1, 2, 3, 4),
    model_type: str = "hist_gb",
    train_end_date: dt.date | None = None,
    test_fraction: float = 0.5,
    gate_method: str = "percentile",
    include_learned: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or (CACHE_DIR / "eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DatasetLoader()
    available_dates = loader.available_init_dates()
    print(f"Total available init dates: {len(available_dates)}")
    if len(available_dates) < 2:
        raise ValueError(
            f"Need at least 2 init dates to split into train and test, found {len(available_dates)}."
        )

    if train_end_date is not None:
        train_dates = [d for d in available_dates if d <= train_end_date]
        test_dates = [d for d in available_dates if d > train_end_date]
    else:
        split_idx = max(1, int(len(available_dates) * (1.0 - test_fraction)))
        train_dates = available_dates[:split_idx]
        test_dates = available_dates[split_idx:]

    print(f"Temporal Split:")
    print(f"  Train: {len(train_dates)} dates ({train_dates[0]} to {train_dates[-1]})")
    print(f"  Test:  {len(test_dates)} dates ({test_dates[0]} to {test_dates[-1]})")

    results_by_lead = {}
    reliability_data_by_lead = {}
    spatial_skill_by_lead = {}
    all_confidences = []
    all_mjo_amp = []
    all_mjo_phase = []
    all_oni = []

    for w in lead_weeks:
        print(f"\n{'=' * 70}\nEvaluating Lead Week {w}\n{'=' * 70}")
        ds_train = loader.extract_lead_week(w, start_date=train_dates[0], end_date=train_dates[-1])
        ds_test = loader.extract_lead_week(w, start_date=test_dates[0], end_date=test_dates[-1])

        print(f"  Samples: Train = {len(ds_train.y):,}, Test = {len(ds_test.y):,}")
        print(f"  Test extreme base rate: {float(np.mean(ds_test.y)):.4%}")

        # Baseline 1: Raw Ensemble
        base_raw = RawEnsembleBaseline().fit(ds_train.X, ds_train.y)
        p_test_raw = base_raw.predict_proba(ds_test.p_raw)

        # Baseline 2: Climatology
        base_clim = ClimatologyBaseline().fit(ds_train.X, ds_train.y)
        p_test_clim = base_clim.predict_proba(len(ds_test.y))

        # Baseline 3: Fixed-strength ML correction
        base_fixed = FixedCorrectionBaseline(model_type=model_type, random_state=42)
        base_fixed.fit(ds_train.X, ds_train.y)
        p_test_fixed = base_fixed.predict_proba(ds_test.X)

        # Baseline 4: Heuristic Spread-gated correction (MVP)
        base_gated = SpreadGatedBaseline(
            base_model=base_fixed,
            gate_method=gate_method,
        )
        base_gated.fit(ds_train.X, ds_train.y, ds_train.spread)
        p_test_gated, conf_heuristic = base_gated.predict_proba(
            ds_test.X, ds_test.spread, return_confidence=True
        )

        m_raw = compute_all_metrics(ds_test.y, p_test_raw, y_ref_prob=p_test_clim)
        m_clim = compute_all_metrics(ds_test.y, p_test_clim, y_ref_prob=p_test_clim)
        m_fixed = compute_all_metrics(ds_test.y, p_test_fixed, y_ref_prob=p_test_clim)
        m_gated = compute_all_metrics(ds_test.y, p_test_gated, y_ref_prob=p_test_clim)

        lead_results = {
            "raw_ensemble": m_raw,
            "climatology": m_clim,
            "fixed_correction": m_fixed,
            "spread_gated": m_gated,
            "delta_bss_gated_vs_fixed": m_gated["brier_skill_score"] - m_fixed["brier_skill_score"],
            "mean_confidence_heuristic": float(np.mean(conf_heuristic)),
        }
        lead_reliability = {
            "raw_ensemble": reliability_curve_data(ds_test.y, p_test_raw),
            "fixed_correction": reliability_curve_data(ds_test.y, p_test_fixed),
            "spread_gated": reliability_curve_data(ds_test.y, p_test_gated),
        }

        print(f"  [B1] Raw Ensemble       : BSS={m_raw['brier_skill_score']:+.4f} | BS={m_raw['brier_score']:.6f} | AUC={m_raw['roc_auc']:.4f}")
        print(f"  [B2] Climatology        : BSS={m_clim['brier_skill_score']:+.4f} | BS={m_clim['brier_score']:.6f}")
        print(f"  [B3] Fixed ML Correction: BSS={m_fixed['brier_skill_score']:+.4f} | BS={m_fixed['brier_score']:.6f} | AUC={m_fixed['roc_auc']:.4f}")
        print(f"  [B4] Heuristic Gated    : BSS={m_gated['brier_skill_score']:+.4f} | BS={m_gated['brier_score']:.6f} | AUC={m_gated['roc_auc']:.4f}")

        if include_learned:
            # Baseline 5: Learned Parametric Gate
            base_param = LearnedParametricGateBaseline(base_model=base_fixed)
            base_param.fit(ds_train.X, ds_train.y, ds_train.spread)
            p_test_param, conf_param = base_param.predict_proba(
                ds_test.X, ds_test.spread, return_confidence=True
            )
            m_param = compute_all_metrics(ds_test.y, p_test_param, y_ref_prob=p_test_clim)
            lead_results["learned_param"] = m_param
            lead_results["mean_confidence_param"] = float(np.mean(conf_param))
            lead_results["delta_bss_param_vs_fixed"] = m_param["brier_skill_score"] - m_fixed["brier_skill_score"]
            print(f"  [B5] Learned Param Gate : BSS={m_param['brier_skill_score']:+.4f} | BS={m_param['brier_score']:.6f} | AUC={m_param['roc_auc']:.4f} (Δ BSS: {lead_results['delta_bss_param_vs_fixed']:+.5f})")

            # Baseline 6: Learned Neural Dual-Head Network (End-to-End PyTorch on GPU)
            base_neural = NeuralGatedBaseline(
                hidden_dim=64,
                epochs=20,
                batch_size=2048,
                learning_rate=1e-3,
                random_state=42,
            )
            base_neural.fit(ds_train.X, ds_train.y)
            p_test_neural, conf_neural = base_neural.predict_proba(ds_test.X, return_confidence=True)
            m_neural = compute_all_metrics(ds_test.y, p_test_neural, y_ref_prob=p_test_clim)
            lead_results["learned_neural"] = m_neural
            lead_results["mean_confidence_neural"] = float(np.mean(conf_neural))
            lead_results["delta_bss_neural_vs_fixed"] = m_neural["brier_skill_score"] - m_fixed["brier_skill_score"]
            lead_reliability["learned_neural"] = reliability_curve_data(ds_test.y, p_test_neural)
            print(f"  [B6] Learned Neural Gate: BSS={m_neural['brier_skill_score']:+.4f} | BS={m_neural['brier_score']:.6f} | AUC={m_neural['roc_auc']:.4f} (Δ BSS: {lead_results['delta_bss_neural_vs_fixed']:+.5f})")

            # Spatial skill calculation per land grid point
            n_land = loader.n_land_cells
            n_test_dates = len(test_dates)
            y_2d = ds_test.y.reshape(n_test_dates, n_land)
            p_pred_2d = p_test_neural.reshape(n_test_dates, n_land)
            p_clim_val = float(np.mean(ds_test.y))
            bs_grid = np.mean((p_pred_2d - y_2d) ** 2, axis=0)
            bs_ref_grid = np.mean((p_clim_val - y_2d) ** 2, axis=0)
            bss_grid = 1.0 - (bs_grid / np.maximum(bs_ref_grid, 1e-8))
            spatial_skill_by_lead[w] = bss_grid

            all_confidences.append(conf_neural)
            all_mjo_amp.append(ds_test.X["mjo_amplitude"].values)
            all_mjo_phase.append(ds_test.X["mjo_phase"].values)
            all_oni.append(ds_test.X["oni"].values)

        results_by_lead[w] = lead_results
        reliability_data_by_lead[w] = lead_reliability

    # Plot diagnostic figures
    plot_results(results_by_lead, reliability_data_by_lead, output_dir)

    if include_learned and spatial_skill_by_lead:
        plot_spatial_skill(spatial_skill_by_lead, loader, output_dir)

        # Physical attribution summary
        conf_cat = np.concatenate(all_confidences)
        mjo_amp_cat = np.concatenate(all_mjo_amp)
        mjo_phase_cat = np.concatenate(all_mjo_phase)
        oni_cat = np.concatenate(all_oni)

        # Subsample for plotting efficiency
        idx_sample = np.random.choice(len(conf_cat), size=min(10000, len(conf_cat)), replace=False)
        phase_means = {}
        for p in range(1, 9):
            mask = mjo_phase_cat == p
            if np.any(mask):
                phase_means[p] = float(np.mean(conf_cat[mask]))

        attribution_data = {
            "confidence": conf_cat[idx_sample],
            "mjo_amplitude": mjo_amp_cat[idx_sample],
            "mjo_phase": mjo_phase_cat[idx_sample],
            "oni": oni_cat[idx_sample],
            "phase_means": phase_means,
            "corr_mjo_amplitude": float(np.corrcoef(mjo_amp_cat[~np.isnan(mjo_amp_cat)], conf_cat[~np.isnan(mjo_amp_cat)])[0, 1]),
            "corr_oni": float(np.corrcoef(oni_cat[~np.isnan(oni_cat)], conf_cat[~np.isnan(oni_cat)])[0, 1]),
        }
        plot_physical_attribution(attribution_data, output_dir)
    else:
        attribution_data = {}

    # Save summary report and JSON
    summary_data = {
        "metadata": {
            "evaluated_at": dt.datetime.now().isoformat(),
            "train_dates": [str(d) for d in train_dates],
            "test_dates": [str(d) for d in test_dates],
            "n_train_dates": len(train_dates),
            "n_test_dates": len(test_dates),
            "model_type": model_type,
            "gate_method": gate_method,
            "include_learned": include_learned,
        },
        "results_by_lead": {
            str(k): v for k, v in results_by_lead.items()
        },
        "attribution": {
            k: v for k, v in attribution_data.items() if isinstance(v, (int, float, dict))
        },
    }
    with open(output_dir / "full_experiment_results.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # Generate Markdown Summary Table
    md_lines = [
        "# S2S Prediction Full-Scale Experiment: Baselines vs. Learnable Weighting Factor",
        "",
        f"- **Train Period**: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} dates, {len(train_dates)*loader.n_land_cells:,} samples/lead)",
        f"- **Test Period**: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} dates, {len(test_dates)*loader.n_land_cells:,} samples/lead)",
        f"- **Total Out-of-Sample Test Instances**: {len(test_dates)*loader.n_land_cells * len(lead_weeks):,}",
        f"- **Base ML Model**: `{model_type}`",
        "",
        "## Performance Metrics (BSS vs Climatology)",
        "",
        "| Lead Week | Raw Ensemble (B1) | Climatology (B2) | Fixed ML (B3) | Heuristic Gated (B4) | Learned Param (B5) | Learned Neural (B6) | Gating Gain (Neural vs Fixed) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for w in lead_weeks:
        r = results_by_lead[w]
        b1 = f"{r['raw_ensemble']['brier_skill_score']:+.4f}"
        b2 = f"{r['climatology']['brier_skill_score']:+.4f}"
        b3 = f"{r['fixed_correction']['brier_skill_score']:+.4f}"
        b4 = f"{r['spread_gated']['brier_skill_score']:+.4f}"
        b5 = f"{r['learned_param']['brier_skill_score']:+.4f}" if "learned_param" in r else "N/A"
        b6 = f"{r['learned_neural']['brier_skill_score']:+.4f}" if "learned_neural" in r else "N/A"
        delta = f"**{r.get('delta_bss_neural_vs_fixed', 0.0):+.4f}**" if "learned_neural" in r else "N/A"
        md_lines.append(f"| Week {w} | {b1} | {b2} | {b3} | {b4} | {b5} | {b6} | {delta} |")

    md_lines.extend([
        "",
        "## Discrimination Metrics (ROC-AUC)",
        "",
        "| Lead Week | Raw Ensemble | Fixed ML | Heuristic Gated | Learned Param | Learned Neural |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for w in lead_weeks:
        r = results_by_lead[w]
        auc_raw = f"{r['raw_ensemble']['roc_auc']:.4f}"
        auc_fix = f"{r['fixed_correction']['roc_auc']:.4f}"
        auc_gate = f"{r['spread_gated']['roc_auc']:.4f}"
        auc_param = f"{r['learned_param']['roc_auc']:.4f}" if "learned_param" in r else "N/A"
        auc_neural = f"{r['learned_neural']['roc_auc']:.4f}" if "learned_neural" in r else "N/A"
        md_lines.append(f"| Week {w} | {auc_raw} | {auc_fix} | {auc_gate} | {auc_param} | {auc_neural} |")

    if attribution_data:
        md_lines.extend([
            "",
            "## Physical Teleconnection Attribution",
            "",
            f"- **Correlation with MJO Amplitude**: `r = {attribution_data.get('corr_mjo_amplitude', float('nan')):+.3f}`",
            f"- **Correlation with ENSO ONI**: `r = {attribution_data.get('corr_oni', float('nan')):+.3f}`",
            "",
            "### Mean Confidence by MJO Phase:",
            "| MJO Phase | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        phase_str = " | ".join(f"{phase_means.get(p, 0.0):.3f}" for p in range(1, 9))
        md_lines.append(f"| Confidence | {phase_str} |")

    md_lines.append("")
    report_text = "\n".join(md_lines)
    with open(output_dir / "full_experiment_report.md", "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nArtifacts and publication figures saved to {output_dir}")
    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-scale S2S prediction experiment.")
    parser.add_argument("--model-type", type=str, default="hist_gb", choices=["hist_gb", "random_forest"])
    parser.add_argument("--test-fraction", type=float, default=0.5, help="Test set fraction (default 0.5)")
    parser.add_argument("--gate-method", type=str, default="percentile", choices=["percentile", "exponential"])
    parser.add_argument("--train-end-date", type=str, default="2018-12-31", help="Cutoff date for training (default: 2018-12-31)")
    parser.add_argument("--no-learned", action="store_true", help="Skip learned weighting models")
    args = parser.parse_args()

    train_end = dt.date.fromisoformat(args.train_end_date) if args.train_end_date else None
    run_comparison(
        lead_weeks=(1, 2, 3, 4),
        model_type=args.model_type,
        train_end_date=train_end,
        test_fraction=args.test_fraction,
        gate_method=args.gate_method,
        include_learned=not args.no_learned,
    )


if __name__ == "__main__":
    main()
