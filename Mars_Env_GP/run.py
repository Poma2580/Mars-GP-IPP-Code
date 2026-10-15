"""Reproduce the two paper figures and their shared-kernel GPy models."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from GPy_true_field import DEFAULT_OUTPUT, DEFAULT_WORKBOOK, ROOT, ROVERS, load_training_data, train_all
from draw_true_field import draw_all, grid


def compare_references(models: dict, workbook_path: Path) -> dict:
    _, _, locations = grid()
    results = {}
    for rover, spec in ROVERS.items():
        with (ROOT / "reference" / spec["legacy_model"]).open("rb") as source:
            old_model = pickle.load(source)
        X, Y = load_training_data(workbook_path, rover)
        if not np.allclose(np.asarray(old_model.X), X, rtol=0, atol=1e-12):
            raise AssertionError(f"{rover}: 旧模型训练坐标与工作簿不一致")
        if not np.allclose(np.asarray(old_model.Y), Y, rtol=0, atol=1e-12):
            raise AssertionError(f"{rover}: 旧模型磁场数值与工作簿不一致")
        old_mean, _ = old_model.predict(locations)
        new_mean, _ = models[rover].predict(locations)
        error = np.asarray(new_mean) - np.asarray(old_mean)
        result = {
            "training_data_matches_legacy_model": True,
            "prediction_max_abs_difference": float(np.max(np.abs(error))),
            "prediction_rmse": float(np.sqrt(np.mean(error**2))),
            "legacy_kernel_variance": float(old_model.kern.variance.values[0]),
            "legacy_lengthscale_degrees": float(old_model.kern.lengthscale.values[0]),
            "legacy_noise_variance": float(old_model.Gaussian_noise.variance.values[0]),
        }
        if rover == "zhurong":
            old_grids = ROOT / "reference" / "legacy_grids"
            result["legacy_grids_max_abs_difference"] = {
                component: float(np.max(np.abs(old_mean[:, index].reshape((31, 31)) - np.load(old_grids / name))))
                for index, (component, name) in enumerate(
                    (("Br", "Br_v2.npy"), ("B2", "Bphi_v2.npy"), ("B3", "Bpsi_v2.npy"))
                )
            }
        results[rover] = result
        print(
            f"{rover}: 新旧预测最大绝对差 {result['prediction_max_abs_difference']:.6g} nT，"
            f"RMSE {result['prediction_rmse']:.6g} nT"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ten-datasets", action="store_true")
    parser.add_argument("--replace-rovers", action="store_true",
                        help="Back up and replace train1/test1 in ten-dataset mode")
    args = parser.parse_args()
    if args.ten_datasets:
        from batch_datasets import DEFAULT_BATCH_WORKBOOK, DEFAULT_BATCH_OUTPUT, run_batch
        run_batch(args.workbook or DEFAULT_BATCH_WORKBOOK,
                  args.output_dir or DEFAULT_BATCH_OUTPUT, args.replace_rovers)
        return
    if args.replace_rovers:
        parser.error("--replace-rovers requires --ten-datasets")
    args.workbook = args.workbook or DEFAULT_WORKBOOK
    args.output_dir = args.output_dir or DEFAULT_OUTPUT
    models = train_all(args.workbook, args.output_dir)
    draw_all(args.workbook, args.output_dir)
    comparisons = compare_references(models, args.workbook)
    parameters_path = args.output_dir / "kernel_parameters.json"
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    for rover in ROVERS:
        parameters[rover]["reference_comparison"] = comparisons[rover]
    parameters_path.write_text(json.dumps(parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
