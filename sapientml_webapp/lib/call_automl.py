import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sapientml import SapientML
from sapientml.util.json_util import JSONEncoder
from sapientml.util.logging import setup_logger
from sapientml_core.explain.main import process as explain
from sapientml_core.generator import add_prefix

logger = setup_logger()


def convert_int64(obj):
    if isinstance(obj, dict):
        return {key: convert_int64(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64(item) for item in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj


def main(args):
    config = json.loads(args.config)

    train_data_path = Path(args.train_data_path)
    train_data = pd.read_csv(train_data_path)

    target_columns = config["target_columns"]
    n_models = config.get("n_models", 3)

    output_dir = Path(config["output_dir"])

    # Streamlit 1.33とnbconvertの組み合わせはグラフ出力がなされないため、
    # EDA用のコードは、fit()では生成せず、別途生成し、実行は別プロセスで行う

    config["add_explanation"] = False
    sml = SapientML(
        target_columns, model_type=args.estimator_name, **({k: v for k, v in config.items() if k != "target_columns"})
    )

    fit_args = ["save_datasets_format", "csv_encoding", "ignore_columns", "output_dir", "test_data"]
    sml.fit(train_data, **({k: v for k, v in config.items() if k in fit_args}))

    explain(
        visualization=True,
        eda=True,
        dataframe=sml.generator.dataset.training_dataframe,
        script_path=(sml.generator.output_dir / add_prefix("final_script.py", sml.generator.config.project_name))
        .absolute()
        .as_posix(),
        target_columns=sml.generator.task.target_columns,
        problem_type=sml.generator.task.task_type,
        ignore_columns=sml.generator.dataset.ignore_columns,
        skeleton=sml.generator._best_pipeline.labels,
        explanation=sml.generator._best_pipeline.pipeline_json,
        run_info=sml.generator.debug_info,
        internal_execution=False,
        timeout=sml.generator.config.timeout_for_test,
        cancel=sml.generator.config.cancel,
    )

    # 別プロセスによるipynbの実行を待つ
    # APIアクセストークンが必要なためフィアルに書き出す
    logger.info("Running the explained notebook...")
    start_time = time.time()

    while True:
        if (output_dir / "final_script.ipynb.out.ipynb").exists():
            logger.info("Saved explained notebook")
            break
        else:
            time.sleep(1)

        if time.time() - start_time > 120:
            logger.info("Failed to execute notebook")
            break

    with open(output_dir / "sml.pkl", "wb") as f:
        pickle.dump(sml, f)

    if not Path(output_dir / "final_script_code_explainability.json").exists():
        script_code_explainability = sml.generator._best_pipeline.pipeline_json
        with open(output_dir / "script_code_explainability.json", "w") as f:
            json.dump(script_code_explainability, f, ensure_ascii=False, indent=2)

        candidates = sml.generator._candidate_scripts
        elements = [t[0] for t in candidates]

        for i in range(n_models):
            with open(output_dir / f"{i+1}_script_code_explainability.json", "w") as f:
                json.dump(elements[i].pipeline_json, f, ensure_ascii=False, indent=2)

    if not Path(output_dir / ".skeleton.json").exists():
        skeleton = sml.generator._best_pipeline.labels
        with open(output_dir / ".skeleton.json", "w") as f:
            json.dump(convert_int64(skeleton), f, ensure_ascii=False, indent=2)

    if not Path(output_dir / "run_info.json").exists():
        with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
            json.dump(sml.generator.debug_info, f, cls=JSONEncoder, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True, help="File name to process")
    parser.add_argument("--estimator_name", type=str, required=True, help="File name to process")
    parser.add_argument("--config", type=str, required=True, help="Config dictionary in JSON format")
    args = parser.parse_args()
    main(args)
