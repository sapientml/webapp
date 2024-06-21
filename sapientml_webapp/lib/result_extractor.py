# Copyright 2023-2024 The SapientML Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import re

import pandas as pd


# @staticmethod
class ResultExtractor:
    def get_preprocess(files):
        code_expl = json.loads(files["script_code_explainability.json"])

        preprocess_list = (
            list(code_expl.get("preprocessing_before_target_separation", []))
            + list(code_expl.get("preprocessing_after_target_separation", []))
            + list(code_expl.get("preprocessing_after_train_test_split", []))
        )

        return preprocess_list

    def get_candidates(files, adaptation_metric):
        model_dict = {}
        for file in files.keys():
            if file.endswith("_script_code_explainability.json"):
                code_info = json.loads(files[file])
                candidate_number = file.split("/")[-1].split("_")[0]
                model_dict[candidate_number] = code_info["model"]["explanation"]["target_component_name"].split(":")[2]
        # Get candidate number and validation score of model from run_info
        run_info = json.loads(files["run_info.json"])

        model_score_dict = {}
        for i in range(1, 4):
            # candidate_number = run_info[str(i)]["run_info"]["filename"].split("_")[0]
            candidate_number = str(i)
            candidate_score = run_info[str(i)]["run_info"]["score"]
            model_score_dict[candidate_number] = candidate_score

        model_df = pd.DataFrame.from_records(list(model_dict.items()), columns=["candidate_number", "model"])
        model_score_df = pd.DataFrame.from_records(
            list(model_score_dict.items()),
            columns=["candidate_number", f"score ({adaptation_metric})_float"],
        )
        skeleton_info = json.loads(files[".skeleton.json"])
        model_info = {k.split(":")[2]: v for k, v in skeleton_info.items() if k.startswith("MODEL:")}
        model_info = pd.DataFrame.from_records(list(model_info.items()), columns=["model", "probability"]).sort_values(
            "probability", ascending=False
        )
        model_info["probability"] /= 2
        model_info = pd.merge(model_info, model_df, on="model", how="left")
        model_info = pd.merge(model_info, model_score_df, on="candidate_number", how="left")
        model_info = model_info[
            [
                "candidate_number",
                "model",
                "probability",
                f"score ({adaptation_metric})_float",
            ]
        ]

        script_info = json.loads(files["script_code_explainability.json"])
        selected_model = script_info["model"]["explanation"]["target_component_name"].split(":")[2]
        score_index = model_info[model_info["model"] == selected_model].index

        model_info[f"score ({adaptation_metric})"] = model_info[f"score ({adaptation_metric})_float"].apply(
            lambda x: "{:.5f}".format(x) if abs(x) < 10000 else "{:.2E}".format(x)
        )
        model_info.drop(f"score ({adaptation_metric})_float", axis=1, inplace=True)

        return model_info, score_index

    def get_columns_info(files):
        # Get the candidatenum ber and validation score of the model from run_info
        run_info = json.loads(files["run_info.json"])

        return {
            "text_columns": run_info["TEXT_COLUMNS"],
            "categorical_columns": run_info["CATEGORICAL_COLS"],
            "date_columns": run_info["DATE_COLUMNS"],
        }

    def remake_run_info(script_dir):
        run_info_path = os.path.join(script_dir, "run_info.json")
        with open(run_info_path, "r") as f:
            run_info_raw = json.load(f)

        run_info = {}
        for i in range(1, 4):
            run_info[str(i)] = {}
            run_info[str(i)]["run_info"] = {}
            run_info[str(i)]["run_info"]["score"] = run_info_raw[str(i)]["run_info"]["score"]
        # To handle both of JSON-serialized Pipeline and str-dumped Pipeline,
        # check the type of `run_info["0"]["content"]` is `str` or not.
        # For details, refer to https://github.com/F-AutoML/sapientml/pull/562
        run_info_content = run_info_raw["1"]["content"]
        if not isinstance(run_info_content, str):
            run_info_content = json.dumps(run_info_content)

        # text_columns_info = re.search(r"_TEXT_COLUMNS = (\[.*?\])", run_info_content)
        text_columns_info = re.search(r"TEXT_COLUMNS = (\[.*?\])\\n", run_info_content)
        # Fix: Comment Please
        text_columns = []
        if text_columns_info:
            text_columns = json.loads(f'"{text_columns_info[1]}"')
        run_info["TEXT_COLUMNS"] = text_columns

        categorical_columns_info = re.search(r"CATEGORICAL_COLS = (\[.*?\])\\n", run_info_content)

        categorical_columns = []
        if categorical_columns_info is not None:
            categorical_columns = json.loads(f'"{categorical_columns_info[1]}"')
        run_info["CATEGORICAL_COLS"] = categorical_columns

        date_columns_info = re.search(r"DATE_COLUMNS = (\[.*?\])\\n", run_info_content)

        date_columns = []
        if date_columns_info is not None:
            date_columns = json.loads(f'"{date_columns_info[1]}"')
        run_info["DATE_COLUMNS"] = date_columns
        with open(run_info_path, "w") as f:
            json.dump(run_info, f, indent=4)
