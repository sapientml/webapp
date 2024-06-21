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
import glob
import os
import re
import shutil
import subprocess
import threading
import zipfile
from pathlib import Path
from typing import Optional

import nbformat
from p2j.p2j import p2j

from .result_extractor import ResultExtractor


class MakeResultThread(threading.Thread):
    def __init__(
        self,
        dirpath: Path,
        csv_name: Optional[str],
        test_csv_name: Optional[str],
        csv_encoding: Optional[str],
        available_file_patterns: list,
        downloadable_files: list,
        in_memory_files: list,
        debug: bool = False,
    ):
        self.dirpath = dirpath
        self.exception = None
        self.csv_name = csv_name
        self.test_csv_name = test_csv_name
        self.csv_encoding = csv_encoding
        self.files = dict()
        self.debug = debug
        self.available_file_patterns = (
            available_file_patterns.copy()
            if available_file_patterns is not None
            else [
                "script_code_explainability.json",
                ".skeleton.json",
                "final_predict.py",
                "final_train.py",
                "permutation_importance.csv",
                "run_info.json",
                "script.ipynb",
                "script.py",
                "training.csv",
                "test.csv",
                "lib",
                "candidates",
                "*.pkl",
            ]
        )
        self.downloadable_files = (
            downloadable_files.copy()
            if downloadable_files is not None
            else [
                "final_predict.py",
                "final_train.py",
                "permutation_importance.csv",
                "run_info.json",
                "script.ipynb",
                "script.py",
                "test.csv",
                "training.csv",
                "lib/sample_dataset.py",
                "candidates/lib/sample_dataset.py",
            ]
        )
        self.in_memory_files = (
            in_memory_files.copy()
            if in_memory_files is not None
            else [
                "result.zip",
                "script_code_explainability.json",
                ".skeleton.json",
                "permutation_importance.csv",
                "run_info.json",
                "script.ipynb",
                "script.py",
                "script.html",
            ]
        )
        threading.Thread.__init__(self)

    def run(self):
        try:
            dirpath = Path(self.dirpath)
            lib_path = dirpath / "lib"
            # if os.path.exists(lib_path / "timeauto.py"):
            #     os.remove(lib_path / "timeauto.py")
            candidate_path = dirpath / "candidates"
            candidate_path.mkdir(parents=True, exist_ok=True)

            if lib_path.exists():
                candidate_lib_path = candidate_path / "lib"
                candidate_lib_path.mkdir(parents=True, exist_ok=True)
                for file in os.listdir(lib_path):
                    if os.path.isdir(os.path.join(lib_path, file)):
                        continue
                    shutil.copyfile(lib_path / file, candidate_lib_path / file)

            script_pattern = "[1-3]_script.py"
            scripts = dirpath.glob(script_pattern)
            for script in scripts:
                shutil.move(str(script), str(candidate_path / script.name))
                if "script.ipynb" in self.available_file_patterns:
                    p2j(
                        candidate_path / script.name,
                        candidate_path / f"{script.name.split('.')[-2]}.ipynb",
                        overwrite=True,
                    )

            json_pattern = "[1-3]_script_code_explainability.json"
            json_files = dirpath.glob(json_pattern)
            for json_file in json_files:
                shutil.move(str(json_file), str(candidate_path / json_file.name))

            # final_script.ipynbが存在する場合に、ファイル名を変更する
            if (dirpath / "final_script.ipynb").exists():
                (dirpath / "final_script.ipynb").rename(dirpath / "script_raw.ipynb")

            explained_notebook_path = Path(dirpath / "final_script.ipynb.out.ipynb")
            data = ""
            if explained_notebook_path.exists():
                with explained_notebook_path.open("r") as f:
                    data = f.read()
                    # if result.image_url:
                    #     data = data.replace(result.image_url, Path(result.image_url).name)
                    #     result.image_url = Path(result.image_url
                    # ).replace(dirpath / Path(result.image_url).name)

                    data = data.replace(str(dirpath), ".")

                with explained_notebook_path.open("w") as f:
                    f.write(data)
                explained_notebook_path.rename(Path(dirpath / "script.ipynb"))

            if Path(dirpath / "final_script.py").exists():
                Path(dirpath / "final_script.py").rename(Path(dirpath / "script.py"))

            ResultExtractor.remake_run_info(dirpath)

            file_names = glob.glob(os.path.join(self.dirpath, "*"))

            available_files = []
            for pattern in self.available_file_patterns:
                available_file_pattern = os.path.join(self.dirpath, pattern)
                available_files.extend(glob.glob(available_file_pattern))
            if self.debug is False:
                for file_name in file_names:
                    if file_name not in available_files:
                        if os.path.isfile(file_name):
                            os.remove(file_name)
                        else:
                            shutil.rmtree(file_name)

            self.downloadable_files += [
                os.path.relpath(p, dirpath) for p in (dirpath / "candidates").glob("*_script.py")
            ] + [os.path.relpath(p, dirpath) for p in (dirpath / "candidates").glob("*_script.ipynb")]

            zip_dir = dirpath / "zip"
            zip_dir.mkdir(parents=True, exist_ok=True)
            (zip_dir / "lib").mkdir(parents=True, exist_ok=True)
            (zip_dir / "candidates" / "lib").mkdir(parents=True, exist_ok=True)
            for file in self.downloadable_files:
                if (dirpath / Path(file)).exists():
                    shutil.copyfile(dirpath / Path(file), zip_dir / Path(file))

            # Rename file training.cs and test.csv if present
            if Path(zip_dir / "training.csv").exists():
                if "https://" not in self.csv_name:
                    Path(zip_dir / "training.csv").rename(zip_dir / self.csv_name)
                    self.downloadable_files.remove("training.csv")
                    self.downloadable_files.append(self.csv_name)

            if self.test_csv_name is not None and Path(zip_dir / "test.csv").exists():
                Path(zip_dir / "test.csv").rename(zip_dir / self.test_csv_name)
                self.downloadable_files.remove("test.csv")
                self.downloadable_files.append(self.test_csv_name)
            self.downloadable_files = [zip_dir / Path(file_name) for file_name in self.downloadable_files]

            with zipfile.ZipFile(file=dirpath / "result.zip", mode="w") as zf:
                for filepath in self.downloadable_files:
                    if filepath.is_file():
                        if os.path.splitext(filepath)[1] == ".py" and self.csv_name is not None:
                            with open(filepath, "r") as f:
                                src = f.read()
                            with open(filepath, "w") as f:
                                src = re.sub(r"r\"/.+/training\.csv", '"training.csv', src)
                                src = re.sub(r"r\"/.+/test\.csv", '"test.csv', src)
                                src = src.replace("training.csv", f"{self.csv_name}")
                                if self.test_csv_name is not None:
                                    src = src.replace("test.csv", f"{self.test_csv_name}")
                                f.write(src)

                        if os.path.splitext(filepath)[1] == ".ipynb" and self.csv_name is not None:
                            with open(filepath, "r", encoding="utf-8") as nb_file:
                                notebook_content = nb_file.read()
                            nb = nbformat.reads(notebook_content, as_version=4)
                            for cell in nb.cells:
                                if cell.cell_type == "code":
                                    if "source" in cell:
                                        cell["source"] = re.sub(
                                            r"r\"/.+/training\.csv", '"training.csv', cell["source"]
                                        )
                                        cell["source"] = re.sub(r"r\"/.+/test\.csv", '"test.csv', cell["source"])
                                        cell["source"] = (
                                            cell["source"]
                                            .replace("training.csv", f"{self.csv_name}")
                                            .replace("test.csv", f"{self.test_csv_name}")
                                        )
                                    if "outputs" in cell:
                                        outputs = cell["outputs"]
                                        for output in outputs:
                                            if "output_type" in output and output["output_type"] == "stream":
                                                output["text"] = re.sub(
                                                    r"/tmp/.+/.+\.py", "scrip.py", output.get("text", "")
                                                )
                            with open(filepath, "w", encoding="utf-8") as new_nb_file:
                                nbformat.write(nb, new_nb_file)
                        zf.write(filepath, arcname=filepath.relative_to(zip_dir))

            if self.debug is False:
                shutil.rmtree(zip_dir)

            # convert ipynb to html to display generate result
            subprocess.run(
                ["jupyter", "nbconvert", "--to", "html", dirpath / "script.ipynb"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            self.in_memory_files += (
                [os.path.relpath(p, dirpath) for p in (dirpath / "candidates").glob("*_script.py")]
                + [os.path.relpath(p, dirpath) for p in (dirpath / "candidates").glob("*_script.ipynb")]
                + [
                    os.path.relpath(p, dirpath)
                    for p in (dirpath / "candidates").glob("*_script_code_explainability.json")
                ]
            )
            for file in self.in_memory_files:
                if os.path.exists(dirpath / file):
                    with open(
                        dirpath / file,
                        "rb" if file.endswith(".zip") else "r",
                        encoding=self.csv_encoding if file == "training.csv" else None,
                    ) as f:
                        self.files[file] = f.read()
                    if self.debug is False:
                        os.remove(dirpath / file)

            if self.debug is False:
                shutil.rmtree(dirpath / "lib")
                shutil.rmtree(dirpath / "candidates")
                for filepath in dirpath.glob("*"):
                    if filepath.is_file():
                        filepath.unlink()

        except Exception as e:
            self.exception = e

    def get_exception(self):
        return self.exception

    def get_files(self):
        return self.files
