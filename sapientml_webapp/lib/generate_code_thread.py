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
import logging
import threading
from contextvars import ContextVar
from pathlib import Path
from uuid import UUID

from sapientml import SapientML

from .utils import convert_int64


class GenerateCodeThread(threading.Thread):
    def __init__(
        self,
        sml: SapientML,
        config: dict,
        log_handler: logging.Handler,
        ctx_uuid: ContextVar[UUID],
        uuid: UUID,
    ):
        self.sml = sml
        self.config = config
        self.result = None
        self.exception = None
        self.log_handler = log_handler
        self.ctx_uuid = ctx_uuid
        self.uuid = uuid
        threading.Thread.__init__(self)

    def run(self):
        try:
            self.ctx_uuid.set(self.uuid)

            fit_args = ["save_datasets_format", "csv_encoding", "ignore_columns", "output_dir", "test_data"]
            self.sml.fit(self.config["training_dataframe"], **({k: v for k, v in self.config.items() if k in fit_args}))

            output_dir = self.config["output_dir"]
            if not Path(output_dir / "final_script_code_explainability.json").exists():
                script_code_explainability = self.sml.generator._best_pipeline.pipeline_json

                with open(output_dir / "script_code_explainability.json", "w") as f:
                    json.dump(script_code_explainability, f, ensure_ascii=False, indent=2)

                candidates = self.sml.generator._candidate_scripts
                elements = [t[0] for t in candidates]

                for i in range(3):
                    # explainability =
                    with open(output_dir / f"{i+1}_script_code_explainability.json", "w") as f:
                        json.dump(elements[i].pipeline_json, f, ensure_ascii=False, indent=2)

            if not Path(output_dir / ".skeleton.json").exists():
                skeleton = self.sml.generator._best_pipeline.labels
                with open(output_dir / ".skeleton.json", "w") as f:
                    json.dump(convert_int64(skeleton), f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.exception = e
        finally:
            pass

    def get_result(self):
        return self.result

    def get_exception(self):
        return self.exception

    def get_sml(self):
        return self.sml

    def trigger_cancel(self):
        self.cancel_token.isTriggered = True
