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
import numpy as np

from .metrics import METRIC_NEEDING_PREDICT_PROBA


def check_needing_predict_proba(metric: str) -> bool:
    ret = [x for x in METRIC_NEEDING_PREDICT_PROBA if metric.lower() == x.lower()]
    if len(ret) == 1:
        return True
    elif metric.startswith("MAP_"):
        return True
    return False


def check_columns(df1_columns, df2_columns):
    for i in range(len(df1_columns)):
        if df1_columns[i] != df2_columns[i]:
            return False
    return True


def convert_int64(obj):
    if isinstance(obj, dict):
        return {key: convert_int64(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64(item) for item in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj
