# Copyright 2023 The SapientML Authors
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
import io
from dataclasses import dataclass, field

import pandas as pd
from sapientml import SapientML

from .generate_code_thread import GenerateCodeThread
from .make_result_thread import MakeResultThread


@dataclass
class QueryParamsState:
    is_tutorial: bool = None
    lang: str = None


@dataclass
class TrainDataState:
    df_train: pd.DataFrame = None
    csv_encoding: str = "UTF-8"
    csv_name: str = "train.csv"
    ex: Exception = None
    ex_msg: str = None


@dataclass
class ConfigurationsState:
    configured: str = None
    config: dict = field(default_factory=dict)


@dataclass
class CodeGenerationState:
    thread_generatecode: GenerateCodeThread = None
    result = None
    ex: Exception = None
    log_stream: io.StringIO = None
    log_message: str = None
    sml: SapientML = None


@dataclass
class MakeResultState:
    thread_makeresult: MakeResultThread = None
    files: dict = None


@dataclass
class PredictState:
    df_predict: pd.DataFrame = None
    load_time: str = None
    ex: Exception = None
    ex_msg: str = None
