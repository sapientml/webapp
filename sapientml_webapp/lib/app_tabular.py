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
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st
from sapientml import SapientML
from sapientml.suggestion import SapientMLSuggestion
from sklearn.model_selection import train_test_split

from . import style_set as style
from .app import Application
from .i18n import use_translation
from .metrics import ADAPTATION_METRIC_FOR_CLASSIFICATION, ADAPTATION_METRIC_FOR_REGRESSION
from .session_state import ConfigurationsState


@dataclass
class SampleState:
    sample: str = " "


@dataclass
class TrainDataState:
    df_train: pd.DataFrame = None
    df_test: pd.DataFrame = None
    csv_encoding: str = "UTF-8"
    csv_name: str = "train.csv"
    ex: Exception = None
    ex_msg: str = None


@dataclass
class PredictedState:
    df_predicted: pd.DataFrame = None
    df_targets_pred: pd.DataFrame = None
    load_time: str = None


class TabularApplication(Application):
    def reset(self, delete_session=None):
        delete_session = [
            "uuid",
            "train_data_state",
            "sample",
            "previous_sample",
            "conf",
            "previous_conf",
            "predict",
            "predicted",
            "code_generation",
            "make_result",
        ]
        super().reset(delete_session)

    def section_chose_sample_datasete(self):
        (lang, is_tutorial) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        if not is_tutorial:
            return None, None, None, None, None

        if "sample" not in st.session_state:
            st.session_state.sample = SampleState()
        if "previous_sample" not in st.session_state:
            st.session_state.previous_sample = SampleState()
        state = st.session_state.sample
        default = st.session_state.previous_sample

        st.info(t("sample_data.sample_select"))

        datasets = {
            " ": " ",
            "Titanic Dataset": t("sample_data.titanic_overview"),
            "Hotel Cancellation": t("sample_data.hotel_overview"),
            "Housing Prices": t("sample_data.housing_overview"),
            "Medical Insurance Charges": t("sample_data.medical_overview"),
        }

        def format_fanc(sample):
            return datasets[sample]

        off_selectbox = False
        if "train_data_state" in st.session_state:
            if st.session_state.train_data_state.df_train is not None:
                off_selectbox = True
        sample = st.selectbox(
            t("sample_data.stb_label"),
            list(datasets.keys()),
            disabled=off_selectbox,
            index=list(datasets.keys()).index(default.sample),
            #   label_visibility="hidden",
            format_func=format_fanc,
        )
        if sample == " ":
            st.stop()
        if sample == "Titanic Dataset":
            source = "[Titanic Dataset](https://www.openml.org/d/40945)"
            sample_training_data_path = "https://github.com/sapientml/sapientml/files/12481088/titanic.csv"
            sample_test_data_path = None
            default_config = {
                "target_columns": ["survived"],
                "task_type": "classification",
                "ignore_columns": ["name", "home.dest"],
            }
            state.sample = sample
        elif sample == "Hotel Cancellation":
            source = "Hotel Cancellation"
            sample_training_data_path = (
                "https://github.com/sapientml/sapientml/files/12617021/train_hotelcancel-prediction.csv"
            )
            sample_test_data_path = (
                "https://github.com/sapientml/sapientml/files/12617033/test_hotelcancel-prediction.csv"
            )
            default_config = {"target_columns": ["Status"], "task_type": "classification", "ignore_columns": ["No."]}
            state.sample = sample
        elif sample == "Housing Prices":
            source = "Housing Prices"
            sample_training_data_path = "https://github.com/sapientml/sapientml/files/12374429/housing-prices.csv"
            sample_test_data_path = None
            default_config = {"target_columns": ["SalePrice"], "task_type": "regression", "ignore_columns": ["Id"]}
            state.sample = sample
        elif sample == "Medical Insurance Charges":
            source = "[Medical Insurance Charges](https://www.kaggle.com/datasets/harishkumardatalab/medical-insurance-price-prediction)"
            sample_training_data_path = (
                "https://github.com/sapientml/sapientml/files/12617660/train_medical-insurance-prediction.csv"
            )
            sample_test_data_path = (
                "https://github.com/sapientml/sapientml/files/12617696/test_medical-insurance-prediction.csv"
            )
            default_config = {"target_columns": ["charges"], "task_type": "regression"}
            state.sample = sample

        return sample, sample_training_data_path, sample_test_data_path, source, default_config

    def section_upload_training_data(self, sample_training_data_path, sample_test_data_path, source):
        (_, is_tutorial) = self.get_query_params()

        if "train_data_state" not in st.session_state:
            st.session_state.train_data_state = TrainDataState()
        state = st.session_state.train_data_state

        df_train, csv_encoding, csv_name = super().section_upload_training_data(
            sample_data_path=sample_training_data_path, source=source
        )

        if is_tutorial and df_train is not None:
            csv_name = sample_training_data_path
            if state.df_test is None:
                if sample_test_data_path is None:
                    df_train, df_test = train_test_split(df_train, random_state=1024)
                    state.df_train = df_train.reset_index(drop=True)
                    state.df_test = df_test.reset_index(drop=True)
                else:
                    state.df_test = pd.read_csv(sample_test_data_path)
        return state.df_train, state.df_test, csv_encoding, csv_name

    def part_advanced_settings(
        self,
        off_param_setting: bool,
        df_train: pd.DataFrame,
        task_type: str,
        default_config: dict,
        max_timeout: int = 120,
        default_timeout: int = 120,
        tutorial_config: dict = None,
    ):
        (lang, is_tutorial) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        df_columns = df_train.columns.to_list()

        if is_tutorial:
            msg_dvanced_settings = f"**{t('configuration.advanced_settings_tutorial')}**"
        else:
            msg_dvanced_settings = f"**{t('configuration.advanced_settings')}**"

        with st.expander(msg_dvanced_settings, expanded=False):
            st.write(f"**{t('configuration.task_settings')}**")

            default_ignore_columns = tutorial_config.get("ignore_columns", None) if is_tutorial else None

            ignore_columns = st.multiselect(
                t("configuration.ignore_columns"),
                df_columns,
                disabled=off_param_setting or is_tutorial,
                default=default_config.get("ignore_columns", default_ignore_columns),
                help=t("configuration.ignore_columns_desc"),
            )
            if task_type == "regression":
                adaptation_metric = st.selectbox(
                    t("configuration.adaptation_metric"),
                    ADAPTATION_METRIC_FOR_REGRESSION,
                    index=ADAPTATION_METRIC_FOR_REGRESSION.index(default_config.get("adaptation_metric", "R2")),
                    disabled=off_param_setting or is_tutorial,
                    help=t("configuration.adaptation_metric_desc"),
                )
            else:
                adaptation_metric = st.selectbox(
                    t("configuration.adaptation_metric"),
                    ADAPTATION_METRIC_FOR_CLASSIFICATION,
                    index=ADAPTATION_METRIC_FOR_CLASSIFICATION.index(default_config.get("adaptation_metric", "F1")),
                    disabled=off_param_setting or is_tutorial,
                    help=t("configuration.adaptation_metric_desc"),
                )
            timeout = st.number_input(
                t("configuration.timeout"),
                min_value=1,
                max_value=max_timeout,
                value=default_config.get("timeout_for_test", default_timeout),
                disabled=off_param_setting or is_tutorial,
                help=t("configuration.timeout_desc"),
            )

            st.write(f"**{t('configuration.train_data_split')}**")
            st.caption(t("configuration.train_data_split_caption"))
            SPLIT_METHOD = ["random", "group", "time"]
            split_method = st.selectbox(
                t("configuration.split_method"),
                SPLIT_METHOD,
                index=SPLIT_METHOD.index(default_config.get("split_method", "random")),
                disabled=off_param_setting or is_tutorial,
            )
            split_column_name = None
            if split_method == "group":
                split_column_name = st.selectbox(
                    t("configuration.split_column_name"),
                    df_columns,
                    index=df_columns.index(default_config.get("split_column_name", df_columns[0])),
                    disabled=off_param_setting or is_tutorial,
                )
            time_split_num = None
            time_split_index = None
            if split_method == "time":
                split_column_name = st.selectbox(
                    t("configuration.split_column_name"),
                    df_columns,
                    index=df_columns.index(default_config.get("split_column_name", df_columns[0])),
                    disabled=off_param_setting or is_tutorial,
                )
                time_split_num = st.number_input(
                    "Time Split Number",
                    min_value=0,
                    max_value=2**32 - 1,
                    value=default_config.get("time_split_num", 5),
                    disabled=off_param_setting or is_tutorial,
                )
                time_split_index = st.number_input(
                    "Time Split Index",
                    min_value=0,
                    max_value=2**32 - 1,
                    value=default_config.get("time_split_index", 4),
                    disabled=off_param_setting or is_tutorial,
                )
            if split_method != "time":
                split_seed = st.number_input(
                    t("configuration.seed"),
                    min_value=0,
                    max_value=2**32 - 1,
                    value=default_config.get("split_seed", 1024),
                    disabled=off_param_setting or is_tutorial,
                )
            else:
                split_seed = "1024"
            split_train_size = st.slider(
                t("configuration.split_train_size"),
                value=int(default_config.get("split_train_size", 0.75) * 100),
                min_value=5,
                max_value=95,
                disabled=off_param_setting or is_tutorial,
                help=t("configuration.split_train_size"),
            )
            split_train_size = split_train_size / 100

            st.write(f'**{t("configuration.hyperparameterTuning")}**')
            st.caption(t("configuration.hyperparameterTuning_desc"))
            hyperparameter_tuning = st.checkbox(
                t("configuration.enable"),
                value=default_config.get("hyperparameter_tuning", False),
                disabled=off_param_setting or is_tutorial,
            )
            trials = None
            random_seed = None
            if hyperparameter_tuning:
                trials = st.number_input(
                    t("configuration.hyperparameterTuningNTrials"),
                    min_value=1,
                    max_value=3600 * 24,
                    value=default_config.get("hyperparameter_tuning_n_trials", 10),
                    disabled=off_param_setting or is_tutorial,
                )
                random_seed = st.number_input(
                    t("configuration.hpo_seed"),
                    min_value=0,
                    max_value=2**32 - 1,
                    value=default_config.get("hyperparameter_tuning_random_state", 1021),
                    disabled=off_param_setting or is_tutorial,
                )

            st.write(t("configuration.explanations"))
            st.caption(t("configuration.explanations_desc"))
            permutation_importance = st.checkbox(
                t("configuration.permutation_importance"),
                value=default_config.get("permutation_importance", True),
                disabled=off_param_setting or is_tutorial,
                help=t("configuration.permutation_importance_decs"),
            )

        if hyperparameter_tuning:
            hyperparameter_tuning_timeout = timeout
            initial_time = 0
        else:
            hyperparameter_tuning_timeout = None
            initial_time = timeout

        config = {
            "ignore_columns": ignore_columns,
            "split_method": split_method,
            "split_seed": split_seed,
            "split_train_size": split_train_size,
            "adaptation_metric": adaptation_metric,
            "initial_timeout": initial_time,
            "timeout_for_test": timeout,
            "permutation_importance": permutation_importance,
            "hyperparameter_tuning": hyperparameter_tuning,
        }

        if split_column_name is not None:
            config["split_column_name"] = split_column_name
        if time_split_num is not None:
            config["time_split_num"] = time_split_num
        if time_split_num is not None:
            config["time_split_index"] = time_split_index
        if trials is not None:
            config["hyperparameter_tuning_n_trials"] = trials
        if hyperparameter_tuning_timeout is not None:
            config["hyperparameter_tuning_timeout"] = hyperparameter_tuning_timeout
        if random_seed is not None:
            config["hyperparameter_tuning_random_state"] = random_seed

        return config

    def section_configure_params(
        self, df_train: pd.DataFrame, csv_encoding: Literal["UTF-8", "SJIS"], tutorial_config: None
    ):
        (lang, is_tutorial) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        if "conf" not in st.session_state:
            st.session_state.conf = ConfigurationsState()
        if "previous_conf" not in st.session_state:
            st.session_state.previous_conf = ConfigurationsState()
        state = st.session_state.conf
        previous_state = st.session_state.previous_conf
        # Configuration
        st.markdown("<a name='set_param'></a>", unsafe_allow_html=True)
        st.write("")
        style.custom_h4(t("header.configuration"))
        st.sidebar.markdown(f"[{t('header.configuration')}](#set_param)")

        if is_tutorial:
            button = f':green[**{t("codegen.start_generation")}**]'
            st.info(
                t("configuration.tutorial").format(
                    button=button,
                    target_columns=tutorial_config["target_columns"],
                    task_type=tutorial_config["task_type"],
                )
            )

        off_param_setting = False if state.configured is None else True
        default_config = previous_state.config
        df_columns = df_train.columns.to_list()

        target_columns = st.multiselect(
            t("configuration.target_column_names"),
            df_columns,
            default=default_config.get("target_columns", None),
            help=t("configuration.target_column_names_desc"),
            disabled=off_param_setting,
        )

        task_suggestion = SapientMLSuggestion(target_columns=target_columns, dataframe=df_train)
        suggested_task = task_suggestion.suggest_task()

        if is_tutorial:
            if target_columns != tutorial_config["target_columns"]:
                st.warning(
                    t("configuration.target_columns_tutorial").format(target_columns=tutorial_config["target_columns"])
                )
                st.stop()
        elif len(target_columns) < 1:
            st.warning(t("configuration.target_column_names_error"))
            st.stop()

        if not target_columns:
            st.stop()

        task_type = st.selectbox(
            t("configuration.task_type"),
            ["classification", "regression"],
            index=0 if suggested_task == "classification" else 1,
            disabled=off_param_setting,
            help=t("configuration.task_type_desc"),
        )

        if is_tutorial:
            if task_type != tutorial_config["task_type"]:
                st.warning(t("configuration.task_type_tutorial").format(task_type=tutorial_config["task_type"]))
                st.stop()

        state.config = self.part_advanced_settings(
            off_param_setting, df_train, task_type, previous_state.config, tutorial_config=tutorial_config
        )

        uuid = st.session_state.uuid
        state.config.update(
            {
                "training_dataframe": df_train,
                "save_datasets_format": "csv",
                "add_explanation": True,
                "csv_encoding": csv_encoding,
                "output_dir": Path(f"./outputs/{uuid}"),
                "target_columns": target_columns,
                "task_type": task_type,
            }
        )

        def on_click():
            state: ConfigurationsState = st.session_state.conf
            state.configured = "sapientml"

        st.button(
            f':green[**{t("codegen.start_generation")}**]',
            on_click=on_click,
            disabled=off_param_setting,
        )

        return state.config, state.configured

    def part_view_predicted(
        self,
        config,
        df_predict,
        df_targets_pred,
    ):
        (lang, is_tutorial) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        st.markdown("<a name='prediction_result'></a>", unsafe_allow_html=True)
        st.write("")
        style.custom_h4(t("header.prediction_result"))
        st.sidebar.markdown(f"[{t('header.prediction_result')}](#prediction_result)")

        st.write(t("prediction_result.data"))

        merge_df_targets_pred = pd.merge(
            df_predict,
            df_targets_pred,
            left_index=True,
            right_index=True,
            suffixes=["_actual", "_predicted"],
        )
        target_columns = config["target_columns"]

        target_exists = set(target_columns).issubset(set(df_predict.columns))

        actual_and_predicted = []
        if target_exists:
            for name in target_columns:
                actual_and_predicted.append(name + "_actual")
                actual_and_predicted.append(name + "_predicted")

        else:
            actual_and_predicted.append(target_columns[0])

        df_predicted = merge_df_targets_pred[actual_and_predicted]
        merge_df_targets_pred = merge_df_targets_pred.drop(actual_and_predicted, axis=1)
        df_predicted = pd.merge(df_predicted, merge_df_targets_pred, left_index=True, right_index=True)

        st.dataframe(df_predicted.head(50))

        # TODO: ダウンロード操作、result_pathをダウンロードすればいいはず

        if not is_tutorial:
            result_csv = df_predicted.to_csv(index=False)
            b64 = base64.b64encode(result_csv.encode("utf-8-sig")).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="result_utf-8-sig.csv">CSV(utf-8 BOM)</a>'
            st.markdown(
                t("prediction_result.download_result_csv").format(download_link=href),
                unsafe_allow_html=True,
            )
        return df_predicted

    def section_predict_by_genrated_code(
        self,
        sml: SapientML,
        df_predict: pd.DataFrame,
        config: dict,
        load_time: str,
    ):
        (lang, is_tutorial) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        if "predicted" not in st.session_state:
            st.session_state.predicted = PredictedState(load_time=load_time)
        state = st.session_state.predicted

        target_columns = config["target_columns"]

        if state.load_time != load_time:
            state.load_time = load_time
            state.df_predicted = None
            state.df_targets_pred = None

        if st.button(f':green[**{t("prediction.btn_predict")}**]'):
            state.df_predicted = None
            state.df_targets_pred = None

            state.df_targets_pred = self.part_predict_by_genrated_code(sml, df_predict)

            if state.df_targets_pred is not None:
                state.df_predicted = self.part_view_predicted(
                    config,
                    df_predict,
                    state.df_targets_pred,
                )

            if set(target_columns).issubset(set(df_predict.columns)):
                self.part_show_metrics(df_predict, df_targets_pred=state.df_targets_pred, config=config)

            if is_tutorial:
                st.info(t("inquiries.tutorial"))
