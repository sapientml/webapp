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
import argparse
import copy
import io
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from contextvars import ContextVar
from pathlib import Path
from typing import Literal
from uuid import UUID

import lib.style_set as style
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from sapientml import SapientML
from sapientml.util.logging import setup_logger
from sklearn import metrics

from .data_grapher import DataGrapher
from .escape_util import escape_csv_columns

# from lib.utils import UUIDContextFilter
from .generate_code_thread import GenerateCodeThread
from .i18n import I18N_DICT, use_translation
from .logging_filter import ModifyLogMessageFilter, UUIDContextFilter
from .make_result_thread import MakeResultThread
from .result_extractor import ResultExtractor
from .session_state import CodeGenerationState, MakeResultState, PredictState, QueryParamsState, TrainDataState
from .utils import check_columns, check_needing_predict_proba


class Application(ABC):
    def __init__(self, I18N_DICT=I18N_DICT):
        # Debug option
        self.I18N_DICT = I18N_DICT
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", "-d", action="store_true")
        args = parser.parse_args()
        self.debug = args.debug

    # @abstractmethod
    def reset(self, delete_session=None):
        # initialize
        if delete_session is None:
            delete_session = [
                "uuid",
                "train_data_state",
                "conf",
                "previous_conf",
                "predict",
                "predicted",
                "code_generation",
                "make_result",
            ]

        for key in st.session_state.keys():
            if key in delete_session:
                del st.session_state[key]

    def get_lang(self):
        query_params = st.experimental_get_query_params()
        selected_lang = query_params["lang"][0] if "lang" in query_params else "en"
        lang = "ja" if selected_lang == "ja" else "en"
        return lang

    def get_tutorial_mode(self):
        query_params = st.experimental_get_query_params()

        # for webapp
        if "is_tutorial" in query_params:
            is_tutorial = True if query_params["is_tutorial"][0] == "True" else False
        # for sandbox
        else:
            selected_page = query_params["page"][0] if "page" in query_params else "tutorial"
            is_tutorial = False if selected_page == "app" else True

        return is_tutorial

    def get_query_params(self):
        # URLパラメータを取得

        is_tutorial = self.get_tutorial_mode()
        lang = self.get_lang()

        if "query_params" not in st.session_state:
            st.session_state.query_params = QueryParamsState(is_tutorial, lang)
        state = st.session_state.query_params

        if state.is_tutorial != is_tutorial:
            state.is_tutorial = is_tutorial
            self.reset()
            st.rerun()

        if state.lang != lang:
            if "previous_conf" in st.session_state:
                st.session_state.previous_conf = copy.deepcopy(st.session_state.conf)
            if "previous_sample" in st.session_state:
                st.session_state.previous_sample = copy.deepcopy(st.session_state.sample)
            state.lang = lang
            st.rerun()

        return lang, is_tutorial

    # Todo: rename initialize_screen
    def configure_page(
        self,
        favicon_path="img/favicon.ico",
        page_title="SapientML",
        layout="wide",
        color="#6fde6f",
    ):
        # set favicon path
        if not os.path.isfile(favicon_path):
            favicon_path = ""

        # Enable wide mode by default
        st.set_page_config(page_title, page_icon=favicon_path, layout=layout)

        css = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

        # style
        self.color = {"background-color": color}

        with open("css/custom-subheader.css", "r") as css_file:
            css_content = css_file.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

    def section_show_logo(self, logo_path="img/logo.png"):
        (lang, _) = self.get_query_params()

        # logo
        # Displaying an image converted into a byte sequence
        image = Image.open("img/logo.png")
        st.sidebar.image(image)

        st.sidebar.write("[HomePage](https://sapientml.io/)")
        st.sidebar.write("[Documentation](https://sapientml.readthedocs.io/)")
        st.sidebar.write("[Repository](https://github.com/sapientml/sapientml)")

    def section_show_sidebar(self):
        (lang, is_tutorial) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        self.section_show_logo()

        def lang_change(lang: Literal["en", "ja"], is_tutorial: bool):
            if lang == "en":
                st.experimental_set_query_params(page="tutorial" if is_tutorial else "app", lang="ja")
            else:
                st.experimental_set_query_params(page="tutorial" if is_tutorial else "app", lang="en")

        def format_func(option):
            return "English" if option == "en" else "日本語"

        langage_list = ["en", "ja"]
        index = langage_list.index(lang)
        st.sidebar.selectbox(
            "Select Language / 言語選択",
            ("en", "ja"),
            index=index,
            key="lang",
            on_change=lang_change,
            args=(lang, is_tutorial),
            format_func=format_func,
        )

        def update_query_params(lang: Literal["en", "ja"], is_tutorial: bool):
            if is_tutorial:
                st.experimental_set_query_params(page="app", lang=lang)
                # reset()
            else:
                st.experimental_set_query_params(page="tutorial", lang=lang)
                # reset()

        if is_tutorial:
            st.sidebar.button(t("sidebar.to_trial"), on_click=update_query_params, args=(lang, True))
        else:
            st.sidebar.button(t("sidebar.to_tutorial"), on_click=update_query_params, args=(lang, False))

        st.sidebar.divider()

    def section_show_introduction(self):
        (lang, is_tutorial) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        if is_tutorial:
            st.info(t("header.welcome"))
            st.write("")
            st.info(t("welcome.description"))

    def section_upload_training_data(
        self,
        sample_data_path="./sample/train.csv",
        source="[Titanic Dataset](https://www.openml.org/d/40945)",
    ):
        (lang, is_tutorial) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        if "train_data_state" not in st.session_state:
            st.session_state.train_data_state = TrainDataState()
        state = st.session_state.train_data_state

        form_disable = state.df_train is not None

        button = f':green[**{t("upload_form.btn_submit")}**]'
        st.write(t("upload_form.description").format(button=button))

        if is_tutorial:
            st.info(t("upload_form.tutorial").format(source=source, button=button))
        with st.form("uploadfile", clear_on_submit=True):
            st.file_uploader(
                t("upload_form.select_file"),
                key="training_data",
                type="csv",
                label_visibility="visible",
                disabled=(form_disable or is_tutorial),
            )
            st.caption(t("upload_form.column_name_rule"))
            st.selectbox(
                t("upload_form.encoding"), ["UTF-8", "SJIS"], disabled=(form_disable or is_tutorial), key="csv_encoding"
            )

            def on_click(sample_data_path):
                state.ex_msg = None
                uploaded = st.session_state.training_data
                csv_encoding = st.session_state.csv_encoding

                if is_tutorial:
                    state.df_train = pd.read_csv(sample_data_path)
                elif uploaded is not None:
                    try:
                        df_train_origin = pd.read_csv(uploaded, encoding=csv_encoding)
                        df_train = state.df_train = escape_csv_columns(df_train_origin)

                        # if backslash is in column name, return error.
                        if df_train is None:
                            state.ex_msg = t("training_data.illigal_column_name_error")
                        else:
                            state.csv_name = uploaded.name
                            state.csv_encoding = csv_encoding
                    except UnicodeDecodeError as e:
                        state.ex = e
                        state.ex_msg = t("prediction.character_code_error")
                elif uploaded is None and state.df_train is None:
                    state.ex_msg = t("upload_form.no_file_error")

            st.form_submit_button(
                f':green[**{t("upload_form.btn_submit")}**]',
                disabled=(form_disable),
                on_click=on_click,
                kwargs={"sample_data_path": sample_data_path},
            )

            if state.ex_msg is not None:
                st.error(state.ex_msg)

        return (state.df_train, state.csv_encoding, state.csv_name)

    # Training Data
    def section_view_training_data(
        self, df_train: pd.DataFrame, source="[Titanic Dataset](https://www.openml.org/d/40945)"
    ):
        (lang, is_tutorial) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        st.markdown("<a name='train_data'></a>", unsafe_allow_html=True)
        st.write("")
        style.custom_h4(t("header.training_data"))
        st.sidebar.markdown(f"[{t('header.training_data')}](#train_data)")

        if is_tutorial:
            st.info(t("training_data.tutorial").format(source=source))

        st.write(t("training_data.data_size").format(columns=len(df_train.axes[1]), rows=len(df_train.axes[0])))

        st.dataframe(df_train.head(50))

    # Configuration
    @abstractmethod
    def part_advanced_settings():
        pass

    @abstractmethod
    def section_configure_params():
        pass

    def section_generate_code(self, config: dict, estimator_name: str):
        (lang, _) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)
        uuid = st.session_state.uuid
        output_dir = config["output_dir"]

        if "code_generation" not in st.session_state:
            st.session_state.code_generation = CodeGenerationState()
        state = st.session_state.code_generation

        thread_generatecode = state.thread_generatecode
        log_stream = state.log_stream
        sml = state.sml

        if state.sml is None:
            config.update({"debug": True})
            sml = state.sml = SapientML(
                config["target_columns"],
                model_type=estimator_name,
                **({k: v for k, v in config.items() if k != "target_columns"}),
            )

            if (state.log_stream is not None) and isinstance(state.log_stream, io.StringIO):
                state.log_stream.close()
            log_stream = state.log_stream = io.StringIO()
            log_handler = logging.StreamHandler(log_stream)
            # ContextVars
            ctx_uuid: ContextVar[UUID] = ContextVar("streamlit_uuid", default=None)
            log_handler.addFilter(UUIDContextFilter(uuid, ctx_uuid))

            log_handler.addFilter(ModifyLogMessageFilter(str(output_dir.resolve())))

            logger = setup_logger()

            logger.addHandler(log_handler)

            thread_generatecode = state.thread_generatecode = GenerateCodeThread(
                sml, config, log_handler, ctx_uuid, uuid
            )
            thread_generatecode.start()

        if state.log_stream is None:
            st.stop()

        # Execution
        st.markdown("<a name='execution'></a>", unsafe_allow_html=True)
        st.write("")
        style.custom_h4(t("header.execution"))
        st.sidebar.markdown(f"[{t('header.execution')}](#execution)")

        if thread_generatecode is not None:
            log_area = st.text("")
            with st.spinner(t("codegen.generating_code")):
                while thread_generatecode.is_alive():
                    if log_stream is not None:
                        log_area.text(re.sub(r"/tmp/.+/", "", log_stream.getvalue()))
                    time.sleep(1)
                if log_stream is not None:
                    log = re.sub(r"/tmp/.+/", "", log_stream.getvalue())
                    state.log_message = log
                    log_area.text(log)

            state.ex = thread_generatecode.get_exception()
            state.sml = thread_generatecode.get_sml()

            if state.ex is not None:
                st.error(t("codegen.failed"))
                import traceback

                tb_str = traceback.format_exception(type(state.ex), state.ex, state.ex.__traceback__)
                tb_str = "".join(tb_str)
                st.error(tb_str)
                st.stop()

            thread_generatecode = state.thread_generatecode = None

            return state.sml

        elif state.sml is not None:
            st.text(state.log_message)
            return state.sml

        else:
            st.stop()

    def section_make_result(
        self,
        csv_name,
        csv_encoding,
        test_csv_name=None,
        available_file_patterns=None,
        downloadable_files=None,
        in_memory_files=None,
    ):
        (lang, _) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        uuid = st.session_state.uuid
        dirpath = Path(f"./outputs/{uuid}")

        if "make_result" not in st.session_state:
            st.session_state.make_result = MakeResultState()
        state = st.session_state.make_result
        if state.files is None:
            # if True:
            thread_makeresult = state.thread_makeresult
            if thread_makeresult is None:
                thread_makeresult = MakeResultThread(
                    dirpath,
                    csv_name,
                    test_csv_name,
                    csv_encoding,
                    available_file_patterns,
                    downloadable_files,
                    in_memory_files,
                    self.debug,
                )
                thread_makeresult.start()
            with st.spinner(t("codegen.result")):
                while thread_makeresult.is_alive():
                    time.sleep(1)
            ex = thread_makeresult.get_exception()
            if ex is not None:
                st.error(t("codegen.failed"))
                st.stop()
            state.files = thread_makeresult.get_files()

        return state.files

    def section_show_generated_code_result(self, files: dict, config: dict, tutorial_disabled: bool = False):
        (lang, is_tutorial) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        # Result
        st.markdown("<a name='automl_result'></a>", unsafe_allow_html=True)
        st.write("")
        style.custom_h4(t("header.result"))
        st.sidebar.markdown(f"[{t('header.result')}](#automl_result)")
        st.write(t("experimental_result.description"))

        if tutorial_disabled:
            is_tutorial = False

        if is_tutorial:
            st.info(t("experimental_result.tutorial"))

        # preprocess
        st.write(f"**{t('experimental_result.selected_preprocess')}**")

        preprocess_list = ResultExtractor.get_preprocess(files)
        if preprocess_list:
            st.write("  →  ".join([t(f"preprocess.{key}") for key in preprocess_list]))
        else:
            st.write(t("preprocess.PREPROCESS:none"))

        # candidate
        st.write(f"**{t('experimental_result.heading_candidates')}**")
        st.write(t("experimental_result.description_candidates"))

        model_info, score_index = ResultExtractor.get_candidates(files, config["adaptation_metric"])
        # style
        color = {"background-color": "#6fde6f"}
        st.write(model_info.head(3).style.set_properties(subset=pd.IndexSlice[score_index, :], **color))

    def section_show_generated_code(self, files: dict, csv_name, test_csv_name=None):
        (lang, is_tutorial) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        script_py = files.get("script.py", None)
        script_html = files.get("script.html", None)

        if script_html is not None:
            with st.expander(f":green[**{t('experimental_result.script_ipynb')}​**]", expanded=True):
                pattern_replacements = [
                    (r"\"/.+/training\.csv", '"training.csv'),
                    (r"\"/.+/test\.csv", '"test.csv'),
                ]

                html = script_html

                for pattern, replacement in pattern_replacements:
                    html = re.sub(pattern, replacement, html)

                if csv_name is not None:
                    html = html.replace("training.csv", f"{csv_name}")

                html = re.sub(r"/tmp/.+/.+\.py", "scrip.py", html)
                st.components.v1.html(html, width=None, height=500, scrolling=True)

        if script_py is not None:
            with st.expander(f":green[**{t('experimental_result.script')}**]", expanded=True):
                pattern_replacements = [
                    (r"r\"/.+/training\.csv", '"training.csv'),
                    (r"r\"/.+/test\.csv", '"test.csv'),
                ]
                code = script_py
                for pattern, replacement in pattern_replacements:
                    code = re.sub(pattern, replacement, code)
                if csv_name is not None:
                    code = code.replace("training.csv", f"{csv_name}")
                if test_csv_name is not None:
                    code = code.replace("test.csv", f"{test_csv_name}")

                st.code(code)
        st.download_button(
            label=t("experimental_result.download"),
            data=files["result.zip"],
            file_name="result.zip",
            disabled=is_tutorial,
        )

    def section_show_model_detail(self, files: dict):
        (lang, is_tutorial) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        df_pi = (
            pd.read_csv(io.StringIO(files["permutation_importance.csv"]))
            if "permutation_importance.csv" in files.keys()
            else None
        )

        # Model Detail
        st.markdown("<a name='model_detail'></a>", unsafe_allow_html=True)
        st.write("")
        style.custom_h4(t("header.model_details"))
        st.sidebar.markdown(f"[{t('header.model_details')}](#model_detail)")

        st.write(t("model_details.description"))
        st.write(t("model_details.description_word"))

        if is_tutorial:
            st.info(t("model_details.tutorial"))
        if df_pi is not None:
            pf = df_pi.sort_values("importance", ascending=False)

            # plotly
            pf_head30 = pf.head(30)
            fig = px.bar(
                pf_head30,
                x="feature",
                y="importance",
                title=t("model_details.feature_importance"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(t("model_details.warning_no_pi_calculation"))

    def section_show_correlation(self, files: dict, config: dict):
        (lang, is_tutorial) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        df_pi = (
            pd.read_csv(io.StringIO(files["permutation_importance.csv"]))
            if "permutation_importance.csv" in files.keys()
            else None
        )

        if df_pi is not None:
            pf = df_pi.sort_values("importance", ascending=False)

        df_train = config["training_dataframe"]

        calculate_PI = True if "permutation_importance.csv" in files.keys() else False

        # Correlation between feature and target column
        st.markdown("<a name='target_features'></a>", unsafe_allow_html=True)
        st.write("")
        style.custom_h4(t("header.correlation_feature_and_target"))
        st.sidebar.markdown(f"[{t('header.correlation_feature_and_target')}](#target_features)")

        st.write(t("correlation_feature_and_target.description"))
        st.write(t("correlation_feature_and_target.description_eda"))
        if is_tutorial:
            st.info(t("correlation_feature_and_target.tutorial"))

        option_target = st.selectbox(
            f'**{t("correlation_feature_and_target.label_selectbox_for_target")}**',
            config["target_columns"],
        )
        features = []

        if calculate_PI and len(pf) == (
            len(df_train.columns) - len(config["ignore_columns"]) - len(config["target_columns"])
        ):
            features = pf["feature"].to_list()
        else:
            for item in df_train.columns:
                if item != option_target and item not in config["ignore_columns"]:
                    features.append(item)

        option = st.selectbox(
            f'**{t("correlation_feature_and_target.label_selectbox_for_feature")}**',
            features,
        )

        colums_info = ResultExtractor.get_columns_info(files)
        fig, discription = DataGrapher.get_fig(df_train, option_target, option, colums_info, config["task_type"], lang)

        if discription:
            st.write(discription)
        st.plotly_chart(fig, use_container_width=True)

    # カラムの数と名前と並び順をチェックする
    def check_columns(self, df_train, df_predict, ignore_columns, target_columns):
        (lang, _) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        df_columns = df_predict.columns.drop(ignore_columns, errors="ignore")
        df_train_columns = df_train.columns.drop(ignore_columns, errors="ignore")

        target_exists = set(target_columns).issubset(set(df_predict.columns))

        if not target_exists:
            df_train_columns = df_train_columns.drop(target_columns)
        if set(df_columns) != set(df_train_columns):
            st.error(t("prediction.data_error_in_number_of_columns"))
            exit_train_data = set(df_train_columns) - set(df_columns)
            exit_upload_data = set(df_columns) - set(df_train_columns)
            with st.expander(t("prediction.data_error_display_details")):
                if exit_train_data != set():
                    st.write(t("prediction.data_error_exit_only_train_data").format(columns=exit_train_data))
                if exit_upload_data != set():
                    st.write(t("prediction.data_error_exit_only_upload_data").format(columns=exit_upload_data))
            return False
        # state.reload = True
        elif not check_columns(df_train_columns, df_columns):
            st.error(t("prediction.data_error_column_order"))
            return False
        else:
            return True

    def section_upload_prediction_data(
        self,
        config: dict,
        sample_test="./sample/test.csv",
        df_test=None,
    ):
        (lang, is_tutorial) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        if "predict" not in st.session_state:
            if config.get("test_data", None) is not None:
                st.session_state.predict = PredictState(df_predict=config["test_data"], load_time=time.time())
            else:
                st.session_state.predict = PredictState()
        state = st.session_state.predict

        df_predict = state.df_predict

        ignore_columns = config["ignore_columns"]
        df_train = config["training_dataframe"]
        target_columns = config["target_columns"]
        csv_encoding = config["csv_encoding"]

        # Prediction

        st.markdown("<a name='prediction'></a>", unsafe_allow_html=True)
        st.write("")
        style.custom_h4(t("header.prediction"))
        st.sidebar.markdown(f"[{t('header.prediction')}](#prediction)")

        st.write(t("prediction.description"))
        if is_tutorial:
            st.info(t("prediction.tutorial"))

        with st.form("upload_prediction_file", clear_on_submit=True):
            st.file_uploader(
                t("prediction.label_file_uploader"),
                key="prediction_data",
                type="csv",
                label_visibility="visible",
                disabled=is_tutorial,
            )

            def on_click():
                state.load_time = str(time.time())
                state.ex_msg = None
                uploaded_file = st.session_state.prediction_data

                df_predict = state.df_predict = None
                if is_tutorial:
                    if df_test is None:
                        df_predict = state.df_predict = pd.read_csv(sample_test)
                    else:
                        df_predict = state.df_predict = df_test

                elif uploaded_file is not None:
                    try:
                        df_predict_origin = pd.read_csv(uploaded_file, encoding=csv_encoding)
                        df_predict = state.df_predict = escape_csv_columns(df_predict_origin)

                        if df_predict is None:
                            state.ex_msg = t("training_data.illigal_column_name_error")

                    except UnicodeDecodeError:
                        state.ex_msg = t("prediction.character_code_error")
                else:
                    state.ex_msg = t("upload_form.no_file_error")

            st.form_submit_button(
                f':green[**{t("upload_form.btn_submit")}**]',
                on_click=on_click,
            )

            if state.ex_msg is not None:
                st.error(state.ex_msg)

            if state.df_predict is not None:
                if not self.check_columns(df_train, state.df_predict, ignore_columns, target_columns):
                    df_predict = state.df_predict = None

        return df_predict, state.load_time

    def section_view_test_data(
        self,
        df_predict: pd.DataFrame,
    ):
        (lang, is_tutorial) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        if is_tutorial:
            st.info(t("prediction.tutorial_pred").format())

        st.write(t("prediction.data_size").format(columns=len(df_predict.axes[1]), rows=len(df_predict.axes[0])))
        st.dataframe(df_predict.head(50))

    def part_predict_by_genrated_code(
        self,
        sml: SapientML,
        df_predict: pd.DataFrame,
    ):
        (lang, _) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)
        state = st.session_state.predicted

        if state is None:
            st.error("invalid state")
            st.stop()

        # 予測の実行
        with st.spinner(t("prediction.predicting")):
            df_targets_pred = state.df_targets_pred = sml.predict(df_predict)

        return df_targets_pred

    @abstractmethod
    def part_view_predicted():
        pass

    def part_show_metrics(
        self,
        df_predict: pd.DataFrame,
        config: dict,
        df_targets_pred,
    ):
        (lang, _) = self.get_query_params()
        t = use_translation(lang, self.I18N_DICT)

        target_columns = config["target_columns"]
        task_type = config["task_type"]
        adaptation_metric = config["adaptation_metric"]

        # Displays metrics if the predict data contains correct answers.
        st.markdown("<a name='metrics'></a>", unsafe_allow_html=True)
        st.write("")
        style.custom_h4(t("header.metrics"))
        st.sidebar.markdown(f"[{t('header.metrics')}](#metrics)")

        if task_type == "regression":
            r2 = metrics.r2_score(df_predict[target_columns], df_targets_pred[target_columns])
            st.write("RESULT: R2 Score:", str(round(r2, 2)))

            __rmse = metrics.mean_squared_error(
                df_predict[target_columns], df_targets_pred[target_columns], squared=False
            )
            st.write("RESULT: RMSE:", str(round(__rmse, 2)))

            __target_test = np.clip(df_predict[target_columns], 0, None)
            __y_pred = np.clip(df_targets_pred[target_columns], 0, None)
            __rmsle = np.sqrt(metrics.mean_squared_log_error(__target_test, __y_pred))
            st.write("RESULT: RMSLE:", str(round(__rmsle, 2)))

            __mae = metrics.mean_absolute_error(df_predict[target_columns], df_targets_pred[target_columns])
            st.write("RESULT: MAE:", str(round(__mae, 2)))

        else:
            if check_needing_predict_proba(adaptation_metric):
                __auc = metrics.roc_auc_score(df_predict[target_columns], df_targets_pred[target_columns])
                st.write("RESULT: AUC Score: " + str(round(__auc, 2)))

            else:
                __f1 = metrics.f1_score(
                    df_predict[target_columns],
                    df_targets_pred[target_columns],
                    average="macro",
                )
                st.write("RESULT: F1 Score: " + str(round(__f1, 2)))

                if len(target_columns) == 1:
                    __accuracy = metrics.accuracy_score(df_predict[target_columns], df_targets_pred[target_columns])
                    st.write("RESULT: Accuracy: " + str(round(__accuracy, 2)))

                else:
                    # TODO Add classfication Method(multiclass-multioutput)
                    pass

    @abstractmethod
    def section_predict_by_genrated_code():
        pass
