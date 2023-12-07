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
from uuid import uuid4

import streamlit as st

from lib.app_tabular import TabularApplication
from lib.i18n import get_dict_tabular, use_translation

if "uuid" not in st.session_state:
    # initialize session state
    st.session_state.uuid = uuid4()


def run_tabular(sb: TabularApplication):
    sb.section_show_introduction()

    (
        sample,
        sample_training_data_path,
        sample_test_data_path,
        source,
        default_config,
    ) = sb.section_chose_sample_datasete()

    sb.I18N_DICT = get_dict_tabular(sample)

    (df_train, df_test, csv_encoding, csv_name) = sb.section_upload_training_data(
        sample_training_data_path, sample_test_data_path, source=source
    )

    if df_train is None:
        st.stop()

    # Training Data
    sb.section_view_training_data(df_train, source=source)

    # Configuration
    (config, estimator_name) = sb.section_configure_params(df_train, csv_encoding, default_config)

    if estimator_name is None:
        st.stop()

    sml = sb.section_generate_code(config, estimator_name=estimator_name)

    if sml is None:
        st.stop()

    files = sb.section_make_result(csv_name, csv_encoding)
    if files is None:
        st.stop()

    sb.section_show_generated_code_result(files, config)

    sb.section_show_generated_code(files, csv_name)

    sb.section_show_model_detail(files)

    sb.section_show_correlation(files, config)

    df_predict, load_time = sb.section_upload_prediction_data(config, df_test=df_test)

    if df_predict is None:
        st.stop()

    sb.section_view_test_data(df_predict)
    sb.section_predict_by_genrated_code(sml, df_predict, config, load_time)


if __name__ == "__main__":
    sb = TabularApplication(get_dict_tabular())

    sb.configure_page()

    sb.section_show_sidebar()

    try:
        run_tabular(sb)

    except Exception:
        (lang, _) = sb.get_query_params()
        t = use_translation(lang, get_dict_tabular())
        st.error(t("error.general"))
        st.stop()
