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
import re
import string

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer

from .i18n import I18N_DICT, use_translation


class DataGrapher:
    def excute_countvectorizer(column_name, df):
        _TEXT_COLUMN = column_name
        __temp_train_data = df

        __simple_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        __temp_train_data = __simple_imputer.fit_transform(__temp_train_data.values.reshape(-1, 1))[:, 0]

        def process_text(__dataserise):
            process_text = [t.lower() for t in __dataserise.tolist()]

            # strip all punctuation
            table = str.maketrans("", "", string.punctuation)
            process_text = [t.translate(table) for t in process_text]
            # convert all numbers in text to 'num'
            process_text = [re.sub(r"\d+", "num", t) for t in process_text]
            __dataserise = process_text
            return __dataserise

        __temp_train_data = pd.Series(process_text(__temp_train_data))

        __countvectorizer = CountVectorizer(max_features=3000)
        vector_train = __countvectorizer.fit_transform(__temp_train_data)
        __feature_names = ["_".join([_TEXT_COLUMN, name]) for name in __countvectorizer.get_feature_names_out()]
        vector_train = pd.DataFrame.sparse.from_spmatrix(
            vector_train, columns=__feature_names, index=__temp_train_data.index
        )

        return vector_train.sparse.to_dense()

    def get_fig(df_train, option_target, option, colums_info, prediction_type, lang):
        fig = None
        description = None
        t = use_translation(lang, I18N_DICT)

        n_category = df_train[option].nunique()
        top_n = 20

        if option in colums_info.get("text_columns"):
            # "option" is a text column. One cell can contain multiple words.

            result_count = DataGrapher.excute_countvectorizer(option, df_train[option])
            option_target_col = df_train[option_target]
            sort_columns = result_count.sum().sort_values(ascending=False).index.to_list()
            top_columns = sort_columns[0:top_n]
            result_count = (result_count > 0).astype(int)

            if prediction_type == "regression":
                fig = go.Figure(layout_title_text=t("correlation_feature_and_target.title_graph_box"))
                y_others = []
                for _col in sort_columns:
                    y0 = option_target_col.loc[result_count[_col] > 0]
                    if _col in top_columns:
                        fig.add_trace(go.Box(y=y0, name=_col))
                    else:
                        y_others.append(y0)
                if y_others:
                    fig.add_trace(go.Box(y=pd.concat(y_others), name="others"))
                    description = t("correlation_feature_and_target.description_graph_text")
                fig.update_yaxes(title_text=option_target)
                fig.update_xaxes(title=option)

            else:
                result_count[option_target] = option_target_col
                result_count = result_count.groupby(option_target).sum().transpose()
                result_count_others = result_count.drop(top_columns).sum()
                result_count = result_count.loc[top_columns]
                result_count.loc["others"] = result_count_others.values.tolist()
                # Fails when there is originally "index" columns.
                result_count = result_count.reset_index().melt(id_vars=["index"])
                fig = px.bar(
                    result_count,
                    x="index",
                    y="value",
                    color=option_target,
                    title=t("correlation_feature_and_target.title_graph_bar"),
                )
                fig.update_xaxes(title=option)
                description = t("correlation_feature_and_target.description_graph_text")

        elif n_category <= top_n or option in colums_info.get("categorical_columns"):
            # "option" is regraded as categorical. One cell must contain only one category.
            if prediction_type == "regression":
                label_freq = df_train[option].value_counts().sort_values(ascending=False).index.to_list()
                count_df = df_train.copy()
                if n_category > top_n:
                    label_freq = label_freq[0:top_n]
                    count_df.loc[~count_df[option].isin(label_freq), option] = "others"
                    label_freq.append("others")
                fig = go.Figure()
                for _col in label_freq:
                    fig.add_trace(
                        go.Box(
                            y=count_df.loc[count_df[option] == _col, option_target],
                            name=_col,
                        )
                    )

                fig.update_layout(
                    title_text=t("correlation_feature_and_target.title_graph_box"),
                    xaxis_title_text=option,
                    yaxis_title_text=option_target,
                )
            else:
                # Fails when there is originally "index" columns.
                count_df = (
                    df_train.reset_index()
                    .pivot_table(
                        values="index",
                        index=option_target,
                        columns=option,
                        aggfunc="count",
                    )
                    .transpose()
                )
                count_df["tmp_sum"] = count_df.sum(axis=1)
                count_df = count_df.sort_values(by="tmp_sum", ascending=False)
                count_df = count_df.drop("tmp_sum", axis=1)
                if n_category > top_n:
                    count_df_top = count_df[:top_n].copy()
                    count_df_top.loc["others"] = count_df[top_n:].sum(axis=0)
                    count_df = count_df_top

                count_df = count_df.reset_index(names=option)
                fig = px.bar(
                    count_df,
                    x=option,
                    y=count_df.columns,
                    title=t("correlation_feature_and_target.title_graph_bar"),
                )

        else:
            # "option" is a continuous column
            tmp_df = df_train[[option, option_target]].copy()
            if option in colums_info.get("date_columns"):
                tmp_df[option] = pd.to_datetime(tmp_df[option], errors="coerce")

            if prediction_type == "regression":
                fig = px.scatter(
                    tmp_df,
                    x=option,
                    y=option_target,
                    title=t("correlation_feature_and_target.title_graph_scatter"),
                    render_mode="svg",
                )
            else:
                fig = px.box(
                    tmp_df,
                    x=option,
                    y=option_target,
                    title=t("correlation_feature_and_target.title_graph_box"),
                    orientation="h",
                )

        return fig, description
