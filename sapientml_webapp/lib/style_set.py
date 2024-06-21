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
import streamlit as st


def custom_h4(title):
    """カスタムサブヘッダー（左側に縦線あり）を表示する関数"""
    st.markdown(
        f"""
        <h4 class="custom-subheader-left">{title}</h4>
        """,
        unsafe_allow_html=True,
    )


def custom_h3(title):
    """
    カスタムサブヘッダーを表示する関数
    """
    st.markdown(
        f"""
        <h3 class="custom-subheader">{title}</h3>
        """,
        unsafe_allow_html=True,
    )
