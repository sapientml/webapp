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
import logging
import re
from contextvars import ContextVar
from uuid import UUID


class UUIDContextFilter(logging.Filter):
    def __init__(self, uuid: UUID, ctx_uuid: ContextVar[UUID]):
        self.uuid = uuid
        self.ctx_uuid = ctx_uuid

    def filter(self, record):
        record.streamlit_uuid = streamlit_uuid = self.ctx_uuid.get()
        return streamlit_uuid == self.uuid


class ModifyLogMessageFilter(logging.Filter):
    def __init__(self, full_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_path = full_path

    def filter(self, record):
        patterns = [
            (r"saved:.+/final_script.ipynb", "Saved notebook."),
            (r"Saved explained notebook in:.+/final_script.ipynb.out.ipynb", "Saved explained notebook."),
            (re.escape(self.full_path), ""),
        ]

        for pattern, replacement in patterns:
            match = re.search(pattern, record.msg)
            if match:
                record.msg = record.msg.replace(match.group(0), replacement)

        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        record.msg = ansi_escape.sub("", record.msg)

        return True
