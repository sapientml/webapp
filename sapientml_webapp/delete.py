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
# limitations under the License
import os
import shutil
import time

dirpath = "/app/outputs"

if not os.path.isdir(dirpath):
    os.mkdir(dirpath)

while True:
    # 24時間以上経過したフォルダを削除
    for dirname in os.listdir("/app/outputs"):
        if os.path.isdir(f"/app/outputs/{dirname}"):
            created_time = os.path.getctime(f"/app/outputs/{dirname}")
            if time.time() - created_time > 60 * 60 * 24:
                shutil.rmtree(f"/app/outputs/{dirname}")

    time.sleep(60 * 60)
