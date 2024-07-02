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
import subprocess
import sys


def main() -> None:

    cmd1 = [sys.executable, "/app/execute.py"]
    subprocess.Popen(
        cmd1,
        shell=False,
        executable=None,
        cwd=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    cmd2 = [sys.executable, "/app/delete.py"]
    subprocess.Popen(
        cmd2,
        shell=False,
        executable=None,
        cwd=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        subprocess.run(
            [
                "/venv/bin/streamlit",
                "run",
                "/app/main.py",
                "--server.address=0.0.0.0",
                "--server.port=8501",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
