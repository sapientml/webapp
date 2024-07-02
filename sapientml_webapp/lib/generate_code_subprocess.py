import json
import re
import subprocess
import threading
import time
from pathlib import Path

LOG_LEVEL = {"DEBUG", "INFO", "WARNING", "ERROR"}


class SubprocessExecutor:
    def __init__(self, debug):
        self.proc = None
        self.log_text = ""
        self.debug = debug
        self.thread = None
        self.full_path = None

    def run_subprocess(self, train_data_path, estimator_name, config):
        self.proc = subprocess.Popen(
            [
                "python",
                "lib/call_automl.py",
                "--train_data_path",
                train_data_path,
                "--estimator_name",
                estimator_name,
                "--config",
                json.dumps(config),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def get_subprocess_output(self):
        while True:
            output = self.proc.stdout.readline()
            if output == "" and self.proc.poll() is not None:
                break
            if self.debug:
                if output:
                    self.log_text += "\n" + output.strip()
            else:
                if any(level in output for level in LOG_LEVEL):
                    self.log_text += "\n" + self.filter(output.strip())

            time.sleep(1)

    def start_automl(self, train_data_path, estimator_name, config):

        output_dir = Path(config["output_dir"])
        self.full_path = str(output_dir.resolve())

        threading.Thread(
            target=self.run_subprocess,
            args=(str(train_data_path), estimator_name, {k: v for k, v in config.items() if k != "training_dataframe"}),
        ).start()
        time.sleep(1)
        self.thread = threading.Thread(target=self.get_subprocess_output)
        self.thread.start()

    def filter(self, record):

        patterns = [
            (r"saved:.+/final_script.ipynb", "Saved notebook."),
            (r"Saved explained notebook in:.+/final_script.ipynb.out.ipynb", "Saved explained notebook."),
            (re.escape(self.full_path), ""),
            (r": request url:.*", ""),
            (r": Traceback.*", ""),
        ]

        for pattern, replacement in patterns:
            match = re.search(pattern, record)
            if match:
                record = record.replace(match.group(0), replacement)

        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        record = ansi_escape.sub("", record)

        return record
