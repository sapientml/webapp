import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

dirpath = "./outputs"

if not os.path.isdir(dirpath):
    os.mkdir(dirpath)


def run_nbconvert(dirname):
    if os.path.isdir(f"{dirpath}/{dirname}"):
        final_script_path = f"{dirpath}/{dirname}/final_script.ipynb"
        script_path = f"{dirpath}/{dirname}/final_script.ipynb.out.ipynb"
        state_path = f"{dirpath}/{dirname}/state"

        # dirnameにfinal_script.ipynbが存在し、script.ipynbが存在しないなら以下の処理を実行する
        if os.path.isfile(final_script_path) and not os.path.isfile(script_path) and not os.path.isfile(state_path):

            with open(state_path, mode="w") as f:
                f.write("doing")

            env = os.environ.copy()

            result = None
            try:
                result = subprocess.run(
                    ["python", "execute_notebook.py", "--workdir", f"{dirpath}/{dirname}"],
                    env=env,
                    timeout=120,
                    stderr=subprocess.PIPE,
                )
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)

            finally:
                if result is not None:
                    if result.returncode == 0:
                        msg = "done"
                    else:
                        msg = f"error: {result.stderr.decode()}"
                    with open(state_path, mode="w") as f:
                        f.write(msg)


while True:
    with ThreadPoolExecutor() as executor:
        executor.map(run_nbconvert, os.listdir(dirpath))

    time.sleep(1)
