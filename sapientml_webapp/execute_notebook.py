import argparse
import os

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def main(args):

    final_script_path = "./final_script.ipynb"
    script_path = "./final_script.ipynb.out.ipynb"
    os.chdir(args.workdir)

    with open(final_script_path, "r") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
    try:
        ep.preprocess(nb)
    except TimeoutError:
        print("The execution exceeded the timeout.")

    with open(script_path, "wt") as f:
        nbformat.write(nb, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, required=True, help="File name to process")
    args = parser.parse_args()
    main(args)
