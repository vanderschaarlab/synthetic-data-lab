# stdlib
import sys
from pathlib import Path
from time import time

# third party
import click
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# tutorials_path = str(Path("../Tutorials").resolve())
# sys.path.append(tutorials_path)

Tutorials_dir = Path(__file__).parents[1] / "Tutorials/"
Tutorials_dir.mkdir(parents=True, exist_ok=True)


def run_notebook(notebook_path: Path) -> None:
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    proc = ExecutePreprocessor(timeout=1800)
    # Will raise on cell error
    proc.preprocess(nb, {"metadata": {"path": Tutorials_dir}})


enabled_tests = [
    "Case Study 1 - Data Modality",
]


@click.command()
@click.option("--nb_dir", type=str, default=".")
def main(nb_dir: Path) -> None:
    start = time()

    nb_dir = Path(nb_dir)

    for nb in nb_dir.rglob("*"):
        if nb.suffix != ".ipynb":
            continue

        if "checkpoint" in nb.name:
            continue

        enabled = False
        for enabled_test in enabled_tests:
            if enabled_test in nb.name:
                enabled = True

        if not enabled:
            continue

        print("Testing", nb.name)
        try:
            run_notebook(nb)
        except BaseException as e:
            print("FAIL", nb, e)

            raise e
        finally:
            print(f"Tutorial {nb} tool {time() - start}")


if __name__ == "__main__":
    main()
