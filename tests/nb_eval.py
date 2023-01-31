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


@click.command()
@click.option("--nb", type=str, default=".")
def main(nb: Path) -> None:
    start = time()
    try:
        run_notebook(nb)
    except BaseException as e:
        print("FAIL", nb, e)

        raise e
    finally:
        print(f"Tutorial {nb} tool {time() - start}")


if __name__ == "__main__":
    main()
