from pathlib import Path

import typer

cli = typer.Typer()


@cli.command()
def main():
    """Convenience command for launching distributed training processes.

    ```python
    torchrun $(nffd) lr test/baselineconfig.py
    ```
    """
    package_root = Path(__file__).parent
    nff_script = package_root / "train.py"
    print(nff_script)


if __name__ == "__main__":
    main()
