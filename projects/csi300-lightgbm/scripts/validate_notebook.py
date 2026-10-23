"""Validate notebook format, Python syntax, and clean publication state without execution."""

import ast
from pathlib import Path

import nbformat


def main():
    project = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(project / 'notebooks/analysis_clean.ipynb', as_version=4)
    nbformat.validate(notebook)
    count = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            ast.parse(cell.source, filename=f'cell-{index}')
            if cell.outputs or cell.execution_count is not None:
                raise ValueError(f'Cell {index}: clear execution outputs/count before publication.')
            count += 1
    print(f'Notebook format, Python syntax, and clean outputs verified for {count} code cells.')


if __name__ == '__main__':
    main()
