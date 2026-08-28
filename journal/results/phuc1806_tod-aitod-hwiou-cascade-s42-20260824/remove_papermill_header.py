# pylint: disable=missing-module-docstring
import re
from nbconvert.preprocessors import Preprocessor


class RemovePapermillHeader(Preprocessor):
  # pylint: disable=missing-class-docstring

  def preprocess(self, notebook, resources):
    pattern = re.compile(r'>An Exception was encountered at.*In \[|'
                         +'Execution using papermill encountered an exception')
    real_cells = []
    for cell in notebook.cells:
      if cell.cell_type == 'markdown' and pattern.search(cell.source):
        continue
      real_cells.append(cell)
    notebook.cells = real_cells
    return notebook, resources

