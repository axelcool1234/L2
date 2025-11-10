# src/l2/extractor.py

from xdsl.dialects import scf


class TransitionExtractor:
    """
    Extracts symbolic transition relation from LoopLang MLIR.
    """

    def __init__(self, module):
        self.module = module
        self.vars = {}  # Maps variable names to Z3 variables
        raise Exception("TransitionExtractor is unimplemented!")

    def extract_from_while(self, while_op: scf.WhileOp):
        """
        Extracts variable names, I(x) T(x, x'), and P(x) from a
        while loop.

        Note: Property P is NOT generated here - it comes from
        %assert statements in the program.

        TODO: Explore Avy for potential generation of property P
        as well. https://arieg.bitbucket.io/pdf/avy.pdf
        """
        raise Exception("TransitionExtractor is unimplemented!")
