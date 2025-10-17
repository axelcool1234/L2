# src/l2/__main__.py

"""
Loop Lang CLI entrypoint.
"""

from xdsl.printer import Printer
from l2 import grammar, L2Transformer
from lark import Lark


def main() -> None:
    parser = Lark(grammar, start="program", parser="lalr")
    # code = """
    # x = 10
    # while @T {
    #   y = x + 2
    # }
    # """
    code = """
    x = @T || @F
    y = x || @T || @F 
    """
    tree = parser.parse(code)
    transformer = L2Transformer()
    printer = Printer()
    transformer.transform(tree)
    printer.print_op(transformer.module)


if __name__ == "__main__":
    main()
