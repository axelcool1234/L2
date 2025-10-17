# src/l2/__main__.py

"""
Loop Lang CLI entrypoint.
"""

from lark import Lark
from xdsl.printer import Printer

from l2 import L2Transformer, grammar


def main() -> None:
    parser = Lark(grammar, start="program", parser="lalr")
    # code = """
    # x = 10
    # while @T {
    #   y = x + 2
    # }
    # """
    code = """
    x = @F || @T
    y = x || @F || @T
    z = 1 + 3
    """
    tree = parser.parse(code)
    transformer = L2Transformer()
    printer = Printer()
    transformer.transform(tree)
    try:
        transformer.module.verify()
    except Exception:
        print("module verification error")
        raise
    printer.print_op(transformer.module)


if __name__ == "__main__":
    main()
