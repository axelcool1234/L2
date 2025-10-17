# src/l2/__main__.py

"""
Loop Lang CLI entrypoint.
"""

from lark import Lark
from xdsl.printer import Printer

from l2 import L2Interpreter, grammar


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
    r = 0
    """
    tree = parser.parse(code)
    interpeter = L2Interpreter()
    printer = Printer()
    interpeter.visit(tree)
    try:
        interpeter.module.verify()
    except Exception:
        print("module verification error")
        raise
    printer.print_op(interpeter.module)


if __name__ == "__main__":
    main()
