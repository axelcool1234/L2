# src/l2/__main__.py

"""
Loop Lang CLI entrypoint.
"""

from lark import Lark
from xdsl.printer import Printer

from l2 import L2Interpreter, grammar


def main() -> None:
    parser = Lark(grammar, start="program", parser="lalr")
    code = """
    x = 10
    y = 0
    while @T {
      y = x + 2
      x = y
    }
    """
    # code = """
    # x = @F || @T
    # y = x || @F || @T
    # z = 1 + 3
    # r = 0
    # a = r
    # a = a + 1
    # while y {
    #     r = r + 1
    #     a = r + 2
    # }
    # """
    tree = parser.parse(code)
    interpreter = L2Interpreter()
    printer = Printer()
    interpreter.visit(tree)
    try:
        interpreter.module.verify()
    except Exception:
        print("module verification error")
        raise
    printer.print_op(interpreter.module)


if __name__ == "__main__":
    main()
