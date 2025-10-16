# src/l2/__main__.py

"""
LoopLang CLI entrypoint.
"""

from l2 import grammar
from lark import Lark


def main() -> None:
    parser = Lark(grammar, start="program", parser="lalr")
    code = """
    x = 10
    while %T {
      y = x + 2
    }
    """
    tree = parser.parse(code)
    print(tree.pretty())


if __name__ == "__main__":
    main()
