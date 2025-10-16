grammar = r"""
?program: stmt+

?stmt: assign
     | loop

assign: VAR "=" expr

loop: "while" expr "{" stmt* "}"

?expr: expr "+" expr   -> add
     | expr "&&" expr  -> and_expr
     | expr "||" expr  -> or_expr
     | VAR
     | INT
     | BOOL

VAR: /[a-zA-Z_][a-zA-Z0-9_]*/
BOOL: "%T" | "%F"

%import common.INT
%import common.WS
%ignore WS
"""
