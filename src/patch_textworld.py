"""
Fix a TextWorld/Python 3.13 incompatibility.

TextWorld's PDDL text generator uses `locals().update(ctx)` before `eval(expr)` to
expose grammar variables. That pattern stopped working in CPython 3.13 — locals()
is now a snapshot — so every ALFWorld reset crashes with `NameError: name 'r' is not defined`.

Import this module once before touching `textworld.*` to rewrite `EvalSymbol.derive`
to pass the grammar variables into `eval` as the locals mapping argument.
"""
import textworld.envs.pddl.textgen as _tg


def _eval_derive(self, context=None):
    context = context or self.context
    variables = dict(context.get("variables", {}))
    value = eval(self.expression, {"__builtins__": {}}, variables)
    return [_tg.TerminalSymbol(value)]


_tg.EvalSymbol.derive = _eval_derive
