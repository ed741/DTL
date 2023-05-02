import dtl


class Invert(dtl.Expr):
    fields = dtl.Expr.fields | {"expr", "skip_zero"}
    
    def __init__(self, expr: dtl.ExprTypeHint, skip_zero:bool =False, **kwargs):
        expr = dtl.Expr.exprInputConversion(expr)
        exprType = expr.type
        typeCheck = exprType.result.matchShape((("t", "t")))
        if typeCheck is None:
            raise ValueError(
                "Marginalisation requires a eta to have 2 exposed dimension (connectivity * vector) and lambda to have 4: (connectivity * connectivity * vector * vector) ")
        super().__init__(expr=expr, skip_zero=skip_zero, **kwargs)
    
    @property
    def type(self) -> dtl.DTLType:
        return self.expr.type
    
    @property
    def operands(self):
        return {"expr": self.expr}
    
    def with_operands(self, operands):
        return self.copy(expr=operands["expr"])
    
    def __str__(self) -> str:
        return f"Invert{'!'if self.skip_zero else ''}({str(self.expr)})"
    
    def shortStr(self) -> str:
        return f"Inv{'!'if self.skip_zero else ''}({self.expr.terminalShortStr()})"
    