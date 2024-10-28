from benchmarking.benchmark import ID_Tuple
from benchmarking.dtlBenchmark import DLTTest

class BasicDTLTest(DLTTest):

    @classmethod
    def get_id_headings(cls) -> list[tuple[str, type[str] | type[int] | type[bool]]]:
        return [("layout", int), ("order", int)]

    @classmethod
    def get_result_headings(
        cls,
    ) -> list[tuple[str, type[int] | type[bool] | type[float]]]:
        return [("correct", bool), ("total_error", float), ("consistent", bool)]

    def get_id(self) -> ID_Tuple:
        return self.layout.number, self.order.number
