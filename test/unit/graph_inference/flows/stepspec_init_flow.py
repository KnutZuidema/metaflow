from metaflow import StepSpec, Parameter
from metaflow.parameters import InitParameter


class StepSpecInitFlow(StepSpec):
    factor = InitParameter("factor", type=int, default=2)
    value = Parameter("value", type=int, default=10)

    def init(self):
        self.computed = self.factor * 10

    def call(self):
        self.result = self.value * self.computed


if __name__ == "__main__":
    StepSpecInitFlow()
