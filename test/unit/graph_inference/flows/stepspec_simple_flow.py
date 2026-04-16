from metaflow import StepSpec, Parameter


class StepSpecSimpleFlow(StepSpec):
    text = Parameter("text", type=str, default="hello")

    def call(self):
        self.output = self.text.upper()


if __name__ == "__main__":
    StepSpecSimpleFlow()
