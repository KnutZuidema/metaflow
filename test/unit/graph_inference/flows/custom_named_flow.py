from metaflow import FlowSpec, step


class CustomNamedFlow(FlowSpec):
    """Flow with non-standard step names to test structural inference."""

    @step
    def begin(self):
        self.x = 1
        self.next(self.middle)

    @step
    def middle(self):
        self.x += 1
        self.next(self.finish)

    @step
    def finish(self):
        self.x += 1


if __name__ == "__main__":
    CustomNamedFlow()
