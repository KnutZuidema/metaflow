"""
Tests for structural inference of start/end steps in FlowGraph.

Verifies that:
- Start step is determined by zero in-edges, end step by zero out-edges
- Steps can have any name (not just "start"/"end")
- Single-step flows are supported (start == end)
- Node types are assigned correctly based on structure
- _graph_info contains start_step/end_step fields
- Lint validation works with structural inference
"""

import pytest
from metaflow import FlowSpec, step, Parameter
from metaflow.lint import linter, LintWarn


# ---------------------------------------------------------------------------
# Flow definitions for testing (must be importable from a real file)
# ---------------------------------------------------------------------------


class StandardFlow(FlowSpec):
    @step
    def start(self):
        self.x = 1
        self.next(self.end)

    @step
    def end(self):
        pass


class CustomNamedLinearFlow(FlowSpec):
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
        pass


class SingleStepFlow(FlowSpec):
    @step
    def only(self):
        self.x = 42


class CustomNamedBranchFlow(FlowSpec):
    @step
    def entry(self):
        self.next(self.a, self.b)

    @step
    def a(self):
        self.next(self.merge)

    @step
    def b(self):
        self.next(self.merge)

    @step
    def merge(self, inputs):
        self.next(self.done)

    @step
    def done(self):
        pass


class CustomNamedForeachFlow(FlowSpec):
    @step
    def init(self):
        self.items = [1, 2, 3]
        self.next(self.process, foreach="items")

    @step
    def process(self):
        self.result = self.input * 2
        self.next(self.collect)

    @step
    def collect(self, inputs):
        self.results = [i.result for i in inputs]
        self.next(self.final)

    @step
    def final(self):
        pass


class SplitStartFlow(FlowSpec):
    """Start step that is also a split."""

    @step
    def origin(self):
        self.next(self.left, self.right)

    @step
    def left(self):
        self.next(self.rejoin)

    @step
    def right(self):
        self.next(self.rejoin)

    @step
    def rejoin(self, inputs):
        self.next(self.terminus)

    @step
    def terminus(self):
        pass


# ---------------------------------------------------------------------------
# Tests: Structural inference
# ---------------------------------------------------------------------------


class TestStartEndInference:
    """Tests for FlowGraph._identify_start_end()."""

    def test_standard_names(self):
        graph = StandardFlow._graph
        assert graph.start_step == "start"
        assert graph.end_step == "end"

    def test_custom_names_linear(self):
        graph = CustomNamedLinearFlow._graph
        assert graph.start_step == "begin"
        assert graph.end_step == "finish"

    def test_single_step(self):
        graph = SingleStepFlow._graph
        assert graph.start_step == "only"
        assert graph.end_step == "only"
        assert graph.start_step == graph.end_step

    def test_custom_names_branch(self):
        graph = CustomNamedBranchFlow._graph
        assert graph.start_step == "entry"
        assert graph.end_step == "done"

    def test_custom_names_foreach(self):
        graph = CustomNamedForeachFlow._graph
        assert graph.start_step == "init"
        assert graph.end_step == "final"

    def test_split_start(self):
        graph = SplitStartFlow._graph
        assert graph.start_step == "origin"
        assert graph.end_step == "terminus"


# ---------------------------------------------------------------------------
# Tests: Node types
# ---------------------------------------------------------------------------


class TestNodeTypes:
    """Tests that node types are assigned based on structure, not names."""

    def test_standard_flow_types(self):
        graph = StandardFlow._graph
        assert graph["start"].type == "start"
        assert graph["end"].type == "end"

    def test_custom_linear_types(self):
        graph = CustomNamedLinearFlow._graph
        assert graph["begin"].type == "start"
        assert graph["middle"].type == "linear"
        assert graph["finish"].type == "end"

    def test_single_step_type_is_end(self):
        """Single-step flow: type is 'end' since it's terminal."""
        graph = SingleStepFlow._graph
        assert graph["only"].type == "end"

    def test_branch_entry_is_split(self):
        """Entry step that splits should keep 'split' type, not be overridden to 'start'."""
        graph = CustomNamedBranchFlow._graph
        assert graph["entry"].type == "split"
        assert graph["merge"].type == "join"
        assert graph["done"].type == "end"

    def test_split_start_keeps_split_type(self):
        """Start step that is also a split must keep 'split' type for lint balance."""
        graph = SplitStartFlow._graph
        assert graph["origin"].type == "split"

    def test_foreach_entry_keeps_foreach_type(self):
        graph = CustomNamedForeachFlow._graph
        assert graph["init"].type == "foreach"
        assert graph["collect"].type == "join"
        assert graph["final"].type == "end"

    def test_custom_flow_in_funcs_out_funcs(self):
        graph = CustomNamedLinearFlow._graph
        assert graph["begin"].in_funcs == []
        assert graph["begin"].out_funcs == ["middle"]
        assert graph["finish"].in_funcs == ["middle"]
        assert graph["finish"].out_funcs == []


# ---------------------------------------------------------------------------
# Tests: output_steps / graph_structure
# ---------------------------------------------------------------------------


class TestOutputSteps:
    """Tests for FlowGraph.output_steps() with structural inference."""

    def test_standard_graph_structure(self):
        steps_info, graph_structure = StandardFlow._graph.output_steps()
        assert graph_structure == ["start", "end"]
        assert set(steps_info.keys()) == {"start", "end"}

    def test_custom_linear_graph_structure(self):
        steps_info, graph_structure = CustomNamedLinearFlow._graph.output_steps()
        assert graph_structure == ["begin", "middle", "finish"]
        assert set(steps_info.keys()) == {"begin", "middle", "finish"}

    def test_single_step_graph_structure(self):
        steps_info, graph_structure = SingleStepFlow._graph.output_steps()
        assert graph_structure == ["only"]
        assert set(steps_info.keys()) == {"only"}

    def test_branch_graph_structure(self):
        steps_info, graph_structure = CustomNamedBranchFlow._graph.output_steps()
        assert graph_structure[0] == "entry"
        assert graph_structure[-1] == "done"
        assert "merge" in graph_structure
        assert set(steps_info.keys()) == {"entry", "a", "b", "merge", "done"}

    def test_steps_info_types_match(self):
        """Steps info type should match the node_to_type mapping."""
        steps_info, _ = CustomNamedLinearFlow._graph.output_steps()
        assert steps_info["begin"]["type"] == "start"
        assert steps_info["middle"]["type"] == "linear"
        assert steps_info["finish"]["type"] == "end"

    def test_split_start_type_in_steps_info(self):
        """When start step is a split, steps_info should show split-static."""
        steps_info, _ = SplitStartFlow._graph.output_steps()
        assert steps_info["origin"]["type"] == "split-static"
        assert steps_info["terminus"]["type"] == "end"

    def test_steps_info_has_next(self):
        steps_info, _ = CustomNamedLinearFlow._graph.output_steps()
        assert steps_info["begin"]["next"] == ["middle"]
        assert steps_info["middle"]["next"] == ["finish"]
        assert steps_info["finish"]["next"] == []


# ---------------------------------------------------------------------------
# Tests: Topological sort
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def test_standard_sort(self):
        graph = StandardFlow._graph
        assert graph.sorted_nodes == ["start", "end"]

    def test_custom_linear_sort(self):
        graph = CustomNamedLinearFlow._graph
        assert graph.sorted_nodes == ["begin", "middle", "finish"]

    def test_single_step_sort(self):
        graph = SingleStepFlow._graph
        assert graph.sorted_nodes == ["only"]

    def test_branch_sort_order(self):
        """Start must come first, end must come last."""
        graph = CustomNamedBranchFlow._graph
        assert graph.sorted_nodes[0] == "entry"
        assert graph.sorted_nodes[-1] == "done"


# ---------------------------------------------------------------------------
# Tests: Lint validation
# ---------------------------------------------------------------------------


class TestLintWithStructuralInference:
    """Tests that lint checks work with structural start/end inference."""

    def test_standard_flow_passes_lint(self):
        linter.run_checks(StandardFlow._graph)

    def test_custom_linear_passes_lint(self):
        linter.run_checks(CustomNamedLinearFlow._graph)

    def test_single_step_passes_lint(self):
        linter.run_checks(SingleStepFlow._graph)

    def test_custom_branch_passes_lint(self):
        linter.run_checks(CustomNamedBranchFlow._graph)

    def test_custom_foreach_passes_lint(self):
        linter.run_checks(CustomNamedForeachFlow._graph)

    def test_split_start_passes_lint(self):
        linter.run_checks(SplitStartFlow._graph)


# ---------------------------------------------------------------------------
# StepSpec flow definitions
# ---------------------------------------------------------------------------

from metaflow import StepSpec
from metaflow.parameters import InitParameter


class SimpleStepSpec(StepSpec):
    text = Parameter("text", type=str, default="hello")

    def call(self):
        self.output = self.text.upper()


class InitCallStepSpec(StepSpec):
    factor = InitParameter("factor", type=int, default=2)
    value = Parameter("value", type=int, default=10)

    def init(self):
        self.computed = self.factor * 10

    def call(self):
        self.result = self.value * self.computed


# ---------------------------------------------------------------------------
# Tests: StepSpec graph construction
# ---------------------------------------------------------------------------


class TestStepSpecGraph:
    """Tests for StepSpec graph construction."""

    def test_step_named_after_class(self):
        graph = SimpleStepSpec._graph
        assert "simplestepspec" in graph.nodes

    def test_single_step_start_end(self):
        graph = SimpleStepSpec._graph
        assert graph.start_step == "simplestepspec"
        assert graph.end_step == "simplestepspec"
        assert graph.start_step == graph.end_step

    def test_step_type_is_end(self):
        graph = SimpleStepSpec._graph
        assert graph["simplestepspec"].type == "end"

    def test_init_call_graph(self):
        graph = InitCallStepSpec._graph
        assert graph.start_step == "initcallstepspec"
        assert graph.end_step == "initcallstepspec"

    def test_output_steps(self):
        steps_info, graph_structure = SimpleStepSpec._graph.output_steps()
        assert "simplestepspec" in steps_info
        assert graph_structure == ["simplestepspec"]

    def test_passes_lint(self):
        linter.run_checks(SimpleStepSpec._graph)

    def test_init_call_passes_lint(self):
        linter.run_checks(InitCallStepSpec._graph)


# ---------------------------------------------------------------------------
# Tests: StepSpec direct invocation
# ---------------------------------------------------------------------------


class TestStepSpecDirectInvocation:
    """Tests for StepSpec direct invocation."""

    def test_basic_call(self):
        s = SimpleStepSpec(use_cli=False)
        s(text="world")
        assert s.output == "WORLD"

    def test_init_and_call(self):
        m = InitCallStepSpec(factor=3)
        m(value=5)
        assert m.result == 150

    def test_multiple_calls(self):
        s = SimpleStepSpec(use_cli=False)
        s(text="a")
        assert s.output == "A"
        s(text="b")
        assert s.output == "B"

    def test_init_state_persists_across_calls(self):
        m = InitCallStepSpec(factor=5)
        m(value=2)
        assert m.result == 100  # 2 * (5*10)
        m(value=3)
        assert m.result == 150  # 3 * (5*10), init not re-run

    def test_default_values(self):
        s = SimpleStepSpec(use_cli=False)
        s()  # use default text="hello"
        assert s.output == "HELLO"

    def test_config_kwargs_imply_direct_mode(self):
        """Passing any kwarg should imply use_cli=False."""
        m = InitCallStepSpec(factor=2)
        m(value=3)
        assert m.result == 60

    def test_init_parameter_defaults(self):
        """InitParameter defaults should be applied when no kwarg given."""
        m = InitCallStepSpec(use_cli=False)  # factor defaults to 2
        m(value=5)
        assert m.result == 100  # 5 * (2*10)

    def test_call_param_rejected_in_constructor(self):
        """Passing a call-phase Parameter to the constructor should error."""
        import pytest

        with pytest.raises(TypeError):
            SimpleStepSpec(text="wrong_place")


# ---------------------------------------------------------------------------
# Tests: StepSpec with Config parameters
# ---------------------------------------------------------------------------


class TestStepSpecWithConfig:
    """Tests for StepSpec with Config parameters."""

    def test_config_default(self):
        from metaflow.user_configs.config_parameters import Config

        class ConfigSpec(StepSpec):
            cfg = Config("cfg", default_value={"key": "val"})

            def call(self):
                self.result = self.cfg

        c = ConfigSpec(use_cli=False)
        c()
        assert c.result == {"key": "val"}

    def test_config_override(self):
        from metaflow.user_configs.config_parameters import Config

        class ConfigSpec2(StepSpec):
            cfg = Config("cfg", default_value={"key": "default"})

            def call(self):
                self.result = self.cfg

        c = ConfigSpec2(cfg={"key": "override"})
        c()
        assert c.result == {"key": "override"}
