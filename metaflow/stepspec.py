"""
StepSpec -- a FlowSpec subclass for single-computation specs.

StepSpec provides a simplified lifecycle: ``init()`` + ``call()``.
Users define ``call()`` (required) and optionally ``init()``, and the
metaclass synthesizes a single ``@step`` method that wraps both.

Parameters are split into two phases:
  - *InitParameters* and *Configs* are provided at construction time.
  - Regular *Parameters* are provided when the instance is called.
"""

import linecache

from .decorators import step as _step_decorator
from .exception import MetaflowException
from .flowspec import FlowSpec, FlowSpecMeta
from .parameters import Parameter
from .user_configs.config_parameters import Config

# Counter for unique pseudo-filenames so each StepSpec subclass gets its own.
_synth_counter = 0


def _get_param_default(param):
    """
    Return the default value for a Parameter (or Config).

    Config stores ``default_value`` into ``param.kwargs["default"]`` (see
    config_parameters.py).  Parameter stores raw defaults in
    ``param._override_kwargs["default"]``.  We check ``_override_kwargs``
    first because it holds the user-supplied value before ``init()``
    processing merges it into ``kwargs``.
    """
    default = param._override_kwargs.get("default")
    if default is not None:
        return default
    return param.kwargs.get("default")


class StepSpecMeta(FlowSpecMeta):
    """Metaclass for StepSpec.

    For user subclasses it:
      1. Validates that ``call()`` exists and no ``@step`` methods are present.
      2. Synthesizes a single ``@step`` method that calls ``init()`` then ``call()``.
      3. Delegates to ``FlowSpecMeta`` for registration and graph building.

    The synthetic step is injected in ``__new__`` (before the class object is
    created) so that the class body already contains the ``@step`` method when
    ``FlowSpecMeta.__init__`` builds the graph.
    """

    _base_class_names = FlowSpecMeta._base_class_names | frozenset({"StepSpec"})

    def __new__(mcs, name, bases, attrs):
        # For the StepSpec base class itself, skip validation / synthesis.
        if name == "StepSpec":
            return super().__new__(mcs, name, bases, attrs)

        # ------------------------------------------------------------------
        # User subclass validation
        # ------------------------------------------------------------------
        if "call" not in attrs:
            raise MetaflowException(
                "StepSpec subclass '%s' must define a call() method." % name
            )

        # Reject @step-decorated methods
        # Use `is True` (not just truthiness) because Config.__getattr__
        # returns a DelayEvaluator for any unknown attribute, which is truthy.
        for attr_name, attr_val in attrs.items():
            if getattr(attr_val, "is_step", False) is True:
                raise MetaflowException(
                    "StepSpec subclass '%s' must not contain @step methods "
                    "(found '%s'). Use call() instead." % (name, attr_name)
                )

        # Reject class names starting with underscore
        if name.startswith("_"):
            raise MetaflowException(
                "StepSpec subclass name '%s' must not start with an underscore." % name
            )

        # ------------------------------------------------------------------
        # Synthesize the @step method and inject into attrs *before*
        # the class is created so FlowSpecMeta.__init__ sees it.
        #
        # FlowGraph inspects source code via inspect.getsourcelines and
        # parses it with the AST.  For a dynamically-created function we
        # need (a) the def-name to match step_name and (b)
        # inspect.getsourcelines to succeed.  We satisfy both by compiling
        # the function from a source string and registering that source in
        # linecache so inspect can find it.
        # ------------------------------------------------------------------
        step_name = name.lower()
        user_init = attrs.get("init", lambda self: None)
        user_call = attrs["call"]

        # We compile the unindented source for exec, but register an
        # indented version in linecache.  FlowGraph._create_nodes calls
        # deindent_docstring on the source returned by inspect, which
        # strips the common leading indent -- so the indented version
        # becomes valid de-indented source for ast.parse.
        exec_source = (
            "def %s(self):\n    _user_init(self)\n    _user_call(self)\n" % step_name
        )
        inspect_source = (
            "    def %s(self):\n        _user_init(self)\n        _user_call(self)\n"
            % step_name
        )

        global _synth_counter
        _synth_counter += 1
        pseudo_file = "<stepspec-%s-%d>" % (step_name, _synth_counter)

        # Register indented source in linecache so inspect.getsourcelines
        # can find it and FlowGraph deindent processing works correctly.
        linecache.cache[pseudo_file] = (
            len(inspect_source),
            None,
            inspect_source.splitlines(True),
            pseudo_file,
        )

        code = compile(exec_source, pseudo_file, "exec")
        _ns = {"_user_init": user_init, "_user_call": user_call}
        exec(code, _ns)
        _synthetic_fn = _ns[step_name]
        _synthetic_fn.__qualname__ = "%s.%s" % (name, step_name)
        attrs[step_name] = _step_decorator(_synthetic_fn)

        return super().__new__(mcs, name, bases, attrs)

    def __init__(cls, name, bases, attrs):
        # ------------------------------------------------------------------
        # Handle the StepSpec base class itself -- we need _flow_state
        # initialised so subclasses can access it during MRO traversal,
        # but we must NOT register it or create synthetic steps.
        # ------------------------------------------------------------------
        if name == "StepSpec":
            type.__init__(cls, name, bases, attrs)
            cls._init_attrs()
            return

        # Store which step name was synthesised
        cls._step_spec_step_name = name.lower()

        # ------------------------------------------------------------------
        # Delegate to FlowSpecMeta for registration + graph building
        # ------------------------------------------------------------------
        super().__init__(name, bases, attrs)


class StepSpec(FlowSpec, metaclass=StepSpecMeta):
    """
    A FlowSpec subclass for single-computation specs.

    Subclasses must define ``call()`` and may optionally define ``init()``.
    Parameters declared as ``InitParameter`` (or ``Config``) are supplied
    at construction; regular ``Parameter`` instances are supplied when the
    object is called.

    Example
    -------
    ::

        class Upper(StepSpec):
            text = Parameter("text", type=str, default="hello")

            def call(self):
                self.output = self.text.upper()

        u = Upper(use_cli=False)
        u(text="world")
        assert u.output == "WORLD"
    """

    def __init__(self, use_cli=None, **kwargs):
        # If kwargs are provided, force direct-invocation mode.
        if kwargs and use_cli is None:
            use_cli = False
        if use_cli is None:
            use_cli = True

        if use_cli:
            super().__init__(use_cli=True)
            return

        # ------------------------------------------------------------------
        # Direct-invocation mode
        # ------------------------------------------------------------------
        # Initialise FlowSpec internals without launching the CLI.
        self.name = self.__class__.__name__
        self._datastore = None
        self._transition = None
        self._cached_input = {}

        # Classify parameters
        init_params = {}  # name -> (var, param)  -- InitParameter
        config_params = {}  # name -> (var, param)  -- Config
        call_params = {}  # name -> (var, param)  -- regular Parameter

        for var, param in self.__class__._get_parameters():
            if param.IS_CONFIG_PARAMETER:
                config_params[param.name] = (var, param)
            elif getattr(param, "IS_INIT_PARAMETER", False):
                init_params[param.name] = (var, param)
            else:
                call_params[param.name] = (var, param)

        # Build valid kwarg names for the constructor
        valid_init_kwargs = set()
        for pname in init_params:
            valid_init_kwargs.add(pname)
        for pname in config_params:
            valid_init_kwargs.add(pname)

        # Reject unknown kwargs
        unknown = set(kwargs.keys()) - valid_init_kwargs
        if unknown:
            raise TypeError(
                "%s() got unexpected keyword argument(s): %s. "
                "Regular Parameters belong in __call__()."
                % (self.__class__.__name__, ", ".join(sorted(unknown)))
            )

        # Set Config defaults, then overrides from kwargs
        for pname, (var, param) in config_params.items():
            val = kwargs.get(pname, _get_param_default(param))
            setattr(self, var, val)

        # Set InitParameter defaults, then overrides from kwargs
        for pname, (var, param) in init_params.items():
            val = kwargs.get(pname, _get_param_default(param))
            setattr(self, var, val)

        # Store call_params info for use in __call__
        self._stepspec_call_params = call_params

        # Run user init
        self.init()

    def init(self):
        """Override to perform initialisation before ``call()``."""
        pass

    def call(self):
        """Override to define the main computation."""
        raise NotImplementedError("Subclasses must implement call()")

    def __call__(self, **kwargs):
        """
        Execute the computation with the given call-phase parameters.

        Parameters are the regular ``Parameter`` instances declared on the class.
        """
        call_params = self._stepspec_call_params

        # Build valid kwarg names
        valid_call_kwargs = {pname for pname in call_params}

        # Reject unknown kwargs
        unknown = set(kwargs.keys()) - valid_call_kwargs
        if unknown:
            raise TypeError(
                "%s() got unexpected keyword argument(s): %s"
                % (self.__class__.__name__, ", ".join(sorted(unknown)))
            )

        # Set parameter values (use default if not provided)
        for pname, (var, param) in call_params.items():
            val = kwargs.get(pname, _get_param_default(param))
            setattr(self, var, val)

        # Run the user's call method
        self.call()
