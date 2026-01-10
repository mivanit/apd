import functools
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar

import torch

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_TORCH_PROFILER_KWARGS: dict = dict(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    profile_memory=True,
    with_stack=True,
    record_shapes=True,
)


def torch_profile(
    *,
    output_path: str | Path = "trace.json",
    profiler_arg_name: str | None = "profiler",
    profiler_kwargs: dict[str, object] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Start a *torch.profiler.profile* session around the decorated function.

    The profiler starts **before** the wrapped function executes and is
    **always** stopped and exported in a *finally* block, guaranteeing a trace
    even if the function raises.

    # Parameters:
     - `output_path : str | Path`
        Path to the Chrome format trace file (defaults to `"trace.json"`).
     - `profiler_arg_name : str | None`
        If not `None`, the profiler instance is injected into the wrapped
        function under this keyword (defaults to `"profiler"`). If `None`, the
        profiler is *not* passed.
     - `profiler_kwargs : dict[str, object] | None`
        Keyword arguments forwarded verbatim to `torch.profiler.profile`.
        (Defaults to `DEFAULT_TORCH_PROFILER_KWARGS`.)

    # Returns:
     - `Callable`
        The decorated function wrapped with profiling.

    # Usage:

    ```python
    @torch_profile(
        profiler_kwargs=dict(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            with_stack=True,
            record_shapes=True,
        ),
        output_path="train_trace.json",
        profiler_arg_name="prof",  # omit or set None to disable passing
    )
    def train_step(batch, *, prof):
        out = model(batch)
        loss = loss_fn(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prof.step()  # mark iteration boundary
    ```
    """
    profiler_kwargs = profiler_kwargs or DEFAULT_TORCH_PROFILER_KWARGS

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:  # type: ignore[misc]
            prof: torch.profiler.profile = torch.profiler.profile(
                **profiler_kwargs  # type: ignore[arg-type]
            )
            prof.start()
            prof.step()
            try:
                if profiler_arg_name is not None:
                    if profiler_arg_name in kwargs:
                        raise TypeError(
                            f"Keyword '{profiler_arg_name}' already present "
                            "when profiler_arg_name is not None."
                        )
                    kwargs[profiler_arg_name] = prof  # type: ignore[index]
                result: R = func(*args, **kwargs)  # type: ignore[arg-type]
            except Exception as e:
                print(e)
                result = e
                pass
            finally:
                prof.step()
                prof.stop()
                prof.export_chrome_trace(str(output_path))
                # after the run finishes
                events = prof.key_averages()
                # ops that allocate the most on GPU
                print(events.table(sort_by="self_cuda_memory_usage", row_limit=30))
            return result

        return wrapper

    return decorator
