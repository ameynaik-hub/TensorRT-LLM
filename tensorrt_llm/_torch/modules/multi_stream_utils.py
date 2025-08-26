from typing import Any, Callable, Optional

import torch

from ..pyexecutor.cuda_graph_runner import is_graph_capturing


def maybe_execute_in_parallel(
        fn0: Callable,
        fn1: Callable,
        event0: torch.cuda.Event,
        event1: torch.cuda.Event,
        aux_stream: Optional[torch.cuda.Stream] = None) -> tuple[Any, Any]:
    """Utility function to run two functions in two cuda streams in parallel. Multi-stream is
    only enabled when cuda graph is turned on because switch stream has extra host overhead.

    This design is mainly for low latency use case. It needs to be improved for max throughput
    use case.
    For simplicity, fn0 and fn1 do not support inputs.

    Args:
        fn0 (Callable): callable for the default stream
        fn1 (Callable): callable for the second stream, aux_stream
        event0 (torch.cuda.Event): cuda event for fn0
        event1 (torch.cuda.Event): cuda event for fn1
        aux_stream (Optional[torch.cuda.Stream]): the second cuda stream for fn1.
            Multi-stream is disabled when aux_stream is None.

    Returns:
        tuple[Any, Any]: the return values of fn0() and fn1()
    """

    do_multi_stream = is_graph_capturing() and aux_stream is not None

    if do_multi_stream:
        event0.record()
        result0 = fn0()

        with torch.cuda.stream(aux_stream):
            event0.wait()
            result1 = fn1()
            event1.record()
        event1.wait()
    else:
        result0 = fn0()
        result1 = fn1()
    return (result0, result1)


def maybe_execute_in_parallel_3streams(
        fn0: Callable,
        fn1: Callable,
        fn2: Callable,
        event0: torch.cuda.Event,
        event1: torch.cuda.Event,
        event2: torch.cuda.Event,
        aux_stream1: Optional[torch.cuda.Stream] = None,
        aux_stream2: Optional[torch.cuda.Stream] = None) -> tuple[Any, Any, Any]:
    """Utility function to run three functions in three cuda streams in parallel. Multi-stream is
    only enabled when cuda graph is turned on because switch stream has extra host overhead.

    This design is mainly for low latency use case. It needs to be improved for max throughput
    use case.
    For simplicity, fn0, fn1, and fn2 do not support inputs.

    Args:
        fn0 (Callable): callable for the default stream
        fn1 (Callable): callable for the second stream, aux_stream1
        fn2 (Callable): callable for the third stream, aux_stream2
        event0 (torch.cuda.Event): cuda event for fn0
        event1 (torch.cuda.Event): cuda event for fn1
        event2 (torch.cuda.Event): cuda event for fn2
        aux_stream1 (Optional[torch.cuda.Stream]): the second cuda stream for fn1.
        aux_stream2 (Optional[torch.cuda.Stream]): the third cuda stream for fn2.
            Multi-stream is disabled when either aux_stream1 or aux_stream2 is None.

    Returns:
        tuple[Any, Any, Any]: the return values of fn0(), fn1(), and fn2()
    """

    do_multi_stream = is_graph_capturing() and aux_stream1 is not None and aux_stream2 is not None

    if do_multi_stream:
        # Start fn0 on default stream
        event0.record()
        result0 = fn0()

        # Start fn1 on aux_stream1
        with torch.cuda.stream(aux_stream1):
            event0.wait()  # Wait for fn0 to complete if there are dependencies
            result1 = fn1()
            event1.record()

        # Start fn2 on aux_stream2 (can run concurrently with fn1)
        with torch.cuda.stream(aux_stream2):
            event0.wait()  # Wait for fn0 to complete if there are dependencies
            result2 = fn2()
            event2.record()

        # Wait for all streams to complete
        event1.wait()
        event2.wait()
    else:
        # Fall back to sequential execution
        result0 = fn0()
        result1 = fn1()
        result2 = fn2()
    
    return (result0, result1, result2)
