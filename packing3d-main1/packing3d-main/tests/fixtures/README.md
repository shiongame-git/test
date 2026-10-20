# GA semantic reference

`ga_semantics.json` was captured from commit `b881039`, before the 2026-10-07
maintenance refactoring, using CPython 3.13.15 and sequential evaluation without
cache. It records both decoders and both BL priorities for the problem, GA
configuration and seed stored at its root.

The fixture records every individual, fitness, placement, generation best,
global best and PP input state. Timings, run IDs, CPU observations and execution
counters are excluded. It is an internal test fixture, not a public history
serialization format. The test runs the same reference with workers 1/2 and
cache OFF/ON.

When the test fails, inspect the changed generations before deciding whether
the algorithm or reference data should change. Reformatting the JSON alone does
not change the expected results.
