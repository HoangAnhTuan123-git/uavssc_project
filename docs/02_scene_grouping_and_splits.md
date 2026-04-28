# Scene grouping and split policy

Main rule:
- keep all runs of the same physical location in the same outer split

Examples:
- `AMtown01`, `AMtown02`, `AMtown03` -> one physical scene group: `AMtown`
- `HKairport01`, `HKairport02`, `HKairport03` -> one physical scene group: `HKairport`

Recommended primary protocol:
- strict cross-scene folds
- one physical scene for test
- one physical scene for validation
- remaining physical scenes for training

Recommended secondary protocol:
- same-scene unseen-run evaluation
- use only as an auxiliary experiment, not the main benchmark claim
