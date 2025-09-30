# reduce
## Subjects
- A: native scan on CPU
- B: native scan on GPU
- C: with bank conflicts, no double buffering
- D: with bank conflicts + double buffering
- E: no bank conflict

## Result
Every data below is the average of 10 consecutive test results.

| Input Size | Subject A | Subject B | Subject C | Subject D | Subject E |
| ---------- | --------- | --------- | --------- | --------- | --------- |
| 1e3        | WIP       | WIP       | WIP       | WIP       | WIP       |
| 1e4        | WIP       | WIP       | WIP       | WIP       | WIP       |
| 1e5        | WIP       | WIP       | WIP       | WIP       | WIP       |
| 1e6        | 3.66ms    | 304.15ms  | 0.058ms*  | WIP       | WIP       |
| 1e7        | WIP       | WIP       | WIP       | WIP       | WIP       |

## Notes
- (*) when measuring GPU times using timestamp query feature, I get many `0.65536ms`, which indicates the query use some fixed precision, so the result for GPU algos may not very be accurate when the workload is light.
