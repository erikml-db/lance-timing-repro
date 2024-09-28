# lance-timing-repro

To run
```
cargo run --release -- --index-version=v1
cargo run --release -- --index-version=v3
```

V1 Results
```
Running query with nprobes: 10, refine_factor: 0
Time: 857.855µs

Running query with nprobes: 10, refine_factor: 10
Time: 1.9901ms

Running query with nprobes: 512, refine_factor: 10
Time: 7.296543ms
```

V3 Results
```
Running query with nprobes: 10, refine_factor: 0
Time: 3.168363ms

Running query with nprobes: 10, refine_factor: 10
Time: 4.349936ms

Running query with nprobes: 512, refine_factor: 10
Time: 126.882095ms
```