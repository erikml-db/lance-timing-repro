# lance-timing-repro

To run
```
cargo run --release -- --index-version=v1
cargo run --release -- --index-version=v3
```

V1 Results
```
Running query with nprobes: 10, refine_factor: 0
Time: 942.829µs

Running query with nprobes: 10, refine_factor: 10
Time: 2.048292ms

Running query with nprobes: 512, refine_factor: 10
Time: 7.607578ms
```

V3 Results
```
Running query with nprobes: 10, refine_factor: 0
Time: 3.442128ms

Running query with nprobes: 10, refine_factor: 10
Time: 4.611809ms

Running query with nprobes: 512, refine_factor: 10
Time: 70.564001ms
```