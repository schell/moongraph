# 🌙 moongraph 📈

`moongraph` is a Rust library for scheduling, managing resources, and running directed acyclic graphs.

In `moongraph`, graph nodes are normal Rust functions and graph edges are function parameters and function results. 

The graph is built automatically by registering each function. `moongraph` figures out how the functions connect by their paramaters and their results.  

`moongraph` validates and schedules nodes to run in parallel where possible, using `rayon` as the underlying parallelizing tech (WIP).

## what

`moongraph` is made up of two parts:

* The scheduler - provided by [`dagga`](https://github.com/schell/dagga)
* The resource manager - provided by [`broomdog`](https://github.com/schell/broomdog)

## why

Scheduling and running DAGs is a common problem and I didn't find any prior art.

## uses

`moongraph` is in use in a number of other libraries (let me know if you use it in yours, and how :))

* As the render graph in [`renderling`](https://github.com/schell/renderling), a scrappy real-time renderer with advanced lighting
* Its scheduler is used for scheduling systems in [`apecs`](https://github.com/schell/apecs), a well-performing ECS with async integration.

## 💚 Sponsor this!

This work will always be free and open source. If you use it (outright or for inspiration), please consider donating.

[💰 Sponsor 💝](https://github.com/sponsors/schell)

## License
Renderling is free and open source. All code in this repository is dual-licensed under either:

    MIT License (LICENSE-MIT or http://opensource.org/licenses/MIT)
    Apache License, Version 2.0 (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)

at your option. This means you can select the license you prefer! This dual-licensing approach
is the de-facto standard in the Rust ecosystem and there are very good reasons to include both.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.
