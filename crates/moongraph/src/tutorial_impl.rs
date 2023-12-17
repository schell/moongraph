#[cfg_attr(doc, aquamarine::aquamarine)]
/// # Intro to Moongraph 🌙📈
///
/// `moongraph` is a library for defining, scheduling and running directed acyclic graphs
/// with shared resources.
///
/// ## DAG
/// A "DAG" is a directed, acyclic graph.
/// Within the `moongraph` library, graph nodes are Rust functions and graph edges are the
/// function's input parameters and output parameters. For example, in the following diagram
/// we represent nodes as circles and edges as rectangles:
///
/// ```mermaid
/// flowchart LR
///     A[Input Types] --> B((Rust Function)) --> C[Output Types]
/// ```
///
/// In this way our graphs are "directed" because the output of one function becomes the
/// input of another function, which leads to a required ordering when executing the graph.
///
/// ```mermaid
/// flowchart LR
///     A[Input Types] --> B((Function A)) --> C[Output Types] -- Become Input Types --> D((Function B))
/// ```
///
/// "Acyclic" means that the graph cannot contain cycles between nodes.
/// What this means in `moongraph` is that the result of one function **cannot** be the input to another
/// function whose result is required input of the first function. For example the following graph would
/// fail to schedule:
///
/// ```mermaid
/// flowchart LR
///     A[A Input Types / B Output Types] --> B((Function A)) --> C[A Output Types] --> D((Function B))
///     D --> A
/// ```
///
/// Obviously this graph cannot be scheduled because it's not possible to determine which function to run
/// first!
/// Attempting to run or schedule a graph with cycles will produce an error.
///
/// ### The benefit of a DAG
/// Once a DAG has been defined, it does not change.
/// For this reason we can create a schedule that runs all nodes in the DAG in an optimal way, running as many nodes as possible in parallel, safely accessing shared resources **without locks** and in a synchronous context.
/// This makes using DAGs an alternative to async runtimes and an alternative to message passing systems.
/// DAGs as a main loop are often found in games, simulations and other contexts where reliable, predictable performance is a top concern.
///
/// So if you're still interested, let's learn about creating DAGs with `moongraph`.
///
/// ## Nodes and Edges
/// Graph nodes are Rust functions.
/// There are two requirements for a Rust function to be used as a graph node.
/// 1. The input parameters must implement [`Edges`](crate::Edges) `+ Any + Send + Sync`.
/// 2. The output must implement [`NodeResults`](crate::NodeResults) `+ Any + Send + Sync`.
///
/// [`Edges`](crate::Edges) is a trait that tells `moongraph` how to construct your node's input
/// from its internal resources (we'll explain more about that when we get talking about the `Graph` type). For now we just need to know about the three ways to access an input parameter:
/// 1. By reference
/// 2. By mutable reference
/// 3. By **move**
///
/// The first two should be familiar from Rust in general and the last is somewhat novel. That is - an input parameter to a node function can be borrowed from the graph, borrowed mutably from the graph, or **moved** out of the graph into the function. To represent each type of access we have three wrappers:
/// 1. [`View`](crate::View) - allows a node function to borrow from the graph.
/// 2. [`ViewMut`](crate::ViewMut) - allows a node function to borrow mutably from the graph.
/// 3. [`Move`](crate::Move) - allows a node function to move a type from the graph into itself.
///
/// With these we can construct our input parameters by wrapping them in a tuple.
/// Here we define a node function that uses all three and possibly returns an error:
///
/// ```rust
/// use moongraph::{View, ViewMut, Move, GraphError};
///
/// fn node_b((s, mut u, f): (View<&'static str>, ViewMut<usize>, Move<f32>)) -> Result<(), GraphError> {
///     *u += f.floor() as usize;
///     println!("s={s}, u={u}, f={f}");
///     Ok(())
/// }
/// ```
///
/// And here we'll define another node that creates `node_b`'s `f32` as a result:
///
/// ```rust
/// # use moongraph::GraphError;
/// fn node_a(_:()) -> Result<(f32,), GraphError> {
///     Ok((42.0,))
/// }
/// ```
///
/// Notice how even though `node_a` doesn't use any input we still have to pass `()`.
/// Also notice how the result is a tuple even though there's only one element `(f32,)`.
/// These are both warts due to constraints in Rust's type system.
/// That's the worst of it though, and it's easy enough to live with.
///
/// ## Graph
/// The top level type you'll interact with is called [`Graph`](crate::Graph).
/// Now that we have our nodes we can start constructing our graph.
/// A [`Graph`](crate::Graph) is made up of two main concepts - [`Execution`](crate::Execution) and [`TypeMap`](crate::TypeMap).
///
/// [`Execution`](crate::Execution) is a collection of node functions and a schedule that determines their running order.
///
/// [`TypeMap`](crate::TypeMap) is a collection of resources (edge types).
///
/// For the most part we won't have to interact with either of these.
/// Instead we'll be adding functions and resources to the graph using `Graph`'s API, which will
/// do this interaction for us, but it's good to understand the concepts.
///
/// So - let's construct a graph using our previously created nodes.
/// For this we'll use the [`graph!`](crate::graph) macro, but you can use the API directly if you like.
///
/// ```rust
/// use moongraph::{View, ViewMut, Move, GraphError, Graph, graph};
///
/// fn node_b((s, mut u, f): (View<&'static str>, ViewMut<usize>, Move<f32>)) -> Result<(), GraphError> {
///     *u += f.floor() as usize;
///     println!("s={s}, u={u}, f={f}");
///     Ok(())
/// }
///
/// fn node_a(_:()) -> Result<(f32,), GraphError> {
///     Ok((42.0,))
/// }
///
/// let mut graph = graph!(node_b, node_a)
///     .with_resource("a big blue crystal") // add the &'static str resource
///     .with_resource(0usize); // add the usize resource
/// // we don't have to add the f32 resource because it comes as the result of node_a
/// ```
///
/// ### Resources
/// Input parameters don't always have to come from the results of functions, in fact they
/// more often come from a set of resources that live inside the graph.
/// Notice how we added the `&'static str` and `usize` resources to the graph explicitly using [`Graph::with_resource`](crate::Graph::with_resource).
/// So to recap - edge types are also known as resources and can come from the results of node functions or can be inserted into the graph manually before executing it.
///
/// ### Execution
/// After the graph is built it can be executed.
/// The order of execution of node functions is determined by the edge types of the functions themselves.
/// For example, since `node_b` takes as input the results of `node_a`, `node_b` must run _after_ `node_a`.
/// [`Graph`](crate::Graph) is smart enough to figure out these types of orderings by itself.
/// It can also determine which node functions access resources _mutably_, and therefore which node functions can be run in parallel.
/// But there are certain scenarios where type information is not enough to determine ordering.
/// For example if we were building a game we might need the physics step to run before rendering, but the types may not describe this and so we would have to state that ordering explicitly.
/// Luckily we can do that using [`graph!`](crate::graph) and the `>` and `<` operators:
/// ```rust
/// use moongraph::{Graph, GraphError, graph};
///
/// fn physics_step(_:()) -> Result<(), GraphError> {
///     println!("physics step goes here");
///     Ok(())
/// }
///
/// fn render(_:()) -> Result<(), GraphError> {
///     println!("rendering goes here");
///     Ok(())
/// }
///
/// let graph = graph!(physics_step < render);
/// ```
/// Alternatively we can use the [`Graph`](crate::Graph) and [`Node`](crate::Node) APIs to do the same thing:
/// ```rust
/// # use moongraph::{IsGraphNode, Graph, GraphError, graph};
/// #
/// # fn physics_step(_:()) -> Result<(), GraphError> {
/// #     println!("physics step goes here");
/// #     Ok(())
/// # }
/// #
/// # fn render(_:()) -> Result<(), GraphError> {
/// #     println!("rendering goes here");
/// #     Ok(())
/// # }
/// #
/// let graph = Graph::default()
///    .with_node(
///        physics_step
///            .into_node()
///            .with_name("physics_step")
///            .run_before("render")
///    )
///    .with_node(render.into_node().with_name("render"));
/// ```
/// You can see that the [`graph!`](crate::graph) macro is doing a lot of heavy lifting.
///
/// Now that we have our graph, we can run it:
/// ```rust
/// # use moongraph::{IsGraphNode, Graph, GraphError, graph};
/// #
/// # fn physics_step(_:()) -> Result<(), GraphError> {
/// #     println!("physics step goes here");
/// #     Ok(())
/// # }
/// #
/// # fn render(_:()) -> Result<(), GraphError> {
/// #     println!("rendering goes here");
/// #     Ok(())
/// # }
/// #
/// let mut graph = graph!(physics_step < render);
/// graph.run().unwrap();
/// ```
///
/// ## Full example
/// Here's another example where we'll implement a number of physics steps followed by an unrelated banking step.
/// After constructing the graph, we'll run the graph and also print out the schedule.
/// ```rust
/// use moongraph::*;
///
/// #[derive(Default)]
/// pub struct Position(f32, f32);
///
/// #[derive(Default)]
/// pub struct Velocity(f32, f32);
///
/// #[derive(Default)]
/// pub struct Acceleration(f32, f32);
///
/// #[derive(Default)]
/// pub struct BankAccount {
///     interest_rate: f32,
///     balance: f32,
/// }
///
/// pub fn add_acceleration((mut v, a): (ViewMut<Velocity>, View<Acceleration>)) -> Result<(), GraphError> {
///     v.0 += a.0;
///     v.1 += a.1;
///     Ok(())
/// }
///
/// pub fn add_velocity((mut p, v): (ViewMut<Position>, View<Velocity>)) -> Result<(), GraphError> {
///     p.0 += v.0;
///     p.1 += v.1;
///     Ok(())
/// }
///
/// pub fn compound_interest(mut acct: ViewMut<BankAccount>) -> Result<(), GraphError> {
///     let increment = acct.interest_rate * acct.balance;
///     acct.balance += increment;
///     Ok(())
/// }
///
/// let mut graph = graph!(
///     add_acceleration,
///     add_velocity,
///     compound_interest
/// )
/// .with_resource(Position(0.0, 0.0))
/// .with_resource(Velocity(0.0, 0.0))
/// .with_resource(Acceleration(1.0, 1.0))
/// .with_resource(BankAccount {
///     interest_rate: 0.02,
///     balance: 1337.0,
/// });
///
/// graph.run().unwrap();
/// let schedule = graph.get_schedule();
/// assert_eq!(
///     vec![
///         vec!["add_velocity", "compound_interest"],
///         vec!["add_acceleration"]
///     ],
///     schedule,
/// );
///
/// graph.run().unwrap();
/// let position = graph.get_resource::<Position>().unwrap().unwrap();
/// assert_eq!((1.0, 1.0), (position.0, position.1));
/// ```
///
/// Notice how the returned schedule shows that `add_velocity` and `compound_interest` can run together in parallel. We call this a "batch". It's possible to run all nodes in a batch at the same time because none of their borrows conflict and there are no explicit ordering constraints between them.
///
///
///
/// ## Conclusion
/// Hopefully by this point you have a better idea what `moongraph` is about and how to use it.
/// For more info please look at the module and type level documentation.
/// If you have any questions please reach out at [the moongraph GitHub repository](https://github.com/schell/moongraph).
///
/// Happy hacking! ☕☕☕
pub mod tutorial {}
