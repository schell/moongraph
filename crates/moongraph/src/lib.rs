//! DAG scheduling, resource managment, and execution.
//!
//! In `moongraph`, nodes are functions with parameters that are accessed
//! immutably, mutably or by move.
//!
//! `moongraph` validates and schedules nodes to run in parallel where possible,
//! using `rayon` as the underlying parallelizing tech (WIP).

use std::{
    any::Any,
    collections::HashMap,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use broomdog::{Loan, LoanMut};
use dagga::Dag;
use snafu::prelude::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub use broomdog::{TypeKey, TypeMap};
pub use dagga::{DaggaError, Node};
pub use moongraph_macros::Edges;

/// All errors.
#[derive(Debug, Snafu)]
pub enum GraphError {
    #[snafu(display("Error while running node: {}", error))]
    RunningNode {
        error: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    #[snafu(display("Error while running local node {error}"))]
    RunningLocalNode { error: String },

    #[snafu(display("Error scheduling the graph: {source}"))]
    Scheduling { source: dagga::DaggaError },

    #[snafu(display("Resource error: {source}"))]
    Resource { source: broomdog::BroomdogErr },

    #[snafu(display("Resource is loaned"))]
    Loaned,

    #[snafu(display("Missing resource '{name}'"))]
    Missing { name: &'static str },

    #[snafu(display("Encountered local function that was not provided or already run"))]
    MissingLocal,
}

type Resource = Box<dyn Any + Send + Sync>;

/// A function wrapper.
///
/// Wraps a function by moving it into a closure. Before running the parameters
/// of the function are constructed from a TypeMap of resources, and the results
/// of the function are packed back into the same TypeMap.
pub struct Function {
    prepare: Box<dyn Fn(&mut TypeMap) -> Result<Resource, GraphError>>,
    run: Option<Box<dyn Fn(Resource) -> Result<Resource, GraphError> + Send + Sync>>,
    save: Box<dyn Fn(Resource, &mut TypeMap) -> Result<(), GraphError>>,
}

impl Function {
    /// Run the function using the given `TypeMap`.
    pub fn run(
        &mut self,
        resources: Resource,
        local: &mut Option<impl FnOnce(Resource) -> Result<Resource, GraphError>>,
    ) -> Result<Resource, GraphError> {
        if let Some(f) = self.run.as_ref() {
            (f)(resources)
        } else {
            let local = local.take().context(MissingLocalSnafu)?;
            (local)(resources)
        }
    }
}

fn missing_local(_: ()) -> Result<(), GraphError> {
    Err(GraphError::MissingLocal)
}

/// Trait for describing types that are made up of graph edges (ie resources).
///
/// Graph edges are the _resources_ that graph nodes (ie functions) consume.
///
/// The `Edges` trait allows the library user to construct types that use
/// resources. This is convenient when the number of resources becomes large
/// and using a tuple becomes unwieldy.
pub trait Edges: Sized {
    /// Keys of all read types used in fields in the implementor.
    fn reads() -> Vec<TypeKey> {
        vec![]
    }

    /// Keys of all write types used in fields in the implementor.
    fn writes() -> Vec<TypeKey> {
        vec![]
    }

    /// Keys of all move types used in fields in the implementor.
    fn moves() -> Vec<TypeKey> {
        vec![]
    }

    /// Attempt to construct the implementor from the given `TypeMap`.
    fn construct(resources: &mut TypeMap) -> Result<Self, GraphError>;
}

impl Edges for () {
    fn construct(_: &mut TypeMap) -> Result<Self, GraphError> {
        Ok(())
    }
}

macro_rules! impl_edges {
    ($($t:ident),+) => {
        impl<$($t: Edges),+> Edges for ($($t,)+) {
            fn construct(resources: &mut TypeMap) -> Result<Self, GraphError> {
                Ok((
                    $( $t::construct(resources)?, )+
                ))
            }

            fn reads() -> Vec<TypeKey> {
                vec![
                    $( $t::reads(), )+
                ].concat()
            }

            fn writes() -> Vec<TypeKey> {
                vec![
                    $( $t::writes(), )+
                ].concat()
            }

            fn moves() -> Vec<TypeKey> {
                vec![
                    $( $t::moves(), )+
                ].concat()
            }
        }
    }
}

impl_edges!(A);
impl_edges!(A, B);
impl_edges!(A, B, C);
impl_edges!(A, B, C, D);
impl_edges!(A, B, C, D, E);
impl_edges!(A, B, C, D, E, F);
impl_edges!(A, B, C, D, E, F, G);
impl_edges!(A, B, C, D, E, F, G, H);
impl_edges!(A, B, C, D, E, F, G, H, I);
impl_edges!(A, B, C, D, E, F, G, H, I, J);
impl_edges!(A, B, C, D, E, F, G, H, I, J, K);
impl_edges!(A, B, C, D, E, F, G, H, I, J, K, L);

/// Trait for describing types that are the result of running a node.
///
/// When a node runs it may result in the creation of graph edges (ie
/// resources). Graph edges are the _resources_ that other nodes (ie functions)
/// consume.
///
/// The `NodeResults` trait allows the library user to emit tuples of resources
/// that will then be stored in the graph for downstream nodes to use as input.
pub trait NodeResults {
    /// All keys of types/resources created.
    fn creates() -> Vec<TypeKey>;

    /// Attempt to pack the implementor's constituent resources into the given
    /// `TypeMap`.
    fn save(self, resources: &mut TypeMap) -> Result<(), GraphError>;
}

impl NodeResults for () {
    fn creates() -> Vec<TypeKey> {
        vec![]
    }

    fn save(self, _: &mut TypeMap) -> Result<(), GraphError> {
        Ok(())
    }
}

macro_rules! impl_node_results {
    ($(($t:ident, $n:tt)),+) => {
        impl<$( $t : Any + Send + Sync ),+> NodeResults for ($($t,)+) {
            fn creates() -> Vec<TypeKey> {
                vec![$( TypeKey::new::<$t>() ),+]
            }

            fn save(self, resources: &mut TypeMap) -> Result<(), GraphError> {
                $( let _ = resources.insert_value( self.$n ); )+
                Ok(())
            }
        }
    }
}

impl_node_results!((A, 0));
impl_node_results!((A, 0), (B, 1));
impl_node_results!((A, 0), (B, 1), (C, 2));
impl_node_results!((A, 0), (B, 1), (C, 2), (D, 3));
impl_node_results!((A, 0), (B, 1), (C, 2), (D, 3), (E, 4));
impl_node_results!((A, 0), (B, 1), (C, 2), (D, 3), (E, 4), (F, 5));
impl_node_results!((A, 0), (B, 1), (C, 2), (D, 3), (E, 4), (F, 5), (G, 6));
impl_node_results!(
    (A, 0),
    (B, 1),
    (C, 2),
    (D, 3),
    (E, 4),
    (F, 5),
    (G, 6),
    (H, 7)
);
impl_node_results!(
    (A, 0),
    (B, 1),
    (C, 2),
    (D, 3),
    (E, 4),
    (F, 5),
    (G, 6),
    (H, 7),
    (I, 8)
);
impl_node_results!(
    (A, 0),
    (B, 1),
    (C, 2),
    (D, 3),
    (E, 4),
    (F, 5),
    (G, 6),
    (H, 7),
    (I, 8),
    (J, 9)
);
impl_node_results!(
    (A, 0),
    (B, 1),
    (C, 2),
    (D, 3),
    (E, 4),
    (F, 5),
    (G, 6),
    (H, 7),
    (I, 8),
    (J, 9),
    (K, 10)
);
impl_node_results!(
    (A, 0),
    (B, 1),
    (C, 2),
    (D, 3),
    (E, 4),
    (F, 5),
    (G, 6),
    (H, 7),
    (I, 8),
    (J, 9),
    (K, 10),
    (L, 11)
);

fn prepare<Input: Edges + Any + Send + Sync>(
    resources: &mut TypeMap,
) -> Result<Resource, GraphError> {
    let input = Input::construct(resources)?;
    Ok(Box::new(input))
}

fn save<Output: NodeResults + Any + Send + Sync>(
    creates: Resource,
    resources: &mut TypeMap,
) -> Result<(), GraphError> {
    let creates = *creates.downcast::<Output>().unwrap();
    creates.save(resources)
}

/// Defines graph nodes.
///
/// A node in the graph is a boxed Rust closure that may do any or all the
/// following:
///
/// * Create resources by returning a result that implements [`NodeResults`].
/// * Consume one or more resources by having a field in the input parameter
///   wrapped in [`Move`]. The resource will not be available in the graph after
///   the node is run.
/// * Read one or more resources by having a field in the input parameter
///   wrapped in [`Read`].
/// * Write one or more resources by having a field in the input parameter
///   wrapped in [`Write`].
///
/// By default `IsGraphNode` is implemented for functions that take one
/// parameter implementing [`Edges`] and returning a `Result` where the "ok"
/// type implements `NodeResults`.
pub trait IsGraphNode<Input, Output> {
    /// Convert the implementor into a `Node`.
    fn into_node(self) -> Node<Function, TypeKey>;
}

impl<
        Input: Edges + Any + Send + Sync,
        Output: NodeResults + Any + Send + Sync,
        F: Fn(Input) -> Result<Output, E> + Send + Sync + 'static,
        E: std::error::Error + Send + Sync + 'static,
    > IsGraphNode<Input, Output> for F
{
    fn into_node(self) -> Node<Function, TypeKey> {
        let prepare = Box::new(prepare::<Input>);
        let save = Box::new(save::<Output>);

        let inner = Box::new(move |resources: Resource| -> Result<Resource, GraphError> {
            let input = *resources.downcast::<Input>().unwrap();
            match (self)(input) {
                Ok(creates) => Ok(Box::new(creates)),
                Err(e) => Err(GraphError::RunningNode { error: Box::new(e) }),
            }
        });
        Node::new(Function {
            prepare,
            run: Some(inner),
            save,
        })
        .with_reads(Input::reads())
        .with_writes(Input::writes())
        .with_moves(Input::moves())
        .with_results(Output::creates())
    }
}

/// Specifies a graph edge/resource that is "moved" by a node.
pub struct Move<T> {
    inner: T,
}

impl<T: Any + Send + Sync> Edges for Move<T> {
    fn moves() -> Vec<TypeKey> {
        vec![TypeKey::new::<T>()]
    }

    fn construct(resources: &mut TypeMap) -> Result<Self, GraphError> {
        let key = TypeKey::new::<T>();
        let inner_loan = resources
            .remove(&key)
            .context(MissingSnafu { name: key.name() })?;
        let value = inner_loan.into_owned(key.name()).context(ResourceSnafu)?;
        // UNWRAP: safe because we got this out as `T`
        let box_t = value.downcast::<T>().unwrap();
        Ok(Move { inner: *box_t })
    }
}

impl<T> Move<T> {
    /// Convert into its inner type.
    pub fn into(self) -> T {
        self.inner
    }
}

impl<T> Deref for Move<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for Move<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// Specifies a graph edge/resource that can be "read" by a node.
pub struct View<T> {
    inner: Loan,
    _phantom: PhantomData<T>,
}

impl<T: Any + Send + Sync> Deref for View<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // UNWRAP: safe because it was constructed with `T`
        self.inner.downcast_ref().unwrap()
    }
}

impl<T: Any + Send + Sync> Edges for View<T> {
    fn reads() -> Vec<TypeKey> {
        vec![TypeKey::new::<T>()]
    }

    fn construct(resources: &mut TypeMap) -> Result<Self, GraphError> {
        let key = TypeKey::new::<T>();
        let inner = resources
            .loan(key)
            .context(ResourceSnafu)?
            .context(MissingSnafu {
                name: std::any::type_name::<T>(),
            })?;
        Ok(View {
            inner,
            _phantom: PhantomData,
        })
    }
}

/// Specifies a graph edge/resource that can be "written" to by a node.
pub struct ViewMut<T> {
    inner: LoanMut,
    _phantom: PhantomData<T>,
}

impl<T: Any + Send + Sync> Deref for ViewMut<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // UNWRAP: safe because it was constructed with `T`
        self.inner.downcast_ref().unwrap()
    }
}

impl<T: Any + Send + Sync> DerefMut for ViewMut<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // UNWRAP: safe because it was constructed with `T`
        self.inner.downcast_mut().unwrap()
    }
}

impl<'a, T: Any + Send + Sync> Edges for ViewMut<T> {
    fn writes() -> Vec<TypeKey> {
        vec![TypeKey::new::<T>()]
    }

    fn construct(resources: &mut TypeMap) -> Result<Self, GraphError> {
        let key = TypeKey::new::<T>();
        let inner = resources
            .loan_mut(key)
            .context(ResourceSnafu)?
            .context(MissingSnafu {
                name: std::any::type_name::<T>(),
            })?;
        Ok(ViewMut {
            inner,
            _phantom: PhantomData,
        })
    }
}

/// An acyclic, directed graph made up of nodes/functions and edges/resources.
///
/// Notably nodes may have additional run requirements added to them besides
/// input/output requirements. These include:
///
/// * barriers
/// * run before node
/// * run after node
///
/// See the module documentation for [`Node`] for more info on constructing
/// nodes with granular constraints.
#[derive(Default)]
pub struct Graph {
    resources: TypeMap,
    barrier: usize,
    unscheduled: Vec<Node<Function, TypeKey>>,
    schedule: Vec<Vec<Node<Function, TypeKey>>>,
}

impl Graph {
    /// Merge two graphs, preferring the right in cases of key collisions.
    ///
    /// The values of `rhs` will override those of `lhs`.
    pub fn merge(mut lhs: Graph, mut rhs: Graph) -> Graph {
        lhs.unschedule();
        rhs.unschedule();
        let Graph {
            resources: mut rhs_resources,
            unscheduled: rhs_nodes,
            barrier: _,
            schedule: _,
        } = rhs;
        lhs.resources
            .extend(std::mem::take(rhs_resources.deref_mut()).into_iter());
        let mut unscheduled: HashMap<String, Node<Function, TypeKey>> = HashMap::default();
        unscheduled.extend(
            lhs.unscheduled
                .into_iter()
                .map(|node| (node.name().to_string(), node)),
        );
        unscheduled.extend(
            rhs_nodes
                .into_iter()
                .map(|node| (node.name().to_string(), node)),
        );
        lhs.unscheduled = unscheduled.into_iter().map(|(_, node)| node).collect();
        lhs.barrier = lhs.barrier.max(rhs.barrier);
        lhs
    }

    /// Unschedule all functions.
    fn unschedule(&mut self) {
        self.unscheduled
            .extend(std::mem::take(&mut self.schedule).into_iter().flatten());
    }

    /// Reschedule all functions.
    ///
    /// If the functions were already scheduled this will unscheduled them
    /// first.
    pub fn reschedule(&mut self) -> Result<(), GraphError> {
        log::trace!("rescheduling the render graph:");
        self.unschedule();
        let all_nodes = std::mem::take(&mut self.unscheduled);
        let dag = all_nodes
            .into_iter()
            .fold(Dag::default(), |dag, node| dag.with_node(node));
        let schedule =
            dag.build_schedule()
                .map_err(|dagga::BuildScheduleError { source, mut dag }| {
                    // we have to put the nodes back so the library user can do debugging
                    for node in dag.take_nodes() {
                        self.add_node(node);
                    }
                    GraphError::Scheduling { source }
                })?;
        let batched_names = schedule.batched_names();
        log::trace!("{:#?}", batched_names);
        self.schedule = schedule.batches;
        Ok(())
    }

    pub fn get_schedule(&self) -> Vec<Vec<&str>> {
        self.schedule
            .iter()
            .map(|batch| batch.iter().map(|node| node.name()).collect())
            .collect()
    }

    /// An iterator over all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &Node<Function, TypeKey>> {
        self.schedule
            .iter()
            .flatten()
            .chain(self.unscheduled.iter())
    }

    /// A mutable iterator over all nodes.
    pub fn nodes_mut(&mut self) -> impl Iterator<Item = &mut Node<Function, TypeKey>> {
        self.schedule
            .iter_mut()
            .flatten()
            .chain(self.unscheduled.iter_mut())
    }

    /// Add multiple nodes to this graph.
    pub fn with_nodes(self, nodes: impl IntoIterator<Item = Node<Function, TypeKey>>) -> Self {
        nodes.into_iter().fold(self, Self::with_node)
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: Node<Function, TypeKey>) {
        self.unscheduled.push(node.runs_after_barrier(self.barrier));
    }

    /// Return a reference to the node with the given name, if possible.
    pub fn get_node(&self, name: impl AsRef<str>) -> Option<&Node<Function, TypeKey>> {
        for node in self.nodes() {
            if node.name() == name.as_ref() {
                return Some(node);
            }
        }
        None
    }

    /// Return a mutable reference to the node with the given name, if possible.
    pub fn get_node_mut(&mut self, name: impl AsRef<str>) -> Option<&mut Node<Function, TypeKey>> {
        for node in self.nodes_mut() {
            if node.name() == name.as_ref() {
                return Some(node);
            }
        }
        None
    }

    /// Remove a node from the graph by name.
    ///
    /// This leaves the graph in an unscheduled state.
    pub fn remove_node(&mut self, name: impl AsRef<str>) -> Option<Node<Function, TypeKey>> {
        self.unschedule();
        let mut may_index = None;
        for (i, node) in self.unscheduled.iter().enumerate() {
            if node.name() == name.as_ref() {
                may_index = Some(i);
            }
        }
        if let Some(i) = may_index.take() {
            Some(self.unscheduled.swap_remove(i))
        } else {
            None
        }
    }

    /// Add a node to the graph.
    pub fn with_node(mut self, node: Node<Function, TypeKey>) -> Self {
        self.add_node(node);
        self
    }

    /// Add a named function to the graph.
    pub fn with_function<Input, Output>(
        mut self,
        name: impl Into<String>,
        f: impl IsGraphNode<Input, Output>,
    ) -> Self {
        self.add_function(name, f);
        self
    }

    /// Add a named function to the graph.
    pub fn add_function<Input, Output>(
        &mut self,
        name: impl Into<String>,
        f: impl IsGraphNode<Input, Output>,
    ) {
        self.add_node(f.into_node().with_name(name));
    }

    /// Return whether the graph contains a node/function with the given name.
    pub fn contains_node(&self, name: impl AsRef<str>) -> bool {
        let name = name.as_ref();
        let search = |node: &Node<Function, TypeKey>| node.name() == name;
        if self.unscheduled.iter().any(search) {
            return true;
        }
        self.schedule.iter().flatten().any(search)
    }

    /// Return whether the graph contains a resource with the parameterized
    /// type.
    pub fn contains_resource<T: Any + Send + Sync>(&self) -> bool {
        let key = TypeKey::new::<T>();
        self.resources.contains_key(&key)
    }

    /// Explicitly insert a resource (an edge) into the graph.
    ///
    /// This will overwrite an existing resource of the same type in the graph.
    pub fn with_resource<T: Any + Send + Sync>(mut self, t: T) -> Self {
        self.add_resource(t);
        self
    }

    /// Explicitly insert a resource (an edge) into the graph.
    ///
    /// This will overwrite an existing resource of the same type in the graph.
    pub fn add_resource<T: Any + Send + Sync>(&mut self, t: T) {
        // UNWRAP: safe because of the guarantees around `insert_value`
        self.resources.insert_value(t).unwrap();
    }

    /// Add a barrier to the graph.
    ///
    /// All nodes added after the barrier will run after nodes added before the
    /// barrier.
    pub fn add_barrier(&mut self) {
        self.barrier += 1;
    }

    /// Add a barrier to the graph.
    ///
    /// All nodes added after the barrier will run after nodes added before the
    /// barrier.
    pub fn with_barrier(mut self) -> Self {
        self.add_barrier();
        self
    }

    /// Add a locally run function to the graph by adding its name, input and
    /// output params.
    ///
    /// There may be only one locally run function.
    ///
    /// If a graph contains a local function the graph _MUST_ be run with
    /// [`Graph::run_with_local`].
    pub fn add_local<Input, Output>(&mut self, name: impl Into<String>)
    where
        Input: Edges + Any + Send + Sync,
        Output: NodeResults + Any + Send + Sync,
    {
        self.add_node(Self::make_local::<Input, Output>().with_name(name));
    }

    pub fn with_local<Input, Output>(mut self, name: impl Into<String>) -> Self
    where
        Input: Edges + Any + Send + Sync,
        Output: NodeResults + Any + Send + Sync,
    {
        self.add_local::<Input, Output>(name);
        self
    }

    pub fn make_local<Input, Output>() -> Node<Function, TypeKey>
    where
        Input: Edges + Any + Send + Sync,
        Output: NodeResults + Any + Send + Sync,
    {
        Node::new(Function {
            prepare: Box::new(prepare::<Input>),
            run: None,
            save: Box::new(save::<Output>),
        })
        .with_reads(Input::reads())
        .with_writes(Input::writes())
        .with_moves(Input::moves())
        .with_results(Output::creates())
    }

    /// Run the graph.
    pub fn run(&mut self) -> Result<(), GraphError> {
        self.run_with_local(missing_local)
    }

    /// Run the graph with the given local function.
    pub fn run_with_local<Input, Output, E>(
        &mut self,
        f: impl FnOnce(Input) -> Result<Output, E>,
    ) -> Result<(), GraphError>
    where
        Input: Edges + Any + Send + Sync,
        Output: NodeResults + Any + Send + Sync,
        E: ToString,
    {
        let mut local = Some(move |resources: Resource| {
            let input = *resources.downcast::<Input>().unwrap();
            match (f)(input) {
                Ok(creates) => Ok(Box::new(creates) as Resource),
                Err(e) => Err(GraphError::RunningLocalNode {
                    error: e.to_string(),
                }),
            }
        });

        if !self.unscheduled.is_empty() {
            self.reschedule()?;
        }

        #[derive(Default)]
        struct Batch<'a, 'b> {
            inputs: Vec<Resource>,
            runs: Vec<&'a Box<dyn Fn(Resource) -> Result<Resource, GraphError> + Send + Sync>>,
            local: Option<(
                Resource,
                Box<dyn FnOnce(Resource) -> Result<Resource, GraphError> + 'b>,
            )>,
        }

        impl<'a, 'b> Batch<'a, 'b> {
            #[cfg(feature = "parallel")]
            fn run(self) -> Vec<Result<Resource, GraphError>> {
                let Batch {
                    inputs,
                    runs,
                    local,
                } = self;
                let mut outputs = inputs
                    .into_par_iter()
                    .zip(runs.into_par_iter())
                    .map(|(input, f)| (f)(input))
                    .collect::<Vec<_>>();
                if let Some((input, f)) = local {
                    outputs.push((f)(input));
                }
                outputs
            }

            #[cfg(not(feature = "parallel"))]
            fn run(self) -> Vec<Result<Resource, GraphError>> {
                let Batch {
                    inputs,
                    runs,
                    local,
                } = self;
                let mut outputs = inputs
                    .into_iter()
                    .zip(runs.into_iter())
                    .map(|(input, f)| (f)(input))
                    .collect::<Vec<_>>();
                if let Some((input, f)) = local {
                    outputs.push((f)(input));
                }
                outputs
            }
        }

        for nodes in self.schedule.iter_mut() {
            let mut batch = Batch::default();
            for node in nodes.iter() {
                let input = (node.inner().prepare)(&mut self.resources)?;
                if let Some(f) = node.inner().run.as_ref() {
                    batch.inputs.push(input);
                    batch.runs.push(f);
                } else {
                    let f = local.take().context(MissingLocalSnafu)?;
                    batch.local = Some((
                        input,
                        Box::new(f) as Box<dyn FnOnce(Resource) -> Result<Resource, GraphError>>,
                    ));
                }
            }

            for (node, output) in nodes.iter().zip(batch.run()) {
                let output = output?;
                (node.inner().save)(output, &mut self.resources)?;
            }

            self.resources.unify().context(ResourceSnafu)?;
        }

        Ok(())
    }

    /// Remove a resource from the graph.
    pub fn remove_resource<T: Any + Send + Sync>(&mut self) -> Result<Option<T>, GraphError> {
        let key = TypeKey::new::<T>();
        if let Some(inner_loan) = self.resources.remove(&key) {
            let value = inner_loan
                .into_owned(key.name())
                .with_context(|_| ResourceSnafu)?;
            let box_t = value.downcast::<T>().ok().with_context(|| LoanedSnafu)?;
            Ok(Some(*box_t))
        } else {
            Ok(None)
        }
    }

    /// Get a reference to a resource in the graph.
    pub fn get_resource<T: Any + Send + Sync>(&self) -> Result<Option<&T>, GraphError> {
        Ok(self.resources.get_value().context(ResourceSnafu)?)
    }

    /// Get a mutable reference to a resource in the graph.
    pub fn get_resource_mut<T: Any + Send + Sync>(&mut self) -> Result<Option<&mut T>, GraphError> {
        Ok(self.resources.get_value_mut().context(ResourceSnafu)?)
    }

    /// Fetch graph edges and visit them with a closure.
    ///
    /// This is like running a one-off graph node, but `S` does not get packed
    /// into the graph as a result resource, instead it is given back to the
    /// callsite.
    ///
    /// ## Note
    /// By design, visiting the graph with a type that uses `Move` in one of its
    /// fields will result in the wrapped type of that field being `move`d
    /// **out** of the graph. The resource will no longer be available
    /// within the graph.
    ///
    /// ```rust
    /// use moongraph::*;
    /// use snafu::prelude::*;
    ///
    /// #[derive(Debug, Snafu)]
    /// enum TestError {}
    ///
    /// #[derive(Edges)]
    /// struct Input {
    ///     num_usize: View<usize>,
    ///     num_f32: ViewMut<f32>,
    ///     num_f64: Move<f64>,
    /// }
    ///
    /// // pack the graph with resources
    /// let mut graph = Graph::default()
    ///     .with_resource(0usize)
    ///     .with_resource(0.0f32)
    ///     .with_resource(0.0f64);
    ///
    /// // visit the graph, reading, modifying and _moving_!
    /// let num_usize = graph.visit(|mut input: Input| {
    ///     *input.num_f32 = 666.0;
    ///     *input.num_f64 += 10.0;
    ///     *input.num_usize
    /// }).unwrap();
    ///
    /// // observe we read usize
    /// assert_eq!(0, num_usize);
    /// assert_eq!(0, *graph.get_resource::<usize>().unwrap().unwrap());
    ///
    /// // observe we modified f32
    /// assert_eq!(666.0, *graph.get_resource::<f32>().unwrap().unwrap());
    ///
    /// // observe we moved f64 out of the graph and it is no longer present
    /// assert!(!graph.contains_resource::<f64>());
    pub fn visit<T: Edges, S>(&mut self, f: impl FnOnce(T) -> S) -> Result<S, GraphError> {
        let t = T::construct(&mut self.resources)?;
        let s = f(t);
        self.resources.unify().context(ResourceSnafu)?;
        Ok(s)
    }

    #[cfg(feature = "dot")]
    /// Save the graph to the filesystem as a dot file to be visualized with
    /// graphiz (or similar).
    pub fn save_graph_dot(&self, path: &str) {
        use dagga::dot::DagLegend;

        let legend =
            DagLegend::new(self.nodes()).with_resources_named(|ty: &TypeKey| ty.name().to_string());
        legend.save_to(path).unwrap();
    }
}

#[macro_export]
macro_rules! node {
    // add the nodes with the given idents if they don't already exist and add a constraint on the
    // first that the first must run before the second
    ($g:ident, $i:ident < $j:ident) => {{
        if let Some(node) = $g.get_node_mut(stringify!($i)) {
            node.add_runs_before(stringify!($j));
        } else {
            g.add_node(
                $i.into_node()
                    .with_name(stringify!($i))
                    .run_before(stringify!($j)),
            );
        }
        if !$g.contains_node(stringify!($j)) {
            g.add_node($j.into_node().with_name(stringify!($i)));
        }
    }};

    // add the nodes with the given idents if they don't already exist and add a constraint on the
    // first that the first must run after the second
    ($g:ident, $i:ident > $j:ident) => {{
        if let Some(node) = $g.get_node_mut(stringify!($i)) {
            node.add_runs_after(stringify!($j));
        } else {
            g.add_node(
                $i.into_node()
                    .with_name(stringify!($i))
                    .run_after(stringify!($j)),
            );
        }
        if !$g.contains_node(stringify!($j)) {
            g.add_node($j.into_node().with_name(stringify!($i)));
        }
    }};
    // add the node with the given ident, using the ident as its name
    ($g:ident, $i:ident) => {
        $g.add_node($i.into_node().with_name(stringify!($i)))
    };
}

#[allow(unused_macros)]
macro_rules! constraint_op {
    (>, $i:ident, $j:ident) => {
        $i.add_runs_after($j)
    };
    (<, $i:ident, $j:ident) => {
        $i.add_runs_before($j)
    };
    (,, $i:ident, $j:ident) => {};
}

#[allow(unused_macros)]
macro_rules! subgraph {
    ($i:ident $op:tt $($tail:tt)*) => {{
        let (mut g, _tail) = subgraph!($($tail)*);
        if let Some(_node) = g.get_node_mut(stringify!($i)) {
            constraint_op!($op, _node, _tail);
        } else {
            g.add_node({
                #[allow(unused_mut)]
                let mut node = $i.into_node().with_name(stringify!($i));
                constraint_op!($op, node, _tail);
                node
            });
        }
        (g, stringify!($i))
    }};

    ($i:ident$(,)?) => {
        (Graph::default().with_node($i.into_node().with_name(stringify!($i))), stringify!($i))
    }
}

#[macro_export]
macro_rules! graph {
    ($($t:tt)*) => {{
        subgraph!($($t)*).0
    }}
}

#[cfg(test)]
mod test {
    use super::*;

    fn create(_: ()) -> Result<(usize,), GraphError> {
        Ok((0,))
    }

    fn edit((mut num,): (ViewMut<usize>,)) -> Result<(), GraphError> {
        *num += 1;
        Ok(())
    }

    fn finish((num,): (Move<usize>,)) -> Result<(), GraphError> {
        assert_eq!(1, num.into(), "edit did not run");
        Ok(())
    }

    #[test]
    fn function_to_node() {
        // sanity test
        let mut graph = Graph::default()
            .with_function("create", create)
            .with_function("edit", edit);
        graph.run().unwrap();
        assert_eq!(1, *graph.get_resource::<usize>().unwrap().unwrap());

        let mut graph = graph.with_function("finish", finish);
        graph.run().unwrap();
        assert!(graph.get_resource::<usize>().unwrap().is_none());
    }

    #[test]
    fn many_inputs_many_outputs() {
        // tests our Edges and NodeResults impl macros
        fn start(_: ()) -> Result<(usize, u32, f32, f64, &'static str, String), GraphError> {
            Ok((0, 0, 0.0, 0.0, "hello", "HELLO".into()))
        }

        fn modify_ints(
            (mut numusize, mut numu32): (ViewMut<usize>, ViewMut<u32>),
        ) -> Result<(), GraphError> {
            *numusize += 1;
            *numu32 += 1;
            Ok(())
        }

        fn modify_floats(
            (mut numf32, mut numf64): (ViewMut<f32>, ViewMut<f64>),
        ) -> Result<(), GraphError> {
            *numf32 += 10.0;
            *numf64 += 10.0;
            Ok(())
        }

        fn modify_strings(
            (mut strstatic, mut strowned): (ViewMut<&'static str>, ViewMut<String>),
        ) -> Result<(), GraphError> {
            *strstatic = "goodbye";
            *strowned = "GOODBYE".into();
            Ok(())
        }

        fn end(
            (nusize, nu32, nf32, nf64, sstatic, sowned): (
                Move<usize>,
                Move<u32>,
                Move<f32>,
                Move<f64>,
                Move<&'static str>,
                Move<String>,
            ),
        ) -> Result<(bool,), GraphError> {
            assert_eq!(1, *nusize);
            assert_eq!(1, *nu32);
            assert_eq!(10.0, *nf32);
            assert_eq!(10.0, *nf64);
            assert_eq!("goodbye", *sstatic);
            assert_eq!("GOODBYE", *sowned);
            Ok((true,))
        }

        let mut graph = Graph::default()
            .with_function("start", start)
            .with_function("modify_ints", modify_ints)
            .with_function("modify_floats", modify_floats)
            .with_function("modify_strings", modify_strings)
            .with_function("end", end);

        graph.reschedule().unwrap();
        let schedule = graph.get_schedule();
        assert_eq!(
            vec![
                vec!["start"],
                vec!["modify_strings", "modify_floats", "modify_ints"],
                vec!["end"]
            ],
            schedule,
            "schedule is wrong"
        );

        graph.run().unwrap();
        let run_was_all_good = graph.get_resource::<bool>().unwrap().unwrap();
        assert!(run_was_all_good, "run was not all good");
    }

    #[test]
    fn can_derive() {
        use crate as moongraph;

        #[derive(Debug, Snafu)]
        enum TestError {}

        #[derive(Edges)]
        struct Input {
            num_usize: View<usize>,
            num_f32: ViewMut<f32>,
            num_f64: Move<f64>,
        }

        type Output = (String, &'static str);

        fn start(_: ()) -> Result<(usize, f32, f64), TestError> {
            Ok((1, 0.0, 10.0))
        }

        fn end(mut input: Input) -> Result<Output, TestError> {
            *input.num_f32 += *input.num_f64 as f32;
            Ok((
                format!("{},{},{}", *input.num_usize, *input.num_f32, *input.num_f64),
                "done",
            ))
        }

        let mut graph = Graph::default()
            .with_function("start", start)
            .with_function("end", end);
        graph.run().unwrap();
        assert_eq!(
            "1,10,10",
            graph.get_resource::<String>().unwrap().unwrap().as_str()
        );
    }

    #[test]
    fn can_visit_and_then_borrow() {
        use crate as moongraph;

        #[derive(Debug, Snafu)]
        enum TestError {}

        #[derive(Edges)]
        struct Input {
            num_usize: View<usize>,
            num_f32: ViewMut<f32>,
            num_f64: Move<f64>,
        }

        let mut graph = Graph::default()
            .with_resource(0usize)
            .with_resource(0.0f32)
            .with_resource(0.0f64);
        let num_usize = graph
            .visit(|mut input: Input| {
                *input.num_f32 = 666.0;
                *input.num_f64 += 10.0;
                *input.num_usize
            })
            .unwrap();
        assert_eq!(0, num_usize);
        assert_eq!(0, *graph.get_resource::<usize>().unwrap().unwrap());
        assert_eq!(666.0, *graph.get_resource::<f32>().unwrap().unwrap());
        assert!(!graph.contains_resource::<f64>());
    }

    #[cfg(feature = "none")]
    #[test]
    fn can_run_local() {
        fn start(_: ()) -> Result<(usize, u32, f32, f64, &'static str, String), GraphError> {
            Ok((0, 0, 0.0, 0.0, "hello", "HELLO".into()))
        }

        fn modify_ints(
            (mut numusize, mut numu32): (ViewMut<usize>, ViewMut<u32>),
        ) -> Result<(), GraphError> {
            *numusize += 1;
            *numu32 += 1;
            Ok(())
        }

        fn modify_floats(
            (mut numf32, mut numf64): (ViewMut<f32>, ViewMut<f64>),
        ) -> Result<(), GraphError> {
            *numf32 += 10.0;
            *numf64 += 10.0;
            Ok(())
        }

        fn modify_strings(
            (mut strstatic, mut strowned): (ViewMut<&'static str>, ViewMut<String>),
        ) -> Result<(), GraphError> {
            *strstatic = "goodbye";
            *strowned = "GOODBYE".into();
            Ok(())
        }

        fn end(
            (nusize, nu32, nf32, nf64, sstatic, sowned): (
                Move<usize>,
                Move<u32>,
                Move<f32>,
                Move<f64>,
                Move<&'static str>,
                Move<String>,
            ),
        ) -> Result<(bool,), GraphError> {
            assert_eq!(1, *nusize);
            assert_eq!(10, *nu32);
            assert_eq!(100.0, *nf32);
            assert_eq!(10.0, *nf64);
            assert_eq!("goodbye", *sstatic);
            assert_eq!("GOODBYE", *sowned);
            Ok((true,))
        }

        let mut graph = graph!(start, modify_ints, modify_floats, modify_strings, end,)
            .with_local::<(ViewMut<u32>, ViewMut<f32>), ()>("local");

        graph.reschedule().unwrap();
        assert_eq!(
            vec![
                vec!["start"],
                vec!["modify_strings", "modify_floats", "modify_ints"],
                vec!["local"],
                vec!["end"]
            ],
            graph.get_schedule(),
            "schedule is wrong"
        );

        let mut my_num = 0.0;
        graph
            .run_with_local(
                |(mut nu32, mut nf32): (ViewMut<u32>, ViewMut<f32>)| -> Result<(), String> {
                    *nu32 *= 10;
                    *nf32 *= 10.0;
                    my_num = *nu32 as f32 + *nf32;
                    Ok(())
                },
            )
            .unwrap();
        let run_was_all_good = graph.get_resource::<bool>().unwrap().unwrap();
        assert!(run_was_all_good, "run was not all good");
        assert_eq!(110.0, my_num, "local did not run");
    }

    #[test]
    fn can_use_graph_macro() {
        fn one(_: ()) -> Result<(), GraphError> {
            log::trace!("one");
            Ok(())
        }
        fn two(mut an_f32: ViewMut<f32>) -> Result<(), GraphError> {
            log::trace!("two");
            *an_f32 += 1.0;
            Ok(())
        }
        fn three(_: ()) -> Result<(), GraphError> {
            log::trace!("three");
            Ok(())
        }

        let _a = graph!(one < two, three, three > two);
        let _b = graph!(one, two);
        let _c = graph!(one < two);
        let _d = graph!(one);
        let _e = graph!(one < two < three);
        let mut g = graph!(one < two < three).with_resource(0.0f32);
        g.reschedule().unwrap();
        let schedule = g.get_schedule();
        assert_eq!(vec![vec!["one"], vec!["two"], vec!["three"]], schedule);
    }

}
