//! DAG scheduling, resource management, and execution.
//!
//! In `moongraph`, nodes are functions with parameters that are accessed
//! immutably, mutably or by move.
//!
//! `moongraph` validates and schedules nodes to run in parallel where possible,
//! using `rayon` as the underlying parallelizing tech.

use std::{
    any::Any,
    collections::HashMap,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use broomdog::{Loan, LoanMut};
use dagga::Dag;
use snafu::prelude::*;

pub use broomdog::{BroomdogErr, TypeKey, TypeMap};
pub use dagga::{DaggaError, Node};
#[cfg(feature = "derive")]
pub use moongraph_macros::Edges;

#[cfg(feature = "tutorial")]
mod tutorial_impl;

#[cfg(feature = "tutorial")]
pub use tutorial_impl::tutorial;

#[cfg(feature = "parallel")]
pub mod rayon_impl;

/// All errors.
#[derive(Debug, Snafu)]
pub enum GraphError {
    #[snafu(display("Error while running local node {error}"))]
    RunningLocalNode { error: String },

    #[snafu(display("Error scheduling the graph: {source}"))]
    Scheduling { source: dagga::DaggaError },

    #[snafu(display("Resource error: {source}"))]
    Resource { source: broomdog::BroomdogErr },

    #[snafu(display("Resource '{type_name}' is loaned"))]
    ResourceLoaned { type_name: &'static str },

    #[snafu(display("Missing resource '{name}'"))]
    Missing { name: &'static str },

    #[snafu(display("Encountered local function that was not provided or already run"))]
    MissingLocal,

    #[snafu(display("Node should be trimmed"))]
    TrimNode,

    #[snafu(display("Unrecoverable error while running node: {source}"))]
    Other {
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },
}

impl From<broomdog::BroomdogErr> for GraphError {
    fn from(source: broomdog::BroomdogErr) -> Self {
        GraphError::Resource { source }
    }
}

impl GraphError {
    pub fn other(err: impl std::error::Error + Send + Sync + 'static) -> Self {
        GraphError::Other {
            source: Box::new(err),
        }
    }
}

/// Returns a result meaning everything is ok and the node should run again next frame.
pub fn ok() -> Result<(), GraphError> {
    Ok(())
}

/// Returns a result meaning everything is ok, but the node should be removed from the graph.
pub fn end() -> Result<(), GraphError> {
    Err(GraphError::TrimNode)
}

/// Returns a result meaning an error occured and the graph cannot recover.
pub fn err(err: impl std::error::Error + Send + Sync + 'static) -> Result<(), GraphError> {
    Err(GraphError::other(err))
}

pub type Resource = Box<dyn Any + Send + Sync>;
pub type FnPrepare = dyn Fn(&mut TypeMap) -> Result<Resource, GraphError>;
pub type FnMutRun = dyn FnMut(Resource) -> Result<Resource, GraphError> + Send + Sync;
pub type FnSave = dyn Fn(Resource, &mut TypeMap) -> Result<(), GraphError>;

/// A function wrapper.
///
/// Wraps a function by moving it into a closure. Before running, the parameters
/// of the function are constructed from a TypeMap of resources. The results
/// of the function are packed back into the same TypeMap.
pub struct Function {
    prepare: Box<FnPrepare>,
    run: Option<Box<FnMutRun>>,
    save: Box<FnSave>,
}

impl Function {
    /// Run the function using the given `TypeMap`.
    pub fn run(
        &mut self,
        resources: Resource,
        local: &mut Option<impl FnOnce(Resource) -> Result<Resource, GraphError>>,
    ) -> Result<Resource, GraphError> {
        if let Some(f) = self.run.as_mut() {
            (f)(resources)
        } else {
            let local = local.take().context(MissingLocalSnafu)?;
            (local)(resources)
        }
    }

    /// Create a new function.
    pub fn new(
        prepare: impl Fn(&mut TypeMap) -> Result<Resource, GraphError> + 'static,
        run: impl Fn(Resource) -> Result<Resource, GraphError> + Send + Sync + 'static,
        save: impl Fn(Resource, &mut TypeMap) -> Result<(), GraphError> + 'static,
    ) -> Self {
        Function {
            prepare: Box::new(prepare),
            run: Some(Box::new(run)),
            save: Box::new(save),
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
///
/// There exists a derive macro [`Edges`](derive@Edges) to help implementing
/// this trait.
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
///   wrapped in [`View`].
/// * Write one or more resources by having a field in the input parameter
///   wrapped in [`ViewMut`].
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
        F: FnMut(Input) -> Result<Output, GraphError> + Send + Sync + 'static,
    > IsGraphNode<Input, Output> for F
{
    fn into_node(mut self) -> Node<Function, TypeKey> {
        let prepare = Box::new(prepare::<Input>);
        let save = Box::new(save::<Output>);

        let inner = Box::new(move |resources: Resource| -> Result<Resource, GraphError> {
            let input = *resources.downcast::<Input>().unwrap();
            match (self)(input) {
                Ok(creates) => Ok(Box::new(creates)),
                Err(e) => Err(e),
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
        match inner_loan.into_owned(key.name()) {
            Ok(value) => {
                // UNWRAP: safe because we got this out as `T`
                let box_t = value.downcast::<T>().unwrap();
                Ok(Move { inner: *box_t })
            }
            Err(loan) => {
                // We really do **not** want to lose any resources
                resources.insert(key, loan);
                let err = ResourceLoanedSnafu {
                    type_name: std::any::type_name::<T>(),
                }
                .build();
                log::error!("{err}");
                Err(err)
            }
        }
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

impl<T: std::fmt::Display> std::fmt::Display for Move<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

/// Used to generate a default value of a resource, if possible.
pub trait Gen<T> {
    fn generate() -> Option<T>;
}

/// Valueless type that represents the ability to generate a resource by
/// default.
pub struct SomeDefault;

impl<T: Default> Gen<T> for SomeDefault {
    fn generate() -> Option<T> {
        Some(T::default())
    }
}

/// Valueless type that represents the **inability** to generate a resource by default.
pub struct NoDefault;

impl<T> Gen<T> for NoDefault {
    fn generate() -> Option<T> {
        None
    }
}

/// Immutably borrowed resource that _may_ be created by default.
///
/// Node functions wrap their parameters in [`View`], [`ViewMut`] or [`Move`].
///
/// `View` has two type parameters:
/// * `T` - The type of the resource.
/// * `G` - The method by which the resource can be generated if it doesn't
///   already exist. By default this is [`SomeDefault`], which denotes creating the
///   resource using its default instance. Another option is [`NoDefault`] which
///   fails to generate the resource.
///
/// ```rust
/// use moongraph::*;
///
/// let mut graph = Graph::default();
/// let default_number = graph.visit(|u: View<usize>| { *u }).map_err(|e| e.to_string());
/// assert_eq!(Ok(0), default_number);
///
/// let no_number = graph.visit(|f: View<f32, NoDefault>| *f);
/// assert!(no_number.is_err());
/// ```
pub struct View<T, G: Gen<T> = SomeDefault> {
    inner: Loan,
    _phantom: PhantomData<(T, G)>,
}

impl<T: Any + Send + Sync, G: Gen<T>> Deref for View<T, G> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // UNWRAP: safe because it was constructed with `T`
        self.inner.downcast_ref().unwrap()
    }
}

impl<T: Any + Send + Sync, G: Gen<T>> Edges for View<T, G> {
    fn reads() -> Vec<TypeKey> {
        vec![TypeKey::new::<T>()]
    }

    fn construct(resources: &mut TypeMap) -> Result<Self, GraphError> {
        let key = TypeKey::new::<T>();
        let inner = match resources.loan(key).context(ResourceSnafu)? {
            Some(inner) => inner,
            None => {
                let t = G::generate().context(MissingSnafu {
                    name: std::any::type_name::<T>(),
                })?;
                // UNWRAP: safe because we know this type was missing, and no other type
                // is stored with this type's type id.
                let _ = resources.insert_value(t).unwrap();
                log::trace!("generated missing {}", std::any::type_name::<T>());
                // UNWRAP: safe because we just inserted
                resources.loan(key).unwrap().unwrap()
            }
        };
        Ok(View {
            inner,
            _phantom: PhantomData,
        })
    }
}

impl<T: std::fmt::Display + Any + Send + Sync, G: Gen<T>> std::fmt::Display for View<T, G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let t: &T = self.inner.downcast_ref().unwrap();
        t.fmt(f)
    }
}

impl<'a, T: Send + Sync + 'static, G: Gen<T>> IntoIterator for &'a View<T, G>
where
    &'a T: IntoIterator,
{
    type Item = <<&'a T as IntoIterator>::IntoIter as Iterator>::Item;

    type IntoIter = <&'a T as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.deref().into_iter()
    }
}

/// A mutably borrowed resource that may be created by default.
///
/// Node functions wrap their parameters in [`View`], [`ViewMut`] or [`Move`].
///
/// `ViewMut` has two type parameters:
/// * `T` - The type of the resource.
/// * `G` - The method by which the resource can be generated if it doesn't
///   already exist. By default this is [`SomeDefault`], which denotes creating
///   the resource using its default implementation. Another option is
///   [`NoDefault`] which fails to generate the resource.
///
/// ```rust
/// use moongraph::*;
///
/// let mut graph = Graph::default();
/// let default_number = graph.visit(|u: ViewMut<usize>| { *u }).map_err(|e| e.to_string());
/// assert_eq!(Ok(0), default_number);
///
/// let no_number = graph.visit(|f: ViewMut<f32, NoDefault>| *f);
/// assert!(no_number.is_err());
/// ```
pub struct ViewMut<T, G: Gen<T> = SomeDefault> {
    inner: LoanMut,
    _phantom: PhantomData<(T, G)>,
}

impl<T: Any + Send + Sync, G: Gen<T>> Deref for ViewMut<T, G> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // UNWRAP: safe because it was constructed with `T`
        self.inner.downcast_ref().unwrap()
    }
}

impl<T: Any + Send + Sync, G: Gen<T>> DerefMut for ViewMut<T, G> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // UNWRAP: safe because it was constructed with `T`
        self.inner.downcast_mut().unwrap()
    }
}

impl<'a, T: Any + Send + Sync, G: Gen<T>> Edges for ViewMut<T, G> {
    fn writes() -> Vec<TypeKey> {
        vec![TypeKey::new::<T>()]
    }

    fn construct(resources: &mut TypeMap) -> Result<Self, GraphError> {
        let key = TypeKey::new::<T>();
        let inner = match resources.loan_mut(key).context(ResourceSnafu)? {
            Some(inner) => inner,
            None => {
                let t = G::generate().context(MissingSnafu {
                    name: std::any::type_name::<T>(),
                })?;
                // UNWRAP: safe because we know this type was missing, and no other type
                // is stored with this type's type id.
                let _ = resources.insert_value(t).unwrap();
                log::trace!("generated missing {}", std::any::type_name::<T>());
                // UNWRAP: safe because we just inserted
                resources.loan_mut(key).unwrap().unwrap()
            }
        };
        Ok(ViewMut {
            inner,
            _phantom: PhantomData,
        })
    }
}

impl<T: std::fmt::Display + Any + Send + Sync, G: Gen<T>> std::fmt::Display for ViewMut<T, G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let t: &T = self.inner.downcast_ref().unwrap();
        t.fmt(f)
    }
}

impl<'a, T: Send + Sync + 'static, G: Gen<T>> IntoIterator for &'a ViewMut<T, G>
where
    &'a T: IntoIterator,
{
    type Item = <<&'a T as IntoIterator>::IntoIter as Iterator>::Item;

    type IntoIter = <&'a T as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.deref().into_iter()
    }
}

/// Contains the nodes/functions and specifies their execution order.
#[derive(Default)]
pub struct Execution {
    barrier: usize,
    unscheduled: Vec<Node<Function, TypeKey>>,
    schedule: Vec<Vec<Node<Function, TypeKey>>>,
}

impl Execution {
    /// Returns the number of nodes.
    pub fn len(&self) -> usize {
        self.unscheduled.len() + self.schedule.iter().map(|batch| batch.len()).sum::<usize>()
    }
}

pub struct BatchResult<'graph> {
    nodes: &'graph mut Vec<Node<Function, TypeKey>>,
    resources: &'graph mut TypeMap,
    results: Vec<Result<Resource, GraphError>>,
}

impl<'graph> BatchResult<'graph> {
    /// Save the results of the batch run to the graph.
    ///
    /// Optionally trim any nodes that report a [`GraphError::TrimNode`] result.
    ///
    /// Optionally unifies resources.
    ///
    /// Returns `true` if any nodes were trimmed.
    pub fn save(
        self,
        should_trim_nodes: bool,
        should_unify_resources: bool,
    ) -> Result<bool, GraphError> {
        let BatchResult {
            nodes,
            resources,
            results,
        } = self;
        let mut trimmed_any = false;
        let mut trimmings = vec![false; nodes.len()];
        for ((node, output), should_trim) in nodes.iter().zip(results).zip(trimmings.iter_mut()) {
            match output {
                Err(GraphError::TrimNode) => {
                    // TrimNode is special in that it is not really an error.
                    // Instead it means that the system which returned TrimNode should be
                    // removed from the graph because its work is done.
                    *should_trim = true;
                    trimmed_any = should_trim_nodes;
                }
                Err(o) => {
                    // A system hit an unrecoverable error.
                    log::error!("node '{}' erred: {}", node.name(), o);
                    return Err(o);
                }
                Ok(output) => (node.inner().save)(output, resources)?,
            }
        }

        if trimmed_any {
            let mut n = 0;
            nodes.retain_mut(|_| {
                let should_trim = trimmings[n];
                n += 1;
                !should_trim
            });
        }

        if should_unify_resources {
            resources.unify().context(ResourceSnafu)?;
        }

        Ok(trimmed_any)
    }
}

pub struct Batch<'graph> {
    nodes: &'graph mut Vec<Node<Function, TypeKey>>,
    resources: &'graph mut TypeMap,
    //inputs: Vec<Resource>,
    //runs: Vec<&'graph Box<dyn Fn(Resource) -> Result<Resource, GraphError> + Send + Sync>>,
    // local: Option<(
    //     Resource,
    //     Box<dyn FnOnce(Resource) -> Result<Resource, GraphError> + 'local>,
    // )>,
}

impl<'a> Batch<'a> {
    /// Create a new [`Batch`] with a local function.
    pub fn new(resources: &'a mut TypeMap, nodes: &'a mut Vec<Node<Function, TypeKey>>) -> Self {
        Batch { resources, nodes }
    }

    #[cfg(feature = "parallel")]
    pub fn run(
        self,
        local: &mut Option<impl FnOnce(Resource) -> Result<Resource, GraphError>>,
    ) -> Result<BatchResult<'a>, GraphError> {
        use rayon::prelude::*;

        let Batch { nodes, resources } = self;

        let mut local_f = None;
        let mut inputs = vec![];
        let mut runs = vec![];
        for node in nodes.iter_mut() {
            let input = (node.inner().prepare)(resources)?;
            if let Some(f) = node.inner_mut().run.as_mut() {
                inputs.push(input);
                runs.push(f);
            } else {
                let f = local.take().context(MissingLocalSnafu)?;
                local_f = Some((
                    input,
                    Box::new(f) as Box<dyn FnOnce(Resource) -> Result<Resource, GraphError>>,
                ));
            }
        }

        let mut results = inputs
            .into_par_iter()
            .zip(runs.into_par_iter())
            .map(|(input, f)| (f)(input))
            .collect::<Vec<_>>();

        if let Some((input, f)) = local_f {
            results.push((f)(input));
        }

        Ok(BatchResult {
            nodes,
            results,
            resources,
        })
    }

    #[cfg(not(feature = "parallel"))]
    pub fn run(
        self,
        local: &mut Option<impl FnOnce(Resource) -> Result<Resource, GraphError>>,
    ) -> Result<BatchResult<'a>, GraphError> {
        let Batch { nodes, resources } = self;

        let mut local_f = None;
        let mut inputs = vec![];
        let mut runs = vec![];
        for node in nodes.iter_mut() {
            let input = (node.inner().prepare)(resources)?;
            if let Some(f) = node.inner_mut().run.as_mut() {
                inputs.push(input);
                runs.push(f);
            } else {
                let f = local.take().context(MissingLocalSnafu)?;
                local_f = Some((
                    input,
                    Box::new(f) as Box<dyn FnOnce(Resource) -> Result<Resource, GraphError>>,
                ));
            }
        }

        let mut outputs = inputs
            .into_iter()
            .zip(runs.into_iter())
            .map(|(input, f)| (f)(input))
            .collect::<Vec<_>>();

        if let Some((input, f)) = local_f {
            outputs.push((f)(input));
        }

        Ok(BatchResult {
            nodes,
            results: outputs,
            resources,
        })
    }
}

/// Provides access to consecutive batches of scheduled nodes/functions.
pub struct Batches<'graph> {
    schedule: std::slice::IterMut<'graph, Vec<Node<Function, TypeKey>>>,
    resources: &'graph mut TypeMap,
}

impl<'graph> Batches<'graph> {
    /// Overwrite the batch's resources, allowing the schedule to operate on a separate
    /// set of resources.
    pub fn set_resources(&mut self, resources: &'graph mut TypeMap) {
        self.resources = resources;
    }

    pub fn next_batch(&mut self) -> Option<Batch> {
        let nodes: &'graph mut Vec<_> = self.schedule.next()?;
        let batch = Batch::new(self.resources, nodes);
        Some(batch)
    }

    /// Attempt to unify resources, returning `true` when unification was successful
    /// or `false` when resources are still loaned.
    pub fn unify(&mut self) -> bool {
        self.resources.unify().is_ok()
    }

    /// Return the number of batches remaining in the schedule.
    pub fn len(&self) -> usize {
        self.schedule.len()
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
    execution: Execution,
}

impl Graph {
    pub fn _resources_mut(&mut self) -> &mut TypeMap {
        &mut self.resources
    }

    /// Creates a graph node from an [`Fn`] closure.
    ///
    /// A node in the graph is a boxed Rust closure that may do any or all the
    /// following:
    ///
    /// * Create resources by returning a result that implements
    ///   [`NodeResults`].
    /// * Consume one or more resources by having a field in the input parameter
    ///   wrapped in [`Move`]. The resource will not be available in the graph
    ///   after the node is run.
    /// * Read one or more resources by having a field in the input parameter
    ///   wrapped in [`View`].
    /// * Write one or more resources by having a field in the input parameter
    ///   wrapped in [`ViewMut`].
    ///
    /// By default `IsGraphNode` is implemented for functions that take one
    /// parameter implementing [`Edges`] and returning a `Result` where the "ok"
    /// type implements `NodeResults`.
    pub fn node<Input, Output, F: IsGraphNode<Input, Output>>(f: F) -> Node<Function, TypeKey> {
        f.into_node()
    }

    /// Creates a graph node without a closure, to be supplied later with
    /// [`Graph::run_with_local`].
    ///
    /// The returned node may be added to a graph and scheduled, allowing
    /// closures with local scope requirements to fit into the graph.
    ///
    /// At this time only one local node is allowed.
    pub fn local<Input, Output>() -> Node<Function, TypeKey>
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

    #[deprecated(
        since = "0.3.3",
        note = "Ambiguous name. Replaced by `interleave_subgraph` and `add_subgraph`. Use \
                Graph::interleave_subgraph instead as a direct replacment."
    )]
    /// Merge two graphs, preferring the right in cases of key collisions.
    ///
    /// The values of `rhs` will override those of `lhs`.
    pub fn merge(mut lhs: Graph, rhs: Graph) -> Graph {
        lhs.interleave_subgraph(rhs);
        lhs
    }

    /// Add a subgraph, preferring the right in cases of key collisions.
    ///
    /// The values of `rhs` will override those of `self`.
    ///
    /// Barriers in each graph will be considered equal. This has the effect
    /// that after adding the subgraph, nodes in `rhs` may run at the same
    /// time as nodes in `lhs` if their barrier matches.
    ///
    /// This is analogous to adding each graph's nodes in an interleaving order,
    /// sorted by their barrier.
    ///
    /// ## Example:
    /// ```rust
    /// use moongraph::{graph, Graph, GraphError, ViewMut};
    ///
    /// fn one(_: ()) -> Result<(), GraphError> {
    ///     log::trace!("one");
    ///     Ok(())
    /// }
    /// fn two(mut an_f32: ViewMut<f32>) -> Result<(), GraphError> {
    ///     log::trace!("two");
    ///     *an_f32 += 1.0;
    ///     Ok(())
    /// }
    /// fn three(_: ()) -> Result<(), GraphError> {
    ///     log::trace!("three");
    ///     Ok(())
    /// }
    /// fn four(_: ()) -> Result<(), GraphError> {
    ///     log::trace!("four");
    ///     Ok(())
    /// }
    ///
    /// let mut one_two = graph!(one < two).with_barrier();
    /// assert_eq!(1, one_two.get_barrier());
    /// let three_four = graph!(three < four);
    /// assert_eq!(0, three_four.get_barrier());
    /// one_two.interleave_subgraph(three_four);
    /// one_two.reschedule().unwrap();
    /// assert_eq!(
    ///     vec![vec!["one", "three"], vec!["four", "two"]],
    ///     one_two.get_schedule()
    /// );
    /// ```
    pub fn interleave_subgraph(&mut self, mut rhs: Graph) -> &mut Self {
        self.unschedule();
        rhs.unschedule();
        let Graph {
            resources: mut rhs_resources,
            execution:
                Execution {
                    unscheduled: rhs_nodes,
                    barrier: _,
                    schedule: _,
                },
        } = rhs;
        self.resources
            .extend(std::mem::take(rhs_resources.deref_mut()).into_iter());
        let mut unscheduled: HashMap<String, Node<Function, TypeKey>> = HashMap::default();
        let lhs_nodes = std::mem::take(&mut self.execution.unscheduled);
        unscheduled.extend(
            lhs_nodes
                .into_iter()
                .map(|node| (node.name().to_string(), node)),
        );
        unscheduled.extend(
            rhs_nodes
                .into_iter()
                .map(|node| (node.name().to_string(), node)),
        );
        self.execution.unscheduled = unscheduled.into_iter().map(|v| v.1).collect();
        self.execution.barrier = self.execution.barrier.max(rhs.execution.barrier);
        self
    }

    /// Add a subgraph, preferring the right in cases of key collisions.
    ///
    /// The values of `rhs` will override those of `self`.
    ///
    /// Barriers will be kept in place, though barriers in `rhs` will be
    /// incremented by `self.barrier`. This has the effect that after adding the
    /// subgraph, nodes in `rhs` will run after the last barrier in `self`,
    /// or later if `rhs` has barriers of its own.
    ///
    /// This is analogous to adding all of the nodes in `rhs`, one by one, to
    /// `self` - while keeping the constraints of `rhs` in place.
    ///
    /// ## Example:
    /// ```rust
    /// use moongraph::{graph, Graph, GraphError, ViewMut};
    ///
    /// fn one(_: ()) -> Result<(), GraphError> {
    ///     log::trace!("one");
    ///     Ok(())
    /// }
    /// fn two(mut an_f32: ViewMut<f32>) -> Result<(), GraphError> {
    ///     log::trace!("two");
    ///     *an_f32 += 1.0;
    ///     Ok(())
    /// }
    /// fn three(_: ()) -> Result<(), GraphError> {
    ///     log::trace!("three");
    ///     Ok(())
    /// }
    /// fn four(_: ()) -> Result<(), GraphError> {
    ///     log::trace!("four");
    ///     Ok(())
    /// }
    ///
    /// let mut one_two = graph!(one < two).with_barrier();
    /// assert_eq!(1, one_two.get_barrier());
    /// let three_four = graph!(three < four);
    /// assert_eq!(0, three_four.get_barrier());
    /// one_two.add_subgraph(three_four);
    /// one_two.reschedule().unwrap();
    /// assert_eq!(
    ///     vec![vec!["one"], vec!["two"], vec!["three"], vec!["four"]],
    ///     one_two.get_schedule()
    /// );
    /// ```
    pub fn add_subgraph(&mut self, mut rhs: Graph) -> &mut Self {
        self.unschedule();
        rhs.unschedule();
        let Graph {
            resources: mut rhs_resources,
            execution:
                Execution {
                    unscheduled: rhs_nodes,
                    barrier: rhs_barrier,
                    schedule: _,
                },
        } = rhs;
        let base_barrier = self.execution.barrier;
        self.execution.barrier = base_barrier + rhs_barrier;
        self.resources
            .extend(std::mem::take(rhs_resources.deref_mut()).into_iter());
        self.execution
            .unscheduled
            .extend(rhs_nodes.into_iter().map(|node| {
                let barrier = node.get_barrier();
                node.with_barrier(base_barrier + barrier)
            }));

        self
    }

    /// Proxy for [`Graph::add_subgraph`] with `self` chaining.
    pub fn with_subgraph(mut self, rhs: Graph) -> Self {
        self.add_subgraph(rhs);
        self
    }

    /// Unschedule all functions.
    fn unschedule(&mut self) {
        self.execution.unscheduled.extend(
            std::mem::take(&mut self.execution.schedule)
                .into_iter()
                .flatten(),
        );
    }

    /// Reschedule all functions.
    ///
    /// If the functions were already scheduled this will unscheduled them
    /// first.
    pub fn reschedule(&mut self) -> Result<(), GraphError> {
        log::trace!("rescheduling the render graph:");
        self.unschedule();
        let all_nodes = std::mem::take(&mut self.execution.unscheduled);
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
        self.execution.schedule = schedule.batches;
        // Order the nodes in each batch by node name so they are deterministic.
        for batch in self.execution.schedule.iter_mut() {
            batch.sort_by(|a, b| a.name().cmp(b.name()));
        }

        let batched_names = self.get_schedule_and_resources();
        log::trace!("{:#?}", batched_names);
        Ok(())
    }

    /// Return the names of scheduled nodes.
    ///
    /// If no nodes have been scheduled this will return an empty vector.
    ///
    /// Use [`Graph::reschedule`] to manually schedule the nodes before calling
    /// this.
    pub fn get_schedule(&self) -> Vec<Vec<&str>> {
        self.execution
            .schedule
            .iter()
            .map(|batch| batch.iter().map(|node| node.name()).collect())
            .collect()
    }

    /// Return the names of scheduled nodes along with the names of their resources.
    ///
    /// If no nodes have been scheduled this will return an empty vector.
    ///
    /// Use [`Graph::reschedule`] to manually schedule the nodes before calling
    /// this.
    pub fn get_schedule_and_resources(&self) -> Vec<Vec<(&str, Vec<&str>)>> {
        self.execution
            .schedule
            .iter()
            .map(|batch| {
                batch
                    .iter()
                    .map(|node| {
                        let name = node.name();
                        let inputs = node
                            .all_inputs()
                            .into_iter()
                            .map(|key| key.name())
                            .collect();
                        (name, inputs)
                    })
                    .collect()
            })
            .collect()
    }

    /// An iterator over all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &Node<Function, TypeKey>> {
        self.execution
            .schedule
            .iter()
            .flatten()
            .chain(self.execution.unscheduled.iter())
    }

    /// A mutable iterator over all nodes.
    pub fn nodes_mut(&mut self) -> impl Iterator<Item = &mut Node<Function, TypeKey>> {
        self.execution
            .schedule
            .iter_mut()
            .flatten()
            .chain(self.execution.unscheduled.iter_mut())
    }

    /// Add multiple nodes to this graph.
    pub fn with_nodes(self, nodes: impl IntoIterator<Item = Node<Function, TypeKey>>) -> Self {
        nodes.into_iter().fold(self, Self::with_node)
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: Node<Function, TypeKey>) {
        self.execution
            .unscheduled
            .push(node.runs_after_barrier(self.execution.barrier));
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
        for (i, node) in self.execution.unscheduled.iter().enumerate() {
            if node.name() == name.as_ref() {
                may_index = Some(i);
            }
        }
        if let Some(i) = may_index.take() {
            Some(self.execution.unscheduled.swap_remove(i))
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
        if self.execution.unscheduled.iter().any(search) {
            return true;
        }
        self.execution.schedule.iter().flatten().any(search)
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
        self.execution.barrier += 1;
    }

    /// Add a barrier to the graph.
    ///
    /// All nodes added after the barrier will run after nodes added before the
    /// barrier.
    pub fn with_barrier(mut self) -> Self {
        self.add_barrier();
        self
    }

    /// Return the current barrier.
    ///
    /// This will be the barrier for any added nodes.
    pub fn get_barrier(&self) -> usize {
        self.execution.barrier
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
        self.add_node(Self::local::<Input, Output>().with_name(name));
    }

    pub fn with_local<Input, Output>(mut self, name: impl Into<String>) -> Self
    where
        Input: Edges + Any + Send + Sync,
        Output: NodeResults + Any + Send + Sync,
    {
        self.add_local::<Input, Output>(name);
        self
    }

    /// Reschedule the graph **only if there are unscheduled nodes**.
    ///
    /// Returns an error if a schedule cannot be built.
    pub fn reschedule_if_necessary(&mut self) -> Result<(), GraphError> {
        if !self.execution.unscheduled.is_empty() {
            self.reschedule()?;
        }
        Ok(())
    }

    /// Return an iterator over prepared schedule-batches.
    ///
    /// The graph should be scheduled ahead of calling this function.
    pub fn batches<'graph>(&'graph mut self) -> Batches {
        Batches {
            schedule: self.execution.schedule.iter_mut(),
            resources: &mut self.resources,
        }
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
        let mut local = Some(Box::new(move |resources: Resource| {
            let input = *resources.downcast::<Input>().unwrap();
            match (f)(input) {
                Ok(creates) => Ok(Box::new(creates) as Resource),
                Err(e) => Err(GraphError::RunningLocalNode {
                    error: e.to_string(),
                }),
            }
        })
            as Box<dyn FnOnce(Resource) -> Result<Resource, GraphError>>);

        self.reschedule_if_necessary()?;

        let mut got_trimmed = false;
        let mut batches = self.batches();
        while let Some(batch) = batches.next_batch() {
            let batch_result = batch.run(&mut local)?;
            let did_trim_batch = batch_result.save(true, true)?;
            got_trimmed = got_trimmed || did_trim_batch;
        }
        if got_trimmed {
            self.reschedule()?;
        }

        Ok(())
    }

    /// Remove a resource from the graph.
    ///
    /// Returns an error if the requested resource is loaned, and cannot be removed.
    pub fn remove_resource<T: Any + Send + Sync>(&mut self) -> Result<Option<T>, GraphError> {
        let key = TypeKey::new::<T>();
        if let Some(inner_loan) = self.resources.remove(&key) {
            match inner_loan.into_owned(key.name()) {
                Ok(value) => {
                    // UNWRAP: safe because we got this out as `T`, and it can only be stored
                    // as `T`
                    let box_t = value.downcast::<T>().unwrap();
                    Ok(Some(*box_t))
                }
                Err(loan) => {
                    self.resources.insert(key, loan);
                    let err = ResourceLoanedSnafu {
                        type_name: std::any::type_name::<T>(),
                    }
                    .build();
                    log::error!("{err}");
                    Err(err)
                }
            }
        } else {
            // There is no such resource
            Ok(None)
        }
    }

    /// Get a reference to a resource in the graph.
    ///
    /// If the resource _does not_ exist `Ok(None)` will be returned.
    pub fn get_resource<T: Any + Send + Sync>(&self) -> Result<Option<&T>, GraphError> {
        Ok(self.resources.get_value().context(ResourceSnafu)?)
    }

    /// Get a mutable reference to a resource in the graph.
    ///
    /// If the resource _does not_ exist `Ok(None)` will be returned.
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

    /// Split the graph into an execution (schedule of functions/nodes) and resources.
    pub fn into_parts(self) -> (Execution, TypeMap) {
        (self.execution, self.resources)
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

    /// Internal function used in the [`graph!`] macro.
    pub fn _add_node_constraint(
        constraint: &str,
        i: &mut Node<Function, TypeKey>,
        j: Option<String>,
    ) {
        //
        match constraint {
            ">" => {
                i.add_runs_after(j.unwrap());
            }
            "<" => {
                i.add_runs_before(j.unwrap());
            }
            _ => {}
        }
    }

    pub fn _last_node(&self) -> Option<String> {
        self.execution
            .unscheduled
            .last()
            .map(|node| node.name().to_string())
    }

    pub fn node_len(&self) -> usize {
        self.execution.len()
    }

    /// Pop off the next batch of nodes from the schedule.
    ///
    /// The graph must be scheduled first.
    pub fn take_next_batch_of_nodes(&mut self) -> Option<Vec<Node<Function, TypeKey>>> {
        if self.execution.schedule.is_empty() {
            None
        } else {
            self.execution.schedule.drain(0..1).next()
        }
    }
}

/// Constructs a [`Graph`] using an intuitive shorthand for node ordering
/// relationships.
///
/// ## Example:
/// ```rust
/// # use moongraph::{Graph, graph, GraphError, ViewMut};
///
/// fn one(_: ()) -> Result<(), GraphError> {
///     log::trace!("one");
///     Ok(())
/// }
/// fn two(mut an_f32: ViewMut<f32>) -> Result<(), GraphError> {
///     log::trace!("two");
///     *an_f32 += 1.0;
///     Ok(())
/// }
/// fn three(_: ()) -> Result<(), GraphError> {
///     log::trace!("three");
///     Ok(())
/// }
///
/// let _a = graph!(one < two, three, three > two);
/// let _b = graph!(one, two);
/// let _c = graph!(one < two);
/// let _d = graph!(one);
/// let _e = graph!(one < two < three);
///
/// let mut g = graph!(one < two < three).with_resource(0.0f32);
/// g.reschedule().unwrap();
/// let schedule = g.get_schedule();
/// assert_eq!(vec![vec!["one"], vec!["two"], vec!["three"]], schedule);
/// ```
#[macro_export]
macro_rules! graph {
    ($i:ident $op:tt $($tail:tt)*) => {{
        let mut g = graph!($($tail)*);
        let tail = g._last_node();
        if let Some(node) = g.get_node_mut(stringify!($i)) {
            Graph::_add_node_constraint(stringify!($op), node, tail);
        } else {
            g.add_node({
                let mut node = Graph::node($i).with_name(stringify!($i));
                Graph::_add_node_constraint(stringify!($op), &mut node, tail);
                node
            });
        }
        g
    }};

    ($i:ident$(,)?) => {
        Graph::default().with_node(Graph::node($i).with_name(stringify!($i)))
    }
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
                vec!["modify_floats", "modify_ints", "modify_strings"],
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
        #[derive(Edges)]
        #[moongraph(crate = crate)]
        struct Input {
            num_usize: View<usize>,
            num_f32: ViewMut<f32>,
            num_f64: Move<f64>,
        }

        type Output = (String, &'static str);

        fn start(_: ()) -> Result<(usize, f32, f64), GraphError> {
            Ok((1, 0.0, 10.0))
        }

        fn end(mut input: Input) -> Result<Output, GraphError> {
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
    // Tests that Gen will generate a default for a missing resource,
    // and that the result will be stored in the graph.
    fn can_generate_view_default() {
        let mut graph = Graph::default();
        let u = graph.visit(|u: View<usize>| *u).unwrap();
        assert_eq!(0, u);

        let my_u = graph.get_resource::<usize>().unwrap();
        assert_eq!(Some(0), my_u.copied());
    }
}
