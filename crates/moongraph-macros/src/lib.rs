//! Provides derive macros for `moongraph::Edges`.
use syn::DeriveInput;

/// Macro for deriving structs that encode a node's edges/resource usage.
///
/// This is the quickest way to get a node up and running that uses a struct as
/// its edges. Simply add `#[derive(Edges)]` on your own structs if each of the
/// fields is one of `View`, `ViewMut` or `Move`.
///
/// ## Example
///
/// ```rust
/// use moongraph::{Edges, ViewMut, View, Move};
///
/// #[derive(Edges)]
/// struct MyData {
///     an_f32: ViewMut<f32>,
///     a_u32: View<u32>,
///     a_str: Move<&'static str>,
/// }
/// ```
#[proc_macro_derive(Edges, attributes(moongraph))]
pub fn derive_edges(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input: DeriveInput = syn::parse_macro_input!(input);
    let maybe_path = match moongraph_macros_syntax::find_path("moongraph", &input) {
        Err(e) => return e.into_compile_error().into(),
        Ok(maybe_path) => maybe_path,
    };
    let path = maybe_path.unwrap_or_else(|| {
        // UNWRAP: safe because we know this will parse
        let path: syn::Path = syn::parse_str("moongraph").unwrap();
        path
    });
    moongraph_macros_syntax::derive_edges(input, path).into()
}
