//! Provides derive macros for `moongraph::Edges` and `moongraph::NodeResults`.
use quote::quote;
use syn::{
    punctuated::Punctuated, token::Comma, Data, DataStruct, DeriveInput, Field, Fields,
    FieldsNamed, FieldsUnnamed, Ident, Type, WhereClause, WherePredicate,
};

fn collect_field_types(fields: &Punctuated<Field, Comma>) -> Vec<Type> {
    fields.iter().map(|x| x.ty.clone()).collect()
}

fn gen_identifiers(fields: &Punctuated<Field, Comma>) -> Vec<Ident> {
    fields.iter().map(|x| x.ident.clone().unwrap()).collect()
}

enum DataType {
    Struct,
    Tuple,
}

fn gen_edges_body(
    ast: &Data,
    name: &Ident,
    path: &syn::Path,
) -> (proc_macro2::TokenStream, Vec<Type>) {
    let (body, fields) = match *ast {
        Data::Struct(DataStruct {
            fields: Fields::Named(FieldsNamed { named: ref x, .. }),
            ..
        }) => (DataType::Struct, x),
        Data::Struct(DataStruct {
            fields: Fields::Unnamed(FieldsUnnamed { unnamed: ref x, .. }),
            ..
        }) => (DataType::Tuple, x),
        _ => panic!("Enums are not supported"),
    };

    let tys = collect_field_types(fields);

    let fetch_return = match body {
        DataType::Struct => {
            let identifiers = gen_identifiers(fields);

            quote! {
                #name {
                    #( #identifiers: #path::Edges::construct(resources)? ),*
                }
            }
        }
        DataType::Tuple => {
            let count = tys.len();
            let fetch = vec![quote! { #path::Edges::construct(resources)? }; count];

            quote! {
                #name ( #( #fetch ),* )
            }
        }
    };

    (fetch_return, tys)
}

/// Macro for deriving structs that encode a node's edges/resource usage.
///
/// This is the quickest way to get a node up and running that uses a struct as
/// its edges. Simply add `#[derive(Edges)]` on your own structs if each of the
/// fields is one of `View`, `ViewMut` or `Move`.
///
/// ## Note:
/// For this to work, `Edges`, `GraphError`, `TypeMap` and `TypeKey` must all be
/// in scope.
///
/// ## Example
/// ```rust
/// use moongraph::{Edges, GraphError, TypeMap, TypeKey, ViewMut, View, Move};
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
    let path = input
        .attrs
        .iter()
        .find_map(|att| -> Option<syn::Path> {
            let mut path = None;
            if att.path().is_ident("moongraph") {
                att.parse_nested_meta(|meta| {
                    if meta.path.is_ident("crate") {
                        let value = meta.value()?;
                        path = Some(value.parse()?);
                        Ok(())
                    } else {
                        Err(meta.error(""))
                    }
                })
                .ok()?;
                path
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            // UNWRAP: safe because we know this will parse
            let path: syn::Path = syn::parse_str("moongraph").unwrap();
            path
        });
    let name = input.ident;
    let (construct_return, tys) = gen_edges_body(&input.data, &name, &path);
    let mut generics = input.generics;
    {
        /// Adds a `Edges` bound on each of the system data types.
        fn constrain_system_data_types(clause: &mut WhereClause, tys: &[Type], path: &syn::Path) {
            for ty in tys.iter() {
                let where_predicate: WherePredicate = syn::parse_quote!(#ty : #path::Edges);
                clause.predicates.push(where_predicate);
            }
        }

        let where_clause = generics.make_where_clause();
        constrain_system_data_types(where_clause, &tys, &path)
    }

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let output = quote! {
        impl #impl_generics #path::Edges for #name #ty_generics #where_clause {
            fn reads() -> Vec<#path::TypeKey> {
                let mut r = Vec::new();
                #({
                    r.extend(<#tys as #path::Edges>::reads());
                })*
                r
            }

            fn writes() -> Vec<#path::TypeKey> {
                let mut r = Vec::new();
                #({
                    r.extend(<#tys as #path::Edges>::writes());
                })*
                r
            }

            fn moves() -> Vec<#path::TypeKey> {
                let mut r = Vec::new();
                #({
                    r.extend(<#tys as #path::Edges>::moves());
                })*
                r
            }

            fn construct(resources: &mut #path::TypeMap) -> Result<Self, #path::GraphError> {
                Ok(#construct_return)
            }
        }
    };

    output.into()
}
