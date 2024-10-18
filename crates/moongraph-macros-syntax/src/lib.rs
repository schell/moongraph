//! Utility functions for writing `moongraph` macros.
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

/// Find the 'path' attribute in an `Edges` derive macro.
///
/// ```ignore
/// #[moongraph(crate = apecs)]
///   ^^^^^^^^^         ^^^^^
///   |                 |
///   outer_attribute   returned path
/// ```
pub fn find_path(
    outer_attribute: &str,
    input: &DeriveInput,
) -> Result<Option<syn::Path>, syn::Error> {
    let mut path = None;
    for att in input.attrs.iter() {
        if att.path().is_ident(outer_attribute) {
            att.parse_nested_meta(|meta| {
                if meta.path.is_ident("crate") {
                    let value = meta.value()?;
                    let parsed_path: syn::Path = value.parse()?;
                    path = Some(parsed_path);
                    Ok(())
                } else {
                    Err(meta.error(""))
                }
            })?;
            break;
        }
    }
    Ok(path)
}

/// Derive `#path::Edges`.
///
/// Useful for re-exporting `moongraph::Edges` and the derive macro.
pub fn derive_edges(input: DeriveInput, path: syn::Path) -> proc_macro2::TokenStream {
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

    quote! {
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
    }
}
