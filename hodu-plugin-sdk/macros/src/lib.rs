//! Procedural macros for hodu-plugin-sdk
//!
//! This crate provides derive macros to reduce boilerplate when writing plugins.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Data, DeriveInput, Fields};

/// Derive macro for creating plugin method handlers with less boilerplate
///
/// This macro generates the necessary wrapper code for async handlers.
///
/// # Example
///
/// ```ignore
/// use hodu_plugin_sdk::PluginMethod;
///
/// #[derive(PluginMethod)]
/// #[method(name = "backend.run")]
/// struct RunHandler;
///
/// impl RunHandler {
///     async fn handle(ctx: Context, params: RunParams) -> Result<RunResult, RpcError> {
///         // implementation
///     }
/// }
/// ```
#[proc_macro_derive(PluginMethod, attributes(method))]
pub fn derive_plugin_method(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;
    let method_name = extract_method_name(&input);

    let expanded = quote! {
        impl #name {
            /// Get the method name for this handler
            pub const METHOD_NAME: &'static str = #method_name;

            /// Register this handler with a plugin server
            pub fn register<S>(server: S) -> S
            where
                S: PluginServerExt,
            {
                server.register_method(Self::METHOD_NAME, Self::handle)
            }
        }
    };

    TokenStream::from(expanded)
}

fn extract_method_name(input: &DeriveInput) -> String {
    for attr in &input.attrs {
        if attr.path().is_ident("method") {
            // Parse #[method(name = "...")] using proper syn parsing
            if let Ok(meta) = attr.meta.require_list() {
                // Try to parse as name = value pair
                if let Ok(nv) = syn::parse2::<syn::MetaNameValue>(meta.tokens.clone()) {
                    if nv.path.is_ident("name") {
                        if let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(lit_str),
                            ..
                        }) = &nv.value
                        {
                            return lit_str.value();
                        }
                    }
                }
                // Also try parsing as just a string literal: #[method("name")]
                if let Ok(lit_str) = syn::parse2::<syn::LitStr>(meta.tokens.clone()) {
                    return lit_str.value();
                }
            }
        }
    }
    // Default: convert struct name to snake_case method name
    to_snake_case(&input.ident.to_string())
}

fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            // to_lowercase() always returns at least one char for any char
            for lower in c.to_lowercase() {
                result.push(lower);
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Attribute macro for simplifying handler function definitions
///
/// # Example
///
/// ```ignore
/// use hodu_plugin_sdk::plugin_handler;
///
/// #[plugin_handler("backend.run")]
/// async fn handle_run(ctx: Context, params: RunParams) -> Result<RunResult, RpcError> {
///     // implementation
/// }
/// ```
#[proc_macro_attribute]
pub fn plugin_handler(attr: TokenStream, item: TokenStream) -> TokenStream {
    let method_name = attr.to_string().trim_matches('"').to_string();
    let input = parse_macro_input!(item as syn::ItemFn);

    let fn_name = &input.sig.ident;
    let fn_block = &input.block;
    let fn_inputs = &input.sig.inputs;
    let fn_output = &input.sig.output;
    let fn_async = &input.sig.asyncness;

    let register_fn_name = format_ident!("register_{}", fn_name);

    let expanded = quote! {
        #fn_async fn #fn_name(#fn_inputs) #fn_output #fn_block

        /// Register this handler with a plugin server (auto-generated)
        pub fn #register_fn_name<F, Fut, P, R>(server: hodu_plugin_sdk::server::PluginServer) -> hodu_plugin_sdk::server::PluginServer
        where
            F: Fn(hodu_plugin_sdk::Context, P) -> Fut + Send + Sync + 'static,
            Fut: std::future::Future<Output = Result<R, hodu_plugin_sdk::rpc::RpcError>> + Send + 'static,
            P: serde::de::DeserializeOwned + Send + 'static,
            R: serde::Serialize + 'static,
        {
            server.method(#method_name, #fn_name)
        }
    };

    TokenStream::from(expanded)
}

/// Macro for defining params structs with automatic serde derives
///
/// # Example
///
/// ```ignore
/// use hodu_plugin_sdk::define_params;
///
/// define_params! {
///     pub struct MyParams {
///         pub path: String,
///         pub options: Option<String>,
///     }
/// }
/// ```
#[proc_macro]
pub fn define_params(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match define_params_impl(&input) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error().into(),
    }
}

fn define_params_impl(input: &DeriveInput) -> Result<TokenStream, syn::Error> {
    let name = &input.ident;
    let vis = &input.vis;

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    input,
                    "define_params only supports structs with named fields",
                ))
            },
        },
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "define_params only supports structs, not enums or unions",
            ))
        },
    };

    let field_defs: Vec<_> = fields
        .iter()
        .map(|f| {
            let name = &f.ident;
            let ty = &f.ty;
            let vis = &f.vis;
            quote! { #vis #name: #ty }
        })
        .collect();

    let expanded = quote! {
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
        #vis struct #name {
            #(#field_defs),*
        }
    };

    Ok(TokenStream::from(expanded))
}

/// Macro for defining result structs with automatic serde derives
#[proc_macro]
pub fn define_result(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match define_params_impl(&input) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error().into(),
    }
}
