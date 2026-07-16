# Add support for custom flag names via a `#[flag_name]` attribute

Flags defined in `bitflags!` currently derive their string representation (used by things like `Display`, iteration, and parsing) directly from the Rust constant identifier. There are cases where the desired external name of a flag cannot match a valid Rust identifier — for example, a single-character name like `"a"`, or a name that mirrors an existing external convention. We need a way to decouple the flag's Rust identifier from the name used when the flag is formatted or otherwise referred to as a string.

Introduce a `#[flag_name = "..."]` attribute that can be applied to individual flag definitions inside the `bitflags!` macro. When present, the provided string literal overrides the identifier-derived name for that flag everywhere the flag's textual name is used.

```rust
bitflags! {
    pub struct MyFlags: u8 {
        #[flag_name = "a"]
        const A = 1;
    }
}
```

## Expected outcomes
1. A `#[flag_name = "<literal>"]` attribute is accepted on any flag constant inside `bitflags!` and compiles cleanly.
2. When a flag carries `#[flag_name = "x"]`, the string `"x"` is used in place of the constant's identifier for all name-based behaviour: formatting/`Display`, flag iteration that yields names, and parsing from strings.
3. Flags without the attribute continue to use their identifier as the name, exactly as before.
4. The attribute works alongside other attributes and doc comments on the same flag without interfering with them.
5. Existing behaviour and public APIs remain fully backward compatible for code that does not use the attribute.

## Constraints
- The attribute value is a plain string literal and is used verbatim. Do not validate, trim, or reject unusual values (e.g. `"_"`, `"a | b"`, or names with leading/trailing whitespace) — passing such names through is acceptable and must not cause a panic or compile error.
- Do not change the numeric representation, bit values, or type parameters of any flag; only the string name is affected.
- Keep the macro ergonomic: the attribute must be optional and independent per flag.

## Implementation notes
- The name substitution needs to flow through the macro expansion into whatever internal structure records each flag's name, so that all name-consuming code paths pick it up from a single source.
- Add test coverage exercising: a flag with `#[flag_name]` compared to one without, round-tripping through formatting and parsing, and at least one "unusual" name value to confirm it is passed through unmodified.

IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.
