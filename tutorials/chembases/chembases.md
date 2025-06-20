# chembases

This package constructs atomic orbital (AO) basis functions using [Polynomials4ML] and verifies them against PySCF.

The AO basis functions take the form of a linear combination of primitive GTOs:

\[
\phi_{lm}(\mathbf{r}) = Y_{lm}(\hat{\mathbf{r}}) \sum_{i=1}^{N} \xi_i \, r^l e^{-\alpha_i r^2}
\]

Both \( \alpha_i \) and \( \xi_i \) are extracted from the PySCF package.

## AO Basis Construction

The function `build_ao(atom::String, basis::String)` constructs an AO basis compatible with PySCF.

### Example
```julia
using chembases

atom = "Be"
basis_set = "cc-pvdz"
basis = build_ao(atom, basis_set)
```

## Saving & Reloading AO Basis

To avoid repeated PySCF calls, one can save all AO basis data to a JSON file and load it later.

### Save Basis
```julia
save_all_bases_to_json(["Be"], ["cc-pvdz"], "ao_basis.json")
```

### Load Basis
With this `.json` file, PySCF is no longer needed.  
All basis sets can be precomputed and stored for lookup when needed.

```julia
basis = load_basis_from_json("ao_basis.json", "Be", "cc-pvdz")
```

[Polynomials4ML]: https://github.com/ACEsuit/Polynomials4ML.jl
