# Convert an AtomicOrbitals object to a dictionary that can be written to JSON
function serialize_basis(basis)
    Dict(
        # Convert each column of ζ and D matrices into a vector (Vector of Vectors)
        "ζ" => [collect(col) for col in eachcol(basis.Dn.ζ)],
        "D" => [collect(col) for col in eachcol(basis.Dn.D)],
        # Convert the `spec` vector of NamedTuples into a list of dictionaries
        "spec" => [Dict("n1" => s.n1, "n2" => s.n2, "l" => s.l, "m" => s.m) for s in basis.spec]
    )
end

# Loop over all elements and basis sets, serialize their AO basis, and save to a single JSON file
function save_all_bases_to_json(element_names, basis_names, filename::String)
    all_data = Dict()

    for atom in element_names
        println("Processing $atom...")
        atom_data = Dict()
        for bset in basis_names
            try
                basis = build_ao(atom, bset)                  # Construct AO basis for atom & basis set
                atom_data[bset] = serialize_basis(basis)     # Convert it to a JSON-compatible dict
            catch e
                @warn "Failed for $atom with $bset: $e"       # Handle failures gracefully
            end
        end
        all_data[atom] = atom_data   # Store all basis sets for this atom
    end

    # Write everything to a JSON file
    open(filename, "w") do io
        JSON.print(io, all_data)
    end
end
