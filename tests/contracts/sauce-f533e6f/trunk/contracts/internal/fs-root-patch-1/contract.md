# Root patch descriptor

Status: internal

An immutable selection of one-based physical root IDs from an authored GMP parent.
`parent_fingerprint` is the 16-digit hexadecimal native content fingerprint
computed after reading the parent and before scaling or binding geometry.
Loading rejects a changed parent, duplicate roots, PML-bearing parents, and
invalid IDs. IDs follow native wave-mesh root order and survive child refinement.

Persist the original parent and this descriptor together. Sparse child GMP
entities retain parent IDs and must not be exported through the compact text
writer. Extraction preserves native curved geometry and its dependencies,
filters inherited boundary memberships, and labels newly exposed faces
`patch_cut`. The descriptor alone does not configure PML or material controls.
