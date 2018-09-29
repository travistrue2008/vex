# vex

`vex` is yet another 3D math library for performing primarily vector and matrix operations.

## Roadmap (Priority's a Wild Ride)

**TODOs**

- Implement reusable code to reduce the growing amount of redundancy (traits, maybe?)
- Check if using base-type values in function calls move their ownership prohibiting further use in their declared scope
- Add a compiler flag for setting the coordinate system (left-handed vs right-handed)
- Add compiler flag for setting the memory mapping matrices
- Add generics to vector and matrix types
- Use `std::mem::swap<T>(...)` in `matrixX::transpose()` functions
- Add serialization support
- Clean up documentation for more consistent use-cases
- Add SIMD support
- Implement quaternion support
- Implement plane support
- Implement rect support

**DONE**

- Convert m[n] lookups with mXX() for memory mapping-agnostic
- Rename vecX/VecX to vector/VectorX to remove collisions with std::vec and std::Vec
- Rename matX/MatX to matrix/MatrixX to remove collisions with std::mat and std::Mat
- Remove local MatX::identity() methods, and rename MatX::new() methods to MatX::identity()
- Rename ::construct(...) methods to ::make(...)
- Add inlining to functions
