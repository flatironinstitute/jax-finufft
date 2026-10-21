# Vendored XLA FFI headers

`xla/ffi/api/{api.h,c_api.h,ffi.h}` and `LICENSE` are copied from
- commit [openxla/xla@9f150f6](https://github.com/openxla/xla/tree/9f150f6b75c08d6ea7b97697c4f393f1a0eb6121/xla/ffi/api), pinned by
- JAX version 0.8.0, providing
- XLA FFI API version 0.1

XLA recommends this vendoring approach: https://github.com/openxla/xla/blob/5f07b925025a1223a75af21ca1707f9022a71f95/xla/ffi/README.md#L62-L64

We will generally try to keep these headers at the oldest version that supports
the latest JAX release.

To update, copy `xla/ffi/api/*.h` out of `jax.ffi.include_dir()` for the
chosen jaxlib version, and update the commit link above. The XLA commit pinned
by a jax release is in `third_party/xla/revision.bzl` (jax ≤ 0.11.1) or
`MODULE.bazel` (jax ≥ 0.11.2) at that release's tag.
