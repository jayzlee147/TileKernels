import torch as _torch

# Workaround for tilelang MFMA codegen bug on ROCm.
# MatrixCoreIntrinEmitter stores dtype attrs as tvm_ffi DataType objects (a str
# subclass). When these are passed to T.tvm_mfma(), the TVM FFI recognizes them
# as DataType rather than plain str, so they don't get auto-converted to StringImm
# (a PrimExpr). This causes:
#   TypeError: Expected Array<ir.PrimExpr> but got Array[index N: DataType]
# Workaround: wrap mfma() to ensure dtype attrs are plain str before the call.
if _torch.version.hip is not None:
    try:
        from tilelang.rocm.intrinsics.mfma_macro_generator import MatrixCoreIntrinEmitter

        _orig_mfma = MatrixCoreIntrinEmitter.mfma

        def _needs_patch(emitter):
            """Check if dtype attrs are DataType objects (str subclass but not pure str)."""
            return type(emitter.a_dtype) is not str

        def _patched_mfma(self, A_local_buf, B_local_buf, C_local_buf, k_inner=0):
            if _needs_patch(self):
                orig_a, orig_b, orig_out = self.a_dtype, self.b_dtype, self.accum_dtype
                self.a_dtype = str(orig_a)
                self.b_dtype = str(orig_b)
                self.accum_dtype = str(orig_out)
                try:
                    return _orig_mfma(self, A_local_buf, B_local_buf, C_local_buf, k_inner)
                finally:
                    self.a_dtype = orig_a
                    self.b_dtype = orig_b
                    self.accum_dtype = orig_out
            return _orig_mfma(self, A_local_buf, B_local_buf, C_local_buf, k_inner)

        MatrixCoreIntrinEmitter.mfma = _patched_mfma
    except (ImportError, OSError):
        pass
