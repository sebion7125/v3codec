# V3 sample codec: fixed-width planar predictive codec with AVX2 batch
# decode (see GOSoundCompressionCacheV3.h / GOSoundCompressionCacheV3Batch.h
# for the format and the measurement history in
# tools/compression/INVESTIGATION.md).
#
# Deliberately self-contained, like src/xorcist: no host header is included
# here, and the two symbols a host project configures (V3CODEC_MAX_CHANNELS,
# V3PredictorMode) are defined locally with GrandOrgue-independent names, so
# this directory can be lifted into its own repository unchanged.
#
# Header-only on purpose (INTERFACE, no compiled .cpp): the kernels are hot,
# per-sample AVX2 code that must stay fully inlined into the caller's own
# translation unit (GOSoundStream.cpp, GOSoundGroupTask.cpp via
# GOSoundHotKernels.h) - a real library boundary here would cost exactly the
# performance this codec exists for. target_include_directories is enough;
# there is nothing to compile or link.
add_library(v3codec INTERFACE)
target_include_directories(v3codec INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
