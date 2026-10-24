# V3 codec

A fixed-width, planar, predictive lossless audio codec with a batched
AVX2 multi-stream decode kernel, built for the case where many
independent, differently-positioned compressed streams must be decoded
concurrently (e.g. a polyphonic software sampler rendering many
simultaneous voices from sample-based audio).

See
[`docs/disclosures/v3-codec-technical-disclosure.md`](../../docs/disclosures/v3-codec-technical-disclosure.md)
(in the host repository this library was extracted from) for the full,
host-independent technical disclosure — format, predictor family, the
batched decode kernel including why the planar layout matters, and the
measured results.

**Status:** extracted from, and still developed alongside, its host
application ([GrandOrgue](https://github.com/GrandOrgue/grandorgue)),
whose `GOSoundHotKernels.h`/`GOSoundStream`/`GOSoundGroupTask` are the
production integration this codec is tuned against. The two symbols a
host project configures (`V3CODEC_MAX_CHANNELS`, `V3PredictorMode`) are
defined here with GrandOrgue-independent names specifically so this
directory needs no change to be lifted into its own repository or
reused by a different host.

## Building

Header-only by design — the AVX2 kernels are hot, per-sample code that
must stay fully inlined into the caller's own translation unit; a real
library boundary here would cost exactly the performance this codec
exists for (see the technical disclosure's "Measured results": the
batching step is the dominant source of speed-up, and it only applies
across translation-unit boundaries if the compiler can still inline
through them). There is nothing to compile or link on its own.

As a CMake `INTERFACE` target (see `CMakeLists.txt`):

```cmake
add_subdirectory(path/to/v3codec)
target_link_libraries(your_target PRIVATE v3codec)
```

This only adds `#include` search path — no object file, no `.so`/`.a`.
`#include "GOSoundCompressionCacheV3.h"` and
`#include "GOSoundCompressionCacheV3Batch.h"` from your own code; both
require a `-mavx2`-capable compiler for the batched kernels (gated at
runtime via `GOSoundCompressionCacheV3Batch::CpuSupported()` — the
binary's own baseline instruction set is untouched, the kernels are
built with a `target("avx2")` attribute and only entered after that
runtime check).

## Minimal usage

Single-stream encode, then decode, bit-exact:

```cpp
#include "GOSoundCompressionCacheV3.h"

std::vector<unsigned char> buf(
  GOSoundCompressionCacheV3::WorstCaseSize(frameCount, channels));

GOSoundCompressionCacheV3 enc;
unsigned written = enc.Encode(
  buf.data(), frameCount, channels, /*order=*/3,
  /*snapshotPositions=*/nullptr, /*numSnapshots=*/0,
  [&](unsigned frame, unsigned channel) { return getSample(frame, channel); },
  /*outSnapshots=*/nullptr);

GOSoundCompressionCacheV3 dec;
dec.DecompressTo(/*position=*/frameCount - 1, buf.data(), channels, /*order=*/3);
// dec.m_value[c] now holds each channel's decoded sample at that position
```

For the resumable-snapshot and 8-lane AVX2 batched-decode APIs (the
mechanisms that make concurrent multi-stream decode and seeking/loop
re-entry practical — see the technical disclosure's parts 3 and 4),
see the extensive doc comments directly in `GOSoundCompressionCacheV3.h`
(`Snapshot`/`LoadSnapshot`) and `GOSoundCompressionCacheV3Batch.h`
(`DecodeBatch`, `DecodeBatchInterleaved`,
`DecodeBatchInterleavedMidCapture`, `DecodeBatchPlanarMidCapture`).

## Tests and reproducing the measured results

The following live in `tools/compression/` in the host repository
(standalone `g++`/`clang++` programs, not part of the host's CMake
build — each file's own header comment has the exact build command):

- `v3_production_roundtrip_check.cpp` — sequential decode, segment-
  boundary resume, and backward-seek bit-exactness against real audio
  files.
- `v3_batch_multivoice_check.cpp` — the 8-lane batched kernel against
  sequential single-stream decode, including phase-misaligned lanes
  and a partial (5-of-8) batch, on real files.
- `codec_fleet_benchmark.cpp` — the throughput/compression-ratio
  benchmark behind the technical disclosure's "Measured results".

## Dependencies and their licenses

None beyond the C++17 standard library and, for the batched kernels,
an x86-64 compiler supporting AVX2 intrinsics (`<immintrin.h>`).

## License

Dual-licensed: the GNU General Public License v3.0 or later (see
`LICENSE`, the canonical GPLv3 text), or a separate commercial license
for proprietary use without the GPL's obligations — contact
**sebion7125@gmail.com**.

Copyright (C) 2026 **Sebion7125**.
