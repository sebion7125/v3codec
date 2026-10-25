# V3 codec

A fixed-width, planar, predictive lossless audio codec with a batched
AVX2 multi-stream decode kernel, built for the case where many
independent, differently-positioned compressed streams must be decoded
concurrently (e.g. a polyphonic software sampler rendering many
simultaneous voices from sample-based audio).

See
[`docs/disclosures/v3-codec-technical-disclosure.md`](docs/disclosures/v3-codec-technical-disclosure.md)
for the full, host-independent technical disclosure — format,
predictor family, the batched decode kernel including why the planar
layout matters, and the measured results. (Also submitted to Technical
Disclosure Commons; mirrored here so it ships with the code it
describes rather than only with the host application it was extracted
from.)

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
enc.Init(); // required: the default constructor leaves state uninitialised
unsigned written = enc.Encode(
  buf.data(), frameCount, channels, /*order=*/3,
  /*snapshotPositions=*/nullptr, /*numSnapshots=*/0,
  [&](unsigned frame, unsigned channel) { return getSample(frame, channel); },
  /*outSnapshots=*/nullptr);

GOSoundCompressionCacheV3 dec;
dec.Init();
dec.DecompressTo(/*position=*/frameCount - 1, buf.data(), channels, /*order=*/3);
// dec.m_value[c] now holds each channel's decoded sample at that position
```

Batched decode of several independent streams at once, the part that
actually carries the codec's speed-up (see the technical disclosure's
"Measured results"): up to `LANES` (8) independently-positioned
streams decode together in one AVX2 pass instead of one at a time.

```cpp
#include "GOSoundCompressionCacheV3.h"
#include "GOSoundCompressionCacheV3Batch.h"

using Lane = GOSoundCompressionCacheV3Batch::Lane;
constexpr unsigned LANES = GOSoundCompressionCacheV3Batch::LANES; // 8

if (!GOSoundCompressionCacheV3Batch::CpuSupported()) {
  // No AVX2 at runtime: decode each stream individually via
  // GOSoundCompressionCacheV3::DecompressTo instead, as above.
}

// LANES independently encoded streams (e.g. LANES organ pipes sounding
// at once), each already Encode()'d into its own buffer exactly as in
// the single-stream example above.
GOSoundCompressionCacheV3 cache[LANES];
Lane lanes[LANES];
for (unsigned i = 0; i < LANES; i++) {
  cache[i].Init(); // required: see the single-stream example above
  // Position each stream at its own current playback position before
  // batching — here, the start; a real caller resumes from wherever
  // that voice last left off via a Snapshot (see LoadSnapshot) rather
  // than always starting at 0.
  cache[i].DecompressTo(
    /*position=*/0, encodedBuf[i].data(), channels, /*order=*/3);
  lanes[i] = {&cache[i], channels};
}

// One interleaved output buffer per channel: out[c][sample * LANES + lane].
const unsigned nSamples = 128; // e.g. one render period
std::vector<int32_t> outStorage[V3CODEC_MAX_CHANNELS];
int32_t *out[V3CODEC_MAX_CHANNELS];
for (unsigned c = 0; c < channels; c++) {
  outStorage[c].resize(nSamples * LANES);
  out[c] = outStorage[c].data();
}

GOSoundCompressionCacheV3Batch::DecodeBatchInterleaved(
  lanes, LANES, nSamples, out);
// out[c][s * LANES + lane] now holds channel c's sample s for each of
// the LANES streams, all decoded together in this one call.
```

All lanes passed to one call must share the same channel count —
grouping differently-channelled streams is the caller's responsibility,
not this kernel's (see the `Lane` struct's own comment). For the
resumable-snapshot mechanism that lets a real caller resume each lane
from wherever it actually left off instead of always decoding from
frame 0 (the mechanism that makes concurrent multi-stream decode and
seeking/loop re-entry practical — see the technical disclosure's parts
3 and 4), see the extensive doc comments directly in
`GOSoundCompressionCacheV3.h` (`Snapshot`/`LoadSnapshot`) and the
remaining `GOSoundCompressionCacheV3Batch.h` entry points
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
