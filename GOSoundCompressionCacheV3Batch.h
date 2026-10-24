/*
 * V3 batch decode: AVX2 multi-stream kernel for the V3 codec. See
 * docs/disclosures/v3-codec-technical-disclosure.md for the
 * host-independent technical description.
 *
 * Copyright (C) 2026 Sebion7125
 *
 * Dual-licensed: use under the GNU General Public License v3.0 or
 * later (see LICENSE), or under a separate commercial license for
 * proprietary use without the GPL's obligations - contact
 * sebion7125@gmail.com for commercial licensing terms.
 */

#ifndef GOSOUNDCOMPRESSIONCACHEV3BATCH_H
#define GOSOUNDCOMPRESSIONCACHEV3BATCH_H

#include <algorithm>
#include <cstdint>
#include <cstring>

/* Same guard as GO_POLYPHASE_AVX2 (GOSoundResample.h): x86 gcc/clang only.
 * The kernel functions carry target("avx2") so the rest of the binary keeps
 * the project's baseline ISA; callers must check CpuSupported() at runtime
 * before calling into the kernel. The standalone tools compile with -mavx2
 * globally, where the attribute is redundant but harmless. */
#if (defined(__GNUC__) || defined(__clang__))                                  \
  && (defined(__x86_64__) || defined(__i386__))
#define GO_V3_BATCH_AVX2 1
#include <immintrin.h>
#endif

#include <cassert>

#include "GOSoundCompressionCacheV3.h"

#ifdef GO_V3_BATCH_AVX2

// AVX2 8-voice batched decode for GOSoundCompressionCacheV3: advances up to
// 8 independent voices' caches together, one AVX2 register holding all 8
// lanes' predictor state per channel, instead of looping DecompressionStep
// 8 times. This is the second half of the codec-speed investigation's
// "Step 1" (see tools/compression/INVESTIGATION.md) - the fixed-width
// format alone (GOSoundCompressionCacheV3's scalar single-voice decode)
// only gains ~1.0-1.9x over legacy; this batching is what the ~4.75x
// figure in the investigation actually depends on.
//
// Relies on GOSoundCompressionCacheV3's block layout being PLANAR PER
// CHANNEL (each channel's residuals contiguous within a block, not
// interleaved sample-by-sample) - an earlier interleaved version of the
// format made every width==3 extraction fall back to a scalar per-sample
// loop for real (stereo) content, because the fast paths below all need
// contiguous memory. With planar-per-channel, every width's extraction is
// a handful of SIMD loads regardless of channel count.
//
// Validated (not assumed) against the real GOSoundCompressionCacheV3
// single-voice decoder, bit-exact, on real stereo GrandOrgue sample files
// deliberately started at different, arbitrary, non-block-aligned
// positions in their own streams (see
// tools/compression/v3_batch_multivoice_check.cpp) - i.e. the realistic
// case of 8 unrelated voices claimed into one batch mid-flight by
// GOSoundGroupTask's scheduler, NOT 8 voices that happen to start together
// (which is all the original codec_fleet_benchmark.cpp prototype ever
// tested), AND a partial (< 8) lane batch, which exercises the lane-0
// padding path below.
//
// GOSoundCompressionCacheV3's predictor is the undamped 2nd-order
// 2*prev - last (cache version 4, 2026-07-13; see the rationale in
// GOSoundCompressionCacheV3.h). The SIMD form is
// _mm256_sub_epi32(_mm256_add_epi32(prev, prev), last) - it must stay
// bit-identical to the scalar decoder and the encoder. (The codec
// originally shipped the damped prev + ((prev - last) >> 1); the
// measured switch to the full slope is smaller AND one op cheaper.)
//
// One real bug found and fixed while building this up - do not undo it:
//
//   1. Padding for numLanes < LANES must use an INDEPENDENT scratch copy
//      of lane 0's cache per unused slot, never a second pointer aliasing
//      the same object: the per-lane loops below mutate *padded[l] in
//      place once per chunk, so aliased slots would advance the same
//      underlying cache multiple times per chunk and corrupt lane 0's
//      real state - a save-before/restore-after of lane 0 around the
//      whole call does NOT fix this, because wrong intermediate state
//      still feeds the shared AVX2 predictor registers during the call
//      itself. Caught by a numLanes==5 test; an all-8-lanes-only test
//      cannot catch it.
class GOSoundCompressionCacheV3Batch {
public:
  static constexpr unsigned LANES = 8;
  static constexpr unsigned BLOCK_SAMPLES = GOSoundCompressionCacheV3::BLOCK_SAMPLES;

  // One lane's live state for a batch call: an independently-positioned
  // cache (own m_blockData/m_ptr/m_blockRemaining/m_curWidth/predictor
  // state) plus its channel count. All lanes passed to one DecodeBatch()
  // call must share the same channel count; grouping voices by matching
  // channel count is the caller's (scheduler's) responsibility, not this
  // kernel's.
  struct Lane {
    GOSoundCompressionCacheV3 *cache;
    unsigned channels;
  };

  /* Runtime gate for the target("avx2") kernel below: the binary's baseline
   * ISA stays untouched, so this must be checked before every entry into
   * DecodeBatch/DecodeBatchInterleaved. */
  static inline bool CpuSupported() {
    static const bool has = __builtin_cpu_supports("avx2");
    return has;
  }

private:
  __attribute__((target("avx2"), always_inline)) static inline void transpose8x8(
    __m256i *m) {
    __m256 r0 = _mm256_castsi256_ps(m[0]);
    __m256 r1 = _mm256_castsi256_ps(m[1]);
    __m256 r2 = _mm256_castsi256_ps(m[2]);
    __m256 r3 = _mm256_castsi256_ps(m[3]);
    __m256 r4 = _mm256_castsi256_ps(m[4]);
    __m256 r5 = _mm256_castsi256_ps(m[5]);
    __m256 r6 = _mm256_castsi256_ps(m[6]);
    __m256 r7 = _mm256_castsi256_ps(m[7]);

    __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    __m256 t7 = _mm256_unpackhi_ps(r6, r7);

    __m256 tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

    m[0] = _mm256_castps_si256(_mm256_permute2f128_ps(tt0, tt4, 0x20));
    m[1] = _mm256_castps_si256(_mm256_permute2f128_ps(tt1, tt5, 0x20));
    m[2] = _mm256_castps_si256(_mm256_permute2f128_ps(tt2, tt6, 0x20));
    m[3] = _mm256_castps_si256(_mm256_permute2f128_ps(tt3, tt7, 0x20));
    m[4] = _mm256_castps_si256(_mm256_permute2f128_ps(tt0, tt4, 0x31));
    m[5] = _mm256_castps_si256(_mm256_permute2f128_ps(tt1, tt5, 0x31));
    m[6] = _mm256_castps_si256(_mm256_permute2f128_ps(tt2, tt6, 0x31));
    m[7] = _mm256_castps_si256(_mm256_permute2f128_ps(tt3, tt7, 0x31));
  }

  /* Fixed integer predictor of a UNIFORM-order batch (the scheduler groups
   * voices by predictor order, so all 8 lanes share one order - see
   * GOSoundStream::GetBucketKey). pred is evaluated BEFORE the per-sample
   * shift, taps v[n-1..n-4] = value/prev/last/hist3. Integer adds and
   * shifts only (3*x = x<<1 + x, 6*x = x<<2 + x<<1) - no multiply, no
   * per-lane coefficient vectors. ORDER is a compile-time template parameter
   * (the caller dispatches ONCE per chunk on batchOrder, not per sample), so
   * each order compiles to a straight-line hardcoded predictor with no branch
   * in the hot loop - identical codegen to the old fixed-order-3 kernel.
   * Bit-identical to the scalar coefficient form. */
  template <unsigned ORDER>
  __attribute__((target("avx2"), always_inline)) static inline __m256i predictOrder(
    __m256i value, __m256i prev, __m256i last, __m256i hist3) {
    /* ADDS, NOT SHIFTS - do not "optimize" 2*x into x<<1 here.
     * On x86 VPADDD issues on three ports (0/1/5), VPSLLD on one (0). In
     * this per-sample loop the shifts therefore serialize on a single port
     * while the adds spread over three. Measured 2026-07-15: the shift form
     * cost ~10% decode throughput against the add form, at identical
     * instruction count - 20 vpslld replacing 22 vpaddd was the whole
     * difference to the old hardcoded kernel. */
    if constexpr (ORDER == 1) { // v[n-1]
      return value;
    } else if constexpr (ORDER == 2) { // 2 v[n-1] - v[n-2]
      return _mm256_sub_epi32(_mm256_add_epi32(value, value), prev);
    } else if constexpr (ORDER == 3) { // 3 v[n-1] - 3 v[n-2] + v[n-3]
      const __m256i v3
        = _mm256_add_epi32(_mm256_add_epi32(value, value), value);
      const __m256i p3 = _mm256_add_epi32(_mm256_add_epi32(prev, prev), prev);
      return _mm256_add_epi32(_mm256_sub_epi32(v3, p3), last);
    } else { // order 4: 4 v[n-1] - 6 v[n-2] + 4 v[n-3] - v[n-4]
      // ADD form (see the port note above) - the removed v7 kernel used
      // shifts here, which serialize on port 0.
      const __m256i v2 = _mm256_add_epi32(value, value);
      const __m256i v4 = _mm256_add_epi32(v2, v2);
      const __m256i p2 = _mm256_add_epi32(prev, prev);
      const __m256i p4 = _mm256_add_epi32(p2, p2);
      const __m256i p6 = _mm256_add_epi32(p4, p2);
      const __m256i l2 = _mm256_add_epi32(last, last);
      const __m256i l4 = _mm256_add_epi32(l2, l2);
      return _mm256_sub_epi32(
        _mm256_add_epi32(_mm256_sub_epi32(v4, p6), l4), hist3);
    }
  }

  // Extracts `m` residuals of one channel from one lane's PLANAR (always
  // contiguous, stride == WIDTH) residual region into a scratch row of
  // int32. Every width is a few SIMD loads now that the format guarantees
  // contiguity regardless of channel count (see the class comment).
  template <int WIDTH>
  __attribute__((target("avx2"), always_inline)) static inline void extractChannelPlanar(
    const unsigned char *ib, unsigned m, int32_t *scratchRow) {
    unsigned j = 0;
    if constexpr (WIDTH == 1) {
      for (; j + 8 <= m; j += 8) {
        const __m128i in8 = _mm_loadl_epi64((const __m128i *)(ib + j));
        const __m256i out = _mm256_cvtepi8_epi32(in8);
        _mm256_storeu_si256((__m256i *)(scratchRow + j), out);
      }
      for (; j < m; j++)
        scratchRow[j] = (int32_t)(int8_t)ib[j];
    } else if constexpr (WIDTH == 2) {
      for (; j + 8 <= m; j += 8) {
        const __m128i in8 = _mm_loadu_si128((const __m128i *)(ib + (size_t)j * 2));
        const __m256i out = _mm256_cvtepi16_epi32(in8);
        _mm256_storeu_si256((__m256i *)(scratchRow + j), out);
      }
      for (; j < m; j++)
        scratchRow[j] = GOSoundCompressionCacheV3::readResidual(ib + (size_t)j * 2, 2);
    } else if constexpr (WIDTH == 3) {
      // PSHUFB-based: shuffle 12 packed bytes (4 samples) into the byte
      // layout [0,b0,b1,b2] per int32 lane in one instruction (0x80
      // selector byte forces zero), then one arithmetic shift
      // sign-extends all 4 at once. Always applicable now (contiguous
      // regardless of channel count), unlike the interleaved format this
      // replaced.
      // non-static so gcc materialises it as a .rodata constant (no
      // thread-safe-init guard on the hot path; the guard showed up as a
      // per-call __cxa_guard_acquire when this kernel is not inlined).
      const __m128i mask = _mm_setr_epi8(
        (char)0x80, 0, 1, 2, (char)0x80, 3, 4, 5,
        (char)0x80, 6, 7, 8, (char)0x80, 9, 10, 11);
      for (; j + 4 <= m; j += 4) {
        const __m128i in = _mm_loadu_si128((const __m128i *)(ib + (size_t)j * 3));
        const __m128i shuffled = _mm_shuffle_epi8(in, mask);
        const __m128i result = _mm_srai_epi32(shuffled, 8);
        _mm_storeu_si128((__m128i *)(scratchRow + j), result);
      }
      for (; j < m; j++)
        scratchRow[j] = GOSoundCompressionCacheV3::readResidual(ib + (size_t)j * 3, 3);
    } else {
      for (; j + 8 <= m; j += 8) {
        const __m256i out = _mm256_loadu_si256((const __m256i *)(ib + (size_t)j * 4));
        _mm256_storeu_si256((__m256i *)(scratchRow + j), out);
      }
      for (; j < m; j++)
        scratchRow[j] = GOSoundCompressionCacheV3::readResidual(ib + (size_t)j * 4, 4);
    }
  }

public:
  // Decodes `nSamples` further samples for each of `numLanes` (<= LANES)
  // independent lanes, writing planar per-lane-per-channel output into
  // out[lane][channel][0..nSamples). Every lane's cache is advanced
  // in-place exactly as if DecompressionStep(channels) had been called
  // nSamples times on it directly (bit-exact, see the class comment).
  // Lanes are NOT assumed phase-aligned: each may already have a
  // different m_blockRemaining/m_curWidth loaded (e.g. from a prior
  // DecompressTo/LoadSnapshot call) - a lane crossing into a new block
  // mid-batch never desyncs another lane still mid-block, because each
  // batch chunk is bounded by the nearest upcoming block boundary of ANY
  // lane.
  //
  // If numLanes < LANES, the unused lanes are padded with independent
  // scratch copies of lane 0's cache (see the class comment, bug 2) so
  // the fixed-8-lane AVX2 path always has real, valid inputs; their
  // decoded output is simply never read, and lane 0's real cache is
  // untouched by the padding.
  static void DecodeBatch(
    Lane lanes[], unsigned numLanes, unsigned nSamples,
    int32_t *const out[][V3CODEC_MAX_CHANNELS]) {
    DecodeBatchImpl<MODE_SPLIT>(
      lanes, numLanes, nSamples, out, nullptr, 0, nullptr, 0);
  }

  // Same decode, but writes each channel to a SIMD-friendly interleaved
  // block cache: out[channel][sample * LANES + lane]. This preserves the
  // original benchmark kernel's key property: each decoded AVX2 register is
  // stored with one contiguous vector store, instead of scattering it back
  // into per-lane buffers.
  static void DecodeBatchInterleaved(
    Lane lanes[], unsigned numLanes, unsigned nSamples,
    int32_t *const out[V3CODEC_MAX_CHANNELS]) {
    DecodeBatchImpl<MODE_INTERLEAVED>(
      lanes, numLanes, nSamples, nullptr, out, 0, nullptr, 0);
  }

  /* DecodeBatchInterleaved with a MID-RUN STATE CAPTURE: decodes nSamples
   * as usual, and additionally stores every real lane's complete decoder
   * state AS IT STANDS AFTER EXACTLY captureAt samples (0 < captureAt <=
   * nSamples) into midStates[0..numLanes). Purpose (2026-07-10 harness
   * finding, see tools/compression/v3_chunked_decode_bench.cpp): the
   * per-span render decode needs the dependent-forward chain point at
   * `span` but the decoded window extends GUARD samples further - without
   * this capture the guard tail had to run scalar per lane
   * (DecompressionStep), which cost more than a third of the whole decode
   * budget at 128-sample spans. With it, one SIMD call covers span+guard
   * and the chain point falls out for one 56-byte copy per lane. */
  static void DecodeBatchInterleavedMidCapture(
    Lane lanes[], unsigned numLanes, unsigned nSamples,
    unsigned captureAt, GOSoundCompressionCacheV3 midStates[],
    int32_t *const out[V3CODEC_MAX_CHANNELS]) {
    DecodeBatchImpl<MODE_INTERLEAVED>(
      lanes, numLanes, nSamples, nullptr, out, captureAt, midStates, 0);
  }

  /* Like DecodeBatchInterleavedMidCapture, but stores PLANAR per-lane
   * runs directly: out[c] + lane*planeStride + sample. The transpose the
   * render side would otherwise pay as a separate pass (variant D) is
   * fused into the store stage AT ZERO register cost: the residual rows
   * are consumed one per predictor step anyway, so each decoded value
   * register simply replaces the residual register it just consumed, and
   * one extra transpose8x8 per 8-sample tile flips the collected outputs
   * lane-planar before storing (bench F2 vs F1:
   * tools/compression/fir_batch_vs_planar_bench.cpp). Sub-8 chunk tails
   * (mid-capture boundaries, block seams) store scalar per lane. */
  static void DecodeBatchPlanarMidCapture(
    Lane lanes[], unsigned numLanes, unsigned nSamples,
    unsigned captureAt, GOSoundCompressionCacheV3 midStates[],
    int32_t *const out[V3CODEC_MAX_CHANNELS], unsigned planeStride) {
    DecodeBatchImpl<MODE_PLANAR>(
      lanes, numLanes, nSamples, nullptr, out, captureAt, midStates,
      planeStride);
  }

private:
  enum { MODE_SPLIT = 0, MODE_INTERLEAVED = 1, MODE_PLANAR = 2 };

  template <int MODE>
  __attribute__((target("avx2"))) static void DecodeBatchImpl(
    Lane lanes[],
    unsigned numLanes,
    unsigned nSamples,
    int32_t *const splitOut[][V3CODEC_MAX_CHANNELS],
    int32_t *const interleavedOut[V3CODEC_MAX_CHANNELS],
    unsigned captureAt,
    GOSoundCompressionCacheV3 *midStates,
    unsigned planeStride) {
    const unsigned channels = lanes[0].channels;

    GOSoundCompressionCacheV3 paddingScratch[LANES];
    GOSoundCompressionCacheV3 *padded[LANES];
    for (unsigned l = 0; l < LANES; l++) {
      if (l < numLanes)
        padded[l] = lanes[l].cache;
      else {
        paddingScratch[l] = *lanes[0].cache;
        padded[l] = &paddingScratch[l];
      }
    }

    alignas(32) int32_t planar[V3CODEC_MAX_CHANNELS][LANES][BLOCK_SAMPLES];
    __m256i value[V3CODEC_MAX_CHANNELS], prev[V3CODEC_MAX_CHANNELS],
      last[V3CODEC_MAX_CHANNELS], hist3[V3CODEC_MAX_CHANNELS];
    /* Uniform-order batch: only order 4 ever touches hist3 (v[n-4]). Gate
     * every hist3 load/store/capture on it - for order <= 3 this is dead
     * work AND an extra history array per state copy vs the fixed-3 kernel
     * (2026-07-15 lesson, kept through the order-4 re-add). */
    const bool needHist3 = padded[0]->m_order >= 4;
    for (unsigned c = 0; c < channels; c++) {
      int32_t v[LANES], p[LANES], la[LANES], h3[LANES];
      for (unsigned l = 0; l < LANES; l++) {
        v[l] = padded[l]->m_value[c];
        p[l] = padded[l]->m_prev[c];
        la[l] = padded[l]->m_last[c];
        if (needHist3)
          h3[l] = padded[l]->m_hist3[c];
      }
      value[c] = _mm256_loadu_si256((const __m256i *)v);
      prev[c] = _mm256_loadu_si256((const __m256i *)p);
      last[c] = _mm256_loadu_si256((const __m256i *)la);
      hist3[c] = needHist3 ? _mm256_loadu_si256((const __m256i *)h3)
                           : _mm256_setzero_si256();
    }
    unsigned done = 0;
    while (done < nSamples) {
      // Refresh any lane whose current block is exhausted.
      for (unsigned l = 0; l < LANES; l++) {
        GOSoundCompressionCacheV3 &c = *padded[l];
        if (c.m_blockRemaining == 0) {
          // Per-block header: [width:1]; the order is implicit and already
          // set (Init/RestoreFrom) - never re-read it here (atomic in the
          // hot refresh path; see GOSoundCompressionCacheV3::Init).
          c.m_curWidth = *c.m_ptr;
          c.m_blockData = c.m_ptr + 1;
          c.m_ptr = c.m_blockData + (size_t)channels * BLOCK_SAMPLES * c.m_curWidth;
          c.m_blockRemaining = BLOCK_SAMPLES;
        }
      }
      // Uniform-order batch: the scheduler groups voices by predictor order
      // (bucket key fold + group flush), so every lane shares one order -
      // dispatch the predictor on it once (hardcoded integer adds, no
      // per-lane muls). The uniformity is the CALLER's contract; a mixed
      // batch would silently decode the minority lanes with the wrong
      // predictor, so pin it in debug builds.
      const unsigned batchOrder = padded[0]->m_order;
#ifndef NDEBUG
      for (unsigned l = 1; l < LANES; l++)
        assert(
          padded[l]->m_order == batchOrder
          && "mixed-order batch: scheduler grouping broken");
#endif
      // Chunk bounded by nSamples left AND the nearest upcoming block
      // boundary of ANY lane - this is what makes phase-misaligned lanes
      // safe: no lane's block can be overrun by another lane's longer
      // remaining count.
      unsigned chunk = nSamples - done;
      // Hand-written min instead of std::min: the latter is a default-target
      // template that gcc refuses to inline into this target("avx2") kernel
      // (same barrier as the FIR accessors), leaving out-of-line calls in the
      // per-chunk loop.
      for (unsigned l = 0; l < LANES; l++) {
        const unsigned rem = padded[l]->m_blockRemaining;
        if (rem < chunk)
          chunk = rem;
      }
      // Force a chunk boundary onto the mid-capture point so the capture
      // below sees fully-updated per-lane position fields.
      if (midStates && done < captureAt) {
        const unsigned cap = captureAt - done;
        if (cap < chunk)
          chunk = cap;
      }

      for (unsigned c = 0; c < channels; c++) {
        for (unsigned l = 0; l < LANES; l++) {
          GOSoundCompressionCacheV3 &cache = *padded[l];
          const unsigned width = cache.m_curWidth;
          const unsigned j = BLOCK_SAMPLES - cache.m_blockRemaining;
          const unsigned char *ib
            = cache.m_blockData + ((size_t)c * BLOCK_SAMPLES + j) * width;
          int32_t *row = planar[c][l];
          switch (width) {
          case 1:
            extractChannelPlanar<1>(ib, chunk, row);
            break;
          case 2:
            extractChannelPlanar<2>(ib, chunk, row);
            break;
          case 3:
            extractChannelPlanar<3>(ib, chunk, row);
            break;
          default:
            extractChannelPlanar<4>(ib, chunk, row);
            break;
          }
        }
      }

      /* Dispatch the predictor order ONCE per chunk (not per sample): the
       * whole inner render path is instantiated per compile-time ORDER, so
       * the predictor is straight-line hardcoded with no hot-loop branch.
       * This is the "swap the whole render path" the fixed modes need - a
       * fixed-order batch runs codegen identical to the old fixed-3 kernel,
       * while Variable dispatches each uniform-order batch on its own order. */
      auto runOrder = [&](auto orderIC) __attribute__((target("avx2"))) {
        constexpr unsigned ORDER = decltype(orderIC)::value;
        const unsigned chunk8 = chunk - (chunk % 8);
        for (unsigned j = 0; j < chunk8; j += 8) {
          for (unsigned c = 0; c < channels; c++) {
            __m256i rows[8];
            for (unsigned l = 0; l < LANES; l++)
              rows[l] = _mm256_loadu_si256((const __m256i *)(planar[c][l] + j));
            transpose8x8(rows);
            for (unsigned k = 0; k < 8; k++) {
              const __m256i pred
                = predictOrder<ORDER>(value[c], prev[c], last[c], hist3[c]);
              // Maintain only the history taps this order actually reads:
              // orders <= 3 never touch hist3 (v[n-4]), so pushing it is dead
              // work AND extra register pressure vs the old fixed-3 kernel.
              if constexpr (ORDER >= 4)
                hist3[c] = last[c];
              if constexpr (ORDER >= 3)
                last[c] = prev[c];
              // prev is maintained for EVERY order: window contract, see
              // the scalar kernel (GOSoundCompressionCacheV3.h).
              prev[c] = value[c];
              value[c] = _mm256_add_epi32(pred, rows[k]);
              if constexpr (MODE == MODE_INTERLEAVED) {
                _mm256_storeu_si256(
                  (__m256i
                     *)(interleavedOut[c] + (size_t)(done + j + k) * LANES),
                  value[c]);
              } else if constexpr (MODE == MODE_PLANAR) {
                /* Zero register cost: rows[k] (this step's residual) is
                 * dead from here on - reuse its slot to collect the
                 * decoded output row for the tile flip below. */
                rows[k] = value[c];
              } else {
                alignas(32) int32_t tmp[LANES];
                _mm256_store_si256((__m256i *)tmp, value[c]);
                for (unsigned l = 0; l < numLanes; l++)
                  splitOut[l][c][done + j + k] = tmp[l];
              }
            }
            if constexpr (MODE == MODE_PLANAR) {
              // Flip the collected 8 output rows lane-planar and store each
              // lane's contiguous 8-sample run.
              transpose8x8(rows);
              for (unsigned l = 0; l < LANES; l++)
                _mm256_storeu_si256(
                  (__m256i
                     *)(interleavedOut[c] + (size_t)l * planeStride + done + j),
                  rows[l]);
            }
          }
        }
        for (unsigned j = chunk8; j < chunk; j++) {
          for (unsigned c = 0; c < channels; c++) {
            alignas(32) int32_t tmp[LANES];
            for (unsigned l = 0; l < LANES; l++)
              tmp[l] = planar[c][l][j];
            __m256i residual = _mm256_load_si256((const __m256i *)tmp);
            const __m256i pred
              = predictOrder<ORDER>(value[c], prev[c], last[c], hist3[c]);
            if constexpr (ORDER >= 4)
              hist3[c] = last[c];
            if constexpr (ORDER >= 3)
              last[c] = prev[c];
            // prev: every order, window contract (see scalar kernel).
            prev[c] = value[c];
            value[c] = _mm256_add_epi32(pred, residual);
            if constexpr (MODE == MODE_INTERLEAVED) {
              _mm256_storeu_si256(
                (__m256i *)(interleavedOut[c] + (size_t)(done + j) * LANES),
                value[c]);
            } else if constexpr (MODE == MODE_PLANAR) {
              _mm256_store_si256((__m256i *)tmp, value[c]);
              for (unsigned l = 0; l < LANES; l++)
                interleavedOut[c][(size_t)l * planeStride + done + j] = tmp[l];
            } else {
              _mm256_store_si256((__m256i *)tmp, value[c]);
              for (unsigned l = 0; l < numLanes; l++)
                splitOut[l][c][done + j] = tmp[l];
            }
          }
        }
      };
      // GO_V3_FORCE_ORDER: single-instance A/B build, see the scalar
      // dispatch in GOSoundCompressionCacheV3.h.
#ifdef GO_V3_FORCE_ORDER
      (void)batchOrder;
      runOrder(std::integral_constant<unsigned, GO_V3_FORCE_ORDER>{});
#else
      switch (batchOrder) {
      case 1:
        runOrder(std::integral_constant<unsigned, 1>{});
        break;
      case 2:
        runOrder(std::integral_constant<unsigned, 2>{});
        break;
      case 4:
        runOrder(std::integral_constant<unsigned, 4>{});
        break;
      default:
        runOrder(std::integral_constant<unsigned, 3>{});
        break;
      }
#endif

      for (unsigned l = 0; l < LANES; l++) {
        GOSoundCompressionCacheV3 &cache = *padded[l];
        cache.m_blockRemaining -= chunk;
        cache.m_position += chunk;
      }
      done += chunk;

      if (midStates && done == captureAt) {
        /* Mid-run state capture (see DecodeBatchInterleavedMidCapture):
         * the lanes' position fields (m_ptr/m_blockData/m_blockRemaining/
         * m_curWidth/m_position) were just updated for this chunk, but
         * the predictor state still lives in the AVX2 registers - dump
         * those into the captured copies. Real lanes only; padding lanes
         * have no caller-visible state. */
        alignas(32) int32_t v[LANES], p[LANES], la[LANES], h3[LANES];

        for (unsigned l = 0; l < numLanes; l++)
          midStates[l] = *padded[l];
        for (unsigned c = 0; c < channels; c++) {
          _mm256_store_si256((__m256i *)v, value[c]);
          _mm256_store_si256((__m256i *)p, prev[c]);
          _mm256_store_si256((__m256i *)la, last[c]);
          if (needHist3)
            _mm256_store_si256((__m256i *)h3, hist3[c]);
          for (unsigned l = 0; l < numLanes; l++) {
            midStates[l].m_value[c] = v[l];
            midStates[l].m_prev[c] = p[l];
            midStates[l].m_last[c] = la[l];
            if (needHist3)
              midStates[l].m_hist3[c] = h3[l];
          }
        }
      }
    }

    for (unsigned c = 0; c < channels; c++) {
      alignas(32) int32_t v[LANES], p[LANES], la[LANES], h3[LANES];
      _mm256_store_si256((__m256i *)v, value[c]);
      _mm256_store_si256((__m256i *)p, prev[c]);
      _mm256_store_si256((__m256i *)la, last[c]);
      if (needHist3)
        _mm256_store_si256((__m256i *)h3, hist3[c]);
      for (unsigned l = 0; l < numLanes; l++) {
        lanes[l].cache->m_value[c] = v[l];
        lanes[l].cache->m_prev[c] = p[l];
        lanes[l].cache->m_last[c] = la[l];
        if (needHist3)
          lanes[l].cache->m_hist3[c] = h3[l];
      }
    }

  }
};

#endif // GO_V3_BATCH_AVX2

#endif
