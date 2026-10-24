/*
 * V3: fixed-width planar predictive audio codec with batched SIMD
 * multi-stream decode. See
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

#ifndef GOSOUNDCOMPRESSIONCACHEV3_H
#define GOSOUNDCOMPRESSIONCACHEV3_H

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

/* Self-contained library config - the only two knobs a host project sets.
 * Deliberately has NO dependency on the embedding project (GrandOrgue's
 * own GOV3PredictorMode/MAX_OUTPUT_CHANNELS stay separate GO-side types;
 * GOSoundAudioSection.h is the adapter that casts between them), so this
 * header and GOSoundCompressionCacheV3Batch.h can be lifted into their own
 * repository unchanged. */
#ifndef V3CODEC_MAX_CHANNELS
#define V3CODEC_MAX_CHANNELS 2
#endif

enum class V3PredictorMode : uint8_t {
  Variable = 0,
  Fixed1 = 1,
  Fixed2 = 2,
  Fixed3 = 3,
  Fixed4 = 4,
};

// Second codec generation for GOSoundCompressionCache's on-disk format.
//
// Undamped 2nd-order predictor (full linear extrapolation):
//   value[n] = residual[n] + 2*prev[n-1] - last[n-1].
// Cache version 4 (2026-07-13): the codec originally shipped the legacy
// codec's DAMPED 2nd-order shape, prev + ((prev - last) >> 1) - a half-
// slope extrapolation inherited unchanged. Measuring predictor orders on
// the whole real Rotterdam set (tools/compression/predictor_order_study,
// predictor_decode_bench) showed the full undamped slope leaves ~2-7%
// smaller residuals - and it is one op CHEAPER to decode (2*prev - last
// is slli+sub; the damped form was sub+shift+add, with the shift on the
// recurrence's critical path). A strict win on both size and decode, so
// the predictor was switched. Higher fixed orders (3-4) shrink more but
// cost a little decode; adaptive LPC helps the loud stops most but its
// 64-bit accumulation loses the decode budget - both declined. Encoder
// and decoder use the same predictor, so decoded PCM stays bit-exact to
// the source samples; only the residual stream changes. This format is a
// new, versioned, from-scratch cache representation; there is no
// requirement (and no benefit) to match legacy's residual stream
// bit-for-bit.
//
// The residual itself is stored at a FIXED byte width (1/2/3/4 bytes)
// chosen once per 1024-sample block from that block's own min/max
// residual, instead of legacy's per-sample self-describing variable-length
// code. This removes the per-sample branch and irregular stride that make
// the legacy format unfriendly to the prefetcher and to SIMD. See
// tools/compression/INVESTIGATION.md for the full measurement history.
//
// Block layout is PLANAR PER CHANNEL, not interleaved: a block is
// [width:1 byte][channel 0's BLOCK_SAMPLES residuals, `width` bytes
// each][channel 1's BLOCK_SAMPLES residuals]...  Every channel's region
// always reserves the FULL BLOCK_SAMPLES slots, even in a section's final,
// possibly-shorter block (the unused tail slots hold unread filler) - this
// keeps a channel's region start a fixed, width-only-dependent offset
// (c * BLOCK_SAMPLES * width) from the block's data start, with no need to
// know the block's true sample count to address it. The waste is at most
// (BLOCK_SAMPLES - trueLength) * channels * width bytes, and only for the
// very last block of a section - negligible against real section sizes.
//
// This is planar specifically so that each channel's residuals are
// byte-contiguous: an EARLIER interleaved design (channel 0 and channel 1's
// residuals alternating per sample) made GOSoundCompressionCacheV3Batch's
// width==3 fast path (a 4-packed-samples-per-load PSHUFB trick) inapplicable
// to real (stereo) content, because it only works on contiguous data -
// falling back to a scalar per-sample loop for the common width==3 case on
// real 24-bit stereo material. Planar-per-channel makes every channel's
// region contiguous regardless of channel count, so the fast path (and,
// with this rewrite, direct SIMD sign-extending loads for widths 1/2/4 too)
// always applies. See GOSoundCompressionCacheV3Batch.h.
//
// API mirrors GOSoundCompressionCache deliberately (Init/DecompressionStep/
// DecompressTo/m_position/m_value/m_prev/m_last) so call sites that are
// templated on the cache type (GOSoundStream's window classes) work
// unchanged with either codec.
class GOSoundCompressionCacheV3 {
public:
  static constexpr unsigned BLOCK_SAMPLES = 1024;

  unsigned m_position;
  int m_value[V3CODEC_MAX_CHANNELS];
  int m_last[V3CODEC_MAX_CHANNELS];
  int m_prev[V3CODEC_MAX_CHANNELS];
  /* Seqlock sequence for cross-thread SNAPSHOTS of this live decode state
   * (GetHistory in the release task copies it while a render worker may be
   * mutating it - found 2026-07-16 via a minidump: a copy torn across the
   * block refresh produced sample index 11232 of max 2048 and a wild read).
   * Writers (single writer per voice) bump it to ODD before a multi-field
   * update and back to EVEN after; readers retry while odd/changed. Fences
   * are compiler-only on x86 (TSO orders stores) - zero hot-path cost, and
   * the field sits in the padding hole after m_prev: sizeof stays 56.
   * NOT serialized (snapshots on disk never contain a mid-update state). */
  std::atomic<uint16_t> m_seq{0};
  /* The per-cache predictor order lives in the same 4-byte padding hole
   * as m_seq (2 + 1 + 1 spare): sizeof stays 56. See the mode comment. */
  uint8_t m_order;

  /* std::atomic deletes the implicit copy operations, but copies of this
   * class are exactly the point (GetHistory's snapshot, the batch decoder's
   * padding scratch, mid-run captures). Copy everything EXCEPT the sequence:
   * a copy is a private snapshot - its own seq starts consistent at 0. */
  GOSoundCompressionCacheV3() = default;
  GOSoundCompressionCacheV3(const GOSoundCompressionCacheV3 &o) { *this = o; }
  GOSoundCompressionCacheV3 &operator=(const GOSoundCompressionCacheV3 &o) {
    m_position = o.m_position;
    for (unsigned c = 0; c < V3CODEC_MAX_CHANNELS; c++) {
      m_value[c] = o.m_value[c];
      m_last[c] = o.m_last[c];
      m_prev[c] = o.m_prev[c];
      // Copied unconditionally (branch-free); only order-4 paths READ it.
      m_hist3[c] = o.m_hist3[c];
    }
    m_order = o.m_order;
    m_blockData = o.m_blockData;
    m_ptr = o.m_ptr;
    m_blockRemaining = o.m_blockRemaining;
    m_curWidth = o.m_curWidth;
    return *this;
  }

  /* Per-section fixed predictor order (cache version 5, 2026-07-13). The
   * predictor is pred[n] = sum PRED_COEF[order][j] * v[n-1-j], the fixed
   * binomial (Pascal) polynomial predictors:
   *   1: v[n-1]                 (hold)
   *   2: 2 v[n-1] - v[n-2]      (undamped linear - the v4 default)
   *   3: 3 v[n-1]-3 v[n-2]+v[n-3]
   *   4: 4 v[n-1]-6 v[n-2]+4 v[n-3]-v[n-4]
   * The encoder picks, per section, the order giving the smallest cache
   * (whole-Rotterdam study: order choice is peak/width-driven, so pure
   * SELECTION among these integer predictors beats any coefficient fit -
   * and integer taps decode with adds/small muls, no float, no 64-bit).
   * Taps map to [m_value, m_prev, m_last, m_hist3] (v[n-1..n-4]) when the
   * predictor is evaluated BEFORE the per-sample state shift. */
  static constexpr int PRED_COEF[5][4] = {
    {0, 0, 0, 0}, {1, 0, 0, 0}, {2, -1, 0, 0}, {3, -3, 1, 0}, {4, -6, 4, -1}};

  /* Global predictor order, set from config before load and constant for the
   * session (it is folded into the cache hash, so a change forces a rebuild).
   * The order is implicit for the whole cache, so the per-block header is
   * just [width:1] - byte-for-byte the v6 stream. */
  // Default Fixed3: the best all-round order.
  static inline std::atomic<uint8_t> s_PredictorMode{
    (uint8_t)V3PredictorMode::Fixed3};

  static void SetPredictorMode(V3PredictorMode mode) {
    s_PredictorMode.store((uint8_t)mode, std::memory_order_relaxed);
  }
  static V3PredictorMode GetPredictorMode() {
    return (V3PredictorMode)s_PredictorMode.load(std::memory_order_relaxed);
  }
  static bool IsVariableOrder() {
    return GetPredictorMode() == V3PredictorMode::Variable;
  }
  // Per-block header is [width:1] in EVERY mode - the order is section
  // metadata (v11 envelope), never per block.
  static constexpr unsigned HeaderBytes() { return 1; }


  // Pointer to the START OF CHANNEL 0's region of the block currently being
  // read (i.e. just past that block's width byte). Channel c's region
  // starts at m_blockData + c * BLOCK_SAMPLES * m_curWidth. Only valid
  // while m_blockRemaining > 0; set together with m_curWidth whenever a new
  // block header is read.
  const unsigned char *m_blockData;

  // Pointer to the NEXT block's width byte - i.e. m_blockData plus every
  // channel's full BLOCK_SAMPLES-wide region for the CURRENT block. Read
  // once the current block's m_blockRemaining reaches 0. Also doubles as
  // the "not yet initialized to a real buffer" / backward-seek-reset marker
  // (nullptr), same role as legacy's m_ptr.
  const unsigned char *m_ptr;

  // Samples remaining in the block currently being read (including the one
  // about to be decoded; always counts down from BLOCK_SAMPLES regardless
  // of the block's true encoded length - see the class comment), and that
  // block's byte width. Both are part of the resumable decode state: a
  // cache captured mid-stream (e.g. a StartSegment snapshot, or by
  // GOSoundStream's read-ahead window) must remember where it is inside
  // the current block.
  unsigned m_blockRemaining;
  unsigned m_curWidth;

  /* Fourth predictor tap v[n-4] - needed only for order 4. Deliberately the
   * LAST field (2026-07-17 cost isolation): every pre-existing field keeps
   * its offset, and orders <= 3 never read or write it (if constexpr), so
   * the fixed-order paths touch bit-for-bit the same cache lines as the
   * 56-byte v10 struct. sizeof grows 56 -> 64; the arena slot (960 B)
   * absorbs it, the sampler stays at exactly 14 lines. NOT part of the
   * 40-byte Snapshot - order-4 seeds keep it in parallel arrays
   * (GOSoundAudioSection, cache v11). */
  int m_hist3[V3CODEC_MAX_CHANNELS];

  static inline int bytesNeededSigned(int32_t minV, int32_t maxV) {
    if (minV >= -128 && maxV <= 127)
      return 1;
    if (minV >= -32768 && maxV <= 32767)
      return 2;
    if (minV >= -8388608 && maxV <= 8388607)
      return 3;
    return 4;
  }

  // width must be a genuine compile-time constant at the call site (all
  // call sites below switch on it) - memcpy with a *runtime* length (e.g.
  // "memcpy(&v, p, width)" with width a plain unsigned) compiles to a real
  // function call on every sample instead of a single inlined load, which
  // was measured to cost ~7x production's decode speed. Never widen these
  // back to a runtime-width helper.
  __attribute__((always_inline)) static inline int32_t readResidual(
    const unsigned char *p, unsigned width) {
    switch (width) {
    case 1:
      return (int32_t)(int8_t)p[0];
    case 2: {
      int16_t v;
      std::memcpy(&v, p, 2);
      return (int32_t)v;
    }
    case 3: {
      uint32_t raw
        = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
      const int32_t shifted = (int32_t)(raw << 8);
      return shifted >> 8;
    }
    default: {
      int32_t v;
      std::memcpy(&v, p, 4);
      return v;
    }
    }
  }

  static inline void writeResidual(
    unsigned char *p, unsigned width, uint32_t v) {
    switch (width) {
    case 1:
      p[0] = (unsigned char)v;
      break;
    case 2:
      std::memcpy(p, &v, 2);
      break;
    case 3:
      p[0] = (unsigned char)v;
      p[1] = (unsigned char)(v >> 8);
      p[2] = (unsigned char)(v >> 16);
      break;
    default:
      std::memcpy(p, &v, 4);
      break;
    }
  }

  inline void Init() {
    m_position = 0;
    m_blockData = nullptr;
    m_ptr = nullptr;
    m_blockRemaining = 0;
    m_curWidth = 0;
    // Default order 2 (the v4 undamped predictor); the section overrides
    /* Fixed modes: the order IS the mode. Variable (0): 0 is not a valid
     * order - every real decode entry (LoadSnapshot, DecompressTo) then
     * overwrites m_order with the SECTION's order; the 3 here is only a
     * safe fallback for order-agnostic resets. Set ONCE here - never per
     * block: reading the atomic in the block refresh cost ~1.1M loads/s
     * at 24k voices AND acted as an optimization barrier in the hot
     * refresh path, which the hardcoded fixed-order build never had. */
    const uint8_t mode = (uint8_t)GetPredictorMode();
    m_order = mode ? mode : 3;
    for (unsigned j = 0; j < V3CODEC_MAX_CHANNELS; j++) {
      m_last[j] = 0;
      m_value[j] = 0;
      m_prev[j] = 0;
      m_hist3[j] = 0;
    }
  }

  inline void DecompressionStep(unsigned channels) {
    if (m_blockRemaining == 0) {
      // Per-block header: [width:1]; the order is implicit and already set
      // (Init/RestoreFrom) - do NOT re-read it here, see Init().
      // Seqlock write side (see m_seq): odd = fields are mid-update.
      const uint16_t seq0 = m_seq.load(std::memory_order_relaxed);
      m_seq.store((uint16_t)(seq0 + 1), std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_release);
      m_curWidth = *m_ptr;
      m_blockData = m_ptr + 1;
      m_ptr = m_blockData + (size_t)channels * BLOCK_SAMPLES * m_curWidth;
      m_blockRemaining = BLOCK_SAMPLES;
      std::atomic_thread_fence(std::memory_order_release);
      m_seq.store((uint16_t)(seq0 + 2), std::memory_order_relaxed);
    }
    const unsigned width = m_curWidth;
    const unsigned j = BLOCK_SAMPLES - m_blockRemaining;
    /* Planar layout: each channel's residuals live in their own region,
     * BLOCK_SAMPLES*width apart - one prefetch per CHANNEL region, not just
     * channel 0's, or every stereo sample stalls on a cold second line
     * (planar's known scalar-decode cost; the batch kernel is where the
     * layout pays off). */
    for (unsigned c = 0; c < channels; c++)
      __builtin_prefetch(
        m_blockData + ((size_t)c * BLOCK_SAMPLES + j) * width + 128, 0, 0);
    // Fixed predictor of this block's order; taps [v[n-1..n-4]] =
    // [m_value, m_prev, m_last, m_hist3] evaluated before the state shift.
    // Dispatch ONCE per sample on m_order (constant per block, so perfectly
    // predicted) into a compile-time-ORDER body: the coefficients become
    // literals (3*x -> lea, no runtime imul) and lower orders skip the
    // history taps they never read - codegen identical to the old hardcoded
    // fixed-order kernel. Bit-identical to the PRED_COEF coefficient form.
    auto step = [&](auto orderIC) {
      constexpr unsigned ORDER = decltype(orderIC)::value;
      for (unsigned c = 0; c < channels; c++) {
        const unsigned char *p
          = m_blockData + ((size_t)c * BLOCK_SAMPLES + j) * width;
        const int32_t residual = readResidual(p, width);
        int32_t pred;
        if constexpr (ORDER == 1)
          pred = m_value[c];
        else if constexpr (ORDER == 2)
          pred = 2 * m_value[c] - m_prev[c];
        else if constexpr (ORDER == 3)
          pred = 3 * m_value[c] - 3 * m_prev[c] + m_last[c];
        else // order 4: 4 v[n-1] - 6 v[n-2] + 4 v[n-3] - v[n-4]
          pred = 4 * m_value[c] - 6 * m_prev[c] + 4 * m_last[c] - m_hist3[c];
        if constexpr (ORDER >= 4)
          m_hist3[c] = m_last[c];
        if constexpr (ORDER >= 3)
          m_last[c] = m_prev[c];
        /* m_prev is maintained for EVERY order - not just >= 2: it is part
         * of the public window contract (StreamCacheWindow::NextSample
         * reads m_prev/m_value as the samples v[n-1]/v[n] for linear
         * interpolation). Gating it on ORDER >= 2 left order 1 serving a
         * constant 0 as v[n-1] - a silent pre-existing bug the four-order
         * test sweep caught 2026-07-17. m_last/m_hist3 stay gated: they
         * have no consumer outside the predictor itself. */
        m_prev[c] = m_value[c];
        m_value[c] = pred + residual;
      }
    };
    /* GO_V3_FORCE_ORDER: A/B experiment build (2026-07-17) - pins the
     * dispatch to one compile-time order so only ONE kernel instance
     * exists in the binary, codegen-equivalent to the hardcoded builds.
     * Measures whether the runtime-order feature itself costs anything
     * (extra instances = I-cache footprint / loop alignment shifts). */
#ifdef GO_V3_FORCE_ORDER
    step(std::integral_constant<unsigned, GO_V3_FORCE_ORDER>{});
#else
    switch (m_order) {
    case 1:
      step(std::integral_constant<unsigned, 1>{});
      break;
    case 2:
      step(std::integral_constant<unsigned, 2>{});
      break;
    case 4:
      step(std::integral_constant<unsigned, 4>{});
      break;
    default:
      step(std::integral_constant<unsigned, 3>{});
      break;
    }
#endif
    m_blockRemaining--;
    m_position++;
  }

  /* `order` is the owning SECTION's predictor order - callers without a
   * better source may pass 0 to keep Init()'s mode-derived default (only
   * correct in the fixed modes; the section-owned call site passes the
   * real order, which is what makes Variable work on this path too). */
  inline void DecompressTo(
    unsigned position,
    const unsigned char *data,
    unsigned channels,
    uint8_t order = 0) {
    if (!m_ptr || m_position > position + 1) {
      Init();
      m_ptr = data;
      if (order)
        m_order = order;
    }
    while (m_position <= position)
      DecompressionStep(channels);
  }

  // A fully resumable decode state, as needed for GOSoundAudioSection's
  // per-loop StartSegment bookmarks: everything DecompressionStep needs to
  // continue decoding correctly from an arbitrary sample position,
  // including one that falls in the middle of a block (a block's byte
  // width is only known once the whole block has been scanned, so this can
  // only be produced by Encode() after the fact, not predicted in advance).
  //
  // `blockDataOffset` follows the same convention GOSoundAudioSection/
  // GOSoundStream already use for the legacy cache's m_ptr: it is a plain
  // offset from the start of the encoded buffer (not a real pointer), so it
  // survives being written to/read from an organ cache file where the real
  // buffer address differs between the encoding run and every later load.
  // LoadSnapshot() rebases it, exactly like
  // GOSoundStream::GoToStartSegment does for the legacy format's
  // "cache.m_ptr = pData + (intptr_t)cache.m_ptr".
  struct Snapshot {
    unsigned position;
    unsigned blockDataOffset;
    int value[V3CODEC_MAX_CHANNELS];
    int prev[V3CODEC_MAX_CHANNELS];
    int last[V3CODEC_MAX_CHANNELS];
    unsigned blockRemaining;
    unsigned curWidth;
    /* NO predictor order here: it is global and session-constant (see
     * s_PredictorMode), so storing it per snapshot only made every snapshot
     * 4 bytes bigger - and snapshots are read on every voice start and every
     * block-state rebuild, i.e. exactly the DRAM traffic the polyphony
     * ceiling is bound by. Measured 2026-07-15: carrying it cost ~7% per
     * voice-frame against the build that never had it. */
  };

  // Rebases a Snapshot onto the real, currently-mapped buffer and loads it
  // into *this, ready to decode forward from `snap.position` via
  // DecompressTo/DecompressionStep. `channels` must match the channel
  // count the section was encoded with (needed to locate the next block's
  // header byte).
  inline void LoadSnapshot(
    const Snapshot &snap,
    const unsigned char *data,
    unsigned channels,
    uint8_t order,
    const int *hist3 = nullptr) {
    /* NO seq bump here: LoadSnapshot on a LIVE state runs inside
     * GoToStartSegment, which holds the seqlock around the WHOLE segment
     * switch (stream fields included). Nested odd/even bumps would make the
     * counter even mid-update and fool readers. Copies have private seqs.
     *
     * `order` is the SECTION's predictor order (stage 1: == the mode for
     * every section; per-section Variable plugs in here). `hist3` is the
     * seed's 4th tap from the section's parallel array - only order-4
     * sections have one; everyone else passes null and never reads the
     * zeroed taps. */
    m_position = snap.position;
    m_blockData = data + snap.blockDataOffset;
    m_ptr
      = m_blockData + (size_t)channels * BLOCK_SAMPLES * snap.curWidth;
    m_blockRemaining = snap.blockRemaining;
    m_curWidth = snap.curWidth;
    m_order = order;
    for (unsigned c = 0; c < V3CODEC_MAX_CHANNELS; c++) {
      m_value[c] = snap.value[c];
      m_prev[c] = snap.prev[c];
      m_last[c] = snap.last[c];
      m_hist3[c] = hist3 ? hist3[c] : 0;
    }
  }

  // ---------------------------------------------------------------------
  // Encoding
  // ---------------------------------------------------------------------
  //
  // Encodes ALL `frames` sample frames of a section in one continuous pass
  // starting at position 0 (matching GOSoundAudioSection::Compress, which
  // encodes one continuous stream for the whole section and treats
  // StartSegments purely as alternate resumable entry points into it, not
  // as separate re-encoded regions). getSample(absoluteIndex, channel)
  // supplies the source PCM values.
  //
  // `snapshotPositions` must be sorted ascending and lists every absolute
  // sample index a resumable Snapshot is needed for (i.e. every
  // StartSegment's start_offset); `outSnapshots` receives one Snapshot per
  // entry, in the same order. Position 0 does not need to be listed
  // specially - list it like any other segment start if a snapshot for it
  // is wanted, since Init()+decode-from-scratch is also always valid.
  //
  // Returns the number of bytes written; `data` must have room for the
  // worst case, WorstCaseSize(frames, channels) bytes.
  // `order` (1..4) selects the fixed predictor (see PRED_COEF); the block
  // header carries no order byte - the order is section metadata.
  // `outSnapHist3` (nullable) receives each snapshot's 4th tap (v[n-4]) in
  // a PARALLEL array - it is deliberately not part of Snapshot (40 bytes,
  // see the struct comment); only order-4 callers pass a buffer.
  template <typename GetSampleFn>
  inline unsigned Encode(
    unsigned char *data,
    unsigned frames,
    unsigned channels,
    unsigned order,
    const unsigned *snapshotPositions,
    unsigned numSnapshots,
    GetSampleFn &&getSample,
    Snapshot *outSnapshots,
    int (*outSnapHist3)[V3CODEC_MAX_CHANNELS] = nullptr) {
    Init();
    const int *const cf = PRED_COEF[order];
    unsigned output_len = 0;
    unsigned nextSnapIn = 0; // index into snapshotPositions/outSnapshots
    int32_t residualBuf[BLOCK_SAMPLES * V3CODEC_MAX_CHANNELS];

    struct Pending {
      unsigned outIdx;
      unsigned j;
      int value[V3CODEC_MAX_CHANNELS];
      int prev[V3CODEC_MAX_CHANNELS];
      int last[V3CODEC_MAX_CHANNELS];
      int hist3[V3CODEC_MAX_CHANNELS];
    };
    std::vector<Pending> pending;
    pending.reserve(numSnapshots);

    for (unsigned base = 0; base < frames; base += BLOCK_SAMPLES) {
      const unsigned n = std::min(BLOCK_SAMPLES, frames - base);
      int32_t minV = 0;
      int32_t maxV = 0;
      pending.clear();

      for (unsigned j = 0; j < n; j++) {
        const unsigned absIdx = base + j;
        if (nextSnapIn < numSnapshots && snapshotPositions[nextSnapIn] == absIdx) {
          Pending p;
          p.outIdx = nextSnapIn;
          p.j = j;
          for (unsigned c = 0; c < channels; c++) {
            p.value[c] = m_value[c];
            p.prev[c] = m_prev[c];
            p.last[c] = m_last[c];
            p.hist3[c] = m_hist3[c];
          }
          pending.push_back(p);
          nextSnapIn++;
        }
        for (unsigned c = 0; c < channels; c++) {
          const int32_t val = getSample(absIdx, c);
          // Fixed predictor of `order`, evaluated BEFORE the state shift
          // (taps v[n-1..n-4] = m_value/m_prev/m_last/m_hist3) - must match
          // DecompressionStep bit-for-bit. cf[3] is 0 for order <= 3, so
          // the hist3 term vanishes there (offline path - no gating).
          const int32_t pred = cf[0] * m_value[c] + cf[1] * m_prev[c]
            + cf[2] * m_last[c] + cf[3] * m_hist3[c];
          const int32_t residual = val - pred;

          residualBuf[c * BLOCK_SAMPLES + j] = residual;
          if (j == 0 && c == 0) {
            minV = maxV = residual;
          } else {
            minV = std::min(minV, residual);
            maxV = std::max(maxV, residual);
          }
          m_hist3[c] = m_last[c];
          m_last[c] = m_prev[c];
          m_prev[c] = m_value[c];
          m_value[c] = val;
        }
      }

      const unsigned width = (unsigned)bytesNeededSigned(minV, maxV);
      // Per-block header: [width:1] only; `order` is implicit == the mode,
      // so this is byte-for-byte the v6 stream.
      data[output_len++] = (unsigned char)width;
      const unsigned blockDataStart = output_len;
      for (unsigned c = 0; c < channels; c++) {
        for (unsigned j = 0; j < n; j++) {
          writeResidual(
            data + output_len, width,
            (uint32_t)residualBuf[c * BLOCK_SAMPLES + j]);
          output_len += width;
        }
        // Filler for this block's unused tail slots (only ever non-zero
        // range on the section's final, possibly-shorter block) - written
        // as 0 so the region is well-defined; never read back since
        // callers never decode past the section's true SampleCount.
        for (unsigned j = n; j < BLOCK_SAMPLES; j++) {
          writeResidual(data + output_len, width, 0);
          output_len += width;
        }
      }

      for (const auto &p : pending) {
        Snapshot &s = outSnapshots[p.outIdx];
        s.position = base + p.j;
        s.blockDataOffset = blockDataStart;
        s.blockRemaining = BLOCK_SAMPLES - p.j;
        s.curWidth = width;
        for (unsigned c = 0; c < channels; c++) {
          s.value[c] = p.value[c];
          s.prev[c] = p.prev[c];
          s.last[c] = p.last[c];
          if (outSnapHist3)
            outSnapHist3[p.outIdx][c] = p.hist3[c];
        }
      }

      m_position = base + n;
    }
    m_blockData = nullptr;
    m_ptr = nullptr;
    m_blockRemaining = 0;
    m_curWidth = 0;
    return output_len;
  }

  // Worst-case byte size for encoding `frames` frames of `channels`
  // channels, used by callers to size the output buffer. Every block
  // reserves the full BLOCK_SAMPLES per channel regardless of the true
  // block length (see the class comment), so this is exact, not just an
  // upper bound, for width==4 blocks - and a true worst case otherwise.
  static inline unsigned WorstCaseSize(unsigned frames, unsigned channels) {
    const unsigned numBlocks = (frames + BLOCK_SAMPLES - 1) / BLOCK_SAMPLES;
    // Block header is 1 byte ([width]); the order is implicit.
    return numBlocks * (HeaderBytes() + BLOCK_SAMPLES * channels * 4);
  }
};

/* SIZE GATES - these two numbers are load-bearing, not incidental.
 *
 * Both structs sit on the polyphony ceiling's critical path: the cache
 * object is embedded per stream (and thus per sampler slot, see
 * SAMPLER_SLOT_SIZE in GOSoundSamplerPool.cpp), and snapshots are read on
 * every voice start and every block-state rebuild - i.e. exactly the DRAM
 * traffic the ceiling is bound by. Growth here is not paid in the code
 * that grew it, it is paid globally and silently.
 *
 * The cost is measured, not hypothetical: the v10 format carried the
 * predictor order per snapshot, four bytes that took Snapshot 40 -> 44 and
 * cost ~7% per voice-frame (24012 voices: 0.818 -> 0.875, see
 * GOSoundCompressionCacheV3.h's Snapshot comment and commit 2652a3063).
 * Order 4 was reintroduced afterwards specifically WITHOUT reopening that
 * cost: m_hist3 went to the END of the class so orders <= 3 keep their
 * offsets, and the order-4 seeds live in index-parallel side arrays rather
 * than in the snapshot (commit 1ed770a63).
 *
 * So: if one of these fires, the fix is almost never to update the number.
 * It is to move the new member out of the hot struct - to the end, to a
 * side array, or to section metadata. Compare 27a165b66, where report
 * statistics accidentally grew GOSoundAudioSection and had to be undone.
 *
 * Conditioned on V3CODEC_MAX_CHANNELS so a channel-count change reports as
 * a deliberate layout change rather than as a spurious failure. */
static_assert(
  V3CODEC_MAX_CHANNELS != 2 || sizeof(GOSoundCompressionCacheV3) == 64,
  "GOSoundCompressionCacheV3 must stay 64 bytes - it is embedded per "
  "stream and per sampler slot; see the size-gate comment above");
static_assert(
  V3CODEC_MAX_CHANNELS != 2
    || sizeof(GOSoundCompressionCacheV3::Snapshot) == 40,
  "Snapshot must stay 40 bytes - growing it to 44 cost ~7% per voice-frame "
  "in v10; see the size-gate comment above");

#endif
