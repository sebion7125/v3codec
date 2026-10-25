# Fixed-width planar predictive audio codec with batched SIMD multi-stream decode

A technical disclosure published to establish prior art and foreclose
patenting of the technique described below. Written to be
understandable and reproducible by a person skilled in digital audio /
DSP engineering, independent of any particular host application.

## Abstract

A lossless audio codec and decoding technique for use cases requiring
many independent, differently-positioned compressed audio streams to be
decoded concurrently on SIMD hardware (e.g. a polyphonic software
sampler). Conventional variable-length per-sample codecs decode with a
data-dependent branch, defeating vectorization. The disclosed codec
instead divides each channel's signal into fixed-length blocks, predicts
each sample from a small family of fixed-order integer linear predictors
(orders 1-4), and stores the residuals at a single fixed byte width per
block (1 to 4 bytes, chosen from that block's own peak residual
magnitude) in a channel-planar, not interleaved, layout. This
fixed-stride, branch-free layout lets up to L independent streams (L =
the SIMD register's lane count) be decoded together in one SIMD register
pass via a two-stage per-lane-extraction-then-transpose kernel, one lane
per stream, even when mutually unaligned; the planar
layout and a specific padding rule for partial batches make this apply
uniformly across channel counts and byte widths. A resumable-snapshot
mechanism lets any stream begin decoding at an arbitrary position
without decoding from the start. Measured on a corpus of 4000+ real
audio files (single-channel material), the technique achieved 4.75x the
decode throughput of a representative fixed-escape-width variable-length
reference codec at a better compression ratio (~1.53x vs ~1.43x). On
real two-channel material, format-only and batching-only contributions
were measured separately at ~1.0-1.9x and ~2.4-2.7x (combined, by
multiplication, roughly 2.4-5.1x, under different material and rounding
conventions than the 4.75x figure); in both cases most of the speed-up
comes from batching, not the format change alone.

## Field

Lossless compression and decompression of pulse-code-modulated (PCM)
audio sample data, specifically optimized for the case where **many
independent, differently-positioned streams of the same compressed
format must be decoded concurrently** on commodity SIMD hardware (e.g. a
software synthesizer or sampler rendering dozens to thousands of
simultaneous voices from a large sample library held in memory).

## Background

Conventional lossless audio codecs for this use case (e.g. variable-length
per-sample entropy coding, such as Shorten/FLAC-style residual coding)
achieve good compression ratios but decode one sample at a time with a
data-dependent branch (the entropy code's length varies sample-to-sample).
This defeats SIMD vectorization and is unfriendly to instruction and data
prefetching, because consecutive samples do not have a fixed, predictable
byte stride. When many such streams must be decoded per audio callback
period, each stream being at an arbitrary, independent playback position,
per-sample branching and variable stride dominate decode cost.

## Summary of the technique

The codec described here trades a small amount of compression ratio for
a byte-addressable, **fixed-stride** residual stream, and specifically
shapes the on-disk/in-memory layout so that **N independent streams can
be decoded together in one SIMD register pass**, including the case
where the N streams are at arbitrary, mutually unaligned playback
positions (not merely N channels of one synchronized source). Measured
on real-world organ-pipe sample material, this reached **4.75× the
decode throughput of a representative fixed-escape-width variable-length
per-sample codec** on a 4000-file single-channel corpus, at an
equal-or-better compression ratio (~1.53× vs ~1.43×); on real
two-channel material the format and batching contributions were
separately measured at ~1.0–1.9× and ~2.4–2.7×, a combined range of
roughly 2.4–5.1× (see "Measured results" for the full breakdown and why
these are reported as two distinct data points rather than one reconfirmed
figure). The technique composes two independent ideas, each necessary;
the batching idea supplies the majority of the speed-up in both cases.

## Detailed description

### 1. Block structure and byte-width selection

The input signal is divided into fixed-length blocks of `B` samples per
channel (a practical value is `B = 1024`). For each block, every channel
of that block occupies a **separate, contiguous region** of the output
buffer ("planar" layout): all of channel 0's residuals for the block,
then all of channel 1's, etc. Each channel's region reserves the full
`B` samples' worth of storage regardless of the block's true length
(relevant only for a stream's final, possibly short block); the wasted
tail space is negligible relative to block size and is never read back.

This is in contrast to a conventional **interleaved** layout
(channel-0-sample, channel-1-sample, channel-0-sample, ...), which is the
natural layout for multi-channel audio but which this technique
deliberately avoids, for reasons given in part 3 below.

Within a block, a **predictor** (part 2) computes an estimate of each
sample from preceding samples of the same channel, and the codec stores
the signed residual (true sample minus prediction). The block header
records a single **byte width** for the whole block, chosen as the
smallest width that can represent every residual in that block without
overflow:

```
w = min { k ∈ {1, 2, 3, 4} :
          -2^(8k-1) ≤ residual[c][i] ≤ 2^(8k-1) - 1
          for every channel c and every sample index i in the block }
```

i.e. `w` is computed once per block from that block's own residual
min/max. All residuals in the block are stored at this fixed width,
addressable by

```
offset(c, i) = channel_index(c) × B × w + sample_index(i) × w
```

relative to the block's data start — an arithmetic computation with no
data-dependent branch, unlike a self-describing variable-length code
where each sample's own length must be read before the next sample can
be located.

Because the width is chosen per block from the actual residual range,
blocks over quiet or smoothly-decaying material converge to the smallest
width (1 byte), while blocks over sharp transients use a wider one —
the codec adapts locally without ever requiring a per-sample escape code.

### 2. Predictor family

The predictor is a fixed-order, integer, feed-forward linear predictor of
one of a small family of orders 1–4, using only the stream's own
previously decoded (or, at encode time, previously seen) sample values
`v[n-1], v[n-2], v[n-3], v[n-4]`:

```
order 1:  pred = v[n-1]
order 2:  pred = 2*v[n-1] -  v[n-2]
order 3:  pred = 3*v[n-1] -3*v[n-2] + v[n-3]
order 4:  pred = 4*v[n-1] -6*v[n-2] +4*v[n-3] - v[n-4]
```

These are the coefficients of the finite-difference (binomial/Pascal)
extrapolating polynomial of the respective order — i.e. each order
assumes the signal continues the polynomial trend established by the
preceding samples, evaluated with integer arithmetic only (no
floating-point, no multiplication by an arbitrary coefficient — every
coefficient is a small integer, so the predictor reduces to shifts, adds
and small integer multiplies).

**This specific family of four fixed integer predictors is itself
well-established prior art** — it is the "fixed predictor" set used by
Shorten (1994) and standardized in FLAC (see RFC 9639 §10.2.5, which
gives the identical four formulas and the identical finite-difference
derivation). Nothing in this disclosure claims these four coefficient
sets themselves as novel. What is disclosed below is an empirical
finding about *which* predictor family remains the better choice under
the specific combination of constraints part 1 and part 3 impose
(a block-wide, peak-driven fixed byte width, decoded by a branch-free
batched SIMD kernel) — not the predictor coefficients.

A key, counter-intuitive, measured property of the source material this
technique was developed against (musical instrument sample recordings):
the choice of block byte-width is driven almost entirely by the **peak**
residual magnitude within a block, not by its variance or spectral
content. Consequently, classical variance/least-squares-optimal
predictors (LPC, minimax fitting, adaptive coefficient estimation) do
not out-perform the fixed small-integer-order family above for this
purpose, because they optimize the wrong objective (expected error, not
worst-case/peak error within a fixed-width block) while costing
materially more to compute per sample. The encoder selects, per stream
(or, in a refinement, per contiguous section of a stream — never per
individual block; every block within one section/stream shares one
order), whichever fixed order among 1–4 minimizes total encoded size;
the decoder is given the chosen order out of band (it is not re-derived
from the residual stream) and applies the single corresponding
fixed-order recursion for the whole section. Because the order is
section-level metadata rather than per-block, the per-block header
never needs to carry it, and the resumable snapshot of part 4 — which
is taken within a section, potentially spanning many blocks — correctly
omits it too: it is supplied once, out of band, alongside the section
rather than once per snapshot.

This finding — that the well-known fixed small-integer-order family
beats variance-optimal alternatives specifically on this corpus, under
a peak-driven fixed-width-block constraint — is part of the disclosed
technique (the empirical justification for the design choice), independent
of the batching technique below; the predictor coefficients themselves
are not.

### 3. Batched multi-stream SIMD decode

This is the dominant contributor to the measured speed-up (see
"Measured results" below: roughly 1.0–1.9× from the fixed-width format
alone vs. a further ~1.5–2.7× multiplicative contribution from batching).

Up to `L` independent streams (`L` = the SIMD register's lane count — 8
for a 256-bit register holding 32-bit residual/sample values) are decoded
**together**, one SIMD instruction operating on all `L` lanes at once,
where **lane `i` holds stream `i`'s current sample**, not `L` samples of
one stream. Critically, the `L` streams need not be synchronized: each
may be at an arbitrary, independent decode position within its own
block, including a different byte-width `w` and a different fractional
offset into its block — exactly the situation that arises when `L`
arbitrary voices of a polyphonic instrument are claimed into a batch at
whatever moment each happens to need its next sample, not `L` channels
of one synchronized multi-microphone recording.

Extraction happens in two stages, because no single SIMD instruction can
gather one value each from `L` genuinely independent memory addresses
(one per stream's own, separately-located buffer) directly into a
lane-aligned register — a true hardware gather instruction was measured
to be *slower* than this two-stage approach on the hardware tested, see
"Variations considered":

**Stage A — per-lane contiguous extraction.** For each of the `L` lanes
*separately* (`L` individual operations, one per lane, each reading only
from that one lane's own buffer), a short run of up to 8 consecutive
residual samples is read from that lane's current position and
sign-extended to 32 bits, into a temporary row of `L` such values. For
byte widths 1, 2 and 4 this is a single SIMD sign-extending load
instruction per lane (reading 8 consecutive same-width residuals from
that lane's own contiguous region at once); for byte-width 3 — the only
non-power-of-two width, needed for 24-bit audio — 4 consecutive residuals
are extracted per instruction via a fixed byte-shuffle instruction (e.g.
`PSHUFB` on x86) that re-packs 12 packed input bytes (4 samples × 3
bytes) into 4 sign-extended 32-bit values in one operation, using a
single, position-independent shuffle mask (because the per-lane
extraction always advances in fixed strides of 4 samples, no
phase-dependent mask variant is needed). This stage is *why* the planar
layout matters: it is only because each lane's own residuals are
guaranteed contiguous (part 1) that this extraction can be a plain wide
load or a single fixed shuffle at all, for any channel count and any of
the four widths.

**Stage B — cross-lane transpose.** The `L` per-lane rows from stage A
each hold up to 8 *consecutive samples of one lane*; the predictor
recursion, however, needs registers holding *one sample position across
all `L` lanes* (lane `i` = stream `i`'s current sample). An 8×8 matrix
transpose — a fixed, data-independent sequence of SIMD unpack/shuffle/
permute instructions operating on general-purpose 256-bit registers,
unrelated to the original residual byte-width — converts the `L`
"lane-major" rows from stage A into `L` "sample-major" registers: after
the transpose, register `j` holds sample offset `j` (0 to 7 within the
extracted run) of every one of the `L` lanes. This transpose is what
actually makes batching `L` *independent*, *unsynchronized* streams
together possible; the predictor recursion of part 2 is then evaluated
**once per sample offset, across all `L` lanes simultaneously**, on
these transposed registers, using ordinary SIMD integer add/subtract
instructions — the small-integer coefficients of the fixed-order
predictor make this a fixed, short sequence of vector instructions with
no per-lane branching, because every lane uses the *same* predictor
order in a given batched call (the scheduler groups streams of matching
predictor order into the same batch, a bucketing step orthogonal to the
kernel itself — see the note on predictor-order fragmentation below).

**Why the planar block layout (part 1) is necessary for stage A above,
not merely convenient.** Storing a lossless codec's compressed data
per-channel rather than sample-interleaved is not itself new — several
existing lossless audio codecs already serialize their compressed
subframes one channel at a time rather than interleaving compressed
samples. What specifically matters here, and what a plain per-channel
serialization does not by itself guarantee, is a **fixed, predictable
per-sample byte stride within a block** that stage A's wide loads and
shuffle can rely on. If a stream's own residuals were stored
channel-interleaved at the sample level (one established alternative
layout for multi-channel audio generally), stage A's per-lane extraction
of several consecutive same-channel residuals would require a non-unit
stride read even within a single lane's own buffer — which, combined
with widths other than a power of two (specifically width 3, the common
case for 24-bit source audio), defeats both the sign-extending wide-load
and the byte-shuffle approach above, forcing a fallback to a scalar
per-sample loop for real multi-channel material. Making each channel's
residuals contiguous within a block (the planar layout of part 1, with
its fixed per-block byte width) is what allows stage A's wide-load /
byte-shuffle extraction to apply uniformly, for any channel count, at
any of the four byte widths — measured, on real two-channel material, to
raise the batching contribution from roughly 1.5–1.9× to roughly
2.4–2.7×, with no change to the single-stream (non-batched) decode cost,
confirming the effect is purely a stage-A property of the data layout,
not of the predictor.

A batch smaller than `L` real streams (the common case at the edges of a
polyphony count that is not a multiple of `L`) is handled by padding the
unused lanes with **independent copies** of one real lane's decoder
state (not pointers/aliases to the same state object) — because the
kernel's internal per-lane bookkeeping (block-header refresh, position
advance) mutates each lane's state object in place once per decoded
chunk; aliasing two "lanes" to one underlying object causes that one
object to be mutated multiple times per chunk, corrupting it. This
padding detail is necessary for a correct implementation and is
disclosed as part of the technique.

### 4. Resumable decode from an arbitrary position

Because each stream may need to begin decoding from an arbitrary sample
position (not only from the start of the compressed stream), the encoder
additionally emits, at a set of caller-specified absolute sample
positions, a small fixed-size **snapshot** of the decoder's complete
recursion state at that position (the last 1–4 decoded sample values,
depending on predictor order, plus the current block's byte offset,
remaining-samples-in-block count and byte width). Resuming decode from
such a snapshot reproduces bit-exactly the same output as decoding
sequentially from the stream's start — this was verified by constructing
snapshots at arbitrary, non-block-aligned sample offsets and comparing
resumed decode against full sequential decode on real file material. This
snapshot mechanism is what allows the technique to support features such
as seeking, looped playback re-entry, and starting many streams at
independent positions without having to decode each stream from its
beginning every time — a practical necessity for the batching technique
above, since batched streams are essentially never phase-aligned at
batch-formation time.

**Start-of-stream warm-up.** At the true start of a stream (no snapshot,
no prior history), the predictor's history taps simply have no real
values to read yet. The disclosed implementation initializes all history
taps to zero and begins decoding normally from the first sample; the
first `order-1` samples of the stream are therefore predicted from a
partially- or fully-zeroed history (e.g. an order-3 stream's very first
sample is predicted as 0, its second from one real sample and two
zeros, and so on) rather than from a special-cased start-up procedure.
This is simple and correct — the encoder computes residuals against the
same zeroed-history convention its decoder uses, so the output remains
bit-exact — at the cost of a slightly larger residual (and thus
occasionally a wider block byte-width) for the first few samples of a
stream, which is negligible against a stream's typical length.

## Measured results

This technique's measured results come in two distinct data points,
from two different benchmark configurations, reported separately rather
than merged into one reconfirmed figure (the second, later measurement
campaign did not exactly reproduce the first's combined throughput
figure under its different — stereo, phase-misaligned, revised rounding
convention — conditions, and the project's own working notes explicitly
caution against restating the first figure as achieved under the
second's conditions).

**(1) Single-channel corpus, combined figure.** On a corpus of 4000+
real musical-instrument (pipe organ) sample recordings (single channel
per file), decode throughput of this technique was measured at **4.75×**
the decode throughput of a representative reference codec on the same
hardware (consumer x86-64, AVX2), with a compression ratio of
approximately **1.53×**, versus the reference codec's ~1.43× on the same
corpus (ratio varies by material; busy, high-energy multi-harmonic
signal content can occasionally favor the reference codec's finer
per-sample escape coding). The reference codec here is specifically a
**fixed, small, per-sample escape-width scheme without entropy coding**
(effectively capped at a small integer number of bytes per sample,
structurally distinct from a Rice/Golomb-style entropy coder such as
FLAC's or Shorten's, which has no such fixed per-sample byte floor) —
the comparison and the ratio advantage above are specific to this class
of reference codec, not a general claim against every variable-length
scheme.

**(2) Two-channel corpus, decomposed figures.** On real two-channel
material, phase-misaligned within each batch to model realistic
usage, the format-only and batching-only contributions to throughput
were measured **separately**: decoding the 8 lanes' individual streams
sequentially, one after another, with the fixed-width/planar format but
without batching them together, measured only about **1.0–1.9×** the
reference codec's throughput; the SIMD batching step's own further,
multiplicative contribution on top of that was measured at **2.4–2.7×**
(see part 3's detailed description of the planar-layout effect on this
figure). Multiplying the two gives an implied combined range of roughly
**2.4–5.1×** for this configuration — broadly consistent with, but not a
direct re-confirmation of, data point (1)'s single 4.75× figure, which
used mono material and an earlier predictor rounding convention. In
both data points, the large majority of the overall speed-up comes from
the batching step, not the fixed-width format change alone.

A second, independent compression-ratio data point from real deployed
use (a complete musical-instrument sample library, loaded in full,
compression ratio read from the resulting in-memory footprint rather
than a benchmark harness): this technique compressed the library to
approximately **56% of its original PCM size (≈1.78× ratio)**, against
the reference codec's ≈67% (≈1.49× ratio, consistent with that codec's
known ≈1.5× structural ceiling) on the identical material — a **16%
reduction in total memory footprint** achieved as a side effect of a
technique whose design objective was decode throughput, not ratio. The
higher ratio on this corpus relative to the 4000-file average above is
consistent with the per-block width adapting all the way down to 1 byte
on long, quiet, smoothly-decaying material (e.g. release tails), a
regime the fixed-width scheme reaches but a 2-bytes-per-sample-minimum
scheme structurally cannot.

## Variations considered (disclosed for completeness)

- **Decode-time regrouping ("bucketing") of an already-batched runtime
  workload by each stream's currently-active byte width**, read from
  each stream's own block header before batching (no format change):
  measured to add at most ~0–10% on top of the base batching technique,
  at the cost of materially more scheduling complexity (re-bucketing
  streams roughly every block); judged not clearly worthwhile as a
  general-purpose refinement, but disclosed as an available extension.
- **A shared, single byte-width across multiple *synchronized* streams
  of the same source event** (e.g. several simultaneous microphone
  perspectives of one acoustic event that are always decoded together as
  a fixed group): forcing one shared width instead of each stream's own
  independently-chosen width was measured to cost negligible compression
  (~0.07% on real multi-microphone material) while enabling a simpler,
  branch-free interleaved decode kernel — legitimate specifically because
  such streams are *known in advance* to always be decoded together,
  unlike the arbitrary, independently-claimed streams of part 3. Due to
  the fixed per-instruction lane count of the SIMD hardware, this
  variant's net benefit depends on how many real streams fill the
  register:

  ```
  cost_per_real_stream = cost_per_lane × (L / N_real)
  ```

  where `L` is the hardware's lane count and `N_real` the number of real
  (non-padding) streams sharing the batch — so it is advantageous only
  when `N_real` is close to `L`; with materially fewer synchronized
  streams than lanes, the independent-stream batching technique of part
  3 remains preferable. Equating the two techniques' per-real-stream
  cost (measured nominal values of ~0.33 ns/stream shared-width vs.
  ~0.38 ns/stream for part 3's technique, on the hardware measured)
  gives a break-even `N_real` of approximately `0.87 × L` — i.e. roughly
  7 of 8 lanes on 256-bit SIMD — below which part 3 is preferable.
- **Interaction between predictor-order selection and `N_real`.** Part 3
  requires every lane in one batched call to share the same predictor
  order, so a scheduler grouping streams into batches must also group
  by order. When the order is fixed session-wide (part 2's basic case),
  this imposes no fragmentation at all — every stream always matches.
  When the order instead varies per section (part 2's refinement), the
  scheduler's pool of streams ready to batch together is implicitly
  split across up to 4 order-groups at any moment, which can reduce the
  realized `N_real` below what it would be without that refinement, and
  interacts directly with the padding behaviour of part 3 and the
  break-even analysis above. This disclosure describes the grouping
  requirement itself (part 3) and this consequence of it, but not any
  particular scheduling policy for forming such groups, which is
  host-application territory (see "Scope note").

## Reference implementation and reproducibility

An open-source reference implementation of this codec — the exact
on-disk block/header format, the single-stream encoder/decoder, the
AVX2 batched multi-stream decode kernel, and the benchmark harness used
to produce the figures in "Measured results" above — is published
alongside this disclosure:

- Source repository: `https://github.com/sebion7125/v3codec` (private
  until this disclosure's own publication; the URL itself does not
  change when visibility is flipped to public)

Where this disclosure describes a mechanism only at the level of a
general technique (e.g. "a batch kernel reads one wide, sign-extending
load per lane"), the reference implementation is the authoritative,
unambiguous specification of every exact detail — byte layout, the
per-width dispatch, the partial-batch padding behaviour of part 3, and
the benchmark methodology and test corpus underlying the measured
results — reproducible by building and running it directly.

## Scope note

This disclosure describes the codec format, predictor family, and the
multi-stream SIMD batched decode technique (including the planar layout
and padding requirement that make it correct and performant) as a
general audio-decoding technique, independent of any specific host
application, operating system, or hardware vendor. It does not disclose,
and makes no claim regarding, any particular application's scheduling,
voice-allocation, or real-time task-management architecture around the
codec — only the codec and its batched decode technique themselves.
