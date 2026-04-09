# NABU JAM

A port of [JAMBO](https://github.com/marspa73/atarijam) (Atari 800, 6502 assembly)
to the NABU Personal Computer, written in C and compiled with z88dk for CP/M.

The original `jam/` folder contains the Atari source unchanged. The NABU port
lives in `nabu/`.

---

## What it is (all directly from JAM)

JAMBO is a self-contained 2-layer neural network language model that runs
entirely on an 8-bit CPU with no floating point and no external libraries.
It takes a text prompt and generates a response one character at a time using
a packed 18 KB weight file trained on short Q&A style exchanges.

Architecture: 192 inputs -> 256 hidden (ReLU) -> 45 output tokens (chars + EOL)

---

## NABU port

Targets CP/M 2.2 and was tested on Cloud CP/M 9.6
on real NABU hardware via the NABU Internet Adapter.

### What changed from the Atari original

- Weights (weights_b2s.bin) embedded as a C array in weights.h using a small
  Python script -- no binary file loading needed at runtime.
- AY-3-8910 sound chip replaces POKEY. Two short beeps mark each inference:
  high pitch when the model starts, low pitch when the response is done.
- Dream mode (idle auto-prompt screensaver) removed.
- VDP colour effects removed.
- Plain CP/M stdio (printf/putchar/fgets) for all text I/O.
- Z80 R register used to seed the pseudo-random number generator at startup.

### What stayed the same

- Full neural network inference -- INT2 L1 weights, INT4 L2 weights, sparse MAC,
  ReLU, 3-phase output bias, confidence gate, repeat detection, EOL boost.
- All weight offsets match jambo.asm as best I can (exactly).
- Same character set, same response behaviour, same fallback messages.
- 50% chance the model re-reads its own answer and generates a follow-up line.

---

## Building

Requires [z88dk](https://github.com/z88dk/z88dk) with the SDCC backend.

From the `nabu/` directory on Windows:

```
build.bat
```

Or directly:

```
zcc +cpm -vn -create-app -compiler=sdcc --opt-code-speed nabujam.c -o NABUJAM
```

A pre-built `NABUJAM.COM` is included for convenience.

---

## Usage

Copy `NABUJAM.COM` to your CP/M drive and run it. Type a prompt and press Enter.
Type `QUIT` to return to CP/M.

```
> 2+3
 5
> COUNT 5
 1 2 3 4 5
> LOVE
 NOT COMPUTABLE-
 BUT REAL!
> ZORK
 EATEN BY GRUE
> QUIT
```

---

## Credits

Original JAMBO by Marek Spanel -- [jam.ag](https://jam.ag) /
[marspa73/atarijam](https://github.com/marspa73/atarijam)

NABU port by Intangybles 2026
