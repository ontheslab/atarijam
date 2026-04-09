/*
 * NABU JAM - Generative language model for NABU PC
 * Port of JAMBO (Atari 800, 6502 assembly) to Z80/z88dk
 * Architecture: 192 input -> 256 hidden (ReLU) -> 45 output (char)
 * Weights: packed INT2 (4 per byte), 18KB blob
 * "C" port for NABU by Intangybles 2026
 *
 * Build: zcc +cpm -vn -create-app -compiler=sdcc --opt-code-speed nabujam.c -o NABUJAM
 * Output: NABUJAM.COM (CP/M 2.2, NABU Personal Computer / Cloud CP/M 9.6)
 *
 * --- Weights ---
 * The 18,439-byte weight blob (weights_b2s.bin) is taken as-is from the
 * original atarijam repo and embedded as a C array in weights.h. Conversion
 * was done with a small Python script:
 *
 *   data = open("../atarijam/jam/weights_b2s.bin", "rb").read()
 *   with open("weights.h", "w") as f:
 *       f.write("static const uint8_t weights[%d] = {\n" % len(data))
 *       f.write(", ".join("0x%02X" % b for b in data))
 *       f.write("\n};\n")
 *
 * No transformation of the data -- all offset constants (W1_OFF, W2_OFF etc.)
 * index directly into the array at runtime, matching jambo.asm as best I can.
 *
 * --- CP/M simplifications vs. the Atari original ---
 * - Dream mode (idle auto-prompt) removed. The Atari version would pick a
 *   random seed phrase after a timeout and feed it to the model unprompted.
 *   Not implimented on CP/M for simplictiy.
 * - VDP colour flash removed.
 *
 * --- Sound ---
 * In place of the Atari POKEY wobble tone, two short AY-3-8910 beeps mark
 * each responce: a high beep when the model starts computing,
 * and a lower beep when the response is complete. A little extra for the NABU ;-)
 * 
 */

#include <stdio.h>
#include <stdint.h>
#include <conio.h>
#include <string.h>

#include "weights.h"

/* =======================================================
 * NABU hardware -- direct port access, no NABU-LIB needed
 * ======================================================= */

/* AY-3-8910 sound chip */
__sfr __at 0x40 IO_AYDATA;
__sfr __at 0x41 IO_AYLATCH;

static void ayWrite(uint8_t reg, uint8_t val)
{
    IO_AYLATCH = reg;
    IO_AYDATA  = val;
}


/* =====================================================
 * Architecture constants
 * ===================================================== */
#define N_IN        192
#define N_HID       256
#define N_OUT       45
#define L1_WHALF    48
#define L2_WHALF    128
#define SEP_CHAR    0x3E    /* '>' */
#define MAX_GEN     15
#define MAX_INPUT   16
#define EOL_IDX     44

#define W1_OFF      0
#define W2_OFF      12288
#define B1_OFF      18048
#define B2_OFF      18304
#define B2S_OFF     18349
#define B2M_OFF     18394

static const uint8_t charset[45] = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-,!?'/";

/* =====================================================
 * Strings
 * ===================================================== */
static const char banner[] =
    "================================\r\n"
    "           NABU JAM\r\n"
    "      Just A NABU Model\r\n"
    "   C port of JAMBO (Atari 800)\r\n"
    "================================\r\n"
    " 18 KB brain. Thinks in chains.\r\n"
    " Try: 2+3, COUNT 5, LOVE, ZORK\r\n"
    " Type QUIT to exit.\r\n\r\n";

static const char *us_msgs[8] = {
    "JUST ASK", "NOT SURE", "HMM", "WHAT",
    "TRY AGAIN", "I JAM", "WHATEVER", "SO CONFUSING"
};
static const uint8_t us_lens[8] = { 8, 8, 3, 4, 9, 5, 8, 12 };

static const char *fb_msgs[3] = {
    "TRY SOMETHING ELSE", "BEATS ME", "NOT IN MY 18 KB"
};
static const uint8_t fb_lens[3] = { 18, 8, 15 };


/* =====================================================
 * Global state
 * ===================================================== */
static uint8_t iline[42];
static uint8_t llen;
static uint8_t ctx[128];
static uint8_t ctxlen;

/* Input feature vector -- laid out contiguously so the W1 column index
 * matches the array index directly.
 * [0..127]   trigram hash buckets
 * [128..191] bag-of-chars and suffix groups */
static uint8_t feature[192];
#define thash   feature
#define bhash   (feature + 128)

static uint8_t  hidbuf[256];
static int16_t  preact[256];
static int16_t  obuf[45];

static uint8_t nz_count;
static uint8_t nz_off[64];
static uint8_t nz_nib[64];
static uint8_t nz_val[64];

static uint8_t dirty_count;
static uint8_t dirty_off[12];
static uint8_t dirty_sgn[12];

static uint8_t gencnt;
static uint8_t lastch;
static uint8_t repchr;
static uint8_t answer_ok;
static uint8_t hint_mode;

static int16_t best_val;
static int16_t second_val;
static uint8_t best_idx;
static uint8_t second_idx;

/* pseudo-random numbers, seeded at startup from the Z80 R register */
static uint16_t rand_state = 0xACE1u;

static uint8_t rand8(void)
{
    uint8_t lsb;
    lsb = (uint8_t)(rand_state & 1u);
    rand_state >>= 1;
    if (lsb) rand_state ^= 0xB400u;
    return (uint8_t)(rand_state & 0xFF);
}

/* Read the Z80 R register -- it ticks on every instruction fetch so its value
 * at startup is a handy source of randomness for seeding the LFSR.
 * The asm block returns via ret, so the C return 0 is never reached.
 * The unreachable return keeps SDCC quiet about missing return values. */
static uint8_t get_r_reg(void)
{
    __asm
    ld  a, r
    ld  l, a
    ld  h, #0
    ret
    __endasm;
    return 0; /* unreachable */
}

/* =======================================================================
 * AY-3-8910 beeps
 * A high beep signals the start of inference, a low beep signals done.
 * The fine register sets pitch -- lower value means higher pitch.
 * The delay loop runs about 30000 iterations, roughly 80 ms at 3.58 MHz.
 * ======================================================================= */
#define AY_FINE_A   0
#define AY_COARSE_A 1
#define AY_MIXER    7
#define AY_VOL_A    8

static void ay_beep(uint8_t fine, uint8_t vol)
{
    uint16_t d;
    ayWrite(AY_COARSE_A, 0);
    ayWrite(AY_FINE_A,   fine);
    ayWrite(AY_MIXER,    0xFEu);
    ayWrite(AY_VOL_A,    vol);
    for (d = 0; d < 30000u; d++);
    ayWrite(AY_VOL_A,    0);
    ayWrite(AY_MIXER,    0xFFu);
}

/* =====================================================
 * CP/M console input
 * ===================================================== */

/* Read a line from stdin. CP/M BDOS handles echo and backspace for us.
 * fflush before fgets is required on CP/M or the prompt may not appear.
 * Strips the trailing newline. Returns the length. */
static uint8_t read_line(uint8_t *buf, uint8_t maxlen)
{
    uint8_t len;
    uint8_t i;
    fflush(stdout);
    if (!fgets((char *)buf, (int)maxlen, stdin)) {
        buf[0] = 0;
        return 0;
    }
    /* Strip trailing newline */
    len = 0;
    for (i = 0; i < maxlen && buf[i]; i++) {
        if (buf[i] == '\r' || buf[i] == '\n') { buf[i] = 0; break; }
        len++;
    }
    return len;
}



/* =====================================================
 * Hash functions
 * ===================================================== */
static void hash_clear(void)
{
    uint8_t i;
    for (i = 0; i < 192u; i++) feature[i] = 0;
}

static uint8_t tri_one(uint8_t c0, uint8_t c1, uint8_t c2)
{
    uint8_t h;
    h = (uint8_t)((uint8_t)((uint8_t)(c0 << 5) - c0)
        + (uint8_t)((uint8_t)(c1 << 3) - c1)
        + c2) & 0x7Fu;
    if (thash[h] < 255u) thash[h]++;
    return h;
}

static void hash_all(void)
{
    uint8_t x;

    if (ctxlen >= 3u) {
        for (x = 0; (uint8_t)(x + 2u) < ctxlen; x++)
            tri_one(ctx[x], ctx[x+1], ctx[x+2]);
    }

    for (x = 0; x < ctxlen; x++) {
        uint8_t b;
        b = ctx[x] & 0x1Fu;
        if (bhash[b] < 255u) bhash[b]++;
    }

    if (gencnt == 0u) return;

    bhash[32u + (ctx[ctxlen-1] & 7u)]++;
    bhash[56u + (gencnt & 7u)]++;

    if (gencnt < 2u) return;

    bhash[40u + (ctx[ctxlen-2] & 7u)]++;
    bhash[48u + ((uint8_t)(ctx[ctxlen-1] - ctx[ctxlen-2]) & 7u)]++;
}

/* =====================================================
 * Weight decode
 * ===================================================== */
static int8_t decode_int2(uint8_t b, uint8_t p)
{
    uint8_t code;
    code = (b >> (uint8_t)(p << 1u)) & 3u;
    if (code == 0u) return  0;
    if (code == 1u) return -1;
    if (code == 2u) return  1;
    return -2;
}

static int8_t decode_int4_hi(uint8_t b)
{
    uint8_t n;
    n = b >> 4;
    if (n & 8u) return (int8_t)((int8_t)n | (int8_t)0xF0);
    return (int8_t)n;
}

static int8_t decode_int4_lo(uint8_t b)
{
    uint8_t n;
    n = b & 0x0Fu;
    if (n & 8u) return (int8_t)((int8_t)n | (int8_t)0xF0);
    return (int8_t)n;
}

/* =====================================================
 * Sparse list builders
 * ===================================================== */
static void build_nz_input(void)
{
    uint8_t i;
    nz_count = 0;
    for (i = 0; i < N_IN; i++) {
        if (feature[i] && nz_count < 64u) {
            nz_val[nz_count] = feature[i];
            nz_off[nz_count] = i >> 2;
            nz_nib[nz_count] = i & 3u;
            nz_count++;
        }
    }
}

static void build_nz_hidden(void)
{
    uint8_t i;
    nz_count = 0;
    i = 0;
    do {
        if (hidbuf[i] && nz_count < 64u) {
            nz_off[nz_count] = i >> 1;
            nz_nib[nz_count] = i & 1u;
            nz_val[nz_count] = hidbuf[i];
            nz_count++;
        }
        i++;
    } while (i != 0);
}

/* =====================================================
 * L1 forward pass
 * ===================================================== */
static void l1_full(void)
{
    uint8_t n;
    build_nz_input();
    n = 0;
    do {
        int16_t acc;
        int8_t  bias;
        uint8_t k;
        acc = 0;
        for (k = 0; k < nz_count; k++) {
            int8_t w;
            w = decode_int2(
                weights[W1_OFF + (uint16_t)n * L1_WHALF + nz_off[k]],
                nz_nib[k]);
            if (w != 0)
                acc += (int16_t)nz_val[k] * (int16_t)w;
        }
        bias = (int8_t)weights[B1_OFF + n];
        acc += (int16_t)bias;
        preact[n] = acc;
        if (acc <= 0)        hidbuf[n] = 0;
        else if (acc >= 255) hidbuf[n] = 255;
        else                 hidbuf[n] = (uint8_t)acc;
        n++;
    } while (n != 0);
}

static void relu_from_preact(void)
{
    uint8_t n;
    n = 0;
    do {
        int16_t v;
        v = preact[n];
        if (v <= 0)        hidbuf[n] = 0;
        else if (v >= 255) hidbuf[n] = 255;
        else               hidbuf[n] = (uint8_t)v;
        n++;
    } while (n != 0);
}

static void apply_delta_col(uint8_t feat_idx, uint8_t is_sub)
{
    uint8_t n;
    uint8_t byte_off;
    uint8_t pair;
    byte_off = feat_idx >> 2;
    pair     = feat_idx & 3u;
    n = 0;
    do {
        int8_t  w;
        int16_t delta;
        w = decode_int2(weights[W1_OFF + (uint16_t)n * L1_WHALF + byte_off], pair);
        if (w != 0) {
            delta = (int16_t)w;
            if (is_sub) delta = (int16_t)-delta;
            preact[n] = (int16_t)(preact[n] + delta);
        }
        n++;
    } while (n != 0);
}

static void l1_incremental(void)
{
    uint8_t old_gencnt;
    uint8_t lc;
    uint8_t pc;

    dirty_count = 0;
    old_gencnt  = gencnt - 1u;

    /* New trigram */
    if (ctxlen >= 3u) {
        uint8_t h;
        h = tri_one(ctx[ctxlen-3], ctx[ctxlen-2], ctx[ctxlen-1]);
        if (dirty_count < 12u) { dirty_off[dirty_count] = h; dirty_sgn[dirty_count++] = 0; }
    }

    /* New bag entry */
    {
        uint8_t b;
        b = ctx[ctxlen-1] & 0x1Fu;
        if (bhash[b] < 255u) bhash[b]++;
        if (dirty_count < 12u) { dirty_off[dirty_count] = (uint8_t)(128u+b); dirty_sgn[dirty_count++] = 0; }
    }

    /* Remove old suffix buckets */
    if (old_gencnt > 0u) {
        pc = ctx[ctxlen-2];
        {
            uint8_t y = pc & 7u;
            if (bhash[32u+y] > 0u) bhash[32u+y]--;
            if (dirty_count < 12u) { dirty_off[dirty_count] = (uint8_t)(160u+y); dirty_sgn[dirty_count++] = 1; }
        }
        {
            uint8_t y = old_gencnt & 7u;
            if (bhash[56u+y] > 0u) bhash[56u+y]--;
            if (dirty_count < 12u) { dirty_off[dirty_count] = (uint8_t)(184u+y); dirty_sgn[dirty_count++] = 1; }
        }
        if (old_gencnt >= 2u) {
            uint8_t ppc = ctx[ctxlen-3];
            {
                uint8_t y = ppc & 7u;
                if (bhash[40u+y] > 0u) bhash[40u+y]--;
                if (dirty_count < 12u) { dirty_off[dirty_count] = (uint8_t)(168u+y); dirty_sgn[dirty_count++] = 1; }
            }
            {
                uint8_t y = (uint8_t)(pc - ppc) & 7u;
                if (bhash[48u+y] > 0u) bhash[48u+y]--;
                if (dirty_count < 12u) { dirty_off[dirty_count] = (uint8_t)(176u+y); dirty_sgn[dirty_count++] = 1; }
            }
        }
    }

    /* Add new suffix buckets */
    lc = ctx[ctxlen-1];
    {
        uint8_t y = lc & 7u;
        bhash[32u+y]++;
        if (dirty_count < 12u) { dirty_off[dirty_count] = (uint8_t)(160u+y); dirty_sgn[dirty_count++] = 0; }
    }
    {
        uint8_t y = gencnt & 7u;
        bhash[56u+y]++;
        if (dirty_count < 12u) { dirty_off[dirty_count] = (uint8_t)(184u+y); dirty_sgn[dirty_count++] = 0; }
    }
    if (gencnt >= 2u) {
        pc = ctx[ctxlen-2];
        {
            uint8_t y = pc & 7u;
            bhash[40u+y]++;
            if (dirty_count < 12u) { dirty_off[dirty_count] = (uint8_t)(168u+y); dirty_sgn[dirty_count++] = 0; }
        }
        {
            uint8_t y = (uint8_t)(lc - pc) & 7u;
            bhash[48u+y]++;
            if (dirty_count < 12u) { dirty_off[dirty_count] = (uint8_t)(176u+y); dirty_sgn[dirty_count++] = 0; }
        }
    }

    {
        uint8_t d;
        for (d = 0; d < dirty_count; d++)
            apply_delta_col(dirty_off[d], dirty_sgn[d]);
    }
    relu_from_preact();
}

/* =====================================================
 * L2 forward pass
 * ===================================================== */
static void l2_sparse(uint8_t gen_pos)
{
    uint8_t o;
    const uint8_t *bias_base;

    build_nz_hidden();

    if (gen_pos < 3u)      bias_base = weights + B2S_OFF;
    else if (gen_pos < 8u) bias_base = weights + B2M_OFF;
    else                   bias_base = weights + B2_OFF;

    for (o = 0; o < N_OUT; o++) {
        int16_t acc;
        int8_t  bias;
        uint8_t k;
        acc = 0;
        for (k = 0; k < nz_count; k++) {
            int8_t  w;
            uint8_t wb;
            wb = weights[W2_OFF + (uint16_t)o * L2_WHALF + nz_off[k]];
            w  = (nz_nib[k] == 0) ? decode_int4_lo(wb) : decode_int4_hi(wb);
            if (w != 0)
                acc += (int16_t)nz_val[k] * (int16_t)w;
        }
        bias = (int8_t)bias_base[o];
        acc += (int16_t)bias;
        obuf[o] = acc;
    }

    if (gen_pos >= 15u)
        obuf[EOL_IDX] += (int16_t)((int16_t)((gen_pos - 15u) << 4) + 24);
}

/* =====================================================
 * Argmax
 * ===================================================== */
static void argmax(void)
{
    uint8_t i;
    best_val   = obuf[0];
    second_val = obuf[0];
    best_idx   = 0;
    second_idx = 0;
    for (i = 1; i < N_OUT; i++) {
        if (obuf[i] > best_val) {
            second_val = best_val;
            second_idx = best_idx;
            best_val   = obuf[i];
            best_idx   = i;
        } else if (obuf[i] > second_val) {
            second_val = obuf[i];
            second_idx = i;
        }
    }
}

/* =====================================================
 * Generation
 * ===================================================== */
static void run_generation(void)
{
    uint8_t done;

    gencnt    = 0;
    repchr    = 0;
    lastch    = 0xFFu;
    answer_ok = 1;
    done      = 0;

    hash_clear();
    hash_all();
    l1_full();

    ay_beep(60u, 10u);   /* high beep -- model is thinking */
    putchar(' ');

    while (!done) {
        l2_sparse(gencnt);
        argmax();

        /* Confidence check: first char only */
        if (gencnt == 0u) {
            int16_t margin;
            margin = (int16_t)(best_val - second_val);
            if (margin < 3) {
                uint8_t idx;
                uint8_t j;
                putchar(' ');
                idx = rand8() & 7u;
                for (j = 0; j < us_lens[idx]; j++) putchar(us_msgs[idx][j]);
                putchar('\r');
                putchar('\n');
                answer_ok = 0;
                done = 1;
                break;
            }
        }

        /* EOL token: end of response */
        if (best_idx == EOL_IDX) {
            done = 1;
            break;
        }

        {
            uint8_t ch;
            ch = charset[best_idx];

            if (ch == lastch) {
                repchr++;
                if (repchr >= 3u) {
                    uint8_t idx;
                    uint8_t j;
                    for (j = 1; j < repchr; j++) {
                        putchar('\b'); putchar(' '); putchar('\b');
                    }
                    putchar('\r');
                    putchar('\n');
                    putchar(' ');
                    idx = rand8() % 3u;
                    for (j = 0; j < fb_lens[idx]; j++) putchar(fb_msgs[idx][j]);
                    putchar('\r');
                    putchar('\n');
                    answer_ok = 0;
                    done = 1;
                    break;
                }
            } else {
                lastch = ch;
                repchr = 1u;
            }

            putchar(ch);

            {
                uint8_t lc;
                lc = ch;
                if (lc >= 'A' && lc <= 'Z') lc = (uint8_t)(lc | 0x20u);
                if (ctxlen < 127u) ctx[ctxlen++] = lc;
            }

            gencnt++;
            if (gencnt >= MAX_GEN) { done = 1; break; }

            l1_incremental();
        }
    }

    fflush(stdout);
    putchar('\r');
    putchar('\n');
    ay_beep(140u, 10u);  /* low beep -- response done */
}

/* =========================================================
 * Hint mode - 50% chance the model re-reads its own answer
 * and generates a follow-up response on the next line.
 * Origonal used 25%
 * ========================================================= */
static void maybe_hint(void)
{
    uint8_t i;
    uint8_t sep_pos;

    if (hint_mode)           return;
    if (!answer_ok)          return;
    if (gencnt < 2u)         return;
    if ((rand8() & 1u) != 0) return;

    sep_pos = 0xFFu;
    for (i = 0; i < ctxlen; i++) {
        if (ctx[i] == SEP_CHAR) { sep_pos = i; break; }
    }
    if (sep_pos == 0xFFu) return;

    {
        uint8_t ans_len;
        ans_len = ctxlen - sep_pos - 1u;
        for (i = 0; i < ans_len; i++) ctx[i] = ctx[sep_pos + 1u + i];
        ctx[i] = SEP_CHAR;
        ctxlen  = i + 1u;
    }

    hint_mode = 1;
    putchar(' ');
    run_generation();
    hint_mode = 0;
}

/* =====================================================
 * Main
 * ===================================================== */
int main(void)
{
    /* Seed the LFSR from the R register */
    rand_state = (uint16_t)get_r_reg() ^ 0xACE1u;
    if (rand_state == 0) rand_state = 0xACE1u;

    printf(banner);

    hint_mode = 0;

    for (;;) {
        uint8_t i;

        putchar('>');
        putchar(' ');
        llen = read_line(iline, 40);

        /* Strip trailing punctuation */
        while (llen > 0u) {
            uint8_t last;
            last = iline[llen-1];
            if (last == '?' || last == '!' || last == '.') {
                llen--;
                iline[llen] = 0;
            } else {
                break;
            }
        }

        if (llen == 0u) continue;

        /* Quit command */
        if ((iline[0]=='q' || iline[0]=='Q') &&
            (iline[1]=='u' || iline[1]=='U') &&
            (iline[2]=='i' || iline[2]=='I') &&
            (iline[3]=='t' || iline[3]=='T') &&
            llen == 4u) return 0;

        if (llen > MAX_INPUT) llen = MAX_INPUT;

        for (i = 0; i < llen; i++) {
            uint8_t c;
            c = iline[i];
            if (c >= 'A' && c <= 'Z') c = (uint8_t)(c | 0x20u);
            ctx[i] = c;
        }
        ctx[llen] = SEP_CHAR;
        ctxlen = llen + 1u;

        run_generation();

        maybe_hint();
    }

    return 0;
}
