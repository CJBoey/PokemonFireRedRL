// fireredsim platform layer: the C library functions that game and platform
// code call while a frame runs.
//
// Two reasons not to call the host C library from inside a frame:
//
//  * Behavior. The retail ROM links agbcc's newlib (third_party/agbcc/libc/
//    string), whose memcpy/strcpy/strcmp copy and compare forward, a word at a
//    time when both pointers are 4-aligned. For overlapping buffers the result
//    depends on that algorithm (glibc's memcpy may copy backwards). The
//    functions below are exact ports of the newlib versions; where the result
//    cannot depend on the algorithm (no overlap; memset) they take a fast path.
//
//  * Determinism. Game code runs on a stack inside the state (.pnstate)
//    (platform/frame.c). glibc is a position-independent shared library at a
//    randomised address: its PIC thunks and lazy binding leave return
//    addresses into libc.so in the stack memory they use, which would make the
//    region's bytes differ between runs.
//
// game/native/mirror_sections.py renames the references of every game and
// platform object (memcpy -> Port_memcpy, ...) and rejects objects that still
// reference one of these functions in the host C library.
//
// Port_memcpy, Port_strcmp and Port_strcpy follow newlib's string/memcpy.c,
// strcmp.c and strcpy.c, which carry newlib's default notice:
//
//   Copyright (c) 1994, 1997 Cygnus Solutions.
//   All rights reserved.
//
//   Redistribution and use in source and binary forms are permitted
//   provided that the above copyright notice and this paragraph are
//   duplicated in all such forms and that any documentation,
//   advertising materials, and other materials related to such
//   distribution and use acknowledge that the software was developed
//   at Cygnus Solutions.  Cygnus Solutions may not be used to
//   endorse or promote products derived from this software without
//   specific prior written permission.
//   THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//   IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

// Keep GCC from turning the loops below back into calls to memcpy/memset.
#pragma GCC optimize ("no-tree-loop-distribute-patterns")

#include <stddef.h>
#include <stdint.h>

void *Port_memcpy(void *dst0, const void *src0, size_t len0);
void *Port_memmove(void *dst0, const void *src0, size_t len0);
void *Port_memset(void *m, int c, size_t n);
int Port_memcmp(const void *a, const void *b, size_t n);
int Port_strcmp(const char *s1, const char *s2);
char *Port_strcpy(char *dst0, const char *src0);
size_t Port_strlen(const char *s);

// newlib's helper macros (string/memcpy.c, strcmp.c, strcpy.c), 32-bit long.
#define UNALIGNED2(X, Y) ((((uintptr_t)(X)) & 3u) | (((uintptr_t)(Y)) & 3u))
#define DETECTNULL(X) (((X) - 0x01010101u) & ~(X) & 0x80808080u)

static inline int Overlap(const void *a, const void *b, size_t n)
{
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return n != 0 && x < y + n && y < x + n;
}

static inline void CopyForward(void *dst, const void *src, size_t n)
{
    __asm__ volatile("rep movsb" : "+D"(dst), "+S"(src), "+c"(n) : : "memory");
}

void *Port_memcpy(void *dst0, const void *src0, size_t len0)
{
    char *dst = dst0;
    const char *src = src0;
    uint32_t *aligned_dst;
    const uint32_t *aligned_src;
    int len = (int)len0;

    if (!Overlap(dst0, src0, len0))
    {
        // Any copy order gives the same bytes.
        CopyForward(dst0, src0, len0);
        return dst0;
    }

    // newlib memcpy (not PREFER_SIZE_OVER_SPEED; agbcc's libc.a is built so).
    if (!(len < 16) && !UNALIGNED2(src, dst))
    {
        aligned_dst = (uint32_t *)dst;
        aligned_src = (const uint32_t *)src;
        while (len >= 16)
        {
            *aligned_dst++ = *aligned_src++;
            *aligned_dst++ = *aligned_src++;
            *aligned_dst++ = *aligned_src++;
            *aligned_dst++ = *aligned_src++;
            len -= 16;
        }
        while (len >= 4)
        {
            *aligned_dst++ = *aligned_src++;
            len -= 4;
        }
        dst = (char *)aligned_dst;
        src = (const char *)aligned_src;
    }
    while (len--)
        *dst++ = *src++;
    return dst0;
}

void *Port_memmove(void *dst0, const void *src0, size_t len0)
{
    // Platform use only (the game does not call memmove): plain memmove.
    unsigned char *d = dst0;
    const unsigned char *s = src0;
    if (d <= s || d >= s + len0)
    {
        CopyForward(dst0, src0, len0);
    }
    else
    {
        while (len0--)
            d[len0] = s[len0];
    }
    return dst0;
}

void *Port_memset(void *m, int c, size_t n)
{
    // The bytes written do not depend on the algorithm.
    void *d = m;
    __asm__ volatile("rep stosb" : "+D"(d), "+c"(n) : "a"(c) : "memory");
    return m;
}

int Port_memcmp(const void *a, const void *b, size_t n)
{
    const unsigned char *x = a, *y = b;
    for (; n; n--, x++, y++)
        if (*x != *y)
            return *x - *y;
    return 0;
}

int Port_strcmp(const char *s1, const char *s2)
{
    // newlib strcmp (word-at-a-time when both are 4-aligned).
    const uint32_t *a1;
    const uint32_t *a2;

    if (!UNALIGNED2(s1, s2))
    {
        a1 = (const uint32_t *)s1;
        a2 = (const uint32_t *)s2;
        while (*a1 == *a2)
        {
            if (DETECTNULL(*a1))
                return 0;
            a1++;
            a2++;
        }
        s1 = (const char *)a1;
        s2 = (const char *)a2;
    }
    while (*s1 != '\0' && *s1 == *s2)
    {
        s1++;
        s2++;
    }
    return (*(const unsigned char *)s1) - (*(const unsigned char *)s2);
}

char *Port_strcpy(char *dst0, const char *src0)
{
    // newlib strcpy (word-at-a-time when both are 4-aligned).
    char *dst = dst0;
    const char *src = src0;
    uint32_t *aligned_dst;
    const uint32_t *aligned_src;

    if (!UNALIGNED2(src, dst))
    {
        aligned_dst = (uint32_t *)dst;
        aligned_src = (const uint32_t *)src;
        while (!DETECTNULL(*aligned_src))
            *aligned_dst++ = *aligned_src++;
        dst = (char *)aligned_dst;
        src = (const char *)aligned_src;
    }
    while ((*dst++ = *src++))
        ;
    return dst0;
}

size_t Port_strlen(const char *s)
{
    const char *p = s;
    while (*p)
        p++;
    return (size_t)(p - s);
}
