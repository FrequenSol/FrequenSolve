# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Canonical JSON text as Sauce hashes it: ``nlohmann::json::dump()``.

Compact separators, object keys in sorted order and UTF-8 strings match
``json.dumps``; doubles do not. nlohmann/json 3.11.3 (Sauce's
``include/json_reader.hpp``) prints a double with Grisu2 digits, which differ
from Python's shortest ``repr`` for about 2% of arbitrary doubles, and uses
exponent notation from ``1e15``. :func:`nlohmann_double` reproduces its
``to_chars`` exactly.
"""

from __future__ import annotations

import json
import math
import struct
from typing import Any, Callable, List, Mapping, Tuple

# Cached powers of ten ``(significand, binary exponent, decimal exponent)`` of
# nlohmann/json 3.11.3's Grisu2 (``dtoa_impl::get_cached_power_for_binary_exponent``).
_GRISU_POWERS = (
    (0xAB70FE17C79AC6CA, -1060, -300),
    (0xFF77B1FCBEBCDC4F, -1034, -292),
    (0xBE5691EF416BD60C, -1007, -284),
    (0x8DD01FAD907FFC3C, -980, -276),
    (0xD3515C2831559A83, -954, -268),
    (0x9D71AC8FADA6C9B5, -927, -260),
    (0xEA9C227723EE8BCB, -901, -252),
    (0xAECC49914078536D, -874, -244),
    (0x823C12795DB6CE57, -847, -236),
    (0xC21094364DFB5637, -821, -228),
    (0x9096EA6F3848984F, -794, -220),
    (0xD77485CB25823AC7, -768, -212),
    (0xA086CFCD97BF97F4, -741, -204),
    (0xEF340A98172AACE5, -715, -196),
    (0xB23867FB2A35B28E, -688, -188),
    (0x84C8D4DFD2C63F3B, -661, -180),
    (0xC5DD44271AD3CDBA, -635, -172),
    (0x936B9FCEBB25C996, -608, -164),
    (0xDBAC6C247D62A584, -582, -156),
    (0xA3AB66580D5FDAF6, -555, -148),
    (0xF3E2F893DEC3F126, -529, -140),
    (0xB5B5ADA8AAFF80B8, -502, -132),
    (0x87625F056C7C4A8B, -475, -124),
    (0xC9BCFF6034C13053, -449, -116),
    (0x964E858C91BA2655, -422, -108),
    (0xDFF9772470297EBD, -396, -100),
    (0xA6DFBD9FB8E5B88F, -369, -92),
    (0xF8A95FCF88747D94, -343, -84),
    (0xB94470938FA89BCF, -316, -76),
    (0x8A08F0F8BF0F156B, -289, -68),
    (0xCDB02555653131B6, -263, -60),
    (0x993FE2C6D07B7FAC, -236, -52),
    (0xE45C10C42A2B3B06, -210, -44),
    (0xAA242499697392D3, -183, -36),
    (0xFD87B5F28300CA0E, -157, -28),
    (0xBCE5086492111AEB, -130, -20),
    (0x8CBCCC096F5088CC, -103, -12),
    (0xD1B71758E219652C, -77, -4),
    (0x9C40000000000000, -50, 4),
    (0xE8D4A51000000000, -24, 12),
    (0xAD78EBC5AC620000, 3, 20),
    (0x813F3978F8940984, 30, 28),
    (0xC097CE7BC90715B3, 56, 36),
    (0x8F7E32CE7BEA5C70, 83, 44),
    (0xD5D238A4ABE98068, 109, 52),
    (0x9F4F2726179A2245, 136, 60),
    (0xED63A231D4C4FB27, 162, 68),
    (0xB0DE65388CC8ADA8, 189, 76),
    (0x83C7088E1AAB65DB, 216, 84),
    (0xC45D1DF942711D9A, 242, 92),
    (0x924D692CA61BE758, 269, 100),
    (0xDA01EE641A708DEA, 295, 108),
    (0xA26DA3999AEF774A, 322, 116),
    (0xF209787BB47D6B85, 348, 124),
    (0xB454E4A179DD1877, 375, 132),
    (0x865B86925B9BC5C2, 402, 140),
    (0xC83553C5C8965D3D, 428, 148),
    (0x952AB45CFA97A0B3, 455, 156),
    (0xDE469FBD99A05FE3, 481, 164),
    (0xA59BC234DB398C25, 508, 172),
    (0xF6C69A72A3989F5C, 534, 180),
    (0xB7DCBF5354E9BECE, 561, 188),
    (0x88FCF317F22241E2, 588, 196),
    (0xCC20CE9BD35C78A5, 614, 204),
    (0x98165AF37B2153DF, 641, 212),
    (0xE2A0B5DC971F303A, 667, 220),
    (0xA8D9D1535CE3B396, 694, 228),
    (0xFB9B7CD9A4A7443C, 720, 236),
    (0xBB764C4CA7A44410, 747, 244),
    (0x8BAB8EEFB6409C1A, 774, 252),
    (0xD01FEF10A657842C, 800, 260),
    (0x9B10A4E5E9913129, 827, 268),
    (0xE7109BFBA19C0C9D, 853, 276),
    (0xAC2820D9623BF429, 880, 284),
    (0x80444B5E7AA7CF85, 907, 292),
    (0xBF21E44003ACDD2D, 933, 300),
    (0x8E679C2F5E44FF8F, 960, 308),
    (0xD433179D9C8CB841, 986, 316),
    (0x9E19DB92B4E31BA9, 1013, 324),
)
_MASK32 = 0xFFFFFFFF


def _grisu_mul(x: int, y: int) -> int:
    """Return the upper 64 bits of ``x * y`` rounded half up (``diyfp::mul``)."""

    u_lo, u_hi, v_lo, v_hi = x & _MASK32, x >> 32, y & _MASK32, y >> 32
    p1, p2 = u_lo * v_hi, u_hi * v_lo
    q = ((u_lo * v_lo) >> 32) + (p1 & _MASK32) + (p2 & _MASK32) + (1 << 31)
    return u_hi * v_hi + (p2 >> 32) + (p1 >> 32) + (q >> 32)


def _grisu_round(
    digits: List[int], dist: int, delta: int, rest: int, ten_k: int
) -> str:
    while (
        rest < dist
        and delta - rest >= ten_k
        and (rest + ten_k < dist or dist - rest > rest + ten_k - dist)
    ):
        digits[-1] -= 1
        rest += ten_k
    return "".join(map(str, digits))


def _grisu2(value: float) -> Tuple[str, int]:
    """Return ``(digits, exponent)`` of a finite positive double like nlohmann's Grisu2."""

    (bits,) = struct.unpack("<Q", struct.pack("<d", value))
    biased, fraction = bits >> 52, bits & ((1 << 52) - 1)
    v_f, v_e = (fraction, -1074) if biased == 0 else (fraction | 1 << 52, biased - 1075)
    plus_f = 2 * v_f + 1
    shift = 64 - plus_f.bit_length()
    plus_f, e = plus_f << shift, v_e - 1 - shift
    if fraction == 0 and biased > 1:
        minus_f = (4 * v_f - 1) << (v_e - 2 - e)
    else:
        minus_f = (2 * v_f - 1) << (v_e - 1 - e)
    v_f <<= 64 - v_f.bit_length()
    # The cached power brings the product's binary exponent into [-60, -32];
    # C++ integer division truncates toward zero.
    f = -61 - e
    product = f * 78913
    k = (product >> 18 if product >= 0 else -(-product >> 18)) + (f > 0)
    cached_f, cached_e, cached_k = _GRISU_POWERS[(307 + k) // 8]
    w = _grisu_mul(v_f, cached_f)
    low = _grisu_mul(minus_f, cached_f) + 1
    high = _grisu_mul(plus_f, cached_f) - 1
    shift = -(e + cached_e + 64)
    one = 1 << shift
    delta, dist = high - low, high - w
    exponent = -cached_k
    integral, fractional = high >> shift, high & (one - 1)
    n = len(str(integral))
    pow10 = 10 ** (n - 1)
    digits: List[int] = []
    while n > 0:
        digits.append(integral // pow10)
        integral %= pow10
        n -= 1
        rest = (integral << shift) + fractional
        if rest <= delta:
            return (
                _grisu_round(digits, dist, delta, rest, pow10 << shift),
                exponent + n,
            )
        pow10 //= 10
    while True:
        fractional *= 10
        digits.append(fractional >> shift)
        fractional &= one - 1
        exponent -= 1
        delta *= 10
        dist *= 10
        if fractional <= delta:
            return _grisu_round(digits, dist, delta, fractional, one), exponent


def nlohmann_double(value: float) -> str:
    """Format a double exactly as ``nlohmann::json::dump()`` does (``to_chars``).

    Grisu2 digits may differ from Python's shortest ``repr`` (for about 2% of
    arbitrary doubles), and magnitudes in ``[1e15, 1e16)`` use exponent
    notation; non-finite values become ``null``.
    """

    if not math.isfinite(value):
        return "null"
    sign = "-" if math.copysign(1.0, value) < 0 else ""
    if value == 0:
        return sign + "0.0"
    digits, exponent = _grisu2(abs(value))
    k = len(digits)
    n = k + exponent
    if k <= n <= 15:
        return sign + digits + "0" * (n - k) + ".0"
    if 0 < n <= 15:
        return sign + digits[:n] + "." + digits[n:]
    if -4 < n <= 0:
        return sign + "0." + "0" * -n + digits
    mantissa = digits if k == 1 else digits[0] + "." + digits[1:]
    return f"{sign}{mantissa}e{'-' if n <= 0 else '+'}{abs(n - 1):02d}"


def _contains_float(payload: Any) -> bool:
    stack = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, float):
            return True
        if isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return False


def _write_canonical(value: Any, write: Callable[[str], Any]) -> None:
    if isinstance(value, Mapping):
        write("{")
        for index, key in enumerate(sorted(value)):
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            write(("," if index else "") + json.dumps(key, ensure_ascii=False) + ":")
            _write_canonical(value[key], write)
        write("}")
    elif isinstance(value, (list, tuple)):
        write("[")
        for index, item in enumerate(value):
            if index:
                write(",")
            _write_canonical(item, write)
        write("]")
    elif isinstance(value, float):
        write(nlohmann_double(value))
    else:
        write(json.dumps(value, ensure_ascii=False))


def canonical_json_text(payload: Any) -> str:
    """Return ``payload`` as Sauce's canonical compact sorted JSON text."""

    if not _contains_float(payload):
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    parts: List[str] = []
    _write_canonical(payload, parts.append)
    return "".join(parts)
