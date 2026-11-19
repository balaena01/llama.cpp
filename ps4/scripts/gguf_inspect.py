import struct, sys, collections

# ggml type -> (block size, bytes per block)
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
}

def main(path):
    f = open(path, "rb")
    def rd(fmt):
        s = struct.calcsize(fmt)
        return struct.unpack("<" + fmt, f.read(s))
    def rstr():
        (n,) = rd("Q")
        return f.read(n).decode("utf-8", "replace")
    def rval(t):
        if t == 0: return rd("B")[0]
        if t == 1: return rd("b")[0]
        if t == 2: return rd("H")[0]
        if t == 3: return rd("h")[0]
        if t == 4: return rd("I")[0]
        if t == 5: return rd("i")[0]
        if t == 6: return rd("f")[0]
        if t == 7: return bool(rd("B")[0])
        if t == 8: return rstr()
        if t == 9:
            (at,) = rd("I"); (n,) = rd("Q")
            vals = [rval(at) for _ in range(n)]
            return vals
        if t == 10: return rd("Q")[0]
        if t == 11: return rd("q")[0]
        if t == 12: return rd("d")[0]
        raise ValueError(t)

    magic = f.read(4)
    (ver,) = rd("I"); (nt,) = rd("Q"); (nkv,) = rd("Q")
    print("magic", magic, "ver", ver, "tensors", nt, "kv", nkv)
    for _ in range(nkv):
        k = rstr(); (t,) = rd("I"); v = rval(t)
        if isinstance(v, list) and len(v) > 64:
            print(f"  {k} = [array len {len(v)}]")
        else:
            print(f"  {k} = {v}")
    tensors = []
    for _ in range(nt):
        name = rstr(); (nd,) = rd("I"); dims = rd("Q" * nd); (ty,) = rd("I"); (off,) = rd("Q")
        tensors.append((name, dims, ty, off))
    total = 0
    by_type = collections.Counter()
    by_kind = collections.Counter()
    for name, dims, ty, off in tensors:
        tn, bs, bb = TYPES.get(ty, (f"t{ty}", 1, 0))
        ne = 1
        for d in dims: ne *= d
        size = ne // bs * bb
        total += size
        by_type[tn] += size
        kind = name.split(".")[-2] if name.startswith("blk.") else name
        by_kind[(kind, tn)] += size
    print("\nTotal tensor bytes: %.3f GiB" % (total / 2**30))
    print("\nBy type:")
    for k, v in by_type.most_common():
        print(f"  {k:8s} {v/2**20:10.1f} MiB")
    print("\nBy tensor kind:")
    for (k, tn), v in sorted(by_kind.items(), key=lambda x: -x[1]):
        print(f"  {k:28s} {tn:8s} {v/2**20:10.1f} MiB")
    print("\nNon-block tensors and block 0/3/4/31 tensors:")
    for name, dims, ty, off in tensors:
        if not name.startswith("blk.") or name.split(".")[1] in ("0", "3", "4", "31"):
            print(f"  {name:40s} {TYPES.get(ty,('?',))[0]:8s} {dims}")
    blks = sorted({int(n.split('.')[1]) for n, *_ in tensors if n.startswith('blk.')})
    print("\nblock ids:", blks[0], "..", blks[-1], "count", len(blks))

main(sys.argv[1])
