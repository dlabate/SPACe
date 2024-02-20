import numpy as np, gmpy2, math, random, time, sympy


def GenPrimes_SieveOfEratosthenes(end):
    # https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
    composites = np.zeros((end,), dtype=np.uint8)
    composites[:2] = 1
    for p, comp in enumerate(composites):
        if comp:
            continue
        if p * p >= len(composites):
            break
        composites[p * p:: p] = 1
    return np.array([i for i in range(len(composites)) if not composites[i]], dtype=np.uint32)


def IsProbablyPrime_MillerRabin(n, *, cnt=32):
    # According to https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test

    if n == 2:
        return True
    if n < 2 or n & 1 == 0:
        return False

    def Num(n):
        return gmpy2.mpz(n)

    n = Num(n)

    d, r = Num(n - 1), 0
    while d & 1 == 0:
        d >>= 1
        r += 1

    nm1 = Num(n - 1)

    for i in range(cnt):
        a = Num(random.randint(2, n - 2))

        x = pow(a, d, n)

        if x == 1 or x == nm1:
            continue

        prp = False
        for j in range(r - 1):
            x *= x
            x %= n
            if x == nm1:
                prp = True
                break
        if prp:
            continue

        return False

    return True


def PrevPrime(N):
    primes = GenPrimes_SieveOfEratosthenes(1 << 17)
    approx_dist = math.log(N)
    sieve_time, miller_time, miller_cnt = 0, 0, 0
    for icycle in range(100):
        sieve_size = round(approx_dist * 1.5)
        sieve_begin = N - sieve_size * (icycle + 1)
        sieve_last = N - sieve_size * icycle
        sieve = np.zeros((1 + sieve_last - sieve_begin,), dtype=np.uint8)
        tb = time.time()
        for p in primes:
            sieve[np.arange(sieve_last - sieve_last % p - sieve_begin, -1, -int(p))] = 1
        sieve_time += time.time() - tb
        tb = time.time()
        for i in range(len(sieve) - 1, -1, -1):
            if sieve[i]:
                continue
            x = sieve_begin + i
            miller_cnt += 1
            if IsProbablyPrime_MillerRabin(x):
                miller_time += time.time() - tb
                print(f'SieveTime {sieve_time:.3f} sec, MillerTime {miller_time:.3f} sec, MillerCnt {miller_cnt}',
                      flush=True)
                return x
        miller_time += time.time() - tb


def PrevPrime_Reference(N):
    # tim = time.time()
    for i in range(1 << 14):
        p = N - i
        if (p & 1) == 0:
            continue
        if sympy.isprime(p):
            # tim = time.time() - tim
            # print(f'ReferenceTime {tim:.3f} sec', flush=True)
            return p


def Main():
    print(PrevPrime_Reference(300))
    # # N = random.randrange(1 << 2048)
    # N = 100
    # p = PrevPrime(N)
    # p0 = PrevPrime_Reference(N)
    # print(p, p0)
    # # print(f'N 2^{math.log2(N):.3f}, PrevPrime offset {N - p}, PrevPrime Reference offset {N - p0}', flush=True)


if __name__ == '__main__':
    Main()