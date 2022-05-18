# %%

import collections


def get_primes(n):
    """Get all prime numbers: 2, 3, 5, 7, 11, ..."""
    primes = []
    for i in range(2, n):
        if all(i % j for j in range(2, int(i ** 0.5) + 1)):
            primes.append(i)
    return primes


def get_prime_factors(n, primes):
    """
    Get prime factors of a number. Example:
        10 = 5 * 2
        16 = 2 * 2 * 2 * 2
    """
    freqs = collections.Counter()
    for prime in primes:
        while n % prime == 0:
            freqs[prime] += 1
            n //= prime
    return freqs


def main(n):
    primes = get_primes(n)
    primes_set = set(primes)
    dp = collections.defaultdict(int, {1: 0, 2: 1})
    for i in range(3, n + 1):
        if i in primes_set:
            # This is prime number. Example:
            # f(7) = 7/6 * f(6) => dp[7] = 1 + dp[6]
            dp[i] = dp[i - 1] + 1
        else:
            # Normal number. Example:
            # f(10) = f(5) * f(2) => dp[10] = dp[5] + dp[2]
            # f(16) = f(2) ** 4 => dp[16] = dp[2] * 4
            prime_factors = get_prime_factors(i, primes)
            for prime, freq in prime_factors.items():
                dp[i] += dp[prime] * freq
    return dp[n]


main(1024)  # => 10

# %%
