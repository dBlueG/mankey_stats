from primelib.prime_numbers import is_prime

def test_false():
    assert is_prime(10) is False

def test_true():
    assert is_prime(11) is True