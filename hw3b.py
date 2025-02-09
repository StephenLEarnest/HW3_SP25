import math


# Gamma function (approximation for non-integer values)
def gamma_function(alpha):
    """Compute the Gamma function for positive real numbers."""
    if alpha.is_integer() and alpha > 0:
        return math.factorial(int(alpha) - 1)
    else:
        # Using Stirling's approximation for Gamma function
        return math.sqrt(2 * math.pi / alpha) * (alpha / math.e) ** alpha


def integrand(u, m):
    """The integrand for the t-distribution."""
    return (1 + u ** 2 / m) ** (-(m + 1) / 2)


def simpsons_rule(func, a, b, n, *args):
    """Apply Simpson's rule for numerical integration."""
    h = (b - a) / n
    sum_odd = sum(func(a + (2 * i + 1) * h, *args) for i in range(n // 2))
    sum_even = sum(func(a + 2 * i * h, *args) for i in range(1, n // 2))
    return (h / 3) * (func(a, *args) + 4 * sum_odd + 2 * sum_even + func(b, *args))


def compute_K_m(m):
    """Calculate K_m for the t-distribution."""
    gamma_m_half = gamma_function(m / 2)
    gamma_m_plus_half = gamma_function(m / 2 + 1 / 2)

    K_m = gamma_m_plus_half / (math.sqrt(m * math.pi) * gamma_m_half)
    return K_m


def compute_probability(m, z):
    """Compute the probability using the t-distribution."""
    # Calculate K_m
    K_m = compute_K_m(m)

    # Use Simpson's rule to integrate the t-distribution from -∞ to z
    integral_value = simpsons_rule(integrand, -100, z, 1000, m)  # Limits from -100 to z, with n=1000 for precision

    # Multiply by K_m to get the final result
    probability = K_m * integral_value
    return probability


def main():
    # Prompt the user for degrees of freedom (m) and z values
    m = int(input("Enter the degrees of freedom (m): "))

    # Test for different z values
    z_values = []
    for i in range(3):
        z = float(input(f"Enter z value {i + 1}: "))
        z_values.append(z)

    # Compute the probabilities for the given z values
    for z in z_values:
        probability = compute_probability(m, z)
        print(f"The probability for m = {m} and z = {z} is: {probability}")


if __name__ == "__main__":
    main()
