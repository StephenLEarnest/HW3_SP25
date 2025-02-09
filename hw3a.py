import math


def normal_pdf(x, mu, sigma):
    """Calculate the normal probability density function at x for mean mu and standard deviation sigma."""
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def simpsons_rule(func, a, b, n, mu, sigma):
    """Approximate the integral of func from a to b using Simpson's 1/3 Rule."""
    if n % 2 == 0:
        n += 1  # Simpson's rule requires an odd number of intervals
    h = (b - a) / n
    integral = func(a, mu, sigma) + func(b, mu, sigma)

    for i in range(1, n, 2):
        integral += 4 * func(a + i * h, mu, sigma)

    for i in range(2, n - 1, 2):
        integral += 2 * func(a + i * h, mu, sigma)

    return (h / 3) * integral


def probability_less_than_c(c, mu, sigma):
    """Calculate P(X < c) using numerical integration."""
    return simpsons_rule(normal_pdf, -float('inf'), c, 1000, mu, sigma)


def probability_greater_than_c(c, mu, sigma):
    """Calculate P(X > c) using numerical integration."""
    return 1 - probability_less_than_c(c, mu, sigma)


def probability_between(c, mu, sigma):
    """Calculate P(μ - (c - μ) < X < μ + (c - μ)) using numerical integration."""
    lower_bound = mu - (c - mu)
    upper_bound = mu + (c - mu)
    return simpsons_rule(normal_pdf, lower_bound, upper_bound, 1000, mu, sigma)


def secant_method(func, x0, x1, mu, sigma, tol=1e-6, max_iter=100):
    """Find the root of the function func using the Secant method."""
    for _ in range(max_iter):
        f0 = func(x0, mu, sigma)
        f1 = func(x1, mu, sigma)
        if abs(f1 - f0) < tol:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, x1 = x1, x2
    return x1


def main():
    # Soliciting input from the user
    print("Welcome to the Probability Calculator")
    choice = input(
        "Are you specifying c and seeking P (enter 'c') or specifying P and seeking c (enter 'P')? ").strip().lower()

    mu = float(input("Enter the mean (μ): "))
    sigma = float(input("Enter the standard deviation (σ): "))

    if choice == 'P':
        P = float(input("Enter the desired probability P (between 0 and 1): "))

        # Case 1: P(x < c)
        def func_less_than(c, mu, sigma):
            return probability_less_than_c(c, mu, sigma) - P

        # Case 2: P(x > c)
        def func_greater_than(c, mu, sigma):
            return probability_greater_than_c(c, mu, sigma) - P

        # Case 3: P(μ - (c - μ) < x < μ + (c - μ))
        def func_between(c, mu, sigma):
            return probability_between(c, mu, sigma) - P

        # Solicit which case the user wants to compute
        case = input("Which probability case would you like to solve for?\n"
                     "1. P(X < c)\n"
                     "2. P(X > c)\n"
                     "3. P(μ - (c - μ) < X < μ + (c - μ))\n"
                     "Enter the case number: ")

        if case == '1':
            c = secant_method(func_less_than, mu - 5 * sigma, mu + 5 * sigma, mu, sigma)
            print(f"The value of c that gives the desired probability is: {c}")
        elif case == '2':
            c = secant_method(func_greater_than, mu - 5 * sigma, mu + 5 * sigma, mu, sigma)
            print(f"The value of c that gives the desired probability is: {c}")
        elif case == '3':
            c = secant_method(func_between, mu, mu + 5 * sigma, mu, sigma)
            print(f"The value of c that gives the desired probability is: {c}")
        else:
            print("Invalid case number.")

    elif choice == 'c':
        c = float(input("Enter the value of c: "))

        # Case 1: P(X < c)
        P_c_less_than = probability_less_than_c(c, mu, sigma)
        print(f"P(X < {c}) = {P_c_less_than}")

        # Case 2: P(X > c)
        P_c_greater_than = probability_greater_than_c(c, mu, sigma)
        print(f"P(X > {c}) = {P_c_greater_than}")

        # Case 3: P(μ - (c - μ) < x < μ + (c - μ))
        P_between = probability_between(c, mu, sigma)
        print(f"P(μ - (c - μ) < X < μ + (c - μ)) = {P_between}")

    else:
        print("Invalid choice! Please choose either 'c' or 'P'.")


if __name__ == "__main__":
    main()
