
# Interest Calculator Program

# import specific method from other code modules for calculating simple interest
from operation import add, multiply, divide

# A Simple Interest Calculator

# Take inputs for simple interest
P = float(input("Enter the principal amount: "))
R = float(input("Enter the rate of interest: "))
T = float(input("Enter the time (in years): "))

# Calculate Simple Interest
simpleinterest = divide(multiply(multiply(P,R),T) , 100)

# Print simple intrest result with 2 decimal places
print("Simple Interest is: %.2f" % (simpleinterest))

# A Compound Interest Calculator

def compound_interest(principal, rate, time):
    # Calculate final amount
    amount = multiply(principal, (add(1, divide(rate, 100)) ** time))
    # Compound Interest = Total - Principal
    compoundinterest = amount - principal
    return compoundinterest, amount


compoundinterest, total = compound_interest(P, R, T)

# Round the results to 2 decimal places and print them
print("Compound Interest is:", round(compoundinterest,2))
print("Total Amount with compound interest:", round(total,2))