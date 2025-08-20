# Accept input from user about the operation needed in numeric values


print("Welcome to the calculator!. Enter first input, followed by operator index as below and then second input!")
print("Operator indexes:")
print("1. Add (+)")
print("2. Subtract (-)")
print("3. Multiply (x)")
print("4. Divide (/)")

inp1 = float(input("Enter first number: "))
choice = input("Enter operator index(1, 2, 3, 4): ")
inp2 = float(input("Enter second number: "))

if choice == '1':
    # Using f-string for formatted output
    print(f"{inp1} plus by {inp2} is: {add(inp1, inp2)}")
elif choice == '2':
    # Using traditional string formatting
   print("%d minus %d is: %d" % (inp1, inp2, subtract(inp1, inp2)))
elif choice == '3':
   print("%d multiplied by %d is: %d" % (inp1, inp2, multiply(inp1, inp2)))
elif choice == '4':
    # Using traditional string formatting for division with decimal precision upto 2 decimal places
    print("%d divided by %d is: %.2f" % (inp1, inp2, divide(inp1, inp2)))
else:
    print("Invalid operator input")