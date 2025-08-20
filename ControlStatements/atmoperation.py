# The program allows to know the account balance, pin verification, and withdrawal facility

def withdraw(withdrawal_amount, account_balance):
    # Withdraw amount
    # Using shorthand augmented assignment operator for a=a-b as a-=b
    account_balance -= withdrawal_amount
    return account_balance

def authorsied_activities(choice_operation, account_balance):
    if choice_operation == 1:   
        print("💰 Your account balance is:", account_balance)
    elif choice_operation == 2:
        # Ask user for withdrawal amount
        withdrawal_amount = int(input("Enter amount to withdraw: "))
            
        if withdrawal_amount <= 0:
            print("❌ Invalid amount entered.")
        elif withdrawal_amount > account_balance:
            print("❌ Insufficient funds! Your balance is:", account_balance)
        
        # Check if withdrawal amount is a multiple of 100 using modulus operator
        elif withdrawal % 100 != 0:
            print("❌ Please enter amount in multiples of 100")
        else:
            account_balance = withdraw(withdrawal_amount, account_balance)
            print("✅ Withdrawal successful!")
            print("💰 Remaining balance:", account_balance)
    else:
        print("❌ Operation not supported.")
    return account_balance


# Lets fix an initial balance to the account

# Initialize account details
account_balance = 100000
account_name = "Rupa Krishna"
account_ifsc = "ICIC0000354"

# Initialize PIN
PIN=294923
# List of allowed choices for ATM operations (avoid tuple as its not mutable)
valid_choices = [1, 2]
valid_choices.append(3)  # Adding exit option and showcasing mutable lists over tuples for this case


print("Welcome to the ATM Withdrawal System!")

# Ask for PIN
entered_pin = int(input("Enter your 6-digit PIN: "))

if entered_pin == PIN:
    print("✅ You are now authorised.")

    # Assume that user is authorised to perform activities and wants to continue with some operations
    continue_choice=1
    choice_operation=1
    # Loop to continue operations until user chooses to exit
    while continue_choice == 1 and choice_operation!=3:
        print("What do you want to do today")
        print("1. Check Balance")
        print("2. Withdraw Money")
        print("3. Exit")
        choice_operation = int(input("Enter your choice (1/2/3): "))

        if choice_operation not in valid_choices:
            print("❌ Invalid choice! Please enter 1, 2, or 3.")
        elif choice_operation == 3:
            print("👋 Thank you for using our ATM service. Goodbye!")
        else:
            # Call function to perform authorised activities as per selection
            account_balance = authorsied_activities(choice_operation,account_balance)
        
        #Check if user wants to do more
        if choice_operation != 3:
            continue_choice = int(input("Do you want to continue with another operation? (1 - yes, 2 - no) "))
        
# PIN validation failed
else:
    print("❌ PIN entered is not correct.")
