from werkzeug.security import generate_password_hash, check_password_hash

# Your desired password
password = "Stocks123!"

# Generate the hash
password_hash = generate_password_hash(password)
print(f"Generated hash: {password_hash}")

# Test that the hash works
is_valid = check_password_hash(password_hash, password)
print(f"Hash validation test: {'Success' if is_valid else 'Failed'}")

# Test with wrong password
is_invalid = check_password_hash(password_hash, "WrongPassword")
print(f"Wrong password test: {'Failed correctly' if not is_invalid else 'Error - validated wrong password'}")