"""
Week 1 Python Homework - Basic Programming Exercises
Student: [Your Name]
Date: 2025-01-16
Class: UG Programming Course
"""

# Exercise 1: Variables and Basic Operations
print("=== Exercise 1: Variables and Basic Operations ===")

# TODO: Create variables for your name, age, and favorite number
student_name = "John Doe"
student_age = 20
favorite_number = 42

# TODO: Calculate and print your age in months
age_in_months = student_age * 12
print(f"My name is {student_name}")
print(f"I am {student_age} years old, which is {age_in_months} months")
print(f"My favorite number is {favorite_number}")

print("\n" + "="*50 + "\n")

# Exercise 2: Conditional Statements
print("=== Exercise 2: Conditional Statements ===")

# TODO: Write a program that checks if a number is positive, negative, or zero
def check_number(num):
    """Check if a number is positive, negative, or zero"""
    if num > 0:
        return "positive"
    elif num < 0:
        return "negative"
    else:
        return "zero"

# Test the function
test_numbers = [10, -5, 0, 3.14, -2.7]
for number in test_numbers:
    result = check_number(number)
    print(f"The number {number} is {result}")

print("\n" + "="*50 + "\n")

# Exercise 3: Loops and Lists
print("=== Exercise 3: Loops and Lists ===")

# TODO: Create a list of your favorite foods and print each one
favorite_foods = ["pizza", "sushi", "chocolate", "pasta", "ice cream"]

print("My favorite foods are:")
for i, food in enumerate(favorite_foods, 1):
    print(f"{i}. {food}")

# TODO: Calculate the sum of numbers from 1 to 10 using a loop
total_sum = 0
for i in range(1, 11):
    total_sum += i

print(f"\nSum of numbers from 1 to 10: {total_sum}")

print("\n" + "="*50 + "\n")

# Exercise 4: Functions
print("=== Exercise 4: Functions ===")

# TODO: Write a function that calculates the area of a rectangle
def calculate_rectangle_area(length, width):
    """Calculate the area of a rectangle"""
    return length * width

# TODO: Write a function that converts Celsius to Fahrenheit
def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit"""
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

# Test the functions
rectangle_area = calculate_rectangle_area(5, 3)
print(f"Area of rectangle (5x3): {rectangle_area}")

temp_celsius = 25
temp_fahrenheit = celsius_to_fahrenheit(temp_celsius)
print(f"{temp_celsius}°C = {temp_fahrenheit}°F")

print("\n" + "="*50 + "\n")

# Exercise 5: String Manipulation
print("=== Exercise 5: String Manipulation ===")

# TODO: Create a function that counts vowels in a string
def count_vowels(text):
    """Count the number of vowels in a string"""
    vowels = "aeiouAEIOU"
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count

# TODO: Create a function that reverses a string
def reverse_string(text):
    """Reverse a string"""
    return text[::-1]

# Test the functions
test_string = "Hello World Programming"
vowel_count = count_vowels(test_string)
reversed_text = reverse_string(test_string)

print(f"Original string: '{test_string}'")
print(f"Number of vowels: {vowel_count}")
print(f"Reversed string: '{reversed_text}'")

print("\n" + "="*50 + "\n")

# Bonus Exercise: List Comprehensions
print("=== Bonus Exercise: List Comprehensions ===")

# TODO: Create a list of squares of even numbers from 1 to 20
even_squares = [x**2 for x in range(1, 21) if x % 2 == 0]
print("Squares of even numbers (1-20):", even_squares)

# TODO: Create a list of lengths of words in a sentence
sentence = "Python programming is fun and educational"
word_lengths = [len(word) for word in sentence.split()]
print(f"Sentence: '{sentence}'")
print("Word lengths:", word_lengths)

print("\n" + "="*50)
print("🎉 All exercises completed successfully! 🎉")
