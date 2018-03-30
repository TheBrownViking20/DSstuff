# Write a python class to convert an integer to a roman numeral
#write a python class to convert a roman to a integer

ROMAN_NUMERAL_TABLE = [
    ("M", 1000), ("CM", 900), ("D", 500),
    ("CD", 400), ("C", 100),  ("XC", 90),
    ("L", 50),   ("XL", 40),  ("X", 10),
    ("IX", 9),   ("V", 5),    ("IV", 4),
    ("I", 1)
]
class Inttoroman():
    def __init__(self,number):
        self.number = number
        
    def convert_to_roman():
        roman_numerals = []
        for numeral, value in ROMAN_NUMERAL_TABLE:
            while value <= self.number:
                self.number -= value
                roman_numerals.append(numeral)
    
        return ''.join(roman_numerals)
    
    
a = Inttoroman(34)
print(a.convert_to_roman())