Format for input.txt
Line 1: Mass of U1 leptoquark
Line 2: Couplings seperated by space
Line 3: 'yes' if single & pair production are to be ignored else 'no'
Next n lines: Coupling values for each coupling seperate by space

Example of input.txt:
1500
LM22L LM32L LM12R
no
1 0.5 1.2
0.5 0.3 0.6
0.4 0.4 0.4

To install modules required, run: pip install -r requirements.txt
To run the python script use: python3 final_calculator.py