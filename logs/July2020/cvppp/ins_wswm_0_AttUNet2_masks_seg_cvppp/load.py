import os
import argparse

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument ("--p", type=str, required=True)
args = parser.parse_args()

path = args.p

cmd1 = "cp " + path + "*.py" + " " + "./" 
cmd2 = "cp " + path + "models/*.py" + " " + "./models/"

os.system (cmd1)
os.system (cmd2) 