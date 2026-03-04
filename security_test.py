
import sys
import os
from tools.code_runner import CodeRunner

runner = CodeRunner()

print("Testing dunder access block...")
res = runner.run("str.__class__")
print(f"Result: {res.error}")

print("\nTesting getattr block...")
res = runner.run("getattr(str, 'lower')")
print(f"Result: {res.error}")

print("\nTesting open write block...")
res = runner.run("open('test.txt', 'w')")
print(f"Result: {res.error}")
