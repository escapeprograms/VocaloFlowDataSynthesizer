import os
import pythonnet
pythonnet.load("coreclr")
import clr

dependency_dir = os.path.join(os.getcwd(), "..", "API", "UtauGenerate", "bin", "Debug", "net9.0", "UtauGenerate.dll")
clr.AddReference(dependency_dir)
from UtauGenerate import Player

player = Player("OpenUtau.Plugin.Builtin.ArpasingPlusPhonemizer")
print("Methods in Player:")
for m in dir(player):
    if not m.startswith("_"):
        print(m)
