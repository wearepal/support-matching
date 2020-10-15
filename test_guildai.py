"""For testing guildai."""
import os
import shlex
import sys

print("Command:")
print(shlex.join(sys.argv))
print()
env = os.environ
if "STARTED_BY_GUILDAI" in env:
    print("Env: STARTED_BY_GUILDAI=" + os.environ["STARTED_BY_GUILDAI"])
    print(f"type: {type(os.environ['STARTED_BY_GUILDAI'])}")
else:
    print("\n".join(env))
