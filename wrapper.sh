#!/bin/bash

#
# This wrapper is meant to be used with guildai. It fixes two problems:
#
#  1. guildai messes with the pythonpath for god-knows-what reasons
#  2. guildai doesn't run in the directory where the source code is
#

# unset pythonpath
unset PYTHONPATH
# move to where the code is
cd .guild/sourcecode
# run
python "$@"
