#!/usr/bin/env bash
#
# A simple smoke test to ensure the core repository logic can execute
# without crashing. It runs a specified model against the first pillar.
#

echo "--- Starting repository smoke test... ---"

# Run the main script via the bootstrap, passing along all arguments
# provided to this script (e.g., --model_version)
# The --smoke flag is always added.
python nighclub_bootstrap.py main.py --smoke "$@"

# Check the exit code of the last command.
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ Smoke test PASSED."
else
  echo "❌ Smoke test FAILED with exit code $EXIT_CODE."
fi

exit $EXIT_CODE 