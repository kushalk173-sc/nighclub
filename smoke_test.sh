#!/usr/bin/env bash
#
# A simple smoke test to ensure the core repository logic can execute
# without crashing. It runs the v1 model against the first pillar.
#

echo "--- Starting repository smoke test... ---"

# Run the main script via the bootstrap to ensure paths are correct.
# We use the --smoke flag which tells main.py to run a minimal test.
python nighclub_bootstrap.py main.py --model_version v1 --smoke

# Check the exit code of the last command.
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ Smoke test PASSED."
else
  echo "❌ Smoke test FAILED with exit code $EXIT_CODE."
fi

exit $EXIT_CODE 