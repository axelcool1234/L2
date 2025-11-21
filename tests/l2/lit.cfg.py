# ruff: noqa

import lit.formats  # pyrefly: ignore
import os
import tempfile

config.name = "L2 tests"  # pyrefly: ignore
config.test_format = lit.formats.ShTest()  # pyrefly: ignore

# Set timeout for tests to prevent infinite runs
config.maxIndividualTestTime = 30  # 30 seconds timeout  # pyrefly: ignore
config.suffixes = [".l2"]  # pyrefly: ignore
config.test_source_root = os.path.dirname(__file__)  # pyrefly: ignore
config.test_exec_root = tempfile.mkdtemp(prefix="lit-")  # pyrefly: ignore
config.recursive = True  # pyrefly: ignore

# Pass through environment variables needed for compilation
config.environment = {}  # pyrefly: ignore
for env_var in ['BIGNUM_RUNTIME_PATH', 'LDFLAGS', 'PATH']:
    if env_var in os.environ:
        config.environment[env_var] = os.environ[env_var]  # pyrefly: ignore

config.substitutions.append(("%l2", "l2"))  # pyrefly: ignore
config.substitutions.append(("%filecheck", "filecheck"))  # pyrefly: ignore
