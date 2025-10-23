import lit.formats
import os
import tempfile

config.name = "L2 tests"
config.test_format = lit.formats.ShTest()
config.suffixes = [".l2"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = tempfile.mkdtemp(prefix="lit-")
config.recursive = True
config.substitutions.append(("%l2", "l2"))
config.substitutions.append(("%filecheck", "filecheck"))
