# Temporary inspection probe file
import inspect
from PAMI.extras.convert.DF2DB import DF2DB

print("SIGNATURE:", inspect.signature(DF2DB.__init__))
print("SOURCE:\n", inspect.getsource(DF2DB.__init__))
