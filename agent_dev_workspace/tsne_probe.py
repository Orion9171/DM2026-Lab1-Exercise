# Temporary probe for sklearn TSNE details
import sklearn
from sklearn.manifold import TSNE
import inspect

print("SKLEARN VERSION:", sklearn.__version__)
print("TSNE INIT SIGNATURE:", inspect.signature(TSNE.__init__))
