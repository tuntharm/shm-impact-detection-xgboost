import sys
import xgboost
import numpy
import pandas

print("Python:", sys.version)
print("XGBoost:", xgboost.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Scikit-learn:", sys.modules['sklearn'].__version__)