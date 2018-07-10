import pandas
import pickle
pickle_in = open("train_output.pkl","rb")
example_dict = pickle.load(pickle_in)
print (example_dict[:5])
