This repository provides a simple GCN-based model trial to predict organic molecule solubilities in water. The file named as "GCN_solubility_ToyProblem.py" is the main code file. It uses the ESOL dataset named as "delaney-processed.csv" located in the folder "data". To be able to run the file and print all the outputs adjusted, please follow these steps: 

1) The necessary libraries and models should be installed on the computer. Some of them are:
     -PyTorch, PyTorch Geometric
     -Rdkit
     -pandas
     -NumPy
     -matplotlib

2) The path to the file providing the dataset may requiring revising according to the folder containing the code file. The corresponding code line is the line 47 in the code file:
     file_path = "CENG562_TermProject/data/delaney-processed.csv"
   
