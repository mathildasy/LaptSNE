import numpy as np
# import load_data

# @attribute class {Agresti,Ashbacher,Auken,Blankenship,Brody,Brown,Bukowsky,CFH,Calvinnme,Chachra,Chandler,Chell,Cholette,Comdet,Corn,Cutey,Davisson,Dent,Engineer,Goonan,Grove,Harp,Hayes,Janson,Johnson,Koenig,Kolln,Lawyeraau,Lee,Lovitt,Mahlers2nd,Mark,McKee,Merritt,Messick,Mitchell,Morrison,Neal,Nigam,Peterson,Power,Riley,Robert,Shea,Sherwin,Taylor,Vernon,Vision,Walters,Wilson}

# # X, y = load_data.EEG(-1,0)
# X = np.load('./data/WAV.npy')
# y = X[:,-1]
# X = X[:,:-1]

# print(X.shape, y.shape)
# print(np.unique(y))

# np.save('./data/WAV_X.npy', X)
# np.save('./data/WAV_Y.npy', y)

# from scipy.io import arff
# import pandas as pd

# data = arff.loadarff('./data/Amazon_initial_50_30_10000.arff')
# df = pd.DataFrame(data[0])

# np.save('./data/Amazon.npy', data)

# X = np.loadtxt('./data/arcene_train.data')
# print(X, X.shape)
# np.save('./data/ARCENE_X.npy', X)
# y = np.loadtxt('./data/arcene_train.labels')
# print(y, y.shape)
# np.save('./data/ARCENE_Y.npy', y)