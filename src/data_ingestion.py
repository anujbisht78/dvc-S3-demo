import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

df=pd.read_csv('data/raw/Placement.csv')

# seperating features and target variable
x=df.drop(columns=['Placed'])
y=df['Placed']

# scaling the features
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

# applying PCA
pca=PCA(n_components=3)

x_pca=pca.fit_transform(x_scaled)

# creating a new dataframe with PCA components
pca_df=pd.DataFrame(data=x_pca, columns=['PCA1', 'PCA2','PCA3'])
pca_df['Placed']=y.values

# saving the processed data
pca_df.to_csv(os.path.join('data','processed','student_performance_pca.csv'), index=False)