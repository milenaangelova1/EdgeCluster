from src.colors import COLORS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('/Users/milenaangelova/git-repo/EdgeCluster/notebooks/total.csv', encoding='utf-8')
print(df.shape)
# df = df[df['segment']<=num_segments]

palette = sns.color_palette('deep', n_colors=len(set(df['cluster'])))

sns.scatterplot(x=df['x'], y=df['y'], hue=df['cluster'], palette=palette)
plt.xticks(np.arange(0, 1.1, 0.1))  
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(loc='center left', bbox_to_anchor=(0.0, -0.3), ncol=5, borderaxespad=0, title='Clusters')
plt.show()