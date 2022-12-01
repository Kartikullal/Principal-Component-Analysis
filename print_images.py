import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns



#Plot top PCA transformation for top 3 eigen values

def scatter_plot(X_train_sample_top3,Y_train_sample):
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection ='3d')

    markers = ['o','v','s','*','^','<','>','P','X','D']
    colors = ['tab:blue', 'tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

    for i in range(10):
        rows = (Y_train_sample == i)
        ax.scatter(xs=X_train_sample_top3[rows,0], 
                ys=X_train_sample_top3[rows,1], 
                zs=X_train_sample_top3[rows,2], label = 'Label:' + str(i), c = colors[i], marker = markers[i], alpha = 0.3)
        
    plt.legend(bbox_to_anchor=(1.25, 0.5), loc='right', borderaxespad=0)
    plt.show()
    fig.savefig('scatter.pdf')

xs