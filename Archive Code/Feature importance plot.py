# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 16:52:51 2023

@author: liy45
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colors

def plot_feature_importance(df, ax=None, top = 5, title_text = None, custom_c = True):
    
    if top is not None:   
        df_fea = df.sort_values('Mean').iloc[-top:]
    else:
        df_fea = df.sort_values('Mean')
        
    # fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(8, 8)) 
    if custom_c:
        cmap = plt.get_cmap("Wistia")
        c = [cmap((p-df_fea['Mean'].min())/(df_fea['Mean'].max() - df_fea['Mean'].min()))  for p in df_fea['Mean']]
        #[{p<0.03: 'blue', 0.03<=p<=0.06: 'yellow', p>0.06: 'red'}[True] for p in df_fea['Mean']]
    else:
        cmap = '#86bf91'
    if ax is None:       
        ax = df_fea.plot.barh(y='Mean', xerr = 'SEM',  color=c, fontsize = 16)
    else:
        df_fea.plot.barh(y='Mean', xerr = 'SEM', ax = ax, color=c, fontsize = 16)
    
    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Switch off ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    # Draw vertical axis lines
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)
        
    ax.legend().set_visible(False)
    
    # Set x-axis label
    ax.set_xlabel("Feature importance score", labelpad=20, weight='bold', size=18)
    
    # Set y-axis label
    ax.set_ylabel("Feautures", labelpad=20, weight='bold', size=16)
    
    ax.set_title(title_text + ' - ' + ml_type, size = 20)
    
    ax.set_xticks([0, 0.05, 0.1, 0.15])
    
    return ax




''' plot importance score from excel'''

df_rename = pd.read_excel(r'D:\OneDrive - Personal\OneDrive\ESCC_LN\ESCC-Neck\Results\Result updated 2023-12-05\Variable names.xlsx')

fig, ax = plt.subplots(nrows = 3, ncols = 2, figsize = (10, 8), sharex = True)
axe = ax.ravel()
for i, ml_type in enumerate(['LR', 'RF', 'XGB', 'LGB', 'KNN', 'SVM']):
    # ml_type = 'LGB'

    df_importance = pd.read_excel(r'D:\OneDrive - Personal\OneDrive\ESCC_LN\ESCC-Neck\Results\Result updated 2023-12-05\Preop only\{}.xlsx'.format(ml_type), 
                                  sheet_name = 'importance', index_col=0)  
    
    a = dict(df_rename.values)
    
    b = df_importance.rename(index=a)
    
    axe[i] = plot_feature_importance(b, ax = axe[i], top = 5, title_text = 'Cervical nodal metastasis')
    
for i, ml_type in enumerate(['Logistic regression', 'Random forest', 'XGBoost', 'LightGBM', ' K-Nearest Neighbors', 'Support vector machines']):
    axe[i].set_ylabel('')
    axe[i].set_xlabel('')
    axe[i].tick_params(axis='y', which='major', labelsize=14)
    axe[i].tick_params(axis='x', which='major', labelsize=14)
    axe[i].set_title(ml_type, size = 16)

fig.supxlabel('Feature importance score', size = 18)
fig.tight_layout()
fig.savefig(r'C:\Users\liy45\Desktop\Task 2.png', dpi=300)
plt.show()
