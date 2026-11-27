import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""Comparing the mutation coverage across and between datasets"""

#load datasets as pandas df
megascale_df = pd.read_parquet('dataset3_single_train.parquet')
s2648_df = pd.read_csv('s2648.csv')
ssym_df = pd.read_csv('ssym.csv')
s669_df = pd.read_csv('S669.csv')

#define set of 20 amino acids
amino_acids = ["A",
               "C",
               "D",
               "E",
               "F",
               "G",
               "H",
               "I",
               "K",
               "L",
               "M",
               "N",
               "P",
               "Q",
               "R",
               "S",
               "T",
               "V",
               "W",
               "Y"]

#function to compute mutation frequency matrix
def generate_frequency_matrix(df, mut_col):
    matrix = pd.crosstab(df[mut_col].str[0], df[mut_col].str[-1], dropna=False)
    matrix = matrix.reindex(index=amino_acids, columns=amino_acids, fill_value=0)

    return matrix

megascale_matrix = generate_frequency_matrix(megascale_df, 'mut_type')
s2648_matrix = generate_frequency_matrix(s2648_df, 'MUT')
s669_matrix = generate_frequency_matrix(s669_df, 'PDB_Mut')
ssym_matrix = generate_frequency_matrix(ssym_df, 'MUT')

# print(f'Megascale matrix:\n {megascale_matrix} \n')
# print(f'Ssym matrix:\n {ssym_matrix} \n')

#normalise matrix so frequencies occur as proportions
megascale_freq = megascale_matrix/len(megascale_df)
s2648_freq = s2648_matrix/len(s2648_df)
s669_freq = s669_matrix/len(s2648_df)
ssym_freq = ssym_matrix/len(ssym_df)

# print(f'Megascale normalised:\n {megascale_freq} \n')
# print(f'Ssym normalised:\n {ssym_freq} \n')

#plot heatmaps to visualise bias (not normalized)
fig, axes = plt.subplots(figsize=(18,7))
sns.heatmap(
    megascale_matrix,
    cmap="Blues",
    cbar=True,
    ax=axes,
    xticklabels=True,
    yticklabels=True,
)
axes.set_title("Megascale Mutational Space Coverage")
axes.set_xlabel("Mutant Residue")
axes.set_ylabel("Wild-Type Residue")
plt.show()

fig1, axes1 = plt.subplots(figsize=(18,7))
sns.heatmap(
    ssym_matrix,
    cmap="Blues",
    cbar=True,
    ax=axes1,
    xticklabels=True,
    yticklabels=True,
)
axes1.set_title("Ssym Mutational Space Coverage")
axes1.set_xlabel("Mutant Residue")
axes1.set_ylabel("Wild-Type Residue")
plt.show()

fig2, axes2 = plt.subplots(figsize=(18,7))
sns.heatmap(
    s2648_matrix,
    cmap="Blues",
    cbar=True,
    ax=axes2,
    xticklabels=True,
    yticklabels=True,
)
axes2.set_title("S2648 Mutational Space Coverage")
axes2.set_xlabel("Mutant Residue")
axes2.set_ylabel("Wild-Type Residue")
plt.show()

fig3, axes3 = plt.subplots(figsize=(18,7))
sns.heatmap(
    s669_matrix,
    cmap="Blues",
    cbar=True,
    ax=axes3,
    xticklabels=True,
    yticklabels=True,
)
axes3.set_title("S669 Mutational Space Coverage")
axes3.set_xlabel("Mutant Residue")
axes3.set_ylabel("Wild-Type Residue")
plt.show()

#plot heatmaps to visualise bias
fig, axes = plt.subplots(2, 2, figsize=(18,7))
vmin = min(megascale_freq.min().min(), s2648_freq.min().min(), s669_freq.min().min(), ssym_freq.min().min())
vmax = max(megascale_freq.max().max(), s2648_freq.max().max(), s669_freq.max().max(), ssym_freq.max().max())
sns.heatmap(
    megascale_freq,
    cmap="Blues",
    vmin=vmin,
    vmax=vmax,
    cbar=True,
    ax=axes[0, 0],
    xticklabels=True,
    yticklabels=True,
)
axes[0, 0].set_title("Megascale Mutational Space Coverage")
axes[0, 0].set_xlabel("Mutant Residue")
axes[0, 0].set_ylabel("Wild-Type Residue")

sns.heatmap(
    ssym_freq,
    cmap="Blues",
    vmin=vmin,
    vmax=vmax,
    cbar=True,
    ax=axes[0, 1],
    xticklabels=True,
    yticklabels=True,
)
axes[0, 1].set_title("Ssym Mutational Space Coverage")
axes[0, 1].set_xlabel("Mutant Residue")
axes[0, 1].set_ylabel("Wild-Type Residue")

sns.heatmap(
    s2648_freq,
    cmap="Blues",
    vmin=vmin,
    vmax=vmax,
    cbar=True,
    ax=axes[1, 0],
    xticklabels=True,
    yticklabels=True,
)
axes[1, 0].set_title("S2648 Mutational Space Coverage")
axes[1, 0].set_xlabel("Mutant Residue")
axes[1, 0].set_ylabel("Wild-Type Residue")

sns.heatmap(
    s669_freq,
    cmap="Blues",
    vmin=vmin,
    vmax=vmax,
    cbar=True,
    ax=axes[1, 1],
    xticklabels=True,
    yticklabels=True,
)
axes[1, 1].set_title("S669 Mutational Space Coverage")
axes[1, 1].set_xlabel("Mutant Residue")
axes[1, 1].set_ylabel("Wild-Type Residue")

plt.tight_layout()
plt.show()

#difference matrices
diff_meg_ssym = megascale_freq - ssym_freq
diff_meg_s669 = megascale_freq - s669_freq
diff_s2648_ssym = s2648_freq - ssym_freq
diff_s2648_s669 = s2648_freq - s669_freq
diff_ssym_s669 = ssym_freq - s669_freq

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

sns.heatmap(
    diff_meg_ssym, 
    cmap="coolwarm", 
    center=0, 
    ax=axes[0, 0], 
    cbar=True,
    xticklabels=True, 
    yticklabels=True
)
axes[0, 0].set_title("Megascale minus Ssym(Blue = More Megascale, Red = More Ssym)")
axes[0, 0].set_xlabel("Mutant Residue")
axes[0, 0].set_ylabel("Wild-Type Residue")

sns.heatmap(
    diff_meg_s669, 
    cmap="coolwarm", 
    center=0, 
    ax=axes[0, 1], 
    cbar=True,
    xticklabels=True, 
    yticklabels=True
)
axes[0, 1].set_title("Megascale minus S669(Blue = More Megascale, Red = More S669)")
axes[0, 1].set_xlabel("Mutant Residue")
axes[0, 1].set_ylabel("Wild-Type Residue")

sns.heatmap(
    diff_s2648_ssym, 
    cmap="coolwarm", 
    center=0, 
    ax=axes[1, 0], 
    cbar=True,
    xticklabels=True, 
    yticklabels=True
)
axes[1, 0].set_title("S2648 minus Ssym(Blue = More S2648, Red = More Ssym)")
axes[1, 0].set_xlabel("Mutant Residue")
axes[1, 0].set_ylabel("Wild-Type Residue")

sns.heatmap(
    diff_s2648_s669, 
    cmap="coolwarm", 
    center=0, 
    ax=axes[1, 1], 
    cbar=True,
    xticklabels=True, 
    yticklabels=True
)
axes[1, 1].set_title("S2648 minus S669(Blue = More S2648, Red = More S669)")
axes[1, 1].set_xlabel("Mutant Residue")
axes[1, 1].set_ylabel("Wild-Type Residue")

plt.tight_layout()
plt.show()