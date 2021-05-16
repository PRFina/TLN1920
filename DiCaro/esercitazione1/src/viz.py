import numpy as np
import matplotlib.pyplot as plt

def plot_concept_matrix(scores, ax, title):
    """ Plot a 2x2 matrix where each element is the avg similarity 
        among the different concept definitions pairs.

    Args:
        scores (numpy.ndarray): a numpy ndarray of shape (2,2) where each element is the avg similarity.
        ax (matplotlib.Axes): axes to draw the plot on.
        title (str): plot title
    """
    print(scores)
    col_labels = ['Generico', 'Specifico']
    row_labels = ['Concreto', 'Astratto']
    line_color = '#2143c2'
    text_color = '#292d33'
    
    ax.set_title(title, color=text_color)

    ax.imshow(np.ones((2,2)), cmap='binary')  # binary cmap to show a white BG

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(col_labels, fontsize='large', color=text_color)
    ax.set_yticklabels(row_labels, fontsize='large', color=text_color)

    for spine in ['top','bottom','right','left']:
        ax.spines[spine].set_color(line_color)
        ax.spines[spine].set_linewidth(1.8)
        ax.spines[spine].set_alpha(0.85)

    ax.axvline(0.5, color=line_color, alpha=0.35)
    ax.axhline(0.5, color=line_color, alpha=0.35)
    
    # draw text for similarity scores
    for i in range(2):
        for j in range(2):
            ax.text(j,i, f"{scores[i,j]:10.4f}", 
                        ha="center", va="center",
                        fontsize='medium',
                        color=text_color)