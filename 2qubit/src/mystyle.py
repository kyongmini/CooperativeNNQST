### @ mystle.py

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console

console = Console()

def print_banner(text: str):
    console.rule(f"[bold cyan]{text}[/bold cyan]")

def print_text(text: str):
    console.rule(f"[bold dim]{text}[/bold dim]")

def my_style(fontsize=12, a_font='STIX Two Text'):
    """Sets a custom matplotlib my style"""

    plt.rcParams.update({
        'font.family': "sans-serif",
        'font.sans-serif': ["DejaVu Sans", "Liberation Sans", "STIX Two Text"],     
        'mathtext.fontset': 'custom',        
        'mathtext.rm': 'Arial',              
        'mathtext.it': 'Arial:italic',       
        'mathtext.bf': 'Arial:bold',         
        'font.size' : fontsize,
        'axes.labelsize' : fontsize,
        'axes.titlesize' : fontsize + 2,
        'xtick.labelsize' : fontsize - 1,
        'ytick.labelsize' : fontsize - 1,
        'legend.fontsize' : fontsize - 1,
        'axes.linewidth' : 1.2,
        'xtick.direction' : 'in',
        'ytick.direction' : 'in',
        'xtick.major.size' : 5,
        'ytick.major.size' : 5,
        'xtick.major.width' : 1.0,
        'ytick.major.width' : 1.0,
        'lines.linewidth' : 2.0,
        'lines.markersize' : 6,
        'legend.frameon' : False,
        'legend.loc' : 'best',
        'figure.dpi' : 300,
        'figure.facecolor' : 'white',
        'axes.facecolor' : 'white',
        'savefig.facecolor' : 'white',
        'savefig.edgecolor' : 'white',
        'savefig.bbox' : 'tight'
        }
    )
