import math
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator, LinearLocator, LogLocator, MultipleLocator
import seaborn as sns

from pulse.caster import list_unique, only_prefix, to_list, to_parsed, parsed_to_str
from pulse.errors import raise_error_if

LABELS = {'acq': 'Acquisition', 'rosenb': 'Rosenbrock', 'styb': 'Styblinski-Tang',
          'best_target_unit_gap': 'Normalized Improvement',
          'curr_input_acq_mae': 'Input MAE Previous Acquisitions',
          'curr_input_best_mae': 'Input MAE Incumbent',
          'curr_input_gap_mae': 'Input MAE Optimality Gap',
          'best_auc': 'Optimality Trajectory'
         }


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap

def get_label(name, label_dict=LABELS):
    return label_dict.get(name, name)

def get_colors(names, sort=True, ascending=True, baseline='sk', field='ik'):
    if isinstance(names, pd.DataFrame):
        names = get_unique_level_values(names, 'name')
    colors = get_colors_group(names, field=field, sort=sort, ascending=ascending)
    colors[baseline] = 'k'
    return colors

def get_colors_group(names, field, group_by=None, sort=True, ascending=True):
    colors = {}
    names = only_prefix(names, field)
    if names:
        if sort:
            names = sorted(names, reverse=not ascending)
        parsed = to_parsed(names, field=field)
        order = 'reverse' if group_by is not None and list(parsed.keys())[0] != group_by else None
        if order:
            names = parsed_to_str(parsed, field=field, order=order, sort=sort, ascending=ascending)
            parsed = to_parsed(names, field=field)
        gi, gj = list(map(lambda x: to_list(list_unique(x)), parsed.values()))
        if len(gi) <= 5 and len(gj) <= 4:
            cmap20b = plt.get_cmap('tab20b')
            cmap20c = plt.get_cmap('tab20c')
            cmap = lambda i, j: cmap20b(3 * 4 + 3 - j) if i == 4 else cmap20c(i * 4 + 3 - j)
        else:
            ccmap = categorical_cmap(len(gi), len(gj))
            cmap = lambda i, j:  ccmap(i * len(gj) + len(gj) - 1 - j)
        colors.update({name: cmap(gi.index(vi), gj.index(vj)) for name, vi, vj in zip(names, *parsed.values())})
    return colors

def get_unique_level_values(df, level: [int, str] = 0, axis=1, to_list=True):
    index = df.index if axis == 0 else df.columns
    values = index.get_level_values(level).unique()
    if to_list:
        values = values.tolist()
    return values

def plot_df(df, names=None, colors=None, disp='median', ax=None, title=None, yscale='linear',
            lw=2.5, legend=True, fontsize_factor=1, ylabel_title=False, fill_between=True,
            error_bars=False, error_disp='median', error_x_interval=25, error_x_between=2, **kwargs):
    raise_error_if(disp not in ['mean', 'median'])
    getlabel = lambda name: get_label(name, label_dict=kwargs.get('label_dict', LABELS))
    if names is None:
        names = get_unique_level_values(df, 'name')
    colors = colors or {}
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    if title:
        ax.set_title(title, fontsize=18 * fontsize_factor)
    elif ylabel_title:
        ax.set_title(getlabel(df.name), fontsize=18 * fontsize_factor)
    ax.set_yscale(yscale)
    idx = df.index.to_list()
    ax.set_xlabel(getlabel(df.index.name), fontsize=14 * fontsize_factor)
    if not ylabel_title:
        ax.set_ylabel(getlabel(df.name), fontsize=14 * fontsize_factor)
    ax.set_xlim([idx[0], idx[-1]])
    vmin = df.xs(disp, axis=1, level='stat').min().min()
    if vmin == 0 and yscale =='log':
        vmin = df.xs(disp, axis=1, level='stat').replace(0, 1).min().min()
    vmax = df.xs(disp, axis=1, level='stat').max().max()
    ymin = math.floor(vmin) if yscale == 'linear' else (10 ** np.floor(np.log10(vmin)) if vmin != 0 else 1e-6)
    ymax = 10 ** np.ceil(np.log10(vmax))
    ax.set_ylim([ymin, ymax])
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    if yscale == 'linear':
        ax.yaxis.set_major_locator(LinearLocator(11))
    elif yscale == 'log':
        ax.yaxis.set_major_locator(LogLocator(10))
    ax.tick_params(axis='x', which='major', labelsize=10 * fontsize_factor)
    ax.tick_params(axis='y', which='major', labelsize=12 * fontsize_factor)
    ax.grid()
    ax.grid(which="minor", alpha=0.2)
    for i, name in enumerate(names):
        sdf = df.xs(name, axis=1, level='name')
        value = sdf[disp].values
        c = colors.get(name)
        options = {} if c is None else dict(color=c)
        ax.plot(idx, value, lw=lw, label=getlabel(name), **options)
        if c is None:
            c = ax.lines[-1].get_color()
            colors[name] = c
            options.update(dict(color=c))
        if fill_between:
            if disp == 'mean':
                delta = sdf['std'].values
                yls = value - delta
                yms = value + delta
            else:
                yls = sdf['q1'].values
                yms = sdf['q3'].values
            ax.fill_between(idx, yls, yms, alpha=0.1, **options)
        if error_bars:
            eidx_start = (i * error_x_between) % error_x_interval
            eidx = idx[eidx_start::error_x_interval]
            value = value[eidx_start::error_x_interval]
            if error_disp == 'mean':
                delta = sdf['std'].values[eidx_start::error_x_interval]
                yls = value - delta
                yms = value + delta
            else:
                yls = sdf['q1'].values[eidx_start::error_x_interval]
                yms = sdf['q3'].values[eidx_start::error_x_interval]
            for x, yl, ym in zip(eidx, yls, yms):
                ax.axvline(x, yl, ym, alpha=0.9, **options)
    if legend:
        ax.legend(bbox_to_anchor=(1.01, 1))
    return colors

def plot_heatmap(df, annot_kws=None, cmap=None, ax=None, figsize=None, title=None, fontsize_factor=1,
                 cbar=False, center=None, **kwargs):
    getlabel = lambda name: get_label(name, label_dict=kwargs.get('label_dict', LABELS))
    if not annot_kws:
        annot_kws = {"size": 14}
    if not cmap:
        cmap = matplotlib.cm.RdYlGn
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize if figsize else (8, 8))
    if center == 'mean':
        center = np.mean(df.values.reshape(-1))
    elif center == 'median':
        center = np.median(df.values.reshape(-1))
    sns.heatmap(df, annot=True, annot_kws=annot_kws, alpha=1.0, center=center, cbar=cbar, cmap=cmap,
                ax=ax, linewidths=.5, **kwargs)
    if title:
        ax.set_title(title, fontsize=18 * fontsize_factor)
    ax.set_xlabel(getlabel(ax.get_xlabel()), fontsize=14 * fontsize_factor)
    ax.set_ylabel(getlabel(ax.get_ylabel()), fontsize=14 * fontsize_factor)
    ax.tick_params(axis='x', which='major', labelsize=12 * fontsize_factor)
    ax.tick_params(axis='y', which='major', labelsize=12 * fontsize_factor)
    return ax