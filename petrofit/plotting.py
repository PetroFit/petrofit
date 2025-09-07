import os
from contextlib import contextmanager

import numpy as np 

from photutils.aperture import EllipticalAperture

from matplotlib import pyplot as plt 

from .config.mpl import (_package_plot_style, FIG_UNIT, FIG_UNIT_MAX_WIDTH, 
                         AXS_BORDER_THICKNESS, TICK_LEN, SMALL_SIZE, MEDIUM_SIZE, 
                         LARGE_SIZE)

__all__ = [
    "package_plot_style",
    "figure", 
    "subplots",
    "mpl_tick_frame",
    "savefig",
    "plot_apertures",
    "arrow",
    "r_arrow",
    "imshow",
    "contour",
    "plot_segments",
    "plot_segment_residual",
    "plot_target",
]

package_plot_style = _package_plot_style

def _get_ax(ax):
    """Get the current axes."""
    with package_plot_style():
        if ax is None:
            return plt.gca()
        return ax

def figure(**kwargs):
    """Create a new figure with package styling applied."""
    with package_plot_style():
        return plt.figure(**kwargs)


def subplots(nrows=1, ncols=1, label_gap=0.25, **kwargs):
    """Create a new subplots with package styling applied."""
    with package_plot_style():
        if 'figsize' not in kwargs:
            if FIG_UNIT * ncols < FIG_UNIT_MAX_WIDTH:
                kwargs.setdefault('figsize', (FIG_UNIT * ncols, FIG_UNIT * nrows))
            elif nrows == ncols:
                kwargs.setdefault('figsize', (FIG_UNIT_MAX_WIDTH, FIG_UNIT_MAX_WIDTH))
            else:
                kwargs.setdefault('figsize', (FIG_UNIT_MAX_WIDTH, (FIG_UNIT_MAX_WIDTH * nrows / ncols) - label_gap))
            
            if label_gap != 0 and 'gridspec_kw' not in kwargs:
                gridspec_kw = {'wspace': label_gap, 'hspace': label_gap}
                kwargs.setdefault('gridspec_kw', gridspec_kw)

        return plt.subplots(nrows=nrows, ncols=ncols, **kwargs)


def savefig(filename, dpi=500):
    """Save the current figure to a file."""
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.05)


def mpl_tick_frame(
        ax=None, minorticks=True, length=None,
        width=None, tick_color=None, tick_fontsize=SMALL_SIZE, 
        direction='in', all_sides=True,
):
    """
    Set the tick parameters for a Matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        The axes to modify. If None, the current axes are used.

    minorticks : bool, optional
        Whether to enable minor ticks. Default is True.

    length : float, optional
        The length of the ticks in points. If None, the default length is used.

    width : float, optional
        The width of the ticks in points. If None, the default width is used.

    tick_fontsize : float, optional
        The font size of the tick labels. Default is SMALL_SIZE.
    """
    ax = _get_ax(ax)

    if minorticks:
        ax.minorticks_on()
    
    if length is None:
        length = TICK_LEN
    if width is None:
        width = AXS_BORDER_THICKNESS

    optional_kwargs = {}
    if tick_color is not None:
        optional_kwargs['color'] = tick_color


    ax.tick_params(
        which='minor', direction=direction, top=all_sides, right=all_sides,
        width=width, length=length/2, labelsize=tick_fontsize,
        **optional_kwargs
    )
    ax.tick_params(
        which='major', direction=direction, top=all_sides, right=all_sides,
        width=width, length=length, labelsize=tick_fontsize,
        **optional_kwargs
    )


def imshow(image, ax=None, tick_frame=True, **kwargs):
    """Wrapper for plt.imshow with package_plot_style context."""
    with package_plot_style():
        ax = _get_ax(ax)
        if tick_frame:
            mpl_tick_frame(ax=ax, direction='out', all_sides=False)
        return ax.imshow(image, **kwargs)
    

def contour(image, ax=None, show_clabel=False, fontsize_clabel=SMALL_SIZE, fmt=None, clabel_kwargs={}, **kwargs):
    """Wrapper for plt.imshow with package_plot_style context."""
    with package_plot_style():
        ax = _get_ax(ax)
        CS = plt.contour(image, **kwargs)
        if show_clabel:
            if fontsize_clabel is not None:
                assert 'fontsize' not in list(clabel_kwargs.keys()), "fontsize_clabel and fontsize in clabel_kwargs cannot be both set."
                clabel_kwargs['fontsize'] = fontsize_clabel
            if fmt is not None:
                assert 'fmt' not in list(clabel_kwargs.keys()), "fmt and fmt in clabel_kwargs cannot be both set."
                clabel_kwargs['fmt'] = fmt
            ax.clabel(CS, CS.levels, **clabel_kwargs)
        return CS


def arrow(p0, p1, c='k', hl=None, hc=None, hstart=0.9, lw=0.25, **kwargs):
    """
    Draw an arrow from p0 to p1.

    Parameters
    ----------
    p0 : array-like
        The starting point of the arrow (x, y).

    p1 : array-like
        The ending point of the arrow (x, y).

    c : color
        The color of the arrow.

    hl : float, optional
        The head length of the arrow. If None, the default length is used.

    hc : color, optional
        The head color of the arrow. If None, the arrow color is used.

    hstart : float, optional
        The starting point of the head. Default is 0.9.

    lw : float, optional
        The line width of the arrow. Default is 0.25.

    kwargs : keyword arguments
        Additional arguments passed to the arrow function.
    """
    p0 = np.array(p0)
    p1 = np.array(p1)

    assert p0.shape == (2,), "p0 must be a 2D point"
    assert p1.shape == (2,), "p1 must be a 2D point"

    d = np.round(p1 - p0, 6)
    
    length = np.linalg.norm(d)
    if length < 1e-4:
        return 
    
    if hc is None:
        hc = c
 
    if hl is None:
        hl = length
        
    plt.arrow(p0[0], p0[1], d[0], d[1],
              head_width=hl*0.0, 
              head_length=hl*0.12, 
              overhang=-0.3, lw=lw,
              length_includes_head=True,
              facecolor=hc, edgecolor=c, aa=True, **kwargs)
    
    plt.arrow(p0[0], p0[1], d[0]*hstart, d[1]*hstart,
              head_width=hl*0.06, 
              head_length=hl*0.12, 
              overhang=-0.3, lw=lw,
              length_includes_head=True,
              facecolor=hc, edgecolor=c, aa=True, **kwargs)
    

def r_arrow(p0, r, theta, **kwargs):
    """
    Draw a right arrow from p0 with length r and angle theta.
    See `petrofit.plotting.arrow` for more details.
    """
    assert np.isscalar(r), "r must be a scalar"
    p0 = np.array(p0)
    d = r * np.array([np.cos(theta), np.sin(theta)])
    p1  = p0 + d
    arrow(p0, p1, **kwargs)


def plot_apertures(
    image=None, apertures=[], vmin=None, vmax=None, color="white", lw=1.5, cmap="Greys_r", ax=None
):
    """
    Plot apertures on image

    Parameters
    ----------
    image : numpy.ndarray
        2D image array.

    apertures : list
        List of photutils Apertures.

    vmin, vmax : float
        vmax and vmin values for plot.

    color : string
        Matplotlib color for the apertures, default=White.

    lw : float
        Line width of aperture outline.

    cmap : string
        Colormap to use for the image.

    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, the current axes are used.
    """
    with package_plot_style():
        ax = _get_ax(ax)
        if image is not None:
            imshow(image, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)

        for aperture in apertures:
            aperture.plot(axes=ax, color=color, lw=lw)

def plot_segments(segm, image=None, vmin=None, vmax=None, alpha=0.5, ax=None, title=None):
    """
    Plot segmented areas over an image (2D array, if provided)
    """
    with package_plot_style():
        ax = _get_ax(ax)

        cmap = segm.make_cmap(seed=np.random.randint(1000000))

        if image is not None:
            ax.imshow(image, vmin=vmin, vmax=vmax, cmap="gist_gray")

        ax.imshow(segm, cmap=cmap, alpha=alpha)

        if title is not None:
            ax.set_title(title)

        ax.set_xlabel("Pixels")
        ax.set_ylabel("Pixels")
        mpl_tick_frame(ax=ax, tick_color='white')


def plot_segment_residual(segm, image, vmin=None, vmax=None, ax=None):
    """
    Plot segment subtracted image (residual)
    """
    with package_plot_style():
        ax = _get_ax(ax)
        temp = image.copy()
        temp[np.where(segm.data != 0)] = 0.0
        ax.imshow(temp, vmin=vmin, vmax=vmax)
        ax.set_xlabel("Pixels")
        ax.set_ylabel("Pixels")
        mpl_tick_frame(ax=ax, tick_color='white')


def plot_target(
     position, image=None, size=None, c="k", lw=None, vmin=None, vmax=None, ax=None, marker_base_size=2,
):
    """
    Plot an image with a target marker.

    Parameters
    ----------
    position : tuple of int
        (x, y) coordinates of the target location.

    image : np.ndarray or object with `data` attribute
        The image to be displayed.

    size : int, optional
        The pixel size around the target to display.
        If not specified, it defaults to the maximum dimension of the image.

    c : str, optional
        Color of the target marker. Default is red (`'r'`).

    lw : int or float, optional
        Line width of the target marker.

    vmin, vmax : int or float, optional
        Values to anchor the colormap.
        
    marker_base_size : int, optional
        Base size of the marker which gets scaled relative to the image size.
        Default is 2.

    Notes
    -----
    The target is plotted as a red '+' at the given position. The displayed
    region is determined by the `size` parameter centered at the target position.
    """

    if size is None:
        size = max(image.shape)
    x, y = position

    # Calculate marker size relative to the average size of the image dimensions
    marker_size = np.mean(image.shape) / 20 * marker_base_size
    with package_plot_style():
        ax = _get_ax(ax)
        if image is not None:
            if hasattr(image, 'data'):
                image = image.data
            imshow(image, vmin=vmin, vmax=vmax)
        ax.plot(x, y, "+", c=c, label="Target", markersize=marker_size, markeredgewidth=lw)
        ax.set_xlim(x - (size / 2.0), x + (size / 2.0))
        ax.set_ylim(y - (size / 2.0), y + (size / 2.0))

def _plot_correction_grid(
        pc,
        x0=None,
        y0=None,
        z0=None,
        cmap="rainbow",
        target_c="black",
        cmap_key="n",
        colorbar_label=None,
        suptitle=None,
        axs=None,
    ):
    """
    Plots a grid of scatter plots for the given data.
    Parameters
    ----------
    x0, y0, z0 : float, optional
    cmap : str, optional
        Colormap to use for the scatter plots (default "hot").
    target_c : str, optional
        Color to use for highlighting the target point (default is "blue").
    cmap_key : str, optional
        Key to use for colormap data from the grid (default is "n").
    colorbar_label : str, optional
        Label for the colorbar (default is None).
    suptitle : str, optional
        Suptitle for the figure (default is None).
    axs : list of matplotlib.axes.Axes, optional
        List of 3 axes to plot on (default is None, which creates new axes).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plots.
    axs : list of matplotlib.axes.Axes
        The list of axes containing the scatter plots.
    Notes
    -----
    This function creates a grid of 3 scatter plots showing the relationships
    between x, y, and z coordinates with color mapping based on `cmap_key`.
    If `x0`, `y0`, and `z0` are provided, the closest point in the grid is
    highlighted and connected to the target point.
    """
    with package_plot_style():
        if axs is None:
            fig, axs = subplots(1, 3)
        else:
            assert len(axs) == 3, "axs should be a list of 3 axis"
            fig = axs[0].figure

        cm = plt.cm.get_cmap(cmap)

        sim_n_list = pc.grid[cmap_key]

        scatter_vmin = min(sim_n_list)
        scatter_vmax = max(sim_n_list)
        
        ax = axs[0]
        sc = ax.scatter(
            pc.x,
            pc.y,
            c=sim_n_list,
            vmin=scatter_vmin,
            vmax=scatter_vmax,
            s=35,
            cmap=cm,
        )
        ax.set_xlabel(r"$r_{{p}}(\eta=0.2)$")
        ax.set_ylabel(r"$r_{{50}}$")
        mpl_tick_frame(ax=ax)

        ax = axs[1]
        sc = ax.scatter(
            pc.x,
            pc.z,
            c=sim_n_list,
            vmin=scatter_vmin,
            vmax=scatter_vmax,
            s=35,
            cmap=cm,
        )
        ax.set_xlabel(r"$r_{{p}}(\eta=0.2)$")
        ax.set_ylabel(r"$C_{2080}$")
        mpl_tick_frame(ax=ax)

        ax = axs[2]
        sc = ax.scatter(
            pc.y,
            pc.z,
            c=sim_n_list,
            vmin=scatter_vmin,
            vmax=scatter_vmax,
            s=35,
            cmap=cm,
        )
        ax.set_xlabel(r"$r_{{50}}$")
        ax.set_ylabel(r"$C_{2080}$")
        mpl_tick_frame(ax=ax)

        if None not in [x0, y0, z0]:
            idx = pc._dr(x0, y0, z0).argmin()
            cx, cy, cz = pc.x[idx], pc.y[idx], pc.z[idx]
            lw = 2
            axs[0].scatter(x0, y0, marker="o", s=200, ec=target_c, fc="None", lw=lw)
            axs[1].scatter(x0, z0, marker="o", s=200, ec=target_c, fc="None", lw=lw)
            axs[2].scatter(y0, z0, marker="o", s=200, ec=target_c, fc="None", lw=lw)

            axs[0].plot([x0, cx], [y0, cy], marker="o", lw=lw, c=target_c)
            axs[1].plot([x0, cx], [z0, cz], marker="o", lw=lw, c=target_c)
            axs[2].plot([y0, cy], [z0, cz], marker="o", lw=lw, c=target_c)
        
        cbar = plt.colorbar(sc, ax=axs[2], pad=0, label=colorbar_label if colorbar_label else cmap_key)

    return fig, axs


def _petrosian_plot_cog(
    p,
    plot_r=True,
    title="Curve of Growth",
    radius_unit="pix",
    flux_unit="",
    ax=None,
    color="tab:blue",
    err_alpha=0.2,
    err_capsize=3,
    show_legend=True,
    legend_fontsize=None,
    ax_fontsize=None,
    tick_fontsize=None,
):
    """
    Plots the Curve of Growth (COG) for the Petrosian profile.

    Parameters
    ----------
    plot_r : bool, optional
        If True, plots radii of interest including Petrosian radius.
        Default is True.

    title : str, optional
        Title for the plot. Default is 'Curve of Growth'.

    radius_unit : str, optional
        Unit for the radius. Default is 'pix'.

    flux_unit  : str, optional
        Unit for the cumulative flux. Default is ''

    ax : matplotlib.axis, optional
        Matplotlib axis object to plot on. If None, creates a new axis.

    color : string
        Matplotlib color for profile.

    err_alpha : float, optional
        Transparency for the error region. Default is 0.2.

    err_capsize : int, optional
        Cap size for the error bars. Default is 3.

    show_legend : bool, optional
        If True, displays the legend. Default is True.

    legend_fontsize : int or float, optional
        Font size for the legend. If None, uses default font size.

    ax_fontsize : int or float, optional
        Font size for the axis labels. If None, uses default font size.

    tick_fontsize : int or float, optional
        Font size for the tick labels. If None, uses default font size.

    Returns
    -------
    ax : matplotlib.axis
        Matplotlib axis object with the plot.
    """

    with package_plot_style():
        radius_unit = "" if radius_unit is None else str(radius_unit)
        if ax is None:
            fig, ax = subplots(1, 1)

        ax.errorbar(
            p.r_list,
            p.flux_list,
            yerr=p.flux_err,
            marker="o",
            capsize=err_capsize,
            c=color,
            label="Data",
        )

        if err_alpha is not None and p.has_petrosian_err and err_alpha > 0:
            ax.fill_between(
                p.r_list,
                p.flux_list - p.flux_err,
                p.flux_list + p.flux_err,
                alpha=err_alpha,
                color=color,
            )

        if plot_r:
            r_half_light = p.r_half_light
            half_flux = p.half_flux
            if not np.isnan(r_half_light) and not np.isnan(half_flux):
                ax.axvline(
                    r_half_light,
                    linestyle="--",
                    c="black",
                    alpha=p._r_plot_alpha,
                    label="$R_{{50}}(L_{{50}}) = {:0.4f}$ {}".format(
                        r_half_light, radius_unit
                    ),
                )
                ax.axhline(
                    half_flux, linestyle="--", c="black", alpha=p._r_plot_alpha
                )
                ax.scatter(
                    r_half_light, half_flux, zorder=6, marker="o", color="tab:orange"
                )

            r_total_flux = p.r_total_flux
            total_flux = p.total_flux
            if not np.isnan(r_total_flux) and not np.isnan(total_flux):
                total_flux_fraction = int(p.total_flux_fraction * 100)
                ax.axvline(
                    r_total_flux,
                    linestyle="-",
                    c="black",
                    alpha=p._r_plot_alpha,
                    label="$R_{{total}}(L_{{{}}}) = {:0.4f}$ {}".format(
                        total_flux_fraction, r_total_flux, radius_unit
                    ),
                )
                ax.axhline(
                    total_flux, linestyle="-", c="black", alpha=p._r_plot_alpha
                )
                ax.scatter(
                    r_total_flux, total_flux, zorder=6, marker="o", color="tab:orange"
                )

        ax.set_title(title, fontsize=ax_fontsize)
        ax.set_xlabel(
            "Aperture Radius" + " [{}]".format(radius_unit) if radius_unit else "",
            fontsize=ax_fontsize,
        )
        ax.set_ylabel(
            r"$L(\leq r)$" + (" [{}]".format(flux_unit) if flux_unit else ""),
            fontsize=ax_fontsize,
        )

        mpl_tick_frame(ax=ax, minorticks=True, tick_fontsize=tick_fontsize)

        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        if show_legend:
            ax.legend(fontsize=legend_fontsize)

        return ax

def _petrosian_imshow(p, position=(0, 0), elong=1.0, theta=0.0, color=None, lw=None, show_legend=True, show_arrow=False, min_arrow_radius=10.0):
    """
    Make 2D plots of elliptical apertures with radii  of `r_half_light`, `r_total_flux`, `r_20` and `r_80`.

    Parameters
    ----------
    position : tuple
        (x, y) center of the apertures.

    elong : float
        Elongation of the aperture.

    theta : float
        The orientation of the aperture in rad.

    color : str
        Color override that will change the color of the apertures in the plot.

    lw : float
        Line width (thickness) of the plotted apertures.
    """
    with package_plot_style():
        ax = plt.gca()

        labels = ["$r_{20}$", "$r_{50}$", "$r_{80}$", f"$r_{{{p.total_flux_fraction*100:0.0f}}}$"]
        radii = [
            p._calculate_fraction_to_r(0.2)[0],
            p.r_half_light,
            p._calculate_fraction_to_r(0.8)[0],
            p.r_total_flux,
        ]
        linestyles = ["dotted", "dashed", "dashdot", "solid"]
        max_r = 0
        for label, r, ls in zip(labels, radii, linestyles):
            if not np.isnan(r) and r > 0:
                a, b = r, r / elong
                EllipticalAperture(position, a, b, theta=theta).plot(
                    label=label,
                    linestyle=ls,
                    color=color if color else 'k',
                    lw=lw,
                )
                max_r = r if r > max_r else max_r

        if show_arrow and max_r > min_arrow_radius:
            r_arrow(position, max_r, theta, color=color if color else 'k', lw=lw if lw is None else lw/2)

        if show_legend:
            ax.legend()

        ax.scatter(*position, marker="+", color=color if color else "k")


def  _petrosian_plot(
    p,
    plot_r=True,
    title="Petrosian Profile",
    radius_unit="pix",
    ax=None,
    color="tab:blue",
    err_alpha=0.2,
    err_capsize=3,
    show_legend=True,
    legend_fontsize=None,
    ax_fontsize=None,
    tick_fontsize=None,
):
    """
    Plots the Petrosian profile.

    Parameters
    ----------
    plot_r : bool, optional
        If True, plots the total flux and half-light radii. Default is True.

    title : str, optional
        Title for the plot. Default is 'Petrosian Profile'.

    radius_unit : str, optional
        Unit for the radius. Default is 'pix'.

    ax : matplotlib.axis, optional
        Matplotlib axis object to plot on. If None, creates a new axis.

    color : string
        Matplotlib color for profile.

    err_alpha : float, optional
        Transparency for the error region. Default is 0.2.

    err_capsize : int, optional
        Cap size for the error bars. Default is 3.

    show_legend : bool, optional
        If True, displays the legend. Default is True.

    legend_fontsize : int or float, optional
        Font size for the legend. If None, uses default font size.

    ax_fontsize : int or float, optional
        Font size for the axis labels. If None, uses default font size.

    tick_fontsize : int or float, optional
        Font size for the tick labels. If None, uses default font size.

    Returns
    -------
    ax : matplotlib.axis
        Matplotlib axis object with the plot.
    """

    with package_plot_style():
        radius_unit = "" if radius_unit is None else str(radius_unit)
        if ax is None:
            fig, ax = subplots(1, 1)

        ax.errorbar(
            p.r_list,
            p.petrosian_list,
            yerr=p.petrosian_err,
            marker="o",
            capsize=err_capsize,
            label="Data",
            color=color,
        )

        if err_alpha is not None and p.has_petrosian_err and err_alpha > 0:
            ax.fill_between(
                p.r_list,
                p.petrosian_list - p.petrosian_err,
                p.petrosian_list + p.petrosian_err,
                alpha=err_alpha,
                color=color,
            )

        r_petrosian = p.r_petrosian
        r_petrosian_err = p.r_petrosian_err

        if plot_r:
            r_color = "black"
            ax.axhline(
                p.eta, linestyle="--", color=r_color, alpha=p._r_plot_alpha
            )
            if not np.isnan(r_petrosian):
                ax.axvline(
                    r_petrosian,
                    linestyle="--",
                    color=r_color,
                    alpha=p._r_plot_alpha,
                    label=r"$R_{{p}}(\eta_{{{}}})={:0.4f}$ {}".format(
                        p.eta, r_petrosian, radius_unit
                    ),
                )
                if not np.isnan(r_petrosian_err):
                    ax.errorbar(
                        r_petrosian,
                        p.eta,
                        xerr=r_petrosian_err,
                        zorder=6,
                        marker="o",
                        capsize=5,
                        lw=3,
                        color="tab:orange",
                    )
                else:
                    ax.scatter(
                        r_petrosian, p.eta, zorder=6, marker="o", color="tab:orange"
                    )

        ax.axhline(0, c="black")
        ax.set_title(title, fontsize=ax_fontsize)
        ax.set_xlabel(
            "Aperture Radius" + " [{}]".format(radius_unit) if radius_unit else "",
            fontsize=ax_fontsize,
        )
        ax.set_ylabel(r"Petrosian Index $\eta(r)$", fontsize=ax_fontsize)

        mpl_tick_frame(ax=ax, minorticks=True, tick_fontsize=tick_fontsize)

        ax.set_xlim(0, None)
        if show_legend:
            ax.legend(fontsize=legend_fontsize)

        return ax