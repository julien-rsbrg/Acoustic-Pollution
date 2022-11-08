# -*- coding: utf-8 -*-


# Python packages
import _env
import matplotlib.pyplot
import numpy
import os

import yaml
from yaml.loader import SafeLoader

# MRG packages

with open('config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)

dst_folder = config["DST_FOLDER"]
dst_folder = os.path.join(dst_folder, "exp_"+str(config["EXPERIMENT_ID"]))
if not (os.path.exists(dst_folder)):
    os.mkdir(dst_folder)


def myimshow(tab, **kwargs):
    """Customized plot."""

    if 'dpi' in kwargs and kwargs['dpi']:
        dpi = kwargs['dpi']
    else:
        dpi = 100

    # -- create figure
    fig = matplotlib.pyplot.figure(dpi=dpi)
    ax = matplotlib.pyplot.axes()

    if 'title' in kwargs and kwargs['title']:
        title = kwargs['title']
    if 'cmap' in kwargs and kwargs['cmap']:
        cmap = kwargs['cmap']
    else:
        cmap = 'jet'
    # if 'clim' in kwargs and kwargs['clim']:
    #    clim = kwargs['clim']
    if 'vmin' in kwargs and kwargs['vmin']:
        vmin = kwargs['vmin']
    if 'vmax' in kwargs and kwargs['vmax']:
        vmax = kwargs['vmax']

    # -- plot curves
    if 'cmap' in kwargs and kwargs['cmap']:
        matplotlib.pyplot.imshow(tab, cmap=cmap)
    else:
        matplotlib.pyplot.imshow(tab, cmap=cmap)
    if 'title' in kwargs and kwargs['title']:
        matplotlib.pyplot.title(title)
    else:
        matplotlib.pyplot.imshow(tab, cmap=cmap)
    if 'colorbar' in kwargs and kwargs['colorbar']:
        matplotlib.pyplot.colorbar()

#    if 'clim' in kwargs and kwargs['clim']:
#        matplotlib.pyplot.clim(clim)
    if 'vmin' in kwargs and kwargs['vmin']:
        matplotlib.pyplot.clim(vmin, vmax)

    if 'filename' in kwargs and kwargs['filename']:
        output_file = kwargs['filename']
        (root, ext) = os.path.splitext(output_file)
        dst_file_path = os.path.join(dst_folder, root + '_plot' + ext)
        matplotlib.pyplot.savefig(dst_file_path, format=ext[1:])
        matplotlib.pyplot.close()
    else:
        matplotlib.pyplot.show()
        matplotlib.pyplot.close()

    matplotlib.pyplot.close(fig)

    return


def _plot_uncontroled_solution(u, chi):
    # def _plot_uncontroled_solution(x_plot, y_plot, x, y, u, chi):
    myimshow(numpy.real(u), title='$\operatorname{Re}(u_{0})$ in $\Omega$',
             colorbar='colorbar', cmap='jet', vmin=-1, vmax=1, filename='fig_u0_re.jpg')
    myimshow(numpy.imag(u), title='$\operatorname{Im}(u_{0})$ in $\Omega$',
             colorbar='colorbar', cmap='jet', vmin=-1, vmax=1, filename='fig_u0_im.jpg')
    myimshow(chi, title='$\chi_{0}$ in $\Omega$', colorbar='colorbar',
             cmap='jet', vmin=-1, vmax=1, filename='fig_chi0_re.jpg')
    # k_begin = 0
    # k_end = len(x) - 1
    # for k in range(k_begin, k_end):
    #     x_plot[k] = k
    #     y_plot[k] = chi[int(y[k]), int(x[k])]
    # matplotlib.pyplot.plot(x_plot, y_plot)
    # matplotlib.pyplot.title('$\chi_{0}$ in $\Omega$')
    # matplotlib.pyplot.show()

    return


def _plot_controled_solution(u, chi):
    # def _plot_controled_solution(x_plot, y_plot, x, y, u, chi):

    myimshow(numpy.real(u), title='$\operatorname{Re}(u_{n})$ in $\Omega$',
             colorbar='colorbar', cmap='jet', vmin=-1, vmax=1, filename='fig_un_re.jpg')
    myimshow(numpy.imag(u), title='$\operatorname{Im}(u_{n})$ in $\Omega$',
             colorbar='colorbar', cmap='jet', vmin=-1, vmax=1, filename='fig_un_im.jpg')
    myimshow(chi, title='$\chi_{n}$ in $\Omega$', colorbar='colorbar',
             cmap='jet', vmin=-1, vmax=1, filename='fig_chin_re.jpg')
    # k_begin = 0
    # k_end = len(x) - 1
    # for k in range(k_begin, k_end):
    #     x_plot[k] = k
    #     y_plot[k] = chi[int(y[k]), int(x[k])]
    # matplotlib.pyplot.plot(x_plot, y_plot)
    # matplotlib.pyplot.title('$\chi_{n}$ in $\Omega$')
    # matplotlib.pyplot.show()

    return


def _plot_error(err):

    myimshow(numpy.real(err), title='$\operatorname{Re}(u_{n}-u_{0})$ in $\Omega$',
             colorbar='colorbar', cmap='jet', vmin=-1, vmax=1, filename='fig_err_real.jpg')
    myimshow(numpy.imag(err), title='$\operatorname{Im}(u_{n}-u_{0})$ in $\Omega$',
             colorbar='colorbar', cmap='jet', vmin=-1, vmax=1, filename='fig_err.jpg')

    return


def _plot_energy_history(energy):

    # , cmap = 'jet')#, vmin = 1e-4, vmax = 1e-0)
    matplotlib.pyplot.plot(numpy.arange(energy.shape[0]), energy[:, 0], 'bo-')
    matplotlib.pyplot.title('Energy')
    # matplotlib.pyplot.colorbar()
    # matplotlib.pyplot.show()
    filename = 'fig_energy_real.jpg'
    dst_file_path = os.path.join(dst_folder, filename)
    matplotlib.pyplot.savefig(dst_file_path)
    matplotlib.pyplot.close()

    return


def _plot_comparison(u0, un):
    figure, axis = matplotlib.pyplot.subplots(1, 4)

    axis[0].imshow(numpy.real(u0),
                   cmap='jet', vmin=-1, vmax=1)
    axis[0].set_title("$\operatorname{Re}(u_{0})$ in $\Omega$")
    axis[0].axis('off')

    axis[1].imshow(numpy.imag(u0),
                   cmap='jet', vmin=-1, vmax=1)
    axis[1].set_title("$\operatorname{Im}(u_{0})$ in $\Omega$")
    axis[1].axis('off')

    axis[2].imshow(numpy.real(un),
                   cmap='jet', vmin=-1, vmax=1)
    axis[2].set_title("$\operatorname{Re}(u_{n})$ in $\Omega$")
    axis[2].axis('off')

    axis[3].imshow(numpy.imag(un),
                   cmap='jet', vmin=-1, vmax=1)
    axis[3].set_title("$\operatorname{Im}(u_{n})$ in $\Omega$")
    axis[3].axis('off')

    file_name = "comparison_u_0_and_n.jpg"
    dst_file_path = os.path.join(dst_folder, file_name)
    matplotlib.pyplot.savefig(dst_file_path)
