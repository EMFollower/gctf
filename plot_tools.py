import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class plot_tools():
    # Plot 2D images
    def plot_2d(data, params):
        plt.figure(figsize=(5, 5))
        if params is not None:
            plt.imshow(data, cmap='gray', vmin=params['vmin'], vmax=params['vmax'])
            plt.xlabel(params['xlabel'])
            plt.ylabel(params['ylabel'])
            plt.title(params['title'])
        else:
            plt.imshow(data, cmap='gray')
        plt.pause(5)

    # Plot logarithmic power spectra
    def plot_LAS_3d(data):
        X = range(np.shape(data)[0])
        Y = X
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, data, cmap='rainbow')
        plt.pause(5)

    # Plot background
    def plot_bg_3d(data):
        X = range(np.shape(data)[0])
        Y = X
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, data, cmap='rainbow')
        plt.pause(5)

    # Plot logarithmic power spectra without noise
    def plot_LAS_no_bg_3d(data):
        X = range(np.shape(data)[0])
        Y = X
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, data, cmap='rainbow')
        plt.pause(5)

    def plot_LAS_no_bg_1d(data, params):
        LAS_no_bg_1d = data
        plt.plot(LAS_no_bg_1d, label='LAS_no_bg')
        plt.title(params['title'])
        plt.pause(5)

    # Plot circular averaged power spectra
    def plot_circAve_1d(data, params):
        plt.figure(figsize=(10, 5))
        resolution = data['resolution']
        amplitude = data['intensity']
        plt.plot(resolution, amplitude, label='Circularly Averaged Amplitude Spectra')
        plt.legend()
        plt.xlabel(params['xlabel'])
        plt.ylabel(params['ylabel'])
        plt.title(params['title'])
        plt.pause(5)

    def plot_circAve_3d(data):
        X = range(np.shape(data)[0])
        Y = X
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, data, cmap='rainbow')
        plt.pause(5)

    # Plot simulated ctf
    def plot_simctf_1d(data, params):
        plt.figure(figsize=(10, 5))
        resolution = data['resolution']

        simctf = data['simctf']
        plt.plot(resolution, simctf, label='simctf')

        Kc = data['Kc']
        plt.plot(resolution, Kc, label='Kc')

        plt.legend()

        plt.xlabel(params['xlabel'])
        plt.ylabel(params['ylabel'])
        plt.title(params['title'])

        plt.pause(5)

    # Plot cross correlation
    def plot_cc_1d(data, params):
        plt.figure(figsize=(10, 5))
        resolution = data['resolution']

        simctf = data['simctf']
        plt.plot(resolution, simctf, label='simctf')

        expctf = data['expctf']
        plt.plot(resolution, expctf, label='expctf')

        plt.legend()
        plt.text(np.max(resolution) * 0.7, 0.35, 'cccs = {}'.format(params['cccs'][0]))

        plt.xlabel(params['xlabel'])
        plt.ylabel(params['ylabel'])
        plt.title(params['title'])
        #plt.ylim((params['vmin'], params['vmax']))

        plt.pause(5)

    # Comparison
    def plot_comparison_1d(data, params):
        fft_power = data['fft_power']
        fft_amp = data['fft_amp']
        LAS = data['LAS']
        bg = data['bg']
        LAS_no_bg = data['LAS_no_bg']
        LAS_circ_ave = data['LAS_circ_ave']

        # plt.plot(fft_power, label='fft_power')
        plt.plot(LAS, label='LAS')
        plt.plot(bg, label='bg')
        # plt.plot(LAS_circ_ave, label='LAS_circ_ave')
        # plt.plot(LAS_no_bg, label='LAS_no_bg')

        plt.legend()

        plt.xlabel(params['xlabel'])
        plt.ylabel(params['ylabel'])
        plt.title(params['title'])
        # plt.ylim((params['vmin'], params['vmax']))

        plt.pause(5)

    # Plot cccs2d
    def plot_cccs_3d(dz_series, theta_a_series, cccs2d, params):
        [dz, theta_a] = np.meshgrid(dz_series, theta_a_series)
        cccs2d = cccs2d.T
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(dz, theta_a, cccs2d, cmap='rainbow')

        plt.title(params['title'])
        plt.xlabel(params['xlabel'])
        plt.ylabel(params['ylabel'])

        plt.pause(5)

    # Plot best_cccs_1d
    def plot_best_cccs_1d(data, params):
        simCTF_1d_x = data['simCTF_1d_x']
        best_ctf_2d = data['best_ctf_2d']
        estimatedctf = data['estimatedctf']

        plt.plot(simCTF_1d_x, best_ctf_2d, label='best_ctf_2d')
        plt.plot(simCTF_1d_x, estimatedctf, label='estimatedctf')
        plt.legend()

        plt.title(params['title'])
        plt.xlabel(params['xlabel'])
        plt.ylabel(params['ylabel'])

        plt.pause(5)
