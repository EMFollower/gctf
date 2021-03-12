import math
import turtle
import os

from scipy import stats

from plot_tools import *


class CTFModel:
    # Initialization
    def __init__(self, **kwargs):
        self.do_ctf_corr = False
        self.Csmm = None
        self.highkV = None
        self.amp_percentage = 0.1
        self.pixel_size_A = None
        self.B_factor = 300
        self.df_search_step = 100
        self.defocusU = 150000
        self.defocusV = 100
        self.bin_size = 3
        self.box_size = 150
        self.tile_size = 512
        for param, value in kwargs.items():
            setattr(self, param, value)

    # Calculate power spectra denoted as |Fs|
    def calculateAveFFT(self, image):

        tile_size = self.tile_size
        nx = np.shape(image)[1]
        ny = np.shape(image)[0]

        if (ny % tile_size) == 0:
            nbox_x = int(ny / tile_size)
            nbox_y = int(nx / tile_size)
        else:
            nbox_x = int(ny / tile_size) + 1
            nbox_y = int(nx / tile_size) + 1

        print("The total number of boxes is: {}".format(nbox_x * nbox_y))

        fft_ave = np.zeros([tile_size, tile_size])

        index_i = np.arange(nbox_x)
        index_j = np.arange(nbox_y)

        [i, j] = np.meshgrid(index_i, index_j)
        # Set box_left
        box_left = i * tile_size
        box_left[np.equal(i, nbox_x - 1)] = nx - tile_size
        # Set box_right
        box_right = (i + 1) * tile_size
        box_right[np.equal(i, nbox_x - 1)] = nx
        # Set box_top
        box_top = j * tile_size
        box_top[np.equal(j, nbox_y - 1)] = ny - tile_size
        # Set box_bottom
        box_bottom = (j + 1) * tile_size
        box_bottom[np.equal(j, nbox_y - 1)] = ny

        for i in index_i:
            for j in index_j:
                box_image = image[box_left[i, j]: box_right[i, j], box_top[i, j]: box_bottom[i, j]]

                fft_box = np.fft.fft2(box_image)
                fft_box = abs(np.fft.fftshift(fft_box))
                fft_ave = fft_ave + fft_box

                print('i = {}, j = {}'.format(i, j))
                params = {'title': '2D accumulated FFT, i = {}, j = {}'.format(i, j),
                          'xlabel': 'x_direction',
                          'ylabel': 'y_direction',
                          'vmin': 0,
                          'vmax': 10000000}
                # plot_tools.plot_2d(fft_ave, params)
        # Calculate power spectra
        fft_power = np.square(fft_ave / (nbox_x * nbox_y))
        # np.savetxt('fft_ave_Fs.txt', fft_ave)
        print('Power spectra Fs: ')
        params = {'title': '2D averaged FFT',
                  'xlabel': 'x_direction',
                  'ylabel': 'y_direction',
                  'vmin': 0,
                  'vmax': 2e11}
        plot_tools.plot_2d(fft_power, params)

        fft = {'fft_amp': fft_ave,
               'fft_power': fft_power}

        return fft

    @staticmethod
    def calculateLAS(Fs):
        LAS = np.log(Fs.copy())
        np.savetxt('LAS.txt', LAS)
        print('LAS: ')
        params = {'title': 'Logarithmic Power Spectra (LPS)',
                  'xlabel': 'x_direction',
                  'ylabel': 'y_direction',
                  'vmin': 0,
                  'vmax': 40}
        # plot_tools.plot_2d(LAS, params)
        # plot_tools.plot_LAS_3d(LAS)

        return LAS

    def calculateBg(self, LAS):
        box_size = self.box_size
        # Box convolution
        tile_size = np.shape(LAS)
        nx = tile_size[1]
        ny = tile_size[0]
        bg = np.zeros([nx, ny])

        half_box = int(box_size / 2)

        index_i = np.arange(nx)
        index_j = np.arange(ny)
        [i, j] = np.meshgrid(index_i, index_j)
        # Set left bound
        left_bound = i - half_box
        left_bound[np.less(left_bound, 0)] = 0
        # Set right bound
        right_bound = i + half_box + 1
        right_bound[np.greater(right_bound, nx)] = nx
        # Set top bound
        top_bound = j - half_box
        top_bound[np.less(top_bound, 0)] = 0
        # Set bottom bound
        bottom_bound = j + half_box + 1
        bottom_bound[np.greater(bottom_bound, ny)] = ny

        for i in index_i:
            for j in index_j:
                bg[i, j] = np.mean(LAS[left_bound[i, j]: right_bound[i, j], top_bound[i, j]: bottom_bound[i, j]])

        print('bg: ')
        params = {'title': 'Background',
                  'xlabel': 'x_direction',
                  'ylabel': 'y_direction',
                  'vmin': 0,
                  'vmax': 20}
        # plot_tools.plot_2d(bg, params)
        # plot_tools.plot_bg_3d(bg)

        return bg

    @staticmethod
    def bgSubtraction(LAS, bg):
        LAS_no_bg = LAS - bg

        print('LAS_no_bg: ')
        params = {'title': 'Background subtracted LAS',
                  'xlabel': 'x_direction',
                  'ylabel': 'y_direction',
                  'vmin': 0,
                  'vmax': 0.8}
        plot_tools.plot_2d(LAS_no_bg, params)
        # plot_tools.plot_LAS_no_bg_3d(LAS_no_bg)
        # center_x = int(np.shape(LAS_no_bg)[1] / 2)
        # center_y = int(np.shape(LAS_no_bg)[0] / 2)
        # plot_tools.plot_LAS_no_bg_1d(LAS_no_bg[center_y, center_x:], params)
        # np.savetxt('LAS_no_bg.txt', LAS_no_bg)

        return LAS_no_bg

    def circAve(self, LAS_no_bg):
        bin_size = self.bin_size
        # Construct the meshgird
        x = np.shape(LAS_no_bg)[1]
        y = np.shape(LAS_no_bg)[0]

        center_x = int(x / 2)
        center_y = int(y / 2)

        [X, Y] = np.meshgrid(np.arange(x) - center_x, np.arange(y) - center_y)
        R = np.sqrt(np.square(X) + np.square(Y))

        # Initialize variables
        radius = np.arange(1, np.max(R), 1)
        intensity = np.zeros(len(radius))
        index = 0
        circ_ave = LAS_no_bg.copy()
        # Construct the mask
        for i in radius:
            mask = np.greater(R, i - bin_size) & np.less(R, i + bin_size)
            # print(mask)
            masked_val = LAS_no_bg[mask]
            intensity[index] = np.mean(masked_val)
            circ_ave[mask] = intensity[index]
            index += 1

        np.savetxt('LAS_circ_ave.txt', circ_ave)
        # Plot 1D profile
        print('circAve: ')
        # print('intensity: {}'.format(np.shape(intensity)))
        data = {'resolution': radius[0: center_x],
                'intensity': intensity[0: center_x]}
        params = {'title': 'Circularly averaged CTF',
                  'xlabel': 'Resolution (1 / Å)',
                  'ylabel': 'Amplitude (a.u.)',
                  'vmin': -1,
                  'vmax': 1}
        # plot_tools.plot_circAve_1d(data, params)

        # Plot 2D averaged amplitude spectra
        params = {'title': 'Cicular Averaged LAS',
                  'xlabel': 'x_direction',
                  'ylabel': 'y_direction',
                  'vmin': -1,
                  'vmax': 1}
        # plot_tools.plot_2d(circ_ave, params)

        # Plot 3D averaged amplitude spectra
        # plot_tools.plot_circAve_3d(LAS_circ_ave)

        # return intensity[0: center_x], LAS_circ_ave
        return circ_ave

    def dfSearch(self):
        nx = np.arange(self.tile_size)
        ny = nx
        center_x = int(len(nx) / 2)
        center_y = int(len(ny) / 2)
        # Calculate the best defocus using 1D cross correlation
        defocus_series = np.arange(self.defocusV, self.defocusU, self.df_search_step)
        cccs = 0
        freq = self.freqSpace()
        freq_x = freq['freq_x']
        s = freq['s']
        Kc = np.exp(-self.B_factor / 4 * s ** 2)
        # Initialize parameters
        best_df = 0
        for df in defocus_series:
            print('df = {}'.format(df))
            simctf = self.simctf(zu=None, zv=None, theta_a=None, df=df)
            simctf_power = simctf['simctf_power']
            # --------------------------------Plot image-------------------------------------
            params = {'title': 'Simulated CTF, defocus = {} Å'.format(df),
                      'xlabel': 'x_direction',
                      'ylabel': 'y_direction',
                      'vmin': -1,
                      'vmax': 1}
            # plot_tools.plot_2d(simctf, params)
            # Plot 1d profile
            simctf_1d_x = freq_x[center_y, center_x:].flatten()
            simctf_1d_y = simctf_power[center_y, center_x:].flatten()
            Kc_1d = Kc[center_y, center_x:].flatten()
            data = {'resolution': simctf_1d_x,
                    'simctf': simctf_1d_y,
                    'Kc': Kc_1d}
            params = {'title': 'Simulated CTF',
                      'xlabel': 'Resolution (1 / Å)',
                      'ylabel': 'Amplitude (a.u.)',
                      'vmin': -1,
                      'vmax': 1}

            # plot_tools.plot_simctf_1d(data, params)
            # -------------------------------------------------------------------------------------
            # Calculate cross-correlation coefficients
            expctf = estimatedCTF['LAS_circ_ave'][center_y, center_x:].flatten()
            cccs_new = self.crossCorrelation1d(simctf_1d_x, simctf_1d_y, expctf, df)
            # print('Cross-correlation coefficient is {} when defocus is {} Å.'.format(cccs_new[0], df))
            if np.abs(cccs_new[0]) > np.abs(cccs):
                cccs = cccs_new[0]
                best_df = df

        print('The best estimated defocus is: {}'.format(best_df))
        print('Corresponding 1D cross-correlation coefficient is: {}'.format(cccs))

        return best_df

    def gridSearch(self, best_df):
        nx = np.arange(self.tile_size)
        ny = nx
        center_x = int(len(nx) / 2)
        center_y = int(len(ny) / 2)
        freq = self.freqSpace()
        freq_x = freq['freq_x']
        simctf_1d_x = freq_x[center_y, center_x:].flatten()
        # Calculate 2D cross correlation
        cccs2d_best = -1
        index = 0
        zu_series = np.arange(best_df - 10000, best_df + 10000, 1000)
        zv_series = np.arange(best_df - 10000, best_df + 10000, 1000)
        theta_a_series = np.arange(0, np.pi, 5 / 180 * np.pi)
        len_zu = len(zu_series)
        len_zv = len(zv_series)
        len_theta_a = len(theta_a_series)
        cccs2d = np.zeros(len_zu * len_zv * len_theta_a)
        freq = self.freqSpace()
        theta = freq['theta']
        # Initialize parameters
        best_zu = 0
        best_zv = 0
        best_theta_a = 0
        best_ctf_2d_power = None
        best_ctf_2d_amp = None

        print('Start 2D refinement...')
        print('The total number of iteration is: {}'.format(len_zu * len_zv * len_theta_a))
        for zu in zu_series:
            for zv in zv_series:
                for theta_a in theta_a_series:
                    if int(index % 500) == 0:
                        print('index = {}'.format(index))
                        print('zu = {}, zv = {}, theta_a = {}'.format(zu, zv, theta_a))
                    simctf = self.simctf(zu, zv, theta_a, df=None)
                    simctf_power = simctf['simctf_power']
                    simctf_amp = simctf['simctf_amp']
                    # plt.imshow(simctf, cmap='gray')
                    # plt.pause(5)
                    cccs2d[index] = self.crossCorrelation2d(simctf_power, estimatedCTF['LAS_no_bg'])
                    if cccs2d[index] > cccs2d_best:
                        cccs2d_best = cccs2d[index]
                        best_zu = zu
                        best_zv = zv
                        best_theta_a = theta_a
                        best_ctf_2d_amp = simctf_amp
                        best_ctf_2d_power = simctf_power
                    index += 1

        cccs2d = np.array(cccs2d)
        # cccs2d = cccs2d.reshape(len(zu_series), len(zv_series), len(theta_a_series))
        # print(np.shape(cccs2d))
        print('Best zu: {}, best_zv: {}, best theta_a: {}, highest cccs2d is {}.'.format(best_zu, best_zv, best_theta_a,
                                                                                         cccs2d_best))

        # Plot best 2D CTF ==========================================================================================
        params = {'title': 'Best simulated 2D CTF(df = {}, zu = {}, zv = {}, {} = {})'.format(best_df, best_zu, best_zv,
                                                                                              r'$\theta_a$',
                                                                                              best_theta_a),
                  'xlabel': 'X_direction',
                  'ylabel': 'Y_direction',
                  'vmax': np.max(best_ctf_2d_power),
                  'vmin': np.min(best_ctf_2d_power)}
        plot_tools.plot_2d(best_ctf_2d_power, params)

        # Plot cccs2d
        params = {'title': 'cccs2d',
                  'xlabel': 'dz',
                  'ylabel': r'$\theta_z$',
                  'zlabel': 'cccs2d'}
        # plot_tools.plot_cccs_3d(zu_series, theta_a_series, cccs2d, params)

        # Plot best 1d curve
        data = {'best_ctf_2d': best_ctf_2d_power[center_y, center_x:].flatten(),
                'simCTF_1d_x': simctf_1d_x,
                'estimatedctf': estimatedCTF['LAS_no_bg'][center_y, center_x:].flatten()}
        params = {'title': '1D curves of the best estimatedCTF',
                  'xlabel': 'Resolution',
                  'ylabel': 'Power spectra'}
        # plot_tools.plot_best_cccs_1d(data, params)
        # ==========================================================================================================
        result = {'cccs2d': cccs2d,
                  'best_ctf_2d_amp': best_ctf_2d_amp,
                  'best_ctf_2d_power': best_ctf_2d_power}

        return result

    def freqSpace(self):
        if self.do_ctf_corr:
            [ny, nx] = np.shape(estimatedCTF['image'])
            nx = np.arange(nx)
            ny = np.arange(ny)
            center_x = int(len(nx) / 2)
            center_y = int(len(ny) / 2)
            nx = (nx - center_x) / len(nx)
            ny = (ny - center_y) / len(ny)
        else:
            nx = np.arange(self.tile_size)
            ny = np.arange(self.tile_size)
            center_x = int(len(nx) / 2)
            center_y = int(len(ny) / 2)
            nx = (nx - center_x) / self.tile_size
            ny = (ny - center_y) / self.tile_size

        theoretical_resolution = 1 / (2 * self.pixel_size_A)

        freq_x = nx * theoretical_resolution
        freq_y = ny * theoretical_resolution

        [freq_x, freq_y] = np.meshgrid(freq_x, freq_y)
        # Calculate frequency
        s = np.sqrt(freq_x ** 2 + freq_y ** 2)
        # Calculate azimuthal angle
        theta = np.arctan2(freq_y, freq_x)
        theta[theta < 0] = theta[theta < 0] + np.pi * 2

        params = {'title': r'$\theta$',
                  'xlabel': 'X_direction',
                  'ylabel': 'Y_direction',
                  'vmax': 360,
                  'vmin': 0}
        # plot_tools.plot_2d(theta / np.pi * 180, params)

        result = {'freq_x': freq_x, 'freq_y': freq_y, 's': s, 'theta': theta}

        return result

    def simctf(self, zu, zv, theta_a, df):
        # The unit of length is angstrom
        # Set parameters
        amp_percentage = self.amp_percentage
        B_factor = self.B_factor
        highV = self.highkV * 1e3
        Cs = self.Csmm * 1e7
        wavelength = 12.2643247 / math.sqrt(highV + highV ** 2 * 0.978466e-6)
        freq = self.freqSpace()
        s = freq['s']
        theta = freq['theta']
        Kc = np.exp(-B_factor / 4 * s ** 2)

        if df is None:
            z = zu * np.square(np.cos(theta - theta_a)) + zv * np.square(np.sin(theta - theta_a))
        else:
            z = df

        gamma_2d = -0.5 * np.pi * Cs * wavelength ** 3 * s ** 4 + np.pi * wavelength * z * s ** 2
        simctf_amp = (-np.sqrt(1 - amp_percentage ** 2) * np.sin(gamma_2d) - amp_percentage * np.cos(gamma_2d)) * Kc
        simctf_power = np.square(simctf_amp)
        simctf = {'simctf_amp': simctf_amp,
                  'simctf_power': simctf_power}

        return simctf

    def bayesianOpt(self, best_df, expctf):
        from hyperopt import hp, fmin, tpe, space_eval
        # from hyperopt.pyll.stochastic import sample

        # Start Bayesian Optimization
        def objective(args):
            zu = args['zu']
            zv = args['zv']
            theta_a = args['theta_a']
            simctf = self.simctf(zu, zv, theta_a, df=None)
            simctf_power = simctf['simctf_power']
            cccs2d = self.crossCorrelation2d(simctf_power, expctf)
            cccs2d = -cccs2d

            return cccs2d

        space = {'zu': hp.uniform('zu', best_df - 10000, best_df + 10000),
                 'zv': hp.uniform('zv', best_df - 10000, best_df + 10000),
                 'theta_a': hp.uniform('theta_a', 0, np.pi)}

        best = fmin(objective, space, algo=tpe.suggest, max_evals=1000)
        print(space_eval(space, best))

        best_simctf = self.simctf(best['zu'], best['zv'], best['theta_a'], df=None)
        best['best_simctf_amp'] = best_simctf['simctf_amp']
        best['best_simctf_power'] = best_simctf['simctf_power']
        params = {'xlabel': 'x_direction',
                  'ylabel': 'y_direction',
                  'title': 'Final simCTF',
                  'vmin': 0,
                  'vmax': 0.8}
        plot_tools.plot_2d(best['best_simctf_power'], params)

        return best

    @staticmethod
    def crossCorrelation1d(simCTF_1d_x, simctf, expctf, df):
        downscale = 1
        psimctf = simctf.copy() * downscale
        pexpctf = expctf.copy()
        plot_range = int(len(simctf) / 2)
        cccs = stats.pearsonr(psimctf, pexpctf)
        data = {'resolution': simCTF_1d_x[10:plot_range],
                'simctf': psimctf[10:plot_range],
                'expctf': pexpctf[10:plot_range]}
        params = {'title': 'Cross Correlation between simulated (df = {} Å) and circularly averaged CTF'.format(df),
                  'xlabel': 'Resolution (1 / Å)',
                  'ylabel': 'Amplitude (a.u.)',
                  'vmin': -0.1,
                  'vmax': np.max(psimctf) * 1.5,
                  'cccs': cccs}
        # plot_tools.plot_cc_1d(data, params)

        return cccs

    @staticmethod
    def crossCorrelation2d(simctf, expctf):
        simctf_mean = np.mean(simctf)
        expctf_mean = np.mean(expctf)
        cov = np.sum((simctf - simctf_mean) * (expctf - expctf_mean))
        simctf_sigma = np.sqrt(np.sum((simctf - simctf_mean) ** 2))
        expctf_sigma = np.sqrt(np.sum((expctf - expctf_mean) ** 2))

        cccs2d = cov / (simctf_sigma * expctf_sigma)

        return cccs2d

    def loadImage(self, image_path):
        image_type = os.path.splitext(image_path)[-1][1:]
        if image_type == 'dm3':
            import dm3_lib as dm3
            dm3f = dm3.DM3(image_path)
            image_data = dm3f.imagedata
            params = {'title': 'Image',
                      'xlabel': 'x_direction',
                      'ylabel': 'y_direction',
                      'vmin': 0,
                      'vmax': np.max(image_data)}
            # plot_tools.plot_2d(image_data, params)
        elif image_type == 'mrc':
            import mrcfile
            with mrcfile.open(image_path) as mrc:
                image_data = mrc.data
            params = None
            plot_tools.plot_2d(image_data, params)

        return image_data

    def ctf_corr(self, best_ctf_2d_para):
        # Calculate theoretical ctf
        self.do_ctf_corr = True
        # self.B_factor = 1000
        zu = best_ctf_2d_para['zu']
        zv = best_ctf_2d_para['zv']
        theta_a = best_ctf_2d_para['theta_a']
        simctf = self.simctf(zu, zv, theta_a, df=None)['simctf_amp']
        # Calculate fft of original image
        image = estimatedCTF['image']
        image_fft_amp = np.fft.fft2(image)
        image_fft_amp = np.fft.fftshift(image_fft_amp)
        # Apply Weiner filter
        fft_corr = image_fft_amp * simctf / (np.square(simctf) + 1)

        fft_corr = np.fft.ifftshift(fft_corr)
        img_corr = np.fft.ifft2(fft_corr)
        plt.imshow(abs(img_corr), cmap='gray')
        plt.show()

    def estimateCTF(self, image_path):
        # tile_size if for FFT, box_size is for estimation of noise using box_convolution, bin_size is used for
        # circular averaging
        image = self.loadImage(image_path)

        fft = self.calculateAveFFT(image)
        fft_power = fft['fft_power']
        fft_amp = fft['fft_amp']

        LAS = self.calculateLAS(fft_power)
        bg = self.calculateBg(LAS)
        LAS_no_bg = self.bgSubtraction(LAS, bg)
        LAS_circ_ave = self.circAve(LAS_no_bg)

        # Compare Fs, LAS, bg and LAS_no_bg
        center = int(self.tile_size / 2)
        data = {'fft_power': fft_power[center, center:],
                'fft_amp': fft_amp[center, center:],
                'LAS': LAS[center, center:],
                'bg': bg[center, center:],
                'LAS_no_bg': LAS_no_bg[center, center:],
                'LAS_circ_ave': LAS_circ_ave[center, center:]}
        params = {'title': 'Comparison among different parameters',
                  'xlabel': 'pixel',
                  'ylabel': 'power spectra',
                  'vmin': -1,
                  'vmax': 1}
        plot_tools.plot_comparison_1d(data, params)

        result = {'image': image,
                  'fft_power': fft_power,
                  'fft_amp': fft_amp,
                  'LAS': 'LAS',
                  'bg': bg,
                  'LAS_no_bg': LAS_no_bg,
                  'LAS_circ_ave': LAS_circ_ave}

        return result


if __name__ == '__main__':
    ctf = CTFModel(Csmm=1e-3, defocusU=150000, defocusV=1000, df_search_step=100, highkV=300, amp_percentage=0.1,
                   pixel_size_A=1.084, B_factor=500, tile_size=512, box_size=130, bin_size=3, do_ctf_corr=False)
    # estimateCTF(self, mic_path, tile_size, box_size, bin_size)
    estimatedCTF = ctf.estimateCTF('Ribosome_4096_4096_32_1_34_A_2_7mm_300kV_10028.mrc')
    best_df_1d = ctf.dfSearch()
    # best_ctf_2d = ctf.gridSearch(best_df_1d)
    best_ctf_2d = ctf.bayesianOpt(best_df_1d, estimatedCTF['LAS_no_bg'])
    ctf.ctf_corr(best_ctf_2d)
    # turtle.done()
