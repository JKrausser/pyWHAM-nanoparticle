import os
import sys
import time
import warnings
import multiprocessing as mp
import freud as fr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import pandas as pd
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from pyquaternion import Quaternion
from scipy.spatial import Voronoi, voronoi_plot_2d
from tools import lmp_tools as lt
from numba import jit
import glob, re


##=================================================##
##=================================================##
# @jit(nopython=True)
def standardBinning(work_data, window_checkpoints, zStartIn, zEndIn, zIntIn, numRealsIn):
    """

    Bins data from different realisations and different umbrella windows into an array.

    :param file_name: colvars input file (.traj) with the z coordinates of the umbrella simulation
    :param zStartIn: smallest bin value
    :param zEndIn: largest bin value
    :param zIntIn: bin size
    :param numRealsIn: number of realisation which will be loaded
    :return: array of histograms (as many windows as have been specified in the simulation)

    """

    test_plot = False

    if zStartIn > np.min(work_data[:, 1]):
        print("WARNING: box range not set small enough!")
    if zEndIn < np.max(work_data[:, 1]):
        print("WARNING: box range not set large enough!")

    num_windows = len(window_checkpoints)

    zBinNumber = int(np.ceil((zEndIn - zStartIn) / zIntIn))
    zHistBuf = np.zeros(shape=(num_windows, zBinNumber, 3))  ## index 1: histogram index 2: bin frequency

    binShift = int(np.ceil(zStartIn / zIntIn))
    for k in range(num_windows):
        for i in range(zBinNumber):
            zHistBuf[k, i, 0] = (i + 0.5) * zIntIn + zStartIn

    for steps in range(num_windows):
        # data_from_window = np.array(work_data[work_data["x0_distZ"] == window_checkpoints[steps]])
        data_from_window = work_data[work_data[:, 3] == window_checkpoints[steps]] ## ToDo obtain this index directly, can't use pandas with numba (still fast enough for now)
        if steps != 0:
            print(f"There are {len(data_from_window)} sample in window {steps} at {window_checkpoints[steps]}.")
        for i in range(len(data_from_window)):
            x_val = data_from_window[i, 1] - zStartIn
            x_bin = int(x_val // zIntIn)
            if x_bin < 0:
                print(steps,i,x_val, x_bin)
                print(data_from_window[i, 1], zStartIn)
                print(zStartIn)
            assert x_bin >= 0
            if x_bin >= 0 and x_bin < zBinNumber:
                zHistBuf[steps, x_bin, 1] += 1 			## binning occurences of position (i.e. yields prob of finding particles at location)
                zHistBuf[steps, x_bin, 2] += data_from_window[i, 2] 		## averaging (binning) other quantity like degree of wrapping

        ## nornalise window in each step individually
        zHistBuf[steps, :, 1] = np.divide(zHistBuf[steps, :, 1], zIntIn * np.sum(zHistBuf[steps, :, 1]))

        # zHistBuf[steps, :, 1] = np.divide(zHistBuf[steps, :, 1], zHistBuf[steps, :, 0])
        # zHistBuf[steps, :, 1] = np.divide(zHistBuf[steps, :, 1], zIntIn * np.sum(zHistBuf[steps, :, 1]), out=np.zeros_like(zHistBuf[steps, :, 1]),where=zHistBuf[steps, :, 1] != 0)  ## mean

    ##ToDo unclarity about norm: stepwise or global?
    zHistBuf[:, :, 2] = np.divide(zHistBuf[:, :, 2], zHistBuf[:, :, 1], out=np.zeros_like(zHistBuf[:, :, 2]), where=zHistBuf[:, :,  1] != 0) ## mean

    # if test_plot:
    #     for steps in range(num_windows):
    #         plt.plot(zHistBuf[steps,:,0], zHistBuf[steps,:,1])
    #
    #     plt.savefig("test.png")
    #     # plt.hist(data_from_window[:,1], bins = 10, density = True)
    #     sys.exit()
    return zHistBuf


#===========================================#
#===========================================#
def array_to_bins_z(data_set, zStartIn, zEndIn, zIntIn):
    """
    Input data set should be z coords in position 0 and the binned quantity in position 1.

    Bin properties of the membrane according to the particles z coordinate (z profile).
    Here, Interval and number of bins is fixed instead of only a fixed number of bins.

    bin_index determines the quantity to be binned from the movie file

    :param data_set:
    :param box:
    :param num_bins:
    :return:
    """

    bin_size = zIntIn
    num_bins = int(np.ceil((zEndIn - zStartIn) / zIntIn))
    binned_grid = np.zeros(shape=(num_bins, 3)) ## last index is grid count


    for i in range(num_bins): ## set centres of bins
        binned_grid[i,0] = (i + 0.5)*bin_size + zStartIn


    for k in range(len(data_set)):
        x_val = data_set[k, 0] - zStartIn ## shift to the positive values
        y_val = data_set[k, 1]
        x_bin = int(x_val//bin_size) ## determine bin number to drop value in
        # print(x_bin, x_val)
        if x_bin >= 0 and x_bin < num_bins: ## bin only data in range, to avoid stray lipids
            binned_grid[x_bin, 1] += y_val
            binned_grid[x_bin,2] += 1

    binned_grid_normed = np.copy(binned_grid)
    binned_grid_normed[:, 1] = np.divide(binned_grid[:,  1], binned_grid[:,  2], out=np.zeros_like(binned_grid[:, 1]), where=binned_grid[:,  2] != 0) ## mean
    binned_grid_normed[:, 2] /= bin_size*len(data_set) ## normalise to 1


    return binned_grid_normed

#===========================================#
#===========================================#

##=================================================##
##=================================================##
@jit(nopython=True)
def whamRecursion(histWindows, zStartIn, zEndIn, zIntIn, recursionsMaxIn, epsConvergeIn, numRealsIn, springConst,
                  window_coords):
    print("Starting WHAM recursion")

    # for i in range(histWindows.shape[0]):
    # 	print(np.sum(histWindows[i,:,1]))
    # 	plt.plot(histWindows[i,:,0], histWindows[i,:,1])
    # plt.savefig("Plots/test_%i.png" % i)
    # plt.close()

    # plt.xlim([40, 50])

    zBinNumber = int(np.ceil((zEndIn - zStartIn) / zIntIn))
    nWindows = histWindows.shape[0]

    assert nWindows == len(window_coords)
    mainHist = np.zeros(shape=(zBinNumber, 2))
    mainHistBuf = np.zeros(shape=(zBinNumber, 2))
    mainHistOld = np.zeros(shape=(zBinNumber, 2))

    mainHist[:, 0] = histWindows[0, :, 0]
    mainHistBuf[:, 0] = histWindows[0, :, 0]
    mainHistOld[:, 0] = histWindows[0, :, 0]

    fWindows = np.zeros(shape=nWindows)  ## start values for the recursion
    fWindowsBuf = np.zeros(shape=nWindows)  ## start values for the recursion
    fWindowsOld = np.zeros(shape=nWindows)

    aWindows = np.zeros(shape=(nWindows, zBinNumber, 2))
    aWindowsSum = np.zeros(shape=(zBinNumber, 2))
    aWindowsSumBuf = np.zeros(shape=(zBinNumber, 2))

    for i in range(nWindows):
        for k in range(zBinNumber):
            aWindows[i, k, 0] = (k + 0.5) * zIntIn + zStartIn

    aWindowsSum[:, 0] = aWindows[0, :, 0]
    aWindowsSumBuf[:, 0] = aWindows[0, :, 0]

    recCount = 0
    while recCount < recursionsMaxIn:
        # print("recursion:", recCount)
        for i in range(nWindows):  ## compute a_i weights
            # springConst = metaFile[i, 2] ## used to be window dependent
            Xi0 = window_coords[i]

            histNorm = np.sum(zIntIn * histWindows[i, :, 1])
            # histNorm = np.sum(histWindows[i, :, 1])
            fWindowBuf = fWindows[i]
            for k in range(zBinNumber):
                aWindows[i, k, 1] = 0.0
                ## try to restrict this scaling only when histogram is nonzero
                if histWindows[i, k, 1] > 0.0:
                    biasPotential = 0.5 * springConst * (aWindows[i, k, 0] - Xi0) ** 2
                    aWindows[i, k, 1] = histNorm * np.exp(- biasPotential + fWindowBuf)

        ## aWindowsSum[:, 1] = np.sum(aWindows[i, :, 1] for i in range(nWindows))
        ## rewriting the above np.sum because it does not work in cython
        aWindowsSumBuf = np.zeros(shape=(zBinNumber, 2))
        for g in range(zBinNumber):
            for i in range(nWindows):
                aWindowsSumBuf[g, 1] += aWindows[i, g, 1]
            aWindowsSum[g, 1] = aWindowsSumBuf[g, 1]

        mainHistBuf = np.zeros(shape=(zBinNumber, 2))
        for k in range(zBinNumber):
            mainHistOld[k, 1] = mainHist[k, 1]
            if aWindowsSum[k, 1] > 0.0:  ## avoid div by zero
                ##mainHist[k, 1] = np.sum(histWindows[i, k, 1] for i in range(nWindows)) / aWindowsSum[k, 1]
                ## same here, replace sum by loop
                for i in range(nWindows):
                    mainHistBuf[k, 1] += histWindows[i, k, 1] / aWindowsSum[k, 1]
            mainHist[k, 1] = mainHistBuf[k, 1]

        fWindowsBuf = np.zeros(shape=nWindows)
        for i in range(nWindows):  ## ToDo  Add interval for integration
            # springConst = metaFile[i, 2]
            Xi0 = window_coords[i]
            fWindowsOld[i] = fWindows[i]
            ## fWindows[i] = -np.log(np.sum(zIntIn * mainHist[k, 1] * np.exp(-0.5 * springConst * (aWindows[i, k, 0] - Xi0) ** 2) for k in range(zBinNumber)))
            ## same here
            for k in range(zBinNumber):
                # fWindowsBuf[i] += zIntIn * mainHist[k, 1] * exp(-0.5 * springConst * (aWindows[i, k, 0] - Xi0) ** 2)
                fWindowsBuf[i] += zIntIn * mainHist[k, 1] * np.exp(-0.5 * springConst * (aWindows[i, k, 0] - Xi0) ** 2)
            fWindowsBuf[i] = -np.log(fWindowsBuf[i])
            fWindows[i] = fWindowsBuf[i]




        fNorm = 0.0
        mainNorm = 0.0
        for i in range(nWindows):
            fNorm += (fWindowsOld[i] - fWindows[i]) ** 2
        for i in range(zBinNumber):
            if (mainHistOld[i, 1] != 0.0 and mainHist[i, 1] != 0.0):
                mainNorm += (np.log(mainHistOld[i, 1]) - np.log(mainHist[i, 1])) ** 2

        # print("Norm convergence: F %f Hist %f" % (fNorm, mainNorm))
        # print("Convergence",fNorm, mainNorm)
        if (mainNorm < epsConvergeIn and fNorm < epsConvergeIn):
            # print("Norm convergence: F %.5f Hist %.5f" % (fNorm, mainNorm))

            print("WHAM converged")
            break
        elif recCount >= recursionsMaxIn - 1:
            print("Stopping WHAM recursion - max steps reached")
            break

        recCount += 1
    #
    #      ##=======##
    #         ## End of recursion
    #         ##=======##
    print("Convergence", fNorm, mainNorm)
    return mainHist


##=================================================##
##=================================================##

def compute_energies(movie_name, config_name, ligands):
    """
    Function to compute the intra-membrane interaction energy
    as well as the interaction energy between the nano partice and
    them membrane.

    :return:
    """
    ## ToDo box should be define at every time step (not yet implemented!!!!)

    work_movie = lt.splitMovie(movie_name, movie_format_mem)  ## split and load movie (coordinates etc)

    work_config = np.array(
        pd.read_csv(config_name, sep="\s+", names=config_format, header=None, dtype=object))
    #
    box_x = np.array(work_config[3, 0:2], dtype=float)
    box_y = np.array(work_config[4, 0:2], dtype=float)
    box_z = np.array(work_config[5, 0:2], dtype=float)
    #
    box = fr.box.Box(np.diff(box_x), np.diff(box_y), np.diff(box_z))  ## define the box

    num_frames = work_movie.shape[0]

    energies = []
    # num_frames = 1
    for frame in range(num_frames):
        # print(frame)

        # if frame%100 == 0:
        #     print(frame)
        # frame = 0
        work_movie_frame = work_movie[frame]
        # work_movie_frame  = work_movie_frame[work_movie_frame[:,1]] ## only np particles
        aq = fr.AABBQuery(box, work_movie_frame[:, 2:5])

        query_points = work_movie_frame[:, 2:5]
        query_result = aq.query(query_points, dict(r_max=4.5))
        nlist = query_result.toNeighborList()

        # nListMod = nlist.copy()

        nListMod = np.transpose(np.array([nlist.query_point_indices, nlist.point_indices, nlist.distances ]))

        ene = zhang_2010_potentials(work_movie_frame, nListMod, ligands)
        energies.append(ene)

    np.savetxt(workingDir + "/Results/test_ene.dat", energies, delimiter=' ')

##=================================================##
##=================================================##
@jit(nopython=True)
def zhang_2010_potentials(movie_frame, n_list, ligands):
    '''
    Implementaiton of membrane interaction potentials from:
    One-particle-thick, solvent-free, coarse-grained model for biological and biomimetic ﬂuid membranes
    PHYSICAL REVIEW E 82, 011905 2010

    Computes intra-membrane interaction as well as interaction between the ligands of the nano particle and the membrane.

    :param movie_frame:
    :param n_list: pre-computed neighbour list
    :param ligands: IDs of the ligands of the nano particle

    :return: potential energy of movie_frame configuration
    '''

    ##Membran-membrane interaction
    r_min = 1.122462
    r_cut = 2.6
    inv_del_r = 1.0/(r_cut - r_min)
    theta_0 = 0.0
    mu = 3
    eps = 4.34
    zeta = 4.0

    ##Membrane-nano particle interaction
    interact_types = list(ligands)
    r_cut_LJ = 1.8
    sigma_LJ = 1.0
    eps_LJ = 12.0

    U_total_mem = 0.0
    U_total_ligands = 0.0
    U_total_central = 0.0

    for i in range(len(n_list)):
        i1 = int(n_list[i,0])
        i2 = int(n_list[i,1])
        type_i = movie_frame[i1, 1]
        type_j = movie_frame[i2, 1]
        r_12_query = n_list[i,2]

        if type_i == 1 and type_j == 1 and i1 < i2:  ##compute membrane membrane interaction
            # r_12_test = LA.norm(movie_frame[i1, 3:6] - movie_frame[i2, 3:6]) ## is distance from AABB quesry the same as from movie
            # assert np.isclose(r_12_query, r_12_test)

            r_12 = movie_frame[i1, 2:5] - movie_frame[i2, 2:5]
            r_12 /= LA.norm(r_12)
            n1 =  movie_frame[i1, 5:8] ## mux muy muz; are normed in LAMMPS output
            n2 =  movie_frame[i2, 5:8] ## mux muy muz; are normed in LAMMPS output
            a = np.dot(np.cross(n1, r_12), np.cross(n2, r_12)) + np.sin(theta_0*np.dot(n2-n1, r_12)) - (np.sin(theta_0))**2
            phi = 1.0 + mu*(a - 1.0)


            if r_12_query < r_min:
                frac_buf = r_min/r_12_query
                frac_buf2 = frac_buf*frac_buf
                u_r = eps*(frac_buf2*frac_buf2 - 2.0*frac_buf2)
                U_total_mem += (u_r + (1.0-phi)*eps)
            elif r_12_query < r_cut:
                # assert r_12_query > r_min
                cos_buf = np.cos(0.5*np.pi*(r_12_query-r_min)*inv_del_r)
                u_a = -eps*cos_buf**(2.0*zeta)
                U_total_mem += u_a*phi

        # if (type_i == 1 and type_j in interact_types) or (type_j == 1 and type_i in interact_types):
        if (type_i == 1 and type_j in interact_types and i1 < i2): ## ligand-membrane interaction
            if r_12_query < r_cut_LJ:
                frac_buf = sigma_LJ / r_12_query
                frac_buf3 = frac_buf*frac_buf*frac_buf
                frac_buf6 = frac_buf3*frac_buf3
                shift = 0.0285368
                int_LJ = 4.0*eps_LJ*(frac_buf6*frac_buf6 - frac_buf6) + 4.0*eps_LJ*shift
                U_total_ligands += int_LJ

        if (type_i == 1 and type_j == 2 and i1 < i2):
            if r_12_query < 4.45:
                frac_buf = 4.0 / r_12_query
                frac_buf3 = frac_buf*frac_buf*frac_buf
                frac_buf6 = frac_buf3*frac_buf3
                shift = 0.249245
                int_LJ = 4.0*100*(frac_buf6*frac_buf6 - frac_buf6) + 4.0*100.0*shift
                if int_LJ < 0.0:
                    print(r_12_query, int_LJ)
                U_total_central += int_LJ


    ## energies are double counted
    return np.array([U_total_mem, U_total_ligands, U_total_central])


##=================================================##
##=================================================##
def rotational_displacement(realsMax):

    cases = ["bud", "nonbud", "uniform_golden"]
    # cases = ["bud", "nonbud"]
    # cases = ["uniform_golden"]
    cases = ["nonbud"]
    cases = ["bud", "nonbud"]
    for c in cases:
        real = 0
        for real in range(0,50):
            reference_vec = np.array([0, 0, 1])
            print(real)

            movie_name = scratch_dir + c + "_output/Results/" + f"243863_12900_0_out_{real}.xyz"
            config_name = scratch_dir + c + "_output/" + "243863_12900_0_data.data"

            if os.path.exists(movie_name):
                pass
            else:
                movie_name = scratch_dir + c + "_output/Results/" + f"run_out_{real}.xyz"
                config_name = scratch_dir + c + "_output/" + f"run_data_{real}.data"

            num_files = len(sorted(glob.glob(scratch_dir + c + "_output/Results/" + "*.xyz")))

            assert num_files == 50

            work_movie = lt.splitMovie(movie_name, movie_format)  ## split and load movie (coordinates etc)

            work_config = np.array(
                pd.read_csv(config_name, sep="\s+", names=config_format, header=None, dtype=object))

            box_x = np.array(work_config[3, 0:2], dtype=float)
            box_y = np.array(work_config[4, 0:2], dtype=float)
            box_z = np.array(work_config[5, 0:2], dtype=float)
            ## ToDo box should vary in size each time step but not impoertant here since computing rotational displacememnt
            box = fr.box.Box(np.diff(box_x), np.diff(box_y), np.diff(box_z))  ## define the box

            num_frames = work_movie.shape[0]
            rotataional_displ = np.zeros(shape =(num_frames, 2))
            for frame in range(num_frames):
                work_movie_frame = work_movie[frame]
                tracking_vec = work_movie_frame[work_movie_frame[:,1] == 3][0,2:5] - work_movie_frame[work_movie_frame[:,1] == 2][0,2:5]
                tracking_vec /= LA.norm(tracking_vec)

                # cross_prod = np.cross(reference_vec, tracking_vec)
                dot_prod = np.dot(reference_vec, tracking_vec)
                angle = np.arccos(dot_prod)
                rotataional_displ[frame, 0] = frame
                rotataional_displ[frame, 1] = angle

                reference_vec = tracking_vec
            np.savetxt(scratch_dir + c + f"_output/Results/rot_disp_sq_{real}.dat", rotataional_displ, delimiter = ' ')

##=================================================##
##=================================================##
def num_membrane_clusters(realsMax):
    '''
    This function computes the number of disconnected clusters made up of membrane particles.
    It should be used as an indicator of whether budding has taken place.
    :param realsMax:
    :return:
    '''


    real = 0
    cases = ["bud"]
    cases = ["bud","bud_fast" ,"nonbud","uniform_golden"]
    cases = ["bud_fast" ,"nonbud","uniform_golden"]



    for c in cases:
        budding_times = []
        for real in range(0,50):
            print(real)
            ## ToDo this if statement is deprecated: movie names do not differ any longer

            movie_name = scratch_dir + c + "_output/Results/" + f"243863_12900_0_out_{real}.xyz"
            config_name = scratch_dir + c + "_output/" + "243863_12900_0_data.data"

            if os.path.exists(movie_name):
                pass
            else:
                movie_name = scratch_dir + c + "_output/Results/" + f"run_out_{real}.xyz"
                config_name = scratch_dir + c + "_output/" + f"run_data_{real}.data"


            num_files = len(sorted(glob.glob(scratch_dir + c + "_output/Results/" + "*.xyz")))

            assert num_files == 50

            # elif c == "nonbud":
            #     movie_name = scratch_dir + c + "_output/Results/" + f"243865_-1_0_out_{real}.xyz"
            #     config_name = scratch_dir + c + "_output/" + "243865_-1_0_data.data"

            work_movie = lt.splitMovie(movie_name, movie_format)  ## split and load movie (coordinates etc)

            test_for_clusters = []
            # work_config = np.array(pd.read_csv(config_name, sep="\s+", names=config_format, header=None, dtype=object))

            num_frames = work_movie.shape[0]


            test_list = np.zeros(shape=num_frames)
            for k in range(num_frames):
                work_movie_frame = work_movie[k]
                membrane_coords = work_movie_frame[work_movie_frame[:, 1] == 1]
                cluster_inds = membrane_coords[:, 5]
                unique_inds, unique_counts = np.unique(cluster_inds, return_counts=True)
                sorted_counts = sorted(unique_counts) ## sort to expose the two largest groups
                num_cluster_thresh = 200
                if len(unique_inds) > 1 and sorted_counts[-1] > num_cluster_thresh and sorted_counts[-2]> num_cluster_thresh:
                    print("inds",unique_inds)
                    print("sort",sorted_counts)
                    assert sorted_counts[-1] + sorted_counts[-2] > 2800
                    budding_times.append([real,k])
                    break
                else:
                    pass
        print(budding_times)
        np.savetxt(scratch_dir + c + f"_output/Results/budding_times_{c}.dat", np.array(budding_times),
                   delimiter=' ')


def degree_of_wrapping(realsMax):
    """
    Function to compute the number of neighbours between the nano particle and the membrane to
    obtain the degree of wrapping. Also to compare with the result obtained trough colvars.
    :param movie:
    :param box:
    :return:
    """
    real = 0
    cases = ["bud", "nonbud"]
    cases = ["uniform_wrong"]
    cases = ["bud", "nonbud"]
    cases = ["uniform_golden"]

    for c in cases:
        for real in range(0,50):
        # for real in range(realsMax):
            print(c, real)
            num_wrapped_list = []

            movie_name = scratch_dir + c + "_output/Results/" + f"243863_12900_0_out_{real}.xyz"
            config_name = scratch_dir + c + "_output/" + "243863_12900_0_data.data"

            if os.path.exists(movie_name):
                pass
            else:
                movie_name = scratch_dir + c + "_output/Results/" + f"run_out_{real}.xyz"
                config_name = scratch_dir + c + "_output/" + f"run_data_{real}.data"

            num_files = len(sorted(glob.glob(scratch_dir + c + "_output/Results/" + "*.xyz")))

            assert num_files == 50

            work_movie = lt.splitMovie(movie_name, movie_format)  ## split and load movie (coordinates etc)

            work_config = np.array(
                pd.read_csv(config_name, sep="\s+", names=config_format, header=None, dtype=object))

            box_x = np.array(work_config[3,0:2], dtype = float)
            box_y = np.array(work_config[4,0:2], dtype = float)
            box_z = np.array(work_config[5,0:2], dtype = float)

            box = fr.box.Box(np.diff(box_x), np.diff(box_y), np.diff(box_z)) ## define the box

            # fig = plt.figure()
            # 	# ax = plt.axes(projection='3d')
            # 	# box.plot(ax=ax)
            # 	# for image in [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]]:
            # 	# 	box.plot(ax=ax, image=image, linestyle='dashed', color='gray')
            # 	# plt.show()

            num_frames = work_movie.shape[0]

            for frame in np.arange(0, work_movie.shape[0], 20):
                work_movie_frame = work_movie[frame]
                # work_movie_frame  = work_movie_frame[work_movie_frame[:,1]] ## only np particles
                aq = fr.AABBQuery(box, work_movie_frame[:,2:5])

                # ax = plt.axes(projection='3d')
                # system.plot(ax=ax)


                # ax = plt.axes(projection='3d')
                # for cluster_id in range(cl.num_clusters):
                # 	cluster_system = fr.AABBQuery(system.box, system.points[cl.cluster_keys[cluster_id]])
                # 	cluster_system.plot(ax=ax, s=10, label="Cluster {}".format(cluster_id))
                # 	print("There are {} points in cluster {}.".format(len(cl.cluster_keys[cluster_id]), cluster_id))
                #
                # ax.set_title('Clusters identified', fontsize=20)
                # ax.legend(loc='best', fontsize=14)
                # ax.tick_params(axis='both', which='both', labelsize=14, size=8)
                # plt.show()

                ## Builiding the Neighbour list
                # query_points = work_movie_frame[work_movie_frame[:,1] > 1][0,2:5] ## query only for nano particles
                query_points = work_movie_frame[:,2:5]
                query_result = aq.query(query_points, dict(r_max=5.05))
                nlist = query_result.toNeighborList()

                nListMod = nlist.copy()

                # test_list = []
                # for i, j in nListMod[:]:
                #     type_i = work_movie_frame[i,1]
                #     type_j = work_movie_frame[j,1]
                #     if type_i == 2 and type_j == 1:
                #         test_list.append([i, j])
                # if len(test_list) > 0: ## if there are neighbours between nano particles and membrane
                #     test_list = np.array(test_list)
                #     num_wrapped = len(np.unique(test_list[:,1])) ## number of individual membrane particles which are in contact with the np
                # else:
                #     num_wrapped = 0
                num_wrapped = eval_nList_central(nListMod[:], work_movie_frame)
                # num_wrapped = eval_nList_active_ligands(nListMod[:], work_movie_frame)

                ## add the centre of mass of the nano particle
                np_centre_of_mass_z = np.mean(work_movie_frame[work_movie_frame[:,1]>1][:,4])
                num_wrapped_list.append([frame, np_centre_of_mass_z, num_wrapped])
                # print(frame, np_centre_of_mass_z, num_wrapped)
            np.savetxt(scratch_dir + c + f"_output/Results/wrapped_list_{c}_{real}.dat", np.array(num_wrapped_list), delimiter=' ')

##=================================================##
##=================================================##
# @jit(nopython=True)
def eval_nList_central(nListModArray,work_movie_frame):
    nList_central = nListModArray[nListModArray[:,0] == 2900]## reduced neighbour list as viewed from central particle
    nList_central = nList_central[nList_central[:, 1] < 2900] ## don't count ligands in neighbour search (import when comparing nano particles with different number of ligands)
    # for k in range(len(nList_central)):
    #     ind_i = nList_central[k, 0] ## particle index in movie file
    #     ind_j = nList_central[k, 0]
    #     type_i = work_movie_frame[ind_i, 1] ## corresponding particle type
    #     type_j = work_movie_frame[ind_j, 1]
    #     assert int(type_i) == 2
    #     test_list.append([ind_i, ind_j])
    # if len(test_list) > 0:  ## if there are neighbours between nano particles and membrane
    #     test_list_array = np.asarray(test_list)
    #     num_wrapped = len(
    #         np.unique(test_list_array[:, 1]))  ## number of individual membrane particles which are in contact with the np
    #     pass
    # else:
    #     num_wrapped = 0
    num_wrapped = len(nList_central)
    return num_wrapped

def eval_nList_active_ligands(nListModArray,work_movie_frame):
    ## get cross neighbour list
    nList_central = nListModArray[nListModArray[:,0] > 2900]## reduced neighbour list as viewed from central particle
    nList_central =  nList_central[nList_central[:, 1] < 2900]
    ##ToDo using the index 2900 instead to checking the types here will cause problems

    num_wrapped = len(np.unique(nList_central[:, 1]))
    return num_wrapped


##=================================================##
##=================================================##
def switch_averaging_range():
    combine = []
    for k in range(0, 20):
        range_min = 100 * (2 * k + 1)
        range_max = 100 * (2 * k + 2)
        for i in range(range_min, range_max):
            combine.append(i)

    combine = np.array(combine)
    return combine
##=================================================##
##=================================================##

# def umbrella_analysis(ex_list):
def umbrella_analysis():
    fig_int, ax_int = plt.subplots(figsize = (100,100))
    ## ToDo add functionality: restrict averaging to later parts of window histograms
    ## ToDo add functionality: output wrapping degree for each individual run

    umbrella_traj = sorted(glob.glob(workingDir + "/*.traj"))

    umbrella_cols = ["step", "wrapping", "E", "x0_wrapping"]
    traj_combine = []

    # new_indices = switch_averaging_range()
    # ex_list = []
    for filename in umbrella_traj:
        print(filename)
        # match = re.search("^.*_(.*).colvars.traj", filename)  ## search for track file at step = stepVal
        # if match:
        # 	stepVal = match.group(1)
        # 	print(stepVal)
        # 	if int(stepVal) in ex_list:
        # 		print("in exclude")
        # 		pass
        # 	else:
        # 		print("adding")
        d_frame = pd.read_csv(filename, names=umbrella_cols, comment='#', header=None, sep="\s+")[ 0:-1]  ## small bug, last row should not be there
        # d_frame = d_frame.iloc[new_indices]
        traj_combine.append(d_frame)

    input_data = pd.concat(traj_combine, axis=0, ignore_index=True)


    work_array = np.array(input_data)
    ## temporarily fiter out zeros
    work_array = work_array[work_array[:, 1] > 0]
    spring_constant = 0.25
    checkpoints = np.unique(input_data["x0_wrapping"])[::-1]
    checkpoints = checkpoints[:-1] ## Exclude first frame, can cause instabilities in WHAM algorithm
    checkpoints = checkpoints[1:] ## Exclude last frame, can cause instabilities in WHAM algorithm
    # checkpoints = np.unique(input_data["x0_wrapping"])[5:] ## skip first 5 windows
    zStart = np.min(checkpoints) - 30.0
    zStart = 0
    zEnd = np.max(checkpoints) + 100.0

    zInt = 0.1
    numRecursions = 1000
    epsilon_converge = 1e-6

    window_histograms = standardBinning(work_array, checkpoints, zStart, zEnd, zInt, 0)

    for i in range(window_histograms.shape[0]):
        ax_int.plot(window_histograms[i, :, 0],  0.05* i + window_histograms[i, :, 1], c='gray')
        ax_int.scatter(+ checkpoints[i],  0.05* i , c='red')
        ax_int.scatter(np.mean(window_histograms[i,:,0][window_histograms[i,:,1] >0]),  0.05* i , c='blue')

    # plt.show()
    fig_int.savefig(workingDir +'/' + plot_name + "hist.png")
    print(workingDir +'/' + plot_name + "hist.png")
    plt.close()

    wham_histogram = whamRecursion(window_histograms, zStart, zEnd, zInt, numRecursions, epsilon_converge, 0,
                                   spring_constant, checkpoints)


    ## test global binnding of wrapping  etc
    binning_input = work_array[:,[1,2]]

    test_binning = array_to_bins_z(binning_input, zStart, zEnd, zInt)








    return wham_histogram, test_binning

    ##=================================================##
    ##=================================================##
    # def main():

    # movie_name = workingDir + "243685_-1_0_out.xyz"
    # log_name = workingDir + "log.lammps"

    # logArray = lt.readLog(log_name)
    # print(logArray)

    # workMovie = lt.splitMovie(movie_name, movie_format) ## split and load movie (coordinates etc)
    # num_frames = len(workMovie)

    # num_np = sum(workMovie[0,:,1] > 1) ## nano particle number
    # np_coords = np.zeros(shape = (workMovie.shape[0], num_np, len(movie_format)))

    # for i in range(num_frames):
    # 	movie_buffer = workMovie[i]
    # 	# print(movie_buffer[movie_buffer[:,1]>1].shape)
    # 	np_coords[i] = movie_buffer[movie_buffer[:,1]>1]

    # z_hist = []
    # for i in range(num_frames):
    # 	z_mean = np.mean(np_coords[i,:,4])
    # 	z_hist.append([i, z_mean])

    # z_hist = np.array(z_hist)

    # plt.hist(z_hist[:,1],bins = 100)
    # plt.show()

    pass
##=================================================##
##=================================================##
def test_umbrella():
    print("Running test...")
    umbrella = umbrella_analysis()
    pmfs = umbrella[0]
    out_pmfs = np.transpose(np.array([pmfs[:, 0], -np.log(pmfs[:, 1], out=np.zeros_like(pmfs[:, 1]), where=pmfs[:, 1] != 0)]))
    np.savetxt("Results/test_pmf.dat", out_pmfs, delimiter=' ')

##=================================================##
##=================================================##
if __name__ == '__main__':

    do_umbrella_analysis = True
    do_wrapping_analysis = False
    do_cluster_analysis = False
    do_rotation_analysis = False

    test_case = True

    movie_format = ['id', 'type', 'x', 'y', 'z', 'c_cls']
    movie_format_mem = ['id', 'type', 'x', 'y', 'z', 'mux', 'muy', 'muz']
    config_format = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']

    workingDir = '..' + '/'
    start_time = time.time()


    scratch_dir = "np_umbrella_nph_rev1/"



    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()



    cases = ["bud"]
    if do_umbrella_analysis:
        for c in cases:
            workingDir = scratch_dir + c + "_output/Results_Umbrella"
            plot_name = c



            umbrella = umbrella_analysis()
            pmfs = umbrella[0]
            wrapping = umbrella[1]


            out_pmfs = np.transpose(np.array([pmfs[:,0],-np.log(pmfs[:, 1], out = np.zeros_like(pmfs[:, 1]), where=pmfs[:, 1]!=0) ]))
            # out_pmfs = np.transpose(np.array([pmfs[:,0],-np.log(pmfs[:, 1]) ]))
            out_wrap = np.transpose(np.array([wrapping[:,0],wrapping[:,1]]))

            np.savetxt(workingDir + '/' + c + "_pmf.dat", out_pmfs, delimiter= ' ')
            np.savetxt(workingDir + '/' + c + "_wrap.dat", out_wrap, delimiter= ' ')

            ax.plot(pmfs[:, 0], -np.log(pmfs[:, 1]), label=c, marker='o')
            ax.legend()


            # ax2.plot(wrapping[:, 1], -np.log(pmfs[:, 1]), label=c, marker='o')
            # ax2.plot(wrapping[:, 0], wrapping[:, 1], label=c, marker='o')
            # ax2.legend()


        fig.savefig("pmfs.png")
        # fig2.savefig("pmfs_wrap.png")

    print("--- %s seconds ---" % (time.time() - start_time))


##=================================================##
##=================================================##










