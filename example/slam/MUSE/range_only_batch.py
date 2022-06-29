import os
import time
import numpy as np
import csv

from geometry.TwoDimension import SE2Pose
from slam.Variables import R2Variable, SE2Variable
from factors.Factors import UnarySE2ApproximateGaussianPriorFactor, \
    SE2RelativeGaussianLikelihoodFactor, SE2R2RangeGaussianLikelihoodFactor
from slam.NFiSAM import NFiSAM, NFiSAMArgs
import matplotlib.pyplot as plt

from utils.Visualization import plot_2d_samples

import scipy.stats

# define data classes
from dataclasses import dataclass


@dataclass
class Range2LM:
    Time: float
    Mean: float
    Cov: float
    ID: int


@dataclass
class Odom2:
    Time: float
    Mean: np.ndarray
    Cov: np.ndarray


if __name__ == '__main__':
    # define the name of the directory to be created
    path = "plots"

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    # load data
    DataFile = "W100_Input.txt"
    ResultFile = "W100_Output.txt"

    # reading csv file
    RangeArray = []
    OdomArray = []
    TimeArray = []
    RuntimeArray = []
    with open(DataFile, 'r') as csvfile:
        # creating a csv reader object
        datareader = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)

        # get total number of rows
        print("Total no. of rows: %d" % datareader.line_num)

        for row in datareader:
            if row[0] == "range_lm2":
                # create range object
                Range = Range2LM(float(row[1]), float(row[2]), float(row[3]), int(float(row[4])))
                RangeArray.append(Range)
                # save timestamp
                TimeArray.append(float(Range.Time))
            elif row[0] == "odom2":
                Mean = np.array([float(row[2]), float(row[3]), float(row[4])])
                Cov = np.zeros((3, 3))
                Cov[0, 0] = float(row[5])
                Cov[1, 1] = float(row[6])
                Cov[2, 2] = float(row[7])
                Odom = Odom2(float(row[1]), Mean, Cov)
                OdomArray.append(Odom)
            else:
                raise RuntimeError("Wrong ID string!")

    # find number of timesteps
    TimeUnique = list(dict.fromkeys(TimeArray))
    NumTime = len(TimeUnique)

    # create basic graph
    args = NFiSAMArgs(posterior_sample_num=500,
                      flow_type="NSF_AR",
                      flow_number=1,
                      flow_iterations=2000,
                      local_sample_num=2000,
                      cuda_training=True,
                      hidden_dim=8,
                      num_knots=9,
                      learning_rate=0.01,
                      elimination_method='pose_first')
    Graph = NFiSAM(args)

    TimeOld = TimeUnique[0] - TimeUnique[1]
    PoseNodeOld = SE2Variable('x' + str(TimeOld))
    IDSet = set()
    PoseArray = []
    for n in range(NumTime):

        # add new pose
        Time = TimeUnique[n]
        PoseNode = SE2Variable('x' + str(Time))
        PoseArray.append(PoseNode)
        Graph.add_node(PoseNode)

        # add prior or odom
        if n == 0:
            # create prior
            PriorPose = SE2Pose(x=0, y=0, theta=np.pi)
            PriorCov = np.identity(3) * (0.1 ** 2)
            PriorCov[2, 2] = 0.01 ** 2
            PriorFactor = UnarySE2ApproximateGaussianPriorFactor(var=PoseNode,
                                                                 prior_pose=PriorPose,
                                                                 covariance=PriorCov)
            # add prior
            Graph.add_factor(PriorFactor)
        else:
            # create odometry factor (as relative pose)
            dT = Time - TimeOld
            PoseNodeOld = SE2Variable('x' + str(TimeOld))
            OdomPose = SE2Pose(x=OdomArray[n - 1].Mean[0] * dT, y=OdomArray[n - 1].Mean[1] * dT,
                               theta=OdomArray[n - 1].Mean[2] * dT)
            OdomCov = OdomArray[n - 1].Cov * (dT ** 2)
            OdomFactor = SE2RelativeGaussianLikelihoodFactor(var1=PoseNodeOld, var2=PoseNode,
                                                             observation=OdomPose,
                                                             covariance=OdomCov)
            # add odometry
            Graph.add_factor(OdomFactor)

        # add ranges
        for Range in RangeArray:
            if Range.Time == Time:
                LMNode = R2Variable('l' + str(Range.ID))
                # create landmark if not exist
                if Range.ID not in IDSet:
                    IDSet.add(Range.ID)
                    LMNode = R2Variable('l' + str(Range.ID))
                    Graph.add_node(LMNode)
                # add range
                LMFactor = SE2R2RangeGaussianLikelihoodFactor(var1=PoseNode,
                                                              var2=LMNode,
                                                              observation=Range.Mean,
                                                              sigma=np.sqrt(Range.Cov))
                Graph.add_factor(LMFactor)

        # solve graph
        Graph.update_physical_and_working_graphs()
        start = time.time()
        Samples = Graph.incremental_inference(timer=[start])
        end = time.time()
        RuntimeArray.append(end - start)

        # plot
        print("Time for phase " + str(n) + " inference " + str(end - start) + " sec")
        plt.figure()
        plot_2d_samples(samples_mapping=Samples, show_plot=True, file_name=path + '/step' + str(n) + '.pdf',
                        legend_on=False, title='Posterior estimation (step ' + str(n) + ')', equal_axis=False,
                        xlim=(-20, 20), ylim=(-20, 20))

        # store old stuff
        PoseNodeOld = PoseNode
        TimeOld = Time

    # get result
    DataPose = []
    DataRuntime = []
    for n in range(len(PoseArray)):
        # get samples for specific pose in SE2
        PoseSamples = Samples.get(PoseArray[n])

        # calculate mean separately for position and orientation
        PoseMean = np.zeros((3, 1))
        PoseMean[0:1] = PoseSamples[:, 0:1].mean(axis=0)
        PoseMean[2] = scipy.stats.circmean(PoseSamples[:, 2], high=np.pi, low=-np.pi, axis=0)

        # pre-calculate error and wrap circular error
        Error = PoseSamples - PoseMean.transpose()
        Error[:, 2] = (Error[:, 2] + np.pi) % (2 * np.pi) - np.pi

        # evaluate covariance
        PoseCov = np.cov(Error.transpose())

        # prepare CSV rows
        DataPose.append(["pose2", TimeUnique[n], PoseMean[0, 0], PoseMean[1, 0], PoseMean[2, 0],
                         PoseCov[0, 0], PoseCov[1, 0], PoseCov[2, 0],
                         PoseCov[0, 1], PoseCov[1, 1], PoseCov[2, 1],
                         PoseCov[0, 2], PoseCov[1, 2], PoseCov[2, 2]])
        DataRuntime.append(["solver_summary", TimeUnique[n], RuntimeArray[n], RuntimeArray[n], 0, 0, 0, 0, 0])

    # write to CSV
    with open(ResultFile, 'w', encoding='UTF8') as f:
        writer = csv.writer(f, delimiter=' ')
        for Row in DataPose:
            writer.writerow(Row)
        for Row in DataRuntime:
            writer.writerow(Row)
