import os
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import *
from metrics import *
from model import social_stgcnn
import copy
import utils
import argparse
import os
import torch
import config
from attrdict import AttrDict
import sys
sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', default=0, type=int)
parser.add_argument('--model_path', default="all", type=str) # Model path
parser.add_argument('--external_test', type=int, default=0)
parser.add_argument('--external', type=int, default=0)

generator = None # The model that generates our paths
loader_test = None

def test(KSTEPS=20):
    global loader_test,generator
    model = generator
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step =0
    for batch in loader_test:
        step+=1
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch


        num_of_objs = obs_traj_rel.shape[1]

        #Forward
        #V_obs = batch,seq,node,feat
        #V_obs_tmp = batch,feat,seq,node
        V_obs_tmp =V_obs.permute(0,3,1,2)

        V_pred,_ = model(V_obs_tmp,A_obs.squeeze())
        # print(V_pred.shape)
        # torch.Size([1, 5, 12, 2])
        # torch.Size([12, 2, 5])
        V_pred = V_pred.permute(0,2,3,1)
        # torch.Size([1, 12, 2, 5])>>seq,node,feat
        # V_pred= torch.rand_like(V_tr).cuda()


        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred,V_tr =  V_pred[:,:num_of_objs,:],V_tr[:,:num_of_objs,:]
        #print(V_pred.shape)

        #For now I have my bi-variate parameters
        #normx =  V_pred[:,:,0:1]
        #normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:,:,2]) #sx
        sy = torch.exp(V_pred[:,:,3]) #sy
        corr = torch.tanh(V_pred[:,:,4]) #corr

        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).cuda()
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]

        mvnormal = torchdist.MultivariateNormal(mean,cov)


        ### Rel to abs
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len

        #Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                 V_x[0,:,:].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                 V_x[-1,:,:].copy())

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for n in range(num_of_objs):
            ade_ls[n]=[]
            fde_ls[n]=[]

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()



            #V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                     V_x[-1,:,:].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

           # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:,n:n+1,:])
                target.append(V_y_rel_to_abs[:,n:n+1,:])
                obsrvs.append(V_x_rel_to_abs[:,n:n+1,:])
                number_of.append(1)

                ade_ls[n].append(ade(pred,target,number_of))
                fde_ls[n].append(fde(pred,target,number_of))

        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls)/len(ade_bigls)
    fde_ = sum(fde_bigls)/len(fde_bigls)
    return ade_,fde_,raw_data_dict


def main(args):
    paths = ['./checkpoint/*social-stgcnn*'] if args.model_path == "all" else [args.model_path]
    KSTEPS=20

    print("*"*50)
    print('Number of samples:',KSTEPS)
    print("*"*50)




    for feta in range(len(paths)):
        ade_ls = []
        fde_ls = []
        path = paths[feta]
        exps = glob.glob(path)
        print('Model being tested are:',exps)

        for exp_path in exps:
            print("*"*50)
            print("Evaluating model:",exp_path)

            model_path = exp_path+'/val_best.pth'
            args_path = exp_path+'/args.pkl'
            with open(args_path,'rb') as f:
                args = pickle.load(f)

            stats= exp_path+'/constant_metrics.pkl'
            with open(stats,'rb') as f:
                cm = pickle.load(f)
            print("Stats:",cm)



            #Data prep
            obs_seq_len = args.obs_seq_len
            pred_seq_len = args.pred_seq_len
            data_set = './datasets/'+"Waymo_small"+'/' #'./datasets/'+args.dataset+'/' # HAAACK RENIVE MEEE

            dset_test = TrajectoryDataset(
                    data_set+'test/',
                    batchSize=1,
                    obs_len=obs_seq_len,
                    pred_len=pred_seq_len,
                    skip=1,norm_lap_matr=True)

            global loader_test
            loader_test = DataLoader(
                    dset_test,
                    batch_size=1,#This is irrelative to the args batch size parameter
                    shuffle =False,
                    num_workers=0)



            #Defining the model
            model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
            output_feat=args.output_size,seq_len=args.obs_seq_len,
            kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()
            model.load_state_dict(torch.load(model_path))

            global generator
            generator = model


            ade_ =999999
            fde_ =999999
            print("Testing ....")
            ad,fd,raw_data_dic_= test()
            ade_= min(ade_,ad)
            fde_ =min(fde_,fd)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            print("ADE:",ade_," FDE:",fde_)




        print("*"*50)

        print("Avg ADE:",sum(ade_ls)/5)
        print("Avg FDE:",sum(fde_ls)/5)


# FLASK SESSION GLOBAL DEFINES
import numpy as np
from flask import Flask, jsonify, request
import json
import io
app = Flask(__name__)

NUM_FRAMES_TO_OBSERVE = 8
NUM_FRAMES_TO_PREDICT = 8
MAIN_AGENT_NAME = "main" # The agent under evaluation
MAIN_AGENT_INDEX = 0

history_pos = {} # History of positons for a given agent name
#----------

# This code runs the inference for one frome
# Input params:
# agentsObservedPos dict of ['agentName'] -> position as np array [2], all agents observed in this frame
# optional: forcedHistoryDict -> same as above but with NUM_FRAMES_OBSERVED o neach agent, allows you to force / set history
# Output : returns the position of the 'main' agent
def DoModelInferenceForFrame(agentsObservedOnFrame, forcedHistoryDict = None):
    global history_pos

    # Update the history if forced param is used
    if forcedHistoryDict != None:
        for key, value in forcedHistoryDict.items():
            assert isinstance(value, np.ndarray), "value is not instance of numpy array"
            assert value.shape is not (NUM_FRAMES_TO_OBSERVE, 2)
            history_pos[key] = value

    # Update the history of agents seen with the new observed values
    for key, value in agentsObservedOnFrame.items():
        # If agent was not already in the history pos, init everything with local value
        if key not in history_pos:
            history_pos[key] = np.tile(value, [NUM_FRAMES_TO_OBSERVE, 1])
        else:  # Else, just put his new pos in the end of history
            values = history_pos[key]
            values[0:NUM_FRAMES_TO_OBSERVE - 1] = values[1:NUM_FRAMES_TO_OBSERVE]
            values[NUM_FRAMES_TO_OBSERVE - 1] = value

    # Do simulation using the model
    # ------------------------------------------

    # Step 1: fill the input
    numAgentsThisFrame = len(agentsObservedOnFrame)

    # Absolute observed trajectories
    obs_traj = np.zeros(shape=(NUM_FRAMES_TO_OBSERVE, numAgentsThisFrame, 2), dtype=np.float32)
    # Zero index is main, others are following
    obs_traj[:, MAIN_AGENT_INDEX, :] = history_pos[MAIN_AGENT_NAME]
    index = 1
    indexToAgentNameMapping = {}
    indexToAgentNameMapping[MAIN_AGENT_INDEX] = MAIN_AGENT_NAME
    for key, value in agentsObservedOnFrame.items():
        if key != MAIN_AGENT_NAME:
            obs_traj[:,index,:] = history_pos[key]
            indexToAgentNameMapping[index] = key
            index += 1

    # Relative observed trajectories
    obs_traj_rel = np.zeros(shape=(NUM_FRAMES_TO_OBSERVE, numAgentsThisFrame, 2), dtype=np.float32)
    obs_traj_rel[1:, :, :] = obs_traj[1:, :, :] - obs_traj[:-1, :, :]

    seq_start_end = np.array([[0, numAgentsThisFrame]])  # We have only 1 batch containing all agents
    # Transform them to torch tensors
    obs_traj = torch.from_numpy(obs_traj).type(torch.float)
    obs_traj_rel = torch.from_numpy(obs_traj_rel).type(torch.float)

     # Permute to the normal input size of the models [num_agents, 2, num_frames]
    obs_traj = obs_traj.permute(1, 2, 0).unsqueeze(0)
    obs_traj_rel = obs_traj_rel.permute(1, 2, 0).unsqueeze(0)

    seq_start_end = torch.from_numpy(seq_start_end)

    V_obs, A_obs = utils.seq_to_graph(obs_traj, obs_traj_rel)

    if len(V_obs.shape) < 4:
        V_obs = V_obs.unsqueeze(0)
    if (len(A_obs.shape) < 4):
        A_obs = A_obs.unsqueeze(0)

    V_obs_tmp = V_obs.permute(0, 3, 1, 2)
    V_obs_tmp = V_obs_tmp.cuda()
    A_obs_tmp = A_obs.squeeze(0).cuda()
    V_pred, _ = generator(V_obs_tmp, A_obs_tmp)
    V_pred = V_pred.permute(0, 2, 3, 1)
    V_pred = V_pred.squeeze(0)

    num_of_objs = obs_traj_rel.shape[1]
    V_pred = V_pred[:, :num_of_objs, :]

    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, 4])  # corr
    cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).cuda()
    cov[:, :, 0, 0] = sx * sx
    cov[:, :, 0, 1] = corr * sx * sy
    cov[:, :, 1, 0] = corr * sx * sy
    cov[:, :, 1, 1] = sy * sy
    mean = V_pred[:, :, 0:2]
    mvnormal = torchdist.MultivariateNormal(mean, cov)
    V_pred = mvnormal.sample()

    V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
    V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze(0).copy(),
                                            V_x[0, :, :].copy())
    V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().copy(),
                                               V_x[-1, :, :].copy())

    #pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
    #pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])


    # Take the first predicted position and add it to history
    pred_traj_fake = V_pred_rel_to_abs
    newMainAgentPos = pred_traj_fake[0][0]  # Agent 0 is our main agent

    return newMainAgentPos


@app.route('/predict', methods=['POST'])
def predict():
    if request.method =='POST':
        # Get the file from request
        #file = request.files['file']
        dataReceived = json.loads(request.data)

        agentsObservedThisFrame = dataReceived['agentsPosThisFrame']
        agentsForcedHistory = dataReceived['agentsForcedHistoryPos'] if 'agentsForcedHistoryPos' in dataReceived else None

        # Read all agents data received
        #-----------------------------------------
        #agentIndex = 1
        agentsObservedPos = {}
        for key,value in agentsObservedThisFrame.items():
            value = np.array(value, dtype=np.float32)
            if key == MAIN_AGENT_NAME:
                agentsObservedPos[MAIN_AGENT_NAME] = value
            else:
                agentsObservedPos[key] = value

        forcedHistoryPos = None
        if agentsForcedHistory is not None:
            forcedHistoryPos = {}
            for key, value in agentsForcedHistory.items():
                value = np.array(value, dtype=np.float32)
                if key == MAIN_AGENT_NAME:
                    forcedHistoryPos[MAIN_AGENT_NAME] = value
                else:
                    forcedHistoryPos[key] = value

        # Then do model inference for agents observed on this frame
        # Get back the new position for main agent and return it to caller
        newMainAgentPos = DoModelInferenceForFrame(agentsObservedPos, forcedHistoryPos)
        return jsonify(newMainAgentPos = str(list(newMainAgentPos)))


def deloyModelForFlaskInference():
    global generator
    #torch.load(args.model_path)

    print("Evaluating model:", args.model_path)

    model_path = args.model_path + '/val_best.pth'
    args_path = args.model_path +'/args.pkl'
    with open(args_path,'rb') as f:
        args_model = pickle.load(f)

    stats= args.model_path + '/constant_metrics.pkl'
    with open(stats,'rb') as f:
        cm = pickle.load(f)
    print("Stats:",cm)

    #Data prep
    obs_seq_len = args_model.obs_seq_len
    pred_seq_len = args_model.pred_seq_len
    data_set = './datasets/'+args_model.dataset+'/'

    #Defining the model
    model = social_stgcnn(n_stgcnn =args_model.n_stgcnn,n_txpcnn=args_model.n_txpcnn,
    output_feat=args_model.output_size,seq_len=args_model.obs_seq_len,
    kernel_size=args_model.kernel_size,pred_seq_len=args_model.pred_seq_len).cuda()
    model.load_state_dict(torch.load(model_path))

    generator = model

def startExternalTest():
    deloyModelForFlaskInference()

    historyMainAgent = np.zeros(shape=(NUM_FRAMES_TO_OBSERVE, 2))
    agentsObservedPos = {MAIN_AGENT_NAME : np.array([0,0], dtype=np.float32)}

    for frameIndex in range(100):
        forcedHistory = None
        if frameIndex == 0:
            forcedHistory = {MAIN_AGENT_NAME : historyMainAgent}

        newMainAgentPos = DoModelInferenceForFrame(agentsObservedPos, forcedHistory)
        print(f"Frame {frameIndex}: {newMainAgentPos}")
        agentsObservedPos[MAIN_AGENT_NAME] = newMainAgentPos

import customFlaskUtils

if __name__ == '__main__':
    args = parser.parse_args()
    config.initDevice(args.use_gpu)

    if True and args.external == True:
        deloyModelForFlaskInference()
        args.portsConfig = customFlaskUtils.PortsConfig(carla=5000, waymo=5001, others=8000)
        customFlaskUtils.runFlask(args, app)

    elif True and args.external_test == True:
        startExternalTest()
    else: # normal evaluation
        main(args)



