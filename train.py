import torch
from torchdiffeq import odeint
from sklearn.decomposition import PCA
import numpy as np
from itertools import combinations
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import argparse
import os
import random
from tqdm import tqdm

from dataset import *
from model import *


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--epoch', default=1000, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='learning rate in training')
    parser.add_argument('--log_dir', default='runs/experiment', help='log directory')
    parser.add_argument('--pca', type=bool, default= True, help='pca already exist.')
    parser.add_argument('--pcanum', default=150, type=int, help='number of PCA components')
    return parser.parse_args()

def main(args):
    n_vertices = 1490 # Hippocampal shape vertex number
    n_reduced_vertices = args.pcanum # default 150 dimensions
    print(f"{n_reduced_vertices=}")
    n_conditions = 1  # condition on disease range in [0, 1] where 0: normal, 1: alzheimer's disease
    
    # Train/Test Dataset Preparation
    torch.manual_seed(42) 
    np.random.seed(42)     
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)  
        torch.cuda.manual_seed_all(42) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False    
    os.makedirs(rf"{args.log_dir}\ckpt", exist_ok=True)
    
    # Load train and test data
    traindata, testdata = train_test_dataset() 

    # TensorBoard Writer Initialization
    writer = SummaryWriter(log_dir=args.log_dir)

    # PCA construction
    if args.pca==False:
        pca = PCA(n_components=n_reduced_vertices)
        sample_data_flat = np.vstack([d[0].numpy() for d in traindata['data']])
        _ = pca.fit_transform(sample_data_flat)

    else: 
        with open(rf"dataset/PCA/pca.pickle","rb") as f:
            pca_dict = pickle.load(f)
            pca = pca_dict['pca']
    
    # PCA dimension reduction for train data
    all_data_flat = np.vstack([d.numpy() for d in traindata['data']])
    reduced_data_flat = pca.transform(all_data_flat)
    reduced_train_data = []
    start_idx = 0
    for d in traindata['data']:
        n_t = len(d)
        reduced_train_data.append(torch.tensor(reduced_data_flat[start_idx:start_idx+n_t], dtype=torch.float32))
        start_idx += n_t

    # PCA dimension reduction for test data
    all_test_data_flat = np.vstack([d.numpy() for d in testdata['data']])
    reduced_test_data_flat = pca.transform(all_test_data_flat)
    
    reduced_test_data = []
    start_idx = 0
    for d in testdata['data']:
        n_t = len(d)
        reduced_test_data.append(torch.tensor(reduced_test_data_flat[start_idx:start_idx+n_t], dtype=torch.float32))
        start_idx += n_t

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_times_list = traindata['time']
    train_cond_list = traindata['cond']
    test_times_list = testdata['time']
    test_cond_list = testdata['cond']
    print(f"{len(traindata['data'])}{traindata['data'][0].shape=}, {len(reduced_train_data)}{reduced_train_data[0].shape=}")
    
    # Train
    def train_brain_ode(reduced_train_data, train_times_list, train_cond_list, 
                         reduced_test_data, test_times_list, test_cond_list, train_real_target, test_real_target, epochs, lr):
        torch.autograd.set_detect_anomaly(True)
        
        latent_dim = n_reduced_vertices
        condition_dim = n_conditions
        
        # Initialize model
        model = ODEFuncWithAttention(latent_dim, condition_dim).to(device)

        # Set up optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoch//5, gamma=0.5)

        n_train_samples = len(reduced_train_data)
        n_test_samples = len(reduced_test_data)
        batch = 2
        
        for epoch in (range(epochs)):
            total_train_loss = 0
            total_mm_train_loss = 0
            model.train()
            
            # shuffle data
            indices = np.random.permutation(n_train_samples)
            reduced_train_data = [reduced_train_data[i] for i in indices]
            train_times_list = [train_times_list[i] for i in indices]
            train_cond_list = [train_cond_list[i] for i in indices]
            train_real_target = [train_real_target[i] for i in indices]
            
            for i in tqdm(range(n_train_samples)):
                
                optimizer.zero_grad()
                
                V_true = reduced_train_data[i].to(device)
                V_real_true = train_real_target[i]
                t = train_times_list[i].to(device)
                c = train_cond_list[i].to(device)
                n_timepoints = len(t)
                if i % batch == 0:
                    loss = 0
                    mm_train_loss = 0
                    n_pairs = 0

                # Forward pass: younger -> older
                for start_idx in (range(n_timepoints)):
                    V_0 = V_true[start_idx:start_idx+1]
                    
                    # Data augmentation: add noise
                    noise_std = 0.01  # noise size (adjustable)
                    scale_noise = random.uniform(0.98, 1.02)
                    noise = torch.randn_like(V_0) * noise_std
                    V_0 = (V_0 + noise) * scale_noise
                    
                    target_times = t[start_idx:]
                    target_cond = c[start_idx:]
                    target_data = (V_true[start_idx:] + noise)*scale_noise
                    
                    V_pred = odeint(lambda t, V: model(t, V, torch.tensor([target_cond[0]]).to(device)), 
                                    V_0, target_times, method='rk4').squeeze(1)
                    # L2 loss in PCA space
                    loss += torch.mean(((V_pred - target_data) ** 2))
                    
                    # Inverse transform to acquire mesh vertices
                    V_pred_vert = pca.inverse_transform(V_pred.clone().detach().cpu().numpy())
                    V_target = pca.inverse_transform(target_data.clone().detach().cpu().numpy())
                    mm_train_loss += np.mean(np.abs(V_pred_vert - V_target)) # loss in Euclidean space
                    
                    n_pairs += 1
                    
                # Backward pass: older -> younger
                for start_idx in (range(n_timepoints)):
                    V_0 = V_true[start_idx:start_idx+1]
                    ## Data augmentation: add noise
                    noise_std = 0.01  # noise size (adjustable)
                    noise = torch.randn_like(V_0) * noise_std
                    scale_noise = random.uniform(0.98, 1.02)
                    V_0 = (V_0 + noise)*scale_noise
                    
                    target_times = torch.flip(t[:start_idx+1],dims=[0])
                    target_cond = torch.flip(c[:start_idx+1],dims=[0])
                    target_data = (torch.flip(V_true[:start_idx+1],dims=[0]) + noise)*scale_noise
                    V_pred = odeint(lambda t, V: model(t, V, torch.tensor([target_cond[0]]).to(device)), 
                                    V_0, target_times, method='rk4').squeeze(1)
                    # L2 loss in PCA space
                    loss += torch.mean(((V_pred - target_data) ** 2))
                    
                    # Inverse transform to acquire mesh vertices
                    V_pred_vert = pca.inverse_transform(V_pred.clone().detach().cpu().numpy())
                    V_target = pca.inverse_transform(target_data.clone().detach().cpu().numpy())
                    mm_train_loss += np.mean(np.abs(V_pred_vert - V_target)) # loss in Euclidean space

                    n_pairs += 1

                if n_pairs > 0 and i % batch == 0:
                    loss /= n_pairs
                    mm_train_loss /= n_pairs
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    total_mm_train_loss += mm_train_loss.item()
                
            avg_train_loss = total_train_loss / n_train_samples
            avg_mm_train_loss =  total_mm_train_loss / n_train_samples
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Train MM Loss: {avg_mm_train_loss:.4f}")
            

            # Test 
            if epoch % 1 == 0:
                model.eval()
                total_test_loss = 0
                total_mm_loss = 0
                total_avg_test_diff_loss = 0
                with torch.no_grad():
                    for i in range(n_test_samples):
                        V_true = reduced_test_data[i].to(device)
                        V_real_true = test_real_target[i] # without pca reconstruction error
                        t = test_times_list[i].to(device)
                        c = test_cond_list[i].to(device)
                        n_timepoints = len(t)
                        
                        test_loss = 0
                        mm_loss = 0
                        avg_test_diff_loss = 0
                        n_pairs = 0
                        #younger -> older
                        for start_idx in (range(n_timepoints)[:-1]):
                            V_0 = V_true[start_idx:start_idx+1]
                            target_times = t[start_idx:]
                            
                            target_cond = c[start_idx:]
                            target_data = V_true[start_idx:]
                            V_pred = odeint(lambda t, V: model(t, V, torch.tensor([target_cond[0]]).to(device)), 
                                            V_0, target_times, method='rk4').squeeze(1)
                            # L2 loss in PCA space
                            test_loss += torch.mean((V_pred[-1] - target_data[-1]) ** 2)
                            
                            # Inverse transform to acquire mesh vertices
                            V_pred_vert = pca.inverse_transform(V_pred.detach().cpu().numpy())
                            V_target = pca.inverse_transform(target_data.detach().cpu().numpy())
                            # V_target = V_real_true[start_idx:].detach().cpu().numpy()
                            mm_loss += np.mean(np.abs(V_pred_vert[-1] - V_target[-1]))
                            avg_test_diff_loss += np.mean(np.abs(V_target[-1] - V_target[0]))
                            
                            n_pairs += 1

                        if n_pairs > 0:
                            test_loss /= n_pairs
                            total_test_loss += test_loss.item()
                            mm_loss /= n_pairs
                            total_mm_loss += mm_loss.item()
                            avg_test_diff_loss /= n_pairs
                            total_avg_test_diff_loss += avg_test_diff_loss.item()
                
                avg_test_loss = total_test_loss / n_test_samples
                avg_mm_loss = total_mm_loss / n_test_samples
                print(f"Epoch {epoch}, Test Loss: {avg_test_loss:.4f}, Test MM Loss: {avg_mm_loss:.4f}")
                print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
                writer.add_scalar('Loss/Train', avg_train_loss, epoch)
                writer.add_scalar('Loss/Test', avg_test_loss, epoch)
                writer.add_scalar('Loss/mmTest', avg_mm_loss, epoch)
                
            # Visualization on tensorboard
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}")
                # Trainset visualization
                for i in range(5):  
                    V_true = reduced_train_data[i].to(device)
                    t = train_times_list[i].to(device)
                    c = train_cond_list[i].to(device)
                    n_timepoints = len(t)

                    V_pred_list = []
                    t_pred_list = []
                    for start_idx, end_idx in combinations(range(n_timepoints), 2):
                        V_0 = V_true[start_idx:start_idx+1]
                        fine_t = torch.linspace(t[start_idx], t[end_idx], steps=20, device=device)
                        V_pred = odeint(lambda t, V: model(t, V, torch.tensor([c[0]]).to(device)), V_0, fine_t, method='rk4')
                        V_pred_list.append(V_pred.squeeze(1))
                        t_pred_list.append(fine_t)

                    fig = plt.figure(figsize=(10, 6))
                    V_true_1d = V_true[:, 0].detach().cpu().numpy()
                    t_np = t.detach().cpu().numpy()
                    plt.scatter(t_np, V_true_1d, c='red', label='Ground Truth', zorder=5)

                    colors = ['blue', 'green', 'orange', 'purple', 'cyan']
                    for idx, (V_pred, t_pred) in enumerate(zip(V_pred_list, t_pred_list)):
                        V_pred_1d = V_pred[:, 0].detach().cpu().numpy()
                        t_pred_np = t_pred.detach().cpu().numpy()
                        plt.plot(t_pred_np, V_pred_1d, c=colors[idx % len(colors)], 
                                label=f'Pred {idx+1}' if idx < 5 else None)

                    plt.xlabel('Time (t)')
                    plt.ylabel('Value (x)')
                    plt.legend()
                    plt.title(f'Sample {i} Predictions')
                    writer.add_figure(f'Predictions/Sample_{i}', fig, global_step=epoch)
                    plt.close(fig)
                # Testset visualization
                for i in range(5):
                    V_true = reduced_test_data[i].to(device)
                    t = test_times_list[i].to(device)
                    c = test_cond_list[i].to(device)
                    n_timepoints = len(t)

                    V_pred_list = []
                    t_pred_list = []
                    for start_idx, end_idx in combinations(range(n_timepoints), 2):
                        V_0 = V_true[start_idx:start_idx+1]
                        fine_t = torch.linspace(t[start_idx], t[end_idx], steps=20, device=device)
                        V_pred = odeint(lambda t, V: model(t, V, torch.tensor([c[0]]).to(device)), V_0, fine_t, method='rk4')
                        V_pred_list.append(V_pred.squeeze(1))
                        t_pred_list.append(fine_t)

                    fig = plt.figure(figsize=(10, 6))
                    V_true_1d = V_true[:, 0].detach().cpu().numpy()
                    t_np = t.detach().cpu().numpy()
                    plt.scatter(t_np, V_true_1d, c='red', label='Ground Truth', zorder=5)

                    colors = ['blue', 'green', 'orange', 'purple', 'cyan']
                    for idx, (V_pred, t_pred) in enumerate(zip(V_pred_list, t_pred_list)):
                        V_pred_1d = V_pred[:, 0].detach().cpu().numpy()
                        t_pred_np = t_pred.detach().cpu().numpy()
                        plt.plot(t_pred_np, V_pred_1d, c=colors[idx % len(colors)], 
                                label=f'Pred {idx+1}' if idx < 5 else None)

                    plt.xlabel('Time (t)')
                    plt.ylabel('Value (x)')
                    plt.legend()
                    plt.title(f'Sample {i} TEST Predictions')
                    writer.add_figure(f'Predictions/TEST_Sample_{i}', fig, global_step=epoch)
                    plt.close(fig)

            # print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Train MM Loss: {avg_mm_train_loss:.4f}")
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Test', avg_test_loss, epoch)
            writer.add_scalar('Loss/mmTest', avg_mm_loss, epoch)
            writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)
            print(f"Epoch {epoch}, Learning rate: {optimizer.param_groups[0]['lr']}")
            scheduler.step()
            torch.save(model.state_dict(), rf"{args.log_dir}\ckpt\model_epoch_{epoch}_{avg_mm_loss:.4f}.pth")
        
        writer.close()
        return model, pca

    # Run Codes
    model, pca = train_brain_ode(reduced_train_data, train_times_list, train_cond_list,
                                  reduced_test_data, test_times_list, test_cond_list, traindata['data'], testdata['data'],
                                  epochs=args.epoch, lr=args.learning_rate)

if __name__ == '__main__':
    args = parse_args()
    main(args)
