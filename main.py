import argparse
import torch.nn as nn

from stgcn import STGCN
from stgcn_utils import *
from deepSVDD import DeepSVDD









parser = argparse.ArgumentParser()
parser.add_argument('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
parser.add_argument('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
parser.add_argument('--optimizer_name', type=str, default='adam', help='choose optimizer type')
parser.add_argument('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')

parser.add_argument('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SVDD network training. Default=0.001')
parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr_milestone', type=int, default=0,help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
# parser.add_argument('--batch_size', type=int, default=25, help='Batch size for mini-batch training.')
parser.add_argument('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
parser.add_argument('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
parser.add_argument('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')


parser.add_argument('--task_id', type=int, default=4, help='bAbI task id')
parser.add_argument('--question_id', type=int, default=0, help='question types')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=25, help='input batch size')
parser.add_argument('--state_dim', type=int, default=4, help='GGNN hidden state size')
parser.add_argument('--n_steps', type=int, default=5, help='propogation steps number of GGNN')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda', type=str,default='true', help='enables cuda')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')






parser.add_argument('--embedding_num', type=int, default=35, help='embedding')






args = parser.parse_args()
args.device = None
args.device = torch.device('cuda')


# if opt.manualSeed is None:
#     opt.manualSeed = random.randint(1, 10000)
# print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)
#
# opt.dataroot = 'babi_data/processed_1/train/%d_graphs.txt' % opt.task_id
#
# if opt.cuda:
#     torch.cuda.manual_seed_all(opt.manualSeed)

def main(args):
    min_loss = -1
    save_root = './checkpoints/'
    torch.manual_seed(7)



    X_dir = "utils/total_nodes.npy"
    label_dir = "utils/total_labels.npy"
    timestamp_dir = "utils/timestamps.npy"
    A_wave, X, labels, means, stds, timestamp = load_traffic_data(X_dir, label_dir, timestamp_dir)



    labels=np.where(labels>0,1,0)

    A = np.repeat(A_wave, len(X))
    A = np.reshape(A, (X.shape[0], A_wave.shape[0], A_wave.shape[1]))

    state = np.random.get_state()

    idx = np.arange(0, len(labels), 1)
    np.random.shuffle(idx)

    # np.random.shuffle(X)
    # np.random.set_state(state)
    # np.random.shuffle(labels)

    train_index = int(0.6 * len(idx))
    split_line1 = idx[:train_index]
    split_line2 = idx[train_index:]
    # split_line2 = int(X.shape[0] * 0.3)

    train_original_data = X[split_line1, :, :, :]
    test_original_data = X[split_line2, :, :, :]
    # test_original_data = X[split_line2:,:, :, :]

    train_original_target = labels[split_line1]
    test_original_target = labels[split_line2]
    # test_original_target = labels[split_line2:]

    train_original_A = A[split_line1]
    test_original_A = A[split_line2]
    # test_original_A = A[split_line2:]

    train_original_timestamp = timestamp[split_line1]
    test_original_timestamp = timestamp[split_line2]


    training_input, training_target, training_A = generate_dataset_new(train_original_data, train_original_target,
                                                                       train_original_A)
    test_input, test_target, test_A = generate_dataset_new(test_original_data, test_original_target, test_original_A)
    # test_input, test_target,test_A = generate_dataset_new(test_original_data,test_original_target,test_original_A)



    A_wave = get_normalized_adj(A_wave)
    A_wave = torch.from_numpy(A_wave)
    A_wave = A_wave.float()

    A_wave = A_wave.to(device=args.device)

    cg = {}
    cg['adj'] = A_wave.detach().cpu().numpy()



    previous_precision = 0
    previous_recall = 0
    previous_f1 = 0



    epoch = args.n_epochs
    nu = args.nu
    embedding_num = args.embedding_num



    num_nodes = A_wave.shape[0]
    num_features = training_input.shape[3]
    num_timesteps_input = training_input.shape[2]


    net = STGCN(num_nodes,
                num_features,
                num_timesteps_input,
                embedding_num).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.5e-3)
    loss_criterion = nn.CrossEntropyLoss()

    training_losses = []
    validation_losses = []
    validation_maes = []

    labels_new = []

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(args)
    deep_SVDD.set_network(net)
    device = 'cuda'


    deep_SVDD.train((A_wave, training_input),
                    optimizer_name=args.optimizer_name,
                    lr=args.lr,
                    n_epochs=epoch,
                    nu=nu,
                    lr_milestones=args.lr_milestone,
                    batch_size=args.batch_size,
                    weight_decay=args.weight_decay,
                    device=device)

    # Test model
    y_test_pred, f1_score_test, precision_score_test, recall_score_test,scores = deep_SVDD.test(
        (A_wave, test_input, test_target), device=device)
    print('precision_test:'+str(precision_score_test)+'  '+'recall_score:'+str(recall_score_test)+'  '+'F1-Score:'+str(f1_score_test))


    cg['split_line1'] = split_line1
    cg['deep_SVDD'] = deep_SVDD
    cg['model_state'] = net.state_dict()
    cg['optimizer'] = optimizer
    cg['test_X'] = test_input
    cg['test_label'] = test_target
    cg['test_pred'] = y_test_pred
    cg['test_time'] = test_original_timestamp
    cg['scores']=scores
    torch.save(cg, save_root + 'STGCN_model' + '.pth')


if __name__ == "__main__":
    main(args)

