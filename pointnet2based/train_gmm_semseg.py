
import argparse
import os
from data_utils.S3DISDataLoader import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from sklearn import datasets
from sklearn.manifold import TSNE

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,2"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)

    color_temp = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120, 130])
    colors = []
    for item in label:
        colors.append(color_temp[item])
    colors = np.array(colors)
    plt.scatter(data[:, 0], data[:, 1], c=colors, s=6.0)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    current_time = time.strftime("%H_%M_%S")
    t_sne_path = ""
    plt.savefig(os.path.join(t_sne_path,current_time+"_t_sne.png"))
    plt.savefig(os.path.join(t_sne_path,current_time+"_t_sne.pdf"))
    return fig

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='gmmseg_pointnet2_seg_msg', help='model name [default: pointnet2_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--loss_func', type=str, default='hera_embedding', help='Loss Strategy,option: nll_loss/hera_embedding/hera_embedding_label/hera_embedding_label_localfun [default: hera_embedding]')
    parser.add_argument('--step_factor', type=float, default=20., help='step factor,option: int[1,50] [default: 20.]')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)


    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    gpus = [0]
    output_gpu = gpus[0]

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    tensorboard_dir = experiment_dir.joinpath("tensorboard/")
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)



    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    log_string("gpu"+str(gpus))

    root = ''
    NUM_CLASSES = 13
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    log_string("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    log_string("start loading test data ...")
    TEST_DATASET = S3DISDataset(split='test', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                                                  pin_memory=False, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=False, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES,class_weight=weights.tolist())
    classifier = torch.nn.DataParallel(classifier,device_ids=gpus,output_device=output_gpu)
    classifier = classifier.cuda()
    criterion = MODEL.get_loss(NUM_CLASSES,BATCH_SIZE,args.loss_func).cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        try:
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/model.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            log_string('Use pretrain model')
        except:
            log_string('No existing model, starting training from scratch...')
            start_epoch = 0
            classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0


    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        log_string('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()
        step = torch.tensor(1).cuda()
        fix_step = torch.tensor(1500.).cuda()
        embedding_step = 0
        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            losses, trans_feat = classifier(points,target,eval=False)
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            loss = losses['loss_ce'].mean(dim=0) + losses['loss_contrast'].mean(dim=0)
            step = step+1.
            loss.backward()
            optimizer.step()

            writer.add_scalar(f'epoch{epoch}total_loss', loss.item(), step)
            writer.add_scalar(f'epoch{epoch}acc', losses['acc_seg'].mean(dim=0).item(), step)


            if( embedding_step!=0 and embedding_step %1==0):

                embedding_data = trans_feat.permute(0,2,1).contiguous().view(-1,64)
                embedding_data = embedding_data.cpu().data.numpy()
                meta = []
                for item in batch_label:
                    meta.append(str(item))
                writer.add_embedding(
                    embedding_data[:4096*2],
                    metadata=meta[:4096*2],
                    global_step=step.item(),
                    tag = f"epoch{epoch}"
                )

                tsne_data = embedding_data[:4096*2]
                tsne_label = batch_label[:4096*2]
                log_string('Computing t-SNE embedding')
                tsne = TSNE(n_components=2, init='pca', random_state=0)
                t0 = time.time()
                result = tsne.fit_transform(tsne_data)
                log_string(f"time cost : {time.time() - t0} ")
                fig = plot_embedding(result, tsne_label,
                                     't-SNE embedding of the digits (time %.2fs)'
                                     % (time.time() - t0))
                writer.add_figure(
                    f"epoch{epoch}",
                    fig,
                    global_step=step.item(),
                    close=False
                )
            embedding_step += 1

            total_correct +=  losses['acc_seg'].mean(dim=0).item()
            loss_sum += loss.item()

        writer.add_scalar('train mean_loss', loss_sum / num_batches, epoch)
        writer.add_scalar("train accuracy", total_correct / num_batches,epoch)
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / num_batches))

        if epoch % 1 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                losses,seg_pred, trans_feat = classifier(points,target,eval=True)
                pred_val = seg_pred.contiguous().cpu().data.numpy()

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = losses['loss_ce'].mean(dim=0) + losses['loss_contrast'].mean(dim=0)

                if (i!=0 and i % 30 == 0):

                    embedding_data = trans_feat.permute(0, 2, 1).contiguous().view(-1, 64)
                    embedding_data = embedding_data.cpu().data.numpy()

                    batch_label_visual = target.view(-1, 1)[:, 0].cpu().data.numpy()
                    tsne_data = embedding_data[:4096 * 2]
                    tsne_label = batch_label_visual[:4096 * 2]

                    print('Computing t-SNE embedding')
                    tsne = TSNE(n_components=2, init='pca', random_state=0)
                    t0 = time.time()
                    result = tsne.fit_transform(tsne_data)
                    print(f"time cost : {time.time() - t0} ")
                    fig = plot_embedding(result, tsne_label,
                                         't-SNE embedding of the digits (time %.2fs)'
                                         % (time.time() - t0))
                    writer.add_figure(
                        f"evel epoch{epoch}",
                        fig,
                        global_step=step.item(),
                        close=False
                    )
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))

            writer.add_scalar("eval point avg class IoU", mIoU, epoch)
            writer.add_scalar('Eval mean_loss', loss_sum / num_batches, epoch)
            writer.add_scalar("eval point avg class acc", np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6)),  epoch)
            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
            writer.add_scalar('Eval mean_loss', loss_sum / num_batches, epoch)
            writer.add_scalar("Eval accuracy", total_correct / float(total_seen), epoch)
            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
