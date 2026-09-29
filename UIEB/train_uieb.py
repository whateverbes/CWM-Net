import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import Config.dataset as dataset
from Config.vgg import *
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from Config.options import opt, device
from Config.models import *
from Config.misc import *
import re
import sys
import json
from pathlib import Path
from pytorch_msssim import ssim

try:
	from torchsummary import summary
except Exception:
	summary = None

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']


if __name__ == '__main__':

	print('[AuthorTrain] hazydir=', opt.hazydir)
	print('[AuthorTrain] cleandir=', opt.cleandir)
	print('[AuthorTrain] ckpts=', opt.checkpoints_dir)
	print('[AuthorTrain] lr={} epochs={} bs={} n={} eval_every={} early_stop={}'.format(
		opt.learning_rate_g, opt.end_epoch, opt.batch_size, opt.num_images,
		opt.eval_every, opt.early_stop_patience))

	_probe = os.path.join(opt.hazydir, 'img1' + opt.img_extension)
	if not os.path.isfile(_probe):
		raise FileNotFoundError(
			'Missing {}. Prepare data first:\n'
			'  python ../prepare_data.py --uieb_root ../Data/UIEB_raw --force'.format(_probe)
		)

	batches = int(opt.num_images / opt.batch_size)
		
	if opt.shared_init_seed >= 0:
		torch.manual_seed(opt.shared_init_seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(opt.shared_init_seed)
		print('[AuthorTrain] shared_init_seed={} (reproducible scratch init)'.format(
			opt.shared_init_seed))

	dicam = DICAM()
	dicam.to(device)
	if summary is not None:
		try:
			summary(dicam,input_size=(3,256,256))
		except Exception as e:
			print('[Warn] torchsummary skipped:', e)

	mae_loss = nn.L1Loss()

	vgg = Vgg19(requires_grad=False).to(device)
	vgg.eval()

	optim_g = optim.Adam(dicam.parameters(), 
						 lr=opt.learning_rate_g, 
						 betas = (opt.beta1, opt.beta2), 
						 weight_decay=opt.wd_g)

		
	dataset = dataset.Dataset_Load(hazy_path=opt.hazydir, 
								   clean_path=opt.cleandir, 
								   transform=dataset.ToTensor()
								   )

	# Author original: num_workers=5, pin_memory=1
	dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=5, pin_memory=True, shuffle=True)
	
	batches = int(opt.num_images / opt.batch_size)

	if not os.path.exists(opt.checkpoints_dir):
		os.makedirs(opt.checkpoints_dir)
	
	models_loaded = getLatestCheckpointName()    
	latest_checkpoint = models_loaded
	
	print('loading model for DICAM ', latest_checkpoint)
	
	if latest_checkpoint == None :
		start_epoch = 1
		print('No checkpoints found for DICAM retraining')
	
	else:
		checkpoint_g = torch.load(os.path.join(opt.checkpoints_dir, latest_checkpoint))    
		start_epoch = checkpoint_g['epoch'] + 1
		dicam.load_state_dict(checkpoint_g['model_state_dict'])
		optim_g.load_state_dict(checkpoint_g['optimizer_state_dict'])
			
		print('Restoring model from checkpoint ' + str(start_epoch))
	
	_ROOT = Path(__file__).resolve().parents[1]
	eval_log_path = Path(opt.checkpoints_dir) / 'train_eval_log.json'
	best_state = {'psnr': -1.0, 'epoch': None, 'ssim': None}
	last_best_epoch = start_epoch - 1

	dicam.train()

	for epoch in range(start_epoch, opt.end_epoch + 1):

		opt.total_mae_loss = 0.0
		opt.total_vgg_loss = 0.0
		opt.total_loss = 0.0
		opt.total_ssim_loss = 0.0
	
		for i_batch, sample_batched in enumerate(dataloader):

			hazy_batch = sample_batched['hazy']
			clean_batch = sample_batched['clean']

			hazy_batch = hazy_batch.to(device)
			clean_batch = clean_batch.to(device)

			
			pred_batch = dicam(hazy_batch)

			batch_mae_loss = torch.mul(opt.lambda_mae, mae_loss(pred_batch, clean_batch))
			batch_mae_loss.backward(retain_graph=True)
			
			batch_ssim_loss = 1-ssim(pred_batch, clean_batch,data_range=1, size_average=True)
			batch_ssim_loss.backward(retain_graph=True)

			clean_vgg_feats = vgg(normalize_batch(clean_batch))
			pred_vgg_feats = vgg(normalize_batch(pred_batch))
			batch_vgg_loss = torch.mul(opt.lambda_vgg, mae_loss(pred_vgg_feats.relu4_3, clean_vgg_feats.relu4_3))
			batch_vgg_loss.backward()
			
			opt.batch_mae_loss = batch_mae_loss.detach().cpu().item()
			opt.total_mae_loss += opt.batch_mae_loss

			opt.batch_ssim_loss = batch_ssim_loss.detach().cpu().item()
			opt.total_ssim_loss += opt.batch_ssim_loss

			opt.batch_vgg_loss = batch_vgg_loss.detach().cpu().item()
			opt.total_vgg_loss += opt.batch_vgg_loss
			
			opt.batch_loss = opt.batch_mae_loss + opt.batch_ssim_loss + opt.batch_vgg_loss
			opt.total_loss += opt.batch_loss
			
			optim_g.step()
			optim_g.zero_grad() 

			print('\r Epoch : ' + str(epoch) + ' | (' + str(i_batch+1) + '/' + str(batches) + ') | l_mae: ' + str(opt.batch_mae_loss/2) + ' | l_ssim: ' + str(1-opt.batch_ssim_loss)+ ' | l_vgg: ' + str(opt.batch_vgg_loss), end='', flush=True)

 

		print('\n\nFinished ep. %d, lr = %.6f, mean_mae = %.6f, mean_ssim = %.6f, mean_vgg = %.6f' % (epoch, get_lr(optim_g), (opt.total_mae_loss / batches)/2,1- (opt.total_ssim_loss / batches), opt.total_vgg_loss / batches))

		ckpt_path = os.path.abspath(os.path.join(opt.checkpoints_dir, 'DICAM_' + str(epoch) + '.pt'))
		payload = {'epoch':epoch,
					'model_state_dict':dicam.state_dict(),
					'optimizer_state_dict':optim_g.state_dict(),
					'mae_loss':opt.total_mae_loss,
					'ssim_loss':opt.total_ssim_loss,
					'vgg_loss':opt.total_vgg_loss,
					'opt':opt,
					'total_loss':opt.total_loss}
		torch.save(payload, ckpt_path)

		do_eval = opt.eval_every > 0 and (epoch % opt.eval_every == 0 or epoch == opt.end_epoch)
		if do_eval:
			sys.path.insert(0, str(_ROOT))
			from eval_paper_protocol import eval_checkpoint
			data_dir = str((_ROOT / 'Data' / 'UIEB').resolve())
			metrics = eval_checkpoint(ckpt_path, data_dir, 'dicam', 'test')
			test_p = metrics['test']['psnr_paper']
			test_s = metrics['test']['ssim_paper']
			print('[Eval] ep {:02d} test_psnr_paper={:.4f} ssim_paper={:.4f}'.format(epoch, test_p, test_s))
			log = []
			if eval_log_path.is_file():
				try:
					log = json.loads(eval_log_path.read_text(encoding='utf-8')).get('log', [])
				except Exception:
					log = []
			log.append({'epoch': epoch, 'test_psnr_paper': test_p, 'test_ssim_paper': test_s})
			if test_p > best_state['psnr']:
				best_state['psnr'] = test_p
				best_state['epoch'] = epoch
				best_state['ssim'] = test_s
				best_path = Path(opt.checkpoints_dir).resolve() / 'DICAM_best_test.pt'
				torch.save(payload, str(best_path))
				print('[Best test] {:.4f} (ep {}) -> {}'.format(test_p, epoch, best_path))
			eval_log_path.write_text(json.dumps({
				'best_test_paper': best_state['psnr'],
				'best_epoch': best_state.get('epoch'),
				'best_ssim_paper': best_state.get('ssim'),
				'log': log,
			}, indent=2), encoding='utf-8')
			if best_state['epoch'] is not None:
				last_best_epoch = int(best_state['epoch'])
			if (opt.early_stop_patience > 0 and best_state['epoch'] is not None
					and epoch - last_best_epoch >= opt.early_stop_patience):
				print('\n[EarlyStop] {} epochs since last best (best ep {}, current ep {}).'.format(
					epoch - last_best_epoch, last_best_epoch, epoch))
				break

	if best_state.get('epoch') is not None:
		print('\n[Done] best test psnr_paper={:.4f} (epoch {}) | {}'.format(
			best_state['psnr'], best_state['epoch'],
			Path(opt.checkpoints_dir).resolve() / 'DICAM_best_test.pt'))
	else:
		print('\n[Done] see DICAM_{}.pt (eval_every=0 during this run)'.format(opt.end_epoch))