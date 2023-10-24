# -*- coding: utf-8 -*-
# @Author: xiegr
# @Date:   2020-08-30 20:57:53
# @Last Modified by:   xiegr
# @Last Modified time: 2021-06-01 22:28:57
import pickle
import dpkt
import random
import numpy as np
from preprocess import protocols
from tqdm import tqdm, trange

ip_features = {'hl':1,'tos':1,'len':2,'df':1,'mf':1,'ttl':1,'p':1}
tcp_features = {'off':1,'flags':1,'win':2}
udp_features = {'ulen':2}
max_byte_len = 50			# L值

def mask(p):
	p.src = b'\x00\x00\x00\x00'
	p.dst = b'\x00\x00\x00\x00'
	p.sum = 0
	p.id = 0
	p.offset = 0

	if isinstance(p.data, dpkt.tcp.TCP):
		p.data.sport = 0
		p.data.dport = 0
		p.data.seq = 0
		p.data.ack = 0
		p.data.sum = 0

	elif isinstance(p.data, dpkt.udp.UDP):
		p.data.sport = 0
		p.data.dport = 0
		p.data.sum = 0

	return p

def pkt2feature(data, k):
	flow_dict = {'train':{}, 'test':{}}

	# train->protocol->flowid->[pkts]
	for p in protocols:						# 每个协议
		flow_dict['train'][p] = []
		flow_dict['test'][p] = []
		all_pkts = []						# 存该协议所有v
		p_keys = list(data[p].keys())		# 所有key

		for flow in p_keys:					# 每个key
			pkts = data[p][flow]			# 每个key对应的v
			all_pkts.extend(pkts)
		random.Random(1024).shuffle(all_pkts)

		for idx in range(len(all_pkts)):
			pkt = mask(all_pkts[idx])		# 循环将该协议所有v的ip地址 端口号等信息mask
			raw_byte = pkt.pack()			# 转成byte

			byte = []
			pos = []
			for x in range(min(len(raw_byte),max_byte_len)):
				byte.append(int(raw_byte[x]))	# 将原始byte中每个值的int值放到byte中
				pos.append(x)					# 并将该值对应的索引添加到pos中

			byte.extend([0]*(max_byte_len-len(byte)))		# 到后面不够max_byte_len的全部补0
			pos.extend([0]*(max_byte_len-len(pos)))			# pos也补0
			# if len(byte) != max_byte_len or len(pos) != max_byte_len:
			# 	print(len(byte), len(pos))
			# 	input()
			if idx in range(k*int(len(all_pkts)*0.1), (k+1)*int(len(all_pkts)*0.1)):
				flow_dict['test'][p].append((byte, pos))
			else:
				flow_dict['train'][p].append((byte, pos))
	return flow_dict

def load_epoch_data(flow_dict, train='train'):
	flow_dict = flow_dict[train]
	x, y, label = [], [], []

	for p in protocols:					# 对每种协议
		pkts = flow_dict[p]				# 拿出该协议对应的所有包
		for byte, pos in pkts:			# 遍历所有包
			x.append(byte)				# x是所有byte组成的数组[]
			y.append(pos)				# y是所有pos组成的数组[]
			label.append(protocols.index(p))		# label是所有协议index的数组

	return np.array(x), np.array(y), np.array(label)[:, np.newaxis]			# 将数组转为nparray (1507995, 50)  (1507995, 50)   (1507995, 1)


if __name__ == '__main__':
	# f = open('flows.pkl','rb')
	# data = pickle.load(f)
	# f.close()

	# print(data.keys())

	# dns = data['dns']
	# # print(list(dns.keys())[:10])

	# # wide dataset contains payload
	# print('================\n',
	# 	len(dns['203.206.160.197.202.89.157.51.17.53.51648'][0]))

	# print('================')
	# flow_dict = pkt2feature(data)
	# x, y, label = train_epoch_data(flow_dict)
	# print(x.shape)
	# print(y.shape)
	# print(label[0])
	with open('pro_flows.pkl','rb') as f:
		data = pickle.load(f)
	# 设置不同1/10范围内的数据作为test集，其他数据作为train集 主要是为了交叉验证  每个数据的格式为(byte, pos)
	for i in trange(10, mininterval=2, \
		desc='  - (Building fold dataset)   ', leave=False):
		flow_dict = pkt2feature(data, i)
		with open('pro_flows_%d_noip_fold.pkl'%i, 'wb') as f:
			pickle.dump(flow_dict, f)