#coding=utf-8
from ast import Num
from sys import executable
from numpy.core.fromnumeric import mean
from numpy.core.numeric import identity
from pandas.core.frame import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
from pandas import concat
import csv,os
import time
import random

# 标签0：文本
# 标签1：图片
# 标签2：朋友圈
# 标签3：视频通话
# 标签4：红包
# 标签5：发送位置
# 标签6：发送视频
# 标签7：发送语音
# 标签8：telegram文本
# 标签9：telegram图片
# 标签10：telegram语音
# 标签11：telegram视频
# 标签12：telegram发送文件
# 标签13：语音通话
# 标签14：读公众号
# 标签15：转账
# 标签16：打开小程序

# ================数据输入预处理=============================================
#------------------Wechat日志载入---------------
# df = pd.read_table(r"C:\Users\wtr\Desktop\抓包数据\文本齐（正确5000）\wenben.log",usecols= [0],sep = " ",header=None)
# pic=pd.read_table(r"C:\Users\wtr\Desktop\抓包数据\图片齐（正确2000）\picture.log",usecols= [0],sep = " ",header=None)
# mom=pd.read_table(r"C:\Users\wtr\Desktop\抓包数据\朋友圈（正确2000）\moments(1).log",usecols= [0],sep = " ",header=None)
# vcall=pd.read_table(r"C:\Users\wtr\Desktop\抓包数据\视频通话（正确1000）\sendvcall第一次500.log",usecols= [0],sep = " ",header=None)
# Red=pd.read_table(r"C:\Users\wtr\Desktop\抓包数据\发红包\sendRedPacket总.log",usecols= [0],sep = " ",header=None)
# loc=pd.read_table(r"C:\Users\wtr\Desktop\抓包数据\发位置\sendLocation总结.log",usecols= [0],sep = " ",header=None)
# vid=pd.read_table(r"C:\Users\wtr\Desktop\抓包数据\视频（正确1500）\video1.log",usecols= [0],sep = " ",header=None)
# acall=pd.read_table(r"C:\Users\wtr\Desktop\抓包数据\语音通话\sendAcall.log",usecols= [0],sep = " ",header=None)
# readnews=pd.read_table(r"C:\Users\wtr\Desktop\抓包数据\读公众号\ReadNews总.log",usecols= [0],sep = " ",header=None)
# transfer=pd.read_table(r"C:\Users\wtr\Desktop\抓包数据\发送转账\sendTransfer总.log",usecols= [0],sep = " ",header=None)
# opmin=pd.read_table(r"D:\Telegram和Whatsapp\小程序\PCAPdroid_22_7月_01_51_57\openminipro总.log",usecols= [0],sep = " ",header=None)
# aud=pd.read_table(r"C:\Users\wtr\Desktop\抓包数据\语音\audio.log",usecols= [0],sep = " ",header=None)
#------------------Telegram日志载入---------------
# Ttext=pd.read_table(r"D:\Telegram和Whatsapp\telegram数据\telegram\文本\message.log",usecols= [0],sep = " ",header=None)
# Tpic=pd.read_table(r"D:\Telegram和Whatsapp\telegram数据\telegram\图片\picture总.log",usecols= [0],sep = " ",header=None)
# Taud=pd.read_table(r"D:\Telegram和Whatsapp\telegram数据\telegram\音频\audio总.log",usecols= [0],sep = " ",header=None)
# Tvid=pd.read_table(r"D:\Telegram和Whatsapp\telegram数据\telegram\视频\video总.log",usecols= [0],sep = " ",header=None)
# Tfile=pd.read_table(r"D:\Telegram和Whatsapp\telegram数据\telegram\文件\file总.log",usecols= [0],sep = " ",header=None)
#-------------------------------------------
#给日志文件(关键动作的时间记录)打上小数点
# def addpoint(x):
    # return x*0.001
# df = df.apply(lambda x: addpoint(x))
# pic= pic.apply(lambda x: addpoint(x))
# mom= mom.apply(lambda x: addpoint(x))
# vcall= vcall.apply(lambda x: addpoint(x))
# Red=Red.apply(lambda x: addpoint(x))
# loc= loc.apply(lambda x: addpoint(x))
# vid= vid.apply(lambda x: addpoint(x))
# aud= aud.apply(lambda x: addpoint(x))
# acall= acall.apply(lambda x: addpoint(x))
# readnews= readnews.apply(lambda x: addpoint(x))
# transfer= transfer.apply(lambda x: addpoint(x))
# opmin= opmin.apply(lambda x: addpoint(x))

# Ttext= Ttext.apply(lambda x: addpoint(x))
# Tpic= Tpic.apply(lambda x: addpoint(x))
# Taud= Taud.apply(lambda x: addpoint(x))
# Tvid= Tvid.apply(lambda x: addpoint(x))
# Tfile= Tfile.apply(lambda x: addpoint(x))
#---------------Telegram数据载入----------------------------
# Tt1 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\文本\文本1.csv', encoding='gbk')#读取第一个文件
# Tp1 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\图片\图片1.csv', encoding='gbk')#读取第二个文件
# Tp2 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\图片\图片2.csv', encoding='gbk')#读取第三个文件
# Tp3 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\图片\图片3.csv', encoding='gbk')#读取第三个文件
# Tp4 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\图片\图片4.csv', encoding='gbk')#读取第三个文件
# # Tp5 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\图片\图片5.csv', encoding='gbk')#读取第三个文件
# Ta1 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\音频\音频1.csv', encoding='gbk')
# Ta2 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\音频\音频2.csv', encoding='gbk')
# Ta3 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\音频\音频3.csv', encoding='gbk')
# Ta4 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\音频\音频4.csv', encoding='gbk')
# Ta5 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\音频\音频5.csv', encoding='gbk')
# Tv1 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\视频\视频1.csv', encoding='gbk')
# Tv2 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\视频\视频2.csv', encoding='gbk')
# Tv3 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\视频\视频3.csv', encoding='gbk')
# Tv4 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\视频\视频4.csv', encoding='gbk')
# Tv5 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\视频\视频5.csv', encoding='gbk')
# Tf1 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\文件\文件1.csv', encoding='gbk')
# Tf2 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\文件\文件2.csv', encoding='gbk')
# Tf3 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\文件\文件3.csv', encoding='gbk')
# Tf4 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\文件\文件4.csv', encoding='gbk')
# Tf5 = pd.read_csv(r'D:\Telegram和Whatsapp\telegram数据\telegram\文件\文件5.csv', encoding='gbk')
# #------------------------微信数据载入---------------------------
# df1 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\文本齐（正确5000）\第一次.csv', encoding='gbk')#读取第一个文件
# df2 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\文本齐（正确5000）\第二次.csv', encoding='gbk')#读取第二个文件
# df3 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\文本齐（正确5000）\第三次.csv', encoding='gbk')#读取第三个文件

# pic1 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\图片齐（正确2000）\图片1.csv', encoding='gbk')
# pic2 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\图片齐（正确2000）\图片2.csv', encoding='gbk')
# pic3 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\图片齐（正确2000）\图片3.csv', encoding='gbk')
# pic4 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\图片齐（正确2000）\图片4.csv', encoding='gbk')

# mom1 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\朋友圈（正确2000）\朋友圈1.csv', encoding='gbk')
# mom2 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\朋友圈（正确2000）\朋友圈2.csv', encoding='gbk')
# mom3 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\朋友圈（正确2000）\朋友圈3.csv', encoding='gbk')
# mom4 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\朋友圈（正确2000）\朋友圈4.csv', encoding='gbk')

# vcall1 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\视频通话（正确1000）\视频通话1.csv', encoding='gbk')
# vcall2 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\视频通话（正确1000）\视频通话2.csv', encoding='gbk')

# Red1 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发红包\红包1.csv', encoding='gbk')
# Red2 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发红包\红包2.csv', encoding='gbk')
# Red3 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发红包\红包3.csv', encoding='gbk')
# Red4 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发红包\红包4.csv', encoding='gbk')
# Red5 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发红包\红包5.csv', encoding='gbk')
# Red6 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发红包\红包6.csv', encoding='gbk')
# Red7 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发红包\红包7.csv', encoding='gbk')
# Red8 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发红包\红包8.csv', encoding='gbk')



# vid1 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\视频（正确1500）\发送视频第一组.csv', encoding='gbk')


# loc1 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发位置\位置1.csv', encoding='gbk')
# loc2 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发位置\位置2.csv', encoding='gbk')
# loc3 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发位置\位置3.csv', encoding='gbk')
# loc4 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发位置\位置4.csv', encoding='gbk')
# loc5 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发位置\位置5.csv', encoding='gbk')
# loc6 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发位置\位置6.csv', encoding='gbk')

# aud1 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\语音\语音第一组.csv', encoding='gbk')
# aud2 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\语音\语音第二组.csv', encoding='gbk')

# acall1= pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\语音通话\语音通话第一组.csv', encoding='gbk')

# readnews1= pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\读公众号\公众号第一组.csv', encoding='gbk')
# readnews2= pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\读公众号\公众号第二组.csv', encoding='gbk')

# transfer1= pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发送转账\发送转账第一组.csv', encoding='gbk')
# transfer2= pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发送转账\发送转账第二组.csv', encoding='gbk')
# transfer3= pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发送转账\发送转账第三组.csv', encoding='gbk')
# transfer4= pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发送转账\发送转账第四组.csv', encoding='gbk')
# transfer5= pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发送转账\发送转账第五组.csv', encoding='gbk')

# opmin1= pd.read_csv(r'D:\Telegram和Whatsapp\小程序\PCAPdroid_22_7月_01_51_57\打开小程序第一组.csv', encoding='gbk')
# opmin2= pd.read_csv(r'D:\Telegram和Whatsapp\小程序\PCAPdroid_22_7月_01_51_57\打开小程序第二组.csv', encoding='gbk')

#-------------------Telegram-------------------------------
# Tte1=Ttext
# Tte2 = Tt1
# Tpi1=Tpic
# Tpi2 = pd.merge(Tp1, Tp2,how='outer')
# Tpi2 = pd.merge(Tpi2, Tp3,how='outer')
# Tpi2 = pd.merge(Tpi2, Tp4,how='outer')
# Tpi2 = pd.merge(Tpi2, Tp5,how='outer')
# Tau1=Taud
# Tau2 = pd.merge(Ta1, Ta2,how='outer')
# Tau2 = pd.merge(Tau2, Ta3,how='outer')
# Tau2 = pd.merge(Tau2, Ta4,how='outer')
# Tau2 = pd.merge(Tau2, Ta5,how='outer')
# Tvi1=Tvid
# Tvi2 = pd.merge(Tv1, Tv2,how='outer')
# Tvi2 = pd.merge(Tvi2, Tv3,how='outer')
# Tvi2 = pd.merge(Tvi2, Tv4,how='outer')
# Tvi2 = pd.merge(Tvi2, Tv5,how='outer')
# Tfi1=Tfile
# Tfi2 = pd.merge(Tf1, Tf2,how='outer')
# Tfi2 = pd.merge(Tfi2, Tf3,how='outer')
# Tfi2 = pd.merge(Tfi2, Tf4,how='outer')
# Tfi2 = pd.merge(Tfi2, Tf5,how='outer')

#----------------------------------------------------------
# file1=df
# file2 = pd.merge(df1, df2,how='outer')
# file2 = pd.merge(file2, df3,how='outer')
# file2.to_csv(r'C:\Users\wtr\Desktop\抓包数据\文本齐（正确5000）\合并后的文本\总文本.csv')

# picc1=pic
# picc2 = pd.merge(pic1, pic2,how='outer')
# picc2 = pd.merge(picc2, pic3,how='outer')
# picc2 = pd.merge(picc2, pic4,how='outer')
# picc2.to_csv(r'C:\Users\wtr\Desktop\抓包数据\图片齐（正确2000）\合并后的图片\总图片.csv')

# momm1=mom
# momm2 = pd.merge(mom1, mom2,how='outer')
# momm2 = pd.merge(momm2, mom3,how='outer')
# momm2 = pd.merge(momm2, mom4,how='outer')
# momm2.to_csv(r'C:\Users\wtr\Desktop\抓包数据\朋友圈（正确2000）\合并后的朋友圈\总朋友圈.csv')

# vcalll1=vcall
# vcalll2 = pd.merge(vcall1, vcall2,how='outer')
# vcalll2.to_csv(r'C:\Users\wtr\Desktop\抓包数据\视频通话（正确1000）\合并后的视频通话\总视频通话.csv')


# RP1=Red
# RP2 = pd.merge(Red1, Red2,how='outer')
# RP2 = pd.merge(RP2, Red3,how='outer')
# RP2 = pd.merge(RP2, Red4,how='outer')
# RP2 = pd.merge(RP2, Red5,how='outer')
# RP2 = pd.merge(RP2, Red6,how='outer')
# RP2 = pd.merge(RP2, Red7,how='outer')
# RP2 = pd.merge(RP2, Red8,how='outer')


# video1=vid
# video2 = vid1



# location1=loc
# location2 = pd.merge(loc1, loc2,how='outer')
# location2 = pd.merge(location2, loc3,how='outer')
# location2 = pd.merge(location2, loc4,how='outer')
# location2 = pd.merge(location2, loc5,how='outer')
# location2 = pd.merge(location2, loc6,how='outer')
# location2.to_csv(r'C:\Users\wtr\Desktop\抓包数据\发位置\合并后的位置\总位置.csv')

# audio1=aud
# audio2 = pd.merge(aud1, aud2,how='outer')
# audio2.to_csv(r'C:\Users\wtr\Desktop\抓包数据\语音\合并后的语音\总语音.csv')

# aca1=acall
# aca2 = acall1

# rn1=readnews
# rn2 = pd.merge(readnews1, readnews2,how='outer')

# trf1 = transfer
# trf2 = pd.merge(transfer1, transfer2,how='outer')
# trf2 = pd.merge(trf2, transfer3,how='outer')
# trf2 = pd.merge(trf2, transfer4,how='outer')
# trf2 = pd.merge(trf2, transfer5,how='outer')

# opm1 = opmin
# opm2 = pd.merge(opmin1, opmin2,how='outer')



# # _______________________________________________________________________________________________________________
# # 以上是数据读写，以下是处理完后的数据
# # _______________________________________________________________________________________________________________
#--------------------Telegram-------------------------------------------
# Tte1=Ttext
# Tte1.columns=['Time']
# Tte3=Tte2.loc[:,['Time']]

# Tpi1=Tpic
# Tpi1.columns=['Time']
# Tpi3=Tpi2.loc[:,['Time']]

# Tau1=Taud
# Tau1.columns=['Time']
# Tau3=Tau2.loc[:,['Time']]

# Tvi1=Tvid
# Tvi1.columns=['Time']
# Tvi3=Tvi2.loc[:,['Time']]

# Tfi1=Tfile
# Tfi1.columns=['Time']
# Tfi3=Tfi2.loc[:,['Time']]

#-----------------------------------------------------------------------
# # loc2 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发位置\位置2.csv', encoding='gbk')
# file1.columns=['Time']
# file3=file2.loc[:,['Time']]
# picc2 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\图片齐（正确2000）\合并后的图片\总图片.csv', encoding='gbk')
# picc1=pic
# picc1.columns=['Time']
# picc3=picc2.loc[:,['Time']]
# momm2 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\朋友圈（正确2000）\合并后的朋友圈\总朋友圈.csv', encoding='gbk')
# momm1=mom
# momm1.columns=['Time']
# momm3=momm2.loc[:,['Time']]

# vcalll2 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\视频通话（正确1000）\合并后的视频通话\总视频通话.csv', encoding='gbk')
# vcalll1=vcall
# vcalll1.columns=['Time']
# vcalll3=vcalll2.loc[:,['Time']]
# # loc2 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发位置\位置2.csv', encoding='gbk')

# RP1.columns=['Time']
# RP3=RP2.loc[:,['Time']]

# video1.columns=['Time']
# video3=video2.loc[:,['Time']]

# location2 = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\发位置\合并后的位置\总位置.csv', encoding='gbk')
# location1=loc
# location1.columns=['Time']
# location3=location2.loc[:,['Time']]

# audio1=aud
# audio1.columns=['Time']
# audio3=audio2.loc[:,['Time']]

# aca1.columns=['Time']
# aca3=aca2.loc[:,['Time']]

# rn1.columns=['Time']
# rn3=rn2.loc[:,['Time']]

# trf1.columns=['Time']
# trf3=trf2.loc[:,['Time']]

# opm1.columns=['Time']
# opm3=opm2.loc[:,['Time']]

#---------------Telegram---------------------------------------
# TT =Tte3['Time'].tolist()
# TTT =Tte1['Time'].tolist()

# PP =Tpi3['Time'].tolist()
# PPP =Tpi1['Time'].tolist()

# AA =Tau3['Time'].tolist()
# AAA =Tau1['Time'].tolist()

# VV =Tvi3['Time'].tolist()
# VVV =Tvi1['Time'].tolist()

# FF =Tfi3['Time'].tolist()
# FFF =Tfi1['Time'].tolist()

# length15 = len(TTT)
# length16 = len(TT)

# length17 = len(PPP)
# length18 = len(PP)

# length19 = len(AAA)
# length20 = len(AA)

# length21 = len(VVV)
# length22 = len(VV)

# length23 = len(FFF)
# length24 = len(FF)
#--------------------------------------------------------------
# a =file3['Time'].tolist()
# b =file1['Time'].tolist()

# c =picc3['Time'].tolist()
# d =picc1['Time'].tolist()

# e =momm3['Time'].tolist()
# f =momm1['Time'].tolist()

# g =vcalll3['Time'].tolist()
# h =vcalll1['Time'].tolist()

# k =RP3['Time'].tolist()
# l =RP1['Time'].tolist()

# k =video3['Time'].tolist()
# l =video1['Time'].tolist()

# mmm =location3['Time'].tolist()
# nnn =location1['Time'].tolist()

# ooo = audio3['Time'].tolist()
# ppp = audio1['Time'].tolist()

# qqq =video3['Time'].tolist()
# rrr =video1['Time'].tolist()

# sss = aca3['Time'].tolist()
# ttt = aca1['Time'].tolist()

# uuu =rn3['Time'].tolist()
# vvv =rn1['Time'].tolist()

# www = trf3['Time'].tolist()
# xxx = trf1['Time'].tolist()

# yyy = opm3['Time'].tolist()
# zzz = opm1['Time'].tolist()




# length1 = len(b)
# length2 = len(a)

# length3 = len(d)
# length4 = len(c)

# length5 = len(f)
# length6 = len(e)

# length7 = len(h)
# length8 = len(g)

# length9 = len(l)
# length10 = len(k)

# length11 = len(nnn)
# length12 = len(mmm)

# length13 = len(ppp)
# length14 = len(ooo)

# length25 = len(rrr)
# length26 = len(qqq)

# length27 = len(ttt)
# length28 = len(sss)

# length29 = len(vvv)
# length30 = len(uuu)

# length31 = len(xxx)
# length32 = len(www)

# length33 = len(zzz)
# length34 = len(yyy)
#-----------------Telegram----------------------------------------
# index7=[]
# for i in range(0, length15):
#     for j in range(0, length16):
#         if TT[j] >= TTT[i]:
#             if(i==0):
#                 index7.append(j)
#                 break    
#             else:
#                 index7.append(j)
#                 index7.append(j+1)
#                 break
# index7.append(length16)
# print(index7)
# kkk=[]
# for i,j in enumerate(index7):
#     if(index7[i]>index7[i-1]):
#         for index ,k in enumerate(index7):
#             if(k>j):
#                 # index7.remove[j:k-1]
#                 kkk.append()
#                 # del index7[i-1:index]
#                 # print(index7)
#                 break
# print(index7)
# index8=[]
# for i in range(0, length17):
#     for j in range(0, length18):
#         if PP[j] >= PPP[i]:
#             if(i==0):
#                 index8.append(j)
#             else:
#                 index8.append(j)
#                 index8.append(j+1)
#             break
# index8.append(length18)

# index9=[]
# for i in range(0, length19):
#     for j in range(0, length20):
#         if AA[j] >= AAA[i]:
#             if(i==0):
#                 index9.append(j)
#             else:
#                 index9.append(j)
#                 index9.append(j+1)
#             break
# index9.append(length20)

# index10=[]
# for i in range(0, length21):
#     for j in range(0, length22):
#         if VV[j] >= VVV[i]:
#             if(i==0):
#                 index10.append(j)
#             else:
#                 index10.append(j)
#                 index10.append(j+1)
#             break
# index10.append(length22)

# index11=[]
# for i in range(0, length23):
#     for j in range(0, length24):
#         if FF[j] >= FFF[i]:
#             if(i==0):
#                 index11.append(j)
#             else:
#                 index11.append(j)
#                 index11.append(j+1)
#             break
# index11.append(length24)

#-----------------------------------------------------------------

# index=[]
# for i in range(0, length1):
#     for j in range(0, length2):
#         if a[j] >= b[i]:
#             if(i==0):
#                 index.append(j)
#             else:
#                 index.append(j)
#                 index.append(j+1)
#             break
# index.append(length2)
# # print(index)

# index1=[]
# for i in range(0, length3):
#     for j in range(0, length4):
#         if c[j] >= d[i]:
#             if(i==0):
#                 index1.append(j)
#             else:
#                 index1.append(j)
#                 index1.append(j+1)
#             break
# index1.append(length4)

# index2=[]
# for i in range(0, length5):
#     for j in range(0, length6):
#         if e[j] >= f[i]:
#             if(i==0):
#                 index2.append(j)
#             else:
#                 index2.append(j)
#                 index2.append(j+1)
#             break
# index2.append(length6)

# index3=[]
# for i in range(0, length7):
#     for j in range(0, length8):
#         if g[j] >= h[i]:
#             if(i==0):
#                 index3.append(j)
#             else:
#                 index3.append(j)
#                 index3.append(j+1)
#             break
# index3.append(length8)

# index4=[]
# for i in range(0, length9):
#     for j in range(0, length10):
#         if k[j] >= l[i]:
#             index4.append(j)
#             break
# index4.append(length10)
# print(index1)

# # print(np.array(index).shape)

# index5=[]
# for i in range(0, length11):
#     for j in range(0, length12):
#         if mmm[j] >= nnn[i]:
#             if(i==0):
#                 index5.append(j)
#             else:
#                 index5.append(j)
#                 index5.append(j+1)
#             break
# index5.append(length12)
# print("ttttt",ttt)
# index5=[]
# for i in range(0, length25):
#     for j in range(0, length26):
#         if qqq[j] >= rrr[i]:
#             if(i==0):
#                 index5.append(j)
#             else:
#                 index5.append(j)
#                 index5.append(j+1)
#             break
# index5.append(length26)
# # print(index5)
# # sss = aca3['Time'].tolist()
# # ttt = aca1['Time'].tolist()
# # print(sss)
# # print("t:",ttt)
# index12=[]
# for i in range(0, length27):
#     for j in range(0, length28):
#         if sss[j] >= ttt[i]:
#             if(i==0):
#                 index12.append(j)
#             else:
#                 index12.append(j)
#                 index12.append(j+1)
#             break
# index12.append(length28)


# index13=[]
# for i in range(0, length29):
#     for j in range(0, length30):
#         if uuu[j] >= vvv[i]:
#             if(i==0):
#                 index13.append(j)
#             else:
#                 index13.append(j)
#                 index13.append(j+1)
#             break
# index13.append(length30)

# index14=[]
# for i in range(0, length31):
#     for j in range(0, length32):
#         if www[j] >= xxx[i]:
#             if(i==0):
#                 index14.append(j)
#             else:
#                 index14.append(j)
#                 index14.append(j+1)
#             break
# index14.append(length32)

# index15=[]
# for i in range(0, length33):
#     for j in range(0, length34):
#         if yyy[j] >= zzz[i]:
#             if(i==0):
#                 index15.append(j)
#             else:
#                 index15.append(j)
#                 index15.append(j+1)
#             break
# index15.append(length34)
# # print(index12)
# def test1(x):
#     onum=[]
#     jnum=[]
#     for i,j in enumerate(x):
#         if i%2==0:
#             onum.append(j)
#         else:
#             jnum.append(j)
#     # return zip(onum,jnum)
#     list0=[]
#     for i,j in zip(onum,jnum):
#         if(i<=j):
#             list0.append([i,j]) 
#     return list0
# for i in range(1,len(index)):
#     if index[i]<index[i-1]:
#         print(i)
# segment1=test1(index)
# print(segment1)
# segment2=test1(index1)
# segment3=test1(index2)
# segment4=test1(index3)
# segment6=test1(index5)
# segment7=test1(index6)
# segment5=test1(index4)
# segment8=test1(index7)
# segment13=test1(index12)
# segment14=test1(index13)
# segment15=test1(index14)
# segment16=test1(index15)
# print(segment13)
# print(segment8)
# print(segment8)
# print(segment8)
# print(np.array(segment8).shape)
# print(segment8)
# ---------------------------------------

# --------------------------------------------
# segment9=test1(index8)
# segment10=test1(index9)
# segment11=test1(index10)
# segment12=test1(index11)

# end1=time.time()
# print('数据读写和生成段的Running time: %s Seconds'%(end1-start1))
#----------------Telegram--------------------------
# start2=time.time()

# Num8=[]
# Cumsize8=[]
# # 双向数据包的数量
# for i,j in segment8:
#     Itv=j-i+1
#     Num8.append(Itv)
# # print(Num4)
# print(np.array(Num8).shape)
# Num8=np.array(Num8)
# # print(Num8)

# fileLength8=Tte2.loc[:,['Length']]
# L8=fileLength8['Length'].tolist()
# # print(L8)
# # for i,j in segment8:
# #     if(len(L8[i:j])==0):
# #         print('i=',i)
# #         print('j=',j)
# # print(L8)
# # print(LL)

# # 累计大小
# for i,j in segment8:
#     cu=sum(L8[i-1:j])
#     Cumsize8.append(cu)
# Cumsize8=np.array(Cumsize8)
# # print(Cumsize8)

# #特征2:最小值，最大值，均值，标准差
# # 最小值
# minimum8=[]
# for i,j in segment8:
#     if(i==j):
#         mm=L8[i-1]
#         minimum8.append(mm)
#     else:
#         mm=min(L8[i-1:j])
#         minimum8.append(mm)
# minimum8=np.array(minimum8)


# # 最大值
# maximum8=[]
# for i,j in segment8:
#     if(i==j):
#         ma=L8[i-1]
#         maximum8.append(ma)
#     else:
#         ma=max(L8[i-1:j])
#         maximum8.append(ma)
# maximum8=np.array(maximum8)
# # print(maximum)

# # 均值
# Mean8=[]
# for i,j in segment8:
#     if(i==j):
#         me=L8[i-1]
#         Mean8.append(me)
#     else:
#         me=mean(L8[i-1:j])
#         Mean8.append(me)
# # print(Mean)
# Mean8=np.array(Mean8)

# # 标准差
# Std8=[]
# for i,j in segment8:
#     if(i==j):
#         Std8.append(0)
#     else:
#         me=np.std(L8[i-1:j])
#         Std8.append(round(me,2))#保留两位
    
# # print(Std)
# Std8=np.array(Std8)

# # 0.25Per
# Per18=[]
# for i,j in segment8:
#     if(i==j):
#         Per18.append(L8[i-1]*0.25)
#     else:
#         cu=sum(L8[i-1:j])
#         Per18.append(cu*0.25)
# Per18=np.array(Per18)

# # 0.5Per
# Per28=[]
# for i,j in segment8:
#     if(i==j):
#         Per28.append(L8[i-1]*0.5)
#     else:
#         cu=sum(L8[i-1:j])
#         Per28.append(cu*0.5)
# Per28=np.array(Per28)

# # 0.75Per
# Per38=[]
# for i,j in segment8:
#     if(i==j):
#         Per38.append(L8[i-1]*0.75)
#     else:
#         cu=sum(L8[i-1:j])
#         Per38.append(cu*0.75)
# Per38=np.array(Per38)

# Num9=[]
# Cumsize9=[]
# # 双向数据包的数量
# for i,j in segment9:
#     Itv=j-i+1
#     Num9.append(Itv)
# # print(Num4)
# print(np.array(Num9).shape)
# Num9=np.array(Num9)
# # print(Num8)

# fileLength9=Tpi2.loc[:,['Length']]
# L9=fileLength9['Length'].tolist()
# # print(L8)
# # for i,j in segment8:
# #     if(len(L8[i:j])==0):
# #         print('i=',i)
# #         print('j=',j)
# # print(L8)
# # print(LL)

# # 累计大小
# for i,j in segment9:
#     cu=sum(L9[i-1:j])
#     Cumsize9.append(cu)
# Cumsize9=np.array(Cumsize9)
# # print(Cumsize8)

# #特征2:最小值，最大值，均值，标准差
# # 最小值
# minimum9=[]
# for i,j in segment9:
#     if(i==j):
#         mm=L9[i-1]
#         minimum9.append(mm)
#     else:
#         mm=min(L9[i-1:j])
#         minimum9.append(mm)
# minimum9=np.array(minimum9)


# # 最大值
# maximum9=[]
# for i,j in segment9:
#     if(i==j):
#         ma=L9[i-1]
#         maximum9.append(ma)
#     else:
#         ma=max(L9[i-1:j])
#         maximum9.append(ma)
# maximum9=np.array(maximum9)
# # print(maximum)

# # 均值
# Mean9=[]
# for i,j in segment9:
#     if(i==j):
#         me=L9[i-1]
#         Mean9.append(me)
#     else:
#         me=mean(L9[i-1:j])
#         Mean9.append(me)
# # print(Mean)
# Mean9=np.array(Mean9)

# # 标准差
# Std9=[]
# for i,j in segment9:
#     if(i==j):
#         Std9.append(0)
#     else:
#         me=np.std(L9[i-1:j])
#         Std9.append(round(me,2))#保留两位
    
# # print(Std)
# Std9=np.array(Std9)

# # 0.25Per
# Per19=[]
# for i,j in segment9:
#     if(i==j):
#         Per19.append(L9[i-1]*0.25)
#     else:
#         cu=sum(L9[i-1:j])
#         Per19.append(cu*0.25)
# Per19=np.array(Per19)

# # 0.5Per
# Per29=[]
# for i,j in segment9:
#     if(i==j):
#         Per29.append(L9[i-1]*0.5)
#     else:
#         cu=sum(L9[i-1:j])
#         Per29.append(cu*0.5)
# Per29=np.array(Per29)

# # 0.75Per
# Per39=[]
# for i,j in segment9:
#     if(i==j):
#         Per39.append(L9[i-1]*0.75)
#     else:
#         cu=sum(L9[i-1:j])
#         Per39.append(cu*0.75)
# Per39=np.array(Per39)

# Num10=[]
# Cumsize10=[]
# # 双向数据包的数量
# for i,j in segment10:
#     Itv=j-i+1
#     Num10.append(Itv)
# # print(Num4)
# print(np.array(Num10).shape)
# Num10=np.array(Num10)
# # print(Num8)

# fileLength10=Tau2.loc[:,['Length']]
# L10=fileLength10['Length'].tolist()
# # print(L8)
# # for i,j in segment8:
# #     if(len(L8[i:j])==0):
# #         print('i=',i)
# #         print('j=',j)
# # print(L8)
# # print(LL)

# # 累计大小
# for i,j in segment10:
#     cu=sum(L10[i-1:j])
#     Cumsize10.append(cu)
# Cumsize10=np.array(Cumsize10)
# # print(Cumsize8)

# #特征2:最小值，最大值，均值，标准差
# # 最小值
# minimum10=[]
# for i,j in segment10:
#     if(i==j):
#         mm=L10[i-1]
#         minimum10.append(mm)
#     else:
#         mm=min(L10[i-1:j])
#         minimum10.append(mm)
# minimum10=np.array(minimum10)


# # 最大值
# maximum10=[]
# for i,j in segment10:
#     if(i==j):
#         ma=L10[i-1]
#         maximum10.append(ma)
#     else:
#         ma=max(L10[i-1:j])
#         maximum10.append(ma)
# maximum10=np.array(maximum10)
# # print(maximum)

# # 均值
# Mean10=[]
# for i,j in segment10:
#     if(i==j):
#         me=L10[i-1]
#         Mean10.append(me)
#     else:
#         me=mean(L10[i-1:j])
#         Mean10.append(me)
# # print(Mean)
# Mean10=np.array(Mean10)

# # 标准差
# Std10=[]
# for i,j in segment10:
#     if(i==j):
#         Std10.append(0)
#     else:
#         me=np.std(L10[i-1:j])
#         Std10.append(round(me,2))#保留两位
    
# # print(Std)
# Std10=np.array(Std10)

# # 0.25Per
# Per110=[]
# for i,j in segment10:
#     if(i==j):
#         Per110.append(L10[i-1]*0.25)
#     else:
#         cu=sum(L10[i-1:j])
#         Per110.append(cu*0.25)
# Per110=np.array(Per110)

# # 0.5Per
# Per210=[]
# for i,j in segment10:
#     if(i==j):
#         Per210.append(L10[i-1]*0.5)
#     else:
#         cu=sum(L10[i-1:j])
#         Per210.append(cu*0.5)
# Per210=np.array(Per210)

# # 0.75Per
# Per310=[]
# for i,j in segment10:
#     if(i==j):
#         Per310.append(L10[i-1]*0.75)
#     else:
#         cu=sum(L10[i-1:j])
#         Per310.append(cu*0.75)
# Per310=np.array(Per310)

# Num11=[]
# Cumsize11=[]
# # 双向数据包的数量
# for i,j in segment11:
#     Itv=j-i+1
#     Num11.append(Itv)
# # print(Num4)
# print(np.array(Num11).shape)
# Num11=np.array(Num11)
# # print(Num8)

# fileLength11=Tvi2.loc[:,['Length']]
# L11=fileLength11['Length'].tolist()
# # print(L8)
# # for i,j in segment8:
# #     if(len(L8[i:j])==0):
# #         print('i=',i)
# #         print('j=',j)
# # print(L8)
# # print(LL)

# # 累计大小
# for i,j in segment11:
#     cu=sum(L11[i-1:j])
#     Cumsize11.append(cu)
# Cumsize11=np.array(Cumsize11)
# # print(Cumsize8)

# #特征2:最小值，最大值，均值，标准差
# # 最小值
# minimum11=[]
# for i,j in segment11:
#     if(i==j):
#         mm=L11[i-1]
#         minimum11.append(mm)
#     else:
#         mm=min(L11[i-1:j])
#         minimum11.append(mm)
# minimum11=np.array(minimum11)


# # 最大值
# maximum11=[]
# for i,j in segment11:
#     if(i==j):
#         ma=L11[i-1]
#         maximum11.append(ma)
#     else:
#         ma=max(L11[i-1:j])
#         maximum11.append(ma)
# maximum11=np.array(maximum11)
# # print(maximum)

# # 均值
# Mean11=[]
# for i,j in segment11:
#     if(i==j):
#         me=L11[i-1]
#         Mean11.append(me)
#     else:
#         me=mean(L11[i-1:j])
#         Mean11.append(me)
# # print(Mean)
# Mean11=np.array(Mean11)

# # 标准差
# Std11=[]
# for i,j in segment11:
#     if(i==j):
#         Std11.append(0)
#     else:
#         me=np.std(L11[i-1:j])
#         Std11.append(round(me,2))#保留两位
    
# # print(Std)
# Std11=np.array(Std11)

# # 0.25Per
# Per111=[]
# for i,j in segment11:
#     if(i==j):
#         Per111.append(L11[i-1]*0.25)
#     else:
#         cu=sum(L11[i-1:j])
#         Per111.append(cu*0.25)
# Per111=np.array(Per111)

# # 0.5Per
# Per211=[]
# for i,j in segment11:
#     if(i==j):
#         Per211.append(L11[i-1]*0.5)
#     else:
#         cu=sum(L11[i-1:j])
#         Per211.append(cu*0.5)
# Per211=np.array(Per211)

# # 0.75Per
# Per311=[]
# for i,j in segment11:
#     if(i==j):
#         Per311.append(L11[i-1]*0.75)
#     else:
#         cu=sum(L11[i-1:j])
#         Per311.append(cu*0.75)
# Per311=np.array(Per311)

# Num12=[]
# Cumsize12=[]
# # 双向数据包的数量
# for i,j in segment12:
#     Itv=j-i+1
#     Num12.append(Itv)
# # print(Num4)
# print(np.array(Num12).shape)
# Num12=np.array(Num12)
# # print(Num8)

# fileLength12=Tfi2.loc[:,['Length']]
# L12=fileLength12['Length'].tolist()
# # print(L8)
# # for i,j in segment8:
# #     if(len(L8[i:j])==0):
# #         print('i=',i)
# #         print('j=',j)
# # print(L8)
# # print(LL)

# # 累计大小
# for i,j in segment12:
#     cu=sum(L12[i-1:j])
#     Cumsize12.append(cu)
# Cumsize12=np.array(Cumsize12)
# # print(Cumsize8)

# #特征2:最小值，最大值，均值，标准差
# # 最小值
# minimum12=[]
# for i,j in segment12:
#     if(i==j):
#         mm=L12[i-1]
#         minimum12.append(mm)
#     else:
#         mm=min(L12[i-1:j])
#         minimum12.append(mm)
# minimum12=np.array(minimum12)


# # 最大值
# maximum12=[]
# for i,j in segment12:
#     if(i==j):
#         ma=L12[i-1]
#         maximum12.append(ma)
#     else:
#         ma=max(L12[i-1:j])
#         maximum12.append(ma)
# maximum12=np.array(maximum12)
# # print(maximum)

# # 均值
# Mean12=[]
# for i,j in segment12:
#     if(i==j):
#         me=L12[i-1]
#         Mean12.append(me)
#     else:
#         me=mean(L12[i-1:j])
#         Mean12.append(me)
# # print(Mean)
# Mean12=np.array(Mean12)

# # 标准差
# Std12=[]
# for i,j in segment12:
#     if(i==j):
#         Std12.append(0)
#     else:
#         me=np.std(L12[i-1:j])
#         Std12.append(round(me,2))#保留两位
    
# # print(Std)
# Std12=np.array(Std12)

# # 0.25Per
# Per112=[]
# for i,j in segment12:
#     if(i==j):
#         Per112.append(L12[i-1]*0.25)
#     else:
#         cu=sum(L12[i-1:j])
#         Per112.append(cu*0.25)
# Per112=np.array(Per112)

# # 0.5Per
# Per212=[]
# for i,j in segment12:
#     if(i==j):
#         Per212.append(L12[i-1]*0.5)
#     else:
#         cu=sum(L12[i-1:j])
#         Per212.append(cu*0.5)
# Per212=np.array(Per212)

# # 0.75Per
# Per312=[]
# for i,j in segment12:
#     if(i==j):
#         Per312.append(L12[i-1]*0.75)
#     else:
#         cu=sum(L12[i-1:j])
#         Per312.append(cu*0.75)
# Per312=np.array(Per312)
# #特征1:双向数据包的数量，和累计大小

# Num=[]
# Cumsize=[]
# # 双向数据包的数量
# for i,j in segment1:
#     Itv=j-i
#     Num.append(Itv)
# print(Num[1999])
# print(Num[2000])
# print(Num[2001])
# Num=np.array(Num)
# # # print(Num)

# fileLength=file2.loc[:,['Length']]
# LL=fileLength['Length'].tolist()
# # # print(LL)


# # # 累计大小
# for i,j in segment1:
#     cu=sum(LL[i:j])
#     Cumsize.append(cu)
# Cumsize=np.array(Cumsize)

# # for i,j in segment1:
# #     print(np.array(LL[i:j])

# # for i,j in segment1:
# #     print(min(LL[i:j]))
# # 特征2:最小值，最大值，均值，标准差
# # 最小值
# # for i,j in segment1:
# #     if not LL[i:j]:
# #         print(i,j)
    

# minimum=[]
# for i,j in segment1:
#     mmin=min(LL[i:j])
#     minimum.append(mmin)
# minimum=np.array(minimum)
# print(minimum)


# # 最大值
# maximum=[]
# for i,j in segment1:
#     ma=max(LL[i:j])
#     maximum.append(ma)
# maximum=np.array(maximum)
# # # # print(maximum)

# # 均值
# Mean=[]
# for i,j in segment1:
#     me=mean(LL[i:j])
#     Mean.append(me)
# # print(Mean)
# Mean=np.array(Mean)

# # 标准差
# Std=[]
# for i,j in segment1:
#     me=np.std(LL[i:j])
#     Std.append(round(me,2))#保留两位
# # print(Std)
# Std=np.array(Std)

# # # # 0.25Per
# Per1=[]
# for i,j in segment1:
#     cu=sum(LL[i:j])
#     Per1.append(cu*0.25)
# Per1=np.array(Per1)

# # # # 0.5Per
# Per2=[]
# for i,j in segment1:
#     cu=sum(LL[i:j])
#     Per2.append(cu*0.5)
# Per2=np.array(Per2)

# # # # 0.75Per
# Per3=[]
# for i,j in segment1:
#     cu=sum(LL[i:j])
#     Per3.append(cu*0.75)
# Per3=np.array(Per3)



# # #特征1:双向数据包的数量，和累计大小

# Num1=[]
# Cumsize1=[]
# # 双向数据包的数量
# for i,j in segment2:
#     Itv=j-i
#     Num1.append(Itv)
# print(np.array(Num1).shape)
# Num1=np.array(Num1)

# fileLength1=picc2.loc[:,['Length']]
# LLL=fileLength1['Length'].tolist()
# # # print(LL)

# # # 累计大小
# for i,j in segment2:
#     cu=sum(LLL[i:j])
#     Cumsize1.append(cu)
# Cumsize1=np.array(Cumsize1)

# #特征2:最小值，最大值，均值，标准差
# # 最小值
# minimum1=[]
# for i,j in segment2:
#     mm=min(LLL[i:j])
#     minimum1.append(mm)
# minimum1=np.array(minimum1)
# for i,j in segment2:
#     if not LLL[i:j]:
#         print(i,j)

# # 最大值
# maximum1=[]
# for i,j in segment2:
#     ma=max(LLL[i:j])
#     maximum1.append(ma)
# # maximum=np.array(maximum)
# # print(maximum)

# # 均值
# Mean1=[]
# for i,j in segment2:
#     me=mean(LLL[i:j])
#     Mean1.append(me)
# # print(Mean)
# Mean1=np.array(Mean1)

# # 标准差
# Std1=[]
# for i,j in segment2:
#     me=np.std(LLL[i:j])
#     Std1.append(round(me,2))#保留两位
# # print(Std)
# # Mean=np.array(Mean)

# # 0.25Per
# Per11=[]
# for i,j in segment2:
#     cu=sum(LLL[i:j])
#     Per11.append(cu*0.25)
# Per11=np.array(Per11)

# # 0.5Per
# Per21=[]
# for i,j in segment2:
#     cu=sum(LLL[i:j])
#     Per21.append(cu*0.5)
# Per21=np.array(Per21)

# # 0.75Per
# Per31=[]
# for i,j in segment2:
#     cu=sum(LLL[i:j])
#     Per31.append(cu*0.75)
# Per31=np.array(Per31)

# Num2=[]
# Cumsize2=[]
# # 双向数据包的数量
# for i,j in segment3:
#     Itv=j-i
#     Num2.append(Itv)
# print(np.array(Num2).shape)
# Num2=np.array(Num2)

# fileLength2=momm2.loc[:,['Length']]
# LLLL=fileLength2['Length'].tolist()
# # print(LL)

# # 累计大小
# for i,j in segment3:
#     cu=sum(LLLL[i:j])
#     Cumsize2.append(cu)
# Cumsize2=np.array(Cumsize2)

# #特征2:最小值，最大值，均值，标准差
# # 最小值
# minimum2=[]
# for i,j in segment3:
#     mm=min(LLLL[i:j])
#     minimum2.append(mm)
# minimum2=np.array(minimum2)


# # 最大值
# maximum2=[]
# for i,j in segment3:
#     ma=max(LLLL[i:j])
#     maximum2.append(ma)
# # maximum=np.array(maximum)
# # print(maximum)

# # 均值
# Mean2=[]
# for i,j in segment3:
#     me=mean(LLLL[i:j])
#     Mean2.append(me)
# # print(Mean)
# Mean2=np.array(Mean2)

# # 标准差
# Std2=[]
# for i,j in segment3:
#     me=np.std(LLLL[i:j])
#     Std2.append(round(me,2))#保留两位
# # print(Std)
# # Mean=np.array(Mean)

# # 0.25Per
# Per12=[]
# for i,j in segment3:
#     cu=sum(LLLL[i:j])
#     Per12.append(cu*0.25)
# Per12=np.array(Per12)

# # 0.5Per
# Per22=[]
# for i,j in segment3:
#     cu=sum(LLLL[i:j])
#     Per22.append(cu*0.5)
# Per22=np.array(Per22)

# # 0.75Per
# Per32=[]
# for i,j in segment3:
#     cu=sum(LLLL[i:j])
#     Per32.append(cu*0.75)
# Per32=np.array(Per32)

# Num3=[]
# Cumsize3=[]
# # 双向数据包的数量
# for i,j in segment4:
#     Itv=j-i
#     Num3.append(Itv)
# print(np.array(Num3).shape)
# Num3=np.array(Num3)


# fileLength3=vcalll2.loc[:,['Length']]
# LLLLL=fileLength3['Length'].tolist()
# # print(LL)

# # 累计大小
# for i,j in segment4:
#     cu=sum(LLLLL[i:j])
#     Cumsize3.append(cu)
# Cumsize3=np.array(Cumsize3)

# #特征2:最小值，最大值，均值，标准差
# # 最小值
# minimum3=[]
# for i,j in segment4:
#     mm=min(LLLLL[i:j])
#     minimum3.append(mm)
# minimum3=np.array(minimum3)


# # 最大值
# maximum3=[]
# for i,j in segment4:
#     ma=max(LLLLL[i:j])
#     maximum3.append(ma)
# # maximum=np.array(maximum)
# # print(maximum)

# # 均值
# Mean3=[]
# for i,j in segment4:
#     me=mean(LLLLL[i:j])
#     Mean3.append(me)
# # print(Mean)
# Mean3=np.array(Mean3)

# # 标准差
# Std3=[]
# for i,j in segment4:
#     me=np.std(LLLLL[i:j])
#     Std3.append(round(me,2))#保留两位
# # print(Std)
# # Mean=np.array(Mean)

# # 0.25Per
# Per13=[]
# for i,j in segment4:
#     cu=sum(LLLLL[i:j])
#     Per13.append(cu*0.25)
# Per13=np.array(Per13)

# # 0.5Per
# Per23=[]
# for i,j in segment4:
#     cu=sum(LLLLL[i:j])
#     Per23.append(cu*0.5)
# Per23=np.array(Per23)

# # 0.75Per
# Per33=[]
# for i,j in segment4:
#     cu=sum(LLLLL[i:j])
#     Per33.append(cu*0.75)
# Per33=np.array(Per33)

# Num4=[]
# Cumsize4=[]
# # 双向数据包的数量
# for i,j in segment5:
#     Itv=j-i
#     Num4.append(Itv)
# # print(Num4)
# print(np.array(Num4).shape)
# Num4=np.array(Num4)
# # print(Num4.shape)

# fileLength4=RP2.loc[:,['Length']]
# LLLLLL=fileLength4['Length'].tolist()
# # # # print(LL)

# # # # 累计大小
# for i,j in segment5:
#     cu=sum(LLLLLL[i:j])
#     Cumsize4.append(cu)
# Cumsize4=np.array(Cumsize4)

# # # #特征2:最小值，最大值，均值，标准差
# # # # 最小值
# minimum4=[]
# for i,j in segment5:
#     mm=min(LLLLLL[i:j])
#     minimum4.append(mm)
# minimum4=np.array(minimum4)


# # # # 最大值
# maximum4=[]
# for i,j in segment5:
#     ma=max(LLLLLL[i:j])
#     maximum4.append(ma)
# maximum4=np.array(maximum4)
# # print(maximum)

# # # # 均值
# Mean4=[]
# for i,j in segment5:
#     me=mean(LLLLLL[i:j])
#     Mean4.append(me)
# # print(Mean)
# Mean4=np.array(Mean4)

# # # # 标准差
# Std4=[]
# for i,j in segment5:
#     me=np.std(LLLLLL[i:j])
#     Std4.append(round(me,2))#保留两位
    
# # # # print(Std)
# # # # Mean=np.array(Mean)

# # # # 0.25Per
# Per14=[]
# for i,j in segment5:
#     cu=sum(LLLLLL[i:j])
#     Per14.append(cu*0.25)
# Per14=np.array(Per14)

# # # # 0.5Per
# Per24=[]
# for i,j in segment5:
#     cu=sum(LLLLLL[i:j])
#     Per24.append(cu*0.5)
# Per24=np.array(Per24)

# # # # 0.75Per
# Per34=[]
# for i,j in segment5:
#     cu=sum(LLLLLL[i:j])
#     Per34.append(cu*0.75)
# Per34=np.array(Per34)

# Num5=[]
# Cumsize5=[]
# # 双向数据包的数量
# for i,j in segment6:
#     Itv=j-i
#     Num5.append(Itv)
# # print(Num4)
# print(np.array(Num5).shape)
# Num5=np.array(Num5)

# fileLength5=location2.loc[:,['Length']]
# LLLLLLL=fileLength5['Length'].tolist()
# # print(LL)

# # 累计大小
# for i,j in segment6:
#     cu=sum(LLLLLLL[i:j])
#     Cumsize5.append(cu)
# Cumsize5=np.array(Cumsize5)

# #特征2:最小值，最大值，均值，标准差
# # 最小值
# minimum5=[]
# for i,j in segment6:
#     mm=min(LLLLLLL[i:j])
#     minimum5.append(mm)
# minimum5=np.array(minimum5)


# # 最大值
# maximum5=[]
# for i,j in segment6:
#     ma=max(LLLLLLL[i:j])
#     maximum5.append(ma)
# # maximum=np.array(maximum)
# # print(maximum)

# # 均值
# Mean5=[]
# for i,j in segment6:
#     me=mean(LLLLLLL[i:j])
#     Mean5.append(me)
# # print(Mean)
# Mean5=np.array(Mean5)

# # 标准差
# Std5=[]
# for i,j in segment6:
#     me=np.std(LLLLLLL[i:j])
#     Std5.append(round(me,2))#保留两位
    
# # print(Std)
# # Mean=np.array(Mean)

# # 0.25Per
# Per15=[]
# for i,j in segment6:
#     cu=sum(LLLLLLL[i:j])
#     Per15.append(cu*0.25)
# Per15=np.array(Per15)

# # 0.5Per
# Per25=[]
# for i,j in segment6:
#     cu=sum(LLLLLLL[i:j])
#     Per25.append(cu*0.5)
# Per25=np.array(Per25)

# # 0.75Per
# Per35=[]
# for i,j in segment6:
#     cu=sum(LLLLLLL[i:j])
#     Per35.append(cu*0.75)
# Per35=np.array(Per35)


# Num7=[]
# Cumsize7=[]
# # 双向数据包的数量
# for i,j in segment7:
#     Itv=j-i
#     Num7.append(Itv)
# # print(Num4)
# print(np.array(Num7).shape)
# Num7=np.array(Num7)

# fileLength7=audio2.loc[:,['Length']]
# LLLLLLLLL=fileLength7['Length'].tolist()
# # print(LL)

# # 累计大小
# for i,j in segment7:
#     cu=sum(LLLLLLLLL[i:j])
#     Cumsize7.append(cu)
# Cumsize7=np.array(Cumsize7)

# #特征2:最小值，最大值，均值，标准差
# # 最小值
# minimum7=[]
# for i,j in segment7:
#     mm=min(LLLLLLLLL[i:j])
#     minimum7.append(mm)
# minimum7=np.array(minimum7)


# # 最大值
# maximum7=[]
# for i,j in segment7:
#     ma=max(LLLLLLLLL[i:j])
#     maximum7.append(ma)
# # maximum=np.array(maximum)
# # print(maximum)

# # 均值
# Mean7=[]
# for i,j in segment7:
#     me=mean(LLLLLLLLL[i:j])
#     Mean7.append(me)
# # print(Mean)
# Mean7=np.array(Mean7)

# # 标准差
# Std7=[]
# for i,j in segment7:
#     me=np.std(LLLLLLLLL[i:j])
#     Std7.append(round(me,2))#保留两位
    
# # print(Std)
# # Mean=np.array(Mean)

# # 0.25Per
# Per17=[]
# for i,j in segment7:
#     cu=sum(LLLLLLLLL[i:j])
#     Per17.append(cu*0.25)
# Per17=np.array(Per17)

# # 0.5Per
# Per27=[]
# for i,j in segment7:
#     cu=sum(LLLLLLLLL[i:j])
#     Per27.append(cu*0.5)
# Per27=np.array(Per27)

# # 0.75Per
# Per37=[]
# for i,j in segment7:
#     cu=sum(LLLLLLLLL[i:j])
#     Per37.append(cu*0.75)
# Per37=np.array(Per37)

# Num6=[]
# for i in range(len(Num3)):
#     k=Num3[i]*2
#     Num6.append(k)
# Num6=np.array(Num6)

# Cumsize6=[]
# for i in range(len(Num6)):
#     k=Cumsize3[i]*2
#     Cumsize6.append(k)
# Cumsize6=np.array(Cumsize6)
# minimum6=[]
# for i in range(len(Num6)):
#     k=minimum3[i]*2
#     minimum6.append(k)
# minimum6=np.array(minimum6)
# maximum6=[]
# for i in range(len(Num6)):
#     k=maximum3[i]*2
#     maximum6.append(k)
# maximum6=np.array(maximum6)
# Mean6=[]
# for i in range(len(Num6)):
#     k=Mean3[i]*2
#     Mean6.append(k)
# Mean6=np.array(Mean6)
# Std6=[]
# for i in range(len(Num6)):
#     k=Std3[i]*2
#     Std6.append(k)
# Std6=np.array(Std6)
# Per16=[]
# for i in range(len(Num6)):
#     k=Per13[i]*2
#     Per16.append(k)
# Per16=np.array(Per16)
# Per26=[]
# for i in range(len(Num6)):
#     k=Per23[i]*2
#     Per26.append(k)
# Per26=np.array(Per26)
# Per36=[]
# for i in range(len(Num6)):
#     k=Per33[i]*2
#     Per36.append(k)
# Per36=np.array(Per36)

# Num6=[]
# Cumsize6=[]
# # 双向数据包的数量
# for i,j in segment6:
#     Itv=j-i+1
#     Num6.append(Itv)
# # print(Num4)
# print(np.array(Num6).shape)
# Num6=np.array(Num6)
# # print(Num8)

# fileLength6=video2.loc[:,['Length']]
# L6=fileLength6['Length'].tolist()
# # print(L8)
# # for i,j in segment8:
# #     if(len(L8[i:j])==0):
# #         print('i=',i)
# #         print('j=',j)
# # print(L8)
# # print(LL)

# # 累计大小
# for i,j in segment6:
#     cu=sum(L6[i-1:j])
#     Cumsize6.append(cu)
# Cumsize6=np.array(Cumsize6)
# # print(Cumsize8)

# #特征2:最小值，最大值，均值，标准差
# # 最小值
# minimum6=[]
# for i,j in segment6:
#     if(i==j):
#         mm=L6[i-1]
#         minimum6.append(mm)
#     else:
#         mm=min(L6[i-1:j])
#         minimum6.append(mm)
# minimum6=np.array(minimum6)


# # 最大值
# maximum6=[]
# for i,j in segment6:
#     if(i==j):
#         ma=L6[i-1]
#         maximum6.append(ma)
#     else:
#         ma=max(L6[i-1:j])
#         maximum6.append(ma)
# maximum6=np.array(maximum6)
# # print(maximum)

# # 均值
# Mean6=[]
# for i,j in segment6:
#     if(i==j):
#         me=L6[i-1]
#         Mean6.append(me)
#     else:
#         me=mean(L6[i-1:j])
#         Mean6.append(me)
# # print(Mean)
# Mean6=np.array(Mean6)

# # 标准差
# Std6=[]
# for i,j in segment6:
#     if(i==j):
#         Std6.append(0)
#     else:
#         me=np.std(L6[i-1:j])
#         Std6.append(round(me,2))#保留两位
    
# # print(Std)
# Std6=np.array(Std6)

# # 0.25Per
# Per16=[]
# for i,j in segment6:
#     if(i==j):
#         Per16.append(L6[i-1]*0.25)
#     else:
#         cu=sum(L6[i-1:j])
#         Per16.append(cu*0.25)
# Per16=np.array(Per16)

# # 0.5Per
# Per26=[]
# for i,j in segment6:
#     if(i==j):
#         Per26.append(L6[i-1]*0.5)
#     else:
#         cu=sum(L6[i-1:j])
#         Per26.append(cu*0.5)
# Per26=np.array(Per26)

# # 0.75Per
# Per36=[]
# for i,j in segment6:
#     if(i==j):
#         Per36.append(L6[i-1]*0.75)
#     else:
#         cu=sum(L6[i-1:j])
#         Per36.append(cu*0.75)
# Per36=np.array(Per36)

# Num13=[]
# Cumsize13=[]
# # 双向数据包的数量
# for i,j in segment13:
#     Itv=j-i+1
#     Num13.append(Itv)
# # print(Num4)
# print(np.array(Num13).shape)
# Num13=np.array(Num13)
# # print(Num8)

# fileLength13=aca2.loc[:,['Length']]
# L13=fileLength13['Length'].tolist()
# # print(L8)
# # for i,j in segment8:
# #     if(len(L8[i:j])==0):
# #         print('i=',i)
# #         print('j=',j)
# # print(L8)
# # print(LL)

# # 累计大小
# for i,j in segment13:
#     cu=sum(L13[i-1:j])
#     Cumsize13.append(cu)
# Cumsize13=np.array(Cumsize13)
# # print(Cumsize8)

# #特征2:最小值，最大值，均值，标准差
# # 最小值
# minimum13=[]
# for i,j in segment13:
#     if(i==j):
#         mm=L13[i-1]
#         minimum13.append(mm)
#     else:
#         mm=min(L13[i-1:j])
#         minimum13.append(mm)
# minimum13=np.array(minimum13)


# # 最大值
# maximum13=[]
# for i,j in segment13:
#     if(i==j):
#         ma=L13[i-1]
#         maximum13.append(ma)
#     else:
#         ma=max(L13[i-1:j])
#         maximum13.append(ma)
# maximum13=np.array(maximum13)
# # print(maximum)

# # 均值
# Mean13=[]
# for i,j in segment13:
#     if(i==j):
#         me=L13[i-1]
#         Mean13.append(me)
#     else:
#         me=mean(L13[i-1:j])
#         Mean13.append(me)
# # print(Mean)
# Mean13=np.array(Mean13)

# # 标准差
# Std13=[]
# for i,j in segment13:
#     if(i==j):
#         Std13.append(0)
#     else:
#         me=np.std(L13[i-1:j])
#         Std13.append(round(me,2))#保留两位
    
# # print(Std)
# Std13=np.array(Std13)

# # 0.25Per
# Per113=[]
# for i,j in segment13:
#     if(i==j):
#         Per113.append(L13[i-1]*0.25)
#     else:
#         cu=sum(L13[i-1:j])
#         Per113.append(cu*0.25)
# Per113=np.array(Per113)

# # 0.5Per
# Per213=[]
# for i,j in segment13:
#     if(i==j):
#         Per213.append(L13[i-1]*0.5)
#     else:
#         cu=sum(L13[i-1:j])
#         Per213.append(cu*0.5)
# Per213=np.array(Per213)

# # 0.75Per
# Per313=[]
# for i,j in segment13:
#     if(i==j):
#         Per313.append(L13[i-1]*0.75)
#     else:
#         cu=sum(L13[i-1:j])
#         Per313.append(cu*0.75)
# Per313=np.array(Per313)

# Num14=[]
# Cumsize14=[]
# # 双向数据包的数量
# for i,j in segment14:
#     Itv=j-i+1
#     Num14.append(Itv)
# # print(Num4)
# print(np.array(Num14).shape)
# Num14=np.array(Num14)
# # print(Num8)

# fileLength14=rn2.loc[:,['Length']]
# L14=fileLength14['Length'].tolist()
# # print(L8)
# # for i,j in segment8:
# #     if(len(L8[i:j])==0):
# #         print('i=',i)
# #         print('j=',j)
# # print(L8)
# # print(LL)

# # 累计大小
# for i,j in segment14:
#     cu=sum(L14[i-1:j])
#     Cumsize14.append(cu)
# Cumsize14=np.array(Cumsize14)
# # print(Cumsize8)

# #特征2:最小值，最大值，均值，标准差
# # 最小值
# minimum14=[]
# for i,j in segment14:
#     if(i==j):
#         mm=L14[i-1]
#         minimum14.append(mm)
#     else:
#         mm=min(L14[i-1:j])
#         minimum14.append(mm)
# minimum14=np.array(minimum14)


# # 最大值
# maximum14=[]
# for i,j in segment14:
#     if(i==j):
#         ma=L14[i-1]
#         maximum14.append(ma)
#     else:
#         ma=max(L14[i-1:j])
#         maximum14.append(ma)
# maximum14=np.array(maximum14)
# # print(maximum)

# # 均值
# Mean14=[]
# for i,j in segment14:
#     if(i==j):
#         me=L14[i-1]
#         Mean14.append(me)
#     else:
#         me=mean(L14[i-1:j])
#         Mean14.append(me)
# # print(Mean)
# Mean14=np.array(Mean14)

# # 标准差
# Std14=[]
# for i,j in segment14:
#     if(i==j):
#         Std14.append(0)
#     else:
#         me=np.std(L14[i-1:j])
#         Std14.append(round(me,2))#保留两位
    
# # print(Std)
# Std14=np.array(Std14)

# # 0.25Per
# Per114=[]
# for i,j in segment14:
#     if(i==j):
#         Per114.append(L14[i-1]*0.25)
#     else:
#         cu=sum(L14[i-1:j])
#         Per114.append(cu*0.25)
# Per114=np.array(Per114)

# # 0.5Per
# Per214=[]
# for i,j in segment14:
#     if(i==j):
#         Per214.append(L14[i-1]*0.5)
#     else:
#         cu=sum(L14[i-1:j])
#         Per214.append(cu*0.5)
# Per214=np.array(Per214)

# # 0.75Per
# Per314=[]
# for i,j in segment14:
#     if(i==j):
#         Per314.append(L14[i-1]*0.75)
#     else:
#         cu=sum(L14[i-1:j])
#         Per314.append(cu*0.75)
# Per314=np.array(Per314)

# Num15=[]
# Cumsize15=[]
# # 双向数据包的数量
# for i,j in segment15:
#     Itv=j-i+1
#     Num15.append(Itv)
# # print(Num4)
# print(np.array(Num15).shape)
# Num15=np.array(Num15)
# # print(Num8)

# fileLength15=trf2.loc[:,['Length']]
# L15=fileLength15['Length'].tolist()
# # print(L8)
# # for i,j in segment8:
# #     if(len(L8[i:j])==0):
# #         print('i=',i)
# #         print('j=',j)
# # print(L8)
# # print(LL)

# # 累计大小
# for i,j in segment15:
#     cu=sum(L15[i-1:j])
#     Cumsize15.append(cu)
# Cumsize15=np.array(Cumsize15)
# # print(Cumsize8)

# #特征2:最小值，最大值，均值，标准差
# # 最小值
# minimum15=[]
# for i,j in segment15:
#     if(i==j):
#         mm=L15[i-1]
#         minimum15.append(mm)
#     else:
#         mm=min(L15[i-1:j])
#         minimum15.append(mm)
# minimum15=np.array(minimum15)


# # 最大值
# maximum15=[]
# for i,j in segment15:
#     if(i==j):
#         ma=L15[i-1]
#         maximum15.append(ma)
#     else:
#         ma=max(L15[i-1:j])
#         maximum15.append(ma)
# maximum15=np.array(maximum15)
# # print(maximum)

# # 均值
# Mean15=[]
# for i,j in segment15:
#     if(i==j):
#         me=L15[i-1]
#         Mean15.append(me)
#     else:
#         me=mean(L15[i-1:j])
#         Mean15.append(me)
# # print(Mean)
# Mean15=np.array(Mean15)

# # 标准差
# Std15=[]
# for i,j in segment15:
#     if(i==j):
#         Std15.append(0)
#     else:
#         me=np.std(L15[i-1:j])
#         Std15.append(round(me,2))#保留两位
    
# # print(Std)
# Std15=np.array(Std15)

# # 0.25Per
# Per115=[]
# for i,j in segment15:
#     if(i==j):
#         Per115.append(L15[i-1]*0.25)
#     else:
#         cu=sum(L15[i-1:j])
#         Per115.append(cu*0.25)
# Per115=np.array(Per115)

# # 0.5Per
# Per215=[]
# for i,j in segment15:
#     if(i==j):
#         Per215.append(L15[i-1]*0.5)
#     else:
#         cu=sum(L15[i-1:j])
#         Per215.append(cu*0.5)
# Per215=np.array(Per215)

# # 0.75Per
# Per315=[]
# for i,j in segment15:
#     if(i==j):
#         Per315.append(L15[i-1]*0.75)
#     else:
#         cu=sum(L15[i-1:j])
#         Per315.append(cu*0.75)
# Per315=np.array(Per315)

# Num16=[]
# Cumsize16=[]
# # 双向数据包的数量
# for i,j in segment16:
#     Itv=j-i+1
#     Num16.append(Itv)
# # print(Num4)
# print(np.array(Num16).shape)
# Num16=np.array(Num16)
# # print(Num8)

# fileLength16=opm2.loc[:,['Length']]
# L16=fileLength16['Length'].tolist()
# # print(L8)
# # for i,j in segment8:
# #     if(len(L8[i:j])==0):
# #         print('i=',i)
# #         print('j=',j)
# # print(L8)
# # print(LL)

# # 累计大小
# for i,j in segment16:
#     cu=sum(L16[i-1:j])
#     Cumsize16.append(cu)
# Cumsize16=np.array(Cumsize16)
# # print(Cumsize8)

# #特征2:最小值，最大值，均值，标准差
# # 最小值
# minimum16=[]
# for i,j in segment16:
#     if(i==j):
#         mm=L16[i-1]
#         minimum16.append(mm)
#     else:
#         mm=min(L16[i-1:j])
#         minimum16.append(mm)
# minimum16=np.array(minimum16)


# # 最大值
# maximum16=[]
# for i,j in segment16:
#     if(i==j):
#         ma=L16[i-1]
#         maximum16.append(ma)
#     else:
#         ma=max(L16[i-1:j])
#         maximum16.append(ma)
# maximum16=np.array(maximum16)
# # print(maximum)

# # 均值
# Mean16=[]
# for i,j in segment16:
#     if(i==j):
#         me=L16[i-1]
#         Mean16.append(me)
#     else:
#         me=mean(L16[i-1:j])
#         Mean16.append(me)
# # print(Mean)
# Mean16=np.array(Mean16)

# # 标准差
# Std16=[]
# for i,j in segment16:
#     if(i==j):
#         Std16.append(0)
#     else:
#         me=np.std(L16[i-1:j])
#         Std16.append(round(me,2))#保留两位
    
# # print(Std)
# Std16=np.array(Std16)

# # 0.25Per
# Per116=[]
# for i,j in segment16:
#     if(i==j):
#         Per116.append(L16[i-1]*0.25)
#     else:
#         cu=sum(L16[i-1:j])
#         Per116.append(cu*0.25)
# Per116=np.array(Per116)

# # 0.5Per
# Per216=[]
# for i,j in segment16:
#     if(i==j):
#         Per216.append(L16[i-1]*0.5)
#     else:
#         cu=sum(L16[i-1:j])
#         Per216.append(cu*0.5)
# Per216=np.array(Per216)

# # 0.75Per
# Per316=[]
# for i,j in segment16:
#     if(i==j):
#         Per316.append(L16[i-1]*0.75)
#     else:
#         cu=sum(L16[i-1:j])
#         Per316.append(cu*0.75)
# Per316=np.array(Per316)


# ##################################################################################################################
# #特征3:100 bin 步长，若大于1500字节，取1500字节

# # def Get_MXN_Array_initx(m,n,x):
# #     return [[x for i in range(m)] for j in range(n)]


# # for i in range(len(segment1)):
# #     row=[]
# #     for j in Num:
# #         for a,b in segment1:
# #          col=[]
# #          if (0<=LL[a:b]<100):
# #             col.append()
# ###################################################################################################################
# label=[]
# label1=[]
# label2=[]
# label3=[]
# label4=[]
# label5=[]
# label6=[]
# label7=[]
# label8=[]
# label9=[]
# label10=[]
# label11=[]
# label12=[]
# label13=[]
# label14=[]
# label15=[]
# label16=[]
# for j in Num8:
    # label8.append(8)
# for j in Num9:
#     label9.append(9)
# for j in Num10:
    # label10.append(10)
# for j in Num11:
#     label11.append(11)
# for j in Num12:
#     label12.append(12)                
# # for i in Num:
# #     label.append(0)
# # for j in Num1:
# #     label1.append(1)
# # for j in Num2:
# #     label2.append(2)
# # for j in Num3:
# #     label3.append(3)
# # for j in Num4:
# #     label4.append(4)
# # for j in Num5:
# #     label5.append(5)
# for j in Num6:
#     label6.append(6)
# # for j in Num7:
# #     label7.append(7)
# for j in Num13:
#     label13.append(13)
# for j in Num14:
#     label14.append(14)
# for j in Num15:
#     label15.append(15)
# for j in Num16:
#     label16.append(16)
# #-----------Telegram-----------------------------------------------
# data8={"Num":Num8,"Cumsize":Cumsize8,"minimum":minimum8,"maximum":maximum8,"mean":Mean8,"standard deviation":Std8,"0.25Per":Per18,"0.5Per":Per28,"0.75Per":Per38,"label":label8}
# data9={"Num":Num9,"Cumsize":Cumsize9,"minimum":minimum9,"maximum":maximum9,"mean":Mean9,"standard deviation":Std9,"0.25Per":Per19,"0.5Per":Per29,"0.75Per":Per39,"label":label9}
# data10={"Num":Num10,"Cumsize":Cumsize10,"minimum":minimum10,"maximum":maximum10,"mean":Mean10,"standard deviation":Std10,"0.25Per":Per110,"0.5Per":Per210,"0.75Per":Per310,"label":label10}
# data11={"Num":Num11,"Cumsize":Cumsize11,"minimum":minimum11,"maximum":maximum11,"mean":Mean11,"standard deviation":Std11,"0.25Per":Per111,"0.5Per":Per211,"0.75Per":Per311,"label":label11}
# data12={"Num":Num12,"Cumsize":Cumsize12,"minimum":minimum12,"maximum":maximum12,"mean":Mean12,"standard deviation":Std12,"0.25Per":Per112,"0.5Per":Per212,"0.75Per":Per312,"label":label12}
# #------------------------------------------------------------------

# # data={"Num":Num,"Cumsize":Cumsize,"minimum":minimum,"maximum":maximum,"mean":Mean,"standard deviation":Std,"0.25Per":Per1,"0.5Per":Per2,"0.75Per":Per3,"label":label}
# # data1={"Num":Num1,"Cumsize":Cumsize1,"minimum":minimum1,"maximum":maximum1,"mean":Mean1,"standard deviation":Std1,"0.25Per":Per11,"0.5Per":Per21,"0.75Per":Per31,"label":label1}
# # data2={"Num":Num2,"Cumsize":Cumsize2,"minimum":minimum2,"maximum":maximum2,"mean":Mean2,"standard deviation":Std2,"0.25Per":Per12,"0.5Per":Per22,"0.75Per":Per32,"label":label2}
# # data3={"Num":Num3,"Cumsize":Cumsize3,"minimum":minimum3,"maximum":maximum3,"mean":Mean3,"standard deviation":Std3,"0.25Per":Per13,"0.5Per":Per23,"0.75Per":Per33,"label":label3}
# # data4={"Num":Num4,"Cumsize":Cumsize4,"minimum":minimum4,"maximum":maximum4,"mean":Mean4,"standard deviation":Std4,"0.25Per":Per14,"0.5Per":Per24,"0.75Per":Per34,"label":label4}
# # data5={"Num":Num5,"Cumsize":Cumsize5,"minimum":minimum5,"maximum":maximum5,"mean":Mean5,"standard deviation":Std5,"0.25Per":Per15,"0.5Per":Per25,"0.75Per":Per35,"label":label5}
# # data7={"Num":Num7,"Cumsize":Cumsize7,"minimum":minimum7,"maximum":maximum7,"mean":Mean7,"standard deviation":Std7,"0.25Per":Per17,"0.5Per":Per27,"0.75Per":Per37,"label":label7}
# data6={"Num":Num6,"Cumsize":Cumsize6,"minimum":minimum6,"maximum":maximum6,"mean":Mean6,"standard deviation":Std6,"0.25Per":Per16,"0.5Per":Per26,"0.75Per":Per36,"label":label6}
# data13={"Num":Num13,"Cumsize":Cumsize13,"minimum":minimum13,"maximum":maximum13,"mean":Mean13,"standard deviation":Std13,"0.25Per":Per113,"0.5Per":Per213,"0.75Per":Per313,"label":label13}
# data14={"Num":Num14,"Cumsize":Cumsize14,"minimum":minimum14,"maximum":maximum14,"mean":Mean14,"standard deviation":Std14,"0.25Per":Per114,"0.5Per":Per214,"0.75Per":Per314,"label":label14}
# data15={"Num":Num15,"Cumsize":Cumsize15,"minimum":minimum15,"maximum":maximum15,"mean":Mean15,"standard deviation":Std15,"0.25Per":Per115,"0.5Per":Per215,"0.75Per":Per315,"label":label15}
# data16={"Num":Num16,"Cumsize":Cumsize16,"minimum":minimum16,"maximum":maximum16,"mean":Mean16,"standard deviation":Std16,"0.25Per":Per116,"0.5Per":Per216,"0.75Per":Per316,"label":label16}

# #---------Telegram-----------------------------------------------------
# featuresTt=pd.DataFrame(data8,index=None)
# featuresTt.to_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\telegram文本特征矩阵.csv')
# featuresTp=pd.DataFrame(data9,index=None)
# featuresTp.to_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\telegram图片特征矩阵.csv')
# featuresTa=pd.DataFrame(data10,index=None)
# featuresTa.to_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\telegram音频特征矩阵.csv')
# featuresTv=pd.DataFrame(data11,index=None)
# featuresTv.to_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\telegram视频特征矩阵.csv')
# featuresTf=pd.DataFrame(data12,index=None)
# featuresTf.to_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\telegram文档特征矩阵.csv')

# #----------------------------------------------------------------------

# featuresfinal = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\telegram加上微信特征矩阵.csv', encoding='gbk',index_col=0)


# # features=pd.DataFrame(data,index=None)
# # featurespic=pd.DataFrame(data1,index=None)
# # # featurespic.drop([1,4,5])
# # featuresmom=pd.DataFrame(data2,index=None)
# # # featuresmom.drop([1,4,5])
# # featuresvcall=pd.DataFrame(data3,index=None)
# # # featuresvcall.drop([1])
# # featuresredpacket=pd.DataFrame(data4,index=None)

# # featuresvideo=pd.DataFrame(data4,index=None)
# # featureslocation=pd.DataFrame(data5,index=None)
# # # featureslocation.drop([1,4,5,22,33])
# # featuressubscript=pd.DataFrame(data6,index=None)

# # featuresaudio=pd.DataFrame(data7,index=None)
# # # featuresaudio.drop([5])

# featuresvideo=pd.DataFrame(data6,index=None)
# featuresacall=pd.DataFrame(data13,index=None)
# featuresreadnews=pd.DataFrame(data14,index=None)
# featurestransfer=pd.DataFrame(data15,index=None)
# featuresopenminipro=pd.DataFrame(data16,index=None)

# featuresvideo.to_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\微信发送视频特征矩阵.csv')
# featuresacall.to_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\微信语音通话特征矩阵.csv')
# featuresreadnews.to_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\微信读公众号特征矩阵.csv')
# featurestransfer.to_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\微信转账特征矩阵.csv')
# featuresopenminipro.to_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\微信打开小程序特征矩阵.csv')

# featuresfinal = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\微信总特征矩阵.csv', encoding='gbk',index_col=0)





# featuresfinal=pd.concat([featuresfinal,featuresvideo],axis=0)
# featuresfinal=pd.concat([featuresfinal,featuresacall],axis=0)
# featuresfinal=pd.concat([featuresfinal,featuresreadnews],axis=0)
# featuresfinal=pd.concat([featuresfinal,featurestransfer],axis=0)
# featuresfinal=pd.concat([featuresfinal,featuresopenminipro],axis=0)

# featuresfinal.to_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\微信总特征矩阵.csv')

# # featuresfinal=pd.concat([featurespic,featuresmom],axis=0)
# # featuresfinal=pd.concat([featuresfinal,featuresmom],axis=0)
# # # print(featuresfinal)
# # featuresfinal=pd.concat([featuresfinal,featuresvcall],axis=0)
# featuresfinal = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\加入新特征矩阵.csv', encoding='gbk',index_col=0)
# # featuresfinal=pd.concat([featuresfinal,featuresredpacket],axis=0)
# # featuresfinal.to_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\加入新特征矩阵.csv')

# # featuresfinal=pd.concat([featuresfinal,featuresvideo],axis=0)
# # featuresfinal=pd.concat([featuresfinal,featureslocation],axis=0)
# # featuresfinal=pd.concat([featuresfinal,featuressubscript],axis=0)
# # featuresfinal=pd.concat([featuresfinal,featuresaudio],axis=0)
# #----------------Telegram---------------------------------------
# featuresfinal=pd.concat([featuresfinal,featuresTt],axis=0)
# featuresfinal=pd.concat([featuresfinal,featuresTp],axis=0)
# featuresfinal=pd.concat([featuresfinal,featuresTa],axis=0)
# featuresfinal=pd.concat([featuresfinal,featuresTv],axis=0)
# featuresfinal=pd.concat([featuresfinal,featuresTf],axis=0)
# #---------------------------------------------------------------

# # featuresfinal.drop(['Unnamed: 0'],axis = 1,inplace=True)
# # featuresfinal.drop(['Unnamed: 0.1'],axis = 1,inplace=True)
# # featuresfinal.drop(['Unnamed: 0.1.1'],axis = 1,inplace=True)
# # print(np.isnan(features).any())
# # featuresfinal=pd.concat([featuresfinal,features],axis=0)
# # featuresfinal = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\总特征矩阵.csv', encoding='gbk',index_col=0)
# # featuresTt = pd.read_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\telegram特征矩阵.csv', encoding='gbk',index_col=0)
# # featuresfinal.drop([10499],inplace=True)
# # featuresfinal.reset_index(drop = True)
# # featuresfinal.drop('Unnamed: 0.1', axis=1,inplace=True)
# featuresfinal.to_csv(r'C:\Users\wtr\Desktop\抓包数据\特征矩阵\telegram加上微信特征矩阵.csv')