import os, sys, essentia, math
import seaborn as sns
import matplotlib.pyplot as plt
import essentia.standard as estd
from essentia.standard import *
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def func_centr(audio,rate):
  spec=[]
  for frame in FrameGenerator(audio, frameSize=1024, hopSize=450, startFromZero=True):
    spec.append(Spectrum()(Windowing(type = 'hamming')(frame)))
  spec=np.array(spec)
  spec=spec.mean(axis=0)
  return (estd.Centroid(range=len(spec))(spec))*rate/1024.

def band_decay(audio, rate, band_no):
  def freq(audio, rate):
    audio=audio[int(0.001*rate):int(1.0*rate)]
    #p,pc=estd.PitchMelodia()(audio)
    #b=p>50
    #ff=(p[np.where(pc==pc.max())]).mean()
    ff,con=PitchYin()(audio)
    print ff
    return ff
    
  def exp_env(audio, step):
    def func(x, a, c):
      return a*np.exp(-c*x)
    max_pos=np.argmax(audio)
    audio1=audio
    audio = audio[np.argmax(audio):]
    step = int(step)
    audio = np.abs(audio)
    envelope = []
    env_x = []
    for i in range(0, len(audio), step):
        env_x += [i+np.argmax(audio[i:i+step])]
        envelope += [np.max(audio[i:i+step])]
    env_x=np.array(env_x)
    envelope = np.array(envelope)
    try:
        popt, pcov = curve_fit(func, env_x, envelope, p0=(1, 1e-3))
    except RuntimeError:
        popt = [envelope[0], 0]
        pcov = []
    xx = np.arange(0, len(audio), 1)
    yy = func(xx, *popt)
    xx=xx+max_pos
    xx=np.append(np.arange(0,max_pos),xx)
    yy=np.append(np.zeros(max_pos),yy)
    plt.plot(xx, yy)
    plt.plot(xx,audio1, color='green')
    start = env_x[np.where(envelope==envelope.max())[0]]
    nf1 = envelope[0:5].mean()
    locs = np.where(envelope<0.1*envelope.max())[0]
    if len(locs)<1:
      stop1 = env_x[-1]
    else:
      stop1 = env_x[locs[np.where(locs > np.where(envelope==envelope.max())[0])][0]]
    locs = np.where(envelope<0.01*envelope.max())[0]
    if len(locs)<1:
      stop2 = env_x[-1]
    else:
      stop2 = env_x[locs[np.where(locs > np.where(envelope==envelope.max())[0])][0]]
    plt.xlabel('Samples')
    plt.ylabel('Absolute Amplitude')
    plt.axis([0,140000, 0, 0.20])
    plt.figure()
    en_mod = np.array(audio1-yy, dtype='float32')
    if len(en_mod)%2>0:
      en_mod=en_mod[:-1]
    print 44100./len(en_mod)
    spectrum=estd.Spectrum()(en_mod)
    plt.plot(spectrum)
    plt.show()
    return stop1-start
  
  if band_no==0:
    x=audio
    popt1 = exp_env(x, 0.05*rate)
    popt1=popt1.tolist()
    popt1.append(freq(audio,rate))
    return np.array(popt1)
  else:
    popt1 = exp_env(audio, 0.05*rate)
    return np.array(popt1)
  
def sustain_durn(audio,rate):
  audio = audio[np.argmax(audio):]
  print len(audio)
  st_en=[]
  for frame in FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
    frame_win=Windowing(type='hamming')(frame)
    st_en.append(pow(frame_win,2).sum())
  st_en=np.array(st_en)
  x = np.arange(len(st_en))*512
  plt.plot(x,st_en)
  plt.show()
  start=np.where(st_en==st_en.max())[0]
  stop=(np.where(st_en < st_en.max()*0.01)[0])[0]
  return stop-start

path = '/media/hitesh/Work/Rohit_RA_2017/frsm_sets'
stroke = sys.argv[1]
category=['Good', 'Bad']


rate=44100.

sets = os.listdir(path)

data_df = pd.DataFrame({})

for seti in sets:
  if not os.path.isdir(path+'/'+seti):
    continue
  for categ in category:
    if not os.path.exists(path+'/'+seti+'/'+stroke+'/'+categ):
      continue
    files_categ = os.listdir(path+'/'+seti+'/'+stroke+'/'+categ) 
    for wave in files_categ:
      fileName = path+'/'+seti+'/'+stroke+'/'+categ+'/'+wave
      #print fileName
      audio=estd.EqloudLoader(filename=fileName)()
      dict_temp={}
      dict_temp['Filename']=wave
      dict_temp['Set']=seti
      dict_temp['Category']=categ
      dict_temp['Decay Rate'] = band_decay(audio, rate, 1)[0]
      dict_temp['Sustain'] = sustain_durn(audio, rate)
      data_df=data_df.append(dict_temp, ignore_index=True)

print data_df

plt.title(stroke)
sns.swarmplot(x="Set", y="Centroid1", hue="Category", data=data_df, palette="Set2", dodge=True)
sns.boxplot(x="Set", y="Centroid1", hue="Category", data=data_df, palette="Set2", dodge=True, boxprops={'facecolor':'None'})

plt.figure()
plt.title(stroke)
ax1 = sns.swarmplot(x="Set", y="Centroid2", hue="Category", data=data_df, palette="Set2", dodge=True)
ax1 = sns.boxplot(x="Set", y="Centroid2", hue="Category", data=data_df, palette="Set2", dodge=True, boxprops={'facecolor':'None'})

plt.figure()
plt.title(stroke)
ax1 = sns.swarmplot(x="Set", y="Centroid3", hue="Category", data=data_df, palette="Set2", dodge=True)
ax1 = sns.boxplot(x="Set", y="Centroid3", hue="Category", data=data_df, palette="Set2", dodge=True, boxprops={'facecolor':'None'})

plt.figure()
plt.title(stroke)
ax1 = sns.swarmplot(x="Set", y="Decay Rate", hue="Category", data=data_df, palette="Set2", dodge=True)
ax1 = sns.boxplot(x="Set", y="Decay Rate", hue="Category", data=data_df, palette="Set2", dodge=True, boxprops={'facecolor':'None'})

plt.show()
