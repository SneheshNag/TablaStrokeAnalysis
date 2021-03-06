import essentia.standard as estd
loader=estd.EqloudLoader(filename='')
audio=loader()
energy=estd.Energy()
en=[]
for frame in estd.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
  en.append(energy(frame))
emax=max(en)
ep=en.index(emax)
th=emax/2
e=[]
for frame in estd.FrameGenerator(audio[:0.6*44100], frameSize=1024, hopSize=512, startFromZero=True):
  e.append(energy(frame))
for i in range(0,len(e)-1):
  d=e[i+1]-e[i]
  if d > 0.01:
    p=e.index(e[i+1])
    if (ep-p) > 2:
      en=e[:p]+en[ep:]
      break
