import numpy as np

class CPCheck:
    def __init__(self, clients, partens, window=10, alpha=0.5, threshold=0.01, dataname="cifar10"):
        self.Win = window
        self.Norms = [0,0,0,0,0,0,0,0,0,0,0,0]
        self.Round = 0
        self.Clients = clients
        self.Threshold = threshold
        Pert = 0.75
        # clients :128 partens:16
        self.CMaxLim = int(clients * Pert) #96
        self.CMinLim = int(partens * 1) #16

        # self.CMaxLim = 16 #96
        # self.CMinLim = 8 #16

        self.Achieve = False
        self.MLim = 5

    def recvInfo(self,Norms):
        self.Round += 1
        AvgNorm = np.mean(Norms)
        self.Norms.append(AvgNorm)

    def WinCheck(self,CNum):
        if CNum == self.CMaxLim:
            self.Achieve = True
        OldNorm = max([np.mean(self.Norms[-self.Win-1:-1]),0.0000001])
        NewNorm = np.mean(self.Norms[-self.Win:])
        
        Is = 0
        if (NewNorm - OldNorm) / OldNorm > self.Threshold or self.Round <= self.MLim:
            Is = 1

        if Is == 1 and self.Achieve == False:
            # if (NewNorm - OldNorm) / OldNorm > self.Threshold * 2:
            #     CNum = min(self.CMaxLim,int(CNum * 1.5))
            # else:
            CNum = min(self.CMaxLim, CNum * 2)
        
        if Is == 0:
            # if (NewNorm - OldNorm) / OldNorm > self.Threshold * 0.5: 
            #     Reduce = max(int(CNum / 1.5),1)
            # else:
            Reduce = max(int(CNum / 2),1)
            CNum = max(self.CMinLim, CNum - Reduce)
        return CNum, Is
