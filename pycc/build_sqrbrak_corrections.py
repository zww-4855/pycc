import numpy as np

def build_FOsqrBrakTriples(): 
    t3resid=wicked_T3corr.build_T3_secondO(self.g,self.o,self.v,self.t2)
    t3resid=wicked_T3corr.antisym_T3(t3resid,self.nocc,self.nvirt)
    t3residOrigContract=t3resid
    t3resid=t3resid.transpose(3,4,5,0,1,2)
    t3=t3resid*self.denoms["D3aa"]
    t3Contract=t3
    t3=t3.transpose(3,4,5,0,1,2)

    energy=wicked_T3corr.getE_squareBrackT(self.g,self.o,self.v,t3,t2_dag)
    print('[T]-based triples energy correction to CC is:',energy)
    return energy

def build_FOsqrBrakSingles():
    netT1=wicked_T3corr.get_netT1_fromT2(self.g,self.o,self.v,self.t2)
    t1_bar=netT1.transpose(1,0)*self.denoms["D1aa"]
    print('shapes',np.shape(netT1),np.shape(t1_bar))
    print('check on [S]',np.einsum("ia,ai->",t1_bar,netT1))
    t1_bar=t1_bar.transpose(1,0)
    t2_bar_resid=wicked_T3corr.get_T2resid_fromT1bar(self.g,self.o,self.v,t1_bar)
    t2_bar_resid=wicked_T3corr.antisym_T2(t2_bar_resid,self.nocc,self.nvirt)
    singles_correction=0.250000000 * np.einsum("abij,ijab->",t2_dag,t2_bar_resid,optimize="optimal")
    print('[S] singles correction to CCD:',singles_correction)
    return singles_correction



def build_T3_secondO(g,o,v,t2):
    #rooovvv = np.zeros((nocc,nocc,nocc,nvir,nvir,nvir))
    rooovvv = -0.250000000 * np.einsum("ilab,jklc->ijkabc",t2,g[o,o,o,v],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("ijad,kdbc->ijkabc",t2,g[o,v,v,v],optimize="optimal")
    return rooovvv



