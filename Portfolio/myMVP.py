import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pylab as plt
from scipy.optimize import minimize
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf

#snippet 2.1
#Marcenko-Pastur pdf
#q=T/N 
def mpPDF(var, q, pts):
    eMin, eMax = var*(1-(1./q)**.5)**2, var*(1+(1./q)**.5)**2 # calc lambda_minus, lambda_plus
    eVal = np.linspace(eMin, eMax, pts) #Return evenly spaced numbers over a specified interval. eVal='lambda'
    #Note: 1.0/2*2 = 1.0 not 0.25=1.0/(2*2)
    pdf = q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5 #np.allclose(np.flip((eMax-eVal)), (eVal-eMin))==True
    pdf = pd.Series(pdf, index=eVal)
    return pdf

#snippet 2.2
#Test Marcenko-Pastur Thm
def getPCA(matrix):
    # Get eVal, eVec from a Hermitian matrix
    eVal, eVec = np.linalg.eig(matrix) #complex Hermitian (conjugate symmetric) or a real symmetric matrix.
    indices = eVal.argsort()[::-1] #arguments for sorting eval desc
    eVal,eVec = eVal[indices],eVec[:,indices]
    eVal = np.diagflat(eVal) # identity matrix with eigenvalues as diagonal
    return eVal,eVec
    
def fitKDE(obs, bWidth=.15, kernel='gaussian', x=None):
    #Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    #print(len(obs.shape) == 1)
    if len(obs.shape) == 1: obs = obs.reshape(-1,1)
    kde = KernelDensity(kernel = kernel, bandwidth = bWidth).fit(obs)
    #print(x is None)
    if x is None: x = np.unique(obs).reshape(-1,1)
    #print(len(x.shape))
    if len(x.shape) == 1: x = x.reshape(-1,1)
    logProb = kde.score_samples(x) # log(density)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf

#snippet 2.3
def getRndCov(nCols, nFacts): #nFacts - contains signal out of nCols
    w = np.random.normal(size=(nCols, nFacts))
    cov = np.dot(w, w.T) #random cov matrix, however not full rank
    cov += np.diag(np.random.uniform(size=nCols)) #full rank cov
    return cov

def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov/np.outer(std,std)
    corr[corr<-1], corr[corr>1] = -1,1 #for numerical errors
    return corr
    
def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov     
    
#snippet 2.4 - fitting the marcenko-pastur pdf - find variance
#Fit error
def errPDFs(var, eVal, q, bWidth, pts=1000):
    var = var[0]
    pdf0 = mpPDF(var, q, pts) #theoretical pdf
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values) #empirical pdf
    sse = np.sum((pdf1-pdf0)**2)
    #print("sse:"+str(sse))
    return sse 
    
# find max random eVal by fitting Marcenko's dist
# and return variance
def findMaxEval(eVal, q, bWidth):
    out = minimize(lambda *x: errPDFs(*x), x0=np.array(0.5), args=(eVal, q, bWidth), bounds=((1E-5, 1-1E-5),))
    print("found errPDFs"+str(out['x'][0]))
    if out['success']: var = out['x'][0]
    else: var=1
    eMax = var*(1+(1./q)**.5)**2
    return eMax, var
    
# code snippet 2.5 - denoising by constant residual eigenvalue
# Remove noise from corr by fixing random eigenvalue
# Operation invariante to trace(Correlation)
# The Trace of a square matrix is the _Sum_ of its eigenvalues
# The Determinate of thematrix is the _Product_ of its eigenvalues
def denoisedCorr(eVal, eVec, nFacts):
    eVal_ = np.diag(eVal).copy()
    eVal_[nFacts:] = eVal_[nFacts:].sum()/float(eVal_.shape[0] - nFacts) #all but 0..i values equals (1/N-i)sum(eVal_[i..N]))
    eVal_ = np.diag(eVal_) #square matrix with eigenvalues as diagonal: eVal_.I
    corr1 = np.dot(eVec, eVal_).dot(eVec.T) #Eigendecomposition of a symmetric matrix: S = QΛQT
    corr1 = cov2corr(corr1) # Rescaling the correlation matrix to have 1s on the main diagonal
    return corr1
    
# code snippet 2.6 - detoning
# ref: mlfinlab/portfolio_optimization/risk_estimators.py
# This method assumes a sorted set of eigenvalues and eigenvectors.
# The market component is the first eigenvector with highest eigenvalue.
# it returns singular correlation matrix: 
# "the detoned correlation matrix is singualar, as a result of eliminating (at least) one eigenvector."
# Page 32
def detoned_corr(corr, eigenvalues, eigenvectors, market_component=1):
    """
    De-tones the de-noised correlation matrix by removing the market component.
    The input is the eigenvalues and the eigenvectors of the correlation matrix and the number
    of the first eigenvalue that is above the maximum theoretical eigenvalue and the number of
    eigenvectors related to a market component.
    :param corr: (np.array) Correlation matrix to detone.
    :param eigenvalues: (np.array) Matrix with eigenvalues on the main diagonal.
    :param eigenvectors: (float) Eigenvectors array.
    :param market_component: (int) Number of fist eigevectors related to a market component. (1 by default)
    :return: (np.array) De-toned correlation matrix.
    """
    
    # Getting the eigenvalues and eigenvectors related to market component
    eigenvalues_mark = eigenvalues[:market_component, :market_component]
    eigenvectors_mark = eigenvectors[:, :market_component]
    
    # Calculating the market component correlation
    corr_mark = np.dot(eigenvectors_mark, eigenvalues_mark).dot(eigenvectors_mark.T)
    
    # Removing the market component from the de-noised correlation matrix
    corr = corr - corr_mark
    
    # Rescaling the correlation matrix to have 1s on the main diagonal
    corr = cov2corr(corr)
    
    return corr
            
def test_detone():
    # ------ Test detone --------
    cov_matrix = np.array([[0.01, 0.002, -0.001],
                           [0.002, 0.04, -0.006],
                           [-0.001, -0.006, 0.01]])
    cor_test = np.corrcoef(cov_matrix, rowvar=0) 
    eVal_test, eVec_test = getPCA(cor_test)
    eMax_test, var_test = findMaxEval(np.diag(eVal_test), q, bWidth=.01)
    nFacts_test = eVal_test.shape[0]-np.diag(eVal_test)[::-1].searchsorted(eMax_test)   
    corr1_test = denoisedCorr(eVal_test, eVec_test, nFacts_test) 
    eVal_denoised_test, eVec_denoised_test = getPCA(corr1_test)
    corr_detoned_denoised_test = detoned_corr(corr1_test, eVal_denoised_test, eVec_denoised_test)       
    eVal_detoned_denoised_test, _ = getPCA(corr_detoned_denoised_test)     
    np.diag(eVal_denoised_test)
    np.diag(eVal_detoned_denoised_test)
    
    expected_detoned_denoised_corr = np.array([ 1.56236229e+00,  1.43763771e+00, -2.22044605e-16])    
    
    np.testing.assert_almost_equal(np.diag(eVal_detoned_denoised_test), expected_detoned_denoised_corr, decimal=4)
    np.testing.assert_almost_equal(sum(np.diag(eVal_denoised_test)), sum(np.diag(eVal_detoned_denoised_test)), decimal=4 )
    
##SNIPPET 2.6 DENOISING BY TARGETED SHRINKAGE
def denoisedCorr2(eVal,eVec,nFacts,alpha=0):
    # Remove noise from corr through targeted shrinkage
    eValL,eVecL=eVal[:nFacts,:nFacts],eVec[:,:nFacts]
    eValR,eVecR=eVal[nFacts:,nFacts:],eVec[:,nFacts:]
    corr0=np.dot(eVecL,eValL).dot(eVecL.T)
    corr1=np.dot(eVecR,eValR).dot(eVecR.T)
    corr2=corr0+alpha*corr1+(1-alpha)*np.diag(np.diag(corr1))
    return corr2    


import numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
#- - - - - - - -- - - - - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - - - - - -- - -
## SNIPPET 4.1 BASE CLUSTERING
def clusterKMeansBase(corr0,maxNumClusters=10,n_init=10):
    x,silh=((1-corr0.fillna(0))/2.)**.5,pd.Series()# observations matrix
    for init in range(n_init):
        for i in xrange(2,maxNumClusters+1):
            kmeans_=KMeans(n_clusters=i,n_jobs=1,n_init=1)
            kmeans_=kmeans_.fit(x)
        silh_=silhouette_samples(x,kmeans_.labels_)
        stat=(silh_.mean()/silh_.std(),silh.mean()/silh.std())
        if np.isnan(stat[1]) or stat[0]>stat[1]:
            silh,kmeans=silh_,kmeans_
    newIdx=np.argsort(kmeans.labels_)
    corr1=corr0.iloc[newIdx] # reorder rows
    corr1=corr1.iloc[:,newIdx] # reorder columns
    clstrs={i:corr0.columns[np.where(kmeans.labels_==i)[0]].tolist() \
        for i in np.unique(kmeans.labels_) } # cluster members
    silh=pd.Series(silh,index=x.index)
    
    return corr1,clstrs,silh

##
## based on https://github.com/andreybabynin/HRP/blob/master/code/hierarchical.py
##
class HRP:
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as mpl
    import scipy.spatial.distance as ssd
    import scipy.cluster.hierarchy as sch

    @classmethod
    def correlDist(cls, corr):
        # A distance matrix based on correlation, where 0<=d[i,j]<=1
        # This is a proper distance metric
        dist = ((1 - corr) / 2.)**.5  # distance matrix
        return dist

    @classmethod
    def getIVP(cls, cov, **kargs):
        # Compute the inverse-variance portfolio
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    @classmethod
    def getQuasiDiag(cls, link):
        # Sort clustered items by distance
        link = link.astype(int)
        sortIx = pd.Series([link[-1, 0], link[-1, 1]])
        numItems = link[-1, 3]  # number of original items
        while sortIx.max() >= numItems:
            sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
            df0 = sortIx[sortIx >= numItems]  # find clusters
            i = df0.index
            j = df0.values - numItems
            sortIx[i] = link[j, 0]  # item 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sortIx = sortIx.append(df0)  # item 2
            sortIx = sortIx.sort_index()  # re-sort
            sortIx.index = range(sortIx.shape[0])  # re-index
        return sortIx.tolist()

    @classmethod
    def getClusterVar(cls, cov,cItems):
        # Compute variance per cluster
        cov_=cov.loc[cItems,cItems] # matrix slice
        w_=cls.getIVP(cov_).reshape(-1,1)
        cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
        return cVar

    @classmethod
    def corr2cov(cls, corr,std):
        cov=corr*np.outer(std,std)
        return cov

    @classmethod
    def getRecBipart(cls, cov, sortIx):
        # Compute HRP alloc
        w = pd.Series(1, index=sortIx)
        cItems = [sortIx]  # initialize all items in one cluster
        while len(cItems) > 0:
            cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
            for i in range(0, len(cItems), 2):  # parse in pairs
                cItems0 = cItems[i]  # cluster 1
                cItems1 = cItems[i + 1]  # cluster 2
                cVar0 = cls.getClusterVar(cov, cItems0)
                cVar1 = cls.getClusterVar(cov, cItems1)
                alpha = 1 - cVar0 / (cVar0 + cVar1)
                w[cItems0] *= alpha  # weight 1
                w[cItems1] *= 1 - alpha  # weight 2
        return w


    @classmethod
    def getHRP(cls, cov, corr, link = None):
        # Construct a hierarchical portfolio
        if link is None:
            dist = cls.correlDist(corr)
            link = sch.linkage(dist, 'single')
        if link is not None:
            cov = cls.corr2cov(corr, np.diag(cov)**.5)
            link = np.array([list(i) for i in link])

        # graph
        '''
        dn = sch.dendrogram(link, labels=list_shares, leaf_rotation=90, distance_sort='descending',
                            count_sort= 'descendent')
        plt.show()
        '''
        sortIx = cls.getQuasiDiag(link)
        sortIx = corr.index[sortIx].tolist()
        hrp = cls.getRecBipart(cov, sortIx)
        return hrp.sort_index()

from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch

def makeNewOutputs(corr, clsters, clstrs2):
    # Copying clusters + combining
    clstrsNew = {}
    for i in clstrs.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs[i])
    for i in clstrs2.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs2[i])
    
    # Reordering according to cluster
    newIdx = [j for i in clstrsNew for j in clstrsNew[i]]
    corrNew = corr0.loc[newIdx, newIdx]
    x = ((1-corr0.fillna(0))/2.)**.5
    kmeans_labels = np.zeros(len(x.columns))
    
    # Creating cluster array
    for i in clstrsNew.keys():
        idxs = [x.index.get_loc(k) for k in clstrsNew[i]]
        kmeans_labels[idxs] = i
        
    silhNew = pd.Series(silhouette_samples(x,kmeans_labels),index=x.index)
    return corrNew, clstrsNew, silhNew
        
def clusterKMeansTop(corr0, maxNumClusters = None, n_init=10):
    if maxNumClusters == None:
        maxNumClusters = corr0.shape[1]-1
    corr1, clstrs, silh = clusterKMeansBase(corr0, maxNumClusters = min(maxNumClusters, corr0.shape[1]-1),n_init=n_init)
    # Calculating Quality for each cluster
    clusterTstats = {i:np.mean(silh[clstrs[i]])/ np.std(silh[clstrs[i]]) for i in clstrs.keys()}
    tStatMean = sum(clusterTstats.values())/len(clusterTstats)
    
    # Selecting the clusters for redo
    redoClusters = [i for i in clusterTstats.keys() if clusterTstats[i]<tStatMean]
    

    
    if len(redoClusters) <= 1:
        return corr1, clstrs, silh
    else:
        keysRedo = [j for i in redoClusters for j in clstrs[i]]
        corrTmp = corr0.loc[keysRedo, keysRedo]
        corr2, clstrs2, silh2 = clusterKMeansTop(corrTmp, maxNumClusters = min(maxNumClusters, corrTmp.shape[1]-1),n_init=n_init)
        corrNew, clstrsNew, silhNew = makeNewOutputs(corr0, {i:clstrs[i] for i in clstrs.keys() if i not in redoClusters}, clstrs2)      
        newTstatMean = np.mean([np.mean(silhNew[clstrsNew[i]])/np.std(silhNew[clstrsNew[i]]) for i in clstrsNew.keys()])
        if newTstatMean <= tStatMean:
            return corr1, clstrs, silh
        else:
            return corrNew, clstrsNew, silhNew
    

def clusterKMeansTop_silhouette(corr0, maxNumClusters = None, n_init=10):
    if maxNumClusters == None:
        maxNumClusters = corr0.shape[1]-1
    corr1, clstrs, silh = clusterKMeansBase_silhouette(corr0, maxNumClusters = min(maxNumClusters, corr0.shape[1]-1),n_init=n_init)
    # Calculating Quality for each cluster
    clusterTstats = {i:np.mean(silh[clstrs[i]]) for i in clstrs.keys()}
    tStatMean = sum(clusterTstats.values())/len(clusterTstats)
    
    # Selecting the clusters for redo
    redoClusters = [i for i in clusterTstats.keys() if clusterTstats[i]<tStatMean]

    if len(redoClusters) <= 1:
        return corr1, clstrs, silh
    else:
        keysRedo = [j for i in redoClusters for j in clstrs[i]]
        print('Keys : ',len(keysRedo))
        corrTmp = corr0.loc[keysRedo, keysRedo]
        tStatMean = np.mean([clusterTstats[i] for i in redoClusters])
        corr2, clstrs2, silh2 = clusterKMeansTop_silhouette(corrTmp, maxNumClusters = min(maxNumClusters, corrTmp.shape[1]-1),n_init=n_init)
        corrNew, clstrsNew, silhNew = makeNewOutputs(corr0, {i:clstrs[i] for i in clstrs.keys() if i not in redoClusters}, clstrs2)
        
        newTstatMean = np.mean([np.mean(silhNew[clstrsNew[i]]) for i in clstrsNew.keys()])
        if newTstatMean <= tStatMean:
            return corr1, clstrs, silh
        else:
            return corrNew, clstrsNew, silhNew
    
    
def clusterKMeansBase(corr0, maxNumClusters=10, n_init=10):
    # calculating distance and create empty array
    x, silh = ((1-corr0.fillna(0))/2.)**0.5, pd.Series(np.nan)
    score_history = []
    for init in range(n_init):
        for i in range(3, maxNumClusters+1):
            kmeans_ = KMeans(n_clusters=i, n_init=1)
            kmeans_ = kmeans_.fit(x)
            silh_ = silhouette_samples(x, kmeans_.labels_)
            stat = (silh_.mean()/silh_.std(), silh.mean()/silh.std())
            score_history.append(stat)
            if np.isnan(stat[1]) or stat[0] > stat[1]:
                silh, kmeans = silh_, kmeans_
    newIdx = np.argsort(kmeans.labels_)
    corr1 = corr0.iloc[newIdx]

    corr1 = corr1.iloc[:, newIdx]
    clusters = {i: corr0.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in np.unique(kmeans.labels_)}
    silh = pd.Series(silh, index=x.index)
    return corr1, clusters, silh


def clusterKMeansBase_silhouette(corr0, maxNumClusters=10, n_init=10):
    # calculating distance and create empty array
    x, silh = ((1-corr0.fillna(0))/2.)**0.5, pd.Series(np.nan)
    score_history = []
    for init in range(n_init):
        for i in range(3, maxNumClusters+1):
            kmeans_ = KMeans(n_clusters=i, n_init=1)
            kmeans_ = kmeans_.fit(x)
            silh_ = silhouette_samples(x, kmeans_.labels_)
            stat = (silh_.mean(), silh.mean())
            if np.isnan(stat[1]) or stat[0] > stat[1]:
                silh, kmeans = silh_, kmeans_
    newIdx = np.argsort(kmeans.labels_)
    corr1 = corr0.iloc[newIdx]

    corr1 = corr1.iloc[:, newIdx]
    clusters = {i: corr0.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in np.unique(kmeans.labels_)}
    silh = pd.Series(silh, index=x.index)
    return corr1, clusters, silh


def clusterKMeansBase_spectral(corr0, maxNumClusters=10, n_init=10):
    # calculating distance and create empty array
    x, silh = ((1-corr0.fillna(0))/2.)**0.5, pd.Series(np.nan)
    score_history = []
    for init in range(n_init):
        for i in range(10, maxNumClusters+1):
            kmeans_ = SpectralClustering(n_clusters=i, n_init=1)
            kmeans_ = kmeans_.fit(x)
            silh_ = silhouette_samples(x, kmeans_.labels_)
            stat = (silh_.mean()/silh_.std(), silh.mean()/silh.std())
            score_history.append(stat)
            if np.isnan(stat[1]) or stat[0] > stat[1]:
                silh, kmeans = silh_, kmeans_
    newIdx = np.argsort(kmeans.labels_)
    corr1 = corr0.iloc[newIdx]

    corr1 = corr1.iloc[:, newIdx]
    clusters = {i: corr0.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in np.unique(kmeans.labels_)}
    silh = pd.Series(silh, index=x.index)
    return corr1, clusters, silh


def clusterKMeansTop_spectral(corr0, maxNumClusters = None, n_init=10):
    if maxNumClusters == None:
        maxNumClusters = corr0.shape[1]-1
    corr1, clstrs, silh = clusterKMeansBase_spectral(corr0, maxNumClusters = min(maxNumClusters, corr0.shape[1]-1),n_init=n_init)
    # Calculating Quality for each cluster
    clusterTstats = {i:np.mean(silh[clstrs[i]])/ np.std(silh[clstrs[i]]) for i in clstrs.keys()}
    tStatMean = sum(clusterTstats.values())/len(clusterTstats)
    
    # Selecting the clusters for redo
    redoClusters = [i for i in clusterTstats.keys() if clusterTstats[i]<tStatMean]
    

    
    if len(redoClusters) <= 1:
        return corr1, clstrs, silh
    else:
        keysRedo = [j for i in redoClusters for j in clstrs[i]]
        corrTmp = corr0.loc[keysRedo, keysRedo]
        corr2, clstrs2, silh2 = clusterKMeansTop_spectral(corrTmp, maxNumClusters = min(maxNumClusters, corrTmp.shape[1]-1),n_init=n_init)
        corrNew, clstrsNew, silhNew = makeNewOutputs(corr0, {i:clstrs[i] for i in clstrs.keys() if i not in redoClusters}, clstrs2)      
        newTstatMean = np.mean([np.mean(silhNew[clstrsNew[i]])/np.std(silhNew[clstrsNew[i]]) for i in clstrsNew.keys()])
        if newTstatMean <= tStatMean:
            return corr1, clstrs, silh
        else:
            return corrNew, clstrsNew, silhNew

def clusterKMeansBase_Birch(corr0, maxNumClusters=10, n_init=10):
    # calculating distance and create empty array
    x, silh = ((1-corr0.fillna(0))/2.)**0.5, pd.Series(np.nan)
    score_history = []
    for init in range(n_init):
        for i in range(10, maxNumClusters+1):
            kmeans_ = Birch(n_clusters=i)
            kmeans_ = kmeans_.fit(x)
            silh_ = silhouette_samples(x, kmeans_.labels_)
            stat = (silh_.mean()/silh_.std(), silh.mean()/silh.std())
            score_history.append(stat)
            if np.isnan(stat[1]) or stat[0] > stat[1]:
                silh, kmeans = silh_, kmeans_
    newIdx = np.argsort(kmeans.labels_)
    corr1 = corr0.iloc[newIdx]

    corr1 = corr1.iloc[:, newIdx]
    clusters = {i: corr0.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in np.unique(kmeans.labels_)}
    silh = pd.Series(silh, index=x.index)
    return corr1, clusters, silh


def clusterKMeansTop_Birch(corr0, maxNumClusters = None, n_init=10):
    if maxNumClusters == None:
        maxNumClusters = corr0.shape[1]-1
    corr1, clstrs, silh = clusterKMeansBase_Birch(corr0, maxNumClusters = min(maxNumClusters, corr0.shape[1]-1))
    # Calculating Quality for each cluster
    clusterTstats = {i:np.mean(silh[clstrs[i]])/ np.std(silh[clstrs[i]]) for i in clstrs.keys()}
    tStatMean = sum(clusterTstats.values())/len(clusterTstats)
    
    # Selecting the clusters for redo
    redoClusters = [i for i in clusterTstats.keys() if clusterTstats[i]<tStatMean]
    

    
    if len(redoClusters) <= 1:
        return corr1, clstrs, silh
    else:
        keysRedo = [j for i in redoClusters for j in clstrs[i]]
        corrTmp = corr0.loc[keysRedo, keysRedo]
        corr2, clstrs2, silh2 = clusterKMeansTop_Birch(corrTmp, maxNumClusters = min(maxNumClusters, corrTmp.shape[1]-1))
        corrNew, clstrsNew, silhNew = makeNewOutputs(corr0, {i:clstrs[i] for i in clstrs.keys() if i not in redoClusters}, clstrs2)      
        newTstatMean = np.mean([np.mean(silhNew[clstrsNew[i]])/np.std(silhNew[clstrsNew[i]]) for i in clstrsNew.keys()])
        if newTstatMean <= tStatMean:
            return corr1, clstrs, silh
        else:
            return corrNew, clstrsNew, silhNew        
# /---------------------------------

def min_var_weight(data):
    """
    input: 
         data: df return data
    output: 
         minimum variance weight      
    """

    sigma_inv = np.linalg.inv(data.cov())
    a = np.ones((len(sigma_inv), 1))

    optimal_weight = (np.dot(sigma_inv, a))  / (np.dot(a.T, np.dot(sigma_inv, a)))

    return optimal_weight


def cum_return(ret, weight):
    """
    input: 
         data: df return data
    output: 
         cumulative return 
    """

    ret_ = np.dot(((1+ret).cumprod(axis=0) -1), weight).flatten()

    return pd.Series(data = ret_ , index=ret.index)    

import numpy as np 
def largest_eigvalue_weight(ret):
    eVal, eVec = np.linalg.eigh(ret.cov())
    idx = eVal.argsort()[::-1]
    eVal = eVal[idx]
    eVec = eVec[:, idx]

    eVec = eVec[:, 0] / eVec[:, 0].sum()

    return ret.columns[idx][0], eVec

def smallest_eigvalue_weight(ret):
    eVal, eVec = np.linalg.eigh(ret.cov())
    idx = eVal.argsort()[::-1]
    eVal = eVal[idx]
    eVec = eVec[:, idx]

    eVec = eVec[:, -1] / eVec[:, -1].sum(axis=0)

    return ret.columns[idx][-1], eVec

        
   
def volatility_opt_fun(weight, ret):
    weight = np.array(weight)
    p_ret = np.sum(ret.mean() * weight) * 250
    p_sigma = np.sqrt(np.dot(weight.T, np.dot(ret.cov() * 250, weight)))
    
    return np.array(p_sigma **2)

import scipy.optimize as opt  
def get_weights_wo_short_sale(ret):
    noa = len(ret.columns)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) -1}]   # sum(weights)=1
    bounds = tuple((0,1) for x in range(noa))
    vol_out = opt.minimize(lambda *x: volatility_opt_fun(*x), noa * [1. / noa], args= (ret), 
                        method='SLSQP', bounds=bounds, constraints=constraints)    
    if vol_out['success']:
        weights = vol_out['x']                    
    else:
        weights = np.nan  

    return weights        

def get_eMaxMin(data, var=1):
    var = 1
    q = data.shape[0] / data.shape[1]
    eMin,eMax=var*(1-(1./q)**.5)**2,var*(1+(1./q)**.5)**2
    return eMin, eMax


def get_month_index(data):
    tmp = data.copy()
    tmp['year'] = tmp.index.year 
    tmp['month'] = tmp.index.month 
    month_idx = np.array(tmp.groupby(['year', 'month'])['MMM'].count().cumsum().values)

    return month_idx

def random_matrix_info(df, lookback=6):
            
    i = 0
    lookback = lookback
    month_idx = get_month_index(df)

    ts  = []
    num_max = []
    num_min  = []
    eMaxLst = []
    eMaxVar = []

    for i in range(len(month_idx)-lookback-1):

        start = month_idx[i]
        end = month_idx[i+lookback]
        df_1 = df.iloc[start:end, :]
        LeVal0, LeVec0 = getPCA(df_1.corr())
        LeMin, LeMax = get_eMaxMin(df_1)
        LeVal01 = np.diag(LeVal0)
        try:
            ts.append(df.index[end])
            num_max.append(len(LeVal01[LeVal01 < LeMin]))
            num_min.append(len(LeVal01[LeVal01 > LeMax]))
            eMaxLst.append(LeVal0.max())

            eMaxVar.append((LeVal01[LeVal01 > LeMax].sum() / LeVal01.sum()))

        except IndexError as e:
            pass 
    df_ex = pd.DataFrame().assign(num_max=num_max, num_min=num_min, eMaxLst=eMaxLst, eMaxVar=eMaxVar)
    df_ex = df_ex.set_index(pd.to_datetime(ts))

    return df_ex 

def random_mat_plot(df_ex, colName, lookback=6):
    fig = plt.figure(figsize=(15,6))

    ax = fig.add_subplot(121)
    df_ex.dow.pct_change().plot(ax=ax, grid=False, c='k', label='Dow return')
    ax.legend()
    ax1 = ax.twinx()
    df_ex[colName].plot(ax=ax1, grid=False, c='b', label=colName, alpha=0.3)
    ax1.legend(loc='lower center')
    ax1.axhline(df_ex[colName].mean(), ls='--', c='r');
    plt.title(f'Random Matrix Stats ({colName}) with Dow (lookback={lookback})');

    ax2 = fig.add_subplot(122)
    df_ex.dow.plot(ax=ax2, grid=False, c='k', label='Dow')
    ax2.legend()
    ax21 = ax2.twinx()
    df_ex[colName].plot(ax=ax21, grid=False, c='b', label=colName, alpha=0.3)
    ax21.legend(loc='lower center')
    ax21.axhline(df_ex[colName].mean(), ls='--', c='r')
    plt.title(f'Random Matrix Stats ({colName}) with Dow (lookback={lookback})');
    