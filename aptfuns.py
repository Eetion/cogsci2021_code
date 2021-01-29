################################################################################
#   AUTHOR: GREGORY HENSELMAN-PETRUSEK
#   CREATED: 2020
#   LAST MODIFIED: 2021
################################################################################

################################################################################
#   DESCRIPTION
#   THIS CODE PROVIDES UTILITIES TO ANALYZE MULTITASKING DATA.
################################################################################



import numpy as np
import itertools
import copy
import sklearn



################################################################################
#   PERFORMANCE LISTS + STRINGS
################################################################################

def pl__pstring(pl):
    """
    :param pl: performance list
    :return: a string that encodes the list
    """
    a                               =   np.array(pl)
    a                               =   np.ndarray.flatten(a)
    pstring                         =   ','.join(str(x) for x in a)
    return pstring

def pstring__pl(pstring):
    a                               =   np.array([int(x) for x in pstring.split(',')])
    a                               =   a.reshape(-1,2)
    pl                              =   [list(a[p]) for p in range(a.shape[0])]
    return pl

#   TEST PASSED 20200317
def check__pl__pstring():
    pll                             =  [[[1,2]],  [[1,2],[3,4]]]
    for pl in pll:
        if not pstring__pl(pl__pstring(pl)) == pl:
            print("error: please check pl__pstring")
            print(pl)
    print("test passed")

################################################################################
#   INDEXING
################################################################################

def dsl_d__ndcomplementfeaturevectors(dsl,d):
    return (np.prod(dsl)/dsl[d]).astype(int)

def dsl_d_f__inda(dsl,d,f):
    """
    :param dsl: array   array of dimension sizes
    :param d:   int     dimension
    :param f:   int     feature
    :return: array of integers; the indices identify trials where dimension d has feature value f; the entries are in sorted order
    """
    fill_size               =   np.prod(dsl[d+1:]).astype(int)
    block_size              =   np.prod(dsl[d:]).astype(int)
    num_blocks              =   np.prod(dsl[:d]).astype(int) # recall that due to indexing conventions this excludes d

    low0                    =   fill_size*f
    hig0                    =   fill_size*(f+1)
    inda                    =   np.zeros((fill_size*num_blocks),dtype=int)
    for p in range(num_blocks):
        inda[p*fill_size : (p+1)*fill_size]   =   np.arange(low0 + p*block_size, hig0 + p*block_size)
    return inda

def check__dsl_d_f__inda():
    """
    TEST PASSED 20200421
    """
    A                       =   np.array(list(itertools.product(range(1),range(3),range(4),range(2),range(7),range(1))))
    dsl                     =   [1,3,4,2,7,1]
    for d in range(6):
        for f in range(dsl[d]):
            ind_true        =   np.nonzero(A[:,d]==f)[0]
            ind_test        =   dsl_d_f__inda(dsl,d,f)
            if not np.array_equal(ind_true,ind_test):
                print("error in check__dsl_d_f__inda")
                print(f"d = {d}, f = {f}, n_rep_ideal = {np.prod(dsl)/dsl[d]}")
                print(f"ind_true.shape = {ind_true.shape}, ind_test.shape = {ind_test.shape}")
                return
    print("test passed")

def dsl_d__sortperma(dsl,d,target='column'):
    if target               ==  'column':
        return np.concatenate([dsl_d_f__inda(dsl,d,f) for f in range(dsl[d])])
    elif target             ==  'column_complement':
        cperma              =   dsl_d__sortperma(dsl,d,target='column')
        block_size          =   int(np.prod(dsl[:d])*np.prod(dsl[d+1:])) # have to convert to integer since np.prod([]) is a float, not an integer
        operma              =   np.concatenate([cperma[x::block_size] for x in range(block_size)])
        return operma

def check__dsl_d__sortperma():
    """
    TEST PASSED 20200423
    """
    dsl                     =   [1,3,4,2,7,1]
    nd                      =   len(dsl)
    A                       =   np.array(list(itertools.product(*[range(x) for x in dsl])))
    for d in range(nd):
        I                   =   np.concatenate((np.arange(d),np.arange(d+1,nd)))

        vals_d              =   np.arange(dsl[d])
        vals_notd           =   np.array(list(itertools.product(*[range(dsl[x]) for x in I])))

        nvals_d             =   dsl[d]
        nvals_notd          =   np.prod([dsl[x] for x in I])

        #   TEST 1
        ind_test            =   dsl_d__sortperma(dsl,d)
        want                =   np.tile( vals_notd, (nvals_d,1))
        have                =   A[ind_test][:,I]
        if not np.array_equal(want ,have ):
            print('error 1')
            print(want)
            print(have)
        want                =   np.repeat(vals_d, nvals_notd, axis = 0)
        have                =   A[ind_test][:,d]
        if not np.array_equal(want ,have):
            print('error 2')
            print(want)
            print(have)
        #   TEST 2
        ind_test            =   dsl_d__sortperma(dsl,d, target='column_complement')
        want                =   np.repeat(vals_notd, nvals_d, axis = 0)
        have                =   A[ind_test][:,I]
        if not np.array_equal(want ,have):
            print('error 3')
            print(want)
            print(have)
        want                =   np.tile( vals_d, (nvals_notd))
        have                =   A[ind_test][:,d]
        if not np.array_equal(want, have):
            print('error 4')
            print(want)
            print(have)

    print("test passed")


#-------------------------------------------------------------------------------
#   APTD --> ARRAY
#-------------------------------------------------------------------------------


def aptd__totallabela(aptd):
    """
    :param aptd:
    :return:
    """
    nuo                     =   aptd['od__u0'][-1]              # number of output units
    nui                     =   aptd['id__u0'][-1]
    nid                     =   len(aptd['id__u0'])-1   # number of input dimensions
    nod                     =   len(aptd['od__u0'])-1   # number of output dimensions
    nstm                    =   aptd['data'].shape[0]   # number of stimuli

    labeltota               =   np.zeros((nstm, nid*nuo))
    for id in range(aptd['nid']):
        labeltota_fillrange =   np.arange(id*nuo, (id+1)*nuo) #  intervals.itv(id, nuo)
        for od in range(aptd['nod']):
            pstring         =   f'{id},{od}'
            labeltota[:,labeltota_fillrange]    =   labeltota[:,labeltota_fillrange] + aptd['label'][pstring]
    return labeltota

#-------------------------------------------------------------------------------
#   APTD --> RESIDUALS
#-------------------------------------------------------------------------------

def aptd_pl__olresidualAFTERlinregonIDENTITYlabels(aptd, pl):
    """
    PREDICTS THE LINEAR INPUT OF EACH OUTPUT UNIT, USING THE LABELS FOR THAT
    UNIT AS A PREDICTOR.  RETURNS THE RESIDUALS OF THESE PATTERNS IN AN ARRAY
    OF SHAPE (# STIMULI) X (# OUTPUT UNITS)
    """

    pstring = pl__pstring(pl)

    if aptd['schema'] == 'rumelhart_multiboth':
        depa = np.matmul(aptd['h1'][pstring], aptd['w1o'])
    elif aptd['schema'] == 'musslick':
        depa = np.matmul(aptd['hl'][pstring], aptd['who'])

    inda = aptd['label'][pstring]

    rsda = np.zeros(depa.shape)  # the residual array
    for colind in range(rsda.shape[1]):
        indcol = inda[:, [colind]]  # parentheses necessary to get correct shape
        depcol = depa[:, [colind]]  # parentheses necessary to get correct shape
        rsda[:,
        [colind]] = depcol - sklearn.linear_model.LinearRegression().fit(indcol,depcol).predict(indcol)
    return rsda




#-------------------------------------------------------------------------------
#   CENTERED REPRESENTATIONS
#-------------------------------------------------------------------------------

#--------------------------------------------------------
#   REPRESENTATIONS
#--------------------------------------------------------

def aptd_ln_pl_id__featurewisecenterrep(aptd, ln, pl, id):
    """
    each tuple of feature values in the extraneous input dimensions determines a
    map (feature values in the relevant input dim) --> hidden activations.
    add a constant offset to each map so that its values center around zero, and
    return this collection of maps
    """
    dsl                     =   aptd['id__nu'] # dsl = dimension size list
    perm                    =   dsl_d__sortperma(dsl, id, target='column_complement')
    block_size              =   dsl[id]
    block_num               =   np.prod([dsl[x] for x in range(len(dsl)) if not x == id])

    lpa                     =   copy.deepcopy(aptd[ln][pl__pstring(pl)]).astype(float) # NECESSARY TO CONVERT THIS TO FLOAT (there have been cases where it was an int ... numpy refused to insert "fractional" values in the for-loop below, to all this were converted to ints by rounding ... took ages to discover what was going wrong)
    lpa                     =   lpa[perm].reshape((block_num,block_size,lpa.shape[1])) # easy way to check this "wraps" correctly: compare np.arange(8).reshape(4,2) to np.arange(8).reshape(4,2).reshape((2,2,2))
    for p in range(block_num):
        lpa[p]              =   lpa[p] - np.mean(lpa[p],axis=0).reshape((1,-1))

    return lpa

def check__aptd_ln_pl_id__taskwisecenterrep():
    """
    PLEASE SEE RELATED CHECK FUNCTION: check__aptd_ln_pl_id__betarep
    """
    print("PLEASE SEE RELATED CHECK FUNCTION: check__aptd_ln_pl_id__betaep")

def aptd_ln_pl0_pl1_id__featurewisecenterrepdist(aptd, ln, pl0, pl1, id):
    rep0                    =   aptd_ln_pl_id__featurewisecenterrep(aptd, ln, pl0, id)
    rep1                    =   aptd_ln_pl_id__featurewisecenterrep(aptd, ln, pl1, id)
    return np.linalg.norm(rep0 - rep1)

#--------------------------------------------------------
#   DOT WITH WEIGHTS
#--------------------------------------------------------

#   THIS MAY NOT BE NECESSARY -- PAUSING THE WRITING OF FUNCTIONS 'TILL WE KNOW MORE

# def aptd_ln_id01_od01_plc__fwcDOTinterferingweights(aptd,ln,id0,id1,od0,od1,plcr):
#
#     if plc == 1:
#         fwcr00              =   aptd_ln_pl_id__featurewisecenterrep(aptd, 'h1',   [[0,od0]],         0)
#         fwcr01              =   aptd_ln_pl_id__featurewisecenterrep(aptd, 'h1',   [[0,od1]],         0)
#         fwcr10              =   aptd_ln_pl_id__featurewisecenterrep(aptd, 'h1',   [[1,od0]],         1)
#         fwcr11              =   aptd_ln_pl_id__featurewisecenterrep(aptd, 'h1',   [[1,od1]],         1)
#
#             dadd(T, f'od {od0}: var(fwc_rep_plc1 * od_weights)',    np.matmul(aptd_ln_pl_id__featurewisecenterrep(aptd, 'h1',   [[0,od0]],         0),   aptd[f'w1{od1}']).var(axis=0).sum() +\
#                                                                     np.matmul(aptd_ln_pl_id__featurewisecenterrep(aptd, 'h1',   [[0,od1]],         0),   aptd[f'w1{od0}']).var(axis=0).sum() +\
#                                                                     np.matmul(aptd_ln_pl_id__featurewisecenterrep(aptd, 'h1',   [[1,od0]],         1),   aptd[f'w1{od1}']).var(axis=0).sum() +\
#                                                                     np.matmul(aptd_ln_pl_id__featurewisecenterrep(aptd, 'h1',   [[1,od1]],         1),   aptd[f'w1{od0}']).var(axis=0).sum()
#                  )
#
#             #   od PRE-RECTIFIED VARIANCE FOR FWC-plc2 REP OF id (TASK pl)
#             dadd(T, f'od {od0}: var(fwc_rep_plc2 * od_weights)',    np.matmul(aptd_ln_pl_id__featurewisecenterrep(aptd, 'h1',   [[0,od0],[1,od1]],0)[x],   aptd[f'w1{od1}']).var(axis=0).sum() +\
#                                                                     np.matmul(aptd_ln_pl_id__featurewisecenterrep(aptd, 'h1',   [[0,od1],[1,od0]],0)[x],   aptd[f'w1{od0}']).var(axis=0).sum() +\
#                                                                     np.matmul(aptd_ln_pl_id__featurewisecenterrep(aptd, 'h1',   [[0,od0],[1,od1]],1)[x],   aptd[f'w1{od0}']).var(axis=0).sum() +\
#                                                                     np.matmul(aptd_ln_pl_id__featurewisecenterrep(aptd, 'h1',   [[0,od1],[1,od0]],1)[x],   aptd[f'w1{od1}']).var(axis=0).sum()
#                  )


#-------------------------------------------------------------------------------
#   BETA REPRESENTATIONS
#-------------------------------------------------------------------------------

#----------------------------
#   PERFORMANCE STATS
#----------------------------

# N                           =   logentrypath__N('/Users/gh10/a/c/l/p/taskhr/data/archives/archive/20200425-162625RMHT-nil2nol2nwu20000plcr1-2')
# aptd                        =   N_pll__aptd(N,rumelP__pll(N,plcr=[1,2]))
# import cProfile
# cProfile.run('aptd_ln_pl_id__meanrep(aptd, "h1", [[0,0]], 0, normalize_by_variance = False)',sort='tottime')
# #   1061 function calls (1060 primitive calls) in 0.003 seconds
#
# cProfile.run('aptd_ln_pl0_pl1_id__meanrepdist(aptd,"h1",[[0,0]],[[0,6]],0,normalize_by_variance=False)',sort='tottime')
# #   2135 function calls (2132 primitive calls) in 0.005 seconds
#
# cProfile.run('aptd_ln_pl0_pl1_id__meanrepdist(aptd,"h1",[[0,0],[1,6]],[[0,6],[1,0]],0,normalize_by_variance=False)',sort='tottime')
# # 2139 function calls (2136 primitive calls) in 0.005 seconds

#----------------------------
#   FUNCTIONS
#----------------------------

def aptd_ln_pl_id__betarep(aptd, ln, pl, id):
    """
    :param aptd:
    :param id:
    :param ln:  layer name
    :param pl:
    :return:
    """
    dsl                     =   aptd['id__nu']
    perm                    =   dsl_d__sortperma(dsl, id, target='column_complement')
    block_size              =   dsl[id]
    block_num               =   np.prod([dsl[x] for x in range(len(dsl)) if not x == id])

    lpa                     =   copy.deepcopy(aptd[ln][pl__pstring(pl)]).astype(float) # NECESSARY TO CONVERT THIS TO FLOAT (there have been cases where it was an int ... numpy refused to insert "fractional" values in the for-loop below, to all this were converted to ints by rounding ... took ages to discover what was going wrong)
    lpa                     =   lpa[perm].reshape((block_num,block_size,lpa.shape[1])) # easy way to check this "wraps" correctly: compare np.arange(8).reshape(4,2) to np.arange(8).reshape(4,2).reshape((2,2,2))
    for p in range(block_num):
        lpa[p]              =   lpa[p] - np.mean(lpa[p],axis=0).reshape((1,-1))

    meanrep                 =   np.mean(lpa,axis=0)

    return meanrep

def check__aptd_ln_pl_id__betarep():
    """
    TEST PASSED 20200424
    """
    dsl                     =   [2,3,4,5,3,10]
    nd                      =   len(dsl)
    repl                    =   [np.random.rand(dsl[p],100) for p in range(nd)]
    repl                    =   [x - np.mean(x, axis=0).reshape((1,-1)) for x in repl]
    A                       =   np.array(list(itertools.product(*[range(x) for x in dsl])))
    repa                    =   np.zeros((A.shape[0],100))
    for p in range(repa.shape[0]):
        for q in range(nd):
            repa[p]         =   repa[p] + repl[q][A[p][q]]

    aptd                    =   {}
    aptd['id__nu']          =   dsl
    aptd['hidden']          =   {}
    aptd['hidden']['0,0']   =   repa
    ln                      =   'hidden'
    pl                      =   [[0,0]]
    for id in range(nd):
        a                   =   aptd_ln_pl_id__betarep(aptd, ln, pl, id)
        err                 =   np.sum(np.abs(a - repl[id]))
        if err > 0.000000000001:
            print(f"error of {err}; please check dimension {id}")
            return
    print(f"test passed; rep means = {[np.linalg.norm(x) for x in repl]}")

def aptd_ln_pl0_pl1_id__betarepdist(aptd, ln, pl0, pl1, id, normalize_by_variance=False):
    """
    :param aptd:
    :param ln: layer name
    :param pl0:
    :param pl1:
    :param id:
    :param normalize_by_variance: True/False
    :return:
    """
    rep0                    =   aptd_ln_pl_id__betarep(aptd, ln, pl0, id)
    rep1                    =   aptd_ln_pl_id__betarep(aptd, ln, pl1, id)
    dist                    =   np.linalg.norm(rep0 - rep1)

    if normalize_by_variance:
        dist                =   dist / (rep0.var(axis=0).sum()+rep0.var(axis=0).sum())
    return dist

#-------------------------------------------------------------------------------
#   SIMPLE MEAN REPS
#-------------------------------------------------------------------------------

def aptd_ln_pl0_pl1_distofmeanreps(aptd, ln, pl0, pl1):
    """
    :param aptd:
    :param ln: layer name
    :param pl0:
    :param pl1:
    :return:
    """
    rep0                    =   np.mean(aptd[ln][  pl__pstring(pl0)  ],axis=0)
    rep1                    =   np.mean(aptd[ln][  pl__pstring(pl1)  ],axis=0)
    return np.linalg.norm(rep0-rep1)


#-------------------------------------------------------------------------------
#   INTERACTION EFFECTS
#-------------------------------------------------------------------------------

#----------------------------
#   PERFORMANCE STATS
#----------------------------

# N                           =   logentrypath__N('/Users/gh10/a/c/l/p/taskhr/data/archives/archive/20200425-162625RMHT-nil2nol2nwu20000plcr1-2')
# aptd                        =   N_pll__aptd(N,rumelP__pll(N,plcr=[1,2]))
# import cProfile
# cProfile.run('aptd_ln_id01_od01_plc__iena(aptd,"h1",0,1,3,9,plc=1).mean()',sort='tottime')

   #       42880 function calls in 0.103 seconds
   # Ordered by: internal time
   # ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   #    840    0.026    0.000    0.077    0.000 aptfuns.py:14(dsl_d_f__inda)
   #   9240    0.019    0.000    0.019    0.000 {built-in method numpy.arange}
   #    420    0.012    0.000    0.097    0.000 aptfuns.py:254(aptd_ln_pl0_pl1_id_f0_f1__iea)
   #   2522    0.012    0.000    0.012    0.000 {method 'reduce' of 'numpy.ufunc' objects}
   #   2520    0.005    0.000    0.020    0.000 fromnumeric.py:73(_wrapreduction)

#----------------------------
#   FULL ARRAYS
#----------------------------

def aptd_ln_pl0_pl1_id_f0_f1__iea(aptd,ln,pl0,pl1,id,f0,f1):
    """
    INTERACTION EFFECT OF SWITCHING BETWEEN TWO TASKS SETS, AND TWO ACTIVE
    FEATURES
    """
    dsl                     =   aptd['id__nu']
    f0ind                   =   dsl_d_f__inda(dsl,id,f0)
    f1ind                   =   dsl_d_f__inda(dsl,id,f1)
    pc0                     =   aptd[ln][pl__pstring(pl0)] # point cloud 0
    pc1                     =   aptd[ln][pl__pstring(pl1)] # point cloud 1
    return (pc0[f0ind]-pc0[f1ind])-(pc1[f0ind]-pc1[f1ind])

def aptd_ln_pl0_pl1_id__iea(aptd,ln,pl0,pl1,id):
    """
    ARRAY OF INTERACTION EFFECTS FOR ALL PAIRS OF FEATURES IN THE SPECIFIED
    INPUT DIMENSION
    """
    dsl                     =   aptd['id__nu']
    L                       =   []
    for (f0,f1) in itertools.combinations(range(dsl[id]),2):
        L.append(aptd_ln_pl0_pl1_id_f0_f1__iea(aptd,ln,pl0,pl1,id,f0,f1))
    return np.concatenate(L,axis=0)

def aptd_ln_id01_od01_plc__iea(aptd,ln,id0,id1,od0,od1,plc=2):
    """
    ARRAY OF INTERACTION EFFECTS FOR ALL TASK + FEATURE SWITCHES INVOLVING THE
    IMPLICATED INPUT/OUTPUT DIMENSIONS
    """
    if plc  ==  1:
        a0                  =   aptd_ln_pl0_pl1_id__iea(aptd,ln,[[id0,od0]],[[id0,od1]],id0)
        a1                  =   aptd_ln_pl0_pl1_id__iea(aptd,ln,[[id1,od0]],[[id1,od1]],id1)
        return np.concatenate((a0,a1),axis=0)
    if plc  ==  2:
        pl0                 =   [[id0,od0],[id1,od1]]
        pl1                 =   [[id0,od1],[id1,od0]]
        a0                  =   aptd_ln_pl0_pl1_id__iea(aptd,ln,pl0,pl1,0)
        a1                  =   aptd_ln_pl0_pl1_id__iea(aptd,ln,pl0,pl1,1)
        return np.concatenate((a0,a1),axis=0)

#----------------------------
#   NORMS
#----------------------------
def aptd_ln_pl0_pl1_id_f0_f1__iena(aptd,ln,pl0,pl1,id,f0,f1):
    iea                     =   aptd_ln_pl0_pl1_id_f0_f1__iea(aptd,ln,pl0,pl1,id,f0,f1)
    return np.linalg.norm(iea,axis=1)

def aptd_ln_pl0_pl1_id__iena(aptd,ln,pl0,pl1,id):
    dsl                     =   aptd['id__nu']
    L                       =   []
    for (f0,f1) in itertools.combinations(range(dsl[id]),2):
        L.append(aptd_ln_pl0_pl1_id_f0_f1__iena(aptd,ln,pl0,pl1,id,f0,f1))
    return np.concatenate(L)

def aptd_ln_id01_od01_plc__iena(aptd,ln,id0,id1,od0,od1,plc=2):
    return np.linalg.norm(aptd_ln_id01_od01_plc__iea(aptd,ln,id0,id1,od0,od1,plc=plc),axis=1)

#-------------------------------------------------------------------------------
#   WEIGHTS
#-------------------------------------------------------------------------------

def aptd_ln_pl0_pl1_id_f0_f1_od__ieDOTa(aptd,ln,pl0,pl1,id,od,f0,f1):
    """
    DOES NNOT ABSOLUTE VAUES
    """
    iea                     =   aptd_ln_pl0_pl1_id_f0_f1__iea(aptd,ln,pl0,pl1,id,f0,f1)
    return np.matmul(iea, aptd[f'w1{od}'])

def aptd_ln_pl0_pl1_id__ieDOTa(aptd,ln,pl0,pl1,id,od):
    """
    DOES NNOT ABSOLUTE VAUES
    """
    dsl                     =   aptd['id__nu']
    L                       =   []
    for (f0,f1) in itertools.combinations(range(dsl[id]),2):
        L.append(aptd_ln_pl0_pl1_id_f0_f1_od__ieDOTa(aptd,ln,pl0,pl1,id,od,f0,f1))
    return np.concatenate(L)

def aptd_ln_id01_od01_plc__ieWDOTa(aptd,ln,id0,id1,od0,od1,plc=2):
    """
    DOES NOT ABSOLUTE VAUES
    CUMULATIVE OVER ALL TASK + FEATURE SWITCHES INVOLVING THE IMPLICATED
    INPUT/OUTPUT DIMENSIONS
    """
    if plc  ==  1:
        a0                  =   aptd_ln_pl0_pl1_id__iea(aptd,ln,[[id0,od0]],[[id0,od1]],id0)
        a1                  =   aptd_ln_pl0_pl1_id__iea(aptd,ln,[[id1,od0]],[[id1,od1]],id1)
    if plc  ==  2:
        pl0                 =   [[id0,od0],[id1,od1]]
        pl1                 =   [[id0,od1],[id1,od0]]
        a0                  =   aptd_ln_pl0_pl1_id__iea(aptd,ln,pl0,pl1,0)
        a1                  =   aptd_ln_pl0_pl1_id__iea(aptd,ln,pl0,pl1,1)
    a                       =   np.concatenate((a0,a1),axis=0)
    w                       =   np.concatenate((aptd[f'w1{od0}'], aptd[f'w1{od1}']) , axis=1  )
    return np.matmul(a,w)





#-------------------------------------------------------------------------------
#   PATTERN CONSTRUCTORS
#-------------------------------------------------------------------------------

#   CURRENTLY UNUSED
#
# def ln0_ln1_ur__aptd_pl__diff(ln0,ln1,ur):
#     """
#     :param ln0: name of layer 1
#     :param ln1: name of layer 2
#     :param ur: unit range
#     :return: a function that takes a performance list pl and returns the matrix
#     activation_pattern_ln0(pl)[:,ur] - activation_pattern_ln0(pl)[:, ur]
#     """
#     def aptd_pl__diff(aptd,pl):
#         pstring         =   pl__pstring(pl)
#         return aptd[ln0][pstring][:,ur]-aptd[ln1][pstring][:,ur]
#     return aptd_pl__diff
#
# def ln_ur__aptd_pl__slice(ln,ur):
#     """
#     :param ln: layer name
#     :param ur: unit range
#     :return: a function that takse a performance list and aptd and returns
#     aptd[:,ur]
#     """
#     def aptd_pl__slice(aptd,pl):
#         return aptd[ln][pl__pstring(pl)][:,ur]
#     return aptd_pl__slice

#-------------------------------------------------------------------------------
#   VARIANCE EXPLAINED
#-------------------------------------------------------------------------------

def inda_depa__varexplained(inda,depa):
    reg                     =   sklearn.linear_model.LinearRegression().fit(inda, depa)
    pred                    =   reg.predict(inda)
    resd                    =   depa - pred
    return np.var(depa,axis=0).sum() - np.var(resd,axis=0).sum()

def aptd_pl_id_od__odvarianceexplainedbyid(aptd,pl,id,od):
    """
    :param aptd:
    :param pl0:
    :param pl1:
    :return: variance in the error of od0 explained by id1 when the network
    performs pl0 and pl1 simultaneously
    """
    ps                      =   pl__pstring(pl)
    depa                    =   aptd['od'][od][ps]
    inda                    =   aptd['id'][id][ps]
    return inda_depa__varexplained(inda,depa)

def aptd_pl_id_od__oderrorvarianceexplainedbyid(aptd,pl,id,od):
    """
    :param aptd:
    :param pl0:
    :param pl1:
    :return: variance in the error of od0 explained by id1 when the network
    performs pl0 and pl1 simultaneously
    """
    ps                      =   pl__pstring(pl)
    depa                    =   aptd['od'][od][ps]-aptd['od_label'][od][ps]
    inda                    =   aptd['id'][id][ps]
    return inda_depa__varexplained(inda,depa)

def aptd_i01_o01__crosstalkvarianceALLFOUR(aptd,id0,id1,od0,od1):
    """
    :param aptd:
    :param id0:
    :param id1:
    :param od0:
    :param od1:
    :return:
    """
    if not id0 < id1:
        print("error in function aptd_i01_o01__crosstalkvarianceALLFOUR: id0 must be lower than id1")
        return
    if not od0 < od1:
        print("error in function aptd_i01_o01__crosstalkvarianceALLFOUR: od0 must be lower than od1")
        return
    varexp                  =   0
    varexp                  =   varexp + aptd_pl_id_od__oderrorvarianceexplainedbyid(aptd,[[id0,od0],[id1,od1]],id0,od1)
    varexp                  =   varexp + aptd_pl_id_od__oderrorvarianceexplainedbyid(aptd,[[id0,od0],[id1,od1]],id1,od0)
    varexp                  =   varexp + aptd_pl_id_od__oderrorvarianceexplainedbyid(aptd,[[id0,od1],[id1,od0]],id0,od0)
    varexp                  =   varexp + aptd_pl_id_od__oderrorvarianceexplainedbyid(aptd,[[id0,od1],[id1,od0]],id1,od1)
    return varexp

def aptd_id01_od01__taskconsistentvarianceexplained(aptd,id0,id1,od0,od1):
    if not id0 < id1:
        print("error in function aptd_i01_o01__crosstalkvarianceALLFOUR: id0 must be lower than id1")
        return
    if not od0 < od1:
        print("error in function aptd_i01_o01__crosstalkvarianceALLFOUR: od0 must be lower than od1")
        return
    varexp                  =   0
    pl                      =   [[id0,od0],[id1,od1]]
    varexp                  =   varexp + aptd_pl_id_od__odvarianceexplainedbyid(aptd,pl,id0,od0)
    varexp                  =   varexp + aptd_pl_id_od__odvarianceexplainedbyid(aptd,pl,id1,od1)
    pl                      =   [[id0,od1],[id1,od0]]
    varexp                  =   varexp + aptd_pl_id_od__odvarianceexplainedbyid(aptd,pl,id0,od1)
    varexp                  =   varexp + aptd_pl_id_od__odvarianceexplainedbyid(aptd,pl,id1,od0)
    return varexp

def aptd_id01_od01__taskINconsistentvarianceexplained(aptd,id0,id1,od0,od1):
    if not id0 < id1:
        print("error in function aptd_i01_o01__crosstalkvarianceALLFOUR: id0 must be lower than id1")
        return
    if not od0 < od1:
        print("error in function aptd_i01_o01__crosstalkvarianceALLFOUR: od0 must be lower than od1")
        return
    varexp                  =   0
    pl                      =   [[id0,od0],[id1,od1]]
    varexp                  =   varexp + aptd_pl_id_od__odvarianceexplainedbyid(aptd,pl,id0,od1)
    varexp                  =   varexp + aptd_pl_id_od__odvarianceexplainedbyid(aptd,pl,id1,od0)
    pl                      =   [[id0,od1],[id1,od0]]
    varexp                  =   varexp + aptd_pl_id_od__odvarianceexplainedbyid(aptd,pl,id0,od0)
    varexp                  =   varexp + aptd_pl_id_od__odvarianceexplainedbyid(aptd,pl,id1,od1)
    return varexp

#-------------------------------------------------------------------------------
#   ERROR
#-------------------------------------------------------------------------------

def aptd_i01_o01__errormeanmagnitudemean(aptd,id0,id1,od0,od1):
    errml                   =   [] # error mean list
    ps                      =   pl__pstring([[id0,od0],[id1,od1]])
    errml.append(aptd['od'][od0][ps].mean(axis=0) - aptd['od_label'][od0][ps].mean(axis=0))
    errml.append(aptd['od'][od1][ps].mean(axis=0) - aptd['od_label'][od1][ps].mean(axis=0))
    ps                      =   pl__pstring([[id0,od1],[id1,od0]])
    errml.append(aptd['od'][od0][ps].mean(axis=0) - aptd['od_label'][od0][ps].mean(axis=0))
    errml.append(aptd['od'][od1][ps].mean(axis=0) - aptd['od_label'][od1][ps].mean(axis=0))
    return np.mean([np.linalg.norm(x) for x in errml])


