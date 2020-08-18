# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 13:16:59 2017

@author: sjurkatis
"""


#from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
cimport cython 
#from cpython cimport array
#import array

DTYPE = np.int
ctypedef np.int_t DTYPE_t

DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t

@cython.boundscheck(False)
@cython.wraparound(False)
def get_ind(double [:] arr, double [:] time):
    cdef: 
        int n = time.shape[0]
        int s = arr.shape[0]
        int begin, end, found, k
        unsigned int ind = 0
        size_t j, i
        np.ndarray[DTYPE_t, ndim=1] left = np.zeros(n, dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=1] right = np.zeros(n, dtype=DTYPE)
        double [:] tmp
        double v

    for j in xrange(n):
        begin = s
        end = 0
        found = 0
        tmp = arr[ind:]
        k = tmp.shape[0]
        for i in xrange(k):
            v = tmp[i] - time[j]
            if not found and v>=0:
                found = 1
                begin = ind + i
            
            if found:
                if v > 0:
                    ind += i
                    end = ind
                    break
                elif i==k-1:
                    end = s
                    break
        
        left[j] = begin
        right[j] = end
        
    return left, right


@cython.boundscheck(False)
@cython.wraparound(False)
def sign_trades_ds1(long[:] P, long[:] V, long[:] Al, long[:] Ar, long[:] Bl, long[:] Br, long[:] runlength, long[:] askp, long[:] bidp, long[:] avdiff, long[:] bvdiff, double[:] atime, double[:] btime, double bar):

    cdef:
        int n = Al.shape[0]
        int anum = askp.shape[0]
        int bnum = bidp.shape[0]                
        int trnum = P.shape[0]
        np.ndarray[DTYPE_t, ndim=1] s = np.zeros(trnum, dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=1] c = np.zeros(trnum, dtype=DTYPE)
        #np.ndarray[DTYPE_t, ndim=1] ask = np.zeros(trnum, dtype=DTYPE)
        #np.ndarray[DTYPE_t, ndim=1] bid = np.zeros(trnum, dtype=DTYPE)
        size_t j,i
        int al, ar, bl, br, last_ask, last_bid, k, d_a, d_b, tmpind, p, v 
        np.ndarray[DTYPE_t, ndim=1] discard_a = np.zeros(anum, dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=1] discard_b = np.zeros(bnum, dtype=DTYPE)              
        int ind = 0
        bint av_match, bv_match
        double upper, lower

    for j in range(n):
        
        al = Al[j]
        ar = Ar[j]
        bl = Bl[j]
        br = Br[j]

        last_ask = askp[al]
        last_bid = bidp[bl]
        for i in range(runlength[j]):
            tmpind = ind+i
            p = P[tmpind]
            v = V[tmpind]
            
            av_match = False
            bv_match = False

            #if ar>0:
            #    ask[tmpind] = last_ask
            #else:
            #    ask[tmpind] = -1

            #if br>0:
            #    bid[tmpind] = last_bid
            #else:
            #    bid[tmpind] = -1
            
            for k in range(ar-al):
                if discard_a[al+k]==0 and askp[al+k]==p and avdiff[al+k]==v:
                    av_match = True
                    d_a = k
                    break

            for k in range(br-bl):
                if discard_b[bl+k]==0 and bidp[bl+k]==p and bvdiff[bl+k]==v:
                    bv_match = True
                    d_b = k
                    break
                
            if av_match and not bv_match:
                s[tmpind] = 1
                c[tmpind] = 1
                #ask[tmpind] = p

                last_ask = askp[al+d_a]
            elif bv_match and not av_match:
                s[tmpind] = -1
                c[tmpind] = 1
                #bid[tmpind] = p

                last_bid = bidp[bl+d_b]
            elif av_match and bv_match:
                
                if atime[al+d_a] < btime[bl+d_b]:
                    s[tmpind] = 1
                    c[tmpind] = 2
                    #ask[tmpind] = p
                                        
                    discard_a[al+d_a] = 1
                    last_ask = askp[al+d_a]

                elif atime[al+d_a] > btime[bl+d_b]:
                    s[tmpind] = -1
                    c[tmpind] = 2
                    #bid[tmpind] = p


                    discard_b[bl+d_b] = 1
                    last_bid = bidp[bl+d_b]
                    
            else:

                # not av_match and not bv_match:
                # most likely hidden order
                if last_ask > last_bid:
                    upper = last_ask*(1-bar) + last_bid*bar
                    lower = last_ask*bar + last_bid*(1-bar)                    
                    #x = (p-last_bid)/(last_ask-last_bid)               
                    
                    if p>upper: #x>1-bar:                  
                        s[tmpind] = 1
                        c[tmpind] = 3
                    elif p<lower: #x<bar:                  
                        s[tmpind] = -1
                        c[tmpind] = 3    
                   
                
        ind +=runlength[j]
    

    return s, c #, ask, bid


@cython.boundscheck(False)
@cython.wraparound(False)
def sign_trades_ds2(long[:] P, long[:] V, long[:] Al, long[:] Ar, long[:] Bl, long[:] Br, long[:] runlength, long[:] askp, long[:] bidp, long[:] avdiff, long[:] bvdiff, double[:] atime, double[:] btime, double bar):

    cdef:
        int n = Al.shape[0]
        int anum = askp.shape[0]
        int bnum = bidp.shape[0]                
        int trnum = P.shape[0]
        np.ndarray[DTYPE_t, ndim=1] s = np.zeros(trnum, dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=1] c = np.zeros(trnum, dtype=DTYPE)
        size_t j,i,l
        int al, ar, bl, br, last_ask, last_bid, k, d_a, d_b, tmpind, p, trV, v
        int ind = 0
        bint av_match, bv_match 
        double upper, lower

    for j in range(n):
        
        al = Al[j]
        ar = Ar[j]
        bl = Bl[j]
        br = Br[j]
       
        last_ask = askp[al]
        last_bid = bidp[bl]
        for i in range(runlength[j]):
            
            tmpind = ind+i
            p = P[tmpind]
            trV = V[tmpind]
            
            av_match = False
            bv_match = False

            for k in range(ar-al):
                if askp[al+k]==p and avdiff[al+k]>=trV:
                    av_match = True
                    d_a = k
                    break

            for k in range(br-bl):
                if bidp[bl+k]==p and bvdiff[bl+k]>=trV:
                    bv_match = True
                    d_b = k
                    break

            if av_match and not bv_match:
                s[tmpind] = 1
                c[tmpind] = 1
                #last_ask = p
                
            elif bv_match and not av_match:
                s[tmpind] = -1
                c[tmpind] = 1
                #last_bid = p

            elif av_match and bv_match:

                if atime[al+d_a] < btime[bl+d_b]:
                
                    s[tmpind] = 1
                    c[tmpind] = 2
                                        
                    avdiff[al+d_a] -= trV
                    #last_ask = p
                    
                elif atime[al+d_a] > btime[bl+d_b]:
                    s[tmpind] = -1
                    c[tmpind] = 2
 
                    bvdiff[bl+d_b] -= trV
                    #last_bid = p                    
                    
            else:
                # there are two possibilites for a visible order to not find a
                # match:
                # (1): a market order trades against the best quote and the next
                #      best quote, due to the size of the market order. The order
                #      book, however, records only best quote and volume before
                #      the market order and the new best quote with corresponding
                #      volume after the completion of the full market order. That is,
                #      we have a price match, but not the corresponding volume 
                #      change match.
                        
                for k in range(ar-al):
                    if al+k+1==anum:
                        break
                    elif askp[al+k]<p and askp[al+k+1]==p: # this correctly checks also the price that we moved to the next interval
                        av_match = True
                        d_a = k+1
                        break
    
                for k in range(br-bl):
                    if bl+k+1==bnum:
                        break
                    elif bidp[bl+k]>p and bidp[bl+k+1]==p:
                        bv_match = True
                        d_b = k+1
                        break
                    
                if av_match and not bv_match:
                    s[tmpind] = 1
                    c[tmpind] = 5
                    #last_ask = p
    
                    
                elif bv_match and not av_match:
                    s[tmpind] = -1
                    c[tmpind] = 5
                    #last_bid = p

                elif av_match and bv_match:
                    if atime[al+d_a] < btime[bl+d_b]:
                        s[tmpind] = 1
                        c[tmpind] = 6
                        #last_ask = p

                    elif atime[al+d_a] > btime[bl+d_b]:
                        s[tmpind] = -1
                        c[tmpind] = 6
                        #last_bid = p
 

                else: 
                    # if there is no match with a price, there is a second possibility
                    # (2): The market order goes through the levels 1 to n>2 of  
                    #      the order book. The prices between level 1 and n are 
                    #      not displayed in the order book. Then all transactions
                    #      taking place between level 1 and n have no price match
                    for k in range(ar-al):
                        if al+k+1==anum:
                            break
                        elif askp[al+k]<p and askp[al+k+1]>p: 
                            av_match = True
                            d_a = k+1
                            break
        
                    for k in range(br-bl):
                        if bl+k+1==bnum:
                            break
                        elif bidp[bl+k]>p and bidp[bl+k+1]<p:
                            bv_match = True
                            d_b = k+1
                            break
                        
                    if av_match and not bv_match:
                        s[tmpind] = 1
                        c[tmpind] = 7
                        #last_ask = p
        
                        
                    elif bv_match and not av_match:
                        s[tmpind] = -1
                        c[tmpind] = 7
                        #last_bid = p
        
    
                    elif av_match and bv_match:                        
                        if atime[al+d_a] < btime[bl+d_b]:
                            s[tmpind] = 1
                            c[tmpind] = 8
                            #last_ask = p

                        elif atime[al+d_a] > btime[bl+d_b]:
                            s[tmpind] = -1
                            c[tmpind] = 8
                            #last_bid = p
 

                    else: # still no match; must be a hidden order
                        # not av_match and not bv_match:
                        # most likely hidden order
                        if last_ask > last_bid:
                            upper = last_ask*(1-bar) + last_bid*bar
                            lower = last_ask*bar + last_bid*(1-bar)                    
                            #x = (p-last_bid)/(last_ask-last_bid)               
                            
                            if p>upper: #x>1-bar:                  
                                s[tmpind] = 1
                                c[tmpind] = 3
                            elif p<lower: #x<bar:                  
                                s[tmpind] = -1
                                c[tmpind] = 3    
                   
                
        ind +=runlength[j]
    

    return s, c


@cython.boundscheck(False)
@cython.wraparound(False)
def sign_trades_ds3(long[:] P, long[:] V, long[:] Al, long[:] Ar, long[:] Bl, long[:] Br, long[:] runlength, long[:] askp, long[:] bidp, long[:] askv, long[:] bidv, double[:] atime, double[:] btime, double bar):

    cdef:
        int n = Al.shape[0]
        int anum = askp.shape[0]
        int bnum = bidp.shape[0]                
        int trnum = P.shape[0]
        np.ndarray[DTYPE_t, ndim=1] s = np.zeros(trnum, dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=1] c = np.zeros(trnum, dtype=DTYPE)
        size_t j,i,l
        int al, ar, bl, br, last_ask, last_bid, k, d_a, d_b, tmpind, p, trV 
        int ind = 0
        bint av_match, bv_match 
        double upper, lower

    for j in range(n):
        
        al = Al[j]
        ar = Ar[j]
        bl = Bl[j]
        br = Br[j]
       
        last_ask = askp[al]
        last_bid = bidp[bl]
        for i in range(runlength[j]):
            
            tmpind = ind+i
            p = P[tmpind]
            trV = V[tmpind]
            
            av_match = False
            bv_match = False

            for k in range(ar-al):
                if askp[al+k]==p and askv[al+k]>=trV:
                    av_match = True
                    d_a = k
                    break

            for k in range(br-bl):
                if bidp[bl+k]==p and bidv[bl+k]>=trV:
                    bv_match = True
                    d_b = k
                    break

            if av_match and not bv_match:
                s[tmpind] = 1
                c[tmpind] = 1
                
            elif bv_match and not av_match:
                s[tmpind] = -1
                c[tmpind] = 1  

            elif av_match and bv_match:
                
                if atime[al+d_a] < btime[bl+d_b]:
                    s[tmpind] = 1
                    c[tmpind] = 2
                    
                    askv[al+d_a] -=trV
                elif atime[al+d_a] > btime[bl+d_b]:
                    s[tmpind] = -1
                    c[tmpind] = 2
                    
                    bidv[bl+d_b] -=trV

            else: # still no match; must be a hidden order
                # not av_match and not bv_match:
                # most likely hidden order
                if last_ask > last_bid:
                    upper = last_ask*(1-bar) + last_bid*bar
                    lower = last_ask*bar + last_bid*(1-bar)                    
                    #x = (p-last_bid)/(last_ask-last_bid)               
                    
                    if p>upper: #x>1-bar:                  
                        s[tmpind] = 1
                        c[tmpind] = 3
                    elif p<lower: #x<bar:                  
                        s[tmpind] = -1
                        c[tmpind] = 3    
                   
                
        ind +=runlength[j]
    
    return s, c


@cython.boundscheck(False)
@cython.wraparound(False)
def tick_rule(long[:] allprices, long[:] prices, long[:] index_p):
    
    cdef:
        int n = prices.shape[0] 
        np.ndarray[DTYPE_t, ndim=1] s = np.zeros(n, dtype=DTYPE)
        int t, p, count, sz, lp
        long [:] lstp
        size_t j

    for j in range(n):
        count = 0
        p = prices[j]
        t = index_p[j]
        lstp = allprices[:t]
        sz = lstp.shape[0]
        s[j] = 0
        while sz>count:
            count += 1
            lp = lstp[sz-count]
            if p>lp:
                s[j] = 1                    
                break
            elif p<lp:
                s[j] = -1                    
                break

    return s #[s[j] for j in range(n)]
        
@cython.boundscheck(False)
@cython.wraparound(False)
def vol_bin(long[:] x, long w):

    cdef:
        long vol = 0
        long g = 0
        int n = x.shape[0]
        np.ndarray[DTYPE_t, ndim=1] group = np.zeros(n, dtype=DTYPE)
        size_t i

    for i in range(n):
        vol += x[i]
        group[i] = g
        if vol>=w:
            g+=1
            vol = 0

    return group 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def concat_runs(long[:] x, bint hj_version=False):

    cdef:
        int n = x.shape[0]
        size_t i, j
        double k
        size_t count = 0  
        np.ndarray[DTYPEf_t, ndim=1] interp = np.zeros(sum(x), dtype=DTYPEf)

    if hj_version:        
        for i in range(n):
            k = x[i]+1.
            for j in range(1,x[i]+1):
                interp[count] = (2*j-1)/(2*k)
                count += 1 
    else:
        for i in range(n):
            k = x[i]+1.
            for j in range(1,x[i]+1):
                interp[count] = j/k
                count += 1 

    return interp