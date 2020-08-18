import numpy as np 
import pandas as pd 
from tradeclassification_c import get_ind, sign_trades_ds1, sign_trades_ds2, sign_trades_ds3, tick_rule, vol_bin, concat_runs
from scipy import stats
import warnings
import statsmodels.api as sm


def get_lastquote(quotes,as_of):

    ind = np.searchsorted(quotes.time.values,as_of, side='left')-1
    last_quote = pd.Series(np.zeros(len(as_of)), index=as_of)
    
    mask = ind>=0
    last_quote.loc[mask] = quotes.loc[ind[mask],'price'].values
    last_quote.loc[~mask]  = np.nan
    return last_quote


def get_midpoint(Ask,Bid,as_of):

    ask = get_lastquote(Ask,as_of)
    bid = get_lastquote(Bid,as_of)

    midpoint = (ask + bid)/2
    midpoint.loc[ask<bid] = np.nan

    return midpoint.to_frame(name='midpoint')


def get_runs(x):
    """Returns group label, start and length of runs of values in x."""

    dx = np.diff(x)
    xi = np.nonzero(np.append(1,dx))[0]
    g_label = np.arange(len(xi))
    run_length = np.diff(np.append(xi,len(x)))
    group = np.repeat(g_label,run_length)
  
    return group, xi, run_length 


def trim_timestamp(x,freq):
    """
    Reduces timestamp precision.
    
    Parameters
    ----------
    x : numpy.ndarray
        1D array of timestamps measured in seconds after midnight.
    freq : float or int
        Frequency to which to timestamp precision should be reduced. 
        Frequency is measured in 10^freq of a second. 
        E.g.:
        freq=0 sets timestamp precision to seconds
        freq=1 sets it to tenth of a second
        freq = log10(0.5) sets it to every second second

    Returns
    -------
    numpy.ndarray

    """

    return np.floor(x*10**freq)/10**freq


def quote_index(q_t,tr_t):
    """Get start and end index of quote times in `q_t` with the same timestamp as trade times in `tr_t`."""

    left, right = get_ind(q_t,tr_t)
    right[left<right] -=1 # last quote cannot be traded on, so shift index 
    left -=1 # consider last quote from before the timestamp of the trade
    left[left<0] = 0    

    return left, right


def interpolate_time(t,freq,hj_version=False):
    """
    Interpolate timestamp precision. 

    .. math:: t_i = t + \frac{i}{N+1} f.

    Parameters
    ----------
    t : array like
        Array containing timestamps represented in seconds after midnight. 
    freq : int
        Timestamp precision as :math:`10^freq`th of a second. 
        E.g. `freq=0` means timestamp is precise to the second, 
        `freq=1` timestamp is precise to 10th of a second.
    hj_version : bool (default : False)
        If true, assigns equal interval length to each timestamp
        following Holden and Jacobsen (2014).

    Returns
    -------
    Array with interpolated timestamp

    References
    ----------
    Holden, C.W. and Jacobsen, S., 2014. Liquidity measurement problems in
    fast, competitive markets: Expensive and cheap solutions. 
    The Journal of Finance, 69(4), pp.1747-1785.
    
    """

    run = np.append(0,np.diff(t))
    runlength = np.append(np.nonzero(run),run.size)
    runlength = np.diff(np.append(0,runlength))

    intertime = concat_runs(runlength,hj_version)    

    return t+intertime*10**(-freq)


def delta_vol(p,v,ask=True):
    """
    Returns change in volume `v` given quotes in `p`.
    For ask quotes the change is given by

    .. math::

        \Delta v^a_j =    
        \begin{cases}
        v^a_j - v^a_{j+1} & \text{if $a_j = a_{j+1}$} \\
        v^a_j           & \text{if $a_j < a_{j+1}$} \\
        -1            & \text{otherwise}. 
        \end{cases} 
    
    For bid quotes the inequality in reversed.

    Parameters
    ----------
    p : numpy array
        Array of quotes
    v : numpy array
        Array of volume at the quotes. p and v must have the same length.
    ask : bool (default, True)
        If True, change in volume is computed for ask quotes, otherwise for bid quotes.

    Returns
    -------
    numpy array with changes in volume

    """
    
    vdiff = np.append(np.diff(v),0)*-1
    
    up = np.less(p[:-1],p[1:]) 
    up = np.append(up,0).astype(int)
    
    down = np.greater(p[:-1],p[1:]) 
    down = np.append(down,0).astype(int)
    if ask:
        vdiff[up==1] = v[up==1] # if ask price went up, volume change is equal to volume previously available
        vdiff[down==1] = -1 # if ask price went down, volume change will not be considered
    else:
        # opposite for the bid
        vdiff[down==1] = v[down==1] 
        vdiff[up==1] = -1 
    
    return vdiff


def fraction_buy(p,dof='estimate'):

    dp = np.diff(np.log(p))
    mask = ~(np.isinf(dp) | np.isnan(dp))
    if dp[mask].shape[0] < 2:
        return np.array([np.nan]*dp.shape[0]) 

    sigma = np.std(dp[mask],ddof=1)
    if sigma == 0:
        return np.array([np.nan]*dp.shape[0])

    x = dp/sigma
    if dof=='estimate':
        dof = stats.t.fit(x[mask],loc=0,scale=1)[0]
    elif dof=='normal':
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in (greater|less)')
            return stats.norm.cdf(x)

    with warnings.catch_warnings():
        return stats.t.cdf(x,dof)   


class TradeClassification:
    """
    Class to classify transactions into buyer- and seller-initiated trades.
    Available methods are the Full-Information algorithm (FI, see Jurkatis, 2020),
    the Lee-Ready algortihm (LR, see Lee and Ready, 1991), the Bulk-Volume 
    classification algorithm (BVC, see Easley et al., 2012), the algorithm of
    Chakrabarty et al. (2007) (CLNV) and the algortihm of Ellis et al. (2000) (EMO).


    Parameters
    ----------
    df_tr : pandas.DataFrame
        Dataframe with transaction data, assumed to be deduplicated, ie. only 
        one record per trade between two counterparties (only relevant for FI and BVC).
        The dataframe must contain at least a `time` column containing the transaction times 
        measured in seconds (i.e. timestamps of precision higher than seconds 
        are expressed as floats) and a `price` column containing the transaction prices. 
        For the FI algorithm the dataframe must also contain a `vol` column with 
        the number of shares exchanged in the transaction.
    Ask : pandas.DataFrame (optional; default None)
        For the FI, LR, EMO and CLNV algorithms order book data is required. 
        The dataframe must contain a `time` column indicating the time of the 
        quote change expressed in seconds and a `price` column with the best ask.
        For the FI algorithm the dataframe must also contain the volume available
        at the best ask.
    Bid : analogous to `Ask`. 
    

    References
    ----------
    Chakrabarty, B., Li, B., Nguyen, V., Van Ness, R.A., 2007. Trade classification
    algorithms for electronic communications network trades. Journal of Banking &
    Finance 31, 3806–3821.

    Easley, D., de Prado, M.M.L., O’Hara, M., 2012. Flow toxicity and liquidity in a
    high-frequency world. Review of Financial Studies 25, 1457–1493.

    Ellis, K., Michaely, R., O’Hara, M., 2000. The accuracy of trade classification rules:
    Evidence from Nasdaq. Journal of Financial and Quantitative Analysis 35, 529–551.

    Jurkatis, S., 2020. Inferring Trade Directions in Fast Markets. Unpublished Mimeo

    Lee, C., Ready, M.J., 1991. Inferring trade direction from intraday data. The Journal
    of Finance 46, 733–746.
    """

    def __init__(self,df_tr,Ask=None,Bid=None):

        self.valid_methods = ['ds_1', 'ds_2', 'ds_3', 'lee_ready', 'bvc', 'emo', 'clnv','true']
        self.df_tr = df_tr
        self.Ask = Ask 
        self.Bid = Bid


    def extract_variables(self,version):
        # get first and last ask/bid quote valid at each trade time
        a_l, a_r = quote_index(self.Ask.time.values,self.df_tr.time.unique()) 
        b_l, b_r = quote_index(self.Bid.time.values,self.df_tr.time.unique())

        # interpolate ask abd bid-quote times
        askit = interpolate_time(self.Ask.time.values,self.freq)
        bidit = interpolate_time(self.Bid.time.values,self.freq)

        # ask price, volume and volume change
        askp = self.Ask['price'].values.astype(int)
        bidp = self.Bid['price'].values.astype(int)

        if version in ['ds_1','ds_2']:
            askv = delta_vol(askp,self.Ask['vol'].values.astype(int))
            bidv = delta_vol(bidp,self.Bid['vol'].values.astype(int),ask=False)
        elif version == 'ds_3':
            askv = self.Ask['vol'].values.astype(int)
            bidv = self.Bid['vol'].values.astype(int)

        # number trades per timestamp, trade prices and volume
        tr_n = self.df_tr[['vol','time']].groupby('time').count().values.flatten().astype(int)
        P = self.df_tr.price.values.astype(int)
        V = self.df_tr['vol'].values.astype(int)

        return [P, V, a_l, a_r, b_l, b_r, tr_n, askp, bidp, askv, bidv, askit, bidit] 

    def reduce_tprecision(self,freq,tcol=None):
        """
        Reduce timesamp precision of data in place.
        If a 'time_org' column does not exist the orginal
        'time' column is renamed as such.
        The new timestamp data is given in the 'time' column.
        
        Parameters
        ----------
        freq : int 
            Frequency to which to timestamp precision should be reduced. 
            Frequency is measured in 10^freq of a second. 
            E.g. freq=0 sets timestamp precision to seconds, freq=1 sets it to tenth of a second.
        tcol : str (default : None)
            Time column on which to perform timestamp precision reduction. If `None`, the 
            precision is reduced on the 'time_org' column. 
        
        """

        if tcol is None:
            tcol = 'time_org'

        if not 'time_org' in self.df_tr.columns:
            self.df_tr.rename(columns={'time': 'time_org'}, inplace=True)

        self.df_tr['time'] = trim_timestamp(self.df_tr[tcol].values,freq)

        if self.Ask is not None:
            if not 'time_org' in self.Ask.columns:
                self.Ask.rename(columns={'time': 'time_org'}, inplace=True)

            self.Ask['time'] = trim_timestamp(self.Ask[tcol].values,freq)

        if self.Bid is not None:            
            if not 'time_org' in self.Bid.columns:
                self.Bid.rename(columns={'time': 'time_org'}, inplace=True)
        
            self.Bid['time'] = trim_timestamp(self.Bid[tcol].values,freq)

        return 

    def rename_timecol(self,tcol=None):
        """
        Rename column given in `tcol` to 'time' column. 
        If `tcol` is None it is set to 'time_org'.
        """

        if tcol is None:
            tcol = 'time_org'
        
        if tcol in self.df_tr.columns:
            if 'time' in self.df_tr.columns:
                self.df_tr.drop(columns='time', inplace=True)
            
            self.df_tr.rename(columns={tcol: 'time'}, inplace=True)
        
        if self.Ask is not None:
            if tcol in self.Ask.columns:
                if 'time' in self.Ask.columns:
                    self.Ask.drop(columns='time', inplace=True)
                
                self.Ask.rename(columns={tcol: 'time'}, inplace=True)
            
        if self.Bid is not None:
            if tcol in self.Bid.columns:
                if 'time' in self.Bid.columns:
                    self.Bid.drop(columns='time', inplace=True)
                
                self.Bid.rename(columns={tcol: 'time'}, inplace=True)

        return

    def classify(self,method,freq,reduce_precision=True,**kwargs):
        """
        Classifies trades into buyer- and seller-initiated.

        Parameters
        ----------
        method : str
            Must be one of the following: 'ds_1', 'ds_2', 'ds_3' (different versions of 
            the FI algorithm), 'lee_ready', 'bvc', 'emo', 'clvn'. See Jurkatis (2020).
        freq : float or int
            Timestamp precision of the data as measured in 10^-freq of a second. 
            E.g., freq=3 corresponds to millisecond timestamps. 
        reduce_precision : bool (default True)
            If True, reduces the precision of the timestamp to the frequency specified in
            `freq`. The original `time` column is renamed to `time_org`.
        kwargs : optional keyword arguments passed to the algorithm.
                 
        Returns
        -------
        None, unless bvc is choosen. 
        If not bvc, classification results is provided in a new column `Initiator` with 1 for
        buyer-initiated trades, -1 for seller-initiated trades and 0 for unclassified 
        trades. An additional `Step` column indicated at which step the the trade was
        classificied. E.g. for lee_ready, 1 means the trade is classified using the 
        quote rule, 2 indicates the use of the tick-test.
        If bvc, returns a pandas.DataFrame with buyer-initiated and total volume over
        the respective classification intervals.

        References
        ----------
        Jurkatis, S., 2020. Inferring Trade Directions in Fast Markets. Unpublished Mimeo

        """

        if not method in self.valid_methods:
            raise ValueError(f"'{method}' is not a valid method; use one of {self.valid_methods}")

        self.freq = freq      
        # rename time colums depending on whether timestamp precision should be reduced
        # or not; and reduce timestamp precision accordingly
        if reduce_precision:
            self.reduce_tprecision(freq,kwargs.get('tcol'))
        else:
            self.rename_timecol(kwargs.get('tcol'))
            

        if method in ['ds_1', 'ds_2', 'ds_3']:
            return self.fi_algo(method,**kwargs)
        elif method == 'lee_ready':
            return self.lee_ready(**kwargs)
        elif method == 'bvc':
            return self.bvc(**kwargs)
        elif method == 'emo':
            return self.emo(**kwargs)
        elif method == 'clnv':
            return self.clnv(**kwargs)
        elif method == 'true':
            return self.true_initiator()


    def fi_algo(self,version,bar=0.3,**kwargs):
        """
        Classify trades using the Full-Information algorithm.

        Parameters
        ----------
        version : str
            Must be one of 'ds_1', 'ds_2' or 'ds_3'. See Jurkatis (2020).
            Note that 'ds_3' corresponds to the version for Data Structure 2
            in the paper and 'ds_2' to Data Structure 3 in the accompanying
            online appendix. 
        bar : float (default 0.3)
            Must be between 0 and 1. Determines the range around the spread 
            midpoint in which the tick-test is used. 

        Returns
        -------
        None. Result is appended to the provided dataframe.

        References
        ----------
        Jurkatis, S., 2020. Inferring Trade Directions in Fast Markets. Unpublished Mimeo      
        
        """

        varpack = self.extract_variables(version)

        # sign trades
        if version == 'ds_1':
            sign, c = sign_trades_ds1(*varpack, bar)
        elif version == 'ds_2':
            sign, c = sign_trades_ds2(*varpack, bar)
        elif version == 'ds_3':
            sign, c = sign_trades_ds3(*varpack, bar)
        else:
            raise ValueError(f"'{version}' is not a valid method; use 'ds_1', 'ds_2' or 'ds_3'.")

        # trade initiator
        self.df_tr['Initiator'] = sign

        # classification step
        self.df_tr['Step'] = c

        # tick rule 
        self.apply_tick()
        self.df_tr.loc[self.df_tr.Step==0,'Step'] = 4

        return 


    def lee_ready(self,interpolate=False,**kwargs):
        """
        Classify trades using the Lee-Ready algorithm.

        Parameters
        ----------      
        interpolate : bool (default False)
            If True, interpolate timestamp precision prior
            to applying the algorithm according to 
            Holden and Jacobsen (2014)

        Returns
        -------
        None. Result is appended to the provided dataframe.

        References
        ----------
        Holden, C.W., Jacobsen, S., 2014. Liquidity measurement problems in fast, com-
        petitive markets: expensive and cheap solutions. The Journal of Finance 69,
        1747–1785.

        Lee, C., Ready, M.J., 1991. Inferring trade direction from intraday data. The Journal
        of Finance 46, 733–746.
        
        """
        
        # cleanup
        self.df_tr.drop(columns='midpoint',errors='ignore',inplace=True)

        if interpolate:
            self.tcol_interpolation()
            timecol = 'time_inter'
        else:
            timecol = 'time'

        midpoint = get_midpoint(self.Ask[[timecol,'price']].rename(columns={'time_inter':'time'}),
                                self.Bid[[timecol,'price']].rename(columns={'time_inter':'time'}),
                                self.df_tr[timecol].unique()
        )

        self.df_tr = self.df_tr.merge(midpoint, left_on=timecol,right_index=True,how='left')
        
        self.df_tr['Initiator'] = 0
        self.df_tr['Step'] = 0

        self.df_tr.loc[self.df_tr.price>self.df_tr.midpoint, 'Initiator'] = 1
        self.df_tr.loc[self.df_tr.price<self.df_tr.midpoint, 'Initiator'] = -1

        self.df_tr.loc[self.df_tr.price!=self.df_tr.midpoint, 'Step'] = 1

        # tick rule
        self.apply_tick()
        self.df_tr.loc[self.df_tr.Step==0,'Step'] = 2

        return 


    def bvc(self,window=1,window_type='time',dof='estimate',start=None,**kwargs):
        """
        Returns the fraction of buyer-initiated volume according to the 
        Bulk-Volume classification algorithm and total volume.

        Parameters
        ----------
        window : float or int (default 1)
            Length of the intervals over which to compute the fraction of
            buyer-inititated volumes.
        window_type : str (default 'time')
            Type of the interval. Must be either 'time', 'vol' or 'per_trade'.
            Specifies which unit the `window` refers to: seconds if 'time' and
            trading volume if 'vol'. If 'per_trade', `window` is ignored and the
            buyer-initiated volume is computed for each individual trade.
        dof : str, int or float (default 'estimate')
            Specifies which distribution to choose to map standardized price 
            changes on the unit-line. If a string is given it must be either 'normal'
            or 'estimate'. If 'normal', the standard Gaussian distribution is chosen. 
            If 'estimate', the student t-distribution is chosen with the degrees of 
            freedom estimated from the array of standardized prices changes. If int
            or float, the provided value is used for the degrees of freedom of the
            t-distribution.
        start: float or int (default None)
            Starting point of the first interval. Only relevant for `window_type` 'time'.
            If None, starting  point for the interval construction is the first mentioned
            timestamp at the given timestamp precision.

        Returns
        -------
        pandas.DataFrame index by the interval number, containing buyer-initiated volume
        and total volume. If 'per_trade', the buyer-initiated volume is equal to 
        total trading volume if the probability of being buyer-initiated is greater 0.5, it 
        is zero if the probability is smaller 0.5 and it is set to -1, if the probability
        of being buyer-initiated is 0.5. 

        References
        ----------
        Easley, D., de Prado, M.M.L., O’Hara, M., 2012. Flow toxicity and liquidity in a
        high-frequency world. Review of Financial Studies 25, 1457–1493.

        """

        if window_type=='per_trade':
            group = np.arange(len(self.df_tr)-1)            
            p = self.df_tr.price.values
        else:
            p0 = self.df_tr.price.iloc[0]
            group = self.create_window(self.df_tr.iloc[1:],window=window,window_type=window_type,start=start)

            # last price per group/window
            ind = np.searchsorted(group,np.unique(group),side='right')-1
            p = np.append(p0,self.df_tr.price.iloc[1:].values[ind])

        buy_frac = pd.DataFrame(fraction_buy(p,dof=dof), index=np.unique(group), columns=['f_b'])
        buy_frac.index.name = 'group'

        self.df_tr['group'] = -1
        self.df_tr.iloc[1:,self.df_tr.columns.get_indexer(['group'])] = group 

        if window_type=='per_trade':
            buy_frac['vol'] = self.df_tr.vol.values[1:]
            buy_frac['buy_vol'] = 0
            buy_frac.loc[buy_frac.f_b>0.5,'buy_vol'] = buy_frac.loc[buy_frac.f_b>0.5,'vol']
            buy_frac.loc[buy_frac.f_b==0.5,'buy_vol'] = -1
        else:
            buy_frac = buy_frac.join( self.df_tr[['group','vol']].groupby('group').sum() ) 
            buy_frac['buy_vol'] = buy_frac.f_b*buy_frac.vol

        return buy_frac


    def buyvolume(self,window=1,window_type='time',start=None,drop_firsttrade=True):
        """
        Returns buyer-initiated volume and total volume over intervals from the 
        individually classified trades. The result can be compared to the output of
        the BVC algorithm.

        Parameters
        ----------
        window : float or int (default 1)
            Length of the intervals over which to compute the fraction of
            buyer-inititated volumes.
        window_type : str (default 'time')
            Type of the interval. Must be either 'time', 'vol' or 'per_trade'.
            Specifies which unit the `window` refers to: seconds if 'time' and
            trading volume if 'vol'. If 'per_trade', `window` is ignored and the
            buyer-initiated volume is computed for each individual trade.
        start: float or int (default None)
            Starting point of the first interval. Only relevant for `window_type` 'time'.
            If None, starting  point for the interval construction is the first mentioned
            timestamp at the given timestamp precision.
        drop_firsttrades : bool (default True)
            If True, first trade is not considered in constructing the interval. This
            choice makes the result comparable to the result from the BVC algorithm
            which uses the first price as the reference starting point to compute the
            between-interval price changes.

        Returns
        -------
        pandas.DataFrame index by the interval number, containing buyer-initiated volume
        and total volume.  

        """


        if not 'Initiator' in self.df_tr.columns:
            raise KeyError("Data do not contain trade initiator label; classify trades first")

        i = 1 if drop_firsttrade else 0
        group = self.create_window(self.df_tr.iloc[i:],window=window,window_type=window_type,start=start)

        self.df_tr['group'] = -1
        self.df_tr.iloc[i:,self.df_tr.columns.get_indexer(['group'])] = group 

        vol = self.df_tr[['group','vol']].groupby('group').sum()
        buyfrac = self.df_tr.loc[self.df_tr.Initiator==1,['group','vol']].groupby('group').sum().rename(columns={'vol': 'buy_vol'})
        buyfrac = buyfrac.join(vol,how='outer').fillna(0)
        
        if drop_firsttrade:
            buyfrac = buyfrac.iloc[i:]

        return buyfrac 


    def create_window(self,df_tr,window=1,window_type='time',start=None):

        if not window_type in ['time','vol']:
            raise ValueError("window type to create intervals must be either 'time' or 'vol'.")

        group = df_tr[window_type].values.astype(int)
        if window_type == 'time':
            if start is not None:
                group = group - start 
            elif self.freq is None:
                group = group - group[0]
            else:
                group = group - np.floor(group[0]*10**self.freq)/10**self.freq

            group = group // window
        else:
            group = vol_bin(group, window)

        return group


    def emo(self,interpolate=False,**kwargs):
        """
        Classify trades using the algorithm of Ellis et al. (2000).

        Parameters
        ----------      
        interpolate : bool (default False)
            If True, interpolate timestamp precision prior
            to applying the algorithm according to 
            Holden and Jacobsen (2014)

        Returns
        -------
        None. Result is appended to the provided dataframe.

        References
        ----------
        Holden, C.W., Jacobsen, S., 2014. Liquidity measurement problems in fast, com-
        petitive markets: expensive and cheap solutions. The Journal of Finance 69,
        1747–1785.

        Ellis, K., Michaely, R., O’Hara, M., 2000. The accuracy of trade classification rules:
        Evidence from Nasdaq. Journal of Financial and Quantitative Analysis 35, 529–551.
        
        """
        
        # cleanup
        self.df_tr.drop(columns=['ask','bid'], errors='ignore',inplace=True)

        if interpolate:
            self.tcol_interpolation()
            timecol = 'time_inter'
        else:
            timecol = 'time'

        lastask = get_lastquote(self.Ask[[timecol,'price']].rename(columns={'time_inter': 'time'}),
                                self.df_tr[timecol].unique()
        )

        lastbid = get_lastquote(self.Bid[[timecol,'price']].rename(columns={'time_inter': 'time'}),
                                self.df_tr[timecol].unique()
        )

        self.df_tr = self.df_tr.merge(lastask.to_frame(name='ask').join(lastbid.to_frame(name='bid'), how='outer'),
                                      left_on=timecol, right_index=True, how='left')

        mask = self.df_tr.ask<=self.df_tr.bid 
        self.df_tr.loc[mask,'ask'] = np.nan 
        self.df_tr.loc[mask,'bid'] = np.nan 

        self.df_tr['Initiator'] = 0
        self.df_tr['Step'] = 0

        self.df_tr.loc[self.df_tr.price==self.df_tr.ask,'Initiator'] = 1 
        self.df_tr.loc[self.df_tr.price==self.df_tr.bid,'Initiator'] = -1

        self.df_tr.loc[(self.df_tr.Initiator==1) | (self.df_tr.Initiator==-1), 'Step'] = 1

        # tick rule
        self.apply_tick()
        self.df_tr.loc[self.df_tr.Step==0,'Step'] = 2

        return 


    def clnv(self,interpolate=False,**kwargs):
        """
        Classify trades using the algorithm of Chakrabarty et al. (2007).

        Parameters
        ----------      
        interpolate : bool (default False)
            If True, interpolate timestamp precision prior
            to applying the algorithm according to 
            Holden and Jacobsen (2014)

        Returns
        -------
        None. Result is appended to the provided dataframe.

        References
        ----------
        Chakrabarty, B., Li, B., Nguyen, V., Van Ness, R.A., 2007. Trade classification
        algorithms for electronic communications network trades. Journal of Banking &
        Finance 31, 3806–3821.
        
        Holden, C.W., Jacobsen, S., 2014. Liquidity measurement problems in fast, com-
        petitive markets: expensive and cheap solutions. The Journal of Finance 69,
        1747–1785.

        """

        # cleanup
        self.df_tr.drop(columns=['ask','bid'], errors='ignore',inplace=True)

        if interpolate:
            self.tcol_interpolation()
            timecol = 'time_inter'
        else:
            timecol = 'time'

        lastask = get_lastquote(self.Ask[[timecol,'price']].rename(columns={'time_inter': 'time'}),
                                self.df_tr[timecol].unique()
        )

        lastbid = get_lastquote(self.Bid[[timecol,'price']].rename(columns={'time_inter': 'time'}),
                                self.df_tr[timecol].unique()
        )

        self.df_tr = self.df_tr.merge(lastask.to_frame(name='ask').join(lastbid.to_frame(name='bid'), how='outer'),
                                      left_on=timecol, right_index=True, how='left')

        mask = self.df_tr.ask<=self.df_tr.bid 
        self.df_tr.loc[mask,'ask'] = np.nan 
        self.df_tr.loc[mask,'bid'] = np.nan 

        self.df_tr['Initiator'] = 0
        self.df_tr['Step'] = 0

        self.df_tr.loc[(self.df_tr.price>0.7*self.df_tr.ask + 0.3*self.df_tr.bid) & (self.df_tr.price<=self.df_tr.ask),'Initiator'] = 1 
        self.df_tr.loc[(self.df_tr.price<0.3*self.df_tr.ask + 0.7*self.df_tr.bid) & (self.df_tr.price>=self.df_tr.bid),'Initiator'] = -1

        self.df_tr.loc[(self.df_tr.Initiator==1) | (self.df_tr.Initiator==-1), 'Step'] = 1

        # tick rule
        self.apply_tick()
        self.df_tr.loc[self.df_tr.Step==0,'Step'] = 2

        return 


    def apply_tick(self):
        """Classify trades using the tick-test. Used in conjunction with one of
        the other algorithms, but can be used standalone if a `Step` column 
        containing only zeros is given in the transaction dataframe."""

        # tick rule 
        mask = self.df_tr.Step==0

        trrest = self.df_tr.loc[mask,['price']].reset_index(drop=False).values.astype(int)
        index_p, prices = trrest[:,0], trrest[:,1]
            
        s = tick_rule(self.df_tr.price.values.astype(int), prices, index_p)

        self.df_tr.loc[mask,'Initiator'] = s

        return

    def true_initiator(self):
        self.df_tr['Initiator'] = self.df_tr.direction*-1
        return 


    def evaluate_bulkclass(self,buyvol,target):
        """
        Evaluate classification result when estimated as a fraction of
        trading volume over time or volume intervals. Criterium follows
        Chakrabarty et al. (2015):

        ..math:: \sum_{i} \min(V_i^B,\hat{V}_i^B) + \min(V_i^S,\hat{V}_i^S) / \sum_i V_i.

        Parameters:
        -----------
        buyvol : pandas.DataFrame
            Estimated buyer initiated volume. Indexed by estimation intervals. 
            Must contain `buy_vol` and `vol` columns to containing the buyer-
            initiated volume and total trading volume for each interval.
        target : pandas.DataFrame
            Same as `buyvol` but with the true buyer-initiated volume

        Returns:
        --------
        float

        References:
        -----------
        Chakrabarty, B., Pascual, R., Shkilko, A., 2015. Evaluating trade classification 
        algorithms: Bulk volume classification versus the tick rule and the Lee-Ready algo-
        rithm. Journal of Financial Markets 25, 52–79.

        """

        vb = np.minimum(buyvol.buy_vol, target.buy_vol).sum()
        vs = np.minimum(buyvol.vol-buyvol.buy_vol, target.vol-target.buy_vol).sum() 

        s = vb + vs 
        vol = target.vol.sum()
        return s/vol

    
    def tcol_interpolation(self):
        """Interpolate original timestamp in all dataframes. Result saved in new column 'time_inter'."""

        self.df_tr['time_inter'] = interpolate_time(self.df_tr.time.values,self.freq,hj_version=True)
        self.Ask['time_inter'] = interpolate_time(self.Ask.time.values,self.freq,hj_version=True)
        self.Bid['time_inter'] = interpolate_time(self.Bid.time.values,self.freq,hj_version=True)
        
        return


    def into_bins(self,n,bin_type='vol'):
        """Split data into n equally sized bins, either by time or by volume."""

        x = self.df_tr[bin_type].values
        if bin_type=='vol':
            x = np.cumsum(x)

        bins = np.linspace(np.min(x),np.max(x), n+1)[1:] 
        group = np.searchsorted(bins,x)

        self.df_tr['group'] = group
        return


    def get_orderimbalance(self,n,bin_type='vol'):
        """
        Returns the order imbalance computed from individually 
        classified trades over `n` data intervals. (To specify
        the length of the intverals rather than the number, use 
        the `buyvolume` method.) 

        Parameters
        ----------
        n : int
            Number of intervals to split the data into.
        bin_type : str (default : 'vol')
            If 'vol', data are split into `n` volume bins. If
            'time', data are split into `n` time bins.
        
        Returns
        -------
        pandas.DataFrame

        """

        self.into_bins(n,bin_type=bin_type)

        V = self.df_tr[['vol','group']].groupby('group').sum()
        Vb = self.df_tr.loc[self.df_tr.Initiator==1,['vol','group']].groupby('group').sum()
        Vs = self.df_tr.loc[self.df_tr.Initiator==-1,['vol','group']].groupby('group').sum()

        oi = (Vb.subtract(Vs,fill_value=0)).divide(V) 
        return oi.rename(columns={'vol': 'oi'})


    def impl_sf(self,iloc=True,tcol='time'):
        """
        Returns the execution costs for each group of consecutive buyer- or
        seller-initiated trades. 

        ..math:: e_i = o_i \sum_{t=1}^{\tau_i} (p_{i,t} - m_i)v_{t,i}

        where `o_i` is the trade direction of the i-th group of consectutive buyer or seller-
        initiated trades (1 for a buy, -1 for a sell order), `{p_it , v_it}` are the transaction
        prices (in log) and volumes of all trades belonging to the i-th group, and `m_i` is 
        the mid-quote (also in log) at the time of the order.

        Parameters
        ----------
        tcol : str (default 'time')
            Determines which column to use to determine the 
            corresponding mid-quote for each group.

        Returns
        -------
        pandas.DataFrame with group label, execution time of the first transaction
        of the group, total trading volume of the group and the execution cost.

        """

        net = self.df_tr[[tcol,'Initiator','vol','price']].rename(columns={'vol': 'net_vol'})           
        
        runs, start, end = get_runs(net.Initiator.values)

        net['group'] = runs
        if iloc:
            as_of = net.iloc[start][tcol].values
        else:
            as_of = net.loc[start,tcol].values

        midpoint = get_midpoint(self.Ask.rename(columns={tcol: 'time'}),self.Bid.rename(columns={tcol: 'time'}),as_of) 
        midpoint['group'] = net.group.unique()
        midpoint.set_index('group',inplace=True)

        net = net.merge(midpoint, left_on='group', right_index=True, how='left')

        net['impl_shortfall'] = (np.log(net.price) - np.log(net.midpoint))*net['net_vol']*net.Initiator

        sf = net[['group','net_vol','impl_shortfall']].groupby('group').sum()
        sf['time'] = as_of 

        return sf


    def estimate_execost(self,sf,params_only=True,quadratic=False):
        """
        Returns the result from a price impact regression.

        ..math:: e_i = \beta_0 + \beta_1 v_i + \eps_i

        Parameters
        ----------
        sf : pandas.DataFrame
            Contains a column 'impl_shortfall' with the transaction
            costs of the i-th order and a column 'net_vol' with
            the total volume of the i-th order.
        params_only : bool (defaul True)
            If True, returns numpy.array with the parameter estimates.
            Otherwise, statsmodel regression result is returned.
        quadratic : bool (default False)
            If True, use v_i^2 as additional regressor.

        Returns
        -------
        numpy.array of parameter estimates or statsmodels regression
        result object.
        
        """    
        mask = pd.notnull(sf.impl_shortfall)
        data = sm.add_constant(sf.loc[mask,['net_vol']].values)
        if quadratic:
            data = np.hstack([data,sf.loc[mask,['net_vol']].values**2])
            
        model = sm.OLS(sf.loc[mask,'impl_shortfall'].values,data)

        res = model.fit() #cov_type='HC3'
        return res.params if params_only else res