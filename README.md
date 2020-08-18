# Trade-Classification-Algorithms

Module to classify financial markets transactions data into 
buyer- and seller-initiated trades. 

Available methods are the Lee-Ready algorithm (Lee and Ready, 1991),
the Bulk-Volume classification algorithm (Easley et al., 2012), the 
algorithm of Ellis et al. (2000), the algorithm of Chakrabarty et al. 
(2007) and the Full-Information algorithm of Jurkatis (2020). 

Also allows the estimation of order imbalances and transaction costs.

# Dependencies
- numpy
- pandas
- cython
- scipy
- warnings
- statsmodels

# Installation
To compile the `tradeclassification_c.pyx` file into the required C file
```
$ python setup.py build_ext -i
```

# Usage
## Trade Classification
```python
from classifytrades import TradeClassification 

tc = TradeClassification(df,Ask=Ask,Bid=Bid)
tc.classifytrades(method='lee_ready', freq=0, reduce_precision=True)

print(tc.df_tr.head())
```
Other method arguments are `'clnv'`, `'emo'`, `'bvc'`, `'ds_1'`, `'ds_2'`, `'ds_3'`.

- `df` : pandas.DataFrame with transaction data. 
Assumed to be deduplicated, ie. only one record per trade between two counterparties (only relevant for FI and BVC).
The dataframe must contain at least a `time` column containing the transaction times measured in seconds (i.e. timestamps of precision higher than seconds are expressed as floats) and a `price` column containing the transaction prices. For the FI algorithm the dataframe must also contain a `vol` column with the number of shares exchanged in the transaction.
- `Ask` : pandas.DataFrame (optional; default None).
For the FI, LR, EMO and CLNV algorithms order book data are required, as well as for computing transaction costs. The dataframe must contain a `time` column indicating the time of the  quote change expressed in seconds and a `price` column with the best ask. For the FI algorithm the dataframe must also contain the volume available at the best ask.
- `Bid` : analogous to `Ask`. 


### The FI algorithm
The FI algorithm comes in three different versions, depending on the data structure (see Jurkatis, 2020).

#### Data Structure 1
For data where each transaction at the ask or bid must have a corresponding reduction in the volume available at the respective quote and where trades and quotes can be assumed to be recorded in the same order in which they were executed use `method = 'ds_1'`.  

#### Data Structure 2
For data where, contrary to DS1, quote changes that are due to the same trade are aggregated, use `method='ds_2'`. Aggregated quote changes mean that, for example, a buy order for 100 shares that is executed against two
standing sell limit-orders for 50 shares each will be reflected in a single change at the ask of a total change in volume of 100 shares, instead of two separate changes of 50 shares.

#### Data Structure 3
If in addition to DS2 one cannot assume that trades and quotes are in the correct order, use `method='ds_3'`.

## Order Imbalances
The module also allows to compute the order imbalance, defined as the buyer-initiated volume minus seller-initiated volume over total volume over a given measurement interval. 

```python
oi = tc.get_orderimbalance(10,bin_type='vol')
```
splits the data into 10 equal volume intervals (individual trades are not broken up between intervals so differences in total volume between the intervals may remain) and computes the order imbalance for each.

To control the length of the intervals rather than the number use
```python
Vb = tc.buyvolume(window=10,window_type='time')
```
The call returns the buyer-initiated volume and total volume for the trading data split into intervals of 10 seconds.

## Transaction Costs
The classification result can also be used to compute the execution costs of each group of consecutive buyer- and seller-initiated trades. 

```python
execost = tc.impl_sf()
```
which can susequently be used in a price impact regression. 

```python
propcost = tc.estimate_execost(execost)
```

# References

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