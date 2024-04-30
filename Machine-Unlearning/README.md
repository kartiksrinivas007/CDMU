

$$

\min L_{D_t} (w + \Sigma)  - \lambda \log |\det(\Sigma)|  \ \ \ \text{without access to $y_t$} 
$$

instead do the following objective

$$
\min L_{D_s} (w + \Sigma)  +  \text{KL}(D_s || D_t)  - \lambda \log |\det(\Sigma)|  \ \ \ \text{without access to $y_t$}
$$


$$

\min L_{D_s^r} (w + \Sigma) + \text{KL}(D_s^r || D_t)  - \lambda \log |\det(\Sigma)|  \ \ \ \text{without access to $y_t$}
$$

the only emaining part I need to scrub is the hessian of the distirution difference of the functions, 
in your anlysis nowehre did you assume that the loss $L_{D_r}$ was required to abe a cross entropy loss on the dataset (except the simple fisheer information part, according to me), 

some importance , you need to scrub the weights that were learnt during the optimization of the distribution divergence $D_{s}^r$ and $D_{s}$, with the fixed distribution $D_t$
Find a scrubbing that removes features that contributed to the KL divergence minimization on the forget set $D_s^f$ and keep the ones that contributed to the minimization on $D_s^f$


- KirkPatrick will tell me how to analyze the hessian
- Koh and Liang will tell me how to retrace my steps back to becomes similar to both 