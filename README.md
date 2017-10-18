Y - binary classification. 0 skewed
F1(int) - skewed 0 category (max 18)
F2(int) - skewed 0 category (max 98)
F3(float) - skewed 0 numerical category (max 29110.04058)
F4(int) - skewed 0 category  (max 8)
** F5(float) - skewed 0 numerical category (max 10.0) very gradual
-- F6(int) - skewed 0 category gigantic outlier / range. (8billion). extremely sparse.
F7(int) - skewed 0 category (max 23)
F8(int) - skewed 0 category (max 9)
F9(int) - skewed 0 category (max 59249). extremely sparse
F10(int) - skewed 0 category (max 54). nice skew 0-10
** F11(int) - category (mean 40), nicely distributed histogram with NO data from two categories (35 and 45).
F12(int) - skewed 0 category (max 12)
F13(int) - skewed 0 category (max 10)
F14(int) - skewed 0 category (max 98). nice 0 skew
F15(int) - skewed 0 category (max 10)
F16(float) - skewed numerical category (max 35807.1)
F17(int) - skewed 0 category (max 10)
*** F18(int) - very beautiful histogram with slight skew. mean around 130
-- F19(float) - skewed 0 numerical category gigantic outlier / range. (3billion). extremely sparse.
F20(int) - skewed 0 category (max 10)
-- F21(int) - skewed 0 category gigantic outlier / range. (600thousand). extremely sparse.
** F22(int) - interesting skew for values below 30. giant mode around 8
F23(float) - skewed 0 numerical category (max 29110.04058)
F24(int) - skewed 0 category (max 10)
F25(int) - skewed 0 category (max 98)
*** F26(int) - again, very beautiful histogram with slight skew. mean around 50
F27(float) - skewed 0 numerical category (max 29110.04058)


### Summary:
Looks like some of the features may be DUPLICATES of  others.
Some of it could also just flat out be noise.
Alot of stuff is skewed toward 0, aside from some of the interesting histogram plots. it's also possible the histogram plots could just be NOISE.

odd pearson R of 0.42 between F22 and F10

Evident correlations:
--positive
F2 and F14 - same idea. two gigantic leverage points.
F2 and F25 - Not really correlated. just two gigantic leverage points.
F3 and F23 - Highly colinear, but not completely. neglible roundoff error. correlation coeff is 1. 
F10 and F22 - 
F14 and F25 - same idea. two gigantic leverage points. 
F18 and F26 - Completely Colinear (pearsonr = 1 LOL) f18 = b0 + b1f26

--negative
F5 and F18 (and 26)
F18 and Y

## unskewing features
general ideas: gigantic uniformish skew unskews well with log
others unskew well with sqrt / boxcox
http://shahramabyari.com/2015/12/21/data-preparation-for-predictive-modeling-resolving-skewness/

F19 seems to unskew nicely with log1p
F22 seems to unskew nicely with np.sqrt

just use boxcox to unskew everything. 
F3 - works extremely nicely
F22, F27 (caution around boundaries), F19
