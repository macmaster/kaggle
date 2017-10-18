Y - binary classification. 0 skewed
F1(int) - skewed 0 category (max 18)
F2(int) - skewed 0 category (max 98)
F3(float) - skewed 0 numerical category (max 29110.04058)
F4(int) - skewed 0 category  (max 8)
* F5(float) - skewed 0 numerical category (max 10.0) very gradual
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
* F22(int) - interesting skew for values below 30. giant mode around 8
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
F2 and F14
F2 and F25
F3 and F23
F14 and F25
F18 and F26
--negative
F5 and F18
F5 and F26
F18 and Y

