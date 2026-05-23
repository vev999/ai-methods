Plan:

1. Przetestujemy kilka róznich klasyfikatorów i porównamy ich skuteczność między sobą 
2. Każdy klasyfikator potem porówanamy po odpowiednim dostosowaniu parametrów (w zależności od ich wartości, pozytywnej, negatwnej itp.)

Porównujemy klasyfikatory GNB, KNN i DT. Macierz konfuzji i test Wilxocona zaprezetentujemy dla baseline i najlepszego wariantu

W pliku clfComparsion.py porównujemy każdy klasyfikator bez żadnych zmian, w clfNormalisationComparison.py porównujemy klasyfikatory po normalizacji danych a w clfSamplingComparision.py wpływ Under- i Oversamplingu. Na koniecu na najskuteczniejszej konfiguracji modelu przeprowadzimy 

Porównamy za pomocą metryk - BAC, F1-score macro i Confusion matrix. Ponadto wykonano test 

Wyniki z clfComparasion.py

Confusion matrix GNB:
[[0.08  0.36  0.45  0.11  0.    0.   ]
 [0.042 0.115 0.511 0.304 0.021 0.008]
 [0.008 0.039 0.666 0.251 0.036 0.001]
 [0.002 0.034 0.288 0.493 0.171 0.012]
 [0.001 0.005 0.064 0.384 0.522 0.024]
 [0.    0.    0.006 0.25  0.683 0.061]]
Confusion matrix KNN:
[[0.    0.    0.57  0.28  0.15  0.   ]
 [0.    0.017 0.619 0.315 0.049 0.   ]
 [0.001 0.008 0.67  0.299 0.022 0.   ]
 [0.    0.009 0.45  0.466 0.074 0.001]
 [0.003 0.016 0.227 0.491 0.257 0.006]
 [0.    0.    0.261 0.506 0.233 0.   ]]
Confusion matrix DT:
[[0.03  0.23  0.41  0.28  0.05  0.   ]
 [0.021 0.128 0.487 0.319 0.045 0.   ]
 [0.004 0.031 0.697 0.236 0.031 0.002]
 [0.003 0.023 0.231 0.628 0.105 0.01 ]
 [0.004 0.014 0.116 0.306 0.532 0.028]
 [0.    0.    0.061 0.4   0.394 0.144]]
-----------------------------------------------------------
          GNB              KNN              DT
--------  ---------------  ---------------  --------------
AC        0.551 +- 0.0246  0.504 +- 0.0249  0.62 +- 0.0223
BAC       0.322 +- 0.0397  0.235 +- 0.0151  0.36 +- 0.0421
F1-SCORE  0.314 +- 0.0404  0.235 +- 0.0169  0.36 +- 0.0417
-----------------------------------------------------------
Difference between GNB and KNN is statistically significant.
Difference between GNB and DT is statistically significant.
Difference between KNN and DT is statistically significant.

Interpretacja wyników:
Na "surowych" danych najlepiej poradził sobie klasyfikator Decision Tree osiągając najwyższe wyniki w każdej z trzech metryk. Najgorszy natomiast był klasyfikator KNN co jest oczzekiwanym rezultatem jako, że jest to klasyfikator najbardziej podatny na niezbalansowane zbiory danych. Z macierzy konfuzji można zaobserwować, że najczęściej mylonymi klasami są 5 i 6 (co również jest oczekiwanym rezultatem ze względu na to, że zbiór jest niezbalansowany i jest tych klas najwięcej), natomiast klasy 3 i 8 (najmniej liczne) są najrzadziej poprawnie rozpoznawanymi klasami. Najlepiej rozpoznawanymi klasami są klasy 5 i 6 a najmniej klasa nr. 3. W celu poprawienia wyników zastosujemy metody normalizacji i samplingu danych, tak aby zbalansować niezbalansowany zbiór oraz "wyrównać" odległości między próbkami dla poprawy KNN. 

Wyniki z clfNormalisationComparision.py

Confusion matrix GNB:
[[0.14  0.3   0.46  0.1   0.    0.   ]
 [0.047 0.119 0.496 0.296 0.026 0.015]
 [0.01  0.04  0.664 0.251 0.033 0.002]
 [0.002 0.033 0.286 0.493 0.174 0.013]
 [0.001 0.007 0.063 0.374 0.519 0.037]
 [0.    0.    0.006 0.233 0.689 0.072]]
Confusion matrix KNN:
[[0.    0.19  0.5   0.31  0.    0.   ]
 [0.    0.036 0.515 0.43  0.019 0.   ]
 [0.003 0.012 0.691 0.277 0.016 0.001]
 [0.    0.006 0.33  0.575 0.087 0.001]
 [0.    0.001 0.166 0.449 0.381 0.003]
 [0.    0.    0.083 0.439 0.478 0.   ]]
Confusion matrix DT:
[[0.06  0.24  0.38  0.29  0.03  0.   ]
 [0.043 0.126 0.489 0.289 0.051 0.002]
 [0.006 0.031 0.699 0.226 0.036 0.002]
 [0.002 0.023 0.244 0.611 0.105 0.014]
 [0.    0.012 0.1   0.316 0.545 0.027]
 [0.    0.006 0.072 0.383 0.372 0.167]]
-----------------------------------------------------------
          GNB              KNN              DT
--------  ---------------  ---------------  ---------------
AC        0.55 +- 0.0272   0.573 +- 0.0284  0.616 +- 0.0259
BAC       0.335 +- 0.0469  0.281 +- 0.0199  0.368 +- 0.0449
F1-SCORE  0.328 +- 0.0496  0.283 +- 0.023   0.363 +- 0.0431
-----------------------------------------------------------
Difference between GNB and KNN is statistically significant.
Difference between GNB and DT is statistically significant.
Difference between KNN and DT is statistically significant.

Interpretaja wyników:

Zgodnie z oczekiwaniami normalizacja pozytywnie wpłyneła na KNN zwiększając jego dokładność w każdej metryce. Niemniej jednak, KNN nadal osiąga najgorze wyniki, co jest skutkiem zbyt dużej ilości cech, przy 11 zbyt wiele punktów jest za blisko siebie przez co dokładność KNN maleje. Niezmiennie najlepiej radzi sobie Decision Tree, co wynika z tego, że podczas uczenia klasyfikator ten sam odnajduje istotność cech, co w przypadku tego zbioru ma duże znaczenie (wykazała to analiza istoności cech).

Wyniki z 