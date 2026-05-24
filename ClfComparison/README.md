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

Jako, że wyniki po normalizacji były generalnie rzecz biorac lepsze, to dalsze testy kontynuowalismy na znormalizowanych danych

Wyniki z clfSamplingComparision.py

Sampling metod: ROS
          GNB              KNN              DT
--------  ---------------  ---------------  ---------------
AC        0.344 +- 0.04    0.508 +- 0.0245  0.611 +- 0.0234
BAC       0.323 +- 0.0691  0.329 +- 0.0337  0.35 +- 0.0382
F1-SCORE  0.225 +- 0.0237  0.297 +- 0.0241  0.345 +- 0.0353
-----------------------------------------------------------
Difference between GNB and KNN is statistically significant.
Difference between GNB and DT is statistically significant.
Difference between KNN and DT is statistically significant.
-----------------------------------------------------------
Sampling metod: RUS
          GNB              KNN              DT
--------  ---------------  ---------------  ---------------
AC        0.295 +- 0.0486  0.308 +- 0.0607  0.281 +- 0.0406
BAC       0.334 +- 0.0766  0.315 +- 0.0589  0.3 +- 0.066
F1-SCORE  0.201 +- 0.0273  0.207 +- 0.0311  0.195 +- 0.0245
-----------------------------------------------------------
Difference between GNB and KNN is not statistically significant.
Difference between GNB and DT is not statistically significant.
Difference between KNN and DT is not statistically significant.
-----------------------------------------------------------
Sampling metod: SMOTE
          GNB              KNN              DT
--------  ---------------  ---------------  ---------------
AC        0.313 +- 0.0301  0.465 +- 0.0289  0.577 +- 0.025
BAC       0.317 +- 0.055   0.352 +- 0.0555  0.357 +- 0.0502
F1-SCORE  0.213 +- 0.0191  0.293 +- 0.0259  0.338 +- 0.0364
-----------------------------------------------------------
Difference between GNB and KNN is statistically significant.
Difference between GNB and DT is statistically significant.
Difference between KNN and DT is statistically significant.
-----------------------------------------------------------
Confusion matrix GNB - ROS:
[[0.26  0.45  0.19  0.1   0.    0.   ]
 [0.342 0.211 0.24  0.138 0.042 0.028]
 [0.144 0.147 0.465 0.135 0.076 0.034]
 [0.085 0.127 0.179 0.221 0.203 0.184]
 [0.03  0.054 0.033 0.112 0.359 0.412]
 [0.    0.028 0.    0.05  0.494 0.428]]
Confusion matrix GNB - RUS:
[[0.43  0.29  0.12  0.09  0.05  0.02 ]
 [0.272 0.306 0.225 0.102 0.062 0.034]
 [0.112 0.215 0.372 0.176 0.085 0.041]
 [0.082 0.164 0.212 0.2   0.191 0.152]
 [0.041 0.074 0.057 0.151 0.323 0.354]
 [0.    0.044 0.017 0.144 0.417 0.378]]
Confusion matrix GNB - SMOTE:
[[0.2   0.47  0.22  0.11  0.    0.   ]
 [0.296 0.342 0.191 0.119 0.042 0.011]
 [0.11  0.262 0.419 0.107 0.079 0.023]
 [0.066 0.221 0.154 0.182 0.219 0.157]
 [0.024 0.071 0.04  0.138 0.36  0.366]
 [0.    0.    0.    0.117 0.483 0.4  ]]
Confusion matrix KNN - ROS:
[[0.02  0.7   0.09  0.19  0.    0.   ]
 [0.081 0.204 0.326 0.317 0.072 0.   ]
 [0.011 0.089 0.574 0.25  0.07  0.007]
 [0.008 0.067 0.243 0.437 0.229 0.016]
 [0.002 0.01  0.06  0.224 0.654 0.051]
 [0.    0.    0.044 0.167 0.706 0.083]]
Confusion matrix KNN - RUS:
[[0.4   0.32  0.12  0.12  0.03  0.01 ]
 [0.26  0.275 0.238 0.125 0.058 0.043]
 [0.098 0.234 0.373 0.195 0.063 0.037]
 [0.068 0.172 0.221 0.231 0.18  0.128]
 [0.031 0.079 0.081 0.185 0.344 0.279]
 [0.    0.044 0.033 0.178 0.483 0.261]]
Confusion matrix KNN - SMOTE:
[[0.19  0.55  0.08  0.18  0.    0.   ]
 [0.13  0.285 0.232 0.291 0.062 0.   ]
 [0.034 0.145 0.528 0.219 0.062 0.012]
 [0.018 0.113 0.219 0.388 0.191 0.07 ]
 [0.007 0.016 0.052 0.204 0.585 0.137]
 [0.    0.    0.044 0.144 0.678 0.133]]
Confusion matrix DT - ROS:
[[0.02  0.29  0.44  0.21  0.04  0.   ]
 [0.032 0.119 0.462 0.319 0.06  0.008]
 [0.006 0.034 0.694 0.233 0.031 0.002]
 [0.004 0.02  0.236 0.613 0.115 0.012]
 [0.001 0.014 0.108 0.325 0.521 0.031]
 [0.011 0.011 0.056 0.433 0.361 0.128]]
Confusion matrix DT - RUS:
[[0.31  0.3   0.15  0.16  0.06  0.02 ]
 [0.275 0.249 0.196 0.155 0.077 0.047]
 [0.136 0.204 0.326 0.207 0.091 0.035]
 [0.099 0.158 0.201 0.223 0.182 0.137]
 [0.04  0.099 0.084 0.169 0.316 0.292]
 [0.028 0.067 0.039 0.189 0.306 0.372]]
Confusion matrix DT - SMOTE:
[[0.06  0.33  0.38  0.22  0.01  0.   ]
 [0.066 0.17  0.409 0.277 0.077 0.   ]
 [0.018 0.074 0.653 0.211 0.041 0.003]
 [0.013 0.052 0.204 0.56  0.143 0.029]
 [0.003 0.024 0.084 0.28  0.544 0.064]
 [0.    0.    0.033 0.356 0.461 0.15 ]]

Interpretacja wyników:
Żadna z metod samplingu nie polepszyła wyników klasyfikatorów. RUS osiągnął najgorsze rezultaty, co wynika z tego że najliczniejsze klasy zostały "ucięte" do rozmiaru najmniejszych, przez co model miał bardzo mało danych do uczenia się. Ciekawą obserwacją dotyczącą RUS jest fakt, że najskuteczniej zwięszył on dokładność modeli w rozpoznawaniu klas skrajnych, ale tak dużym kosztem dokładności rozpoznawania klas środkowych (4-7), że końcowo wyniki z tej metody samplingu są najgorsze. ROS i SMOTE również nie zwiększyły całkowitej dokładności klasyfikatorów. Może to wynikać z tego, że zarówno SMOTE jak i ROS działają na zasadzie dodawania syntetycznych próbek na podstawie istniejących już - problem jest taki, że próbek klasy 3 jest bardzo mało i są one w "jednym miejscu" więc dodanie wiecej próbek w tym miejscu nie zmienia dokładności modelu.

Porównanie statystycznej istotności wyników normalizacji i samplingu do baseline:


Próba optymalizacji każdego z klasyfikatorów:

