# Etap 2 - porównanie klasyfikatorów (normalizacja, sampling, SelectKBest)

## Metryki oceny

Wyniki porównywane są za pomocą trzech metryk:
- **F1-score macro** — główna metryka, traktuje każdą klasę równo niezależnie od liczebności
- **Balanced Accuracy (BAC)** — pomocnicza, neutralizuje efekt dominacji klas 5 i 6
- **Accuracy (AC)** — kontekstowa, pokazuje jak myląca jest przy niezbalansowanym zbiorze
- **Confusion matrix** — diagnostyczna, pokazuje gdzie konkretnie modele się mylą

Istotność statystyczna różnic między wariantami sprawdzana jest testem Wilcoxona (dla alphy = 0.05),
ze względu na brak rozkładu normalnego w danych.

Wszystkie eksperymenty przeprowadzono przy użyciu RepeatedStratifiedKFold
(n_splits=5, n_repeats=10), co daje 50 foldów na klasyfikator.

## Eksperymenty

### 1. Baseline vs Normalizacja (`normalizationBaselineComp.py`)

Porównanie klasyfikatorów na surowych danych oraz po zastosowaniu StandardScaler.

**Wyniki baseline:**
'''
Wariant: Baseline
Confusion matrix GNB:
[[0.11  0.38  0.41  0.1   0.    0.   ]
 [0.047 0.117 0.515 0.289 0.021 0.011]
 [0.009 0.042 0.659 0.252 0.037 0.   ]
 [0.002 0.034 0.287 0.498 0.169 0.01 ]
 [0.    0.008 0.062 0.39  0.52  0.02 ]
 [0.    0.    0.    0.261 0.683 0.056]]
Confusion matrix KNN:
[[0.    0.    0.53  0.36  0.11  0.   ]
 [0.    0.015 0.611 0.321 0.053 0.   ]
 [0.001 0.008 0.671 0.298 0.022 0.   ]
 [0.    0.008 0.464 0.457 0.071 0.   ]
 [0.001 0.014 0.225 0.494 0.26  0.006]
 [0.    0.    0.239 0.478 0.283 0.   ]]
Confusion matrix DT:
[[0.02  0.24  0.52  0.16  0.06  0.   ]
 [0.028 0.123 0.491 0.302 0.055 0.002]
 [0.007 0.033 0.704 0.216 0.038 0.002]
 [0.001 0.02  0.24  0.623 0.107 0.009]
 [0.003 0.009 0.097 0.319 0.546 0.028]
 [0.    0.006 0.072 0.372 0.389 0.161]]
-----------------------------------------------------------
          GNB              KNN              DT
--------  ---------------  ---------------  ---------------
AC        0.549 +- 0.0265  0.501 +- 0.021   0.623 +- 0.0228
BAC       0.327 +- 0.0498  0.234 +- 0.0128  0.363 +- 0.04
F1-SCORE  0.319 +- 0.0518  0.234 +- 0.0142  0.359 +- 0.0385
-----------------------------------------------------------
Difference between GNB and KNN is statistically significant.
Difference between GNB and DT is statistically significant.
Difference between KNN and DT is statistically significant.

Wariant: Normalization
Confusion matrix GNB:
[[0.08  0.36  0.46  0.1   0.    0.   ]
 [0.049 0.096 0.534 0.292 0.019 0.009]
 [0.01  0.039 0.666 0.249 0.035 0.   ]
 [0.002 0.033 0.293 0.489 0.173 0.011]
 [0.    0.005 0.065 0.384 0.525 0.021]
 [0.    0.    0.    0.228 0.706 0.067]]
Confusion matrix KNN:
[[0.    0.13  0.52  0.35  0.    0.   ]
 [0.    0.034 0.502 0.445 0.019 0.   ]
 [0.003 0.01  0.693 0.277 0.016 0.   ]
 [0.001 0.006 0.327 0.578 0.087 0.001]
 [0.    0.002 0.157 0.453 0.388 0.   ]
 [0.    0.    0.094 0.489 0.417 0.   ]]
Confusion matrix DT:
[[0.02  0.22  0.47  0.27  0.02  0.   ]
 [0.04  0.119 0.458 0.321 0.06  0.002]
 [0.005 0.031 0.699 0.228 0.034 0.003]
 [0.002 0.022 0.24  0.621 0.104 0.011]
 [0.001 0.011 0.105 0.31  0.547 0.026]
 [0.    0.006 0.094 0.383 0.378 0.139]]
-----------------------------------------------------------
          GNB              KNN              DT
--------  ---------------  ---------------  ---------------
AC        0.548 +- 0.0246  0.575 +- 0.0233  0.619 +- 0.0237
BAC       0.321 +- 0.0439  0.282 +- 0.0158  0.357 +- 0.0365
F1-SCORE  0.313 +- 0.0416  0.285 +- 0.0189  0.358 +- 0.0382
-----------------------------------------------------------
Difference between GNB and KNN is statistically significant.
Difference between GNB and DT is statistically significant.
Difference between KNN and DT is statistically significant.
-----------------------------------------------------------
Wilcoxon: baseline vs normalization
GNB: normalization is not  significantly different than baseline.
KNN: normalization is  significantly different than baseline.
DT: normalization is not  significantly different than baseline.
'''

**Wnioski:**
Na surowych danych najlepiej poradził sobie DT, najgorzej KNN. Jest to oczekiwany wynik —
KNN opiera się na odległościach w przestrzeni cech, przez co jest szczególnie wrażliwy na
niezbalansowane skale cech. KNN jest jedynym klasyfikatorem w którym można zaobserwować
statystycznie znaczącą rożnice po normalizacji a bez niej. Dla DT i GNB nie zaobsrwowano
znaczącej różnicy, co wynika ze specyfikacji tych klasyfikatorów.

Z macierzy konfuzji wynika że wszystkie modele najczęściej mylą klasy 5 i 6, a klasy skrajne 
(3 i 8) są rozpoznawane najsłabiej — KNN nie rozpoznaje ich prawie wcale.

Jako, że wyniki po normalizacji były lepsze, to dalsze testy kontynuowalismy na znormalizowanych danych

### 2. Metody samplingu (`clfSamplingComparision.py`)

Na danych znormalizowanych przetestowano trzy metody balansowania zbioru:
Random Over Sampling (ROS), Random Under Sampling (RUS) i SMOTE.

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

**Wnioski:**
Żadna metoda samplingu nie poprawiła wyników względem samej normalizacji. RUS osiągnął
najgorsze wyniki — usunięcie próbek klas dominujących sprowadza zbiór treningowy do
rozmiaru najmniejszej klasy (10 próbek), co drastycznie ogranicza ilość danych do uczenia.

ROS i SMOTE nie przyniosły poprawy pomimo wyrównania liczebności klas. Może to wynikać z 
tego, że wina różnych klas mają te same cechy - wobec tego dodanie kolejnych w tej samej przestrzeni 
nie poduje zwiększenia dokładności modelu, a zwiększenie szumu danych. Dodatkowo, probllem jest w
tym, że jakość wina zależy od gustu ludzi je oceniających, czego nie można ująć w danych. Fizykochemiczne
parametry win mogą tylko z przybliżeniem określić jego jakość.

Ciekawą obserwacją jest to że RUS poprawił rozpoznawanie klas skrajnych (3 i 8)
kosztem drastycznego pogorszenia klas środkowych. Potwierdza to tezę o nakładaniu się cech win o różnych
klasach - po wyrównaniu liczby próbek w przestrzeni wszystkie próbki są na tyle blisko siebie, że model 
z łatwością je myli.

### 3. Selekcja cech — SelectKBest (`normalizationSelectKBest.py`)

Na danych znormalizowanych przetestowano SelectKBest z k=6. Wartość k dobrano
na podstawie wcześniejszej analizy istotności cech (F-score), z której wynika że
6 spośród 11 cech ma znaczący wpływ na jakość wina, a pozostałe 5 znikomy.

Wariant: SelectKBest
Confusion matrix GNB:
[[0.13  0.17  0.52  0.18  0.    0.   ]
 [0.028 0.091 0.611 0.258 0.011 0.   ]
 [0.005 0.015 0.737 0.229 0.014 0.001]
 [0.001 0.015 0.343 0.517 0.119 0.005]
 [0.    0.003 0.065 0.499 0.422 0.012]
 [0.    0.    0.006 0.433 0.494 0.067]]
Confusion matrix KNN:
[[0.08  0.13  0.48  0.31  0.    0.   ]
 [0.    0.068 0.494 0.421 0.017 0.   ]
 [0.002 0.012 0.716 0.258 0.012 0.   ]
 [0.    0.008 0.325 0.582 0.083 0.001]
 [0.    0.001 0.147 0.431 0.416 0.006]
 [0.    0.    0.106 0.511 0.383 0.   ]]
Confusion matrix DT:
[[0.03  0.16  0.49  0.27  0.05  0.   ]
 [0.025 0.17  0.409 0.311 0.083 0.002]
 [0.006 0.031 0.699 0.232 0.029 0.002]
 [0.005 0.028 0.232 0.615 0.109 0.011]
 [0.001 0.021 0.1   0.312 0.535 0.032]
 [0.    0.006 0.061 0.344 0.433 0.156]]
-----------------------------------------------------------
          GNB              KNN              DT
--------  ---------------  ---------------  ---------------
AC        0.577 +- 0.0216  0.592 +- 0.0198  0.617 +- 0.0296
BAC       0.328 +- 0.0484  0.31 +- 0.0322   0.367 +- 0.0421
F1-SCORE  0.334 +- 0.0548  0.319 +- 0.0423  0.363 +- 0.041
-----------------------------------------------------------
Difference between GNB and KNN is not statistically significant.
Difference between GNB and DT is statistically significant.
Difference between KNN and DT is statistically significant.

Wariant: Normalization
Confusion matrix GNB:
[[0.07  0.39  0.42  0.12  0.    0.   ]
 [0.042 0.121 0.517 0.279 0.03  0.011]
 [0.008 0.045 0.66  0.25  0.036 0.001]
 [0.002 0.033 0.292 0.487 0.172 0.013]
 [0.    0.01  0.061 0.382 0.518 0.03 ]
 [0.    0.    0.    0.278 0.661 0.061]]
Confusion matrix KNN:
[[0.    0.21  0.44  0.35  0.    0.   ]
 [0.    0.03  0.53  0.421 0.019 0.   ]
 [0.002 0.011 0.691 0.277 0.018 0.001]
 [0.    0.005 0.327 0.581 0.085 0.001]
 [0.    0.002 0.155 0.45  0.391 0.002]
 [0.    0.    0.067 0.45  0.483 0.   ]]
Confusion matrix DT:
[[0.02  0.13  0.54  0.28  0.03  0.   ]
 [0.04  0.109 0.464 0.33  0.055 0.002]
 [0.002 0.035 0.701 0.226 0.035 0.002]
 [0.003 0.029 0.241 0.612 0.106 0.01 ]
 [0.002 0.011 0.102 0.297 0.561 0.027]
 [0.    0.011 0.05  0.4   0.394 0.144]]
-----------------------------------------------------------
          GNB              KNN              DT
--------  ---------------  ---------------  ---------------
AC        0.545 +- 0.0224  0.576 +- 0.0229  0.618 +- 0.0283
BAC       0.32 +- 0.0425   0.282 +- 0.0138  0.358 +- 0.0371
F1-SCORE  0.313 +- 0.042   0.285 +- 0.0157  0.353 +- 0.0362
-----------------------------------------------------------
Difference between GNB and KNN is statistically significant.
Difference between GNB and DT is statistically significant.
Difference between KNN and DT is statistically significant.
-----------------------------------------------------------
Wilcoxon: baseline vs normalization
GNB: SelectKBest is not significantly different than plain normalization.
KNN: SelectKBest is significantly different than plain normalization.
DT: SelectKBest is not significantly different than plain normalization.

**Wnioski:**
SelectKBest poprawił wyniki wszystkich trzech klasyfikatorów, jednak poprawa istotna 
statystycznie wystąpiła tylko dla KNN. Jest to oczekiwany efekt ze względu na zasadę 
działania klasyfikatora KNN - traktuje każdą cechę jako równie ważną przy obliczaniu 
odległości, więc usunięcie nieistotnych cech bezpośrednio poprawia jakość miary odległości.

DT samodzielnie odkrywa istotność cech podczas budowy drzewa, więc SelectKBest
wnosi tu minimalną zmianę. Skuteczność GNB wzrosła bardziej niż DT, ale nie istotnie statystycznie. 

## Podsumowanie

Spośród wszystkich przetestowanych konfiguracji najlepsze wyniki F1 macro osiągnął
DT z normalizacją i SelectKBest (k=6), jednak żadna metoda nie przyniosła
przełomowej poprawy. Główną przeszkodą jest nakładanie się klas w przestrzeni
cech — co wynika z subiektywnej oceny jakości win, które zależą głownie od smaku wina
(którego nie da się ująć w danych liczbowych) a nie od samych cech fizycznochemicznych. 
 
Z testów Wilcoxona można zauwazyć, że KNN reaguje najsilniej na każdą transformację danych 
(normalizacja, selekcja cech), podczas gdy DT pozostaje najbardziej stabilny i osiąga 
najlepsze wyniki niezależnie od wariantu.