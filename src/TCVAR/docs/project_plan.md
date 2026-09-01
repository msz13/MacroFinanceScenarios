### Problem:
warianty:
- porównanie forecasting performance modeli
- zrozumienie wpływu trendów na wycenę i wzrost earnings

Planowane analizy:
- USA tcvar porównanie modeli
- international tcvar dla g7 lub części
- model z polską

Long term asset returns scenarios - long term macro trends
We deal with long term return forecast, in tradition of barberis, cambpel i nni. 
we connect to tradition of building block of asset retuns.
We introduce trends as in norges and acwi

problem - z czego wynikaja obecne wyceny
- zmienila sie fair value - wiec nie bedzie mean reversion lub niewielkie, zmienila sie bo:
  - zmienily sie dlugookresowe oczekiwania  co do wzrostu, mogl sie zmienic prognoza wzrostu pkb, albo udzial zysko w pkb
  - zmienila sie dlugookresowa stopa dyskontowa
- nie zmienila sie fair value - bedziemy mieli mean reversion 

#### Analises contribution
- wpisuje się w nurt, gdy dy nie jest stacjonarne, podlega trendom, szukania fair vale cape, ale wcześniejsze próby badały tylko pd i dividends growth (valuation ratios and cashflow trens, score dirven model, decomposycje trendu dy jednoczynnikowe), albo vars bez trendow (vanguard davis, ale rozszerzenie new trends)
- wpisuje się nurt sum of the parts, decomposition
- wpisuje się w debatę what drives
- nowocia jest nie model obiektywny oczekiwn a subiektywny



Powónwujemy rone specyfikacje trendow z modelem bez trendow - villani.

-	Long term extension of sum of the parts model
-	#### Modele do porównania w forecasts:
  - MVormal - basowy
  -	Steady state bvar with SV
  -	Steady state bvar - tradycyjny - returns i pd i zmienne macro
  -	TC VAR-SV – jeden trend dla zmiennej – wtedy jest podobne do steady state bvar z time varing state
  -	Zwykłdy TVP-VAR – czy identygikacja trendów coś daje
  -	TC-VAR with SV
  -	Ewentualnie oba bez SV 
  -    bvar dla dy or cape, z average div yeld and earnings growth
  -    

-	#### In sample and Out of sample comparison
o	Porównanie moments of simulated and real data dla całej próbki
o	Predictive likehood /Marginal predictive likehood - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2507827
Marginalized Predictive Likelihood Comparisons of Linear Gaussian State-Space Models with Applications to DSGE, DSGE-VAR, and VAR Models
o	Porównanie simulated moments dla kilku wybranych punktów, np. przed i po kryzysie, gdy parametry są oszacowane dla całej próbki
- The density forecasts, constructed by fitting a Gaussian kernel to the simulated forecasts, are evaluated with the log predictive score (LPS). - dla symulowanych danych oblicza się kernel density
o	Porfolio utility dla parametrów oszacowanych dla całej próbki
-	#### out of smaple
o	Moments of simulated data in periods
o	Percentailse of simulated data in periods
o	Forecast error and density – to samo co marginal predictive likehod
o	Porfolio utlility 
o	Different correlation on different regimes (macro regimes, stgflations, rozwój, itd) – impulse response functions
o	Inne założenia aktywów, fat tails, itd
- #### simulated data test
- simulate data from steady state bvar, and check tcvar, and simulate from tcvar, and estimate SS-BVAR, simulate with and withoud Stochastic volatility.

#### Pomysły na wariacje modelu"
- dane
  - tylko cash flow i valuations, div gorwth/dy or eps growth/ep + r star estimated elsware np. hlw
  - tylko asset returns
  - variables levels zamiast changes
  - pd changes zamiast level
  - dodatkowo konsumpcja
  - dodatkowo expected volatility/equity risk premium
  - survey data
  - yeld curce - rożne maturity yeld curve
- rodzaje modeli
  - tc var
  - trend + ar_cycle  + corelated innovations
  - time varing coefficient params
- różne measurement equtations - definicje trendów poszczególnych variables,
  - earnigns growth tylko gdp/consumtion growth trend vs addtional earnigns growth trend
  - dy/pe jako expectatons, równe definicje expectations:
    - trends + cycle
    - loadings * trends +cycle
    - objective expectations - jak w present value model, parametry wynikają z danych
    - trend + measurement error/ measurement error as MA
    - extrapolative belief
    - extrapolative belief + AR
    - change of believes - v*(gd - Egd_{t-1})
  - priors 
    - intial trend values (means) - albo mean of presample albo wartość obliczone inna metodą, np. hodrick prescot filter
    - initial volatility state - unconditioanl mean and variance of ar process
    - volatility params - ar coefficient i variance fix, or estimated
  - long term yelds as nielson siegel yeld curve (Natural rate chimera .. art)
- #### wyzwania
  - explosive trends - mogą się odchylić do nierealnych wartości rozwiązania:
    - trend ar proces,
    - tighter priors on trend innovation,
    - trend truncated normal bounded do jakiś wartości, np. trend inflacji dla USA  (0%-4%/-1%-5%)  
