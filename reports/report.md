# Chronological stream of thought (in italian)

## Il problema e il metodo 

Il problema \
Si tratta di un problema di inversione in cui abbiamo la funzione $f: X \rightarrow Y$, abbiamo le y osservate e vogliamo ritrovare x. La cosa più semplice sarebbe invertire f su y_obs. Questo non ci porterebbe al risultato esatto in quanto abbiamo y_obs e non y reali, ma ci porterebbe comunque ad un buon risultato. Tuttavia, f è complicata e non invertibile. x ha 5 dimensioni, mentre y ne ha 3, parliamo quindi di 
$$
x = {x_1,x_2,x_3,x_4,x_5}
$$
$$
y = {y_1,y_2,y_3}
$$

Il metodo \
Il metodo quindi conside nel minimizzare su x l'oggetto 
$$
\hat{x} = \argmin_{x} |f(x) - y_{obs}|
$$
o più precisamente 
$$
\hat{x} = \argmin_{x} (|f_1(x) - V_{p,obs}|+ W_1 |f_2(x) - V_{s,obs}| + W_2 |\frac{1}{f_3(x)} - \frac{1}{\sigma_{obs}} )
$$
dove $W_1$ e $W_2$ sono dei pesi che aiutano l'ottimizzazione. 

L'ottimizzazione \
L'oggetto è intricato per questo la scelta converge sugli algoritmi evolutivi. Gli autori del paper hanno usato Simulate Annelaing (SA) mentre noi abbiamo provato PSO e CMAES. 

## Risultati degli evolutivi e discussion
I risultati \
PSO è più efficente e più efficace. Infatti i risultati di PSO sono migliori di quelli di SA su più o meno tutte le metriche, più o meno in ogni casistica. 

Commento sui risultati \
Una cosa interessante è che la loss_hat (cioè la loss sull'oggetto) è circa zero. Questo vuol dire PSO è in grado di trovare i minimi dell'oggetto. Tuttavia, le soluzioni presentano errori. Questo fa supporre due possibilità 
- il landscape dell'oggetto non è descrittivo 
- il landscape presenta molteplici minimi, di cui uno solo è quello che minimizzare la loss_real (cioè la differenza con i veri x)

## Symbolic Regression per un differente landscape
Idea della Symbolic Regression \
L'idea è che quella di usare f e prendere molti x1,x2,x3,...,xn e con f troviamo y1,y2,y3,...,yn, quindi abbiamo $D = \{xi, yi\}_i$. A questo punto trainiamo SR su D per ottenere 3 funzioni 
$$
g_j: X \rightarrow Y_j
$$
con $j \in {1,2,3}$. Ora possiamo ottimizzare 
$$
\hat{x} = \argmin_{x} (|g_1(x) - V_{p,obs}|+ W_1 |g_2(x) - V_{s,obs}| + W_2 |\frac{1}{g_3(x)} - \frac{1}{\sigma_{obs}} )
$$

Perché la SR? \
Siccome una delle ipotesi è che il landscape dell'oggetto della ottimizzazione non fosse descrittivo, abbiamo provato con la SR a ricreare una nuova forward più parsimoniosa per creare un landscape diverso, eventualemente più semplice nel quale 
- 1. Potremmebero descrivere meglio il vero landscape 
- 2. Essendo più smooth potremmo utilizzare metodi di ottimizzazione più forti come least_squares o differential_evolution  

SR strong + least_squares \
Il primo tentativo è stato usare questa procedura senza imporre il vincolo della parsimonia nell $g$ prodotte dalla SR. In seguito all'ottimizzazione abbiamo osservato i risultati e risultavano abbastanza peggiore di SA e PSO.

SR parisimoniosa + differential_evolution \
Il secondo tentativo è stato una SR con vincolo di parsimonia per creare un landscape ancora più smooth e permetterci possibilmente di usare un metodo di ottimizzazione con più garanzie o più costoso. Abbiamo usato differential evolution. I risultati sono ancora stati notevolmente peggiori di quelli ottenuti da SA e PSO.

## ...

## Comprendere il landscape con PSO

Comprendere il landscape \
Per tale motivo abbiamo lanciato PSO con 30 seed diversi, salvato gli archivi di tutte le soluzioni sufficientemente vicino a zero e confrontato i risultati. Tutti i risultati che portavano a loss_hat quasi zero variavano abbastanza su ogni parametro. Tuttavia, esisteva sempre un seed oracle che effettivamente trovava un x_hat che era quasi identico a x. Questo porta a credere che il landscape sia sufficientemente descrittivo in quanto esiste un minimo del landscape che corrisponde al vero x. 

