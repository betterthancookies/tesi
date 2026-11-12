# SAGE-smart 2, round per round

Versione attuale (`sage_smart2_socfc`); numeri di β = 0.1. Testo da portare nel report.

## Round 0: l'iscrizione (nessun training)

All'inizio ognuno dei 100 telefoni si presenta al server e gli dice tre cose:
- **dati statici**: quanto è veloce (tier), quanto è grande la batteria e quanti campioni di dati ha;
- **SoC iniziale**: quanto è carico adesso;
- **niente sul carico di lavoro**: il server non sa ancora quanto il telefono è usato dal suo proprietario e assume la media della popolazione, 0.118.

Con questi dati il server fissa la **scadenza** del round: il tempo mediano che serve ai telefoni per fare il lavoro standard (5 epoche con B = 128). Viene **132 s** e resta fissa per tutta la run.

## Ogni round, in 7 passi

**1. Previsione (la stima del SoC).** Per ogni telefono il server ricorda l'ultimo stato riportato e quando l'ha ricevuto. Da lì prevede il SoC di adesso, senza chiedere niente a nessuno:
- quanto si è scaricato nel frattempo, usando un carico che parte dall'ultimo valore riportato e si avvicina alla media di quel telefono in circa 7 minuti (τ = 450 s);
- quanto è probabile che nel frattempo l'abbiano messo in carica, e se stava già caricando, quanto ha recuperato (1C, fino al 100%).

**2. Classifica.** Ogni telefono riceve un punteggio:
- punteggio = 0.4 × SoC previsto + 0.6 × divergenza + 0.3 × anzianità;
- **divergenza**: quanto le classi dei suoi dati sono diverse da quelle degli altri. Si conosce solo dopo che il telefono si è allenato almeno una volta; prima vale la media;
- **anzianità**: da quanti round non viene scelto, divisi per 25, con massimo 1. Chi non è mai stato scelto ha già l'anzianità massima.

**3. Quanti posti.** Il numero di posti è round + 1: 2 al round 1, 3 al round 2, e così via fino a 20 dal round 19. Il tetto è il 20% dei telefoni ancora vivi.

**4. Proposte (la negoziazione).** Il server non manda il modello ai primi 20, ma una **proposta** ai primi 30 della classifica (1.5 volte i posti):
- **10% non raggiungibili**: circa il 10% non risponde, perché è fuori copertura o spento. Il server aspetta 1 s e poi va avanti senza di loro. Non c'è nessun broadcast: lo stato si scopre solo da chi risponde.
- **Chi risponde** manda il suo stato di adesso (SoC, carico, se sta caricando) e dice sì o no.
- **Il rifiuto** arriva solo per energia: se fare il minimo (2 epoche) lo porterebbe sotto il 20% di batteria. Chi sta caricando accetta sempre.
- **Seconda ondata**: se i sì non bastano a riempire i posti, ci sono al massimo 2 ondate.

**5. Scelta fresca.** Fra chi ha detto sì, il server tiene i 20 migliori con lo **stato appena ricevuto**:
- punteggio fresco = (0.4 × SoC di adesso + 0.6 × divergenza + 0.3 × anzianità) × (1 − carico di adesso);
- chi sta caricando conta come SoC 1, perché allenarsi non gli costa batteria;
- un telefono occupato pesa meno, perché nella scadenza farebbe meno passi;
- gli altri che avevano accettato sono **rinviati**: non perdono l'anzianità e il round dopo sono in cima.

**6. Envelope e training sul telefono.** Ai 20 scelti il server manda il modello e un **envelope**: E fra 2 e un massimo calcolato per quel telefono (al più 10), B fra 16, 32, 64 e 128, scadenza 132 s, riserva 20%. Il telefono poi:
- **misura la loss** su 256 campioni. Loss alta vuol dire che i suoi dati sono "nuovi" per il modello, quindi conviene lavorare di più;
- **sceglie E** in proporzione a quella loss e a quanta batteria ha sopra la riserva;
- **sceglie B** partendo dal più piccolo: più passi di SGD nello stesso tempo;
- **lavoro parziale**: se nemmeno 2 epoche stanno nella scadenza, fa solo i passi che ci stanno;
- rimanda il modello aggiornato e un report: passi fatti, distribuzione delle classi, SoC finale.

**7. Aggregazione e aggiornamenti.** Il server:
- **fa la media FedAvg** dei modelli, pesata per i campioni visti davvero: un lavoro parziale pesa meno;
- **aggiorna** la divergenza di chi si è allenato (dalle sue classi), il suo stato noto e la sua anzianità, che torna a 0;
- **valuta** il modello sul test set.

Nel frattempo il **mondo** fa il suo corso:
- tutti i telefoni consumano in idle per tutta la durata del round, cioè 132 s più le attese della negoziazione;
- chi si è allenato paga training, download e upload in 4G, e chi ha risposto paga lo scambio della proposta;
- **ricariche**: qualcuno viene messo in carica a caso, con più probabilità se è scarico;
- **scarichi**: chi arriva a 0 è morto ed esce dal gioco.

## Cosa succede di particolare nei primi round

- **Pochi posti**: 2, 3, 4… fino a 20 al round 19.
- **Esplorazione**: chi non è mai stato scelto ha l'anzianità al massimo e sale in classifica. Nei primi round il server passa così in rassegna telefoni nuovi e intanto impara la loro divergenza e il loro carico.
- **Stato noto vecchio**: all'inizio il server sa solo il SoC dell'iscrizione e il carico medio. La previsione e le risposte alle proposte lo rendono via via più preciso: già al round 3 l'errore sul carico scende da 0.118 a 0.086.
