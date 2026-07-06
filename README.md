# Capacitated Vehicle Routing Problem (CVRP) - Algoritmo Immunologico (IA)

Questo progetto implementa un **Algoritmo Immunologico (IA)** ispirato al principio della selezione clonale (CLONALG) per risolvere il *Capacitated Vehicle Routing Problem (CVRP)*.

Il progetto è stato sviluppato per il corso di **Heuristics & Metaheuristics for Optimization & Learning** (Prof. Mario Pavone). (Algoritmo 2 tra le scelte proposte).

## Struttura della Repository

- `src/`: Contiene il codice sorgente dell'algoritmo immunologico e degli operatori genetici/immunitari (ipermutazione, sostituzione, ecc.).
- `instances/`: Contiene i dataset delle istanze utilizzate per i test (Set A, B, E, P). Le istanze devono essere scaricate da CVRPLIB.
- `results/`: Directory in cui vengono salvati i risultati degli esperimenti (file JSON, grafici di convergenza, tabelle riassuntive).
- `report/`: Contiene la relazione finale del progetto scritta in LaTeX.
- `main.py`: Entry point per eseguire test rapidi o per eseguire l'algoritmo su una singola istanza.
- `run_experiments.py`: Script principale per eseguire il batch completo di esperimenti per tutte le istanze, calcolando best, mean, std e altri parametri.
- `interface/`: **Interfaccia grafica in Tkinter** per configurare i parametri e seguire l'esperimento in tempo reale (vedere sezione sotto).
- `progetto-20260701.pdf`: Documento originale contenente le specifiche del progetto.

## Requisiti

Assicurati di avere `Python 3` installato. Puoi installare le dipendenze necessarie eseguendo:

```bash
pip install -r requirements.txt
```

## Download Istanze

Prima di avviare gli esperimenti, scarica le istanze dal portale [CVRPLIB](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/instances). I file con estensione `.vrp` vanno estratti e posizionati all'interno della cartella `instances/`. Le istanze richieste dal progetto sono:

- **Set A**: `A-n45-k7`, `A-n60-k9`, `A-n80-k10`
- **Set B**: `B-n56-k7`, `B-n66-k9`, `B-n78-k10`
- **Set E**: `E-n76-k8`, `E-n101-k14`
- **Set P**: `P-n50-k10`, `P-n101-k4`

## Esecuzione degli Esperimenti

Il protocollo sperimentale del progetto richiede l'esecuzione di **5 run** per ogni istanza con un limite massimo di **350.000 valutazioni** della funzione di fitness (FE).

### 1. Esecuzione batch completa

Per avviare tutti gli esperimenti e generare automaticamente i grafici e le tabelle riassuntive (sia testuali che LaTeX) nella cartella `results/`, esegui:

```bash
python run_experiments.py
```

*(Nota: L'esecuzione di tutte le run su tutte le istanze richiederà diverso tempo)*

### 2. Esecuzione su singola istanza

Tramite `main.py` è possibile testare l'algoritmo su una specifica istanza:

```bash
python main.py --instance A-n45-k7
```

### 3. Test Rapido (Debug)

Per verificare il corretto funzionamento dell'algoritmo con un budget ridotto di Fitness Evaluation (FE = 10.000) su un'istanza di default, eseguire:

```bash
python main.py --quick
```

## Protocollo Sperimentale e Statistiche

In conformità alle richieste del progetto, gli script di output calcoleranno le seguenti metriche per ogni istanza testata:

- **Best**: la migliore soluzione trovata (miglior costo) tra tutte le 5 run.
- **Mean**: la media dei costi delle migliori soluzioni di ciascun run.
- **Standard Deviation**: la deviazione standard delle soluzioni trovate.
- **Satisfability**: numero di città soddisfatte.
- **Avg Iterations**: numero medio di iterazioni necessarie per raggiungere la migliore soluzione.
- **Gap %**: lo scostamento percentuale rispetto alla *Best Known Solution (BKS)* dell'istanza.

## Interfaccia Grafica (Tkinter)

Per configurare i parametri e osservare l'esperimento in tempo reale
(convergenza live, rotte delle soluzioni migliori, log colorato, statistiche
live) è disponibile un'interfaccia Tkinter inclusa nella cartella `interface/`.

### Avvio

Dalla **root del progetto** (`CVRP-IA/`), uno qualsiasi di questi comandi
funziona: il launcher &#232; configurato per risolvere correttamente i moduli
sia eseguito come modulo sia come script diretto.

```bash
# Forma canonica (consigliata)
python -m interface

# Equivalente: script diretto
python interface/app.py

# Su Windows (percorso con backslash)
python .\interface\app.py
```

Tutte le dipendenze (Tkinter + matplotlib) sono gi&#224; soddisfatte dai requisiti
del progetto. *Non serve installare nulla di nuovo* — Tkinter &#232; incluso
nella libreria standard di Python.

> **Errore frequente**: lanciare lo script da una directory diversa da
> quella del progetto, o dimenticarsi del `-m` eseguendo `app.py`
> direttamente, può dare `ModuleNotFoundError: No module named 'interface'`
> o `'src'`. In tal caso, spostati nella root del progetto e rilancia con
> uno dei comandi qui sopra.

### Cosa offre l'interfaccia

- **Toolbar superiore**: selezione istanza dal menu a tendina (vengono
  elencati automaticamente i file `.vrp` in `instances/`), caricamento di
  un file `.vrp` custom, pulsanti **▶ Avvia / ■ Stop / ↻ Reset parametri /
  💾 Salva log**.
- **Pannello parametri** (sinistra): form con spinbox per tutti i
  parametri del CLONALG —
  `n_runs`, `max_fe`, `pop_size`, `clone_factor`, `beta`, `rho`,
  `replacement_rate`, `local_search_top_k`, `seed`.
- **Tab "Convergenza live"**: grafico matplotlib con una curva per ogni
  run (axis *FE → best cost*) e linea orizzontale di riferimento sul BKS.
  Il grafico viene aggiornato ~5 volte al secondo e mostra in tempo reale
  dove stanno convergendo le soluzioni.
- **Tab "Miglior soluzione"**: visualizzazione delle rotte della soluzione
  migliore trovata finora sulla mappa delle coordinate (depot in rosso,
  clienti in blu, ogni rotta in un colore distinto). Si aggiorna ogni volta
  che l'algoritmo trova un nuovo best.
- **Pannello stato corrente**: FE correnti, generazione, miglior costo,
  tempo trascorso, barra di avanzamento percentuale.
- **Tabella di riepilogo per run**: una riga per ogni run completata
  (best, numero rotte, clienti serviti, generazione del best, tempo).
- **Riepilogo finale**: best / mean / std / gap% / soddisfazione / tempo totale.
- **Log eventi** colorato in fondo (info / success / warning / error).
  Buffer limitato a 1000 righe; esportabile su file con **💾 Salva log**.

### Come funziona (threading)

- Il calcolo gira in un thread separato (`ExperimentWorker`) — l'interfaccia
  resta sempre reattiva, anche durante run lunghe.
- Stop interrompe la run corrente in modo *graceful*; la run interrotta
  viene scartata dai riassunti, le altre vengono conservate.
- I parametri sono validati prima dell'avvio (range, tipo) per prevenire
  errori a runtime.

### Argomenti CLI

```bash
python -m interface                    # usa ./instances
python -m interface --instances-dir D  # usa una directory diversa
```
