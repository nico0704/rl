# Second Mini-Project

## Vorgeschlagene Methoden: 
### REINFORCE
1. Was ist das?
    - REINFORCE ist eien Policy Gradient Methode, die direkt die Policy optimiert, ohne eine Wertfunktion wie bei DQN zu verwenden
    - Idee ist, Aktionen, die zu einer hohen Belohnung führen, wahrscheinlicher zu machen
2. Wie funktioniert es?
    - der Agent sammelt eine Episode (eine komplette Abfolge von Zuständen, Aktionen und Belohnungen)
    - nach Abschluss der Episode wird gesamte Belohnung berechnet
    - basierend uaf Belohnung werden Gewichte der Policy angepasst:
        - Aktionen, die zu positiven Belohnungen geführt haben, werden wahrscheinlicher gemacht
        - Aktionen, die negative Ergebnisse brachten, werden unwahrscheinlicher
3. Vorteile: 
    - einfach und leicht zu implementieren
    - direktes Optimieren der Policy
4. Nachteile:
    - kann hohe Varianz in Gradienten haben (-> Training dauert länger)
    - instabil, wenn Umgebung komplex ist

### TRPO (Trust Region Policy Optimization)
1. Was ist das?
    - TRPO it eine verbesserte Policy Gradient Methode, die dafür sorgt, dass die Policy bei jedem Update nicht zu stark verändert wird -> dadurch wird Training stabiler
2. Wie funktioniert es?
    - es verwendet ein mathematisches Konzept -> Trust Region:
        - bei jedem Update darf Policy nicht zu weit von alten entfernt sein
        - dies wird durch zusätzliche Einschränkungen in Optimierung erzwungen
3. Vorteile:
    - stabiles Training
    - besonders gut für komplexe Umgebungen geeignet
4. Nachteile:
    - komplex in Implementierung 
    - benötigt mehr Rechenleistung als REINFORCE 

### Natural Policy Gradient 
1. Was ist das?
    - eine Methode, die eine "natürliche" Metrik für die Optimierung nutzt -> das bedeutet, dass sie die Geometrie der Policy berücksichtigt, um bessere Updates zu machen
2. Wie funktioniert es?
    - anstatt nur den Gradienten zu verwenden, wird eine natürliche Geometrie-Metrik (Fisher Information Matrix) berechnet
    - sorgt für effizientere & stabilere Schritte in Richtung der optimalen Policy
3. Vorteile
    - bessere & stabilere Ergebnisse als bei REINFORCE
    - weniger Varianz in Gradienten
4. Nachteile
    - muss Fisher Information Matrix berechnen -> kann rechenintensiv sein

### PPO (Proximal Policy Optimization)
1. Was ist das?
    - PPO ist eine der beliebtesten & häufigstem verwendeten Algorithmen -> ist eine Weiterentwicklung von TRPO, aber einfacher & effizienter
2. Wie funktioniert es?
    - PPO sorgt dafür, dass Policy bei jedem Update in kleinen Schritten angepasst wird
    - es verwendet spezielle Verlustfunktion, die sicherstellt, dass Policy nicht zu stark verändert wird
3. Vorteile
    - einfach und stabil
    - funktioniert gut in vielen Umgebungen
4. Nachteile
    - nicht optimal für Umgebungen mit sehr hoher Varianz

### DDPG (Deep Deterministic Policy Gradient)
1. Was ist das?
    - speziell für Umgebungen mit kontinuierlichen Aktionsräumen (wie Robotersteuerung -> kontinuierliche Aktionen wie Drehwinkel)
2. Wie funktioniert es?
    - kombiniert die Policy-Gradient-Methode mit einer Q-Learning-Methode
    - Algorithmus trainiert zwei neuronale Netze
        - Actor: Policy, die Aktionen auswählt
        - Critic: Wertfunktion, die die Qualität der Aktionen bewertet
3. Vorteile
    - gut geeignet für kontinuierliche Aktionsräume
    - liefert gute Ergebnisse in physikalischen Simulationen
4. Nachteile
    - kann instabil sein, wenn Hyperparameter nicht gut gewählt sind
    - oft langsamer als diskrete Methoden wir PPO

### TD3 (Twin Delayed Deep Deterministic Policy Gradient)
1. Was ist das?
    - verbesserte Version von DDPG,, die einige Schwächen von DDPG behebt
2. Wie funktioniert es?
    - fügt zwei wichtige Verbesserungen zu DDPG hinzu:
        - zwei Q-Netzwerke: verwendet Wertfunktionen (Critics), um Verzerrungen bei Bewertung zu vermeiden
        - Delayed Updates: der Actor (Policy) wird seltener aktualisiert, um stabilere Ergebnisse zu erzielen
3. Vorteile
    - stabiler als DDPG
    - bessere Ergebnisse bei kontinuierlichen Aufgaben
4. Nachteile
    - etwas komplexer zu implementieren 




# Tetris
1. Agent & Environment
    - Agent: the rl-algorithm (example: PPO) learns which action should be executed in which state
    - Environment: the tetris-game -> simulates game rules and returns the new state and reward after each action
2. State:
    - game-board: 2d-Array with every entry containing if cell is used or unused -> 0 for unused and 1 for used
    - current game-block: needs information about form, position and orientation
    (- next game-block) 
3. Actions:
    - moving: left and right
    - rotate: 90 degrees
    - fast move-down: move block directly to lowest possible position
    - move-down normally: might be automatically?
4. Reward:
    - remove lines: reward for every removed line (more lines at the same time = higher reward) -> something like +10 for one line, +30 for 2 lines
    - good placing: small reward for placing the block with small gaps
    - game over: strongly negative reward for game over (-100)
5. Goal: 
    - remove as many lines as possible
    - delay game over as long as possible
    - maximize reward
6. Training:
    - observation: agent observes current state of gameboard
    - action: agent selects action based on current policy
    - environment: returns new state and reward
    - policy: adapting policy for better future-decision
7. Policy Gradient Algorithm: PPO 
    - apparently good in high dimensional states
    - apparently robust for sparse reward