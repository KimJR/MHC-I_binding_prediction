# MHC-I binding prediction
Universität Tübingen  
Computational Immunomics  
SS 2018  

Benjamin Hartert, Kim Julia Radke, Merle Kammer


### Abstract
Einen sehr wichtigen Teil in der Immunabwehr stellt die Antigenprozessierung dar. Die Vorhersage der Bindung von Peptiden, welche durch Prozessierung von Antigenen entstehen, sind besonders interessant in der Entwicklung von Impfstoffen. Das im Folgenden erarbeitete Modell zur Bindevorhersage von Peptidenan MHC-I Molekülen wird mit Hilfe von maschinellem lernen realisiert. Gearbeitet wird mit Python3.6.5 unter Verwendung der MachineLearning-Bibliothek scikit-learn. Zu Beginn wird der Klassifizierer Gradient Bossting ausgewählt. Anschließend wird eine Codierung für die Peptide implementiert. Der unbalancierte Datensatz wird durch Einführung einer Gewichtung balanciert. Damit bessere Vorhersagen möglich sind werden die Parameter des Gradient Boosting Klassifizierers mittels Gridsearch optimiertund eine Kreuzvalidierung angewendet. Abschließend wird das Modell mit einer ROC-Kurve anhand von Spezfität, Sensitivität und MMC Werten ausgwertet.

### Results  

<img src="/Images/Results.png" alt="Drawing" width="450"/>
<table>
  <thead>
    <tr>
      <th> </th>
      <th>TPR</th>
      <th>TNR</th>
      <th>PPV</th>
      <th>ACC</th>
      <th>MCC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Wert</td>
      <td>0.75</td>
      <td>0.96</td>
      <td>0.82</td>
      <td>0.92</td>
      <td>0.74</td>
    </tr>
  </tbody>
</table>

True positive rate (TPR), True negative rate (TNR), Prediction (PPV), Accuracy (ACC), Matthews Korrelationskoeffizient (MCC)
