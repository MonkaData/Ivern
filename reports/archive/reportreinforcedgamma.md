# Rapport - Assistant Radiologue Virtuel

> **Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.**

## Contexte du projet

Prototype pedagogique d'assistant radiologique virtuel pour radiographies thoraciques frontales. Le systeme produit un JSON structure avec trois classes possibles : `normal`, `suspicion_opacite`, `incertain`.

## Dataset utilise

- Source : RSNA Pneumonia Detection Challenge (Kaggle)
- Labels source : `/workspace/projet/data/eval/labels.csv`
- Cas evalues : 30
- Repartition : normal=15, suspicion_opacite=15

## Configurations comparees

- `gemma3_baseline`

## Tableau des metriques

| Configuration | n | Accuracy | Macro-F1 | Sensibilite | Specificite | Incertain % | JSON valide % | FP | FN | JSON invalides | Hallucinations taggees | Latence moyenne (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma3_baseline | 30 | 0.5333 | 0.4444 | 0.9333 | 0.1333 | 0.0 | 100.0 | 13 | 1 | 0 | 0 | 17460.4 |

## Matrices de confusion

### gemma3_baseline

- Classes : ['normal', 'suspicion_opacite', 'incertain']
- Matrice : `[[2, 13, 0], [1, 14, 0], [0, 0, 0]]`

## Synthese d'erreurs

- false_positives: 13
- false_negatives: 1
- uncertain_cases: 0
- json_invalid_cases: 0
- technical_errors: 0
- hallucination_flags: 0

## Cas commentes (24)

### Faux negatifs

- case=`case_027` modele=`gemma3_baseline` gt=`suspicion_opacite` pred=`normal` raison=`ok` json_valid=`True` findings=`["Deux cathéters périphériques visibles (un dans la veine subclavia gauche et l'autre dans la veine fémorale droite)", 'Silhouette pulmonaire claire', "Absence d'opacités évidentes"]` limitations=`['Cette analyse est basée sur une seule radiographie et ne peut pas exclure des pathologies subtiles.']`

### Faux positifs

- case=`case_001` modele=`gemma3_baseline` gt=`normal` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`['Opacité suspecte dans le lobe inférieur du poumon droit.', 'Silhouette pulmonaire légèrement altérée.', "Absence de signes d'effusion pleurale immédiate."]` limitations=`["Cette analyse est basée sur une seule radiographie et ne peut pas exclure d'autres causes possibles.  L'interprétation clinique doit tenir compte du contexte clinique du patient."]`
- case=`case_005` modele=`gemma3_baseline` gt=`normal` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`['Opacité suspecte dans le lobe supérieur droit du poumon.', 'Silhouette pulmonaire légèrement altérée.', 'Silhouette abdominale bien définie.']` limitations=`["Cette analyse est basée sur une seule radiographie et ne peut pas exclure d'autres pathologies. Une évaluation plus approfondie avec d'autres examens d'imagerie est nécessaire."]`
- case=`case_007` modele=`gemma3_baseline` gt=`normal` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`["Présence d'opacités pulmonaires bilatérales", 'Possible signe de consolidation', "Présence d'un tube endotraché"]` limitations=`["Cette analyse est basée sur une seule radiographie et ne peut pas exclure d'autres pathologies. Une évaluation clinique et d'autres examens d'imagerie sont nécessaires pour confirmer le diagnostic."]`
- case=`case_008` modele=`gemma3_baseline` gt=`normal` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`['Opacité suspecte dans le lobe inférieur du poumon droit.', 'Silhouette pulmonaire légèrement altérée.', 'Lignes de densité dans le parenchyme pulmonaire.']` limitations=`["L'image est une radiographie frontale et ne permet pas de visualiser les structures en profondeur.  L'interprétation est basée sur l'observation visuelle et ne peut remplacer un examen plus approfondi."]`
- case=`case_009` modele=`gemma3_baseline` gt=`normal` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`["Présence d'opacités dans les bases pulmonaires bilatérales.", 'Silhouette pulmonaire légèrement altérée.', "Absence de signes d'effusion pleurale."]` limitations=`["L'interprétation est basée sur une seule radiographie et ne peut exclure d'autres pathologies. Une évaluation plus approfondie avec des examens complémentaires est nécessaire."]`
- case=`case_010` modele=`gemma3_baseline` gt=`normal` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`['Opacité suspecte dans le lobe supérieur droit du poumon.', 'Silhouette pulmonaire légèrement altérée.', "Présence d'une structure dense dans le lobe supérieur droit."]` limitations=`["Cette analyse est basée sur une seule radiographie et ne peut pas exclure d'autres pathologies.  Un examen plus approfondi, tel qu'une tomodensitométrie, est nécessaire pour confirmer le diagnostic."]`
- case=`case_012` modele=`gemma3_baseline` gt=`normal` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`['Opacité significative dans le lobe inférieur du poumon droit.', 'Silhouette pulmonaire altérée.', 'Absence de détails clairs dans les champs pulmonaires.']` limitations=`["La résolution de l'image peut limiter la capacité à identifier des anomalies subtiles. L'absence d'images de contrôle (par exemple, radiographie antérieure) rend difficile l'évaluation de l'évolution."]`

### Reussites

- case=`case_002` modele=`gemma3_baseline` gt=`suspicion_opacite` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`["Présence d'une opacité suspecte dans le lobe inférieur gauche du poumon.", 'Légère opacification diffuse dans le lobe supérieur droit.', 'Silhouette pulmonaire légèrement altérée.']` limitations=`["L'image est une radiographie frontale et ne permet pas de distinguer les causes de l'opacification. Une tomographie par ordinateur ou une scanner pulmonaire seraient nécessaires pour une évaluation plus précise."]`
- case=`case_003` modele=`gemma3_baseline` gt=`suspicion_opacite` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`["Présence d'opacités dans les bases pulmonaires bilatérales.", 'Silhouette cardiaque normale.', "Traces de marqueurs d'imagerie (VK, R) visibles."]` limitations=`["L'interprétation est basée sur une seule radiographie et ne tient pas compte du contexte clinique du patient."]`
- case=`case_004` modele=`gemma3_baseline` gt=`normal` pred=`normal` raison=`ok` json_valid=`True` findings=`['Silhouette thoracique bien définie', 'Les côtes semblent intactes', "Absence d'opacités évidentes"]` limitations=`['Cette analyse est basée sur une seule radiographie et ne peut pas exclure des pathologies subtiles. Une évaluation clinique complète est nécessaire.']`
- case=`case_006` modele=`gemma3_baseline` gt=`suspicion_opacite` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`['Opacité diffuse dans le parenchyme pulmonaire, particulièrement au niveau du lobe supérieur droit.', "Présence d'une opacité suspecte dans le bas du poumon droit.", 'Silhouette pulmonaire légèrement altérée.']` limitations=`["L'interprétation est basée sur une seule radiographie et ne tient pas compte de l'anamnèse du patient ou d'autres examens complémentaires."]`
- case=`case_011` modele=`gemma3_baseline` gt=`suspicion_opacite` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`['Opacités bilatérales dans les champs pulmonaires', "Présence d'une opacité diffuse", 'Silhouette pulmonaire altérée']` limitations=`["Cette analyse est basée sur une seule image et ne peut pas exclure d'autres pathologies. Un examen plus approfondi est nécessaire pour établir un diagnostic précis."]`
- case=`case_013` modele=`gemma3_baseline` gt=`suspicion_opacite` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`["Présence d'une opacité dense dans le bas du poumon droit.", 'Silhouette pulmonaire légèrement altérée.', 'Lignes de carène pulmonaires visibles.']` limitations=`["Cette analyse est basée sur une seule radiographie et ne peut pas exclure d'autres pathologies.  L'interprétation clinique doit tenir compte du contexte clinique du patient."]`
- case=`case_014` modele=`gemma3_baseline` gt=`suspicion_opacite` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`['Opacité suspecte dans le lobe supérieur droit du poumon.', 'Silhouette pulmonaire légèrement altérée.', 'Lignes de carène pulmonaires visibles.']` limitations=`["L'image est une radiographie frontale et ne permet pas de visualiser les tissus mous. L'interprétation est basée sur l'observation visuelle et ne remplace pas un examen clinique et des examens complémentaires."]`
- case=`case_015` modele=`gemma3_baseline` gt=`suspicion_opacite` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`['Présence de quelques opacités floues dans les bases pulmonaires.', 'Silhouette pulmonaire légèrement altérée.', "Absence de signes d'effusion pleurale ou d'pneumothorax."]` limitations=`["L'interprétation est basée sur une seule radiographie et ne peut exclure d'autres pathologies. Une évaluation clinique et des examens complémentaires sont nécessaires pour confirmer le diagnostic."]`
- case=`case_016` modele=`gemma3_baseline` gt=`suspicion_opacite` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`["Présence d'opacités pulmonaires bilatérales, plus marquées dans l'hémithorax droit.", 'Silhouette pulmonaire légèrement altérée.', "Absence de signes d'effusion pleurale."]` limitations=`["L'interprétation est basée sur une seule radiographie et ne peut exclure d'autres pathologies. Une évaluation clinique et des examens complémentaires sont nécessaires pour confirmer le diagnostic."]`
- case=`case_018` modele=`gemma3_baseline` gt=`suspicion_opacite` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`['Opacité suspecte dans le lobe supérieur droit du poumon.', 'Silhouette pulmonaire légèrement altérée.', "Présence d'un cathéter veineux central."]` limitations=`["Cette radiographie est une vue frontale et ne permet pas de visualiser les structures en profondeur. L'interprétation doit être complétée par d'autres examens d'imagerie et un examen clinique."]`
- case=`case_019` modele=`gemma3_baseline` gt=`suspicion_opacite` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`['Opacité diffuse dans les deux champs pulmonaires', "Présence d'un cathéter droit"]` limitations=`['Cette analyse est basée sur une seule image et ne tient pas compte du contexte clinique du patient.']`
- case=`case_023` modele=`gemma3_baseline` gt=`suspicion_opacite` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`["Présence d'une opacité diffuse dans les deux champs pulmonaires.", 'Silhouette cardiaque normale.', 'Traces de médiastin normales.']` limitations=`["Cette analyse est basée sur une seule radiographie et ne peut pas exclure d'autres pathologies. Un examen plus approfondi, y compris des radiographies complémentaires, est nécessaire pour un diagnostic précis."]`
- case=`case_024` modele=`gemma3_baseline` gt=`normal` pred=`normal` raison=`ok` json_valid=`True` findings=`['Silhouette pulmonaire claire', 'Traces de côtes bien définies', 'Ombres dures minimales dans les bases pulmonaires']` limitations=`["L'interprétation est basée sur une seule radiographie et ne peut pas exclure des pathologies subtiles."]`
- case=`case_026` modele=`gemma3_baseline` gt=`suspicion_opacite` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`["Présence d'une opacité suspecte dans le lobe supérieur droit du poumon.", 'Silhouette pulmonaire légèrement altérée.', "Absence de signes d'effusion pleurale immédiate."]` limitations=`["Cette analyse est basée sur une seule radiographie et ne peut pas exclure d'autres pathologies.  L'interprétation clinique doit tenir compte du contexte clinique du patient."]`
- case=`case_028` modele=`gemma3_baseline` gt=`suspicion_opacite` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`["Présence d'opacités pulmonaires bilatérales", 'Potentiel infiltrat pulmonaire', 'Silhouette pulmonaire légèrement altérée']` limitations=`["Cette analyse est basée sur une seule radiographie et ne peut pas exclure d'autres pathologies. Un examen plus approfondi est nécessaire pour établir un diagnostic précis."]`
- case=`case_029` modele=`gemma3_baseline` gt=`suspicion_opacite` pred=`suspicion_opacite` raison=`ok` json_valid=`True` findings=`['Opacité suspecte dans le lobe supérieur droit du poumon.', "Présence d'une opacité diffuse dans le lobe inférieur droit.", 'Silhouette pulmonaire légèrement altérée.']` limitations=`["L'image est une radiographie frontale et ne permet pas de distinguer les causes de l'opacité. Une tomographie par ordinateur ou une scanner serait nécessaire pour une évaluation plus précise."]`

## Registre d'erreurs

Aucun commentaire manuel enregistre.

## Limites

- Prototype pedagogique, non destine au diagnostic.
- La classe `incertain` est une sortie de securite du systeme, pas un label Kaggle.
- Les performances dependent fortement du chargement correct des modeles Hugging Face et de CUDA.
- Le dashboard peut etre montre a partir de runs pre-calcules sans inference live.

## Conformite et securite

- Voir `ETHICS.md` pour le cadrage non clinique, les garde-fous et les limites.
- Voir `DATASET_USAGE.md` pour la provenance des donnees et les conditions d'usage.

## Conclusion

Le projet demontre une chaine d'ingenierie complete : preparation du dataset RSNA, pretraitement DICOM/PNG, sortie JSON structuree, journalisation SQLite, evaluation multi-configurations et analyse d'erreurs, sans revendiquer un usage clinique.
