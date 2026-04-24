PS C:\Codes\CAPSTONE\DermaScan\AI\train> python two_way_train.py       
Loading cached embeddings...
  Loaded acne-blackheads-mild: 219 embeddings
  Loaded acne-blackheads-moderate: 176 embeddings
  Loaded acne-blackheads-severe: 214 embeddings
  Loaded acne-cyst: 310 embeddings
  Loaded acne-fungal-mild: 349 embeddings
  Loaded acne-fungal-severe: 258 embeddings
  Loaded acne-nodules: 309 embeddings
  Loaded acne-papules-mild: 697 embeddings
  Loaded acne-papules-moderate: 390 embeddings
  Loaded acne-papules-severe: 319 embeddings
  Loaded acne-pustules-mild: 588 embeddings
  Loaded acne-pustules-moderate: 328 embeddings
  Loaded acne-pustules-severe: 497 embeddings
  Loaded acne-whiteheads-mild: 316 embeddings
  Loaded acne-whiteheads-moderate: 243 embeddings
  Loaded acne-whiteheads-severe: 215 embeddings
  Loaded eczema-mild: 511 embeddings
  Loaded eczema-moderate: 334 embeddings
  Loaded eczema-severe: 281 embeddings
  Loaded enlarged-pores-mild: 568 embeddings
  Loaded enlarged-pores-moderate: 670 embeddings
  Loaded enlarged-pores-severe: 677 embeddings
  Loaded melasma-mild: 512 embeddings
  Loaded melasma-moderate: 603 embeddings
  Loaded melasma-severe: 474 embeddings
  Loaded milia-mild: 209 embeddings
  Loaded milia-moderate: 490 embeddings
  Loaded milia-severe: 165 embeddings
  Loaded normal-skin: 344 embeddings
  Loaded out-of-scope: 386 embeddings
  Loaded post-inflammatory-erythema-mild: 350 embeddings
  Loaded post-inflammatory-erythema-moderate: 138 embeddings
  Loaded post-inflammatory-erythema-severe: 318 embeddings
  Loaded post-inflammatory-hyperpigmentation-mild: 407 embeddings
  Loaded post-inflammatory-hyperpigmentation-moderate: 435 embeddings
  Loaded post-inflammatory-hyperpigmentation-severe: 261 embeddings
  Skipping psoriasis

Total: (13561, 6144)

==================================================
Training Stage 1: Condition Classifier
==================================================
Conditions (15): ['acne-blackheads', 'acne-cyst', 'acne-fungal', 'acne-nodules', 'acne-papules', 'acne-pustules', 'acne-whiteheads', 'eczema', 'enlarged-pores', 'melasma', 'milia', 'normal-skin', 'out-of-scope', 'post-inflammatory-erythema', 'post-inflammatory-hyperpigmentation']
Stage 1 Train: 100.00%
Stage 1 Test:  95.02%

Per-class report:
                                     precision    recall  f1-score   support

                    acne-blackheads       0.97      0.93      0.95       122
                          acne-cyst       0.81      0.81      0.81        62
                        acne-fungal       0.96      1.00      0.98       121
                       acne-nodules       0.79      0.94      0.86        62
                       acne-papules       0.95      0.90      0.93       281
                      acne-pustules       0.96      0.99      0.97       283
                    acne-whiteheads       0.96      0.99      0.97       155
                             eczema       0.95      0.94      0.95       225
                     enlarged-pores       0.99      0.98      0.99       383
                            melasma       0.98      0.96      0.97       318
                              milia       0.96      0.95      0.96       173
                        normal-skin       0.92      0.97      0.94        69
                       out-of-scope       0.97      0.99      0.98        77
         post-inflammatory-erythema       0.89      0.84      0.87       161
post-inflammatory-hyperpigmentation       0.93      0.96      0.94       221

                           accuracy                           0.95      2713
                          macro avg       0.93      0.94      0.94      2713
                       weighted avg       0.95      0.95      0.95      2713

Confusion matrix saved.
Stage 1 saved.

==================================================
Training Stage 2: Severity Classifiers
==================================================

[acne-blackheads] 609 samples, classes: ['mild', 'moderate', 'severe']
  Train: 99.59% | Test: 89.34%
              precision    recall  f1-score   support

        mild       0.95      0.95      0.95        44
    moderate       0.81      0.86      0.83        35
      severe       0.90      0.86      0.88        43

    accuracy                           0.89       122
   macro avg       0.89      0.89      0.89       122
weighted avg       0.89      0.89      0.89       122

  Saved: ../trained_data_two_stage_apr_23\stage2_acne-blackheads.pkl

[acne-cyst] No severity variants — skipping

[acne-fungal] 607 samples, classes: ['mild', 'severe']
  Train: 99.18% | Test: 98.36%
              precision    recall  f1-score   support

        mild       0.99      0.99      0.99        70
      severe       0.98      0.98      0.98        52

    accuracy                           0.98       122
   macro avg       0.98      0.98      0.98       122
weighted avg       0.98      0.98      0.98       122

  Saved: ../trained_data_two_stage_apr_23\stage2_acne-fungal.pkl

[acne-nodules] No severity variants — skipping

[acne-papules] 1406 samples, classes: ['mild', 'moderate', 'severe']
  Train: 99.38% | Test: 86.52%
              precision    recall  f1-score   support

        mild       0.94      0.92      0.93       140
    moderate       0.74      0.85      0.79        78
      severe       0.88      0.77      0.82        64

    accuracy                           0.87       282
   macro avg       0.85      0.84      0.85       282
weighted avg       0.87      0.87      0.87       282

  Saved: ../trained_data_two_stage_apr_23\stage2_acne-papules.pkl

[acne-pustules] 1413 samples, classes: ['mild', 'moderate', 'severe']
  Train: 99.12% | Test: 94.70%
              precision    recall  f1-score   support

        mild       0.96      0.94      0.95       118
    moderate       0.88      0.91      0.90        66
      severe       0.98      0.98      0.98        99

    accuracy                           0.95       283
   macro avg       0.94      0.94      0.94       283
weighted avg       0.95      0.95      0.95       283

  Saved: ../trained_data_two_stage_apr_23\stage2_acne-pustules.pkl

[acne-whiteheads] 774 samples, classes: ['mild', 'moderate', 'severe']
  Train: 97.25% | Test: 86.45%
              precision    recall  f1-score   support

        mild       0.88      0.84      0.86        63
    moderate       0.78      0.86      0.82        49
      severe       0.95      0.91      0.93        43

    accuracy                           0.86       155
   macro avg       0.87      0.87      0.87       155
weighted avg       0.87      0.86      0.87       155

  Saved: ../trained_data_two_stage_apr_23\stage2_acne-whiteheads.pkl

[eczema] 1126 samples, classes: ['mild', 'moderate', 'severe']
  Train: 98.33% | Test: 84.07%
              precision    recall  f1-score   support

        mild       0.87      0.82      0.84       103
    moderate       0.91      0.90      0.90        67
      severe       0.73      0.82      0.77        56

    accuracy                           0.84       226
   macro avg       0.84      0.84      0.84       226
weighted avg       0.85      0.84      0.84       226

  Saved: ../trained_data_two_stage_apr_23\stage2_eczema.pkl

[enlarged-pores] 1915 samples, classes: ['mild', 'moderate', 'severe']
  Train: 92.95% | Test: 86.68%
              precision    recall  f1-score   support

        mild       0.84      0.77      0.80       114
    moderate       0.83      0.90      0.86       134
      severe       0.93      0.91      0.92       135

    accuracy                           0.87       383
   macro avg       0.87      0.86      0.86       383
weighted avg       0.87      0.87      0.87       383

  Saved: ../trained_data_two_stage_apr_23\stage2_enlarged-pores.pkl

[melasma] 1589 samples, classes: ['mild', 'moderate', 'severe']
  Train: 97.25% | Test: 94.03%
              precision    recall  f1-score   support

        mild       0.91      0.91      0.91       102
    moderate       0.93      0.92      0.92       121
      severe       0.99      1.00      0.99        95

    accuracy                           0.94       318
   macro avg       0.94      0.94      0.94       318
weighted avg       0.94      0.94      0.94       318

  Saved: ../trained_data_two_stage_apr_23\stage2_melasma.pkl

[milia] 864 samples, classes: ['mild', 'moderate', 'severe']
  Train: 100.00% | Test: 74.57%
              precision    recall  f1-score   support

        mild       0.85      0.79      0.81        42
    moderate       0.77      0.81      0.79        98
      severe       0.53      0.52      0.52        33

    accuracy                           0.75       173
   macro avg       0.72      0.70      0.71       173
weighted avg       0.75      0.75      0.75       173

  Saved: ../trained_data_two_stage_apr_23\stage2_milia.pkl

[normal-skin] No severity variants — skipping

[out-of-scope] No severity variants — skipping

[post-inflammatory-erythema] 806 samples, classes: ['mild', 'moderate', 'severe']
  Train: 100.00% | Test: 79.63%
              precision    recall  f1-score   support

        mild       0.95      0.86      0.90        70
    moderate       0.46      0.68      0.55        28
      severe       0.86      0.78      0.82        64

    accuracy                           0.80       162
   macro avg       0.76      0.77      0.76       162
weighted avg       0.83      0.80      0.81       162

  Saved: ../trained_data_two_stage_apr_23\stage2_post-inflammatory-erythema.pkl

[post-inflammatory-hyperpigmentation] 1103 samples, classes: ['mild', 'moderate', 'severe']
  Train: 99.66% | Test: 93.67%
              precision    recall  f1-score   support

        mild       0.94      0.94      0.94        82
    moderate       0.94      0.95      0.95        87
      severe       0.92      0.90      0.91        52

    accuracy                           0.94       221
   macro avg       0.93      0.93      0.93       221
weighted avg       0.94      0.94      0.94       221

  Saved: ../trained_data_two_stage_apr_23\stage2_post-inflammatory-hyperpigmentation.pkl

==================================================
DONE
==================================================
Stage 1 conditions: 15
Stage 2 severity models trained: 11
Conditions with no severity model: {'acne-cyst', 'out-of-scope', 'normal-skin', 'acne-nodules'}
PS C:\Codes\CAPSTONE\DermaScan\AI\train> 