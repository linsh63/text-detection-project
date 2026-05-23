# Baseline 对比实验结果

| name | description | accuracy | macro_f1 | f1_spam | precision_spam | recall_spam | pr_auc | roc_auc | recall_at_precision_90 | recall_at_precision_95 | false_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| char_tfidf_linear_svm | 字符级 TF-IDF(1-3gram) + Linear SVM | 0.9870 | 0.9654 | 0.9380 | 0.9021 | 0.9768 | 0.9900 | 0.9986 | 0.9768 | 0.9553 | 0.0119 |
| word_tfidf_linear_svm | 词级 TF-IDF(1-2gram) + Linear SVM | 0.9840 | 0.9573 | 0.9236 | 0.8896 | 0.9603 | 0.9754 | 0.9938 | 0.9536 | 0.9040 | 0.0133 |
| char_tfidf_logreg | 字符级 TF-IDF(1-3gram) + Logistic Regression | 0.9773 | 0.9417 | 0.8960 | 0.8324 | 0.9702 | 0.9770 | 0.9967 | 0.9603 | 0.8858 | 0.0219 |
| word_tfidf_logreg | 词级 TF-IDF(1-2gram) + Logistic Regression | 0.9725 | 0.9297 | 0.8749 | 0.8070 | 0.9553 | 0.9475 | 0.9912 | 0.8940 | 0.7086 | 0.0256 |
