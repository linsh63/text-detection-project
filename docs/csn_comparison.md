# 字符相似性网络优化对比

| name | description | clean_accuracy | clean_f1_spam | clean_recall_spam | clean_recall_at_precision_95 | adv_recall_spam | adv_f1_spam | adv_false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| char_csn_aug_tfidf_linear_svm | 字符相似性归一化 + 关键词增强 + 字符级 TF-IDF + Linear SVM | 0.9835 | 0.9225 | 0.9752 | 0.9454 | 1.0000 | 1.0000 | 0.0000 |
| char_csn_aug_tfidf_logreg | 字符相似性归一化 + 关键词增强 + 字符级 TF-IDF + Logistic Regression | 0.9742 | 0.8832 | 0.9702 | 0.8874 | 1.0000 | 1.0000 | 0.0000 |
| char_csn_tfidf_logreg | 字符相似性归一化 + 字符级 TF-IDF(1-3gram) + Logistic Regression | 0.9773 | 0.8960 | 0.9702 | 0.8858 | 0.1815 | 0.3072 | 221.0000 |
| char_tfidf_logreg | 字符级 TF-IDF(1-3gram) + Logistic Regression | 0.9773 | 0.8960 | 0.9702 | 0.8858 | 0.1444 | 0.2524 | 231.0000 |
| char_csn_tfidf_linear_svm | 字符相似性归一化 + 字符级 TF-IDF(1-3gram) + Linear SVM | 0.9870 | 0.9380 | 0.9768 | 0.9553 | 0.1037 | 0.1879 | 242.0000 |
| char_tfidf_linear_svm | 字符级 TF-IDF(1-3gram) + Linear SVM | 0.9870 | 0.9380 | 0.9768 | 0.9553 | 0.0667 | 0.1250 | 252.0000 |
