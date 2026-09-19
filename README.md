# 🚢 Titanic - Data Science və Machine Learning Layihəsi

Bu layihə klassik Titanic dataseti üzərində kəşfiyyat xarakterli data analizindən (EDA) başlayaraq, xüsusiyyət mühəndisliyi (Feature Engineering), maşın öyrənməsi modellərinin tənzimlənməsi (Hyperparameter Tuning), dərin öyrənmə (Keras MLP), səhv analizi və canlı Streamlit veb-tətbiqinin hazırlanmasına qədər olan tam Data Science prosesini əhatə edir.

---

## 📌 Layihə Haqqında Qısa Məlumat
- **Məsələnin Növü:** Supervised Learning (Binary Classification / İkili Təsnifat)
- **Hədəf Dəyişəni (Target):** `Survived` (0 = Həlak oldu, 1 = Sağ qaldı)
- **Ən Yüksək Dəqiqlik (Accuracy):** **82.12%** (Keras Multi-Layer Perceptron) və **81.56%** (Random Forest / XGBoost)

---

## 🛠️ Data Boru Kəməri və Mərhələlər (Pipeline)

1. **Datanın Təmizlənməsi (Data Cleaning & Imputation):**
   - `Age` (Yaş) sütunundakı boşluqlar median dəyər ilə dolduruldu.
   - `Embarked` (Liman) sütunundakı boşluqlar mode (ən çox təkrarlanan) dəyər ilə dolduruldu.
   - Yüksək əksiklik dərəcəsinə (~77%) görə `Cabin` sütunu silindi.

2. **Xüsusiyyət Mühəndisliyi və Encoding:**
   - Yeni `FamilySize` (`SibSp` + `Parch` + 1) sütunu yaradıldı.
   - `Sex` (Binary) və `Embarked` (One-Hot Encoding) sütunları ədədə çevrildi.
   - Dərin öyrənmə modeli üçün ədədi xüsusiyyətlər `StandardScaler` ilə normallaşdırıldı.

3. **Modellərin Qiymətləndirilməsi və Müqayisəsi:**

| Model | Dəqiqlik (%) |
| :--- | :--- |
| **Keras Neyron Şəbəkəsi (MLP)** | **82.12%** |
| **XGBoost Classifier** | **81.56%** |
| **Random Forest (Tuned)** | **81.56%** |
| **LightGBM Classifier** | **81.00%** |
| **Logistic Regression** | **80.45%** |

4. **Səhv Analizi (Error Analysis):**
   - False Positive (15 nəfər) və False Negative (18 nəfər) hallar təhlil edildi.
   - **Əsas nəticə:** Model güclü tarixsel sinif fərqi (class bias) səbəbindən 3-cü sinifdə sağ qalan kişiləri yanlış təxmin etməyə meyllidir.

---

## 🚀 İnteraktiv Streamlit Veb-Tətbiqi
Layihə istifadəçilərin sərnişin parametrlərini (Yaş, Bilet Sinfi, Cinsiyyət, Qiymət) daxil edərək real vaxt rejimində sağ qalma ehtimalını hesablayan Streamlit interfeysinə malikdir.

```bash
# Veb tətbiqi lokal olaraq başlatmaq üçün:
streamlit run app.py
