# THE STUDY OF BREAST CANCER
# INTRODUCTION
This is a simple project, my very first, completed during my academic internship in May 2023. In this project, I performed Exploratory Data Analysis (EDA) and fitted a Logistic Regression predictive model to the breast cancer data, aiming to predict who will have a recurrence of cancer after undergoing some treatments.

# DATASET
In this project I use breast cancer data, that contains 10 columns they are explained as follows.

**Variable Characteristics**

**AGE**
In the provided data, the "age" column represents the age range of the patients. The age ranges are categorized into different groups, such as:
*	30-39: Patients between the ages of 30 and 39.
*	40-49: Patients between the ages of 40 and 49.
*	50-59: Patients between the ages of 50 and 59.
*	60-69: Patients between the ages of 60 and 69.
*	70-79: Patients between the ages of 70 and 79.
 

**MENOPAUSE**
In the given dataset, "menopause" is one of the attributes or features describing the patients. It indicates the menopausal status of the patients. Menopause refers to the natural biological process in women when they permanently stop menstruating and are no longer able to conceive.

The "menopause" feature in the dataset has three categories:

"premeno":
 This category indicates that the patient has not yet reached menopause and is still experiencing regular menstrual cycles. The term "premeno" stands for premenopausal, meaning the patient is in the premenopausal stage.
 
"ge40":
 This category indicates that the patient has gone through menopause and is in the postmenopausal stage. The term "ge40" stands for greater than or equal to 40, implying that the patient's age at the time of menopause was 40 or older.

"lt40": 
This category indicates that the patient has gone through menopause, but her age at the time of menopause was less than 40. The term "lt40" stands for less than 40.
By analyzing this attribute in the dataset, researchers or analysts can study the association between menopausal status and other variables to understand how menopause may impact the occurrence or recurrence of breast cancer.

**TUMOR-SIZE**
In the given dataset, "tumor-size" is one of the attributes describing the patients. It provides information about the size of the tumor observed in the patients' breast cancer cases.
Each category represents a range of tumor sizes in millimeters. For example, "15-19" indicates that the tumor size falls between 15mm and 19mm, "35-39" indicates a tumor size between 35mm and 39mm, and so on.

**INV-NODES**
In the given dataset, "inv-nodes" is one of the attributes describing the patients. It provides information about the number of axillary lymph nodes involved or affected by breast cancer.
The category "0-2" indicates that the patient has between 0 to 2 axillary lymph nodes involved by breast cancer. This means that the cancer cells have spread to a maximum of 2 nearby lymph nodes.

**What is lymph Node?**
A lymph node is a small, bean-shaped structure found throughout the body that plays a crucial role in the immune system. 
In breast cancer, lymph nodes act as filters, trapping and containing cancer cells that may have spread from the primary tumor. Their examination helps determine the stage of cancer and guides treatment decisions.

**NODE-CAPS**
It provides information about the presence or absence of encapsulation or involvement of the tumor with the lymph nodes.
"yes": It suggests that the tumor is contained within a capsule or barrier, which may help restrict its spread.
"no": It implies that the tumor is not enclosed within a capsule and may have a higher likelihood of spreading beyond the lymph nodes.

**DEG-MALIG**
In the given dataset, the "deg-malig" attribute represents the degree of malignancy or the severity of the tumor.
The "deg-malig" feature provides information about the histological grade or cellular characteristics of the tumor cells.
It is measured on a scale from 1 to 3, where a higher value indicates a higher degree of malignancy or aggressiveness.

**BREAST**
In the given dataset, the "breast" attribute describes the location of the tumor within the breast.
The "breast" feature indicates whether the tumor is located in the right or left breast.

**BREAST QUAD**
It provides information about the specific quadrant or region of the breast affected by cancer. Understanding the breast quad helps determine the extent and spread of the tumor, aiding in treatment planning and prognosis evaluation. 
The provided data contains information on the breast quad for each case, including left_up, central, left_low, right_low and right_up.

**IRRADIAT**
"Irradiat" in the given data refers to whether or not the patient has received radiation therapy as part of their breast cancer treatment. It is represented as "yes" or "no." 
Radiation therapy involves using high-energy rays to target and destroy cancer cells. The information on irradiat helps assess the specific treatment approach taken for each patient and its potential impact on disease management and recurrence.

**CLASS** (Target Variable)
In the given data, "Class" represents the outcome or classification of breast cancer for each patient. It indicates whether the patient experienced a recurrence of the cancer ("recurrence-events") or did not have a recurrence ("no-recurrence-events"). 
The "no-recurrence-events" class signifies that the patient did not have a recurrence of breast cancer after receiving radiation therapy.

# METHODS & MODELS
**Chi-Square Test:**
The Chi-Square test is a statistical method used to determine if there is a significant association between two categorical variables. It helps us understand whether the observed distribution of data differs from the expected distribution and is commonly used in contingency table analysis. This test is valuable for exploring relationships between categorical variables and assessing if these relationships are statistically significant.

**Correlation:**
Correlation measures the strength and direction of a linear relationship between two continuous variables. The correlation coefficient ranges from -1 to 1, where -1 indicates a perfect negative correlation, 1 indicates a perfect positive correlation, and 0 implies no linear correlation. This statistical tool helps us understand how changes in one variable relate to changes in another, providing insights into patterns and dependencies in the data.

**Logistic Regression:**
Logistic Regression is a statistical model used for binary classification problems. It predicts the probability of an event occurring (like a patient having a disease) based on one or more predictor variables. Unlike linear regression, logistic regression uses the logistic function to constrain predictions between 0 and 1, making it suitable for tasks where the outcome is binary. This method is widely employed in fields like medicine and finance for making predictions and understanding the factors influencing a particular outcome.

**Accuracy Score:**
Accuracy score is a metric used to assess the performance of a classification model. It represents the ratio of correctly predicted instances to the total instances in the dataset. The formula is (number of correct predictions / total number of predictions). A higher accuracy score, usually expressed as a percentage, indicates better model performance in correctly classifying instances across different classes.

# PROCEDURE
**`Data Preprocessing`**

In this data preprocessing section, I performed the following steps:

**Missing Values:** I checked for missing values and found one in the 'inv-nodes' column. Since there was only one missing value, I opted to remove that specific data point from the dataset.

**Data Encoding:** For the given data, only the 'deg-malig' column contains numerical data, while the remaining columns consist of text data (strings). To address this, I applied label encoding to all nine columns, excluding the 'deg-malig' column, to convert text data into numerical format.

**`Feature Selection`**

In this Feature Selection phase, I conducted both the Chi-Square test and correlation matrix to select feature variables that significantly contribute to predicting the class variable (recurrent or non-recurrent cancer).

**Chi-Square Test:**
First, I applied the Chi-Square test to each individual feature variable with the target variable. At the end of the analysis, I identified only four feature variables (irradiate, inv-nodes, node-caps, and deg-malig) out of the nine that significantly contribute. Consequently, I removed the other feature variables (age, tumor_size, breast, breast_quad, menopause) that did not contribute significantly to the model.

**Correlation Matrix:**
Following the Chi-Square test, I created a correlation heatmap to assess whether the feature variables exhibited multicollinearity. Fortunately, none of the variables showed multicollinearity.

**Multicollinearity :**
Multicollinearity refers to a situation in regression analysis where two or more independent variables in a model are highly correlated, making it challenging to discern their individual effects on the dependent variable. High multicollinearity may obscure the true relationships between variables, making it difficult to interpret the model accurately. Detecting and addressing multicollinearity is crucial for ensuring the robustness and reliability of regression models.

**`Model Fitting`**

Next, I split the data into training and testing sets, with 80% for training and 20% for testing. Afterward, I fitted a logistic regression model to the training data and used that model to predict the test data. The calculated prediction accuracy for this model is 86.21%.

# CONCLUSION
my project involved thorough data preprocessing, including handling missing values, label encoding, and feature selection through Chi-Square and correlation tests. The logistic regression model achieved an accuracy of 86.21% on predicting breast cancer recurrence in the test set, highlighting the effectiveness of the chosen features.
