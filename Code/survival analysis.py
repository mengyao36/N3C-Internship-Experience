# Import Packages
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import count, sum
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import year, month, dayofmonth
from pyspark.sql.functions import col, when, isnull
from pyspark.sql.functions import round
from pyspark.sql.functions import lit
from pyspark.sql.functions import datediff
from pyspark.sql.functions import coalesce
from pyspark.sql.functions import least
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.plotting import plot_lifetimes


# I. Prepare the Table for Survival Anlysis and Kaplan-Meier Curve
### Patient ID: person_id
### Covid Group: comparison_group (inclduing covid negative, covid mild, and covid severe)
### Study Period Start Date (Covid Positive/Negative First Diagnosed Date): index_date
### Study Period End Date (2 Years after Covid Positive/Negative First Diagnosed Date): post_index_date
### Overall Earliest CKD Clinical Observation Date: first_ckd_date
### Overall Earliest Cilinical Observation Date: first_observation_date
def prep_for_km_ckd(ckd_cohort): 

    df = ckd_cohort.select('person_id', 'comparison_group', 'index_date', 'post_index_date', 'first_ckd_date', 'first_observation_date')

    return df

## Create `event` and `event_duration` Columns for Survival Anlysis and Kaplan-Meier Curve
def km_ckd(prep_for_km_ckd):

    df = prep_for_km_ckd

    df = df.withColumn('compare_ckd_and_study_period',
                                                 when((col('first_ckd_date') >= col('index_date')) &
                                                      (col('first_ckd_date') <= col('post_index_date')), 'in')
                                                 .when(col('first_ckd_date') < col('index_date'), 'before')
                                                 .when(col('first_ckd_date') > col('post_index_date'), 'after')
                                                 .otherwise(None))

    df = df.withColumn('compare_first_observation_date_and_study_period',
                                                 when((col('first_observation_date') >= col('index_date')) &
                                                      (col('first_observation_date') <= col('post_index_date')), "in")
                                                 .when(col('first_observation_date') < col('index_date'), 'before')
                                                 .when(col('first_observation_date') > col('post_index_date'), 'after')
                                                 .otherwise(None))

    df = df.withColumn('event_end_date',
                    when(col('compare_ckd_and_study_period') == 'in', col('first_ckd_date'))
                    .when(col('compare_ckd_and_study_period') == 'after', col('post_index_date'))
                    .when(col('first_ckd_date').isNull() & (col('compare_first_observation_date_and_study_period') == 'in'), col('first_observation_date'))
                    .when(col('first_ckd_date').isNull() & (col('compare_first_observation_date_and_study_period') == 'after'), col('post_index_date'))
                    .otherwise(None))

    # Create `event` Column (happens - 1, censored - 0)
    df = df.withColumn('event', when(col('compare_ckd_and_study_period') == 'in', 1).otherwise(0))

    # create `event_duration` Column
    df = df.withColumn('event_duration', datediff(col('event_end_date'), col('index_date')))

    return df


# II. Apply Survival Analysis and Generate Kaplan-Meier Curve
## For Overall Population without Comparing Different Covid Condition Groups
def km_ckd_no_compare(km_ckd):

    # Convert PySpark Data to Pandas Data Format
    df = km_ckd.toPandas()

    # Create KaplanMeierFitter object
    kmf = KaplanMeierFitter()

    D = df['event_duration']
    E = df['event']

    ax = plt.subplot()
    kmf.fit(D,E)
    # Show At-Risk Table at bottom
    kmf.plot_survival_function(ax = ax, at_risk_counts = True)
    
    kmf.plot(title='Kaplan-Meier Survival Curve - CKD')

    plt.show()

## Compare Differernt Covid Condition Groups
def km_ckd_compare(km_ckd):

    df = km_ckd.toPandas()

    D = df['event_duration']
    E = df['event']
    group1 = (df.comparison_group == 'covid negative')
    label1 = 'COVID Negative'
    group2 = (df.comparison_group == 'covid mild')
    label2 = 'Mild COVID'
    group3 = (df.comparison_group == 'covid severe')
    label3 = 'Severe COVID'

    def km_and_logrank(group1, group2, group3, D, E, label1, label2, label3):

        D1=D[group1]
        E1=E[group1]
    
        D2=D[group2]
        E2=E[group2]

        D3=D[group3]
        E3=E[group3]

        kmf = KaplanMeierFitter()

        ax = plt.subplot(111)
        ax = kmf.fit(D1, E1, label=f"{label1}").plot(ax=ax, ci_show=True, show_censors=False, color='blue')
        ax = kmf.fit(D2, E2, label=f"{label2}").plot(ax=ax, ci_show=True, show_censors=False, color='orange')
        ax = kmf.fit(D3, E3, label=f"{label3}").plot(ax=ax, ci_show=True, show_censors=False, color='red')
        ax.set_xlabel('Time')
        ax.set_ylabel('Survival Rate')
        plt.ylim([0.8, 1.0])

    km_and_logrank(group1, group2, group3, D, E, label1, label2, label3)

    plt.show()

## Create Kaplan-Meier Curve for Different Covid Condition Groups Seperately
### Extarct Covid Negative Patient Data and Create Figure
def km_ckd_negative(km_ckd):
    
    df = km_ckd.where(km_ckd.comparison_group == 'covid negative')

    return df

def ckd_negative(km_ckd_negative):

    df = km_ckd_negative.toPandas()

    kmf = KaplanMeierFitter()

    D = df['event_duration']
    E = df['event']

    ax = plt.subplot(111)
    kmf.fit(D,E)
    kmf.plot_survival_function(ax = ax, ci_show=True, show_censors=False, at_risk_counts = True, color='blue')
    plt.ylim([0.95, 1.0])
    
    kmf.plot(title='KM - CKD for Covid Negative Group', color = 'blue')

    plt.show()

### Extarct Covid Mild Patient Data and Create Figure
def km_ckd_mild(km_ckd):
    
    df = km_ckd.where(km_ckd.comparison_group == 'covid mild')

    return df

def ckd_mild(km_ckd_mild):

    df = km_ckd_mild.toPandas()

    kmf = KaplanMeierFitter()

    D = df['event_duration']
    E = df['event']

    ax = plt.subplot(111)
    kmf.fit(D,E)
    kmf.plot_survival_function(ax = ax, ci_show=True, show_censors=False, at_risk_counts = True, color='orange')
    plt.ylim([0.95, 1.0])
    
    kmf.plot(title='KM - CKD for Covid Mild Group', color = 'orange')

    plt.show()

### Extarct Covid Severe Patient Data and Create Figure
def km_ckd_severe(km_ckd):
    
    df = km_ckd.where(km_ckd.comparison_group == 'covid severe')

    return df

def ckd_severe(km_ckd_severe):

    df = km_ckd_severe.toPandas()

    kmf = KaplanMeierFitter()

    D = df['event_duration']
    E = df['event']

    ax = plt.subplot(111)
    kmf.fit(D,E)
    kmf.plot_survival_function(ax = ax, ci_show=True, show_censors=False, at_risk_counts = True, color='red')
    plt.ylim([0.95, 1.0])
    
    kmf.plot(title='KM - CKD for Covid Severe Group', color = 'red')

    plt.show()