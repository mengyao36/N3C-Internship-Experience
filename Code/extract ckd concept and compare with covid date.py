# Import Pacakages
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.functions import count
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import year, month, dayofmonth
from pyspark.sql.window import Window
from pyspark.sql.functions import col, when, isnull
from pyspark.sql.functions import round
from pyspark.sql.functions import lit


# I. Extract Unqiue Concept ID for CKD Concept (Including ESRD, Kidney Transplant, and Renal Replacement Therapy Concepts)
## Get ESRD Concept and Keep Selected Columns from `concept_set_members` Table
def end_stage_renal_disease(concept_set_members):

    df = concept_set_members.filter(concept_set_members['codeset_id'] == 000000000)
    df = df.select('codeset_id', 'concept_id', 'concept_set_name', 'concept_name')

    return df

## Get Kidney Transplant Concept and Keep Selected Columns from `concept_set_members` Table
def kidney_transplant(concept_set_members):

    df = concept_set_members.filter(concept_set_members['codeset_id'] == 111111111)
    df = df.select('codeset_id', 'concept_id', 'concept_set_name', 'concept_name')

    return df

## Get Renal Replacement Therapy Concept and Keep Selected Columns from `concept_set_members` Table
def renal_replacement_therapy(concept_set_members):

    df = concept_set_members.filter(concept_set_members['codeset_id'] == 222222222)
    df = df.select('codeset_id', 'concept_id', 'concept_set_name', 'concept_name')

    return df

## Combine ESRD, Kidney Transplant, and Renal Replacement Therapy Concepts into One CKD Concept from `concept_set_members` Table
def ckd_unique_concept_id(end_stage_renal_disease, kidney_transplant, renal_replacement_therapy):
    
    df = kidney_transplant.union(end_stage_renal_disease).union(renal_replacement_therapy)
    df = df.select('concept_id')

    return df


# II. Link CKD Concept with Other Clinical Related Tables (Including Measurement, Condition, Procedure, and Observation)
## Combine `ckd_unique_concept_id` and `measurement` Tables
def measurement_ckd(measurement, ckd_unique_concept_id):
    
    ckd = ckd_unique_concept_id
    df = measurement.join(ckd, measurement['measurement_concept_id'] == ckd['concept_id'], 'inner')

    return df

## Combine `ckd_unique_concept_id` and `condition` Tables
def condition_ckd(condition, ckd_unique_concept_id):
    
    ckd = ckd_unique_concept_id
    df = condition.join(ckd, condition['condition_concept_id'] == ckd['concept_id'], 'inner')

    return df

## Combine `ckd_unique_concept_id` and `procedure` Tables
def procedure_ckd(procedure, ckd_unique_concept_id):
    
    ckd = ckd_unique_concept_id
    df = procedure.join(ckd, procedure['procedure_concept_id'] == ckd['concept_id'], 'inner')

    return df

## Combine `ckd_unique_concept_id` and `observation` Tables
def observation_ckd(observation, ckd_unique_concept_id):
    
    ckd = ckd_unique_concept_id
    df = observation.join(ckd, observation['observation_concept_id'] == ckd['concept_id'], 'inner')

    return df


# III. Define the Overall Earliest Clinical Observation Date for Patients with CKD Condition
## Define the Earliest Measurement Date
def measurement_earliest_ckd(measurement_ckd):

    df = measurement_ckd

    df_with_date = df.withColumn(
        'date',
        F.to_date('measurement_date', 'yyyy-MM-dd'))

    df_agg = (df_with_date
        .groupBy('person_id')
        .agg(F.min('date').alias('first_date')))

    return df_agg

## Define the Earliest Condition Date
def condition_earliest_ckd(condition_ckd):

    df = condition_ckd

    df_with_date = df.withColumn(
        'date',
        F.to_date('condition_date', 'yyyy-MM-dd'))

    df_agg = (df_with_date
        .groupBy('person_id')
        .agg(F.min('date').alias('first_date')))

    return df_agg

## Define the Earliest Procedure Date
def procedure_earliest_ckd(procedure_ckd):

    df = procedure_ckd

    df_with_date = df.withColumn(
        'date',
        F.to_date('procedure_date', 'yyyy-MM-dd'))

    df_agg = (df_with_date
        .groupBy('person_id')
        .agg(F.min('date').alias('first_date')))

    return df_agg

## Define the Earliest Observation Date
def observation_earliest_ckd(observation_ckd):

    df = observation_ckd

    df_with_date = df.withColumn(
        'date',
        F.to_date('observation_date', 'yyyy-MM-dd'))

    df_agg = (df_with_date
        .groupBy('person_id')
        .agg(F.min('date').alias('first_date')))

    return df_agg

## Get the Overall Earliest Clinical Observation Date for Patients with CKD Condition
def combined_ckd_first_date(measurement_ckd, condition_ckd, procedure_ckd, observation_ckd):
    
    df1 = measurement_ckd
    df2 = condition_ckd
    df3 = procedure_ckd
    df4 = observation_ckd

    df = df1.union(df2).union(df3).union(df4)

    df_agg = (df
        .groupBy('person_id')
        .agg(F.min('first_date').alias('first_ckd_date')))

    return df_agg


# IV. Extract the First Covid Positive Diagnose Date for Patients Using `covid_patient_summary` Table
def covid_date(covid_patient_summary):
    
    df = covid_patient_summary.select(col('person_id'), col('covid_first_lab_date'))
    df = df.withColumnRenamed('covid_first_lab_date', 'covid_index_date')

    return df


# V. Convert Overall Earliest CKD Clinical Observation Date Table and First Covid Positiive Diagnose Date into One Table
def covid_and_ckd_dates(combined_ckd_first_date, covid_date):
    
    df = combined_ckd_first_date.join(covid_date, 'person_id', 'full')

    return df


# VI. Compare Overall Earliest CKD Clinical Observation Date and First Covid Positiive/Negative Diagnose Date
## Compare Overall Earliest CKD Clinical Observation Date and First Covid Positiive Date
def compare_ckd_and_covid_date(covid_and_ckd_dates):
    
    df = covid_and_ckd_dates.withColumn('pre_covid_index_date',
                                            when(isnull(col('first_ckd_date')) | isnull(col('covid_index_date')), 'not available')
                                            .when(col('first_ckd_date') < col('covid_index_date'), 'yes')
                                            .otherwise('no'))

    df = df.withColumn('ckd', when(col('pre_covid_index_date') == 'yes', lit('ckd pre covid'))
                             .otherwise(when(col('pre_covid_index_date') == 'no', lit('ckd post covid'))
                             .otherwise('ckd/covid date unavailable')))
    
    return df

## Compare Overall Earliest CKD Clinical Observation Date and First Covid Negtaive Date
def compare_ckd_and_covid_negative_date(covid_and_ckd_dates, covid_neg_people):
    
    df = covid_and_ckd_dates.join(covid_neg_people, 'person_id', 'full')

    df = df.withColumn('pre_covid_negative_index_date',
                                            when(isnull(col('first_ckd_date')) | isnull(col('covid_negative_index_date')), 'not available')
                                            .when(col('first_ckd_date') < col('covid_negative_index_date'), 'yes')
                                            .otherwise('no'))

    df = df.withColumn('ckd', when(col('pre_covid_negative_index_date') == 'yes', lit('ckd pre covid negative'))
                             .otherwise(when(col('pre_covid_negative_index_date') == 'no', lit('ckd post covid negative'))
                             .otherwise('ckd/covid negative date unavailable')))
    
    return df